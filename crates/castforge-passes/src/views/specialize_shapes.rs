use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Dim, Module, TensorType, Type};
use indexmap::IndexMap;

/// Symbolic & Dynamic Shape Specialization & Inference Pass (`specialize-shapes`).
///
/// Inspired by Modular MAX Engine's symbolic shape specialization (`reify-shapes`):
/// - Binds symbolic dimensions (`tensor<BxSx128xf32>`) and dynamic dimensions
///   (`tensor<?x128xf32>`) using user-supplied symbol bindings (e.g. `B=4,S=128,default=1`).
/// - Propagates static shapes forward through `stablehlo` operations (`dot_general`, `transpose`,
///   `reshape` with a single `?` dimension, `reduce`, `concatenate`, `slice`, `composite`, and
///   elementwise ops).
/// - Updates `func.func` function signatures so downstream Arena Memory Planning (`MemoryPlan`),
///   Roofline analysis, and target codegen operate on fully static shapes.
pub struct SpecializeShapesPass {
    pub default_dim: i64,
    pub bindings: IndexMap<String, i64>,
}

impl Default for SpecializeShapesPass {
    fn default() -> Self {
        Self {
            default_dim: 1,
            bindings: IndexMap::new(),
        }
    }
}

impl SpecializeShapesPass {
    fn resolve_dim(&self, dim: &Dim) -> (Dim, bool) {
        match dim {
            Dim::Static(n) => (Dim::Static(*n), false),
            Dim::Dynamic => (Dim::Static(self.default_dim), true),
            Dim::Symbolic(sym) => {
                let val = self.bindings.get(sym).copied().unwrap_or(self.default_dim);
                (Dim::Static(val), true)
            }
        }
    }

    fn resolve_tensor_type(&self, t: &TensorType) -> (TensorType, bool) {
        let mut changed = false;
        let shape = t
            .shape
            .iter()
            .map(|d| {
                let (nd, c) = self.resolve_dim(d);
                changed |= c;
                nd
            })
            .collect();
        (
            TensorType {
                shape,
                dtype: t.dtype.clone(),
                encoding: t.encoding.clone(),
            },
            changed,
        )
    }
}

impl Pass for SpecializeShapesPass {
    fn name(&self) -> &'static str {
        "specialize-shapes"
    }

    fn description(&self) -> &'static str {
        "Binds dynamic/symbolic dimensions (?, B, S) and propagates static shapes across the IR"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        // 1. Specialize all block arguments first
        for &block_id in &blocks {
            let args = module.blocks[block_id].arguments.clone();
            for arg_val in args {
                if let Some(t) = module.values[arg_val].ty.as_tensor() {
                    let (new_t, ch) = self.resolve_tensor_type(t);
                    if ch {
                        module.values[arg_val].ty = Type::Tensor(new_t);
                        modified += 1;
                    }
                }
            }
        }

        // 2. Forward shape inference & specialization across all operations in each block
        for &block_id in &blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(&res_val) = op.results.first() else {
                    continue;
                };
                let Some(cur_res_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                    continue;
                };

                let mut inferred_shape: Option<Vec<Dim>> = None;

                if let Some(hlo) = StableHloOp::from_op(&op) {
                    match hlo {
                        StableHloOp::Add { lhs, .. }
                        | StableHloOp::Subtract { lhs, .. }
                        | StableHloOp::Multiply { lhs, .. }
                        | StableHloOp::Divide { lhs, .. }
                        | StableHloOp::Maximum { lhs, .. }
                        | StableHloOp::Minimum { lhs, .. }
                        | StableHloOp::Exp { operand: lhs, .. }
                        | StableHloOp::Tanh { operand: lhs, .. }
                        | StableHloOp::Rsqrt { operand: lhs, .. }
                        | StableHloOp::Convert { operand: lhs, .. } => {
                            if let Some(in_t) = module.values[lhs].ty.as_tensor() {
                                inferred_shape = Some(in_t.shape.clone());
                            }
                        }
                        StableHloOp::Select { on_true, .. } => {
                            if let Some(in_t) = module.values[on_true].ty.as_tensor() {
                                inferred_shape = Some(in_t.shape.clone());
                            }
                        }
                        StableHloOp::Transpose {
                            operand,
                            permutation,
                            ..
                        } => {
                            if let Some(in_t) = module.values[operand].ty.as_tensor()
                                && permutation.iter().all(|&p| (p as usize) < in_t.shape.len())
                            {
                                inferred_shape = Some(
                                    permutation
                                        .iter()
                                        .map(|&p| in_t.shape[p as usize].clone())
                                        .collect(),
                                );
                            }
                        }
                        StableHloOp::Reshape { operand, .. } => {
                            if let Some(in_t) = module.values[operand].ty.as_tensor()
                                && let Some(total_elems) = in_t.num_elements()
                            {
                                // If target shape has exactly one non-static dim, infer it!
                                let non_static_count = cur_res_ty
                                    .shape
                                    .iter()
                                    .filter(|d| !matches!(d, Dim::Static(_)))
                                    .count();
                                if non_static_count == 1 {
                                    let known_prod: i64 = cur_res_ty
                                        .shape
                                        .iter()
                                        .filter_map(|d| match d {
                                            Dim::Static(n) => Some(*n),
                                            _ => None,
                                        })
                                        .product();
                                    if known_prod > 0 && total_elems % known_prod == 0 {
                                        let missing = total_elems / known_prod;
                                        inferred_shape = Some(
                                            cur_res_ty
                                                .shape
                                                .iter()
                                                .map(|d| match d {
                                                    Dim::Static(n) => Dim::Static(*n),
                                                    _ => Dim::Static(missing),
                                                })
                                                .collect(),
                                        );
                                    }
                                }
                            }
                        }
                        StableHloOp::DotGeneral {
                            lhs,
                            rhs,
                            dimension_numbers: dims,
                            ..
                        } => {
                            if let (Some(lt), Some(rt)) = (
                                module.values[lhs].ty.as_tensor(),
                                module.values[rhs].ty.as_tensor(),
                            ) && lt.shape.len() == 2
                                && rt.shape.len() == 2
                            {
                                let m_dim = lt.shape[0].clone();
                                let n_dim = if dims.rhs_contracting_dimensions == [1] {
                                    rt.shape[0].clone()
                                } else {
                                    rt.shape[1].clone()
                                };
                                inferred_shape = Some(vec![m_dim, n_dim]);
                            }
                        }
                        StableHloOp::Reduce {
                            inputs, dimensions, ..
                        } => {
                            if let Some(&first_in) = inputs.first()
                                && let Some(in_t) = module.values[first_in].ty.as_tensor()
                            {
                                let kept: Vec<Dim> = in_t
                                    .shape
                                    .iter()
                                    .enumerate()
                                    .filter(|(idx, _)| !dimensions.contains(&(*idx as i64)))
                                    .map(|(_, d)| d.clone())
                                    .collect();
                                inferred_shape = Some(kept);
                            }
                        }
                        StableHloOp::Concatenate {
                            inputs, dimension, ..
                        } => {
                            let d = dimension as usize;
                            if let Some(&first_in) = inputs.first()
                                && let Some(first_t) = module.values[first_in].ty.as_tensor()
                                && d < first_t.shape.len()
                            {
                                let mut cat_sum = 0i64;
                                let mut all_static = true;
                                for &inp in &inputs {
                                    if let Some(t) = module.values[inp].ty.as_tensor()
                                        && let Some(Dim::Static(n)) = t.shape.get(d)
                                    {
                                        cat_sum += *n;
                                    } else {
                                        all_static = false;
                                        break;
                                    }
                                }
                                if all_static {
                                    let mut new_shape = first_t.shape.clone();
                                    new_shape[d] = Dim::Static(cat_sum);
                                    inferred_shape = Some(new_shape);
                                }
                            }
                        }
                        StableHloOp::Slice {
                            start_indices,
                            limit_indices,
                            strides,
                            ..
                        } => {
                            if start_indices.len() == limit_indices.len() {
                                let mut s_shape = Vec::with_capacity(start_indices.len());
                                for (i, &start) in start_indices.iter().enumerate() {
                                    let limit = limit_indices[i];
                                    let stride = strides.get(i).copied().unwrap_or(1).max(1);
                                    s_shape
                                        .push(Dim::Static((limit - start + stride - 1) / stride));
                                }
                                inferred_shape = Some(s_shape);
                            }
                        }
                        StableHloOp::Composite { name, inputs, .. } => {
                            if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 2 {
                                if let (Some(lt), Some(rt)) = (
                                    module.values[inputs[0]].ty.as_tensor(),
                                    module.values[inputs[1]].ty.as_tensor(),
                                ) && lt.shape.len() == 2
                                    && rt.shape.len() == 2
                                {
                                    inferred_shape =
                                        Some(vec![lt.shape[0].clone(), rt.shape[1].clone()]);
                                }
                            } else if let Some(&first_in) = inputs.first()
                                && let Some(in_t) = module.values[first_in].ty.as_tensor()
                            {
                                inferred_shape = Some(in_t.shape.clone());
                            }
                        }
                        _ => {}
                    }
                }

                let candidate_ty = TensorType {
                    shape: inferred_shape.unwrap_or_else(|| cur_res_ty.shape.clone()),
                    dtype: cur_res_ty.dtype.clone(),
                    encoding: cur_res_ty.encoding.clone(),
                };
                let (resolved_ty, _) = self.resolve_tensor_type(&candidate_ty);
                if resolved_ty != cur_res_ty {
                    module.values[res_val].ty = Type::Tensor(resolved_ty);
                    modified += 1;
                }
            }
        }

        // 3. Synchronize `func.func` `function_type` attribute with specialized args & returns
        for fn_op_id in module.functions() {
            let Some(reg_id) = module.ops[fn_op_id].regions.first().copied() else {
                continue;
            };
            let Some(entry_blk) = module.regions[reg_id].blocks.first().copied() else {
                continue;
            };
            let new_inputs: Vec<Type> = module.blocks[entry_blk]
                .arguments
                .iter()
                .map(|&v| module.values[v].ty.clone())
                .collect();

            let mut new_results: Option<Vec<Type>> = None;
            for &op_id in &module.blocks[entry_blk].ops {
                let op = &module.ops[op_id];
                if op.is_op("func", "return") || op.is_op("stablehlo", "return") {
                    new_results = Some(
                        op.operands
                            .iter()
                            .map(|&v| module.values[v].ty.clone())
                            .collect(),
                    );
                }
            }

            if let Some(Attribute::Type(Type::Function { inputs, results })) = module.ops[fn_op_id]
                .attributes
                .get("function_type")
                .cloned()
            {
                let res_vec = new_results.unwrap_or(results.clone());
                if inputs != new_inputs || results != res_vec {
                    module.ops[fn_op_id].attributes.insert(
                        "function_type".to_string(),
                        Attribute::Type(Type::Function {
                            inputs: new_inputs,
                            results: res_vec,
                        }),
                    );
                    modified += 1;
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
