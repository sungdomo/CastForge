//! Symbolic & Dynamic Shape Specialization, Batch Resizing, and Shape Bucketing
//! (`specialize-shapes`).

use crate::ir::{Attribute, BlockId, Dim, Module, OpName, TensorType, Type, ValueId};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// Unified Shape Specialization, Batch Resizing, Bucketing & Token Pruning (`specialize-shapes`).
///
/// Consolidates `specialize-shapes`, `resize-batch`, `bucket-shapes`, and dynamic sequence token
/// pruning/merging (`token_prune_ratio`).
pub struct UnifiedSpecializeShapesPass {
    pub default_dim: i64,
    pub bindings: IndexMap<String, i64>,
    pub batch: Option<i64>,
    pub buckets: Option<String>,
    pub token_prune_ratio: Option<f32>,
}

impl Default for UnifiedSpecializeShapesPass {
    fn default() -> Self {
        Self {
            default_dim: 1,
            bindings: IndexMap::new(),
            batch: None,
            buckets: None,
            token_prune_ratio: None,
        }
    }
}

impl Pass for UnifiedSpecializeShapesPass {
    fn name(&self) -> &'static str {
        "specialize-shapes"
    }

    fn description(&self) -> &'static str {
        "Unified shape specialization, static batch resizing, bucketing, and token pruning"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let spec_helper = SpecializeShapesPass {
            default_dim: self.default_dim,
            bindings: self.bindings.clone(),
        };
        let mut modified = specialize_and_resize_single_traversal(
            module,
            &spec_helper,
            true,
            self.batch,
            self.token_prune_ratio,
        );
        let mut created = 0usize;
        if let Some(ref buckets) = self.buckets {
            let b_res = BucketShapesPass {
                buckets: buckets.clone(),
            }
            .run(module)?;
            created += b_res.ops_created;
            modified += b_res.ops_modified;
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || created > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn specialize_and_resize_single_traversal(
    module: &mut Module,
    spec: &SpecializeShapesPass,
    do_specialize: bool,
    target_batch: Option<i64>,
    token_prune_ratio: Option<f32>,
) -> usize {
    let mut modified = 0usize;
    let new_batch = target_batch.map(|b| b.max(1));
    let prune_r = token_prune_ratio.map(|r| r.clamp(0.05, 0.95));

    let blocks: Vec<BlockId> = module.blocks.keys().collect();
    for &block_id in &blocks {
        let args = module.blocks[block_id].arguments.clone();
        let ops = module.block_ops(block_id);

        let mut weight_rhs_vals: HashSet<ValueId> = HashSet::new();
        if new_batch.is_some() {
            for &op_id in &ops {
                if let Some(op) = module.ops.get(op_id)
                    && op.is_op("stablehlo", "dot_general")
                    && op.operands.len() >= 2
                {
                    weight_rhs_vals.insert(op.operands[1]);
                }
            }
        }

        if do_specialize {
            for &arg_val in &args {
                if let Some(t) = module.values[arg_val].ty.as_tensor() {
                    let (new_t, ch) = spec.resolve_tensor_type(t);
                    if ch {
                        module.values[arg_val].ty = Type::Tensor(new_t);
                        modified += 1;
                    }
                }
            }
        }

        let old_batch = if let Some(nb) = new_batch {
            args.first()
                .and_then(|&fa| leading_static_dim(module, fa))
                .filter(|&ob| ob != nb)
        } else {
            None
        };

        if let (Some(ob), Some(nb)) = (old_batch, new_batch) {
            for &arg_id in &args {
                if !weight_rhs_vals.contains(&arg_id) && set_leading_dim(module, arg_id, ob, nb) {
                    modified += 1;
                }
            }
        }

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };

            if do_specialize
                && let Some(&res_val) = op.results.first()
                && let Some(cur_res_ty) = module.values[res_val].ty.as_tensor().cloned()
            {
                let inferred_shape = infer_op_result_shape(module, &op, &cur_res_ty);
                let candidate_ty = TensorType {
                    shape: inferred_shape.unwrap_or_else(|| cur_res_ty.shape.clone()),
                    dtype: cur_res_ty.dtype.clone(),
                    encoding: cur_res_ty.encoding.clone(),
                };
                let (resolved_ty, _) = spec.resolve_tensor_type(&candidate_ty);
                if resolved_ty != cur_res_ty {
                    module.values[res_val].ty = Type::Tensor(resolved_ty);
                    modified += 1;
                }
            }

            if let (Some(ob), Some(nb)) = (old_batch, new_batch)
                && !op.is_op("func", "return")
            {
                for &res_id in &op.results {
                    if !weight_rhs_vals.contains(&res_id) && set_leading_dim(module, res_id, ob, nb)
                    {
                        modified += 1;
                        if let Some(Attribute::DenseElements { ty, .. }) =
                            module.ops[op_id].attributes.get_mut("value")
                            && !ty.shape.is_empty()
                        {
                            ty.shape[0] = Dim::Static(nb);
                        }
                    }
                }
            }

            if let Some(r) = prune_r {
                let is_target = op.is_op("stablehlo", "dot_general")
                    || (op.is_op("stablehlo", "composite")
                        && op
                            .attributes
                            .get("name")
                            .and_then(|a| a.as_str())
                            .is_some_and(|n| n.contains("attention") || n.contains("matmul")));
                if is_target {
                    let seq_len = op
                        .operands
                        .first()
                        .and_then(|&v| module.values[v].ty.as_tensor())
                        .and_then(|t| match t.shape.first() {
                            Some(Dim::Static(d)) => Some(*d),
                            _ => None,
                        })
                        .unwrap_or(1);
                    let kept = ((seq_len as f32) * (1.0 - r)).ceil().max(1.0) as i64;
                    let op_mut = &mut module.ops[op_id];
                    op_mut.attributes.insert(
                        "castforge.token_prune_ratio".to_string(),
                        Attribute::Float(r as f64),
                    );
                    op_mut.attributes.insert(
                        "castforge.token_merging".to_string(),
                        Attribute::String("tome_bipartite".to_string()),
                    );
                    op_mut
                        .attributes
                        .insert("castforge.kept_tokens".to_string(), Attribute::Int(kept));
                    modified += 1;
                }
            }
        }
    }

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

    modified
}

/// Symbolic & Dynamic Shape Specialization & Inference Pass (`specialize-shapes`).
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
        let modified = specialize_and_resize_single_traversal(module, self, true, None, None);
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn infer_op_result_shape(
    module: &Module,
    op: &crate::ir::Operation,
    cur_res_ty: &TensorType,
) -> Option<Vec<Dim>> {
    let mut inferred_shape: Option<Vec<Dim>> = None;

    if let Some(hlo) = StableHloOp::from_op(op) {
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
                        s_shape.push(Dim::Static((limit - start + stride - 1) / stride));
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
                        inferred_shape = Some(vec![lt.shape[0].clone(), rt.shape[1].clone()]);
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

    inferred_shape
}

/// Static Batch Dimension Resizer Pass (`resize-batch`).
#[derive(Debug, Clone)]
pub struct ResizeBatchPass {
    /// Target leading batch size (`>= 1`, default `4`).
    pub batch: i64,
}

impl Default for ResizeBatchPass {
    fn default() -> Self {
        Self { batch: 4 }
    }
}

impl Pass for ResizeBatchPass {
    fn name(&self) -> &'static str {
        "resize-batch"
    }

    fn description(&self) -> &'static str {
        "Rewrites the leading batch dimension across activations while preserving 2D GEMM weights"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let dummy = SpecializeShapesPass::default();
        let modified =
            specialize_and_resize_single_traversal(module, &dummy, false, Some(self.batch), None);
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn leading_static_dim(module: &Module, vid: ValueId) -> Option<i64> {
    module.values[vid]
        .ty
        .as_tensor()
        .and_then(|t| match t.shape.first() {
            Some(Dim::Static(d)) => Some(*d),
            _ => None,
        })
}

fn set_leading_dim(module: &mut Module, vid: ValueId, old_batch: i64, new_batch: i64) -> bool {
    if let Type::Tensor(ref mut t) = module.values[vid].ty
        && !t.shape.is_empty()
        && t.shape[0] == Dim::Static(old_batch)
    {
        t.shape[0] = Dim::Static(new_batch);
        return true;
    }
    false
}

/// Multi-Bucket Shape Cloner Pass (`bucket-shapes`).
#[derive(Debug, Clone)]
pub struct BucketShapesPass {
    /// Semicolon/comma-separated batch sizes to specialize (default `"1;4;8"`).
    pub buckets: String,
}

impl Default for BucketShapesPass {
    fn default() -> Self {
        Self {
            buckets: "1;4;8".to_string(),
        }
    }
}

impl Pass for BucketShapesPass {
    fn name(&self) -> &'static str {
        "bucket-shapes"
    }

    fn description(&self) -> &'static str {
        "Clones the entry function into batch-specialized bucket variants (@fn_batch_B)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&src_fn_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let batch_sizes: Vec<i64> = self
            .buckets
            .split([',', ';'])
            .filter_map(|s| s.trim().parse::<i64>().ok())
            .filter(|&b| b >= 1)
            .collect();
        if batch_sizes.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let src_sym = module.ops[src_fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("main")
            .to_string();
        let src_reg = module.ops[src_fn_id].regions[0];
        let src_blk = module.regions[src_reg].blocks[0];
        let src_args = module.blocks[src_blk].arguments.clone();
        let src_ops = module.blocks[src_blk].ops.clone();

        let old_batch = src_args
            .first()
            .and_then(|&a| module.values[a].ty.as_tensor())
            .and_then(|t| match t.shape.first() {
                Some(Dim::Static(d)) => Some(*d),
                _ => None,
            })
            .unwrap_or(1);

        let mut weight_rhs_vals: HashSet<ValueId> = HashSet::new();
        for &op_id in &src_ops {
            let op = &module.ops[op_id];
            if op.is_op("stablehlo", "dot_general") && op.operands.len() >= 2 {
                weight_rhs_vals.insert(op.operands[1]);
            }
        }

        let mod_body = module.body_block();
        let mut created_fns = 0usize;

        for &b_size in &batch_sizes {
            let dst_blk = module.create_block(None);
            let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
            let mut in_types = Vec::with_capacity(src_args.len());

            for &arg_id in &src_args {
                let ssa = module.values[arg_id].ssa_name.clone();
                let mut ty = module.values[arg_id].ty.clone();
                if !weight_rhs_vals.contains(&arg_id) {
                    resize_type_batch(&mut ty, old_batch, b_size);
                }
                in_types.push(ty.clone());
                let new_arg = module.add_block_argument(dst_blk, ssa, ty);
                val_map.insert(arg_id, new_arg);
            }

            let mut out_types = Vec::new();
            for &op_id in &src_ops {
                let op = module.ops[op_id].clone();
                let mapped_operands: SmallVec<[ValueId; 4]> = op
                    .operands
                    .iter()
                    .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                    .collect();

                if op.is_op("func", "return") {
                    for &opnd in &mapped_operands {
                        out_types.push(module.values[opnd].ty.clone());
                    }
                }

                let mut res_specs = Vec::with_capacity(op.results.len());
                for &old_r in &op.results {
                    let ssa = module.values[old_r].ssa_name.clone();
                    let mut r_ty = module.values[old_r].ty.clone();
                    if !weight_rhs_vals.contains(&old_r) {
                        resize_type_batch(&mut r_ty, old_batch, b_size);
                    }
                    res_specs.push((ssa, r_ty));
                }

                let mut attrs = op.attributes.clone();
                if let Some(Attribute::DenseElements { ty, .. }) = attrs.get_mut("value")
                    && op
                        .results
                        .first()
                        .is_some_and(|r| !weight_rhs_vals.contains(r))
                    && !ty.shape.is_empty()
                    && ty.shape[0] == Dim::Static(old_batch)
                {
                    ty.shape[0] = Dim::Static(b_size);
                }

                let new_op_id = module.add_operation(
                    dst_blk,
                    op.name,
                    mapped_operands,
                    res_specs,
                    attrs,
                    Vec::new(),
                    op.custom_syntax,
                    op.location,
                );
                for (&old_r, &new_r) in op.results.iter().zip(module.ops[new_op_id].results.iter())
                {
                    val_map.insert(old_r, new_r);
                }
            }

            let dst_reg = module.create_region(vec![dst_blk]);
            let mut fn_attrs = module.ops[src_fn_id].attributes.clone();
            fn_attrs.insert(
                "sym_name".to_string(),
                Attribute::String(format!("{src_sym}_batch_{b_size}")),
            );
            fn_attrs.insert(
                "function_type".to_string(),
                Attribute::Type(Type::Function {
                    inputs: in_types,
                    results: out_types,
                }),
            );
            let loc = module.ops[src_fn_id].location.clone();
            module.add_operation(
                mod_body,
                OpName::new("func", "func"),
                SmallVec::new(),
                Vec::new(),
                fn_attrs,
                vec![dst_reg],
                true,
                loc,
            );
            created_fns += 1;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created_fns > 0,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: created_fns,
        })
    }
}

fn resize_type_batch(ty: &mut Type, old_batch: i64, new_batch: i64) {
    if let Type::Tensor(ref mut t) = *ty
        && !t.shape.is_empty()
        && (t.shape[0] == Dim::Static(old_batch)
            || matches!(t.shape[0], Dim::Dynamic | Dim::Symbolic(_)))
    {
        t.shape[0] = Dim::Static(new_batch);
    }
}
