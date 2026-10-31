//! Symbolic & Dynamic Shape Specialization, Batch Resizing, and Shape Bucketing
//! (`specialize-shapes`).

pub mod bucketing;

pub use bucketing::{BucketShapesPass, ResizeBatchPass};

use crate::ir::{Attribute, BlockId, Dim, Module, TensorType, Type, ValueId};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use std::collections::HashSet;

/// Unified Shape Specialization, Batch Resizing, Bucketing & Token Pruning (`specialize-shapes`).
///
/// Consolidates `specialize-shapes`, `resize-batch`, `bucket-shapes`, and dynamic sequence token
/// pruning/merging (`token_prune_ratio`).
pub struct SpecializeShapesPass {
    pub default_dim: i64,
    pub bindings: IndexMap<String, i64>,
    pub batch: Option<i64>,
    pub buckets: Option<String>,
    pub token_prune_ratio: Option<f32>,
}

impl Default for SpecializeShapesPass {
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

impl Pass for SpecializeShapesPass {
    fn name(&self) -> &'static str {
        "specialize-shapes"
    }

    fn description(&self) -> &'static str {
        "Unified shape specialization, static batch resizing, bucketing, and token pruning"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &["resize-batch", "bucket-shapes", "prune-tokens"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let modified = specialize_and_resize_single_traversal(module, self, true, None, None);
        let mut steps = vec![PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        }];
        if let Some(ratio) = self.token_prune_ratio {
            steps.push(PruneTokensPass { ratio }.run(module)?);
        }
        if let Some(batch) = self.batch {
            steps.push(ResizeBatchPass { batch }.run(module)?);
        }
        if let Some(ref buckets) = self.buckets {
            steps.push(
                BucketShapesPass {
                    buckets: buckets.clone(),
                }
                .run(module)?,
            );
        }
        Ok(PassResult::aggregate(self.name(), &steps))
    }
}

/// ToMe / FastV Dynamic Sequence Token Pruning Pass (`prune-tokens`).
pub struct PruneTokensPass {
    pub ratio: f32,
}

impl Default for PruneTokensPass {
    fn default() -> Self {
        Self { ratio: 0.5 }
    }
}

impl Pass for PruneTokensPass {
    fn name(&self) -> &'static str {
        "prune-tokens"
    }

    fn description(&self) -> &'static str {
        "Applies ToMe/FastV dynamic sequence token pruning and bipartite token merging"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let dummy = SpecializeShapesPass::default();
        let modified =
            specialize_and_resize_single_traversal(module, &dummy, false, None, Some(self.ratio));
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

pub(crate) fn specialize_and_resize_single_traversal(
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

        let weight_rhs_vals: HashSet<ValueId> = if new_batch.is_some() {
            collect_non_batch_weight_operands(module, &ops)
        } else {
            HashSet::new()
        };

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
                if matches!(
                    name.as_str(),
                    "castforge.fused_matmul_epilogue" | "castforge.fused_qmatmul"
                ) && inputs.len() >= 2
                {
                    if let (Some(lt), Some(rt)) = (
                        module.values[inputs[0]].ty.as_tensor(),
                        module.values[inputs[1]].ty.as_tensor(),
                    ) && lt.shape.len() == 2
                        && rt.shape.len() == 2
                    {
                        let packed = op
                            .attributes
                            .get("packed_rhs")
                            .is_some_and(|a| matches!(a, Attribute::Bool(true)))
                            || op
                                .attributes
                                .get("rhs_contracting_dimensions")
                                .and_then(|a| a.as_int_slice())
                                .is_some_and(|v| v == [1]);
                        let n_dim = if packed {
                            rt.shape[0].clone()
                        } else {
                            rt.shape[1].clone()
                        };
                        inferred_shape = Some(vec![lt.shape[0].clone(), n_dim]);
                    }
                } else if name == "castforge.fused_gated_mlp" && inputs.len() >= 4 {
                    if let (Some(lt), Some(dt)) = (
                        module.values[inputs[0]].ty.as_tensor(),
                        module.values[inputs[3]].ty.as_tensor(),
                    ) && lt.shape.len() == 2
                        && dt.shape.len() == 2
                    {
                        inferred_shape = Some(vec![lt.shape[0].clone(), dt.shape[1].clone()]);
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

pub(crate) fn collect_non_batch_weight_operands(
    module: &Module,
    ops: &[crate::ir::OpId],
) -> HashSet<ValueId> {
    let mut weight_rhs_vals = HashSet::new();
    for &op_id in ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if (op.is_op("stablehlo", "dot_general") || op.is_op("stablehlo", "convolution"))
            && op.operands.len() >= 2
        {
            weight_rhs_vals.insert(op.operands[1]);
        } else if op.is_op("stablehlo", "composite") {
            let comp_name = op
                .attributes
                .get("name")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            match comp_name {
                "castforge.fused_matmul_epilogue" | "castforge.fused_qmatmul" => {
                    let has_res = op
                        .attributes
                        .get("epilogue")
                        .and_then(|a| a.as_str())
                        .is_some_and(|s| s.ends_with("_residual"));
                    let end = if has_res {
                        op.operands.len().saturating_sub(1)
                    } else {
                        op.operands.len()
                    };
                    for &w in op.operands.get(1..end).unwrap_or(&[]) {
                        weight_rhs_vals.insert(w);
                    }
                }
                "castforge.fused_gated_mlp"
                | "castforge.fused_rmsnorm"
                | "castforge.fused_layernorm" => {
                    for &w in op.operands.get(1..).unwrap_or(&[]) {
                        weight_rhs_vals.insert(w);
                    }
                }
                "castforge.fused_residual_rmsnorm" => {
                    for &w in op.operands.get(2..).unwrap_or(&[]) {
                        weight_rhs_vals.insert(w);
                    }
                }
                "castforge.fused_moe" => {
                    for &w in op.operands.get(1..3).unwrap_or(&[]) {
                        weight_rhs_vals.insert(w);
                    }
                }
                _ => {}
            }
        }
    }
    weight_rhs_vals
}
