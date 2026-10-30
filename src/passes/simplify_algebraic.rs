//! Unified Algebraic Simplification, Constant Folding, Conditional Propagation (SCCP),
//! and E-Graph Factoring/FMA Pass (`simplify-algebraic`), executed in a single forward
//! dataflow traversal.

use crate::ir::{
    Attribute, BlockId, Dim, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
    format_dense_f32_literal,
};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashSet;

/// Unified Algebraic Simplification, Constant Folding & E-Graph Factoring/FMA Pass
/// (`simplify-algebraic`).
pub struct SimplifyAlgebraicPass {
    pub fold_constants: bool,
    pub include_conditional: bool,
    pub egraph: bool,
    pub fuse_fma: bool,
}

impl Default for SimplifyAlgebraicPass {
    fn default() -> Self {
        Self {
            fold_constants: true,
            include_conditional: true,
            egraph: false,
            fuse_fma: true,
        }
    }
}

impl Pass for SimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Unified constant folding, SCCP, algebraic identities, e-graph factoring & FMA extraction"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut dead_ops: HashSet<OpId> = HashSet::new();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if StableHloOp::from_op(op).is_none() {
                    continue;
                }

                // 1. Constant folding & SCCP (`compare` / `select` + arithmetic/views)
                if self.fold_constants {
                    match try_fold_constant_op(module, op_id, self.include_conditional) {
                        ConstFoldOutcome::Modified => {
                            modified += 1;
                            continue;
                        }
                        ConstFoldOutcome::Replaced => {
                            dead_ops.insert(op_id);
                            removed += 1;
                            continue;
                        }
                        ConstFoldOutcome::Unchanged => {}
                    }
                }

                // 2. Algebraic identities, commutative canonicalization, and reassociation
                match try_simplify_algebraic_op(module, op_id) {
                    AlgOutcome::ModifiedAndDone => {
                        modified += 1;
                        continue;
                    }
                    AlgOutcome::Replaced(swapped_before) => {
                        if swapped_before {
                            modified += 1;
                        }
                        dead_ops.insert(op_id);
                        removed += 1;
                        continue;
                    }
                    AlgOutcome::SwappedOnly => {
                        modified += 1;
                    }
                    AlgOutcome::Unchanged => {}
                }

                // 3. E-Graph distributive factoring (`a*b +/- a*c -> a*(b +/- c)`) & FMA
                if self.egraph {
                    let (c, m) = try_egraph_and_fma_op(module, block_id, op_id, self.fuse_fma);
                    created += c;
                    modified += m;
                }
            }

            if !dead_ops.is_empty() {
                module.blocks[block_id]
                    .ops
                    .retain(|id| !dead_ops.contains(id));
                for dead_id in dead_ops {
                    module.ops.remove(dead_id);
                }
            }
        }

        if created > 0 || modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// Constant query utilities used across passes (`const_splat` and `const_f32_vec`).
pub struct FoldConstantsPass;

impl FoldConstantsPass {
    pub fn const_splat(module: &Module, val_id: ValueId) -> Option<f64> {
        if let ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                let attr = op_node.attributes.get("value")?;
                if let Some(v) = attr.as_splat_f64() {
                    return Some(v);
                }
                let n = module.values[val_id]
                    .ty
                    .as_tensor()
                    .and_then(|t| t.num_elements())
                    .unwrap_or(1) as usize;
                let vals = module.tensor_store.resolve_f32_attr(attr, n)?;
                let first = *vals.first()?;
                if vals.iter().all(|&x| (x - first).abs() < 1e-7) {
                    return Some(first as f64);
                }
            }
        }
        None
    }

    pub fn const_f32_vec(module: &Module, val_id: ValueId) -> Option<Vec<f32>> {
        if let ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                let n = module.values[val_id]
                    .ty
                    .as_tensor()
                    .and_then(|t| t.num_elements())
                    .unwrap_or(1) as usize;
                let attr = op_node.attributes.get("value")?;
                return module.tensor_store.resolve_f32_attr(attr, n);
            }
        }
        None
    }
}

enum ConstFoldOutcome {
    Unchanged,
    Modified,
    Replaced,
}

fn try_fold_constant_op(
    module: &mut Module,
    op_id: OpId,
    include_conditional: bool,
) -> ConstFoldOutcome {
    let Some(op) = module.ops.get(op_id) else {
        return ConstFoldOutcome::Unchanged;
    };
    let Some(hlo) = StableHloOp::from_op(op) else {
        return ConstFoldOutcome::Unchanged;
    };

    if include_conditional {
        match &hlo {
            StableHloOp::Compare {
                lhs,
                rhs,
                direction,
                result,
            } => {
                if let (Some(a), Some(b)) = (
                    FoldConstantsPass::const_splat(module, *lhs),
                    FoldConstantsPass::const_splat(module, *rhs),
                ) {
                    let cond = match direction.as_str() {
                        "EQ" => (a - b).abs() < 1e-7,
                        "NE" => (a - b).abs() >= 1e-7,
                        "LT" => a < b,
                        "LE" => a <= b,
                        "GT" => a > b,
                        "GE" => a >= b,
                        _ => false,
                    };
                    if let Some(res_ty) = module.values[*result].ty.as_tensor().cloned() {
                        let lit = if cond { "1" } else { "0" };
                        let mut attrs = IndexMap::new();
                        attrs.insert(
                            "value".to_string(),
                            Attribute::DenseElements {
                                ty: res_ty,
                                raw_literal: lit.to_string(),
                            },
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "constant");
                        op_mut.operands.clear();
                        op_mut.attributes = attrs;
                        return ConstFoldOutcome::Modified;
                    }
                }
            }
            StableHloOp::Select {
                pred,
                on_true,
                on_false,
                result,
            } => {
                let chosen = if on_true == on_false {
                    Some(*on_true)
                } else {
                    FoldConstantsPass::const_splat(module, *pred)
                        .map(|v| if v != 0.0 { *on_true } else { *on_false })
                };
                if let Some(winner) = chosen {
                    module.replace_all_uses(*result, winner);
                    return ConstFoldOutcome::Replaced;
                }
            }
            _ => {}
        }
    }

    let folded_vec: Option<(ValueId, Vec<f32>)> = match hlo {
        StableHloOp::Add { lhs, rhs, result } => {
            match (
                FoldConstantsPass::const_f32_vec(module, lhs),
                FoldConstantsPass::const_f32_vec(module, rhs),
            ) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    Some((result, a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()))
                }
                _ => None,
            }
        }
        StableHloOp::Subtract { lhs, rhs, result } => {
            match (
                FoldConstantsPass::const_f32_vec(module, lhs),
                FoldConstantsPass::const_f32_vec(module, rhs),
            ) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    Some((result, a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()))
                }
                _ => None,
            }
        }
        StableHloOp::Multiply { lhs, rhs, result } => {
            match (
                FoldConstantsPass::const_f32_vec(module, lhs),
                FoldConstantsPass::const_f32_vec(module, rhs),
            ) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    Some((result, a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()))
                }
                _ => None,
            }
        }
        StableHloOp::Divide { lhs, rhs, result } => {
            match (
                FoldConstantsPass::const_f32_vec(module, lhs),
                FoldConstantsPass::const_f32_vec(module, rhs),
            ) {
                (Some(a), Some(b)) if a.len() == b.len() && b.iter().all(|&y| y.abs() > 1e-12) => {
                    Some((result, a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()))
                }
                _ => None,
            }
        }
        StableHloOp::Maximum { lhs, rhs, result } => {
            match (
                FoldConstantsPass::const_f32_vec(module, lhs),
                FoldConstantsPass::const_f32_vec(module, rhs),
            ) {
                (Some(a), Some(b)) if a.len() == b.len() => Some((
                    result,
                    a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect(),
                )),
                _ => None,
            }
        }
        StableHloOp::Minimum { lhs, rhs, result } => {
            match (
                FoldConstantsPass::const_f32_vec(module, lhs),
                FoldConstantsPass::const_f32_vec(module, rhs),
            ) {
                (Some(a), Some(b)) if a.len() == b.len() => Some((
                    result,
                    a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect(),
                )),
                _ => None,
            }
        }
        StableHloOp::Rsqrt { operand, result } => FoldConstantsPass::const_f32_vec(module, operand)
            .map(|v| (result, v.into_iter().map(|x| 1.0 / x.sqrt()).collect())),
        StableHloOp::Exp { operand, result } => FoldConstantsPass::const_f32_vec(module, operand)
            .map(|v| (result, v.into_iter().map(|x| x.exp()).collect())),
        StableHloOp::Tanh { operand, result } => FoldConstantsPass::const_f32_vec(module, operand)
            .map(|v| (result, v.into_iter().map(|x| x.tanh()).collect())),
        StableHloOp::Convert { operand, result } | StableHloOp::Reshape { operand, result } => {
            FoldConstantsPass::const_f32_vec(module, operand).map(|v| (result, v))
        }
        StableHloOp::Transpose {
            operand,
            permutation,
            result,
        } => {
            let in_ty = module.values[operand].ty.as_tensor().cloned();
            let out_ty = module.values[result].ty.as_tensor().cloned();
            match (
                in_ty,
                out_ty,
                FoldConstantsPass::const_f32_vec(module, operand),
            ) {
                (Some(in_t), Some(out_t), Some(in_vals)) => {
                    transpose_f32_nd(&in_vals, &in_t.shape, &out_t.shape, &permutation)
                        .map(|vals| (result, vals))
                }
                _ => None,
            }
        }
        StableHloOp::BroadcastInDim {
            operand, result, ..
        } => FoldConstantsPass::const_splat(module, operand).map(|v| {
            let n = module.values[result]
                .ty
                .as_tensor()
                .and_then(|t| t.num_elements())
                .unwrap_or(1) as usize;
            (result, vec![v as f32; n])
        }),
        _ => None,
    };

    if let Some((res_val, vals)) = folded_vec
        && let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned()
    {
        let raw_literal = format_dense_f32_literal(&vals, &tensor_ty.shape);
        let mut attrs = IndexMap::new();
        attrs.insert(
            "value".to_string(),
            Attribute::DenseElements {
                ty: tensor_ty,
                raw_literal,
            },
        );
        let op_mut = &mut module.ops[op_id];
        op_mut.name = OpName::new("stablehlo", "constant");
        op_mut.operands.clear();
        op_mut.attributes = attrs;
        return ConstFoldOutcome::Modified;
    }

    ConstFoldOutcome::Unchanged
}

enum AlgOutcome {
    Unchanged,
    SwappedOnly,
    ModifiedAndDone,
    Replaced(bool),
}

fn try_simplify_algebraic_op(module: &mut Module, op_id: OpId) -> AlgOutcome {
    let Some(op) = module.ops.get(op_id) else {
        return AlgOutcome::Unchanged;
    };
    let Some(mut hlo) = StableHloOp::from_op(op) else {
        return AlgOutcome::Unchanged;
    };

    let mut swapped = false;
    match &mut hlo {
        StableHloOp::Add { lhs, rhs, .. }
        | StableHloOp::Multiply { lhs, rhs, .. }
        | StableHloOp::Maximum { lhs, rhs, .. }
        | StableHloOp::Minimum { lhs, rhs, .. }
            if is_constant(module, *lhs) && !is_constant(module, *rhs) =>
        {
            std::mem::swap(lhs, rhs);
            module.ops[op_id].operands.swap(0, 1);
            swapped = true;
        }
        _ => {}
    }

    // Reassociation of splat constants: `(x + c1) + c2 -> x + (c1 + c2)`
    // and `(x * c1) * c2 -> x * (c1 * c2)`
    match hlo {
        StableHloOp::Add { lhs, rhs, .. } => {
            if let Some(c2) = FoldConstantsPass::const_splat(module, rhs)
                && let ValueDef::OpResult { op: lhs_op_id, .. } = module.values[lhs].def
                && let Some(lhs_op) = module.ops.get(lhs_op_id)
                && let Some(StableHloOp::Add {
                    lhs: x,
                    rhs: c1_val,
                    ..
                }) = StableHloOp::from_op(lhs_op)
                && let Some(c1) = FoldConstantsPass::const_splat(module, c1_val)
                && let ValueDef::OpResult { op: c2_op_id, .. } = module.values[rhs].def
                && let Some(c2_ty) = module.values[rhs].ty.as_tensor().cloned()
            {
                let sum_lit = format_splat(c1 + c2);
                module.ops[c2_op_id].attributes.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: c2_ty,
                        raw_literal: sum_lit,
                    },
                );
                module.ops[op_id].operands[0] = x;
                return AlgOutcome::ModifiedAndDone;
            }
        }
        StableHloOp::Multiply { lhs, rhs, .. } => {
            if let Some(c2) = FoldConstantsPass::const_splat(module, rhs)
                && let ValueDef::OpResult { op: lhs_op_id, .. } = module.values[lhs].def
                && let Some(lhs_op) = module.ops.get(lhs_op_id)
                && let Some(StableHloOp::Multiply {
                    lhs: x,
                    rhs: c1_val,
                    ..
                }) = StableHloOp::from_op(lhs_op)
                && let Some(c1) = FoldConstantsPass::const_splat(module, c1_val)
                && let ValueDef::OpResult { op: c2_op_id, .. } = module.values[rhs].def
                && let Some(c2_ty) = module.values[rhs].ty.as_tensor().cloned()
            {
                let prod_lit = format_splat(c1 * c2);
                module.ops[c2_op_id].attributes.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: c2_ty,
                        raw_literal: prod_lit,
                    },
                );
                module.ops[op_id].operands[0] = x;
                return AlgOutcome::ModifiedAndDone;
            }
        }
        _ => {}
    }

    // Self-subtraction `subtract(x, x) -> constant 0`
    if let StableHloOp::Subtract { lhs, rhs, result } = hlo
        && lhs == rhs
        && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
    {
        let op_mut = &mut module.ops[op_id];
        op_mut.name = OpName::new("stablehlo", "constant");
        op_mut.operands.clear();
        op_mut.attributes.clear();
        op_mut.attributes.insert(
            "value".to_string(),
            Attribute::DenseElements {
                ty: res_ty,
                raw_literal: "0.0".to_string(),
            },
        );
        return AlgOutcome::ModifiedAndDone;
    }

    let replacement: Option<(ValueId, ValueId)> = match hlo {
        StableHloOp::Add { lhs, rhs, result } => {
            if const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                Some((result, lhs))
            } else if const_attr(module, lhs).is_some_and(|a| a.is_splat_zero()) {
                Some((result, rhs))
            } else {
                None
            }
        }
        StableHloOp::Subtract { lhs, rhs, result } => {
            if const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                Some((result, lhs))
            } else {
                None
            }
        }
        StableHloOp::Multiply { lhs, rhs, result } => {
            if const_attr(module, rhs).is_some_and(|a| a.is_splat_one())
                || const_attr(module, lhs).is_some_and(|a| a.is_splat_zero())
            {
                Some((result, lhs))
            } else if const_attr(module, lhs).is_some_and(|a| a.is_splat_one())
                || const_attr(module, rhs).is_some_and(|a| a.is_splat_zero())
            {
                Some((result, rhs))
            } else {
                None
            }
        }
        StableHloOp::Divide { lhs, rhs, result } => {
            if const_attr(module, rhs).is_some_and(|a| a.is_splat_one()) {
                Some((result, lhs))
            } else {
                None
            }
        }
        StableHloOp::Maximum { lhs, rhs, result } | StableHloOp::Minimum { lhs, rhs, result } => {
            if lhs == rhs {
                Some((result, lhs))
            } else {
                None
            }
        }
        StableHloOp::Select {
            on_true,
            on_false,
            result,
            ..
        } => {
            if on_true == on_false {
                Some((result, on_true))
            } else {
                None
            }
        }
        StableHloOp::Convert { operand, result } => {
            if module.values[operand].ty == module.values[result].ty {
                Some((result, operand))
            } else {
                None
            }
        }
        StableHloOp::Slice {
            operand,
            start_indices,
            limit_indices,
            strides,
            result,
        } => {
            let is_full = module.values[operand].ty == module.values[result].ty
                && start_indices.iter().all(|&s| s == 0)
                && strides.iter().all(|&s| s == 1)
                && module.values[operand].ty.as_tensor().is_some_and(|t| {
                    t.shape.len() == limit_indices.len()
                        && t.shape
                            .iter()
                            .zip(limit_indices.iter())
                            .all(|(d, &lim)| *d == Dim::Static(lim))
                });
            if is_full {
                Some((result, operand))
            } else {
                None
            }
        }
        _ => None,
    };

    if let Some((res, rep)) = replacement {
        module.replace_all_uses(res, rep);
        return AlgOutcome::Replaced(swapped);
    }

    if swapped {
        AlgOutcome::SwappedOnly
    } else {
        AlgOutcome::Unchanged
    }
}

fn try_egraph_and_fma_op(
    module: &mut Module,
    block_id: BlockId,
    op_id: OpId,
    fuse_fma: bool,
) -> (usize, usize) {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return (0, 0);
    };
    let (lhs, rhs, is_sub) = match StableHloOp::from_op(&op) {
        Some(StableHloOp::Add { lhs, rhs, .. }) => (lhs, rhs, false),
        Some(StableHloOp::Subtract { lhs, rhs, .. }) => (lhs, rhs, true),
        _ => return (0, 0),
    };

    if let (Some((l0, l1)), Some((r0, r1))) = (
        match_mul_operands(module, lhs),
        match_mul_operands(module, rhs),
    ) {
        let factored = if l0 == r0 {
            Some((l0, l1, r1))
        } else if l0 == r1 {
            Some((l0, l1, r0))
        } else if l1 == r0 {
            Some((l1, l0, r1))
        } else if l1 == r1 {
            Some((l1, l0, r0))
        } else {
            None
        };
        if let Some((common, b_val, c_val)) = factored {
            let out_ty = module.values[op.results[0]].ty.clone();
            let inner_op_name = if is_sub {
                OpName::new("stablehlo", "subtract")
            } else {
                OpName::new("stablehlo", "add")
            };
            let bc_val = insert_op_before(
                module,
                block_id,
                op_id,
                inner_op_name,
                &[b_val, c_val],
                out_ty,
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "multiply");
            op_mut.operands = SmallVec::from_slice(&[common, bc_val]);
            op_mut.attributes.clear();
            return (1, 1);
        }
    }

    if fuse_fma && !is_sub {
        let fma_match = match_mul_operands(module, lhs)
            .map(|(a, b)| (a, b, rhs))
            .or_else(|| match_mul_operands(module, rhs).map(|(a, b)| (a, b, lhs)));
        if let Some((a_val, b_val, c_val)) = fma_match {
            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fma".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fma_impl".to_string()),
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[a_val, b_val, c_val]);
            op_mut.attributes = attrs;
            return (0, 1);
        }
    }

    (0, 0)
}

fn const_attr(module: &Module, val_id: ValueId) -> Option<&Attribute> {
    if let ValueDef::OpResult { op, .. } = module.values[val_id].def {
        let op_node = module.ops.get(op)?;
        if op_node.is_op("stablehlo", "constant") {
            return op_node.attributes.get("value");
        }
    }
    None
}

fn is_constant(module: &Module, val_id: ValueId) -> bool {
    const_attr(module, val_id).is_some()
}

fn format_splat(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{:.1}", v.round())
    } else {
        format!("{v}")
    }
}

fn transpose_f32_nd(
    in_vals: &[f32],
    in_shape: &[Dim],
    out_shape: &[Dim],
    perm: &[i64],
) -> Option<Vec<f32>> {
    let rank = in_shape.len();
    if rank == 0 || out_shape.len() != rank || perm.len() != rank {
        return None;
    }
    let mut in_dims = Vec::with_capacity(rank);
    let mut out_dims = Vec::with_capacity(rank);
    for d in in_shape {
        if let Dim::Static(v) = d {
            in_dims.push(*v as usize);
        } else {
            return None;
        }
    }
    for d in out_shape {
        if let Dim::Static(v) = d {
            out_dims.push(*v as usize);
        } else {
            return None;
        }
    }
    let total: usize = in_dims.iter().product();
    if in_vals.len() != total {
        return None;
    }
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
    }
    let mut out = vec![0.0_f32; total];
    for (out_flat, out_elem) in out.iter_mut().enumerate() {
        let mut rem = out_flat;
        let mut out_coord = vec![0usize; rank];
        for i in (0..rank).rev() {
            let d = out_dims[i].max(1);
            out_coord[i] = rem % d;
            rem /= d;
        }
        let mut in_flat = 0usize;
        for (out_axis, &in_axis) in perm.iter().enumerate() {
            let ia = in_axis as usize;
            if ia >= rank {
                return None;
            }
            in_flat += out_coord[out_axis] * in_strides[ia];
        }
        *out_elem = in_vals[in_flat];
    }
    Some(out)
}

fn match_mul_operands(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: mul_id, .. } = module.values[val_id].def else {
        return None;
    };
    let mul_op = module.ops.get(mul_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(mul_op)? else {
        return None;
    };
    Some((lhs, rhs))
}

fn insert_op_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    out_ty: crate::ir::Type,
) -> ValueId {
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
        operands: SmallVec::from_slice(operands),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(val_id);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, new_op_id);
    } else {
        block_ops.push(new_op_id);
    }
    val_id
}
