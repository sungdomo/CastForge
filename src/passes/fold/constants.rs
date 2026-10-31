//! Compile-time constant folding and conditional constant propagation (`fold-constants`).

use crate::ir::{
    Attribute, Dim, Module, OpId, OpName, ValueDef, ValueId, format_dense_f32_literal,
};
use crate::passes::{Pass, PassResult, SimplifyAlgebraicPass};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;

/// Constant Folding & Conditional Constant Propagation (SCCP) Pass (`fold-constants`).
#[derive(Debug, Clone, Default)]
pub struct FoldConstantsPass;

impl Pass for FoldConstantsPass {
    fn name(&self) -> &'static str {
        "fold-constants"
    }

    fn description(&self) -> &'static str {
        "Evaluates pure operations on constant inputs at compile time (including SCCP)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(SimplifyAlgebraicPass {
            fold_constants: true,
            include_conditional: true,
            egraph: false,
            fuse_fma: false,
        }
        .run(module)?
        .with_name(self.name()))
    }
}

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

pub(crate) enum ConstFoldOutcome {
    Unchanged,
    Modified,
    Replaced,
}

pub(crate) fn try_fold_constant_op(
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
