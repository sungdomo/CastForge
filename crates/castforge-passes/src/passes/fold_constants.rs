use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Dim, Module, OpName, ValueDef, ValueId, format_dense_f32_literal,
};
use indexmap::IndexMap;

/// 9. Unified Constant Folding & Conditional Propagation Pass (`fold-constants`)
pub struct FoldConstantsPass {
    pub include_conditional: bool,
}

impl Default for FoldConstantsPass {
    fn default() -> Self {
        Self {
            include_conditional: true,
        }
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

impl Pass for FoldConstantsPass {
    fn name(&self) -> &'static str {
        "fold-constants"
    }

    fn description(&self) -> &'static str {
        "Constant folding & conditional propagation (arithmetic, shape, compare/select)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                // 1. Conditional folding (`compare` & `select`) when enabled
                if self.include_conditional {
                    match &hlo {
                        StableHloOp::Compare {
                            lhs,
                            rhs,
                            direction,
                            result,
                        } => {
                            if let (Some(a), Some(b)) = (
                                Self::const_splat(module, *lhs),
                                Self::const_splat(module, *rhs),
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
                                if let Some(res_ty) = module.values[*result].ty.as_tensor().cloned()
                                {
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
                                    modified += 1;
                                    continue;
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
                                Self::const_splat(module, *pred)
                                    .map(|v| if v != 0.0 { *on_true } else { *on_false })
                            };
                            if let Some(winner) = chosen {
                                module.replace_all_uses(*result, winner);
                                module.remove_op(block_id, op_id);
                                removed += 1;
                                continue;
                            }
                        }
                        _ => {}
                    }
                }

                // 2. General tensor & splat constant folding
                let folded_vec: Option<(ValueId, Vec<f32>)> = match hlo {
                    StableHloOp::Add { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Subtract { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Multiply { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Divide { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b))
                                if a.len() == b.len() && b.iter().all(|&y| y.abs() > 1e-12) =>
                            {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Maximum { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
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
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => Some((
                                result,
                                a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect(),
                            )),
                            _ => None,
                        }
                    }
                    StableHloOp::Rsqrt { operand, result } => Self::const_f32_vec(module, operand)
                        .map(|v| (result, v.into_iter().map(|x| 1.0 / x.sqrt()).collect())),
                    StableHloOp::Exp { operand, result } => Self::const_f32_vec(module, operand)
                        .map(|v| (result, v.into_iter().map(|x| x.exp()).collect())),
                    StableHloOp::Tanh { operand, result } => Self::const_f32_vec(module, operand)
                        .map(|v| (result, v.into_iter().map(|x| x.tanh()).collect())),
                    StableHloOp::Convert { operand, result }
                    | StableHloOp::Reshape { operand, result } => {
                        Self::const_f32_vec(module, operand).map(|v| (result, v))
                    }
                    StableHloOp::Transpose {
                        operand,
                        permutation,
                        result,
                    } => {
                        let in_ty = module.values[operand].ty.as_tensor().cloned();
                        let out_ty = module.values[result].ty.as_tensor().cloned();
                        match (in_ty, out_ty, Self::const_f32_vec(module, operand)) {
                            (Some(in_t), Some(out_t), Some(in_vals)) => {
                                transpose_f32_nd(&in_vals, &in_t.shape, &out_t.shape, &permutation)
                                    .map(|vals| (result, vals))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::BroadcastInDim {
                        operand, result, ..
                    } => Self::const_splat(module, operand).map(|v| {
                        let n = module.values[result]
                            .ty
                            .as_tensor()
                            .and_then(|t| t.num_elements())
                            .unwrap_or(1) as usize;
                        (result, vec![v as f32; n])
                    }),
                    _ => None,
                };

                if let Some((res_val, vals)) = folded_vec {
                    let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                        continue;
                    };
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
                    modified += 1;
                }
            }
        }

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
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
