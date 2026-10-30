//! Submodule `layernorm_softmax` of `fuse_norm`.

use super::*;
use crate::ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use crate::passes::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// LayerNorm Fusion Pass (`fuse-layernorm`).
pub struct FuseLayerNormPass;

impl Pass for FuseLayerNormPass {
    fn name(&self) -> &'static str {
        "fuse-layernorm"
    }

    fn description(&self) -> &'static str {
        "Fuses (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta into fused_layernorm"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let (created, modified) =
            fuse_norm_single_traversal(module, false, false, true, false, false, 0);
        let mut removed = 0usize;
        if created > 0 || modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

pub(super) fn rewrite_to_layernorm(
    module: &mut Module,
    op_id: crate::ir::OpId,
    x_val: ValueId,
    gamma_opt: Option<ValueId>,
    beta_opt: Option<ValueId>,
    eps: f64,
) {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.fused_layernorm".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("fused_layernorm_impl".to_string()),
    );
    attrs.insert("eps".to_string(), Attribute::Float(eps));

    let mut operands = SmallVec::from_slice(&[x_val]);
    if let Some(g) = gamma_opt {
        operands.push(g);
    }
    if let Some(b) = beta_opt {
        operands.push(b);
    }

    let op_mut = &mut module.ops[op_id];
    op_mut.name = OpName::new("stablehlo", "composite");
    op_mut.operands = operands;
    op_mut.attributes = attrs;
}

pub(super) fn match_scaled_layernorm(
    module: &Module,
    val_id: ValueId,
) -> Option<(ValueId, ValueId, f64)> {
    let ValueDef::OpResult { op: mul_id, .. } = module.values[val_id].def else {
        return None;
    };
    let mul_op = module.ops.get(mul_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(mul_op)? else {
        return None;
    };
    match_core_layernorm_val(module, lhs)
        .map(|(x, eps)| (x, rhs, eps))
        .or_else(|| match_core_layernorm_val(module, rhs).map(|(x, eps)| (x, lhs, eps)))
}

pub(super) fn match_core_layernorm_val(module: &Module, val_id: ValueId) -> Option<(ValueId, f64)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match_core_layernorm_op(module, op)
}

pub(super) fn match_core_layernorm_op(
    module: &Module,
    op: &crate::ir::Operation,
) -> Option<(ValueId, f64)> {
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    match_centered_and_inv_std(module, lhs, rhs)
        .or_else(|| match_centered_and_inv_std(module, rhs, lhs))
}

fn match_centered_and_inv_std(
    module: &Module,
    centered_cand: ValueId,
    inv_std_cand: ValueId,
) -> Option<(ValueId, f64)> {
    let ValueDef::OpResult { op: sub_id, .. } = module.values[centered_cand].def else {
        return None;
    };
    let sub_op = module.ops.get(sub_id)?;
    let StableHloOp::Subtract {
        lhs: x_val,
        rhs: mean_bcast,
        ..
    } = StableHloOp::from_op(sub_op)?
    else {
        return None;
    };
    if unwrap_reduce_mean_input(module, mean_bcast)? != x_val {
        return None;
    }

    let mut cur = inv_std_cand;
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def
        && let Some(b_op) = module.ops.get(b_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        cur = operand;
    }
    let ValueDef::OpResult { op: rsqrt_id, .. } = module.values[cur].def else {
        return None;
    };
    let rsqrt_op = module.ops.get(rsqrt_id)?;
    let StableHloOp::Rsqrt {
        operand: rsqrt_in, ..
    } = StableHloOp::from_op(rsqrt_op)?
    else {
        return None;
    };

    let mut eps = 1e-5_f64;
    let mut var_val = rsqrt_in;
    if let ValueDef::OpResult { op: add_id, .. } = module.values[rsqrt_in].def
        && let Some(add_op) = module.ops.get(add_id)
        && let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(add_op)
    {
        if let Some(c) = FoldConstantsPass::const_splat(module, rhs) {
            eps = c;
            var_val = lhs;
        } else if let Some(c) = FoldConstantsPass::const_splat(module, lhs) {
            eps = c;
            var_val = rhs;
        }
    }

    let sq_val = unwrap_reduce_mean_input(module, var_val)?;
    let ValueDef::OpResult { op: sq_id, .. } = module.values[sq_val].def else {
        return None;
    };
    let sq_op = module.ops.get(sq_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(sq_op)? else {
        return None;
    };
    if lhs == centered_cand && rhs == centered_cand {
        Some((x_val, eps))
    } else {
        None
    }
}

fn unwrap_reduce_mean_input(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let mut cur = val_id;
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def
        && let Some(b_op) = module.ops.get(b_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        cur = operand;
    }
    if let ValueDef::OpResult { op: div_id, .. } = module.values[cur].def
        && let Some(div_op) = module.ops.get(div_id)
    {
        match StableHloOp::from_op(div_op) {
            Some(StableHloOp::Divide { lhs, .. }) => cur = lhs,
            Some(StableHloOp::Multiply { lhs, rhs, .. }) => {
                cur = if FoldConstantsPass::const_splat(module, rhs).is_some() {
                    lhs
                } else {
                    rhs
                };
            }
            _ => {}
        }
    }
    let ValueDef::OpResult { op: red_id, .. } = module.values[cur].def else {
        return None;
    };
    let red_op = module.ops.get(red_id)?;
    let StableHloOp::Reduce { inputs, .. } = StableHloOp::from_op(red_op)? else {
        return None;
    };
    inputs.first().copied()
}

/// Unified Softmax, LogSoftmax & Chunked Linear Cross-Entropy Pass (`fuse-softmax`).
pub struct FuseSoftmaxPass {
    pub kind: String,
    pub chunk_size: i64,
}

impl Default for FuseSoftmaxPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
            chunk_size: 1024,
        }
    }
}

impl Pass for FuseSoftmaxPass {
    fn name(&self) -> &'static str {
        "fuse-softmax"
    }

    fn description(&self) -> &'static str {
        "Unified Softmax, LogSoftmax, and chunked Linear Cross-Entropy fusion"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let do_ce = k == "cross_entropy" || k == "ce" || k == "linear_cross_entropy";
        let do_sm = !do_ce;

        let (created, modified) =
            fuse_norm_single_traversal(module, false, false, false, do_sm, do_ce, self.chunk_size);
        let mut removed = 0usize;
        if created > 0 || modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// Chunked Fused Linear Classifier + Cross-Entropy Loss Pass (`fuse-linear-cross-entropy`).
pub struct FuseLinearCrossEntropyPass {
    pub chunk_size: i64,
}

impl Default for FuseLinearCrossEntropyPass {
    fn default() -> Self {
        Self { chunk_size: 1024 }
    }
}

impl Pass for FuseLinearCrossEntropyPass {
    fn name(&self) -> &'static str {
        "fuse-linear-cross-entropy"
    }

    fn description(&self) -> &'static str {
        "Fuses classifier GEMM + online log-sum-exp Cross-Entropy to avoid materializing logits"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            for op_id in module.block_ops(block_id) {
                let Some(op) = module.ops.get_mut(op_id) else {
                    continue;
                };
                if op.is_op("stablehlo", "dot_general")
                    || (op.is_op("stablehlo", "composite")
                        && op
                            .attributes
                            .get("name")
                            .and_then(|a| a.as_str())
                            .is_some_and(|n| n.contains("log_softmax") || n.contains("matmul")))
                {
                    op.attributes.insert(
                        "castforge.fused_linear_cross_entropy".to_string(),
                        Attribute::Bool(true),
                    );
                    op.attributes.insert(
                        "castforge.ce_chunk_size".to_string(),
                        Attribute::Int(self.chunk_size.max(64)),
                    );
                    op.attributes.insert(
                        "castforge.materialize_logits".to_string(),
                        Attribute::Bool(false),
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

pub(crate) fn try_fuse_softmax_at_op(
    module: &mut Module,
    op_id: crate::ir::OpId,
    op: &crate::ir::Operation,
) -> bool {
    if let Some(StableHloOp::Divide {
        lhs: exp_val,
        rhs: denom_val,
        ..
    }) = StableHloOp::from_op(op)
        && let ValueDef::OpResult { op: exp_op_id, .. } = module.values[exp_val].def
        && let Some(exp_op) = module.ops.get(exp_op_id)
        && let Some(StableHloOp::Exp {
            operand: exp_in, ..
        }) = StableHloOp::from_op(exp_op)
        && let Some((reduce_axis, _, _)) = match_broadcast_reduce_of_exp(module, denom_val, exp_val)
    {
        let x_input = unwrap_optional_max_sub(module, exp_in).unwrap_or(exp_in);
        let mut attrs = IndexMap::new();
        attrs.insert(
            "name".to_string(),
            Attribute::String("castforge.fused_softmax".to_string()),
        );
        attrs.insert(
            "decomposition".to_string(),
            Attribute::SymbolRef("fused_softmax_impl".to_string()),
        );
        attrs.insert("axis".to_string(), Attribute::Int(reduce_axis));

        let op_mut = &mut module.ops[op_id];
        op_mut.name = OpName::new("stablehlo", "composite");
        op_mut.operands = SmallVec::from_slice(&[x_input]);
        op_mut.attributes = attrs;
        return true;
    }
    false
}

pub(super) fn rewrite_to_log_softmax(
    module: &mut Module,
    op_id: crate::ir::OpId,
    x_val: ValueId,
    axis: i64,
) {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.fused_log_softmax".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("fused_log_softmax_impl".to_string()),
    );
    attrs.insert("axis".to_string(), Attribute::Int(axis));

    let op_mut = &mut module.ops[op_id];
    op_mut.name = OpName::new("stablehlo", "composite");
    op_mut.operands = SmallVec::from_slice(&[x_val]);
    op_mut.attributes = attrs;
}

pub(super) fn match_log_sum_exp_sub(
    module: &Module,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<(ValueId, i64)> {
    let mut cur = rhs;
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def
        && let Some(b_op) = module.ops.get(b_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        cur = operand;
    }
    if let ValueDef::OpResult { op: log_id, .. } = module.values[cur].def
        && let Some(log_op) = module.ops.get(log_id)
        && log_op.is_op("stablehlo", "log")
        && log_op.operands.len() == 1
    {
        cur = log_op.operands[0];
    } else {
        return None;
    }
    let ValueDef::OpResult { op: red_id, .. } = module.values[cur].def else {
        return None;
    };
    let red_op = module.ops.get(red_id)?;
    let StableHloOp::Reduce {
        inputs, dimensions, ..
    } = StableHloOp::from_op(red_op)?
    else {
        return None;
    };
    let exp_val = *inputs.first()?;
    let ValueDef::OpResult { op: exp_id, .. } = module.values[exp_val].def else {
        return None;
    };
    let exp_op = module.ops.get(exp_id)?;
    let StableHloOp::Exp { operand, .. } = StableHloOp::from_op(exp_op)? else {
        return None;
    };
    if operand != lhs {
        return None;
    }
    let axis = dimensions.first().copied().unwrap_or(1);
    Some((lhs, axis))
}

pub(super) fn match_broadcast_reduce_of_exp(
    module: &Module,
    denom_val: ValueId,
    exp_val: ValueId,
) -> Option<(i64, Option<crate::ir::OpId>, crate::ir::OpId)> {
    let mut cur_val = denom_val;
    let mut bcast_id = None;

    if let ValueDef::OpResult { op: b_op_id, .. } = module.values[cur_val].def
        && let Some(b_op) = module.ops.get(b_op_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        cur_val = operand;
        bcast_id = Some(b_op_id);
    }

    let ValueDef::OpResult { op: red_op_id, .. } = module.values[cur_val].def else {
        return None;
    };
    let red_op = module.ops.get(red_op_id)?;
    let StableHloOp::Reduce {
        inputs, dimensions, ..
    } = StableHloOp::from_op(red_op)?
    else {
        return None;
    };

    if inputs.first().copied() != Some(exp_val) {
        return None;
    }
    let axis = dimensions.first().copied().unwrap_or(-1);
    Some((axis, bcast_id, red_op_id))
}

pub(super) fn unwrap_optional_max_sub(module: &Module, exp_in: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: sub_op_id, .. } = module.values[exp_in].def else {
        return None;
    };
    let sub_op = module.ops.get(sub_op_id)?;
    let StableHloOp::Subtract { lhs: x, rhs, .. } = StableHloOp::from_op(sub_op)? else {
        return None;
    };

    let mut cur = rhs;
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def
        && let Some(b_op) = module.ops.get(b_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        cur = operand;
    }
    let ValueDef::OpResult { op: red_id, .. } = module.values[cur].def else {
        return None;
    };
    let red_op = module.ops.get(red_id)?;
    if let Some(StableHloOp::Reduce { inputs, .. }) = StableHloOp::from_op(red_op)
        && inputs.first().copied() == Some(x)
    {
        return Some(x);
    }
    None
}
