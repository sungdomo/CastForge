use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Transformer Macro-Fusion Pass: Fuses decomposed Softmax subgraphs
/// (`exp(x) / broadcast_in_dim(reduce_sum(exp(x)))` or with max-subtraction)
/// into a single `stablehlo.composite` operation (`"castforge.fused_softmax"`).
pub struct FuseSoftmaxPass;

impl Pass for FuseSoftmaxPass {
    fn name(&self) -> &'static str {
        "fuse-softmax"
    }

    fn description(&self) -> &'static str {
        "Fuses exp / broadcast(reduce_sum(exp)) into stablehlo.composite (fused_softmax)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Divide {
                    lhs: exp_val,
                    rhs: denom_val,
                    ..
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                // Match `exp_val = stablehlo.exponential(shifted_or_x)`
                let ValueDef::OpResult { op: exp_op_id, .. } = module.values[exp_val].def else {
                    continue;
                };
                let Some(exp_op) = module.ops.get(exp_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Exp {
                    operand: exp_in, ..
                }) = StableHloOp::from_op(exp_op)
                else {
                    continue;
                };

                // Match `denom_val = broadcast_in_dim(reduce(exp_val, init))` (or direct `reduce`)
                let Some((reduce_axis, bcast_op_id, reduce_op_id)) =
                    match_broadcast_reduce_of_exp(module, denom_val, exp_val)
                else {
                    continue;
                };

                // Optional: `exp_in` may be `subtract(x, broadcast_in_dim(reduce_max(x)))`
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
                created += 1;

                let _ = (bcast_op_id, reduce_op_id);
            }
        }

        if created > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

fn match_broadcast_reduce_of_exp(
    module: &Module,
    denom_val: ValueId,
    exp_val: ValueId,
) -> Option<(i64, Option<castforge_ir::OpId>, castforge_ir::OpId)> {
    let mut cur_val = denom_val;
    let mut bcast_id = None;

    if let ValueDef::OpResult { op: b_op_id, .. } = module.values[cur_val].def {
        if let Some(b_op) = module.ops.get(b_op_id) {
            if let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op) {
                cur_val = operand;
                bcast_id = Some(b_op_id);
            }
        }
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

fn unwrap_optional_max_sub(module: &Module, exp_in: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: sub_op_id, .. } = module.values[exp_in].def else {
        return None;
    };
    let sub_op = module.ops.get(sub_op_id)?;
    let StableHloOp::Subtract { lhs: x, rhs, .. } = StableHloOp::from_op(sub_op)? else {
        return None;
    };

    let mut cur = rhs;
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def {
        if let Some(b_op) = module.ops.get(b_id) {
            if let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op) {
                cur = operand;
            }
        }
    }
    let ValueDef::OpResult { op: red_id, .. } = module.values[cur].def else {
        return None;
    };
    let red_op = module.ops.get(red_id)?;
    if let Some(StableHloOp::Reduce { inputs, .. }) = StableHloOp::from_op(red_op) {
        if inputs.first().copied() == Some(x) {
            return Some(x);
        }
    }
    None
}
