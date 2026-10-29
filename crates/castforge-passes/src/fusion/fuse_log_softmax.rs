use crate::{DeadCodeEliminationPass, FuseSoftmaxPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Transformer Macro-Fusion Pass: LogSoftmax (`fuse-log-softmax`).
///
/// Recognizes `stablehlo.log(castforge.fused_softmax(x))` as well as
/// `subtract(shifted_x, broadcast_in_dim(log(reduce_sum(exp(shifted_x)))))` and fuses them into a
/// single numerically stable `stablehlo.composite` operation (`"castforge.fused_log_softmax"`).
pub struct FuseLogSoftmaxPass;

impl Pass for FuseLogSoftmaxPass {
    fn name(&self) -> &'static str {
        "fuse-log-softmax"
    }

    fn description(&self) -> &'static str {
        "Fuses log(softmax(x)) or x - log(reduce_sum(exp(x))) into fused_log_softmax"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let sm_res = FuseSoftmaxPass.run(module)?;
        let mut created = 0usize;
        let mut removed = sm_res.ops_removed;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };

                // Case 1: `stablehlo.log(castforge.fused_softmax(x))`
                if op.is_op("stablehlo", "log") && op.operands.len() == 1 {
                    let sm_val = op.operands[0];
                    if let ValueDef::OpResult { op: sm_id, .. } = module.values[sm_val].def
                        && let Some(sm_op) = module.ops.get(sm_id)
                        && let Some(StableHloOp::Composite { name, inputs, .. }) =
                            StableHloOp::from_op(sm_op)
                        && name == "castforge.fused_softmax"
                        && !inputs.is_empty()
                    {
                        let x_val = inputs[0];
                        let axis = match sm_op.attributes.get("axis") {
                            Some(Attribute::Int(a)) => *a,
                            _ => 1,
                        };
                        rewrite_to_log_softmax(module, op_id, x_val, axis);
                        created += 1;
                        continue;
                    }
                }

                // Case 2: `subtract(x, broadcast_in_dim(log(reduce_sum(exp(x)))))`
                if let Some(StableHloOp::Subtract { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                    && let Some((x_val, axis)) = match_log_sum_exp_sub(module, lhs, rhs)
                {
                    rewrite_to_log_softmax(module, op_id, x_val, axis);
                    created += 1;
                }
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

fn rewrite_to_log_softmax(
    module: &mut Module,
    op_id: castforge_ir::OpId,
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

fn match_log_sum_exp_sub(module: &Module, lhs: ValueId, rhs: ValueId) -> Option<(ValueId, i64)> {
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
