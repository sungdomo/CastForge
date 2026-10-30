//! Unified Softmax, LogSoftmax & Chunked Linear Cross-Entropy Fusion (`fuse-softmax`).

use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Softmax, LogSoftmax & Chunked Linear Cross-Entropy Pass (`fuse-softmax`).
///
/// - `kind = "softmax" | "log_softmax" | "all"` (default): Fuses `exp(x) / sum(exp(x))` into
///   `castforge.fused_softmax` and `log(softmax(x))` / `x - log(sum(exp(x)))` into
///   `castforge.fused_log_softmax`.
/// - `kind = "cross_entropy"`: Fuses classifier `dot_general` + online log-sum-exp cross-entropy
///   into `castforge.fused_linear_cross_entropy` with `castforge.ce_chunk_size`.
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
        if k == "cross_entropy" || k == "ce" || k == "linear_cross_entropy" {
            let mut res = FuseLinearCrossEntropyPass {
                chunk_size: self.chunk_size,
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }

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

                let Some((reduce_axis, bcast_op_id, reduce_op_id)) =
                    match_broadcast_reduce_of_exp(module, denom_val, exp_val)
                else {
                    continue;
                };

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

        let mut log_created = 0usize;
        let blocks2: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks2 {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
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
                        log_created += 1;
                        continue;
                    }
                }
                if let Some(StableHloOp::Subtract { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                    && let Some((x_val, axis)) = match_log_sum_exp_sub(module, lhs, rhs)
                {
                    rewrite_to_log_softmax(module, op_id, x_val, axis);
                    log_created += 1;
                }
            }
        }
        if log_created > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
            created += log_created;
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

fn match_broadcast_reduce_of_exp(
    module: &Module,
    denom_val: ValueId,
    exp_val: ValueId,
) -> Option<(i64, Option<castforge_ir::OpId>, castforge_ir::OpId)> {
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

fn unwrap_optional_max_sub(module: &Module, exp_in: ValueId) -> Option<ValueId> {
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
