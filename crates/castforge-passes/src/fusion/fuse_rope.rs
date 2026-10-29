//! Rotary Position Embedding (RoPE) fusion pass (`fuse-rope`).

use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// AI Optimization Pass: Rotary Position Embedding Fusion (`fuse-rope`).
///
/// Detects decomposed RoPE subgraphs:
/// `add(multiply(x, cos), multiply(concatenate(neg(slice_hi(x)), slice_lo(x)), sin))`
/// and fuses them into a single `stablehlo.composite` (`"castforge.fused_rope"`).
pub struct FuseRopePass;

struct RopeMatch {
    x: ValueId,
    cos: ValueId,
    sin: ValueId,
}

impl Pass for FuseRopePass {
    fn name(&self) -> &'static str {
        "fuse-rope"
    }

    fn description(&self) -> &'static str {
        "Fuses decomposed Rotary Position Embedding (RoPE) subgraphs into stablehlo.composite"
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
                let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };

                let matched =
                    match_rope_pair(module, lhs, rhs).or_else(|| match_rope_pair(module, rhs, lhs));
                let Some(RopeMatch { x, cos, sin }) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_rope".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_rope_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x, cos, sin]);
                op_mut.attributes = attrs;
                created += 1;
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

fn match_rope_pair(
    module: &Module,
    mul_cos_val: ValueId,
    mul_sin_val: ValueId,
) -> Option<RopeMatch> {
    let (a0, a1) = match_multiply_operands(module, mul_cos_val)?;
    let (b0, b1) = match_multiply_operands(module, mul_sin_val)?;

    for &(rot_cand, sin_cand) in &[(b0, b1), (b1, b0)] {
        if let Some(x) = match_rotate_half(module, rot_cand) {
            if a0 == x {
                return Some(RopeMatch {
                    x,
                    cos: a1,
                    sin: sin_cand,
                });
            }
            if a1 == x {
                return Some(RopeMatch {
                    x,
                    cos: a0,
                    sin: sin_cand,
                });
            }
        }
    }
    None
}

fn match_multiply_operands(module: &Module, val: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    Some((lhs, rhs))
}

/// Matches `concatenate([neg(slice(x, hi)), slice(x, lo)])` and returns `x`.
fn match_rotate_half(module: &Module, val: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: cat_id, .. } = module.values[val].def else {
        return None;
    };
    let cat_op = module.ops.get(cat_id)?;
    let StableHloOp::Concatenate { inputs, .. } = StableHloOp::from_op(cat_op)? else {
        return None;
    };
    if inputs.len() != 2 {
        return None;
    }

    let neg_slice_val = match_negated_value(module, inputs[0])?;
    let x_hi = match_slice_source(module, neg_slice_val)?;
    let x_lo = match_slice_source(module, inputs[1])?;
    if x_hi == x_lo { Some(x_lo) } else { None }
}

fn match_negated_value(module: &Module, val: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if op.is_op("stablehlo", "negate") {
        return op.operands.first().copied();
    }
    if let Some(StableHloOp::Subtract { lhs, rhs, .. }) = StableHloOp::from_op(op)
        && is_zero_constant(module, lhs)
    {
        return Some(rhs);
    }
    None
}

fn is_zero_constant(module: &Module, val: ValueId) -> bool {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return false;
    };
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::Constant { value, .. }) = StableHloOp::from_op(op) else {
        return false;
    };
    match value {
        Attribute::Float(f) => f.abs() < 1e-9,
        Attribute::Int(i) => i == 0,
        Attribute::DenseElements { raw_literal, .. } => raw_literal
            .trim()
            .parse::<f64>()
            .is_ok_and(|f| f.abs() < 1e-9),
        _ => false,
    }
}

fn match_slice_source(module: &Module, val: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    let StableHloOp::Slice { operand, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    Some(operand)
}
