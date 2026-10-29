use crate::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Transformer Macro-Fusion Pass: SwiGLU (`fuse-swiglu`).
///
/// Recognizes the SwiGLU gated MLP subgraph:
///   `neg = subtract(0, gate)`
///   `exp = exponential(neg)`
///   `denom = add(exp, 1)` (or `add(1, exp)`)
///   `silu = divide(gate, denom)`
///   `out = multiply(silu, up)` (or `multiply(up, silu)`)
/// and fuses it into a single `stablehlo.composite` operation (`"castforge.fused_swiglu"`,
/// `decomposition = @fused_swiglu_impl`) taking `(gate, up)`.
pub struct FuseSwiGluPass;

impl Pass for FuseSwiGluPass {
    fn name(&self) -> &'static str {
        "fuse-swiglu"
    }

    fn description(&self) -> &'static str {
        "Fuses SwiGLU (gate / (1 + exp(-gate)) * up) into stablehlo.composite (fused_swiglu)"
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
                let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };

                let matched = match_silu_val(module, lhs)
                    .map(|g| (g, rhs))
                    .or_else(|| match_silu_val(module, rhs).map(|g| (g, lhs)));
                let Some((gate_val, up_val)) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_swiglu".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_swiglu_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[gate_val, up_val]);
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

/// Matches `silu(x) = divide(x, add(1.0, exponential(subtract(0.0, x))))` and returns `Some(x)`.
pub fn match_silu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: div_op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let div_op = module.ops.get(div_op_id)?;
    let StableHloOp::Divide {
        lhs: gate,
        rhs: denom,
        ..
    } = StableHloOp::from_op(div_op)?
    else {
        return None;
    };

    let ValueDef::OpResult { op: add_op_id, .. } = module.values[denom].def else {
        return None;
    };
    let add_op = module.ops.get(add_op_id)?;
    let StableHloOp::Add {
        lhs: a1, rhs: a2, ..
    } = StableHloOp::from_op(add_op)?
    else {
        return None;
    };

    let exp_val = if FoldConstantsPass::const_splat(module, a2)
        .is_some_and(|v| (v - 1.0).abs() < 1e-6)
    {
        a1
    } else if FoldConstantsPass::const_splat(module, a1).is_some_and(|v| (v - 1.0).abs() < 1e-6) {
        a2
    } else {
        return None;
    };

    let ValueDef::OpResult { op: exp_op_id, .. } = module.values[exp_val].def else {
        return None;
    };
    let exp_op = module.ops.get(exp_op_id)?;
    let StableHloOp::Exp {
        operand: neg_val, ..
    } = StableHloOp::from_op(exp_op)?
    else {
        return None;
    };

    let ValueDef::OpResult { op: sub_op_id, .. } = module.values[neg_val].def else {
        return None;
    };
    let sub_op = module.ops.get(sub_op_id)?;
    let StableHloOp::Subtract {
        lhs: zero_val,
        rhs: sub_rhs,
        ..
    } = StableHloOp::from_op(sub_op)?
    else {
        return None;
    };

    if sub_rhs != gate {
        return None;
    }
    if !FoldConstantsPass::const_splat(module, zero_val).is_some_and(|v| v.abs() < 1e-7) {
        return None;
    }

    Some(gate)
}
