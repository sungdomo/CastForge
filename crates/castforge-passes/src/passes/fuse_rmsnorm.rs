use crate::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Transformer Macro-Fusion Pass: Fuses decomposed RMSNorm subgraphs
/// (`x * broadcast(rsqrt(mean(x * x) + eps))` + optional `multiply(..., weight)`)
/// into a single `stablehlo.composite` operation (`"castforge.fused_rmsnorm"`).
pub struct FuseRmsNormPass;

impl Pass for FuseRmsNormPass {
    fn name(&self) -> &'static str {
        "fuse-rmsnorm"
    }

    fn description(&self) -> &'static str {
        "Fuses x * rsqrt(mean(x^2) + eps) * weight into stablehlo.composite (fused_rmsnorm)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            // Pass 1: Match `multiply(multiply(x, rsqrt_bcast), weight)`
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };

                let matched = match_scaled_rmsnorm(module, lhs, rhs)
                    .or_else(|| match_scaled_rmsnorm(module, rhs, lhs));
                let Some((x_val, weight_val, eps_f64)) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_rmsnorm".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_rmsnorm_impl".to_string()),
                );
                attrs.insert("eps".to_string(), Attribute::Float(eps_f64));

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x_val, weight_val]);
                op_mut.attributes = attrs;
                created += 1;
            }

            // Pass 2: Match unscaled `multiply(x, rsqrt_bcast)` not already consumed
            if created == 0 {
                let ops2 = module.block_ops(block_id);
                for op_id in ops2 {
                    let Some(op) = module.ops.get(op_id) else {
                        continue;
                    };
                    let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op)
                    else {
                        continue;
                    };
                    let matched = match_x_and_rsqrt(module, lhs, rhs)
                        .or_else(|| match_x_and_rsqrt(module, rhs, lhs));
                    let Some((x_val, eps_f64)) = matched else {
                        continue;
                    };

                    let mut attrs = IndexMap::new();
                    attrs.insert(
                        "name".to_string(),
                        Attribute::String("castforge.fused_rmsnorm".to_string()),
                    );
                    attrs.insert(
                        "decomposition".to_string(),
                        Attribute::SymbolRef("fused_rmsnorm_impl".to_string()),
                    );
                    attrs.insert("eps".to_string(), Attribute::Float(eps_f64));

                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "composite");
                    op_mut.operands = SmallVec::from_slice(&[x_val]);
                    op_mut.attributes = attrs;
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

fn match_scaled_rmsnorm(
    module: &Module,
    norm_cand: ValueId,
    weight_cand: ValueId,
) -> Option<(ValueId, ValueId, f64)> {
    let ValueDef::OpResult { op: norm_op_id, .. } = module.values[norm_cand].def else {
        return None;
    };
    let norm_op = module.ops.get(norm_op_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(norm_op)? else {
        return None;
    };
    let (x_val, eps) =
        match_x_and_rsqrt(module, lhs, rhs).or_else(|| match_x_and_rsqrt(module, rhs, lhs))?;
    Some((x_val, weight_cand, eps))
}

fn match_x_and_rsqrt(
    module: &Module,
    x_cand: ValueId,
    rsqrt_cand: ValueId,
) -> Option<(ValueId, f64)> {
    let mut cur = rsqrt_cand;
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def {
        if let Some(b_op) = module.ops.get(b_id) {
            if let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op) {
                cur = operand;
            }
        }
    }

    let ValueDef::OpResult {
        op: rsqrt_op_id, ..
    } = module.values[cur].def
    else {
        return None;
    };
    let rsqrt_op = module.ops.get(rsqrt_op_id)?;
    let StableHloOp::Rsqrt {
        operand: rsqrt_in, ..
    } = StableHloOp::from_op(rsqrt_op)?
    else {
        return None;
    };

    // `rsqrt_in` may be `add(mean_or_red, eps)` or directly `mean_or_red`
    let mut eps = 1e-5_f64;
    let mut mean_val = rsqrt_in;
    if let ValueDef::OpResult { op: add_id, .. } = module.values[rsqrt_in].def {
        if let Some(add_op) = module.ops.get(add_id) {
            if let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(add_op) {
                if let Some(c) = FoldConstantsPass::const_splat(module, rhs) {
                    eps = c;
                    mean_val = lhs;
                } else if let Some(c) = FoldConstantsPass::const_splat(module, lhs) {
                    eps = c;
                    mean_val = rhs;
                }
            }
        }
    }

    // `mean_val` may be `divide(reduce_val, N)` or `multiply(reduce_val, inv_N)` or `reduce_val`
    let mut red_val = mean_val;
    if let ValueDef::OpResult { op: div_id, .. } = module.values[mean_val].def {
        if let Some(div_op) = module.ops.get(div_id) {
            match StableHloOp::from_op(div_op) {
                Some(StableHloOp::Divide { lhs, .. }) => red_val = lhs,
                Some(StableHloOp::Multiply { lhs, rhs, .. }) => {
                    red_val = if FoldConstantsPass::const_splat(module, rhs).is_some() {
                        lhs
                    } else {
                        rhs
                    };
                }
                _ => {}
            }
        }
    }

    let ValueDef::OpResult { op: red_id, .. } = module.values[red_val].def else {
        return None;
    };
    let red_op = module.ops.get(red_id)?;
    let StableHloOp::Reduce { inputs, .. } = StableHloOp::from_op(red_op)? else {
        return None;
    };
    let sq_val = *inputs.first()?;

    let ValueDef::OpResult { op: sq_id, .. } = module.values[sq_val].def else {
        return None;
    };
    let sq_op = module.ops.get(sq_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(sq_op)? else {
        return None;
    };
    if lhs == x_cand && rhs == x_cand {
        Some((x_cand, eps))
    } else {
        None
    }
}
