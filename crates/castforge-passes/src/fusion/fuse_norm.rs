//! Unified Normalization Fusion (`fuse-norm`): Residual+RMSNorm, RMSNorm, and LayerNorm.

use crate::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Normalization & Softmax/Cross-Entropy Fusion Pass (`fuse-norm`).
pub struct FuseNormPass {
    pub kind: String,
    pub chunk_size: i64,
}

impl Default for FuseNormPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
            chunk_size: 1024,
        }
    }
}

impl Pass for FuseNormPass {
    fn name(&self) -> &'static str {
        "fuse-norm"
    }

    fn description(&self) -> &'static str {
        "Unified normalization & softmax fusion (RMSNorm, LayerNorm, Softmax, Cross-Entropy)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        if k == "all" || k == "residual_rmsnorm" || k == "residual" {
            steps.push(FuseResidualNormPass.run(module)?);
        }
        if k == "all" || k == "rmsnorm" || k == "rms" {
            steps.push(FuseRmsNormPass.run(module)?);
        }
        if k == "all" || k == "layernorm" || k == "ln" {
            steps.push(FuseLayerNormPass.run(module)?);
        }
        if k == "all" || k == "softmax" || k == "log_softmax" {
            steps.push(
                crate::FuseSoftmaxPass {
                    kind: "all".to_string(),
                    chunk_size: self.chunk_size,
                }
                .run(module)?,
            );
        } else if k == "cross_entropy" || k == "ce" || k == "linear_cross_entropy" {
            steps.push(
                crate::FuseSoftmaxPass {
                    kind: "cross_entropy".to_string(),
                    chunk_size: self.chunk_size,
                }
                .run(module)?,
            );
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Residual-Add + RMSNorm Fusion Pass (`fuse-residual-norm`).
pub struct FuseResidualNormPass;

impl Pass for FuseResidualNormPass {
    fn name(&self) -> &'static str {
        "fuse-residual-norm"
    }

    fn description(&self) -> &'static str {
        "Fuses add(x, residual) + RMSNorm into stablehlo.composite (fused_residual_rmsnorm)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let rms_res = FuseRmsNormPass.run(module)?;
        let mut created = rms_res.ops_created;
        let mut removed = rms_res.ops_removed;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if name != "castforge.fused_rmsnorm" || inputs.is_empty() {
                    continue;
                }

                let h_val = inputs[0];
                let ValueDef::OpResult { op: add_op_id, .. } = module.values[h_val].def else {
                    continue;
                };
                let Some(add_op) = module.ops.get(add_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add {
                    lhs: x, rhs: res, ..
                }) = StableHloOp::from_op(add_op)
                else {
                    continue;
                };

                let mut new_operands = SmallVec::from_slice(&[x, res]);
                for &extra in inputs.iter().skip(1) {
                    new_operands.push(extra);
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.operands = new_operands;
                op_mut.attributes.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_residual_rmsnorm".to_string()),
                );
                op_mut.attributes.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_residual_rmsnorm_impl".to_string()),
                );
                modified += 1;
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
            created += modified;
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

/// RMSNorm Fusion Pass (`fuse-rmsnorm`).
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
    if let ValueDef::OpResult { op: b_id, .. } = module.values[cur].def
        && let Some(b_op) = module.ops.get(b_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        cur = operand;
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

    let mut eps = 1e-5_f64;
    let mut mean_val = rsqrt_in;
    if let ValueDef::OpResult { op: add_id, .. } = module.values[rsqrt_in].def
        && let Some(add_op) = module.ops.get(add_id)
        && let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(add_op)
    {
        if let Some(c) = FoldConstantsPass::const_splat(module, rhs) {
            eps = c;
            mean_val = lhs;
        } else if let Some(c) = FoldConstantsPass::const_splat(module, lhs) {
            eps = c;
            mean_val = rhs;
        }
    }

    let mut red_val = mean_val;
    if let ValueDef::OpResult { op: div_id, .. } = module.values[mean_val].def
        && let Some(div_op) = module.ops.get(div_id)
    {
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
                let matched = match_scaled_layernorm(module, lhs)
                    .map(|(x, g, eps)| (x, Some(g), Some(rhs), eps))
                    .or_else(|| {
                        match_scaled_layernorm(module, rhs)
                            .map(|(x, g, eps)| (x, Some(g), Some(lhs), eps))
                    });
                let Some((x_val, gamma_opt, beta_opt, eps)) = matched else {
                    continue;
                };
                rewrite_to_layernorm(module, op_id, x_val, gamma_opt, beta_opt, eps);
                created += 1;
            }

            if created == 0 {
                let ops2 = module.block_ops(block_id);
                for op_id in ops2 {
                    let Some((x_val, gamma_val, eps)) = module.ops.get(op_id).and_then(|op| {
                        let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)?
                        else {
                            return None;
                        };
                        match_core_layernorm_val(module, lhs)
                            .map(|(x, e)| (x, rhs, e))
                            .or_else(|| {
                                match_core_layernorm_val(module, rhs).map(|(x, e)| (x, lhs, e))
                            })
                    }) else {
                        continue;
                    };
                    rewrite_to_layernorm(module, op_id, x_val, Some(gamma_val), None, eps);
                    created += 1;
                }
            }

            if created == 0 {
                let ops3 = module.block_ops(block_id);
                for op_id in ops3 {
                    let Some((x_val, eps)) = module
                        .ops
                        .get(op_id)
                        .and_then(|op| match_core_layernorm_op(module, op))
                    else {
                        continue;
                    };
                    rewrite_to_layernorm(module, op_id, x_val, None, None, eps);
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

fn rewrite_to_layernorm(
    module: &mut Module,
    op_id: castforge_ir::OpId,
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

fn match_scaled_layernorm(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId, f64)> {
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

fn match_core_layernorm_val(module: &Module, val_id: ValueId) -> Option<(ValueId, f64)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match_core_layernorm_op(module, op)
}

fn match_core_layernorm_op(
    module: &Module,
    op: &castforge_ir::Operation,
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
