//! Unified Normalization Fusion (`fuse-norm`): Residual+RMSNorm, RMSNorm, and LayerNorm.

pub mod layernorm_softmax;

pub use layernorm_softmax::*;

use crate::ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use crate::passes::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
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
        let do_res_rms = k == "all" || k == "residual_rmsnorm" || k == "residual";
        let do_rms = do_res_rms || k == "rmsnorm" || k == "rms";
        let do_ln = k == "all" || k == "layernorm" || k == "ln";
        let do_sm = k == "all" || k == "softmax" || k == "log_softmax";
        let do_ce = k == "cross_entropy" || k == "ce" || k == "linear_cross_entropy";

        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;

        if do_rms || do_ln || do_sm {
            let (c, m) = fuse_norm_single_traversal(module, do_rms, do_res_rms, do_ln, do_sm);
            created += c;
            modified += m;
        }

        if do_ce {
            let ce_res = FuseLinearCrossEntropyPass {
                chunk_size: self.chunk_size,
            }
            .run(module)?;
            modified += ce_res.ops_modified;
        }

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

pub(super) fn fuse_norm_single_traversal(
    module: &mut Module,
    do_rms: bool,
    do_res_rms: bool,
    do_ln: bool,
    do_sm: bool,
) -> (usize, usize) {
    let mut created = 0usize;
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        let mut rms_in_block = 0usize;
        let mut ln_in_block = 0usize;

        for &op_id in &ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };

            if do_rms
                && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                && let Some((x_val, weight_val, eps_f64)) = match_scaled_rmsnorm(module, lhs, rhs)
                    .or_else(|| match_scaled_rmsnorm(module, rhs, lhs))
            {
                let (m_res, c_res) = rewrite_rmsnorm_or_residual(
                    module,
                    op_id,
                    x_val,
                    Some(weight_val),
                    eps_f64,
                    do_res_rms,
                );
                modified += m_res;
                created += c_res;
                rms_in_block += 1;
                continue;
            }

            if do_ln
                && let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                && let Some((x_val, gamma_opt, beta_opt, eps)) = match_scaled_layernorm(module, lhs)
                    .map(|(x, g, eps)| (x, Some(g), Some(rhs), eps))
                    .or_else(|| {
                        match_scaled_layernorm(module, rhs)
                            .map(|(x, g, eps)| (x, Some(g), Some(lhs), eps))
                    })
            {
                rewrite_to_layernorm(module, op_id, x_val, gamma_opt, beta_opt, eps);
                created += 1;
                ln_in_block += 1;
                continue;
            }

            if do_sm {
                if let Some(StableHloOp::Divide {
                    lhs: exp_val,
                    rhs: denom_val,
                    ..
                }) = StableHloOp::from_op(&op)
                    && let ValueDef::OpResult { op: exp_op_id, .. } = module.values[exp_val].def
                    && let Some(exp_op) = module.ops.get(exp_op_id)
                    && let Some(StableHloOp::Exp {
                        operand: exp_in, ..
                    }) = StableHloOp::from_op(exp_op)
                    && let Some((reduce_axis, _, _)) =
                        match_broadcast_reduce_of_exp(module, denom_val, exp_val)
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
                    created += 1;
                    continue;
                }

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

                if let Some(StableHloOp::Subtract { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                    && let Some((x_val, axis)) = match_log_sum_exp_sub(module, lhs, rhs)
                {
                    rewrite_to_log_softmax(module, op_id, x_val, axis);
                    created += 1;
                    continue;
                }
            }
        }

        if (do_rms && rms_in_block == 0) || (do_ln && ln_in_block == 0) {
            for &op_id in &ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };

                if do_rms
                    && rms_in_block == 0
                    && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                    && let Some((x_val, eps_f64)) = match_x_and_rsqrt(module, lhs, rhs)
                        .or_else(|| match_x_and_rsqrt(module, rhs, lhs))
                {
                    let (m_res, c_res) = rewrite_rmsnorm_or_residual(
                        module, op_id, x_val, None, eps_f64, do_res_rms,
                    );
                    modified += m_res;
                    created += c_res;
                    rms_in_block += 1;
                    continue;
                }

                if do_ln && ln_in_block == 0 {
                    if let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                        && let Some((x_val, gamma_val, eps)) = match_core_layernorm_val(module, lhs)
                            .map(|(x, e)| (x, rhs, e))
                            .or_else(|| {
                                match_core_layernorm_val(module, rhs).map(|(x, e)| (x, lhs, e))
                            })
                    {
                        rewrite_to_layernorm(module, op_id, x_val, Some(gamma_val), None, eps);
                        created += 1;
                        ln_in_block += 1;
                        continue;
                    }
                    if let Some((x_val, eps)) = match_core_layernorm_op(module, &op) {
                        rewrite_to_layernorm(module, op_id, x_val, None, None, eps);
                        created += 1;
                        ln_in_block += 1;
                    }
                }
            }
        }
    }

    (created, modified)
}

fn rewrite_rmsnorm_or_residual(
    module: &mut Module,
    op_id: crate::ir::OpId,
    x_val: ValueId,
    weight_opt: Option<ValueId>,
    eps_f64: f64,
    allow_residual: bool,
) -> (usize, usize) {
    let mut attrs = IndexMap::new();
    let mut operands = SmallVec::new();
    let mut modified = 0usize;
    let mut created = 1usize;

    if allow_residual
        && let ValueDef::OpResult { op: add_op_id, .. } = module.values[x_val].def
        && let Some(add_op) = module.ops.get(add_op_id)
        && let Some(StableHloOp::Add {
            lhs: x, rhs: res, ..
        }) = StableHloOp::from_op(add_op)
    {
        operands.push(x);
        operands.push(res);
        if let Some(w) = weight_opt {
            operands.push(w);
        }
        attrs.insert(
            "name".to_string(),
            Attribute::String("castforge.fused_residual_rmsnorm".to_string()),
        );
        attrs.insert(
            "decomposition".to_string(),
            Attribute::SymbolRef("fused_residual_rmsnorm_impl".to_string()),
        );
        modified += 1;
        created += 1;
    } else {
        operands.push(x_val);
        if let Some(w) = weight_opt {
            operands.push(w);
        }
        attrs.insert(
            "name".to_string(),
            Attribute::String("castforge.fused_rmsnorm".to_string()),
        );
        attrs.insert(
            "decomposition".to_string(),
            Attribute::SymbolRef("fused_rmsnorm_impl".to_string()),
        );
    }
    attrs.insert("eps".to_string(), Attribute::Float(eps_f64));

    let op_mut = &mut module.ops[op_id];
    op_mut.name = OpName::new("stablehlo", "composite");
    op_mut.operands = operands;
    op_mut.attributes = attrs;
    (modified, created)
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
