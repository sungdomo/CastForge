//! LoRA (`bake-lora`) and Weight-Decomposed Low-Rank Adaptation (`bake-dora`) baking passes.

use crate::ir::{Attribute, BlockId, DType, Dim, Module, ValueDef, ValueId};
use crate::passes::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;

/// LoRA Adapter Baking Pass (`bake-lora`).
pub struct BakeLoraPass {
    pub alpha: f32,
}

impl Default for BakeLoraPass {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Pass for BakeLoraPass {
    fn name(&self) -> &'static str {
        "bake-lora"
    }

    fn description(&self) -> &'static str {
        "Fuses LoRA adapter branches W + alpha * (A @ B) into base weight matrices at compile time"
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
                let Some(StableHloOp::Add {
                    lhs,
                    rhs,
                    result: add_res,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let matched = match_base_and_lora(module, lhs, rhs, self.alpha)
                    .or_else(|| match_base_and_lora(module, rhs, lhs, self.alpha));
                let Some(LoraMatch {
                    base_dot_val,
                    w_op_id,
                    a_op_id,
                    b_op_id,
                    k,
                    r,
                    n,
                    eff_alpha,
                }) = matched
                else {
                    continue;
                };

                let (Some(w_attr), Some(a_attr), Some(b_attr)) = (
                    module.ops[w_op_id].attributes.get("value").cloned(),
                    module.ops[a_op_id].attributes.get("value").cloned(),
                    module.ops[b_op_id].attributes.get("value").cloned(),
                ) else {
                    continue;
                };

                let (Some(mut w_vals), Some(a_vals), Some(b_vals)) = (
                    module.tensor_store.resolve_f32_attr(&w_attr, k * n),
                    module.tensor_store.resolve_f32_attr(&a_attr, k * r),
                    module.tensor_store.resolve_f32_attr(&b_attr, r * n),
                ) else {
                    continue;
                };

                for row in 0..k {
                    for col in 0..n {
                        let mut delta = 0.0_f32;
                        for p in 0..r {
                            delta += a_vals[row * r + p] * b_vals[p * n + col];
                        }
                        w_vals[row * n + col] += eff_alpha * delta;
                    }
                }

                let Some(w_res_val) = module.ops[w_op_id].results.first().copied() else {
                    continue;
                };
                let Some(w_ty) = module.values[w_res_val].ty.as_tensor().cloned() else {
                    continue;
                };
                let ssa = module.values[w_res_val].ssa_name.clone();
                let new_w_attr =
                    module
                        .tensor_store
                        .update_f32_constant(ssa.as_deref(), &w_attr, w_ty, w_vals);
                module.ops[w_op_id]
                    .attributes
                    .insert("value".to_string(), new_w_attr);

                module.replace_all_uses(add_res, base_dot_val);
                module.remove_op(block_id, op_id);
                modified += 1;
                removed += 1;
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

struct LoraMatch {
    base_dot_val: ValueId,
    w_op_id: crate::ir::OpId,
    a_op_id: crate::ir::OpId,
    b_op_id: crate::ir::OpId,
    k: usize,
    r: usize,
    n: usize,
    eff_alpha: f32,
}

fn match_base_and_lora(
    module: &Module,
    base_cand: ValueId,
    lora_cand: ValueId,
    default_alpha: f32,
) -> Option<LoraMatch> {
    if module.use_count(base_cand) != 1 {
        return None;
    }
    let ValueDef::OpResult { op: base_op_id, .. } = module.values[base_cand].def else {
        return None;
    };
    let base_op = module.ops.get(base_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: x_base,
        rhs: w_val,
        ..
    } = StableHloOp::from_op(base_op)?
    else {
        return None;
    };
    if module.use_count(w_val) != 1 {
        return None;
    }
    let (w_op_id, k, n) = const_2d_dims(module, w_val)?;

    let mut dot_b_val = lora_cand;
    let mut eff_alpha = default_alpha;
    if let ValueDef::OpResult { op: m_id, .. } = module.values[lora_cand].def
        && let Some(m_op) = module.ops.get(m_id)
        && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(m_op)
    {
        if let Some(s) = FoldConstantsPass::const_splat(module, rhs) {
            dot_b_val = lhs;
            eff_alpha = s as f32;
        } else if let Some(s) = FoldConstantsPass::const_splat(module, lhs) {
            dot_b_val = rhs;
            eff_alpha = s as f32;
        }
    }

    let ValueDef::OpResult {
        op: dot_b_op_id, ..
    } = module.values[dot_b_val].def
    else {
        return None;
    };
    let dot_b_op = module.ops.get(dot_b_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: dot_a_val,
        rhs: b_val,
        ..
    } = StableHloOp::from_op(dot_b_op)?
    else {
        return None;
    };
    let (b_op_id, r_b, n_b) = const_2d_dims(module, b_val)?;
    if n_b != n {
        return None;
    }

    let ValueDef::OpResult {
        op: dot_a_op_id, ..
    } = module.values[dot_a_val].def
    else {
        return None;
    };
    let dot_a_op = module.ops.get(dot_a_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: x_lora,
        rhs: a_val,
        ..
    } = StableHloOp::from_op(dot_a_op)?
    else {
        return None;
    };
    if x_lora != x_base {
        return None;
    }
    let (a_op_id, k_a, r_a) = const_2d_dims(module, a_val)?;
    if k_a != k || r_a != r_b {
        return None;
    }

    Some(LoraMatch {
        base_dot_val: base_cand,
        w_op_id,
        a_op_id,
        b_op_id,
        k,
        r: r_a,
        n,
        eff_alpha,
    })
}

fn const_2d_dims(module: &Module, val_id: ValueId) -> Option<(crate::ir::OpId, usize, usize)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if !op.is_op("stablehlo", "constant") {
        return None;
    }
    let t = module.values[val_id].ty.as_tensor()?;
    if t.dtype != DType::F32 || t.shape.len() != 2 {
        return None;
    }
    let (Dim::Static(r), Dim::Static(c)) = (&t.shape[0], &t.shape[1]) else {
        return None;
    };
    Some((op_id, *r as usize, *c as usize))
}

/// Weight-Decomposed Low-Rank Adaptation (DoRA) Baking Pass (`bake-dora`).
pub struct BakeDoraPass {
    pub alpha: f32,
}

impl Default for BakeDoraPass {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Pass for BakeDoraPass {
    fn name(&self) -> &'static str {
        "bake-dora"
    }

    fn description(&self) -> &'static str {
        "Bakes DoRA adapters via column-wise directional normalization and magnitude rescaling"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(run_bake_dora(module, self.alpha)?.with_name(self.name()))
    }
}

fn run_bake_dora(module: &mut Module, alpha: f32) -> Result<PassResult> {
    let mut pre_mags: std::collections::HashMap<crate::ir::OpId, Vec<f32>> =
        std::collections::HashMap::new();
    for op_id in module.live_ops() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
            continue;
        }
        let res_id = op.results[0];
        let Some(t_ty) = module.values[res_id].ty.as_tensor() else {
            continue;
        };
        if t_ty.dtype != DType::F32 || t_ty.shape.len() != 2 {
            continue;
        }
        let (Dim::Static(k), Dim::Static(n)) = (&t_ty.shape[0], &t_ty.shape[1]) else {
            continue;
        };
        let (k, n) = (*k as usize, *n as usize);
        if k == 0 || n == 0 {
            continue;
        }
        if let Some(val_attr) = op.attributes.get("value")
            && let Some(vals) = module.tensor_store.resolve_f32_attr(val_attr, k * n)
        {
            let mut mags = Vec::with_capacity(n);
            for col in 0..n {
                let mut sq_sum = 0.0_f32;
                for row in 0..k {
                    let v = vals[row * n + col];
                    sq_sum += v * v;
                }
                mags.push(sq_sum.sqrt().max(1e-8));
            }
            pre_mags.insert(op_id, mags);
        }
    }

    let lora_res = BakeLoraPass { alpha }.run(module)?;
    let mut modified = lora_res.ops_modified;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "constant")
                || op.results.is_empty()
                || op
                    .attributes
                    .get("castforge.dora_baked")
                    .and_then(|a| a.as_bool())
                    == Some(true)
            {
                continue;
            }
            let res_id = op.results[0];
            let Some(t_ty) = module.values[res_id].ty.as_tensor().cloned() else {
                continue;
            };
            if t_ty.dtype != DType::F32 || t_ty.shape.len() != 2 {
                continue;
            }
            let (Dim::Static(k), Dim::Static(n)) = (&t_ty.shape[0], &t_ty.shape[1]) else {
                continue;
            };
            let (k, n) = (*k as usize, *n as usize);
            if k == 0 || n == 0 {
                continue;
            }
            let Some(val_attr) = op.attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut vals) = module.tensor_store.resolve_f32_attr(&val_attr, k * n) else {
                continue;
            };

            let orig_mags = pre_mags.get(&op_id);
            let mut col_norms = Vec::with_capacity(n);
            for col in 0..n {
                let mut sq_sum = 0.0_f32;
                for row in 0..k {
                    let v = vals[row * n + col];
                    sq_sum += v * v;
                }
                let norm = sq_sum.sqrt().max(1e-8);
                let mag = orig_mags.and_then(|m| m.get(col).copied()).unwrap_or(norm);
                col_norms.push(mag as f64);
                // DoRA directional normalization + pre-LoRA magnitude rescaling:
                // W'_{:, j} = m_j * (V_{:, j} / ||V_{:, j}||_2)
                for row in 0..k {
                    let dir = vals[row * n + col] / norm;
                    vals[row * n + col] = mag * dir;
                }
            }

            let ssa = module.values[res_id].ssa_name.clone();
            let new_val_attr =
                module
                    .tensor_store
                    .update_f32_constant(ssa.as_deref(), &val_attr, t_ty, vals);
            let op_mut = &mut module.ops[op_id];
            op_mut.attributes.insert("value".to_string(), new_val_attr);
            op_mut
                .attributes
                .insert("castforge.dora_baked".to_string(), Attribute::Bool(true));
            op_mut.attributes.insert(
                "castforge.dora_col_norms".to_string(),
                Attribute::DenseF64Array(col_norms.clone()),
            );
            for (_, user_op) in module.ops.iter_mut() {
                if user_op.is_op("stablehlo", "dot_general") && user_op.operands.contains(&res_id) {
                    user_op
                        .attributes
                        .insert("castforge.dora_baked".to_string(), Attribute::Bool(true));
                    user_op.attributes.insert(
                        "castforge.dora_col_norms".to_string(),
                        Attribute::DenseF64Array(col_norms.clone()),
                    );
                }
            }
            modified += 1;
        }
    }

    Ok(PassResult {
        pass_name: "merge-weights".to_string(),
        changed: modified > 0 || lora_res.changed,
        ops_removed: lora_res.ops_removed,
        ops_modified: modified,
        ops_created: lora_res.ops_created,
    })
}
