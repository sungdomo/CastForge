use crate::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Module, TensorBacking, ValueDef, ValueId,
    format_dense_f32_literal,
};

/// Weight Surgery Pass: Fuses Low-Rank Adapter (LoRA) residual branches
/// `add(dot_general(x, W), multiply(dot_general(dot_general(x, A), B), alpha))`
/// (or unscaled `add(dot_general(x, W), dot_general(dot_general(x, A), B))`)
/// directly into the base weight matrix `W' = W + alpha * (A @ B)` at compile time.
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
                let Some((base_dot_val, w_op_id, a_op_id, b_op_id, k, r, n, eff_alpha)) = matched
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

                // Compute W'[row, col] = W[row, col] + eff_alpha * sum_p(A[row, p] * B[p, col])
                for row in 0..k {
                    for col in 0..n {
                        let mut delta = 0.0_f32;
                        for p in 0..r {
                            delta += a_vals[row * r + p] * b_vals[p * n + col];
                        }
                        w_vals[row * n + col] += eff_alpha * delta;
                    }
                }

                // Update W in-place (either in TensorStore or DenseElements)
                let Some(w_res_val) = module.ops[w_op_id].results.first().copied() else {
                    continue;
                };
                let Some(w_ty) = module.values[w_res_val].ty.as_tensor().cloned() else {
                    continue;
                };
                match &w_attr {
                    Attribute::TensorRef { key, .. } => {
                        if let Some(entry) = module.tensor_store.get_mut(key) {
                            entry.backing = TensorBacking::F32Data(w_vals);
                        }
                    }
                    _ => {
                        let raw_literal = format_dense_f32_literal(&w_vals, &w_ty.shape);
                        module.ops[w_op_id].attributes.insert(
                            "value".to_string(),
                            Attribute::DenseElements {
                                ty: w_ty,
                                raw_literal,
                            },
                        );
                    }
                }

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

fn match_base_and_lora(
    module: &Module,
    base_cand: ValueId,
    lora_cand: ValueId,
    default_alpha: f32,
) -> Option<(
    ValueId,
    castforge_ir::OpId,
    castforge_ir::OpId,
    castforge_ir::OpId,
    usize,
    usize,
    usize,
    f32,
)> {
    // 1. `base_cand` must be `dot_general(x, W)` where `W` is a 2D constant `[K, N]`
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
    let (w_op_id, k, n) = const_2d_dims(module, w_val)?;

    // 2. `lora_cand` may be `multiply(dot_b, alpha_const)` or directly `dot_b`
    let mut dot_b_val = lora_cand;
    let mut eff_alpha = default_alpha;
    if let ValueDef::OpResult { op: m_id, .. } = module.values[lora_cand].def {
        if let Some(m_op) = module.ops.get(m_id) {
            if let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(m_op) {
                if let Some(s) = FoldConstantsPass::const_splat(module, rhs) {
                    dot_b_val = lhs;
                    eff_alpha = s as f32;
                } else if let Some(s) = FoldConstantsPass::const_splat(module, lhs) {
                    dot_b_val = rhs;
                    eff_alpha = s as f32;
                }
            }
        }
    }

    // 3. `dot_b_val` must be `dot_general(dot_a_val, B)` where `B` is a 2D constant `[R, N]`
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

    // 4. `dot_a_val` must be `dot_general(x_lora, A)` where `x_lora == x_base` and `A` is `[K, R]`
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

    Some((base_cand, w_op_id, a_op_id, b_op_id, k, r_a, n, eff_alpha))
}

fn const_2d_dims(module: &Module, val_id: ValueId) -> Option<(castforge_ir::OpId, usize, usize)> {
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
