use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Module, TensorBacking, ValueDef, ValueId,
    format_dense_f32_literal,
};

/// AI Optimization Pass: Absorbs constant normalization scale (`stablehlo.multiply %x, %scale`)
/// into an adjacent `stablehlo.dot_general` constant weight matrix
/// (`W'[k, n] = scale[k] * W[k, n]`), eliminating the runtime elementwise multiplication.
pub struct FoldNormWeightsPass;

impl Pass for FoldNormWeightsPass {
    fn name(&self) -> &'static str {
        "fold-norm-weights"
    }

    fn description(&self) -> &'static str {
        "Absorbs constant RMSNorm/LayerNorm scale multiply into adjacent dot_general weights"
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
                let Some(StableHloOp::DotGeneral {
                    lhs,
                    rhs,
                    dimension_numbers,
                    ..
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                // Standard 2D matmul contracting dim 1 of `lhs` with dim 0 of `rhs`
                if dimension_numbers.lhs_contracting_dimensions != vec![1]
                    || dimension_numbers.rhs_contracting_dimensions != vec![0]
                {
                    continue;
                }

                // `rhs` must be a single-use constant 2D f32 tensor `[K, N]`
                if module.use_count(rhs) != 1 {
                    continue;
                }
                let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs].def else {
                    continue;
                };
                let Some(w_ty) = module.values[rhs].ty.as_tensor().cloned() else {
                    continue;
                };
                if w_ty.dtype != DType::F32 || w_ty.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(k_dim), Dim::Static(n_dim)) = (&w_ty.shape[0], &w_ty.shape[1])
                else {
                    continue;
                };
                let (k, n) = (*k_dim as usize, *n_dim as usize);

                // `lhs` must be produced by `stablehlo.multiply` of `(x, scale_const)`
                let ValueDef::OpResult { op: mul_op_id, .. } = module.values[lhs].def else {
                    continue;
                };
                let Some(mul_op) = module.ops.get(mul_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Multiply {
                    lhs: mul_a,
                    rhs: mul_b,
                    ..
                }) = StableHloOp::from_op(mul_op)
                else {
                    continue;
                };

                let Some((x_val, scale_vec)) = extract_x_and_scale(module, mul_a, mul_b, k)
                    .or_else(|| extract_x_and_scale(module, mul_b, mul_a, k))
                else {
                    continue;
                };

                let Some(w_op) = module.ops.get(w_op_id) else {
                    continue;
                };
                if !w_op.is_op("stablehlo", "constant") {
                    continue;
                }
                let Some(w_attr) = w_op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(mut w_vals) = module.tensor_store.resolve_f32_attr(&w_attr, k * n) else {
                    continue;
                };

                // Absorb scale along contracting dim `k`: W'[row, col] = scale[row] * W[row, col]
                for row in 0..k {
                    let s = scale_vec[row];
                    for col in 0..n {
                        w_vals[row * n + col] *= s;
                    }
                }

                // Update `w_op` (or `TensorStore` if `dense_resource`)
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

                // Bypass `mul_op` in `dot_general`
                module.ops[op_id].operands[0] = x_val;
                modified += 2;
            }
        }

        if modified > 0 {
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

fn extract_x_and_scale(
    module: &Module,
    x_cand: ValueId,
    scale_cand: ValueId,
    k: usize,
) -> Option<(ValueId, Vec<f32>)> {
    // Allow `scale_cand` to be either a direct `stablehlo.constant` or a `broadcast_in_dim` of one
    let mut const_val = scale_cand;
    if let ValueDef::OpResult { op: b_op_id, .. } = module.values[scale_cand].def
        && let Some(b_op) = module.ops.get(b_op_id)
        && let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(b_op)
    {
        const_val = operand;
    }

    let ValueDef::OpResult { op: c_op_id, .. } = module.values[const_val].def else {
        return None;
    };
    let c_op = module.ops.get(c_op_id)?;
    if !c_op.is_op("stablehlo", "constant") {
        return None;
    }
    let attr = c_op.attributes.get("value")?;

    // Case 1: splat constant
    if let Some(splat) = attr.as_splat_f64() {
        return Some((x_cand, vec![splat as f32; k]));
    }

    let num_elems = module.values[const_val]
        .ty
        .as_tensor()
        .and_then(|t| t.num_elements())
        .unwrap_or(k as i64) as usize;
    let raw = module.tensor_store.resolve_f32_attr(attr, num_elems)?;

    if raw.len() == 1 {
        Some((x_cand, vec![raw[0]; k]))
    } else if raw.len() == k {
        Some((x_cand, raw))
    } else if raw.len() % k == 0 {
        // Check if rows are identical (e.g. `[M, K]` constant with identical rows `[1, K]`)
        let first_row = &raw[..k];
        if raw
            .chunks_exact(k)
            .all(|row| row.iter().zip(first_row).all(|(a, b)| (a - b).abs() < 1e-6))
        {
            Some((x_cand, first_row.to_vec()))
        } else {
            None
        }
    } else {
        None
    }
}
