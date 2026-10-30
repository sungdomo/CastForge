//! Compile-Time Weight Folding & Pre-Packing Pass (`fold-weights`).
//!
//! Consolidates `fold-norm-weights` (`mode=norm`), `fold-batchnorm` (`mode=batchnorm`), and
//! `pack-weights` (`mode=pack`).

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, TensorBacking, TensorEntry, TensorType, Type,
    ValueDef, ValueId, format_dense_f32_literal, format_dense_i8_literal, parse_dense_f32_literal,
};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use std::collections::HashSet;

/// Unified Compile-Time Weight Folding, Pre-Packing, Externalization & Pruning Pass
/// (`fold-weights`).
///
/// - `mode = "batchnorm"` (default): Absorbs pre-GEMM RMSNorm/LayerNorm scales and post-Conv/GEMM
///   BatchNorm scales into constant weight tensors.
/// - `mode = "norm"`: Absorbs pre-GEMM RMSNorm/LayerNorm scales only.
/// - `mode = "pack"`: Pre-packs 2D constant RHS weights from `[K, N]` to `[N, K]` (`packed_rhs =
///   true`) for unit-stride SIMD dot products.
/// - `mode = "externalize"` / `mode = "inline"`: Externalizes inline `dense<...>` constants to
///   `dense_resource<key>` in `TensorStore` (or inlines back to `dense<...>`).
/// - `mode = "prune" | "structured" | "2to4" | "activation_sparse"`: Structured L1 channel
///   pruning, NVIDIA 2:4 semi-structured sparsity, or TEAL/CATS dynamic activation sparsity.
/// - `mode = "all"`: Runs both BatchNorm/Norm folding and weight pre-packing.
pub struct FoldWeightsPass {
    pub mode: String,
    pub min_elements: usize,
    pub inline: bool,
    pub sparsity: f32,
}

impl Default for FoldWeightsPass {
    fn default() -> Self {
        Self {
            mode: "batchnorm".to_string(),
            min_elements: 2,
            inline: false,
            sparsity: 0.5,
        }
    }
}

impl Pass for FoldWeightsPass {
    fn name(&self) -> &'static str {
        "fold-weights"
    }

    fn description(&self) -> &'static str {
        "Unified weight folding, SIMD pre-packing, SafeTensors externalization & pruning"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut modified = 0usize;
        let mut removed = 0usize;
        let mut created = 0usize;

        if matches!(
            m.as_str(),
            "prune" | "structured" | "2to4" | "activation_sparse" | "activation" | "teal"
        ) {
            let prune_mode = if m == "prune" {
                "structured".to_string()
            } else {
                m
            };
            let mut r = PruneWeightsPass {
                mode: prune_mode,
                sparsity: self.sparsity,
            }
            .run(module)?;
            r.pass_name = self.name().to_string();
            return Ok(r);
        }

        if matches!(m.as_str(), "externalize" | "ext" | "inline") || self.inline {
            let r = crate::passes::ExternalizeWeightsPass {
                min_elements: self.min_elements,
                inline: self.inline || m == "inline",
            }
            .run(module)?;
            modified += r.ops_modified;
            removed += r.ops_removed;
            created += r.ops_created;
        } else if matches!(m.as_str(), "norm" | "fold_norm") {
            let r = FoldNormWeightsPass.run(module)?;
            modified += r.ops_modified;
            removed += r.ops_removed;
            created += r.ops_created;
        } else if matches!(m.as_str(), "batchnorm" | "bn" | "all") {
            let r = FoldBatchNormPass.run(module)?;
            modified += r.ops_modified;
            removed += r.ops_removed;
            created += r.ops_created;
        }

        if matches!(m.as_str(), "pack" | "pack_weights" | "all") {
            let r = PackWeightsPass.run(module)?;
            modified += r.ops_modified;
            removed += r.ops_removed;
            created += r.ops_created;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0 || created > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// Normalization Scale Weight Folding Pass (`fold-norm-weights`).
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

                if dimension_numbers.lhs_contracting_dimensions != vec![1]
                    || dimension_numbers.rhs_contracting_dimensions != vec![0]
                {
                    continue;
                }

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

                for row in 0..k {
                    let s = scale_vec[row];
                    for col in 0..n {
                        w_vals[row * n + col] *= s;
                    }
                }

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
    } else if raw.len().is_multiple_of(k) {
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

/// Inference BatchNorm & Linear Scale Folding Pass (`fold-batchnorm`).
#[derive(Default)]
pub struct FoldBatchNormPass;

impl Pass for FoldBatchNormPass {
    fn name(&self) -> &'static str {
        "fold-batchnorm"
    }

    fn description(&self) -> &'static str {
        "Absorbs BatchNorm / channel scale into preceding convolution or dot_general weights"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let pre_norm_res = FoldNormWeightsPass.run(module)?;
        let mut modified = pre_norm_res.ops_modified;
        let mut removed = pre_norm_res.ops_removed;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Multiply {
                    lhs: a,
                    rhs: b,
                    result: mul_out,
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if let Some(()) = try_fold_bn_scale(module, a, b, mul_out)
                    .or_else(|| try_fold_bn_scale(module, b, a, mul_out))
                {
                    modified += 1;
                }
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0 || pre_norm_res.changed,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn try_fold_bn_scale(
    module: &mut Module,
    linear_val: ValueId,
    scale_val: ValueId,
    mul_out: ValueId,
) -> Option<()> {
    let ValueDef::OpResult { op: lin_op_id, .. } = module.values[linear_val].def else {
        return None;
    };
    let lin_op = module.ops.get(lin_op_id)?.clone();
    let is_dot = lin_op.is_op("stablehlo", "dot_general");
    let is_conv = lin_op.is_op("stablehlo", "convolution");
    if (!is_dot && !is_conv) || lin_op.operands.len() < 2 {
        return None;
    }

    let w_val = lin_op.operands[1];
    let ValueDef::OpResult { op: w_op_id, .. } = module.values[w_val].def else {
        return None;
    };
    let w_op = module.ops.get(w_op_id)?.clone();
    if !w_op.is_op("stablehlo", "constant") {
        return None;
    }
    let w_ty = module.values[w_val].ty.as_tensor()?.clone();
    let w_elems = w_ty.num_elements()? as usize;
    let w_attr = w_op.attributes.get("value")?.clone();
    let mut w_data = module.tensor_store.resolve_f32_attr(&w_attr, w_elems)?;

    let ValueDef::OpResult { op: s_op_id, .. } = module.values[scale_val].def else {
        return None;
    };
    let s_op = module.ops.get(s_op_id)?.clone();
    let mut scale_src_val = scale_val;
    if let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(&s_op) {
        scale_src_val = operand;
    }
    let ValueDef::OpResult {
        op: sc_const_id, ..
    } = module.values[scale_src_val].def
    else {
        return None;
    };
    let sc_op = module.ops.get(sc_const_id)?.clone();
    if !sc_op.is_op("stablehlo", "constant") {
        return None;
    }
    let sc_ty = module.values[scale_src_val].ty.as_tensor()?.clone();
    let sc_elems = sc_ty.num_elements()? as usize;
    let sc_attr = sc_op.attributes.get("value")?.clone();
    let sc_data = module.tensor_store.resolve_f32_attr(&sc_attr, sc_elems)?;

    let out_channels = match w_ty.shape.last()? {
        Dim::Static(c) if *c > 0 => *c as usize,
        _ => return None,
    };
    if !w_elems.is_multiple_of(out_channels) {
        return None;
    }
    let rows = w_elems / out_channels;
    for r in 0..rows {
        for c in 0..out_channels {
            let s = sc_data[c % sc_data.len()];
            w_data[r * out_channels + c] *= s;
        }
    }

    let new_w_attr = match w_attr {
        Attribute::TensorRef { key, ty } => {
            if let Some(entry) = module.tensor_store.get_mut(&key) {
                entry.backing = TensorBacking::F32Data(w_data);
            }
            Attribute::TensorRef { key, ty }
        }
        _ => Attribute::DenseElements {
            ty: w_ty.clone(),
            raw_literal: format_dense_f32_literal(&w_data, &w_ty.shape),
        },
    };
    module.ops[w_op_id]
        .attributes
        .insert("value".to_string(), new_w_attr);
    module.ops[lin_op_id].attributes.insert(
        "castforge.batchnorm_folded".to_string(),
        Attribute::Bool(true),
    );
    module.replace_all_uses(mul_out, linear_val);
    Some(())
}

/// Compile-Time Weight Pre-Packing & Layout Transformation Pass (`pack-weights`).
pub struct PackWeightsPass;

impl Pass for PackWeightsPass {
    fn name(&self) -> &'static str {
        "pack-weights"
    }

    fn description(&self) -> &'static str {
        "Pre-packs 2D constant RHS weights from [K, N] to [N, K] for unit-stride SIMD MatMul"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut packed_const_ops: HashSet<OpId> = HashSet::new();
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };

                let is_cand = match StableHloOp::from_op(op) {
                    Some(StableHloOp::DotGeneral {
                        dimension_numbers: dims,
                        ..
                    }) => {
                        dims.lhs_batching_dimensions.is_empty()
                            && dims.rhs_batching_dimensions.is_empty()
                            && dims.lhs_contracting_dimensions == [1]
                            && dims.rhs_contracting_dimensions == [0]
                    }
                    Some(StableHloOp::Composite { name, inputs, .. })
                        if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 =>
                    {
                        let rhs_c = op
                            .attributes
                            .get("rhs_contracting_dimensions")
                            .and_then(|a| match a {
                                Attribute::DenseIntArray(v) => Some(v.as_slice()),
                                _ => None,
                            })
                            .unwrap_or(&[0]);
                        rhs_c == [0]
                    }
                    _ => false,
                };
                if !is_cand {
                    continue;
                }

                let Some(&rhs_val) = module.ops[op_id].operands.get(1) else {
                    continue;
                };
                let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs_val].def else {
                    continue;
                };

                if !packed_const_ops.contains(&w_op_id) {
                    let Some(w_op) = module.ops.get(w_op_id) else {
                        continue;
                    };
                    if !w_op.is_op("stablehlo", "constant") {
                        continue;
                    }
                    let Some(w_ty) = module.values[rhs_val].ty.as_tensor().cloned() else {
                        continue;
                    };
                    if w_ty.dtype != DType::F32 || w_ty.shape.len() != 2 {
                        continue;
                    }
                    let (Dim::Static(k_i64), Dim::Static(n_i64)) = (&w_ty.shape[0], &w_ty.shape[1])
                    else {
                        continue;
                    };
                    let (k, n) = (*k_i64 as usize, *n_i64 as usize);
                    let Some(w_attr) = w_op.attributes.get("value").cloned() else {
                        continue;
                    };
                    let Some(w_vals) = module.tensor_store.resolve_f32_attr(&w_attr, k * n) else {
                        continue;
                    };

                    let mut packed_vals = vec![0.0_f32; n * k];
                    for row_k in 0..k {
                        for col_n in 0..n {
                            packed_vals[col_n * k + row_k] = w_vals[row_k * n + col_n];
                        }
                    }

                    let packed_ty = TensorType {
                        shape: vec![Dim::Static(n as i64), Dim::Static(k as i64)],
                        dtype: DType::F32,
                        encoding: w_ty.encoding.clone(),
                    };

                    match &w_attr {
                        Attribute::TensorRef { key, .. } => {
                            if let Some(entry) = module.tensor_store.get_mut(key) {
                                entry.ty = packed_ty.clone();
                                entry.backing = TensorBacking::F32Data(packed_vals);
                            }
                            module.ops[w_op_id].attributes.insert(
                                "value".to_string(),
                                Attribute::TensorRef {
                                    key: key.clone(),
                                    ty: packed_ty.clone(),
                                },
                            );
                        }
                        _ => {
                            let raw_literal =
                                format_dense_f32_literal(&packed_vals, &packed_ty.shape);
                            module.ops[w_op_id].attributes.insert(
                                "value".to_string(),
                                Attribute::DenseElements {
                                    ty: packed_ty.clone(),
                                    raw_literal,
                                },
                            );
                        }
                    }
                    module.values[rhs_val].ty = Type::Tensor(packed_ty);
                    packed_const_ops.insert(w_op_id);
                    modified += 1;
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert(
                    "lhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![1]),
                );
                op_mut.attributes.insert(
                    "rhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![1]),
                );
                op_mut
                    .attributes
                    .insert("packed_rhs".to_string(), Attribute::Bool(true));
                modified += 1;
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

/// Externalize / Inline Weights Pass (`externalize-weights`).
pub struct ExternalizeWeightsPass {
    pub min_elements: usize,
    pub inline: bool,
}

impl Default for ExternalizeWeightsPass {
    fn default() -> Self {
        Self {
            min_elements: 2,
            inline: false,
        }
    }
}

impl Pass for ExternalizeWeightsPass {
    fn name(&self) -> &'static str {
        "externalize-weights"
    }

    fn description(&self) -> &'static str {
        "Converts inline dense<...> weights to dense_resource<key> in TensorStore (or inline=true)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_op("stablehlo", "constant") || op.results.len() != 1 {
                    continue;
                }
                let val_id = op.results[0];
                let Some(tensor_ty) = module.values[val_id].ty.as_tensor().cloned() else {
                    continue;
                };
                let num_elems = tensor_ty.num_elements().unwrap_or(1) as usize;
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };

                if self.inline {
                    if let Attribute::TensorRef { .. } = &val_attr {
                        let raw_literal = if matches!(tensor_ty.dtype, DType::QuantUniform { .. }) {
                            let Some(q_vals) =
                                module.tensor_store.resolve_i8_attr(&val_attr, num_elems)
                            else {
                                continue;
                            };
                            format_dense_i8_literal(&q_vals, &tensor_ty.shape)
                        } else {
                            let Some(f_vals) =
                                module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                            else {
                                continue;
                            };
                            format_dense_f32_literal(&f_vals, &tensor_ty.shape)
                        };
                        module.ops[op_id].attributes.insert(
                            "value".to_string(),
                            Attribute::DenseElements {
                                ty: tensor_ty,
                                raw_literal,
                            },
                        );
                        modified += 1;
                    }
                } else if let Attribute::DenseElements { raw_literal, .. } = &val_attr {
                    if num_elems < self.min_elements {
                        continue;
                    }
                    let Some(vals) = parse_dense_f32_literal(raw_literal, num_elems) else {
                        continue;
                    };
                    let key = module.values[val_id]
                        .ssa_name
                        .clone()
                        .unwrap_or_else(|| format!("weight_{modified}"));
                    module.tensor_store.insert(TensorEntry::from_f32(
                        key.clone(),
                        tensor_ty.clone(),
                        vals,
                    ));
                    module.ops[op_id].attributes.insert(
                        "value".to_string(),
                        Attribute::TensorRef { key, ty: tensor_ty },
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

/// Unified Weight Pruning & Structured Sparsity Pass (`prune-weights`).
pub struct PruneWeightsPass {
    pub mode: String,
    pub sparsity: f32,
}

impl Default for PruneWeightsPass {
    fn default() -> Self {
        Self {
            mode: "structured".to_string(),
            sparsity: 0.5,
        }
    }
}

impl Pass for PruneWeightsPass {
    fn name(&self) -> &'static str {
        "prune-weights"
    }

    fn description(&self) -> &'static str {
        "Unified weight & activation sparsity (mode=structured|2to4|activation_sparse)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut res = if m == "2to4" {
            Sparsify2To4Pass.run(module)?
        } else if matches!(m.as_str(), "activation_sparse" | "activation" | "teal") {
            run_activation_sparsity(module, self.sparsity)?
        } else {
            StructuredPruningPass {
                sparsity: self.sparsity,
            }
            .run(module)?
        };
        res.pass_name = self.name().to_string();
        Ok(res)
    }
}

fn run_activation_sparsity(module: &mut Module, sparsity: f32) -> Result<PassResult> {
    let ratio = sparsity.clamp(0.05, 0.95);
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();
    for block_id in blocks {
        for op_id in module.block_ops(block_id) {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let is_gemm = matches!(
                StableHloOp::from_op(op),
                Some(StableHloOp::DotGeneral { .. })
            ) || (op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n.starts_with("castforge.fused_")));
            if !is_gemm {
                continue;
            }
            let attrs = &mut module.ops[op_id].attributes;
            attrs.insert(
                "castforge.activation_sparsity".to_string(),
                Attribute::Float(ratio as f64),
            );
            attrs.insert(
                "castforge.sparse_activation_mode".to_string(),
                Attribute::String("top_k_magnitude".to_string()),
            );
            modified += 1;
        }
    }
    Ok(PassResult {
        pass_name: "prune-weights".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

/// Structured Channel Pruning Pass (`prune-structured`).
pub struct StructuredPruningPass {
    pub sparsity: f32,
}

impl Default for StructuredPruningPass {
    fn default() -> Self {
        Self { sparsity: 0.5 }
    }
}

impl Pass for StructuredPruningPass {
    fn name(&self) -> &'static str {
        "prune-structured"
    }

    fn description(&self) -> &'static str {
        "Structured L1 channel pruning on 2D weight matrices in dot_general (sparsity=0.5)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    rhs,
                    result: dot_res,
                    ..
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let ValueDef::OpResult {
                    op: weight_op_id, ..
                } = module.values[rhs].def
                else {
                    continue;
                };
                let Some(weight_op) = module.ops.get(weight_op_id) else {
                    continue;
                };
                if !weight_op.is_op("stablehlo", "constant") {
                    continue;
                }

                let Some(w_ty) = module.values[rhs].ty.as_tensor().cloned() else {
                    continue;
                };
                if w_ty.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(rows_i64), Dim::Static(cols_i64)) =
                    (&w_ty.shape[0], &w_ty.shape[1])
                else {
                    continue;
                };
                let rows = *rows_i64 as usize;
                let cols = *cols_i64 as usize;
                if cols <= 1 {
                    continue;
                }

                let keep_cols =
                    ((cols as f32) * (1.0 - self.sparsity.clamp(0.0, 0.9))).round() as usize;
                let keep_cols = keep_cols.clamp(1, cols);
                if keep_cols == cols {
                    continue;
                }

                let Some(val_attr) = weight_op.attributes.get("value").cloned() else {
                    continue;
                };
                let resource_key = match &val_attr {
                    Attribute::TensorRef { key, .. } => Some(key.clone()),
                    _ => None,
                };
                let Some(f32_vals) = module.tensor_store.resolve_f32_attr(&val_attr, rows * cols)
                else {
                    continue;
                };

                let (pruned_vals, _kept_cols) =
                    prune_2d_columns_by_l1(&f32_vals, rows, cols, keep_cols);

                let new_w_ty = TensorType {
                    shape: vec![Dim::Static(rows as i64), Dim::Static(keep_cols as i64)],
                    dtype: w_ty.dtype.clone(),
                    encoding: None,
                };
                let new_literal = format_dense_f32_literal(&pruned_vals, &new_w_ty.shape);

                let store_key = resource_key.clone().unwrap_or_else(|| {
                    module.values[rhs]
                        .ssa_name
                        .clone()
                        .unwrap_or_else(|| "pruned_weight".to_string())
                });

                module.values[rhs].ty = Type::Tensor(new_w_ty.clone());
                let updated_attr = if let Some(key) = resource_key {
                    Attribute::TensorRef {
                        key,
                        ty: new_w_ty.clone(),
                    }
                } else {
                    Attribute::DenseElements {
                        ty: new_w_ty.clone(),
                        raw_literal: new_literal,
                    }
                };
                module.ops[weight_op_id]
                    .attributes
                    .insert("value".to_string(), updated_attr);
                module
                    .tensor_store
                    .insert(TensorEntry::from_f32(store_key, new_w_ty, pruned_vals));

                if let Some(mut dot_out_ty) = module.values[dot_res].ty.as_tensor().cloned() {
                    if let Some(last_dim) = dot_out_ty.shape.last_mut() {
                        *last_dim = Dim::Static(keep_cols as i64);
                    }
                    let updated_out_ty = Type::Tensor(dot_out_ty);
                    module.values[dot_res].ty = updated_out_ty.clone();

                    for fn_op_id in module.functions() {
                        if let Some(Attribute::Type(Type::Function { inputs, results })) = module
                            .ops[fn_op_id]
                            .attributes
                            .get("function_type")
                            .cloned()
                            && results.len() == 1
                        {
                            module.ops[fn_op_id].attributes.insert(
                                "function_type".to_string(),
                                Attribute::Type(Type::Function {
                                    inputs,
                                    results: vec![updated_out_ty.clone()],
                                }),
                            );
                        }
                    }
                }

                modified += 1;
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

/// NVIDIA 2:4 Semi-Structured Weight Sparsifier Pass (`sparsify-2to4`).
#[derive(Debug, Clone, Default)]
pub struct Sparsify2To4Pass;

impl Pass for Sparsify2To4Pass {
    fn name(&self) -> &'static str {
        "sparsify-2to4"
    }

    fn description(&self) -> &'static str {
        "Enforces 2:4 semi-structured sparsity on weight tensors and annotates dot_general ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut rhs_weights: HashSet<ValueId> = HashSet::new();

        for (_, op) in module.ops.iter_mut() {
            if op.is_op("stablehlo", "dot_general") && op.operands.len() >= 2 {
                rhs_weights.insert(op.operands[1]);
                if op.attributes.get("sparsity").and_then(|a| a.as_str()) != Some("2:4") {
                    op.attributes
                        .insert("sparsity".to_string(), Attribute::String("2:4".to_string()));
                    modified += 1;
                }
            }
        }

        let mut const_ops = Vec::new();
        for &vid in &rhs_weights {
            if let ValueDef::OpResult { op: prod_op, .. } = module.values[vid].def
                && module.ops[prod_op].is_op("stablehlo", "constant")
            {
                const_ops.push((prod_op, vid));
            }
        }

        for (op_id, vid) in const_ops {
            let Some(t_ty) = module.values[vid].ty.as_tensor().cloned() else {
                continue;
            };
            let n = t_ty.num_elements().unwrap_or(1).max(1) as usize;
            let Some(val_attr) = module.ops[op_id].attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut vals) = module.tensor_store.resolve_f32_attr(&val_attr, n) else {
                continue;
            };
            apply_2to4_mask(&mut vals);

            match val_attr {
                Attribute::TensorRef { key, ty } => {
                    module.tensor_store.insert(TensorEntry::from_f32(
                        key.clone(),
                        TensorType::new(ty.shape.clone(), DType::F32),
                        vals,
                    ));
                    modified += 1;
                }
                Attribute::DenseElements { ty, .. } => {
                    let lit = format_dense_f32_literal(&vals, &ty.shape);
                    module.ops[op_id].attributes.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty,
                            raw_literal: lit,
                        },
                    );
                    modified += 1;
                }
                _ => {}
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

/// Zeroes out the 2 smallest-magnitude elements in every contiguous 4-element chunk of `vals`.
pub fn apply_2to4_mask(vals: &mut [f32]) {
    for chunk in vals.chunks_mut(4) {
        if chunk.len() < 4 {
            continue;
        }
        let mut indexed: [(usize, f32); 4] = [
            (0, chunk[0].abs()),
            (1, chunk[1].abs()),
            (2, chunk[2].abs()),
            (3, chunk[3].abs()),
        ];
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        chunk[indexed[0].0] = 0.0;
        chunk[indexed[1].0] = 0.0;
    }
}

/// Structured pruning of a 2D weight matrix `[rows, cols]` along its output columns (`dim = 1`),
/// keeping the top `keep_cols` columns with highest L1 importance norm.
/// Returns `(pruned_f32_data, kept_column_indices)`.
pub fn prune_2d_columns_by_l1(
    values: &[f32],
    rows: usize,
    cols: usize,
    keep_cols: usize,
) -> (Vec<f32>, Vec<usize>) {
    if rows == 0 || cols == 0 || values.len() < rows * cols {
        return (values.to_vec(), (0..cols).collect());
    }
    let keep_cols = keep_cols.clamp(1, cols);
    let mut col_scores: Vec<(usize, f32)> = (0..cols)
        .map(|c| {
            let mut l1 = 0.0f32;
            for r in 0..rows {
                l1 += values[r * cols + c].abs();
            }
            (c, l1)
        })
        .collect();

    col_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept_indices: Vec<usize> = col_scores
        .into_iter()
        .take(keep_cols)
        .map(|(idx, _)| idx)
        .collect();
    kept_indices.sort_unstable();

    let mut pruned = Vec::with_capacity(rows * keep_cols);
    for r in 0..rows {
        for &c in &kept_indices {
            pruned.push(values[r * cols + c]);
        }
    }
    (pruned, kept_indices)
}
