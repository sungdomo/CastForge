//! Compile-Time Weight Folding & Pre-Packing Pass (`fold-weights`).
//!
//! Consolidates `fold-norm-weights` (`mode=norm`), `fold-batchnorm` (`mode=batchnorm`), and
//! `pack-weights` (`mode=pack`).

pub mod prune;

pub use prune::*;

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, TensorType, Type, ValueDef, ValueId,
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

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "fold-norm-weights",
            "fold-batchnorm",
            "pack-weights",
            "externalize-weights",
            "prune-weights",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        if matches!(
            m.as_str(),
            "prune" | "structured" | "2to4" | "activation_sparse" | "activation" | "teal"
        ) {
            let prune_mode = if m == "prune" {
                "structured".to_string()
            } else {
                m
            };
            return Ok(PruneWeightsPass {
                mode: prune_mode,
                sparsity: self.sparsity,
            }
            .run(module)?
            .with_name(self.name()));
        }

        if matches!(m.as_str(), "externalize" | "ext" | "inline") || self.inline {
            return Ok(crate::passes::ExternalizeWeightsPass {
                min_elements: self.min_elements,
                inline: self.inline || m == "inline",
            }
            .run(module)?
            .with_name(self.name()));
        }

        let mut steps = Vec::new();
        if matches!(m.as_str(), "norm" | "fold_norm") {
            steps.push(FoldNormWeightsPass.run(module)?);
        } else if matches!(m.as_str(), "batchnorm" | "bn" | "all") {
            steps.push(FoldBatchNormPass.run(module)?);
        }
        if matches!(m.as_str(), "pack" | "pack_weights" | "all") {
            steps.push(PackWeightsPass.run(module)?);
        }
        Ok(PassResult::aggregate(self.name(), &steps))
    }
}

fn fold_weights_single_traversal(
    module: &mut Module,
    do_norm: bool,
    do_bn: bool,
    do_pack: bool,
) -> Result<(usize, usize)> {
    let mut modified = 0usize;
    let mut removed = 0usize;
    let mut packed_const_ops: HashSet<OpId> = HashSet::new();
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            if do_norm && try_fold_norm_weight_at_op(module, op_id) {
                modified += 2;
            }
            if do_bn && try_fold_bn_at_op(module, op_id) {
                modified += 1;
            }
            if do_pack {
                modified += try_pack_weight_at_op(module, op_id, &mut packed_const_ops);
            }
        }
    }

    if (do_norm || do_bn) && modified > 0 {
        let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
        removed += dce_res.ops_removed;
    }

    Ok((modified, removed))
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
        let (modified, removed) = fold_weights_single_traversal(module, true, false, false)?;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn try_fold_norm_weight_at_op(module: &mut Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::DotGeneral {
        lhs,
        rhs,
        dimension_numbers,
        ..
    }) = StableHloOp::from_op(op)
    else {
        return false;
    };

    if dimension_numbers.lhs_contracting_dimensions != vec![1]
        || dimension_numbers.rhs_contracting_dimensions != vec![0]
    {
        return false;
    }

    if module.use_count(rhs) != 1 {
        return false;
    }
    let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs].def else {
        return false;
    };
    let Some(w_ty) = module.values[rhs].ty.as_tensor().cloned() else {
        return false;
    };
    if w_ty.dtype != DType::F32 || w_ty.shape.len() != 2 {
        return false;
    }
    let (Dim::Static(k_dim), Dim::Static(n_dim)) = (&w_ty.shape[0], &w_ty.shape[1]) else {
        return false;
    };
    let (k, n) = (*k_dim as usize, *n_dim as usize);

    let ValueDef::OpResult { op: mul_op_id, .. } = module.values[lhs].def else {
        return false;
    };
    let Some(mul_op) = module.ops.get(mul_op_id) else {
        return false;
    };
    let Some(StableHloOp::Multiply {
        lhs: mul_a,
        rhs: mul_b,
        ..
    }) = StableHloOp::from_op(mul_op)
    else {
        return false;
    };

    let Some((x_val, scale_vec)) = extract_x_and_scale(module, mul_a, mul_b, k)
        .or_else(|| extract_x_and_scale(module, mul_b, mul_a, k))
    else {
        return false;
    };

    let Some(w_op) = module.ops.get(w_op_id) else {
        return false;
    };
    if !w_op.is_op("stablehlo", "constant") {
        return false;
    }
    let Some(w_attr) = w_op.attributes.get("value").cloned() else {
        return false;
    };
    let Some(mut w_vals) = module.tensor_store.resolve_f32_attr(&w_attr, k * n) else {
        return false;
    };

    for row in 0..k {
        let s = scale_vec[row];
        for col in 0..n {
            w_vals[row * n + col] *= s;
        }
    }

    let ssa = module.values[rhs].ssa_name.clone();
    let new_w_attr = module
        .tensor_store
        .update_f32_constant(ssa.as_deref(), &w_attr, w_ty, w_vals);
    module.ops[w_op_id]
        .attributes
        .insert("value".to_string(), new_w_attr);

    module.ops[op_id].operands[0] = x_val;
    true
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
        let (modified, removed) = fold_weights_single_traversal(module, true, true, false)?;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn try_fold_bn_at_op(module: &mut Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return false;
    };
    let Some(StableHloOp::Multiply {
        lhs: a,
        rhs: b,
        result: mul_out,
    }) = StableHloOp::from_op(&op)
    else {
        return false;
    };
    try_fold_bn_scale(module, a, b, mul_out)
        .or_else(|| try_fold_bn_scale(module, b, a, mul_out))
        .is_some()
}

fn try_fold_bn_scale(
    module: &mut Module,
    linear_val: ValueId,
    scale_val: ValueId,
    mul_out: ValueId,
) -> Option<()> {
    if module.use_count(linear_val) != 1 {
        return None;
    }
    let ValueDef::OpResult { op: lin_op_id, .. } = module.values[linear_val].def else {
        return None;
    };
    let lin_op = module.ops.get(lin_op_id)?.clone();
    let is_dot = lin_op.is_op("stablehlo", "dot_general");
    let is_conv = lin_op.is_op("stablehlo", "convolution");
    if (!is_dot && !is_conv) || lin_op.operands.len() < 2 {
        return None;
    }
    let packed_rhs = matches!(
        lin_op.attributes.get("packed_rhs"),
        Some(Attribute::Bool(true))
    );

    let w_val = lin_op.operands[1];
    if module.use_count(w_val) != 1 {
        return None;
    }
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

    if packed_rhs && w_ty.shape.len() == 2 {
        let out_channels = match w_ty.shape.first()? {
            Dim::Static(c) if *c > 0 => *c as usize,
            _ => return None,
        };
        let k_cols = w_elems / out_channels;
        for c in 0..out_channels {
            let s = sc_data[c % sc_data.len()];
            for r in 0..k_cols {
                w_data[c * k_cols + r] *= s;
            }
        }
    } else {
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
    }

    let ssa = module.values[w_val].ssa_name.clone();
    let new_w_attr = module
        .tensor_store
        .update_f32_constant(ssa.as_deref(), &w_attr, w_ty, w_data);
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
        let (modified, _) = fold_weights_single_traversal(module, false, false, true)?;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn try_pack_weight_at_op(
    module: &mut Module,
    op_id: OpId,
    packed_const_ops: &mut HashSet<OpId>,
) -> usize {
    let Some(op) = module.ops.get(op_id) else {
        return 0;
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
        return 0;
    }

    let Some(&rhs_val) = module.ops[op_id].operands.get(1) else {
        return 0;
    };
    let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs_val].def else {
        return 0;
    };

    let mut modified = 0usize;
    if !packed_const_ops.contains(&w_op_id) {
        let Some(w_op) = module.ops.get(w_op_id) else {
            return 0;
        };
        if !w_op.is_op("stablehlo", "constant") {
            return 0;
        }
        let Some(w_ty) = module.values[rhs_val].ty.as_tensor().cloned() else {
            return 0;
        };
        if w_ty.dtype != DType::F32 || w_ty.shape.len() != 2 {
            return 0;
        }
        let (Dim::Static(k_i64), Dim::Static(n_i64)) = (&w_ty.shape[0], &w_ty.shape[1]) else {
            return 0;
        };
        let (k, n) = (*k_i64 as usize, *n_i64 as usize);
        let Some(w_attr) = w_op.attributes.get("value").cloned() else {
            return 0;
        };
        let Some(w_vals) = module.tensor_store.resolve_f32_attr(&w_attr, k * n) else {
            return 0;
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

        let ssa = module.values[rhs_val].ssa_name.clone();
        let new_w_attr = module.tensor_store.update_f32_constant(
            ssa.as_deref(),
            &w_attr,
            packed_ty.clone(),
            packed_vals,
        );
        module.ops[w_op_id]
            .attributes
            .insert("value".to_string(), new_w_attr);
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
    modified
}
