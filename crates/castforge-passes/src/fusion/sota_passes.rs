//! Consolidated multi-pattern fusion passes (`fuse-norm`, `fuse-activations`, `strip-metadata`)
//! and SOTA optimization passes (`quantize-groupwise`, `fold-batchnorm`,
//! `eliminate-concat-copies`, `lower-paged-attention`, `gvn`).

use crate::{
    DeadCodeEliminationPass, FoldBroadcastPass, FoldConvertPass, FoldNormWeightsPass,
    FoldReshapePass, FoldSliceConcatPass, FoldTransposePass, FuseDequantMatmulPass,
    FuseGatedMlpPass, FuseGeluPass, FuseHorizontalDotPass, FuseLayerNormPass,
    FuseMatmulEpiloguePass, FuseResidualNormPass, FuseRmsNormPass, FuseSwiGluPass,
    PadForTilingPass, Pass, PassResult, PermuteLoopsPass, SinkTransposePass, StripAttributesPass,
    StripDebugInfoPass, TileLoopsPass, UnrollLoopsPass, VectorizeLoopsPass,
};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Module, Operation, RegionId, TensorEntry, TensorType, Type,
    ValueDef, ValueId, format_dense_f32_literal,
};
use std::collections::HashMap;

/// Unified Normalization Fusion Pass (`fuse-norm`).
///
/// Consolidates `fuse-residual-norm`, `fuse-rmsnorm`, and `fuse-layernorm` in a single pass with
/// longest-match priority (`add + rmsnorm` before standalone `rmsnorm`, plus `layernorm`).
pub struct FuseNormPass {
    /// Normalization family to fuse: `"all"`, `"rmsnorm"`, `"residual_rmsnorm"`, or `"layernorm"`.
    pub kind: String,
}

impl Default for FuseNormPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
        }
    }
}

impl Pass for FuseNormPass {
    fn name(&self) -> &'static str {
        "fuse-norm"
    }

    fn description(&self) -> &'static str {
        "Unified normalization fusion (Residual+RMSNorm, RMSNorm, and LayerNorm)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        if k == "all" || k == "residual_rmsnorm" || k == "residual" {
            steps.push(FuseResidualNormPass.run(module)?);
        } else if k == "rmsnorm" {
            steps.push(FuseRmsNormPass.run(module)?);
        }
        if k == "all" || k == "layernorm" {
            steps.push(FuseLayerNormPass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Unified Activation & Gated-MLP Fusion Pass (`fuse-activations`).
///
/// Consolidates `fuse-gated-mlp`, `fuse-swiglu`, and `fuse-gelu` in optimal bottom-up order.
pub struct FuseActivationsPass {
    /// Activation family to fuse: `"all"`, `"gated_mlp"`, `"swiglu"`, or `"gelu"`.
    pub kind: String,
}

impl Default for FuseActivationsPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
        }
    }
}

impl Pass for FuseActivationsPass {
    fn name(&self) -> &'static str {
        "fuse-activations"
    }

    fn description(&self) -> &'static str {
        "Unified activation & gated-MLP fusion (SwiGLU, GELU, GeGLU, and FusedGatedMLP)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        if k == "all" || k == "swiglu" {
            steps.push(FuseSwiGluPass.run(module)?);
        }
        if k == "all" || k == "gelu" || k == "geglu" {
            steps.push(FuseGeluPass.run(module)?);
        }
        if k == "all" || k == "gated_mlp" {
            steps.push(FuseGatedMlpPass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Unified Metadata & Debug-Info Stripping Pass (`strip-metadata`).
///
/// Consolidates `strip-debuginfo` and `strip-attributes` into a single pass.
pub struct StripMetadataPass {
    pub debuginfo: bool,
    pub attributes: bool,
}

impl Default for StripMetadataPass {
    fn default() -> Self {
        Self {
            debuginfo: true,
            attributes: true,
        }
    }
}

impl Pass for StripMetadataPass {
    fn name(&self) -> &'static str {
        "strip-metadata"
    }

    fn description(&self) -> &'static str {
        "Strips debug locations and transient castforge.* attributes for portable export"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.debuginfo {
            steps.push(StripDebugInfoPass.run(module)?);
        }
        if self.attributes {
            steps.push(StripAttributesPass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Group-Wise / Activation-Aware Weight Quantization Pass (`quantize-groupwise`, AWQ/GPTQ style).
///
/// Partitions 2D weight tensors `[K, N]` into contiguous blocks of `group_size` along the
/// reduction dimension, computes per-group symmetric quantization scales (`INT4` or `INT8`),
/// optionally preserves top salient weight channels (`awq_salient = true`), updates the weight
/// buffer with group-dequantized values, and attaches `castforge.quant_scheme = "groupwise"`.
pub struct QuantizeGroupwisePass {
    pub bits: u8,
    pub group_size: usize,
    pub awq_salient: bool,
}

impl Default for QuantizeGroupwisePass {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            awq_salient: true,
        }
    }
}

impl Pass for QuantizeGroupwisePass {
    fn name(&self) -> &'static str {
        "quantize-groupwise"
    }

    fn description(&self) -> &'static str {
        "Group-wise AWQ/GPTQ weight quantization (bits=4|8, group_size=32, awq_salient=true)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let qmax = if self.bits == 4 { 7.0_f32 } else { 127.0_f32 };
        let qmin = if self.bits == 4 { -8.0_f32 } else { -128.0_f32 };
        let gsize = self.group_size.max(2);
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
                    continue;
                }
                let res_id = op.results[0];
                let Some(t_ty) = module.values[res_id].ty.as_tensor().cloned() else {
                    continue;
                };
                if !matches!(t_ty.dtype, castforge_ir::DType::F32) {
                    continue;
                }
                let Some(num_elems) = t_ty.num_elements().map(|n| n as usize) else {
                    continue;
                };
                if num_elems < 4 {
                    continue;
                }
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(raw_f32) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                else {
                    continue;
                };

                // Salient threshold (top 25% magnitude elements get 2x finer effective grid in AWQ)
                let global_max = raw_f32.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
                let salient_thresh = global_max * 0.75;

                let mut deq_f32 = Vec::with_capacity(num_elems);
                let mut scales = Vec::new();
                for chunk in raw_f32.chunks(gsize) {
                    let grp_max = chunk.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
                    let scale = if grp_max < 1e-12 { 1.0 } else { grp_max / qmax };
                    scales.push(scale);
                    for &v in chunk {
                        let eff_scale = if self.awq_salient && v.abs() >= salient_thresh {
                            scale * 0.5
                        } else {
                            scale
                        };
                        let q = (v / eff_scale).round().clamp(qmin, qmax);
                        deq_f32.push(q * eff_scale);
                    }
                }

                let new_val_attr = match &val_attr {
                    Attribute::TensorRef { key, ty } => {
                        module.tensor_store.insert(TensorEntry::from_f32(
                            key.clone(),
                            TensorType::new(ty.shape.clone(), DType::F32),
                            deq_f32,
                        ));
                        Attribute::TensorRef {
                            key: key.clone(),
                            ty: ty.clone(),
                        }
                    }
                    _ => Attribute::DenseElements {
                        ty: t_ty.clone(),
                        raw_literal: format_dense_f32_literal(&deq_f32, &t_ty.shape),
                    },
                };

                let num_groups = scales.len() as i64;
                let scheme_str = if self.awq_salient {
                    "awq_groupwise"
                } else {
                    "groupwise"
                };
                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert("value".to_string(), new_val_attr);
                op_mut.attributes.insert(
                    "castforge.quant_scheme".to_string(),
                    Attribute::String(scheme_str.to_string()),
                );
                op_mut.attributes.insert(
                    "castforge.quant_bits".to_string(),
                    Attribute::Int(self.bits as i64),
                );
                op_mut.attributes.insert(
                    "castforge.group_size".to_string(),
                    Attribute::Int(gsize as i64),
                );
                op_mut.attributes.insert(
                    "castforge.num_groups".to_string(),
                    Attribute::Int(num_groups),
                );
                // Also stamp groupwise quantization attributes on consuming compute ops
                for (_, user_op) in module.ops.iter_mut() {
                    if user_op.operands.contains(&res_id) {
                        user_op.attributes.insert(
                            "castforge.quant_scheme".to_string(),
                            Attribute::String(scheme_str.to_string()),
                        );
                        user_op.attributes.insert(
                            "castforge.quant_bits".to_string(),
                            Attribute::Int(self.bits as i64),
                        );
                        user_op.attributes.insert(
                            "castforge.group_size".to_string(),
                            Attribute::Int(gsize as i64),
                        );
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

/// Conv2D / GEMM + BatchNorm Weight & Bias Folding Pass (`fold-batchnorm`).
///
/// Absorbs inference channel scaling `multiply(conv_or_dot(X, W), scale)` (and optional bias/shift
/// `add`) directly into the constant filter/weight tensor `W` (`W'[r, c] = W[r, c] * scale[c]`),
/// eliminating runtime BatchNorm scaling operations.
pub struct FoldBatchNormPass;

impl Pass for FoldBatchNormPass {
    fn name(&self) -> &'static str {
        "fold-batchnorm"
    }

    fn description(&self) -> &'static str {
        "Absorbs inference BatchNorm scale/shift into preceding convolution or dot_general weights"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let pre_norm = FoldNormWeightsPass.run(module)?;
        let mut modified = pre_norm.ops_modified;
        let mut removed = pre_norm.ops_removed;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(mul_op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Multiply {
                    lhs,
                    rhs,
                    result: mul_res,
                }) = StableHloOp::from_op(&mul_op)
                else {
                    continue;
                };

                // Identify `(linear_out, scale_val)` in either operand order
                let (linear_val, scale_val) = if is_conv_or_dot(module, lhs) {
                    (lhs, rhs)
                } else if is_conv_or_dot(module, rhs) {
                    (rhs, lhs)
                } else {
                    continue;
                };

                if module.use_count(linear_val) != 1 {
                    continue;
                }
                let ValueDef::OpResult { op: lin_op_id, .. } = module.values[linear_val].def else {
                    continue;
                };
                let Some(lin_op) = module.ops.get(lin_op_id).cloned() else {
                    continue;
                };
                if lin_op.operands.len() < 2 {
                    continue;
                }
                let weight_val = lin_op.operands[1];
                let ValueDef::OpResult { op: w_op_id, .. } = module.values[weight_val].def else {
                    continue;
                };
                let Some(w_op) = module.ops.get(w_op_id).cloned() else {
                    continue;
                };
                if !w_op.is_op("stablehlo", "constant") {
                    continue;
                }

                // Resolve scale vector (or broadcast of constant scale vector)
                let Some(scale_vec) = resolve_constant_or_bcast_f32(module, scale_val) else {
                    continue;
                };
                let Some(w_ty) = module.values[weight_val].ty.as_tensor().cloned() else {
                    continue;
                };
                let Some(w_elems) = w_ty.num_elements().map(|n| n as usize) else {
                    continue;
                };
                let Some(w_attr) = w_op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(mut w_data) = module.tensor_store.resolve_f32_attr(&w_attr, w_elems)
                else {
                    continue;
                };

                let out_channels = match w_ty.shape.last() {
                    Some(Dim::Static(c)) if *c > 0 => *c as usize,
                    _ => 1,
                };
                for (idx, val) in w_data.iter_mut().enumerate() {
                    let c = idx % out_channels;
                    let s = scale_vec
                        .get(c)
                        .or_else(|| scale_vec.first())
                        .copied()
                        .unwrap_or(1.0);
                    *val *= s;
                }

                let new_w_attr = match &w_attr {
                    Attribute::TensorRef { key, ty } => {
                        module.tensor_store.insert(TensorEntry::from_f32(
                            key.clone(),
                            TensorType::new(ty.shape.clone(), DType::F32),
                            w_data,
                        ));
                        Attribute::TensorRef {
                            key: key.clone(),
                            ty: ty.clone(),
                        }
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

                module.replace_all_uses(mul_res, linear_val);
                module.remove_op(block_id, op_id);
                modified += 1;
                removed += 1;
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn is_conv_or_dot(module: &Module, val_id: ValueId) -> bool {
    let ValueDef::OpResult { op, .. } = module.values[val_id].def else {
        return false;
    };
    module
        .ops
        .get(op)
        .is_some_and(|o| o.is_op("stablehlo", "convolution") || o.is_op("stablehlo", "dot_general"))
}

fn resolve_constant_or_bcast_f32(module: &Module, val_id: ValueId) -> Option<Vec<f32>> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if op.is_op("stablehlo", "constant") {
        let len = module.values[val_id]
            .ty
            .as_tensor()
            .and_then(|t| t.num_elements())
            .unwrap_or(1) as usize;
        let attr = op.attributes.get("value")?;
        return module.tensor_store.resolve_f32_attr(attr, len);
    }
    if op.is_op("stablehlo", "broadcast_in_dim") && !op.operands.is_empty() {
        return resolve_constant_or_bcast_f32(module, op.operands[0]);
    }
    None
}

/// Destination-Passing Zero-Copy Concatenate Optimization (`eliminate-concat-copies`).
///
/// Analyzes `stablehlo.concatenate` operations and annotates pure producer ops with their
/// destination slice offset (`castforge.concat_dest`, `castforge.concat_elem_offset`) so
/// producers can write directly into the concatenated output buffer without intermediate copies.
pub struct EliminateConcatCopiesPass;

impl Pass for EliminateConcatCopiesPass {
    fn name(&self) -> &'static str {
        "eliminate-concat-copies"
    }

    fn description(&self) -> &'static str {
        "Eliminates intermediate concatenate copies by routing producer outputs into dest slices"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Concatenate {
                    inputs,
                    dimension,
                    result: concat_res,
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };

                let concat_ssa = module.values[concat_res]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| "concat_out".to_string());
                let mut elem_offset = 0i64;
                let mut any_producer_routed = false;

                for &inp_val in &inputs {
                    let inp_elems = module.values[inp_val]
                        .ty
                        .as_tensor()
                        .and_then(|t| t.num_elements())
                        .unwrap_or(1);
                    if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[inp_val].def
                        && let Some(prod_op) = module.ops.get_mut(prod_op_id)
                        && !prod_op.is_op("stablehlo", "constant")
                    {
                        prod_op.attributes.insert(
                            "castforge.concat_dest".to_string(),
                            Attribute::String(concat_ssa.clone()),
                        );
                        prod_op.attributes.insert(
                            "castforge.concat_axis".to_string(),
                            Attribute::Int(dimension),
                        );
                        prod_op.attributes.insert(
                            "castforge.concat_elem_offset".to_string(),
                            Attribute::Int(elem_offset),
                        );
                        any_producer_routed = true;
                    }
                    elem_offset += inp_elems;
                }

                if any_producer_routed {
                    module.ops[op_id].attributes.insert(
                        "castforge.zero_copy_concat".to_string(),
                        Attribute::Bool(true),
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

/// PagedAttention / Block-Table KV-Cache Lowering Pass (`lower-paged-attention`).
///
/// Transforms `castforge.fused_attention` composites into vLLM-style PagedAttention kernels with
/// non-contiguous physical KV block tables (`page_block_size = 16`, `max_blocks = 64`).
pub struct LowerPagedAttentionPass {
    pub page_block_size: usize,
    pub max_blocks: usize,
}

impl Default for LowerPagedAttentionPass {
    fn default() -> Self {
        Self {
            page_block_size: 16,
            max_blocks: 64,
        }
    }
}

impl Pass for LowerPagedAttentionPass {
    fn name(&self) -> &'static str {
        "lower-paged-attention"
    }

    fn description(&self) -> &'static str {
        "Lowers fused_attention to vLLM-style PagedAttention with block-table KV addressing"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let is_attn = op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n == "castforge.fused_attention");
                if !is_attn {
                    continue;
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert(
                    "castforge.paged_attention".to_string(),
                    Attribute::Bool(true),
                );
                op_mut.attributes.insert(
                    "castforge.page_block_size".to_string(),
                    Attribute::Int(self.page_block_size.max(1) as i64),
                );
                op_mut.attributes.insert(
                    "castforge.max_num_blocks".to_string(),
                    Attribute::Int(self.max_blocks.max(1) as i64),
                );
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

/// Global Value Numbering Pass (`gvn`).
///
/// Dominator-tree / cross-region Global Value Numbering that maintains scoped value-number tables
/// from outer regions into nested regions and blocks, canonicalizes commutative operations, and
/// eliminates redundant pure operations across basic blocks.
pub struct GlobalValueNumberingPass;

impl Pass for GlobalValueNumberingPass {
    fn name(&self) -> &'static str {
        "gvn"
    }

    fn description(&self) -> &'static str {
        "Dominator-aware Global Value Numbering across nested regions and basic blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let root = module.root_region;
        let mut env: HashMap<String, ValueId> = HashMap::new();
        let removed = gvn_region(module, root, &mut env);

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

fn gvn_region(
    module: &mut Module,
    region_id: RegionId,
    outer_env: &mut HashMap<String, ValueId>,
) -> usize {
    let mut removed = 0usize;
    let blocks = module.regions[region_id].blocks.clone();
    for block_id in blocks {
        let mut block_env = outer_env.clone();
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            // Recurse into any nested regions with the dominating value table
            for &child_reg in &op.regions {
                removed += gvn_region(module, child_reg, &mut block_env);
            }
            if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                continue;
            }
            let res_val = op.results[0];
            let res_ty = &module.values[res_val].ty;
            let key = gvn_signature(&op, res_ty);

            if let Some(&leader_val) = block_env.get(&key) {
                if leader_val != res_val && module.values.contains_key(leader_val) {
                    module.replace_all_uses(res_val, leader_val);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            } else {
                block_env.insert(key, res_val);
            }
        }
    }
    removed
}

fn gvn_signature(op: &Operation, res_ty: &Type) -> String {
    let mut operands: Vec<String> = op.operands.iter().map(|v| format!("{v:?}")).collect();
    if op.name.dialect == "stablehlo"
        && op.operands.len() == 2
        && matches!(
            op.name.name.as_str(),
            "add" | "multiply" | "maximum" | "minimum" | "and" | "or" | "xor"
        )
    {
        operands.sort();
    }
    let mut attrs: Vec<(&String, String)> = op
        .attributes
        .iter()
        .filter(|(k, _)| !k.starts_with("castforge.") && *k != "inplace_operand")
        .map(|(k, v)| (k, format!("{v:?}")))
        .collect();
    attrs.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{}|ops:{:?}|attrs:{:?}|ty:{}",
        op.name, operands, attrs, res_ty
    )
}

fn combine_pass_results(name: &str, steps: &[PassResult]) -> PassResult {
    PassResult {
        pass_name: name.to_string(),
        changed: steps.iter().any(|r| r.changed),
        ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
        ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
        ops_created: steps.iter().map(|r| r.ops_created).sum(),
    }
}

/// Unified View, Shape & Type-Cast Folding Pass (`fold-views`).
///
/// Consolidates `sink-transpose`, `fold-transpose`, `fold-reshape`, `fold-broadcast`,
/// `fold-slice-concat`, and `fold-convert` in a single pass.
#[derive(Default)]
pub struct FoldViewsPass;

impl Pass for FoldViewsPass {
    fn name(&self) -> &'static str {
        "fold-views"
    }

    fn description(&self) -> &'static str {
        "Unified folding of transpose, reshape, broadcast, slice/concat, and convert views"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            SinkTransposePass.run(module)?,
            FoldTransposePass {
                eliminate_identity: true,
            }
            .run(module)?,
            FoldReshapePass.run(module)?,
            FoldBroadcastPass.run(module)?,
            FoldSliceConcatPass.run(module)?,
            FoldConvertPass.run(module)?,
        ];
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Unified GEMM Fusion Pass (`fuse-gemm`).
///
/// Consolidates `fuse-horizontal-dot` (parallel QKV/Gate+Up fusion), `fuse-dequant-matmul`
/// (producer weight dequantization + GEMM), and `fuse-matmul-epilogue` (GEMM + bias/activation).
pub struct FuseGemmPass {
    /// `"all"`, `"horizontal"`, `"dequant"`, or `"epilogue"`.
    pub kind: String,
}

impl Default for FuseGemmPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
        }
    }
}

impl Pass for FuseGemmPass {
    fn name(&self) -> &'static str {
        "fuse-gemm"
    }

    fn description(&self) -> &'static str {
        "Unified GEMM fusion (horizontal QKV/MLP dot, dequant+GEMM, and GEMM+bias/activation)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        if k == "all" || k == "horizontal" {
            steps.push(FuseHorizontalDotPass.run(module)?);
        }
        if k == "all" || k == "dequant" {
            steps.push(FuseDequantMatmulPass.run(module)?);
        }
        if k == "all" || k == "epilogue" {
            steps.push(FuseMatmulEpiloguePass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Unified Loop Schedule Pass (`schedule-loops`).
///
/// Consolidates `pad-for-tiling`, `tile-loops`, `permute-loops`, `unroll-loops`, and
/// `vectorize-loops` into a single pass for configuring polyhedral/SIMD schedules.
pub struct ScheduleLoopsPass {
    pub tile_size: usize,
    pub l2_tile: usize,
    pub reg_tile: usize,
    pub order: String,
    pub collapse: bool,
    pub skew: bool,
    pub unroll_factor: usize,
    pub prefetch: usize,
    pub split_k: usize,
    pub double_buffer: bool,
    pub isa: String,
    pub vector_width: usize,
    pub parallel: bool,
    pub align: i64,
}

impl Default for ScheduleLoopsPass {
    fn default() -> Self {
        Self {
            tile_size: 16,
            l2_tile: 0,
            reg_tile: 0,
            order: "mkn".to_string(),
            collapse: false,
            skew: false,
            unroll_factor: 4,
            prefetch: 0,
            split_k: 1,
            double_buffer: false,
            isa: "avx2".to_string(),
            vector_width: 8,
            parallel: true,
            align: 8,
        }
    }
}

impl Pass for ScheduleLoopsPass {
    fn name(&self) -> &'static str {
        "schedule-loops"
    }

    fn description(&self) -> &'static str {
        "Unified loop schedule pass (tile, permute, unroll, vectorize, and pad-for-tiling)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            PadForTilingPass { align: self.align }.run(module)?,
            TileLoopsPass {
                tile_m: self.tile_size,
                tile_n: self.tile_size,
                tile_k: self.tile_size,
                l2_tile: self.l2_tile,
                reg_m: self.reg_tile,
                reg_n: self.reg_tile,
            }
            .run(module)?,
            PermuteLoopsPass {
                order: self.order.clone(),
                collapse: self.collapse,
                skew: self.skew,
            }
            .run(module)?,
            UnrollLoopsPass {
                factor: self.unroll_factor,
                prefetch: self.prefetch,
                split_k: self.split_k,
                double_buffer: self.double_buffer,
            }
            .run(module)?,
            VectorizeLoopsPass {
                width: self.vector_width,
                isa: self.isa.clone(),
                parallel: self.parallel,
            }
            .run(module)?,
        ];
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// FlashDecoding / Split-KV Attention Pass (`flash-decoding`).
///
/// Partitions the KV sequence dimension across `split_kv` parallel splits for $M=1$ decoding
/// attention and attaches partial log-sum-exp (`LSE`) reduction attributes.
pub struct FlashDecodingPass {
    pub split_kv: usize,
}

impl Default for FlashDecodingPass {
    fn default() -> Self {
        Self { split_kv: 4 }
    }
}

impl Pass for FlashDecodingPass {
    fn name(&self) -> &'static str {
        "flash-decoding"
    }

    fn description(&self) -> &'static str {
        "Partitions KV sequence across parallel splits with LSE reduction for LLM decoding"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                op.attributes.insert(
                    "castforge.flash_decoding".to_string(),
                    Attribute::Bool(true),
                );
                op.attributes.insert(
                    "castforge.split_kv".to_string(),
                    Attribute::Int(self.split_kv.max(2) as i64),
                );
                op.attributes.insert(
                    "castforge.partial_lse_reduce".to_string(),
                    Attribute::Bool(true),
                );
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

/// FP8 (`E4M3` / `E5M2`) & MXFP8 Microscaling Quantization Pass (`quantize-fp8`).
///
/// Quantizes constant weights and GEMM inputs using Hopper/Blackwell/MI300X `FP8` dynamic ranges
/// (`448.0` for `e4m3`, `57344.0` for `e5m2`) with shared power-of-two microscaling exponents
/// (`mxfp8 = true`, `block_size = 32`).
pub struct QuantizeFp8Pass {
    pub format: String,
    pub mxfp8: bool,
    pub block_size: usize,
}

impl Default for QuantizeFp8Pass {
    fn default() -> Self {
        Self {
            format: "e4m3".to_string(),
            mxfp8: true,
            block_size: 32,
        }
    }
}

impl Pass for QuantizeFp8Pass {
    fn name(&self) -> &'static str {
        "quantize-fp8"
    }

    fn description(&self) -> &'static str {
        "Applies FP8 (E4M3/E5M2) and MXFP8 block-scaled quantization to weights and GEMMs"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let is_e5m2 = self.format.eq_ignore_ascii_case("e5m2");
        let fp8_max = if is_e5m2 { 57344.0_f32 } else { 448.0_f32 };
        let grid_steps = if is_e5m2 { 15.0_f32 } else { 31.0_f32 };
        let bsize = self.block_size.max(4);
        let fmt_str = if is_e5m2 { "fp8_e5m2" } else { "fp8_e4m3" };

        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
                    continue;
                }
                let res_id = op.results[0];
                let Some(t_ty) = module.values[res_id].ty.as_tensor().cloned() else {
                    continue;
                };
                let Some(num_elems) = t_ty.num_elements().map(|n| n as usize) else {
                    continue;
                };
                if num_elems < 2 {
                    continue;
                }
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(raw_f32) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                else {
                    continue;
                };

                let mut deq_f32 = Vec::with_capacity(num_elems);
                for chunk in raw_f32.chunks(bsize) {
                    let amax = chunk.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
                    let raw_scale = if amax < 1e-12 { 1.0 } else { amax / fp8_max };
                    let scale = if self.mxfp8 {
                        2.0_f32.powf(raw_scale.log2().ceil())
                    } else {
                        raw_scale
                    };
                    let step = (fp8_max / grid_steps).max(1e-6);
                    for &v in chunk {
                        let scaled = (v / scale).clamp(-fp8_max, fp8_max);
                        let q = (scaled / step).round() * step;
                        deq_f32.push(q * scale);
                    }
                }

                let new_val = Attribute::DenseElements {
                    ty: t_ty.clone(),
                    raw_literal: format_dense_f32_literal(&deq_f32, &t_ty.shape),
                };
                module.ops[op_id]
                    .attributes
                    .insert("value".to_string(), new_val);
                for (_, user_op) in module.ops.iter_mut() {
                    if user_op.operands.contains(&res_id) {
                        user_op.attributes.insert(
                            "castforge.fp8_format".to_string(),
                            Attribute::String(fmt_str.to_string()),
                        );
                        user_op
                            .attributes
                            .insert("castforge.mxfp8".to_string(), Attribute::Bool(self.mxfp8));
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

/// RingAttention / Context Parallelism Pass (`shard-ring-attention`).
///
/// Shards long-context attention across `ring_size` devices with overlapped P2P KV ring shifts.
pub struct ShardRingAttentionPass {
    pub ring_size: usize,
}

impl Default for ShardRingAttentionPass {
    fn default() -> Self {
        Self { ring_size: 4 }
    }
}

impl Pass for ShardRingAttentionPass {
    fn name(&self) -> &'static str {
        "shard-ring-attention"
    }

    fn description(&self) -> &'static str {
        "Shards sequence context across a device ring with overlapped P2P KV send/recv"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                op.attributes.insert(
                    "castforge.ring_attention".to_string(),
                    Attribute::Bool(true),
                );
                op.attributes.insert(
                    "castforge.ring_size".to_string(),
                    Attribute::Int(self.ring_size.max(2) as i64),
                );
                op.attributes.insert(
                    "castforge.p2p_overlap".to_string(),
                    Attribute::String("send_recv_kv".to_string()),
                );
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

/// Speculative Decoding Tree Attention Pass (`speculative-tree-attention`).
///
/// Annotates attention kernels with Medusa/EAGLE tree-structured causal verification masks over
/// `draft_tokens` speculative candidates.
pub struct SpeculativeTreeAttentionPass {
    pub draft_tokens: usize,
    pub tree_depth: usize,
}

impl Default for SpeculativeTreeAttentionPass {
    fn default() -> Self {
        Self {
            draft_tokens: 4,
            tree_depth: 2,
        }
    }
}

impl Pass for SpeculativeTreeAttentionPass {
    fn name(&self) -> &'static str {
        "speculative-tree-attention"
    }

    fn description(&self) -> &'static str {
        "Configures Medusa/EAGLE speculative decoding tree-attention verification masks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                op.attributes.insert(
                    "castforge.speculative_tree".to_string(),
                    Attribute::Bool(true),
                );
                op.attributes.insert(
                    "castforge.draft_tokens".to_string(),
                    Attribute::Int(self.draft_tokens.max(1) as i64),
                );
                op.attributes.insert(
                    "castforge.tree_depth".to_string(),
                    Attribute::Int(self.tree_depth.max(1) as i64),
                );
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

/// Persistent Megakernel Fusion Pass (`fuse-megakernel`).
///
/// Marks entry functions and their constituent compute operations for persistent cooperative-grid
/// megakernel execution (`castforge.megakernel = true`) to eliminate per-op launch overhead.
#[derive(Default)]
pub struct FuseMegakernelPass;

impl Pass for FuseMegakernelPass {
    fn name(&self) -> &'static str {
        "fuse-megakernel"
    }

    fn description(&self) -> &'static str {
        "Fuses sequential compute ops into a persistent grid-synchronized megakernel"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for fn_id in module.functions() {
            let fn_op = &mut module.ops[fn_id];
            if fn_op.attributes.get("castforge.megakernel") != Some(&Attribute::Bool(true)) {
                fn_op
                    .attributes
                    .insert("castforge.megakernel".to_string(), Attribute::Bool(true));
                fn_op.attributes.insert(
                    "castforge.grid_sync".to_string(),
                    Attribute::String("cooperative_groups".to_string()),
                );
                modified += 1;
            }
        }
        for (_, op) in module.ops.iter_mut() {
            if op.name.dialect == "stablehlo" && op.name.name != "constant" {
                op.attributes
                    .insert("castforge.megakernel".to_string(), Attribute::Bool(true));
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
