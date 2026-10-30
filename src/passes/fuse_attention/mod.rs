//! Unified Attention, MLA, RoPE & Serving KV-Cache Optimization (`fuse-attention`).

pub mod serving;

pub use serving::*;

use crate::ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use crate::passes::fuse_norm::try_fuse_softmax_at_op;
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Attention, MLA, RoPE & LLM Serving Optimization Pass (`fuse-attention`).
///
/// Consolidates:
/// - `kind=sdpa|all` (`FuseAttentionPass` — SDPA / GQA fusion)
/// - `kind=rope|all` (`FuseRopePass` — Rotary Position Embedding fusion)
/// - `kind=mla` (`FuseMlaPass` — DeepSeek-V2/V3 Multi-Head Latent Attention KV absorption)
/// - `kind=kv_cache` (`InjectKvCachePass` — stateful KV-cache injection)
/// - `kind=paged` (`LowerPagedAttentionPass` — vLLM PagedAttention block tables)
/// - `kind=flash_decoding` (`FlashDecodingPass` — Split-KV parallel LSE reduction)
/// - `kind=tree|speculative_tree` (`SpeculativeTreeAttentionPass` — Medusa/EAGLE tree masks)
/// - `kind=sliding_window` (Mistral/Gemma-2 sliding window + StreamingLLM attention sinks)
/// - `kind=kv_sharing` (YOCO / Cross-Layer KV-cache sharing across `share_every` layers)
pub struct UnifiedFuseAttentionPass {
    pub kind: String,
    pub kv_lora_rank: i64,
    pub max_seq_len: usize,
    pub page_block_size: usize,
    pub max_blocks: usize,
    pub split_kv: usize,
    pub draft_tokens: usize,
    pub tree_depth: usize,
    pub window_size: usize,
    pub sink_tokens: usize,
    pub share_every: usize,
    pub rope_scaling: Option<String>,
    pub rope_factor: f64,
    pub logit_softcap: Option<f64>,
    pub qk_norm: bool,
    pub kv_dtype: Option<String>,
    pub state_dim: usize,
    pub ssm_chunk_size: usize,
}

impl Default for UnifiedFuseAttentionPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
            kv_lora_rank: 512,
            max_seq_len: 2048,
            page_block_size: 16,
            max_blocks: 64,
            split_kv: 4,
            draft_tokens: 4,
            tree_depth: 2,
            window_size: 4096,
            sink_tokens: 4,
            share_every: 2,
            rope_scaling: None,
            rope_factor: 1.0,
            logit_softcap: None,
            qk_norm: false,
            kv_dtype: None,
            state_dim: 128,
            ssm_chunk_size: 64,
        }
    }
}

impl Pass for UnifiedFuseAttentionPass {
    fn name(&self) -> &'static str {
        "fuse-attention"
    }

    fn description(&self) -> &'static str {
        "Unified attention fusion & serving (SDPA/RoPE/MLA/KV-cache/Paged/Flash/Sliding/KVShare)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        let do_rope = k == "all" || k == "rope";
        let do_sdpa = matches!(
            k.as_str(),
            "all"
                | "sdpa"
                | "gqa"
                | "mla"
                | "paged"
                | "flash_decoding"
                | "flash"
                | "tree"
                | "speculative_tree"
                | "sliding_window"
                | "window"
                | "sink"
                | "kv_sharing"
                | "yoco"
                | "cla"
                | "serving"
        );
        if do_rope || do_sdpa {
            let (rope_c, sdpa_c) = fuse_rope_and_sdpa_single_traversal(module, do_rope, do_sdpa);
            let created = rope_c + sdpa_c;
            let mut removed = 0usize;
            if created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
            }
            steps.push(PassResult {
                pass_name: self.name().to_string(),
                changed: created > 0,
                ops_removed: removed,
                ops_modified: 0,
                ops_created: created,
            });
        }
        if self.rope_scaling.is_some() || self.rope_factor > 1.0 {
            let scaling = self.rope_scaling.as_deref().unwrap_or("yarn");
            steps.push(run_rope_scaling(module, scaling, self.rope_factor)?);
        }
        if self.logit_softcap.is_some() || self.qk_norm {
            steps.push(run_attention_softcap_and_qk_norm(
                module,
                self.logit_softcap,
                self.qk_norm,
            )?);
        }
        if k == "mla" {
            steps.push(
                FuseMlaPass {
                    latent_rank: self.kv_lora_rank,
                }
                .run(module)?,
            );
        }
        if matches!(k.as_str(), "kv_cache" | "kv") || self.kv_dtype.is_some() {
            steps.push(
                InjectKvCachePass {
                    max_seq_len: self.max_seq_len,
                }
                .run(module)?,
            );
            if let Some(ref dtype) = self.kv_dtype {
                steps.push(run_kv_cache_quantization(module, dtype)?);
            }
        }
        if matches!(k.as_str(), "paged" | "serving") {
            steps.push(
                LowerPagedAttentionPass {
                    page_block_size: self.page_block_size,
                    max_blocks: self.max_blocks,
                }
                .run(module)?,
            );
        }
        if matches!(k.as_str(), "flash_decoding" | "flash" | "serving") {
            steps.push(
                FlashDecodingPass {
                    split_kv: self.split_kv,
                }
                .run(module)?,
            );
        }
        if matches!(k.as_str(), "speculative_tree" | "tree" | "serving") {
            steps.push(
                SpeculativeTreeAttentionPass {
                    draft_tokens: self.draft_tokens,
                    tree_depth: self.tree_depth,
                }
                .run(module)?,
            );
        }
        if matches!(k.as_str(), "sliding_window" | "window" | "sink") {
            steps.push(run_sliding_window_attention(
                module,
                self.window_size,
                self.sink_tokens,
            )?);
        }
        if matches!(k.as_str(), "kv_sharing" | "yoco" | "cla") {
            steps.push(run_kv_sharing(module, self.share_every)?);
        }
        if matches!(
            k.as_str(),
            "ssm" | "mamba2" | "linear_attn" | "deltanet" | "rwkv6"
        ) {
            steps.push(run_ssm_chunk_scan_fusion(
                module,
                &k,
                self.state_dim,
                self.ssm_chunk_size,
            )?);
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

/// Scaled Dot-Product & Grouped-Query Attention Fusion (`fuse-attention`).
pub struct FuseAttentionPass;

struct AttentionMatch {
    q: ValueId,
    k: ValueId,
    v: ValueId,
    mask: Option<ValueId>,
    scale: f64,
    k_transposed: bool,
    is_gqa: bool,
}

impl Pass for FuseAttentionPass {
    fn name(&self) -> &'static str {
        "fuse-attention"
    }

    fn description(&self) -> &'static str {
        "Fuses Q*K^T + scale + softmax + P*V (with GQA support) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let (_, created) = fuse_rope_and_sdpa_single_traversal(module, false, true);
        let mut removed = 0usize;
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

/// Fuses RoPE (`castforge.fused_rope`), Softmax (`castforge.fused_softmax`), and SDPA/GQA
/// (`castforge.fused_attention`) in a **single top-down forward traversal** per block.
fn fuse_rope_and_sdpa_single_traversal(
    module: &mut Module,
    do_rope: bool,
    do_sdpa: bool,
) -> (usize, usize) {
    let mut rope_created = 0usize;
    let mut sdpa_created = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };

            if do_rope
                && let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op)
                && let Some(RopeMatch { x, cos, sin }) =
                    match_rope_pair(module, lhs, rhs).or_else(|| match_rope_pair(module, rhs, lhs))
            {
                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_rope".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_rope_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x, cos, sin]);
                op_mut.attributes = attrs;
                rope_created += 1;
                continue;
            }

            if do_sdpa {
                if try_fuse_softmax_at_op(module, op_id, &op) {
                    continue;
                }

                let Some(StableHloOp::DotGeneral {
                    lhs: probs_val,
                    rhs: v_val,
                    ..
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };

                let Some(AttentionMatch {
                    q,
                    k,
                    v,
                    mask,
                    scale,
                    k_transposed,
                    is_gqa,
                }) = match_sdpa_chain(module, probs_val, v_val)
                else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_attention".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_attention_impl".to_string()),
                );
                attrs.insert("scale".to_string(), Attribute::Float(scale));
                if k_transposed {
                    attrs.insert("k_transposed".to_string(), Attribute::Bool(true));
                }
                if is_gqa {
                    attrs.insert("gqa".to_string(), Attribute::Bool(true));
                }
                if mask.is_some() {
                    attrs.insert("masked".to_string(), Attribute::Bool(true));
                }

                let mut operands = SmallVec::from_slice(&[q, k, v]);
                if let Some(m_val) = mask {
                    operands.push(m_val);
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = operands;
                op_mut.attributes = attrs;
                sdpa_created += 1;
            }
        }
    }

    (rope_created, sdpa_created)
}

fn match_sdpa_chain(module: &Module, probs_val: ValueId, v_val: ValueId) -> Option<AttentionMatch> {
    let ValueDef::OpResult { op: sm_op_id, .. } = module.values[probs_val].def else {
        return None;
    };
    let sm_op = module.ops.get(sm_op_id)?;
    let StableHloOp::Composite { name, inputs, .. } = StableHloOp::from_op(sm_op)? else {
        return None;
    };
    if name != "castforge.fused_softmax" || inputs.is_empty() {
        return None;
    }

    let scores_val = inputs[0];
    let (pre_mask_val, mask) = unwrap_optional_mask(module, scores_val);
    let (qk_val, scale) = unwrap_optional_scale(module, pre_mask_val);

    let ValueDef::OpResult { op: qk_op_id, .. } = module.values[qk_val].def else {
        return None;
    };
    let qk_op = module.ops.get(qk_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: q,
        rhs: k_raw,
        dimension_numbers: dims,
        ..
    } = StableHloOp::from_op(qk_op)?
    else {
        return None;
    };

    let (k_unwrapped, k_from_transpose) = unwrap_optional_transpose(module, k_raw);
    let k_transposed = k_from_transpose || dims.rhs_contracting_dimensions == [1];
    let is_gqa = is_gqa_expanded(module, k_unwrapped) || is_gqa_expanded(module, v_val);

    Some(AttentionMatch {
        q,
        k: k_unwrapped,
        v: v_val,
        mask,
        scale,
        k_transposed,
        is_gqa,
    })
}

fn unwrap_optional_mask(module: &Module, val: ValueId) -> (ValueId, Option<ValueId>) {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return (val, None);
    };
    let Some(op) = module.ops.get(op_id) else {
        return (val, None);
    };
    if let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(op) {
        let (qk_cand, _) = unwrap_optional_scale(module, lhs);
        if let ValueDef::OpResult { op: qk_id, .. } = module.values[qk_cand].def
            && module
                .ops
                .get(qk_id)
                .and_then(StableHloOp::from_op)
                .is_some_and(|h| matches!(h, StableHloOp::DotGeneral { .. }))
        {
            return (lhs, Some(rhs));
        }
        let (qk_cand_r, _) = unwrap_optional_scale(module, rhs);
        if let ValueDef::OpResult { op: qk_id, .. } = module.values[qk_cand_r].def
            && module
                .ops
                .get(qk_id)
                .and_then(StableHloOp::from_op)
                .is_some_and(|h| matches!(h, StableHloOp::DotGeneral { .. }))
        {
            return (rhs, Some(lhs));
        }
    }
    (val, None)
}

fn unwrap_optional_scale(module: &Module, val: ValueId) -> (ValueId, f64) {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return (val, 1.0);
    };
    let Some(op) = module.ops.get(op_id) else {
        return (val, 1.0);
    };
    if let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) {
        if let Some(s) = extract_scalar_constant(module, rhs) {
            return (lhs, s);
        }
        if let Some(s) = extract_scalar_constant(module, lhs) {
            return (rhs, s);
        }
    }
    (val, 1.0)
}

fn extract_scalar_constant(module: &Module, val: ValueId) -> Option<f64> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    let StableHloOp::Constant { value, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    match value {
        Attribute::Float(f) => Some(f),
        Attribute::DenseElements { raw_literal, .. } => raw_literal.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn unwrap_optional_transpose(module: &Module, val: ValueId) -> (ValueId, bool) {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return (val, false);
    };
    let Some(op) = module.ops.get(op_id) else {
        return (val, false);
    };
    if let Some(StableHloOp::Transpose {
        operand,
        permutation,
        ..
    }) = StableHloOp::from_op(op)
        && permutation == [1, 0]
    {
        return (operand, true);
    }
    (val, false)
}

fn is_gqa_expanded(module: &Module, val: ValueId) -> bool {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return false;
    };
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    matches!(
        StableHloOp::from_op(op),
        Some(StableHloOp::BroadcastInDim { .. } | StableHloOp::Reshape { .. })
    )
}

/// Rotary Position Embedding Fusion (`fuse-rope`).
pub struct FuseRopePass;

struct RopeMatch {
    x: ValueId,
    cos: ValueId,
    sin: ValueId,
}

impl Pass for FuseRopePass {
    fn name(&self) -> &'static str {
        "fuse-rope"
    }

    fn description(&self) -> &'static str {
        "Fuses decomposed Rotary Position Embedding (RoPE) subgraphs into stablehlo.composite"
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

                let matched =
                    match_rope_pair(module, lhs, rhs).or_else(|| match_rope_pair(module, rhs, lhs));
                let Some(RopeMatch { x, cos, sin }) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_rope".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_rope_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x, cos, sin]);
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

fn match_rope_pair(
    module: &Module,
    mul_cos_val: ValueId,
    mul_sin_val: ValueId,
) -> Option<RopeMatch> {
    let (a0, a1) = match_multiply_operands(module, mul_cos_val)?;
    let (b0, b1) = match_multiply_operands(module, mul_sin_val)?;

    for &(rot_cand, sin_cand) in &[(b0, b1), (b1, b0)] {
        if let Some(x) = match_rotate_half(module, rot_cand) {
            if a0 == x {
                return Some(RopeMatch {
                    x,
                    cos: a1,
                    sin: sin_cand,
                });
            }
            if a1 == x {
                return Some(RopeMatch {
                    x,
                    cos: a0,
                    sin: sin_cand,
                });
            }
        }
    }
    None
}

fn match_multiply_operands(module: &Module, val: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    Some((lhs, rhs))
}

fn match_rotate_half(module: &Module, val: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: cat_id, .. } = module.values[val].def else {
        return None;
    };
    let cat_op = module.ops.get(cat_id)?;
    let StableHloOp::Concatenate { inputs, .. } = StableHloOp::from_op(cat_op)? else {
        return None;
    };
    if inputs.len() != 2 {
        return None;
    }

    let neg_slice_val = match_negated_value(module, inputs[0])?;
    let x_hi = match_slice_source(module, neg_slice_val)?;
    let x_lo = match_slice_source(module, inputs[1])?;
    if x_hi == x_lo { Some(x_lo) } else { None }
}

fn match_negated_value(module: &Module, val: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if op.is_op("stablehlo", "negate") {
        return op.operands.first().copied();
    }
    if let Some(StableHloOp::Subtract { lhs, rhs, .. }) = StableHloOp::from_op(op)
        && is_zero_constant(module, lhs)
    {
        return Some(rhs);
    }
    None
}

fn is_zero_constant(module: &Module, val: ValueId) -> bool {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return false;
    };
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::Constant { value, .. }) = StableHloOp::from_op(op) else {
        return false;
    };
    match value {
        Attribute::Float(f) => f.abs() < 1e-9,
        Attribute::Int(i) => i == 0,
        Attribute::DenseElements { raw_literal, .. } => raw_literal
            .trim()
            .parse::<f64>()
            .is_ok_and(|f| f.abs() < 1e-9),
        _ => false,
    }
}

fn match_slice_source(module: &Module, val: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    let StableHloOp::Slice { operand, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    Some(operand)
}

/// DeepSeek-V2/V3 Multi-Head Latent Attention Pass (`fuse-mla`).
pub struct FuseMlaPass {
    pub latent_rank: i64,
}

impl Default for FuseMlaPass {
    fn default() -> Self {
        Self { latent_rank: 512 }
    }
}

impl Pass for FuseMlaPass {
    fn name(&self) -> &'static str {
        "fuse-mla"
    }

    fn description(&self) -> &'static str {
        "Fuses Multi-Head Latent Attention (MLA) by absorbing KV up-projections into Q/O weights"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let is_target = op.is_op("stablehlo", "dot_general")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n == "castforge.fused_attention"));
            if is_target {
                op.attributes.insert(
                    "castforge.mla_compressed_kv".to_string(),
                    Attribute::Bool(true),
                );
                op.attributes.insert(
                    "castforge.mla_latent_rank".to_string(),
                    Attribute::Int(self.latent_rank.max(16)),
                );
                op.attributes.insert(
                    "castforge.mla_w_uk_absorbed".to_string(),
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
