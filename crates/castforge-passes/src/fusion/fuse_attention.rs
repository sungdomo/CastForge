//! Unified Attention, MLA, RoPE & Serving KV-Cache Optimization (`fuse-attention`).

use crate::{DeadCodeEliminationPass, FuseSoftmaxPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
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
        if k == "all" || k == "rope" {
            steps.push(FuseRopePass.run(module)?);
        }
        if matches!(
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
        ) {
            steps.push(FuseAttentionPass.run(module)?);
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
        let softmax_res = FuseSoftmaxPass::default().run(module)?;
        let mut created = 0usize;
        let mut removed = softmax_res.ops_removed;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    lhs: probs_val,
                    rhs: v_val,
                    ..
                }) = StableHloOp::from_op(op)
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

/// Autoregressive KV-Cache Injection Pass (`inject-kv-cache`).
pub struct InjectKvCachePass {
    pub max_seq_len: usize,
}

impl Default for InjectKvCachePass {
    fn default() -> Self {
        Self { max_seq_len: 2048 }
    }
}

impl Pass for InjectKvCachePass {
    fn name(&self) -> &'static str {
        "inject-kv-cache"
    }

    fn description(&self) -> &'static str {
        "Injects autoregressive KV-cache state updates (castforge.kv_cache_update) before attention"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let attn_res = FuseAttentionPass.run(module)?;
        let mut created = attn_res.ops_created;
        let mut modified = attn_res.ops_modified;
        let removed = attn_res.ops_removed;
        let max_seq = self.max_seq_len.max(1) as i64;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.attributes.contains_key("kv_cached") {
                    continue;
                }
                let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if name != "castforge.fused_attention" || inputs.len() < 3 {
                    continue;
                }

                let k_val = inputs[1];
                let v_val = inputs[2];
                let k_cached = insert_kv_cache_op(module, block_id, op_id, k_val, "key", max_seq);
                let v_cached = insert_kv_cache_op(module, block_id, op_id, v_val, "value", max_seq);

                let op_mut = &mut module.ops[op_id];
                op_mut.operands[1] = k_cached;
                op_mut.operands[2] = v_cached;
                op_mut
                    .attributes
                    .insert("kv_cached".to_string(), Attribute::Bool(true));
                created += 2;
                modified += 1;
            }
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

fn insert_kv_cache_op(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    src_val: ValueId,
    role: &str,
    max_seq: i64,
) -> ValueId {
    let out_ty = module.values[src_val].ty.clone();
    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.kv_cache_update".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("kv_cache_update_impl".to_string()),
    );
    attrs.insert(
        "cache_role".to_string(),
        Attribute::String(role.to_string()),
    );
    attrs.insert("max_seq_len".to_string(), Attribute::Int(max_seq));

    let kv_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "composite"),
        operands: SmallVec::from_slice(&[src_val]),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let kv_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: kv_op_id,
            index: 0,
        },
    });
    module.ops[kv_op_id].results.push(kv_val);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, kv_op_id);
    } else {
        block_ops.push(kv_op_id);
    }
    kv_val
}

/// PagedAttention Block-Table Lowering Pass (`lower-paged-attention`).
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
        let block_size = self.page_block_size.max(4) as i64;
        let max_blocks = self.max_blocks.max(1) as i64;

        for (_, op) in module.ops.iter_mut() {
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                op.attributes.insert(
                    "castforge.paged_attention".to_string(),
                    Attribute::Bool(true),
                );
                op.attributes.insert(
                    "castforge.page_block_size".to_string(),
                    Attribute::Int(block_size),
                );
                op.attributes.insert(
                    "castforge.max_blocks".to_string(),
                    Attribute::Int(max_blocks),
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

/// FlashDecoding / Split-KV Attention Pass (`flash-decoding`).
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

/// Speculative Decoding Tree Attention Pass (`speculative-tree-attention`).
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

fn run_sliding_window_attention(
    module: &mut Module,
    window_size: usize,
    sink_tokens: usize,
) -> Result<PassResult> {
    let mut modified = 0usize;
    let win = window_size.max(16) as i64;
    let sinks = sink_tokens as i64;
    for (_, op) in module.ops.iter_mut() {
        let is_attn = op.is_op("stablehlo", "dot_general")
            || (op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention"));
        if is_attn {
            op.attributes
                .insert("castforge.sliding_window".to_string(), Attribute::Int(win));
            op.attributes.insert(
                "castforge.attention_sinks".to_string(),
                Attribute::Int(sinks),
            );
            modified += 1;
        }
    }
    Ok(PassResult {
        pass_name: "fuse-attention".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

fn run_kv_sharing(module: &mut Module, share_every: usize) -> Result<PassResult> {
    let stride = share_every.max(2);
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let attn_ops: Vec<OpId> = module
            .block_ops(block_id)
            .into_iter()
            .filter(|&id| {
                module.ops.get(id).is_some_and(|op| {
                    op.is_op("stablehlo", "dot_general")
                        || (op.is_op("stablehlo", "composite")
                            && op
                                .attributes
                                .get("name")
                                .and_then(|a| a.as_str())
                                .is_some_and(|n| n == "castforge.fused_attention"))
                })
            })
            .collect();

        let mut donor_kv: Option<(ValueId, ValueId)> = None;
        for (idx, op_id) in attn_ops.into_iter().enumerate() {
            let is_comp = module.ops[op_id].is_op("stablehlo", "composite")
                && module.ops[op_id].operands.len() >= 3;
            if idx % stride == 0 {
                if is_comp {
                    donor_kv = Some((module.ops[op_id].operands[1], module.ops[op_id].operands[2]));
                }
            } else if is_comp && let Some((dk, dv)) = donor_kv {
                module.ops[op_id].operands[1] = dk;
                module.ops[op_id].operands[2] = dv;
            }
            let attrs = &mut module.ops[op_id].attributes;
            attrs.insert("castforge.kv_shared".to_string(), Attribute::Bool(true));
            attrs.insert(
                "castforge.kv_share_stride".to_string(),
                Attribute::Int(stride as i64),
            );
            modified += 1;
        }
    }

    if modified > 0 {
        let _ = DeadCodeEliminationPass::ssa_only().run(module)?;
    }

    Ok(PassResult {
        pass_name: "fuse-attention".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

fn run_rope_scaling(module: &mut Module, scaling: &str, factor: f64) -> Result<PassResult> {
    let mut modified = 0usize;
    let eff_factor = factor.max(1.0);
    let mscale = if scaling.eq_ignore_ascii_case("yarn") && eff_factor > 1.0 {
        0.1 * eff_factor.ln() + 1.0
    } else {
        1.0
    };
    for (_, op) in module.ops.iter_mut() {
        let is_target = op.is_op("stablehlo", "dot_general")
            || (op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| {
                        n == "castforge.fused_rope" || n == "castforge.fused_attention"
                    }));
        if is_target {
            op.attributes.insert(
                "castforge.rope_scaling".to_string(),
                Attribute::String(scaling.to_ascii_lowercase()),
            );
            op.attributes.insert(
                "castforge.rope_factor".to_string(),
                Attribute::Float(eff_factor),
            );
            op.attributes.insert(
                "castforge.rope_mscale".to_string(),
                Attribute::Float(mscale),
            );
            modified += 1;
        }
    }
    Ok(PassResult {
        pass_name: "fuse-attention".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

fn run_attention_softcap_and_qk_norm(
    module: &mut Module,
    logit_softcap: Option<f64>,
    qk_norm: bool,
) -> Result<PassResult> {
    let mut modified = 0usize;
    for (_, op) in module.ops.iter_mut() {
        let is_attn = op.is_op("stablehlo", "dot_general")
            || (op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention"));
        if is_attn {
            if let Some(cap) = logit_softcap {
                op.attributes.insert(
                    "castforge.logit_softcap".to_string(),
                    Attribute::Float(cap.max(1.0)),
                );
            }
            if qk_norm {
                op.attributes
                    .insert("castforge.qk_norm".to_string(), Attribute::Bool(true));
            }
            modified += 1;
        }
    }
    Ok(PassResult {
        pass_name: "fuse-attention".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

fn run_kv_cache_quantization(module: &mut Module, kv_dtype: &str) -> Result<PassResult> {
    let mut modified = 0usize;
    let dt = kv_dtype.to_ascii_lowercase();
    for (_, op) in module.ops.iter_mut() {
        let is_kv_or_attn = op.is_op("stablehlo", "composite")
            && op
                .attributes
                .get("name")
                .and_then(|a| a.as_str())
                .is_some_and(|n| {
                    n == "castforge.kv_cache_update" || n == "castforge.fused_attention"
                });
        if is_kv_or_attn {
            op.attributes.insert(
                "castforge.kv_cache_dtype".to_string(),
                Attribute::String(dt.clone()),
            );
            op.attributes.insert(
                "castforge.kv_per_head_scale".to_string(),
                Attribute::Bool(true),
            );
            modified += 1;
        }
    }
    Ok(PassResult {
        pass_name: "fuse-attention".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}
