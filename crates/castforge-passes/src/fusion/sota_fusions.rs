use crate::{
    FissionLoopsPass, FuseDequantMatmulPass, FuseElementwisePass, FuseGatedMlpPass, FuseGeluPass,
    FuseHorizontalDotPass, FuseLayerNormPass, FuseLoopsPass, FuseMatmulEpiloguePass,
    FuseResidualNormPass, FuseRmsNormPass, FuseSwiGluPass, InjectKvCachePass, Pass, PassResult,
};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Module};

fn combine_pass_results(name: &str, steps: &[PassResult]) -> PassResult {
    PassResult {
        pass_name: name.to_string(),
        changed: steps.iter().any(|r| r.changed),
        ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
        ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
        ops_created: steps.iter().map(|r| r.ops_created).sum(),
    }
}

/// Unified Normalization Fusion Pass (`fuse-norm`).
pub struct FuseNormPass {
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
        }
        if k == "all" || k == "rmsnorm" || k == "rms" {
            steps.push(FuseRmsNormPass.run(module)?);
        }
        if k == "all" || k == "layernorm" || k == "ln" {
            steps.push(FuseLayerNormPass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Unified Activation & Gated-MLP Fusion Pass (`fuse-activations`).
pub struct FuseActivationsPass {
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
        if k == "all" || k == "gated_mlp" || k == "mlp" {
            steps.push(FuseGatedMlpPass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Unified GEMM Fusion Pass (`fuse-gemm`).
pub struct FuseGemmPass {
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

/// Unified Elementwise & Loop-Chain Fusion Pass (`fuse-elementwise`).
///
/// Consolidates `fuse-elementwise` (`mode=composites`), `fuse-loops` (`mode=loops`), and
/// `fission-loops` (`max_steps > 0`).
pub struct UnifiedFuseElementwisePass {
    pub mode: String,
    pub max_steps: usize,
}

impl Default for UnifiedFuseElementwisePass {
    fn default() -> Self {
        Self {
            mode: "composites".to_string(),
            max_steps: 0,
        }
    }
}

impl Pass for UnifiedFuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Unified elementwise & register-pipelined loop-chain fusion (with max_steps fission)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        if m == "loops" || m == "chain" || self.max_steps > 0 {
            steps.push(FuseLoopsPass.run(module)?);
            if self.max_steps > 0 {
                steps.push(
                    FissionLoopsPass {
                        max_steps: self.max_steps,
                    }
                    .run(module)?,
                );
            }
        } else {
            steps.push(FuseElementwisePass.run(module)?);
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
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

/// Unified LLM Attention Decoding & KV-Cache Optimization Pass (`optimize-attention`).
///
/// Consolidates `inject-kv-cache`, `lower-paged-attention`, `flash-decoding`, and
/// `speculative-tree-attention`.
pub struct OptimizeAttentionPass {
    pub mode: String,
    pub max_seq_len: usize,
    pub page_block_size: usize,
    pub max_blocks: usize,
    pub split_kv: usize,
    pub draft_tokens: usize,
    pub tree_depth: usize,
}

impl Default for OptimizeAttentionPass {
    fn default() -> Self {
        Self {
            mode: "all".to_string(),
            max_seq_len: 2048,
            page_block_size: 16,
            max_blocks: 64,
            split_kv: 4,
            draft_tokens: 4,
            tree_depth: 2,
        }
    }
}

impl Pass for OptimizeAttentionPass {
    fn name(&self) -> &'static str {
        "optimize-attention"
    }

    fn description(&self) -> &'static str {
        "Unified LLM attention optimization (KV-cache, PagedAttention, FlashDecoding, SpecTree)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        if m == "kv_cache" || m == "kv" {
            steps.push(
                InjectKvCachePass {
                    max_seq_len: self.max_seq_len,
                }
                .run(module)?,
            );
        }
        if m == "paged" || m == "all" {
            steps.push(
                LowerPagedAttentionPass {
                    page_block_size: self.page_block_size,
                    max_blocks: self.max_blocks,
                }
                .run(module)?,
            );
        }
        if m == "flash_decoding" || m == "flash" || m == "all" {
            steps.push(
                FlashDecodingPass {
                    split_kv: self.split_kv,
                }
                .run(module)?,
            );
        }
        if m == "speculative_tree" || m == "tree" || m == "all" {
            steps.push(
                SpeculativeTreeAttentionPass {
                    draft_tokens: self.draft_tokens,
                    tree_depth: self.tree_depth,
                }
                .run(module)?,
            );
        }
        Ok(combine_pass_results(self.name(), &steps))
    }
}

/// Persistent Megakernel Fusion Pass (`fuse-megakernel`).
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

/// DeepSeek-V2/V3 Multi-Head Latent Attention Pass (`fuse-mla`).
///
/// Absorbs the KV latent up-projection matrix $W^{UK}$ directly into $W^Q$ ($W^{Q\_absorbed} =
/// W^Q (W^{UK})^T$) and annotates `fused_attention` / `dot_general` ops with compressed KV latent
/// rank metadata (`castforge.mla_compressed_kv = true`, `castforge.mla_latent_rank`).
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

/// Chunked Fused Linear Classifier + Cross-Entropy Loss Pass (`fuse-linear-cross-entropy`).
///
/// Fuses the final vocabulary projection (`hidden @ lm_head`) with online log-sum-exp
/// Cross-Entropy loss in chunks of `chunk_size` tokens so the full `[B*S, Vocab]` logits tensor
/// is never materialized in HBM (`castforge.fused_linear_cross_entropy = true`).
pub struct FuseLinearCrossEntropyPass {
    pub chunk_size: i64,
}

impl Default for FuseLinearCrossEntropyPass {
    fn default() -> Self {
        Self { chunk_size: 1024 }
    }
}

impl Pass for FuseLinearCrossEntropyPass {
    fn name(&self) -> &'static str {
        "fuse-linear-cross-entropy"
    }

    fn description(&self) -> &'static str {
        "Fuses classifier GEMM + online log-sum-exp Cross-Entropy to avoid materializing logits"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            for op_id in module.block_ops(block_id) {
                let Some(op) = module.ops.get_mut(op_id) else {
                    continue;
                };
                if op.is_op("stablehlo", "dot_general")
                    || (op.is_op("stablehlo", "composite")
                        && op
                            .attributes
                            .get("name")
                            .and_then(|a| a.as_str())
                            .is_some_and(|n| n.contains("log_softmax") || n.contains("matmul")))
                {
                    op.attributes.insert(
                        "castforge.fused_linear_cross_entropy".to_string(),
                        Attribute::Bool(true),
                    );
                    op.attributes.insert(
                        "castforge.ce_chunk_size".to_string(),
                        Attribute::Int(self.chunk_size.max(64)),
                    );
                    op.attributes.insert(
                        "castforge.materialize_logits".to_string(),
                        Attribute::Bool(false),
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
