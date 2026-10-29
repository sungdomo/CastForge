use crate::{
    AutodiffPass, BakeLoraPass, BisectPassesPass, BucketShapesPass, BufferizeInplacePass,
    CalibrateQuantizePass, CanonicalizePass, CommonSubexpressionEliminationPass, ComposeFuncsPass,
    ConvertLayoutPass, DeadCodeEliminationPass, DecomposeCompositesPass, DedupFunctionsPass,
    EgraphSimplifyPass, EliminateConcatCopiesPass, ExternalizeWeightsPass, ExtractSubgraphPass,
    FissionLoopsPass, FlashDecodingPass, FoldBatchNormPass, FoldConstantsPass, FoldViewsPass,
    FreezeInputsPass, FuseActivationsPass, FuseAttentionPass, FuseElementwisePass, FuseGemmPass,
    FuseLoopsPass, FuseMapReducePass, FuseMegakernelPass, FuseMoePass, FuseNormPass, FuseRopePass,
    FuseSoftmaxPass, GlobalValueNumberingPass, HashIrPass, HoistAllReducePass, HoistConstantsPass,
    InjectKvCachePass, InlinerPass, InstrumentProfilingPass, IrEditPass, LicmPass,
    LowerConvToGemmPass, LowerPagedAttentionPass, MergeMethod, MergeWeightsPass,
    MixedPrecisionPass, NormalizeSsaPass, OffloadActivationsPass, OptO1Pass, OptO2Pass, OptO3Pass,
    OutlineSubgraphsPass, PackWeightsPass, ParameterizeConstantsPass, Pass, PassOptions,
    PrintOpStatsPass, QuantizeFp8Pass, QuantizeGroupwisePass, QuantizeWeightsPass, ReduceIrPass,
    RematerializePass, ResizeBatchPass, SanitizeNumericsPass, ScheduleLoopsPass,
    ScheduleStreamsPass, ShardRingAttentionPass, ShardSpmdPass, SimplifyAlgebraicPass,
    Sparsify2To4Pass, SpecializeShapesPass, SpeculativeTreeAttentionPass, SplitLargeGemmPass,
    StripMetadataPass, StructuredPruningPass, SymbolDcePass, TapIntermediatesPass,
    TaskArithmeticPass, TopologicalSortPass, UnrollCallsPass, VerifyEquivPass, VerifyShapesPass,
};
use anyhow::{Result, anyhow};
use castforge_ir::DType;
use indexmap::IndexMap;

pub type PassFactory = Box<dyn Fn(&PassOptions) -> Result<Box<dyn Pass>> + Send + Sync>;

pub struct PassDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub factory: PassFactory,
}

/// Registry holding all available passes for dynamic lookup by CLI, command name, or config file.
pub struct PassRegistry {
    passes: IndexMap<&'static str, PassDescriptor>,
}

impl Default for PassRegistry {
    fn default() -> Self {
        Self::with_builtin_passes()
    }
}

impl PassRegistry {
    pub fn new() -> Self {
        Self {
            passes: IndexMap::new(),
        }
    }

    /// Create a registry pre-populated with all built-in CastForge passes.
    pub fn with_builtin_passes() -> Self {
        let mut reg = Self::new();
        reg.register(
            "dce",
            "Dead Code Elimination (removes unused pure SSA ops and dead private symbols)",
            |opts| {
                let include_symbols = opts.get("symbols").map(|v| v != "false").unwrap_or(true);
                Ok(Box::new(DeadCodeEliminationPass { include_symbols }))
            },
        );
        reg.register(
            "cse",
            "Common Subexpression Elimination (deduplicates identical pure operations)",
            |_opts| Ok(Box::new(CommonSubexpressionEliminationPass)),
        );
        reg.register(
            "gvn",
            "Dominator-aware Global Value Numbering across nested regions and basic blocks",
            |_opts| Ok(Box::new(GlobalValueNumberingPass)),
        );
        reg.register(
            "fold-views",
            "Unified folding of transpose, reshape, broadcast, slice/concat, and convert views",
            |_opts| Ok(Box::new(FoldViewsPass)),
        );
        reg.register(
            "simplify-algebraic",
            "Algebraic identity simplification (x+0->x, x*1->x, x*0->0, identity convert/slice)",
            |_opts| Ok(Box::new(SimplifyAlgebraicPass)),
        );
        reg.register(
            "egraph-simplify",
            "Applies distributive algebraic factoring (a*b + a*c -> a*(b+c)) and FMA extraction",
            |opts| {
                let fuse_fma = opts
                    .get("fuse_fma")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                Ok(Box::new(EgraphSimplifyPass { fuse_fma }))
            },
        );
        reg.register(
            "fold-constants",
            "Constant folding & conditional propagation (arithmetic, shape, compare/select)",
            |opts| {
                let include_conditional = opts
                    .get("conditional")
                    .map(|v| v != "false")
                    .unwrap_or(true);
                Ok(Box::new(FoldConstantsPass {
                    include_conditional,
                }))
            },
        );
        reg.register(
            "verify-shapes",
            "Verifies StableHLO tensor shapes and rank consistency across all ops",
            |_opts| Ok(Box::new(VerifyShapesPass)),
        );
        reg.register(
            "quantize-weights",
            "Post-training symmetric weight quantization (bits=8|4) with uniform_dequantize",
            |opts| {
                let bits = opts
                    .get("bits")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(8);
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(QuantizeWeightsPass { bits, min_elements }))
            },
        );
        reg.register(
            "quantize-groupwise",
            "Group-wise AWQ/GPTQ weight quantization (bits=4|8, group_size=32, awq_salient=true)",
            |opts| {
                let bits = opts
                    .get("bits")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(4);
                let group_size = opts
                    .get("group_size")
                    .or_else(|| opts.get("group"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32);
                let awq_salient = opts
                    .get("awq_salient")
                    .map(|s| s != "false")
                    .unwrap_or(true);
                Ok(Box::new(QuantizeGroupwisePass {
                    bits,
                    group_size,
                    awq_salient,
                }))
            },
        );
        reg.register(
            "quantize-fp8",
            "Applies FP8 (E4M3/E5M2) and MXFP8 block-scaled quantization to weights and GEMMs",
            |opts| {
                let format = opts
                    .get("format")
                    .cloned()
                    .unwrap_or_else(|| "e4m3".to_string());
                let mxfp8 = opts.get("mxfp8").map(|s| s != "false").unwrap_or(true);
                let block_size = opts
                    .get("block_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32);
                Ok(Box::new(QuantizeFp8Pass {
                    format,
                    mxfp8,
                    block_size,
                }))
            },
        );
        reg.register(
            "calibrate-quantize",
            "Calibrates activation dynamic ranges and inserts static W8A8 quantization scales",
            |opts| {
                let bits = opts
                    .get("bits")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(8);
                let num_samples = opts
                    .get("samples")
                    .or_else(|| opts.get("num_samples"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                Ok(Box::new(CalibrateQuantizePass { bits, num_samples }))
            },
        );
        reg.register(
            "mixed-precision",
            "Applies Automatic Mixed Precision (f16/bf16) casts to GEMM/Conv ops",
            |opts| {
                let dtype_str = opts.get("dtype").map(|s| s.to_ascii_lowercase());
                let target_dtype = match dtype_str.as_deref() {
                    Some("bf16") | Some("bfloat16") => DType::BF16,
                    _ => DType::F16,
                };
                Ok(Box::new(MixedPrecisionPass { target_dtype }))
            },
        );
        reg.register(
            "prune-structured",
            "Structured L1 channel pruning on 2D weight matrices in dot_general (sparsity=0.5)",
            |opts| {
                let sparsity = opts
                    .get("sparsity")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(StructuredPruningPass { sparsity }))
            },
        );
        reg.register(
            "sparsify-2to4",
            "Applies NVIDIA Ampere/Hopper 2:4 fine-grained structured sparsity to 2D GEMM weights",
            |_opts| Ok(Box::new(Sparsify2To4Pass)),
        );
        reg.register(
            "merge-weights",
            "Merges paired constant weight tensors (method=linear|slerp|ties, t=0.5)",
            |opts| {
                let method = match opts.get("method").map(|s| s.to_lowercase()).as_deref() {
                    Some("slerp") => MergeMethod::Slerp,
                    Some("ties") => MergeMethod::Ties,
                    _ => MergeMethod::Linear,
                };
                let t = opts
                    .get("t")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(MergeWeightsPass { method, t }))
            },
        );
        reg.register(
            "bake-lora",
            "Fuses LoRA adapter branches W + alpha * (A @ B) into base weight matrices",
            |opts| {
                let alpha = opts
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                Ok(Box::new(BakeLoraPass { alpha }))
            },
        );
        reg.register(
            "task-arithmetic",
            "Applies multi-checkpoint task vector arithmetic and DARE pruning (lambda, drop_rate)",
            |opts| {
                let lambda = opts
                    .get("lambda")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let drop_rate = opts
                    .get("drop_rate")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                Ok(Box::new(TaskArithmeticPass { lambda, drop_rate }))
            },
        );
        reg.register(
            "fold-batchnorm",
            "Absorbs RMSNorm/LayerNorm/BatchNorm scales into adjacent conv or dot_general weights",
            |_opts| Ok(Box::new(FoldBatchNormPass)),
        );
        reg.register(
            "pack-weights",
            "Pre-packs 2D constant RHS weights from [K, N] to [N, K] for unit-stride SIMD MatMul",
            |_opts| Ok(Box::new(PackWeightsPass)),
        );
        reg.register(
            "externalize-weights",
            "Converts inline dense<...> weights to dense_resource<key> in TensorStore",
            |opts| {
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let inline = opts.get("inline").is_some_and(|v| v == "true");
                Ok(Box::new(ExternalizeWeightsPass {
                    min_elements,
                    inline,
                }))
            },
        );
        reg.register(
            "fuse-elementwise",
            "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite",
            |_opts| Ok(Box::new(FuseElementwisePass)),
        );
        reg.register(
            "fuse-gemm",
            "Unified GEMM fusion (horizontal QKV/MLP dot, dequant+GEMM, and GEMM+bias/activation)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .cloned()
                    .unwrap_or_else(|| "all".to_string());
                Ok(Box::new(FuseGemmPass { kind }))
            },
        );
        reg.register(
            "fuse-norm",
            "Unified normalization fusion (Residual+RMSNorm, RMSNorm, and LayerNorm)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .cloned()
                    .unwrap_or_else(|| "all".to_string());
                Ok(Box::new(FuseNormPass { kind }))
            },
        );
        reg.register(
            "fuse-activations",
            "Unified activation & gated-MLP fusion (SwiGLU, GELU, GeGLU, and FusedGatedMLP)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .cloned()
                    .unwrap_or_else(|| "all".to_string());
                Ok(Box::new(FuseActivationsPass { kind }))
            },
        );
        reg.register(
            "fuse-softmax",
            "Fuses Softmax and LogSoftmax subgraphs into stablehlo.composite",
            |_opts| Ok(Box::new(FuseSoftmaxPass)),
        );
        reg.register(
            "fuse-rope",
            "Fuses decomposed Rotary Position Embedding (RoPE) subgraphs into stablehlo.composite",
            |_opts| Ok(Box::new(FuseRopePass)),
        );
        reg.register(
            "fuse-attention",
            "Fuses Q*K^T + scale + softmax + P*V (with GQA support) into stablehlo.composite",
            |_opts| Ok(Box::new(FuseAttentionPass)),
        );
        reg.register(
            "fuse-moe",
            "Fuses routed Mixture-of-Experts (MoE) expert GEMMs and gate combine into fused_moe",
            |_opts| Ok(Box::new(FuseMoePass)),
        );
        reg.register(
            "fuse-map-reduce",
            "Fuses elementwise map producers directly into stablehlo.reduce (fused_map_reduce)",
            |_opts| Ok(Box::new(FuseMapReducePass)),
        );
        reg.register(
            "fuse-megakernel",
            "Fuses sequential compute ops into a persistent grid-synchronized megakernel",
            |_opts| Ok(Box::new(FuseMegakernelPass)),
        );
        reg.register(
            "decompose-composites",
            "Expands stablehlo.composite operations back into primitive StableHLO ops",
            |_opts| Ok(Box::new(DecomposeCompositesPass)),
        );
        reg.register(
            "canonicalize",
            "Composite canonicalization pass (fold-views + simplify-algebraic + cse + dce)",
            |_opts| Ok(Box::new(CanonicalizePass)),
        );
        reg.register(
            "inliner",
            "Inlines func.call operations by splicing callee bodies into caller blocks",
            |_opts| Ok(Box::new(InlinerPass)),
        );
        reg.register(
            "symbol-dce",
            "Symbol Dead Code Elimination (removes unreachable private func.func symbols)",
            |_opts| Ok(Box::new(SymbolDcePass)),
        );
        reg.register(
            "topological-sort",
            "Reorders operations inside blocks to satisfy SSA def-before-use ordering",
            |_opts| Ok(Box::new(TopologicalSortPass)),
        );
        reg.register(
            "hoist-constants",
            "Hoists stablehlo.constant ops to block entry and deduplicates identical constants",
            |_opts| Ok(Box::new(HoistConstantsPass)),
        );
        reg.register(
            "licm",
            "Loop/Region-Invariant Code Motion (hoists invariant pure ops out of nested regions)",
            |_opts| Ok(Box::new(LicmPass)),
        );
        reg.register(
            "specialize-shapes",
            "Binds dynamic/symbolic dimensions (?, B, S) and propagates static shapes across IR",
            |opts| {
                let mut default_dim = 1i64;
                let mut bindings = IndexMap::new();
                for (k, v) in opts {
                    if let Ok(n) = v.parse::<i64>() {
                        if k == "default" || k == "?" {
                            default_dim = n;
                        } else {
                            bindings.insert(k.clone(), n);
                        }
                    }
                }
                Ok(Box::new(SpecializeShapesPass {
                    default_dim,
                    bindings,
                }))
            },
        );
        reg.register(
            "resize-batch",
            "Rewrites leading batch dimension across activations while preserving 2D GEMM weights",
            |opts| {
                let batch = opts
                    .get("batch")
                    .or_else(|| opts.get("size"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(4);
                Ok(Box::new(ResizeBatchPass { batch }))
            },
        );
        reg.register(
            "bucket-shapes",
            "Synthesizes multi-bucket shape-specialized functions (buckets=1;4;8) for dispatch",
            |opts| {
                let buckets = opts
                    .get("buckets")
                    .cloned()
                    .unwrap_or_else(|| "1;4;8".to_string());
                Ok(Box::new(BucketShapesPass { buckets }))
            },
        );
        reg.register(
            "convert-layout",
            "Inserts boundary layout-conversion transposes (NCHW<->NHWC / row<->col) on ops",
            |opts| {
                let layout = opts
                    .get("layout")
                    .or_else(|| opts.get("target"))
                    .cloned()
                    .unwrap_or_else(|| "nhwc".to_string());
                Ok(Box::new(ConvertLayoutPass { layout }))
            },
        );
        reg.register(
            "split-large-gemm",
            "Splits wide dot_general GEMMs (N > max_n) into sliced sub-GEMMs + concatenate",
            |opts| {
                let max_n = opts
                    .get("max_n")
                    .or_else(|| opts.get("limit"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(2);
                Ok(Box::new(SplitLargeGemmPass { max_n }))
            },
        );
        reg.register(
            "lower-conv-to-gemm",
            "Lowers stablehlo.convolution into im2col + stablehlo.dot_general GEMM",
            |_opts| Ok(Box::new(LowerConvToGemmPass)),
        );
        reg.register(
            "bufferize-inplace",
            "Annotates elementwise ops whose input buffer dies at the op for in-place arena reuse",
            |_opts| Ok(Box::new(BufferizeInplacePass)),
        );
        reg.register(
            "rematerialize",
            "Recomputes long-lived cheap activations before late consumers to save memory",
            |opts| {
                let min_distance = opts
                    .get("min_distance")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(3);
                Ok(Box::new(RematerializePass { min_distance }))
            },
        );
        reg.register(
            "offload-activations",
            "Inserts castforge.host_offload (D2H/H2D pinned host memory) on long-lived tensors",
            |opts| {
                let min_span = opts
                    .get("min_span")
                    .or_else(|| opts.get("min_distance"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(OffloadActivationsPass { min_span }))
            },
        );
        reg.register(
            "eliminate-concat-copies",
            "Eliminates intermediate concatenate copies by routing producer outputs into dest",
            |_opts| Ok(Box::new(EliminateConcatCopiesPass)),
        );
        reg.register(
            "shard-spmd",
            "Partitions GEMM ops for Megatron-style SPMD tensor parallelism and inserts all_reduce",
            |opts| {
                let num_devices = opts
                    .get("devices")
                    .or_else(|| opts.get("num_devices"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(ShardSpmdPass { num_devices }))
            },
        );
        reg.register(
            "shard-ring-attention",
            "Shards sequence context across a device ring with overlapped P2P KV send/recv",
            |opts| {
                let ring_size = opts
                    .get("ring_size")
                    .or_else(|| opts.get("devices"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                Ok(Box::new(ShardRingAttentionPass { ring_size }))
            },
        );
        reg.register(
            "hoist-all-reduce",
            "Coalesces linear all_reduce sums, eliminates redundant collectives, and overlaps",
            |_opts| Ok(Box::new(HoistAllReducePass)),
        );
        reg.register(
            "schedule-streams",
            "Assigns independent branches to concurrent streams with stream_wait barriers",
            |opts| {
                let num_streams = opts
                    .get("streams")
                    .or_else(|| opts.get("num_streams"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(ScheduleStreamsPass { num_streams }))
            },
        );
        reg.register(
            "inject-kv-cache",
            "Injects autoregressive KV-cache updates (castforge.kv_cache_update) before attention",
            |opts| {
                let max_seq_len = opts
                    .get("max_seq_len")
                    .or_else(|| opts.get("seq_len"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2048);
                Ok(Box::new(InjectKvCachePass { max_seq_len }))
            },
        );
        reg.register(
            "lower-paged-attention",
            "Lowers fused_attention to vLLM-style PagedAttention with block-table KV addressing",
            |opts| {
                let page_block_size = opts
                    .get("page_block_size")
                    .or_else(|| opts.get("block_size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let max_blocks = opts
                    .get("max_blocks")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(64);
                Ok(Box::new(LowerPagedAttentionPass {
                    page_block_size,
                    max_blocks,
                }))
            },
        );
        reg.register(
            "flash-decoding",
            "Partitions KV sequence across parallel splits with LSE reduction for LLM decoding",
            |opts| {
                let split_kv = opts
                    .get("split_kv")
                    .or_else(|| opts.get("splits"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                Ok(Box::new(FlashDecodingPass { split_kv }))
            },
        );
        reg.register(
            "speculative-tree-attention",
            "Configures Medusa/EAGLE speculative decoding tree-attention verification masks",
            |opts| {
                let draft_tokens = opts
                    .get("draft_tokens")
                    .or_else(|| opts.get("tokens"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let tree_depth = opts
                    .get("tree_depth")
                    .or_else(|| opts.get("depth"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(SpeculativeTreeAttentionPass {
                    draft_tokens,
                    tree_depth,
                }))
            },
        );
        reg.register(
            "autodiff",
            "Synthesizes reverse-mode VJP gradient ops in StableHLO (optional SGD step via lr=...)",
            |opts| {
                let learning_rate = opts
                    .get("lr")
                    .or_else(|| opts.get("learning_rate"))
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                Ok(Box::new(AutodiffPass { learning_rate }))
            },
        );
        reg.register(
            "schedule-loops",
            "Unified loop schedule pass (tile, permute, unroll, vectorize, and pad-for-tiling)",
            |opts| {
                let tile_size = opts
                    .get("size")
                    .or_else(|| opts.get("tile"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let l2_tile = opts
                    .get("l2_size")
                    .or_else(|| opts.get("l2_tile"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let reg_tile = opts
                    .get("reg_tile")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let order = opts
                    .get("order")
                    .cloned()
                    .unwrap_or_else(|| "mkn".to_string());
                let collapse = opts
                    .get("collapse")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let skew = opts
                    .get("skew")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let unroll_factor = opts
                    .get("factor")
                    .or_else(|| opts.get("unroll"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let prefetch = opts
                    .get("prefetch")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let split_k = opts
                    .get("split_k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1);
                let double_buffer = opts
                    .get("double_buffer")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let isa = opts
                    .get("isa")
                    .cloned()
                    .unwrap_or_else(|| "avx2".to_string());
                let default_w = match isa.to_ascii_lowercase().as_str() {
                    "avx512" | "avx-512" | "avx512f" => 16,
                    "neon" | "aarch64" | "arm64" | "wasm" | "wasm32" | "simd128" => 4,
                    _ => 8,
                };
                let vector_width = opts
                    .get("width")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(default_w);
                let parallel = opts
                    .get("parallel")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(true);
                let align = opts
                    .get("align")
                    .or_else(|| opts.get("multiple"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(8);
                Ok(Box::new(ScheduleLoopsPass {
                    tile_size,
                    l2_tile,
                    reg_tile,
                    order,
                    collapse,
                    skew,
                    unroll_factor,
                    prefetch,
                    split_k,
                    double_buffer,
                    isa,
                    vector_width,
                    parallel,
                    align,
                }))
            },
        );
        reg.register(
            "fuse-loops",
            "Fuses multi-op elementwise loop chains into a single register-pipelined loop",
            |_opts| Ok(Box::new(FuseLoopsPass)),
        );
        reg.register(
            "fission-loops",
            "Distributes long fused_loop_chain ops into balanced sub-loops (max_steps=2)",
            |opts| {
                let max_steps = opts
                    .get("max_steps")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(FissionLoopsPass { max_steps }))
            },
        );
        reg.register(
            "ir-edit",
            "Surgical IR editing pass (rename_op=from:to, set_attr=op:key:val)",
            |opts| {
                let rename_op = opts.get("rename_op").cloned();
                let annotate_op = opts.get("annotate_op").cloned();
                Ok(Box::new(IrEditPass {
                    rename_op,
                    annotate_op,
                }))
            },
        );
        reg.register(
            "extract-subgraph",
            "Extracts backward dataflow slice for a target SSA value into a standalone function",
            |opts| {
                let target = opts
                    .get("target")
                    .or_else(|| opts.get("output"))
                    .cloned()
                    .unwrap_or_default();
                let stop_at = opts
                    .get("stop_at")
                    .or_else(|| opts.get("inputs"))
                    .cloned()
                    .unwrap_or_default();
                Ok(Box::new(ExtractSubgraphPass { target, stop_at }))
            },
        );
        reg.register(
            "outline-subgraphs",
            "Outlines GEMM/composite subgraphs into private func.func kernels + func.call sites",
            |opts| {
                let mode = opts
                    .get("mode")
                    .cloned()
                    .unwrap_or_else(|| "all".to_string());
                Ok(Box::new(OutlineSubgraphsPass { mode }))
            },
        );
        reg.register(
            "unroll-calls",
            "Unrolls repeated func.call invocations (repeat=N) for stacked layer blocks",
            |opts| {
                let repeat = opts
                    .get("repeat")
                    .or_else(|| opts.get("count"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(UnrollCallsPass { repeat }))
            },
        );
        reg.register(
            "compose-funcs",
            "Synthesizes a composed entry function chaining two module functions (f2(f1(x)))",
            |opts| {
                let first = opts.get("first").cloned().unwrap_or_default();
                let second = opts.get("second").cloned().unwrap_or_default();
                let entry_name = opts
                    .get("entry")
                    .or_else(|| opts.get("name"))
                    .cloned()
                    .unwrap_or_else(|| "composed_main".to_string());
                Ok(Box::new(ComposeFuncsPass {
                    first,
                    second,
                    entry_name,
                }))
            },
        );
        reg.register(
            "dedup-functions",
            "Merges structurally identical func.func definitions and rewrites func.call sites",
            |_opts| Ok(Box::new(DedupFunctionsPass)),
        );
        reg.register(
            "freeze-inputs",
            "Binds func.func input arguments to in-body stablehlo.constant operations",
            |opts| {
                let args = opts.get("args").cloned().unwrap_or_default();
                let value = opts
                    .get("value")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(FreezeInputsPass { args, value }))
            },
        );
        reg.register(
            "parameterize-constants",
            "Hoists in-body stablehlo.constant weight tensors into explicit func.func arguments",
            |opts| {
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(ParameterizeConstantsPass { min_elements }))
            },
        );
        reg.register(
            "tap-intermediates",
            "Appends intermediate SSA activations to func.return for multi-output layer inspection",
            |opts| {
                let values = opts
                    .get("values")
                    .or_else(|| opts.get("targets"))
                    .cloned()
                    .unwrap_or_default();
                Ok(Box::new(TapIntermediatesPass { values }))
            },
        );
        reg.register(
            "reduce-ir",
            "Delta-reduces a function to a minimal reproducer of at most max_ops operations",
            |opts| {
                let max_ops = opts
                    .get("max_ops")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1);
                let keep_op = opts.get("keep_op").cloned().unwrap_or_default();
                Ok(Box::new(ReduceIrPass { max_ops, keep_op }))
            },
        );
        reg.register(
            "bisect-passes",
            "Verifies IR invariants and stamps a numbered bisection checkpoint on entry func",
            |opts| {
                let limit = opts
                    .get("limit")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1);
                Ok(Box::new(BisectPassesPass { limit }))
            },
        );
        reg.register(
            "sanitize-numerics",
            "Injects castforge.nan_guard clamps on numerically sensitive ops and function outputs",
            |opts| {
                let clamp_max = opts
                    .get("clamp_max")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(65504.0);
                let eps = opts
                    .get("eps")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1e-6);
                Ok(Box::new(SanitizeNumericsPass { clamp_max, eps }))
            },
        );
        reg.register(
            "normalize-ssa",
            "Renames SSA values in canonical topological order and sorts operation attributes",
            |opts| {
                let prefix = opts
                    .get("prefix")
                    .cloned()
                    .unwrap_or_else(|| "v".to_string());
                Ok(Box::new(NormalizeSsaPass { prefix }))
            },
        );
        reg.register(
            "hash-ir",
            "Computes FNV-1a 64-bit structural hash of functions and stamps castforge.ir_hash",
            |_opts| Ok(Box::new(HashIrPass)),
        );
        reg.register(
            "verify-equiv",
            "Verifies structural/signature invariants across pass chains",
            |opts| {
                let tolerance = opts
                    .get("tol")
                    .or_else(|| opts.get("tolerance"))
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1e-4);
                Ok(Box::new(VerifyEquivPass { tolerance }))
            },
        );
        reg.register(
            "instrument-profiling",
            "Injects castforge.profile_region timing markers around target compute operations",
            |opts| {
                let filter = opts
                    .get("filter")
                    .or_else(|| opts.get("ops"))
                    .cloned()
                    .unwrap_or_else(|| "gemm".to_string());
                Ok(Box::new(InstrumentProfilingPass { filter }))
            },
        );
        reg.register(
            "print-op-stats",
            "Computes operation counts, parameter sizes, FLOPs, and Roofline arithmetic intensity",
            |opts| {
                let ridge = opts
                    .get("ridge")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(10.0);
                Ok(Box::new(PrintOpStatsPass {
                    ridge_flops_per_byte: ridge,
                }))
            },
        );
        reg.register(
            "strip-metadata",
            "Strips debug locations and transient castforge.* attributes for portable export",
            |opts| {
                let debuginfo = opts.get("debuginfo").map(|s| s != "false").unwrap_or(true);
                let attributes = opts.get("attributes").map(|s| s != "false").unwrap_or(true);
                Ok(Box::new(StripMetadataPass {
                    debuginfo,
                    attributes,
                }))
            },
        );
        reg.register(
            "O1",
            "Preset O1: Inlining, fold-constants, LICM, GVN, fold-views, canonicalize, symbol-dce",
            |_opts| Ok(Box::new(OptO1Pass)),
        );
        reg.register(
            "O2",
            "Preset O2: O1 + LoRA/BatchNorm weight folding + unified Transformer/GEMM fusions",
            |_opts| Ok(Box::new(OptO2Pass)),
        );
        reg.register(
            "O3",
            "Preset O3: Shape specialization + O2 fusions + weight packing + inplace arena",
            |opts| Ok(Box::new(build_o3_preset("O3", opts))),
        );
        reg.register(
            "llm-opt",
            "Preset llm-opt: O3 + MoE fusion + egraph-simplify + hoist-all-reduce + zero-copy",
            |opts| Ok(Box::new(build_o3_preset("llm-opt", opts))),
        );
        reg
    }

    pub fn register<F>(&mut self, name: &'static str, description: &'static str, factory: F)
    where
        F: Fn(&PassOptions) -> Result<Box<dyn Pass>> + Send + Sync + 'static,
    {
        self.passes.insert(
            name,
            PassDescriptor {
                name,
                description,
                factory: Box::new(factory),
            },
        );
    }

    pub fn create_pass(&self, name: &str, opts: &PassOptions) -> Result<Box<dyn Pass>> {
        let desc = self.passes.get(name).ok_or_else(|| {
            anyhow!("Unknown pass `{name}`. Run `castforge list-passes` to see available passes.")
        })?;
        (desc.factory)(opts)
    }

    pub fn list(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.passes.values().map(|d| (d.name, d.description))
    }
}

fn build_o3_preset(preset_name: &'static str, opts: &PassOptions) -> OptO3Pass {
    let mut default_dim = 1i64;
    let mut bindings = IndexMap::new();
    for (k, v) in opts {
        if let Ok(n) = v.parse::<i64>() {
            if k == "default" || k == "?" {
                default_dim = n;
            } else {
                bindings.insert(k.clone(), n);
            }
        }
    }
    OptO3Pass {
        preset_name,
        default_dim,
        bindings,
    }
}
