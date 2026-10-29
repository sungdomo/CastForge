use crate::{
    AutodiffPass, BakeLoraPass, BisectPassesPass, BucketShapesPass, BufferizeInplacePass,
    CalibrateQuantizePass, CanonicalizePass, CommonSubexpressionEliminationPass, ComposeFuncsPass,
    ConvertLayoutPass, DeadCodeEliminationPass, DecomposeCompositesPass, DedupFunctionsPass,
    EgraphSimplifyPass, ExternalizeWeightsPass, ExtractSubgraphPass, FissionLoopsPass,
    FoldBroadcastPass, FoldConstantsPass, FoldConvertPass, FoldNormWeightsPass, FoldReshapePass,
    FoldSliceConcatPass, FoldTransposePass, FreezeInputsPass, FuseAttentionPass,
    FuseDequantMatmulPass, FuseElementwisePass, FuseGatedMlpPass, FuseGeluPass,
    FuseHorizontalDotPass, FuseLayerNormPass, FuseLogSoftmaxPass, FuseLoopsPass, FuseMapReducePass,
    FuseMatmulEpiloguePass, FuseMoePass, FuseResidualNormPass, FuseRmsNormPass, FuseRopePass,
    FuseSoftmaxPass, FuseSwiGluPass, HashIrPass, HoistAllReducePass, HoistConstantsPass,
    InjectKvCachePass, InlinerPass, InstrumentProfilingPass, IrEditPass, LicmPass,
    LowerConvToGemmPass, MergeWeightsPass, MixedPrecisionPass, NormalizeSsaPass,
    OffloadActivationsPass, OptO1Pass, OptO2Pass, OptO3Pass, OutlineSubgraphsPass, PackWeightsPass,
    PadForTilingPass, ParameterizeConstantsPass, Pass, PassOptions, PermuteLoopsPass,
    PrintOpStatsPass, QuantizeWeightsPass, ReduceIrPass, RematerializePass, ResizeBatchPass,
    SanitizeNumericsPass, SccpPass, ScheduleStreamsPass, ShardSpmdPass, SimplifyAlgebraicPass,
    SinkTransposePass, Sparsify2To4Pass, SpecializeShapesPass, SplitLargeGemmPass,
    StripAttributesPass, StripDebugInfoPass, StructuredPruningPass, SymbolDcePass,
    TapIntermediatesPass, TaskArithmeticPass, TileLoopsPass, TopologicalSortPass, UnrollCallsPass,
    UnrollLoopsPass, VectorizeLoopsPass, VerifyEquivPass, VerifyShapesPass,
};
use anyhow::{Result, anyhow};
use castforge_ir::{DType, MergeMethod};
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
            "fold-transpose",
            "Folds chained stablehlo.transpose ops and eliminates identity transposes",
            |opts| {
                let eliminate_identity = opts
                    .get("eliminate_identity")
                    .map(|v| v != "false")
                    .unwrap_or(true);
                Ok(Box::new(FoldTransposePass { eliminate_identity }))
            },
        );
        reg.register(
            "fold-reshape",
            "Folds chained stablehlo.reshape ops and eliminates identity reshapes",
            |_opts| Ok(Box::new(FoldReshapePass)),
        );
        reg.register(
            "simplify-algebraic",
            "Algebraic identity simplification (x+0->x, x*1->x, x*0->0, identity convert/slice)",
            |_opts| Ok(Box::new(SimplifyAlgebraicPass)),
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
            "fuse-elementwise",
            "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite",
            |_opts| Ok(Box::new(FuseElementwisePass)),
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
            "canonicalize",
            "Composite canonicalization pass (fold-transpose + fold-reshape + cse + dce)",
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
            "sccp",
            "Sparse Conditional Constant Propagation (evaluates compare/select and constants)",
            |_opts| Ok(Box::new(SccpPass)),
        );
        reg.register(
            "topological-sort",
            "Reorders operations inside blocks to satisfy SSA def-before-use ordering",
            |_opts| Ok(Box::new(TopologicalSortPass)),
        );
        reg.register(
            "strip-debuginfo",
            "Strips source location metadata (loc(...)) from all operations in the module",
            |_opts| Ok(Box::new(StripDebugInfoPass)),
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
            "sink-transpose",
            "Sinks stablehlo.transpose through elementwise ops toward constants and producers",
            |_opts| Ok(Box::new(SinkTransposePass)),
        );
        reg.register(
            "fuse-matmul-epilogue",
            "Fuses dot_general + bias add (+ optional relu/tanh) into stablehlo.composite",
            |_opts| Ok(Box::new(FuseMatmulEpiloguePass)),
        );
        reg.register(
            "fold-norm-weights",
            "Absorbs constant RMSNorm/LayerNorm scale multiply into adjacent dot_general weights",
            |_opts| Ok(Box::new(FoldNormWeightsPass)),
        );
        reg.register(
            "fuse-softmax",
            "Fuses exp / broadcast(reduce_sum(exp)) into stablehlo.composite (fused_softmax)",
            |_opts| Ok(Box::new(FuseSoftmaxPass)),
        );
        reg.register(
            "fuse-rmsnorm",
            "Fuses x * rsqrt(mean(x^2) + eps) * weight into stablehlo.composite (fused_rmsnorm)",
            |_opts| Ok(Box::new(FuseRmsNormPass)),
        );
        reg.register(
            "decompose-composites",
            "Expands stablehlo.composite operations back into primitive StableHLO ops",
            |_opts| Ok(Box::new(DecomposeCompositesPass)),
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
            "fold-broadcast",
            "Composes chained broadcast_in_dim ops, removes identity broadcasts, hoists unary ops",
            |_opts| Ok(Box::new(FoldBroadcastPass)),
        );
        reg.register(
            "fold-slice-concat",
            "Composes nested slices and folds slices of concatenate ops directly to source inputs",
            |_opts| Ok(Box::new(FoldSliceConcatPass)),
        );
        reg.register(
            "fuse-horizontal-dot",
            "Horizontally fuses parallel dot_general ops sharing an LHS input (e.g. Q/K/V)",
            |_opts| Ok(Box::new(FuseHorizontalDotPass)),
        );
        reg.register(
            "fold-convert",
            "Eliminates redundant or round-trip stablehlo.convert casts and folds constant casts",
            |_opts| Ok(Box::new(FoldConvertPass)),
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
            "fuse-swiglu",
            "Fuses SwiGLU (gate / (1 + exp(-gate)) * up) into stablehlo.composite (fused_swiglu)",
            |_opts| Ok(Box::new(FuseSwiGluPass)),
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
            "pack-weights",
            "Pre-packs 2D constant RHS weights from [K, N] to [N, K] for unit-stride SIMD MatMul",
            |_opts| Ok(Box::new(PackWeightsPass)),
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
            "bufferize-inplace",
            "Annotates elementwise ops whose input buffer dies at the op for in-place arena reuse",
            |_opts| Ok(Box::new(BufferizeInplacePass)),
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
            "fuse-dequant-matmul",
            "Fuses uniform_dequantize + dot_general into on-the-fly quantized GEMM (fused_qmatmul)",
            |_opts| Ok(Box::new(FuseDequantMatmulPass)),
        );
        reg.register(
            "fuse-residual-norm",
            "Fuses add(x, residual) + RMSNorm into stablehlo.composite (fused_residual_rmsnorm)",
            |_opts| Ok(Box::new(FuseResidualNormPass)),
        );
        reg.register(
            "fuse-layernorm",
            "Fuses (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta into fused_layernorm",
            |_opts| Ok(Box::new(FuseLayerNormPass)),
        );
        reg.register(
            "fuse-gelu",
            "Fuses GELU and gated GeGLU (gelu(gate) * up) into stablehlo.composite",
            |_opts| Ok(Box::new(FuseGeluPass)),
        );
        reg.register(
            "fuse-gated-mlp",
            "Fuses dual gate/up dot_general + slices + SwiGLU/GeGLU into fused_gated_mlp",
            |_opts| Ok(Box::new(FuseGatedMlpPass)),
        );
        reg.register(
            "fuse-map-reduce",
            "Fuses elementwise map producers directly into stablehlo.reduce (fused_map_reduce)",
            |_opts| Ok(Box::new(FuseMapReducePass)),
        );
        reg.register(
            "fuse-log-softmax",
            "Fuses log(softmax(x)) or x - log(reduce_sum(exp(x))) into fused_log_softmax",
            |_opts| Ok(Box::new(FuseLogSoftmaxPass)),
        );
        reg.register(
            "lower-conv-to-gemm",
            "Lowers stablehlo.convolution into im2col + stablehlo.dot_general GEMM",
            |_opts| Ok(Box::new(LowerConvToGemmPass)),
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
            "fuse-moe",
            "Fuses routed Mixture-of-Experts (MoE) expert GEMMs and gate combine into fused_moe",
            |_opts| Ok(Box::new(FuseMoePass)),
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
            "tile-loops",
            "Translates GEMM ops by attaching L1/L2 cache and register tiling schedule attributes",
            |opts| {
                let size = opts
                    .get("size")
                    .or_else(|| opts.get("tile"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let tile_m = opts
                    .get("tile_m")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(size);
                let tile_n = opts
                    .get("tile_n")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(size);
                let tile_k = opts
                    .get("tile_k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(size);
                let l2_tile = opts
                    .get("l2_size")
                    .or_else(|| opts.get("l2_tile"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let reg = opts
                    .get("reg_tile")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let reg_m = opts
                    .get("reg_m")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(reg);
                let reg_n = opts
                    .get("reg_n")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(reg);
                Ok(Box::new(TileLoopsPass {
                    tile_m,
                    tile_n,
                    tile_k,
                    l2_tile,
                    reg_m,
                    reg_n,
                }))
            },
        );
        reg.register(
            "permute-loops",
            "Translates loop nests by attaching permutation (mnk/mkn/nkm), collapse, or skew attrs",
            |opts| {
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
                Ok(Box::new(PermuteLoopsPass {
                    order,
                    collapse,
                    skew,
                }))
            },
        );
        reg.register(
            "unroll-loops",
            "Translates reduction loops by attaching unroll, prefetch, split-k, and double-buffer",
            |opts| {
                let factor = opts
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
                Ok(Box::new(UnrollLoopsPass {
                    factor,
                    prefetch,
                    split_k,
                    double_buffer,
                }))
            },
        );
        reg.register(
            "vectorize-loops",
            "Translates compute ops by attaching SIMD vectorization, ISA (avx2/neon), and parallel",
            |opts| {
                let isa = opts
                    .get("isa")
                    .cloned()
                    .unwrap_or_else(|| "scalar".to_string());
                let default_w = if isa.eq_ignore_ascii_case("neon") {
                    4
                } else {
                    8
                };
                let width = opts
                    .get("width")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(default_w);
                let parallel = opts
                    .get("parallel")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(VectorizeLoopsPass {
                    width,
                    isa,
                    parallel,
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
            "dedup-functions",
            "Merges structurally identical func.func definitions and rewrites func.call sites",
            |_opts| Ok(Box::new(DedupFunctionsPass)),
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
            "sparsify-2to4",
            "Applies NVIDIA Ampere/Hopper 2:4 fine-grained structured sparsity to 2D GEMM weights",
            |_opts| Ok(Box::new(Sparsify2To4Pass)),
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
            "pad-for-tiling",
            "Computes tile-aligned (M, K, N) multiples for GEMM ops to eliminate remainder tails",
            |opts| {
                let align = opts
                    .get("align")
                    .or_else(|| opts.get("multiple"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(8);
                Ok(Box::new(PadForTilingPass { align }))
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
            "hoist-all-reduce",
            "Coalesces linear all_reduce sums, eliminates redundant collectives, and overlaps",
            |_opts| Ok(Box::new(HoistAllReducePass)),
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
            "strip-attributes",
            "Strips transient castforge.* and schedule attributes for portable StableHLO export",
            |_opts| Ok(Box::new(StripAttributesPass)),
        );
        reg.register(
            "O1",
            "Preset O1: Inlining, SCCP, LICM, fixed-point canonicalization, symbol-dce, toposort",
            |_opts| Ok(Box::new(OptO1Pass)),
        );
        reg.register(
            "O2",
            "Preset O2: O1 + LoRA/Norm weight folding + Transformer/MatMul operator fusions",
            |_opts| Ok(Box::new(OptO2Pass)),
        );
        reg.register(
            "O3",
            "Preset O3: Shape specialization + O2 fusions + weight packing + inplace arena",
            |opts| Ok(Box::new(build_o3_preset("O3", opts))),
        );
        reg.register(
            "llm-opt",
            "Preset llm-opt: Full LLM shape specialization, fusion, weight packing & inplace arena",
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
