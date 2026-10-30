use crate::{
    AutodiffPass, CanonicalizePass, DeadCodeEliminationPass, DecomposeCompositesPass,
    FoldConstantsPass, FoldViewsPass, FoldWeightsPass, FuseActivationsPass, FuseGemmPass,
    FuseNormPass, IrEditPass, OptO1Pass, OptO2Pass, OptO3Pass, OptimizeMemoryPass, Pass,
    PassOptions, PruneWeightsPass, QuantizePass, ScheduleLoopsPass, ShardDistributedPass,
    TopologicalSortPass, UnifiedCsePass, UnifiedFuseAttentionPass, UnifiedFuseElementwisePass,
    UnifiedMergeWeightsPass, UnifiedOutlineSubgraphsPass, UnifiedProfileIrPass,
    UnifiedSimplifyAlgebraicPass, UnifiedSpecializeShapesPass, UnifiedVerifyIrPass,
};
use anyhow::{Result, anyhow};
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

    /// Create a registry pre-populated with all 29 built-in CastForge passes and presets.
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
            "Unified CSE, dominator-aware GVN, LICM, and entry constant hoisting",
            |opts| {
                let gvn = opts.get("gvn").map(|v| v != "false").unwrap_or(true);
                let hoist_constants = opts
                    .get("hoist_constants")
                    .or_else(|| opts.get("hoist"))
                    .is_some_and(|v| v == "true" || v == "1");
                let licm = opts
                    .get("licm")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                Ok(Box::new(UnifiedCsePass {
                    gvn,
                    hoist_constants,
                    licm,
                }))
            },
        );
        reg.register(
            "fold-views",
            "Unified folding of transpose, reshape, broadcast, slice/concat, convert, and layout",
            |opts| {
                let layout = opts.get("layout").or_else(|| opts.get("target")).cloned();
                let propagate = opts
                    .get("propagate")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(FoldViewsPass { layout, propagate }))
            },
        );
        reg.register(
            "simplify-algebraic",
            "Unified algebraic identities (x+0->x, x*1->x, x*0->0) and e-graph FMA/factoring",
            |opts| {
                let egraph = opts
                    .get("egraph")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                let fuse_fma = opts
                    .get("fuse_fma")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                Ok(Box::new(UnifiedSimplifyAlgebraicPass { egraph, fuse_fma }))
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
            "quantize",
            "Unified quantization & precision (weight|awq|fp8|nvfp4|f16|bf16|w8a8|smooth|hadamard)",
            |opts| {
                let scheme = opts
                    .get("scheme")
                    .or_else(|| opts.get("mode"))
                    .or_else(|| opts.get("dtype"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts.contains_key("alpha") {
                            "smoothquant".to_string()
                        } else if opts.contains_key("salient_ratio") {
                            "awq".to_string()
                        } else if opts.contains_key("group_size")
                            || opts.contains_key("awq_salient")
                        {
                            "groupwise".to_string()
                        } else if opts.contains_key("format") || opts.contains_key("mxfp8") {
                            "fp8".to_string()
                        } else if opts.contains_key("samples") || opts.contains_key("num_samples") {
                            "w8a8".to_string()
                        } else if opts.contains_key("block_size") && !opts.contains_key("bits") {
                            "hadamard".to_string()
                        } else {
                            "weight".to_string()
                        }
                    });
                let default_bits = if matches!(
                    scheme.as_str(),
                    "groupwise" | "awq" | "gptq" | "fp4" | "mxfp4" | "nvfp4"
                ) {
                    4
                } else {
                    8
                };
                let bits = opts
                    .get("bits")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(default_bits);
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let group_size = opts
                    .get("group_size")
                    .or_else(|| opts.get("group"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32);
                let awq_salient = opts
                    .get("awq_salient")
                    .map(|s| s != "false")
                    .unwrap_or(true);
                let salient_ratio = opts
                    .get("salient_ratio")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.01);
                let format = opts
                    .get("format")
                    .or_else(|| opts.get("dtype"))
                    .cloned()
                    .unwrap_or_else(|| "e4m3".to_string());
                let mxfp8 = opts.get("mxfp8").map(|s| s != "false").unwrap_or(true);
                let block_size = opts
                    .get("block_size")
                    .or_else(|| opts.get("size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32);
                let samples = opts
                    .get("samples")
                    .or_else(|| opts.get("num_samples"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let alpha = opts
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                let hadamard = opts
                    .get("hadamard")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(QuantizePass {
                    scheme,
                    bits,
                    min_elements,
                    group_size,
                    awq_salient,
                    salient_ratio,
                    format,
                    mxfp8,
                    block_size,
                    samples,
                    alpha,
                    hadamard,
                }))
            },
        );
        reg.register(
            "prune-weights",
            "Unified weight & activation sparsity (mode=structured|2to4|activation_sparse)",
            |opts| {
                let mode = opts
                    .get("mode")
                    .cloned()
                    .unwrap_or_else(|| "structured".to_string());
                let sparsity = opts
                    .get("sparsity")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(PruneWeightsPass { mode, sparsity }))
            },
        );
        reg.register(
            "merge-weights",
            "Unified model merging & adapter baking (method=linear|slerp|ties|dare|lora)",
            |opts| {
                let method = opts.get("method").cloned().unwrap_or_else(|| {
                    if opts.contains_key("alpha") {
                        "lora".to_string()
                    } else if opts.contains_key("lambda") || opts.contains_key("drop_rate") {
                        "dare".to_string()
                    } else {
                        "linear".to_string()
                    }
                });
                let t = opts
                    .get("t")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                let alpha = opts
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let lambda = opts
                    .get("lambda")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let drop_rate = opts
                    .get("drop_rate")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                Ok(Box::new(UnifiedMergeWeightsPass {
                    method,
                    t,
                    alpha,
                    lambda,
                    drop_rate,
                }))
            },
        );
        reg.register(
            "fold-weights",
            "Unified weight folding, SIMD pre-packing & SafeTensors externalization",
            |opts| {
                let inline = opts.get("inline").is_some_and(|v| v == "true" || v == "1");
                let mode = opts
                    .get("mode")
                    .or_else(|| opts.get("kind"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if inline {
                            "inline".to_string()
                        } else if opts.contains_key("min_elements") {
                            "externalize".to_string()
                        } else {
                            "batchnorm".to_string()
                        }
                    });
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(FoldWeightsPass {
                    mode,
                    min_elements,
                    inline,
                }))
            },
        );
        reg.register(
            "fuse-elementwise",
            "Unified elementwise, map-reduce, megakernel, and register loop-chain fusion",
            |opts| {
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts.contains_key("max_steps") {
                        "loops".to_string()
                    } else {
                        "composites".to_string()
                    }
                });
                let max_steps = opts
                    .get("max_steps")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                Ok(Box::new(UnifiedFuseElementwisePass { mode, max_steps }))
            },
        );
        reg.register(
            "fuse-gemm",
            "Unified GEMM fusion (horizontal QKV/MLP dot, dequant+GEMM, and GEMM+bias/act/res)",
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
            "Unified normalization & softmax fusion (RMSNorm, LayerNorm, Softmax, Cross-Entropy)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .or_else(|| opts.get("mode"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts.contains_key("chunk_size") || opts.contains_key("chunk") {
                            "cross_entropy".to_string()
                        } else {
                            "all".to_string()
                        }
                    });
                let chunk_size = opts
                    .get("chunk_size")
                    .or_else(|| opts.get("chunk"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(1024);
                Ok(Box::new(FuseNormPass { kind, chunk_size }))
            },
        );
        reg.register(
            "fuse-activations",
            "Unified activation, gated-MLP & MoE fusion (SwiGLU, GELU, GeGLU, GatedMLP, MoE)",
            |opts| {
                let shared_expert = opts
                    .get("shared_expert")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let kind = opts
                    .get("kind")
                    .or_else(|| opts.get("mode"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if shared_expert
                            || opts.contains_key("top_k")
                            || opts.contains_key("capacity_factor")
                        {
                            "moe".to_string()
                        } else {
                            "all".to_string()
                        }
                    });
                let top_k = opts
                    .get("top_k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let capacity_factor = opts
                    .get("capacity_factor")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.25);
                Ok(Box::new(FuseActivationsPass {
                    kind,
                    shared_expert,
                    top_k,
                    capacity_factor,
                }))
            },
        );
        reg.register(
            "fuse-attention",
            "Unified attention fusion & serving (SDPA/RoPE/MLA/KV-cache/Paged/Flash/Sliding/Share)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .or_else(|| opts.get("mode"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts.contains_key("kv_lora_rank")
                            || opts.contains_key("latent_rank")
                            || opts.contains_key("rank")
                        {
                            "mla".to_string()
                        } else if opts.contains_key("share_every") {
                            "kv_sharing".to_string()
                        } else if opts.contains_key("window_size")
                            || opts.contains_key("sink_tokens")
                        {
                            "sliding_window".to_string()
                        } else if opts.contains_key("page_block_size")
                            || opts.contains_key("block_size")
                            || opts.contains_key("max_blocks")
                        {
                            "paged".to_string()
                        } else if opts.contains_key("split_kv") || opts.contains_key("splits") {
                            "flash_decoding".to_string()
                        } else if opts.contains_key("draft_tokens")
                            || opts.contains_key("tokens")
                            || opts.contains_key("tree_depth")
                            || opts.contains_key("depth")
                        {
                            "tree".to_string()
                        } else if opts.contains_key("max_seq_len") || opts.contains_key("seq_len") {
                            "kv_cache".to_string()
                        } else {
                            "all".to_string()
                        }
                    });
                let kv_lora_rank = opts
                    .get("kv_lora_rank")
                    .or_else(|| opts.get("latent_rank"))
                    .or_else(|| opts.get("rank"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(512);
                let max_seq_len = opts
                    .get("max_seq_len")
                    .or_else(|| opts.get("seq_len"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2048);
                let page_block_size = opts
                    .get("page_block_size")
                    .or_else(|| opts.get("block_size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let max_blocks = opts
                    .get("max_blocks")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(64);
                let split_kv = opts
                    .get("split_kv")
                    .or_else(|| opts.get("splits"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
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
                let window_size = opts
                    .get("window_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4096);
                let sink_tokens = opts
                    .get("sink_tokens")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let share_every = opts
                    .get("share_every")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(UnifiedFuseAttentionPass {
                    kind,
                    kv_lora_rank,
                    max_seq_len,
                    page_block_size,
                    max_blocks,
                    split_kv,
                    draft_tokens,
                    tree_depth,
                    window_size,
                    sink_tokens,
                    share_every,
                }))
            },
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
            "topological-sort",
            "Reorders operations for SSA def-before-use and optional peak-memory minimization",
            |opts| {
                let mode = opts
                    .get("mode")
                    .cloned()
                    .unwrap_or_else(|| "standard".to_string());
                Ok(Box::new(TopologicalSortPass { mode }))
            },
        );
        reg.register(
            "specialize-shapes",
            "Unified symbolic shape specialization, static batch resizing, and bucket dispatch",
            |opts| {
                let mut default_dim = 1i64;
                let mut bindings = IndexMap::new();
                let batch = opts
                    .get("batch")
                    .or_else(|| opts.get("size"))
                    .and_then(|s| s.parse::<i64>().ok());
                let buckets = opts.get("buckets").cloned();
                for (k, v) in opts {
                    if k == "batch" || k == "size" || k == "buckets" {
                        continue;
                    }
                    if let Ok(n) = v.parse::<i64>() {
                        if k == "default" || k == "?" {
                            default_dim = n;
                        } else {
                            bindings.insert(k.clone(), n);
                        }
                    }
                }
                Ok(Box::new(UnifiedSpecializeShapesPass {
                    default_dim,
                    bindings,
                    batch,
                    buckets,
                }))
            },
        );
        reg.register(
            "optimize-memory",
            "Unified memory pass (mode=all|inplace|zero_copy|remat|offload, min_distance)",
            |opts| {
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts.contains_key("min_distance") {
                        "remat".to_string()
                    } else if opts.contains_key("min_span") {
                        "offload".to_string()
                    } else {
                        "all".to_string()
                    }
                });
                let min_distance = opts
                    .get("min_distance")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(3);
                let min_span = opts
                    .get("min_span")
                    .or_else(|| opts.get("min_distance"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(OptimizeMemoryPass {
                    mode,
                    min_distance,
                    min_span,
                }))
            },
        );
        reg.register(
            "shard-distributed",
            "Unified SPMD TP, Ring CP, AllReduce coalescing, 1F1B PP, and multi-stream",
            |opts| {
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts.contains_key("ring_size") {
                        "ring_cp".to_string()
                    } else if opts.contains_key("stages")
                        || opts.contains_key("microbatches")
                        || opts.contains_key("schedule")
                    {
                        "pp".to_string()
                    } else if opts.contains_key("streams") || opts.contains_key("num_streams") {
                        "streams".to_string()
                    } else {
                        "tp".to_string()
                    }
                });
                let devices = opts
                    .get("devices")
                    .or_else(|| opts.get("num_devices"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let ring_size = opts
                    .get("ring_size")
                    .or_else(|| opts.get("devices"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let coalesce = opts
                    .get("coalesce")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                let stages = opts
                    .get("stages")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let microbatches = opts
                    .get("microbatches")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let schedule = opts
                    .get("schedule")
                    .cloned()
                    .unwrap_or_else(|| "1f1b".to_string());
                let streams = opts
                    .get("streams")
                    .or_else(|| opts.get("num_streams"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(ShardDistributedPass {
                    mode,
                    devices,
                    ring_size,
                    coalesce,
                    stages,
                    microbatches,
                    schedule,
                    streams,
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
            "Unified loop schedule, conv-to-gemm & tensor-core pass (tile, permute, unroll, max_n)",
            |opts| {
                let tile_size = opts
                    .get("size")
                    .or_else(|| opts.get("tile_size"))
                    .or_else(|| opts.get("tile").filter(|t| !t.contains('x')))
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
                let tensor_cores = opts
                    .get("tensor_cores")
                    .or_else(|| opts.get("arch"))
                    .map(|s| s.to_string())
                    .filter(|s| s != "none" && !s.is_empty());
                let mma_tile = opts
                    .get("mma_tile")
                    .or_else(|| opts.get("tile").filter(|t| t.contains('x')))
                    .map(|t| {
                        let parts: Vec<i64> =
                            t.split('x').filter_map(|p| p.parse::<i64>().ok()).collect();
                        if parts.len() == 3 {
                            (parts[0], parts[1], parts[2])
                        } else {
                            (16, 16, 16)
                        }
                    })
                    .unwrap_or((16, 16, 16));
                let max_n = opts
                    .get("max_n")
                    .or_else(|| opts.get("limit"))
                    .and_then(|s| s.parse::<i64>().ok());
                let lower_conv = opts
                    .get("lower_conv")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
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
                    tensor_cores,
                    mma_tile,
                    max_n,
                    lower_conv,
                }))
            },
        );
        reg.register(
            "ir-edit",
            "Unified IR surgery, freezing, tapping, SSA normalization, and metadata stripping",
            |opts| {
                let rename_op = opts.get("rename_op").cloned();
                let annotate_op = opts
                    .get("annotate_op")
                    .or_else(|| opts.get("set_attr"))
                    .cloned();
                let unfreeze = opts
                    .get("unfreeze")
                    .or_else(|| opts.get("parameterize"))
                    .is_some_and(|v| v == "true" || v == "1")
                    || opts.contains_key("min_elements");
                let freeze = opts.get("freeze").is_some_and(|v| v == "true" || v == "1")
                    || opts.contains_key("args")
                    || opts.contains_key("value");
                let args = opts.get("args").cloned().unwrap_or_default();
                let value = opts
                    .get("value")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.5);
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let tap = opts
                    .get("tap")
                    .or_else(|| opts.get("values"))
                    .or_else(|| opts.get("targets"))
                    .cloned();
                let ssa = opts
                    .get("ssa")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or_else(|| opts.contains_key("prefix"));
                let prefix = opts
                    .get("prefix")
                    .cloned()
                    .unwrap_or_else(|| "v".to_string());
                let debuginfo = opts
                    .get("debuginfo")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let attributes = opts
                    .get("attributes")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(IrEditPass {
                    rename_op,
                    annotate_op,
                    freeze,
                    unfreeze,
                    args,
                    value,
                    min_elements,
                    tap,
                    ssa,
                    prefix,
                    debuginfo,
                    attributes,
                }))
            },
        );
        reg.register(
            "outline-subgraphs",
            "Unified kernel outlining, inlining, extraction, dedup, composition & call unrolling",
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
                let max_ops = opts.get("max_ops").and_then(|s| s.parse::<usize>().ok());
                let keep_op = opts.get("keep_op").cloned().unwrap_or_default();
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if max_ops.is_some() || !keep_op.is_empty() {
                        "reduce".to_string()
                    } else if !target.is_empty() || !stop_at.is_empty() {
                        "extract".to_string()
                    } else if opts.get("inline").is_some_and(|v| v == "true" || v == "1") {
                        "inline".to_string()
                    } else {
                        "all".to_string()
                    }
                });
                let dedup = opts
                    .get("dedup")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                let first = opts.get("first").cloned().unwrap_or_default();
                let second = opts.get("second").cloned().unwrap_or_default();
                let entry_name = opts
                    .get("entry")
                    .or_else(|| opts.get("name"))
                    .cloned()
                    .unwrap_or_else(|| "composed_main".to_string());
                let repeat = opts
                    .get("repeat")
                    .or_else(|| opts.get("count"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                Ok(Box::new(UnifiedOutlineSubgraphsPass {
                    mode,
                    dedup,
                    first,
                    second,
                    entry_name,
                    repeat,
                    target,
                    stop_at,
                    max_ops,
                    keep_op,
                }))
            },
        );
        reg.register(
            "verify-ir",
            "Unified IR verifier, structural FNV-1a hasher, equivalence, sanitizer & bisection",
            |opts| {
                let sanitize = opts
                    .get("sanitize")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or_else(|| opts.contains_key("clamp_max") || opts.contains_key("eps"));
                let clamp_max = opts
                    .get("clamp_max")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(65504.0);
                let eps = opts
                    .get("eps")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1e-6);
                let shapes = opts
                    .get("shapes")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                let equiv = opts
                    .get("equiv")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!sanitize);
                let tolerance = opts
                    .get("tol")
                    .or_else(|| opts.get("tolerance"))
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1e-4);
                let hash = opts
                    .get("hash")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!sanitize);
                let bisect_limit = opts
                    .get("bisect")
                    .or_else(|| opts.get("limit"))
                    .and_then(|s| s.parse::<usize>().ok());
                Ok(Box::new(UnifiedVerifyIrPass {
                    shapes,
                    equiv,
                    tolerance,
                    hash,
                    bisect_limit,
                    sanitize,
                    clamp_max,
                    eps,
                }))
            },
        );
        reg.register(
            "profile-ir",
            "Unified Roofline FLOPs/intensity analysis and profiling region instrumentation",
            |opts| {
                let stats = opts
                    .get("stats")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                let ridge_flops_per_byte = opts
                    .get("ridge")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(10.0);
                let instrument = opts
                    .get("instrument")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or_else(|| opts.contains_key("filter") || opts.contains_key("ops"));
                let filter = opts
                    .get("filter")
                    .or_else(|| opts.get("ops"))
                    .cloned()
                    .unwrap_or_else(|| "gemm".to_string());
                Ok(Box::new(UnifiedProfileIrPass {
                    stats,
                    ridge_flops_per_byte,
                    instrument,
                    filter,
                }))
            },
        );
        reg.register(
            "O1",
            "Preset O1: Inlining, fold-constants, LICM, GVN, fold-views, canonicalize, dce",
            |_opts| Ok(Box::new(OptO1Pass)),
        );
        reg.register(
            "O2",
            "Preset O2: O1 + LoRA/BatchNorm weight folding + unified Transformer/GEMM fusions",
            |_opts| Ok(Box::new(OptO2Pass)),
        );
        reg.register(
            "O3",
            "Preset O3: Shape specialization + O2 + MoE/FMA/zero-copy + weight packing + arena",
            |opts| Ok(Box::new(build_o3_preset("O3", opts))),
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
