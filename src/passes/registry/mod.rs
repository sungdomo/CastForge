//! Built-in [`PassRegistry`] and pass factory registration (`src/passes/registry`).

mod lowering;

use super::*;

impl Default for PassRegistry {
    fn default() -> Self {
        Self::with_builtin_passes()
    }
}

impl PassRegistry {
    /// Create a registry pre-populated with all built-in single-traversal CastForge passes and
    /// `O1`/`O2`/`O3` optimization presets.
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
            "Unified constant folding, SCCP, algebraic identities, e-graph factoring & FMA",
            |opts| {
                let fold_constants = opts
                    .get("fold_constants")
                    .or_else(|| opts.get("constants"))
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                let include_conditional = opts
                    .get("conditional")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                let const_only = opts
                    .get("mode")
                    .is_some_and(|m| m == "constants" || m == "fold");
                let egraph = opts
                    .get("egraph")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(!const_only);
                let fuse_fma = opts
                    .get("fuse_fma")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!const_only);
                Ok(Box::new(UnifiedSimplifyAlgebraicPass {
                    fold_constants,
                    include_conditional,
                    egraph,
                    fuse_fma,
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
            "Unified weight folding, SIMD pre-packing, SafeTensors externalization & pruning",
            |opts| {
                let inline = opts.get("inline").is_some_and(|v| v == "true" || v == "1");
                let mode = opts
                    .get("mode")
                    .or_else(|| opts.get("kind"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if inline {
                            "inline".to_string()
                        } else if opts.contains_key("sparsity") {
                            "structured".to_string()
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
                let sparsity = opts
                    .get("sparsity")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(FoldWeightsPass {
                    mode,
                    min_elements,
                    inline,
                    sparsity,
                }))
            },
        );
        reg.register(
            "prune-weights",
            "Compile-time weight pruning (mode=structured|2to4|activation_sparse, sparsity)",
            |opts| {
                let mode = opts
                    .get("mode")
                    .or_else(|| opts.get("kind"))
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
            "externalize-weights",
            "Externalize or inline weight constants between IR and SafeTensors TensorStore",
            |opts| {
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let inline = opts.get("inline").is_some_and(|v| v == "true" || v == "1");
                Ok(Box::new(ExternalizeWeightsPass {
                    min_elements,
                    inline,
                }))
            },
        );
        reg.register(
            "fuse-elementwise",
            "Unified elementwise, activation (SwiGLU/GELU/GeGLU/MoE), map-reduce & loop fusion",
            |opts| {
                let shared_expert = opts
                    .get("shared_expert")
                    .is_some_and(|s| s == "true" || s == "1");
                let mode = opts
                    .get("mode")
                    .or_else(|| opts.get("kind"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts
                            .get("decompose")
                            .is_some_and(|v| v == "true" || v == "1")
                        {
                            "decompose".to_string()
                        } else if shared_expert
                            || opts.contains_key("top_k")
                            || opts.contains_key("capacity_factor")
                        {
                            "moe".to_string()
                        } else if opts.contains_key("max_steps") {
                            "loops".to_string()
                        } else {
                            "composites".to_string()
                        }
                    });
                let max_steps = opts
                    .get("max_steps")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let top_k = opts
                    .get("top_k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let capacity_factor = opts
                    .get("capacity_factor")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.25);
                Ok(Box::new(UnifiedFuseElementwisePass {
                    mode,
                    max_steps,
                    shared_expert,
                    top_k,
                    capacity_factor,
                }))
            },
        );
        reg.register(
            "decompose-composites",
            "Expands castforge.* composite operations into primitive StableHLO ops",
            |_opts| Ok(Box::new(DecomposeCompositesPass)),
        );
        reg.register(
            "fuse-loops",
            "Fuses multi-step elementwise chains into castforge.fused_elementwise_chain",
            |_opts| Ok(Box::new(FuseLoopsPass)),
        );
        reg.register(
            "fission-loops",
            "Splits oversized fused_elementwise_chain ops at max_steps register-pressure limits",
            |opts| {
                let max_steps = opts
                    .get("max_steps")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(3);
                Ok(Box::new(FissionLoopsPass { max_steps }))
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
                        } else if opts.contains_key("max_seq_len")
                            || opts.contains_key("seq_len")
                            || opts.contains_key("kv_dtype")
                        {
                            "kv_cache".to_string()
                        } else if opts.contains_key("rope_scaling")
                            || opts.contains_key("rope_factor")
                        {
                            "rope".to_string()
                        } else if opts.contains_key("state_dim") {
                            "ssm".to_string()
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
                let rope_scaling = opts.get("rope_scaling").cloned();
                let rope_factor = opts
                    .get("rope_factor")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);
                let logit_softcap = opts
                    .get("logit_softcap")
                    .or_else(|| opts.get("softcap"))
                    .and_then(|s| s.parse::<f64>().ok());
                let qk_norm = opts.get("qk_norm").is_some_and(|v| v == "true" || v == "1");
                let kv_dtype = opts.get("kv_dtype").cloned();
                let state_dim = opts
                    .get("state_dim")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(128);
                let ssm_chunk_size = opts
                    .get("chunk_size")
                    .or_else(|| opts.get("ssm_chunk_size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(64);
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
                    rope_scaling,
                    rope_factor,
                    logit_softcap,
                    qk_norm,
                    kv_dtype,
                    state_dim,
                    ssm_chunk_size,
                }))
            },
        );
        reg.register(
            "canonicalize",
            "Composite canonicalization & topological/peak-memory scheduler (mode=all|min_memory)",
            |opts| {
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts
                        .get("min_memory")
                        .is_some_and(|v| v == "true" || v == "1")
                    {
                        "min_memory".to_string()
                    } else {
                        "all".to_string()
                    }
                });
                let toposort = opts
                    .get("toposort")
                    .is_some_and(|v| v == "true" || v == "1");
                Ok(Box::new(CanonicalizePass { mode, toposort }))
            },
        );
        lowering::register_lowering_passes(&mut reg);
        reg
    }
}
