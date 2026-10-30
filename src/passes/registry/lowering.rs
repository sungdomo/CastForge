//! Lowering, distributed, memory, surgery, verification, and preset registrations for
//! [`PassRegistry`].

use super::*;
use indexmap::IndexMap;

pub(super) fn register_lowering_passes(reg: &mut PassRegistry) {
    reg.register(
        "specialize-shapes",
        "Unified shape specialization, static batch resizing, bucketing, and token pruning",
        |opts| {
            let mut default_dim = 1i64;
            let mut bindings = IndexMap::new();
            let batch = opts
                .get("batch")
                .or_else(|| opts.get("size"))
                .and_then(|s| s.parse::<i64>().ok());
            let buckets = opts.get("buckets").cloned();
            let token_prune_ratio = opts
                .get("token_prune_ratio")
                .or_else(|| opts.get("prune_ratio"))
                .and_then(|s| s.parse::<f32>().ok());
            for (k, v) in opts {
                if matches!(
                    k.as_str(),
                    "batch" | "size" | "buckets" | "token_prune_ratio" | "prune_ratio"
                ) {
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
            Ok(Box::new(SpecializeShapesPass {
                default_dim,
                bindings,
                batch,
                buckets,
                token_prune_ratio,
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
        "rematerialize",
        "Activation rematerialization / checkpointing for long-lived intermediates",
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
        "Inserts host-device DRAM offload markers on long-lived activations",
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
        "shard-distributed",
        "Unified distributed parallelism (TP, Ring-CP, PP, FSDP, EP, streams, & reverse VJP)",
        |opts| {
            let learning_rate = opts
                .get("lr")
                .or_else(|| opts.get("learning_rate"))
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0);
            let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                if opts.contains_key("lr") || opts.contains_key("learning_rate") {
                    "autodiff".to_string()
                } else if opts.contains_key("ring_size") {
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
                .unwrap_or(!matches!(mode.as_str(), "autodiff" | "vjp" | "sgd"));
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
                learning_rate,
            }))
        },
    );
    reg.register(
        "autodiff",
        "Reverse-mode automatic differentiation (VJP backward graph + SGD update)",
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
        "hoist-all-reduce",
        "Coalesces adjacent all_reduce ops and annotates compute-communication overlap",
        |_opts| Ok(Box::new(HoistAllReducePass)),
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
            let warp_specialized = opts
                .get("warp_specialized")
                .map(|s| s == "true" || s == "1")
                .unwrap_or(false);
            let tmem_columns = opts
                .get("tmem_columns")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(512);
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
                warp_specialized,
                tmem_columns,
            }))
        },
    );
    reg.register(
        "lower-conv-to-gemm",
        "Lowers 1x1 NHWC 2D convolutions into reshape + dot_general GEMMs",
        |_opts| Ok(Box::new(LowerConvToGemmPass)),
    );
    reg.register(
        "split-large-gemm",
        "Splits wide dot_general RHS weights along N into cache-friendly chunks",
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
        "normalize-ir",
        "Unified IR normalization, subgraph outlining/inlining/extraction, freezing & tapping",
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
            let mode = opts.get("mode").cloned().or_else(|| {
                if opts.get("inline").is_some_and(|v| v == "true" || v == "1") {
                    Some("inline".to_string())
                } else if opts.get("outline").is_some_and(|v| v == "true" || v == "1") {
                    Some("outline".to_string())
                } else if max_ops.is_some() || !keep_op.is_empty() {
                    Some("reduce".to_string())
                } else if !target.is_empty() || !stop_at.is_empty() {
                    Some("extract".to_string())
                } else {
                    None
                }
            });
            let dedup = opts
                .get("dedup")
                .map(|s| s != "false" && s != "0")
                .unwrap_or_else(|| {
                    matches!(
                        mode.as_deref(),
                        Some("outline" | "all" | "gemm" | "composites" | "dedup")
                    )
                });
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
            Ok(Box::new(NormalizeIrPass {
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
        "inline-calls",
        "Inlines func.call sites by cloning callee body operations into the caller",
        |_opts| Ok(Box::new(InlineCallsPass)),
    );
    reg.register(
        "outline-subgraphs",
        "Outlines composite/GEMM ops into reusable private functions and deduplicates",
        |opts| {
            let mode = opts
                .get("mode")
                .cloned()
                .unwrap_or_else(|| "all".to_string());
            let dedup = opts
                .get("dedup")
                .map(|s| s != "false" && s != "0")
                .unwrap_or(true);
            Ok(Box::new(OutlineSubgraphsPass {
                mode,
                dedup,
                ..Default::default()
            }))
        },
    );
    reg.register(
        "extract-subgraph",
        "Slices the backward dependency cone of target SSA values",
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
            Ok(Box::new(ExtractSubgraphPass {
                target,
                stop_at,
                ..Default::default()
            }))
        },
    );
    reg.register(
        "reduce-ir",
        "Delta-debugging IR reducer (truncates function body to max_ops or keep_op)",
        |opts| {
            let max_ops = opts
                .get("max_ops")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2);
            let keep_op = opts.get("keep_op").cloned().unwrap_or_default();
            Ok(Box::new(ReduceIrPass { max_ops, keep_op }))
        },
    );
    reg.register(
        "verify-ir",
        "Unified IR verifier, structural hasher, equivalence, sanitizer, bisection & profiling",
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
            let instrument = opts
                .get("instrument")
                .map(|s| s == "true" || s == "1")
                .unwrap_or_else(|| opts.contains_key("filter") || opts.contains_key("ops"));
            let profile = opts
                .get("profile")
                .or_else(|| opts.get("stats"))
                .map(|s| s != "false" && s != "0")
                .unwrap_or_else(|| opts.contains_key("ridge"));
            let filter = opts
                .get("filter")
                .or_else(|| opts.get("ops"))
                .cloned()
                .unwrap_or_else(|| "gemm".to_string());
            let ridge_flops_per_byte = opts
                .get("ridge")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(10.0);
            let shapes = opts
                .get("shapes")
                .map(|s| s != "false" && s != "0")
                .unwrap_or(true);
            let equiv = opts
                .get("equiv")
                .map(|s| s != "false" && s != "0")
                .unwrap_or(!sanitize && !instrument);
            let tolerance = opts
                .get("tol")
                .or_else(|| opts.get("tolerance"))
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1e-4);
            let hash = opts
                .get("hash")
                .map(|s| s != "false" && s != "0")
                .unwrap_or(!sanitize && !instrument);
            let bisect_limit = opts
                .get("bisect")
                .or_else(|| opts.get("limit"))
                .and_then(|s| s.parse::<usize>().ok());
            Ok(Box::new(VerifyIrPass {
                shapes,
                equiv,
                tolerance,
                hash,
                bisect_limit,
                sanitize,
                clamp_max,
                eps,
                profile,
                instrument,
                filter,
                ridge_flops_per_byte,
            }))
        },
    );
    reg.register(
        "sanitize-numerics",
        "Inserts FP16/BF16 finite clamps and non-zero division guards",
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
        "instrument-profiling",
        "Wraps target ops in castforge.profile_region timing markers",
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
        "O1",
        "Preset O1: Inlining, simplify-algebraic, LICM, GVN, fold-views, canonicalize, dce",
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
