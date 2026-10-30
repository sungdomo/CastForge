//! Single-optimization leaf pass registrations for [`PassRegistry`].

use super::*;
use crate::ir::DType;

pub(super) fn register_leaf_passes(reg: &mut PassRegistry) {
    reg.register(
        "topological-sort",
        "Reorders operations for SSA def-before-use and optional peak-memory minimization",
        |opts| {
            let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                if opts
                    .get("min_memory")
                    .is_some_and(|v| v == "true" || v == "1")
                {
                    "min_memory".to_string()
                } else {
                    "standard".to_string()
                }
            });
            Ok(Box::new(TopologicalSortPass { mode }))
        },
    );
    reg.register(
        "hoist-loop-invariants",
        "Hoists loop- and region-invariant pure operations out of nested regions (LICM)",
        |_opts| Ok(Box::new(HoistLoopInvariantsPass)),
    );
    reg.register(
        "hoist-constants",
        "Hoists stablehlo.constant operations to block entry and deduplicates identical constants",
        |_opts| Ok(Box::new(HoistConstantsPass)),
    );
    reg.register(
        "fold-constants",
        "Evaluates pure operations on constant inputs at compile time (including SCCP)",
        |_opts| Ok(Box::new(FoldConstantsPass)),
    );
    reg.register(
        "fuse-fma",
        "Applies e-graph distributive factoring (a*b + a*c -> a*(b+c)) and extracts castforge.fma",
        |_opts| Ok(Box::new(FuseFmaPass)),
    );
    reg.register(
        "convert-layout",
        "Converts spatial/matrix layouts (nhwc/nchw/col_major) and propagates layout across ops",
        |opts| {
            let layout = opts
                .get("layout")
                .cloned()
                .unwrap_or_else(|| "nhwc".to_string());
            let propagate = opts
                .get("propagate")
                .is_some_and(|v| v == "true" || v == "1");
            Ok(Box::new(ConvertLayoutPass { layout, propagate }))
        },
    );
    reg.register(
        "resize-batch",
        "Rewrites the leading batch dimension across activations while preserving 2D GEMM weights",
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
        "Clones the entry function into batch-specialized bucket variants (@fn_batch_B)",
        |opts| {
            let buckets = opts
                .get("buckets")
                .cloned()
                .unwrap_or_else(|| "1;4;8".to_string());
            Ok(Box::new(BucketShapesPass { buckets }))
        },
    );
    reg.register(
        "prune-tokens",
        "Applies ToMe/FastV dynamic sequence token pruning and bipartite token merging",
        |opts| {
            let ratio = opts
                .get("ratio")
                .or_else(|| opts.get("token_prune_ratio"))
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.5);
            Ok(Box::new(PruneTokensPass { ratio }))
        },
    );
    reg.register(
        "quantize-weights",
        "Per-tensor symmetric INT8/INT4 weight quantization with uniform_dequantize",
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
        "Group-wise INT4/INT8 weight quantization with AWQ salient-channel protection",
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
            let salient_ratio = opts
                .get("salient_ratio")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.01);
            Ok(Box::new(QuantizeGroupwisePass {
                bits,
                group_size,
                awq_salient,
                salient_ratio,
            }))
        },
    );
    reg.register(
        "quantize-fp8",
        "FP8 (E4M3FN/E5M2) and OCP Microscaling MXFP8 block-scaled quantization",
        |opts| {
            let format = opts
                .get("format")
                .or_else(|| opts.get("dtype"))
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
        "quantize-fp4",
        "Quantizes weights to Blackwell NVFP4 / OCP MXFP4 E2M1 4-bit float microscaling blocks",
        |opts| {
            let scheme = opts
                .get("scheme")
                .or_else(|| opts.get("format"))
                .cloned()
                .unwrap_or_else(|| "nvfp4".to_string());
            let block_size = opts
                .get("block_size")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);
            Ok(Box::new(QuantizeFp4Pass { scheme, block_size }))
        },
    );
    reg.register(
        "calibrate-quantize",
        "Profile-guided static W8A8/W4A8 calibration and quantization",
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
        "smooth-quantize",
        "Migrates activation outlier difficulty into weights via SmoothQuant alpha scaling",
        |opts| {
            let bits = opts
                .get("bits")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(8);
            let alpha = opts
                .get("alpha")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.5);
            let num_samples = opts
                .get("samples")
                .or_else(|| opts.get("num_samples"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4);
            Ok(Box::new(SmoothQuantizePass {
                bits,
                alpha,
                num_samples,
            }))
        },
    );
    reg.register(
        "insert-qat-ste",
        "Applies fake-quantization with Straight-Through Estimator (QAT-STE) for training",
        |opts| {
            let bits = opts
                .get("bits")
                .and_then(|s| s.parse::<u8>().ok())
                .unwrap_or(8);
            Ok(Box::new(InsertQatStePass { bits }))
        },
    );
    reg.register(
        "rotate-hadamard",
        "Applies orthogonal Walsh-Hadamard rotation (QuaRot/SpinQuant) to suppress outliers",
        |opts| {
            let block_size = opts
                .get("block_size")
                .or_else(|| opts.get("size"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4);
            Ok(Box::new(RotateHadamardPass { block_size }))
        },
    );
    reg.register(
        "lower-mixed-precision",
        "Lowers heavy GEMM/Conv ops to F16/BF16 while keeping reductions and norms in F32",
        |opts| {
            let target_dtype = if opts
                .get("dtype")
                .or_else(|| opts.get("format"))
                .is_some_and(|s| s.eq_ignore_ascii_case("bf16"))
            {
                DType::BF16
            } else {
                DType::F16
            };
            Ok(Box::new(LowerMixedPrecisionPass { target_dtype }))
        },
    );
    reg.register(
        "bake-lora",
        "Folds low-rank LoRA adapters (W + alpha * A @ B) into base weight constants",
        |opts| {
            let alpha = opts
                .get("alpha")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            Ok(Box::new(BakeLoraPass { alpha }))
        },
    );
    reg.register(
        "bake-dora",
        "Bakes DoRA adapters via column-wise directional normalization and magnitude rescaling",
        |opts| {
            let alpha = opts
                .get("alpha")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            Ok(Box::new(BakeDoraPass { alpha }))
        },
    );
    reg.register(
        "merge-task-vectors",
        "Merges fine-tuned task vectors into base weights via Task Arithmetic and DARE",
        |opts| {
            let lambda = opts
                .get("lambda")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.0);
            let drop_rate = opts
                .get("drop_rate")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0);
            Ok(Box::new(MergeTaskVectorsPass { lambda, drop_rate }))
        },
    );
    reg.register(
        "fold-norm-weights",
        "Absorbs constant RMSNorm/LayerNorm scale multiply into adjacent dot_general weights",
        |_opts| Ok(Box::new(FoldNormWeightsPass)),
    );
    reg.register(
        "fold-batchnorm",
        "Absorbs BatchNorm / channel scale into preceding convolution or dot_general weights",
        |_opts| Ok(Box::new(FoldBatchNormPass)),
    );
    reg.register(
        "pack-weights",
        "Pre-packs 2D constant RHS weights from [K, N] to [N, K] for unit-stride SIMD MatMul",
        |_opts| Ok(Box::new(PackWeightsPass)),
    );
    reg.register(
        "prune-structured",
        "Structured L1 channel pruning on 2D weight matrices in dot_general (sparsity=0.5)",
        |opts| {
            let sparsity = opts
                .get("sparsity")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.5);
            Ok(Box::new(PruneStructuredPass { sparsity }))
        },
    );
    reg.register(
        "prune-2to4",
        "Prunes 2D weight matrices to NVIDIA 2:4 semi-structured sparsity (2 non-zeros per 4)",
        |_opts| Ok(Box::new(Prune2To4Pass)),
    );
    reg.register(
        "prune-activations",
        "Applies TEAL/CATS dynamic top-k magnitude activation sparsity to GEMM inputs",
        |opts| {
            let sparsity = opts
                .get("sparsity")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.5);
            Ok(Box::new(PruneActivationsPass { sparsity }))
        },
    );
    reg.register(
        "fuse-map-reduce",
        "Fuses elementwise map producers directly into stablehlo.reduce (fused_map_reduce)",
        |_opts| Ok(Box::new(FuseMapReducePass)),
    );
    reg.register(
        "fuse-megakernel",
        "Fuses sequential ops into a single persistent grid-synced megakernel launch",
        |_opts| Ok(Box::new(FuseMegakernelPass)),
    );
    reg.register(
        "fuse-activations",
        "Unified activation, gated-MLP & MoE fusion (SwiGLU, GELU, GeGLU, GatedMLP, MoE)",
        |opts| {
            let shared_expert = opts
                .get("shared_expert")
                .is_some_and(|s| s == "true" || s == "1");
            let kind = opts
                .get("kind")
                .or_else(|| opts.get("mode"))
                .cloned()
                .unwrap_or_else(|| "all".to_string());
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
        "fuse-swiglu",
        "Fuses SwiGLU (gate / (1 + exp(-gate)) * up) into stablehlo.composite (fused_swiglu)",
        |_opts| Ok(Box::new(FuseSwiGluPass)),
    );
    reg.register(
        "fuse-gelu",
        "Fuses GELU and GeGLU subgraphs into stablehlo.composite (fused_gelu / fused_geglu)",
        |_opts| Ok(Box::new(FuseGeluPass)),
    );
    reg.register(
        "fuse-gated-mlp",
        "Fuses gate/up projections + SwiGLU/GeGLU + down projection into fused_gated_mlp",
        |_opts| Ok(Box::new(FuseGatedMlpPass)),
    );
    reg.register(
        "fuse-moe",
        "Fuses Top-K router + expert gated MLPs into castforge.fused_moe",
        |opts| {
            let shared_expert = opts
                .get("shared_expert")
                .is_some_and(|s| s == "true" || s == "1");
            let top_k = opts
                .get("top_k")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2);
            let capacity_factor = opts
                .get("capacity_factor")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(1.25);
            Ok(Box::new(FuseMoePass {
                shared_expert,
                top_k,
                capacity_factor,
            }))
        },
    );
    reg.register(
        "fuse-horizontal-dot",
        "Horizontally fuses parallel dot_general ops sharing an LHS input (Q/K/V projections)",
        |_opts| Ok(Box::new(FuseHorizontalDotPass)),
    );
    reg.register(
        "fuse-dequant-matmul",
        "Fuses uniform_dequantize + dot_general into on-the-fly quantized GEMM (fused_qmatmul)",
        |_opts| Ok(Box::new(FuseDequantMatmulPass)),
    );
    reg.register(
        "fuse-matmul-epilogue",
        "Fuses dot_general + bias add (+ relu/tanh/silu/scale epilogue) into composite",
        |_opts| Ok(Box::new(FuseMatmulEpiloguePass)),
    );
    reg.register(
        "fuse-residual-norm",
        "Fuses add(x, residual) + RMSNorm into stablehlo.composite (fused_residual_rmsnorm)",
        |_opts| Ok(Box::new(FuseResidualNormPass)),
    );
    reg.register(
        "fuse-rmsnorm",
        "Fuses x * rsqrt(mean(x^2) + eps) * weight into stablehlo.composite (fused_rmsnorm)",
        |_opts| Ok(Box::new(FuseRmsNormPass)),
    );
    reg.register(
        "fuse-layernorm",
        "Fuses (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta into fused_layernorm",
        |_opts| Ok(Box::new(FuseLayerNormPass)),
    );
    reg.register(
        "fuse-softmax",
        "Unified Softmax, LogSoftmax, and chunked Linear Cross-Entropy fusion",
        |opts| {
            let kind = opts
                .get("kind")
                .or_else(|| opts.get("mode"))
                .cloned()
                .unwrap_or_else(|| "all".to_string());
            let chunk_size = opts
                .get("chunk_size")
                .or_else(|| opts.get("chunk"))
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(1024);
            Ok(Box::new(FuseSoftmaxPass { kind, chunk_size }))
        },
    );
    reg.register(
        "fuse-linear-cross-entropy",
        "Fuses classifier GEMM + online log-sum-exp Cross-Entropy to avoid materializing logits",
        |opts| {
            let chunk_size = opts
                .get("chunk_size")
                .or_else(|| opts.get("chunk"))
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(1024);
            Ok(Box::new(FuseLinearCrossEntropyPass { chunk_size }))
        },
    );
    reg.register(
        "fuse-rope",
        "Fuses decomposed Rotary Position Embedding (RoPE) subgraphs into stablehlo.composite",
        |_opts| Ok(Box::new(FuseRopePass)),
    );
    reg.register(
        "fuse-mla",
        "Fuses Multi-Head Latent Attention (MLA) by absorbing KV up-projections into Q/O weights",
        |opts| {
            let latent_rank = opts
                .get("latent_rank")
                .or_else(|| opts.get("kv_lora_rank"))
                .or_else(|| opts.get("rank"))
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(512);
            Ok(Box::new(FuseMlaPass { latent_rank }))
        },
    );
    reg.register(
        "inject-kv-cache",
        "Injects autoregressive KV-cache state updates (castforge.kv_cache_update)",
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
        "lower-flash-decoding",
        "Partitions KV sequence across parallel splits with LSE reduction for LLM decoding",
        |opts| {
            let split_kv = opts
                .get("split_kv")
                .or_else(|| opts.get("splits"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4);
            Ok(Box::new(LowerFlashDecodingPass { split_kv }))
        },
    );
    reg.register(
        "lower-tree-attention",
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
            Ok(Box::new(LowerTreeAttentionPass {
                draft_tokens,
                tree_depth,
            }))
        },
    );
    reg.register(
        "bufferize-inplace",
        "Annotates elementwise ops whose input buffer is on its last use for in-place reuse",
        |_opts| Ok(Box::new(BufferizeInplacePass)),
    );
    reg.register(
        "eliminate-concat-copies",
        "Eliminates intermediate concatenate copies via zero-copy destination subviews",
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
        "shard-pipeline-parallel",
        "Partitions ops across pipeline stages with 1F1B / Zero-Bubble microbatch schedules",
        |opts| {
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
            Ok(Box::new(ShardPipelineParallelPass {
                stages,
                microbatches,
                schedule,
            }))
        },
    );
    reg.register(
        "shard-fsdp",
        "Shards GEMM weights with ZeRO-3/FSDP all_gather prefetch and reduce_scatter grad sync",
        |opts| {
            let num_devices = opts
                .get("devices")
                .or_else(|| opts.get("num_devices"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2);
            Ok(Box::new(ShardFsdpPass { num_devices }))
        },
    );
    reg.register(
        "shard-experts",
        "Shards MoE/MLP experts across devices with all_to_all token dispatch and combine",
        |opts| {
            let num_devices = opts
                .get("devices")
                .or_else(|| opts.get("num_devices"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(2);
            Ok(Box::new(ShardExpertsPass { num_devices }))
        },
    );
    reg.register(
        "schedule-streams",
        "Assigns independent branches to concurrent execution streams with stream_wait barriers",
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
        "tile-loops",
        "Translates GEMM ops by attaching L1/L2 cache and register tiling schedule attributes",
        |opts| {
            let tile_size = opts
                .get("size")
                .or_else(|| opts.get("tile_size"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16);
            let tile_m = opts
                .get("tile_m")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(tile_size);
            let tile_n = opts
                .get("tile_n")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(tile_size);
            let tile_k = opts
                .get("tile_k")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(tile_size);
            let l2_tile = opts
                .get("l2_tile")
                .or_else(|| opts.get("l2_size"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let reg_m = opts
                .get("reg_m")
                .or_else(|| opts.get("reg_tile"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let reg_n = opts
                .get("reg_n")
                .or_else(|| opts.get("reg_tile"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
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
        "Translates compute ops by attaching SIMD vectorization, ISA, and parallel attrs",
        |opts| {
            let isa = opts
                .get("isa")
                .cloned()
                .unwrap_or_else(|| "avx2".to_string());
            let width = opts
                .get("width")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8);
            let parallel = opts
                .get("parallel")
                .map(|s| s == "true" || s == "1")
                .unwrap_or(true);
            Ok(Box::new(VectorizeLoopsPass {
                width,
                isa,
                parallel,
            }))
        },
    );
    reg.register(
        "lower-tensor-cores",
        "Lowers GEMMs to GPU Tensor Core / Matrix Core instructions (WMMA, WGMMA, tcgen05, MFMA)",
        |opts| {
            let arch = opts
                .get("arch")
                .or_else(|| opts.get("tensor_cores"))
                .cloned()
                .unwrap_or_else(|| "hopper_wgmma".to_string());
            let warp_specialized = opts
                .get("warp_specialized")
                .map(|s| s == "true" || s == "1")
                .unwrap_or(false);
            let tmem_columns = opts
                .get("tmem_columns")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(512);
            Ok(Box::new(LowerTensorCoresPass {
                arch,
                mma_m: 16,
                mma_n: 16,
                mma_k: 16,
                warp_specialized,
                tmem_columns,
            }))
        },
    );
    reg.register(
        "freeze-inputs",
        "Freezes function input arguments into inline constant tensors",
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
        "Hoists inline weight constants into explicit function arguments",
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
        "Exposes intermediate SSA values as additional function return outputs",
        |opts| {
            let values = opts
                .get("values")
                .or_else(|| opts.get("tap"))
                .or_else(|| opts.get("targets"))
                .cloned()
                .unwrap_or_default();
            Ok(Box::new(TapIntermediatesPass { values }))
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
        "strip-debuginfo",
        "Strips source location metadata (loc(...)) from all operations in the module",
        |_opts| Ok(Box::new(StripDebugInfoPass)),
    );
    reg.register(
        "strip-attributes",
        "Strips transient castforge.* and schedule attributes for pure portable StableHLO export",
        |_opts| Ok(Box::new(StripAttributesPass)),
    );
    reg.register(
        "outline-kernels",
        "Outlines GEMM and composite subgraphs into private func.func kernels + func.call sites",
        |opts| {
            let mode = opts
                .get("mode")
                .cloned()
                .unwrap_or_else(|| "all".to_string());
            Ok(Box::new(OutlineKernelsPass { mode }))
        },
    );
    reg.register(
        "dedup-functions",
        "Deduplicates structurally identical private functions and rewrites call sites",
        |_opts| Ok(Box::new(DedupFunctionsPass)),
    );
    reg.register(
        "compose-functions",
        "Chains two functions (@first -> @second) into a composed entry pipeline function",
        |opts| {
            let first = opts.get("first").cloned().unwrap_or_default();
            let second = opts.get("second").cloned().unwrap_or_default();
            let entry_name = opts
                .get("entry")
                .or_else(|| opts.get("name"))
                .cloned()
                .unwrap_or_else(|| "composed_main".to_string());
            Ok(Box::new(ComposeFunctionsPass {
                first,
                second,
                entry_name,
            }))
        },
    );
    reg.register(
        "unroll-calls",
        "Unrolls repeated func.call chains or self-recurrent blocks by a static repeat factor",
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
        "verify-shapes",
        "Verifies StableHLO static/dynamic tensor shape and element-type invariants",
        |_opts| Ok(Box::new(VerifyShapesPass)),
    );
    reg.register(
        "hash-ir",
        "Computes a 64-bit SSA-invariant structural hash and stamps it onto the entry function",
        |_opts| Ok(Box::new(HashIrPass)),
    );
    reg.register(
        "verify-equiv",
        "Verifies structural SSA integrity and function I/O signature consistency in a pipeline",
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
        "bisect-passes",
        "Verifies IR invariants and stamps a numbered bisection checkpoint on the entry function",
        |opts| {
            let limit = opts
                .get("limit")
                .or_else(|| opts.get("bisect"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);
            Ok(Box::new(BisectPassesPass { limit }))
        },
    );
    reg.register(
        "print-op-stats",
        "Computes operation counts, parameter sizes, FLOPs, and Roofline arithmetic intensity",
        |opts| {
            let ridge_flops_per_byte = opts
                .get("ridge")
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(10.0);
            Ok(Box::new(PrintOpStatsPass {
                ridge_flops_per_byte,
            }))
        },
    );
}
