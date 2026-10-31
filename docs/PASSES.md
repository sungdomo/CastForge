# CastForge — Pass Reference (115 Graph Passes + Loop IR Optimizer)

CastForge provides **115 built-in optimization, fusion, quantization, and surgery passes** across
9 domain packages in [`src/passes/`](../src/passes) (`38` top-level/composite passes and presets +
`77` single-optimization leaf passes), plus [`OptimizeLoopIrPass`](../src/passes/loops/opt.rs)
(`optimize-loop-ir`) on lowered loop nests.

- **Composite Passes** (`Pass::is_composite() == true`): Orchestrate an ordered sequence of
  lower-level passes (`Pass::child_passes()`).
- **Leaf Passes**: Perform a single focused transformation and can be invoked directly via
  `-p "<leaf-pass>"` or inspected with `castforge list-passes --tree`.

---

## Domain Overview

| # | Package (`src/passes/`) | Representative Composite Passes | Leaf Passes |
| :-: | :--- | :--- | :---: |
| 1 | **`canonicalize/`** | `dce`, `cse`, `simplify-algebraic`, `canonicalize`, `fold-views` | 6 |
| 2 | **`shapes/`** | `specialize-shapes` | 3 |
| 3 | **`weights/`** | `merge-weights`, `fold-weights`, `prune-weights`, `externalize-weights` | 9 |
| 4 | **`quantize/`** | `quantize` | 10 |
| 5 | **`fuse/`** | `fuse-elementwise`, `fuse-gemm`, `fuse-norm`, `fuse-attention` | 26 |
| 6 | **`memory/`** | `optimize-memory`, `rematerialize`, `offload-activations` | 2 |
| 7 | **`distributed/`** | `shard-distributed`, `autodiff`, `coalesce-all-reduce` | 6 |
| 8 | **`loops/`** | `schedule-loops`, `lower-conv-to-gemm`, `split-large-gemm` | 6 |
| 9 | **`normalize/`** | `normalize-ir`, `inline-calls`, `outline-subgraphs`, `verify-ir` | 15 |
| 10 | **`registry/`** | `O1`, `O2`, `O3` (Optimization Presets) | — |

---

## 1. Canonicalization & Algebraic Simplification (`src/passes/canonicalize/`)

### Composite & Top-Level Passes
- **`dce` (`DeadCodeEliminationPass` — `dce.rs`)**
  - *Options*: `symbols=true|false` (default: `true`)
  - *Effect*: Reverse-topological $O(N)$ Dead Code Elimination of unused SSA operations and
    unreachable private `func.func` symbols.
- **`cse` (`CsePass` — `cse.rs`)**
  - *Options*: `gvn=true|false`, `licm=true|false`, `hoist_constants=true|false`
  - *Effect*: Common Subexpression Elimination, dominator-aware Global Value Numbering (`GVN`),
    Loop-Invariant Code Motion (`LICM`), and constant hoisting.
- **`simplify-algebraic` (`SimplifyAlgebraicPass` — `algebraic.rs`)**
  - *Options*: `fold_constants=true|false`, `conditional=true|false`, `egraph=true|false`,
    `fuse_fma=true|false`, `mode=constants|all`
  - *Effect*: Constant folding, Sparse Conditional Constant Propagation (`SCCP`), algebraic
    identities (`x + 0 -> x`, `x * 1 -> x`, `x - x -> 0`), reassociation, factoring, and FMA fusion.
- **`fold-views` (`FoldViewsPass` — `views.rs`)**
  - *Options*: `layout=nhwc|nchw|col_major`, `propagate=true|false`
  - *Effect*: Folds redundant `transpose`, `reshape`, `broadcast_in_dim`, `slice`, `concatenate`,
    and `convert` ops, and propagates target tensor layouts.
- **`canonicalize` (`CanonicalizePass` — `driver.rs`)**
  - *Options*: `mode=all|toposort|min_memory`, `toposort=true|false`
  - *Effect*: Fixed-point canonicalizer running `fold-views`, `simplify-algebraic`, `cse`, `dce`,
    and Sethi-Ullman minimum-live-memory topological scheduling (`mode=min_memory`).

### Leaf Passes
`hoist-loop-invariants`, `hoist-constants`, `fold-constants`, `fuse-fma`, `topological-sort`,
`convert-layout`.

---

## 2. Shape Specialization (`src/passes/shapes/`)

### Composite Pass
- **`specialize-shapes` (`SpecializeShapesPass` — `specialize.rs`)**
  - *Options*: `B=val,S=val,default=val`, `batch=N`, `buckets=1;4;8`, `token_prune_ratio=0.5`
  - *Effect*: Binds symbolic/dynamic dimensions (`?` or `B`) to static integers, resizes batch
    dimensions across activations, emits multi-bucket shape-specialized dispatchers, and performs
    ToMe/FastV bipartite sequence token merging (`token_prune_ratio`).

### Leaf Passes
`resize-batch`, `bucket-shapes`, `prune-tokens`.

---

## 3. Weight Merging, Folding & Pruning (`src/passes/weights/`)

### Composite & Top-Level Passes
- **`merge-weights` (`MergeWeightsPass` — `merge.rs`)**
  - *Options*: `method=linear|slerp|ties|dare|lora|dora`, `t=0.5`, `alpha=1.0`, `lambda=1.0`,
    `drop_rate=0.0`
  - *Effect*: Merges checkpoint weights via Linear, SLERP, TIES, or DARE task arithmetic, and bakes
    low-rank LoRA (`W + alpha * A @ B`) or DoRA adapters into base weights at compile time.
- **`fold-weights` (`FoldWeightsPass` — `fold.rs`)**
  - *Options*: `mode=all|batchnorm|norm|pack`
  - *Effect*: Absorbs RMSNorm/LayerNorm scales (`mode=norm`) and BatchNorm parameters
    (`mode=batchnorm`) into adjacent weights, and pre-packs 2D RHS weights `[K, N] -> [N, K]` for
    unit-stride SIMD dot products (`mode=pack`).
- **`prune-weights` (`PruneWeightsPass` — `prune.rs`)**
  - *Options*: `mode=structured|2to4|activation_sparse`, `sparsity=0.5`
  - *Effect*: L1 structured column pruning, NVIDIA 2:4 semi-structured sparsity (`mode=2to4`), and
    dynamic top-$k$ activation sparsity (`mode=activation_sparse`).
- **`externalize-weights` (`ExternalizeWeightsPass` — `prune.rs`)**
  - *Options*: `min_elements=2`, `inline=true|false`
  - *Effect*: Moves inline `dense<...>` constants into `TensorStore` (`dense_resource<key>`) or
    inlines external weights back into the IR (`inline=true`).

### Leaf Passes
`bake-lora`, `bake-dora`, `merge-task-vectors`, `fold-norm-weights`, `fold-batchnorm`,
`pack-weights`, `prune-structured`, `prune-2to4`, `prune-activations`.

---

## 4. Quantization & Mixed Precision (`src/passes/quantize/`)

### Composite Pass
- **`quantize` (`QuantizePass` — `schemes.rs`)**
  - *Options*: `scheme=weight|groupwise|awq|fp8|nvfp4|f16|bf16|w8a8|smoothquant|hadamard|qat_ste`,
    `bits=8|4`, `group_size=32`, `salient_ratio=0.01`, `format=e4m3|e5m2`, `mxfp8=true`,
    `alpha=0.5`, `hadamard=true|false`
  - *Effect*: Unified quantization suite supporting per-tensor `INT8`/`INT4` PTQ, group-wise
    AWQ/GPTQ (`scheme=groupwise|awq`), Hopper/Blackwell `FP8`/`MXFP8`, 4-bit microscaling
    `NVFP4`/`MXFP4`, `F16`/`BF16` mixed precision, calibrated `W8A8`, SmoothQuant migration,
    QuaRot/SpinQuant Hadamard rotations, and QAT Straight-Through Estimator nodes.

### Leaf Passes
`quantize-weights`, `quantize-groupwise`, `quantize-fp8`, `quantize-nvfp4`,
`convert-mixed-precision`, `calibrate-quantization`, `quantize-dynamic`, `smoothquant`,
`rotate-hadamard`, `insert-qat-ste`.

---

## 5. Operator, GEMM & Attention Fusion (`src/passes/fuse/`)

### Composite & Top-Level Passes
- **`fuse-elementwise` (`FuseElementwisePass` — `elementwise.rs`)**
  - *Options*: `mode=all|composites|map_reduce|activations|swiglu|gelu|gated_mlp|moe|megakernel`,
    `top_k=2`, `capacity_factor=1.25`
  - *Effect*: Fuses `add+max`, `SwiGLU`, `GELU`, `GeGLU`, horizontal `FusedGatedMLP`, routed/shared
    Mixture of Experts (`mode=moe`), elementwise-into-reduce (`mode=map_reduce`), and persistent
    megakernels.
- **`fuse-gemm` (`FuseGemmPass` — `gemm.rs`)**
  - *Options*: `kind=all|horizontal|dequant|epilogue`
  - *Effect*: Horizontally merges parallel `dot_general` ops sharing an LHS activation, fuses
    dequantization into matmul (`fused_qmatmul`), and fuses bias, activation, and residual add into
    `castforge.fused_matmul_epilogue`.
- **`fuse-norm` (`FuseNormPass` — `norm.rs`)**
  - *Options*: `kind=all|rmsnorm|residual_rmsnorm|layernorm|softmax|log_softmax|cross_entropy`
  - *Effect*: Recognizes subgraph patterns for `RMSNorm`, `Residual+RMSNorm`, `LayerNorm`,
    numerically stable `Softmax` / `LogSoftmax`, and fused linear cross-entropy.
- **`fuse-attention` (`FuseAttentionPass` — `attention.rs`)**
  - *Options*: `kind=all|sdpa|rope|mla|ssm|mamba2|kv_cache|paged|flash_decoding|tree`,
    `rope_scaling=yarn|ntk|llama3`, `kv_dtype=fp8|int4`, `page_block_size=16`, `window_size=4096`
  - *Effect*: Fuses Rotary Position Embeddings (`RoPE`), Flash/GQA Scaled Dot-Product Attention
    (`SDPA`), DeepSeek Multi-Head Latent Attention (`MLA`), Mamba-2 SSD / Linear Attention (`SSM`),
    KV-cache injection, PagedAttention, FlashDecoding, speculative tree attention, sliding-window
    attention with sink tokens, and cross-layer KV sharing.
- **`fuse-loops` & `split-loops` (`FuseLoopsPass`, `SplitLoopsPass` — `loops.rs`)**
  - *Options*: `max_steps=3` (for `split-loops`)
  - *Effect*: Collapses single-use elementwise chains into `castforge.fused_elementwise_chain` or
    splits long chains to bound register pressure.
- **`decompose-composites` (`DecomposeCompositesPass` — `decompose.rs`)**
  - *Effect*: Lowers all `castforge.*` composite operations back into standard primitive StableHLO
    operations.

### Leaf Passes
`fuse-add-max`, `fuse-activations`, `fuse-swiglu`, `fuse-gelu`, `fuse-gated-mlp`, `fuse-moe`,
`fuse-map-reduce`, `fuse-megakernel`, `fuse-horizontal-dot`, `fuse-dequant-matmul`,
`fuse-matmul-epilogue`, `fuse-rmsnorm`, `fuse-residual-norm`, `fuse-layernorm`, `fuse-softmax`,
`fuse-cross-entropy`, `fuse-rope`, `fuse-sdpa`, `fuse-mla`, `fuse-ssm`, `inject-kv-cache`,
`rewrite-paged-attention`, `rewrite-flash-decoding`, `fuse-speculative-tree`,
`fuse-sliding-window`, `share-kv-cache`.

---

## 6. Arena Memory, Rematerialization & Offloading (`src/passes/memory/`)

### Composite & Top-Level Passes
- **`optimize-memory` (`OptimizeMemoryPass` — `bufferize.rs`)**
  - *Options*: `mode=all|inplace|zero_copy|selective_remat`
  - *Effect*: Annotates last-use elementwise inputs for zero-overhead in-place buffer reuse,
    eliminates intermediate `concatenate` copies via destination passing, and marks selective
    activation checkpoints.
- **`rematerialize` (`RematerializePass` — `remat.rs`)**
  - *Options*: `min_distance=3`
  - *Effect*: Recomputes inexpensive elementwise producers near late consumers to shrink peak
    activation memory.
- **`offload-activations` (`OffloadActivationsPass` — `remat.rs`)**
  - *Options*: `min_span=2`
  - *Effect*: Inserts asynchronous host-memory offload/prefetch markers (`castforge.host_offload`)
    on long-lived activation tensors.

### Leaf Passes
`bufferize-inplace`, `eliminate-concat-copies`.

---

## 7. Distributed Sharding, Streams & Autodiff (`src/passes/distributed/`)

### Composite & Top-Level Passes
- **`shard-distributed` (`ShardDistributedPass` — `spmd.rs`)**
  - *Options*: `mode=tp|ring_cp|pp|fsdp|zero3|ep|streams`, `devices=2`, `ring_size=4`, `stages=2`,
    `microbatches=4`, `schedule=1f1b`, `streams=2`
  - *Effect*: 1D Megatron Tensor Parallelism (`tp`), RingAttention Context Parallelism (`ring_cp`),
    Expert Parallelism (`ep`), FSDP/ZeRO-3 weight sharding (`fsdp`/`zero3`), 1F1B pipeline
    parallelism (`pp`), and multi-stream overlap (`streams`).
- **`autodiff` (`AutodiffPass` — `autodiff.rs`)**
  - *Options*: `optimizer=sgd|adamw`, `lr=0.01`, `beta1=0.9`, `beta2=0.999`, `weight_decay=0.01`
  - *Effect*: Synthesizes reverse-mode VJP backward graphs and in-place SGD/AdamW optimizer steps.
- **`coalesce-all-reduce` (`CoalesceAllReducePass` — `spmd.rs`)**
  - *Effect*: Fuses adjacent `all_reduce` collectives and schedules compute-communication overlap.

### Leaf Passes
`shard-spmd`, `shard-ring-attention`, `shard-pipeline`, `shard-fsdp`, `shard-experts`,
`assign-streams`.

---

## 8. Polyhedral Loop Scheduling & Lowering (`src/passes/loops/`)

### Composite & Top-Level Passes
- **`schedule-loops` (`ScheduleLoopsPass` — `schedule.rs`)**
  - *Options*: `align=8`, `size=16`, `l2_size=64`, `reg_tile=4`, `order=mkn`, `collapse=true`,
    `skew=true`, `factor=4`, `prefetch=2`, `split_k=2`, `double_buffer=true`,
    `isa=avx2|avx512|neon|sve|rvv|wasm`, `tensor_cores=wgmma|tcgen05|wmma|mfma|simdgroup`
  - *Effect*: Tile-alignment padding, multi-level L1/L2/register tiling, loop interchange/skewing,
    unrolling, software prefetching, Split-K reduction, shared-memory double buffering, SIMD
    vectorization, and GPU Tensor Core (`WMMA`/`WGMMA`/`tcgen05`/`MFMA`) lowering.
- **`lower-conv-to-gemm` (`LowerConvToGemmPass` — `gemm_conv.rs`)**
  - *Effect*: Lowers `1x1` NHWC 2D convolutions into `reshape + dot_general + reshape`.
- **`split-large-gemm` (`SplitLargeGemmPass` — `gemm_conv.rs`)**
  - *Options*: `max_n=256`
  - *Effect*: Splits wide `dot_general` RHS matrices along $N$ into cache-friendly column panels.
- **`optimize-loop-ir` (`OptimizeLoopIrPass` — `opt.rs`)**
  - *Effect*: Runs loop fusion, fission, interchange, parallel collapse, unroll-and-jam, loop
    unrolling, and SIMD vectorization directly on lowered `LoopModule` ASTs.

### Leaf Passes
`pad-for-tiling`, `tile-loops`, `permute-loops`, `unroll-loops`, `vectorize-loops`,
`lower-to-tensor-cores`.

---

## 9. IR Normalization, Verification & Tooling (`src/passes/normalize/`)

### Composite & Top-Level Passes
- **`verify-ir` (`VerifyIrPass` — `verify.rs`)**
  - *Options*: `shapes=true`, `equiv=true`, `tol=1e-4`, `hash=true`, `bisect=N`, `profile=true`
  - *Effect*: Verifies SSA dominance and StableHLO shapes, checks numerical equivalence, computes
    deterministic FNV-1a structural hashes, bisects passes, and emits Roofline cost reports.
- **`normalize-ir` (`NormalizeIrPass` — `surgery.rs`)**
  - *Options*: `rename_op=from:to`, `annotate_op=op:k=v`, `freeze=true`, `unfreeze=true`,
    `tap=%0;%1`, `ssa=true,prefix=v`, `debuginfo=true`
  - *Effect*: Canonical SSA renumbering, op renaming/annotation, input freezing, intermediate tensor
    tapping, and metadata stripping.
- **`inline-calls` (`InlineCallsPass` — `inliner.rs`)**
  - *Effect*: Inlines `func.call` operations by cloning callee bodies into the caller.
- **`outline-subgraphs` (`OutlineSubgraphsPass` — `outline.rs`)**
  - *Options*: `mode=all|gemm|composites`
  - *Effect*: Outlines repeated subgraphs into reusable functions and deduplicates identical
    functions.
- **`extract-subgraph` (`ExtractSubgraphPass` — `inliner.rs`)**
  - *Options*: `target=%v,stop_at=%in`
  - *Effect*: Extracts the backward dependency cone of target SSA values into a standalone module.
- **`reduce-ir` (`ReduceIrPass` — `inliner.rs`)**
  - *Options*: `max_ops=N`, `keep_op=...`
  - *Effect*: Delta-debugging IR reducer that shrinks failing graphs to minimal reproducers.
- **`sanitize-numerics` (`SanitizeNumericsPass` — `verify.rs`)**
  - *Options*: `clamp_max=65504.0`, `eps=1e-6`
  - *Effect*: Inserts `castforge.nan_guard` finite clamps (`FP16`/`BF16`) and division-by-zero
    guards.
- **`instrument-profiling` (`InstrumentProfilingPass` — `verify.rs`)**
  - *Options*: `filter=gemm|all`
  - *Effect*: Wraps target operations in `castforge.profile_region` timing markers.

### Leaf Passes
`rename-ops`, `annotate-ops`, `freeze-inputs`, `unfreeze-constants`, `tap-intermediates`,
`normalize-ssa`, `strip-metadata`, `outline-kernels`, `dedup-functions`, `unroll-calls`,
`verify-shapes`, `verify-equiv`, `bisect-passes`, `hash-ir`, `profile-roofline`.

---

## 10. Optimization Presets (`src/passes/registry/presets.rs`)

- **`O1` (`OptO1Pass`)**:
  - *Pipeline*: `inline-calls`, `simplify-algebraic`, `cse`, `fold-views`, `canonicalize`, `dce`,
    `topological-sort`.
  - *Use Case*: Fast compile-time cleanup and algebraic canonicalization.
- **`O2` (`OptO2Pass`)**:
  - *Pipeline*: `O1` + `merge-weights(lora)` + `fold-weights(batchnorm)` + `fuse-norm` +
    `fuse-attention` + `fuse-activations` + `fuse-softmax` + `fuse-gemm` + `fuse-elementwise`.
  - *Use Case*: General deep learning and Transformer inference optimization.
- **`O3` (`OptO3Pass`)**:
  - *Pipeline*: `specialize-shapes` + `O2` + `fuse-activations(moe)` +
    `simplify-algebraic(egraph,fma)` + `shard-distributed(coalesce)` + `fold-weights(pack)` +
    `optimize-memory(all)` + `topological-sort`.
  - *Use Case*: Maximum performance for LLM and production deployment.
