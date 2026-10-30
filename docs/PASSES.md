# CastForge — Complete Reference of All 115 Hierarchical Graph IR Passes (+ LoopIR Pass)

CastForge provides **115 built-in modular A-to-B IR passes & presets** (`38 top-level/composite
passes & presets + 77 single-optimization leaf passes`) across `src/passes/` and
`src/ir/verify.rs` (plus the unified **`OptimizeLoopIrPass`** (`optimize-loop-ir`) in
`src/codegen/`).

- **Leaf passes** perform a single optimization (or tightly related single-traversal family) in a
  single path and may invoke tightly coupled cleanup passes such as
  `DeadCodeEliminationPass::ssa_only()`.
- **Higher-level composite passes** (`Pass::is_composite() == true`, e.g., `canonicalize`,
  `specialize-shapes`, `quantize`, `merge-weights`, `fold-weights`, `prune-weights`,
  `fuse-elementwise`, `fuse-gemm`, `fuse-norm`, `fuse-attention`, `optimize-memory`,
  `shard-distributed`, `schedule-loops`, `normalize-ir`, `outline-subgraphs`, `verify-ir`, `O1`,
  `O2`, `O3`) contain an ordered sequence of lower-level passes (`Pass::child_passes()`) aggregated
  via `PassResult::aggregate` / `PassResult::run_sequence`.

---

## Summary by Category

| Category | Top | Leaf | Section |
| :--- | :---: | :---: | :--- |
| **1. General MLIR & Algebraic Passes (+ `O1`/`O2`/`O3`)** | 7 | +5 | Section 1 |
| **2. Views, Shape Specialization & Layout** | 2 | +4 | Section 2 |
| **3. Weight Surgery, Pruning & Quantization** | 5 | +18 | Section 3 |
| **4. Transformer, GEMM & Operator Fusions** | 7 | +21 | Section 4 |
| **5. Memory Planning, Remat & Offload** | 3 | +2 | Section 5 |
| **6. Distributed Parallelism, Streams & Autodiff** | 3 | +6 | Section 6 |
| **7. Polyhedral Loop Scheduling & Tensor Cores** | 3 | +6 | Section 7 |
| **8. IR Surgery, Verification & Tooling** | 8 | +15 | Section 8 |
| **9. Polyhedral / Imperative LoopIR** | +1 | — | Section 9 |

*(Total: **115 registered Graph IR passes/presets** in `PassRegistry` + **`OptimizeLoopIrPass`**.)*

---

## 1. General MLIR & Algebraic Optimization Passes (`src/passes/`, 7 Entries)

### 1.1 `dce` (`DeadCodeEliminationPass` — `dce/mod.rs`)
- **Options:** `symbols=true|false` (default: `true`)
- **Explanation:** Single-traversal $O(N)$ liveness-driven Dead Code Elimination that sweeps
  operations in reverse topological order (marking operands of live ops on the fly without a
  separate forward use-counting pass) and eliminates unreachable private `func.func` symbols when
  `symbols=true`.

### 1.2 `cse` (`CsePass` — `cse/mod.rs`)
- **Options:** `gvn=true|false` (default: `true`), `licm=true|false` (default: `true`),
  `hoist_constants=true|false` (default: `false`)
- **Explanation:** Unified Common Subexpression Elimination (`CSE`), dominator-aware cross-region
  Global Value Numbering (`GVN`), Loop/Region-Invariant Code Motion (`LICM`), and optional
  block-entry constant hoisting/deduplication.

### 1.3 `simplify-algebraic` (`SimplifyAlgebraicPass` — `simplify_algebraic/mod.rs`)
- **Options:** `fold_constants=true|false` (default: `true`), `conditional=true|false`
  (default: `true`), `egraph=true|false` (default: `true`), `fuse_fma=true|false` (default: `true`),
  `mode=constants|all`
- **Explanation:** Unified algebraic simplifier and constant evaluator combining compile-time
  constant folding & Sparse Conditional Constant Propagation (`SCCP` on `compare`/`select`),
  identity rewrites (`x + 0 -> x`, `x - x -> 0`, `x * 1 -> x`, `x * 0 -> 0`, `max(x, x) -> x`),
  constant reassociation (`(x + c1) + c2 -> x + (c1 + c2)`), distributive factoring
  (`a * b + a * c -> a * (b + c)`), and FMA extraction (`a * b + c -> castforge.fma(a, b, c)`).

### 1.4 `canonicalize` (`CanonicalizePass` — `canonicalize/mod.rs`)
- **Options:** `mode=all|toposort|min_memory` (default: `all`), `toposort=true|false`
- **Explanation:** Fixed-point canonicalizer combining `fold-views`, `simplify-algebraic`, `cse`,
  `dce`, and topological / Sethi-Ullman minimum-live-tensor priority scheduling
  (`mode=min_memory`, `castforge.min_memory_scheduled = true`).

### 1.5–1.7 Optimization Pipeline Presets (`O1`, `O2`, `O3` — `canonicalize/presets.rs`)
- **`O1` (`OptO1Pass`):** Inlining, `simplify-algebraic`, `cse(licm=true)`, `fold-views`,
  `canonicalize(toposort=true)`, and `dce`.
- **`O2` (`OptO2Pass`):** `O1` + weight folding (`merge-weights(method=lora)`, `fold-weights`) +
  unified Transformer/GEMM fusions (`fuse-norm`, `fuse-attention`, `fuse-elementwise`, `fuse-gemm`).
- **`O3` (`OptO3Pass`):** `specialize-shapes` + `O2` + `fuse-elementwise(mode=moe)` +
  `simplify-algebraic` + `shard-distributed` + `optimize-memory` + `fold-weights(mode=pack)`.

---

## 2. Views, Shape Specialization & Layout Passes (`src/passes/`, 2 Passes)

### 2.1 `fold-views` (`FoldViewsPass` — `fold_views/mod.rs`)
- **Options:** `layout=nhwc|nchw|col_major`, `propagate=true|false`
- **Explanation:** Unified folding of `transpose` (sinking & permutation composition), `reshape`,
  `broadcast_in_dim`, `slice`/`concatenate`, round-trip `convert` casts, and spatial/matrix layout
  conversion & global layout propagation (`layout=nhwc,propagate=true`).

### 2.2 `specialize-shapes` (`SpecializeShapesPass` — `specialize_shapes/mod.rs`)
- **Options:** `B=val,S=val,default=val`, `batch=N`, `buckets=1;4;8`, `token_prune_ratio=0.5`
- **Explanation:** Unified shape specialization engine supporting symbolic dimension binding
  (`B=4`), static batch resizing across activations (`batch=8`), multi-bucket shape-specialized
  function dispatch (`buckets=1;4;8`), and **ToMe / FastV / StreamingLLM dynamic sequence token
  pruning & bipartite merging** (`token_prune_ratio=0.5`, `castforge.token_merging =
  "tome_bipartite"`).

---

## 3. Weight Surgery, Pruning & Quantization Passes (`src/passes/`, 5 Passes)

### 3.1 `quantize` (`QuantizePass` — `quantize/mod.rs`)
- **Options:** `scheme=weight|groupwise|awq|fp8|nvfp4|f16|bf16|w8a8|smoothquant|hadamard|qat_ste`
  (`dtype=f16|bf16`, `bits=8|4`, `group_size=32`, `salient_ratio=0.01`, `format=e4m3|e5m2`,
  `mxfp8=true`, `block_size=32`, `samples=4`, `alpha=0.5`, `hadamard=true|false`)
- **Explanation:** Unified precision lowering & quantization pass consolidating per-tensor `i8`/`i4`
  PTQ (`scheme=weight`), AWQ/GPTQ group-wise weight quantization (`scheme=groupwise|awq`) with
  **AWQ salient-channel protection** (`salient_ratio=0.01`), Hopper/Blackwell `FP8` & `MXFP8`
  (`scheme=fp8`), **Blackwell `NVFP4` / `MXFP4` E2M1 4-bit float microscaling**
  (`scheme=nvfp4|mxfp4`), Automatic Mixed Precision (`dtype=f16|bf16`), calibrated `W8A8`,
  **SmoothQuant** (`scheme=smoothquant`), **QuaRot / SpinQuant Hadamard rotation**
  (`scheme=hadamard` or `hadamard=true`), and **QAT Straight-Through Estimator**
  (`scheme=qat_ste`).

### 3.2 `merge-weights` (`MergeWeightsPass` — `merge_weights/mod.rs`)
- **Options:** `method=linear|slerp|ties|dare|lora|dora`, `t=0.5`, `alpha=1.0`, `lambda=1.0`,
  `drop_rate=0.0`
- **Explanation:** Unified model merging and adapter baking pass supporting Linear, SLERP, TIES,
  DARE task-vector arithmetic (`method=dare`), compile-time LoRA adapter baking (`method=lora`), and
  **DoRA (Weight-Decomposed Low-Rank Adaptation)** directional column normalization & magnitude
  baking (`method=dora`, `castforge.dora_baked = true`).

### 3.3 `fold-weights` (`FoldWeightsPass` — `fold_weights/mod.rs`)
- **Options:** `mode=all|batchnorm|norm|pack` (default: `batchnorm`)
- **Explanation:** Single-traversal compile-time weight folding & SIMD pre-packing pass absorbing
  pre-GEMM RMSNorm/LayerNorm scales (`mode=norm`), post-Conv/GEMM BatchNorm scales
  (`mode=batchnorm`), and pre-transposing 2D constant RHS weights `[K, N] -> [N, K]` (`mode=pack` or
  `mode=all`).

### 3.4 `prune-weights` (`PruneWeightsPass` — `fold_weights/prune.rs`)
- **Options:** `mode=structured|2to4|activation_sparse` (default: `structured`), `sparsity=0.5`
- **Explanation:** Single-traversal compile-time L1 structured column pruning (`mode=structured`),
  NVIDIA Ampere/Hopper 2:4 semi-structured sparsity (`mode=2to4`), and **TEAL / CATS dynamic
  top-$k$ activation sparsity** (`mode=activation_sparse`).

### 3.5 `externalize-weights` (`ExternalizeWeightsPass` — `fold_weights/prune.rs`)
- **Options:** `min_elements=2`, `inline=true|false`
- **Explanation:** Single-traversal weight storage pass externalizing inline `dense<...>` constants
  into `dense_resource<key>` handles in `TensorStore` (or inlining back with `inline=true`).

---

## 4. Transformer, GEMM & Operator Fusion Passes (`src/passes/`, 7 Passes)

### 4.1 `fuse-elementwise` (`FuseElementwisePass` — `fuse_elementwise/mod.rs`)
- **Options:** `mode=all|composites|map_reduce|activations|swiglu|gelu|gated_mlp|moe|megakernel`
  (default: `composites`), `shared_expert=true|false`, `top_k=2`, `capacity_factor=1.25`
- **Explanation:** Single-traversal elementwise, activation, Gated-MLP, MoE, and map-reduce fusion
  engine supporting `add+max` (`mode=composites`), horizontal gate/up `dot_general` + `SwiGLU` /
  `GELU` / `GeGLU` / `FusedGatedMLP` (`mode=activations|swiglu|gelu|gated_mlp`), routed/shared
  **Mixture of Experts (`mode=moe`)**, and map-into-reduce fusion (`mode=map_reduce`).

### 4.2 `decompose-composites` (`DecomposeCompositesPass` — `fuse_elementwise/decompose.rs`)
- **Explanation:** Single-traversal composite-to-primitive lowering pass expanding all
  `castforge.*` `stablehlo.composite` operations back into canonical primitive StableHLO operations.

### 4.3 `fuse-loops` (`FuseLoopsPass` — `fuse_elementwise/loops.rs`)
- **Explanation:** Single-traversal loop-chain fusion pass collapsing linear single-use elementwise
  producer-consumer chains into register-pipelined `castforge.fused_elementwise_chain` composites.

### 4.4 `split-loops` (`SplitLoopsPass` — `fuse_elementwise/loops.rs`)
- **Options:** `max_steps=3`
- **Explanation:** Single-traversal register-pressure loop splitting pass splitting oversized
  `castforge.fused_elementwise_chain` composites into chunks of at most `max_steps` operations.

### 4.5 `fuse-gemm` (`FuseGemmPass` — `fuse_gemm/mod.rs`)
- **Options:** `kind=all|horizontal|dequant|epilogue` (default: `all`)
- **Explanation:** Single-traversal GEMM fusion combining horizontal parallel QKV/MLP dot fusion
  with **compile-time constant weight concatenation**
  (`castforge.horizontal_weights_folded = true`),
  on-the-fly dequant+GEMM (`fused_qmatmul`), and chained GEMM + bias + activation + **residual
  add** epilogue fusion (`epilogue = "bias_relu_residual"`, `castforge.has_residual = true`).

### 4.6 `fuse-norm` (`FuseNormPass` — `fuse_norm/mod.rs`)
- **Options:** `kind=all|rmsnorm|residual_rmsnorm|layernorm|softmax|log_softmax|cross_entropy`
  (default: `all`), `chunk_size=1024`
- **Explanation:** Unified normalization & softmax fusion combining `Residual+RMSNorm`, `RMSNorm`,
  `LayerNorm`, `Softmax`, `LogSoftmax`, and **Liger-Kernel / Flash-Cross-Entropy**
  (`kind=cross_entropy`, `castforge.fused_linear_cross_entropy = true`).

### 4.7 `fuse-attention` (`FuseAttentionPass` — `fuse_attention/mod.rs`)
- **Options:** `kind=all|sdpa|rope|mla|ssm|mamba2|kv_cache|paged|flash_decoding|tree|sliding_window`
  (`kv_lora_rank=512`, `state_dim=128`, `chunk_size=64`, `max_seq_len=2048`, `page_block_size=16`,
  `max_blocks=64`, `split_kv=4`, `draft_tokens=4`, `tree_depth=2`, `window_size=4096`,
  `sink_tokens=4`, `share_every=2`, `rope_scaling=yarn|ntk|llama3`, `rope_factor=8.0`,
  `logit_softcap=50.0`, `qk_norm=true`, `kv_dtype=fp8|int4`)
- **Explanation:** Single-traversal `RoPE` + `Softmax` + `SDPA`/`GQA` fusion pass (with **YaRN /
  NTK-Aware / Llama-3 frequency scaling**, **Gemma-2/Grok logit soft-capping & QK-Norm**), plus
  **DeepSeek MLA** (`kind=mla`), **Mamba-2 SSD / RWKV-6 / Gated-DeltaNet Linear Attention**
  (`kind=ssm|mamba2`), stateful KV-cache injection with **FP8 / INT4 KV-cache quantization**
  (`kv_dtype=fp8|int4`), vLLM PagedAttention (`kind=paged`), Split-KV FlashDecoding
  (`kind=flash_decoding`), speculative tree attention (`kind=tree`), **Sliding-Window + Attention
  Sinks** (`kind=sliding_window`), and **YOCO / CLA Cross-Layer KV-Cache Sharing**
  (`kind=kv_sharing`).

---

## 5. Memory Planning, Rematerialization & Offloading (`src/passes/`, 3 Passes)

1. **`optimize-memory` (`OptimizeMemoryPass` — `optimize_memory/mod.rs`):** Single-traversal
   in-place arena buffer donation (`inplace`) and destination-passing zero-copy `concatenate`
   elimination (`zero_copy`), plus **Selective Activation Checkpointing** (`mode=selective_remat`).
2. **`rematerialize` (`RematerializePass` — `optimize_memory/remat.rs`):** Single-traversal
   activation rematerialization / checkpointing (`min_distance=3`) recomputing cheap elementwise
   producers before late consumers.
3. **`offload-activations` (`OffloadActivationsPass` — `optimize_memory/remat.rs`):**
   Single-traversal pinned host-memory D2H/H2D offloading (`min_span=2`) inserting
   `castforge.host_offload` markers.

---

## 6. Distributed Parallelism, Streams & Autodiff (`src/passes/`, 3 Passes)

1. **`shard-distributed` (`ShardDistributedPass` — `shard_distributed/mod.rs`):** Single-traversal
   distributed parallelism & stream pass (`mode=tp|ring_cp|pp|fsdp|zero3|ep|streams`, `devices=2`,
   `ring_size=4`, `stages=2`, `microbatches=4`, `schedule=1f1b`, `streams=2`) consolidating 1D
   Megatron Tensor Parallelism (`tp`), RingAttention Context Parallelism (`ring_cp`), **Expert
   Parallelism (`mode=ep`)**, **ZeRO-3 / FSDP parameter sharding** (`mode=fsdp|zero3`), 1F1B/
   Zero-Bubble Pipeline Parallelism (`pp`), and multi-stream async execution (`streams`).
2. **`autodiff` (`AutodiffPass` — `shard_distributed/autodiff.rs`):** Single-traversal
   reverse-mode VJP automatic differentiation (`dot_general`, `add`, `subtract`, `multiply`,
   `divide`, `tanh`, `exponential`, `log`, `sqrt`, `rsqrt`, `sine`, `cosine`, `maximum`, `minimum`,
   `negate`, `reshape`, `transpose`, `broadcast_in_dim`, `reduce`) & in-place SGD or **AdamW**
   weight update synthesis (`optimizer=sgd|adamw`, `lr=0.01`, `beta1=0.9`, `beta2=0.999`,
   `weight_decay=0.01`, `eps=1e-8`).
3. **`coalesce-all-reduce` (`CoalesceAllReducePass` — `shard_distributed/mod.rs`):**
   Single-traversal `all_reduce` coalescer and compute-communication overlap scheduler.

---

## 7. Polyhedral Loop Scheduling & Tensor Cores (`src/passes/`, 3 Passes)

1. **`schedule-loops` (`ScheduleLoopsPass` — `schedule_loops/mod.rs`):** Single-traversal loop
   schedule, tile alignment padding (`align=8`), L1/L2/register tiling (`size=16,l2_size=64,
   reg_tile=4`), loop permutation/skewing (`order=mkn,collapse=true,skew=true`), unrolling/
   pipelining/Split-K/double-buffering (`factor=4,prefetch=2,split_k=2,double_buffer=true`),
   multi-arch SIMD vectorization (`isa=avx2|avx512|neon|sve|rvv|wasm`), and GPU Tensor/Matrix Core
   lowering (`tensor_cores=wgmma|tcgen05|blackwell|wmma|mfma|simdgroup,mma_tile=16x16x16,
   warp_specialized=true,tmem_columns=512`).
2. **`lower-conv-to-gemm` (`LowerConvToGemmPass` — `schedule_loops/lowering.rs`):**
   Single-traversal lowering of `1x1` NHWC 2D `stablehlo.convolution` operations into `reshape +
   dot_general + reshape`.
3. **`split-large-gemm` (`SplitLargeGemmPass` — `schedule_loops/lowering.rs`):** Single-traversal
   wide GEMM column splitter (`max_n=256`) partitioning wide `dot_general` RHS weights along $N$.

---

## 8. IR Surgery, Verification, Profiling & Tooling (`src/passes/` & `src/ir/`, 8 Passes)

1. **`normalize-ir` (`NormalizeIrPass` — `normalize_ir/mod.rs`):** Single-traversal surgical IR
   normalizer and editor (`rename_op=from:to`, `annotate_op=op:k=v`, `freeze=true,args=arg0:1.0`,
   `unfreeze=true,min_elements=2`, `tap=%0;%1`, `ssa=true,prefix=v`, `debuginfo=true,
   attributes=true`, `first=...`, `second=...`, `repeat=N`).
2. **`inline-calls` (`InlineCallsPass` — `normalize_ir/inliner.rs`):** Single-traversal function
   inliner replacing `func.call` sites with cloned callee body operations.
3. **`outline-subgraphs` (`OutlineSubgraphsPass` — `normalize_ir/outline.rs`):**
   Single-traversal subgraph outliner and structural deduplicator (`mode=all|gemm|composites`).
4. **`extract-subgraph` (`ExtractSubgraphPass` — `normalize_ir/inliner.rs`):** Single-traversal
   backward dependency cone slicer (`target=%v,stop_at=%in`).
5. **`reduce-ir` (`ReduceIrPass` — `normalize_ir/inliner.rs`):** Single-traversal delta-debugging
   IR reducer (`max_ops=N` or `keep_op=...`).
6. **`verify-ir` (`VerifyIrPass` — `src/ir/verify.rs`):** Read-only IR verifier,
   structural FNV-1a hasher, numerical equivalence checker, pass bisector, and Roofline profiler
   (`shapes=true,equiv=true,tol=1e-4,hash=true,bisect=N,profile=true,ridge=10.0`).
7. **`sanitize-numerics` (`SanitizeNumericsPass` — `src/ir/verify.rs`):** Single-traversal
   numerical sanitizer inserting `castforge.nan_guard` finite clamps and division-by-zero guards
   (`clamp_max=65504.0,eps=1e-6`).
8. **`instrument-profiling` (`InstrumentProfilingPass` — `src/ir/verify.rs`):**
   Single-traversal profiling instrumenter wrapping target ops in `castforge.profile_region`
   (`filter=gemm|all`).

---

## 9. Polyhedral / Imperative LoopIR Pass (`src/codegen/`, 1 Unified Pass)

- **`optimize-loop-ir` (`OptimizeLoopIrPass`):** Unified imperative `LoopModule` transformation pass
  orchestrating `LoopFusionPass`, `LoopFissionPass`, `LoopInterchangePass`,
  `LoopCollapseParallelPass`, `LoopUnrollAndJamPass`, `LoopUnrollPass`, and `LoopVectorizePass`.

---

## 10. Complete Reference of All 77 Single-Optimization Leaf Passes

Every composite pass (`Pass::child_passes()`) can also be invoked at the fine-grained leaf level via
`-p "<leaf-pass>"` or inspected with `castforge list-passes --tree`:

- **`cse`:** `hoist-loop-invariants`, `hoist-constants`
- **`simplify-algebraic`:** `fold-constants`, `fuse-fma`
- **`canonicalize`:** `topological-sort` (`mode=standard|min_memory`)
- **`fold-views`:** `convert-layout` (`layout=nhwc|nchw|col_major,propagate=true`)
- **`specialize-shapes`:** `resize-batch`, `bucket-shapes`, `prune-tokens`
- **`quantize`:** `quantize-weights`, `quantize-groupwise`, `quantize-fp8`, `quantize-nvfp4`,
  `convert-mixed-precision`, `calibrate-quantization`, `quantize-dynamic`, `smoothquant`,
  `rotate-hadamard`, `insert-qat-ste`
- **`merge-weights`:** `bake-lora`, `bake-dora`, `merge-task-vectors`
- **`fold-weights`:** `fold-norm-weights`, `fold-batchnorm`, `pack-weights`
- **`prune-weights`:** `prune-structured`, `prune-2to4`, `prune-activations`
- **`fuse-elementwise`:** `fuse-add-max`, `fuse-activations`, `fuse-swiglu`, `fuse-gelu`,
  `fuse-gated-mlp`, `fuse-moe`, `fuse-map-reduce`, `fuse-megakernel`
- **`fuse-gemm`:** `fuse-horizontal-dot`, `fuse-dequant-matmul`, `fuse-matmul-epilogue`
- **`fuse-norm`:** `fuse-rmsnorm`, `fuse-residual-norm`, `fuse-layernorm`, `fuse-softmax`,
  `fuse-cross-entropy`
- **`fuse-attention`:** `fuse-rope`, `fuse-sdpa`, `fuse-mla`, `fuse-ssm`, `inject-kv-cache`,
  `rewrite-paged-attention`, `rewrite-flash-decoding`, `fuse-speculative-tree`,
  `fuse-sliding-window`, `share-kv-cache`
- **`optimize-memory`:** `bufferize-inplace`, `eliminate-concat-copies`
- **`shard-distributed`:** `shard-spmd`, `shard-ring-attention`, `shard-pipeline`, `shard-fsdp`,
  `shard-experts`, `assign-streams`
- **`schedule-loops`:** `pad-for-tiling`, `tile-loops`, `permute-loops`, `unroll-loops`,
  `vectorize-loops`, `lower-to-tensor-cores`
- **`normalize-ir`:** `rename-ops`, `annotate-ops`, `freeze-inputs`, `unfreeze-constants`,
  `tap-intermediates`, `normalize-ssa`, `strip-metadata`
- **`outline-subgraphs`:** `outline-kernels`, `dedup-functions`, `unroll-calls`
- **`verify-ir`:** `verify-shapes`, `verify-equiv`, `bisect-passes`, `hash-ir`,
  `AnalyzeCostPass` (`profile-roofline`)
