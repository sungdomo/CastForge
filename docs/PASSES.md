# CastForge — Complete Reference of All 35 Graph IR Passes (+ Unified LoopIR Pass)

CastForge provides **35 built-in modular A-to-B IR passes & presets** (`32 passes + 3 presets
O1/O2/O3`) in `castforge-passes` (plus the unified **`OptimizeLoopIrPass`** (`optimize-loop-ir`) in
`castforge-codegen`). Every pass implements the unified `Pass` and `Translator` traits and maps
1-to-1 to a clean source file in 8 domain subdirectories under `crates/castforge-passes/src/`
(`general/`, `views/`, `weights/`, `fusion/`, `memory/`, `distributed/`, `loops/`, `tooling/`).

---

## Summary by Category

| Category | Count | Directory / Section |
| :--- | :---: | :--- |
| **1. General MLIR & Algebraic Passes (+ `O1`/`O2`/`O3`)** | 11 | `src/general/` (Section 1) |
| **2. Views, Shape Specialization & Layout** | 2 | `src/views/` (Section 2) |
| **3. Weight Surgery, Pruning & Quantization** | 5 | `src/weights/` (Section 3) |
| **4. Transformer, GEMM & Operator Fusions** | 7 | `src/fusion/` (Section 4) |
| **5. Memory Planning, Remat & Offload** | 1 | `src/memory/` (Section 5) |
| **6. Distributed Parallelism, Streams & Autodiff** | 2 | `src/distributed/` (Section 6) |
| **7. Polyhedral Loop Scheduling & Tensor Cores** | 1 | `src/loops/` (Section 7) |
| **8. IR Surgery, Verification & Tooling** | 6 | `src/tooling/` (Section 8) |
| **9. Polyhedral / Imperative LoopIR** | +1 | `castforge-codegen` (Section 9) |

*(Total: **35 registered Graph IR passes/presets** in `PassRegistry` + **`OptimizeLoopIrPass`**.)*

---

## 1. General MLIR & Algebraic Optimization Passes (`src/general/`, 11 Entries)

### 1.1 `dce` (`DeadCodeEliminationPass` — `general/dce.rs`)
- **Options:** `symbols=true|false` (default: `true`)
- **Explanation:** Single-pass $O(N)$ reference-counted Dead Code Elimination
  (`Module::compute_use_counts`) that removes unused side-effect-free operations in reverse
  topological order and eliminates unreachable private `func.func` symbols when `symbols=true`.

### 1.2 `cse` (`UnifiedCsePass` — `general/cse.rs`)
- **Options:** `gvn=true|false` (default: `true`), `hoist_constants=true|false` (default: `false`)
- **Explanation:** Unified Common Subexpression Elimination (`CSE`), dominator-aware cross-region
  Global Value Numbering (`GVN`), and optional block-entry constant hoisting/deduplication.

### 1.3 `simplify-algebraic` (`UnifiedSimplifyAlgebraicPass` — `general/simplify_algebraic.rs`)
- **Options:** `egraph=true|false` (default: `true`), `fuse_fma=true|false` (default: `true`)
- **Explanation:** Unified algebraic simplifier combining identity rewrites (`x + 0 -> x`,
  `x - x -> 0`, `x * 1 -> x`, `x * 0 -> 0`, `max(x, x) -> x`), constant reassociation
  (`(x + c1) + c2 -> x + (c1 + c2)`), distributive factoring (`a * b + a * c -> a * (b + c)`), and
  FMA extraction (`a * b + c -> castforge.fma(a, b, c)`).

### 1.4 `fold-constants` (`FoldConstantsPass` — `general/fold_constants.rs`)
- **Options:** `conditional=true|false` (default: `true`)
- **Explanation:** Evaluates unary, binary, shape (`transpose`, `reshape`, `slice`, `concatenate`,
  `broadcast_in_dim`), and reduction ops on compile-time constants, plus Sparse Conditional Constant
  Propagation (`SCCP`) on constant `compare`/`select` branches.

### 1.5 `canonicalize` (`CanonicalizePass` — `general/canonicalize.rs`)
- **Options:** None
- **Explanation:** Fixed-point canonicalizer combining `fold-views`, `simplify-algebraic`, `cse`,
  `fold-constants`, and `dce`.

### 1.6 `licm` (`LicmPass` — `general/licm.rs`)
- **Options:** None
- **Explanation:** Loop/Region-Invariant Code Motion out of nested regions.

### 1.7 `inliner` (`InlinerPass` — `general/inliner.rs`)
- **Options:** None
- **Explanation:** Inlines single-block `func.call` invocations into caller blocks.

### 1.8 `topological-sort` (`TopologicalSortPass` — `general/topological_sort.rs`)
- **Options:** `mode=standard|min_memory` (default: `standard`)
- **Explanation:** Reorders operations inside each basic block for SSA def-before-use (`standard`),
  or applies a Sethi-Ullman minimum-live-tensor priority schedule (`mode=min_memory`) that schedules
  operations freeing large intermediate buffers before operations allocating new ones
  (`castforge.min_memory_scheduled = true`).

### 1.9–1.11 Optimization Pipeline Presets (`O1`, `O2`, `O3` — `general/canonicalize.rs`)
- **`O1` (`OptO1Pass`):** Inlining, `fold-constants`, `licm`, `cse`, `fold-views`, `canonicalize`,
  `dce`, and `topological-sort`.
- **`O2` (`OptO2Pass`):** `O1` + weight folding (`merge-weights(method=lora)`, `fold-weights`) +
  unified Transformer/GEMM fusions (`fuse-norm`, `fuse-attention`, `fuse-activations`,
  `fuse-softmax`, `fuse-gemm`, `fuse-elementwise`).
- **`O3` (`OptO3Pass`):** `specialize-shapes` + `O2` + `fuse-activations(kind=moe)` +
  `simplify-algebraic` + `shard-distributed` + `optimize-memory` + `fold-weights(mode=pack)`.

---

## 2. Views, Shape Specialization & Layout Passes (`src/views/`, 2 Passes)

### 2.1 `fold-views` (`FoldViewsPass` — `views/fold_views.rs`)
- **Options:** `layout=nhwc|nchw|col_major`, `propagate=true|false`
- **Explanation:** Unified folding of `transpose` (sinking & permutation composition), `reshape`,
  `broadcast_in_dim`, `slice`/`concatenate`, round-trip `convert` casts, and spatial/matrix layout
  conversion & global layout propagation (`layout=nhwc,propagate=true`).

### 2.2 `specialize-shapes` (`UnifiedSpecializeShapesPass` — `views/specialize_shapes.rs`)
- **Options:** `B=val,S=val,default=val`, `batch=N`, `buckets=1;4;8`
- **Explanation:** Unified shape specialization engine supporting symbolic dimension binding
  (`B=4`), static batch resizing across activations (`batch=8`), and multi-bucket shape-specialized
  function dispatch (`buckets=1;4;8`).

---

## 3. Weight Surgery, Pruning & Quantization Passes (`src/weights/`, 5 Passes)

### 3.1 `quantize` (`QuantizePass` — `weights/quantize.rs`)
- **Options:** `scheme=weight|groupwise|awq|fp8|nvfp4|mxfp4|f16|bf16|w8a8|smoothquant|hadamard`,
  `dtype=f16|bf16`, `bits=8|4`, `group_size=32`, `salient_ratio=0.01`, `format=e4m3|e5m2`,
  `mxfp8=true`, `block_size=32`, `samples=4`, `alpha=0.5`, `hadamard=true|false`
- **Explanation:** Unified precision lowering & quantization pass consolidating per-tensor `i8`/`i4`
  PTQ (`scheme=weight`), AWQ/GPTQ group-wise weight quantization (`scheme=groupwise|awq`) with
  **AWQ salient-channel protection** (`salient_ratio=0.01`), Hopper/Blackwell `FP8` & `MXFP8`
  (`scheme=fp8`), **Blackwell `NVFP4` / `MXFP4` E2M1 4-bit float microscaling**
  (`scheme=nvfp4|mxfp4`), Automatic Mixed Precision (`dtype=f16|bf16`), calibrated `W8A8`,
  **SmoothQuant** (`scheme=smoothquant`), and **QuaRot / SpinQuant Hadamard rotation**
  (`scheme=hadamard` or `hadamard=true`).

### 3.2 `prune-weights` (`PruneWeightsPass` — `weights/prune_weights.rs`)
- **Options:** `mode=structured|2to4|activation_sparse` (default: `structured`), `sparsity=0.5`
- **Explanation:** Unified weight & activation sparsity pass supporting L1 structured column pruning
  (`mode=structured`), NVIDIA Ampere/Hopper 2:4 structured sparsity (`mode=2to4`), and **dynamic
  top-$k$ activation sparsity** (`mode=activation_sparse`).

### 3.3 `merge-weights` (`UnifiedMergeWeightsPass` — `weights/merge_weights.rs`)
- **Options:** `method=linear|slerp|ties|dare|lora`, `t=0.5`, `alpha=1.0`, `lambda=1.0`,
  `drop_rate=0.0`
- **Explanation:** Unified model merging and adapter baking pass supporting Linear, SLERP, TIES,
  DARE task-vector arithmetic (`method=dare`), and compile-time LoRA adapter baking (`method=lora`).

### 3.4 `fold-weights` (`FoldWeightsPass` — `weights/fold_weights.rs`)
- **Options:** `mode=batchnorm|norm|pack|all` (default: `batchnorm`)
- **Explanation:** Unified compile-time weight folding & pre-packing pass absorbing pre-GEMM
  RMSNorm/LayerNorm scales (`mode=norm`), post-Conv/GEMM BatchNorm scales (`mode=batchnorm`), and
  pre-transposing 2D constant RHS weights `[K, N] -> [N, K]` (`mode=pack`, `packed_rhs = true`) for
  unit-stride SIMD dot products.

### 3.5 `externalize-weights` (`ExternalizeWeightsPass` — `weights/externalize_weights.rs`)
- **Options:** `min_elements=2`, `inline=false`
- **Explanation:** Moves inline `dense<...>` constants to `dense_resource<key>` in `TensorStore`
  (or inlines them back when `inline=true`).

---

## 4. Transformer, GEMM & Operator Fusion Passes (`src/fusion/`, 7 Passes)

### 4.1 `fuse-elementwise` (`UnifiedFuseElementwisePass` — `fusion/fuse_elementwise.rs`)
- **Options:** `mode=composites|map_reduce|loops|megakernel|all` (default: `composites`),
  `max_steps=N`
- **Explanation:** Unified elementwise fusion engine supporting composite producer-consumer pair
  fusion (`mode=composites`), map-into-reduce fusion (`mode=map_reduce`), register-pipelined loop
  chain fusion/fission (`mode=loops`), and persistent cooperative-grid megakernel fusion
  (`mode=megakernel`).

### 4.2 `fuse-gemm` (`FuseGemmPass` — `fusion/fuse_gemm.rs`)
- **Options:** `kind=all|horizontal|dequant|epilogue` (default: `all`)
- **Explanation:** Unified GEMM fusion combining horizontal parallel QKV/MLP dot fusion,
  on-the-fly dequant+GEMM (`fused_qmatmul`), and chained GEMM + bias + activation + **residual
  add** epilogue fusion (`epilogue = "bias_relu_residual"`, `castforge.has_residual = true`).

### 4.3 `fuse-norm` (`FuseNormPass` — `fusion/fuse_norm.rs`)
- **Options:** `kind=all|rmsnorm|residual_rmsnorm|layernorm` (default: `all`)
- **Explanation:** Unified normalization fusion (`Residual+RMSNorm`, `RMSNorm`, and `LayerNorm`).

### 4.4 `fuse-activations` (`FuseActivationsPass` — `fusion/fuse_activations.rs`)
- **Options:** `kind=all|swiglu|gelu|gated_mlp|moe` (default: `all`), `shared_expert=true|false`,
  `top_k=2`, `capacity_factor=1.25`
- **Explanation:** Unified activation, gated-MLP, and routed/shared-expert **Mixture-of-Experts
  (`MoE`)** fusion (`SwiGLU`, `GELU`, `GeGLU`, `FusedGatedMLP`, and `castforge.fused_moe`).

### 4.5 `fuse-softmax` (`FuseSoftmaxPass` — `fusion/fuse_softmax.rs`)
- **Options:** `kind=softmax|log_softmax|cross_entropy|all` (default: `all`), `chunk_size=1024`
- **Explanation:** Unified Softmax, LogSoftmax, and **Liger-Kernel / Flash-Cross-Entropy**
  (`kind=cross_entropy`, `castforge.fused_linear_cross_entropy = true`) composite fusion.

### 4.6 `fuse-attention` (`UnifiedFuseAttentionPass` — `fusion/fuse_attention.rs`)
- **Options:** `kind=all|sdpa|rope|mla|kv_cache|paged|flash_decoding|tree|sliding_window|kv_sharing`
  (`kv_lora_rank=512`, `max_seq_len=2048`, `page_block_size=16`, `max_blocks=64`, `split_kv=4`,
  `draft_tokens=4`, `tree_depth=2`, `window_size=4096`, `sink_tokens=4`, `share_every=2`)
- **Explanation:** Unified attention subgraph fusion & serving pass combining `RoPE`, `SDPA`/`GQA`,
  **DeepSeek MLA** (`kind=mla`), stateful KV-cache injection (`kind=kv_cache`), vLLM PagedAttention
  (`kind=paged`), Split-KV FlashDecoding (`kind=flash_decoding`), speculative tree attention
  (`kind=tree`), **Sliding-Window + Attention Sinks** (`kind=sliding_window`), and **YOCO / CLA
  Cross-Layer KV-Cache Sharing** (`kind=kv_sharing,share_every=2`).

### 4.7 `decompose-composites` (`DecomposeCompositesPass` — `fusion/decompose_composites.rs`)
- **Options:** None
- **Explanation:** Expands `stablehlo.composite` ops back to primitive StableHLO ops.

---

## 5. Memory Planning, Rematerialization & Offloading (`src/memory/`, 1 Pass)

- **`optimize-memory` (`OptimizeMemoryPass` — `memory/optimize_memory.rs`):** Unified memory
  optimization pass (`mode=all|inplace|zero_copy|remat|offload`, `min_distance=3`, `min_span=2`)
  consolidating in-place arena buffer donation (`inplace`), destination-passing zero-copy
  `concatenate` elimination (`zero_copy`), activation rematerialization/checkpointing (`remat`), and
  pinned host-memory D2H/H2D offloading (`offload`).

---

## 6. Distributed Parallelism, Streams & Autodiff (`src/distributed/`, 2 Passes)

- **`shard-distributed` (`ShardDistributedPass` — `distributed/shard_distributed.rs`):** Unified
  distributed parallelism & stream pass (`mode=tp|ring_cp|pp|streams`, `devices=2`, `ring_size=4`,
  `coalesce=true`, `stages=2`, `microbatches=4`, `schedule=1f1b`, `streams=2`) consolidating 1D
  Megatron Tensor Parallelism (`tp`), RingAttention Context Parallelism (`ring_cp`), `all_reduce`
  coalescing, 1F1B/Zero-Bubble Pipeline Parallelism (`pp`), and multi-stream async execution.
- **`autodiff` (`AutodiffPass` — `distributed/autodiff.rs`):** Synthesizes reverse-mode VJP backward
  gradient operations and optional in-graph SGD updates (`lr=...`).

---

## 7. Polyhedral Loop Scheduling & Tensor Cores (`src/loops/`, 1 Pass)

- **`schedule-loops` (`ScheduleLoopsPass` — `loops/schedule_loops.rs`):** Unified loop schedule,
  2D `convolution` to `im2col + dot_general` lowering (`lower_conv=true`), wide GEMM column
  splitting (`max_n=256`), tile alignment padding (`align=8`), L1/L2/register tiling (`size=16,
  l2_size=64,reg_tile=4`), loop permutation/skewing (`order=mkn,collapse=true,skew=true`),
  unrolling/pipelining/Split-K/double-buffering (`factor=4,prefetch=2,split_k=2,
  double_buffer=true`), multi-arch SIMD vectorization (`isa=avx2|avx512|neon|sve|rvv|wasm`), and GPU
  Tensor/Matrix Core lowering (`tensor_cores=wgmma|wmma|mfma|simdgroup,mma_tile=16x16x16`).

---

## 8. IR Surgery, Verification, Profiling & Tooling (`src/tooling/`, 6 Passes)

1. **`ir-edit` (`IrEditPass` — `tooling/ir_edit.rs`):** Unified surgical IR editor consolidating
   op renaming (`rename_op=from:to`), attribute injection (`annotate_op=op:k=v`), input freezing/
   unfreezing (`freeze=true,args=arg0:1.0`, `unfreeze=true,min_elements=2`), and intermediate value
   tapping (`tap=%0;%1`).
2. **`extract-subgraph` (`ExtractSubgraphPass` — `tooling/extract_subgraph.rs`):** Unified backward
   dataflow cone slicer (`target=%v,stop_at=%in`) and bugpoint-style delta IR reducer
   (`max_ops=1,keep_op=...`).
3. **`outline-subgraphs` (`UnifiedOutlineSubgraphsPass` — `tooling/outline_subgraphs.rs`):** Unified
   function/subgraph surgery pass (`mode=all|gemm|composites|dedup|compose|unroll`, `dedup=true`,
   `first=...`, `second=...`, `repeat=N`) consolidating kernel outlining, LLVM `-mergefunc` function
   deduplication, function composition (`f2(f1(x))`), and bounded `func.call` unrolling.
4. **`normalize-ir` (`NormalizeIrPass` — `tooling/normalize_ir.rs`):** Unified canonical SSA renamer
   (`ssa=true,prefix=v`) and metadata/location stripper (`debuginfo=true,attributes=true`).
5. **`verify-ir` (`UnifiedVerifyIrPass` — `tooling/verify_ir.rs`):** Unified IR verification,
   numerical sanitizer & hashing pass (`shapes=true,equiv=true,tol=1e-4,hash=true,bisect=N,
   sanitize=true,clamp_max=65504.0`) consolidating shape verification, signature equivalence,
   64-bit FNV-1a structural hashing, `castforge.nan_guard` finite clamps, and pass bisection.
6. **`profile-ir` (`UnifiedProfileIrPass` — `tooling/profile_ir.rs`):** Unified Roofline
   FLOPs/arithmetic-intensity analysis (`stats=true,ridge=10.0`) and `castforge.profile_region`
   timer instrumentation (`instrument=true,filter=gemm`).

---

## 9. Polyhedral / Imperative LoopIR Pass (`castforge-codegen`, 1 Unified Pass)

- **`optimize-loop-ir` (`OptimizeLoopIrPass`):** Unified imperative `LoopModule` transformation pass
  orchestrating `LoopFusionPass`, `LoopFissionPass`, `LoopInterchangePass`,
  `LoopCollapseParallelPass`, `LoopUnrollAndJamPass`, `LoopUnrollPass`, and `LoopVectorizePass`.
