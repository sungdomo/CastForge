# CastForge — Complete Reference of All 25 Graph IR Passes (+ Unified LoopIR Pass)

CastForge provides **25 built-in modular A-to-B IR passes & presets** (`22 passes + 3 presets
O1/O2/O3`) in `castforge-passes` (plus the unified **`OptimizeLoopIrPass`** (`optimize-loop-ir`) in
`castforge-codegen`). Every pass implements the unified `Pass` and `Translator` traits and maps
**100% 1-to-1** to a dedicated source file (`22` `.rs` files) in 8 domain subdirectories under
`crates/castforge-passes/src/` (`general/`, `views/`, `weights/`, `fusion/`, `memory/`,
`distributed/`, `loops/`, `tooling/`).

---

## Summary by Category

| Category | Count | Directory / Section |
| :--- | :---: | :--- |
| **1. General MLIR & Algebraic Passes (+ `O1`/`O2`/`O3`)** | 7 | `src/general/` (Section 1) |
| **2. Views, Shape Specialization & Layout** | 2 | `src/views/` (Section 2) |
| **3. Weight Surgery, Pruning & Quantization** | 4 | `src/weights/` (Section 3) |
| **4. Transformer, GEMM & Operator Fusions** | 5 | `src/fusion/` (Section 4) |
| **5. Memory Planning, Remat & Offload** | 1 | `src/memory/` (Section 5) |
| **6. Distributed Parallelism, Streams & Autodiff** | 2 | `src/distributed/` (Section 6) |
| **7. Polyhedral Loop Scheduling & Tensor Cores** | 1 | `src/loops/` (Section 7) |
| **8. IR Surgery, Verification & Tooling** | 3 | `src/tooling/` (Section 8) |
| **9. Polyhedral / Imperative LoopIR** | +1 | `castforge-codegen` (Section 9) |

*(Total: **25 registered Graph IR passes/presets** in `PassRegistry` + **`OptimizeLoopIrPass`**.)*

---

## 1. General MLIR & Algebraic Optimization Passes (`src/general/`, 7 Entries)

### 1.1 `dce` (`DeadCodeEliminationPass` — `general/dce.rs`)
- **Options:** `symbols=true|false` (default: `true`)
- **Explanation:** Single-pass $O(N)$ reference-counted Dead Code Elimination
  (`Module::compute_use_counts`) that removes unused side-effect-free operations in reverse
  topological order and eliminates unreachable private `func.func` symbols when `symbols=true`.

### 1.2 `cse` (`UnifiedCsePass` — `general/cse.rs`)
- **Options:** `gvn=true|false` (default: `true`), `licm=true|false` (default: `true`),
  `hoist_constants=true|false` (default: `false`)
- **Explanation:** Unified Common Subexpression Elimination (`CSE`), dominator-aware cross-region
  Global Value Numbering (`GVN`), Loop/Region-Invariant Code Motion (`LICM`), and optional
  block-entry constant hoisting/deduplication.

### 1.3 `simplify-algebraic` (`UnifiedSimplifyAlgebraicPass` — `general/simplify_algebraic.rs`)
- **Options:** `fold_constants=true|false` (default: `true`), `conditional=true|false`
  (default: `true`), `egraph=true|false` (default: `true`), `fuse_fma=true|false` (default: `true`),
  `mode=constants|all`
- **Explanation:** Unified algebraic simplifier and constant evaluator combining compile-time
  constant folding & Sparse Conditional Constant Propagation (`SCCP` on `compare`/`select`),
  identity rewrites (`x + 0 -> x`, `x - x -> 0`, `x * 1 -> x`, `x * 0 -> 0`, `max(x, x) -> x`),
  constant reassociation (`(x + c1) + c2 -> x + (c1 + c2)`), distributive factoring
  (`a * b + a * c -> a * (b + c)`), and FMA extraction (`a * b + c -> castforge.fma(a, b, c)`).

### 1.4 `canonicalize` (`CanonicalizePass` — `general/canonicalize.rs`)
- **Options:** `mode=all|toposort|min_memory` (default: `all`), `toposort=true|false`
- **Explanation:** Fixed-point canonicalizer combining `fold-views`, `simplify-algebraic`, `cse`,
  `dce`, and topological / Sethi-Ullman minimum-live-tensor priority scheduling
  (`mode=min_memory`, `castforge.min_memory_scheduled = true`).

### 1.5–1.7 Optimization Pipeline Presets (`O1`, `O2`, `O3` — `general/canonicalize.rs`)
- **`O1` (`OptO1Pass`):** Inlining, `simplify-algebraic`, `cse(licm=true)`, `fold-views`,
  `canonicalize(toposort=true)`, and `dce`.
- **`O2` (`OptO2Pass`):** `O1` + weight folding (`merge-weights(method=lora)`, `fold-weights`) +
  unified Transformer/GEMM fusions (`fuse-norm`, `fuse-attention`, `fuse-activations`, `fuse-gemm`,
  `fuse-elementwise`).
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
- **Options:** `B=val,S=val,default=val`, `batch=N`, `buckets=1;4;8`, `token_prune_ratio=0.5`
- **Explanation:** Unified shape specialization engine supporting symbolic dimension binding
  (`B=4`), static batch resizing across activations (`batch=8`), multi-bucket shape-specialized
  function dispatch (`buckets=1;4;8`), and **ToMe / FastV / StreamingLLM dynamic sequence token
  pruning & bipartite merging** (`token_prune_ratio=0.5`, `castforge.token_merging =
  "tome_bipartite"`).

---

## 3. Weight Surgery, Pruning & Quantization Passes (`src/weights/`, 4 Passes)

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
- **Options:** `mode=batchnorm|norm|pack|externalize|inline|all` (default: `batchnorm`),
  `min_elements=2`, `inline=true|false`
- **Explanation:** Unified compile-time weight folding, pre-packing & SafeTensors storage pass
  absorbing pre-GEMM RMSNorm/LayerNorm scales (`mode=norm`), post-Conv/GEMM BatchNorm scales
  (`mode=batchnorm`), pre-transposing 2D constant RHS weights `[K, N] -> [N, K]` (`mode=pack`,
  `packed_rhs = true`), and externalizing inline `dense<...>` constants to `dense_resource<key>` in
  `TensorStore` (`mode=externalize`) or inlining them back (`mode=inline`).

---

## 4. Transformer, GEMM & Operator Fusion Passes (`src/fusion/`, 5 Passes)

### 4.1 `fuse-elementwise` (`UnifiedFuseElementwisePass` — `fusion/fuse_elementwise.rs`)
- **Options:** `mode=composites|map_reduce|loops|megakernel|decompose|all` (default: `composites`),
  `max_steps=N`, `decompose=true|false`
- **Explanation:** Unified elementwise fusion & decomposition engine supporting composite
  producer-consumer pair fusion (`mode=composites`), map-into-reduce fusion (`mode=map_reduce`),
  register-pipelined loop chain fusion/fission (`mode=loops`), persistent cooperative-grid
  megakernel fusion (`mode=megakernel`), and composite-to-primitive decomposition
  (`mode=decompose`).

### 4.2 `fuse-gemm` (`FuseGemmPass` — `fusion/fuse_gemm.rs`)
- **Options:** `kind=all|horizontal|dequant|epilogue` (default: `all`)
- **Explanation:** Unified GEMM fusion combining horizontal parallel QKV/MLP dot fusion with
  **compile-time constant weight concatenation** (`castforge.horizontal_weights_folded = true`),
  on-the-fly dequant+GEMM (`fused_qmatmul`), and chained GEMM + bias + activation + **residual
  add** epilogue fusion (`epilogue = "bias_relu_residual"`, `castforge.has_residual = true`).

### 4.3 `fuse-norm` (`FuseNormPass` — `fusion/fuse_norm.rs`)
- **Options:** `kind=all|rmsnorm|residual_rmsnorm|layernorm|softmax|log_softmax|cross_entropy`
  (default: `all`), `chunk_size=1024`
- **Explanation:** Unified normalization & softmax fusion combining `Residual+RMSNorm`, `RMSNorm`,
  `LayerNorm`, `Softmax`, `LogSoftmax`, and **Liger-Kernel / Flash-Cross-Entropy**
  (`kind=cross_entropy`, `castforge.fused_linear_cross_entropy = true`).

### 4.4 `fuse-activations` (`FuseActivationsPass` — `fusion/fuse_activations.rs`)
- **Options:** `kind=all|swiglu|gelu|gated_mlp|moe` (default: `all`), `shared_expert=true|false`,
  `top_k=2`, `capacity_factor=1.25`
- **Explanation:** Unified activation, gated-MLP, and routed/shared-expert **Mixture-of-Experts
  (`MoE`)** fusion (`SwiGLU`, `GELU`, `GeGLU`, `FusedGatedMLP`, and `castforge.fused_moe`).

### 4.5 `fuse-attention` (`UnifiedFuseAttentionPass` — `fusion/fuse_attention.rs`)
- **Options:** `kind=all|sdpa|rope|mla|kv_cache|paged|flash_decoding|tree|sliding_window|kv_sharing`
  (`kv_lora_rank=512`, `max_seq_len=2048`, `page_block_size=16`, `max_blocks=64`, `split_kv=4`,
  `draft_tokens=4`, `tree_depth=2`, `window_size=4096`, `sink_tokens=4`, `share_every=2`,
  `rope_scaling=yarn|ntk|llama3`, `rope_factor=8.0`, `logit_softcap=50.0`, `qk_norm=true`,
  `kv_dtype=fp8|int4`)
- **Explanation:** Unified attention subgraph fusion & serving pass combining `RoPE` (with
  **YaRN / NTK-Aware / Llama-3 long-context frequency scaling**), `SDPA`/`GQA` (with **Gemma-2/Grok
  logit soft-capping & QK-Norm**), **DeepSeek MLA** (`kind=mla`), stateful KV-cache injection with
  **FP8 / INT4 KV-cache quantization** (`kv_dtype=fp8|int4`), vLLM PagedAttention (`kind=paged`),
  Split-KV FlashDecoding (`kind=flash_decoding`), speculative tree attention (`kind=tree`),
  **Sliding-Window + Attention Sinks** (`kind=sliding_window`), and **YOCO / CLA Cross-Layer
  KV-Cache Sharing** (`kind=kv_sharing,share_every=2`).

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
  distributed parallelism & stream pass (`mode=tp|ring_cp|pp|fsdp|zero3|ep|streams`, `devices=2`,
  `ring_size=4`, `coalesce=true`, `stages=2`, `microbatches=4`, `schedule=1f1b`, `streams=2`)
  consolidating 1D Megatron Tensor Parallelism (`tp`), RingAttention Context Parallelism
  (`ring_cp`), **Expert Parallelism (`mode=ep`) with `all_to_all` token dispatch/combine**,
  **ZeRO-3 / FSDP parameter sharding with `all_gather` prefetch & `reduce_scatter` gradient sync**
  (`mode=fsdp|zero3`), `all_reduce` coalescing, 1F1B/Zero-Bubble Pipeline Parallelism (`pp`), and
  multi-stream async execution.
- **`autodiff` (`AutodiffPass` — `distributed/autodiff.rs`):** Synthesizes reverse-mode VJP
  backward gradient operations and optional in-graph SGD updates (`lr=...`).

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

## 8. IR Surgery, Verification, Profiling & Tooling (`src/tooling/`, 3 Passes)

1. **`ir-edit` (`IrEditPass` — `tooling/ir_edit.rs`):** Unified surgical IR editor & canonicalizer
   consolidating op renaming (`rename_op=from:to`), attribute injection (`annotate_op=op:k=v`),
   input freezing/unfreezing (`freeze=true,args=arg0:1.0`, `unfreeze=true,min_elements=2`),
   intermediate value tapping (`tap=%0;%1`), canonical SSA renaming (`ssa=true,prefix=v`), and
   debug/schedule metadata stripping (`debuginfo=true,attributes=true`).
2. **`outline-subgraphs` (`UnifiedOutlineSubgraphsPass` — `tooling/outline_subgraphs.rs`):**
   Unified function/subgraph surgery pass (`mode=all|gemm|composites|inline|extract|reduce|dedup|
   compose|unroll`, `target=%v`, `stop_at=%in`, `max_ops=1`, `keep_op=...`, `dedup=true`,
   `first=...`, `second=...`, `repeat=N`) consolidating kernel outlining, function inlining
   (`mode=inline`), backward dataflow cone slicing (`mode=extract`), delta IR reduction
   (`mode=reduce`), LLVM `-mergefunc` function deduplication, function composition, and bounded
   `func.call` unrolling.
3. **`verify-ir` (`UnifiedVerifyIrPass` — `tooling/verify_ir.rs`):** Unified IR verification,
   numerical sanitizer, structural hashing & Roofline profiling pass (`shapes=true,equiv=true,
   tol=1e-4,hash=true,bisect=N,sanitize=true,clamp_max=65504.0,profile=true,instrument=true,
   filter=gemm,ridge=10.0`) consolidating shape verification, signature equivalence, 64-bit FNV-1a
   structural hashing, `castforge.nan_guard` finite clamps, pass bisection, Roofline
   FLOPs/arithmetic-intensity analysis, and `castforge.profile_region` timer instrumentation.

---

## 9. Polyhedral / Imperative LoopIR Pass (`castforge-codegen`, 1 Unified Pass)

- **`optimize-loop-ir` (`OptimizeLoopIrPass`):** Unified imperative `LoopModule` transformation pass
  orchestrating `LoopFusionPass`, `LoopFissionPass`, `LoopInterchangePass`,
  `LoopCollapseParallelPass`, `LoopUnrollAndJamPass`, `LoopUnrollPass`, and `LoopVectorizePass`.
