# CastForge — Complete Reference of All 21 Graph IR Passes (+ Unified LoopIR Pass)

CastForge provides **21 built-in modular A-to-B IR passes & presets** (`18 passes + 3 presets
O1/O2/O3`) across `src/passes/` and `src/ir/verify_ir/` (plus the unified **`OptimizeLoopIrPass`**
(`optimize-loop-ir`) in `src/codegen/`). Core pass execution infrastructure (`Pass`, `PassManager`,
`PassRegistry`) and `verify-ir` live in `src/ir/`, while all transformation passes and
`PassRegistry::with_builtin_passes()` are organized into focused pass modules and sub-packages
(`src/passes/<pass>/`) under `src/passes/`.

---

## Summary by Category

| Category | Count | Section |
| :--- | :---: | :--- |
| **1. General MLIR & Algebraic Passes (+ `O1`/`O2`/`O3`)** | 7 | Section 1 |
| **2. Views, Shape Specialization & Layout** | 2 | Section 2 |
| **3. Weight Surgery, Pruning & Quantization** | 3 | Section 3 |
| **4. Transformer, GEMM & Operator Fusions** | 4 | Section 4 |
| **5. Memory Planning, Remat & Offload** | 1 | Section 5 |
| **6. Distributed Parallelism, Streams & Autodiff** | 1 | Section 6 |
| **7. Polyhedral Loop Scheduling & Tensor Cores** | 1 | Section 7 |
| **8. IR Surgery, Verification & Tooling** | 2 | Section 8 |
| **9. Polyhedral / Imperative LoopIR** | +1 | `src/codegen/` (Section 9) |

*(Total: **21 registered Graph IR passes/presets** in `PassRegistry` + **`OptimizeLoopIrPass`**.)*

---

## 1. General MLIR & Algebraic Optimization Passes (`src/passes/`, 7 Entries)

### 1.1 `dce` (`DeadCodeEliminationPass` — `dce.rs`)
- **Options:** `symbols=true|false` (default: `true`)
- **Explanation:** Single-pass $O(N)$ reference-counted Dead Code Elimination
  (`Module::compute_use_counts`) that removes unused side-effect-free operations in reverse
  topological order and eliminates unreachable private `func.func` symbols when `symbols=true`.

### 1.2 `cse` (`UnifiedCsePass` — `cse.rs`)
- **Options:** `gvn=true|false` (default: `true`), `licm=true|false` (default: `true`),
  `hoist_constants=true|false` (default: `false`)
- **Explanation:** Unified Common Subexpression Elimination (`CSE`), dominator-aware cross-region
  Global Value Numbering (`GVN`), Loop/Region-Invariant Code Motion (`LICM`), and optional
  block-entry constant hoisting/deduplication.

### 1.3 `simplify-algebraic` (`UnifiedSimplifyAlgebraicPass` — `simplify_algebraic.rs`)
- **Options:** `fold_constants=true|false` (default: `true`), `conditional=true|false`
  (default: `true`), `egraph=true|false` (default: `true`), `fuse_fma=true|false` (default: `true`),
  `mode=constants|all`
- **Explanation:** Unified algebraic simplifier and constant evaluator combining compile-time
  constant folding & Sparse Conditional Constant Propagation (`SCCP` on `compare`/`select`),
  identity rewrites (`x + 0 -> x`, `x - x -> 0`, `x * 1 -> x`, `x * 0 -> 0`, `max(x, x) -> x`),
  constant reassociation (`(x + c1) + c2 -> x + (c1 + c2)`), distributive factoring
  (`a * b + a * c -> a * (b + c)`), and FMA extraction (`a * b + c -> castforge.fma(a, b, c)`).

### 1.4 `canonicalize` (`CanonicalizePass` — `canonicalize.rs`)
- **Options:** `mode=all|toposort|min_memory` (default: `all`), `toposort=true|false`
- **Explanation:** Fixed-point canonicalizer combining `fold-views`, `simplify-algebraic`, `cse`,
  `dce`, and topological / Sethi-Ullman minimum-live-tensor priority scheduling
  (`mode=min_memory`, `castforge.min_memory_scheduled = true`).

### 1.5–1.7 Optimization Pipeline Presets (`O1`, `O2`, `O3` — `canonicalize.rs`)
- **`O1` (`OptO1Pass`):** Inlining, `simplify-algebraic`, `cse(licm=true)`, `fold-views`,
  `canonicalize(toposort=true)`, and `dce`.
- **`O2` (`OptO2Pass`):** `O1` + weight folding (`merge-weights(method=lora)`, `fold-weights`) +
  unified Transformer/GEMM fusions (`fuse-norm`, `fuse-attention`, `fuse-elementwise`, `fuse-gemm`).
- **`O3` (`OptO3Pass`):** `specialize-shapes` + `O2` + `fuse-elementwise(mode=moe)` +
  `simplify-algebraic` + `shard-distributed` + `optimize-memory` + `fold-weights(mode=pack)`.

---

## 2. Views, Shape Specialization & Layout Passes (`src/passes/`, 2 Passes)

### 2.1 `fold-views` (`FoldViewsPass` — `fold_views.rs`)
- **Options:** `layout=nhwc|nchw|col_major`, `propagate=true|false`
- **Explanation:** Unified folding of `transpose` (sinking & permutation composition), `reshape`,
  `broadcast_in_dim`, `slice`/`concatenate`, round-trip `convert` casts, and spatial/matrix layout
  conversion & global layout propagation (`layout=nhwc,propagate=true`).

### 2.2 `specialize-shapes` (`UnifiedSpecializeShapesPass` — `specialize_shapes.rs`)
- **Options:** `B=val,S=val,default=val`, `batch=N`, `buckets=1;4;8`, `token_prune_ratio=0.5`
- **Explanation:** Unified shape specialization engine supporting symbolic dimension binding
  (`B=4`), static batch resizing across activations (`batch=8`), multi-bucket shape-specialized
  function dispatch (`buckets=1;4;8`), and **ToMe / FastV / StreamingLLM dynamic sequence token
  pruning & bipartite merging** (`token_prune_ratio=0.5`, `castforge.token_merging =
  "tome_bipartite"`).

---

## 3. Weight Surgery, Pruning & Quantization Passes (`src/passes/`, 3 Passes)

### 3.1 `quantize` (`QuantizePass` — `quantize.rs`)
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

### 3.2 `merge-weights` (`UnifiedMergeWeightsPass` — `merge_weights.rs`)
- **Options:** `method=linear|slerp|ties|dare|lora|dora`, `t=0.5`, `alpha=1.0`, `lambda=1.0`,
  `drop_rate=0.0`
- **Explanation:** Unified model merging and adapter baking pass supporting Linear, SLERP, TIES,
  DARE task-vector arithmetic (`method=dare`), compile-time LoRA adapter baking (`method=lora`), and
  **DoRA (Weight-Decomposed Low-Rank Adaptation)** directional column normalization & magnitude
  baking (`method=dora`, `castforge.dora_baked = true`).

### 3.3 `fold-weights` (`FoldWeightsPass` — `fold_weights.rs`)
- **Options:** `mode=batchnorm|norm|pack|externalize|inline|prune|structured|2to4|activation_sparse`
  (default: `batchnorm`), `min_elements=2`, `inline=true|false`, `sparsity=0.5`
- **Explanation:** Unified compile-time weight folding, pre-packing, SafeTensors storage &
  sparsity pass absorbing pre-GEMM RMSNorm/LayerNorm scales (`mode=norm`), post-Conv/GEMM BatchNorm
  scales (`mode=batchnorm`), pre-transposing 2D constant RHS weights `[K, N] -> [N, K]`
  (`mode=pack`), externalizing inline `dense<...>` constants to `dense_resource<key>` in
  `TensorStore` (`mode=externalize` / `inline`), L1 structured column pruning
  (`mode=prune|structured`), NVIDIA Ampere/Hopper 2:4 structured sparsity (`mode=2to4`), and
  **dynamic top-$k$ activation sparsity** (`mode=activation_sparse`).

---

## 4. Transformer, GEMM & Operator Fusion Passes (`src/passes/`, 4 Passes)

### 4.1 `fuse-elementwise` (`UnifiedFuseElementwisePass` — `fuse_elementwise.rs`)
- **Options:** `mode=composites|map_reduce|loops|megakernel|decompose|activations|swiglu|gelu|moe`
  (default: `composites`), `max_steps=N`, `decompose=true|false`, `shared_expert=true|false`,
  `top_k=2`, `capacity_factor=1.25`
- **Explanation:** Unified elementwise, activation, Gated-MLP, and MoE fusion engine supporting
  composite producer-consumer pair fusion (`mode=composites`), `SwiGLU` / `GELU` / `GeGLU` /
  `FusedGatedMLP` (`mode=activations|swiglu|gelu|gated_mlp`), routed/shared-expert **Mixture of
  Experts (`mode=moe`)**, map-into-reduce fusion (`mode=map_reduce`), register-pipelined loop chain
  fusion/fission (`mode=loops`), persistent cooperative-grid megakernel fusion (`mode=megakernel`),
  and composite-to-primitive decomposition (`mode=decompose`).

### 4.2 `fuse-gemm` (`FuseGemmPass` — `fuse_gemm.rs`)
- **Options:** `kind=all|horizontal|dequant|epilogue` (default: `all`)
- **Explanation:** Unified GEMM fusion combining horizontal parallel QKV/MLP dot fusion with
  **compile-time constant weight concatenation** (`castforge.horizontal_weights_folded = true`),
  on-the-fly dequant+GEMM (`fused_qmatmul`), and chained GEMM + bias + activation + **residual
  add** epilogue fusion (`epilogue = "bias_relu_residual"`, `castforge.has_residual = true`).

### 4.3 `fuse-norm` (`FuseNormPass` — `fuse_norm.rs`)
- **Options:** `kind=all|rmsnorm|residual_rmsnorm|layernorm|softmax|log_softmax|cross_entropy`
  (default: `all`), `chunk_size=1024`
- **Explanation:** Unified normalization & softmax fusion combining `Residual+RMSNorm`, `RMSNorm`,
  `LayerNorm`, `Softmax`, `LogSoftmax`, and **Liger-Kernel / Flash-Cross-Entropy**
  (`kind=cross_entropy`, `castforge.fused_linear_cross_entropy = true`).

### 4.4 `fuse-attention` (`UnifiedFuseAttentionPass` — `fuse_attention.rs`)
- **Options:** `kind=all|sdpa|rope|mla|ssm|mamba2|kv_cache|paged|flash_decoding|tree|sliding_window`
  (`kv_lora_rank=512`, `state_dim=128`, `chunk_size=64`, `max_seq_len=2048`, `page_block_size=16`,
  `max_blocks=64`, `split_kv=4`, `draft_tokens=4`, `tree_depth=2`, `window_size=4096`,
  `sink_tokens=4`, `share_every=2`, `rope_scaling=yarn|ntk|llama3`, `rope_factor=8.0`,
  `logit_softcap=50.0`, `qk_norm=true`, `kv_dtype=fp8|int4`)
- **Explanation:** Unified attention & state-space subgraph fusion pass combining `RoPE` (with
  **YaRN / NTK-Aware / Llama-3 frequency scaling**), `SDPA`/`GQA` (with **Gemma-2/Grok logit
  soft-capping & QK-Norm**), **DeepSeek MLA** (`kind=mla`), **Mamba-2 SSD / RWKV-6 / Gated-DeltaNet
  Linear Attention chunk-scan fusion** (`kind=ssm|mamba2|linear_attn,state_dim=128,chunk_size=64`),
  stateful KV-cache injection with **FP8 / INT4 KV-cache quantization** (`kv_dtype=fp8|int4`), vLLM
  PagedAttention (`kind=paged`), Split-KV FlashDecoding (`kind=flash_decoding`), speculative tree
  attention (`kind=tree`), **Sliding-Window + Attention Sinks** (`kind=sliding_window`), and **YOCO
  / CLA Cross-Layer KV-Cache Sharing** (`kind=kv_sharing`).

---

## 5. Memory Planning, Rematerialization & Offloading (`src/passes/`, 1 Pass)

- **`optimize-memory` (`OptimizeMemoryPass` — `optimize_memory.rs`):** Unified memory
  optimization pass (`mode=all|inplace|zero_copy|remat|selective_remat|offload`, `min_distance=3`,
  `min_span=2`) consolidating in-place arena buffer donation (`inplace`), destination-passing
  zero-copy `concatenate` elimination (`zero_copy`), activation rematerialization (`remat`),
  **Selective Activation Checkpointing** (`mode=selective_remat`, checkpointing compute-heavy GEMMs
  while recomputing memory-bound activations/norms), and pinned host-memory D2H/H2D offloading
  (`offload`).

---

## 6. Distributed Parallelism, Streams & Autodiff (`src/passes/`, 1 Pass)

- **`shard-distributed` (`ShardDistributedPass` — `shard_distributed.rs`):** Unified
  distributed parallelism, stream & reverse-mode AD pass
  (`mode=tp|ring_cp|pp|fsdp|zero3|ep|streams|autodiff`, `devices=2`, `ring_size=4`, `coalesce=true`,
  `stages=2`, `microbatches=4`, `schedule=1f1b`, `streams=2`, `lr=0.0`) consolidating 1D Megatron
  Tensor Parallelism (`tp`), RingAttention Context Parallelism (`ring_cp`), **Expert Parallelism
  (`mode=ep`) with `all_to_all` dispatch/combine**, **ZeRO-3 / FSDP parameter sharding**
  (`mode=fsdp|zero3`), `all_reduce` coalescing, 1F1B/Zero-Bubble Pipeline Parallelism (`pp`),
  multi-stream async execution, and **reverse-mode VJP automatic differentiation & SGD updates**
  (`mode=autodiff`).

---

## 7. Polyhedral Loop Scheduling & Tensor Cores (`src/passes/`, 1 Pass)

- **`schedule-loops` (`ScheduleLoopsPass` — `schedule_loops.rs`):** Unified loop schedule,
  2D `convolution` to `im2col + dot_general` lowering (`lower_conv=true`), wide GEMM column
  splitting (`max_n=256`), tile alignment padding (`align=8`), L1/L2/register tiling (`size=16,
  l2_size=64,reg_tile=4`), loop permutation/skewing (`order=mkn,collapse=true,skew=true`),
  unrolling/pipelining/Split-K/double-buffering (`factor=4,prefetch=2,split_k=2,
  double_buffer=true`), multi-arch SIMD vectorization (`isa=avx2|avx512|neon|sve|rvv|wasm`), and GPU
  Tensor/Matrix Core lowering (`tensor_cores=wgmma|tcgen05|blackwell|wmma|mfma|simdgroup,
  mma_tile=16x16x16,warp_specialized=true,tmem_columns=512`), including **NVIDIA Blackwell
  `tcgen05.mma` Tensor Memory (`TMEM`) allocation and TMA producer/consumer warp specialization**.

---

## 8. IR Surgery, Verification, Profiling & Tooling (`src/passes/`, 2 Passes)

1. **`ir-edit` (`IrEditPass` — `ir_edit.rs`):** Unified surgical IR editor, canonicalizer
   and subgraph outliner (`mode=edit|outline|all|gemm|composites|inline|extract|reduce|dedup|
   compose|unroll`, `rename_op=from:to`, `annotate_op=op:k=v`, `freeze=true,args=arg0:1.0`,
   `unfreeze=true,min_elements=2`, `tap=%0;%1`, `ssa=true,prefix=v`, `debuginfo=true,
   attributes=true`, `target=%v`, `stop_at=%in`, `max_ops=1`, `keep_op=...`, `dedup=true`,
   `first=...`, `second=...`, `repeat=N`).
2. **`verify-ir` (`UnifiedVerifyIrPass` — `verify_ir.rs`):** Unified IR verification,
   numerical sanitizer, structural hashing & Roofline profiling pass (`shapes=true,equiv=true,
   tol=1e-4,hash=true,bisect=N,sanitize=true,clamp_max=65504.0,profile=true,instrument=true,
   filter=gemm,ridge=10.0`) consolidating shape verification, signature equivalence, 64-bit FNV-1a
   structural hashing, `castforge.nan_guard` finite clamps, pass bisection, Roofline
   FLOPs/arithmetic-intensity analysis, and `castforge.profile_region` timer instrumentation.

---

## 9. Polyhedral / Imperative LoopIR Pass (`src/codegen/`, 1 Unified Pass)

- **`optimize-loop-ir` (`OptimizeLoopIrPass`):** Unified imperative `LoopModule` transformation pass
  orchestrating `LoopFusionPass`, `LoopFissionPass`, `LoopInterchangePass`,
  `LoopCollapseParallelPass`, `LoopUnrollAndJamPass`, `LoopUnrollPass`, and `LoopVectorizePass`.
