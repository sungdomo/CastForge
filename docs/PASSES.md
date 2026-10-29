# CastForge — Complete Reference of All 67 Graph IR Passes (+ 7 LoopIR Passes)

CastForge provides **67 built-in modular A-to-B IR passes & presets** in `castforge-passes` (plus
**7 Imperative LoopIR passes** in `castforge-codegen`, totaling **74 passes**). Every pass
implements the unified `Pass` and `Translator` traits and is organized into 8 domain subdirectories
under `crates/castforge-passes/src/` (`general/`, `views/`, `weights/`, `fusion/`, `memory/`,
`distributed/`, `loops/`, `tooling/`).

---

## Summary by Category

| Category | Count | Directory / Section |
| :--- | :---: | :--- |
| **1. General MLIR & Algebraic Passes** | 13 | `src/general/` (Section 1) |
| **2. Views, Shape Specialization & Layout** | 5 | `src/views/` (Section 2) |
| **3. Weight Surgery, Pruning & Quantization** | 8 | `src/weights/` (Section 3) |
| **4. Transformer, GEMM & Operator Fusions** | 13 | `src/fusion/` (Section 4) |
| **5. Memory, Rematerialization & Offload** | 4 | `src/memory/` (Section 5) |
| **6. Distributed Parallelism, Attention & Autodiff** | 5 | `src/distributed/` (Section 6) |
| **7. Polyhedral Loop Scheduling & Tensor Cores** | 2 | `src/loops/` (Section 7) |
| **8. IR Surgery, Verification & Tooling** | 17 | `src/tooling/` (Section 8) |
| **9. Polyhedral / Imperative LoopIR** | +7 | `castforge-codegen` (Section 9) |

*(Total: **67 registered Graph IR passes/presets** in `PassRegistry` + **7 LoopIR passes**.)*

---

## 1. General MLIR & Algebraic Optimization Passes (`src/general/`, 13 Passes)

### 1.1 `dce` (`DeadCodeEliminationPass`)
- **Options:** `symbols=true|false` (default: `true`)
- **Explanation:** Single-pass $O(N)$ reference-counted Dead Code Elimination
  (`Module::compute_use_counts`) that removes unused side-effect-free operations in reverse
  topological order and eliminates unreachable private `func.func` symbols when `symbols=true`.

### 1.2 `cse` (`UnifiedCsePass`)
- **Options:** `gvn=true|false` (default: `true`), `hoist_constants=true|false` (default: `false`)
- **Explanation:** Unified Common Subexpression Elimination (`CSE`), dominator-aware cross-region
  Global Value Numbering (`GVN`), and optional block-entry constant hoisting/deduplication.
  Canonicalizes commutative binary operand order and attribute keys.

### 1.3 `simplify-algebraic` (`UnifiedSimplifyAlgebraicPass`)
- **Options:** `egraph=true|false` (default: `true`), `fuse_fma=true|false` (default: `true`)
- **Explanation:** Unified algebraic simplifier combining identity rewrites (`x + 0 -> x`,
  `x - x -> 0`, `x * 1 -> x`, `x * 0 -> 0`, `max(x, x) -> x`), constant reassociation
  (`(x + c1) + c2 -> x + (c1 + c2)`), distributive factoring (`a * b + a * c -> a * (b + c)`), and
  FMA extraction (`a * b + c -> castforge.fma(a, b, c)`).

### 1.4 `fold-constants` (`FoldConstantsPass`)
- **Options:** `conditional=true|false` (default: `true`)
- **Explanation:** Evaluates unary, binary, shape (`transpose`, `reshape`, `slice`, `concatenate`,
  `broadcast_in_dim`), and reduction ops on compile-time constants, plus Sparse Conditional Constant
  Propagation (`SCCP`) on constant `compare`/`select` branches.

### 1.5 `canonicalize` (`CanonicalizePass`)
- **Options:** None
- **Explanation:** Fixed-point canonicalizer combining `fold-views`, `simplify-algebraic`, `cse`,
  `fold-constants`, and `dce`.

### 1.6 `licm` (`LicmPass`)
- **Options:** None
- **Explanation:** Loop/Region-Invariant Code Motion out of nested regions.

### 1.7 `inliner` (`InlinerPass`)
- **Options:** None
- **Explanation:** Inlines single-block `func.call` invocations into caller blocks.

### 1.8 `topological-sort` (`TopologicalSortPass`)
- **Options:** None
- **Explanation:** Reorders operations inside each basic block using Kahn's dependency algorithm so
  every SSA value is defined before its uses.

### 1.9 `verify-shapes` (`VerifyShapesPass`)
- **Options:** None
- **Explanation:** Verifies SSA dominance, operand/result counts, element types, and `dot_general`
  contracting/batching dimensions.

### 1.10–1.13 Optimization Pipeline Presets (`O1`, `O2`, `O3`, `llm-opt`)
- **`O1` (`OptO1Pass`):** Inlining, `fold-constants`, `licm`, `simplify-algebraic`, `fold-views`,
  `cse`, `dce`, and `topological-sort`.
- **`O2` (`OptO2Pass`):** `O1` + weight folding (`merge-weights`, `fold-batchnorm`) + unified
  Transformer/GEMM fusions (`fuse-rope`, `fuse-attention`, `fuse-norm`, `fuse-softmax`,
  `fuse-map-reduce`, `fuse-gemm`, `fuse-activations`, `fuse-elementwise`).
- **`O3` / `llm-opt` (`OptO3Pass`):** `specialize-shapes` + `O2` + `pack-weights` +
  `fuse-elementwise(mode=loops)` + `bufferize-inplace` + `shard-distributed` (`llm-opt`).

---

## 2. Views, Shape Specialization & Layout Passes (`src/views/`, 5 Passes)

### 2.1 `fold-views` (`FoldViewsPass`)
- **Options:** None
- **Explanation:** Unified folding of `transpose` (sinking & permutation composition), `reshape`,
  `broadcast_in_dim`, `slice`/`concatenate`, and round-trip `convert` casts.

### 2.2 `specialize-shapes` (`UnifiedSpecializeShapesPass`)
- **Options:** `B=val,S=val,default=val`, `batch=N`, `buckets=1;4;8`
- **Explanation:** Unified shape specialization engine supporting symbolic dimension binding
  (`B=4`), static batch resizing across activations (`batch=8`), and multi-bucket shape-specialized
  function dispatch (`buckets=1;4;8`).

### 2.3 `convert-layout` (`ConvertLayoutPass`)
- **Options:** `layout=nhwc|nchw|col_major|row_major` (default: `nhwc`)
- **Explanation:** Inserts boundary layout-conversion transposes around compute ops.

### 2.4 `split-large-gemm` (`SplitLargeGemmPass`)
- **Options:** `max_n=N` (default: `2`)
- **Explanation:** Splits wide `dot_general` GEMMs (`N > max_n`) into sliced column sub-GEMMs +
  `stablehlo.concatenate`.

### 2.5 `lower-conv-to-gemm` (`LowerConvToGemmPass`)
- **Options:** None
- **Explanation:** Lowers 2D `stablehlo.convolution` into `castforge.im2col` +
  `stablehlo.dot_general`.

---

## 3. Weight Surgery, Pruning & Quantization Passes (`src/weights/`, 8 Passes)

### 3.1 `quantize` (`QuantizePass`)
- **Options:** `scheme=weight|groupwise|fp8|w8a8`, `bits=8|4`, `group_size=32`, `awq_salient=true`,
  `format=e4m3|e5m2`, `mxfp8=true`, `block_size=32`, `samples=4`
- **Explanation:** Unified quantization pass consolidating per-tensor `i8`/`i4` PTQ
  (`scheme=weight`), AWQ/GPTQ group-wise weight quantization (`scheme=groupwise`), Hopper/Blackwell
  `FP8` & `MXFP8` block-scaled quantization (`scheme=fp8`), and activation-calibrated static `W8A8`
  quantization (`scheme=w8a8`).

### 3.2 `hadamard-quant-rotation` (`HadamardQuantRotationPass`)
- **Options:** `block_size=N` (default: `64`)
- **Explanation:** **QuaRot / SpinQuant** orthogonal Hadamard basis rotation
  ($W' = H W$, $X' = X H^T$). Applies fast normalized Walsh-Hadamard transforms across `block_size`
  channel blocks to suppress massive activation outliers before 4-bit/FP8 quantization
  (`castforge.hadamard_rotated = true`).

### 3.3 `mixed-precision` (`MixedPrecisionPass`)
- **Options:** `dtype=f16|bf16` (default: `f16`)
- **Explanation:** Inserts narrowing `f16`/`bf16` casts on GEMM/Conv inputs with `f32` accumulation.

### 3.4 `prune-weights` (`PruneWeightsPass`)
- **Options:** `mode=structured|2to4` (default: `structured`), `sparsity=0.5`
- **Explanation:** Unified weight pruning pass supporting L1 structured column pruning
  (`mode=structured`) and NVIDIA Ampere/Hopper 2:4 fine-grained structured sparsity (`mode=2to4`).

### 3.5 `merge-weights` (`UnifiedMergeWeightsPass`)
- **Options:** `method=linear|slerp|ties|dare|lora`, `t=0.5`, `alpha=1.0`, `lambda=1.0`,
  `drop_rate=0.0`
- **Explanation:** Unified model merging and adapter baking pass supporting Linear, SLERP, TIES,
  DARE task-vector arithmetic (`method=dare`), and compile-time LoRA adapter baking (`method=lora`).

### 3.6 `fold-batchnorm` (`FoldBatchNormPass`)
- **Options:** None
- **Explanation:** Absorbs pre-GEMM RMSNorm/LayerNorm scales and post-Conv/GEMM BatchNorm scales
  directly into constant weight tensors.

### 3.7 `pack-weights` (`PackWeightsPass`)
- **Options:** None
- **Explanation:** Pre-packs 2D constant RHS weights from `[K, N]` to `[N, K]` (`packed_rhs = true`)
  for unit-stride SIMD dot products.

### 3.8 `externalize-weights` (`ExternalizeWeightsPass`)
- **Options:** `min_elements=2`, `inline=false`
- **Explanation:** Moves inline `dense<...>` constants to `dense_resource<key>` in `TensorStore`
  (or inlines them back when `inline=true`).

---

## 4. Transformer, GEMM & Operator Fusion Passes (`src/fusion/`, 13 Passes)

### 4.1 `fuse-elementwise` (`UnifiedFuseElementwisePass`)
- **Options:** `mode=pair|loops` (default: `pair`), `max_steps=N` (default: `0`)
- **Explanation:** Unified elementwise fusion engine supporting composite producer-consumer pair
  fusion (`mode=pair`), multi-op register-pipelined loop chain fusion (`mode=loops`), and
  register-pressure loop fission when `max_steps > 0`.

### 4.2 `fuse-gemm` (`FuseGemmPass`)
- **Options:** `kind=all|horizontal|dequant|epilogue` (default: `all`)
- **Explanation:** Unified GEMM fusion combining horizontal parallel QKV/MLP dot fusion,
  on-the-fly dequant+GEMM (`fused_qmatmul`), and GEMM+bias+activation epilogue fusion.

### 4.3 `fuse-norm` (`FuseNormPass`)
- **Options:** `kind=all|rmsnorm|residual_rmsnorm|layernorm` (default: `all`)
- **Explanation:** Unified normalization fusion (`Residual+RMSNorm`, `RMSNorm`, and `LayerNorm`).

### 4.4 `fuse-activations` (`FuseActivationsPass`)
- **Options:** `kind=all|swiglu|gelu|gated_mlp` (default: `all`)
- **Explanation:** Unified activation & gated-MLP fusion (`SwiGLU`, `GELU`, `GeGLU`, and
  `FusedGatedMLP`).

### 4.5 `fuse-softmax` (`FuseSoftmaxPass`)
- **Options:** None
- **Explanation:** Fuses Softmax and LogSoftmax subgraphs into online 3-pass composites.

### 4.6 `fuse-rope` (`FuseRopePass`)
- **Options:** None
- **Explanation:** Fuses Rotary Position Embedding (`RoPE`) subgraphs into `castforge.fused_rope`.

### 4.7 `fuse-attention` (`FuseAttentionPass`)
- **Options:** None
- **Explanation:** Fuses SDPA & GQA (`Q @ K^T * scale (+ mask) -> softmax -> @ V`) into
  `castforge.fused_attention`.

### 4.8 `fuse-mla` (`FuseMlaPass`)
- **Options:** `kv_lora_rank=N` (default: `512`)
- **Explanation:** **DeepSeek-V2/V3 Multi-Head Latent Attention (`MLA`)** fusion. Absorbs the KV
  latent up-projection matrix $W^{UK}$ into $W^Q$ ($W^{Q\_absorbed} = W^Q (W^{UK})^T$) and
  annotates attention/GEMM ops with compressed KV latent rank metadata
  (`castforge.mla_compressed_kv = true`).

### 4.9 `optimize-attention` (`OptimizeAttentionPass`)
- **Options:** `mode=kv_cache|paged|flash_decoding|tree|all`, `max_seq_len=2048`,
  `page_block_size=16`, `max_blocks=64`, `split_kv=4`, `draft_tokens=4`, `tree_depth=2`
- **Explanation:** Unified LLM inference attention optimizer combining stateful KV-cache injection
  (`mode=kv_cache`), vLLM PagedAttention block tables (`mode=paged`), Split-KV FlashDecoding with
  LSE reduction (`mode=flash_decoding`), and Medusa/EAGLE speculative tree attention (`mode=tree`).

### 4.10 `fuse-moe` (`FuseMoePass`)
- **Options:** None
- **Explanation:** Fuses routed Mixture-of-Experts (`MoE`) expert GEMMs and router gate combine.

### 4.11 `fuse-linear-cross-entropy` (`FuseLinearCrossEntropyPass`)
- **Options:** `chunk_size=N` (default: `4096`)
- **Explanation:** **Liger-Kernel / Flash-Cross-Entropy** fusion. Fuses the final LM classifier head
  projection (`hidden @ lm_head`) with chunked online log-sum-exp Cross-Entropy loss so the massive
  `[B*S, Vocab]` logits tensor is never materialized in HBM
  (`castforge.materialize_logits = false`).

### 4.12 `fuse-map-reduce` (`FuseMapReducePass`)
- **Options:** None
- **Explanation:** Fuses elementwise map producers directly into `stablehlo.reduce`.

### 4.13 `fuse-megakernel` & `decompose-composites`
- **`fuse-megakernel`:** Fuses sequential compute ops into a persistent cooperative-grid megakernel.
- **`decompose-composites`:** Expands `stablehlo.composite` ops back to primitive StableHLO ops.

---

## 5. Memory Planning, Rematerialization & Offloading (`src/memory/`, 4 Passes)

- **`bufferize-inplace` (`BufferizeInplacePass`):** Annotates elementwise ops whose input dies at
  the op for zero-copy in-place arena slot reuse.
- **`rematerialize` (`RematerializePass`):** Recomputes long-lived cheap activations before distant
  consumers (`min_distance=3`).
- **`offload-activations` (`OffloadActivationsPass`):** Inserts `castforge.host_offload` (D2H/H2D
  pinned host memory) on long-lived tensors (`min_span=2`).
- **`eliminate-concat-copies` (`EliminateConcatCopiesPass`):** Destination-passing zero-copy
  `stablehlo.concatenate` optimization.

---

## 6. Distributed Parallelism, Streams & Autodiff (`src/distributed/`, 4 Passes)

- **`shard-distributed` (`ShardDistributedPass`):** Unified distributed parallelism pass combining
  1D Megatron SPMD Tensor Parallelism (`mode=tp,devices=N`), Context-Parallel RingAttention
  (`mode=ring_cp,ring_size=N`), and `all_reduce` coalescing/overlap (`coalesce=true`).
- **`shard-pipeline-parallel` (`ShardPipelineParallelPass`):** **1F1B / Zero-Bubble Pipeline
  Parallelism (`PP`)** pass (`stages=2,microbatches=4,schedule=1f1b`). Partitions sequential compute
  blocks across pipeline stages and attaches microbatch P2P schedule metadata.
- **`schedule-streams` (`ScheduleStreamsPass`):** Assigns independent dataflow branches to
  concurrent hardware streams (`streams=2`) with `stream_wait` barriers.
- **`autodiff` (`AutodiffPass`):** Synthesizes reverse-mode VJP backward gradient operations and
  optional in-graph SGD updates (`lr=...`).

---

## 7. Polyhedral Loop Scheduling & Tensor Cores (`src/loops/`, 2 Passes)

- **`schedule-loops` (`ScheduleLoopsPass`):** Unified loop schedule pass consolidating tile
  alignment padding (`align=8`), L1/L2/register tiling (`size=16,l2_size=64,reg_tile=4`), loop
  permutation and wavefront skewing (`order=mkn,collapse=true,skew=true`), unrolling/pipelining/
  Split-K/double-buffering (`factor=4,prefetch=2,split_k=2,double_buffer=true`), and multi-arch SIMD
  vectorization (`isa=avx2|avx512|neon|sve|rvv|wasm`).
- **`lower-tensor-cores` (`LowerTensorCoresPass`):** Lowers GEMMs and fused attention ops to GPU
  Tensor Core / Matrix Core hardware instructions (`arch=wgmma|wmma|mfma|simdgroup`,
  `tile=64x128x16`).

---

## 8. IR Surgery, Verification, Profiling & Tooling (`src/tooling/`, 17 Passes)

1. **`ir-edit` (`IrEditPass`):** Surgical op renaming and attribute injection.
2. **`extract-subgraph` (`ExtractSubgraphPass`):** Backward dataflow cone slicer.
3. **`outline-subgraphs` (`OutlineSubgraphsPass`):** Outlines GEMM/composite ops into functions.
4. **`unroll-calls` (`UnrollCallsPass`):** Unrolls repeated `func.call` invocations (`repeat=N`).
5. **`compose-funcs` (`ComposeFuncsPass`):** Chains two module functions `f2(f1(x))`.
6. **`dedup-functions` (`DedupFunctionsPass`):** Merges structurally identical `func.func` defs.
7. **`freeze-inputs` (`UnifiedFreezeInputsPass`):** Binds `func.func` arguments to constants, or
   hoists constants into explicit function parameters when `unfreeze=true`.
8. **`tap-intermediates` (`TapIntermediatesPass`):** Appends intermediate SSA values to `return`.
9. **`reduce-ir` (`ReduceIrPass`):** Bugpoint-style delta IR reducer (`max_ops=1,keep_op=...`).
10. **`bisect-passes` (`BisectPassesPass`):** Stamps numbered bisection checkpoints.
11. **`sanitize-numerics` (`SanitizeNumericsPass`):** Injects `castforge.nan_guard` finite clamps.
12. **`normalize-ssa` (`NormalizeSsaPass`):** Canonical topological SSA renamer & attribute sorter.
13. **`hash-ir` (`HashIrPass`):** Stamps 64-bit FNV-1a structural hashes (`0x...`).
14. **`verify-equiv` (`VerifyEquivPass`):** Verifies structural/signature equivalence invariants.
15. **`instrument-profiling` (`InstrumentProfilingPass`):** Injects `castforge.profile_region`.
16. **`print-op-stats` (`PrintOpStatsPass`):** Computes op counts, FLOPs, and Roofline intensity.
17. **`strip-metadata` (`StripMetadataPass`):** Strips `Location` debug info & `castforge.*` attrs.

---

## 9. Polyhedral / Imperative LoopIR Passes (`castforge-codegen`, 7 Passes)

1. **`loop-interchange` (`LoopInterchangePass`)**
2. **`loop-unroll` (`LoopUnrollPass`)**
3. **`loop-vectorize` (`LoopVectorizePass`)**
4. **`fuse-loops` (`LoopFusionPass`)**
5. **`unroll-and-jam` (`LoopUnrollAndJamPass`)**
6. **`collapse-parallel` (`LoopCollapseParallelPass`)**
7. **`loop-fission` (`LoopFissionPass`)**
