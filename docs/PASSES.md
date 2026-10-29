# CastForge — Complete Reference of All 99 Graph IR Passes (+ 7 LoopIR Passes)

CastForge provides **99 built-in modular A-to-B IR passes & presets** in `castforge-passes` (plus
**7 Imperative LoopIR passes** in `castforge-codegen`, totaling **106 passes**). Every pass
implements the unified `Pass` and `Translator` traits and can be invoked via:

- **CLI pipeline:** `castforge opt model.mlir -p "pass-name(opt1=val1,opt2=val2)"`
- **Translator chain:** `castforge chain model.mlir -s "pass-a | pass-b | lower-to-loops"`
- **Declarative config:** `pipeline.toml` or `pipeline.json`

---

## Summary by Category

| Category | Count | Section |
| :--- | :---: | :--- |
| **1. General MLIR & Algebraic Passes** | 20 | Section 1 |
| **2. Weight Surgery & Quantization** | 13 | Section 2 |
| **3. Transformer & Operator Fusions** | 19 | Section 3 |
| **4. Shape, Layout & GEMM Lowering** | 7 | Section 4 |
| **5. Memory, Rematerialization & Offload** | 4 | Section 5 |
| **6. Distributed SPMD, Streams & Autodiff** | 6 | Section 6 |
| **7. Graph Schedule & Loop-Chain Passes** | 6 | Section 7 |
| **8. IR Surgery, Verification & Tooling** | 20 | Section 8 |
| **9. Optimization Pipeline Presets** | 4 | Section 9 |
| **10. Polyhedral / Imperative LoopIR** | +7 | Section 10 |

*(Total: **99 registered Graph IR passes/presets** in `PassRegistry` + **7 LoopIR passes**.)*

---

## 1. General MLIR & Algebraic Optimization Passes (19 Passes)

### 1.1 `dce` (`DeadCodeEliminationPass`)
- **Options:** `symbols=true|false` (default: `true`)
- **Explanation:** Computes a single-pass $O(N)$ reference count table
  (`Module::compute_use_counts`) and sweeps blocks in reverse topological order, removing
  side-effect-free operations whose SSA results have zero uses and decrementing operand reference
  counts in $O(1)$. When `symbols=true`, also runs `symbol-dce`.

### 1.2 `cse` (`CommonSubexpressionEliminationPass`)
- **Options:** None
- **Explanation:** Deduplicates identical pure operations within each basic block. Canonicalizes
  operand order for commutative binary ops (`add`, `multiply`, `maximum`, `minimum`, `and`, `or`,
  `xor`) so `add(%a, %b)` and `add(%b, %a)` share a single result, and sorts non-transient
  attributes by key so attribute insertion order never prevents deduplication.

### 1.3 `canonicalize` (`CanonicalizePass`)
- **Options:** None
- **Explanation:** Composite fixed-point canonicalizer combining `fold-transpose`, `fold-reshape`,
  `cse`, and `dce`.

### 1.4 `simplify-algebraic` (`SimplifyAlgebraicPass`)
- **Options:** None
- **Explanation:** Applies algebraic identities (`x + 0 -> x`, `x - 0 -> x`, `x - x -> 0`,
  `x * 1 -> x`, `x * 0 -> 0`, `x / 1 -> x`, `max(x, x) -> x`, `min(x, x) -> x`), reassociates
  chained constant additions/multiplications (`(x + c1) + c2 -> x + (c1 + c2)`), and eliminates
  identity `convert` and full-range `slice` operations.

### 1.5 `egraph-simplify` (`EgraphSimplifyPass`)
- **Options:** `fuse_fma=true|false` (default: `true`)
- **Explanation:** Applies equality-saturation-inspired algebraic rewrites: factors distributive
  expressions (`a * b + a * c -> a * (b + c)`, reducing 3 ops to 2) and extracts fused
  multiply-add composites (`a * b + c -> castforge.fma(a, b, c)`).

### 1.6 `fold-constants` (`FoldConstantsPass`)
- **Options:** `conditional=true|false` (default: `true`)
- **Explanation:** Evaluates unary, binary, shape (`transpose`, `reshape`, `slice`, `concatenate`,
  `broadcast_in_dim`), and reduction operations whose operands are compile-time constants. When
  `conditional=true`, also evaluates constant `compare` and `select` branches.

### 1.7 `sccp` (`SccpPass`)
- **Options:** None
- **Explanation:** Sparse Conditional Constant Propagation (`fold-constants(conditional=true)`
  followed by `dce`), pruning dead conditional branches and propagating folded constants.

### 1.8 `fold-transpose` (`FoldTransposePass`)
- **Options:** `eliminate_identity=true|false` (default: `true`)
- **Explanation:** Composes chained `stablehlo.transpose` operations by multiplying their
  permutation vectors (`p_combined[i] = p1[p2[i]]`) and removes identity permutations
  (`[0, 1, ..., R-1]`).

### 1.9 `sink-transpose` (`SinkTransposePass`)
- **Options:** None
- **Explanation:** Pushes `stablehlo.transpose` backward through unary elementwise operations
  (`exponential`, `tanh`, `rsqrt`, `negate`, `abs`, `convert`) so transposes reach constant
  weights where `fold-constants` folds them at compile time.

### 1.10 `fold-reshape` (`FoldReshapePass`)
- **Options:** None
- **Explanation:** Collapses chains of `reshape(reshape(x))` into a single `reshape` and eliminates
  no-op reshapes where input and output tensor types are identical.

### 1.11 `fold-broadcast` (`FoldBroadcastPass`)
- **Options:** None
- **Explanation:** Composes nested `broadcast_in_dim` operations, eliminates identity broadcasts,
  and hoists unary elementwise operations before `broadcast_in_dim` so unary math executes on the
  smaller unbroadcasted tensor.

### 1.12 `fold-slice-concat` (`FoldSliceConcatPass`)
- **Options:** None
- **Explanation:** Composes nested `stablehlo.slice` operations (`start = s1 + s2 * stride1`) and
  simplifies `slice(concatenate(a, b, ...))` directly to the underlying source input when the
  slice window falls cleanly within one concatenated operand.

### 1.13 `fold-convert` (`FoldConvertPass`)
- **Options:** None
- **Explanation:** Eliminates identity `convert(x : T) -> T`, collapses narrowing-widening chains,
  and folds `convert` on `stablehlo.constant` tensors at compile time.

### 1.14 `hoist-constants` (`HoistConstantsPass`)
- **Options:** None
- **Explanation:** Hoists all `stablehlo.constant` operations to the top of their basic block and
  deduplicates identical constants within the block.

### 1.15 `licm` (`LicmPass`)
- **Options:** None
- **Explanation:** Loop/Region-Invariant Code Motion. Hoists pure operations inside nested regions
  whose operands are defined outside the region into the parent block before the region-holding op.

### 1.16 `inliner` (`InlinerPass`)
- **Options:** None
- **Explanation:** Inlines single-block `func.call` invocations by cloning the callee's operations
  into the caller block, remapping block arguments to call operands, and replacing call results.

### 1.17 `symbol-dce` (`SymbolDcePass`)
- **Options:** None
- **Explanation:** Performs call-graph reachability analysis from public entry functions and removes
  unreachable private `func.func` definitions.

### 1.18 `topological-sort` (`TopologicalSortPass`)
- **Options:** None
- **Explanation:** Reorders operations within each basic block using Kahn's dependency algorithm so
  every SSA value is defined before its uses while keeping terminators (`return`) at the end.

### 1.19 `verify-shapes` (`VerifyShapesPass`)
- **Options:** None
- **Explanation:** Verifies SSA dominance, operand/result counts, element types, and `dot_general`
  contracting/batching dimension compatibility across all operations.

### 1.20 `gvn` (`GlobalValueNumberingPass`)
- **Options:** None
- **Explanation:** Dominator-tree / cross-region Global Value Numbering (`GVN`). Propagates
  canonical value numbers from outer regions into nested regions and basic blocks, canonicalizes
  commutative binary operations, and eliminates redundant computations across basic blocks.

---

## 2. Weight Surgery & Quantization Passes (13 Passes)

### 2.1 `quantize-weights` (`QuantizeWeightsPass`)
- **Options:** `bits=8|4` (default: `8`), `min_elements=N` (default: `2`)
- **Explanation:** Performs symmetric per-tensor post-training quantization (`PTQ`) on `f32`
  weight constants into `!quant.uniform<i8:f32, scale:0>` (or `i4`), storing quantized `i8` bytes
  in `TensorStore` and inserting `stablehlo.uniform_dequantize`.

### 2.2 `calibrate-quantize` (`CalibrateQuantizePass`)
- **Options:** `bits=8|4` (default: `8`), `samples=N` (default: `4`)
- **Explanation:** Runs the reference interpreter over `samples` synthetic calibration inputs to
  measure activation dynamic ranges (`max(|x|)`), computes static quantization scales, and inserts
  `castforge.static_quantize` (`W8A8` / `W4A4`) nodes before GEMM operations.

### 2.3 `mixed-precision` (`MixedPrecisionPass`)
- **Options:** `dtype=f16|bf16` (default: `f16`)
- **Explanation:** Automatic Mixed Precision (`AMP`). Inserts narrowing `stablehlo.convert` casts
  to `f16` or `bf16` on `dot_general` and `convolution` inputs while maintaining `f32`
  accumulation (`precision = "f32_accum"`).

### 2.4 `prune-structured` (`StructuredPruningPass`)
- **Options:** `sparsity=0.0..1.0` (default: `0.5`)
- **Explanation:** Computes L1 column norms across 2D weight matrices in `dot_general`, prunes the
  lowest-importance columns according to `sparsity`, shrinks the GEMM output dimension, and inserts
  a reconstruction projection so downstream tensor shapes remain valid.

### 2.5 `sparsify-2to4` (`Sparsify2To4Pass`)
- **Options:** None
- **Explanation:** Applies NVIDIA Ampere/Hopper **2:4 fine-grained structured sparsity** to 2D
  constant weight matrices: within every group of 4 consecutive elements, zeros out the 2 smallest
  magnitudes, builds a 2-bit-per-element nibble mask, and attaches `castforge.sparse_2to4 = true`.

### 2.6 `merge-weights` (`MergeWeightsPass`)
- **Options:** `method=linear|slerp|ties` (default: `linear`), `t=0.0..1.0` (default: `0.5`)
- **Explanation:** Merges paired weight tensors (`*_a` and `*_b` or `add(mul(A, 1-t), mul(B, t))`)
  using Linear interpolation, Spherical Linear Interpolation (`SLERP`), or `TIES`-Merging.

### 2.7 `bake-lora` (`BakeLoraPass`)
- **Options:** `alpha=f32` (default: `1.0`)
- **Explanation:** Detects Low-Rank Adaptation (`LoRA`) subgraphs—either weight-space
  `W_base + alpha * (A @ B)` or parallel activation-space `dot(X, W) + dot(dot(X, A), B)`—and
  bakes the low-rank adapter directly into `W_base` at compile time, eliminating runtime LoRA
  overhead completely.

### 2.8 `task-arithmetic` (`TaskArithmeticPass`)
- **Options:** `lambda=f32` (default: `1.0`), `drop_rate=0.0..1.0` (default: `0.0`)
- **Explanation:** Fuses task-vector additions (`W_base + lambda * (W_ft - W_base)`) with optional
  **DARE** (Drop And REscale) deterministic magnitude-hashed delta pruning (`1 / (1 - p)` rescale).

### 2.9 `fold-norm-weights` (`FoldNormWeightsPass`)
- **Options:** None
- **Explanation:** Absorbs constant RMSNorm/LayerNorm scale vectors (`gamma`) directly into the
  rows of an immediately following `dot_general` weight matrix (`W'[k, n] = gamma[k] * W[k, n]`).

### 2.10 `pack-weights` (`PackWeightsPass`)
- **Options:** None
- **Explanation:** Transposes 2D constant RHS weight matrices in `dot_general`,
  `fused_matmul_epilogue`, `fused_qmatmul`, and `fused_gated_mlp` from row-major `[K, N]` to
  N-major `[N, K]` (`packed_rhs = true`), turning inner-loop $K$-reductions into unit-stride SIMD
  dot products.

### 2.11 `externalize-weights` (`ExternalizeWeightsPass`)
- **Options:** `min_elements=N` (default: `2`), `inline=true|false` (default: `false`)
- **Explanation:** Moves inline `dense<...>` weight literals with at least `min_elements` into
  `Module::tensor_store` as `dense_resource<w_N>` handles (or inlines them back when `inline=true`).

### 2.12 `quantize-groupwise` (`QuantizeGroupwisePass`)
- **Options:** `bits=4|8` (default: `4`), `group_size=N` (default: `32`), `awq_salient=true|false`
- **Explanation:** Group-wise / Activation-Aware Weight Quantization (`AWQ` / `GPTQ` style).
  Partitions 2D weight tensors into contiguous blocks of `group_size`, computes per-group symmetric
  quantization scales, preserves top salient channels at $2\times$ grid resolution when
  `awq_salient=true`, and stamps `castforge.quant_scheme = "awq_groupwise"`.

### 2.13 `fold-batchnorm` (`FoldBatchNormPass`)
- **Options:** None
- **Explanation:** Absorbs inference BatchNorm channel scaling (`multiply(conv_or_dot(X, W),
  scale)`) directly into the preceding `stablehlo.convolution` or `stablehlo.dot_general` constant
  filter/weight tensor (`W'[r, c] = W[r, c] * scale[c]`), eliminating runtime BatchNorm overhead.

---

## 3. Transformer & Operator Fusion Passes (19 Passes)

### 3.1 `fuse-elementwise` (`FuseElementwisePass`)
- **Options:** None
- **Explanation:** Fuses `add + maximum` (bias + ReLU) into `stablehlo.composite`
  (`castforge.fused_add_max`).

### 3.2 `fuse-matmul-epilogue` (`FuseMatmulEpiloguePass`)
- **Options:** None
- **Explanation:** Fuses `dot_general + add` (+ optional `relu`, `tanh`, `silu`, `gelu`, or
  `scale`) into `castforge.fused_matmul_epilogue` so bias and activation are applied in registers
  before writing the output tile to memory.

### 3.3 `fuse-dequant-matmul` (`FuseDequantMatmulPass`)
- **Options:** None
- **Explanation:** Fuses `uniform_dequantize(W_q) + dot_general` (and optional bias/activation
  epilogue) into `castforge.fused_qmatmul` (`W8A32` / `W4A32`), dequantizing `i8`/`i4` weights on
  the fly in registers without materializing an `f32` weight buffer.

### 3.4 `fuse-horizontal-dot` (`FuseHorizontalDotPass`)
- **Options:** None
- **Explanation:** Horizontally fuses 2 or more parallel `dot_general` ops sharing the same LHS
  activation (such as Transformer $Q, K, V$ or Gate/Up projections) into a single wide
  `dot_general` with concatenated constant weights plus `slice` views.

### 3.5 `fuse-softmax` (`FuseSoftmaxPass`)
- **Options:** None
- **Explanation:** Fuses decomposed `exp(x - max) / broadcast(reduce_sum(exp(x - max)))` subgraphs
  into `castforge.fused_softmax`, emitted as a numerically stable 3-pass online softmax loop.

### 3.6 `fuse-log-softmax` (`FuseLogSoftmaxPass`)
- **Options:** None
- **Explanation:** Fuses `log(softmax(x))` or `x - broadcast(log(reduce_sum(exp(x))))` into
  `castforge.fused_log_softmax` using the numerically stable log-sum-exp trick.

### 3.7 `fuse-rmsnorm` (`FuseRmsNormPass`)
- **Options:** None
- **Explanation:** Fuses `x * broadcast(rsqrt(mean(x^2) + eps)) * weight` into
  `castforge.fused_rmsnorm`.

### 3.8 `fuse-residual-norm` (`FuseResidualNormPass`)
- **Options:** None
- **Explanation:** Fuses residual stream addition `h = add(x, residual)` followed by `RMSNorm(h)`
  into `castforge.fused_residual_rmsnorm`.

### 3.9 `fuse-layernorm` (`FuseLayerNormPass`)
- **Options:** None
- **Explanation:** Fuses full mean-and-variance LayerNormalization
  `(x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta` into `castforge.fused_layernorm`.

### 3.10 `fuse-swiglu` (`FuseSwiGluPass`)
- **Options:** None
- **Explanation:** Fuses `gate * sigmoid(gate) * up` (`(gate / (1 + exp(-gate))) * up`) into
  `castforge.fused_swiglu`.

### 3.11 `fuse-gelu` (`FuseGeluPass`)
- **Options:** None
- **Explanation:** Fuses tanh-approximated Gaussian Error Linear Unit (`0.5 * x * (1 + tanh(...))`)
  into `castforge.fused_gelu`, and gated `gelu(gate) * up` into `castforge.fused_geglu`.

### 3.12 `fuse-gated-mlp` (`FuseGatedMlpPass`)
- **Options:** None
- **Explanation:** Fuses horizontally concatenated Gate+Up `dot_general` + `slice` +
  `fused_swiglu`/`fused_geglu` into a single `castforge.fused_gated_mlp` kernel that eliminates the
  intermediate wide `[M, 2N]` buffer.

### 3.13 `fuse-rope` (`FuseRopePass`)
- **Options:** None
- **Explanation:** Fuses Rotary Position Embedding (`RoPE`) slice/negate/concat/cos/sin subgraphs
  into `castforge.fused_rope`.

### 3.14 `fuse-attention` (`FuseAttentionPass`)
- **Options:** None
- **Explanation:** Fuses Scaled Dot-Product Attention (`Q @ K^T * scale (+ mask) -> softmax -> @ V`)
  and Grouped-Query Attention (`GQA`) head broadcasting into `castforge.fused_attention`
  (lowered to online FlashAttention-style tiled loops without materializing the $N \times N$ score
  matrix).

### 3.15 `fuse-moe` (`FuseMoePass`)
- **Options:** None
- **Explanation:** Fuses Mixture-of-Experts (`MoE`) dual expert GEMMs and router gate weighting
  (`dot(x, E0) * g0 + dot(x, E1) * g1`) into `castforge.fused_moe`.

### 3.16 `fuse-map-reduce` (`FuseMapReducePass`)
- **Options:** None
- **Explanation:** Fuses unary/square map operations (`x * x`, `exp(x)`, `tanh(x)`, `rsqrt(x)`)
  directly into a consuming `stablehlo.reduce` as `castforge.fused_map_reduce`.

### 3.17 `decompose-composites` (`DecomposeCompositesPass`)
- **Options:** None
- **Explanation:** Inverse lowering pass that expands `stablehlo.composite` ops (`fused_add_max`,
  `fused_matmul_epilogue`, `fused_softmax`, `fused_rmsnorm`) back into standard primitive
  StableHLO operations for portable export.

### 3.18 `fuse-norm` (`FuseNormPass`)
- **Options:** `kind=all|rmsnorm|residual_rmsnorm|layernorm` (default: `all`)
- **Explanation:** Consolidated normalization fusion engine combining `fuse-residual-norm`,
  `fuse-rmsnorm`, and `fuse-layernorm` with longest-match priority in a single pipeline stage.

### 3.19 `fuse-activations` (`FuseActivationsPass`)
- **Options:** `kind=all|swiglu|gelu|gated_mlp` (default: `all`)
- **Explanation:** Consolidated activation and gated-MLP fusion engine combining `fuse-swiglu`,
  `fuse-gelu` (GELU + GeGLU), and `fuse-gated-mlp` in bottom-up order.

---

## 4. Shape Specialization, Layout & GEMM Lowering Passes (7 Passes)

### 4.1 `specialize-shapes` (`SpecializeShapesPass`)
- **Options:** `B=val,S=val,default=val` (default: `1`)
- **Explanation:** Binds dynamic (`?`) and symbolic (`B`, `S`, `N`) tensor dimensions on function
  parameters and propagates concrete static shapes across all operations in the module.

### 4.2 `resize-batch` (`ResizeBatchPass`)
- **Options:** `batch=N` (default: `4`)
- **Explanation:** Rewrites the leading batch dimension (`dim 0`) of activation tensors and
  function signatures to `batch` while preserving 2D GEMM weight matrices (`[K, N]`) and updating
  `slice`/`broadcast_in_dim` bounds.

### 4.3 `bucket-shapes` (`BucketShapesPass`)
- **Options:** `buckets=1;4;8` (default: `"1;4;8"`)
- **Explanation:** Clones the entry function into specialized `@main_bucket_<B>` functions for
  each batch bucket in `buckets` with static shapes propagated per bucket.

### 4.4 `convert-layout` (`ConvertLayoutPass`)
- **Options:** `layout=nhwc|nchw|col_major|row_major` (default: `nhwc`)
- **Explanation:** Inserts boundary layout-conversion `stablehlo.transpose` operations around
  target compute ops and stamps `castforge.data_layout`.

### 4.5 `pad-for-tiling` (`PadForTilingPass`)
- **Options:** `align=N` (default: `8`)
- **Explanation:** Computes tile-aligned dimension multiples (`ceil(D / align) * align`) for GEMM
  `(M, K, N)` dimensions and attaches `castforge.padded_m/k/n` attributes to eliminate scalar
  remainder tail loops.

### 4.6 `split-large-gemm` (`SplitLargeGemmPass`)
- **Options:** `max_n=N` (default: `2`)
- **Explanation:** Splits wide `dot_general` GEMMs (`N > max_n`) along the output column dimension
  into `stablehlo.slice` weight chunks, narrower `dot_general` sub-ops, and a final
  `stablehlo.concatenate`.

### 4.7 `lower-conv-to-gemm` (`LowerConvToGemmPass`)
- **Options:** None
- **Explanation:** Lowers 2D `stablehlo.convolution` (NHWC/HWC) into an explicit `castforge.im2col`
  patch-extraction composite followed by a `stablehlo.dot_general` GEMM so convolutions inherit all
  GEMM tiling, packing, and SIMD optimizations.

---

## 5. Memory Planning, Rematerialization & Offloading Passes (3 Passes)

### 5.1 `bufferize-inplace` (`BufferizeInplacePass`)
- **Options:** None
- **Explanation:** Analyzes block liveness and annotates elementwise ops whose input buffer has its
  final use at that op (`last_use == step`) with `inplace_operand = <idx>` for zero-copy in-place
  arena slot reuse.

### 5.2 `rematerialize` (`RematerializePass`)
- **Options:** `min_distance=N` (default: `3`)
- **Explanation:** Gradient/activation checkpointing. Clones cheap unary elementwise producers
  (`exponential`, `tanh`, `rsqrt`, `negate`, `abs`, `convert`, `reshape`) right before distant
  consumers (`distance >= min_distance`) so the original activation buffer dies early.

### 5.3 `offload-activations` (`OffloadActivationsPass`)
- **Options:** `min_span=N` (default: `2`)
- **Explanation:** Inserts paired `castforge.offload_to_host` (D2H DMA) and
  `castforge.prefetch_to_device` (H2D DMA) custom calls around activations whose live range exceeds
  `min_span` instructions.

### 5.4 `eliminate-concat-copies` (`EliminateConcatCopiesPass`)
- **Options:** None
- **Explanation:** Destination-passing zero-copy concatenation optimization. Annotates pure
  producer operands of `stablehlo.concatenate` with `castforge.concat_dest` and
  `castforge.concat_elem_offset` so producers write directly into destination sub-slices.

---

## 6. Distributed SPMD, Async Streams & Autodiff Passes (6 Passes)

### 6.1 `shard-spmd` (`ShardSpmdPass`)
- **Options:** `devices=N` (default: `2`)
- **Explanation:** Megatron-LM style Tensor Parallelism (`TP`). Annotates GEMM operations with
  column/row parallel partitioning across `devices` ranks and inserts a cross-replica
  `stablehlo.all_reduce` synchronization op.

### 6.2 `hoist-all-reduce` (`HoistAllReducePass`)
- **Options:** None
- **Explanation:** Distributes `add(all_reduce(x), all_reduce(y)) -> all_reduce(add(x, y))` to
  halve collective communication calls, collapses idempotent `all_reduce(all_reduce(x))`, and marks
  remaining collectives for async compute overlap (`castforge.async_collective = "overlap"`).

### 6.3 `schedule-streams` (`ScheduleStreamsPass`)
- **Options:** `streams=N` (default: `2`)
- **Explanation:** Assigns independent dataflow branches to concurrent hardware execution streams
  (`stream_id = 0..N-1`) and attaches `stream_wait` synchronization attributes at join nodes.

### 6.4 `inject-kv-cache` (`InjectKvCachePass`)
- **Options:** `max_seq_len=N` (default: `2048`)
- **Explanation:** Injects `castforge.kv_cache_update` composites on $K$ and $V$ inputs of
  `fused_attention` for stateful autoregressive LLM token decoding.

### 6.5 `lower-paged-attention` (`LowerPagedAttentionPass`)
- **Options:** `page_block_size=N` (default: `16`), `max_blocks=N` (default: `64`)
- **Explanation:** Lowers `castforge.fused_attention` into vLLM-style **PagedAttention** with
  non-contiguous block-table KV addressing (`castforge.paged_attention = true`).

### 6.6 `autodiff` (`AutodiffPass`)
- **Options:** `lr=f32` (default: `0.0`)
- **Explanation:** Reverse-mode Automatic Differentiation (`VJP`). Synthesizes backward adjoint
  operations (`dL/dX`, `dL/dW`) in StableHLO and, when `lr > 0`, emits in-graph Stochastic Gradient
  Descent (`W_new = W - lr * dW`) updates.

---

## 7. Graph-Level Schedule & Loop-Chain Passes (6 Passes)

### 7.1 `tile-loops` (`TileLoopsPass`)
- **Options:** `size=N` (default: `16`), `tile_m`, `tile_n`, `tile_k`, `l2_size`, `reg_m`, `reg_n`
- **Explanation:** Attaches hierarchical L1 (`tile_m/n/k`), L2 (`l2_tile`), and register
  microkernel (`reg_m/n`) blocking attributes to GEMM operations.

### 7.2 `permute-loops` (`PermuteLoopsPass`)
- **Options:** `order=mnk|mkn|nkm|knm` (default: `mkn`), `collapse=bool`, `skew=bool`
- **Explanation:** Attaches loop interchange (`loop_order`), outer-loop collapsing
  (`collapse_loops`), and polyhedral wavefront skewing (`polyhedral_skew`) attributes.

### 7.3 `unroll-loops` (`UnrollLoopsPass`)
- **Options:** `factor=N` (default: `4`), `prefetch=N`, `split_k=N`, `double_buffer=bool`
- **Explanation:** Attaches inner-loop unrolling (`unroll_factor`), software prefetch distance
  (`prefetch_distance`), Split-K reduction parallelism (`split_k`), and shared-memory ping-pong
  double buffering (`double_buffer`) attributes.

### 7.4 `vectorize-loops` (`VectorizeLoopsPass`)
- **Options:** `isa=scalar|avx2|avx512|neon|sve|rvv|wasm`, `width=N`, `parallel=bool`
- **Explanation:** Attaches explicit SIMD lane width (`vector_width`), target vector ISA
  (`simd_isa`), and multi-core parallelization (`parallel_loops`) attributes.

### 7.5 `fuse-loops` (`FuseLoopsPass`)
- **Options:** None
- **Explanation:** Fuses arbitrary-length chains of elementwise unary/binary operations into a
  single register-pipelined `castforge.fused_loop_chain` composite (`steps = "add(0,1);tanh(s0)"`).

### 7.6 `fission-loops` (`FissionLoopsPass`)
- **Options:** `max_steps=N` (default: `2`)
- **Explanation:** Splits overly long `castforge.fused_loop_chain` composites (`steps > max_steps`)
  into balanced sub-chains to prevent register spilling.

---

## 8. IR Surgery, Verification, Profiling & Tooling Passes (19 Passes)

### 8.1 `ir-edit` (`IrEditPass`)
- **Options:** `rename_op=old:new`, `annotate_op=op:key:val`
- **Explanation:** Surgical pattern-based operation renaming and attribute injection from the CLI.

### 8.2 `extract-subgraph` (`ExtractSubgraphPass`)
- **Options:** `target=%val`, `stop_at=%in1;%in2`
- **Explanation:** Extracts the backward dataflow slice producing `target` (stopping at cut-set
  values `stop_at`) into a standalone function.

### 8.3 `outline-subgraphs` (`OutlineSubgraphsPass`)
- **Options:** `mode=all|composites|gemm` (default: `all`)
- **Explanation:** Outlines matching operations into private `@outlined_<op>_<idx>` functions and
  replaces the inline ops with `func.call` sites.

### 8.4 `unroll-calls` (`UnrollCallsPass`)
- **Options:** `repeat=N` (default: `2`)
- **Explanation:** Replicates `func.call` invocations `repeat` times in a sequential chain (when
  domain and codomain match) or parallel fan-out + average combine.

### 8.5 `compose-funcs` (`ComposeFuncsPass`)
- **Options:** `first=@f1`, `second=@f2`, `entry=composed_main`
- **Explanation:** Synthesizes a pipeline wrapper function `@composed_main(%x) { %t = call @f1(%x);
  %y = call @f2(%t); return %y }`.

### 8.6 `dedup-functions` (`DedupFunctionsPass`)
- **Options:** None
- **Explanation:** Computes canonical structural signatures of all `func.func` definitions,
  eliminates duplicate functions, and rewrites `func.call` callees to the canonical function.

### 8.7 `freeze-inputs` (`FreezeInputsPass`)
- **Options:** `args=0;1`, `value=f64` (default: `0.5`)
- **Explanation:** Binds selected `func.func` parameters to in-body `stablehlo.constant` tensors
  and removes the frozen arguments from the function signature.

### 8.8 `parameterize-constants` (`ParameterizeConstantsPass`)
- **Options:** `min_elements=N` (default: `2`)
- **Explanation:** Inverse of `freeze-inputs`: hoists in-body `stablehlo.constant` weights into
  explicit `%w_param_N` function arguments so weights can be supplied externally at runtime.

### 8.9 `tap-intermediates` (`TapIntermediatesPass`)
- **Options:** `values=%v1;%v2|all`
- **Explanation:** Appends intermediate SSA activations to `func.return` and updates the `func.func`
  result signature so hidden layer activations can be inspected directly.

### 8.10 `reduce-ir` (`ReduceIrPass`)
- **Options:** `max_ops=N` (default: `1`), `keep_op=pattern`
- **Explanation:** Automated `llvm-reduce` / Bugpoint-style delta reducer that shrinks a function
  body down to at most `max_ops` operations while preserving `keep_op` and valid SSA types.

### 8.11 `bisect-passes` (`BisectPassesPass`)
- **Options:** `limit=N` (default: `1`)
- **Explanation:** Verifies module & shape invariants at the current pipeline point and increments
  a `castforge.bisect_step` checkpoint counter on the entry function.

### 8.12 `sanitize-numerics` (`SanitizeNumericsPass`)
- **Options:** `clamp_max=f64` (default: `65504.0`), `eps=f64` (default: `1e-6`)
- **Explanation:** Injects `castforge.nan_guard` composites after numerically sensitive operations
  (`divide`, `rsqrt`, `exponential`, `log`, `fused_softmax`, `fused_rmsnorm`) to scrub `NaN`/`Inf`.

### 8.13 `normalize-ssa` (`NormalizeSsaPass`)
- **Options:** `prefix=v` (default: `"v"`)
- **Explanation:** Renames all SSA values in deterministic topological order (`%arg0..`, `%v0..`)
  and sorts operation attributes alphabetically for canonical diffing.

### 8.14 `hash-ir` (`HashIrPass`)
- **Options:** None
- **Explanation:** Computes deterministic 64-bit FNV-1a structural hashes of every `func.func` and
  stamps `castforge.ir_hash = "0x..."`.

### 8.15 `verify-equiv` (`VerifyEquivPass`)
- **Options:** `tol=f64` (default: `1e-4`)
- **Explanation:** Verifies structural and numerical equivalence invariants across pass chains and
  stamps `castforge.verified_equiv = true`.

### 8.16 `instrument-profiling` (`InstrumentProfilingPass`)
- **Options:** `filter=gemm|composites|all` (default: `gemm`)
- **Explanation:** Injects `stablehlo.custom_call @castforge.profile_trace` timing probes after
  matching compute operations, lowered to `std::time::Instant` / `clock_gettime` traces in codegen.

### 8.17 `print-op-stats` (`PrintOpStatsPass`)
- **Options:** `ridge=f64` (default: `10.0`)
- **Explanation:** Computes op histogram, parameter memory footprint, FLOPs, and Roofline
  arithmetic intensity (`ComputeBound` vs. `MemoryBound`).

### 8.18 `strip-debuginfo` (`StripDebugInfoPass`)
- **Options:** None
- **Explanation:** Resets all source `Location` metadata to `Location::Unknown` for clean MLIR
  output.

### 8.19 `strip-attributes` (`StripAttributesPass`)
- **Options:** None
- **Explanation:** Strips all compiler-internal `castforge.*` and schedule attributes while
  preserving standard MLIR and StableHLO semantics for strict upstream portability.

### 8.20 `strip-metadata` (`StripMetadataPass`)
- **Options:** `debuginfo=true|false` (default: `true`), `attributes=true|false` (default: `true`)
- **Explanation:** Unified pass combining `strip-debuginfo` and `strip-attributes` in a single
  stage.

---

## 9. Turnkey Optimization Pipeline Presets (4 Presets)

- **`O1` (`OptO1Pass`):** Inlining (`inliner`), Sparse Conditional Constant Propagation (`sccp`),
  Loop-Invariant Code Motion (`licm`), fixed-point algebraic/shape canonicalization
  (`simplify-algebraic`, `fold-transpose`, `fold-reshape`, `fold-broadcast`, `fold-slice-concat`,
  `fold-convert`, `hoist-constants`, `cse`, `dce`), `symbol-dce`, and `topological-sort`.
- **`O2` (`OptO2Pass`):** Runs `O1`, weight-level fusions (`bake-lora`, `task-arithmetic`,
  `sink-transpose`, `fold-constants`, `fold-norm-weights`, `fuse-horizontal-dot`), and full
  Transformer/MatMul operator fusions (`fuse-rope`, `fuse-attention`, `fuse-residual-norm`,
  `fuse-rmsnorm`, `fuse-layernorm`, `fuse-softmax`, `fuse-log-softmax`, `fuse-map-reduce`,
  `fuse-gated-mlp`, `fuse-swiglu`, `fuse-gelu`, `fuse-dequant-matmul`, `fuse-matmul-epilogue`,
  `fuse-elementwise`), followed by `O1` cleanup.
- **`O3` / `llm-opt` (`OptO3Pass`):** Runs `specialize-shapes`, `O2`, `pack-weights` (`[K, N] ->
  [N, K]` unit-stride RHS packing), `fuse-loops` (register-pipelined elementwise loop fusion),
  `bufferize-inplace` (zero-copy arena buffer donation), and final `dce` + `topological-sort`.

---

## 10. Polyhedral / Imperative LoopIR Passes (`castforge-codegen`, 7 Passes)

These 7 passes operate directly on `castforge_codegen::LoopModule` (`loop-ir -> loop-ir`
translators):

1. **`loop-interchange` (`LoopInterchangePass`):** Reorders nested `i, j, k`GEMM loops into
   cache-friendly `i, k, j` (`mkn`) order so inner-loop array accesses have unit stride.
2. **`loop-unroll` (`LoopUnrollPass`):** Attaches `LoopAnnotation::Unroll(factor)` to innermost
   reduction and elementwise loops.
3. **`loop-vectorize` (`LoopVectorizePass`):** Attaches `LoopAnnotation::Vectorize(width)` and
   `LoopAnnotation::SimdIsa(isa)` to innermost unit-stride loops.
4. **`fuse-loops` (`LoopFusionPass`):** Fuses adjacent `LoopStmt::For` loops with identical bounds
   `[lower, upper, step]` into a single loop body.
5. **`unroll-and-jam` (`LoopUnrollAndJamPass`):** Unrolls an outer spatial loop and jams the
   duplicated inner $K$-reduction statements together to maximize register reuse.
6. **`collapse-parallel` (`LoopCollapseParallelPass`):** Collapses perfectly nested parallel outer
   loops (`for i in 0..M { for j in 0..N }`) into a single flattened parallel loop (`0..M*N`).
7. **`loop-fission` (`LoopFissionPass`):** Distributes a multi-statement loop body into separate
   loops when independent statement groups exceed the register pressure threshold.
