# CastForge

**CastForge** is a pure-Rust AI Model IR Transformation, Optimization, Weight Surgery,
Visualization, and Code Generation toolset built around an MLIR-compatible Dual-Plane
Intermediate Representation (**Compute Graph + TensorStore**) with first-class **StableHLO**
support.

## Documentation

- [**System Architecture & Design Document (`docs/DESIGN.md`)**](docs/DESIGN.md): Deep dive into
  CastForge's Dual-Plane IR (`Module` + `TensorStore`), modular A-to-B `Translator` architecture,
  static arena memory planning, polyhedral `LoopModule` IR, multi-architecture SIMD profiles, and
  pluggable `CodegenBackend` registry.
- [**Complete Reference of All 91 Compiler Passes (`docs/PASSES.md`)**](docs/PASSES.md): Detailed
  explanation, CLI options, and IR transformation rules for all 91 Graph IR passes/presets and 7
  LoopIR passes.

## Workspace Architecture

- [`crates/castforge-ir`](crates/castforge-ir): Arena-allocated MLIR-compatible IR (`Module`,
  `Operation`, `Value`, `Block`, `Region`, `Type`, `Attribute`, `Location`), structural verifier,
  binary `.safetensors` zero-copy reader/writer, and the `TensorStore` weight plane (Linear/SLERP/
  TIES merging, L1 structured pruning, symmetric `i8`/`i4` quantization).
- [`crates/castforge-dialect-stablehlo`](crates/castforge-dialect-stablehlo): Strongly-typed
  `stablehlo` and `func` dialect views (`StableHloOp`, `DotDimensionNumbers`, `FuncView`) and
  shape verifiers.
- [`crates/castforge-mlir`](crates/castforge-mlir): Zero-C++ MLIR parser & printer supporting both
  **MLIR Generic Form** (`"stablehlo.add"(...)`) and **Pretty/Custom Syntax**
  (`stablehlo.add %0, %1 : tensor<...>`), `!quant.uniform` quantized tensor types,
  `dense_resource<key>` external weight references, plus symmetric **GGUF v2/v3**, **ONNX-JSON**,
  and **SafeTensors** model importers & exporters.
- [`crates/castforge-passes`](crates/castforge-passes): Unified `Pass`, `PassRegistry`, and
  `PassManager` with **91 built-in A-to-B IR translator modules & presets** supporting static Rust
  construction, CLI pipeline strings (`-p`, comma- or `|`-separated), Unix pipe (`stdin -> stdout`)
  invocation, declarative TOML config files (`--config`), and step-by-step pass tracing.
- [`crates/castforge-viz`](crates/castforge-viz): Computation graph exporter for **Mermaid**,
  **Graphviz DOT**, **JSON**, unified **MLIR Snapshot Diffing**, and a self-contained **Interactive
  HTML Pass Timeline, Roofline & Arena Memory Liveness Debugger**.
- [`crates/castforge-codegen`](crates/castforge-codegen): Liveness analyzer (`LivenessAnalysis`),
  greedy interval-coloring arena buffer planner (`MemoryPlan`), progressive `LoopModule` IR,
  pluggable `CodegenBackend` & `BackendRegistry`, multi-architecture SIMD profiles (`x86_64`
  AVX2/AVX-512, `aarch64` NEON/SVE, `riscv64` RVV, `wasm32` SIMD128), Roofline & native JIT
  schedule auto-tuner, in-memory reference IR interpreter, and 7 built-in codegen targets:
  **Rust**, **C99**, **WebGPU WGSL**, **NVIDIA CUDA (`.cu`)**, **Apple Metal (`.metal`)**, **AMD
  ROCm HIP (`.hip.cpp`)**, and **Khronos OpenCL (`.cl`)**, plus **C/C++ FFI Header (`.h`)** and
  **Native Shared Libraries (`.so`/`.dylib`)**.
- [`crates/castforge-cli`](crates/castforge-cli): The `castforge` command-line tool (`chain`,
  `opt`, `viz`, `codegen`, `autotune`, `import`, `profile`, `run`, `list-passes`, plus direct pass
  subcommands).

## Built-in A-to-B Translator Modules & Presets (91 Passes)

| Pass Name | Category | Description |
| :--- | :--- | :--- |
| `dce` | General IR | Dead Code Elimination (unused pure SSA ops & dead symbols) |
| `cse` | General IR | Common Subexpression Elimination (deduplicates pure ops) |
| `canonicalize` | General IR | Fixed-point algebraic, constant, layout, slice/concat, CSE, DCE |
| `inliner` | General IR | Inlines `func.call` ops by splicing callee bodies into callers |
| `symbol-dce` | General IR | Removes unreachable `private` `func.func` symbols |
| `sccp` | General IR | Sparse Conditional Constant Propagation (`compare`/`select`) |
| `licm` | General IR | Loop/Region-Invariant Code Motion out of nested regions |
| `hoist-constants` | General IR | Hoists `stablehlo.constant` ops to entry and deduplicates |
| `topological-sort` | General IR | Reorders operations inside blocks for SSA def-before-use |
| `strip-debuginfo` | General IR | Strips source location metadata (`loc(...)`) from all ops |
| `ir-edit` | General IR | Surgical IR editing (`rename_op=from:to`, `annotate_op=k:v`) |
| `specialize-shapes` | Shapes | Binds dynamic/symbolic dims (`?`, `B`, `S`) & infers shapes |
| `verify-shapes` | Diagnostics | Verifies StableHLO tensor shapes and rank consistency |
| `print-op-stats` | Diagnostics | Computes op counts, FLOPs, bytes, and Roofline intensity |
| `fold-constants` | Folding | Constant folding for arithmetic, N-D transpose, and shape ops |
| `simplify-algebraic` | Folding | Identities (`x+0`, `x-x`, `x*1`, `x/1`), reassociation, order |
| `fold-transpose` | Layout | Composes chained `transpose` ops and removes identity perms |
| `sink-transpose` | Layout | Sinks `transpose` through elementwise ops into constants |
| `fold-reshape` | Layout | Composes chained `reshape` ops and removes identity reshapes |
| `fold-broadcast` | Layout | Composes `broadcast_in_dim`, folds splats, hoists unary ops |
| `fold-slice-concat` | Layout | Composes nested `slice` ops and folds `slice(concatenate)` |
| `fold-convert` | Layout | Eliminates redundant/round-trip `convert` casts (`f16->f32->f16`) |
| `pack-weights` | Layout | Pre-packs 2D RHS weights `[K,N]->[N,K]` for unit-stride SIMD |
| `fuse-elementwise` | Fusion | Fuses `add` + `maximum` into `stablehlo.composite` |
| `fuse-loops` | Fusion | Fuses multi-op elementwise loop chains into 1 register loop |
| `fuse-map-reduce` | Fusion | Fuses elementwise map producers into `stablehlo.reduce` |
| `fission-loops` | Lowering | Splits oversized fused loop chains (`max_steps`) for registers |
| `fuse-matmul-epilogue` | Fusion | Fuses `dot_general` + bias + `relu`/`tanh`/`silu`/`gelu` |
| `fuse-dequant-matmul` | Fusion | Fuses `uniform_dequantize` + `dot_general` into qGEMM |
| `fuse-horizontal-dot` | Fusion | Horizontally fuses parallel `dot_general` ops (e.g. Q/K/V) |
| `fuse-gated-mlp` | Fusion | Fuses dual gate/up `dot_general` + `slice`s + SwiGLU/GeGLU |
| `fuse-softmax` | Fusion | Fuses `exp(x) / broadcast(reduce_sum(exp(x)))` into composite |
| `fuse-log-softmax` | Fusion | Fuses `log(softmax(x))` into online `fused_log_softmax` |
| `fuse-rmsnorm` | Fusion | Fuses `x * rsqrt(mean(x^2) + eps) * weight` into composite |
| `fuse-residual-norm` | Fusion | Fuses `add(x, residual)` + RMSNorm into `fused_residual_rmsnorm` |
| `fuse-layernorm` | Fusion | Fuses `(x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta` |
| `fuse-swiglu` | Fusion | Fuses SwiGLU (`gate / (1 + exp(-gate)) * up`) into composite |
| `fuse-gelu` | Fusion | Fuses Tanh-approx GELU and gated GeGLU (`gelu(gate) * up`) |
| `fuse-rope` | Fusion | Fuses Rotary Position Embedding (`RoPE`) into composite |
| `fuse-attention` | Fusion | Fuses SDPA & GQA (`softmax(Q @ K^T * scale) @ V`) |
| `lower-conv-to-gemm` | Lowering | Lowers 2D `convolution` (NHWC) to `im2col` + `dot_general` |
| `autodiff` | Training | Synthesizes reverse-mode VJP gradients & optional SGD step |
| `rematerialize` | Memory | Recomputes long-lived cheap activations to shrink arena memory |
| `shard-spmd` | Distributed | Megatron-style 1D SPMD tensor parallelism + `all_reduce` |
| `inject-kv-cache` | Inference | Injects `kv_cache_update` state ops before fused attention |
| `egraph-simplify` | Folding | Distributive factoring (`a*b+a*c -> a*(b+c)`) & FMA extraction |
| `mixed-precision` | Precision | Automatic Mixed Precision (`f16`/`bf16` GEMM + `f32` norms) |
| `fuse-moe` | Fusion | Fuses routed Mixture-of-Experts (MoE) expert GEMMs & gates |
| `calibrate-quantize` | Precision | Calibrates activation ranges & inserts static `W8A8` scales |
| `schedule-streams` | Concurrency | Assigns parallel branches to streams + `stream_wait` |
| `tile-loops` | Loop Schedule | Attaches L1/L2 cache & register tiling attrs to GEMM ops |
| `permute-loops` | Loop Schedule | Attaches loop order (`mnk`/`mkn`/`nkm`), collapse, or skew |
| `unroll-loops` | Loop Schedule | Attaches unroll, prefetch, split-k, & double-buffer attrs |
| `vectorize-loops` | Loop Schedule | Attaches SIMD lane width, ISA (`avx2`/`neon`), & parallel |
| `extract-subgraph` | IR Slicing | Slices backward dataflow cone (`target=%v,stop_at=%in`) |
| `outline-subgraphs` | IR Partition | Outlines GEMM/composite ops to `@outlined_kernel_k` |
| `reduce-ir` | IR Debugging | Delta-reduces function to minimal `max_ops` reproducer |
| `bisect-passes` | IR Debugging | Verifies invariants & stamps bisection step checkpoints |
| `sanitize-numerics` | IR Sanitizer | Injects `castforge.nan_guard` finite clamps on ops |
| `compose-funcs` | IR Linker | Chains two module functions `f2(f1(x))` into entry func |
| `freeze-inputs` | IR Surgery | Binds `func.func` arguments to in-body constants |
| `parameterize-constants` | IR Surgery | Hoists in-body weight constants to `func.func` args |
| `tap-intermediates` | IR Probe | Appends intermediate SSA values to `func.return` |
| `resize-batch` | IR Resizer | Rewrites leading batch dim while preserving 2D weights |
| `dedup-functions` | IR Dedup | Merges structurally identical `func.func` definitions |
| `normalize-ssa` | IR Canonical | Renames SSA values (`%arg0`, `%v0`) & sorts attributes |
| `hash-ir` | IR Hashing | Stamps 64-bit FNV-1a structural hash (`castforge.ir_hash`) |
| `verify-equiv` | IR Verifier | Verifies structural/signature invariants across chains |
| `sparsify-2to4` | Weight Surgery | Applies NVIDIA Ampere/Hopper 2:4 structured sparsity |
| `bucket-shapes` | IR Bucketing | Synthesizes multi-bucket shape-specialized functions |
| `convert-layout` | Layout | Canonicalizes 4D `convolution` between `NCHW` and `NHWC` |
| `offload-activations` | Memory | Inserts `castforge.host_offload` D2H/H2D on long-lived ops |
| `pad-for-tiling` | Loop Schedule | Computes tile-aligned `(M, K, N)` multiples (`align=8`) |
| `split-large-gemm` | IR Partition | Splits wide `dot_general` (`N > max_n`) into sub-GEMMs |
| `unroll-calls` | IR Unroller | Unrolls repeated `func.call` invocations (`repeat=N`) |
| `hoist-all-reduce` | Distributed | Coalesces linear `all_reduce` sums & enables overlap |
| `instrument-profiling` | IR Profiler | Wraps compute ops in `castforge.profile_region` timers |
| `strip-attributes` | IR Cleaner | Strips transient `castforge.*` & schedule attributes |
| `decompose-composites` | Lowering | Expands `stablehlo.composite` ops to primitive StableHLO |
| `bufferize-inplace` | Lowering | Annotates elementwise ops for in-place arena buffer reuse |
| `fold-norm-weights` | Weight Surgery | Absorbs RMSNorm/LayerNorm scale weights into MatMuls |
| `bake-lora` | Weight Surgery | Fuses LoRA branches `W + alpha * (A @ B)` into base weights |
| `task-arithmetic` | Weight Surgery | Applies multi-checkpoint task vectors & DARE pruning |
| `merge-weights` | Weight Surgery | Merges paired weights (`method=linear\|slerp\|ties,t=0.5`) |
| `prune-structured` | Weight Surgery | Structured L1 channel pruning on weights (`sparsity=0.5`) |
| `quantize-weights` | Weight Surgery | Symmetric `i8`/`i4` quantization + `uniform_dequantize` |
| `externalize-weights` | Weight I/O | Moves inline `dense<...>` to `dense_resource<key>` |
| `O1` | Preset | Inlining, SCCP, LICM, canonicalization, symbol-dce, toposort |
| `O2` | Preset | `O1` + LoRA/Norm folding + Transformer/MatMul operator fusions |
| `O3` / `llm-opt` | Preset | Shape specialization + `O2` + weight packing + inplace arena |

### Non-Pass Ecosystem A-to-B Translators (`TranslatorChain` / `castforge chain`)

| Translator Stage | Signature (`A -> B`) | Description |
| :--- | :--- | :--- |
| `parse-mlir` / `print-mlir` | `text <-> ir` | MLIR parser and printer |
| `import-onnx` / `export-onnx` | `text <-> ir` | Symmetric ONNX-JSON graph importer & exporter |
| `import-gguf` / `export-gguf` | `bytes <-> ir` | Symmetric GGUF v3 binary importer & exporter |
| `export-safetensors` | `ir -> bytes` | SafeTensors weight serializer |
| `lower-to-loops` | `ir -> text` | Progressive Tensor IR to `LoopModule` lowering |
| `autotune-schedule` | `ir -> ir` | Stamps winning Roofline schedule attrs onto IR |
| `codegen-rust` / `codegen-c` | `ir -> text` | Standalone Rust & C99 kernel emitter |
| `codegen-wgsl` / `codegen-cuda` | `ir -> text` | WebGPU WGSL & NVIDIA CUDA `.cu` GPU emitter |
| `codegen-msl` / `codegen-hip` | `ir -> text` | Apple Metal `.metal` & AMD ROCm `.hip.cpp` |
| `codegen-opencl` / `codegen-header` | `ir -> text` | OpenCL `.cl` & C/C++ `.h` FFI header |
| `compile-lib` | `ir -> bytes` | Compiles C99 kernel to native `.so`/`.dylib` |
| `snapshot-ir` / `ir-diff` | `ir -> ir` / `text` | Captures IR snapshot & emits unified MLIR diff |
| `roofline-report` / `memory-plan` | `ir -> text` | Emits JSON Roofline cost & arena memory plan |
| `viz-mermaid` / `viz-dot` / `viz-html` | `ir -> text` | Graph visualization & HTML debugger |

## Quick Start

```bash
# 1. List all 91 registered A-to-B translator modules and pipeline presets
cargo run --bin castforge -- list-passes

# 2. Concatenate simple A-to-B translators via standard Unix pipes (stdin -> stdout)
cat examples/llama_block.mlir \
  | cargo run -q --bin castforge -- fuse-rmsnorm \
  | cargo run -q --bin castforge -- fuse-attention \
  | cargo run -q --bin castforge -- tile-loops(size=16) \
  | cargo run -q --bin castforge -- vectorize-loops(isa=avx2) \
  | cargo run -q --bin castforge -- codegen -t rust

# 3. Or run a concatenated TranslatorChain in-process via `castforge chain`
cargo run --bin castforge -- chain examples/llama_block.mlir \
  -s "fuse-rmsnorm | fuse-attention | tile-loops(size=16) | codegen-rust"

# 4. Optimize an end-to-end LLaMA block with automatic numerical verification
cargo run --bin castforge -- opt examples/llama_block.mlir -p "O3" --verify-numerics

# 5. Auto-tune polyhedral/HPC loop schedules (analytical Roofline or native CPU JIT)
cargo run --bin castforge -- autotune examples/llama_block.mlir -p "O3" --bench-jit

# 6. End-to-End Weight Surgery + SafeTensors binary export
cargo run --bin castforge -- opt examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),externalize-weights" \
  --save-weights examples/merged_weights.safetensors

# 7. Generate an Interactive HTML Pass Timeline, Roofline & Arena Memory Debugger
cargo run --bin castforge -- viz examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),quantize-weights(bits=8)" \
  --format html -o examples/weight_surgery_trace.html

# 8. Lower the optimized model to standalone Rust, C99, WGSL, CUDA, or Metal
cargo run --bin castforge -- codegen examples/weight_surgery.mlir \
  -p "canonicalize,fuse-matmul-epilogue" --target rust --tile-size 16

# 9. Profile Roofline FLOPs, memory bandwidth traffic, and arithmetic intensity
cargo run --bin castforge -- profile examples/llama_block.mlir -p "O3"
```
