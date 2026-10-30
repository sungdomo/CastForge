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
- [**Complete Reference of All 21 Passes (`docs/PASSES.md`)**](docs/PASSES.md):
  Detailed explanation, CLI options, and IR transformation rules for all 21 Graph IR passes/presets
  (`18 passes + 3 presets O1/O2/O3`) and unified `OptimizeLoopIrPass` (`optimize-loop-ir`).

## Project Directory Structure (`src/`)

- [`src/ir`](src/ir): Arena-allocated MLIR-compatible Dual-Plane IR (`Module`, `Operation`, `Value`,
  `Block`, `Region`, `Type`, `Attribute`, `Location`), core pass & pipeline infrastructure (`Pass`,
  `PassManager`, `PassRegistry`, `PipelineConfig` in `pass_manager.rs`), unified IR verification,
  structural hashing, equivalence, sanitization & Roofline profiling (`verify_ir/`), `TensorStore`
  weight plane & `.safetensors` zero-copy I/O (`tensor_store.rs`), strongly-typed `StableHloOp` and
  `FuncView` dialect views (`stablehlo.rs`), zero-C++ MLIR parser (`parser.rs`) & printer
  (`mlir.rs`), symmetric **GGUF v2/v3**, **ONNX-JSON**, and **SafeTensors** model importers &
  exporters (`import.rs`), and computation graph visualization exporters (**Mermaid**, **Graphviz
  DOT**, **JSON**, **IR Diff**, and **Interactive HTML Pass Timeline Debugger** in `viz.rs`).
- [`src/passes`](src/passes): Built-in `PassRegistry::with_builtin_passes()`
  (`src/passes/registry/`) and **21 canonical single-traversal A-to-B IR translator modules &
  presets** (`18 passes + 3 presets`), organized into modular pass packages (`src/passes/<pass>/`)
  and concise pass modules under `src/passes/`, supporting static Rust construction, CLI pipeline
  strings (`-p`), Unix pipes (`stdin -> stdout`), declarative TOML config files (`--config`), and
  step-by-step tracing.
- [`src/codegen`](src/codegen): Liveness analyzer (`LivenessAnalysis`), greedy interval-coloring
  arena buffer planner (`MemoryPlan`), progressive `LoopModule` IR with unified
  `OptimizeLoopIrPass`, pluggable `CodegenBackend` & `BackendRegistry`, multi-architecture SIMD
  profiles (`x86_64` AVX2/AVX-512, `aarch64` NEON/SVE, `riscv64` RVV, `wasm32` SIMD128), Roofline &
  native JIT schedule auto-tuner, in-memory reference IR interpreter, and 7 built-in codegen
  targets: **Rust**, **C99**, **WebGPU WGSL**, **NVIDIA CUDA (`.cu`)**, **Apple Metal (`.metal`)**,
  **AMD ROCm HIP (`.hip.cpp`)**, and **Khronos OpenCL (`.cl`)**, plus **C/C++ FFI Header (`.h`)**
  and **Native Shared Libraries (`.so`/`.dylib`)**.
- [`src/cli`](src/cli): The `castforge` command-line tool (`chain`, `opt`, `viz`, `codegen`,
  `autotune`, `import`, `profile`, `run`, `list-passes`, plus direct pass subcommands), invoked by
  [`src/main.rs`](src/main.rs) and exported alongside all modules in [`src/lib.rs`](src/lib.rs).
- [`tests/`](tests): End-to-end integration test suites (`ai_opt_tests.rs`, `codegen_tests.rs`,
  `pipeline_tests.rs`).

## Built-in A-to-B Translator Modules & Presets (21 Canonical Entries)

| Pass Name | Category | Description |
| :--- | :--- | :--- |
| `dce` | General IR | $O(N)$ ref-counted Dead Code Elimination (SSA ops & dead symbols) |
| `cse` | General IR | Unified CSE, dominator-aware GVN, LICM & entry constant hoisting |
| `canonicalize` | General IR | Fixed-point canonicalizer & `toposort` / `min_memory` scheduler |
| `simplify-algebraic` | General IR | Identities, constant folding, SCCP, reassociation & FMA |
| `fold-views` | Views & Shapes | Folds `transpose`, `reshape`, `broadcast`, `slice` & layout |
| `specialize-shapes` | Views & Shapes | Symbolic shape bind, batch resize, buckets & token prune |
| `quantize` | Weights | Quantize (`weight\|awq\|fp8\|nvfp4\|f16\|bf16\|w8a8\|smooth\|qat_ste`) |
| `merge-weights` | Weights | Model merging & adapters (`linear\|slerp\|ties\|dare\|lora\|dora`) |
| `fold-weights` | Weights | Norm/BatchNorm folding, `[N,K]` pack, `externalize` & `prune\|2to4` |
| `fuse-elementwise` | Fusion | Composites, activations/MoE, map-reduce, loops & `decompose` |
| `fuse-gemm` | Fusion | Unified GEMM (`horizontal\|dequant\|epilogue` + constant weight concat) |
| `fuse-norm` | Fusion | Norm & Softmax (`rmsnorm\|residual_rmsnorm\|layernorm\|softmax\|ce`) |
| `fuse-attention` | Fusion | Attn (`sdpa\|rope\|mla\|ssm\|kv_cache\|paged\|flash\|sliding`) |
| `optimize-memory` | Memory | Unified memory (`all\|inplace\|zero_copy\|remat\|selective_remat`) |
| `shard-distributed` | Distributed | TP, Ring CP, EP, ZeRO-3/FSDP, 1F1B PP, streams & `autodiff` |
| `schedule-loops` | Loops | Unified tiling, permutation, unrolling, SIMD, conv & `wgmma\|tcgen05` |
| `ir-edit` | Tooling | Op rename/annotate, freeze/unfreeze, tap, SSA normalize, outline & inline |
| `verify-ir` | Tooling | Shape/equiv verifier, FNV-1a hasher, NaN sanitizer, bisector & profiler |
| `O1` | Preset | Inlining, SCCP, LICM, view/algebraic canonicalization, DCE |
| `O2` | Preset | `O1` + LoRA/BatchNorm folding + Transformer/GEMM fusions |
| `O3` | Preset | Shape specialization + `O2` + MoE/FMA/zero-copy + weight packing |

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
# 1. List all 21 registered A-to-B translator passes and pipeline presets
cargo run --bin castforge -- list-passes

# 2. Concatenate simple A-to-B translators via standard Unix pipes (stdin -> stdout)
cat examples/llama_block.mlir \
  | cargo run -q --bin castforge -- fuse-norm \
  | cargo run -q --bin castforge -- fuse-attention \
  | cargo run -q --bin castforge -- schedule-loops(size=16,isa=avx2) \
  | cargo run -q --bin castforge -- codegen -t rust

# 3. Or run a concatenated TranslatorChain in-process via `castforge chain`
cargo run --bin castforge -- chain examples/llama_block.mlir \
  -s "fuse-norm | fuse-attention | schedule-loops(size=16,isa=avx2) | codegen-rust"

# 4. Optimize an end-to-end LLaMA block with automatic numerical verification
cargo run --bin castforge -- opt examples/llama_block.mlir -p "O3" --verify-numerics

# 5. Auto-tune polyhedral/HPC loop schedules (analytical Roofline or native CPU JIT)
cargo run --bin castforge -- autotune examples/llama_block.mlir -p "O3" --bench-jit

# 6. End-to-End Weight Surgery + SafeTensors binary export
cargo run --bin castforge -- opt examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),fold-weights(mode=externalize)" \
  --save-weights examples/merged_weights.safetensors

# 7. Generate an Interactive HTML Pass Timeline, Roofline & Arena Memory Debugger
cargo run --bin castforge -- viz examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),fold-weights(mode=prune,sparsity=0.5),quantize(bits=8)" \
  --format html -o examples/weight_surgery_trace.html

# 8. Lower the optimized model to standalone Rust, C99, WGSL, CUDA, or Metal
cargo run --bin castforge -- codegen examples/weight_surgery.mlir \
  -p "canonicalize,fuse-gemm" --target rust --tile-size 16

# 9. Profile Roofline FLOPs, memory bandwidth traffic, and arithmetic intensity
cargo run --bin castforge -- profile examples/llama_block.mlir -p "O3"
```
