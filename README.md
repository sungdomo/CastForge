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
- [**Complete Reference of All 25 Passes (`docs/PASSES.md`)**](docs/PASSES.md):
  Detailed explanation, CLI options, and IR transformation rules for all 25 Graph IR passes/presets
  (`22 passes + 3 presets O1/O2/O3`) and unified `OptimizeLoopIrPass` (`optimize-loop-ir`).

## Workspace Architecture

- [`crates/castforge-ir`](crates/castforge-ir): Arena-allocated MLIR-compatible IR (`Module`,
  `Operation`, `Value`, `Block`, `Region`, `Type`, `Attribute`, `Location`), structural verifier,
  binary `.safetensors` zero-copy reader/writer, and the `TensorStore` weight plane.
- [`crates/castforge-dialect-stablehlo`](crates/castforge-dialect-stablehlo): Strongly-typed
  `stablehlo` and `func` dialect views (`StableHloOp`, `DotDimensionNumbers`, `FuncView`) and
  shape verifiers.
- [`crates/castforge-mlir`](crates/castforge-mlir): Zero-C++ MLIR parser & printer supporting both
  **MLIR Generic Form** (`"stablehlo.add"(...)`) and **Pretty/Custom Syntax**
  (`stablehlo.add %0, %1 : tensor<...>`), `!quant.uniform` quantized tensor types,
  `dense_resource<key>` external weight references, plus symmetric **GGUF v2/v3**, **ONNX-JSON**,
  and **SafeTensors** model importers & exporters.
- [`crates/castforge-passes`](crates/castforge-passes): Unified `Pass`, `PassRegistry`, and
  `PassManager` with **25 canonical A-to-B IR translator modules & presets** (`22 passes + 3
  presets`) organized with **100% 1-to-1 pass-to-file mapping** (`22` `.rs` files) into 8 domain
  subdirectories (`general/`, `views/`, `weights/`, `fusion/`, `memory/`, `distributed/`, `loops/`,
  `tooling/`), supporting static Rust construction, CLI pipeline strings (`-p`), Unix pipes
  (`stdin -> stdout`), declarative TOML config files (`--config`), and step-by-step tracing.
- [`crates/castforge-viz`](crates/castforge-viz): Computation graph exporter for **Mermaid**,
  **Graphviz DOT**, **JSON**, unified **MLIR Snapshot Diffing**, and a self-contained **Interactive
  HTML Pass Timeline, Roofline & Arena Memory Liveness Debugger**.
- [`crates/castforge-codegen`](crates/castforge-codegen): Liveness analyzer (`LivenessAnalysis`),
  greedy interval-coloring arena buffer planner (`MemoryPlan`), progressive `LoopModule` IR with
  unified `OptimizeLoopIrPass`, pluggable `CodegenBackend` & `BackendRegistry`, multi-architecture
  SIMD profiles (`x86_64` AVX2/AVX-512, `aarch64` NEON/SVE, `riscv64` RVV, `wasm32` SIMD128),
  Roofline & native JIT schedule auto-tuner, in-memory reference IR interpreter, and 7 built-in
  codegen targets: **Rust**, **C99**, **WebGPU WGSL**, **NVIDIA CUDA (`.cu`)**, **Apple Metal
  (`.metal`)**, **AMD ROCm HIP (`.hip.cpp`)**, and **Khronos OpenCL (`.cl`)**, plus **C/C++ FFI
  Header (`.h`)** and **Native Shared Libraries (`.so`/`.dylib`)**.
- [`crates/castforge-cli`](crates/castforge-cli): The `castforge` command-line tool (`chain`,
  `opt`, `viz`, `codegen`, `autotune`, `import`, `profile`, `run`, `list-passes`, plus direct pass
  subcommands).

## Built-in A-to-B Translator Modules & Presets (25 Canonical Entries)

| Pass Name | Category | Description |
| :--- | :--- | :--- |
| `dce` | General IR | $O(N)$ ref-counted Dead Code Elimination (SSA ops & dead symbols) |
| `cse` | General IR | Unified CSE, dominator-aware GVN, LICM & entry constant hoisting |
| `canonicalize` | General IR | Fixed-point canonicalizer & `toposort` / `min_memory` scheduler |
| `simplify-algebraic` | General IR | Identities, constant folding, SCCP, reassociation & FMA |
| `fold-views` | Views & Shapes | Folds `transpose`, `reshape`, `broadcast`, `slice` & layout |
| `specialize-shapes` | Views & Shapes | Symbolic shape bind, batch resize, buckets & token prune |
| `quantize` | Weights | Quantize (`weight\|awq\|fp8\|nvfp4\|f16\|bf16\|w8a8\|smooth\|hadamard`) |
| `prune-weights` | Weights | Unified structured L1, `2:4` & dynamic `activation_sparse` pruning |
| `merge-weights` | Weights | Unified model merging & adapters (`linear\|slerp\|ties\|dare\|lora`) |
| `fold-weights` | Weights | Norm/BatchNorm folding, `[N,K]` SIMD packing & `externalize` |
| `fuse-elementwise` | Fusion | Composites, map-reduce, megakernel, loop chains & `decompose` |
| `fuse-gemm` | Fusion | Unified GEMM (`horizontal\|dequant\|epilogue` + constant weight concat) |
| `fuse-norm` | Fusion | Norm & Softmax (`rmsnorm\|residual_rmsnorm\|layernorm\|softmax\|ce`) |
| `fuse-activations` | Fusion | Unified Activation, Gated-MLP & routed/shared-expert MoE fusion |
| `fuse-attention` | Fusion | Attn (`sdpa\|rope\|mla\|kv_cache\|paged\|flash\|sliding\|share`) |
| `optimize-memory` | Memory | Unified memory (`mode=all\|inplace\|zero_copy\|remat\|offload`) |
| `shard-distributed` | Distributed | Unified SPMD TP, Ring CP, EP, ZeRO-3/FSDP, 1F1B PP & streams |
| `autodiff` | Distributed | Synthesizes reverse-mode VJP gradients & optional SGD step |
| `schedule-loops` | Loops | Unified tiling, permutation, unrolling, SIMD, conv & TensorCore |
| `ir-edit` | Tooling | Op rename/annotate, freeze/unfreeze, tap, SSA normalize & strip |
| `outline-subgraphs` | Tooling | Outlining, inlining, slice extract, delta reduce, dedup & unroll |
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
# 1. List all 29 registered A-to-B translator modules and pipeline presets
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
  -p "merge-weights(method=slerp,t=0.5),prune-weights(sparsity=0.5),quantize(bits=8)" \
  --format html -o examples/weight_surgery_trace.html

# 8. Lower the optimized model to standalone Rust, C99, WGSL, CUDA, or Metal
cargo run --bin castforge -- codegen examples/weight_surgery.mlir \
  -p "canonicalize,fuse-gemm" --target rust --tile-size 16

# 9. Profile Roofline FLOPs, memory bandwidth traffic, and arithmetic intensity
cargo run --bin castforge -- profile examples/llama_block.mlir -p "O3"
```
