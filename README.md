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
- [**Complete Reference of All 35 Passes (`docs/PASSES.md`)**](docs/PASSES.md):
  Detailed explanation, CLI options, and IR transformation rules for all 35 Graph IR passes/presets
  (`32 passes + 3 presets O1/O2/O3`) and unified `OptimizeLoopIrPass` (`optimize-loop-ir`).

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
  `PassManager` with **35 canonical A-to-B IR translator modules & presets** organized with strict
  1-to-1 source-file mapping into 8 domain subdirectories (`general/`, `views/`, `weights/`,
  `fusion/`, `memory/`, `distributed/`, `loops/`, `tooling/`), supporting static Rust construction,
  CLI pipeline strings (`-p`), Unix pipes (`stdin -> stdout`), declarative TOML config files
  (`--config`), and step-by-step tracing.
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

## Built-in A-to-B Translator Modules & Presets (35 Canonical Passes)

| Pass Name | Category | Description |
| :--- | :--- | :--- |
| `dce` | General IR | $O(N)$ ref-counted Dead Code Elimination (SSA ops & dead symbols) |
| `cse` | General IR | Unified CSE, dominator-aware GVN, and entry constant hoisting |
| `canonicalize` | General IR | Fixed-point algebraic, constant, view folding, CSE, and DCE |
| `inliner` | General IR | Inlines `func.call` ops by splicing callee bodies into callers |
| `licm` | General IR | Loop/Region-Invariant Code Motion out of nested regions |
| `topological-sort` | General IR | Def-before-use & peak-memory-minimizing (`min_memory`) sort |
| `fold-constants` | General IR | Constant folding & conditional `compare`/`select` SCCP |
| `simplify-algebraic` | General IR | Algebraic identities, reassociation & e-graph FMA/factor |
| `fold-views` | Views & Shapes | Folds `transpose`, `reshape`, `broadcast`, `slice` & layout |
| `specialize-shapes` | Views & Shapes | Symbolic shape binding, batch resize & bucketing |
| `quantize` | Weights | Quantize (`weight\|awq\|fp8\|nvfp4\|f16\|bf16\|w8a8\|smooth\|hadamard`) |
| `prune-weights` | Weights | Unified structured L1, `2:4` & dynamic `activation_sparse` pruning |
| `merge-weights` | Weights | Unified model merging & adapters (`linear\|slerp\|ties\|dare\|lora`) |
| `fold-weights` | Weights | Unified pre-GEMM Norm/BatchNorm scale folding & `[N,K]` SIMD packing |
| `externalize-weights` | Weights | Moves inline `dense<...>` to `dense_resource<key>` |
| `fuse-elementwise` | Fusion | Unified elementwise composites, map-reduce, megakernel & loops |
| `fuse-gemm` | Fusion | Unified GEMM (`horizontal\|dequant\|epilogue` + chained residual) |
| `fuse-norm` | Fusion | Unified Norm fusion (`kind=all\|rmsnorm\|residual_rmsnorm\|layernorm`) |
| `fuse-activations` | Fusion | Unified Activation, Gated-MLP & routed/shared-expert MoE fusion |
| `fuse-softmax` | Fusion | Unified Softmax, LogSoftmax & chunked Linear Cross-Entropy |
| `fuse-attention` | Fusion | Attn (`sdpa\|rope\|mla\|kv_cache\|paged\|flash\|sliding\|share`) |
| `decompose-composites` | Fusion | Expands `stablehlo.composite` ops to primitive StableHLO |
| `optimize-memory` | Memory | Unified memory (`mode=all\|inplace\|zero_copy\|remat\|offload`) |
| `shard-distributed` | Distributed | Unified SPMD TP, Ring CP, `all_reduce`, 1F1B PP & streams |
| `autodiff` | Distributed | Synthesizes reverse-mode VJP gradients & optional SGD step |
| `schedule-loops` | Loops | Unified tiling, permutation, unrolling, SIMD, conv & TensorCore |
| `ir-edit` | Tooling | Unified op rename/annotate, input freeze/unfreeze & value tapping |
| `extract-subgraph` | Tooling | Backward dataflow cone slicer & `max_ops` delta IR reducer |
| `outline-subgraphs` | Tooling | Unified outlining, deduplication, composition & unrolling |
| `normalize-ir` | Tooling | Canonical SSA renaming (`ssa=true`) & metadata stripping |
| `verify-ir` | Tooling | Unified shape/equiv verifier, FNV-1a hasher, NaN sanitizer & bisector |
| `profile-ir` | Tooling | Unified Roofline FLOPs/intensity analysis & profiling timers |
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
# 1. List all 35 registered A-to-B translator modules and pipeline presets
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
  -p "merge-weights(method=slerp,t=0.5),prune-weights(sparsity=0.5),externalize-weights" \
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
