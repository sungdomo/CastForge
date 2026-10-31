# CastForge

> **Zero-C++ MLIR & StableHLO compiler in pure Rust: 115+ graph optimization, fusion, quantization,
> and weight-surgery passes with standalone CPU/GPU codegen.**

**CastForge** is a self-contained AI compiler and model surgery engine written in pure Rust. It
combines an arena-allocated **MLIR / StableHLO compute graph** (`Module`) with a zero-copy **binary
weight store** (`TensorStore`), allowing you to import models, run composable optimization
pipelines, inspect transformations visually, and generate dependency-free CPU and GPU kernels.

## Highlights

- **Zero C++ / LLVM Build Tax**: Pure Rust implementation (`cargo build --release` in seconds).
- **Dual-Plane IR**: Separates symbolic graph topology (`Module`) from binary weights
  (`TensorStore`) with lazy `.safetensors` file indexing.
- **115 Composable Passes**: 9 domain packages covering algebraic simplification, shape
  specialization, weight merging/pruning, quantization, operator fusion, arena memory planning,
  distributed sharding, polyhedral loop scheduling, and IR verification.
- **Multi-Format Importers & Exporters**: Symmetric support for **MLIR / StableHLO**, **ONNX-JSON**,
  **GGUF v2/v3**, **SafeTensors**, and **HuggingFace `config.json`**.
- **7 Codegen Backends**: Emits standalone **Rust**, **C99**, **WebGPU (WGSL)**, **NVIDIA CUDA**,
  **Apple Metal (MSL)**, **AMD ROCm (HIP)**, and **OpenCL** kernels, plus C/C++ headers (`.h`) and
  native shared libraries (`.so` / `.dylib`).
- **Embeddable C ABI & Python SDK**: Link directly via [`include/castforge.h`](include/castforge.h)
  or call from Python via [`bindings/python/castforge.py`](bindings/python/castforge.py).

## Quick Start

```bash
# 1. Optimize a LLaMA Transformer block with preset O3 and verify numerical equivalence
cargo run --bin castforge -- opt examples/llama_block.mlir -p "O3" --verify-numerics

# 2. Compose stages via Unix pipes (stdin -> stdout)
cat examples/llama_block.mlir \
  | cargo run -q --bin castforge -- fuse-norm \
  | cargo run -q --bin castforge -- fuse-attention \
  | cargo run -q --bin castforge -- codegen -t rust

# 3. Run an in-process stage chain (import -> optimize -> lower -> emit code)
cargo run --bin castforge -- chain examples/llama_block.mlir \
  -s "fuse-norm | fuse-attention | schedule-loops(size=16,isa=avx2) | codegen-rust"

# 4. Auto-tune loop schedules against hardware Roofline models (h100 | a100 | m3-max)
cargo run --bin castforge -- autotune examples/llama_block.mlir -p "O3" --arch h100

# 5. Export an optimized model to GGUF v3, SafeTensors, or ONNX-JSON
cargo run --bin castforge -- export examples/llama_block.mlir -p "O2" --format gguf -o model.gguf

# 6. Generate an interactive HTML pass timeline and memory arena debugger
cargo run --bin castforge -- viz examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),quantize(bits=8)" \
  --format html -o examples/weight_surgery_trace.html

# 7. Browse all 115 passes and their composite -> leaf hierarchy
cargo run --bin castforge -- list-passes --tree
```

## Optimization Domains & Presets

CastForge organizes its 115 passes into 9 domain packages plus 3 optimization presets (`O1`, `O2`,
`O3`). See [**`docs/PASSES.md`**](docs/PASSES.md) for the complete pass catalog and CLI options.

| Domain | Key Passes | Purpose |
| :--- | :--- | :--- |
| **`canonicalize`** | `dce`, `cse`, `simplify-algebraic` | Algebraic folding, GVN/CSE, topo-sort |
| **`shapes`** | `fold-views`, `specialize-shapes` | View folding & dynamic shape binding |
| **`weights`** | `merge-weights`, `fold-weights`, `prune-weights` | LoRA, SLERP/TIES, 2:4 pruning |
| **`quantize`** | `quantize`, `recast-mixed-precision` | INT8/INT4/FP8/NF4/AWQ/GPTQ PTQ |
| **`fuse`** | `fuse-norm`, `fuse-attention`, `fuse-gemm` | RMSNorm, RoPE, SDPA, SwiGLU, epilogues |
| **`memory`** | `optimize-memory`, `rematerialize` | In-place buffers & checkpointing |
| **`distributed`** | `shard-distributed`, `autodiff` | SPMD sharding, ring attn, VJP autodiff |
| **`loops`** | `schedule-loops`, `lower-conv-to-gemm` | Tiling, interchange, unroll, SIMD |
| **`normalize`** | `verify-ir`, `reduce-ir`, `inline-calls` | Verification, bisection, IR reducer |
| **`registry`** | `O1`, `O2`, `O3` | Turnkey optimization pipelines |

## Supported Formats & Codegen Targets

| Category | Supported Formats & Targets |
| :--- | :--- |
| **Model Formats** | MLIR / StableHLO, ONNX-JSON, GGUF v2/v3, SafeTensors, HF `config.json` |
| **CPU Codegen** | **Rust** (`scalar`/`avx2`/`avx512`/`neon`/`sve`/`rvv`), **C99**, `.so` |
| **GPU Codegen** | **WebGPU WGSL**, **NVIDIA CUDA**, **Apple Metal**, **ROCm HIP**, **OpenCL** |
| **Visualizers** | HTML Timeline Debugger, Mermaid, Graphviz DOT, JSON, Roofline & Arena Reports |

## Repository Structure

```text
CastForge/
├── src/
│   ├── ir/             # MLIR/StableHLO AST, parser/printer, TensorStore, LoopModule
│   ├── passes/         # 9 pass packages + PassRegistry, PassManager, O1/O2/O3 presets
│   ├── codegen/        # Liveness arena planner, Roofline model, interpreter, 7 backends
│   ├── cli/            # CLI parser, stage chain runner, and inspection commands
│   └── ffi.rs          # Panic-safe C ABI (`extern "C"`)
├── include/            # C/C++ header (`castforge.h`)
├── bindings/python/    # Zero-dependency Python ctypes SDK (`castforge.py`)
├── config/             # Pass catalog (`passes.toml`) & hardware profiles (`targets.toml`)
├── docs/               # Architecture (`DESIGN.md`) and pass reference (`PASSES.md`)
├── examples/           # Sample `.mlir` workloads and pipeline configs
├── tests/              # 10 integration, property, fuzz, negative, and scaling test suites
└── benches/            # Compiler micro-benchmark suite (`compiler_bench.rs`)
```

## Documentation

- [**Architecture & Design (`docs/DESIGN.md`)**](docs/DESIGN.md): Dual-Plane IR, static memory
  planning, polyhedral Loop IR, and backend codegen architecture.
- [**Pass Reference (`docs/PASSES.md`)**](docs/PASSES.md): Detailed guide to all 115 composite and
  leaf optimization passes.
- [**Contributing (`CONTRIBUTING.md`)**](CONTRIBUTING.md) &
  [**Security Policy (`SECURITY.md`)**](SECURITY.md).
