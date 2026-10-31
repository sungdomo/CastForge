# CastForge — System Architecture & Design

**CastForge** is a zero-C++ MLIR and StableHLO compiler, weight surgery engine, polyhedral loop
optimizer, and multi-target code generator written in pure Rust.

---

## 1. Architectural Principles

1. **Dual-Plane Intermediate Representation (`Module` + `TensorStore`)**
   CastForge separates symbolic graph topology from bulk model parameters inside
   [`castforge::ir::Module`](../src/ir/mlir.rs):
   - **Compute Graph Plane**: Arena-allocated (`slotmap::SlotMap`) SSA operations (`OpId`), values
     (`ValueId`), basic blocks (`BlockId`), and regions (`RegionId`).
   - **Weight Plane (`TensorStore`)**: Dedicated storage holding `f32` buffers, quantized `i8`
     tensors (`TensorBacking::QuantizedI8`), or lazy zero-copy `.safetensors` file slices
     (`TensorBacking::ExternalFile`).

2. **Two-Tier Compilation (`Module` -> `LoopModule` -> Target Code)**
   - **Graph Level (`Module`)**: Algebraic simplification, shape specialization, weight surgery,
     quantization, operator fusion, SPMD sharding, and reverse-mode autodiff run on the SSA graph.
   - **Loop Level (`LoopModule`)**: Lowered tensor ops become imperative loop nests
     (`LoopStmt::For`) where polyhedral tiling, loop interchange, unroll-and-jam, fusion/fission,
     and SIMD vectorization run before target code emission.

3. **Composable Pipeline Stages (`Translator` & `PassManager`)**
   Every importer, pass, lowering step, visualizer, and code generator implements the unified
   [`Translator`](../src/ir/translator.rs) interface over `Payload` (`Text`, `Bytes`, `Ir`),
   allowing stages to be composed in-memory (`TranslatorChain`) or across Unix pipes.

---

## 2. End-to-End Compilation Pipeline

```text
  +--------------------+   +-------------------+   +-------------------+
  |  Model Frontends   |   |   Weight Shards   |   |  StableHLO MLIR   |
  | ONNX / GGUF / HF   |   |   (.safetensors)  |   |     (.mlir)       |
  +---------+----------+   +---------+---------+   +---------+---------+
            |                        |                       |
            +------------------------+-----------------------+
                                     |
                                     v
                     +-------------------------------+
                     |     castforge::ir::Module     |
                     |  - SSA Compute Graph (SlotMap)|
                     |  - Weight Plane (TensorStore) |
                     +---------------+---------------+
                                     |
               +---------------------+---------------------+
               |     9 Optimization Packages (115 Passes)  |
               |  canonicalize | shapes | weights          |
               |  quantize     | fuse   | memory           |
               |  distributed  | loops  | normalize        |
               +---------------------+---------------------+
                                     |
                                     v
                     +-------------------------------+
                     | Static Liveness & Arena Plan  |
                     |  - compute_module_liveness()  |
                     |  - build_memory_plan()        |
                     +---------------+---------------+
                                     |
                                     v
                     +-------------------------------+
                     |   castforge::ir::LoopModule   |
                     |  - OptimizeLoopIrPass         |
                     |    (tile, interchange, unroll,|
                     |     vectorize, jam, fission)  |
                     +---------------+---------------+
                                     |
                                     v
                     +-------------------------------+
                     |       BackendRegistry         |
                     |  - CPU: Rust, C99 (7 ISAs)    |
                     |  - GPU: WGSL, CUDA, Metal,    |
                     |         ROCm HIP, OpenCL      |
                     +-------------------------------+
```

---

## 3. Codebase Organization

- **Core IR ([`src/ir/`](../src/ir))**: `Module` arena (`mlir.rs`), `StableHloOp` dialect views
  (`stablehlo.rs`), MLIR parser/printer (`parser.rs`, `printer.rs`), `TensorStore`
  (`tensor_store.rs`), `LoopModule` (`loops.rs`), visualizers (`viz.rs`), and
  ONNX/GGUF/SafeTensors/HF formats (`formats/`).
- **Passes ([`src/passes/`](../src/passes))**: `PassRegistry`, `PassManager`, structured
  `PassTelemetry`, `O1`/`O2`/`O3` presets (`registry/`), and 9 domain pass packages
  (`canonicalize`, `shapes`, `weights`, `quantize`, `fuse`, `memory`, `distributed`, `loops`,
  `normalize`).
- **Codegen ([`src/codegen/`](../src/codegen))**: Liveness & interval-coloring arena planner
  (`memory_plan.rs`), Roofline cost model (`cost_model.rs`), reference interpreter (`eval.rs`),
  schedule autotuner (`autotune.rs`), and 7 target backends (`targets/`).
- **CLI ([`src/cli/`](../src/cli))**: Command-line interface (`opt`, `chain`, `codegen`,
  `autotune`, `viz`, `profile`, `import`, `export`, `link`, `run`, `list-passes`).
- **FFI & SDK ([`src/ffi.rs`](../src/ffi.rs))**: Panic-safe `extern "C"` ABI
  (`include/castforge.h`) and Python `ctypes` wrapper (`bindings/python/castforge.py`).

---

## 4. Dual-Plane IR Design (`src/ir/`)

### 4.1 SSA Compute Graph (`mlir.rs`)
- **SlotMap Arenas**: Operations (`OpId`), SSA values (`ValueId`), blocks (`BlockId`), and regions
  (`RegionId`) live in dense `slotmap::SlotMap` arenas inside `Module`.
- **Zero-Heap Small Vectors**: `Operation::operands` (`SmallVec<[ValueId; 4]>`) and
  `Operation::results` (`SmallVec<[ValueId; 2]>`) avoid heap allocation for standard instructions.
- **Deterministic Attributes**: `Operation::attributes` uses `IndexMap<String, Attribute>` for
  deterministic MLIR printing and structural hashing (`hash-ir`).

### 4.2 Weight Plane (`tensor_store.rs`)
- **Inline vs. External Weights**: Small scalars/constants stay inline as `dense<...>` attributes,
  while weight matrices use `dense_resource<key>` handles referencing `Module::tensor_store`.
- **Lazy SafeTensors Indexing**: `TensorStore::load_safetensors_lazy_file` reads only the JSON
  header from disk and populates `TensorBacking::ExternalFile { path, offset, length }` descriptors
  so multi-gigabyte checkpoints can be inspected or sliced without eager RAM allocation.

### 4.3 Input Hardening & Safety Quotas
- **Checked Tensor Arithmetic**: `TensorType::num_elements()` and `TensorType::checked_byte_len()`
  use checked multiplication (`checked_mul`) and enforce `TensorType::MAX_TENSOR_ELEMENTS`
  (`1,000,000,000` elements).
- **Bounded Region Nesting**: `MlirParser` enforces `MAX_REGION_NESTING_DEPTH = 64` to prevent
  stack overflow on deeply nested regions.
- **Binary Header Quotas**: SafeTensors headers are capped at 100 MiB and GGUF tensor/metadata
  counts are capped at `65,536` entries with bounds-checked slice reads.

---

## 5. Static Liveness & Arena Memory Planning (`src/codegen/memory_plan.rs`)

1. **Block Liveness Analysis**: `compute_block_liveness` assigns each operation a topological index
   `1..=N` and computes `LiveInterval { first_def, last_use, ... }` for every SSA tensor value.
2. **In-Place Bufferization (`bufferize-inplace`)**: Elementwise ops whose input tensor has its
   final use at that instruction (`last_use == step`) and matches the output byte size receive an
   `inplace_operand` annotation, reusing the input buffer with zero extra memory.
3. **Greedy Interval Coloring**: `build_memory_plan` packs all intermediate buffers into a single
   contiguous `arena: &mut [f32]` allocation, reusing slots across non-overlapping live intervals.

---

## 6. Loop IR & Multi-Target Code Generation (`src/codegen/`)

### 6.1 Imperative Loop IR (`src/ir/loops.rs`, `src/passes/loops/`)
`lower_to_loop_ir` converts tensor operations into `LoopKernel`s composed of `LoopStmt` nodes
(`For`, `Alloc`, `Assign`, `Comment`) annotated with `Parallel`, `Vectorize(width)`, `Unroll(n)`,
and `SimdIsa`. `OptimizeLoopIrPass` performs loop fusion, fission, interchange, parallel collapse,
unroll-and-jam, and vectorization directly on `LoopModule`.

### 6.2 Target Backends (`src/codegen/targets/`)

| Target | ID | Output Format & Key Capabilities |
| :--- | :--- | :--- |
| **Rust** | `rust` | Standalone Rust (`pub fn main`) with static `arena: &mut [f32]` |
| **C99** | `c` | Portable C99 with OpenMP + `AVX2`/`AVX-512`/`NEON`/`SVE`/`RVV`/`SIMD128` |
| **WebGPU** | `wgsl` | `@compute @workgroup_size(...)` WGSL compute shaders |
| **CUDA** | `cuda` | NVIDIA `.cu` kernels with WMMA Tensor Cores & `__shared__` tiling |
| **Metal** | `msl` | Apple Silicon `.metal` shaders with `[[thread_position_in_grid]]` |
| **ROCm HIP** | `hip` | AMD `.hip.cpp` kernels with `<hip/hip_runtime.h>` |
| **OpenCL** | `opencl` | Khronos OpenCL 1.2+ `.cl` kernels with `__kernel` & `__global` buffers |

---

## 7. Verification, Profiling & Telemetry

- **IR & Shape Verifier (`verify-ir`)**: Checks SSA dominance, block ownership, and StableHLO
  shape/dtype compatibility (`verify_stablehlo_shapes`) after each pass.
- **Reference Interpreter (`src/codegen/eval.rs`)**: Evaluates all StableHLO ops and `castforge.*`
  fused composites in memory, powering `--verify-numerics` and automated pass bisection
  (`bisect_module_snapshots`).
- **Roofline Cost Model (`src/codegen/cost_model.rs`)**: Computes FLOPs, memory traffic,
  arithmetic intensity (`FLOPs/Byte`), and ridge-point bottleneck classification (`ComputeBound` vs.
  `MemoryBound`) across hardware profiles (`h100`, `a100`, `m3-max`, `avx512`, `neon`).
- **Structured Telemetry (`PassTelemetry`)**: `PassManager::run_with_telemetry` records per-pass
  execution time (`elapsed_us`), live op counts before/after, and mutation counters as JSON.
