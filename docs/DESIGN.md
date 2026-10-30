# CastForge v1.0.0 — System Architecture & Design Document

**CastForge** is a pure-Rust, zero-LLVM/C++-dependency MLIR + StableHLO AI compiler, weight
surgery engine, polyhedral loop optimizer, and multi-architecture code generator.

---

## 1. Core Architectural Philosophy

CastForge is built around three foundational design principles:

1. **Everything is a Modular A-to-B Translator (`Translator` + `Payload`)**
   Every compiler component—frontend model importers (`.mlir`, `.safetensors`, `.onnx.json`,
   `.pt.json`, `.gguf`), each of the **91 Graph IR passes & presets**, each of the **7 LoopIR
   passes**, lowering stages (`ir -> loop-ir`), and backend code emitters (`loop-ir -> code`)—
   implements the unified `castforge_ir::Translator` trait over a polymorphic
   `castforge_ir::Payload`. Any stage can be composed into a `TranslatorChain` via CLI
   (`castforge chain -s "A | B | C"`) or Rust API.
2. **Dual-Plane Intermediate Representation (SSA Compute Graph + Weight Plane)**
   Traditional compilers treat multi-gigabyte model weights either as giant AST literals or as
   opaque external pointers. CastForge decouples them into two synchronized planes inside
   `castforge_ir::Module`:
   - **Compute Graph Plane:** Arena-allocated (`slotmap::SlotMap`) SSA operations, values, blocks,
     and regions (`OpId`, `ValueId`, `BlockId`, `RegionId`).
   - **Weight Plane (`TensorStore`):** Dedicated storage holding inline `f32`/`i64` tensors,
     quantized `i8`/`i4` buffers (`TensorData::QuantizedI8`), or lazy zero-copy `SafeTensors` shard
     references (`TensorData::ExternalFile`).
3. **Two-Tier IR Lowering (`Module` $\to$ `LoopModule` $\to$ Multi-Arch Target Source)**
   High-level algebraic, fusion, quantization, and graph surgery passes operate on the SSA
   `Module`. Schedule and loop-nest transformations (`loop-interchange`, `loop-unroll`,
   `loop-vectorize`, `fuse-loops`, `unroll-and-jam`, `collapse-parallel`, `loop-fission`) operate
   on an explicit polyhedral/imperative `LoopModule` before target-specific code generation.

---

## 2. End-to-End Compiler Pipeline

```text
  +--------------------+   +-------------------+   +-------------------+
  |  Model Frontends   |   |   Weight Shards   |   |  StableHLO MLIR   |
  | ONNX / Torch / GGUF|   |   (.safetensors)  |   |     (.mlir)       |
  +---------+----------+   +---------+---------+   +---------+---------+
            |                        |                       |
            +------------------------+-----------------------+
                                     |
                                     v
                     +-------------------------------+
                     |    castforge_ir::Module       |
                     |  - SSA Compute Graph (SlotMap)|
                     |  - Weight Plane (TensorStore) |
                     +---------------+---------------+
                                     |
               +---------------------+---------------------+
               |  91 Graph IR Passes & Presets (O1/O2/O3)  |
               |  - Algebraic / Canonicalization / SCCP    |
               |  - Weight Surgery (Quant/Prune/Merge/LoRA)|
               |  - LLM & Transformer Operator Fusions     |
               |  - Distributed SPMD / Autodiff / Memory   |
               +---------------------+---------------------+
                                     |
                                     v
                     +-------------------------------+
                     | Static Liveness & Arena Plan  |
                     |  - compute_module_liveness()  |
                     |  - build_memory_plan() O(N)   |
                     +---------------+---------------+
                                     |
                                     v
                     +-------------------------------+
                     |   castforge_codegen::LoopIR   |
                     |  - 7 LoopIR Passes            |
                     |    (interchange, unroll,      |
                     |     vectorize, jam, fission)  |
                     +---------------+---------------+
                                     |
                                     v
                     +-------------------------------+
                     |    Pluggable BackendRegistry  |
                     |  - CPU: Rust, C99 (7 SIMD ISAs|
                     |    Scalar/AVX2/AVX512/NEON/   |
                     |    SVE/RVV/WasmSimd128 + JIT) |
                     |  - GPU: WGSL, CUDA, HIP,      |
                     |    Metal (MSL), OpenCL        |
                     +-------------------------------+
```

---

## 3. Source Module Topology (`src/`)

- **`src/ir`**: Core arena-backed SSA data structures (`Module`, `Operation`, `Value`, `Block`,
  `Region`), `Type`/`DType`/`Dim`, `TensorStore` (`tensor_store.rs`), strongly-typed `StableHloOp`
  dialect views (`stablehlo.rs`), MLIR parser/printer (`parser.rs`, `mlir.rs`), ONNX/GGUF
  importers/exporters (`import.rs`), and `Payload`/`Translator`/`TranslatorChain`.
- **`src/passes`**: `Pass` trait, `PassManager`, `PassRegistry`, declarative TOML/JSON
  `PipelineConfig`, and **21 built-in Graph IR transformation passes and presets**.
- **`src/codegen`**: Liveness analysis, static arena memory planner (`MemoryPlan`), Roofline cost
  model, reference interpreter (`evaluate_module`), `LoopModule` + **`OptimizeLoopIrPass`**,
  `CodegenBackend` + `BackendRegistry` (7 built-in targets + 7 CPU SIMD ISAs), autotuner, and
  native C JIT runner.
- **`src/viz`**: Graphviz DOT (`to_dot`) and self-contained interactive HTML5/SVG step-by-step pass
  debugger (`to_interactive_trace_html`).
- **`src/cli`**: Unified `castforge` command-line tool exposing `opt`, `translate`, `codegen`,
  `autotune`, `jit`, `verify`, `stats`, `inspect`, `viz`, `diff`, and `list-passes`.

---

## 4. Dual-Plane IR Details (`src/ir`)

### 4.1 SSA Compute Graph Plane
- **Arena Storage:** Operations (`OpId`), SSA values (`ValueId`), basic blocks (`BlockId`), and
  regions (`RegionId`) are stored in dense `slotmap::SlotMap` arenas inside `Module`.
- **SmallVec Operands & Results:** `Operation::operands` (`SmallVec<[ValueId; 4]>`) and
  `Operation::results` (`SmallVec<[ValueId; 2]>`) avoid heap allocations for common unary, binary,
  and ternary operations.
- **Deterministic Attributes:** `Operation::attributes` uses `IndexMap<String, Attribute>` to
  preserve insertion order during MLIR round-tripping while supporting canonical key sorting in
  passes like `cse`, `normalize-ssa`, and `hash-ir`.
- **$O(N)$ Reference Counting:** `Module::compute_use_counts()` computes operand reference counts
  across all active blocks in a single $O(N)$ sweep, enabling linear-time Dead Code Elimination.

### 4.2 Weight Plane (`TensorStore`) & Weight Surgery (`src/passes/`)
- **Pure IR Storage (`src/ir`):** Small constants remain inline in MLIR attributes, while large
  weight matrices use `dense_resource<key>` handles pointing into `Module::tensor_store`.
  `src/ir/tensor_store.rs` loads and saves `TensorStore` entries directly to/from HuggingFace
  `.safetensors` files.
- **Weight Surgery Kernels (`quantize.rs`, `fold_weights.rs`, `merge_weights.rs`):**
  - `quantize_symmetric_f32`: Symmetric per-tensor `i8` (`[-128, 127]`) and `i4` (`[-8, 7]`)
    quantization emitting `!quant.uniform<i8:f32, scale:zp>`.
  - `prune_2d_columns_by_l1`: Structured output-column pruning by L1 norm with zero-dimension
    safety guards.
  - `merge_f32_tensors`: Linear interpolation, Spherical Linear Interpolation (`SLERP`), and
    `TIES`-Merging (Trim, Elect Sign, Disjoint Merge).

---

## 5. Static Liveness & Arena Memory Planning (`src/codegen`)

1. **Block Liveness Analysis (`liveness.rs`):**
   `compute_block_liveness` assigns each operation in a block a topological step index `1..=N` and
   computes `LiveInterval { first_def, last_use, is_constant, is_block_arg, is_returned }` for
   every tensor SSA value.
2. **In-Place Buffer Donation (`bufferize-inplace`):**
   Elementwise operations whose input tensor has its final use at that exact instruction
   (`last_use == step`) and matches the output byte size are annotated with
   `inplace_operand = <idx>`.
3. **Greedy Interval Coloring & $O(1)$ Lookup (`memory_plan.rs`):**
   `build_memory_plan` allocates intermediate tensors into a single contiguous `arena: &mut [f32]`
   workspace. Non-overlapping live intervals share the same `ArenaSlot` (preferring minimal slot
   growth), while `inplace_operand` donors reuse the donor's slot at zero additional byte cost.
   `MemoryPlan::allocation_for(ValueId)` indexes allocations via an internal `HashMap<ValueId,
   usize>` for $O(1)$ lookup during codegen.

---

## 6. Polyhedral Loop IR & Pluggable Multi-Arch Codegen

### 6.1 Imperative Loop IR (`loop_ir.rs`)
Before emitting target code, `lower_module_to_loop_ir` lowers the SSA `Module` into a
target-independent `LoopModule` consisting of `LoopKernel`s and recursive `LoopStmt` nodes:
- `LoopStmt::For { var, lower, upper, step, annotations, body }` with `LoopAnnotation` tags
  (`Parallel`, `Vectorize(w)`, `SimdIsa(name)`, `Unroll(u)`, `GpuThreadIdx(axis)`).
- Structured statements (`Assign`, `Accumulate`, `IfZero`, `ExprStmt`, `Comment`) and scalar/index
  expressions (`LoopExpr::Load`, `Var`, `LitF32`, `Binary`, `Call`).

### 6.2 Multi-Architecture CPU SIMD & Pluggable GPU Backends (`backend.rs`)
- **Pluggable `CodegenBackend` Trait & `BackendRegistry`:**
  Every backend implements `CodegenBackend::emit_from_loop_ir(&LoopModule, &TargetConfig)` (with a
  default `emit_module(&Module, &TargetConfig)` pipeline that runs `lower_module_to_loop_ir` +
  `optimize_loop_module`). Users can register custom hardware targets at runtime via
  `BackendRegistry::register` or wrap them as `Translator` stages via `CustomBackendTranslator`.
- **7 Built-In Target Backends:**
  1. `rust`: Dependency-free Rust (`pub fn main(...)`) with shared `arena: &mut [f32]`.
  2. `c`: Portable C99 (`#pragma omp parallel for`, `#pragma omp simd`) with architecture-specific
     SIMD intrinsics across **7 Vector ISAs**:
     - `SimdIsa::Scalar` (1 lane)
     - `SimdIsa::Avx2` (`<immintrin.h>`, `__m256`, `_mm256_fmadd_ps`, 8 lanes)
     - `SimdIsa::Avx512` (`<immintrin.h>`, `__m512`, `_mm512_fmadd_ps`, 16 lanes)
     - `SimdIsa::Neon` (`<arm_neon.h>`, `float32x4_t`, `vfmaq_f32`, 4 lanes)
     - `SimdIsa::Sve` (`<arm_sve.h>`, `svfloat32_t`, `svmla_f32_m`, scalable lanes)
     - `SimdIsa::Rvv` (`<riscv_vector.h>`, `vfloat32m1_t`, `__riscv_vfmacc_vv_f32m1`, RVV 1.0)
     - `SimdIsa::WasmSimd128` (`<wasm_simd128.h>`, `v128_t`, `wasm_f32x4_mul`, 4 lanes)
  3. `wgsl`: WebGPU compute shaders (`@compute @workgroup_size(...)`).
  4. `cuda`: NVIDIA CUDA C++ (`.cu`) with WMMA Tensor Core helpers and `__shared__` double buffers.
  5. `hip`: AMD ROCm HIP C++ (`.hip.cpp`) with `<hip/hip_runtime.h>` and `hipLaunchKernelGGL`.
  6. `metal`: Apple Silicon Metal Shading Language (`.metal`) with `[[thread_position_in_grid]]`.
  7. `opencl`: Khronos OpenCL C 1.2+ (`.cl`) with `__kernel`, `__global`, and `get_global_id(0)`.

---

## 7. Verification, Bisection & Roofline Cost Modeling

- **Shape & Dominance Verification (`verify.rs`):** Every pass in `PassManager` can automatically
  invoke `module.verify()` + `verify_stablehlo_shapes(&module)` after execution.
- **Reference Interpreter (`eval.rs`):** Evaluates all StableHLO operations and all `castforge.*`
  fused composites on deterministic synthetic inputs. `verify_module_numerics` verifies numerical
  equivalence within tolerance (e.g. `1e-4`), and `bisect_module_snapshots` pinpoints the exact
  pass where numerical drift occurs in a single baseline evaluation.
- **Roofline Performance Estimator (`cost_model.rs`):** Computes per-op and whole-module FLOPs,
  memory traffic bytes (accounting for weight quantization and arena reuse), operational intensity
  ($\text{FLOPs}/\text{Byte}$), and ridge-point bottleneck classification (`ComputeBound` vs.
  `MemoryBound`).
