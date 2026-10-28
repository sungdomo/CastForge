# CastForge

**CastForge** is a pure-Rust AI Model IR Transformation, Optimization, Weight Surgery,
Visualization, and Code Generation toolset built around an MLIR-compatible Dual-Plane
Intermediate Representation (**Compute Graph + TensorStore**) with first-class **StableHLO**
support.

## Workspace Architecture

- [`crates/castforge-ir`](crates/castforge-ir): Arena-allocated MLIR-compatible IR (`Module`,
  `Operation`, `Value`, `Block`, `Region`, `Type`, `Attribute`, `Location`), structural verifier,
  and the `TensorStore` weight plane (Linear/SLERP/TIES merging, L1 structured pruning, symmetric
  `i8`/`i4` quantization).
- [`crates/castforge-dialect-stablehlo`](crates/castforge-dialect-stablehlo): Strongly-typed
  `stablehlo` and `func` dialect views (`StableHloOp`, `DotDimensionNumbers`, `FuncView`) and
  shape verifiers.
- [`crates/castforge-mlir`](crates/castforge-mlir): Zero-C++ MLIR parser & printer supporting both
  **MLIR Generic Form** (`"stablehlo.add"(...)`) and **Pretty/Custom Syntax**
  (`stablehlo.add %0, %1 : tensor<...>`), including `!quant.uniform` quantized tensor types.
- [`crates/castforge-passes`](crates/castforge-passes): Unified `Pass`, `PassRegistry`, and
  `PassManager` supporting static Rust construction, CLI pipeline strings (`-p`), direct
  subcommand invocation, declarative TOML config files (`--config`), and step-by-step pass tracing.
- [`crates/castforge-viz`](crates/castforge-viz): Computation graph exporter for **Mermaid**,
  **Graphviz DOT**, **JSON**, and a self-contained **Interactive HTML Pass Timeline Debugger**.
- [`crates/castforge-codegen`](crates/castforge-codegen): Static memory/buffer planner
  (`MemoryPlan`) and multi-target source code generator for **Rust**, **C99**, and **WebGPU WGSL**.
- [`crates/castforge-cli`](crates/castforge-cli): The `castforge` command-line tool.

## Built-in Transformation Passes

| Pass Name | Description |
| :--- | :--- |
| `dce` | Dead Code Elimination (removes unused pure operations) |
| `cse` | Common Subexpression Elimination (deduplicates identical pure operations) |
| `fold-transpose` | Folds chained `stablehlo.transpose` ops and eliminates identity transposes |
| `fold-reshape` | Folds chained `stablehlo.reshape` ops and eliminates identity reshapes |
| `simplify-algebraic` | Algebraic identity simplification (`x+0->x`, `x*1->x`, `x*0->0`) |
| `fold-constants` | Constant folding for arithmetic and shape ops on constant splat tensors |
| `verify-shapes` | Verifies StableHLO tensor shapes and rank consistency across all ops |
| `fuse-elementwise` | Fuses elementwise producer-consumer pairs into `stablehlo.composite` |
| `ir-edit` | Surgical IR editing pass (`rename_op=from:to`, `annotate_op=op:key:val`) |
| `merge-weights` | Merges constant weight tensors (`method=linear\|slerp\|ties,t=0.5`) |
| `prune-structured` | Structured L1 channel pruning on 2D weights (`sparsity=0.5`) |
| `quantize-weights` | Symmetric weight quantization (`bits=8\|4`) + `uniform_dequantize` |
| `canonicalize` | Fixed-point algebraic, constant, transpose, reshape, CSE, and DCE pass |
| `inliner` | Inlines `func.call` operations by splicing callee bodies into caller blocks |
| `symbol-dce` | Removes unreachable `private` `func.func` symbols from the module |
| `sccp` | Sparse Conditional Constant Propagation (`compare`/`select` + constants) |
| `topological-sort` | Reorders operations inside blocks to satisfy SSA def-before-use ordering |
| `strip-debuginfo` | Strips source location metadata (`loc(...)`) from all operations |
| `print-op-stats` | Prints operation counts, parameter sizes, and MatMul FLOP estimates |

## Quick Start

```bash
# 1. List all registered passes
cargo run --bin castforge -- list-passes

# 2. Run a pass pipeline via CLI flag (-p) or TOML config (--config)
cargo run --bin castforge -- opt examples/mlp_redundant.mlir --config examples/pipeline.toml

# 3. End-to-End Weight Surgery: SLERP Merge -> 50% Structured Pruning -> INT8 Quantization
cargo run --bin castforge -- opt examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),quantize-weights(bits=8)"

# 4. Generate an Interactive HTML Pass Timeline & Graph Debugger
cargo run --bin castforge -- viz examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),quantize-weights(bits=8)" \
  --format html -o examples/weight_surgery_trace.html

# 5. Lower the optimized/quantized model to standalone Rust, C99, or WebGPU WGSL code
cargo run --bin castforge -- codegen examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),quantize-weights(bits=8)" \
  --target rust
```
