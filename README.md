# CastForge

**CastForge** is a pure-Rust AI Model IR Transformation, Optimization, Weight Surgery,
Visualization, and Code Generation toolset built around an MLIR-compatible Dual-Plane
Intermediate Representation (**Compute Graph + TensorStore**) with first-class **StableHLO**
support.

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
  (`stablehlo.add %0, %1 : tensor<...>`), including `!quant.uniform` quantized tensor types and
  `dense_resource<key>` external weight references.
- [`crates/castforge-passes`](crates/castforge-passes): Unified `Pass`, `PassRegistry`, and
  `PassManager` with **36 built-in compiler passes** supporting static Rust construction, CLI
  pipeline strings (`-p`), direct subcommand invocation, declarative TOML config files (`--config`),
  and step-by-step pass tracing.
- [`crates/castforge-viz`](crates/castforge-viz): Computation graph exporter for **Mermaid**,
  **Graphviz DOT**, **JSON**, and a self-contained **Interactive HTML Pass Timeline, Roofline &
  Arena Memory Liveness Debugger**.
- [`crates/castforge-codegen`](crates/castforge-codegen): Liveness analyzer (`LivenessAnalysis`),
  greedy interval-coloring arena buffer planner (`MemoryPlan`), in-memory reference IR interpreter
  & numerical verifier (`eval_module`), and multi-target source code generator for **Rust**,
  **C99**, and **WebGPU WGSL**.
- [`crates/castforge-cli`](crates/castforge-cli): The `castforge` command-line tool (`opt`, `viz`,
  `codegen`, `run`, `list-passes`, plus direct pass subcommands).

## Built-in Transformation Passes (36 Passes)

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
| `fuse-matmul-epilogue` | Fusion | Fuses `dot_general` + bias (`relu`/`tanh`/`silu`/`scale`) |
| `fuse-horizontal-dot` | Fusion | Horizontally fuses parallel `dot_general` ops (e.g. Q/K/V) |
| `fuse-softmax` | Fusion | Fuses `exp(x) / broadcast(reduce_sum(exp(x)))` into composite |
| `fuse-rmsnorm` | Fusion | Fuses `x * rsqrt(mean(x^2) + eps) * weight` into composite |
| `fuse-swiglu` | Fusion | Fuses SwiGLU (`gate / (1 + exp(-gate)) * up`) into composite |
| `decompose-composites` | Lowering | Expands `stablehlo.composite` ops to primitive StableHLO |
| `fold-norm-weights` | Weight Surgery | Absorbs RMSNorm/LayerNorm scale weights into MatMuls |
| `bake-lora` | Weight Surgery | Fuses LoRA branches `W + alpha * (A @ B)` into base weights |
| `task-arithmetic` | Weight Surgery | Applies multi-checkpoint task vectors & DARE pruning |
| `merge-weights` | Weight Surgery | Merges paired weights (`method=linear\|slerp\|ties,t=0.5`) |
| `prune-structured` | Weight Surgery | Structured L1 channel pruning on weights (`sparsity=0.5`) |
| `quantize-weights` | Weight Surgery | Symmetric `i8`/`i4` quantization + `uniform_dequantize` |
| `externalize-weights` | Weight I/O | Moves inline `dense<...>` to `dense_resource<key>` |

## Quick Start

```bash
# 1. List all 33 registered passes
cargo run --bin castforge -- list-passes

# 2. Run an optimization pipeline with automatic numerical verification
cargo run --bin castforge -- opt examples/mlp_redundant.mlir \
  -p "canonicalize,fuse-matmul-epilogue" --verify-numerics

# 3. End-to-End Weight Surgery + SafeTensors binary export
cargo run --bin castforge -- opt examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),externalize-weights" \
  --save-weights examples/merged_weights.safetensors

# 4. Generate an Interactive HTML Pass Timeline, Roofline & Arena Memory Debugger
cargo run --bin castforge -- viz examples/weight_surgery.mlir \
  -p "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),quantize-weights(bits=8)" \
  --format html -o examples/weight_surgery_trace.html

# 5. Lower the optimized model to standalone Rust, C99, or WebGPU WGSL with Arena Buffer Reuse
cargo run --bin castforge -- codegen examples/weight_surgery.mlir \
  -p "canonicalize,fuse-matmul-epilogue" --target rust --tile-size 16

# 6. Profile Roofline FLOPs, memory bandwidth traffic, and arithmetic intensity
cargo run --bin castforge -- profile examples/mlp_redundant.mlir \
  -p "canonicalize,fuse-matmul-epilogue"
```
