# Changelog

All notable changes to **CastForge** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dedicated `passes::serving` submodule hierarchy (`kv_attention`, `memory_pools`, `loop_buffers`,
  `cacheblend`, `transpose_sinking`, `fleet_directory`, `collectives`, `tiered_storage`,
  `mutable_loads`, `index_views`, `dependencies`, `device_graphs`, `kernel_stubs`, `routines`,
  `module_hashing`, `effect_chains`, `parameters`, `tenant_quotas`, and `shape_fallbacks`).
- Standard RustCrypto (`aes-gcm`, `hkdf`, `sha2`, `subtle`) authenticated encryption and per-tenant
  key derivation for L2 KV cache frame serialization.
- Hardware Roofline profile aliases (`h100`, `a100`, `b200`, `m1`–`m4`, `m3-max`, `mi250`,
  `mi300`, `mi300x`) in `config/targets.toml` and `cost_model`.

### Fixed
- Escaped and validated untrusted IR attributes (`profile_label`, `fused_loop_chain` step specs,
  and HTML visualization frames) against source/script injection.
- Replaced silent no-op / zero fallbacks on unsupported operations in codegen emitters and the
  reference evaluator with explicit error reporting.

## [1.0.0] - 2026-10-31

### Added
- **Zero-C++ MLIR & StableHLO Arena Engine (`src/ir/`)**:
  - SlotMap-backed IR (`Module`, `Region`, `Block`, `Operation`, `Value`) with generic MLIR and
    StableHLO pretty-syntax parsing and deterministic round-trip printing.
  - Binary Weight Plane (`TensorStore`) supporting `F32`, `F16`, `BF16`, `I8`/`U8`, uniform
    quantized types (`!quant.uniform`), and lazy `ExternalFile` `.safetensors` indexing.
  - Symmetric model importers and exporters for **ONNX-JSON**, **GGUF v2/v3**, **SafeTensors**,
    and **HuggingFace `config.json`** (`Llama`, `Mistral`, `Qwen2`, `Gemma`, `GPT-2`).
- **97 Composable Optimization & Surgery Passes (`src/passes/`)**:
  - Organized into 9 domain packages (`canonicalize`, `shapes`, `weights`, `quantize`, `fuse`,
    `memory`, `distributed`, `loops`, `normalize`) plus declarative `config/passes.toml` catalog.
  - Built-in optimization presets (`O1`, `O2`, `O3`, `llm-opt`) and structured per-pass JSON
    telemetry (`PassManager::run_with_telemetry`).
- **Polyhedral Loop IR & 6 Code Generation Backends (`src/codegen/`)**:
  - Liveness-based greedy interval coloring arena planner and in-place bufferizer.
  - Standalone code generation for **Rust** (scalar + AVX2/NEON/AVX-512), **C99**, **WGSL**,
    **CUDA**, **Metal**, and **NVIDIA PTX**, plus native JIT compilation (`rustc`/`cc`) and an
    in-memory StableHLO + Loop IR reference interpreter.
- **C Foreign Function Interface (`src/ffi.rs`, `include/castforge.h`) & Python SDK**:
  - Panic-safe `extern "C"` ABI (`castforge_optimize_mlir`, `castforge_compile_mlir`) and
    zero-dependency Python `ctypes` wrapper (`bindings/python/castforge.py`).
