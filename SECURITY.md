# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Input Hardening Guarantees

CastForge is designed to safely process untrusted model artifacts (`.mlir`, `.safetensors`,
`.gguf`, `.onnx.json`, and `.json`):

1. **Checked Dimension & Byte-Offset Arithmetic**: Tensor element counts and byte offsets are
   validated using checked integer arithmetic (`checked_mul` / `checked_add`) with a hard cap of
   `TensorType::MAX_TENSOR_ELEMENTS` (`1,000,000,000` elements) and a 100 MiB header limit on
   `.safetensors` metadata.
2. **Bounded Region Recursion Depth**: The MLIR parser enforces `MAX_REGION_NESTING_DEPTH` (`64`)
   to prevent call-stack exhaustion on deeply nested regions.
3. **Panic-Safe C FFI Boundary**: All `extern "C"` entry points in `src/ffi.rs` wrap execution in
   `std::panic::catch_unwind` to guarantee that panics never cross FFI boundaries.

## Reporting a Vulnerability

If you discover a security vulnerability (such as an out-of-bounds read, integer overflow, or
unbounded allocation triggered by malformed input), please report it privately via GitHub Security
Advisories on the repository rather than opening a public issue.
