/*
 * CastForge C/C++ Embedding Header (`include/castforge.h`)
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */

#ifndef CASTFORGE_H
#define CASTFORGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns the static NUL-terminated CastForge library version string (e.g., "1.0.0").
 */
const char *castforge_version(void);

/**
 * Frees a heap-allocated NUL-terminated string returned by CastForge FFI functions.
 */
void castforge_free_string(char *ptr);

/**
 * Optimizes an MLIR text string using `pipeline_spec` (e.g., "O2", "llm-opt", or "cse,dce").
 * Writes the optimized MLIR string to `*out_mlir` and optional JSON pass telemetry to
 * `*out_telemetry_json` (pass NULL to skip).
 *
 * Returns 0 on success or non-zero on error (writing the error message to `*out_error`).
 */
int32_t castforge_optimize_mlir(
    const char *input_mlir,
    const char *pipeline_spec,
    char **out_mlir,
    char **out_telemetry_json,
    char **out_error
);

/**
 * Compiles an MLIR text string through `pipeline_spec` and emits target code for `target_name`
 * ("rust", "c", "wgsl", "cuda", "metal", "ptx") into `*out_code`.
 *
 * Returns 0 on success or non-zero on error (writing the error message to `*out_error`).
 */
int32_t castforge_compile_mlir(
    const char *input_mlir,
    const char *pipeline_spec,
    const char *target_name,
    char **out_code,
    char **out_error
);

#ifdef __cplusplus
}
#endif

#endif /* CASTFORGE_H */
