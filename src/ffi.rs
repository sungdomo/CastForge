//! C Foreign Function Interface (`extern "C"` ABI) for embedding CastForge in C, C++, Python,
//! Go, or other host runtimes (`src/ffi.rs`).

use crate::codegen::{BackendRegistry, TargetConfig};
use crate::ir::{parse_mlir, print_mlir};
use crate::passes::{PassManager, PassRegistry};
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

const VERSION_CSTR: &[u8] = b"1.0.0\0";

/// Returns a static NUL-terminated version string (`"1.0.0"`).
#[unsafe(no_mangle)]
pub extern "C" fn castforge_version() -> *const c_char {
    VERSION_CSTR.as_ptr().cast::<c_char>()
}

/// Frees a NUL-terminated string previously allocated and returned by CastForge FFI functions.
///
/// # Safety
///
/// `ptr` must be either null or a pointer previously returned by `castforge_compile_mlir` or
/// `castforge_optimize_mlir` that has not yet been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn castforge_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

/// Optimizes an MLIR text string using `pipeline_spec` (e.g., `"O2"`, `"llm-opt"`, or
/// `"cse,dce"`) and writes the resulting MLIR text to `*out_mlir` and optional JSON telemetry to
/// `*out_telemetry_json`.
///
/// Returns `0` on success or a non-zero status code on failure (populating `*out_error` when
/// non-null).
///
/// # Safety
///
/// - `input_mlir` and `pipeline_spec` must be valid, NUL-terminated UTF-8 C strings.
/// - `out_mlir` must be a valid non-null pointer to a `*mut c_char`.
/// - `out_telemetry_json` and `out_error` may be null or valid pointers to `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn castforge_optimize_mlir(
    input_mlir: *const c_char,
    pipeline_spec: *const c_char,
    out_mlir: *mut *mut c_char,
    out_telemetry_json: *mut *mut c_char,
    out_error: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        optimize_mlir_inner(input_mlir, pipeline_spec, out_mlir, out_telemetry_json)
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(msg)) => {
            unsafe { write_c_string(out_error, &msg) };
            1
        }
        Err(_) => {
            unsafe { write_c_string(out_error, "panic caught across FFI boundary") };
            2
        }
    }
}

/// Compiles an MLIR text string through `pipeline_spec` (or empty string for no passes) and emits
/// source code for `target_name` (`"rust"`, `"c"`, `"wgsl"`, `"cuda"`, `"metal"`, `"ptx"`).
///
/// Returns `0` on success or a non-zero status code on failure (populating `*out_error` when
/// non-null).
///
/// # Safety
///
/// - `input_mlir`, `pipeline_spec`, and `target_name` must be valid NUL-terminated C strings.
/// - `out_code` must be a valid non-null pointer to `*mut c_char`.
/// - `out_error` may be null or a valid pointer to `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn castforge_compile_mlir(
    input_mlir: *const c_char,
    pipeline_spec: *const c_char,
    target_name: *const c_char,
    out_code: *mut *mut c_char,
    out_error: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        compile_mlir_inner(input_mlir, pipeline_spec, target_name, out_code)
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(msg)) => {
            unsafe { write_c_string(out_error, &msg) };
            1
        }
        Err(_) => {
            unsafe { write_c_string(out_error, "panic caught across FFI boundary") };
            2
        }
    }
}

unsafe fn optimize_mlir_inner(
    input_mlir: *const c_char,
    pipeline_spec: *const c_char,
    out_mlir: *mut *mut c_char,
    out_telemetry_json: *mut *mut c_char,
) -> Result<(), String> {
    if input_mlir.is_null() || pipeline_spec.is_null() || out_mlir.is_null() {
        return Err("Null pointer passed to castforge_optimize_mlir".to_string());
    }
    let src = unsafe { CStr::from_ptr(input_mlir) }
        .to_str()
        .map_err(|e| format!("Invalid UTF-8 in input_mlir: {e}"))?;
    let spec = unsafe { CStr::from_ptr(pipeline_spec) }
        .to_str()
        .map_err(|e| format!("Invalid UTF-8 in pipeline_spec: {e}"))?
        .trim();

    let mut module = parse_mlir(src).map_err(|e| e.to_string())?;
    let mut pm = build_pass_manager(spec)?;
    let telemetry = pm
        .run_with_telemetry(&mut module)
        .map_err(|e| e.to_string())?;

    let printed = print_mlir(&module);
    unsafe { write_c_string(out_mlir, &printed) };

    if !out_telemetry_json.is_null() {
        let json = PassManager::telemetry_to_json(&telemetry).map_err(|e| e.to_string())?;
        unsafe { write_c_string(out_telemetry_json, &json) };
    }
    Ok(())
}

unsafe fn compile_mlir_inner(
    input_mlir: *const c_char,
    pipeline_spec: *const c_char,
    target_name: *const c_char,
    out_code: *mut *mut c_char,
) -> Result<(), String> {
    if input_mlir.is_null()
        || pipeline_spec.is_null()
        || target_name.is_null()
        || out_code.is_null()
    {
        return Err("Null pointer passed to castforge_compile_mlir".to_string());
    }
    let src = unsafe { CStr::from_ptr(input_mlir) }
        .to_str()
        .map_err(|e| format!("Invalid UTF-8 in input_mlir: {e}"))?;
    let spec = unsafe { CStr::from_ptr(pipeline_spec) }
        .to_str()
        .map_err(|e| format!("Invalid UTF-8 in pipeline_spec: {e}"))?
        .trim();
    let target_str = unsafe { CStr::from_ptr(target_name) }
        .to_str()
        .map_err(|e| format!("Invalid UTF-8 in target_name: {e}"))?
        .trim();

    let mut module = parse_mlir(src).map_err(|e| e.to_string())?;
    if !spec.is_empty() {
        let mut pm = build_pass_manager(spec)?;
        pm.run(&mut module).map_err(|e| e.to_string())?;
    }

    let cg_reg = BackendRegistry::with_builtin_backends();
    let backend = cg_reg
        .get(target_str)
        .ok_or_else(|| format!("Unknown codegen target `{target_str}`"))?;
    let cfg = TargetConfig::default();
    let code = backend
        .emit_module(&module, &cfg)
        .map_err(|e| e.to_string())?;
    unsafe { write_c_string(out_code, &code) };
    Ok(())
}

fn build_pass_manager(spec: &str) -> Result<PassManager, String> {
    let reg = PassRegistry::with_builtin_passes();
    if spec.is_empty() {
        return Ok(PassManager::new());
    }
    PassManager::from_pipeline_str(&reg, spec).map_err(|e| e.to_string())
}

unsafe fn write_c_string(dst: *mut *mut c_char, text: &str) {
    if dst.is_null() {
        return;
    }
    let clean = text.replace('\0', " ");
    let cstr = CString::new(clean).unwrap_or_default();
    unsafe {
        *dst = cstr.into_raw();
    }
}
