//! Pluggable Multi-Architecture Codegen Backend Trait ([`CodegenBackend`]) and Dynamic Target
//! Registry ([`BackendRegistry`]).
//!
//! Adding a new codegen target to CastForge only requires implementing
//! [`CodegenBackend::emit_from_loop_ir`] over the 4 statements of [`LoopModule`] (`Comment`,
//! `Alloc`, `For`, `Assign`) and registering the struct with [`BackendRegistry::register`].

use crate::codegen::targets::{
    emit_c_from_loop_ir, emit_c_module_with_config, emit_cuda_from_loop_ir,
    emit_cuda_module_with_config, emit_hip_from_loop_ir, emit_hip_module_with_config,
    emit_msl_from_loop_ir, emit_msl_module_with_config, emit_opencl_from_loop_ir,
    emit_opencl_module_with_config, emit_rust_from_loop_ir, emit_rust_module_with_config,
    emit_wgsl_from_loop_ir_with_workgroup,
};
pub use crate::codegen::targets::{emit_wgsl_module, emit_wgsl_module_with_config};
use crate::codegen::{LoopModule, LoopStmt, TargetConfig, lower_to_loop_ir};
use crate::ir::{Module, Payload, Translator};
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::fmt::Write as _;

/// Trait for pluggable target code generators in CastForge.
///
/// Backends that operate on the lowered [`LoopModule`] (Progressive Loop IR) only need to
/// implement [`CodegenBackend::emit_from_loop_ir`]; [`CodegenBackend::emit_module`] automatically
/// lowers [`Module`] to [`LoopModule`] via [`lower_to_loop_ir`] by default.
pub trait CodegenBackend: Send + Sync {
    /// Canonical lowercase target identifier (e.g. `"rust"`, `"c"`, `"wgsl"`, `"cuda"`, `"msl"`,
    /// `"hip"`, `"opencl"`).
    fn name(&self) -> &'static str;

    /// Standard source file extension (e.g. `"rs"`, `"c"`, `"wgsl"`, `"cu"`, `"metal"`,
    /// `"hip.cpp"`, `"cl"`).
    fn file_extension(&self) -> &'static str;

    /// Human-readable description of the target architecture and runtime.
    fn description(&self) -> &'static str;

    /// Emits target source code from a lowered [`LoopModule`] and [`TargetConfig`].
    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, config: &TargetConfig) -> Result<String>;

    /// Emits target source code from a Tensor-level [`Module`] and [`TargetConfig`].
    ///
    /// By default, lowers `module` to [`LoopModule`] and delegates to
    /// [`CodegenBackend::emit_from_loop_ir`].
    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        let loop_mod = lower_to_loop_ir(module, config);
        self.emit_from_loop_ir(&loop_mod, config)
    }
}

/// Standalone Rust Backend (`rust`, `.rs`).
#[derive(Debug, Default)]
pub struct RustBackend;

impl CodegenBackend for RustBackend {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn file_extension(&self) -> &'static str {
        "rs"
    }

    fn description(&self) -> &'static str {
        "Standalone Rust (Scalar / AVX2 / AVX-512 / NEON / SVE / RVV / WASM SIMD128)"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, _config: &TargetConfig) -> Result<String> {
        emit_rust_from_loop_ir(loop_mod)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_rust_module_with_config(module, config)
    }
}

/// Portable C99 Backend (`c`, `.c`).
#[derive(Debug, Default)]
pub struct C99Backend;

impl CodegenBackend for C99Backend {
    fn name(&self) -> &'static str {
        "c"
    }

    fn file_extension(&self) -> &'static str {
        "c"
    }

    fn description(&self) -> &'static str {
        "Portable C99 with OpenMP & multi-arch SIMD intrinsics (AVX2/AVX-512/NEON/SVE/RVV/WASM)"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, _config: &TargetConfig) -> Result<String> {
        emit_c_from_loop_ir(loop_mod)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_c_module_with_config(module, config)
    }
}

/// WebGPU WGSL Compute Shader Backend (`wgsl`, `.wgsl`).
#[derive(Debug, Default)]
pub struct WgslBackend;

impl CodegenBackend for WgslBackend {
    fn name(&self) -> &'static str {
        "wgsl"
    }

    fn file_extension(&self) -> &'static str {
        "wgsl"
    }

    fn description(&self) -> &'static str {
        "WebGPU WGSL compute shaders with workgroup shared memory tiling"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, config: &TargetConfig) -> Result<String> {
        emit_wgsl_from_loop_ir_with_workgroup(loop_mod, config.wgsl_workgroup_size)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_wgsl_module_with_config(module, config)
    }
}

/// NVIDIA CUDA C++ Backend (`cuda`, `.cu`).
#[derive(Debug, Default)]
pub struct CudaBackend;

impl CodegenBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn file_extension(&self) -> &'static str {
        "cu"
    }

    fn description(&self) -> &'static str {
        "NVIDIA CUDA C++ (__global__ kernels + host launch wrapper)"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, config: &TargetConfig) -> Result<String> {
        emit_cuda_from_loop_ir(loop_mod, config)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_cuda_module_with_config(module, config)
    }
}

/// Apple Metal Shading Language Backend (`msl`, `.metal`).
#[derive(Debug, Default)]
pub struct MslBackend;

impl CodegenBackend for MslBackend {
    fn name(&self) -> &'static str {
        "msl"
    }

    fn file_extension(&self) -> &'static str {
        "metal"
    }

    fn description(&self) -> &'static str {
        "Apple Silicon Metal Shading Language (MSL) compute kernels"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, _config: &TargetConfig) -> Result<String> {
        emit_msl_from_loop_ir(loop_mod)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_msl_module_with_config(module, config)
    }
}

/// AMD ROCm HIP Backend (`hip`, `.hip.cpp`), implemented directly over [`LoopModule`].
#[derive(Debug, Default)]
pub struct HipBackend;

impl CodegenBackend for HipBackend {
    fn name(&self) -> &'static str {
        "hip"
    }

    fn file_extension(&self) -> &'static str {
        "hip.cpp"
    }

    fn description(&self) -> &'static str {
        "AMD ROCm HIP C++ (__global__ kernels + hipLaunchKernelGGL host wrapper)"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, config: &TargetConfig) -> Result<String> {
        emit_hip_from_loop_ir(loop_mod, config)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_hip_module_with_config(module, config)
    }
}

/// Khronos OpenCL C Backend (`opencl`, `.cl`), implemented directly over [`LoopModule`].
#[derive(Debug, Default)]
pub struct OpenClBackend;

impl CodegenBackend for OpenClBackend {
    fn name(&self) -> &'static str {
        "opencl"
    }

    fn file_extension(&self) -> &'static str {
        "cl"
    }

    fn description(&self) -> &'static str {
        "Portable Khronos OpenCL C (__kernel compute kernels for multi-vendor GPUs/FPGAs)"
    }

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, _config: &TargetConfig) -> Result<String> {
        emit_opencl_from_loop_ir(loop_mod)
    }

    fn emit_module(&self, module: &Module, config: &TargetConfig) -> Result<String> {
        emit_opencl_module_with_config(module, config)
    }
}

/// Helper used by [`LoopModule`]-based backends (`HipBackend`, `OpenClBackend`, and custom user
/// backends) to emit C-family GPU kernel statements from [`LoopStmt`].
pub fn emit_c_like_loop_stmt(
    out: &mut String,
    stmt: &LoopStmt,
    ret_name: &str,
    thread_var: &str,
    indent: usize,
    map_outer_for_to_thread: bool,
) -> Result<()> {
    let pad = "    ".repeat(indent);
    match stmt {
        LoopStmt::Comment(c) => writeln!(out, "{pad}// {c}")?,
        LoopStmt::Alloc { name, len } if name != ret_name => {
            writeln!(out, "{pad}float {name}[{len}];")?;
        }
        LoopStmt::Alloc { .. } => {}
        LoopStmt::For {
            var,
            lower,
            upper,
            step,
            body,
            ..
        } => {
            if map_outer_for_to_thread && *lower == 0 && *step == 1 {
                writeln!(out, "{pad}if ({thread_var} < {upper}) {{")?;
                writeln!(out, "{pad}    const int {var} = {thread_var};")?;
                for inner in body {
                    emit_c_like_loop_stmt(out, inner, ret_name, thread_var, indent + 1, false)?;
                }
                writeln!(out, "{pad}}}")?;
            } else {
                writeln!(
                    out,
                    "{pad}for (int {var} = {lower}; {var} < {upper}; {var} += {step}) {{"
                )?;
                for inner in body {
                    emit_c_like_loop_stmt(out, inner, ret_name, thread_var, indent + 1, false)?;
                }
                writeln!(out, "{pad}}}")?;
            }
        }
        LoopStmt::Assign { dst, index, expr } => {
            writeln!(out, "{pad}{dst}[{index}] = {expr};")?;
        }
    }
    Ok(())
}

/// Dynamic Registry of [`CodegenBackend`] targets (`rust`, `c`, `wgsl`, `cuda`, `msl`, `hip`,
/// `opencl`, plus any user-registered custom backend).
pub struct BackendRegistry {
    order: Vec<&'static str>,
    backends: HashMap<&'static str, Box<dyn CodegenBackend>>,
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::with_builtin_backends()
    }
}

impl BackendRegistry {
    /// Creates an empty [`BackendRegistry`].
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            backends: HashMap::new(),
        }
    }

    /// Creates a [`BackendRegistry`] pre-populated with all 7 built-in multi-arch backends.
    pub fn with_builtin_backends() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(RustBackend));
        reg.register(Box::new(C99Backend));
        reg.register(Box::new(WgslBackend));
        reg.register(Box::new(CudaBackend));
        reg.register(Box::new(MslBackend));
        reg.register(Box::new(HipBackend));
        reg.register(Box::new(OpenClBackend));
        reg
    }

    /// Registers a new [`CodegenBackend`] implementation.
    pub fn register(&mut self, backend: Box<dyn CodegenBackend>) {
        let name = backend.name();
        if !self.order.contains(&name) {
            self.order.push(name);
        }
        self.backends.insert(name, backend);
    }

    /// Looks up a registered [`CodegenBackend`] by name (case-insensitive, supporting aliases
    /// such as `"c99"`, `"metal"`, `"rocm"`, `"cl"`).
    pub fn get(&self, name: &str) -> Option<&dyn CodegenBackend> {
        let key = match name.trim().to_ascii_lowercase().as_str() {
            "rs" | "rust" => "rust",
            "c" | "c99" => "c",
            "wgsl" | "webgpu" => "wgsl",
            "cuda" | "cu" | "nvptx" => "cuda",
            "msl" | "metal" => "msl",
            "hip" | "rocm" => "hip",
            "opencl" | "cl" => "opencl",
            other => return self.backends.get(other).map(|b| b.as_ref()),
        };
        self.backends.get(key).map(|b| b.as_ref())
    }

    /// Generates target code for `target_name` using the registered [`CodegenBackend`].
    pub fn generate(
        &self,
        target_name: &str,
        module: &Module,
        config: &TargetConfig,
    ) -> Result<String> {
        let backend = self.get(target_name).ok_or_else(|| {
            anyhow!(
                "Unknown codegen target `{target_name}`. Available targets: {}",
                self.order.join(", ")
            )
        })?;
        backend.emit_module(module, config)
    }

    /// Lists all registered backends as `(name, file_extension, description)`.
    pub fn list(&self) -> impl Iterator<Item = (&'static str, &'static str, &'static str)> + '_ {
        self.order.iter().filter_map(|&k| {
            self.backends
                .get(k)
                .map(|b| (b.name(), b.file_extension(), b.description()))
        })
    }
}

/// Adapter wrapping any [`CodegenBackend`] into a composable [`Translator`] (`Ir -> Text`).
pub struct CustomBackendTranslator {
    stage_name: String,
    backend: Box<dyn CodegenBackend>,
    config: TargetConfig,
}

impl CustomBackendTranslator {
    /// Wraps `backend` and `config` as a [`Translator`] named `codegen-<backend.name()>`.
    pub fn new(backend: Box<dyn CodegenBackend>, config: TargetConfig) -> Self {
        Self {
            stage_name: format!("codegen-{}", backend.name()),
            backend,
            config,
        }
    }
}

impl Translator for CustomBackendTranslator {
    fn name(&self) -> &str {
        &self.stage_name
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        let src = self.backend.emit_module(&module, &self.config)?;
        Ok(Payload::Text(src))
    }
}
