//! Pluggable Multi-Architecture Codegen Backend Trait ([`CodegenBackend`]) and Dynamic Target
//! Registry ([`BackendRegistry`]).
//!
//! Adding a new codegen target to CastForge only requires implementing
//! [`CodegenBackend::emit_from_loop_ir`] over the 4 statements of [`LoopModule`] (`Comment`,
//! `Alloc`, `For`, `Assign`) and registering the struct with [`BackendRegistry::register`].

use crate::loop_ir::{
    LoopModule, LoopStmt, emit_cuda_module_with_config, emit_msl_module_with_config,
    lower_to_loop_ir,
};
use crate::target_c::emit_c_module_with_config;
use crate::target_rust::emit_rust_module_with_config;
use crate::{TargetConfig, build_memory_plan, tensor_len, tensor_shape_dims, value_ident};
use anyhow::{Result, anyhow};
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{DType, Module, OpId, Payload, Translator, ValueId};
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
        Ok(loop_mod.to_string())
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
        Ok(loop_mod.to_string())
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

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, _config: &TargetConfig) -> Result<String> {
        Ok(loop_mod.to_string())
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

    fn emit_from_loop_ir(&self, loop_mod: &LoopModule, _config: &TargetConfig) -> Result<String> {
        Ok(loop_mod.to_string())
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
        Ok(loop_mod.to_string())
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
        let block_dim = config.wgsl_workgroup_size.max(64);
        let mut out = String::new();
        writeln!(out, "// Auto-generated by CastForge AMD ROCm HIP Backend")?;
        writeln!(out, "#include <hip/hip_runtime.h>")?;
        writeln!(out, "#include <math.h>\n")?;

        for kernel in &loop_mod.kernels {
            let mut sig_params: Vec<String> = kernel
                .params
                .iter()
                .map(|(n, _)| format!("const float* __restrict__ {n}"))
                .collect();
            sig_params.push(format!("float* __restrict__ {}", kernel.output.0));
            writeln!(
                out,
                "extern \"C\" __global__ void {}_hip_kernel({}) {{",
                kernel.name,
                sig_params.join(", ")
            )?;
            writeln!(
                out,
                "    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);"
            )?;
            for stmt in &kernel.body {
                emit_c_like_loop_stmt(&mut out, stmt, &kernel.output.0, "tid", 1, true)?;
            }
            writeln!(out, "}}\n")?;

            let mut host_params: Vec<String> = kernel
                .params
                .iter()
                .map(|(n, _)| format!("const float* {n}"))
                .collect();
            host_params.push(format!("float* {}", kernel.output.0));
            let arg_names: Vec<String> = kernel
                .params
                .iter()
                .map(|(n, _)| n.clone())
                .chain(std::iter::once(kernel.output.0.clone()))
                .collect();
            let total = kernel.output.1.max(1);
            writeln!(
                out,
                "extern \"C\" void {}_hip_launch({}) {{",
                kernel.name,
                host_params.join(", ")
            )?;
            writeln!(out, "    const unsigned int threads = {block_dim}u;")?;
            writeln!(
                out,
                "    const unsigned int blocks = ({total}u + threads - 1u) / threads;"
            )?;
            writeln!(
                out,
                "    hipLaunchKernelGGL({}_hip_kernel, dim3(blocks), dim3(threads), 0, 0, {});",
                kernel.name,
                arg_names.join(", ")
            )?;
            writeln!(out, "}}\n")?;
        }
        Ok(out)
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
        let mut out = String::new();
        writeln!(out, "/* Auto-generated by CastForge OpenCL C Backend */")?;

        for kernel in &loop_mod.kernels {
            let mut sig_params: Vec<String> = kernel
                .params
                .iter()
                .map(|(n, _)| format!("__global const float* restrict {n}"))
                .collect();
            sig_params.push(format!("__global float* restrict {}", kernel.output.0));
            writeln!(
                out,
                "__kernel void {}_opencl_kernel({}) {{",
                kernel.name,
                sig_params.join(", ")
            )?;
            writeln!(out, "    const int gid = (int)get_global_id(0);")?;
            for stmt in &kernel.body {
                emit_c_like_loop_stmt(&mut out, stmt, &kernel.output.0, "gid", 1, true)?;
            }
            writeln!(out, "}}\n")?;
        }
        Ok(out)
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

/// WebGPU WGSL Compute Shader Generator (`CodegenTarget::Wgsl`)
pub fn emit_wgsl_module(module: &Module) -> Result<String> {
    emit_wgsl_module_with_config(module, &TargetConfig::default())
}

/// WebGPU WGSL Compute Shader Generator parameterized by `TargetConfig`.
pub fn emit_wgsl_module_with_config(module: &Module, config: &TargetConfig) -> Result<String> {
    let mut out = String::new();
    let plan = build_memory_plan(module);
    writeln!(
        out,
        "// Auto-generated by CastForge Codegen (Target: WebGPU WGSL)"
    )?;
    writeln!(
        out,
        "// Constant Weights: {} B | Activation Workspace: {} B \
         (Naive: {} B, Saved: {} B across {} arena slots)",
        plan.total_constant_bytes,
        plan.total_activation_bytes,
        plan.naive_activation_bytes,
        plan.saved_activation_bytes(),
        plan.reusable_slot_count
    )?;

    if let Some(&fn_op_id) = module.functions().first() {
        emit_wgsl_func(module, fn_op_id, config, &mut out)?;
    }

    Ok(out)
}

fn emit_wgsl_func(
    module: &Module,
    fn_op_id: OpId,
    config: &TargetConfig,
    out: &mut String,
) -> Result<()> {
    let fn_op = &module.ops[fn_op_id];
    let region_id = fn_op.regions[0];
    let block_id = module.regions[region_id].blocks[0];
    let block = &module.blocks[block_id];

    let mut names: HashMap<ValueId, String> = HashMap::new();
    let mut counter = 0usize;

    for (b_idx, &arg_id) in block.arguments.iter().enumerate() {
        let name = format!("input_{b_idx}");
        names.insert(arg_id, name.clone());
        writeln!(
            out,
            "@group(0) @binding({b_idx}) var<storage, read> {name}: array<f32>;"
        )?;
    }
    let out_binding = block.arguments.len();
    writeln!(
        out,
        "@group(0) @binding({out_binding}) var<storage, read_write> output_0: array<f32>;\n"
    )?;

    // Emit module-scope constant arrays (`const<private>` / `var<private>`)
    for &op_id in &block.ops {
        let op = &module.ops[op_id];
        if !op.is_op("stablehlo", "constant") {
            continue;
        }
        let Some(&res_val) = op.results.first() else {
            continue;
        };
        let res_name = value_ident(module, res_val, &mut names, &mut counter);
        let res_len = tensor_len(module, res_val);
        let Some(t_ty) = module.values[res_val].ty.as_tensor() else {
            continue;
        };
        let val_attr = op.attributes.get("value");
        if let DType::QuantUniform { .. } = &t_ty.dtype {
            let q_vals = val_attr
                .and_then(|a| module.tensor_store.resolve_i8_attr(a, res_len))
                .unwrap_or_else(|| vec![0; res_len]);
            let elems: Vec<String> = q_vals.iter().map(|v| format!("{v}i")).collect();
            writeln!(
                out,
                "const {res_name}: array<i32, {res_len}> = array<i32, {res_len}>({});",
                elems.join(", ")
            )?;
        } else {
            let vals = val_attr
                .and_then(|a| module.tensor_store.resolve_f32_attr(a, res_len))
                .unwrap_or_else(|| vec![0.0; res_len]);
            let elems: Vec<String> = vals.iter().map(|v| format!("{v:.6}")).collect();
            writeln!(
                out,
                "const {res_name}: array<f32, {res_len}> = array<f32, {res_len}>({});",
                elems.join(", ")
            )?;
        }
    }

    let wg_size = config.wgsl_workgroup_size.max(1);
    if config.is_tiled() {
        let tile_elems = (config.tile_m * config.tile_k).max(1);
        writeln!(out, "var<workgroup> wg_tile_a: array<f32, {tile_elems}>;")?;
        writeln!(out, "var<workgroup> wg_tile_b: array<f32, {tile_elems}>;")?;
    }
    writeln!(out, "\n@compute @workgroup_size({wg_size})")?;
    writeln!(
        out,
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{"
    )?;
    writeln!(out, "    let idx = gid.x;")?;
    writeln!(out, "    if (idx >= arrayLength(&output_0)) {{ return; }}")?;

    for &op_id in &block.ops {
        let op = &module.ops[op_id];
        if op.is_op("stablehlo", "constant") {
            continue;
        }
        if op.is_op("func", "return") {
            if let Some(&ret_val) = op.operands.first() {
                let r_name = value_ident(module, ret_val, &mut names, &mut counter);
                writeln!(out, "    output_0[idx] = {r_name}[idx];")?;
            }
            continue;
        }

        let Some(&res_val) = op.results.first() else {
            continue;
        };
        let res_name = value_ident(module, res_val, &mut names, &mut counter);
        let res_len = tensor_len(module, res_val);

        if op.is_op("stablehlo", "uniform_dequantize") {
            let src_val = op.operands[0];
            let src_name = value_ident(module, src_val, &mut names, &mut counter);
            let (scale, zp) = match module.values[src_val].ty.as_tensor().map(|t| &t.dtype) {
                Some(DType::QuantUniform {
                    scale, zero_point, ..
                }) => (scale.parse::<f32>().unwrap_or(1.0), *zero_point),
                _ => (1.0, 0),
            };
            writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
            writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
            writeln!(
                out,
                "        {res_name}[i] = f32({src_name}[i] - {zp}i) * {scale:.6};"
            )?;
            writeln!(out, "    }}")?;
            continue;
        }

        if let Some(hlo) = StableHloOp::from_op(op) {
            match hlo {
                StableHloOp::Add { lhs, rhs, .. }
                | StableHloOp::Subtract { lhs, rhs, .. }
                | StableHloOp::Multiply { lhs, rhs, .. }
                | StableHloOp::Divide { lhs, rhs, .. }
                | StableHloOp::Maximum { lhs, rhs, .. }
                | StableHloOp::Minimum { lhs, rhs, .. } => {
                    let a = value_ident(module, lhs, &mut names, &mut counter);
                    let b = value_ident(module, rhs, &mut names, &mut counter);
                    let expr = match op.name.name.as_str() {
                        "add" => format!("{a}[i] + {b}[i]"),
                        "subtract" => format!("{a}[i] - {b}[i]"),
                        "multiply" => format!("{a}[i] * {b}[i]"),
                        "divide" => format!("{a}[i] / {b}[i]"),
                        "maximum" => format!("max({a}[i], {b}[i])"),
                        _ => format!("min({a}[i], {b}[i])"),
                    };
                    writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                    writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                    writeln!(out, "        {res_name}[i] = {expr};")?;
                    writeln!(out, "    }}")?;
                }
                StableHloOp::Exp { operand, .. }
                | StableHloOp::Tanh { operand, .. }
                | StableHloOp::Rsqrt { operand, .. }
                | StableHloOp::Convert { operand, .. }
                | StableHloOp::Reshape { operand, .. } => {
                    let x = value_ident(module, operand, &mut names, &mut counter);
                    let expr = match op.name.name.as_str() {
                        "exponential" => format!("exp({x}[i])"),
                        "tanh" => format!("tanh({x}[i])"),
                        "rsqrt" => format!("inverseSqrt({x}[i])"),
                        _ => format!("{x}[i]"),
                    };
                    writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                    writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                    writeln!(out, "        {res_name}[i] = {expr};")?;
                    writeln!(out, "    }}")?;
                }
                StableHloOp::DotGeneral {
                    lhs,
                    rhs,
                    dimension_numbers: dims,
                    ..
                } => {
                    let packed = dims.rhs_contracting_dimensions == [1];
                    let a = value_ident(module, lhs, &mut names, &mut counter);
                    let b = value_ident(module, rhs, &mut names, &mut counter);
                    let lhs_dims = module.values[lhs]
                        .ty
                        .as_tensor()
                        .map(tensor_shape_dims)
                        .unwrap_or_else(|| vec![1, 1]);
                    let rhs_dims = module.values[rhs]
                        .ty
                        .as_tensor()
                        .map(tensor_shape_dims)
                        .unwrap_or_else(|| vec![1, 1]);
                    let m = *lhs_dims.first().unwrap_or(&1);
                    let k = *lhs_dims.get(1).unwrap_or(&1);
                    let n = if packed {
                        *rhs_dims.first().unwrap_or(&1)
                    } else {
                        *rhs_dims.get(1).unwrap_or(&1)
                    };
                    let b_idx = if packed {
                        format!("n * {k}u + k")
                    } else {
                        format!("k * {n}u + n")
                    };
                    writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                    writeln!(out, "    for (var m = 0u; m < {m}u; m = m + 1u) {{")?;
                    writeln!(out, "        for (var n = 0u; n < {n}u; n = n + 1u) {{")?;
                    writeln!(out, "            var acc: f32 = 0.0;")?;
                    writeln!(out, "            for (var k = 0u; k < {k}u; k = k + 1u) {{")?;
                    writeln!(
                        out,
                        "                acc = acc + {a}[m * {k}u + k] * {b}[{b_idx}];"
                    )?;
                    writeln!(out, "            }}")?;
                    writeln!(out, "            {res_name}[m * {n}u + n] = acc;")?;
                    writeln!(out, "        }}")?;
                    writeln!(out, "    }}")?;
                }
                StableHloOp::Composite { name, inputs, .. } => {
                    if name == "castforge.fused_add_max" && inputs.len() == 3 {
                        let a = value_ident(module, inputs[0], &mut names, &mut counter);
                        let b = value_ident(module, inputs[1], &mut names, &mut counter);
                        let c = value_ident(module, inputs[2], &mut names, &mut counter);
                        writeln!(
                            out,
                            "    // Lowered from stablehlo.composite (castforge.fused_add_max)"
                        )?;
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                        writeln!(out, "        {res_name}[i] = max({a}[i] + {b}[i], {c}[i]);")?;
                        writeln!(out, "    }}")?;
                    } else if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 {
                        let a = value_ident(module, inputs[0], &mut names, &mut counter);
                        let b = value_ident(module, inputs[1], &mut names, &mut counter);
                        let bias = value_ident(module, inputs[2], &mut names, &mut counter);
                        let lhs_dims = module.values[inputs[0]]
                            .ty
                            .as_tensor()
                            .map(tensor_shape_dims)
                            .unwrap_or_else(|| vec![1, 1]);
                        let rhs_dims = module.values[inputs[1]]
                            .ty
                            .as_tensor()
                            .map(tensor_shape_dims)
                            .unwrap_or_else(|| vec![1, 1]);
                        let m = *lhs_dims.first().unwrap_or(&1);
                        let k = *lhs_dims.get(1).unwrap_or(&1);
                        let n = *rhs_dims.get(1).unwrap_or(&1);
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var m = 0u; m < {m}u; m = m + 1u) {{")?;
                        writeln!(out, "        for (var n = 0u; n < {n}u; n = n + 1u) {{")?;
                        writeln!(out, "            var acc: f32 = 0.0;")?;
                        writeln!(out, "            for (var k = 0u; k < {k}u; k = k + 1u) {{")?;
                        writeln!(
                            out,
                            "                acc = acc + {a}[m * {k}u + k] * {b}[k * {n}u + n];"
                        )?;
                        writeln!(out, "            }}")?;
                        writeln!(out, "            let flat = m * {n}u + n;")?;
                        writeln!(
                            out,
                            "            {res_name}[flat] = max(acc + {bias}[flat], 0.0);"
                        )?;
                        writeln!(out, "        }}")?;
                        writeln!(out, "    }}")?;
                    } else if name == "castforge.fused_swiglu" && inputs.len() >= 2 {
                        let g = value_ident(module, inputs[0], &mut names, &mut counter);
                        let u = value_ident(module, inputs[1], &mut names, &mut counter);
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                        writeln!(
                            out,
                            "        {res_name}[i] = ({g}[i] / (1.0 + exp(-{g}[i]))) * {u}[i];"
                        )?;
                        writeln!(out, "    }}")?;
                    } else if name == "castforge.fused_rope" && inputs.len() >= 3 {
                        let x = value_ident(module, inputs[0], &mut names, &mut counter);
                        let c_arr = value_ident(module, inputs[1], &mut names, &mut counter);
                        let s_arr = value_ident(module, inputs[2], &mut names, &mut counter);
                        let dims = module.values[res_val]
                            .ty
                            .as_tensor()
                            .map(tensor_shape_dims)
                            .unwrap_or_else(|| vec![1, res_len]);
                        let cols = *dims.last().unwrap_or(&res_len).max(&2);
                        let rows = (res_len / cols).max(1);
                        let half = cols / 2;
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var r = 0u; r < {rows}u; r = r + 1u) {{")?;
                        writeln!(out, "        for (var c = 0u; c < {cols}u; c = c + 1u) {{")?;
                        writeln!(out, "            let idx = r * {cols}u + c;")?;
                        writeln!(
                            out,
                            "            let rot = select(\
                             {x}[idx - {half}u], -{x}[idx + {half}u], c < {half}u);"
                        )?;
                        writeln!(
                            out,
                            "            {res_name}[idx] = \
                             {x}[idx] * {c_arr}[idx] + rot * {s_arr}[idx];"
                        )?;
                        writeln!(out, "        }}")?;
                        writeln!(out, "    }}")?;
                    } else if name == "castforge.fused_gelu" && !inputs.is_empty() {
                        let x = value_ident(module, inputs[0], &mut names, &mut counter);
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                        writeln!(out, "        let v = {x}[i];")?;
                        writeln!(
                            out,
                            "        {res_name}[i] = 0.5 * v * \
                             (1.0 + tanh(0.7978846 * (v + 0.044715 * v * v * v)));"
                        )?;
                        writeln!(out, "    }}")?;
                    } else if name == "castforge.fused_geglu" && inputs.len() >= 2 {
                        let g = value_ident(module, inputs[0], &mut names, &mut counter);
                        let u = value_ident(module, inputs[1], &mut names, &mut counter);
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                        writeln!(out, "        let v = {g}[i];")?;
                        writeln!(
                            out,
                            "        let gv = 0.5 * v * \
                             (1.0 + tanh(0.7978846 * (v + 0.044715 * v * v * v)));"
                        )?;
                        writeln!(out, "        {res_name}[i] = gv * {u}[i];")?;
                        writeln!(out, "    }}")?;
                    } else {
                        let a = value_ident(module, inputs[0], &mut names, &mut counter);
                        writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                        writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                        writeln!(out, "        {res_name}[i] = {a}[i];")?;
                        writeln!(out, "    }}")?;
                    }
                }
                _ => {
                    let a = op
                        .operands
                        .first()
                        .map(|&v| value_ident(module, v, &mut names, &mut counter))
                        .unwrap_or_else(|| "input_0".to_string());
                    writeln!(out, "    var {res_name}: array<f32, {res_len}>;")?;
                    writeln!(out, "    for (var i = 0u; i < {res_len}u; i = i + 1u) {{")?;
                    writeln!(out, "        {res_name}[i] = {a}[i];")?;
                    writeln!(out, "    }}")?;
                }
            }
        }
    }

    writeln!(out, "}}")?;
    Ok(())
}
