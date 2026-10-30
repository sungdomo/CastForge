//! Liveness analyzer, interval-coloring arena [`MemoryPlan`], progressive [`LoopModule`] IR,
//! Roofline & native JIT schedule auto-tuner, reference IR interpreter, and multi-target code
//! generator (Rust, C99, WebGPU WGSL, NVIDIA CUDA, Apple Metal, C/C++ header, and native shared
//! library) (`castforge-codegen`).

pub mod autotune;
pub mod backend;
pub mod cost_model;
pub mod eval;
pub mod jit;
pub mod liveness;
pub mod loop_ir;
pub mod memory_plan;
pub mod target_c;
pub mod target_config;
pub mod target_rust;
pub mod target_wgsl;

pub use crate::autotune::{
    AutotuneCandidate, AutotuneReport, AutotuneScheduleTranslator, apply_schedule_to_module,
    autotune_module,
};
pub use crate::backend::{
    BackendRegistry, C99Backend, CodegenBackend, CudaBackend, CustomBackendTranslator, HipBackend,
    MslBackend, OpenClBackend, RustBackend, WgslBackend, emit_c_like_loop_stmt,
};
pub use crate::cost_model::{
    BottleneckKind, HardwareRoofline, ModuleCostReport, OpCostProfile, analyze_module_cost,
    analyze_module_cost_with_roofline,
};
pub use crate::eval::{
    BisectReport, BisectStep, NumericVerificationReport, bisect_module_snapshots,
    evaluate_function, evaluate_module, generate_synthetic_inputs, verify_module_numerics,
};
pub use crate::jit::{JitBenchmarkResult, autotune_module_jit, jit_execute_and_benchmark};
pub use crate::liveness::{LiveInterval, compute_block_liveness, compute_module_liveness};
pub use crate::loop_ir::{
    CHeaderTranslator, LoopKernel, LoopModule, LoopPass, LoopStmt, OptimizeLoopIrPass,
    SharedLibTranslator, TensorToLoopTranslator, compile_shared_lib, emit_c_header,
    emit_cuda_module_with_config, emit_msl_module_with_config, lower_to_loop_ir,
};
pub use crate::memory_plan::{BufferAllocation, MemoryPlan, build_memory_plan};
pub use crate::target_config::{GemmLoopOrder, SimdIsa, TargetConfig};
use anyhow::Result;
use castforge_ir::{Dim, Module, TensorType, ValueId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodegenTarget {
    Rust,
    C,
    Wgsl,
    Cuda,
    Msl,
    Hip,
    OpenCl,
}

impl CodegenTarget {
    /// Parses a target identifier string (`"rust"`, `"c"`, `"wgsl"`, `"cuda"`, `"msl"`, `"hip"`,
    /// `"opencl"`).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "c" | "c99" => Self::C,
            "wgsl" | "webgpu" => Self::Wgsl,
            "cuda" | "cu" | "nvptx" => Self::Cuda,
            "msl" | "metal" => Self::Msl,
            "hip" | "rocm" => Self::Hip,
            "opencl" | "cl" => Self::OpenCl,
            _ => Self::Rust,
        }
    }

    /// Returns the canonical lowercase target name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::C => "c",
            Self::Wgsl => "wgsl",
            Self::Cuda => "cuda",
            Self::Msl => "msl",
            Self::Hip => "hip",
            Self::OpenCl => "opencl",
        }
    }
}

/// Generate executable source code from a CastForge `Module` for the requested `CodegenTarget`.
pub fn generate_code(module: &Module, target: CodegenTarget) -> Result<String> {
    generate_code_with_config(module, target, &TargetConfig::default())
}

/// Generate executable source code parameterized by `TargetConfig` via [`BackendRegistry`].
pub fn generate_code_with_config(
    module: &Module,
    target: CodegenTarget,
    config: &TargetConfig,
) -> Result<String> {
    let reg = BackendRegistry::with_builtin_backends();
    reg.generate(target.as_str(), module, config)
}

/// A-to-B Backend Codegen Translator (`Ir(Module) -> Text(Rust|C99|WGSL|CUDA|MSL|HIP|OpenCL)`).
pub struct CodegenTranslator {
    pub target: CodegenTarget,
    pub config: TargetConfig,
}

impl CodegenTranslator {
    /// Creates a [`CodegenTranslator`] for `target` with default [`TargetConfig`].
    pub fn new(target: CodegenTarget) -> Self {
        Self {
            target,
            config: TargetConfig::default(),
        }
    }

    /// Creates a [`CodegenTranslator`] for `target` and `config`.
    pub fn with_config(target: CodegenTarget, config: TargetConfig) -> Self {
        Self { target, config }
    }
}

impl castforge_ir::Translator for CodegenTranslator {
    fn name(&self) -> &str {
        match self.target {
            CodegenTarget::Rust => "codegen-rust",
            CodegenTarget::C => "codegen-c",
            CodegenTarget::Wgsl => "codegen-wgsl",
            CodegenTarget::Cuda => "codegen-cuda",
            CodegenTarget::Msl => "codegen-msl",
            CodegenTarget::Hip => "codegen-hip",
            CodegenTarget::OpenCl => "codegen-opencl",
        }
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> Result<castforge_ir::Payload> {
        let module = input.into_ir()?;
        let code = generate_code_with_config(&module, self.target, &self.config)?;
        Ok(castforge_ir::Payload::Text(code))
    }
}

/// A-to-B Roofline Cost Report Translator (`Ir(Module) -> Text(JSON)`).
#[derive(Default)]
pub struct RooflineReportTranslator {
    pub roofline: HardwareRoofline,
}

impl castforge_ir::Translator for RooflineReportTranslator {
    fn name(&self) -> &str {
        "roofline-report"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> Result<castforge_ir::Payload> {
        let module = input.into_ir()?;
        let report = analyze_module_cost_with_roofline(&module, &self.roofline);
        let json = serde_json::to_string_pretty(&report)?;
        Ok(castforge_ir::Payload::Text(json))
    }
}

/// A-to-B Static Memory Plan Translator (`Ir(Module) -> Text(JSON)`).
#[derive(Default)]
pub struct MemoryPlanTranslator;

impl castforge_ir::Translator for MemoryPlanTranslator {
    fn name(&self) -> &str {
        "memory-plan"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> Result<castforge_ir::Payload> {
        let module = input.into_ir()?;
        let plan = build_memory_plan(&module);
        let json = serde_json::to_string_pretty(&plan)?;
        Ok(castforge_ir::Payload::Text(json))
    }
}

pub(crate) fn sanitize_ident(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "v_anon".to_string()
    } else if cleaned.chars().next().unwrap().is_ascii_digit() {
        format!("v_{cleaned}")
    } else {
        cleaned
    }
}

pub(crate) fn value_ident(
    module: &Module,
    val_id: ValueId,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
) -> String {
    if let Some(existing) = names.get(&val_id) {
        return existing.clone();
    }
    let ident = if let Some(ssa) = &module.values[val_id].ssa_name {
        sanitize_ident(ssa)
    } else {
        *counter += 1;
        format!("v_{counter}")
    };
    names.insert(val_id, ident.clone());
    ident
}

pub(crate) fn tensor_len(module: &Module, val_id: ValueId) -> usize {
    module.values[val_id]
        .ty
        .as_tensor()
        .and_then(|t| t.num_elements())
        .unwrap_or(1)
        .max(1) as usize
}

pub(crate) fn tensor_shape_dims(t: &TensorType) -> Vec<usize> {
    t.shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) if *n > 0 => *n as usize,
            _ => 1,
        })
        .collect()
}
