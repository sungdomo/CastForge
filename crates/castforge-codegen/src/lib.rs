pub mod autotune;
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
pub use crate::cost_model::{
    BottleneckKind, HardwareRoofline, ModuleCostReport, OpCostProfile, analyze_module_cost,
    analyze_module_cost_with_roofline,
};
pub use crate::eval::{
    NumericVerificationReport, evaluate_function, evaluate_module, generate_synthetic_inputs,
    verify_module_numerics,
};
pub use crate::jit::{JitBenchmarkResult, autotune_module_jit, jit_execute_and_benchmark};
pub use crate::liveness::{LiveInterval, compute_block_liveness, compute_module_liveness};
pub use crate::loop_ir::{
    CHeaderTranslator, LoopKernel, LoopModule, LoopStmt, SharedLibTranslator,
    TensorToLoopTranslator, compile_shared_lib, emit_c_header, emit_cuda_module_with_config,
    emit_msl_module_with_config, lower_to_loop_ir,
};
pub use crate::memory_plan::{BufferAllocation, MemoryPlan, build_memory_plan};
use crate::target_c::emit_c_module_with_config;
pub use crate::target_config::{GemmLoopOrder, SimdIsa, TargetConfig};
use crate::target_rust::emit_rust_module_with_config;
use crate::target_wgsl::emit_wgsl_module_with_config;
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
}

/// Generate executable source code from a CastForge `Module` for the requested `CodegenTarget`.
pub fn generate_code(module: &Module, target: CodegenTarget) -> Result<String> {
    generate_code_with_config(module, target, &TargetConfig::default())
}

/// Generate executable source code parameterized by hardware `TargetConfig` (tiling, SIMD, WGSL).
pub fn generate_code_with_config(
    module: &Module,
    target: CodegenTarget,
    config: &TargetConfig,
) -> Result<String> {
    match target {
        CodegenTarget::Rust => emit_rust_module_with_config(module, config),
        CodegenTarget::C => emit_c_module_with_config(module, config),
        CodegenTarget::Wgsl => emit_wgsl_module_with_config(module, config),
        CodegenTarget::Cuda => emit_cuda_module_with_config(module, config),
        CodegenTarget::Msl => emit_msl_module_with_config(module, config),
    }
}

/// A-to-B Backend Codegen Translator (`Ir(Module) -> Text(Rust | C99 | WGSL | CUDA | MSL)`).
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
