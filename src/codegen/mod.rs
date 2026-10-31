//! Liveness analyzer, interval-coloring arena [`MemoryPlan`], progressive [`LoopModule`] IR,
//! Roofline & native JIT schedule auto-tuner, reference IR interpreter, and multi-target code
//! generator (Rust, C99, WebGPU WGSL, NVIDIA CUDA, Apple Metal, C/C++ header, and native shared
//! library) (`castforge::codegen`).

pub mod autotune;
pub mod backend;
pub mod config;
pub mod cost_model;
pub mod eval;
pub mod memory_plan;
pub mod targets;

pub use crate::ir::loops::{LoopKernel, LoopModule, LoopStmt};
pub use crate::passes::loops::{
    LoopPass, OptimizeLoopIrPass, OptimizeLoopIrTranslator, TensorToLoopTranslator,
    lower_to_loop_ir,
};
pub use autotune::{
    AutotuneCandidate, AutotuneReport, AutotuneScheduleTranslator, JitBenchmarkResult,
    apply_schedule_to_module, autotune_module, autotune_module_jit, jit_execute_and_benchmark,
};
pub use backend::{
    BackendRegistry, C99Backend, CodegenBackend, CudaBackend, CustomBackendTranslator, HipBackend,
    MslBackend, OpenClBackend, RustBackend, WgslBackend, emit_c_like_loop_stmt,
};
pub use config::*;
pub use cost_model::{
    BottleneckKind, GemmLoopOrder, HardwareRoofline, ModuleCostReport, OpCostProfile, SimdIsa,
    TargetConfig, analyze_module_cost, analyze_module_cost_with_roofline,
};
pub use eval::{
    BisectReport, BisectStep, NumericVerificationReport, bisect_module_snapshots,
    compare_output_tensors, evaluate_function, evaluate_module, generate_synthetic_inputs,
    verify_module_numerics,
};
pub use memory_plan::{
    BufferAllocation, LiveInterval, MemoryPlan, build_function_memory_plan, build_memory_plan,
    compute_block_liveness, compute_module_liveness,
};
pub use targets::{
    CHeaderTranslator, SharedLibTranslator, compile_shared_lib, emit_c_from_loop_ir, emit_c_header,
    emit_c_module, emit_c_module_with_config, emit_cuda_from_loop_ir, emit_cuda_module_with_config,
    emit_hip_from_loop_ir, emit_hip_module_with_config, emit_msl_from_loop_ir,
    emit_msl_module_with_config, emit_opencl_from_loop_ir, emit_opencl_module_with_config,
    emit_rust_from_loop_ir, emit_rust_module, emit_rust_module_with_config, emit_wgsl_from_loop_ir,
    emit_wgsl_module, emit_wgsl_module_with_config,
};
