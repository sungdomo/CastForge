//! Target-specific code emitters (`c`, `gpu_c`, `rust`, `wgsl`).

pub mod c;
pub mod gpu_c;
pub mod rust;
pub mod wgsl;

pub use c::{
    CHeaderTranslator, SharedLibTranslator, compile_shared_lib, emit_c_from_loop_ir, emit_c_header,
    emit_c_module, emit_c_module_with_config,
};
pub use gpu_c::{
    emit_cuda_from_loop_ir, emit_cuda_module_with_config, emit_hip_from_loop_ir,
    emit_hip_module_with_config, emit_msl_from_loop_ir, emit_msl_module_with_config,
    emit_opencl_from_loop_ir, emit_opencl_module_with_config,
};
pub use rust::{emit_rust_from_loop_ir, emit_rust_module, emit_rust_module_with_config};
pub use wgsl::{
    emit_wgsl_from_loop_ir, emit_wgsl_from_loop_ir_with_workgroup, emit_wgsl_module,
    emit_wgsl_module_with_config,
};
