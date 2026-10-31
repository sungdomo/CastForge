//! CastForge: Zero-C++ MLIR & StableHLO compiler in pure Rust with 115+ graph optimization,
//! fusion, quantization, and weight-surgery passes and standalone CPU/GPU codegen.

pub mod cli;
pub mod codegen;
pub mod ffi;
pub mod ir;
pub mod passes;

pub use ir::{formats, parser, printer, stablehlo, viz};
pub use passes::normalize::verify;
