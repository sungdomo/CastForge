//! CastForge: Pure-Rust AI Model IR Transformation, Optimization, Visualization, and Code
//! Generation Suite.

pub mod cli;
pub mod codegen;
pub mod ir;
pub mod passes;

pub use ir::{formats, parser, printer, stablehlo, viz};
pub use passes::verify_ir as verify;
