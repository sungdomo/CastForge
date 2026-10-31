//! Built-in [`PassRegistry`], [`PassManager`], and optimization presets (`src/passes/registry`).

pub mod families;
mod leaf;
mod lowering;
pub mod manager;
pub mod presets;

pub use manager::*;
pub use presets::*;
