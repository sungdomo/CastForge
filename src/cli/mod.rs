//! Command-line interface (`castforge`) for the CastForge AI Model IR Transformation,
//! Optimization, Visualization, and Code Generation Suite.

pub mod args;
pub mod chain;
pub mod commands;

pub use args::{Cli, Commands};
pub use chain::build_translator_chain;
pub use commands::run;
