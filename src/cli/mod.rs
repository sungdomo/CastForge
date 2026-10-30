//! Command-line interface (`castforge`) for the CastForge AI Model IR Transformation,
//! Optimization, Visualization, and Code Generation Suite.

pub mod args;
pub mod chain;
pub mod commands;
pub mod inspect;

pub use args::{Cli, Commands};
pub use chain::{
    LoadWeightsTranslator, build_translator_chain, build_translator_chain_with_weights,
};
pub use commands::run;
pub use inspect::{
    ArchProfileInfo, FunctionSummary, ModuleInspectionReport, TargetBackendInfo,
    TranslatorStageInfo, WeightTensorSummary, inspect_module, run_inspect, run_list_targets,
};
