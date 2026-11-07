//! Built-in [`PassRegistry`], [`PassManager`], and optimization presets (`src/passes/registry`).

pub mod families;
mod leaf;
mod lowering;
pub mod manager;
pub mod presets;

pub use manager::{
    Pass, PassCatalog, PassCatalogEntry, PassDescriptor, PassFactory, PassManager, PassOptionSpec,
    PassOptions, PassRegistry, PassResult, PassStepConfig, PassTelemetry, PassTraceStep,
    PassTranslator, PipelineConfig, PipelineSettings, parse_pass_spec_item, pass_catalog,
    split_pipeline_spec,
};
pub use presets::{OptO1Pass, OptO2Pass, OptO3Pass, OptimizeServingPass};
