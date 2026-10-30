//! Compiler diagnostics, IR surgery, subgraph outlining, verification, and profiling passes.

pub mod ir_edit;
pub mod outline_subgraphs;
pub mod verify_ir;

pub use ir_edit::{
    FreezeInputsPass, IrEditPass, NormalizeIrPass, NormalizeSsaPass, ParameterizeConstantsPass,
    StripAttributesPass, StripDebugInfoPass, StripMetadataPass, TapIntermediatesPass,
    UnifiedFreezeInputsPass,
};
pub use outline_subgraphs::{
    ComposeFuncsPass, DedupFunctionsPass, ExtractSubgraphPass, InlinerPass, OutlineSubgraphsPass,
    ReduceIrPass, UnifiedOutlineSubgraphsPass, UnrollCallsPass, link_modules,
};
pub use verify_ir::{
    BisectPassesPass, HashIrPass, InstrumentProfilingPass, PrintOpStatsPass, RooflineSummary,
    SanitizeNumericsPass, UnifiedProfileIrPass, UnifiedVerifyIrPass, VerifyEquivPass,
    VerifyShapesPass, analyze_roofline, compute_structural_hash,
};
