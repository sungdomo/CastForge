//! Compiler diagnostics, IR surgery, subgraph outlining, verification, and profiling passes.

pub mod ir_edit;
pub mod outline_subgraphs;
pub mod profile_ir;
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
pub use profile_ir::{
    InstrumentProfilingPass, PrintOpStatsPass, RooflineSummary, UnifiedProfileIrPass,
    analyze_roofline,
};
pub use verify_ir::{
    BisectPassesPass, HashIrPass, SanitizeNumericsPass, UnifiedVerifyIrPass, VerifyEquivPass,
    VerifyShapesPass, compute_structural_hash,
};
