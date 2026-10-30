//! Compiler diagnostics, IR surgery, subgraph outlining, normalization, verification, and profiling
//! passes.

pub mod extract_subgraph;
pub mod ir_edit;
pub mod normalize_ir;
pub mod outline_subgraphs;
pub mod profile_ir;
pub mod verify_ir;

pub use extract_subgraph::{ExtractSubgraphPass, ReduceIrPass};
pub use ir_edit::{
    FreezeInputsPass, IrEditPass, ParameterizeConstantsPass, TapIntermediatesPass,
    UnifiedFreezeInputsPass,
};
pub use normalize_ir::{
    NormalizeIrPass, NormalizeSsaPass, StripAttributesPass, StripDebugInfoPass, StripMetadataPass,
};
pub use outline_subgraphs::{
    ComposeFuncsPass, DedupFunctionsPass, OutlineSubgraphsPass, UnifiedOutlineSubgraphsPass,
    UnrollCallsPass, link_modules,
};
pub use profile_ir::{
    InstrumentProfilingPass, PrintOpStatsPass, RooflineSummary, UnifiedProfileIrPass,
    analyze_roofline,
};
pub use verify_ir::{
    BisectPassesPass, HashIrPass, SanitizeNumericsPass, UnifiedVerifyIrPass, VerifyEquivPass,
    VerifyShapesPass, compute_structural_hash,
};
