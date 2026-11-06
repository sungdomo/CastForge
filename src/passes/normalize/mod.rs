//! Unified IR Normalization, Subgraph Outlining/Inlining/Extraction, Input Freezing,
//! Parameterization, Activation Tapping & Verification (`normalize-ir`, `verify-ir`).

pub mod inliner;
pub mod outline;
pub mod surgery;
pub mod verify;

pub use inliner::{DedupFunctionsPass, ExtractSubgraphPass, InlineCallsPass, ReduceIrPass};
pub use outline::{
    ComposeFunctionsPass, OutlineKernelsPass, OutlineSubgraphsPass, UnrollCallsPass, link_modules,
};
pub use surgery::{
    FreezeInputsPass, NormalizeIrPass, NormalizeSsaPass, ParameterizeConstantsPass,
    StripAttributesPass, StripDebugInfoPass, TapIntermediatesPass,
};
pub use verify::{
    BisectPassesPass, HashIrPass, InstrumentProfilingPass, PrintOpStatsPass, RooflineSummary,
    SanitizeNumericsPass, VerifyEquivPass, VerifyIrPass, VerifyShapesPass, analyze_roofline,
    compute_structural_hash,
};
