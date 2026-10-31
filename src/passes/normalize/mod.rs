//! Unified IR Normalization, Subgraph Outlining/Inlining/Extraction, Input Freezing,
//! Parameterization, Activation Tapping & Verification (`normalize-ir`, `verify-ir`).

pub mod inliner;
pub mod outline;
pub mod surgery;
pub mod verify;

pub use inliner::*;
pub use outline::*;
pub use surgery::*;
pub use verify::*;
