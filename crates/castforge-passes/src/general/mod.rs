//! General compiler optimization passes (DCE, CSE, algebraic simplification, constant folding,
//! canonicalization, optimization presets, and topological sorting).

pub mod canonicalize;
pub mod cse;
pub mod dce;
pub mod simplify_algebraic;

pub use canonicalize::{CanonicalizePass, OptO1Pass, OptO2Pass, OptO3Pass, TopologicalSortPass};
pub use cse::{
    CommonSubexpressionEliminationPass, GlobalValueNumberingPass, HoistConstantsPass, LicmPass,
    UnifiedCsePass,
};
pub use dce::{DeadCodeEliminationPass, SymbolDcePass};
pub use simplify_algebraic::{
    EgraphSimplifyPass, FoldConstantsPass, SimplifyAlgebraicPass, UnifiedSimplifyAlgebraicPass,
};
