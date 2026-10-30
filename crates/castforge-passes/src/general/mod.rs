//! General compiler optimization passes (DCE, CSE, algebraic simplification, constant folding,
//! inlining, LICM, canonicalization, optimization presets, and topological sorting).

pub mod canonicalize;
pub mod cse;
pub mod dce;
pub mod fold_constants;
pub mod inliner;
pub mod licm;
pub mod simplify_algebraic;
pub mod topological_sort;

pub use canonicalize::{CanonicalizePass, OptO1Pass, OptO2Pass, OptO3Pass};
pub use cse::{
    CommonSubexpressionEliminationPass, GlobalValueNumberingPass, HoistConstantsPass,
    UnifiedCsePass,
};
pub use dce::{DeadCodeEliminationPass, SymbolDcePass};
pub use fold_constants::FoldConstantsPass;
pub use inliner::InlinerPass;
pub use licm::LicmPass;
pub use simplify_algebraic::{
    EgraphSimplifyPass, SimplifyAlgebraicPass, UnifiedSimplifyAlgebraicPass,
};
pub use topological_sort::TopologicalSortPass;
