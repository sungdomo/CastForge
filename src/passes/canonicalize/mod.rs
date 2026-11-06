//! Composite Canonicalization (`canonicalize`), CSE/GVN/LICM (`cse`), Dead Code Elimination
//! (`dce`), Algebraic Simplification (`simplify-algebraic`), Constant Folding (`fold-constants`),
//! View/Layout Folding (`fold-views`, `convert-layout`), and Topological / Peak-Memory Scheduling.

pub mod algebraic;
pub mod constants;
pub mod cse;
pub mod dce;
pub mod driver;
pub mod layout;
pub mod views;

pub use algebraic::SimplifyAlgebraicPass;
pub use constants::FoldConstantsPass;
pub use cse::{CsePass, HoistConstantsPass, HoistLoopInvariantsPass};
pub use dce::DeadCodeEliminationPass;
pub use driver::{CanonicalizePass, TopologicalSortPass};
pub use layout::ConvertLayoutPass;
pub use views::FoldViewsPass;
pub(crate) use views::fold_all_views_single_pass;
