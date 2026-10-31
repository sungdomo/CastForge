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

pub use algebraic::*;
pub use constants::*;
pub use cse::*;
pub use dce::*;
pub use driver::*;
pub use layout::*;
pub use views::*;
