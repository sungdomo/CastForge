//! Unified compile-time constant, view/layout, and weight/pruning folding passes
//! (`fold-constants`, `fold-views`, `convert-layout`, `fold-weights`, `externalize-weights`,
//! `prune-weights`).

pub mod constants;
pub mod layout;
pub mod prune;
pub mod views;
pub mod weights;

pub use constants::*;
pub use layout::*;
pub use prune::*;
pub use views::*;
pub use weights::*;
