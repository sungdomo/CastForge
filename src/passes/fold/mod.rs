//! Unified compile-time constant, view/layout, weight/pruning, and adapter/checkpoint folding
//! passes (`fold-constants`, `fold-views`, `convert-layout`, `fold-weights`, `externalize-weights`,
//! `prune-weights`, `merge-weights`, `bake-lora`, `bake-dora`).

pub mod constants;
pub mod layout;
pub mod lora;
pub mod merge;
pub mod prune;
pub mod views;
pub mod weights;

pub use constants::*;
pub use layout::*;
pub use lora::*;
pub use merge::*;
pub use prune::*;
pub use views::*;
pub use weights::*;
