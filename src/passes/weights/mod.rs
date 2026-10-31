//! Unified compile-time weight folding, pre-packing, SafeTensors externalization,
//! structured/2:4 sparsity pruning, checkpoint merging, and LoRA/DoRA adapter baking
//! (`fold-weights`, `externalize-weights`, `prune-weights`, `merge-weights`, `bake-lora`,
//! `bake-dora`).

pub mod fold;
pub mod lora;
pub mod merge;
pub mod prune;

pub use fold::*;
pub use lora::*;
pub use merge::*;
pub use prune::*;
