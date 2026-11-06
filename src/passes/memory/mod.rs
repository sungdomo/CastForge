//! Unified Memory Optimization, In-Place Bufferization, Rematerialization & Host Offloading
//! (`optimize-memory`, `rematerialize`, `plan-memory-arena`).

pub mod bufferize;
pub mod remat;

pub use bufferize::{BufferizeInplacePass, EliminateConcatCopiesPass, OptimizeMemoryPass};
pub use remat::{OffloadActivationsPass, RematerializePass};
