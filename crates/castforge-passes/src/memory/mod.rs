//! Memory planning, in-place bufferization, zero-copy concat, rematerialization, and offloading
//! (`optimize-memory`).

pub mod optimize_memory;

pub use optimize_memory::{
    BufferizeInplacePass, EliminateConcatCopiesPass, OffloadActivationsPass, OptimizeMemoryPass,
    RematerializePass,
};
