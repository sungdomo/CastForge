//! Symbolic & Dynamic Shape Specialization, Batch Resizing, and Shape Bucketing
//! (`specialize-shapes`, `bucket-shapes`).

pub mod bucketing;
pub mod specialize;

pub use bucketing::{BucketShapesPass, ResizeBatchPass};
pub use specialize::{PruneTokensPass, SpecializeShapesPass};
pub(crate) use specialize::{
    collect_non_batch_weight_operands, specialize_and_resize_single_traversal,
};
