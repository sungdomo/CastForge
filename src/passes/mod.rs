//! Built-in A-to-B IR optimization, fusion, quantization, loop-scheduling, and surgery passes
//! (`castforge::passes`).
//!
//! Core pass execution and verification infrastructure ([`Pass`], [`PassManager`],
//! [`PassRegistry`], [`VerifyIrPass`]) lives in [`crate::ir`] and is re-exported here for
//! ergonomics.

pub mod canonicalize;
pub mod cse;
pub mod dce;
pub mod fold_views;
pub mod fold_weights;
pub mod fuse_attention;
pub mod fuse_elementwise;
pub mod fuse_gemm;
pub mod fuse_norm;
pub mod merge_weights;
pub mod normalize_ir;
pub mod optimize_memory;
pub mod quantize;
pub mod registry;
pub mod schedule_loops;
pub mod shard_distributed;
pub mod simplify_algebraic;
pub mod specialize_shapes;

pub use crate::ir::{pass_manager::*, verify::*};
pub use canonicalize::*;
pub use cse::*;
pub use dce::*;
pub use fold_views::*;
pub use fold_weights::*;
pub use fuse_attention::*;
pub use fuse_elementwise::*;
pub use fuse_gemm::*;
pub use fuse_norm::*;
pub use merge_weights::*;
pub use normalize_ir::*;
pub use optimize_memory::*;
pub use quantize::*;
pub use schedule_loops::*;
pub use shard_distributed::*;
pub use simplify_algebraic::*;
pub use specialize_shapes::*;
