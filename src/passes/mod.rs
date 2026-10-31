//! Built-in A-to-B IR optimization, fusion, quantization, loop-scheduling, and surgery passes
//! (`castforge::passes`).
//!
//! Core pass execution and verification infrastructure ([`Pass`], [`PassManager`],
//! [`PassRegistry`], [`VerifyIrPass`]) lives in [`crate::ir`] and is re-exported here for
//! ergonomics.

pub mod canonicalize;
pub mod distributed;
pub mod fold;
pub mod fuse;
pub mod loops;
pub mod manager;
pub mod memory;
pub mod normalize;
pub mod quantize;
pub mod registry;
pub mod shapes;

pub use canonicalize::*;
pub use distributed::*;
pub use fold::*;
pub use fuse::*;
pub use loops::*;
pub use manager::*;
pub use memory::*;
pub use normalize::*;
pub use quantize::*;
pub use shapes::*;
