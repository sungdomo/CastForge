//! Unified operator fusion and composite decomposition passes (`fuse-attention`,
//! `fuse-elementwise`, `fuse-gemm`, `fuse-norm`, `fuse-fma`, `decompose-composites`).

pub mod activations;
pub mod attention;
pub mod decompose;
pub mod decompose_helpers;
pub mod elementwise;
pub mod epilogue;
pub mod gemm;
pub mod layernorm_softmax;
pub mod loops;
pub mod norm;
pub mod serving;

pub use activations::*;
pub use attention::*;
pub use decompose::*;
pub use elementwise::*;
pub use gemm::*;
pub use layernorm_softmax::*;
pub use loops::*;
pub use norm::*;
pub use serving::*;
