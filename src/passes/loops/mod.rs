//! Unified Polyhedral, SIMD, Convolution Lowering, Tensor-Core Loop Scheduling & Loop-IR Lowering
//! (`schedule-loops`, `lower-conv-to-gemm`, `lower-to-loops`, `optimize-loop-ir`).

pub mod gemm_conv;
pub mod lower;
pub mod opt;
pub mod schedule;

pub use gemm_conv::{LowerConvToGemmPass, PadForTilingPass, SplitLargeGemmPass};
pub use lower::{TensorToLoopTranslator, lower_to_loop_ir};
pub use opt::{LoopPass, OptimizeLoopIrPass, OptimizeLoopIrTranslator};
pub use schedule::{
    LowerTensorCoresPass, PermuteLoopsPass, ScheduleLoopsPass, TileLoopsPass, UnrollLoopsPass,
    VectorizeLoopsPass,
};
