//! Polyhedral, SIMD, Convolution Lowering, and Tensor-Core loop scheduling (`schedule-loops`).

pub mod schedule_loops;

pub use schedule_loops::{
    LowerConvToGemmPass, LowerTensorCoresPass, PadForTilingPass, PermuteLoopsPass,
    ScheduleLoopsPass, SplitLargeGemmPass, TileLoopsPass, UnrollLoopsPass, VectorizeLoopsPass,
};
