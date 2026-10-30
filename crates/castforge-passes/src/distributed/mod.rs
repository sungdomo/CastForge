//! Distributed parallelism, collective coalescing, and stream scheduling (`shard-distributed`).

pub mod autodiff;
pub mod shard_distributed;

pub use autodiff::AutodiffPass;
pub use shard_distributed::{
    HoistAllReducePass, ScheduleStreamsPass, ShardDistributedPass, ShardPipelineParallelPass,
    ShardRingAttentionPass, ShardSpmdPass,
};
