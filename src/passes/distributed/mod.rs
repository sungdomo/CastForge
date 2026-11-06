//! Unified Distributed Parallelism, Collectives, Stream Scheduling & Autodiff
//! (`shard-distributed`, `autodiff-backward`).

pub mod autodiff;
pub mod spmd;

pub use autodiff::AutodiffPass;
pub use spmd::{
    CoalesceAllReducePass, ScheduleStreamsPass, ShardDistributedPass, ShardExpertsPass,
    ShardFsdpPass, ShardPipelineParallelPass, ShardRingAttentionPass, ShardSpmdPass,
};
