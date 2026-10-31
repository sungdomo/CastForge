//! Unified Distributed Parallelism, Collectives, Stream Scheduling & Autodiff
//! (`shard-distributed`, `autodiff-backward`).

pub mod autodiff;
pub mod spmd;

pub use autodiff::*;
pub use spmd::*;
