//! Serving runtime subsystems and graph compiler optimization passes.

pub mod cacheblend;
pub mod collectives;
pub mod dependencies;
pub mod device_graphs;
pub mod dialect_lowering;
pub mod effect_chains;
pub mod fleet_directory;
pub mod graph_blobs_and_poison;
pub mod graph_finalization;
pub mod index_views;
pub mod kernel_stubs;
pub mod kv_attention;
pub mod kv_pools;
pub mod loop_buffers;
pub mod map_input_output_fusion;
pub mod memory_pools;
pub mod model_outlining_and_inlining;
pub mod module_hashing;
pub mod mp_session_and_host_completion;
pub mod mutable_loads;
pub mod packing_shapes_and_range_propagation;
pub mod page_agent_and_trace_recorder;
pub mod parameters;
pub mod rdma_and_cache_control;
pub mod routines;
pub mod shape_fallbacks;
pub mod shape_staticization;
pub mod shared_region_orchestrator;
pub mod stub_merge_and_layout_inference;
pub mod tenant_quotas;
pub mod tiered_storage;
pub mod transpose_sinking;

pub(crate) use crate::passes::fuse::kv_serving::checked_lcm_u64;
pub use crate::passes::fuse::{
    EngineKvFormat, InjectKvCachePass, JengaPoolGeometry, KvAttentionMaskKind, KvLayerGroup,
    KvPageSpec, LowerFlashDecodingPass, LowerPagedAttentionPass, LowerTreeAttentionPass,
    analyze_kv_layer_groups, longest_full_attention_hit, longest_joint_prefix_hit,
    longest_sliding_window_hit, plan_jenga_kv_geometry,
};
pub use cacheblend::*;
pub use collectives::*;
pub use dependencies::*;
pub use device_graphs::*;
pub use dialect_lowering::*;
pub use effect_chains::*;
pub use fleet_directory::*;
pub use graph_blobs_and_poison::*;
pub use graph_finalization::*;
pub use index_views::*;
pub use kernel_stubs::*;
pub use kv_attention::*;
pub use kv_pools::*;
pub use loop_buffers::*;
pub use map_input_output_fusion::*;
pub use memory_pools::*;
pub use model_outlining_and_inlining::*;
pub use module_hashing::*;
pub use mp_session_and_host_completion::*;
pub use mutable_loads::*;
pub use packing_shapes_and_range_propagation::*;
pub use page_agent_and_trace_recorder::*;
pub use parameters::*;
pub use rdma_and_cache_control::*;
pub use routines::*;
pub use shape_fallbacks::*;
pub use shape_staticization::*;
pub use shared_region_orchestrator::*;
pub use stub_merge_and_layout_inference::*;
pub use tenant_quotas::*;
pub use tiered_storage::*;
pub use transpose_sinking::*;
