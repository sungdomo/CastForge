//! Serving runtime subsystems and graph compiler optimization passes.

pub mod blend_and_algebraic;
pub mod chains_and_parameters;
pub mod device_graph_and_stubs;
pub mod fleet_and_collective;
pub mod kv_attention;
pub mod memory_and_loops;
pub mod quotas_and_fallbacks;
pub mod routines_and_hashing;
pub mod storage_and_mutable;
pub mod views_and_dependencies;

pub use blend_and_algebraic::*;
pub use chains_and_parameters::*;
pub use device_graph_and_stubs::*;
pub use fleet_and_collective::*;
pub use kv_attention::*;
pub use memory_and_loops::*;
pub use quotas_and_fallbacks::*;
pub use routines_and_hashing::*;
pub use storage_and_mutable::*;
pub use views_and_dependencies::*;

use blend_and_algebraic::{
    extract_single_int_attr, is_sinkable_binary_elementwise, is_sinkable_unary_elementwise,
};
use kv_attention::checked_lcm_u64;
use memory_and_loops::extract_int_list_attr;
use storage_and_mutable::{count_value_uses_in_block, extract_u64_attr};
