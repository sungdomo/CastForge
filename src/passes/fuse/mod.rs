//! Unified operator fusion and composite decomposition passes (`fuse-attention`,
//! `fuse-elementwise`, `fuse-gemm`, `fuse-norm`, `fuse-fma`, `decompose-composites`).

pub mod activations;
pub mod attention;
pub mod decompose;
pub mod decompose_helpers;
pub mod elementwise;
pub mod epilogue;
pub mod gemm;
pub mod kv_serving;
pub mod layernorm_softmax;
pub mod loops;
pub mod norm;

pub use activations::{
    FuseActivationsPass, FuseGatedMlpPass, FuseGeluPass, FuseMoePass, FuseSwiGluPass,
    match_gelu_op, match_gelu_val, match_silu_val,
};
pub use attention::{FuseAttentionPass, FuseMlaPass, FuseRopePass};
pub use decompose::DecomposeCompositesPass;
pub use elementwise::{FuseAddMaxPass, FuseElementwisePass, FuseFmaPass, FuseMapReducePass};
pub(crate) use elementwise::{try_fuse_add_max_at_op, try_fuse_map_reduce_at_op};
pub use gemm::{
    FuseDequantMatmulPass, FuseGemmPass, FuseHorizontalDotPass, FuseMatmulEpiloguePass,
};
pub use kv_serving::{
    EngineKvFormat, InjectKvCachePass, JengaPoolGeometry, KvAttentionMaskKind, KvLayerGroup,
    KvPageSpec, LowerFlashDecodingPass, LowerPagedAttentionPass, LowerTreeAttentionPass,
    analyze_kv_layer_groups, longest_full_attention_hit, longest_joint_prefix_hit,
    longest_sliding_window_hit, plan_jenga_kv_geometry,
};
pub use layernorm_softmax::{FuseLayerNormPass, FuseLinearCrossEntropyPass, FuseSoftmaxPass};
pub use loops::{FuseLoopsPass, FuseMegakernelPass, SplitLoopsPass};
pub use norm::{FuseNormPass, FuseResidualNormPass, FuseRmsNormPass};
