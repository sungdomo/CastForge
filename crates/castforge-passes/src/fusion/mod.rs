//! Operator and subgraph fusion passes (`fuse-attention`, `fuse-norm`, `fuse-activations`,
//! `fuse-gemm`, `fuse-elementwise`).

pub mod fuse_activations;
pub mod fuse_attention;
pub mod fuse_elementwise;
pub mod fuse_gemm;
pub mod fuse_norm;

pub use fuse_activations::{
    FuseActivationsPass, FuseGatedMlpPass, FuseGeluPass, FuseMoePass, FuseSwiGluPass,
};
pub use fuse_attention::{
    FlashDecodingPass, FuseAttentionPass, FuseMlaPass, FuseRopePass, InjectKvCachePass,
    LowerPagedAttentionPass, OptimizeAttentionPass, SpeculativeTreeAttentionPass,
    UnifiedFuseAttentionPass,
};
pub use fuse_elementwise::{
    DecomposeCompositesPass, FissionLoopsPass, FuseElementwisePass, FuseLoopsPass,
    FuseMapReducePass, FuseMegakernelPass, UnifiedFuseElementwisePass,
};
pub use fuse_gemm::{
    FuseDequantMatmulPass, FuseGemmPass, FuseHorizontalDotPass, FuseMatmulEpiloguePass,
};
pub use fuse_norm::{
    FuseLayerNormPass, FuseLinearCrossEntropyPass, FuseNormPass, FuseResidualNormPass,
    FuseRmsNormPass, FuseSoftmaxPass,
};
