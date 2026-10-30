//! Operator and subgraph fusion passes (`fuse-attention`, `fuse-norm`, `fuse-activations`,
//! `fuse-gemm`, `fuse-elementwise`, `fuse-softmax`, `decompose-composites`).

pub mod decompose_composites;
pub mod fuse_activations;
pub mod fuse_attention;
pub mod fuse_elementwise;
pub mod fuse_gemm;
pub mod fuse_norm;
pub mod fuse_softmax;

pub use decompose_composites::DecomposeCompositesPass;
pub use fuse_activations::{
    FuseActivationsPass, FuseGatedMlpPass, FuseGeluPass, FuseMoePass, FuseSwiGluPass,
};
pub use fuse_attention::{
    FlashDecodingPass, FuseAttentionPass, FuseMlaPass, FuseRopePass, InjectKvCachePass,
    LowerPagedAttentionPass, OptimizeAttentionPass, SpeculativeTreeAttentionPass,
    UnifiedFuseAttentionPass,
};
pub use fuse_elementwise::{
    FissionLoopsPass, FuseElementwisePass, FuseLoopsPass, FuseMapReducePass, FuseMegakernelPass,
    UnifiedFuseElementwisePass,
};
pub use fuse_gemm::{
    FuseDequantMatmulPass, FuseGemmPass, FuseHorizontalDotPass, FuseMatmulEpiloguePass,
};
pub use fuse_norm::{FuseLayerNormPass, FuseNormPass, FuseResidualNormPass, FuseRmsNormPass};
pub use fuse_softmax::{FuseLinearCrossEntropyPass, FuseSoftmaxPass};
