pub mod decompose_composites;
pub mod fuse_attention;
pub mod fuse_dequant_matmul;
pub mod fuse_elementwise;
pub mod fuse_gated_mlp;
pub mod fuse_gelu;
pub mod fuse_horizontal_dot;
pub mod fuse_layernorm;
pub mod fuse_map_reduce;
pub mod fuse_matmul_epilogue;
pub mod fuse_moe;
pub mod fuse_residual_norm;
pub mod fuse_rmsnorm;
pub mod fuse_rope;
pub mod fuse_softmax;
pub mod fuse_swiglu;
pub mod sota_fusions;

pub use decompose_composites::DecomposeCompositesPass;
pub use fuse_attention::FuseAttentionPass;
pub use fuse_dequant_matmul::FuseDequantMatmulPass;
pub use fuse_elementwise::FuseElementwisePass;
pub use fuse_gated_mlp::FuseGatedMlpPass;
pub use fuse_gelu::FuseGeluPass;
pub use fuse_horizontal_dot::FuseHorizontalDotPass;
pub use fuse_layernorm::FuseLayerNormPass;
pub use fuse_map_reduce::FuseMapReducePass;
pub use fuse_matmul_epilogue::FuseMatmulEpiloguePass;
pub use fuse_moe::FuseMoePass;
pub use fuse_residual_norm::FuseResidualNormPass;
pub use fuse_rmsnorm::FuseRmsNormPass;
pub use fuse_rope::FuseRopePass;
pub use fuse_softmax::FuseSoftmaxPass;
pub use fuse_swiglu::FuseSwiGluPass;
pub use sota_fusions::{
    FlashDecodingPass, FuseActivationsPass, FuseGemmPass, FuseLinearCrossEntropyPass,
    FuseMegakernelPass, FuseMlaPass, FuseNormPass, LowerPagedAttentionPass, OptimizeAttentionPass,
    SpeculativeTreeAttentionPass, UnifiedFuseElementwisePass,
};
