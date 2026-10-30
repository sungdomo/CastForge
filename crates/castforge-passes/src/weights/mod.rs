//! Weight quantization, pruning, merging, folding, and storage passes.

pub mod externalize_weights;
pub mod fold_weights;
pub mod merge_weights;
pub mod ops;
pub mod prune_weights;
pub mod quantize;

pub use externalize_weights::ExternalizeWeightsPass;
pub use fold_weights::{FoldBatchNormPass, FoldNormWeightsPass, FoldWeightsPass, PackWeightsPass};
pub use merge_weights::{
    BakeLoraPass, MergeWeightsPass, TaskArithmeticPass, UnifiedMergeWeightsPass,
};
pub use ops::{MergeMethod, merge_f32_tensors, prune_2d_columns_by_l1, quantize_symmetric_f32};
pub use prune_weights::{
    PruneWeightsPass, Sparsify2To4Pass, StructuredPruningPass, apply_2to4_mask,
};
pub use quantize::{
    CalibrateQuantizePass, HadamardQuantRotationPass, MixedPrecisionPass, QuantizeFp8Pass,
    QuantizeGroupwisePass, QuantizePass, QuantizeWeightsPass,
};
