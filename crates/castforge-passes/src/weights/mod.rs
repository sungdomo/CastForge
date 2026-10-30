//! Weight quantization, pruning, merging, and folding passes.

pub mod fold_weights;
pub mod merge_weights;
pub mod prune_weights;
pub mod quantize;

pub use fold_weights::{
    ExternalizeWeightsPass, FoldBatchNormPass, FoldNormWeightsPass, FoldWeightsPass,
    PackWeightsPass,
};
pub use merge_weights::{
    BakeLoraPass, MergeMethod, MergeWeightsPass, TaskArithmeticPass, UnifiedMergeWeightsPass,
    merge_f32_tensors,
};
pub use prune_weights::{
    PruneWeightsPass, Sparsify2To4Pass, StructuredPruningPass, apply_2to4_mask,
    prune_2d_columns_by_l1,
};
pub use quantize::{
    CalibrateQuantizePass, HadamardQuantRotationPass, MixedPrecisionPass, QuantizeFp8Pass,
    QuantizeGroupwisePass, QuantizePass, QuantizeWeightsPass, quantize_symmetric_f32,
};
