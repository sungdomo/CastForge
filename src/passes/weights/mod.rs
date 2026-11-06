//! Unified compile-time weight folding, pre-packing, SafeTensors externalization,
//! structured/2:4 sparsity pruning, checkpoint merging, and LoRA/DoRA adapter baking
//! (`fold-weights`, `externalize-weights`, `prune-weights`, `merge-weights`, `bake-lora`,
//! `bake-dora`).

pub mod fold;
pub mod lora;
pub mod merge;
pub mod prune;

pub use fold::{FoldBatchNormPass, FoldNormWeightsPass, FoldWeightsPass, PackWeightsPass};
pub use lora::{BakeDoraPass, BakeLoraPass};
pub use merge::{
    MergeMethod, MergeTaskVectorsPass, MergeWeightsPass, PairwiseMergeWeightsPass,
    merge_f32_tensors,
};
pub use prune::{
    ExternalizeWeightsPass, Prune2To4Pass, PruneActivationsPass, PruneStructuredPass,
    PruneWeightsPass, apply_2to4_mask, prune_2d_columns_by_l1,
};
