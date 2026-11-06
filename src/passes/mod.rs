//! Built-in A-to-B IR optimization, fusion, quantization, loop-scheduling, and surgery passes
//! (`castforge::passes`).
//!
//! Core pass execution and verification infrastructure ([`Pass`], [`PassManager`],
//! [`PassRegistry`], [`VerifyIrPass`]) lives in [`crate::ir`] and is re-exported here for
//! ergonomics.

pub mod canonicalize;
pub mod distributed;
pub mod fuse;
pub mod loops;
pub mod memory;
pub mod normalize;
pub mod quantize;
pub mod registry;
#[cfg(feature = "serving")]
pub mod serving;
pub mod shapes;
pub mod weights;

pub use canonicalize::{
    CanonicalizePass, ConvertLayoutPass, CsePass, DeadCodeEliminationPass, FoldConstantsPass,
    FoldViewsPass, HoistConstantsPass, HoistLoopInvariantsPass, SimplifyAlgebraicPass,
    TopologicalSortPass,
};
pub use distributed::{
    AutodiffPass, CoalesceAllReducePass, ScheduleStreamsPass, ShardDistributedPass,
    ShardExpertsPass, ShardFsdpPass, ShardPipelineParallelPass, ShardRingAttentionPass,
    ShardSpmdPass,
};
pub use fuse::{
    DecomposeCompositesPass, EngineKvFormat, FuseActivationsPass, FuseAddMaxPass,
    FuseAttentionPass, FuseDequantMatmulPass, FuseElementwisePass, FuseFmaPass, FuseGatedMlpPass,
    FuseGeluPass, FuseGemmPass, FuseHorizontalDotPass, FuseLayerNormPass,
    FuseLinearCrossEntropyPass, FuseLoopsPass, FuseMapReducePass, FuseMatmulEpiloguePass,
    FuseMegakernelPass, FuseMlaPass, FuseMoePass, FuseNormPass, FuseResidualNormPass,
    FuseRmsNormPass, FuseRopePass, FuseSoftmaxPass, FuseSwiGluPass, InjectKvCachePass,
    JengaPoolGeometry, KvAttentionMaskKind, KvLayerGroup, KvPageSpec, LowerFlashDecodingPass,
    LowerPagedAttentionPass, LowerTreeAttentionPass, SplitLoopsPass, analyze_kv_layer_groups,
    longest_full_attention_hit, longest_joint_prefix_hit, longest_sliding_window_hit,
    match_gelu_op, match_gelu_val, match_silu_val, plan_jenga_kv_geometry,
};
pub use loops::{
    LoopPass, LowerConvToGemmPass, LowerTensorCoresPass, OptimizeLoopIrPass,
    OptimizeLoopIrTranslator, PadForTilingPass, PermuteLoopsPass, ScheduleLoopsPass,
    SplitLargeGemmPass, TensorToLoopTranslator, TileLoopsPass, UnrollLoopsPass, VectorizeLoopsPass,
    lower_to_loop_ir,
};
pub use memory::{
    BufferizeInplacePass, EliminateConcatCopiesPass, OffloadActivationsPass, OptimizeMemoryPass,
    RematerializePass,
};
pub use normalize::{
    BisectPassesPass, ComposeFunctionsPass, DedupFunctionsPass, ExtractSubgraphPass,
    FreezeInputsPass, HashIrPass, InlineCallsPass, InstrumentProfilingPass, NormalizeIrPass,
    NormalizeSsaPass, OutlineKernelsPass, OutlineSubgraphsPass, ParameterizeConstantsPass,
    PrintOpStatsPass, ReduceIrPass, RooflineSummary, SanitizeNumericsPass, StripAttributesPass,
    StripDebugInfoPass, TapIntermediatesPass, UnrollCallsPass, VerifyEquivPass, VerifyIrPass,
    VerifyShapesPass, analyze_roofline, compute_structural_hash, link_modules,
};
pub use quantize::{
    CalibrateQuantizePass, InsertQatStePass, LowerMixedPrecisionPass, QuantizeFp4Pass,
    QuantizeFp8Pass, QuantizeGroupwisePass, QuantizePass, QuantizeWeightsPass, RotateHadamardPass,
    SmoothQuantizePass, quantize_symmetric_f32,
};
pub use registry::{
    OptO1Pass, OptO2Pass, OptO3Pass, Pass, PassCatalog, PassCatalogEntry, PassDescriptor,
    PassFactory, PassManager, PassOptionSpec, PassOptions, PassRegistry, PassResult,
    PassStepConfig, PassTelemetry, PassTraceStep, PassTranslator, PipelineConfig, PipelineSettings,
    parse_pass_spec_item, pass_catalog, split_pipeline_spec,
};
#[cfg(feature = "serving")]
pub use serving::*;
pub use shapes::{BucketShapesPass, PruneTokensPass, ResizeBatchPass, SpecializeShapesPass};
pub use weights::{
    BakeDoraPass, BakeLoraPass, ExternalizeWeightsPass, FoldBatchNormPass, FoldNormWeightsPass,
    FoldWeightsPass, MergeMethod, MergeTaskVectorsPass, MergeWeightsPass, PackWeightsPass,
    PairwiseMergeWeightsPass, Prune2To4Pass, PruneActivationsPass, PruneStructuredPass,
    PruneWeightsPass, apply_2to4_mask, merge_f32_tensors, prune_2d_columns_by_l1,
};
