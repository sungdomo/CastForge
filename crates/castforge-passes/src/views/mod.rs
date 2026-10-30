//! View, shape, and layout transformation passes (`fold-views`, `specialize-shapes`).

pub mod fold_views;
pub mod specialize_shapes;

pub use fold_views::{
    ConvertLayoutPass, FoldBroadcastPass, FoldConvertPass, FoldReshapePass, FoldSliceConcatPass,
    FoldTransposePass, FoldViewsPass, SinkTransposePass,
};
pub use specialize_shapes::{
    BucketShapesPass, ResizeBatchPass, SpecializeShapesPass, UnifiedSpecializeShapesPass,
};
