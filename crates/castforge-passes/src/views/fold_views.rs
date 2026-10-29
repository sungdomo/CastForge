use crate::{
    BucketShapesPass, FoldBroadcastPass, FoldConvertPass, FoldReshapePass, FoldSliceConcatPass,
    FoldTransposePass, Pass, PassResult, ResizeBatchPass, SinkTransposePass, SpecializeShapesPass,
};
use anyhow::Result;
use castforge_ir::Module;
use indexmap::IndexMap;

/// Unified View, Shape & Type-Cast Folding Pass (`fold-views`).
///
/// Consolidates `sink-transpose`, `fold-transpose`, `fold-reshape`, `fold-broadcast`,
/// `fold-slice-concat`, and `fold-convert` in a single pass.
#[derive(Default)]
pub struct FoldViewsPass;

impl Pass for FoldViewsPass {
    fn name(&self) -> &'static str {
        "fold-views"
    }

    fn description(&self) -> &'static str {
        "Unified folding of transpose, reshape, broadcast, slice/concat, and convert views"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            SinkTransposePass.run(module)?,
            FoldTransposePass {
                eliminate_identity: true,
            }
            .run(module)?,
            FoldReshapePass.run(module)?,
            FoldBroadcastPass.run(module)?,
            FoldSliceConcatPass.run(module)?,
            FoldConvertPass.run(module)?,
        ];
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Unified Shape Specialization, Batch Resizing & Multi-Bucket Dispatch Pass (`specialize-shapes`).
///
/// Consolidates symbolic/dynamic dimension binding (`SpecializeShapesPass`), static batch
/// dimension resizing (`ResizeBatchPass`), and multi-bucket dispatch cloning (`BucketShapesPass`).
#[derive(Default)]
pub struct UnifiedSpecializeShapesPass {
    pub default_dim: i64,
    pub bindings: IndexMap<String, i64>,
    pub batch: Option<i64>,
    pub buckets: Option<String>,
}

impl Pass for UnifiedSpecializeShapesPass {
    fn name(&self) -> &'static str {
        "specialize-shapes"
    }

    fn description(&self) -> &'static str {
        "Unified shape specialization, batch resizing (batch=N), and bucketing (buckets=1;4;8)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if let Some(ref bkts) = self.buckets
            && !bkts.is_empty()
        {
            steps.push(
                BucketShapesPass {
                    buckets: bkts.clone(),
                }
                .run(module)?,
            );
        }
        if let Some(b) = self.batch
            && b > 0
        {
            steps.push(ResizeBatchPass { batch: b }.run(module)?);
        }
        if !self.bindings.is_empty() || steps.is_empty() {
            steps.push(
                SpecializeShapesPass {
                    default_dim: if self.default_dim > 0 {
                        self.default_dim
                    } else {
                        1
                    },
                    bindings: self.bindings.clone(),
                }
                .run(module)?,
            );
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}
