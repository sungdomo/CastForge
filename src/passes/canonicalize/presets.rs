//! Standard compiler optimization level presets (`O1`, `O2`, `O3`).

use super::{CanonicalizePass, TopologicalSortPass};
use crate::ir::Module;
use crate::passes::{
    CsePass, DeadCodeEliminationPass, FoldViewsPass, FoldWeightsPass, FuseActivationsPass,
    FuseAddMaxPass, FuseAttentionPass, FuseGemmPass, FuseNormPass, FuseSoftmaxPass,
    InlineCallsPass, MergeWeightsPass, OptimizeMemoryPass, Pass, PassResult, ShardDistributedPass,
    SimplifyAlgebraicPass, SpecializeShapesPass,
};
use anyhow::Result;
use indexmap::IndexMap;

/// Level 1 Optimization Preset (`O1`):
/// Runs inlining, constant folding, LICM, GVN/CSE, view folding, canonicalization, DCE, and
/// topological sort.
pub struct OptO1Pass;

impl Pass for OptO1Pass {
    fn name(&self) -> &'static str {
        "O1"
    }

    fn description(&self) -> &'static str {
        "Preset O1: Inlining, simplify-algebraic, LICM, CSE/GVN, fold-views, canonicalize, dce"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "inline-calls",
            "simplify-algebraic",
            "cse",
            "fold-views",
            "canonicalize",
            "dce",
            "topological-sort",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            InlineCallsPass.run(module)?,
            SimplifyAlgebraicPass::default().run(module)?,
            CsePass::default().run(module)?,
            FoldViewsPass::default().run(module)?,
            CanonicalizePass::default().run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
            TopologicalSortPass::default().run(module)?,
        ];
        Ok(PassResult::aggregate(self.name(), &steps))
    }
}

/// Level 2 Optimization Preset (`O2`):
/// Runs `O1` followed by weight folding (`merge-weights(method=lora)`, `fold-weights`),
/// unified macro-fusions (`fuse-norm`, `fuse-attention`, `fuse-activations`, `fuse-softmax`,
/// `fuse-gemm`, `fuse-elementwise`), and post-fusion cleanup.
pub struct OptO2Pass;

impl Pass for OptO2Pass {
    fn name(&self) -> &'static str {
        "O2"
    }

    fn description(&self) -> &'static str {
        "Preset O2: O1 + LoRA/BatchNorm weight folding + unified Transformer/GEMM fusions"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "O1",
            "fold-views",
            "merge-weights",
            "fold-weights",
            "fuse-norm",
            "fuse-attention",
            "fuse-activations",
            "fuse-softmax",
            "fuse-gemm",
            "fuse-elementwise",
            "canonicalize",
            "dce",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            OptO1Pass.run(module)?,
            FoldViewsPass::default().run(module)?,
            MergeWeightsPass {
                method: "lora".to_string(),
                alpha: 1.0,
                ..Default::default()
            }
            .run(module)?,
            FoldWeightsPass {
                mode: "batchnorm".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseNormPass::default().run(module)?,
            FuseAttentionPass {
                kind: "rope".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseActivationsPass {
                kind: "swiglu".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseActivationsPass {
                kind: "gelu".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseActivationsPass {
                kind: "gated_mlp".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseAttentionPass {
                kind: "sdpa".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseSoftmaxPass::default().run(module)?,
            FuseGemmPass::default().run(module)?,
            FuseAddMaxPass.run(module)?,
            CanonicalizePass::default().run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
        ];
        Ok(PassResult::aggregate(self.name(), &steps))
    }
}

/// Level 3 Optimization Preset (`O3`):
/// Runs symbolic/dynamic shape specialization, `O2` operator fusions, `fuse-activations(kind=moe)`,
/// `simplify-algebraic(fuse_fma=true)`, `shard-distributed`, `optimize-memory`,
/// `fold-weights(mode=pack)`, and peak-memory topological sorting.
pub struct OptO3Pass {
    pub preset_name: &'static str,
    pub default_dim: i64,
    pub bindings: IndexMap<String, i64>,
}

impl Default for OptO3Pass {
    fn default() -> Self {
        Self {
            preset_name: "O3",
            default_dim: 1,
            bindings: IndexMap::new(),
        }
    }
}

impl Pass for OptO3Pass {
    fn name(&self) -> &'static str {
        self.preset_name
    }

    fn description(&self) -> &'static str {
        "Preset O3: Shape specialization + O2 + MoE/FMA/zero-copy + weight packing + arena"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "specialize-shapes",
            "O2",
            "fuse-activations",
            "simplify-algebraic",
            "shard-distributed",
            "fold-weights",
            "optimize-memory",
            "topological-sort",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut spec_pass = SpecializeShapesPass {
            default_dim: self.default_dim,
            bindings: self.bindings.clone(),
            batch: None,
            buckets: None,
            token_prune_ratio: None,
        };
        let steps = vec![
            spec_pass.run(module)?,
            OptO2Pass.run(module)?,
            FuseActivationsPass {
                kind: "moe".to_string(),
                ..Default::default()
            }
            .run(module)?,
            SimplifyAlgebraicPass {
                fold_constants: true,
                include_conditional: true,
                egraph: true,
                fuse_fma: true,
            }
            .run(module)?,
            ShardDistributedPass {
                mode: "coalesce".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FoldWeightsPass {
                mode: "pack".to_string(),
                ..Default::default()
            }
            .run(module)?,
            OptimizeMemoryPass {
                mode: "all".to_string(),
                ..Default::default()
            }
            .run(module)?,
            TopologicalSortPass::default().run(module)?,
        ];
        Ok(PassResult::aggregate(self.name(), &steps))
    }
}
