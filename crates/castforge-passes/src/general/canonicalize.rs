//! Composite Canonicalization (`canonicalize`) & Optimization Presets (`O1`, `O2`, `O3`).

use crate::{
    DeadCodeEliminationPass, FoldConstantsPass, FoldViewsPass, FoldWeightsPass,
    FuseActivationsPass, FuseElementwisePass, FuseGemmPass, FuseNormPass, FuseSoftmaxPass,
    InlinerPass, LicmPass, OptimizeMemoryPass, Pass, PassResult, ShardDistributedPass,
    TopologicalSortPass, UnifiedCsePass, UnifiedFuseAttentionPass, UnifiedMergeWeightsPass,
    UnifiedSimplifyAlgebraicPass, UnifiedSpecializeShapesPass,
};
use anyhow::Result;
use castforge_ir::Module;
use indexmap::IndexMap;

/// Composite Canonicalize Pass (`canonicalize`).
pub struct CanonicalizePass;

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn description(&self) -> &'static str {
        "Composite canonicalization pass (constants, algebraic, views, cse, dce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let steps = [
                UnifiedCsePass::default().run(module)?,
                FoldConstantsPass::default().run(module)?,
                UnifiedSimplifyAlgebraicPass::default().run(module)?,
                FoldViewsPass::default().run(module)?,
                DeadCodeEliminationPass::default().run(module)?,
            ];

            let iter_changed = steps.iter().any(|r| r.changed);
            total.changed |= iter_changed;
            total.ops_removed += steps.iter().map(|r| r.ops_removed).sum::<usize>();
            total.ops_modified += steps.iter().map(|r| r.ops_modified).sum::<usize>();
            if !iter_changed {
                break;
            }
        }

        Ok(total)
    }
}

/// Level 1 Optimization Preset (`O1`):
/// Runs inlining, constant folding, LICM, GVN/CSE, view folding, canonicalization, DCE, and
/// topological sort.
pub struct OptO1Pass;

impl Pass for OptO1Pass {
    fn name(&self) -> &'static str {
        "O1"
    }

    fn description(&self) -> &'static str {
        "Preset O1: Inlining, fold-constants, LICM, CSE/GVN, fold-views, canonicalize, dce"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            InlinerPass.run(module)?,
            FoldConstantsPass {
                include_conditional: true,
            }
            .run(module)?,
            LicmPass.run(module)?,
            UnifiedCsePass::default().run(module)?,
            FoldViewsPass::default().run(module)?,
            CanonicalizePass.run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
            TopologicalSortPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
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

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            OptO1Pass.run(module)?,
            FoldViewsPass::default().run(module)?,
            UnifiedMergeWeightsPass {
                method: "lora".to_string(),
                alpha: 1.0,
                ..Default::default()
            }
            .run(module)?,
            FoldWeightsPass {
                mode: "batchnorm".to_string(),
            }
            .run(module)?,
            FuseNormPass::default().run(module)?,
            UnifiedFuseAttentionPass {
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
            UnifiedFuseAttentionPass {
                kind: "sdpa".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseSoftmaxPass::default().run(module)?,
            FuseGemmPass::default().run(module)?,
            FuseElementwisePass.run(module)?,
            CanonicalizePass.run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
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

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut spec_pass = UnifiedSpecializeShapesPass {
            default_dim: self.default_dim,
            bindings: self.bindings.clone(),
            batch: None,
            buckets: None,
        };
        let steps = vec![
            spec_pass.run(module)?,
            OptO2Pass.run(module)?,
            FuseActivationsPass {
                kind: "moe".to_string(),
                ..Default::default()
            }
            .run(module)?,
            UnifiedSimplifyAlgebraicPass {
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
            }
            .run(module)?,
            OptimizeMemoryPass {
                mode: "all".to_string(),
                ..Default::default()
            }
            .run(module)?,
            TopologicalSortPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

fn aggregate_results(name: &str, steps: &[PassResult]) -> PassResult {
    PassResult {
        pass_name: name.to_string(),
        changed: steps.iter().any(|r| r.changed),
        ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
        ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
        ops_created: steps.iter().map(|r| r.ops_created).sum(),
    }
}
