//! Pre-built end-to-end optimization pipeline presets (`O1`, `O2`, `O3`, `llm-opt`).

use crate::{
    BakeLoraPass, BufferizeInplacePass, CanonicalizePass, DeadCodeEliminationPass,
    EgraphSimplifyPass, EliminateConcatCopiesPass, FoldBatchNormPass, FoldNormWeightsPass,
    FuseActivationsPass, FuseAttentionPass, FuseDequantMatmulPass, FuseElementwisePass,
    FuseMatmulEpiloguePass, FuseMoePass, FuseNormPass, FuseRopePass, FuseSoftmaxPass,
    GlobalValueNumberingPass, HoistAllReducePass, InlinerPass, LicmPass, PackWeightsPass, Pass,
    PassResult, SccpPass, SinkTransposePass, SpecializeShapesPass, SymbolDcePass,
    TopologicalSortPass,
};
use anyhow::Result;
use castforge_ir::Module;
use indexmap::IndexMap;

/// Level 1 Optimization Preset (`O1`):
/// Runs inlining, SCCP, LICM, GVN, fixed-point canonicalization, symbol DCE, and topological sort.
pub struct OptO1Pass;

impl Pass for OptO1Pass {
    fn name(&self) -> &'static str {
        "O1"
    }

    fn description(&self) -> &'static str {
        "Preset O1: Inlining, SCCP, LICM, GVN, canonicalization, symbol-dce, toposort"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            InlinerPass.run(module)?,
            SccpPass.run(module)?,
            LicmPass.run(module)?,
            GlobalValueNumberingPass.run(module)?,
            CanonicalizePass.run(module)?,
            SymbolDcePass.run(module)?,
            TopologicalSortPass.run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

/// Level 2 Optimization Preset (`O2`):
/// Runs `O1` followed by weight folding (`bake-lora`, `fold-norm-weights`, `fold-batchnorm`),
/// layout sinking, unified macro-fusions (`fuse-norm`, `fuse-rope`, `fuse-activations`,
/// `fuse-attention`, `fuse-softmax`, `fuse-matmul-epilogue`, `fuse-dequant-matmul`,
/// `fuse-elementwise`), and post-fusion cleanup.
pub struct OptO2Pass;

impl Pass for OptO2Pass {
    fn name(&self) -> &'static str {
        "O2"
    }

    fn description(&self) -> &'static str {
        "Preset O2: O1 + LoRA/Norm/BatchNorm weight folding + unified Transformer/MatMul fusions"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            OptO1Pass.run(module)?,
            SinkTransposePass.run(module)?,
            BakeLoraPass { alpha: 1.0 }.run(module)?,
            FoldNormWeightsPass.run(module)?,
            FoldBatchNormPass.run(module)?,
            FuseNormPass::default().run(module)?,
            FuseRopePass.run(module)?,
            FuseActivationsPass::default().run(module)?,
            FuseAttentionPass.run(module)?,
            FuseSoftmaxPass.run(module)?,
            FuseMatmulEpiloguePass.run(module)?,
            FuseDequantMatmulPass.run(module)?,
            FuseElementwisePass.run(module)?,
            CanonicalizePass.run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

/// Level 3 (`O3`) & Distributed/MoE LLM Optimization Preset (`llm-opt`):
/// Runs symbolic/dynamic shape specialization, `O2` operator fusions, compile-time RHS weight
/// pre-packing (`pack-weights`), in-place arena buffer donation (`bufferize-inplace`), and
/// topological sorting. When invoked as `llm-opt`, additionally runs `fuse-moe`,
/// `egraph-simplify`, `hoist-all-reduce`, and `eliminate-concat-copies`.
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
        "Preset O3 / llm-opt: Shape specialization + O2 fusions + weight packing + inplace arena"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut spec_pass = SpecializeShapesPass {
            default_dim: self.default_dim,
            bindings: self.bindings.clone(),
        };
        let mut steps = vec![spec_pass.run(module)?, OptO2Pass.run(module)?];
        if self.preset_name == "llm-opt" {
            steps.push(FuseMoePass.run(module)?);
            steps.push(EgraphSimplifyPass { fuse_fma: true }.run(module)?);
            steps.push(HoistAllReducePass.run(module)?);
            steps.push(EliminateConcatCopiesPass.run(module)?);
        }
        steps.push(PackWeightsPass.run(module)?);
        steps.push(BufferizeInplacePass.run(module)?);
        steps.push(TopologicalSortPass.run(module)?);
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
