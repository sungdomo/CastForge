//! Pre-built end-to-end optimization pipeline presets (`O1`, `O2`, `O3`, `llm-opt`).

use crate::{
    BakeLoraPass, BufferizeInplacePass, CanonicalizePass, DeadCodeEliminationPass,
    FoldNormWeightsPass, FuseAttentionPass, FuseDequantMatmulPass, FuseElementwisePass,
    FuseGatedMlpPass, FuseGeluPass, FuseLayerNormPass, FuseMatmulEpiloguePass,
    FuseResidualNormPass, FuseRmsNormPass, FuseRopePass, FuseSoftmaxPass, FuseSwiGluPass,
    InlinerPass, LicmPass, PackWeightsPass, Pass, PassResult, SccpPass, SinkTransposePass,
    SpecializeShapesPass, SymbolDcePass, TopologicalSortPass,
};
use anyhow::Result;
use castforge_ir::Module;
use indexmap::IndexMap;

/// Level 1 Optimization Preset (`O1`):
/// Runs inlining, SCCP, LICM, fixed-point canonicalization, symbol DCE, and topological sort.
pub struct OptO1Pass;

impl Pass for OptO1Pass {
    fn name(&self) -> &'static str {
        "O1"
    }

    fn description(&self) -> &'static str {
        "Preset O1: Inlining, SCCP, LICM, fixed-point canonicalization, symbol-dce, toposort"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            InlinerPass.run(module)?,
            SccpPass.run(module)?,
            LicmPass.run(module)?,
            CanonicalizePass.run(module)?,
            SymbolDcePass.run(module)?,
            TopologicalSortPass.run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

/// Level 2 Optimization Preset (`O2`):
/// Runs `O1` followed by weight folding (`bake-lora`, `fold-norm-weights`), layout sinking,
/// macro-fusions (`rmsnorm`, `rope`, `swiglu`, `attention`, `softmax`, `matmul-epilogue`,
/// `elementwise`), and post-fusion cleanup.
pub struct OptO2Pass;

impl Pass for OptO2Pass {
    fn name(&self) -> &'static str {
        "O2"
    }

    fn description(&self) -> &'static str {
        "Preset O2: O1 + LoRA/Norm weight folding + Transformer/MatMul operator fusions"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            OptO1Pass.run(module)?,
            SinkTransposePass.run(module)?,
            BakeLoraPass { alpha: 1.0 }.run(module)?,
            FoldNormWeightsPass.run(module)?,
            FuseRmsNormPass.run(module)?,
            FuseResidualNormPass.run(module)?,
            FuseLayerNormPass.run(module)?,
            FuseRopePass.run(module)?,
            FuseGatedMlpPass.run(module)?,
            FuseSwiGluPass.run(module)?,
            FuseGeluPass.run(module)?,
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

/// Level 3 / LLM Full Optimization Preset (`O3` and `llm-opt`):
/// Runs symbolic/dynamic shape specialization, `O2` operator fusions, compile-time RHS weight
/// pre-packing (`pack-weights`), in-place arena buffer donation (`bufferize-inplace`), and
/// topological sorting.
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
        let steps = [
            spec_pass.run(module)?,
            OptO2Pass.run(module)?,
            PackWeightsPass.run(module)?,
            BufferizeInplacePass.run(module)?,
            TopologicalSortPass.run(module)?,
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
