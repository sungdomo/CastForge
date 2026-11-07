//! Standard compiler optimization level presets (`O1`, `O2`, `O3`).

use crate::ir::Module;
use crate::passes::{
    CanonicalizePass, CsePass, DeadCodeEliminationPass, FoldViewsPass, FoldWeightsPass,
    FuseActivationsPass, FuseAddMaxPass, FuseAttentionPass, FuseGemmPass, FuseNormPass,
    FuseSoftmaxPass, InlineCallsPass, MergeWeightsPass, OptimizeMemoryPass, Pass, PassResult,
    ShardDistributedPass, SimplifyAlgebraicPass, SpecializeShapesPass, TopologicalSortPass,
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

/// Unified serving IR optimization pass (`optimize-serving`):
/// Orchestrates parametric dependency topological sorting, hanging chain overhang joins,
/// non-aliasing mutable chain relaxation, MGP device-graph staging and collective partitioning,
/// routine lowering, mutable buffer elision/parallelization, kernel stub deduplication,
/// single-caller index-constant raising, and call-graph slot globalization.
pub struct OptimizeServingPass {
    /// Execution mode (`"all"`, `"topology"`, `"device_graph"`, `"routines"`, `"memory"`, or
    /// `"stubs"`).
    pub mode: String,
}

impl Default for OptimizeServingPass {
    fn default() -> Self {
        Self {
            mode: "all".to_string(),
        }
    }
}

impl Pass for OptimizeServingPass {
    fn name(&self) -> &'static str {
        "optimize-serving"
    }

    fn description(&self) -> &'static str {
        "Unified serving pass (mode=all|topology|device_graph|routines|memory|stubs)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        #[cfg(not(feature = "serving"))]
        {
            let _ = module;
            Ok(PassResult::unchanged(self.name()))
        }
        #[cfg(feature = "serving")]
        {
            use crate::passes::serving::{
                AsyncLaunchDispatch, canonicalize_kernel_stubs_for_merge,
                deduplicate_kernel_stubs_bottom_up, elide_redundant_mutable_buffer_ops,
                fixup_hanging_chain_overhangs, fuse_mutable_loads_into_kernels,
                globalize_mgp_call_slot_indices, lower_buffer_concat_in_place,
                lower_compiled_routines_with_init_guards, lower_guards_to_async_routines,
                lower_map_kernels_to_routines, lower_primitives_to_async_routines,
                parallelize_multi_use_mutable_loads, partition_device_graph_collective,
                raise_back_single_caller_index_constants,
                relax_non_aliasing_mutable_chain_sequences, stage_device_graph_regions,
                topologically_sort_block_with_parameters, verify_device_graph_kernel_registrations,
            };

            let mode = self.mode.as_str();
            let run_all = mode.is_empty() || mode == "all";
            let mut modified = 0usize;

            if run_all || mode == "topology" {
                let block_ids: Vec<_> = module.blocks.keys().collect();
                for blk_id in block_ids {
                    modified += topologically_sort_block_with_parameters(module, blk_id)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }
                modified += fixup_hanging_chain_overhangs(module);
                modified += relax_non_aliasing_mutable_chain_sequences(module);
            }

            if run_all || mode == "device_graph" {
                let _ = verify_device_graph_kernel_registrations(module)
                    .map_err(|e| anyhow::anyhow!(e))?;
                modified += stage_device_graph_regions(module);
                modified +=
                    partition_device_graph_collective(module).map_err(|e| anyhow::anyhow!(e))?;
            }

            if run_all || mode == "routines" {
                modified += lower_guards_to_async_routines(module);
                modified += lower_map_kernels_to_routines(module);
                let (c, w) =
                    lower_primitives_to_async_routines(module, AsyncLaunchDispatch::Immediate);
                modified += c + w;
                modified += lower_compiled_routines_with_init_guards(module)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }

            if run_all || mode == "memory" {
                modified += elide_redundant_mutable_buffer_ops(module);
                modified += parallelize_multi_use_mutable_loads(module);
                modified += fuse_mutable_loads_into_kernels(module);
                modified += lower_buffer_concat_in_place(module, 64);
            }

            if run_all || mode == "stubs" {
                modified += canonicalize_kernel_stubs_for_merge(module);
                modified += deduplicate_kernel_stubs_bottom_up(module);
                modified += raise_back_single_caller_index_constants(module);
                modified +=
                    globalize_mgp_call_slot_indices(module).map_err(|e| anyhow::anyhow!(e))?;
            }

            Ok(PassResult {
                pass_name: self.name().to_string(),
                changed: modified > 0,
                ops_removed: 0,
                ops_modified: modified,
                ops_created: 0,
            })
        }
    }
}
