//! Unified Elementwise, Map-Reduce, Loop-Chain Fusion & Composite Decomposition
//! (`fuse-elementwise`).
//!
//! Organized into cohesive submodules:
//! - [`mod.rs`](self): `FuseElementwisePass`, `FuseMapReducePass`, `FuseAddMaxPass`
//! - [`loops`]: `FuseLoopsPass`, `SplitLoopsPass`, `FuseMegakernelPass`
//! - [`decompose`]: `DecomposeCompositesPass`
//! - [`decompose_helpers`]: Primitive StableHLO expansion builders for composite decomposition
//! - [`activations`]: `FuseActivationsPass`, `FuseSwiGluPass`, `FuseGeluPass`, `FuseGatedMlpPass`,
//!   `FuseMoePass`

pub mod activations;
pub mod decompose;
pub mod decompose_helpers;
pub mod loops;

pub use activations::*;
pub use decompose::*;
pub use loops::*;

use crate::ir::{Attribute, BlockId, Module, OpId, OpName, ValueDef, ValueId};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Elementwise, Activation, Gated-MLP, MoE, Map-Reduce, Loop-Chain Fusion & Composite
/// Decomposition Pass (`fuse-elementwise`).
pub struct FuseElementwisePass {
    pub mode: String,
    pub max_steps: usize,
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for FuseElementwisePass {
    fn default() -> Self {
        Self {
            mode: "composites".to_string(),
            max_steps: 0,
            shared_expert: false,
            top_k: 2,
            capacity_factor: 1.25,
        }
    }
}

impl Pass for FuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Unified elementwise, activation (SwiGLU/GELU/GeGLU/MoE), map-reduce & loop-chain fusion"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        if matches!(m.as_str(), "decompose" | "expand" | "lower") {
            let mut r = DecomposeCompositesPass.run(module)?;
            r.pass_name = self.name().to_string();
            return Ok(r);
        }
        if m == "loops" || m == "chain" || self.max_steps > 0 {
            let mut steps = vec![FuseLoopsPass.run(module)?];
            if self.max_steps > 0 {
                steps.push(
                    SplitLoopsPass {
                        max_steps: self.max_steps,
                    }
                    .run(module)?,
                );
            }
            return Ok(PassResult {
                pass_name: self.name().to_string(),
                changed: steps.iter().any(|r| r.changed),
                ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
                ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
                ops_created: steps.iter().map(|r| r.ops_created).sum(),
            });
        }
        if m == "megakernel" || m == "persistent" {
            let mut r = FuseMegakernelPass.run(module)?;
            r.pass_name = self.name().to_string();
            return Ok(r);
        }

        let do_all = m == "all";
        let do_activations = do_all
            || matches!(
                m.as_str(),
                "activations" | "swiglu" | "gelu" | "geglu" | "gated_mlp" | "mlp" | "moe"
            )
            || self.shared_expert;
        let do_mlp = do_all || matches!(m.as_str(), "activations" | "gated_mlp" | "mlp");
        let do_swiglu = do_mlp || m == "swiglu";
        let do_gelu = do_mlp || m == "gelu" || m == "geglu";
        let do_moe = do_all || m == "activations" || m == "moe" || self.shared_expert;
        let do_map_reduce = do_all || m == "map_reduce" || m == "reduce";
        let do_add_max = do_all || (!do_activations && !do_map_reduce);

        let (created, modified, mut removed) = fuse_elementwise_and_activations_single_traversal(
            module,
            do_mlp,
            do_swiglu,
            do_gelu,
            do_moe,
            do_add_max,
            do_map_reduce,
            self.shared_expert,
            self.top_k,
            self.capacity_factor,
        );

        if (do_activations || do_map_reduce) && (created > 0 || modified > 0) {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// Producer-into-Reduction Fusion Pass (`fuse-map-reduce`).
pub struct FuseMapReducePass;

impl Pass for FuseMapReducePass {
    fn name(&self) -> &'static str {
        "fuse-map-reduce"
    }

    fn description(&self) -> &'static str {
        "Fuses elementwise map producers directly into stablehlo.reduce (fused_map_reduce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let (created, _, mut removed) = fuse_elementwise_and_activations_single_traversal(
            module, false, false, false, false, false, true, false, 2, 1.25,
        );
        if created > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

pub(crate) fn try_fuse_map_reduce_at_op(module: &mut Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return false;
    };
    let Some(StableHloOp::Reduce {
        inputs,
        init_values,
        dimensions,
        ..
    }) = StableHloOp::from_op(&op)
    else {
        return false;
    };
    let reducer = op
        .attributes
        .get("reducer")
        .and_then(|a| a.as_str())
        .unwrap_or("stablehlo.add")
        .to_string();
    let Some(&mapped_val) = inputs.first() else {
        return false;
    };
    let Some(&init_val) = init_values.first() else {
        return false;
    };
    if module.use_count(mapped_val) != 1 {
        return false;
    }
    let Some((src_x, map_kind)) = match_map_producer(module, mapped_val) else {
        return false;
    };

    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.fused_map_reduce".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("fused_map_reduce_impl".to_string()),
    );
    attrs.insert(
        "map_op".to_string(),
        Attribute::String(map_kind.to_string()),
    );
    attrs.insert("reducer".to_string(), Attribute::String(reducer));
    attrs.insert(
        "dimensions".to_string(),
        Attribute::DenseIntArray(dimensions),
    );

    let op_mut = &mut module.ops[op_id];
    op_mut.name = OpName::new("stablehlo", "composite");
    op_mut.operands = SmallVec::from_slice(&[src_x, init_val]);
    op_mut.attributes = attrs;
    true
}

fn match_map_producer(module: &Module, val_id: ValueId) -> Option<(ValueId, &'static str)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match StableHloOp::from_op(op)? {
        StableHloOp::Multiply { lhs, rhs, .. } if lhs == rhs => Some((lhs, "square")),
        StableHloOp::Exp { operand, .. } => Some((operand, "exponential")),
        StableHloOp::Tanh { operand, .. } => Some((operand, "tanh")),
        StableHloOp::Rsqrt { operand, .. } => Some((operand, "rsqrt")),
        _ => None,
    }
}

/// Fuse Elementwise Add+Max helper (`stablehlo.add` + `stablehlo.maximum` -> composite).
pub struct FuseAddMaxPass;

impl Pass for FuseAddMaxPass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let (created, _, removed) = fuse_elementwise_and_activations_single_traversal(
            module, false, false, false, false, true, false, false, 2, 1.25,
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

pub(crate) fn try_fuse_add_max_at_op(module: &mut Module, block_id: BlockId, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::Maximum {
        lhs,
        rhs,
        result: max_res,
    }) = StableHloOp::from_op(op)
    else {
        return false;
    };

    let ValueDef::OpResult { op: add_op_id, .. } = module.values[lhs].def else {
        return false;
    };
    if module.use_count(lhs) != 1 {
        return false;
    }
    let Some(add_op) = module.ops.get(add_op_id) else {
        return false;
    };
    let Some(StableHloOp::Add {
        lhs: add_a,
        rhs: add_b,
        ..
    }) = StableHloOp::from_op(add_op)
    else {
        return false;
    };

    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.fused_add_max".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("fused_add_max_impl".to_string()),
    );

    let max_op_mut = &mut module.ops[op_id];
    max_op_mut.name = OpName::new("stablehlo", "composite");
    max_op_mut.operands = SmallVec::from_slice(&[add_a, add_b, rhs]);
    max_op_mut.attributes = attrs;
    let _ = max_res;

    module.remove_op(block_id, add_op_id);
    true
}
