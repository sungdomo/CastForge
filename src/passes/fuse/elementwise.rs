//! Unified Elementwise, Map-Reduce, Loop-Chain Fusion & Composite Decomposition
//! (`fuse-elementwise`).

use super::activations::*;
use super::decompose::*;
use super::loops::*;
use crate::ir::{Attribute, BlockId, Module, OpId, OpName, ValueDef, ValueId};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult, SimplifyAlgebraicPass};
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

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "fuse-activations",
            "fuse-swiglu",
            "fuse-gelu",
            "fuse-gated-mlp",
            "fuse-moe",
            "fuse-map-reduce",
            "fuse-loops",
            "split-loops",
            "fuse-megakernel",
            "decompose-composites",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        if matches!(m.as_str(), "decompose" | "expand" | "lower") {
            return Ok(DecomposeCompositesPass.run(module)?.with_name(self.name()));
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
            return Ok(PassResult::aggregate(self.name(), &steps));
        }
        if m == "megakernel" || m == "persistent" {
            return Ok(FuseMegakernelPass.run(module)?.with_name(self.name()));
        }
        if matches!(
            m.as_str(),
            "activations" | "swiglu" | "gelu" | "geglu" | "gated_mlp" | "mlp" | "moe"
        ) || (self.shared_expert && m != "all")
        {
            let kind = if m == "activations" {
                "all".to_string()
            } else {
                m
            };
            return Ok(FuseActivationsPass {
                kind,
                shared_expert: self.shared_expert,
                top_k: self.top_k,
                capacity_factor: self.capacity_factor,
            }
            .run(module)?
            .with_name(self.name()));
        }
        if m == "map_reduce" || m == "reduce" {
            return Ok(FuseMapReducePass.run(module)?.with_name(self.name()));
        }
        if m == "all" {
            let steps = [
                FuseActivationsPass {
                    kind: "all".to_string(),
                    shared_expert: self.shared_expert,
                    top_k: self.top_k,
                    capacity_factor: self.capacity_factor,
                }
                .run(module)?,
                FuseMapReducePass.run(module)?,
                FuseAddMaxPass.run(module)?,
            ];
            return Ok(PassResult::aggregate(self.name(), &steps));
        }

        Ok(FuseAddMaxPass.run(module)?.with_name(self.name()))
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

    let mut matched = None;
    for (add_cand, other_opnd) in [(lhs, rhs), (rhs, lhs)] {
        let ValueDef::OpResult { op: add_op_id, .. } = module.values[add_cand].def else {
            continue;
        };
        if module.use_count(add_cand) != 1 {
            continue;
        };
        let Some(add_op) = module.ops.get(add_op_id) else {
            continue;
        };
        if let Some(StableHloOp::Add {
            lhs: add_a,
            rhs: add_b,
            ..
        }) = StableHloOp::from_op(add_op)
        {
            matched = Some((add_op_id, add_a, add_b, other_opnd));
            break;
        }
    }
    let Some((add_op_id, add_a, add_b, clamp_opnd)) = matched else {
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
    max_op_mut.operands = SmallVec::from_slice(&[add_a, add_b, clamp_opnd]);
    max_op_mut.attributes = attrs;
    let _ = max_res;

    module.remove_op(block_id, add_op_id);
    true
}

/// E-Graph Distributive Factoring & Fused Multiply-Add Extraction Pass (`fuse-fma`).
#[derive(Debug, Clone, Default)]
pub struct FuseFmaPass;

impl Pass for FuseFmaPass {
    fn name(&self) -> &'static str {
        "fuse-fma"
    }

    fn description(&self) -> &'static str {
        "Applies e-graph distributive factoring (a*b + a*c -> a*(b+c)) and extracts castforge.fma"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(SimplifyAlgebraicPass {
            fold_constants: false,
            include_conditional: false,
            egraph: true,
            fuse_fma: true,
        }
        .run(module)?
        .with_name(self.name()))
    }
}
