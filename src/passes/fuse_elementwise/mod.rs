//! Unified Elementwise, Map-Reduce, Loop-Chain Fusion & Composite Decomposition
//! (`fuse-elementwise`).
//!
//! Organized into cohesive submodules:
//! - [`mod.rs`](self): `UnifiedFuseElementwisePass`, `FuseMapReducePass`, `FuseElementwisePass`
//! - [`loops`]: `FuseLoopsPass`, `FissionLoopsPass`, `FuseMegakernelPass`
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

use crate::ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Elementwise, Activation, Gated-MLP, MoE, Map-Reduce, Loop-Chain Fusion & Composite
/// Decomposition Pass (`fuse-elementwise`).
pub struct UnifiedFuseElementwisePass {
    pub mode: String,
    pub max_steps: usize,
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for UnifiedFuseElementwisePass {
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

impl Pass for UnifiedFuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Unified elementwise, activation (SwiGLU/GELU/GeGLU/MoE), map-reduce & loop-chain fusion"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        if matches!(
            m.as_str(),
            "activations" | "swiglu" | "gelu" | "geglu" | "gated_mlp" | "mlp" | "moe"
        ) || self.shared_expert
        {
            let kind = if m == "activations" || m == "composites" {
                "all".to_string()
            } else {
                m
            };
            steps.push(
                FuseActivationsPass {
                    kind,
                    shared_expert: self.shared_expert,
                    top_k: self.top_k,
                    capacity_factor: self.capacity_factor,
                }
                .run(module)?,
            );
        } else if matches!(m.as_str(), "decompose" | "expand" | "lower") {
            steps.push(DecomposeCompositesPass.run(module)?);
        } else if m == "loops" || m == "chain" || self.max_steps > 0 {
            steps.push(FuseLoopsPass.run(module)?);
            if self.max_steps > 0 {
                steps.push(
                    FissionLoopsPass {
                        max_steps: self.max_steps,
                    }
                    .run(module)?,
                );
            }
        } else if m == "map_reduce" || m == "reduce" {
            steps.push(FuseMapReducePass.run(module)?);
        } else if m == "megakernel" || m == "persistent" {
            steps.push(FuseMegakernelPass.run(module)?);
        } else if m == "all" {
            steps.push(FuseElementwisePass.run(module)?);
            steps.push(FuseMapReducePass.run(module)?);
            steps.push(FuseActivationsPass::default().run(module)?);
        } else {
            steps.push(FuseElementwisePass.run(module)?);
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
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Reduce {
                    inputs,
                    init_values,
                    dimensions,
                    ..
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                let reducer = op
                    .attributes
                    .get("reducer")
                    .and_then(|a| a.as_str())
                    .unwrap_or("stablehlo.add")
                    .to_string();
                let Some(&mapped_val) = inputs.first() else {
                    continue;
                };
                let Some(&init_val) = init_values.first() else {
                    continue;
                };
                if module.use_count(mapped_val) != 1 {
                    continue;
                }
                let Some((src_x, map_kind)) = match_map_producer(module, mapped_val) else {
                    continue;
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
                created += 1;
            }
        }

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

/// Fuse Elementwise Pass (`stablehlo.add` + `stablehlo.maximum` -> `stablehlo.composite`).
pub struct FuseElementwisePass;

impl Pass for FuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Maximum {
                    lhs,
                    rhs,
                    result: max_res,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let ValueDef::OpResult { op: add_op_id, .. } = module.values[lhs].def else {
                    continue;
                };
                if module.use_count(lhs) != 1 {
                    continue;
                }
                let Some(add_op) = module.ops.get(add_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add {
                    lhs: add_a,
                    rhs: add_b,
                    ..
                }) = StableHloOp::from_op(add_op)
                else {
                    continue;
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
                created += 1;

                module.remove_op(block_id, add_op_id);
                removed += 1;
            }
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
