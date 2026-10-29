use crate::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Transformer Macro-Fusion Pass: GELU & GeGLU (`fuse-gelu`).
///
/// Recognizes Tanh-Approximation GELU subgraphs:
///   `gelu(x) = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)))`
/// as well as gated GeGLU (`gelu(gate) * up`) and fuses them into a single
/// `stablehlo.composite` operation (`"castforge.fused_geglu"` or `"castforge.fused_gelu"`).
pub struct FuseGeluPass;

impl Pass for FuseGeluPass {
    fn name(&self) -> &'static str {
        "fuse-gelu"
    }

    fn description(&self) -> &'static str {
        "Fuses GELU and gated GeGLU (gelu(gate) * up) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            // Pass 1: Match gated GeGLU `multiply(gelu(gate), up)`
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };
                // Ensure this `multiply` is not itself the outer multiply of a standalone GELU
                if match_gelu_op(module, op).is_some() {
                    continue;
                }
                let matched = match_gelu_val(module, lhs)
                    .map(|g| (g, rhs))
                    .or_else(|| match_gelu_val(module, rhs).map(|g| (g, lhs)));
                let Some((gate_val, up_val)) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_geglu".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_geglu_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[gate_val, up_val]);
                op_mut.attributes = attrs;
                created += 1;
            }

            if created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
            }

            // Pass 2: Match standalone GELU `0.5 * x * (1 + tanh(...))`
            let ops2 = module.block_ops(block_id);
            let mut standalone_created = 0usize;
            for op_id in ops2 {
                let Some(x_val) = module
                    .ops
                    .get(op_id)
                    .and_then(|op| match_gelu_op(module, op))
                else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_gelu".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_gelu_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x_val]);
                op_mut.attributes = attrs;
                standalone_created += 1;
            }

            if standalone_created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
                created += standalone_created;
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

/// Matches a value defined by a Tanh-Approximation GELU subgraph and returns its input `x`.
pub fn match_gelu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match_gelu_op(module, op)
}

pub fn match_gelu_op(module: &Module, op: &castforge_ir::Operation) -> Option<ValueId> {
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    // Collect 3 multiplicative leaves (`0.5`, `x`, `1 + tanh(...)`)
    let mut factors = Vec::with_capacity(3);
    collect_mul_factors(module, lhs, &mut factors, 2);
    collect_mul_factors(module, rhs, &mut factors, 2);
    if factors.len() != 3 {
        return None;
    }

    let half_idx = factors.iter().position(|&v| {
        FoldConstantsPass::const_splat(module, v).is_some_and(|c| (c - 0.5).abs() < 1e-4)
    })?;

    for i in 0..3 {
        if i == half_idx {
            continue;
        }
        let other_idx = 3 - half_idx - i;
        let x_cand = factors[i];
        let one_plus_tanh_cand = factors[other_idx];
        if match_one_plus_tanh_of_x(module, one_plus_tanh_cand, x_cand) {
            return Some(x_cand);
        }
    }
    None
}

fn collect_mul_factors(module: &Module, val: ValueId, out: &mut Vec<ValueId>, depth: usize) {
    if depth > 0
        && let ValueDef::OpResult { op: op_id, .. } = module.values[val].def
        && let Some(op) = module.ops.get(op_id)
        && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op)
    {
        collect_mul_factors(module, lhs, out, depth - 1);
        collect_mul_factors(module, rhs, out, depth - 1);
        return;
    }
    out.push(val);
}

fn match_one_plus_tanh_of_x(module: &Module, val_id: ValueId, x_val: ValueId) -> bool {
    let ValueDef::OpResult { op: add_id, .. } = module.values[val_id].def else {
        return false;
    };
    let Some(add_op) = module.ops.get(add_id) else {
        return false;
    };
    let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(add_op) else {
        return false;
    };
    let tanh_val = if FoldConstantsPass::const_splat(module, lhs)
        .is_some_and(|c| (c - 1.0).abs() < 1e-4)
    {
        rhs
    } else if FoldConstantsPass::const_splat(module, rhs).is_some_and(|c| (c - 1.0).abs() < 1e-4) {
        lhs
    } else {
        return false;
    };

    let ValueDef::OpResult { op: tanh_id, .. } = module.values[tanh_val].def else {
        return false;
    };
    let Some(tanh_op) = module.ops.get(tanh_id) else {
        return false;
    };
    let Some(StableHloOp::Tanh { operand: inner, .. }) = StableHloOp::from_op(tanh_op) else {
        return false;
    };

    // `inner = 0.79788456 * (x + 0.044715 * x^3)`
    let ValueDef::OpResult { op: mul_id, .. } = module.values[inner].def else {
        return false;
    };
    let Some(mul_op) = module.ops.get(mul_id) else {
        return false;
    };
    let Some(StableHloOp::Multiply {
        lhs: m1, rhs: m2, ..
    }) = StableHloOp::from_op(mul_op)
    else {
        return false;
    };
    let poly_val = if FoldConstantsPass::const_splat(module, m1)
        .is_some_and(|c| (c - 0.79788456).abs() < 1e-2)
    {
        m2
    } else if FoldConstantsPass::const_splat(module, m2)
        .is_some_and(|c| (c - 0.79788456).abs() < 1e-2)
    {
        m1
    } else {
        return false;
    };

    let ValueDef::OpResult { op: p_add_id, .. } = module.values[poly_val].def else {
        return false;
    };
    let Some(p_add_op) = module.ops.get(p_add_id) else {
        return false;
    };
    let Some(StableHloOp::Add {
        lhs: p1, rhs: p2, ..
    }) = StableHloOp::from_op(p_add_op)
    else {
        return false;
    };
    let cubic_val = if p1 == x_val {
        p2
    } else if p2 == x_val {
        p1
    } else {
        return false;
    };

    let mut cubic_factors = Vec::new();
    collect_mul_factors(module, cubic_val, &mut cubic_factors, 3);
    if cubic_factors.len() != 4 {
        return false;
    }
    let x_count = cubic_factors.iter().filter(|&&v| v == x_val).count();
    let coeff_ok = cubic_factors.iter().any(|&v| {
        FoldConstantsPass::const_splat(module, v).is_some_and(|c| (c - 0.044715).abs() < 1e-3)
    });
    x_count == 3 && coeff_ok
}
