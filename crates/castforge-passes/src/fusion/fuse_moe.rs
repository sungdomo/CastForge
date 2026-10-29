use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Mixture-of-Experts (MoE) Routing & Expert Fusion Pass (`fuse-moe`).
///
/// Recognizes dual routed expert projections combined by router gates:
///   `e0 = dot_general(x, w_e0)`
///   `e1 = dot_general(x, w_e1)`
///   `y = add(multiply(e0, g0), multiply(e1, g1))`
/// and fuses the entire subgraph into a single `stablehlo.composite` operation
/// (`"castforge.fused_moe"`, `decomposition = @fused_moe_impl`, `num_experts = 2`)
/// taking `(x, w_e0, w_e1, g0, g1)`.
pub struct FuseMoePass;

impl Pass for FuseMoePass {
    fn name(&self) -> &'static str {
        "fuse-moe"
    }

    fn description(&self) -> &'static str {
        "Fuses routed Mixture-of-Experts (MoE) expert GEMMs and gate combine into fused_moe"
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
                let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op) else {
                    continue;
                };
                let Some((x0, w0, g0)) = match_gated_expert(module, lhs) else {
                    continue;
                };
                let Some((x1, w1, g1)) = match_gated_expert(module, rhs) else {
                    continue;
                };
                if x0 != x1 {
                    continue;
                }

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_moe".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_moe_impl".to_string()),
                );
                attrs.insert("num_experts".to_string(), Attribute::Int(2));

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x0, w0, w1, g0, g1]);
                op_mut.attributes = attrs;
                created += 1;
            }
        }

        if created > 0 {
            let dce = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: created,
            ops_created: created,
        })
    }
}

fn match_gated_expert(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
    let ValueDef::OpResult { op: mul_id, .. } = module.values[val_id].def else {
        return None;
    };
    let mul_op = module.ops.get(mul_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(mul_op)? else {
        return None;
    };
    match_dot_2d(module, lhs)
        .map(|(x, w)| (x, w, rhs))
        .or_else(|| match_dot_2d(module, rhs).map(|(x, w)| (x, w, lhs)))
}

fn match_dot_2d(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: dot_id, .. } = module.values[val_id].def else {
        return None;
    };
    let dot_op = module.ops.get(dot_id)?;
    let StableHloOp::DotGeneral { lhs, rhs, .. } = StableHloOp::from_op(dot_op)? else {
        return None;
    };
    Some((lhs, rhs))
}
