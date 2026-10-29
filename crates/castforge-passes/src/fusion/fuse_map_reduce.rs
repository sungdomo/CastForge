use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Producer-into-Reduction Fusion Pass (`fuse-map-reduce`).
///
/// Fuses single-consumer elementwise map operations (`multiply(x, x)`, `exponential(x)`,
/// `tanh(x)`, `rsqrt(x)`) directly into downstream `stablehlo.reduce` operations as
/// `"castforge.fused_map_reduce"`, accumulating `map(x[r, c])` in registers without
/// materializing the intermediate `[M, N]` tensor in memory.
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
                if count_uses(module, mapped_val) != 1 {
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

fn count_uses(module: &Module, val_id: ValueId) -> usize {
    module
        .ops
        .values()
        .flat_map(|op| op.operands.iter())
        .filter(|&&v| v == val_id)
        .count()
}
