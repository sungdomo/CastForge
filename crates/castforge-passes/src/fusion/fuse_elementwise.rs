use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef};
use indexmap::IndexMap;
use smallvec::SmallVec;

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

                // Check if `lhs` is produced by `stablehlo.add` with a single use
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

                // Rewrite `max_op` in-place into `stablehlo.composite` representing fused bias+relu
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

                // Remove the now-dead `add` op
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
