use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{BlockId, Module, ValueDef};

/// 4. Fold Reshape Pass (`reshape(reshape(x))` -> `reshape(x)` & identity reshape removal)
pub struct FoldReshapePass;

impl Pass for FoldReshapePass {
    fn name(&self) -> &'static str {
        "fold-reshape"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.reshape ops and eliminates identity reshapes"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Reshape { operand, result }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut current_operand = operand;
                if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def {
                    if let Some(prod_op) = module.ops.get(prod_op_id) {
                        if let Some(StableHloOp::Reshape {
                            operand: base_input,
                            ..
                        }) = StableHloOp::from_op(prod_op)
                        {
                            current_operand = base_input;
                            module.ops[op_id].operands[0] = current_operand;
                            modified += 1;
                        }
                    }
                }

                // If input type and output type of reshape are identical, eliminate it!
                if module.values[current_operand].ty == module.values[result].ty {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
