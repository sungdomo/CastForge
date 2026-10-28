use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{BlockId, Module, OpId};
use std::collections::HashMap;

/// 2. Common Subexpression Elimination (CSE)
pub struct CommonSubexpressionEliminationPass;

impl Pass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Common Subexpression Elimination (deduplicates identical pure operations)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let mut seen: HashMap<String, OpId> = HashMap::new();
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                    continue;
                }
                let res_val = op.results[0];
                let res_ty = &module.values[res_val].ty;
                let sig = format!(
                    "{}|operands:{:?}|attrs:{:?}|ty:{}",
                    op.name, op.operands, op.attributes, res_ty
                );

                if let Some(&existing_op_id) = seen.get(&sig) {
                    let existing_res = module.ops[existing_op_id].results[0];
                    module.replace_all_uses(res_val, existing_res);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                } else {
                    seen.insert(sig, op_id);
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}
