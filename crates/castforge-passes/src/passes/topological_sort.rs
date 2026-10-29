use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{BlockId, Module, OpId, ValueDef};
use std::collections::HashSet;

/// Topological Sort Pass (`topological-sort`).
///
/// Reorders operations inside each block so all SSA definitions precede their uses.
pub struct TopologicalSortPass;

impl Pass for TopologicalSortPass {
    fn name(&self) -> &'static str {
        "topological-sort"
    }

    fn description(&self) -> &'static str {
        "Reorders operations inside blocks to satisfy SSA def-before-use ordering"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let orig_ops = module.block_ops(block_id);
            if orig_ops.len() <= 1 {
                continue;
            }
            let block_op_set: HashSet<OpId> = orig_ops.iter().copied().collect();
            let mut visited: HashSet<OpId> = HashSet::new();
            let mut sorted: Vec<OpId> = Vec::with_capacity(orig_ops.len());
            let mut terminators: Vec<OpId> = Vec::new();

            fn visit(
                op_id: OpId,
                module: &Module,
                block_op_set: &HashSet<OpId>,
                visited: &mut HashSet<OpId>,
                sorted: &mut Vec<OpId>,
            ) {
                if !visited.insert(op_id) {
                    return;
                }
                if let Some(op) = module.ops.get(op_id) {
                    for &operand in &op.operands {
                        if let ValueDef::OpResult { op: prod_op, .. } = module.values[operand].def
                            && block_op_set.contains(&prod_op)
                        {
                            visit(prod_op, module, block_op_set, visited, sorted);
                        }
                    }
                }
                sorted.push(op_id);
            }

            for &op_id in &orig_ops {
                let is_term = module
                    .ops
                    .get(op_id)
                    .is_some_and(|o| o.is_op("func", "return") || o.is_op("stablehlo", "return"));
                if is_term {
                    terminators.push(op_id);
                } else {
                    visit(op_id, module, &block_op_set, &mut visited, &mut sorted);
                }
            }
            sorted.extend(terminators);

            if sorted != orig_ops {
                module.blocks[block_id].ops = sorted;
                modified += 1;
            }
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
