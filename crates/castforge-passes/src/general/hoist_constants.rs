use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{BlockId, Module, OpId};
use std::collections::HashMap;

/// Hoist & Deduplicate Constants Pass (`hoist-constants`).
///
/// Moves all `stablehlo.constant` operations in each block to the top of the block and
/// deduplicates identical constant definitions (`(ResultType, Attribute)`), replacing redundant
/// constants with the canonical hoisted SSA value.
pub struct HoistConstantsPass;

impl Pass for HoistConstantsPass {
    fn name(&self) -> &'static str {
        "hoist-constants"
    }

    fn description(&self) -> &'static str {
        "Hoists stablehlo.constant ops to block entry and deduplicates identical constants"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut seen_consts: HashMap<(String, String), castforge_ir::ValueId> = HashMap::new();
            let mut const_ops: Vec<OpId> = Vec::new();
            let mut non_const_ops: Vec<OpId> = Vec::new();

            for op_id in ops.iter().copied() {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if op.is_op("stablehlo", "constant")
                    && let Some(&res_val) = op.results.first()
                    && let Some(val_attr) = op.attributes.get("value")
                {
                    let key = (
                        module.values[res_val].ty.to_string(),
                        format!("{val_attr:?}"),
                    );
                    if let Some(&canon_val) = seen_consts.get(&key) {
                        module.replace_all_uses(res_val, canon_val);
                        module.remove_op(block_id, op_id);
                        removed += 1;
                    } else {
                        seen_consts.insert(key, res_val);
                        const_ops.push(op_id);
                    }
                } else {
                    non_const_ops.push(op_id);
                }
            }

            let mut new_order = Vec::with_capacity(const_ops.len() + non_const_ops.len());
            new_order.extend(const_ops);
            new_order.extend(non_const_ops);

            if module.blocks[block_id].ops != new_order {
                module.blocks[block_id].ops = new_order;
                modified += 1;
            }
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
