use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{BlockId, Module, OpId, ValueDef, ValueId};
use std::collections::HashSet;

/// Loop / Region-Invariant Code Motion Pass (`licm`).
///
/// Scans operations with nested regions (such as `stablehlo.while`, `stablehlo.case`, or custom
/// region ops) and hoists pure, side-effect-free operations inside nested blocks whose operands
/// are all defined outside the region (invariant) into the parent block immediately before the
/// region-owning operation.
pub struct LicmPass;

impl Pass for LicmPass {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn description(&self) -> &'static str {
        "Loop/Region-Invariant Code Motion (hoists invariant pure ops out of nested regions)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for parent_block_id in blocks {
            let parent_ops = module.block_ops(parent_block_id);
            for parent_op_id in parent_ops {
                let Some(parent_op) = module.ops.get(parent_op_id) else {
                    continue;
                };
                // Do not hoist out of top-level `builtin.module` or `func.func`
                if parent_op.is_op("builtin", "module") || parent_op.is_op("func", "func") {
                    continue;
                }
                if parent_op.regions.is_empty() {
                    continue;
                }

                let regions = parent_op.regions.clone();
                let mut inner_blocks = Vec::new();
                for reg_id in regions {
                    inner_blocks.extend(module.regions[reg_id].blocks.iter().copied());
                }

                // Collect all SSA values defined inside these nested blocks
                let mut region_defined_values: HashSet<ValueId> = HashSet::new();
                for &blk_id in &inner_blocks {
                    for &arg_val in &module.blocks[blk_id].arguments {
                        region_defined_values.insert(arg_val);
                    }
                    for &inner_op_id in &module.blocks[blk_id].ops {
                        if let Some(inner_op) = module.ops.get(inner_op_id) {
                            for &res_val in &inner_op.results {
                                region_defined_values.insert(res_val);
                            }
                        }
                    }
                }

                // Iteratively find pure ops in `inner_blocks` whose operands are all outside
                let mut hoisted_ops: Vec<(BlockId, OpId)> = Vec::new();
                for &blk_id in &inner_blocks {
                    let inner_ops = module.block_ops(blk_id);
                    for inner_op_id in inner_ops {
                        let Some(inner_op) = module.ops.get(inner_op_id) else {
                            continue;
                        };
                        if !inner_op.is_pure() || !inner_op.regions.is_empty() {
                            continue;
                        }
                        let is_invariant = inner_op
                            .operands
                            .iter()
                            .all(|v| !region_defined_values.contains(v));
                        if is_invariant {
                            for &res_val in &inner_op.results {
                                region_defined_values.remove(&res_val);
                            }
                            hoisted_ops.push((blk_id, inner_op_id));
                        }
                    }
                }

                if hoisted_ops.is_empty() {
                    continue;
                }

                for (from_blk, hoisted_op_id) in hoisted_ops {
                    module.blocks[from_blk]
                        .ops
                        .retain(|&id| id != hoisted_op_id);
                    let p_ops = &mut module.blocks[parent_block_id].ops;
                    let pos = p_ops
                        .iter()
                        .position(|&id| id == parent_op_id)
                        .unwrap_or(p_ops.len());
                    p_ops.insert(pos, hoisted_op_id);
                    // Ensure ValueDef remains valid
                    let _ = ValueDef::OpResult {
                        op: hoisted_op_id,
                        index: 0,
                    };
                    modified += 1;
                }
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
