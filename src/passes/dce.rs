use crate::ir::{BlockId, Module, OpId, RegionId, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use std::collections::HashSet;

/// Dead Code Elimination Pass (`dce` — $O(N)$ liveness-driven SSA + Symbol DCE in a single
/// reverse topological traversal).
pub struct DeadCodeEliminationPass {
    pub include_symbols: bool,
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self {
            include_symbols: true,
        }
    }
}

impl DeadCodeEliminationPass {
    pub fn ssa_only() -> Self {
        Self {
            include_symbols: false,
        }
    }
}

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn description(&self) -> &'static str {
        "Dead Code Elimination (removes unused pure SSA ops and dead private symbols)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut live_values: HashSet<ValueId> = HashSet::with_capacity(module.values.len());
        let mut referenced_symbols: HashSet<String> = HashSet::new();
        let mut visited_blocks: HashSet<BlockId> = HashSet::with_capacity(module.blocks.len());

        let funcs = module.functions();
        let mut private_funcs: Vec<OpId> = Vec::new();

        // 1. Sweep all non-private functions in reverse topological order first so their
        //    live call/composite references populate `referenced_symbols`.
        for &fn_op_id in &funcs {
            let op = &module.ops[fn_op_id];
            let is_private =
                op.attributes.get("sym_visibility").and_then(|a| a.as_str()) == Some("private");
            if is_private {
                private_funcs.push(fn_op_id);
                continue;
            }
            let regions = op.regions.clone();
            for reg_id in regions.into_iter().rev() {
                removed += sweep_region_reverse(
                    module,
                    reg_id,
                    self.include_symbols,
                    &mut live_values,
                    &mut referenced_symbols,
                    &mut visited_blocks,
                );
            }
        }

        // 2. Process private functions: either eliminate unreferenced private symbols outright
        //    or sweep referenced private function bodies in reverse.
        let root_block = module.body_block();
        let mut dead_fns: HashSet<OpId> = HashSet::new();
        for fn_op_id in private_funcs.into_iter().rev() {
            let op = &module.ops[fn_op_id];
            let sym_name = op
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            if self.include_symbols && !referenced_symbols.contains(sym_name) {
                let regions = op.regions.clone();
                for reg_id in regions {
                    let fn_blocks = module.regions[reg_id].blocks.clone();
                    for blk_id in fn_blocks {
                        visited_blocks.insert(blk_id);
                        for inner_op in module.blocks[blk_id].ops.drain(..) {
                            module.ops.remove(inner_op);
                        }
                    }
                }
                dead_fns.insert(fn_op_id);
            } else {
                let regions = op.regions.clone();
                for reg_id in regions.into_iter().rev() {
                    removed += sweep_region_reverse(
                        module,
                        reg_id,
                        self.include_symbols,
                        &mut live_values,
                        &mut referenced_symbols,
                        &mut visited_blocks,
                    );
                }
            }
        }

        if !dead_fns.is_empty() {
            removed += dead_fns.len();
            module.blocks[root_block]
                .ops
                .retain(|id| !dead_fns.contains(id));
            for fn_id in dead_fns {
                module.ops.remove(fn_id);
            }
        }

        // 3. Sweep any standalone/non-function blocks not yet visited.
        visited_blocks.insert(root_block);
        let all_blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in all_blocks.into_iter().rev() {
            if !visited_blocks.contains(&block_id) {
                removed += sweep_block_reverse(
                    module,
                    block_id,
                    self.include_symbols,
                    &mut live_values,
                    &mut referenced_symbols,
                    &mut visited_blocks,
                );
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

fn sweep_region_reverse(
    module: &mut Module,
    region_id: RegionId,
    include_symbols: bool,
    live_values: &mut HashSet<ValueId>,
    referenced_symbols: &mut HashSet<String>,
    visited_blocks: &mut HashSet<BlockId>,
) -> usize {
    let Some(reg) = module.regions.get(region_id) else {
        return 0;
    };
    let blocks = reg.blocks.clone();
    let mut removed = 0usize;
    for block_id in blocks.into_iter().rev() {
        removed += sweep_block_reverse(
            module,
            block_id,
            include_symbols,
            live_values,
            referenced_symbols,
            visited_blocks,
        );
    }
    removed
}

fn sweep_block_reverse(
    module: &mut Module,
    block_id: BlockId,
    include_symbols: bool,
    live_values: &mut HashSet<ValueId>,
    referenced_symbols: &mut HashSet<String>,
    visited_blocks: &mut HashSet<BlockId>,
) -> usize {
    if !visited_blocks.insert(block_id) {
        return 0;
    }
    let mut removed = 0usize;
    let mut dead_in_block: HashSet<OpId> = HashSet::new();
    let ops = module.blocks[block_id].ops.clone();

    for op_id in ops.into_iter().rev() {
        let Some(op) = module.ops.get(op_id).cloned() else {
            continue;
        };
        let is_dead = op.is_pure()
            && !op.results.is_empty()
            && op.results.iter().all(|res| !live_values.contains(res));
        if is_dead {
            for reg_id in &op.regions {
                if let Some(reg) = module.regions.get(*reg_id) {
                    let inner_blocks = reg.blocks.clone();
                    for blk_id in inner_blocks {
                        visited_blocks.insert(blk_id);
                        for inner_op in module.blocks[blk_id].ops.drain(..) {
                            module.ops.remove(inner_op);
                        }
                    }
                }
            }
            dead_in_block.insert(op_id);
            continue;
        }

        for &operand in &op.operands {
            live_values.insert(operand);
        }
        if include_symbols {
            if op.is_op("func", "call")
                && let Some(c) = op.attributes.get("callee").and_then(|a| a.as_str())
            {
                referenced_symbols.insert(c.to_string());
            }
            if op.is_op("stablehlo", "composite")
                && let Some(d) = op.attributes.get("decomposition").and_then(|a| a.as_str())
            {
                referenced_symbols.insert(d.to_string());
            }
        }
        for reg_id in op.regions.into_iter().rev() {
            removed += sweep_region_reverse(
                module,
                reg_id,
                include_symbols,
                live_values,
                referenced_symbols,
                visited_blocks,
            );
        }
    }

    if !dead_in_block.is_empty() {
        removed += dead_in_block.len();
        module.blocks[block_id]
            .ops
            .retain(|id| !dead_in_block.contains(id));
        for dead_id in dead_in_block {
            module.ops.remove(dead_id);
        }
    }
    removed
}
