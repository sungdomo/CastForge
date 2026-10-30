use crate::ir::{BlockId, Module, OpId, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Dead Code Elimination Pass (`dce` — $O(N)$ reference-counted SSA + Symbol DCE in a single
/// reverse sweep with batch `retain` compaction).
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
        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        let mut use_counts: HashMap<ValueId, usize> = HashMap::with_capacity(module.values.len());
        let mut referenced_symbols: HashSet<String> = HashSet::new();

        for &block_id in &blocks {
            for &op_id in &module.blocks[block_id].ops {
                if let Some(op) = module.ops.get(op_id) {
                    for &operand in &op.operands {
                        *use_counts.entry(operand).or_insert(0) += 1;
                    }
                    if self.include_symbols {
                        if op.is_op("func", "call")
                            && let Some(c) = op.attributes.get("callee").and_then(|a| a.as_str())
                        {
                            referenced_symbols.insert(c.to_string());
                        }
                        if op.is_op("stablehlo", "composite")
                            && let Some(d) =
                                op.attributes.get("decomposition").and_then(|a| a.as_str())
                        {
                            referenced_symbols.insert(d.to_string());
                        }
                    }
                }
            }
        }

        for &block_id in blocks.iter().rev() {
            let mut dead_in_block: HashSet<OpId> = HashSet::new();
            for &op_id in module.blocks[block_id].ops.iter().rev() {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_pure() || op.results.is_empty() {
                    continue;
                }
                let all_unused = op
                    .results
                    .iter()
                    .all(|res| use_counts.get(res).copied().unwrap_or(0) == 0);
                if all_unused {
                    for &operand in &op.operands {
                        if let Some(cnt) = use_counts.get_mut(&operand) {
                            *cnt = cnt.saturating_sub(1);
                        }
                    }
                    dead_in_block.insert(op_id);
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
        }

        if self.include_symbols {
            let root_block = module.body_block();
            let mut dead_fns: HashSet<OpId> = HashSet::new();
            for fn_op_id in module.functions() {
                let op = &module.ops[fn_op_id];
                let is_private =
                    op.attributes.get("sym_visibility").and_then(|a| a.as_str()) == Some("private");
                let sym_name = op
                    .attributes
                    .get("sym_name")
                    .and_then(|a| a.as_str())
                    .unwrap_or("");
                if is_private && !referenced_symbols.contains(sym_name) {
                    let regions = op.regions.clone();
                    for reg_id in regions {
                        let fn_blocks = module.regions[reg_id].blocks.clone();
                        for blk_id in fn_blocks {
                            for inner_op in module.blocks[blk_id].ops.drain(..) {
                                module.ops.remove(inner_op);
                            }
                        }
                    }
                    dead_fns.insert(fn_op_id);
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
