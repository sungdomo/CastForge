use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{BlockId, Module, OpId};
use std::collections::HashSet;

/// Dead Code Elimination Pass (`dce` — $O(N)$ reference-counted SSA DCE + optional Symbol DCE).
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
        let mut use_counts = module.compute_use_counts();

        loop {
            let mut pass_removed = 0usize;
            for &block_id in &blocks {
                let ops = module.block_ops(block_id);
                let mut dead_in_block: Vec<OpId> = Vec::new();
                for &op_id in ops.iter().rev() {
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
                        dead_in_block.push(op_id);
                        pass_removed += 1;
                    }
                }
                for op_id in dead_in_block {
                    module.remove_op(block_id, op_id);
                }
            }
            removed += pass_removed;
            if pass_removed == 0 {
                break;
            }
        }

        if self.include_symbols {
            let sym_res = SymbolDcePass.run(module)?;
            removed += sym_res.ops_removed;
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

/// Symbol Dead Code Elimination Pass (`symbol-dce`).
///
/// Eliminates unreachable `private` functions from the module.
pub struct SymbolDcePass;

impl Pass for SymbolDcePass {
    fn name(&self) -> &'static str {
        "symbol-dce"
    }

    fn description(&self) -> &'static str {
        "Symbol Dead Code Elimination (removes unreachable private func.func symbols)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut referenced_symbols = HashSet::new();
        for (_, op) in &module.ops {
            if op.is_op("func", "call")
                && let Some(callee) = op.attributes.get("callee").and_then(|a| a.as_str())
            {
                referenced_symbols.insert(callee.to_string());
            }
            if op.is_op("stablehlo", "composite")
                && let Some(decomp) = op.attributes.get("decomposition").and_then(|a| a.as_str())
            {
                referenced_symbols.insert(decomp.to_string());
            }
        }

        let root_block = module.body_block();
        let mut removed = 0usize;
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
                    let blocks = module.regions[reg_id].blocks.clone();
                    for blk_id in blocks {
                        let inner_ops = module.block_ops(blk_id);
                        for inner_op in inner_ops {
                            module.remove_op(blk_id, inner_op);
                        }
                    }
                }
                module.remove_op(root_block, fn_op_id);
                removed += 1;
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
