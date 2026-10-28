use crate::{Pass, PassResult, SymbolDcePass};
use anyhow::Result;
use castforge_ir::{BlockId, Module};

/// 1. Dead Code Elimination (DCE — SSA values + optional Symbol DCE)
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

        loop {
            let mut pass_removed = 0usize;
            for &block_id in &blocks {
                let ops = module.block_ops(block_id);
                // Iterate in reverse so chains of dead ops are cleaned up quickly
                for &op_id in ops.iter().rev() {
                    let Some(op) = module.ops.get(op_id) else {
                        continue;
                    };
                    if !op.is_pure() || op.results.is_empty() {
                        continue;
                    }
                    let all_unused = op.results.iter().all(|&res| module.use_count(res) == 0);
                    if all_unused {
                        module.remove_op(block_id, op_id);
                        pass_removed += 1;
                    }
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
