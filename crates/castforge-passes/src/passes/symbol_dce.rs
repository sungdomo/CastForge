use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::Module;
use std::collections::HashSet;

/// 15. Symbol Dead Code Elimination Pass (`symbol-dce`)
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
            if op.is_op("func", "call") {
                if let Some(callee) = op.attributes.get("callee").and_then(|a| a.as_str()) {
                    referenced_symbols.insert(callee.to_string());
                }
            }
            if op.is_op("stablehlo", "composite") {
                if let Some(decomp) = op.attributes.get("decomposition").and_then(|a| a.as_str()) {
                    referenced_symbols.insert(decomp.to_string());
                }
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
