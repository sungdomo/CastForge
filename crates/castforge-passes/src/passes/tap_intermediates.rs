//! Intermediate Activation Tapper Pass (`tap-intermediates`: `Ir -> Ir`).
//!
//! Appends selected (or all non-constant) intermediate SSA values to `func.return` and updates
//! `function_type` to a multi-result signature `-> (tensor<...>, tensor<...>, ...)` for
//! layer-by-layer activation inspection and parity verification.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, Type, ValueId};
use std::collections::HashSet;

/// Intermediate Activation Tapper Pass (`tap-intermediates`).
#[derive(Debug, Clone, Default)]
pub struct TapIntermediatesPass {
    /// Semicolon/comma-separated SSA value names to append to `func.return` (empty = all compute).
    pub values: String,
}

impl Pass for TapIntermediatesPass {
    fn name(&self) -> &'static str {
        "tap-intermediates"
    }

    fn description(&self) -> &'static str {
        "Appends intermediate SSA activations to func.return for multi-output layer inspection"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let Some(&ret_op_id) = ops
            .iter()
            .find(|&&id| module.ops[id].is_op("func", "return"))
        else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let filter: HashSet<String> = self
            .values
            .split([',', ';'])
            .map(|s| s.trim().trim_start_matches('%').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut tapped: Vec<ValueId> = Vec::new();
        for &op_id in &ops {
            if op_id == ret_op_id || module.ops[op_id].is_op("stablehlo", "constant") {
                continue;
            }
            for &res in &module.ops[op_id].results {
                let ssa = module.values[res].ssa_name.as_deref().unwrap_or("");
                let matched = if filter.is_empty() {
                    true
                } else {
                    filter.contains(ssa)
                };
                if matched && !module.ops[ret_op_id].operands.contains(&res) {
                    tapped.push(res);
                }
            }
        }

        if tapped.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        for &vid in &tapped {
            module.ops[ret_op_id].operands.push(vid);
        }

        let inputs: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let results: Vec<Type> = module.ops[ret_op_id]
            .operands
            .iter()
            .map(|&v| module.values[v].ty.clone())
            .collect();
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function { inputs, results }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: tapped.len(),
        })
    }
}
