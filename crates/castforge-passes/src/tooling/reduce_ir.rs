//! Delta IR Reducer Pass (`reduce-ir`: `Ir -> Ir`).
//!
//! Inspired by `mlir-reduce` and `bugpoint`: shrinks a function down to a minimal valid
//! reproducer containing at most `max_ops` compute operations (optionally filtered by `keep_op`),
//! automatically promoting any cut incoming operands to `func.func` block arguments and rewiring
//! `func.return`.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpId, Type, ValueDef, ValueId};
use smallvec::smallvec;
use std::collections::HashSet;

/// Delta IR Reducer Pass (`reduce-ir`).
#[derive(Debug, Clone)]
pub struct ReduceIrPass {
    /// Maximum number of compute operations to keep (default: `1`).
    pub max_ops: usize,
    /// Optional substring filter on operation name (e.g. `"dot_general"`, `"add"`).
    pub keep_op: String,
}

impl Default for ReduceIrPass {
    fn default() -> Self {
        Self {
            max_ops: 1,
            keep_op: String::new(),
        }
    }
}

impl Pass for ReduceIrPass {
    fn name(&self) -> &'static str {
        "reduce-ir"
    }

    fn description(&self) -> &'static str {
        "Delta-reduces a function to a minimal reproducer of at most max_ops operations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let compute_ops: Vec<OpId> = ops
            .iter()
            .copied()
            .filter(|&id| !module.ops[id].is_op("func", "return"))
            .collect();
        let limit = self.max_ops.max(1);
        if compute_ops.len() <= limit && self.keep_op.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let filter = self.keep_op.trim().to_ascii_lowercase();
        let mut retained: Vec<OpId> = compute_ops
            .iter()
            .copied()
            .filter(|&id| {
                if filter.is_empty() {
                    !module.ops[id].is_op("stablehlo", "constant")
                } else {
                    module.ops[id]
                        .name
                        .to_string()
                        .to_ascii_lowercase()
                        .contains(&filter)
                }
            })
            .take(limit)
            .collect();

        if retained.is_empty()
            && let Some(&first_op) = compute_ops.first()
        {
            retained.push(first_op);
        }

        let Some(&last_op) = retained.last() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let Some(&ret_val) = module.ops[last_op].results.first() else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let keep_set: HashSet<OpId> = retained.iter().copied().collect();
        let mut to_promote: Vec<ValueId> = Vec::new();
        for &op_id in &retained {
            for &opnd in &module.ops[op_id].operands {
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[opnd].def
                    && !keep_set.contains(&prod_op)
                    && !to_promote.contains(&opnd)
                {
                    to_promote.push(opnd);
                }
            }
        }

        for &vid in &to_promote {
            let ssa = module.values[vid].ssa_name.clone();
            let ty = module.values[vid].ty.clone();
            let new_arg = module.add_block_argument(block_id, ssa, ty);
            module.replace_all_uses(vid, new_arg);
        }

        let mut removed = 0usize;
        for &op_id in &ops {
            if module.ops[op_id].is_op("func", "return") {
                module.ops[op_id].operands = smallvec![ret_val];
                continue;
            }
            if !keep_set.contains(&op_id) {
                module.remove_op(block_id, op_id);
                removed += 1;
            }
        }

        let in_types: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let out_types = vec![module.values[ret_val].ty.clone()];
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: in_types,
                results: out_types,
            }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || !to_promote.is_empty(),
            ops_removed: removed,
            ops_modified: 1,
            ops_created: to_promote.len(),
        })
    }
}

/// Pass-Pipeline Bisector Checkpoint Pass (`bisect-passes`: `Ir -> Ir`).
///
/// Inspired by `clang -opt-bisect-limit`: verifies IR structural invariants and records a
/// numbered bisection checkpoint attribute (`castforge.bisect_step`) on the entry function.
#[derive(Debug, Clone)]
pub struct BisectPassesPass {
    /// Checkpoint step limit or label (default: `1`).
    pub limit: usize,
}

impl Default for BisectPassesPass {
    fn default() -> Self {
        Self { limit: 1 }
    }
}

impl Pass for BisectPassesPass {
    fn name(&self) -> &'static str {
        "bisect-passes"
    }

    fn description(&self) -> &'static str {
        "Verifies IR invariants and stamps a numbered bisection checkpoint on the entry function"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        module.verify()?;
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let prev = module.ops[fn_op_id]
            .attributes
            .get("castforge.bisect_step")
            .and_then(|a| match a {
                Attribute::Int(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(0);
        let next = (prev + 1).min(self.limit.max(1));
        module.ops[fn_op_id].attributes.insert(
            "castforge.bisect_step".to_string(),
            Attribute::Int(next as i64),
        );
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: 0,
        })
    }
}
