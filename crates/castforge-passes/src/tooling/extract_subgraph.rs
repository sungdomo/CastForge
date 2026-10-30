//! Unified Subgraph Extractor & Delta IR Reducer Pass (`extract-subgraph`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpId, Type, ValueDef, ValueId};
use smallvec::smallvec;
use std::collections::HashSet;

/// Unified Subgraph Extractor & Delta IR Reducer Pass (`extract-subgraph`).
///
/// - When `max_ops.is_some()` or `keep_op` is non-empty: runs delta IR reduction (`ReduceIrPass`)
///   to shrink the function to at most `max_ops` operations.
/// - Otherwise: slices out the backward dataflow cone feeding `target` and stops at `stop_at`.
#[derive(Debug, Clone, Default)]
pub struct ExtractSubgraphPass {
    /// Target SSA value name (e.g. `"mm"` or `"%mm"`) or 0-based compute op index.
    pub target: String,
    /// Optional semicolon/comma-separated SSA value names to cut and promote to function inputs.
    pub stop_at: String,
    /// Optional maximum number of compute operations to keep (delta reduction mode).
    pub max_ops: Option<usize>,
    /// Optional substring filter on operation name in delta reduction mode.
    pub keep_op: String,
}

impl Pass for ExtractSubgraphPass {
    fn name(&self) -> &'static str {
        "extract-subgraph"
    }

    fn description(&self) -> &'static str {
        "Extracts backward dataflow slice or delta-reduces function to a minimal reproducer"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        if self.max_ops.is_some() || !self.keep_op.is_empty() {
            let mut res = ReduceIrPass {
                max_ops: self.max_ops.unwrap_or(1),
                keep_op: self.keep_op.clone(),
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }

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
        if compute_ops.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let clean_target = self.target.trim().trim_start_matches('%');
        let mut target_val: Option<ValueId> = None;

        if !clean_target.is_empty() {
            for &op_id in &compute_ops {
                for &res in &module.ops[op_id].results {
                    if module.values[res].ssa_name.as_deref() == Some(clean_target) {
                        target_val = Some(res);
                        break;
                    }
                }
            }
            if target_val.is_none()
                && let Ok(idx) = clean_target.parse::<usize>()
                && let Some(&op_id) = compute_ops.get(idx)
            {
                target_val = module.ops[op_id].results.first().copied();
            }
        } else {
            for &op_id in &compute_ops {
                if !module.ops[op_id].is_op("stablehlo", "constant")
                    && let Some(&res) = module.ops[op_id].results.first()
                {
                    target_val = Some(res);
                    break;
                }
            }
        }

        let Some(target_vid) = target_val else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let stop_names: HashSet<String> = self
            .stop_at
            .split([',', ';'])
            .map(|s| s.trim().trim_start_matches('%').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut keep_ops: HashSet<OpId> = HashSet::new();
        let mut promoted_vals: Vec<ValueId> = Vec::new();
        let mut visited_vals: HashSet<ValueId> = HashSet::new();
        let mut worklist = vec![target_vid];

        while let Some(vid) = worklist.pop() {
            if !visited_vals.insert(vid) {
                continue;
            }
            let ssa = module.values[vid].ssa_name.as_deref().unwrap_or("");
            if vid != target_vid && stop_names.contains(ssa) {
                if matches!(module.values[vid].def, ValueDef::OpResult { .. }) {
                    promoted_vals.push(vid);
                }
                continue;
            }
            if let ValueDef::OpResult { op: prod_op, .. } = module.values[vid].def {
                keep_ops.insert(prod_op);
                for &opnd in &module.ops[prod_op].operands {
                    worklist.push(opnd);
                }
            }
        }

        for &p_vid in &promoted_vals {
            let ssa = module.values[p_vid].ssa_name.clone();
            let ty = module.values[p_vid].ty.clone();
            let new_arg = module.add_block_argument(block_id, ssa, ty);
            module.replace_all_uses(p_vid, new_arg);
        }

        let mut removed = 0usize;
        let mut modified = 0usize;
        for &op_id in &ops {
            if module.ops[op_id].is_op("func", "return") {
                if module.ops[op_id].operands.as_slice() != [target_vid] {
                    module.ops[op_id].operands = smallvec![target_vid];
                    modified += 1;
                }
                continue;
            }
            if !keep_ops.contains(&op_id) {
                module.remove_op(block_id, op_id);
                removed += 1;
            }
        }

        let in_types: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let out_types = vec![module.values[target_vid].ty.clone()];
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: in_types,
                results: out_types,
            }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || modified > 0 || !promoted_vals.is_empty(),
            ops_removed: removed,
            ops_modified: modified,
            ops_created: promoted_vals.len(),
        })
    }
}

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
