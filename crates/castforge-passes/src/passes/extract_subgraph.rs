//! Subgraph Extractor / Model Slicer Pass (`extract-subgraph`: `Ir -> Ir`).
//!
//! Slices out the backward dataflow cone feeding a target SSA value or operation index,
//! promotes any cut incoming edges (`stop_at`) into `func.func` block arguments, rewires
//! `func.return` to the target value, and prunes all operations outside the slice.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpId, Type, ValueDef, ValueId};
use smallvec::smallvec;
use std::collections::HashSet;

/// Subgraph Extractor Pass (`extract-subgraph`).
#[derive(Debug, Clone, Default)]
pub struct ExtractSubgraphPass {
    /// Target SSA value name (e.g. `"mm"` or `"%mm"`) or 0-based compute op index.
    pub target: String,
    /// Optional semicolon/comma-separated SSA value names to cut and promote to function inputs.
    pub stop_at: String,
}

impl Pass for ExtractSubgraphPass {
    fn name(&self) -> &'static str {
        "extract-subgraph"
    }

    fn description(&self) -> &'static str {
        "Extracts the backward dataflow slice for a target SSA value into a standalone function"
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
