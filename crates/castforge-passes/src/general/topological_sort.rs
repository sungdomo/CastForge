//! Topological Sort & Peak-Memory-Minimizing Scheduler Pass (`topological-sort`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Module, OpId, ValueDef, ValueId};
use std::collections::{HashMap, HashSet};

/// Topological Sort & Peak-Memory Scheduler Pass (`topological-sort`).
///
/// - `mode = "standard"` (default): Reorders operations inside each block to satisfy SSA
///   def-before-use ordering while preserving original relative order where possible.
/// - `mode = "min_memory"`: Uses a Sethi-Ullman / minimum-live-tensor priority schedule that
///   prioritizes operations freeing large intermediate tensors before operations allocating new
///   large tensors, reducing peak arena footprint.
pub struct TopologicalSortPass {
    pub mode: String,
}

impl Default for TopologicalSortPass {
    fn default() -> Self {
        Self {
            mode: "standard".to_string(),
        }
    }
}

impl Pass for TopologicalSortPass {
    fn name(&self) -> &'static str {
        "topological-sort"
    }

    fn description(&self) -> &'static str {
        "Reorders operations for SSA def-before-use and optional peak-memory minimization"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let min_mem = matches!(
            self.mode.to_ascii_lowercase().as_str(),
            "min_memory" | "memory" | "sethi_ullman"
        );
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let orig_ops = module.block_ops(block_id);
            if orig_ops.len() <= 1 {
                continue;
            }
            let sorted = if min_mem {
                schedule_min_memory(module, &orig_ops)
            } else {
                schedule_dfs(module, &orig_ops)
            };

            if sorted != orig_ops || min_mem {
                if min_mem
                    && let Some(&target_op) = sorted.iter().rev().find(|&&id| {
                        module.ops.get(id).is_some_and(|o| {
                            !o.is_op("func", "return")
                                && !o.is_op("stablehlo", "return")
                                && !o.is_op("stablehlo", "constant")
                        })
                    })
                {
                    module.ops[target_op].attributes.insert(
                        "castforge.min_memory_scheduled".to_string(),
                        Attribute::Bool(true),
                    );
                }
                module.blocks[block_id].ops = sorted;
                modified += 1;
            }
        }

        if min_mem {
            for fn_id in module.functions() {
                module.ops[fn_id].attributes.insert(
                    "castforge.min_memory_scheduled".to_string(),
                    Attribute::Bool(true),
                );
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

fn schedule_dfs(module: &Module, orig_ops: &[OpId]) -> Vec<OpId> {
    let block_op_set: HashSet<OpId> = orig_ops.iter().copied().collect();
    let mut visited: HashSet<OpId> = HashSet::new();
    let mut sorted: Vec<OpId> = Vec::with_capacity(orig_ops.len());
    let mut terminators: Vec<OpId> = Vec::new();

    fn visit(
        op_id: OpId,
        module: &Module,
        block_op_set: &HashSet<OpId>,
        visited: &mut HashSet<OpId>,
        sorted: &mut Vec<OpId>,
    ) {
        if !visited.insert(op_id) {
            return;
        }
        if let Some(op) = module.ops.get(op_id) {
            for &operand in &op.operands {
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[operand].def
                    && block_op_set.contains(&prod_op)
                {
                    visit(prod_op, module, block_op_set, visited, sorted);
                }
            }
        }
        sorted.push(op_id);
    }

    for &op_id in orig_ops {
        let is_term = module
            .ops
            .get(op_id)
            .is_some_and(|o| o.is_op("func", "return") || o.is_op("stablehlo", "return"));
        if is_term {
            terminators.push(op_id);
        } else {
            visit(op_id, module, &block_op_set, &mut visited, &mut sorted);
        }
    }
    sorted.extend(terminators);
    sorted
}

fn schedule_min_memory(module: &Module, orig_ops: &[OpId]) -> Vec<OpId> {
    let block_op_set: HashSet<OpId> = orig_ops.iter().copied().collect();
    let mut non_term_ops: Vec<OpId> = Vec::new();
    let mut terminators: Vec<OpId> = Vec::new();
    for &op_id in orig_ops {
        let is_term = module
            .ops
            .get(op_id)
            .is_some_and(|o| o.is_op("func", "return") || o.is_op("stablehlo", "return"));
        if is_term {
            terminators.push(op_id);
        } else {
            non_term_ops.push(op_id);
        }
    }

    let mut in_degree: HashMap<OpId, usize> = HashMap::new();
    let mut users: HashMap<OpId, Vec<OpId>> = HashMap::new();
    let mut remaining_uses: HashMap<ValueId, usize> = HashMap::new();

    for &op_id in &non_term_ops {
        in_degree.insert(op_id, 0);
        if let Some(op) = module.ops.get(op_id) {
            let mut preds: HashSet<OpId> = HashSet::new();
            for &v in &op.operands {
                *remaining_uses.entry(v).or_insert(0) += 1;
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[v].def
                    && block_op_set.contains(&prod_op)
                    && prod_op != op_id
                {
                    preds.insert(prod_op);
                }
            }
            *in_degree.get_mut(&op_id).unwrap() = preds.len();
            for p in preds {
                users.entry(p).or_default().push(op_id);
            }
        }
    }
    for &term_id in &terminators {
        if let Some(op) = module.ops.get(term_id) {
            for &v in &op.operands {
                *remaining_uses.entry(v).or_insert(0) += 1;
            }
        }
    }

    let orig_pos: HashMap<OpId, usize> = non_term_ops
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut ready: Vec<OpId> = non_term_ops
        .iter()
        .copied()
        .filter(|id| in_degree.get(id).copied().unwrap_or(0) == 0)
        .collect();
    let mut scheduled: Vec<OpId> = Vec::with_capacity(orig_ops.len());

    while !ready.is_empty() {
        ready.sort_by_key(|&op_id| {
            let op = &module.ops[op_id];
            let is_const = op.is_op("stablehlo", "constant");
            let alloc_bytes: i64 = if is_const {
                0
            } else {
                op.results.iter().map(|&r| val_bytes(module, r)).sum()
            };
            let mut freed_bytes: i64 = 0;
            for &v in &op.operands {
                if remaining_uses.get(&v).copied().unwrap_or(0) == 1 {
                    freed_bytes += val_bytes(module, v);
                }
            }
            let net_delta = alloc_bytes - freed_bytes;
            let const_rank = i32::from(!is_const);
            (
                const_rank,
                net_delta,
                orig_pos.get(&op_id).copied().unwrap_or(0),
            )
        });

        let chosen = ready.remove(0);
        scheduled.push(chosen);

        if let Some(op) = module.ops.get(chosen) {
            for &v in &op.operands {
                if let Some(cnt) = remaining_uses.get_mut(&v) {
                    *cnt = cnt.saturating_sub(1);
                }
            }
        }
        if let Some(succs) = users.get(&chosen) {
            for &succ in succs {
                if let Some(deg) = in_degree.get_mut(&succ) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        ready.push(succ);
                    }
                }
            }
        }
    }

    for &op_id in &non_term_ops {
        if !scheduled.contains(&op_id) {
            scheduled.push(op_id);
        }
    }
    scheduled.extend(terminators);
    scheduled
}

fn val_bytes(module: &Module, val_id: ValueId) -> i64 {
    module.values[val_id]
        .ty
        .as_tensor()
        .and_then(|t| {
            let elems = t.num_elements()?;
            Some(((elems.max(1) * (t.dtype.bit_width() as i64)) + 7) / 8)
        })
        .unwrap_or(4)
}
