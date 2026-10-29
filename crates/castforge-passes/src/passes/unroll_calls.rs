//! Bounded Call Unroller (`unroll-calls`) and Collective Communication Optimizer
//! (`hoist-all-reduce`) Passes (`Ir -> Ir`).
//!
//! - [`UnrollCallsPass`] (`unroll-calls`) chains repeated `func.call` invocations (`repeat=N`) by
//!   feeding the output of step `i` into step `i + 1` when input/output types match, composing
//!   directly with `inliner` to unroll stacked Transformer blocks.
//! - [`HoistAllReducePass`] (`hoist-all-reduce`) merges linear collective sums
//!   `add(all_reduce(a), all_reduce(b)) -> all_reduce(add(a, b))`, eliminates redundant chained
//!   `all_reduce(all_reduce(x))`, and marks collective ops with `async_overlap = true`.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpId, OpName, ValueDef};
use smallvec::SmallVec;

/// Bounded Call-Chain Unroller Pass (`unroll-calls`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct UnrollCallsPass {
    /// Total number of chained call iterations for each self-compatible `func.call` (default `2`).
    pub repeat: usize,
}

impl Default for UnrollCallsPass {
    fn default() -> Self {
        Self { repeat: 2 }
    }
}

impl Pass for UnrollCallsPass {
    fn name(&self) -> &'static str {
        "unroll-calls"
    }

    fn description(&self) -> &'static str {
        "Unrolls repeated func.call invocations (repeat=N) for stacked layer blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let repeat = self.repeat.max(1);
        if repeat <= 1 {
            return Ok(PassResult::unchanged(self.name()));
        }
        let mut created = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("func", "call") || op.operands.is_empty() || op.results.len() != 1 {
                    continue;
                }
                let orig_res = op.results[0];
                let res_ty = module.values[orig_res].ty.clone();
                if module.values[op.operands[0]].ty != res_ty {
                    continue;
                }

                let mut cur_val = orig_res;
                let mut new_call_ids: Vec<OpId> = Vec::with_capacity(repeat - 1);
                for step in 1..repeat {
                    let mut next_operands: SmallVec<[_; 4]> = op.operands.clone();
                    next_operands[0] = cur_val;
                    let mut attrs = op.attributes.clone();
                    attrs.insert("unroll_step".to_string(), Attribute::Int(step as i64));

                    let call_id = module.add_operation(
                        block_id,
                        OpName::new("func", "call"),
                        next_operands,
                        vec![(Some(format!("call_u{step}")), res_ty.clone())],
                        attrs,
                        Vec::new(),
                        true,
                        op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    cur_val = module.ops[call_id].results[0];
                    new_call_ids.push(call_id);
                    created += 1;
                }

                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == op_id || new_call_ids.contains(&other_id) {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = cur_val;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id]
                        .ops
                        .splice((pos + 1)..(pos + 1), new_call_ids);
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: 0,
            ops_modified: created,
            ops_created: created,
        })
    }
}

/// Collective All-Reduce Coalescer & Async Overlap Pass (`hoist-all-reduce`: `Ir -> Ir`).
#[derive(Debug, Clone, Default)]
pub struct HoistAllReducePass;

impl Pass for HoistAllReducePass {
    fn name(&self) -> &'static str {
        "hoist-all-reduce"
    }

    fn description(&self) -> &'static str {
        "Coalesces linear all_reduce sums, eliminates redundant collectives, and enables overlap"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };

                // 1. Annotate all_reduce ops with async_overlap = true and collapse chained ones
                if is_all_reduce(&op) {
                    if op.attributes.get("async_overlap") != Some(&Attribute::Bool(true)) {
                        module.ops[op_id]
                            .attributes
                            .insert("async_overlap".to_string(), Attribute::Bool(true));
                        modified += 1;
                    }
                    if let Some(&src) = op.operands.first()
                        && let ValueDef::OpResult { op: prod_id, .. } = module.values[src].def
                        && module.ops.get(prod_id).is_some_and(is_all_reduce)
                    {
                        let res = op.results[0];
                        module.replace_all_uses(res, src);
                        module.remove_op(block_id, op_id);
                        removed += 1;
                        continue;
                    }
                }

                // 2. Coalesce `add(all_reduce(a), all_reduce(b))` -> `all_reduce(add(a, b))`
                if op.is_op("stablehlo", "add") && op.operands.len() == 2 && op.results.len() == 1 {
                    let (lhs, rhs) = (op.operands[0], op.operands[1]);
                    let (
                        ValueDef::OpResult { op: l_op_id, .. },
                        ValueDef::OpResult { op: r_op_id, .. },
                    ) = (module.values[lhs].def, module.values[rhs].def)
                    else {
                        continue;
                    };
                    if l_op_id == r_op_id {
                        continue;
                    }
                    let (Some(l_op), Some(r_op)) = (
                        module.ops.get(l_op_id).cloned(),
                        module.ops.get(r_op_id).cloned(),
                    ) else {
                        continue;
                    };
                    if !is_all_reduce(&l_op)
                        || !is_all_reduce(&r_op)
                        || l_op.operands.is_empty()
                        || r_op.operands.is_empty()
                    {
                        continue;
                    }
                    let raw_a = l_op.operands[0];
                    let raw_b = r_op.operands[0];
                    let add_res = op.results[0];

                    // Rewrite `add` to operate on raw_a, raw_b, then wrap with 1 all_reduce
                    module.ops[op_id].operands[0] = raw_a;
                    module.ops[op_id].operands[1] = raw_b;

                    let mut ar_attrs = l_op.attributes.clone();
                    ar_attrs.insert("async_overlap".to_string(), Attribute::Bool(true));
                    ar_attrs.insert("coalesced_collectives".to_string(), Attribute::Int(2));
                    let out_ty = module.values[add_res].ty.clone();

                    let ar_id = module.add_operation(
                        block_id,
                        l_op.name.clone(),
                        SmallVec::from_slice(&[add_res]),
                        vec![(Some("ar_coalesced".to_string()), out_ty)],
                        ar_attrs,
                        Vec::new(),
                        l_op.custom_syntax,
                        op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    let ar_res = module.ops[ar_id].results[0];

                    for (other_id, other_op) in module.ops.iter_mut() {
                        if other_id == op_id || other_id == ar_id {
                            continue;
                        }
                        for opnd in &mut other_op.operands {
                            if *opnd == add_res {
                                *opnd = ar_res;
                            }
                        }
                    }
                    if let Some(pos) = module.blocks[block_id]
                        .ops
                        .iter()
                        .position(|&id| id == op_id)
                    {
                        module.blocks[block_id].ops.insert(pos + 1, ar_id);
                    }
                    modified += 1;
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn is_all_reduce(op: &castforge_ir::Operation) -> bool {
    op.is_op("stablehlo", "all_reduce")
        || (op.is_op("stablehlo", "composite")
            && op.attributes.get("name").and_then(|a| a.as_str()) == Some("castforge.all_reduce"))
}
