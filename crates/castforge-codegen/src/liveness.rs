use castforge_ir::{BlockId, Module, ValueDef, ValueId};
use serde::Serialize;
use std::collections::HashMap;

/// Live range `[first_def, last_use]` (in block instruction indices) for an SSA tensor value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LiveInterval {
    pub value: ValueId,
    pub first_def: usize,
    pub last_use: usize,
    pub is_constant: bool,
    pub is_block_arg: bool,
    pub is_returned: bool,
}

/// Computes live intervals `[first_def, last_use]` for all tensor values inside `block_id`.
pub fn compute_block_liveness(module: &Module, block_id: BlockId) -> Vec<LiveInterval> {
    let block = &module.blocks[block_id];
    let num_ops = block.ops.len();
    let mut intervals: HashMap<ValueId, LiveInterval> = HashMap::new();
    let mut order: Vec<ValueId> = Vec::new();

    // 1. Block arguments are live starting at step 0
    for &arg_id in &block.arguments {
        if module.values[arg_id].ty.as_tensor().is_some() {
            order.push(arg_id);
            intervals.insert(
                arg_id,
                LiveInterval {
                    value: arg_id,
                    first_def: 0,
                    last_use: 0,
                    is_constant: false,
                    is_block_arg: true,
                    is_returned: false,
                },
            );
        }
    }

    // 2. Scan operations in topological execution order (step = idx + 1)
    for (idx, &op_id) in block.ops.iter().enumerate() {
        let step = idx + 1;
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let is_const = op.is_op("stablehlo", "constant");
        let is_ret = op.is_op("func", "return") || op.is_op("stablehlo", "return");

        for &res_id in &op.results {
            if module.values[res_id].ty.as_tensor().is_some() {
                order.push(res_id);
                intervals.insert(
                    res_id,
                    LiveInterval {
                        value: res_id,
                        first_def: step,
                        last_use: step,
                        is_constant: is_const,
                        is_block_arg: false,
                        is_returned: false,
                    },
                );
            }
        }

        for &operand_id in &op.operands {
            if let Some(iv) = intervals.get_mut(&operand_id) {
                iv.last_use = iv.last_use.max(step);
                if is_ret {
                    iv.is_returned = true;
                    iv.last_use = num_ops + 1;
                }
            }
        }
    }

    order
        .into_iter()
        .filter_map(|v| intervals.remove(&v))
        .collect()
}

/// Computes live intervals across all function entry blocks in `module`.
pub fn compute_module_liveness(module: &Module) -> HashMap<ValueId, LiveInterval> {
    let mut all = HashMap::new();
    for fn_op_id in module.functions() {
        let fn_op = &module.ops[fn_op_id];
        if let Some(&reg_id) = fn_op.regions.first()
            && let Some(&blk_id) = module.regions[reg_id].blocks.first()
        {
            for iv in compute_block_liveness(module, blk_id) {
                all.insert(iv.value, iv);
            }
        }
    }
    // Fallback for any standalone values not in a function block
    for (val_id, val) in &module.values {
        if val.ty.as_tensor().is_none() || all.contains_key(&val_id) {
            continue;
        }
        let is_const = match val.def {
            ValueDef::OpResult { op, .. } => module
                .ops
                .get(op)
                .is_some_and(|o| o.is_op("stablehlo", "constant")),
            _ => false,
        };
        all.insert(
            val_id,
            LiveInterval {
                value: val_id,
                first_def: 0,
                last_use: 0,
                is_constant: is_const,
                is_block_arg: matches!(val.def, ValueDef::BlockArgument { .. }),
                is_returned: false,
            },
        );
    }
    all
}
