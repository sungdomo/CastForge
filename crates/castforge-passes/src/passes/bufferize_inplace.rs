//! In-place buffer donation and destination-passing style pass (`bufferize-inplace`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Module, OpId, ValueDef, ValueId};
use std::collections::HashMap;

/// AI Optimization Pass: In-Place Buffer Donation (`bufferize-inplace`).
///
/// Analyzes SSA tensor liveness in each basic block and annotates elementwise operations
/// whose intermediate input buffer dies at that exact instruction (`last_use == step`)
/// with `inplace_operand = <idx>`. Downstream memory planning and code generation reuse
/// the donor's arena slot in-place without allocating a second ping-pong buffer.
pub struct BufferizeInplacePass;

impl Pass for BufferizeInplacePass {
    fn name(&self) -> &'static str {
        "bufferize-inplace"
    }

    fn description(&self) -> &'static str {
        "Annotates elementwise ops whose input buffer dies at the op for in-place arena reuse"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            modified += annotate_block_inplace(module, block_id);
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

fn annotate_block_inplace(module: &mut Module, block_id: BlockId) -> usize {
    let op_ids = module.blocks[block_id].ops.clone();
    let num_ops = op_ids.len();
    let mut first_def: HashMap<ValueId, usize> = HashMap::new();
    let mut last_use: HashMap<ValueId, usize> = HashMap::new();
    let mut is_candidate_donor: HashMap<ValueId, bool> = HashMap::new();

    for (idx, &op_id) in op_ids.iter().enumerate() {
        let step = idx + 1;
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let is_const = op.is_op("stablehlo", "constant");
        let is_ret = op.is_op("func", "return") || op.is_op("stablehlo", "return");

        for &res_id in &op.results {
            if module.values[res_id].ty.as_tensor().is_some() {
                first_def.insert(res_id, step);
                last_use.insert(res_id, step);
                let is_arg = matches!(module.values[res_id].def, ValueDef::BlockArgument { .. });
                is_candidate_donor.insert(res_id, !is_const && !is_arg);
            }
        }

        for &operand_id in &op.operands {
            if let Some(lu) = last_use.get_mut(&operand_id) {
                *lu = (*lu).max(step);
                if is_ret {
                    *lu = num_ops + 1;
                }
            }
        }
    }

    let mut modified = 0usize;
    for (idx, &op_id) in op_ids.iter().enumerate() {
        let step = idx + 1;
        if !is_inplace_eligible_op(module, op_id) {
            continue;
        }
        let Some(&res_id) = module.ops[op_id].results.first() else {
            continue;
        };
        let Some(res_bytes) = tensor_byte_size(module, res_id) else {
            continue;
        };

        let operands = module.ops[op_id].operands.clone();
        let mut chosen_idx: Option<usize> = None;
        for (op_idx, &cand_id) in operands.iter().enumerate() {
            if !is_candidate_donor.get(&cand_id).copied().unwrap_or(false) {
                continue;
            }
            let def_step = first_def.get(&cand_id).copied().unwrap_or(step);
            let use_step = last_use.get(&cand_id).copied().unwrap_or(0);
            if def_step < step
                && use_step == step
                && tensor_byte_size(module, cand_id) == Some(res_bytes)
            {
                chosen_idx = Some(op_idx);
                break;
            }
        }

        if let Some(op_idx) = chosen_idx {
            let attr_val = Attribute::Int(op_idx as i64);
            let op_mut = &mut module.ops[op_id];
            if op_mut.attributes.get("inplace_operand") != Some(&attr_val) {
                op_mut
                    .attributes
                    .insert("inplace_operand".to_string(), attr_val);
                modified += 1;
            }
        }
    }

    modified
}

fn is_inplace_eligible_op(module: &Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    if op.name.dialect != "stablehlo" || op.results.len() != 1 {
        return false;
    }
    match op.name.name.as_str() {
        "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum" | "exponential"
        | "tanh" | "rsqrt" | "negate" | "abs" | "convert" | "reshape" => true,
        "composite" => matches!(
            op.attributes.get("name").and_then(|a| a.as_str()),
            Some("castforge.fused_add_max" | "castforge.fused_swiglu")
        ),
        _ => false,
    }
}

fn tensor_byte_size(module: &Module, val_id: ValueId) -> Option<usize> {
    let t = module.values[val_id].ty.as_tensor()?;
    let elems = t.num_elements()? as usize;
    Some((elems.max(1) * t.dtype.bit_width()).div_ceil(8))
}
