//! Unified Memory Optimization Pass (`optimize-memory`).
//!
//! Consolidates:
//! - `mode=inplace|all` (`BufferizeInplacePass` — in-place arena buffer donation)
//! - `mode=zero_copy|all` (`EliminateConcatCopiesPass` — zero-copy destination-slice concat)
//! - `mode=remat` (`RematerializePass` — activation rematerialization / checkpointing)
//! - `mode=offload` (`OffloadActivationsPass` — host-device DRAM activation offload)

pub mod remat;

pub use remat::{OffloadActivationsPass, RematerializePass};

use remat::run_selective_remat;

use crate::ir::{Attribute, BlockId, Module, OpId, ValueDef, ValueId};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;

/// Unified Memory Optimization Pass (`optimize-memory`).
pub struct OptimizeMemoryPass {
    pub mode: String,
    pub min_distance: usize,
    pub min_span: usize,
}

impl Default for OptimizeMemoryPass {
    fn default() -> Self {
        Self {
            mode: "all".to_string(),
            min_distance: 3,
            min_span: 2,
        }
    }
}

impl Pass for OptimizeMemoryPass {
    fn name(&self) -> &'static str {
        "optimize-memory"
    }

    fn description(&self) -> &'static str {
        "Unified memory optimization (in-place donation, zero-copy concat, remat, host offload)"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "bufferize-inplace",
            "eliminate-concat-copies",
            "rematerialize",
            "offload-activations",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        match m.as_str() {
            "inplace" => Ok(BufferizeInplacePass.run(module)?.with_name(self.name())),
            "zero_copy" | "concat" => Ok(EliminateConcatCopiesPass
                .run(module)?
                .with_name(self.name())),
            "remat" | "rematerialize" => Ok(RematerializePass {
                min_distance: self.min_distance,
            }
            .run(module)?
            .with_name(self.name())),
            "selective_remat" | "selective" => {
                Ok(run_selective_remat(module, self.min_distance)?.with_name(self.name()))
            }
            "offload" => Ok(OffloadActivationsPass {
                min_span: self.min_span,
            }
            .run(module)?
            .with_name(self.name())),
            _ => PassResult::run_sequence(
                self.name(),
                module,
                &mut [&mut BufferizeInplacePass, &mut EliminateConcatCopiesPass],
            ),
        }
    }
}

fn optimize_memory_inplace_and_zero_copy(
    module: &mut Module,
    do_inplace: bool,
    do_zero_copy: bool,
) -> usize {
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();
    for block_id in blocks {
        if do_inplace {
            modified += annotate_block_inplace_and_concat(module, block_id, do_zero_copy);
        } else if do_zero_copy {
            for op_id in module.block_ops(block_id) {
                modified += try_eliminate_concat_copy_at_op(module, op_id);
            }
        }
    }
    modified
}

/// In-Place Buffer Donation Pass (`bufferize-inplace`).
pub struct BufferizeInplacePass;

impl Pass for BufferizeInplacePass {
    fn name(&self) -> &'static str {
        "bufferize-inplace"
    }

    fn description(&self) -> &'static str {
        "Annotates elementwise ops whose input buffer dies at the op for in-place arena reuse"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let modified = optimize_memory_inplace_and_zero_copy(module, true, false);
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn annotate_block_inplace_and_concat(
    module: &mut Module,
    block_id: BlockId,
    do_zero_copy: bool,
) -> usize {
    let op_ids = module.blocks[block_id].ops.clone();
    let mut seen_later_use: std::collections::HashSet<ValueId> =
        std::collections::HashSet::with_capacity(op_ids.len());
    let mut modified = 0usize;

    for op_id in op_ids.into_iter().rev() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if op.is_op("func", "return") || op.is_op("stablehlo", "return") {
            for &operand_id in &op.operands {
                seen_later_use.insert(operand_id);
            }
            continue;
        }

        if do_zero_copy {
            modified += try_eliminate_concat_copy_at_op(module, op_id);
        }

        if is_inplace_eligible_op(module, op_id)
            && let Some(&res_id) = module.ops[op_id].results.first()
            && let Some(res_bytes) = tensor_byte_size(module, res_id)
        {
            let operands = module.ops[op_id].operands.clone();
            let mut chosen_idx: Option<usize> = None;
            for (op_idx, &cand_id) in operands.iter().enumerate() {
                if seen_later_use.contains(&cand_id) {
                    continue;
                }
                let ValueDef::OpResult { op: def_op_id, .. } = module.values[cand_id].def else {
                    continue;
                };
                if def_op_id == op_id
                    || module
                        .ops
                        .get(def_op_id)
                        .is_none_or(|d| d.is_op("stablehlo", "constant"))
                {
                    continue;
                }
                if tensor_byte_size(module, cand_id) == Some(res_bytes) {
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

        for &operand_id in &module.ops[op_id].operands {
            seen_later_use.insert(operand_id);
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

/// Zero-Copy Concatenation Elimination Pass (`eliminate-concat-copies`).
#[derive(Default)]
pub struct EliminateConcatCopiesPass;

impl Pass for EliminateConcatCopiesPass {
    fn name(&self) -> &'static str {
        "eliminate-concat-copies"
    }

    fn description(&self) -> &'static str {
        "Eliminates intermediate concatenate copies by routing producer outputs into dest slices"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let modified = optimize_memory_inplace_and_zero_copy(module, false, true);
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn try_eliminate_concat_copy_at_op(module: &mut Module, op_id: OpId) -> usize {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return 0;
    };
    let Some(StableHloOp::Concatenate {
        inputs,
        dimension,
        result,
    }) = StableHloOp::from_op(&op)
    else {
        return 0;
    };
    if inputs.is_empty() {
        return 0;
    }

    let dest_name = module.values[result]
        .ssa_name
        .clone()
        .unwrap_or_else(|| format!("{result:?}"));
    let mut elem_offset = 0i64;
    let mut routed_any = false;
    let mut modified = 0usize;
    for &inp in &inputs {
        let inp_elems = module.values[inp]
            .ty
            .as_tensor()
            .and_then(|t| t.num_elements())
            .unwrap_or(0);
        if let ValueDef::OpResult { op: prod_id, .. } = module.values[inp].def
            && let Some(prod_op) = module.ops.get_mut(prod_id)
            && prod_op.is_pure()
            && !prod_op.is_op("stablehlo", "constant")
        {
            prod_op.attributes.insert(
                "castforge.concat_dest".to_string(),
                Attribute::String(dest_name.clone()),
            );
            prod_op.attributes.insert(
                "castforge.concat_axis".to_string(),
                Attribute::Int(dimension),
            );
            prod_op.attributes.insert(
                "castforge.concat_elem_offset".to_string(),
                Attribute::Int(elem_offset),
            );
            routed_any = true;
            modified += 1;
        }
        elem_offset += inp_elems;
    }

    if routed_any && let Some(cat_op) = module.ops.get_mut(op_id) {
        cat_op.attributes.insert(
            "castforge.zero_copy_concat".to_string(),
            Attribute::Bool(true),
        );
    }
    modified
}
