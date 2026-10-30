//! Unified Memory Optimization Pass (`optimize-memory`).
//!
//! Consolidates:
//! - `mode=inplace|all` (`BufferizeInplacePass` — in-place arena buffer donation)
//! - `mode=zero_copy|all` (`EliminateConcatCopiesPass` — zero-copy destination-slice concat)
//! - `mode=remat` (`RematerializePass` — activation rematerialization / checkpointing)
//! - `mode=offload` (`OffloadActivationsPass` — host-device DRAM activation offload)

use crate::ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;

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

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let do_inplace = m == "all" || m == "inplace";
        let do_zero_copy = m == "all" || m == "zero_copy" || m == "concat";
        if do_inplace || do_zero_copy {
            let modified = optimize_memory_inplace_and_zero_copy(module, do_inplace, do_zero_copy);
            return Ok(PassResult {
                pass_name: self.name().to_string(),
                changed: modified > 0,
                ops_removed: 0,
                ops_modified: modified,
                ops_created: 0,
            });
        }
        let mut steps = Vec::new();
        if m == "remat" || m == "rematerialize" {
            steps.push(
                RematerializePass {
                    min_distance: self.min_distance,
                }
                .run(module)?,
            );
        }
        if m == "selective_remat" || m == "selective" {
            steps.push(run_selective_remat(module, self.min_distance)?);
        }
        if m == "offload" {
            steps.push(
                OffloadActivationsPass {
                    min_span: self.min_span,
                }
                .run(module)?,
            );
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
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
        if do_zero_copy {
            modified += try_eliminate_concat_copy_at_op(module, op_id);
        }
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

/// Activation Rematerialization / Gradient Checkpointing Pass (`rematerialize`).
pub struct RematerializePass {
    pub min_distance: usize,
}

impl Default for RematerializePass {
    fn default() -> Self {
        Self { min_distance: 3 }
    }
}

impl Pass for RematerializePass {
    fn name(&self) -> &'static str {
        "rematerialize"
    }

    fn description(&self) -> &'static str {
        "Recomputes long-lived cheap elementwise activations before late consumers to save memory"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut remat_plans = Vec::new();

            for (def_pos, &def_op_id) in ops.iter().enumerate() {
                let Some(def_op) = module.ops.get(def_op_id) else {
                    continue;
                };
                if !is_cheap_remat_candidate(def_op) || def_op.results.len() != 1 {
                    continue;
                }
                let val_id = def_op.results[0];

                let mut use_positions = Vec::new();
                for (u_pos, &u_op_id) in ops.iter().enumerate().skip(def_pos + 1) {
                    if let Some(u_op) = module.ops.get(u_op_id)
                        && !u_op.is_op("func", "return")
                        && u_op.operands.contains(&val_id)
                    {
                        use_positions.push((u_pos, u_op_id));
                    }
                }
                if use_positions.is_empty() {
                    continue;
                }

                let &(last_pos, last_user_op) = use_positions.last().unwrap();
                let dist = last_pos.saturating_sub(def_pos);
                let crosses_heavy = ops[def_pos + 1..last_pos]
                    .iter()
                    .any(|&mid_id| module.ops.get(mid_id).is_some_and(is_heavy_barrier));

                if dist >= self.min_distance.max(2) && (crosses_heavy || use_positions.len() >= 2) {
                    remat_plans.push((def_op_id, val_id, last_user_op));
                }
            }

            for (def_op_id, old_val, user_op_id) in remat_plans {
                let Some(orig_op) = module.ops.get(def_op_id).cloned() else {
                    continue;
                };
                let out_ty = module.values[old_val].ty.clone();
                let mut new_attrs = orig_op.attributes.clone();
                new_attrs.insert("rematerialized".to_string(), Attribute::Bool(true));

                let new_op_id = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: orig_op.name.clone(),
                    operands: orig_op.operands.clone(),
                    results: SmallVec::new(),
                    attributes: new_attrs,
                    regions: Vec::new(),
                    custom_syntax: orig_op.custom_syntax,
                    location: Location::Unknown,
                });
                let new_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: out_ty,
                    def: ValueDef::OpResult {
                        op: new_op_id,
                        index: 0,
                    },
                });
                module.ops[new_op_id].results.push(new_val);

                let block_ops = &mut module.blocks[block_id].ops;
                if let Some(pos) = block_ops.iter().position(|&id| id == user_op_id) {
                    block_ops.insert(pos, new_op_id);
                } else {
                    block_ops.push(new_op_id);
                }

                for opnd in &mut module.ops[user_op_id].operands {
                    if *opnd == old_val {
                        *opnd = new_val;
                    }
                }
                created += 1;
                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn is_cheap_remat_candidate(op: &Operation) -> bool {
    if op.name.dialect != "stablehlo"
        || op.attributes.contains_key("rematerialized")
        || op.operands.is_empty()
    {
        return false;
    }
    matches!(
        op.name.name.as_str(),
        "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "maximum"
            | "minimum"
            | "tanh"
            | "exponential"
            | "logistic"
            | "rsqrt"
            | "sqrt"
            | "negate"
            | "broadcast_in_dim"
            | "reshape"
    )
}

fn is_heavy_barrier(op: &Operation) -> bool {
    op.name.dialect == "stablehlo"
        && matches!(
            op.name.name.as_str(),
            "dot_general" | "convolution" | "composite" | "reduce"
        )
}

/// Host-Device Activation Offloader Pass (`offload-activations`).
#[derive(Debug, Clone)]
pub struct OffloadActivationsPass {
    pub min_span: usize,
}

impl Default for OffloadActivationsPass {
    fn default() -> Self {
        Self { min_span: 2 }
    }
}

impl Pass for OffloadActivationsPass {
    fn name(&self) -> &'static str {
        "offload-activations"
    }

    fn description(&self) -> &'static str {
        "Inserts castforge.host_offload composite markers on long-lived intermediate activations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let min_span = self.min_span.max(1);
        let mut created = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            let mut def_idx: HashMap<ValueId, (usize, OpId)> = HashMap::new();
            let mut last_use_idx: HashMap<ValueId, usize> = HashMap::new();

            for (idx, &op_id) in ops.iter().enumerate() {
                let op = &module.ops[op_id];
                for &opnd in &op.operands {
                    last_use_idx.insert(opnd, idx);
                }
                if !op.is_op("stablehlo", "constant")
                    && !(op.is_op("stablehlo", "composite")
                        && op.attributes.get("name").and_then(|a| a.as_str())
                            == Some("castforge.host_offload"))
                {
                    for &res in &op.results {
                        def_idx.insert(res, (idx, op_id));
                    }
                }
            }

            let mut candidates: Vec<(ValueId, OpId, usize)> = Vec::new();
            for (&vid, &(d_i, prod_op)) in &def_idx {
                if let Some(&u_i) = last_use_idx.get(&vid)
                    && u_i.saturating_sub(d_i) >= min_span
                {
                    candidates.push((vid, prod_op, u_i - d_i));
                }
            }

            for (orig_res, prod_op, span) in candidates {
                let res_ty = module.values[orig_res].ty.clone();
                let base_ssa = module.values[orig_res]
                    .ssa_name
                    .as_deref()
                    .unwrap_or("act")
                    .to_string();
                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.host_offload".to_string()),
                );
                attrs.insert("live_span".to_string(), Attribute::Int(span as i64));

                let loc = module.ops[prod_op].location.clone();
                let offload_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "composite"),
                    smallvec![orig_res],
                    vec![(Some(format!("{base_ssa}_offload")), res_ty)],
                    attrs,
                    Vec::new(),
                    false,
                    loc,
                );
                module.blocks[block_id].ops.pop();
                let offload_res = module.ops[offload_op_id].results[0];

                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == prod_op || other_id == offload_op_id {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = offload_res;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == prod_op)
                {
                    module.blocks[block_id].ops.insert(pos + 1, offload_op_id);
                }
                created += 1;
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

fn run_selective_remat(module: &mut Module, min_distance: usize) -> Result<PassResult> {
    let remat_res = RematerializePass { min_distance }.run(module)?;
    let mut modified = remat_res.ops_modified;

    for (_, op) in module.ops.iter_mut() {
        if op.is_op("stablehlo", "constant")
            || op.is_op("func", "func")
            || op.is_op("func", "return")
        {
            continue;
        }
        let is_compute_heavy = op.is_op("stablehlo", "dot_general")
            || op.is_op("stablehlo", "convolution")
            || (op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n.contains("matmul") || n.contains("qmatmul")));
        let policy = if is_compute_heavy {
            "checkpoint"
        } else {
            "recompute"
        };
        op.attributes.insert(
            "castforge.selective_remat".to_string(),
            Attribute::String(policy.to_string()),
        );
        modified += 1;
    }

    Ok(PassResult {
        pass_name: "optimize-memory".to_string(),
        changed: modified > 0 || remat_res.changed,
        ops_removed: remat_res.ops_removed,
        ops_modified: modified,
        ops_created: remat_res.ops_created,
    })
}
