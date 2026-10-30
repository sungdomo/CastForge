//! Activation Rematerialization (`rematerialize`), Selective Rematerialization,
//! and Host-Device DRAM Offloading (`offload-activations`).

use crate::ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;

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

pub(super) fn run_selective_remat(module: &mut Module, min_distance: usize) -> Result<PassResult> {
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
