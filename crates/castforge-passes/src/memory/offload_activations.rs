//! Host-Device Activation Offloader Pass (`offload-activations`: `Ir -> Ir`).
//!
//! Inspired by DeepSpeed ZeRO-Offload and XLA Host Memory Offload: identifies long-lived
//! intermediate activation tensors (whose liveness span across operations is `>= min_span`) and
//! wraps them in `stablehlo.composite` (`"castforge.host_offload"`) markers so codegen and memory
//! planners can offload them to host DRAM during intermediate computation.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpId, OpName, ValueId};
use indexmap::IndexMap;
use smallvec::smallvec;
use std::collections::HashMap;

/// Host-Device Activation Offloader Pass (`offload-activations`).
#[derive(Debug, Clone)]
pub struct OffloadActivationsPass {
    /// Minimum operation span (`last_use_idx - def_idx`) to trigger offloading (default `2`).
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
