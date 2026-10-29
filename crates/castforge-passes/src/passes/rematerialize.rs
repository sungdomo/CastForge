use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{BlockId, Location, Module, Operation, Value, ValueDef};
use smallvec::SmallVec;

/// Activation Rematerialization / Gradient Checkpointing Pass (`rematerialize`).
///
/// Scans each block for cheap pure elementwise/unary operations whose result value has a long
/// live range spanning across heavy compute barriers (`dot_general`, `convolution`, or fused
/// composites) or spanning at least `min_distance` operations. Clones/recomputes the cheap
/// activation immediately before its late consumer (`rematerialized = true`), shortening its
/// live interval so the arena memory planner can reuse its buffer slot earlier.
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

                // Find all use positions in the same block
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
                new_attrs.insert(
                    "rematerialized".to_string(),
                    castforge_ir::Attribute::Bool(true),
                );

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
