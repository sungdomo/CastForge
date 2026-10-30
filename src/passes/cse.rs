use crate::ir::{BlockId, Module, OpId, Operation, RegionId, Type, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Unified Common Subexpression Elimination, Dominator GVN, LICM & Constant Hoisting (`cse`),
/// executed in a single top-down region/block traversal.
pub struct CsePass {
    pub gvn: bool,
    pub hoist_constants: bool,
    pub licm: bool,
}

impl Default for CsePass {
    fn default() -> Self {
        Self {
            gvn: true,
            hoist_constants: false,
            licm: true,
        }
    }
}

impl Pass for CsePass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Unified CSE, cross-region Dominator GVN, LICM, and constant hoisting/deduplication"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut modified = 0usize;
        let mut value_leader: HashMap<ValueId, ValueId> = HashMap::new();

        for fn_id in module.functions() {
            let regions: Vec<RegionId> = module.ops[fn_id].regions.clone();
            let mut env: HashMap<String, ValueId> = HashMap::new();
            for reg_id in regions {
                let (rem, modif) = self.process_region(module, reg_id, &mut env, &mut value_leader);
                removed += rem;
                modified += modif;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

impl CsePass {
    fn process_region(
        &self,
        module: &mut Module,
        region_id: RegionId,
        outer_env: &mut HashMap<String, ValueId>,
        value_leader: &mut HashMap<ValueId, ValueId>,
    ) -> (usize, usize) {
        let mut removed = 0usize;
        let mut modified = 0usize;
        let Some(reg) = module.regions.get(region_id) else {
            return (0, 0);
        };
        let blocks: Vec<BlockId> = reg.blocks.clone();

        for block_id in blocks {
            let mut block_env = if self.gvn {
                outer_env.clone()
            } else {
                HashMap::new()
            };
            let ops = module.block_ops(block_id);
            let mut const_ops: Vec<OpId> = Vec::new();
            let mut non_const_ops: Vec<OpId> = Vec::with_capacity(ops.len());
            let mut dead_ops: Vec<OpId> = Vec::new();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };

                if self.licm
                    && !op.regions.is_empty()
                    && !op.is_op("builtin", "module")
                    && !op.is_op("func", "func")
                {
                    let hoisted = extract_region_invariant_ops(module, &op.regions);
                    for hoisted_op_id in hoisted {
                        modified += 1;
                        let Some(h_op) = module.ops.get(hoisted_op_id).cloned() else {
                            continue;
                        };
                        if self.try_gvn_dedup(
                            module,
                            hoisted_op_id,
                            &h_op,
                            &mut block_env,
                            value_leader,
                            &mut dead_ops,
                        ) {
                            removed += 1;
                            continue;
                        }
                        if self.hoist_constants && h_op.is_op("stablehlo", "constant") {
                            const_ops.push(hoisted_op_id);
                        } else {
                            non_const_ops.push(hoisted_op_id);
                        }
                    }
                }

                for &nested_reg in &op.regions {
                    let (rem, modif) =
                        self.process_region(module, nested_reg, &mut block_env, value_leader);
                    removed += rem;
                    modified += modif;
                }

                if self.try_gvn_dedup(
                    module,
                    op_id,
                    &op,
                    &mut block_env,
                    value_leader,
                    &mut dead_ops,
                ) {
                    removed += 1;
                    continue;
                }

                if self.hoist_constants && op.is_op("stablehlo", "constant") {
                    const_ops.push(op_id);
                } else {
                    non_const_ops.push(op_id);
                }
            }

            let mut new_order = Vec::with_capacity(const_ops.len() + non_const_ops.len());
            new_order.extend(const_ops);
            new_order.extend(non_const_ops);

            if !dead_ops.is_empty() || module.blocks[block_id].ops != new_order {
                if dead_ops.is_empty() && module.blocks[block_id].ops != new_order {
                    modified += 1;
                }
                module.blocks[block_id].ops = new_order;
                for dead_id in dead_ops {
                    module.ops.remove(dead_id);
                }
            }

            if self.gvn {
                *outer_env = block_env;
            }
        }

        (removed, modified)
    }

    fn try_gvn_dedup(
        &self,
        module: &mut Module,
        op_id: OpId,
        op: &Operation,
        block_env: &mut HashMap<String, ValueId>,
        value_leader: &mut HashMap<ValueId, ValueId>,
        dead_ops: &mut Vec<OpId>,
    ) -> bool {
        if op.is_pure() && op.results.len() == 1 && op.regions.is_empty() {
            let res_val = op.results[0];
            let res_ty = module.values[res_val].ty.clone();
            let sig = gvn_signature(op, &res_ty, value_leader);
            if let Some(&leader_val) = block_env.get(&sig)
                && leader_val != res_val
            {
                value_leader.insert(res_val, leader_val);
                module.replace_all_uses(res_val, leader_val);
                dead_ops.push(op_id);
                return true;
            }
            value_leader.insert(res_val, res_val);
            block_env.insert(sig, res_val);
        }
        false
    }
}

fn extract_region_invariant_ops(module: &mut Module, regions: &[RegionId]) -> Vec<OpId> {
    let mut inner_blocks = Vec::new();
    for &reg_id in regions {
        if let Some(reg) = module.regions.get(reg_id) {
            inner_blocks.extend(reg.blocks.iter().copied());
        }
    }

    let mut region_defined_values: HashSet<ValueId> = HashSet::new();
    for &blk_id in &inner_blocks {
        for &arg_val in &module.blocks[blk_id].arguments {
            region_defined_values.insert(arg_val);
        }
        for &inner_op_id in &module.blocks[blk_id].ops {
            if let Some(inner_op) = module.ops.get(inner_op_id) {
                for &res_val in &inner_op.results {
                    region_defined_values.insert(res_val);
                }
            }
        }
    }

    let mut hoisted_ops: Vec<(BlockId, OpId)> = Vec::new();
    for &blk_id in &inner_blocks {
        let inner_ops = module.block_ops(blk_id);
        for inner_op_id in inner_ops {
            let Some(inner_op) = module.ops.get(inner_op_id) else {
                continue;
            };
            if !inner_op.is_pure() || !inner_op.regions.is_empty() {
                continue;
            };
            let is_invariant = inner_op
                .operands
                .iter()
                .all(|v| !region_defined_values.contains(v));
            if is_invariant {
                for &res_val in &inner_op.results {
                    region_defined_values.remove(&res_val);
                }
                hoisted_ops.push((blk_id, inner_op_id));
            }
        }
    }

    let mut out = Vec::with_capacity(hoisted_ops.len());
    for (from_blk, hoisted_op_id) in hoisted_ops {
        module.blocks[from_blk]
            .ops
            .retain(|&id| id != hoisted_op_id);
        out.push(hoisted_op_id);
    }
    out
}

fn is_commutative_binary_op(op: &Operation) -> bool {
    op.name.dialect == "stablehlo"
        && op.operands.len() == 2
        && matches!(
            op.name.name.as_str(),
            "add" | "multiply" | "maximum" | "minimum" | "and" | "or" | "xor"
        )
}

fn gvn_signature(
    op: &Operation,
    res_ty: &Type,
    value_leader: &HashMap<ValueId, ValueId>,
) -> String {
    let mut ops_vec: Vec<String> = op
        .operands
        .iter()
        .map(|v| {
            let canon = value_leader.get(v).copied().unwrap_or(*v);
            format!("{canon:?}")
        })
        .collect();
    if is_commutative_binary_op(op) {
        ops_vec.sort();
    }
    let mut attrs: Vec<(&String, String)> = op
        .attributes
        .iter()
        .filter(|(k, _)| !k.starts_with("castforge.") && *k != "inplace_operand")
        .map(|(k, v)| (k, format!("{v:?}")))
        .collect();
    attrs.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{}|operands:{:?}|attrs:{:?}|ty:{}",
        op.name, ops_vec, attrs, res_ty
    )
}
