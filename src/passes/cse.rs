use crate::ir::{BlockId, Module, OpId, Operation, RegionId, Type, ValueDef, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Unified Common Subexpression Elimination, Dominator GVN, LICM & Constant Hoisting (`cse`).
pub struct UnifiedCsePass {
    pub gvn: bool,
    pub hoist_constants: bool,
    pub licm: bool,
}

impl Default for UnifiedCsePass {
    fn default() -> Self {
        Self {
            gvn: true,
            hoist_constants: false,
            licm: true,
        }
    }
}

impl Pass for UnifiedCsePass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Unified CSE, cross-region Dominator GVN, LICM, and constant hoisting/deduplication"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.hoist_constants {
            steps.push(HoistConstantsPass.run(module)?);
        }
        if self.licm {
            steps.push(crate::passes::LicmPass.run(module)?);
        }
        if self.gvn {
            steps.push(GlobalValueNumberingPass.run(module)?);
        } else {
            steps.push(CommonSubexpressionEliminationPass.run(module)?);
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

/// Common Subexpression Elimination Pass (`cse`).
pub struct CommonSubexpressionEliminationPass;

impl Pass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Common Subexpression Elimination (deduplicates identical pure operations)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let mut seen: HashMap<String, OpId> = HashMap::new();
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                    continue;
                }
                let res_val = op.results[0];
                let res_ty = &module.values[res_val].ty;
                let sig = canonical_cse_signature(op, res_ty);

                if let Some(&existing_op_id) = seen.get(&sig) {
                    let existing_res = module.ops[existing_op_id].results[0];
                    module.replace_all_uses(res_val, existing_res);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                } else {
                    seen.insert(sig, op_id);
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// Hoist & Deduplicate Constants Pass (`hoist-constants`).
pub struct HoistConstantsPass;

impl Pass for HoistConstantsPass {
    fn name(&self) -> &'static str {
        "hoist-constants"
    }

    fn description(&self) -> &'static str {
        "Hoists stablehlo.constant ops to block entry and deduplicates identical constants"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut seen_consts: HashMap<(String, String), ValueId> = HashMap::new();
            let mut const_ops: Vec<OpId> = Vec::new();
            let mut non_const_ops: Vec<OpId> = Vec::new();

            for op_id in ops.iter().copied() {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if op.is_op("stablehlo", "constant")
                    && let Some(&res_val) = op.results.first()
                    && let Some(val_attr) = op.attributes.get("value")
                {
                    let key = (
                        module.values[res_val].ty.to_string(),
                        format!("{val_attr:?}"),
                    );
                    if let Some(&canon_val) = seen_consts.get(&key) {
                        module.replace_all_uses(res_val, canon_val);
                        module.remove_op(block_id, op_id);
                        removed += 1;
                    } else {
                        seen_consts.insert(key, res_val);
                        const_ops.push(op_id);
                    }
                } else {
                    non_const_ops.push(op_id);
                }
            }

            let mut new_order = Vec::with_capacity(const_ops.len() + non_const_ops.len());
            new_order.extend(const_ops);
            new_order.extend(non_const_ops);

            if module.blocks[block_id].ops != new_order {
                module.blocks[block_id].ops = new_order;
                modified += 1;
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

/// Global Value Numbering Pass (`gvn`).
#[derive(Default)]
pub struct GlobalValueNumberingPass;

impl Pass for GlobalValueNumberingPass {
    fn name(&self) -> &'static str {
        "gvn"
    }

    fn description(&self) -> &'static str {
        "Dominator-aware Global Value Numbering across nested regions and basic blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut value_leader: HashMap<ValueId, ValueId> = HashMap::new();
        for fn_id in module.functions() {
            let regions: Vec<RegionId> = module.ops[fn_id].regions.clone();
            let mut env: HashMap<String, ValueId> = HashMap::new();
            for reg_id in regions {
                removed += gvn_region(module, reg_id, &mut env, &mut value_leader);
            }
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

fn gvn_region(
    module: &mut Module,
    region_id: RegionId,
    outer_env: &mut HashMap<String, ValueId>,
    value_leader: &mut HashMap<ValueId, ValueId>,
) -> usize {
    let mut removed = 0usize;
    let Some(reg) = module.regions.get(region_id) else {
        return 0;
    };
    let blocks: Vec<BlockId> = reg.blocks.clone();
    for block_id in blocks {
        let mut block_env = outer_env.clone();
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            for &nested_reg in &op.regions {
                removed += gvn_region(module, nested_reg, &mut block_env, value_leader);
            }
            if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                continue;
            }
            let res_val = op.results[0];
            let res_ty = module.values[res_val].ty.clone();
            let sig = gvn_signature(&op, &res_ty, value_leader);
            if let Some(&leader_val) = block_env.get(&sig)
                && leader_val != res_val
            {
                value_leader.insert(res_val, leader_val);
                module.replace_all_uses(res_val, leader_val);
                module.remove_op(block_id, op_id);
                removed += 1;
            } else {
                value_leader.insert(res_val, res_val);
                block_env.insert(sig, res_val);
            }
        }
        *outer_env = block_env;
    }
    removed
}

fn is_commutative_binary_op(op: &Operation) -> bool {
    op.name.dialect == "stablehlo"
        && op.operands.len() == 2
        && matches!(
            op.name.name.as_str(),
            "add" | "multiply" | "maximum" | "minimum" | "and" | "or" | "xor"
        )
}

fn canonical_cse_signature(op: &Operation, res_ty: &Type) -> String {
    let mut operand_keys: Vec<String> = op
        .operands
        .iter()
        .map(|v: &ValueId| format!("{v:?}"))
        .collect();
    if is_commutative_binary_op(op) {
        operand_keys.sort();
    }

    let mut attr_pairs: Vec<(&String, String)> = op
        .attributes
        .iter()
        .filter(|(k, _)| !k.starts_with("castforge.") && *k != "inplace_operand")
        .map(|(k, v)| (k, format!("{v:?}")))
        .collect();
    attr_pairs.sort_by(|a, b| a.0.cmp(b.0));

    format!(
        "{}|operands:{:?}|attrs:{:?}|ty:{}",
        op.name, operand_keys, attr_pairs, res_ty
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

/// Loop / Region-Invariant Code Motion Pass (`licm`).
pub struct LicmPass;

impl Pass for LicmPass {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn description(&self) -> &'static str {
        "Loop/Region-Invariant Code Motion (hoists invariant pure ops out of nested regions)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for parent_block_id in blocks {
            let parent_ops = module.block_ops(parent_block_id);
            for parent_op_id in parent_ops {
                let Some(parent_op) = module.ops.get(parent_op_id) else {
                    continue;
                };
                if parent_op.is_op("builtin", "module") || parent_op.is_op("func", "func") {
                    continue;
                }
                if parent_op.regions.is_empty() {
                    continue;
                }

                let regions = parent_op.regions.clone();
                let mut inner_blocks = Vec::new();
                for reg_id in regions {
                    inner_blocks.extend(module.regions[reg_id].blocks.iter().copied());
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
                        }
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

                if hoisted_ops.is_empty() {
                    continue;
                }

                for (from_blk, hoisted_op_id) in hoisted_ops {
                    module.blocks[from_blk]
                        .ops
                        .retain(|&id| id != hoisted_op_id);
                    let p_ops = &mut module.blocks[parent_block_id].ops;
                    let pos = p_ops
                        .iter()
                        .position(|&id| id == parent_op_id)
                        .unwrap_or(p_ops.len());
                    p_ops.insert(pos, hoisted_op_id);
                    let _ = ValueDef::OpResult {
                        op: hoisted_op_id,
                        index: 0,
                    };
                    modified += 1;
                }
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
