use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Location, Module, OpId, OpName, Type, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;

/// 5. Fuse Elementwise Pass (`stablehlo.add` + `stablehlo.maximum` -> `stablehlo.composite`)
pub struct FuseElementwisePass;

impl Pass for FuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Maximum {
                    lhs,
                    rhs,
                    result: max_res,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                // Check if `lhs` is produced by `stablehlo.add` with a single use
                let ValueDef::OpResult { op: add_op_id, .. } = module.values[lhs].def else {
                    continue;
                };
                if module.use_count(lhs) != 1 {
                    continue;
                }
                let Some(add_op) = module.ops.get(add_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add {
                    lhs: add_a,
                    rhs: add_b,
                    ..
                }) = StableHloOp::from_op(add_op)
                else {
                    continue;
                };

                // Rewrite `max_op` in-place into `stablehlo.composite` representing fused bias+relu
                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_add_max".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_add_max_impl".to_string()),
                );

                let max_op_mut = &mut module.ops[op_id];
                max_op_mut.name = OpName::new("stablehlo", "composite");
                max_op_mut.operands = SmallVec::from_slice(&[add_a, add_b, rhs]);
                max_op_mut.attributes = attrs;
                let _ = max_res;
                created += 1;

                // Remove the now-dead `add` op
                module.remove_op(block_id, add_op_id);
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

/// 6. Surgical IR Edit Pass (`ir-edit`)
pub struct IrEditPass {
    /// Format: `"old_dialect.old_op:new_dialect.new_op"`
    pub rename_op: Option<String>,
    /// Format: `"dialect.op:attr_key:attr_value"`
    pub annotate_op: Option<String>,
}

impl Pass for IrEditPass {
    fn name(&self) -> &'static str {
        "ir-edit"
    }

    fn description(&self) -> &'static str {
        "Surgical IR editing pass (rename_op=from:to, annotate_op=op:key:val)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;

        if let Some(rule) = &self.rename_op {
            if let Some((from, to)) = rule.split_once(':') {
                let from_name = OpName::parse(from.trim());
                let to_name = OpName::parse(to.trim());
                for (_, op) in &mut module.ops {
                    if op.name == from_name {
                        op.name = to_name.clone();
                        modified += 1;
                    }
                }
            }
        }

        if let Some(rule) = &self.annotate_op {
            let parts: Vec<&str> = rule.splitn(3, ':').collect();
            if parts.len() == 3 {
                let target_op = OpName::parse(parts[0].trim());
                let attr_key = parts[1].trim().to_string();
                let attr_val = Attribute::String(parts[2].trim().to_string());
                for (_, op) in &mut module.ops {
                    if op.name == target_op {
                        op.attributes.insert(attr_key.clone(), attr_val.clone());
                        modified += 1;
                    }
                }
            }
        }

        let _ = Location::Unknown;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// 14. Function Call Inliner Pass (`inliner`)
/// Inlines `func.call @callee(...)` operations into their caller blocks.
pub struct InlinerPass;

impl Pass for InlinerPass {
    fn name(&self) -> &'static str {
        "inliner"
    }

    fn description(&self) -> &'static str {
        "Inlines func.call operations by splicing callee bodies into caller blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut created = 0usize;

        let mut func_map: HashMap<String, OpId> = HashMap::new();
        for fn_op_id in module.functions() {
            if let Some(name) = module.ops[fn_op_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
            {
                func_map.insert(name.to_string(), fn_op_id);
            }
        }

        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for call_op_id in ops {
                let Some(call_op) = module.ops.get(call_op_id).cloned() else {
                    continue;
                };
                if !call_op.is_op("func", "call") {
                    continue;
                }
                let Some(callee_name) = call_op.attributes.get("callee").and_then(|a| a.as_str())
                else {
                    continue;
                };
                let Some(&callee_fn_id) = func_map.get(callee_name) else {
                    continue;
                };
                let callee_reg = module.ops[callee_fn_id].regions[0];
                let callee_block = module.regions[callee_reg].blocks[0];
                if callee_block == block_id {
                    continue;
                }

                let callee_args = module.blocks[callee_block].arguments.clone();
                if callee_args.len() != call_op.operands.len() {
                    continue;
                }

                let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
                for (&arg_val, &caller_val) in callee_args.iter().zip(call_op.operands.iter()) {
                    val_map.insert(arg_val, caller_val);
                }

                let callee_ops = module.blocks[callee_block].ops.clone();
                let mut new_op_ids = Vec::new();
                let mut return_values = Vec::new();

                for src_op_id in callee_ops {
                    let src_op = module.ops[src_op_id].clone();
                    if src_op.is_op("func", "return") {
                        for &ret_opnd in &src_op.operands {
                            let mapped = val_map.get(&ret_opnd).copied().unwrap_or(ret_opnd);
                            return_values.push(mapped);
                        }
                        continue;
                    }

                    let mapped_operands: SmallVec<[ValueId; 4]> = src_op
                        .operands
                        .iter()
                        .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                        .collect();

                    let res_types: Vec<(Option<String>, Type)> = src_op
                        .results
                        .iter()
                        .map(|&r| {
                            let v = &module.values[r];
                            let inlined_name = v
                                .ssa_name
                                .as_ref()
                                .map(|n| format!("inl_{callee_name}_{n}"));
                            (inlined_name, v.ty.clone())
                        })
                        .collect();

                    let cloned_id = module.add_operation(
                        block_id,
                        src_op.name.clone(),
                        mapped_operands,
                        res_types,
                        src_op.attributes.clone(),
                        Vec::new(),
                        src_op.custom_syntax,
                        src_op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    for (&old_res, &new_res) in src_op
                        .results
                        .iter()
                        .zip(module.ops[cloned_id].results.iter())
                    {
                        val_map.insert(old_res, new_res);
                    }
                    new_op_ids.push(cloned_id);
                    created += 1;
                }

                for (&call_res, &ret_val) in call_op.results.iter().zip(return_values.iter()) {
                    module.replace_all_uses(call_res, ret_val);
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == call_op_id)
                {
                    module.blocks[block_id].ops.splice(pos..pos, new_op_ids);
                }
                module.remove_op(block_id, call_op_id);
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}
