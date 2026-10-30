//! Submodule `inliner` of `normalize_ir`.

use crate::ir::{Attribute, BlockId, Module, OpId, Type, ValueDef, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// Identical Function Deduplicator Pass (`dedup-functions`: `Ir -> Ir`).
#[derive(Debug, Clone, Default)]
pub struct DedupFunctionsPass;

impl Pass for DedupFunctionsPass {
    fn name(&self) -> &'static str {
        "dedup-functions"
    }

    fn description(&self) -> &'static str {
        "Merges structurally identical func.func definitions and rewrites func.call sites"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let funcs = module.functions();
        if funcs.len() < 2 {
            return Ok(PassResult::unchanged(self.name()));
        }

        let mut sig_to_sym: HashMap<String, String> = HashMap::new();
        let mut redirect_map: HashMap<String, String> = HashMap::new();
        let mut duplicate_fn_ids: Vec<OpId> = Vec::new();

        for (idx, &fn_id) in funcs.iter().enumerate() {
            let sym = module.ops[fn_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .unwrap_or("func")
                .to_string();
            let fingerprint = fingerprint_function(module, fn_id);
            if let Some(canon_sym) = sig_to_sym.get(&fingerprint) {
                if idx > 0 {
                    redirect_map.insert(sym, canon_sym.clone());
                    duplicate_fn_ids.push(fn_id);
                }
            } else {
                sig_to_sym.insert(fingerprint, sym);
            }
        }

        if duplicate_fn_ids.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let mut modified_calls = 0usize;
        for (_, op) in module.ops.iter_mut() {
            if !op.is_op("func", "call") {
                continue;
            }
            if let Some(callee) = op
                .attributes
                .get("callee")
                .and_then(|a| a.as_str())
                .map(str::to_string)
                && let Some(canon) = redirect_map.get(&callee)
            {
                op.attributes
                    .insert("callee".to_string(), Attribute::String(canon.clone()));
                modified_calls += 1;
            }
        }

        let mod_body = module.body_block();
        let removed = duplicate_fn_ids.len();
        for fn_id in duplicate_fn_ids {
            module.remove_op(mod_body, fn_id);
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: removed,
            ops_modified: modified_calls,
            ops_created: 0,
        })
    }
}

fn fingerprint_function(module: &Module, fn_id: OpId) -> String {
    let mut fp = String::new();
    let op = &module.ops[fn_id];
    let Some(&reg_id) = op.regions.first() else {
        return fp;
    };
    let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
        return fp;
    };
    let block = &module.blocks[blk_id];

    let mut val_slots: HashMap<ValueId, String> = HashMap::new();
    for (i, &arg_id) in block.arguments.iter().enumerate() {
        let ty = &module.values[arg_id].ty;
        let _ = writeln!(fp, "arg({i}):{ty}");
        val_slots.insert(arg_id, format!("a{i}"));
    }

    for (op_idx, &child_id) in block.ops.iter().enumerate() {
        let child = &module.ops[child_id];
        let opnds: Vec<String> = child
            .operands
            .iter()
            .map(|&v| {
                val_slots
                    .get(&v)
                    .cloned()
                    .unwrap_or_else(|| "?".to_string())
            })
            .collect();
        let res_tys: Vec<String> = child
            .results
            .iter()
            .enumerate()
            .map(|(r_i, &r)| {
                val_slots.insert(r, format!("r{op_idx}_{r_i}"));
                module.values[r].ty.to_string()
            })
            .collect();
        let mut attr_pairs: Vec<(&String, &Attribute)> = child.attributes.iter().collect();
        attr_pairs.sort_by_key(|(k, _)| *k);
        let attrs_str: Vec<String> = attr_pairs
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let _ = writeln!(
            fp,
            "op({}):{}({})->[{}]{{{}}}",
            op_idx,
            child.name,
            opnds.join(","),
            res_tys.join(","),
            attrs_str.join(",")
        );
    }
    fp
}

/// Function Call Inliner Pass (`inline-calls`).
pub struct InlineCallsPass;

impl Pass for InlineCallsPass {
    fn name(&self) -> &'static str {
        "inline-calls"
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

/// Unified Subgraph Extractor & Delta IR Reducer Pass (`extract-subgraph`).
#[derive(Debug, Clone, Default)]
pub struct ExtractSubgraphPass {
    pub target: String,
    pub stop_at: String,
    pub max_ops: Option<usize>,
    pub keep_op: String,
}

impl Pass for ExtractSubgraphPass {
    fn name(&self) -> &'static str {
        "extract-subgraph"
    }

    fn description(&self) -> &'static str {
        "Extracts backward dataflow slice or delta-reduces function to a minimal reproducer"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        if self.max_ops.is_some() || !self.keep_op.is_empty() {
            let mut res = ReduceIrPass {
                max_ops: self.max_ops.unwrap_or(1),
                keep_op: self.keep_op.clone(),
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }

        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let compute_ops: Vec<OpId> = ops
            .iter()
            .copied()
            .filter(|&id| !module.ops[id].is_op("func", "return"))
            .collect();
        if compute_ops.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let clean_target = self.target.trim().trim_start_matches('%');
        let mut target_val: Option<ValueId> = None;

        if !clean_target.is_empty() {
            for &op_id in &compute_ops {
                for &res in &module.ops[op_id].results {
                    if module.values[res].ssa_name.as_deref() == Some(clean_target) {
                        target_val = Some(res);
                        break;
                    }
                }
            }
            if target_val.is_none()
                && let Ok(idx) = clean_target.parse::<usize>()
                && let Some(&op_id) = compute_ops.get(idx)
            {
                target_val = module.ops[op_id].results.first().copied();
            }
        } else {
            for &op_id in &compute_ops {
                if !module.ops[op_id].is_op("stablehlo", "constant")
                    && let Some(&res) = module.ops[op_id].results.first()
                {
                    target_val = Some(res);
                    break;
                }
            }
        }

        let Some(target_vid) = target_val else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let stop_names: HashSet<String> = self
            .stop_at
            .split([',', ';'])
            .map(|s| s.trim().trim_start_matches('%').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut keep_ops: HashSet<OpId> = HashSet::new();
        let mut promoted_vals: Vec<ValueId> = Vec::new();
        let mut visited_vals: HashSet<ValueId> = HashSet::new();
        let mut worklist = vec![target_vid];

        while let Some(vid) = worklist.pop() {
            if !visited_vals.insert(vid) {
                continue;
            }
            let ssa = module.values[vid].ssa_name.as_deref().unwrap_or("");
            if vid != target_vid && stop_names.contains(ssa) {
                if matches!(module.values[vid].def, ValueDef::OpResult { .. }) {
                    promoted_vals.push(vid);
                }
                continue;
            }
            if let ValueDef::OpResult { op: prod_op, .. } = module.values[vid].def {
                keep_ops.insert(prod_op);
                for &opnd in &module.ops[prod_op].operands {
                    worklist.push(opnd);
                }
            }
        }

        for &p_vid in &promoted_vals {
            let ssa = module.values[p_vid].ssa_name.clone();
            let ty = module.values[p_vid].ty.clone();
            let new_arg = module.add_block_argument(block_id, ssa, ty);
            module.replace_all_uses(p_vid, new_arg);
        }

        let mut removed = 0usize;
        let mut modified = 0usize;
        for &op_id in &ops {
            if module.ops[op_id].is_op("func", "return") {
                if module.ops[op_id].operands.as_slice() != [target_vid] {
                    module.ops[op_id].operands = smallvec![target_vid];
                    modified += 1;
                }
                continue;
            }
            if !keep_ops.contains(&op_id) {
                module.remove_op(block_id, op_id);
                removed += 1;
            }
        }

        let in_types: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let out_types = vec![module.values[target_vid].ty.clone()];
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: in_types,
                results: out_types,
            }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || modified > 0 || !promoted_vals.is_empty(),
            ops_removed: removed,
            ops_modified: modified,
            ops_created: promoted_vals.len(),
        })
    }
}

/// Delta IR Reducer Pass (`reduce-ir`).
#[derive(Debug, Clone)]
pub struct ReduceIrPass {
    pub max_ops: usize,
    pub keep_op: String,
}

impl Default for ReduceIrPass {
    fn default() -> Self {
        Self {
            max_ops: 1,
            keep_op: String::new(),
        }
    }
}

impl Pass for ReduceIrPass {
    fn name(&self) -> &'static str {
        "reduce-ir"
    }

    fn description(&self) -> &'static str {
        "Delta-reduces a function to a minimal reproducer of at most max_ops operations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let compute_ops: Vec<OpId> = ops
            .iter()
            .copied()
            .filter(|&id| !module.ops[id].is_op("func", "return"))
            .collect();
        let limit = self.max_ops.max(1);
        if compute_ops.len() <= limit && self.keep_op.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let filter = self.keep_op.trim().to_ascii_lowercase();
        let mut retained: Vec<OpId> = compute_ops
            .iter()
            .copied()
            .filter(|&id| {
                if filter.is_empty() {
                    !module.ops[id].is_op("stablehlo", "constant")
                } else {
                    module.ops[id]
                        .name
                        .to_string()
                        .to_ascii_lowercase()
                        .contains(&filter)
                }
            })
            .take(limit)
            .collect();

        if retained.is_empty()
            && let Some(&first_op) = compute_ops.first()
        {
            retained.push(first_op);
        }

        let Some(&last_op) = retained.last() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let Some(&ret_val) = module.ops[last_op].results.first() else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let keep_set: HashSet<OpId> = retained.iter().copied().collect();
        let mut to_promote: Vec<ValueId> = Vec::new();
        for &op_id in &retained {
            for &opnd in &module.ops[op_id].operands {
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[opnd].def
                    && !keep_set.contains(&prod_op)
                    && !to_promote.contains(&opnd)
                {
                    to_promote.push(opnd);
                }
            }
        }

        for &vid in &to_promote {
            let ssa = module.values[vid].ssa_name.clone();
            let ty = module.values[vid].ty.clone();
            let new_arg = module.add_block_argument(block_id, ssa, ty);
            module.replace_all_uses(vid, new_arg);
        }

        let mut removed = 0usize;
        for &op_id in &ops {
            if module.ops[op_id].is_op("func", "return") {
                module.ops[op_id].operands = smallvec![ret_val];
                continue;
            }
            if !keep_set.contains(&op_id) {
                module.remove_op(block_id, op_id);
                removed += 1;
            }
        }

        let in_types: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let out_types = vec![module.values[ret_val].ty.clone()];
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: in_types,
                results: out_types,
            }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || !to_promote.is_empty(),
            ops_removed: removed,
            ops_modified: 1,
            ops_created: to_promote.len(),
        })
    }
}
