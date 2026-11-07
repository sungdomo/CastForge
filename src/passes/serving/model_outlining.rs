//! Subgraph model outlining to init and execute callables.
//!
//! Builds the `mgp.model` call graph, detects cycles across `mgp.call` edges
//! (reporting the exact cycle path starting from the re-entered ancestor),
//! propagates runtime weight-name prefixes into `is_subgraph = true` models,
//! splits each model in post-order into `mef.callable` `$init` and `$exec`
//! callables, rewrites `mgp.call` into `grt.invoke` in both `init` and
//! `execute`, and wraps top-level non-subgraph models in `mef.model`.

use std::collections::{BTreeMap, HashMap};

use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

use crate::ir::mlir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, RegionId, Type, Value, ValueDef,
    ValueId,
};

fn make_op_name(full: &str) -> OpName {
    if let Some((dialect, name)) = full.split_once('.') {
        OpName::new(dialect, name)
    } else {
        OpName::new("builtin", full)
    }
}

fn is_terminator_name(full: &str) -> bool {
    matches!(
        full,
        "mef.output" | "mgp.yield" | "mo.output" | "func.return" | "return"
    )
}

/// Summary of a model outlined by [`outline_models_to_mef_callables`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutlinedModelInfo {
    /// Original model symbol name.
    pub model_name: String,
    /// Generated init callable symbol name (`"<model_name>$init"`).
    pub init_callable_name: String,
    /// Generated execute callable symbol name (`"<model_name>$exec"`).
    pub exec_callable_name: String,
    /// Whether this model was a subgraph (`is_subgraph == true`).
    pub is_subgraph: bool,
}

/// Extracts `(init_region, init_block, exec_region, exec_block)` from an
/// `mgp.model` operation.
fn get_model_regions_and_blocks(
    module: &Module,
    model_op_id: OpId,
) -> Result<(RegionId, BlockId, RegionId, BlockId), String> {
    let op = module
        .ops
        .get(model_op_id)
        .ok_or_else(|| "model operation not found".to_string())?;
    if op.regions.len() < 2 {
        return Err("mgp.model requires both init and execute regions".to_string());
    }
    let init_reg = op.regions[0];
    let exec_reg = op.regions[1];
    let init_blk = module
        .regions
        .get(init_reg)
        .and_then(|r| r.blocks.first().copied())
        .ok_or_else(|| "mgp.model init region has no entry block".to_string())?;
    let exec_blk = module
        .regions
        .get(exec_reg)
        .and_then(|r| r.blocks.first().copied())
        .ok_or_else(|| "mgp.model execute region has no entry block".to_string())?;
    Ok((init_reg, init_blk, exec_reg, exec_blk))
}

/// Inserts a block argument at `insert_index` in `block_id`, updating the
/// `ValueDef::BlockArgument { index }` of all subsequent arguments.
fn insert_block_argument_at(
    module: &mut Module,
    block_id: BlockId,
    insert_index: usize,
    ty: Type,
) -> ValueId {
    let arg_len = module.blocks.get(block_id).map_or(0, |b| b.arguments.len());
    let clamped_idx = insert_index.min(arg_len);
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty,
        def: ValueDef::BlockArgument {
            block: block_id,
            index: clamped_idx as u32,
        },
    });
    if let Some(block) = module.blocks.get_mut(block_id) {
        block.arguments.insert(clamped_idx, val_id);
        let updated_args = block.arguments.clone();
        for (i, &aid) in updated_args.iter().enumerate() {
            if let Some(val) = module.values.get_mut(aid) {
                val.def = ValueDef::BlockArgument {
                    block: block_id,
                    index: i as u32,
                };
            }
        }
    }
    val_id
}

/// Rewrites placeholder `mgp.buffer.constant.external` operations (`e_is_placeholder == true`)
/// inside `block_id` by prepending `mgp.string.concat(prefix_val, raw_name)`.
fn prefix_placeholder_external_constants(
    module: &mut Module,
    block_id: BlockId,
    prefix_val: ValueId,
) {
    let Some(block) = module.blocks.get(block_id) else {
        return;
    };
    let orig_ops = block.ops.clone();
    let mut new_ops = Vec::with_capacity(orig_ops.len() + 2);
    let str_ty = Type::Opaque("!mgp.opaque<\"mojo_string\">".to_string());

    for op_id in orig_ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let is_placeholder = op.name.full_name() == "mgp.buffer.constant.external"
            && matches!(
                op.attributes.get("e_is_placeholder"),
                Some(Attribute::Bool(true))
            );
        if !is_placeholder || op.operands.is_empty() {
            new_ops.push(op_id);
            continue;
        }
        let raw_name_val = op.operands[0];
        let concat_op_id = module.ops.insert_with_key(|id| Operation {
            id,
            name: make_op_name("mgp.string.concat"),
            operands: smallvec![prefix_val, raw_name_val],
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: Location::Unknown,
        });
        let concat_ty = str_ty.clone();
        let concat_res = module.values.insert_with_key(|vid| Value {
            id: vid,
            ssa_name: None,
            ty: concat_ty,
            def: ValueDef::OpResult {
                op: concat_op_id,
                index: 0,
            },
        });
        if let Some(cop) = module.ops.get_mut(concat_op_id) {
            cop.results.push(concat_res);
        }
        if let Some(ext_op) = module.ops.get_mut(op_id) {
            ext_op.operands[0] = concat_res;
        }
        new_ops.push(concat_op_id);
        new_ops.push(op_id);
    }

    if let Some(block_mut) = module.blocks.get_mut(block_id) {
        block_mut.ops = new_ops;
    }
}

/// Outlines `mgp.model` operations in `module` into `mef.callable` `$init` and
/// `$exec` operations, lowering `mgp.call` into `grt.invoke` and wrapping
/// non-subgraph models in `mef.model`.
///
/// # Errors
///
/// Returns `Err` if duplicate model symbol names exist, a call references a
/// missing model, or a cycle is detected in the subgraph call graph (reporting
/// the cycle members starting from the re-entered ancestor).
pub fn outline_models_to_mef_callables(
    module: &mut Module,
) -> Result<Vec<OutlinedModelInfo>, String> {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();

    let mut models_by_name: BTreeMap<String, OpId> = BTreeMap::new();
    let mut entry_models: Vec<String> = Vec::new();
    let mut is_subgraph_map: HashMap<String, bool> = HashMap::new();

    for &op_id in &top_ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if op.name.full_name() != "mgp.model" {
            continue;
        }
        let sym_name = match op
            .attributes
            .get("sym_name")
            .or_else(|| op.attributes.get("name"))
        {
            Some(Attribute::String(s)) => s.clone(),
            _ => return Err("mgp.model is missing 'sym_name' attribute".to_string()),
        };
        if models_by_name.insert(sym_name.clone(), op_id).is_some() {
            return Err(format!(
                "two models with the same symbol name existed: {sym_name}"
            ));
        }
        let is_subgraph = matches!(
            op.attributes.get("is_subgraph"),
            Some(Attribute::Bool(true))
        );
        is_subgraph_map.insert(sym_name.clone(), is_subgraph);
        if !is_subgraph {
            entry_models.push(sym_name);
        }
    }

    if models_by_name.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Local prefix propagation for `is_subgraph == true` models.
    let mut exec_prefix_vals: HashMap<String, ValueId> = HashMap::new();
    let str_ty = Type::Opaque("!mgp.opaque<\"mojo_string\">".to_string());

    for (name, &model_id) in &models_by_name {
        if !is_subgraph_map.get(name).copied().unwrap_or(false) {
            continue;
        }
        let (_, init_blk, _, exec_blk) = get_model_regions_and_blocks(module, model_id)?;
        let implicit_count = match module
            .ops
            .get(model_id)
            .and_then(|op| op.attributes.get("implicit_args_count"))
        {
            Some(&Attribute::Int(n)) if n >= 0 => n as usize,
            _ => 0,
        };
        let exec_prefix =
            insert_block_argument_at(module, exec_blk, implicit_count, str_ty.clone());
        exec_prefix_vals.insert(name.clone(), exec_prefix);

        let init_prefix = module.add_block_argument(init_blk, None, str_ty.clone());
        prefix_placeholder_external_constants(module, init_blk, init_prefix);
        prefix_placeholder_external_constants(module, exec_blk, exec_prefix);

        // Append `init_prefix` to the init block terminator operands so exec receives it.
        if let Some(term_id) = module
            .blocks
            .get(init_blk)
            .and_then(|b| b.ops.last().copied())
            && let Some(term_op) = module.ops.get_mut(term_id)
            && is_terminator_name(&term_op.name.full_name())
        {
            term_op.operands.push(init_prefix);
        }
    }

    // Step 2: Post-order recursive `rewrite_and_seal` with cycle detection.
    #[derive(Clone)]
    struct SealedCallableInfo {
        init_name: String,
        exec_name: String,
        init_result_types: Vec<Type>,
    }

    let mut sealed_map: HashMap<String, SealedCallableInfo> = HashMap::new();
    let mut outlined_order: Vec<OutlinedModelInfo> = Vec::new();

    #[allow(clippy::too_many_arguments)]
    fn rewrite_and_seal(
        module: &mut Module,
        root_blk: BlockId,
        model_name: &str,
        models_by_name: &BTreeMap<String, OpId>,
        is_subgraph_map: &HashMap<String, bool>,
        exec_prefix_vals: &HashMap<String, ValueId>,
        ancestors: &mut Vec<String>,
        sealed_map: &mut HashMap<String, SealedCallableInfo>,
        outlined_order: &mut Vec<OutlinedModelInfo>,
    ) -> Result<SealedCallableInfo, String> {
        if let Some(existing) = sealed_map.get(model_name) {
            return Ok(existing.clone());
        }
        if let Some(cycle_start) = ancestors.iter().position(|a| a == model_name) {
            let cycle_nodes = &ancestors[cycle_start..];
            return Err(format!(
                "cycle detected in subgraph call graph: {}",
                cycle_nodes.join(" -> ")
            ));
        }

        let &model_id = models_by_name
            .get(model_name)
            .ok_or_else(|| format!("unknown model '{model_name}'"))?;
        let (init_reg, init_blk, exec_reg, exec_blk) =
            get_model_regions_and_blocks(module, model_id)?;

        ancestors.push(model_name.to_string());

        // Find all `mgp.call` ops in `exec_blk`.
        let exec_ops = module
            .blocks
            .get(exec_blk)
            .map(|b| b.ops.clone())
            .unwrap_or_default();
        let mut call_sites: Vec<(OpId, String, String)> = Vec::new();
        for &op_id in &exec_ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.name.full_name() != "mgp.call" {
                continue;
            }
            let callee = match op
                .attributes
                .get("callee")
                .or_else(|| op.attributes.get("symbol"))
            {
                Some(Attribute::String(s)) => s.trim_start_matches('@').to_string(),
                Some(Attribute::SymbolRef(s)) => s.trim_start_matches('@').to_string(),
                _ => return Err("mgp.call missing 'callee' attribute".to_string()),
            };
            let prefix = match op.attributes.get("prefix") {
                Some(Attribute::String(s)) => s.clone(),
                _ => String::new(),
            };
            let _ = rewrite_and_seal(
                module,
                root_blk,
                &callee,
                models_by_name,
                is_subgraph_map,
                exec_prefix_vals,
                ancestors,
                sealed_map,
                outlined_order,
            )?;
            call_sites.push((op_id, callee, prefix));
        }

        // Thread child init invokes into parent init block and child exec invokes into exec block.
        if !call_sites.is_empty() {
            let str_ty = Type::Opaque("!mgp.opaque<\"mojo_string\">".to_string());
            let init_args: Vec<ValueId> = module
                .blocks
                .get(init_blk)
                .map(|b| b.arguments.clone())
                .unwrap_or_default();
            let init_term_id = module
                .blocks
                .get(init_blk)
                .and_then(|b| b.ops.last().copied())
                .ok_or_else(|| "init block missing terminator".to_string())?;

            // Remove terminator temporarily so we can append before it.
            if let Some(ib) = module.blocks.get_mut(init_blk) {
                ib.ops.pop();
            }

            let mut init_chains: SmallVec<[ValueId; 4]> = SmallVec::new();
            let mut per_call_exec_bundles: Vec<Vec<ValueId>> = Vec::with_capacity(call_sites.len());
            let mut per_call_init_arity: Vec<usize> = Vec::with_capacity(call_sites.len());

            for (_call_op_id, callee_name, prefix_str) in &call_sites {
                let target_info = sealed_map
                    .get(callee_name)
                    .cloned()
                    .ok_or_else(|| format!("unsealed callee '{callee_name}'"))?;

                let mut str_attrs = IndexMap::new();
                str_attrs.insert("value".to_string(), Attribute::String(prefix_str.clone()));
                let str_create_id = module.add_operation(
                    init_blk,
                    make_op_name("mgp.string.create"),
                    SmallVec::new(),
                    vec![(None, str_ty.clone())],
                    str_attrs,
                    Vec::new(),
                    false,
                    Location::Unknown,
                );
                let mut prefix_val = module.ops[str_create_id].results[0];

                if let Some(&parent_prefix) = exec_prefix_vals.get(model_name) {
                    let concat_id = module.add_operation(
                        init_blk,
                        make_op_name("mgp.string.concat"),
                        smallvec![parent_prefix, prefix_val],
                        vec![(None, str_ty.clone())],
                        IndexMap::new(),
                        Vec::new(),
                        false,
                        Location::Unknown,
                    );
                    prefix_val = module.ops[concat_id].results[0];
                }

                let mut invoke_operands: SmallVec<[ValueId; 4]> =
                    init_args.iter().copied().collect();
                invoke_operands.push(prefix_val);
                if let Some(&first_arg) = init_args.first() {
                    init_chains.push(first_arg);
                }

                let mut invoke_attrs = IndexMap::new();
                invoke_attrs.insert(
                    "callee".to_string(),
                    Attribute::String(target_info.init_name.clone()),
                );
                let res_specs: Vec<(Option<String>, Type)> = target_info
                    .init_result_types
                    .iter()
                    .cloned()
                    .map(|t| (None, t))
                    .collect();
                let init_invoke_id = module.add_operation(
                    init_blk,
                    make_op_name("grt.invoke"),
                    invoke_operands,
                    res_specs,
                    invoke_attrs,
                    Vec::new(),
                    false,
                    Location::Unknown,
                );
                let invoke_results: Vec<ValueId> =
                    module.ops[init_invoke_id].results.iter().copied().collect();
                per_call_init_arity.push(invoke_results.len());

                let insertion_base = module.ops[init_term_id].operands.len();
                module.ops[init_term_id]
                    .operands
                    .extend(invoke_results.iter().copied());

                let mut exec_bundle = Vec::with_capacity(target_info.init_result_types.len());
                for (i, r_ty) in target_info.init_result_types.iter().enumerate() {
                    let exec_arg = insert_block_argument_at(
                        module,
                        exec_blk,
                        insertion_base + i,
                        r_ty.clone(),
                    );
                    exec_bundle.push(exec_arg);
                }
                per_call_exec_bundles.push(exec_bundle);
            }

            // Join init chains into the init terminator's first operand.
            if !module.ops[init_term_id].operands.is_empty() {
                let parent_chain = module.ops[init_term_id].operands[0];
                init_chains.push(parent_chain);
                let chain_create_id = module.add_operation(
                    init_blk,
                    make_op_name("grt.chain.create"),
                    init_chains,
                    vec![(None, Type::Opaque("!grt.chain".to_string()))],
                    IndexMap::new(),
                    Vec::new(),
                    false,
                    Location::Unknown,
                );
                let merged_chain = module.ops[chain_create_id].results[0];
                module.ops[init_term_id].operands[0] = merged_chain;
            }

            if let Some(ib) = module.blocks.get_mut(init_blk) {
                ib.ops.push(init_term_id);
            }

            // Rewrite each `mgp.call` in `exec_blk` into `grt.invoke` targeting `<callee>$exec`.
            for (idx, (call_op_id, callee_name, _prefix)) in call_sites.iter().enumerate() {
                let target_info = sealed_map
                    .get(callee_name)
                    .cloned()
                    .ok_or_else(|| format!("unsealed callee '{callee_name}'"))?;
                let call_operands: Vec<ValueId> =
                    module.ops[*call_op_id].operands.iter().copied().collect();
                let drop_count = per_call_init_arity[idx].saturating_sub(1);
                let explicit_args = if call_operands.len() >= drop_count {
                    &call_operands[drop_count..]
                } else {
                    &call_operands[..]
                };

                let mut exec_invoke_operands: SmallVec<[ValueId; 4]> =
                    per_call_exec_bundles[idx].iter().copied().collect();
                exec_invoke_operands.extend(explicit_args.iter().copied());

                if let Some(call_mut) = module.ops.get_mut(*call_op_id) {
                    call_mut.name = make_op_name("grt.invoke");
                    call_mut.operands = exec_invoke_operands;
                    call_mut.attributes.insert(
                        "callee".to_string(),
                        Attribute::String(target_info.exec_name),
                    );
                }
            }
        }

        // Collect init and exec result types from their block terminators.
        let init_result_types: Vec<Type> = module
            .blocks
            .get(init_blk)
            .and_then(|b| b.ops.last().copied())
            .and_then(|tid| module.ops.get(tid))
            .map(|t| {
                t.operands
                    .iter()
                    .filter_map(|&vid| module.values.get(vid).map(|v| v.ty.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let exec_result_types: Vec<Type> = module
            .blocks
            .get(exec_blk)
            .and_then(|b| b.ops.last().copied())
            .and_then(|tid| module.ops.get(tid))
            .map(|t| {
                t.operands
                    .iter()
                    .filter_map(|&vid| module.values.get(vid).map(|v| v.ty.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let init_arg_types: Vec<Type> = module
            .blocks
            .get(init_blk)
            .map(|b| {
                b.arguments
                    .iter()
                    .filter_map(|&vid| module.values.get(vid).map(|v| v.ty.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let exec_arg_types: Vec<Type> = module
            .blocks
            .get(exec_blk)
            .map(|b| {
                b.arguments
                    .iter()
                    .filter_map(|&vid| module.values.get(vid).map(|v| v.ty.clone()))
                    .collect()
            })
            .unwrap_or_default();

        let init_callable_name = format!("{model_name}$init");
        let exec_callable_name = format!("{model_name}$exec");

        let mut init_callable_attrs = IndexMap::new();
        init_callable_attrs.insert(
            "sym_name".to_string(),
            Attribute::String(init_callable_name.clone()),
        );
        init_callable_attrs.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: init_arg_types.clone(),
                results: init_result_types.clone(),
            }),
        );
        module.add_operation(
            root_blk,
            make_op_name("mef.callable"),
            SmallVec::new(),
            Vec::new(),
            init_callable_attrs,
            vec![init_reg],
            false,
            Location::Unknown,
        );

        let mut exec_callable_attrs = IndexMap::new();
        exec_callable_attrs.insert(
            "sym_name".to_string(),
            Attribute::String(exec_callable_name.clone()),
        );
        exec_callable_attrs.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: exec_arg_types.clone(),
                results: exec_result_types.clone(),
            }),
        );
        module.add_operation(
            root_blk,
            make_op_name("mef.callable"),
            SmallVec::new(),
            Vec::new(),
            exec_callable_attrs,
            vec![exec_reg],
            false,
            Location::Unknown,
        );

        let is_subgraph = is_subgraph_map.get(model_name).copied().unwrap_or(false);
        if !is_subgraph {
            // Create wrapper `mef.model` that invokes `$init` and `$exec`.
            let wrap_init_blk = module.create_block(Some("init".to_string()));
            let mut wrap_init_args: SmallVec<[ValueId; 4]> = SmallVec::new();
            for ty in &init_arg_types {
                wrap_init_args.push(module.add_block_argument(wrap_init_blk, None, ty.clone()));
            }
            let mut inv_i_attrs = IndexMap::new();
            inv_i_attrs.insert(
                "callee".to_string(),
                Attribute::String(init_callable_name.clone()),
            );
            let inv_i_id = module.add_operation(
                wrap_init_blk,
                make_op_name("grt.invoke"),
                wrap_init_args,
                init_result_types
                    .iter()
                    .cloned()
                    .map(|t| (None, t))
                    .collect(),
                inv_i_attrs,
                Vec::new(),
                false,
                Location::Unknown,
            );
            let inv_i_res: SmallVec<[ValueId; 4]> =
                module.ops[inv_i_id].results.iter().copied().collect();
            module.add_operation(
                wrap_init_blk,
                make_op_name("mef.output"),
                inv_i_res,
                Vec::new(),
                IndexMap::new(),
                Vec::new(),
                false,
                Location::Unknown,
            );
            let wrap_init_reg = module.create_region(vec![wrap_init_blk]);

            let wrap_exec_blk = module.create_block(Some("execute".to_string()));
            let mut wrap_exec_args: SmallVec<[ValueId; 4]> = SmallVec::new();
            for ty in &exec_arg_types {
                wrap_exec_args.push(module.add_block_argument(wrap_exec_blk, None, ty.clone()));
            }
            let mut inv_e_attrs = IndexMap::new();
            inv_e_attrs.insert(
                "callee".to_string(),
                Attribute::String(exec_callable_name.clone()),
            );
            let inv_e_id = module.add_operation(
                wrap_exec_blk,
                make_op_name("grt.invoke"),
                wrap_exec_args,
                exec_result_types
                    .iter()
                    .cloned()
                    .map(|t| (None, t))
                    .collect(),
                inv_e_attrs,
                Vec::new(),
                false,
                Location::Unknown,
            );
            let inv_e_res: SmallVec<[ValueId; 4]> =
                module.ops[inv_e_id].results.iter().copied().collect();
            module.add_operation(
                wrap_exec_blk,
                make_op_name("mef.output"),
                inv_e_res,
                Vec::new(),
                IndexMap::new(),
                Vec::new(),
                false,
                Location::Unknown,
            );
            let wrap_exec_reg = module.create_region(vec![wrap_exec_blk]);

            let mut mef_attrs = IndexMap::new();
            mef_attrs.insert(
                "sym_name".to_string(),
                Attribute::String(model_name.to_string()),
            );
            module.add_operation(
                root_blk,
                make_op_name("mef.model"),
                SmallVec::new(),
                Vec::new(),
                mef_attrs,
                vec![wrap_init_reg, wrap_exec_reg],
                false,
                Location::Unknown,
            );
        }

        // Unlink and erase the original `mgp.model` operation (without deleting its moved regions).
        if let Some(rb) = module.blocks.get_mut(root_blk) {
            rb.ops.retain(|&oid| oid != model_id);
        }
        module.ops.remove(model_id);

        ancestors.pop();

        let sealed_info = SealedCallableInfo {
            init_name: init_callable_name.clone(),
            exec_name: exec_callable_name.clone(),
            init_result_types,
        };
        sealed_map.insert(model_name.to_string(), sealed_info.clone());
        outlined_order.push(OutlinedModelInfo {
            model_name: model_name.to_string(),
            init_callable_name,
            exec_callable_name,
            is_subgraph,
        });
        Ok(sealed_info)
    }

    let mut ancestors = Vec::new();
    for entry_name in &entry_models {
        rewrite_and_seal(
            module,
            root_blk,
            entry_name,
            &models_by_name,
            &is_subgraph_map,
            &exec_prefix_vals,
            &mut ancestors,
            &mut sealed_map,
            &mut outlined_order,
        )?;
    }

    Ok(outlined_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outline_empty_module() {
        let mut module = Module::new(Some("empty".to_string()));
        let outlined = outline_models_to_mef_callables(&mut module).expect("empty outline");
        assert!(outlined.is_empty());
    }
}
