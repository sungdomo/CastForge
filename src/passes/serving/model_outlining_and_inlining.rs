//! Subgraph model outlining to init/execute callables, subgraph call-graph
//! inlining with runtime slot offsetting, and overhanging chain fixup.
//!
//! Covers:
//! 1. **Subgraph Model Outlining (`outline_models_to_mef_callables`)**:
//!    Builds the `mgp.model` call graph, detects cycles across `mgp.call` edges
//!    (reporting the exact cycle path starting from the re-entered ancestor),
//!    propagates runtime weight-name prefixes into `is_subgraph = true` models,
//!    splits each model in post-order into `mef.callable` `$init` and `$exec`
//!    callables, rewrites `mgp.call` into `grt.invoke` in both `init` and
//!    `execute`, and wraps top-level non-subgraph models in `mef.model`.
//! 2. **Subgraph Call-Graph Inlining with Runtime Slot Offsetting
//!    (`offset_runtime_slots`, `inline_mgp_calls`)**:
//!    Detects call-graph cycles and missing `mgp.context.create` operations,
//!    clones callee `init` and `execute` blocks into the caller in bottom-up
//!    order while offsetting `buffer_slot` and `runtime_slot` attributes by the
//!    caller's accumulated slot usage, updates the caller's `num_slots`, and
//!    erases dead inlined subgraph models.
//! 3. **Overhanging Chain Fixup (`fixup_overhang_chains`)**:
//!    Finds operations with unused chain/result values in each block (taking at
//!    most one unused result per operation since a failing operation sets all
//!    its results to error) and joins all hanging chains into the block
//!    terminator's chain operand via `grt.chain.create`.

use std::collections::{BTreeMap, HashMap, HashSet};

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

// ---------------------------------------------------------------------------
// 1. Subgraph Model Outlining to Init/Execute Callables
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 2. Subgraph Call Inlining with Runtime Slot Offsetting
// ---------------------------------------------------------------------------

/// Offsets the runtime/buffer slot index attributes (`buffer_slot`,
/// `b_buffer_slot`, `runtime_slot`, `a_runtime_slot`, `slot`) of `op_id` and
/// any nested operations by `slot_offset`.
pub fn offset_runtime_slots(module: &mut Module, op_id: OpId, slot_offset: u64) {
    if slot_offset == 0 {
        return;
    }
    let Some(op) = module.ops.get_mut(op_id) else {
        return;
    };
    let full = op.name.full_name();
    let is_slotted_op = matches!(
        full.as_str(),
        "mgp.buffer.cache"
            | "mgp.runtime.create"
            | "mgp.module.load"
            | "mgp.execute_generic"
            | "mgp.generic.execute"
    );
    if is_slotted_op {
        for key in [
            "buffer_slot",
            "b_buffer_slot",
            "runtime_slot",
            "a_runtime_slot",
            "slot",
        ] {
            if let Some(Attribute::Int(old_slot)) = op.attributes.get_mut(key) {
                *old_slot = (*old_slot).saturating_add(slot_offset as i64);
            }
        }
    }
    let nested_regions = op.regions.clone();
    for reg_id in nested_regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        let blocks = reg.blocks.clone();
        for blk_id in blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            let child_ops = blk.ops.clone();
            for child_op in child_ops {
                offset_runtime_slots(module, child_op, slot_offset);
            }
        }
    }
}

/// Inlines `mgp.call` subgraph invocations into caller `mgp.model` operations in
/// bottom-up call-graph order, offsetting runtime slots and erasing dead
/// `is_subgraph = true` models.
///
/// # Errors
///
/// Returns `Err` if a cycle is detected in the `mgp.model` call graph, a model
/// is missing an `mgp.context.create` operation in its `init` block, or a
/// callee device context argument cannot be mapped.
pub fn inline_mgp_calls(module: &mut Module) -> Result<usize, String> {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();

    let mut models_by_name: BTreeMap<String, OpId> = BTreeMap::new();
    for &op_id in &top_ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if op.name.full_name() == "mgp.model"
            && let Some(Attribute::String(name)) = op
                .attributes
                .get("sym_name")
                .or_else(|| op.attributes.get("name"))
        {
            models_by_name.insert(name.clone(), op_id);
        }
    }

    if models_by_name.len() <= 1 {
        return Ok(0);
    }

    // Build call edges and check every model has `mgp.context.create` with `num_slots`.
    let mut call_edges: BTreeMap<String, Vec<(OpId, String, String)>> = BTreeMap::new();
    let mut model_ctx_op: HashMap<String, (OpId, u64)> = HashMap::new();

    for (name, &model_id) in &models_by_name {
        let (_, init_blk, _, exec_blk) = get_model_regions_and_blocks(module, model_id)?;
        let init_ops = module.blocks[init_blk].ops.clone();
        let mut found_ctx: Option<(OpId, u64)> = None;
        for &oid in &init_ops {
            if let Some(op) = module.ops.get(oid)
                && op.name.full_name() == "mgp.context.create"
            {
                let slots = match op.attributes.get("num_slots") {
                    Some(&Attribute::Int(s)) if s >= 0 => s as u64,
                    _ => 0,
                };
                found_ctx = Some((oid, slots));
                break;
            }
        }
        let Some(ctx_pair) = found_ctx else {
            return Err(format!("unable to find context in model @{name}"));
        };
        model_ctx_op.insert(name.clone(), ctx_pair);

        let mut edges = Vec::new();
        for &oid in &module.blocks[exec_blk].ops {
            let Some(op) = module.ops.get(oid) else {
                continue;
            };
            if op.name.full_name() == "mgp.call" {
                let callee = match op
                    .attributes
                    .get("callee")
                    .or_else(|| op.attributes.get("symbol"))
                {
                    Some(Attribute::String(s)) => s.trim_start_matches('@').to_string(),
                    Some(Attribute::SymbolRef(s)) => s.trim_start_matches('@').to_string(),
                    _ => return Err("mgp.call missing 'callee' attribute".to_string()),
                };
                if !models_by_name.contains_key(&callee) {
                    return Err(format!("failed to find symbol @{callee}"));
                }
                let prefix = match op.attributes.get("prefix") {
                    Some(Attribute::String(s)) => s.clone(),
                    _ => String::new(),
                };
                edges.push((oid, callee, prefix));
            }
        }
        call_edges.insert(name.clone(), edges);
    }

    // Topological post-order sort with cycle detection.
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: Vec<String> = Vec::new();
    let mut post_order: Vec<String> = Vec::new();

    fn dfs_order(
        node: &str,
        call_edges: &BTreeMap<String, Vec<(OpId, String, String)>>,
        visited: &mut HashSet<String>,
        on_stack: &mut Vec<String>,
        post_order: &mut Vec<String>,
    ) -> Result<(), String> {
        if let Some(pos) = on_stack.iter().position(|n| n == node) {
            let involved: Vec<String> = on_stack[pos..].iter().map(|s| format!("@{s}")).collect();
            return Err(format!(
                "Cycle detected in MGP call graph, cannot inline. Involved models: {}",
                involved.join(", ")
            ));
        }
        if visited.contains(node) {
            return Ok(());
        }
        on_stack.push(node.to_string());
        if let Some(edges) = call_edges.get(node) {
            for (_, callee, _) in edges {
                dfs_order(callee, call_edges, visited, on_stack, post_order)?;
            }
        }
        on_stack.pop();
        visited.insert(node.to_string());
        post_order.push(node.to_string());
        Ok(())
    }

    for name in models_by_name.keys() {
        dfs_order(
            name,
            &call_edges,
            &mut visited,
            &mut on_stack,
            &mut post_order,
        )?;
    }

    let mut total_inlined = 0usize;
    let mut inlined_counts: HashMap<String, usize> = HashMap::new();

    for caller_name in &post_order {
        let edges = call_edges.get(caller_name).cloned().unwrap_or_default();
        if edges.is_empty() {
            continue;
        }
        let caller_id = models_by_name[caller_name];
        let (_, parent_init_blk, _, parent_exec_blk) =
            get_model_regions_and_blocks(module, caller_id)?;
        let (parent_ctx_op_id, mut caller_slots_used) = model_ctx_op[caller_name];
        let parent_ctx_val = module.ops[parent_ctx_op_id]
            .results
            .first()
            .copied()
            .ok_or_else(|| "mgp.context.create missing result".to_string())?;

        for (call_op_id, callee_name, prefix_str) in edges {
            let callee_id = models_by_name[&callee_name];
            let (_, sub_init_blk, _, sub_exec_blk) =
                get_model_regions_and_blocks(module, callee_id)?;
            let (sub_ctx_op_id, sub_slots_used) = model_ctx_op[&callee_name];
            let sub_ctx_val = module.ops[sub_ctx_op_id]
                .results
                .first()
                .copied()
                .ok_or_else(|| "subgraph mgp.context.create missing result".to_string())?;

            let slot_offset = caller_slots_used;

            // Map subgraph init block args to parent init block args.
            let mut init_map: HashMap<ValueId, ValueId> = HashMap::new();
            init_map.insert(sub_ctx_val, parent_ctx_val);

            let parent_init_args = module.blocks[parent_init_blk].arguments.clone();
            let sub_init_args = module.blocks[sub_init_blk].arguments.clone();
            if let (Some(&s0), Some(&p0)) = (sub_init_args.first(), parent_init_args.first()) {
                init_map.insert(s0, p0);
            }
            for &s_arg in &sub_init_args {
                let s_ty = module.values[s_arg].ty.clone();
                if matches!(&s_ty, Type::Opaque(t) if t.contains("device.context")) {
                    for &p_arg in &parent_init_args {
                        if module.values[p_arg].ty == s_ty {
                            init_map.insert(s_arg, p_arg);
                        }
                    }
                    if !init_map.contains_key(&s_arg) {
                        return Err(format!(
                            "failed to map device context for argument in @{callee_name}"
                        ));
                    }
                }
            }

            // Temporarily pop parent init terminator to insert cloned subgraph init ops before it.
            let parent_init_term = module.blocks[parent_init_blk]
                .ops
                .pop()
                .ok_or_else(|| "parent init block missing terminator".to_string())?;

            let str_ty = Type::Opaque("!mgp.opaque<\"mojo_string\">".to_string());
            let mut p_attrs = IndexMap::new();
            p_attrs.insert("value".to_string(), Attribute::String(prefix_str));
            let prefix_op_id = module.add_operation(
                parent_init_blk,
                make_op_name("mgp.string.create"),
                SmallVec::new(),
                vec![(None, str_ty.clone())],
                p_attrs,
                Vec::new(),
                false,
                Location::Unknown,
            );
            let prefix_val = module.ops[prefix_op_id].results[0];

            let sub_init_ops = module.blocks[sub_init_blk].ops.clone();
            for sub_op_id in sub_init_ops {
                let Some(sub_op) = module.ops.get(sub_op_id).cloned() else {
                    continue;
                };
                let sub_full = sub_op.name.full_name();
                if sub_full == "mgp.context.create" {
                    continue;
                }
                if is_terminator_name(&sub_full) {
                    if let Some(&sub_out_chain_orig) = sub_op.operands.first() {
                        let sub_out_chain = init_map
                            .get(&sub_out_chain_orig)
                            .copied()
                            .unwrap_or(sub_out_chain_orig);
                        if !module.ops[parent_init_term].operands.is_empty() {
                            let parent_out_chain = module.ops[parent_init_term].operands[0];
                            let chain_id = module.add_operation(
                                parent_init_blk,
                                make_op_name("grt.chain.create"),
                                smallvec![parent_out_chain, sub_out_chain],
                                vec![(None, Type::Opaque("!grt.chain".to_string()))],
                                IndexMap::new(),
                                Vec::new(),
                                false,
                                Location::Unknown,
                            );
                            module.ops[parent_init_term].operands[0] =
                                module.ops[chain_id].results[0];
                        }
                    }
                    continue;
                }

                let mut mapped_operands: SmallVec<[ValueId; 4]> = sub_op
                    .operands
                    .iter()
                    .map(|&v| init_map.get(&v).copied().unwrap_or(v))
                    .collect();

                let is_placeholder_ext = sub_full == "mgp.buffer.constant.external"
                    && matches!(
                        sub_op.attributes.get("e_is_placeholder"),
                        Some(Attribute::Bool(true))
                    );
                if is_placeholder_ext && !mapped_operands.is_empty() {
                    let raw_name = mapped_operands[0];
                    let concat_id = module.add_operation(
                        parent_init_blk,
                        make_op_name("mgp.string.concat"),
                        smallvec![prefix_val, raw_name],
                        vec![(None, str_ty.clone())],
                        IndexMap::new(),
                        Vec::new(),
                        false,
                        Location::Unknown,
                    );
                    mapped_operands[0] = module.ops[concat_id].results[0];
                }

                let res_specs: Vec<(Option<String>, Type)> = sub_op
                    .results
                    .iter()
                    .map(|&r| (None, module.values[r].ty.clone()))
                    .collect();
                let cloned_id = module.add_operation(
                    parent_init_blk,
                    sub_op.name,
                    mapped_operands,
                    res_specs,
                    sub_op.attributes,
                    Vec::new(),
                    sub_op.custom_syntax,
                    sub_op.location,
                );
                offset_runtime_slots(module, cloned_id, slot_offset);
                for (&old_r, &new_r) in sub_op
                    .results
                    .iter()
                    .zip(module.ops[cloned_id].results.iter())
                {
                    init_map.insert(old_r, new_r);
                }
            }

            module.blocks[parent_init_blk].ops.push(parent_init_term);

            // Clone subgraph execute ops at `call_op_id` in `parent_exec_blk`.
            let call_operands: Vec<ValueId> =
                module.ops[call_op_id].operands.iter().copied().collect();
            let sub_exec_args = module.blocks[sub_exec_blk].arguments.clone();
            let mut exec_map: HashMap<ValueId, ValueId> = HashMap::new();
            for (&s_arg, &c_opd) in sub_exec_args.iter().zip(call_operands.iter()) {
                exec_map.insert(s_arg, c_opd);
            }

            let sub_exec_ops = module.blocks[sub_exec_blk].ops.clone();
            let mut inlined_exec_ops: Vec<OpId> = Vec::new();
            let mut term_mapped_results: Vec<ValueId> = Vec::new();

            for sub_op_id in sub_exec_ops {
                let Some(sub_op) = module.ops.get(sub_op_id).cloned() else {
                    continue;
                };
                let sub_full = sub_op.name.full_name();
                if is_terminator_name(&sub_full) {
                    term_mapped_results = sub_op
                        .operands
                        .iter()
                        .map(|&v| exec_map.get(&v).copied().unwrap_or(v))
                        .collect();
                    continue;
                }
                let mapped_operands: SmallVec<[ValueId; 4]> = sub_op
                    .operands
                    .iter()
                    .map(|&v| exec_map.get(&v).copied().unwrap_or(v))
                    .collect();
                let res_specs: Vec<(Option<String>, Type)> = sub_op
                    .results
                    .iter()
                    .map(|&r| (None, module.values[r].ty.clone()))
                    .collect();
                let cloned_id = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: sub_op.name,
                    operands: mapped_operands,
                    results: SmallVec::new(),
                    attributes: sub_op.attributes,
                    regions: Vec::new(),
                    custom_syntax: sub_op.custom_syntax,
                    location: sub_op.location,
                });
                for (idx, (_, ty)) in res_specs.into_iter().enumerate() {
                    let r_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty,
                        def: ValueDef::OpResult {
                            op: cloned_id,
                            index: idx as u32,
                        },
                    });
                    module.ops[cloned_id].results.push(r_val);
                }
                offset_runtime_slots(module, cloned_id, slot_offset);
                for (&old_r, &new_r) in sub_op
                    .results
                    .iter()
                    .zip(module.ops[cloned_id].results.iter())
                {
                    exec_map.insert(old_r, new_r);
                }
                inlined_exec_ops.push(cloned_id);
            }

            // Replace call results with `term_mapped_results` and Splice `inlined_exec_ops`.
            let call_results: Vec<ValueId> =
                module.ops[call_op_id].results.iter().copied().collect();
            for (&old_res, &new_res) in call_results.iter().zip(term_mapped_results.iter()) {
                module.replace_all_uses(old_res, new_res);
            }

            let orig_parent_exec_ops = module.blocks[parent_exec_blk].ops.clone();
            let mut new_parent_exec_ops =
                Vec::with_capacity(orig_parent_exec_ops.len() + inlined_exec_ops.len());
            for oid in orig_parent_exec_ops {
                if oid == call_op_id {
                    new_parent_exec_ops.extend(inlined_exec_ops.iter().copied());
                } else {
                    new_parent_exec_ops.push(oid);
                }
            }
            module.blocks[parent_exec_blk].ops = new_parent_exec_ops;
            module.remove_op_unlinked(call_op_id);

            caller_slots_used += sub_slots_used;
            *inlined_counts.entry(callee_name).or_insert(0) += 1;
            total_inlined += 1;
        }

        // Update caller's `mgp.context.create` `num_slots` and `model_ctx_op`.
        if let Some(ctx_op) = module.ops.get_mut(parent_ctx_op_id) {
            ctx_op.attributes.insert(
                "num_slots".to_string(),
                Attribute::Int(caller_slots_used as i64),
            );
        }
        model_ctx_op.insert(caller_name.clone(), (parent_ctx_op_id, caller_slots_used));
    }

    // Erase inlined subgraph models that have no remaining uses.
    for (name, count) in inlined_counts {
        if count == 0 {
            continue;
        }
        let Some(&model_id) = models_by_name.get(&name) else {
            continue;
        };
        let is_subgraph = matches!(
            module
                .ops
                .get(model_id)
                .and_then(|op| op.attributes.get("is_subgraph")),
            Some(Attribute::Bool(true))
        );
        if is_subgraph {
            module.remove_op(root_blk, model_id);
        }
    }

    Ok(total_inlined)
}

// ---------------------------------------------------------------------------
// 3. Overhanging Chain Fixup
// ---------------------------------------------------------------------------

/// Scans each block in `module` for non-terminator operations with unused
/// results (collecting at most one unused result per operation since a failing
/// operation sets all its results to error) and joins all hanging chains into
/// the block terminator's chain dependency via `grt.chain.create`.
///
/// Returns the total number of hanging operation results wired into block
/// terminators.
pub fn fixup_overhang_chains(module: &mut Module) -> usize {
    let use_counts = module.compute_use_counts();
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut fixed_count = 0usize;

    for block_id in block_ids {
        let Some(block) = module.blocks.get(block_id) else {
            continue;
        };
        if block.ops.len() < 2 {
            continue;
        }
        let ops = block.ops.clone();
        let Some(&term_id) = ops.last() else {
            continue;
        };
        let is_term = module
            .ops
            .get(term_id)
            .is_some_and(|op| is_terminator_name(&op.name.full_name()));
        if !is_term {
            continue;
        }

        let mut hanging_chains: SmallVec<[ValueId; 4]> = SmallVec::new();
        for &op_id in &ops[..ops.len() - 1] {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            for &res in &op.results {
                if use_counts.get(&res).copied().unwrap_or(0) == 0 {
                    hanging_chains.push(res);
                    break;
                }
            }
        }

        if hanging_chains.is_empty() {
            continue;
        }
        fixed_count += hanging_chains.len();

        // Pop terminator, insert `grt.chain.create`, and attach to terminator.
        if let Some(blk_mut) = module.blocks.get_mut(block_id) {
            blk_mut.ops.pop();
        }
        if let Some(&first_term_opd) = module.ops[term_id].operands.first() {
            hanging_chains.insert(0, first_term_opd);
        }
        let chain_op_id = module.add_operation(
            block_id,
            make_op_name("grt.chain.create"),
            hanging_chains,
            vec![(None, Type::Opaque("!grt.chain".to_string()))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let merged_chain = module.ops[chain_op_id].results[0];
        if module.ops[term_id].operands.is_empty() {
            module.ops[term_id].operands.push(merged_chain);
        } else {
            module.ops[term_id].operands[0] = merged_chain;
        }
        if let Some(blk_mut) = module.blocks.get_mut(block_id) {
            blk_mut.ops.push(term_id);
        }
    }

    fixed_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixup_overhang_chains_one_per_op() {
        let mut module = Module::new(Some("overhang".to_string()));
        let blk = module.create_block(Some("entry".to_string()));
        let c0 = module.add_block_argument(blk, None, Type::Opaque("!grt.chain".to_string()));

        // Op with two unused results: only one should be added to hanging chains.
        let _op1 = module.add_operation(
            blk,
            make_op_name("mgp.custom_async"),
            smallvec![c0],
            vec![
                (None, Type::Opaque("!grt.chain".to_string())),
                (None, Type::Opaque("!mgp.buffer".to_string())),
            ],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let term = module.add_operation(
            blk,
            make_op_name("mef.output"),
            smallvec![c0],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let reg = module.create_region(vec![blk]);
        let root_blk = module.body_block();
        module.add_operation(
            root_blk,
            make_op_name("func.func"),
            SmallVec::new(),
            Vec::new(),
            IndexMap::new(),
            vec![reg],
            false,
            Location::Unknown,
        );

        let fixed = fixup_overhang_chains(&mut module);
        assert_eq!(fixed, 1);
        assert_ne!(module.ops[term].operands[0], c0);
    }
}
