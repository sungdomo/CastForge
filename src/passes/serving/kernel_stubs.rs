//! Device-graph build stubs, model subgraph inlining, and rowwise legalization.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, TensorType, Type, Value,
    ValueDef, ValueId,
};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

// ============================================================================
// Device-Graph Recording Stub Generator & Builder Pool Alloc
// Lowering
// ============================================================================

#[derive(Debug, Clone)]
pub(super) struct RecordingStubInfo {
    stub_name: String,
    context_arg_index: usize,
}

pub(super) fn is_producer_dispatch_op(op: &Operation) -> bool {
    matches!(
        op.name.full_name().as_str(),
        "mogg.call_opaque" | "mogg.call.foreach" | "mogg.call.foreach_tile" | "mogg.call.view_copy"
    )
}

pub(super) fn create_recording_stub_function(
    module: &mut Module,
    orig_func_id: OpId,
    orig_sym_name: &str,
) -> Result<RecordingStubInfo, String> {
    let Some(&orig_reg) = module.ops[orig_func_id].regions.first() else {
        return Err(format!("kernel stub '{orig_sym_name}' has no region"));
    };
    let Some(&orig_blk) = module.regions[orig_reg].blocks.first() else {
        return Err(format!("kernel stub '{orig_sym_name}' has no entry block"));
    };
    let orig_args = module.blocks[orig_blk].arguments.clone();
    let orig_ops = module.blocks[orig_blk].ops.clone();

    let mut producer_op_opt: Option<OpId> = None;
    for &op_id in &orig_ops {
        if is_producer_dispatch_op(&module.ops[op_id]) {
            producer_op_opt = Some(op_id);
            break;
        }
    }
    let Some(producer_op) = producer_op_opt else {
        return Err(format!(
            "device-graph recording stub: '{orig_sym_name}' has no producer dispatch"
        ));
    };

    // Determine which block argument of `orig_blk` acts as the launch context.
    let mut context_arg_index: Option<usize> = None;
    if let Some(Attribute::Int(idx)) = module.ops[producer_op].attributes.get("context_arg_index")
        && *idx >= 0
        && (*idx as usize) < orig_args.len()
    {
        context_arg_index = Some(*idx as usize);
    } else {
        for &opd in module.ops[producer_op].operands.iter().rev() {
            if let Some(pos) = orig_args.iter().position(|&a| a == opd) {
                context_arg_index = Some(pos);
                break;
            }
        }
    }
    let Some(ctx_arg_idx) = context_arg_index else {
        return Err(format!(
            "device-graph recording stub: launch context in '{orig_sym_name}' is not a block arg"
        ));
    };

    let use_build_graph = module.ops[producer_op].name.full_name() == "mogg.call_opaque"
        && matches!(
            module.ops[producer_op].attributes.get("has_build_graph"),
            Some(Attribute::Bool(true))
        );

    let loc = module.ops[orig_func_id].location.clone();
    let new_blk = module.blocks.insert_with_key(|id| crate::ir::Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let mut val_map: IndexMap<ValueId, ValueId> = IndexMap::new();
    for (arg_i, &old_arg) in orig_args.iter().enumerate() {
        let arg_ty = module.values[old_arg].ty.clone();
        let new_arg = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: arg_ty,
            def: ValueDef::BlockArgument {
                block: new_blk,
                index: arg_i as u32,
            },
        });
        module.blocks[new_blk].arguments.push(new_arg);
        val_map.insert(old_arg, new_arg);
    }

    let launch_arg = module.blocks[new_blk].arguments[ctx_arg_idx];

    // If not `use_build_graph`, emit `mgp.device_graph.recording_context` at block start.
    let recording_ctx_val = if !use_build_graph {
        let rec_ty = module.values[launch_arg].ty.clone();
        let rec_op = module.ops.insert_with_key(|id| Operation {
            id,
            name: OpName::new("mgp", "device_graph.recording_context"),
            operands: smallvec::smallvec![launch_arg],
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: loc.clone(),
        });
        let rec_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: rec_ty,
            def: ValueDef::OpResult {
                op: rec_op,
                index: 0,
            },
        });
        module.ops[rec_op].results.push(rec_val);
        module.blocks[new_blk].ops.push(rec_op);
        Some(rec_val)
    } else {
        None
    };

    for &op_id in &orig_ops {
        let full_name = module.ops[op_id].name.full_name();
        if use_build_graph && full_name == "mogg.with_trace" {
            continue;
        }
        if op_id == producer_op {
            let mut mapped_opds: SmallVec<[ValueId; 4]> = module.ops[op_id]
                .operands
                .iter()
                .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                .collect();

            if use_build_graph {
                // Stamp any `mogg.tuple.create` feeding the build_graph dispatch with
                // `variadic_abi = "stable"` when marked `is_stable_variadic = true`.
                for &in_val in &mapped_opds {
                    if let ValueDef::OpResult { op: def_op, .. } = module.values[in_val].def
                        && module.ops[def_op].name.full_name() == "mogg.tuple.create"
                        && matches!(
                            module.ops[def_op].attributes.get("is_stable_variadic"),
                            Some(Attribute::Bool(true))
                        )
                    {
                        module.ops[def_op].attributes.insert(
                            "variadic_abi".to_string(),
                            Attribute::String("stable".to_string()),
                        );
                    }
                }
                let mut bg_attrs = module.ops[op_id].attributes.clone();
                bg_attrs.shift_remove("has_build_graph");
                if !bg_attrs.contains_key("has_epilogue_fusion") {
                    bg_attrs.insert("has_epilogue_fusion".to_string(), Attribute::Bool(false));
                }
                let bg_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mogg", "call.build_graph"),
                    operands: mapped_opds,
                    results: SmallVec::new(),
                    attributes: bg_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                module.blocks[new_blk].ops.push(bg_op);
            } else {
                if let Some(rec_v) = recording_ctx_val {
                    for opd in &mut mapped_opds {
                        if *opd == launch_arg {
                            *opd = rec_v;
                        }
                    }
                }
                let orig_name = module.ops[op_id].name.clone();
                let orig_attrs = module.ops[op_id].attributes.clone();
                let orig_res = module.ops[op_id].results.clone();
                let cloned_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: orig_name,
                    operands: mapped_opds,
                    results: SmallVec::new(),
                    attributes: orig_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                for (r_idx, &old_r) in orig_res.iter().enumerate() {
                    let r_ty = module.values[old_r].ty.clone();
                    let new_r = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: r_ty,
                        def: ValueDef::OpResult {
                            op: cloned_op,
                            index: r_idx as u32,
                        },
                    });
                    module.ops[cloned_op].results.push(new_r);
                    val_map.insert(old_r, new_r);
                }
                module.blocks[new_blk].ops.push(cloned_op);
            }
            continue;
        }

        if full_name == "func.return" || full_name == "mogg.output" {
            let ret_name = module.ops[op_id].name.clone();
            let ret_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: ret_name,
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            module.blocks[new_blk].ops.push(ret_op);
            continue;
        }

        let mapped_opds: SmallVec<[ValueId; 4]> = module.ops[op_id]
            .operands
            .iter()
            .map(|&v| val_map.get(&v).copied().unwrap_or(v))
            .collect();
        let orig_name = module.ops[op_id].name.clone();
        let orig_attrs = module.ops[op_id].attributes.clone();
        let orig_res = module.ops[op_id].results.clone();
        let cloned_op = module.ops.insert_with_key(|id| Operation {
            id,
            name: orig_name,
            operands: mapped_opds,
            results: SmallVec::new(),
            attributes: orig_attrs,
            regions: Vec::new(),
            custom_syntax: false,
            location: loc.clone(),
        });
        for (r_idx, &old_r) in orig_res.iter().enumerate() {
            let r_ty = module.values[old_r].ty.clone();
            let new_r = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: r_ty,
                def: ValueDef::OpResult {
                    op: cloned_op,
                    index: r_idx as u32,
                },
            });
            module.ops[cloned_op].results.push(new_r);
            val_map.insert(old_r, new_r);
        }
        module.blocks[new_blk].ops.push(cloned_op);
    }

    let new_reg = module.regions.insert_with_key(|id| crate::ir::Region {
        id,
        blocks: vec![new_blk],
    });
    let stub_name = format!("{orig_sym_name}_build_graph");
    let mut func_attrs = module.ops[orig_func_id].attributes.clone();
    func_attrs.insert("sym_name".to_string(), Attribute::String(stub_name.clone()));
    let new_func_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("func", "func"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: func_attrs,
        regions: vec![new_reg],
        custom_syntax: false,
        location: loc,
    });

    // Insert the generated stub right after `orig_func_id` in its parent block.
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for b_id in block_ids {
        if let Some(pos) = module.blocks[b_id]
            .ops
            .iter()
            .position(|&id| id == orig_func_id)
        {
            module.blocks[b_id].ops.insert(pos + 1, new_func_id);
            break;
        }
    }

    Ok(RecordingStubInfo {
        stub_name,
        context_arg_index: ctx_arg_idx,
    })
}

/// Generates `@<callee>_build_graph` recording sibling stubs for `mogg.call`
/// sites inside device-graph regions (or functions marked `is_device_graph = true`),
/// retargets the calls to pass the enclosing graph builder in the launch-context
/// slot, and rewrites `mgp.buffer.alloc` inside those regions into
/// `mgp.device_graph.alloc` bound to the enclosing graph builder.
pub fn generate_device_graph_build_stubs(module: &mut Module) -> Result<usize, String> {
    let mut sym_to_func: IndexMap<String, OpId> = IndexMap::new();
    for f_id in module.functions() {
        if let Some(Attribute::String(name)) = module.ops[f_id].attributes.get("sym_name") {
            sym_to_func.insert(name.clone(), f_id);
        }
    }

    // Collect `(block_id, builder_value)` pairs for all device-graph blocks.
    let mut graph_blocks: Vec<(BlockId, ValueId)> = Vec::new();
    for f_id in module.functions() {
        let is_dg_func = matches!(
            module.ops[f_id].attributes.get("is_device_graph"),
            Some(Attribute::Bool(true))
        );
        for &reg_id in &module.ops[f_id].regions {
            for &blk_id in &module.regions[reg_id].blocks {
                if is_dg_func && let Some(&b_arg) = module.blocks[blk_id].arguments.first() {
                    graph_blocks.push((blk_id, b_arg));
                }
                for &op_id in &module.blocks[blk_id].ops {
                    let op_name = module.ops[op_id].name.full_name();
                    if op_name == "mgp.device_graph.collective"
                        || op_name == "mgp.device_graph.staging"
                        || op_name == "mgp.device_graph.dependencies"
                    {
                        for &inner_reg in &module.ops[op_id].regions {
                            for &inner_blk in &module.regions[inner_reg].blocks {
                                if let Some(&b_arg) = module.blocks[inner_blk].arguments.first() {
                                    graph_blocks.push((inner_blk, b_arg));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut callee_to_stub: IndexMap<String, RecordingStubInfo> = IndexMap::new();
    let mut rewritten_count = 0usize;

    for (blk_id, builder_val) in graph_blocks {
        let ops = module.blocks[blk_id].ops.clone();
        for op_id in ops {
            let full_name = module.ops[op_id].name.full_name();
            if full_name == "mogg.call" {
                let callee = module.ops[op_id]
                    .attributes
                    .get("callee")
                    .and_then(Attribute::as_str)
                    .map(|s| s.trim_start_matches('@').to_string())
                    .ok_or_else(|| "mogg.call missing 'callee' attribute".to_string())?;

                let stub_info = if let Some(existing) = callee_to_stub.get(&callee) {
                    existing.clone()
                } else {
                    let orig_f = *sym_to_func.get(&callee).ok_or_else(|| {
                        format!("recording stub call: '{callee}' does not name a kernel stub")
                    })?;
                    let info = create_recording_stub_function(module, orig_f, &callee)?;
                    callee_to_stub.insert(callee.clone(), info.clone());
                    info
                };

                if stub_info.context_arg_index >= module.ops[op_id].operands.len() {
                    return Err(format!(
                        "recording stub call: expected operand {} to be the launch context",
                        stub_info.context_arg_index
                    ));
                }
                module.ops[op_id].operands[stub_info.context_arg_index] = builder_val;
                module.ops[op_id]
                    .attributes
                    .insert("callee".to_string(), Attribute::String(stub_info.stub_name));
                rewritten_count += 1;
            } else if full_name == "mgp.buffer.alloc" {
                module.ops[op_id].name = OpName::new("mgp", "device_graph.alloc");
                module.ops[op_id].operands.push(builder_val);
                rewritten_count += 1;
            }
        }
    }

    Ok(rewritten_count)
}

// ============================================================================
// Bottom-Up Model Call-Graph Slot Globalization & Subgraph
// Inlining
// ============================================================================

/// Offsets the runtime slot attribute (`b_buffer_slot` on `mgp.buffer.cache`,
/// or `a_runtime_slot` on `mgp.runtime.create`, `mgp.module.load`, and
/// `mgp.execute.generic`) of `op` by `slot_offset`.
pub fn offset_runtime_slots_in_op(op: &mut Operation, slot_offset: i64) {
    let attr_key = match op.name.full_name().as_str() {
        "mgp.buffer.cache" => Some("b_buffer_slot"),
        "mgp.runtime.create" | "mgp.module.load" | "mgp.execute.generic" => Some("a_runtime_slot"),
        _ => None,
    };
    if let Some(key) = attr_key
        && let Some(Attribute::Int(old_slot)) = op.attributes.get(key).cloned()
    {
        op.attributes
            .insert(key.to_string(), Attribute::Int(old_slot + slot_offset));
    }
}

pub(super) fn find_model_context_create(module: &Module, func_id: OpId) -> Option<(OpId, i64)> {
    for &reg_id in &module.ops[func_id].regions {
        for &blk_id in &module.regions[reg_id].blocks {
            for &op_id in &module.blocks[blk_id].ops {
                if module.ops[op_id].name.full_name() == "mgp.context.create" {
                    let slots = match module.ops[op_id].attributes.get("num_slots") {
                        Some(Attribute::Int(n)) => *n,
                        _ => 0,
                    };
                    return Some((op_id, slots));
                }
            }
        }
    }
    None
}

pub(super) fn topo_sort_model_call_graph(
    models: &[String],
    edges: &IndexMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    // 0 = unvisited, 1 = visiting (on stack), 2 = visited
    let mut state: IndexMap<String, u8> = IndexMap::new();
    let mut post_order: Vec<String> = Vec::new();
    let mut stack_path: Vec<String> = Vec::new();

    fn dfs(
        node: &str,
        edges: &IndexMap<String, Vec<String>>,
        state: &mut IndexMap<String, u8>,
        post_order: &mut Vec<String>,
        stack_path: &mut Vec<String>,
    ) -> Result<(), String> {
        state.insert(node.to_string(), 1);
        stack_path.push(node.to_string());
        if let Some(callees) = edges.get(node) {
            for callee in callees {
                match state.get(callee).copied().unwrap_or(0) {
                    1 => {
                        return Err(format!(
                            "cycle detected in model call graph, cannot inline: {} -> {}",
                            stack_path.join(" -> "),
                            callee
                        ));
                    }
                    0 => dfs(callee, edges, state, post_order, stack_path)?,
                    _ => {}
                }
            }
        }
        stack_path.pop();
        state.insert(node.to_string(), 2);
        post_order.push(node.to_string());
        Ok(())
    }

    for m in models {
        if state.get(m).copied().unwrap_or(0) == 0 {
            dfs(m, edges, &mut state, &mut post_order, &mut stack_path)?;
        }
    }
    Ok(post_order)
}

/// Performs bottom-up model call-graph analysis across `mgp.call` sites in
/// `module`, rejecting call-graph cycles and either:
/// - (`inline_subgraphs == false`): globalizing subgraph slot indices by
///   emitting an `index.constant` slot offset operand for each `mgp.call` and
///   marking `is_globalized_index = true`; or
/// - (`inline_subgraphs == true`): inlining subgraph bodies at each `mgp.call`
///   site with runtime slot attributes offset by the accumulated parent slot
///   usage, prefixing placeholder `mgp.buffer.constant_external` weight names
///   via `mgp.string.create` + `mgp.string.concat`, updating the parent
///   `mgp.context.create`'s `num_slots`, and erasing unused subgraph models.
pub fn globalize_and_inline_model_subgraphs(
    module: &mut Module,
    inline_subgraphs: bool,
) -> Result<usize, String> {
    let func_ids = module.functions();
    if func_ids.len() <= 1 {
        return Ok(0);
    }

    let mut sym_to_func: IndexMap<String, OpId> = IndexMap::new();
    let mut model_names: Vec<String> = Vec::new();
    for &f_id in &func_ids {
        if let Some(Attribute::String(sym)) = module.ops[f_id].attributes.get("sym_name") {
            sym_to_func.insert(sym.clone(), f_id);
            model_names.push(sym.clone());
        }
    }

    let mut call_edges: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut has_any_mgp_call = false;
    for (sym, &f_id) in &sym_to_func {
        let mut callees: Vec<String> = Vec::new();
        for &reg_id in &module.ops[f_id].regions {
            for &blk_id in &module.regions[reg_id].blocks {
                for &op_id in &module.blocks[blk_id].ops {
                    if module.ops[op_id].name.full_name() == "mgp.call" {
                        has_any_mgp_call = true;
                        let callee = module.ops[op_id]
                            .attributes
                            .get("callee")
                            .and_then(Attribute::as_str)
                            .map(|s| s.trim_start_matches('@').to_string())
                            .ok_or_else(|| "mgp.call missing 'callee' attribute".to_string())?;
                        if !sym_to_func.contains_key(&callee) {
                            return Err(format!("failed to find symbol @{callee}"));
                        }
                        if !callees.contains(&callee) {
                            callees.push(callee);
                        }
                    }
                }
            }
        }
        call_edges.insert(sym.clone(), callees);
    }

    if !has_any_mgp_call {
        return Ok(0);
    }

    // Bottom-up order (callees before callers) and cycle check.
    let bottom_up_models = topo_sort_model_call_graph(&model_names, &call_edges)?;

    // Verify every model in the call graph has an `mgp.context.create` op.
    let mut model_ctx_info: IndexMap<String, (OpId, i64)> = IndexMap::new();
    for sym in &bottom_up_models {
        let f_id = sym_to_func[sym];
        let Some(info) = find_model_context_create(module, f_id) else {
            return Err(format!("unable to find context in model @{sym}"));
        };
        model_ctx_info.insert(sym.clone(), info);
    }

    let mut processed_calls = 0usize;
    let mut inlined_counts: IndexMap<String, usize> = IndexMap::new();

    for caller_sym in &bottom_up_models {
        let caller_f = sym_to_func[caller_sym];
        let (caller_ctx_op, mut model_slot_usage) = model_ctx_info[caller_sym];
        let caller_ctx_val = module.ops[caller_ctx_op].results.first().copied();
        let Some(&caller_reg) = module.ops[caller_f].regions.first() else {
            continue;
        };
        let Some(&caller_blk) = module.regions[caller_reg].blocks.first() else {
            continue;
        };

        let call_ops: Vec<OpId> = module.blocks[caller_blk]
            .ops
            .iter()
            .copied()
            .filter(|&id| module.ops[id].name.full_name() == "mgp.call")
            .collect();
        if call_ops.is_empty() {
            continue;
        }

        for call_op in call_ops {
            let callee_sym = module.ops[call_op]
                .attributes
                .get("callee")
                .and_then(Attribute::as_str)
                .unwrap_or("")
                .trim_start_matches('@')
                .to_string();
            let (_, callee_slots_used) = model_ctx_info[&callee_sym];
            let slot_offset = model_slot_usage;
            let call_loc = module.ops[call_op].location.clone();

            if !inline_subgraphs {
                let const_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("index", "constant"),
                    operands: SmallVec::new(),
                    results: SmallVec::new(),
                    attributes: {
                        let mut attrs = IndexMap::new();
                        attrs.insert("value".to_string(), Attribute::Int(slot_offset));
                        attrs
                    },
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: call_loc,
                });
                let idx_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: Type::Tensor(TensorType::new(vec![Dim::Static(1)], DType::I64)),
                    def: ValueDef::OpResult {
                        op: const_op,
                        index: 0,
                    },
                });
                module.ops[const_op].results.push(idx_val);
                if let Some(pos) = module.blocks[caller_blk]
                    .ops
                    .iter()
                    .position(|&id| id == call_op)
                {
                    module.blocks[caller_blk].ops.insert(pos, const_op);
                }
                module.ops[call_op].operands.push(idx_val);
                module.ops[call_op]
                    .attributes
                    .insert("is_globalized_index".to_string(), Attribute::Bool(true));
                model_slot_usage += callee_slots_used;
                processed_calls += 1;
                continue;
            }

            // Full subgraph inlining mode:
            let callee_f = sym_to_func[&callee_sym];
            let (callee_ctx_op, _) = model_ctx_info[&callee_sym];
            let callee_reg = module.ops[callee_f].regions[0];
            let callee_blk = module.regions[callee_reg].blocks[0];
            let callee_args = module.blocks[callee_blk].arguments.clone();
            let callee_ops = module.blocks[callee_blk].ops.clone();
            let call_operands = module.ops[call_op].operands.clone();

            let mut val_map: IndexMap<ValueId, ValueId> = IndexMap::new();
            for (&c_arg, &c_opd) in callee_args.iter().zip(call_operands.iter()) {
                val_map.insert(c_arg, c_opd);
            }
            if let (Some(parent_ctx_v), Some(&sub_ctx_v)) =
                (caller_ctx_val, module.ops[callee_ctx_op].results.first())
            {
                val_map.insert(sub_ctx_v, parent_ctx_v);
            }

            let prefix_opt = module.ops[call_op]
                .attributes
                .get("prefix")
                .and_then(Attribute::as_str)
                .map(ToString::to_string);

            let mut inserted_ops: Vec<OpId> = Vec::new();
            let mut prefix_val_opt: Option<ValueId> = None;
            if let Some(ref prefix_str) = prefix_opt {
                let str_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mgp", "string.create"),
                    operands: SmallVec::new(),
                    results: SmallVec::new(),
                    attributes: {
                        let mut attrs = IndexMap::new();
                        attrs.insert("value".to_string(), Attribute::String(prefix_str.clone()));
                        attrs
                    },
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: call_loc.clone(),
                });
                let p_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: Type::Tensor(TensorType::new(vec![Dim::Static(1)], DType::I64)),
                    def: ValueDef::OpResult {
                        op: str_op,
                        index: 0,
                    },
                });
                module.ops[str_op].results.push(p_val);
                inserted_ops.push(str_op);
                prefix_val_opt = Some(p_val);
            }

            let mut term_mapped_results: Vec<ValueId> = Vec::new();
            for &sub_op in &callee_ops {
                if sub_op == callee_ctx_op {
                    continue;
                }
                let sub_name = module.ops[sub_op].name.full_name();
                if sub_name == "func.return" || sub_name == "mef.output" {
                    term_mapped_results = module.ops[sub_op]
                        .operands
                        .iter()
                        .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                        .collect();
                    continue;
                }

                let mut mapped_opds: SmallVec<[ValueId; 4]> = module.ops[sub_op]
                    .operands
                    .iter()
                    .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                    .collect();

                if sub_name == "mgp.buffer.constant_external"
                    && matches!(
                        module.ops[sub_op].attributes.get("e_is_placeholder"),
                        Some(Attribute::Bool(true))
                    )
                    && let Some(p_val) = prefix_val_opt
                    && let Some(name_opd) = mapped_opds.first_mut()
                {
                    let raw_name = *name_opd;
                    let raw_ty = module.values[raw_name].ty.clone();
                    let concat_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mgp", "string.concat"),
                        operands: smallvec::smallvec![p_val, raw_name],
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: call_loc.clone(),
                    });
                    let concat_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: raw_ty,
                        def: ValueDef::OpResult {
                            op: concat_op,
                            index: 0,
                        },
                    });
                    module.ops[concat_op].results.push(concat_val);
                    inserted_ops.push(concat_op);
                    *name_opd = concat_val;
                }

                let orig_name = module.ops[sub_op].name.clone();
                let orig_attrs = module.ops[sub_op].attributes.clone();
                let orig_res = module.ops[sub_op].results.clone();
                let cloned_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: orig_name,
                    operands: mapped_opds,
                    results: SmallVec::new(),
                    attributes: orig_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: call_loc.clone(),
                });
                offset_runtime_slots_in_op(&mut module.ops[cloned_op], slot_offset);
                for (r_idx, &old_r) in orig_res.iter().enumerate() {
                    let r_ty = module.values[old_r].ty.clone();
                    let new_r = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: r_ty,
                        def: ValueDef::OpResult {
                            op: cloned_op,
                            index: r_idx as u32,
                        },
                    });
                    module.ops[cloned_op].results.push(new_r);
                    val_map.insert(old_r, new_r);
                }
                inserted_ops.push(cloned_op);
            }

            let call_results = module.ops[call_op].results.clone();
            let mut result_replacements: IndexMap<ValueId, ValueId> = IndexMap::new();
            for (&old_res, &new_res) in call_results.iter().zip(term_mapped_results.iter()) {
                result_replacements.insert(old_res, new_res);
            }

            if let Some(pos) = module.blocks[caller_blk]
                .ops
                .iter()
                .position(|&id| id == call_op)
            {
                module.blocks[caller_blk]
                    .ops
                    .splice(pos..=pos, inserted_ops);
            }
            for &op_id in &module.blocks[caller_blk].ops {
                for opd in &mut module.ops[op_id].operands {
                    if let Some(&rep) = result_replacements.get(opd) {
                        *opd = rep;
                    }
                }
            }
            module.ops.remove(call_op);

            model_slot_usage += callee_slots_used;
            *inlined_counts.entry(callee_sym).or_insert(0) += 1;
            processed_calls += 1;
        }

        if inline_subgraphs {
            module.ops[caller_ctx_op]
                .attributes
                .insert("num_slots".to_string(), Attribute::Int(model_slot_usage));
            if let Some(entry) = model_ctx_info.get_mut(caller_sym) {
                entry.1 = model_slot_usage;
            }
        }
    }

    if inline_subgraphs {
        for (sym, count) in inlined_counts {
            if count == 0 {
                continue;
            }
            let f_id = sym_to_func[&sym];
            let is_subgraph = matches!(
                module.ops[f_id].attributes.get("is_subgraph"),
                Some(Attribute::Bool(true))
            );
            if !is_subgraph {
                continue;
            }
            let mut still_used = false;
            for &other_f in &module.functions() {
                if other_f == f_id {
                    continue;
                }
                for &r_id in &module.ops[other_f].regions {
                    for &b_id in &module.regions[r_id].blocks {
                        for &op_id in &module.blocks[b_id].ops {
                            if module.ops[op_id].name.full_name() == "mgp.call"
                                && module.ops[op_id]
                                    .attributes
                                    .get("callee")
                                    .and_then(Attribute::as_str)
                                    .is_some_and(|s| s.trim_start_matches('@') == sym)
                            {
                                still_used = true;
                            }
                        }
                    }
                }
            }
            if !still_used {
                let all_blks: Vec<BlockId> = module.blocks.keys().collect();
                for b_id in all_blks {
                    module.blocks[b_id].ops.retain(|&id| id != f_id);
                }
                module.ops.remove(f_id);
            }
        }
    }

    Ok(processed_calls)
}

// ============================================================================
// Rowwise Kernel Legalizer (Primary Input Selection, Monoid
// Binding, Row Store Grouping, and Launch Parameter Synthesis)
// ============================================================================

pub(super) const CHEAP_ROWWISE_ELEMENT_COST: u32 = 1;
pub(super) const EXPENSIVE_ROWWISE_ELEMENT_COST: u32 = 16;

pub(super) fn dtype_bitwidth(ty: &Type) -> u32 {
    match ty {
        Type::Tensor(t) => t.dtype.bit_width() as u32,
        _ => 32,
    }
}

pub(super) fn rowwise_call_elementwise_cost(op: &Operation) -> u32 {
    if matches!(
        op.attributes.get("is_expensive"),
        Some(Attribute::Bool(true))
    ) {
        return EXPENSIVE_ROWWISE_ELEMENT_COST;
    }
    let mo_name = op
        .attributes
        .get("mo_op_name")
        .and_then(Attribute::as_str)
        .unwrap_or("");
    if matches!(
        mo_name,
        "mo.div" | "mo.exp" | "mo.log" | "mo.tanh" | "mo.erf" | "mo.rsqrt" | "mo.sin" | "mo.cos"
    ) {
        EXPENSIVE_ROWWISE_ELEMENT_COST
    } else {
        CHEAP_ROWWISE_ELEMENT_COST
    }
}

/// Legalizes `mogg.call.rowwise` operations across `module` in four stages:
/// 1. Selects the primary input capture from the first `mogg.rowwise.reduce`
///    fold's element-index `mogg.tensor.load`, rewriting matching loads into
///    `mogg.rowwise.primary` and setting `primary` and `is_cached`.
/// 2. Binds monoid reduction metadata (`fields` and widest `accum_bitwidth`).
/// 3. Groups row-indexed `mogg.tensor.store` operations into a trailing
///    `mogg.rowwise.emit` region.
/// 4. Validates that no stray stores remain outside `mogg.rowwise.emit` and
///    computes launch attributes (`num_phases`, `dtype_size`,
///    `computationally_expensive`, `simd_monoid`, and optional `cols`).
pub fn legalize_rowwise_kernels(
    module: &mut Module,
    expensive_cost_threshold: u32,
) -> Result<usize, String> {
    let mut legalized_count = 0usize;
    let all_ops: Vec<OpId> = module.ops.keys().collect();
    let rowwise_ops: Vec<OpId> = all_ops
        .into_iter()
        .filter(|&id| {
            module
                .ops
                .get(id)
                .is_some_and(|op| op.name.full_name() == "mogg.call.rowwise")
        })
        .collect();

    for rw_op in rowwise_ops {
        let Some(&rw_reg) = module.ops[rw_op].regions.first() else {
            return Err("mogg.call.rowwise has no body region".to_string());
        };
        let Some(&rw_blk) = module.regions[rw_reg].blocks.first() else {
            return Err("mogg.call.rowwise has no entry block".to_string());
        };
        let rw_args = module.blocks[rw_blk].arguments.clone();
        if rw_args.len() < 2 {
            return Err(
                "mogg.call.rowwise body must have at least 1 capture arg and 1 row index arg"
                    .to_string(),
            );
        }
        let capture_args = &rw_args[..rw_args.len() - 1];
        let row_arg = rw_args[rw_args.len() - 1];

        let fold_ops: Vec<OpId> = module.blocks[rw_blk]
            .ops
            .iter()
            .copied()
            .filter(|&id| module.ops[id].name.full_name() == "mogg.rowwise.reduce")
            .collect();
        if fold_ops.is_empty() {
            return Err("mogg.call.rowwise has no fold".to_string());
        }

        // Stage 1: Select primary input capture if not already set.
        if !module.ops[rw_op].attributes.contains_key("primary") {
            let first_fold = fold_ops[0];
            let Some(&first_fold_reg) = module.ops[first_fold].regions.first() else {
                return Err("mogg.rowwise.reduce has no body region".to_string());
            };
            let first_fold_blk = module.regions[first_fold_reg].blocks[0];
            let Some(&first_elem_idx) = module.blocks[first_fold_blk].arguments.first() else {
                return Err("mogg.rowwise.reduce body missing element index argument".to_string());
            };

            let mut primary_idx_opt: Option<usize> = None;
            for &f_op in &module.blocks[first_fold_blk].ops {
                if module.ops[f_op].name.full_name() == "mogg.tensor.load"
                    && module.ops[f_op].operands.len() >= 2
                    && module.ops[f_op].operands[1] == first_elem_idx
                {
                    let tensor_opd = module.ops[f_op].operands[0];
                    if let Some(pos) = capture_args.iter().position(|&c| c == tensor_opd) {
                        primary_idx_opt = Some(pos);
                        break;
                    }
                }
            }
            let Some(primary_idx) = primary_idx_opt else {
                return Err(
                    "mogg.call.rowwise has no input its first fold loads at the element index"
                        .to_string(),
                );
            };

            let primary_tensor = capture_args[primary_idx];
            let mut folds_reading_primary = 0usize;
            for &fold_op in &fold_ops {
                let fold_reg = module.ops[fold_op].regions[0];
                let fold_blk = module.regions[fold_reg].blocks[0];
                let Some(&elem_idx) = module.blocks[fold_blk].arguments.first() else {
                    continue;
                };
                let mut matched_loads = Vec::new();
                for &f_op in &module.blocks[fold_blk].ops {
                    if module.ops[f_op].name.full_name() == "mogg.tensor.load"
                        && module.ops[f_op].operands.len() >= 2
                        && module.ops[f_op].operands[0] == primary_tensor
                        && module.ops[f_op].operands[1] == elem_idx
                    {
                        matched_loads.push(f_op);
                    }
                }
                if !matched_loads.is_empty() {
                    folds_reading_primary += 1;
                    for load_op in matched_loads {
                        module.ops[load_op].name = OpName::new("mogg.rowwise", "primary");
                        module.ops[load_op].operands.clear();
                    }
                }
            }

            module.ops[rw_op]
                .attributes
                .insert("primary".to_string(), Attribute::Int(primary_idx as i64));
            module.ops[rw_op].attributes.insert(
                "is_cached".to_string(),
                Attribute::Bool(folds_reading_primary > 1),
            );
        }

        // Stage 2: Bind monoids and compute widest accumulator bitwidth.
        let mut widest_accum_bits = 0u32;
        let mut widest_monoid = String::new();
        let mut widest_state_bits = 0u32;
        let mut total_element_cost = 0u32;

        for &fold_op in &fold_ops {
            let monoid_name = module.ops[fold_op]
                .attributes
                .get("monoid")
                .and_then(Attribute::as_str)
                .ok_or_else(|| "mogg.rowwise.reduce missing 'monoid' attribute".to_string())?
                .to_string();

            let num_res = module.ops[fold_op].results.len();
            if let Some(Attribute::Array(expected_fields)) =
                module.ops[fold_op].attributes.get("monoid_fields").cloned()
            {
                if expected_fields.len() != num_res {
                    return Err(format!(
                        "rowwise.reduce has {num_res} results, but monoid has {} output fields",
                        expected_fields.len()
                    ));
                }
                module.ops[fold_op]
                    .attributes
                    .insert("fields".to_string(), Attribute::Array(expected_fields));
            } else {
                let default_fields: Vec<Attribute> = (0..num_res)
                    .map(|i| Attribute::String(format!("val_{i}")))
                    .collect();
                module.ops[fold_op]
                    .attributes
                    .insert("fields".to_string(), Attribute::Array(default_fields));
            }

            let mut fold_state_bits = 0u32;
            let res_ids = module.ops[fold_op].results.clone();
            for r_id in res_ids {
                let bits = dtype_bitwidth(&module.values[r_id].ty);
                fold_state_bits += bits;
                widest_accum_bits = widest_accum_bits.max(bits);
            }
            if widest_monoid.is_empty() || fold_state_bits > widest_state_bits {
                widest_state_bits = fold_state_bits;
                widest_monoid = monoid_name;
            }

            let fold_reg = module.ops[fold_op].regions[0];
            let fold_blk = module.regions[fold_reg].blocks[0];
            for &f_op in &module.blocks[fold_blk].ops {
                if module.ops[f_op].name.full_name() == "mogg.call.elementwise" {
                    total_element_cost += rowwise_call_elementwise_cost(&module.ops[f_op]);
                }
            }
        }

        if widest_accum_bits > 0 {
            module.ops[rw_op].attributes.insert(
                "accum_bitwidth".to_string(),
                Attribute::Int(i64::from(widest_accum_bits)),
            );
        }

        // Stage 3: Group row-indexed `mogg.tensor.store` ops into a trailing `mogg.rowwise.emit`.
        let mut row_stores: Vec<OpId> = Vec::new();
        let mut stray_store = false;
        for &op_id in &module.blocks[rw_blk].ops {
            if module.ops[op_id].name.full_name() == "mogg.tensor.store" {
                if module.ops[op_id].operands.get(1).copied() == Some(row_arg) {
                    row_stores.push(op_id);
                } else {
                    stray_store = true;
                }
            }
        }
        if stray_store {
            return Err(
                "mogg.call.rowwise stores outside its emit; only reduce-shaped bodies are supported"
                    .to_string(),
            );
        }

        if !row_stores.is_empty() {
            let rw_loc = module.ops[rw_op].location.clone();
            module.blocks[rw_blk]
                .ops
                .retain(|id| !row_stores.contains(id));
            let emit_blk = module.blocks.insert_with_key(|id| crate::ir::Block {
                id,
                label: None,
                arguments: Vec::new(),
                ops: row_stores,
            });
            let emit_reg = module.regions.insert_with_key(|id| crate::ir::Region {
                id,
                blocks: vec![emit_blk],
            });
            let emit_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg.rowwise", "emit"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: vec![emit_reg],
                custom_syntax: false,
                location: rw_loc,
            });
            let insert_pos = module.blocks[rw_blk]
                .ops
                .iter()
                .position(|&id| {
                    let n = module.ops[id].name.full_name();
                    n == "mogg.yield" || n == "func.return"
                })
                .unwrap_or(module.blocks[rw_blk].ops.len());
            module.blocks[rw_blk].ops.insert(insert_pos, emit_op);
        }

        // Stage 4: Set launch parameters.
        module.ops[rw_op]
            .attributes
            .insert("num_phases".to_string(), Attribute::Int(1));
        module.ops[rw_op]
            .attributes
            .insert("dtype_size".to_string(), Attribute::Int(0));
        module.ops[rw_op].attributes.insert(
            "computationally_expensive".to_string(),
            Attribute::Bool(total_element_cost >= expensive_cost_threshold),
        );
        module.ops[rw_op]
            .attributes
            .insert("simd_monoid".to_string(), Attribute::String(widest_monoid));
        if let Some(Attribute::Int(cols)) = module.ops[rw_op].attributes.get("static_cols").cloned()
        {
            module.ops[rw_op]
                .attributes
                .insert("cols".to_string(), Attribute::Int(cols));
        }

        legalized_count += 1;
    }

    Ok(legalized_count)
}

// ============================================================================
// Dynamic File-Backed Zero-Copy L2 Storage Agent, L1 Page
// Index Mapper, and 2-Level Hash-Sharded Atomic Publish Planner
// ============================================================================

/// Computes the registered L1 page indices covering `[raw_addr, raw_addr + mem_size)`
/// for a zero-copy storage transfer, verifying that both `raw_addr` and `mem_size`
/// are aligned to `l1_align_bytes`.
pub fn compute_l1_page_indices(
    raw_addr: usize,
    mem_size: usize,
    l1_align_bytes: usize,
) -> Result<Vec<usize>, String> {
    if l1_align_bytes == 0 {
        return Err("l1_align_bytes must be > 0".to_string());
    }
    if !raw_addr.is_multiple_of(l1_align_bytes) {
        return Err(format!(
            "raw address {raw_addr} is not aligned to page size {l1_align_bytes}"
        ));
    }
    if mem_size == 0 || !mem_size.is_multiple_of(l1_align_bytes) {
        return Err(format!(
            "memory size {mem_size} is not a positive multiple of page size {l1_align_bytes}"
        ));
    }
    let base_page = raw_addr / l1_align_bytes;
    let num_pages = mem_size / l1_align_bytes;
    Ok((0..num_pages).map(|i| base_page + i).collect())
}

/// Formats a deterministic binary storage filename
/// (`{model}_{rank:08x}_{group:x}_{hash}[@{salt}].bin` with `/` in `model_name`
/// escaped as `--`).
pub fn format_dynamic_storage_filename(
    model_name: &str,
    kv_rank: u32,
    group_id: u32,
    chunk_hash_hex: &str,
    cache_salt: &str,
) -> Result<String, String> {
    if model_name.is_empty() {
        return Err("model_name must be non-empty".to_string());
    }
    if chunk_hash_hex.len() < 4 || !chunk_hash_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("chunk_hash_hex must be at least 4 hex digits".to_string());
    }
    let escaped_model = model_name.replace('/', "--");
    let suffix = if cache_salt.is_empty() {
        String::new()
    } else {
        format!("@{cache_salt}")
    };
    Ok(format!(
        "{escaped_model}_{kv_rank:08x}_{group_id:x}_{chunk_hash_hex}{suffix}.bin"
    ))
}

/// Computes the relative storage path (and optional 2-level `<h[0..2]>/<h[2..4]>`
/// shard subdirectory) for a cache object key. The 2-level shard bucket depends
/// solely on `chunk_hash_hex`, keeping salted variants colocated in the same
/// shard directory.
pub fn object_key_to_sharded_relpath(
    model_name: &str,
    kv_rank: u32,
    group_id: u32,
    chunk_hash_hex: &str,
    cache_salt: &str,
    shard_dirs: bool,
) -> Result<(Option<String>, String), String> {
    let filename =
        format_dynamic_storage_filename(model_name, kv_rank, group_id, chunk_hash_hex, cache_salt)?;
    if !shard_dirs {
        return Ok((None, filename));
    }
    let l1 = &chunk_hash_hex[0..2];
    let l2 = &chunk_hash_hex[2..4];
    let subdir = format!("{l1}/{l2}");
    let relpath = format!("{subdir}/{filename}");
    Ok((Some(subdir), relpath))
}

/// Builds per-page `(file_offset, page_size, fd)` transfer descriptors for a
/// single file of `file_size` bytes.
pub fn build_single_file_xfer_descs(
    fd: i32,
    file_size: usize,
    page_size: usize,
) -> Result<Vec<(usize, usize, i32)>, String> {
    if page_size == 0 || file_size == 0 || !file_size.is_multiple_of(page_size) {
        return Err(format!(
            "file_size {file_size} must be a positive multiple of page_size {page_size}"
        ));
    }
    let num_pages = file_size / page_size;
    Ok((0..num_pages)
        .map(|page_idx| (page_idx * page_size, page_size, fd))
        .collect())
}

/// Returns `true` if `filename` matches the `<final_name>.tmp.<token>` pattern
/// created during atomic write-to-temp-then-rename staging.
#[must_use]
pub fn is_orphan_temp_storage_file(filename: &str) -> bool {
    if let Some((prefix, token)) = filename.rsplit_once(".tmp.") {
        !prefix.is_empty() && !token.is_empty()
    } else {
        false
    }
}

/// Planned atomic file write operation for a dynamic zero-copy L2 storage agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicFileStorePlan {
    /// Whether the target 2-level shard directory needs to be created first.
    pub needs_mkdir: Option<String>,
    /// Temporary staging file path (`<final_path>.tmp.<nonce>`).
    pub temp_path: String,
    /// Final published file path renamed atomically after DMA completion.
    pub final_path: String,
    /// Total file size in bytes (`mem_indices.len() * l1_align_bytes`).
    pub file_size: usize,
}

/// Dynamic file-backed L2 storage agent planner managing L1 memory page
/// registration, 2-level directory sharding cache, and atomic temp-to-final
/// store publication.
#[derive(Debug, Clone)]
pub struct DynamicFileStorageAgent {
    base_dir: String,
    memory_type: String,
    l1_align_bytes: usize,
    shard_dirs: bool,
    use_direct_io: bool,
    l1_xfer_descs: Vec<(usize, usize, u32)>,
    created_subdirs: std::collections::BTreeSet<String>,
}

impl DynamicFileStorageAgent {
    /// Initializes a dynamic file-backed storage agent for `device` (`"cpu"` ->
    /// `"DRAM"`, accelerator -> `"VRAM"`) and precomputes L1 page descriptors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_dir: &str,
        device: &str,
        buffer_ptr: usize,
        buffer_size: usize,
        l1_align_bytes: usize,
        device_id: u32,
        shard_dirs: bool,
        use_direct_io: bool,
    ) -> Result<Self, String> {
        if base_dir.is_empty() {
            return Err("base_dir must be non-empty".to_string());
        }
        if l1_align_bytes == 0 || buffer_size == 0 || !buffer_size.is_multiple_of(l1_align_bytes) {
            return Err("buffer_size must be a positive multiple of l1_align_bytes".to_string());
        }
        let memory_type = if device.eq_ignore_ascii_case("cpu") {
            "DRAM".to_string()
        } else {
            "VRAM".to_string()
        };
        let num_pages = buffer_size / l1_align_bytes;
        let l1_xfer_descs = (0..num_pages)
            .map(|i| (buffer_ptr + i * l1_align_bytes, l1_align_bytes, device_id))
            .collect();
        Ok(Self {
            base_dir: base_dir.trim_end_matches('/').to_string(),
            memory_type,
            l1_align_bytes,
            shard_dirs,
            use_direct_io,
            l1_xfer_descs,
            created_subdirs: std::collections::BTreeSet::new(),
        })
    }

    /// Returns the registered memory type (`"DRAM"` or `"VRAM"`).
    #[must_use]
    pub fn memory_type(&self) -> &str {
        &self.memory_type
    }

    /// Returns whether direct I/O (`O_DIRECT`) is enabled for file transfers.
    #[must_use]
    pub fn use_direct_io(&self) -> bool {
        self.use_direct_io
    }

    /// Returns the number of registered L1 memory pages.
    #[must_use]
    pub fn registered_page_count(&self) -> usize {
        self.l1_xfer_descs.len()
    }

    /// Plans an atomic write-to-temp-then-rename store operation, caching
    /// created 2-level shard subdirectories so each bucket is created at most
    /// once on the store hot path.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_atomic_store(
        &mut self,
        mem_indices: &[usize],
        model_name: &str,
        kv_rank: u32,
        group_id: u32,
        chunk_hash_hex: &str,
        cache_salt: &str,
        nonce_hex: &str,
    ) -> Result<AtomicFileStorePlan, String> {
        if mem_indices.is_empty() {
            return Err("mem_indices must be non-empty".to_string());
        }
        if nonce_hex.is_empty() {
            return Err("nonce_hex must be non-empty".to_string());
        }
        let (subdir_opt, relpath) = object_key_to_sharded_relpath(
            model_name,
            kv_rank,
            group_id,
            chunk_hash_hex,
            cache_salt,
            self.shard_dirs,
        )?;
        let final_path = format!("{}/{relpath}", self.base_dir);
        let temp_path = format!("{final_path}.tmp.{nonce_hex}");
        let needs_mkdir = if let Some(rel_sub) = subdir_opt {
            let full_sub = format!("{}/{rel_sub}", self.base_dir);
            if self.created_subdirs.insert(full_sub.clone()) {
                Some(full_sub)
            } else {
                None
            }
        } else {
            None
        };
        Ok(AtomicFileStorePlan {
            needs_mkdir,
            temp_path,
            final_path,
            file_size: mem_indices.len() * self.l1_align_bytes,
        })
    }
}

/// Wraps the body of top-level functions in `castforge.try` / except regions based on their ABI:
/// - When `"abi" = "runtime"` (`"mogg"` or `"runtime"`): moves every body op except existing
///   `castforge.try` and the final terminator into a `castforge.try` op's `try` region ending with
///   `castforge.yield`, populates the `except` region with an `!error` block argument and a
///   `castforge.raise(%err)` op, and wires `castforge.try` results into the function terminator.
/// - When `"abi" = "graph"` (`"mgp"` or `"graph"`) and `"throws" = true`: moves the body ops into
///   the `try` region, and populates the `except` region with one `castforge.async_error(%err)` per
///   try output followed by `castforge.yield` so async error tokens and placeholder outputs flow to
///   the terminator.
///
/// Returns the number of functions wrapped in a `castforge.try` op.
pub fn add_try_except_blocks(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, OpName, Region};

    let mut wrapped = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let abi = module.ops[func_id]
            .attributes
            .get("abi")
            .and_then(|a| a.as_str())
            .unwrap_or("")
            .to_string();
        let is_runtime_abi = abi == "runtime" || abi == "mogg";
        let is_graph_abi = abi == "graph" || abi == "mgp";
        let throws = module.ops[func_id]
            .attributes
            .get("throws")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);

        if !is_runtime_abi && !(is_graph_abi && throws) {
            continue;
        }
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let body_reg = module.ops[func_id].regions[0];
        if module.regions[body_reg].blocks.is_empty() {
            continue;
        }
        let entry_block = module.regions[body_reg].blocks[0];
        let block_ops = module.blocks[entry_block].ops.clone();
        if block_ops.is_empty() {
            continue;
        }

        // Skip if already wrapped in `castforge.try`.
        if block_ops
            .iter()
            .any(|&op_id| module.ops[op_id].is_op("castforge", "try"))
        {
            continue;
        }

        let term_id = block_ops[block_ops.len() - 1];
        let term_operands: SmallVec<[ValueId; 4]> = module.ops[term_id].operands.clone();
        let term_types: Vec<Type> = term_operands
            .iter()
            .map(|&v| module.values[v].ty.clone())
            .collect();
        let loc = module.ops[func_id].location.clone();

        // Create `try` and `except` regions and their single blocks.
        let try_block = module.blocks.insert_with_key(|bid| Block {
            id: bid,
            label: None,
            arguments: Vec::new(),
            ops: Vec::new(),
        });
        let try_region = module.regions.insert_with_key(|rid| Region {
            id: rid,
            blocks: vec![try_block],
        });

        let except_block = module.blocks.insert_with_key(|bid| Block {
            id: bid,
            label: None,
            arguments: Vec::new(),
            ops: Vec::new(),
        });
        let err_val = module.values.insert_with_key(|vid| Value {
            id: vid,
            ssa_name: None,
            ty: Type::Opaque("!castforge.error".to_string()),
            def: ValueDef::BlockArgument {
                block: except_block,
                index: 0,
            },
        });
        module.blocks[except_block].arguments.push(err_val);
        let except_region = module.regions.insert_with_key(|rid| Region {
            id: rid,
            blocks: vec![except_block],
        });

        // Move non-terminator ops into `try_block` and append `castforge.yield`.
        let mut moved_ops = Vec::new();
        for &op_id in &block_ops[..block_ops.len() - 1] {
            moved_ops.push(op_id);
        }
        let try_yield_loc = loc.clone();
        let try_yield_operands = term_operands.clone();
        let try_yield_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("castforge", "yield"),
            operands: try_yield_operands,
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: try_yield_loc,
        });
        moved_ops.push(try_yield_id);
        module.blocks[try_block].ops = moved_ops;

        // Populate `except_block` according to ABI.
        if is_runtime_abi {
            let raise_loc = loc.clone();
            let raise_id = module.ops.insert_with_key(|oid| Operation {
                id: oid,
                name: OpName::new("castforge", "raise"),
                operands: SmallVec::from_slice(&[err_val]),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: raise_loc,
            });
            module.blocks[except_block].ops.push(raise_id);
        } else {
            let mut except_yield_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
            for out_ty in &term_types {
                let async_err_loc = loc.clone();
                let async_err_id = module.ops.insert_with_key(|oid| Operation {
                    id: oid,
                    name: OpName::new("castforge", "async_error"),
                    operands: SmallVec::from_slice(&[err_val]),
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: async_err_loc,
                });
                let res_val = module.values.insert_with_key(|vid| Value {
                    id: vid,
                    ssa_name: None,
                    ty: out_ty.clone(),
                    def: ValueDef::OpResult {
                        op: async_err_id,
                        index: 0,
                    },
                });
                module.ops[async_err_id].results.push(res_val);
                module.blocks[except_block].ops.push(async_err_id);
                except_yield_vals.push(res_val);
            }
            let except_yield_loc = loc.clone();
            let except_yield_id = module.ops.insert_with_key(|oid| Operation {
                id: oid,
                name: OpName::new("castforge", "yield"),
                operands: except_yield_vals,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: except_yield_loc,
            });
            module.blocks[except_block].ops.push(except_yield_id);
        }

        // Create the `castforge.try` op and wire its results into `term_id`.
        let try_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("castforge", "try"),
            operands: SmallVec::new(),
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: vec![try_region, except_region],
            custom_syntax: false,
            location: loc,
        });
        let mut try_results: SmallVec<[ValueId; 4]> = SmallVec::new();
        for (idx, out_ty) in term_types.into_iter().enumerate() {
            let v_id = module.values.insert_with_key(|vid| Value {
                id: vid,
                ssa_name: None,
                ty: out_ty,
                def: ValueDef::OpResult {
                    op: try_id,
                    index: idx as u32,
                },
            });
            module.ops[try_id].results.push(v_id);
            try_results.push(v_id);
        }
        module.ops[term_id].operands = try_results;
        module.blocks[entry_block].ops = vec![try_id, term_id];
        wrapped += 1;
    }

    wrapped
}

/// Resolves `castforge.buffer_constant` (or `mgp.buffer_constant`) ops inside functions carrying a
/// `"constant_resources"` array attribute by looking up each op's `"value"` attribute in the
/// function's `"constant_resources"` table, rewriting the op in place to
/// `castforge.resource_constant` with `"resource_index" = index`, and replacing the function's
/// `"constant_resources"` entries with unit placeholders (`Attribute::Opaque("unit")`) so
/// downstream codegen reads only the resource count without retaining large constant attributes.
pub fn lookup_constant_resources(module: &mut Module) -> Result<usize, String> {
    use crate::ir::mlir::OpName;

    let mut rewritten = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let resources = match module.ops[func_id].attributes.get("constant_resources") {
            Some(Attribute::Array(arr)) => arr.clone(),
            _ => continue,
        };

        let mut body_ops = Vec::new();
        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                body_ops.extend(module.blocks[blk_id].ops.iter().copied());
            }
        }

        for op_id in body_ops {
            let op_name = module.ops[op_id].name.full_name();
            if op_name != "castforge.buffer_constant" && op_name != "mgp.buffer_constant" {
                continue;
            }
            let val_attr = module.ops[op_id]
                .attributes
                .get("value")
                .cloned()
                .ok_or_else(|| {
                    "buffer_constant is missing required 'value' attribute".to_string()
                })?;

            let resource_idx = resources
                .iter()
                .position(|item| item == &val_attr)
                .ok_or_else(|| "The constant resource can't be found in the kernel".to_string())?;

            module.ops[op_id].name = OpName::new("castforge", "resource_constant");
            module.ops[op_id].attributes.shift_remove("value");
            module.ops[op_id].attributes.insert(
                "resource_index".to_string(),
                Attribute::Int(resource_idx as i64),
            );
            rewritten += 1;
        }

        let unit_placeholders = vec![Attribute::Opaque("unit".to_string()); resources.len()];
        module.ops[func_id].attributes.insert(
            "constant_resources".to_string(),
            Attribute::Array(unit_placeholders),
        );
    }

    Ok(rewritten)
}
