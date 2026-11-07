//! Declarative routine graph-blob fusion (`form_graph_blobs`).

use crate::ir::{
    Attribute, Block, BlockId, Location, Module, OpId, OpName, Operation, Region, Type, Value,
    ValueDef, ValueId,
};
use crate::passes::serving::kv_attention::fuse_locations;
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

fn is_effect_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Opaque(s) if s.contains("chain") || s.contains("buffer") || s.contains("token")
    )
}

fn is_terminator_op(name: &str) -> bool {
    matches!(
        name,
        "func.return"
            | "return"
            | "mo.output"
            | "mo.yield"
            | "scf.yield"
            | "mogg.routine.return"
            | "mgp.device_graph.staging.yield"
            | "mgp.device_graph.collective.yield"
    )
}

fn collect_read_values(module: &Module, op_id: OpId) -> IndexSet<ValueId> {
    let mut read_vals: IndexSet<ValueId> = IndexSet::new();
    let Some(op) = module.ops.get(op_id) else {
        return read_vals;
    };
    for &opd in &op.operands {
        read_vals.insert(opd);
    }
    let mut defined_inside: IndexSet<ValueId> = IndexSet::new();
    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &arg in &blk.arguments {
                defined_inside.insert(arg);
            }
            for &inner_id in &blk.ops {
                if let Some(inner_op) = module.ops.get(inner_id) {
                    for &res in &inner_op.results {
                        defined_inside.insert(res);
                    }
                }
            }
        }
    }
    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &inner_id in &blk.ops {
                if let Some(inner_op) = module.ops.get(inner_id) {
                    for &opd in &inner_op.operands {
                        if !defined_inside.contains(&opd) {
                            read_vals.insert(opd);
                        }
                    }
                }
            }
        }
    }
    read_vals
}

/// Identifies SSA values in `blk_id` that are execution-invariant across runs (constants without
/// external mutable aliases, and non-branching/non-parallel operations without effect types whose
/// read values are all execution-invariant).
pub fn find_invariant_values(module: &Module, blk_id: BlockId) -> IndexSet<ValueId> {
    let mut invariant: IndexSet<ValueId> = IndexSet::new();
    let Some(blk) = module.blocks.get(blk_id) else {
        return invariant;
    };
    for &op_id in &blk.ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let full = op.name.full_name();
        let is_const = full.ends_with(".constant")
            || full == "stablehlo.constant"
            || full == "mo.constant"
            || full == "mo.constant.external";
        if is_const {
            let has_alias = matches!(op.attributes.get("has_alias"), Some(Attribute::Bool(true)));
            if full != "mo.constant.external" || !has_alias {
                for &res in &op.results {
                    invariant.insert(res);
                }
            }
            continue;
        }
        let is_branch_or_parallel = matches!(
            full.as_str(),
            "mo.if" | "scf.if" | "mo.while" | "scf.while" | "mo.parallel"
        );
        if is_branch_or_parallel {
            continue;
        }
        let has_effect_opd = op
            .operands
            .iter()
            .any(|&v| is_effect_type(&module.values[v].ty));
        let has_effect_res = op
            .results
            .iter()
            .any(|&v| is_effect_type(&module.values[v].ty));
        if has_effect_opd || has_effect_res {
            continue;
        }
        let reads = collect_read_values(module, op_id);
        if !reads.is_empty() && reads.iter().all(|v| invariant.contains(v)) {
            for &res in &op.results {
                invariant.insert(res);
            }
        }
    }
    invariant
}

/// Collects declarative kernel names (`mogg.kernel_decl` with `execute_device = true`) declared at
/// the module top level.
pub fn collect_declarative_kernel_decls(module: &Module) -> IndexSet<String> {
    let mut decls = IndexSet::new();
    let root_blk = module.body_block();
    if let Some(blk) = module.blocks.get(root_blk) {
        for &op_id in &blk.ops {
            if let Some(op) = module.ops.get(op_id)
                && op.name.full_name() == "mogg.kernel_decl"
                && matches!(
                    op.attributes.get("execute_device"),
                    Some(Attribute::Bool(true))
                )
                && let Some(Attribute::String(sym)) = op
                    .attributes
                    .get("sym_name")
                    .or_else(|| op.attributes.get("symbol"))
            {
                decls.insert(sym.clone());
            }
        }
    }
    decls
}

/// Checks whether an `mogg.routine` operation `op_id` can join a declarative `mogg.graph_blob`:
/// 1. Must be an `mogg.routine` with non-empty tensor inputs and outputs, no effect chain operand
///    or result, and no non-empty `output_param_decls`.
/// 2. At least one output must not be in `invariant` (init-only routines stay out of blobs).
/// 3. Every `!mogg.dim` block argument in its body must have a non-empty `ssa_name`, and its body
///    must contain exactly one call (`mogg.call.opaque` or `mogg.kernel.invoke`) targeting a
///    declarative `execute_device` kernel in `declarative_kernels` (or marked `execute_device =
///    true` on the call).
pub fn can_join_graph_blob(
    module: &Module,
    op_id: OpId,
    declarative_kernels: &IndexSet<String>,
    invariant: &IndexSet<ValueId>,
) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    if op.name.full_name() != "mogg.routine" || op.results.is_empty() || op.regions.is_empty() {
        return false;
    }
    if let Some(Attribute::Array(decls)) = op.attributes.get("output_param_decls")
        && !decls.is_empty()
    {
        return false;
    }
    if op
        .operands
        .iter()
        .any(|&v| !matches!(module.values[v].ty, Type::Tensor(_)))
        || op
            .results
            .iter()
            .any(|&v| !matches!(module.values[v].ty, Type::Tensor(_)))
    {
        return false;
    }
    if op.results.iter().all(|v| invariant.contains(v)) {
        return false;
    }

    let reg_id = op.regions[0];
    let Some(reg) = module.regions.get(reg_id) else {
        return false;
    };
    let Some(&body_blk_id) = reg.blocks.first() else {
        return false;
    };
    let Some(body_blk) = module.blocks.get(body_blk_id) else {
        return false;
    };

    for &arg_val in &body_blk.arguments {
        if matches!(&module.values[arg_val].ty, Type::Opaque(s) if s == "!mogg.dim")
            && module.values[arg_val]
                .ssa_name
                .as_ref()
                .is_none_or(String::is_empty)
        {
            return false;
        }
    }

    let mut found_call: Option<&Operation> = None;
    for &inner_id in &body_blk.ops {
        let Some(inner_op) = module.ops.get(inner_id) else {
            continue;
        };
        let iname = inner_op.name.full_name();
        if matches!(iname.as_str(), "mogg.call.opaque" | "mogg.kernel.invoke") {
            if found_call.is_some() {
                return false;
            }
            found_call = Some(inner_op);
        }
    }
    let Some(call_op) = found_call else {
        return false;
    };
    if matches!(
        call_op.attributes.get("execute_device"),
        Some(Attribute::Bool(true))
    ) {
        return true;
    }
    let callee = call_op
        .attributes
        .get("callee")
        .or_else(|| call_op.attributes.get("symbol"));
    matches!(callee, Some(Attribute::String(s)) if declarative_kernels.contains(s))
}

fn routine_device_ref(op: &Operation) -> String {
    match op.attributes.get("device_ref") {
        Some(Attribute::String(s)) => s.clone(),
        _ => "cpu:0".to_string(),
    }
}

/// Fuses groups of declarative `mogg.routine` operations (sequences and siblings alike on the same
/// device) into a single `mogg.routine` whose body is an `mogg.graph_blob` containing
/// `mogg.call.execute_device` invocations.
///
/// A blob closes immediately at any member whose result is read by an operation outside the blob
/// (other than the block terminator), ensuring no dataflow path leaves a blob and re-enters it.
/// Returns the number of fused `mogg.graph_blob` routines created.
pub fn form_graph_blobs(module: &mut Module) -> usize {
    let declarative_kernels = collect_declarative_kernel_decls(module);
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut blobs_formed = 0usize;

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        if orig_ops.len() < 2 {
            continue;
        }
        let invariant = find_invariant_values(module, blk_id);

        let mut val_users: IndexMap<ValueId, Vec<(BlockId, OpId)>> = IndexMap::new();
        for (b_id, b) in &module.blocks {
            for &o_id in &b.ops {
                if let Some(o) = module.ops.get(o_id) {
                    for &opd in &o.operands {
                        val_users.entry(opd).or_default().push((b_id, o_id));
                    }
                }
            }
        }

        let is_read_outside = |module: &Module, member_id: OpId, dev: &str| -> bool {
            let Some(member_op) = module.ops.get(member_id) else {
                return true;
            };
            for &res in &member_op.results {
                let Some(users) = val_users.get(&res) else {
                    continue;
                };
                for &(user_blk, user_op_id) in users {
                    if user_blk != blk_id {
                        return true;
                    }
                    let Some(user_op) = module.ops.get(user_op_id) else {
                        return true;
                    };
                    if is_terminator_op(&user_op.name.full_name()) {
                        continue;
                    }
                    if !can_join_graph_blob(module, user_op_id, &declarative_kernels, &invariant)
                        || routine_device_ref(user_op) != dev
                    {
                        return true;
                    }
                }
            }
            false
        };

        let mut groups_to_fuse: Vec<Vec<OpId>> = Vec::new();
        let mut current_blob: Vec<OpId> = Vec::new();
        let mut current_dev = String::new();

        for &op_id in &orig_ops {
            if can_join_graph_blob(module, op_id, &declarative_kernels, &invariant) {
                let dev = routine_device_ref(&module.ops[op_id]);
                if !current_blob.is_empty() && dev != current_dev {
                    if current_blob.len() > 1 {
                        groups_to_fuse.push(std::mem::take(&mut current_blob));
                    } else {
                        current_blob.clear();
                    }
                }
                current_dev = dev.clone();
                current_blob.push(op_id);
                if is_read_outside(module, op_id, &dev) {
                    if current_blob.len() > 1 {
                        groups_to_fuse.push(std::mem::take(&mut current_blob));
                    } else {
                        current_blob.clear();
                    }
                }
            } else if current_blob.len() > 1 {
                groups_to_fuse.push(std::mem::take(&mut current_blob));
            } else {
                current_blob.clear();
            }
        }
        if current_blob.len() > 1 {
            groups_to_fuse.push(current_blob);
        }

        for members in groups_to_fuse {
            fuse_routine_blob_group(module, blk_id, &members);
            blobs_formed += 1;
        }
    }

    blobs_formed
}

fn fuse_routine_blob_group(module: &mut Module, blk_id: BlockId, members: &[OpId]) {
    let Some(&last_id) = members.last() else {
        return;
    };
    let mut locs: Vec<Location> = Vec::with_capacity(members.len());
    let mut external_inputs: Vec<ValueId> = Vec::new();
    let mut input_seen: IndexSet<ValueId> = IndexSet::new();
    let mut all_results: Vec<ValueId> = Vec::new();
    let mut member_result_set: IndexSet<ValueId> = IndexSet::new();
    let mut any_deferred = false;
    let mut shared_dim_names: IndexSet<String> = IndexSet::new();
    let dev_ref = routine_device_ref(&module.ops[last_id]);

    for &m_id in members {
        let Some(m_op) = module.ops.get(m_id) else {
            continue;
        };
        locs.push(m_op.location.clone());
        if matches!(
            m_op.attributes.get("dispatch"),
            Some(Attribute::String(s)) if s == "deferred"
        ) {
            any_deferred = true;
        }
        for &inp in &m_op.operands {
            if !member_result_set.contains(&inp) && input_seen.insert(inp) {
                external_inputs.push(inp);
            }
        }
        for &res in &m_op.results {
            member_result_set.insert(res);
            all_results.push(res);
        }
        if let Some(&r_id) = m_op.regions.first()
            && let Some(reg) = module.regions.get(r_id)
            && let Some(&b_id) = reg.blocks.first()
            && let Some(b) = module.blocks.get(b_id)
        {
            for &arg in &b.arguments {
                if matches!(&module.values[arg].ty, Type::Opaque(s) if s == "!mogg.dim")
                    && let Some(ref name) = module.values[arg].ssa_name
                {
                    shared_dim_names.insert(name.clone());
                }
            }
        }
    }

    let fused_loc = fuse_locations(&locs);
    let routine_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let ptr_ty = Type::Opaque("!mogg.pointer".to_string());
    let dim_ty = Type::Opaque("!mogg.dim".to_string());
    let ctx_ty = Type::Opaque("!mogg.context".to_string());

    for idx in 0..(external_inputs.len() + all_results.len()) {
        let arg_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: ptr_ty.clone(),
            def: ValueDef::BlockArgument {
                block: routine_blk,
                index: idx as u32,
            },
        });
        module.blocks[routine_blk].arguments.push(arg_val);
    }
    for dim_name in &shared_dim_names {
        let idx = module.blocks[routine_blk].arguments.len() as u32;
        let dim_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: Some(dim_name.clone()),
            ty: dim_ty.clone(),
            def: ValueDef::BlockArgument {
                block: routine_blk,
                index: idx,
            },
        });
        module.blocks[routine_blk].arguments.push(dim_val);
    }
    let ctx_idx = module.blocks[routine_blk].arguments.len() as u32;
    let ctx_arg = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: ctx_ty,
        def: ValueDef::BlockArgument {
            block: routine_blk,
            index: ctx_idx,
        },
    });
    module.blocks[routine_blk].arguments.push(ctx_arg);

    let blob_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    for &m_id in members {
        let Some(m_op) = module.ops.get(m_id).cloned() else {
            continue;
        };
        if let Some(&r_id) = m_op.regions.first()
            && let Some(reg) = module.regions.get(r_id).cloned()
            && let Some(&b_id) = reg.blocks.first()
            && let Some(b) = module.blocks.get(b_id).cloned()
        {
            for &inner_id in &b.ops {
                let Some(inner_op) = module.ops.get(inner_id).cloned() else {
                    continue;
                };
                let iname = inner_op.name.full_name();
                if matches!(iname.as_str(), "mogg.call.opaque" | "mogg.kernel.invoke") {
                    let callee = inner_op
                        .attributes
                        .get("callee")
                        .or_else(|| inner_op.attributes.get("symbol"))
                        .cloned()
                        .unwrap_or_else(|| Attribute::String("kernel".to_string()));
                    let mut exec_attrs = IndexMap::new();
                    exec_attrs.insert("callee".to_string(), callee);
                    let exec_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mogg", "call.execute_device"),
                        operands: SmallVec::from_slice(&[ctx_arg]),
                        results: SmallVec::new(),
                        attributes: exec_attrs,
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: inner_op.location.clone(),
                    });
                    module.blocks[blob_blk].ops.push(exec_op);
                }
            }
        }
    }

    let blob_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![blob_blk],
    });
    let blob_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg", "graph_blob"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: vec![blob_reg],
        custom_syntax: false,
        location: fused_loc.clone(),
    });
    module.blocks[routine_blk].ops.push(blob_op);

    let routine_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![routine_blk],
    });

    let mut fused_attrs = IndexMap::new();
    fused_attrs.insert("device_ref".to_string(), Attribute::String(dev_ref));
    fused_attrs.insert(
        "dispatch".to_string(),
        Attribute::String(
            if any_deferred {
                "deferred"
            } else {
                "immediate"
            }
            .to_string(),
        ),
    );
    fused_attrs.insert(
        "blob_member_count".to_string(),
        Attribute::Int(members.len() as i64),
    );
    fused_attrs.insert(
        "shared_dim_names".to_string(),
        Attribute::Array(
            shared_dim_names
                .iter()
                .map(|s| Attribute::String(s.clone()))
                .collect(),
        ),
    );

    let fused_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg", "routine"),
        operands: SmallVec::from_slice(&external_inputs),
        results: SmallVec::new(),
        attributes: fused_attrs,
        regions: vec![routine_reg],
        custom_syntax: false,
        location: fused_loc,
    });

    for (r_idx, &old_res) in all_results.iter().enumerate() {
        let r_ty = module.values[old_res].ty.clone();
        let new_res = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: r_ty,
            def: ValueDef::OpResult {
                op: fused_op,
                index: r_idx as u32,
            },
        });
        module.ops[fused_op].results.push(new_res);
        module.replace_all_uses(old_res, new_res);
    }

    let member_set: IndexSet<OpId> = members.iter().copied().collect();
    if let Some(blk_mut) = module.blocks.get_mut(blk_id) {
        let mut new_ops = Vec::with_capacity(blk_mut.ops.len());
        for &id in &blk_mut.ops {
            if id == last_id {
                new_ops.push(fused_op);
            } else if !member_set.contains(&id) {
                new_ops.push(id);
            }
        }
        blk_mut.ops = new_ops;
    }
    for &m_id in members {
        module.ops.remove(m_id);
    }
}
