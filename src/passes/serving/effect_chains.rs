//! Chain sequence canonicalization, parallel unbundling, and layout inference.

#[allow(unused_imports)]
use super::*;
use crate::ir::{Attribute, BlockId, Module, OpId, Type, ValueId};

pub(super) fn is_chain_type(ty: &Type) -> bool {
    matches!(ty, Type::Opaque(s) if s == "!mo.chain" || s.starts_with("!mo.chain<"))
}

pub(super) fn is_mutable_operand_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Opaque(s)
            if s.starts_with("!mo.buffer")
                || s.starts_with("!mo.opaque")
                || s.starts_with("!mogg.buffer")
    )
}

pub(super) fn extract_opaque_symbol(ty: &Type) -> Option<&str> {
    if let Type::Opaque(s) = ty
        && let Some(rest) = s.strip_prefix("!mo.opaque<")
    {
        return rest.strip_suffix('>');
    }
    None
}

pub(super) fn single_in_out_chains(
    module: &Module,
    op_id: OpId,
) -> Option<(usize, ValueId, ValueId)> {
    let op = module.ops.get(op_id)?;
    if op.name.full_name() == "mo.chain.create" {
        return None;
    }
    let mut in_chains = Vec::new();
    for (idx, &v) in op.operands.iter().enumerate() {
        if let Some(val) = module.values.get(v)
            && is_chain_type(&val.ty)
        {
            in_chains.push((idx, v));
        }
    }
    let mut out_chains = Vec::new();
    for &r in &op.results {
        if let Some(val) = module.values.get(r)
            && is_chain_type(&val.ty)
        {
            out_chains.push(r);
        }
    }
    if in_chains.len() == 1 && out_chains.len() == 1 {
        Some((in_chains[0].0, in_chains[0].1, out_chains[0]))
    } else {
        None
    }
}

pub(super) fn collect_mutable_and_write_operands(
    module: &Module,
    op_id: OpId,
) -> (
    std::collections::HashSet<ValueId>,
    std::collections::HashSet<ValueId>,
    std::collections::HashSet<String>,
) {
    let mut mutable_set = std::collections::HashSet::new();
    let mut write_set = std::collections::HashSet::new();
    let mut opaque_syms = std::collections::HashSet::new();
    let Some(op) = module.ops.get(op_id) else {
        return (mutable_set, write_set, opaque_syms);
    };
    let is_read_only_load = op.name.full_name() == "mo.mutable.load"
        || matches!(
            op.attributes.get("memory_effect"),
            Some(Attribute::String(s)) if s == "read"
        );
    for &v in &op.operands {
        if let Some(val) = module.values.get(v)
            && is_mutable_operand_type(&val.ty)
        {
            mutable_set.insert(v);
            if !is_read_only_load {
                write_set.insert(v);
            }
            if let Some(sym) = extract_opaque_symbol(&val.ty) {
                opaque_syms.insert(sym.to_string());
            }
        }
    }
    (mutable_set, write_set, opaque_syms)
}

pub(super) fn can_reorder_chained_mutable_ops(module: &Module, op1: OpId, op2: OpId) -> bool {
    for oid in [op1, op2] {
        if matches!(
            module.ops[oid].attributes.get("multi_device"),
            Some(Attribute::Bool(true))
        ) {
            return false;
        }
    }
    let (mut1, write1, opq1) = collect_mutable_and_write_operands(module, op1);
    let (mut2, write2, opq2) = collect_mutable_and_write_operands(module, op2);
    if mut1.iter().any(|v| write2.contains(v)) || mut2.iter().any(|v| write1.contains(v)) {
        return false;
    }
    if opq1.iter().any(|s| opq2.contains(s)) {
        return false;
    }
    true
}

/// Relaxes execution chains across independent buffer-mutating operations in the same block,
/// joining their output chains via `mo.chain.create` when no write-after-read, read-after-write,
/// write-after-write, multi-device, or opaque-alias hazard exists.
///
/// Returns the number of chain dependencies relaxed.
pub fn canonicalize_chain_sequences(module: &mut Module) -> usize {
    use crate::ir::mlir::{Location, OpName, Operation, Value, ValueDef};

    let mut relaxed = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let mut idx = 0usize;
        while idx < module.blocks[bid].ops.len() {
            let op_id = module.blocks[bid].ops[idx];
            let Some((in_chain_operand_idx, in_chain_val, out_chain_val)) =
                single_in_out_chains(module, op_id)
            else {
                idx += 1;
                continue;
            };
            let Some(ValueDef::OpResult {
                op: parent_op_id, ..
            }) = module.values.get(in_chain_val).map(|v| v.def)
            else {
                idx += 1;
                continue;
            };
            if !module.blocks[bid].ops[..idx].contains(&parent_op_id) {
                idx += 1;
                continue;
            }
            let Some((_, parent_in_chain, parent_out_chain)) =
                single_in_out_chains(module, parent_op_id)
            else {
                idx += 1;
                continue;
            };
            if !can_reorder_chained_mutable_ops(module, op_id, parent_op_id) {
                idx += 1;
                continue;
            }

            module.ops[op_id].operands[in_chain_operand_idx] = parent_in_chain;
            let chain_ty = module.values[out_chain_val].ty.clone();
            let merged_chain_val = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::BlockArgument {
                    block: bid,
                    index: 0,
                },
            });
            module.values[merged_chain_val].id = merged_chain_val;
            let merge_op_id = module.ops.insert(Operation {
                id: OpId::default(),
                name: OpName::new("mo", "chain.create"),
                operands: smallvec::smallvec![out_chain_val, parent_out_chain],
                results: smallvec::smallvec![merged_chain_val],
                attributes: indexmap::IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: Location::Unknown,
            });
            module.ops[merge_op_id].id = merge_op_id;
            module.values[merged_chain_val].def = ValueDef::OpResult {
                op: merge_op_id,
                index: 0,
            };

            let all_ops: Vec<OpId> = module.ops.keys().collect();
            for other_id in all_ops {
                if other_id == merge_op_id || other_id == op_id {
                    continue;
                }
                for opnd in &mut module.ops[other_id].operands {
                    if *opnd == out_chain_val {
                        *opnd = merged_chain_val;
                    }
                }
            }
            module.blocks[bid].ops.insert(idx + 1, merge_op_id);
            relaxed += 1;
            idx += 2;
        }
    }
    relaxed
}

/// Eliminates redundant `mo.mutable.store` -> `mo.mutable.load` pairs and dead overwritten
/// `mo.mutable.store` -> `mo.mutable.store` pairs on the same buffer when connected by a
/// single-use execution chain.
///
/// Returns the total number of redundant mutable operations eliminated.
pub fn elide_redundant_mutable_ops(module: &mut Module) -> usize {
    let mut elided = 0usize;
    loop {
        let mut changed_in_iter = false;
        let use_counts = {
            let mut counts: std::collections::HashMap<ValueId, usize> =
                std::collections::HashMap::new();
            for op in module.ops.values() {
                for &v in &op.operands {
                    *counts.entry(v).or_insert(0) += 1;
                }
            }
            counts
        };

        let block_ids: Vec<BlockId> = module.blocks.keys().collect();
        for bid in block_ids {
            let ops_snapshot = module.blocks[bid].ops.clone();
            for &op_id in &ops_snapshot {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.name.full_name() == "mo.mutable.load"
                    && op.operands.len() >= 2
                    && op.results.len() >= 2
                {
                    let in_chain = op.operands[0];
                    let load_buf = op.operands[1];
                    if use_counts.get(&in_chain).copied().unwrap_or(0) != 1 {
                        continue;
                    }
                    let Some(crate::ir::mlir::ValueDef::OpResult { op: store_id, .. }) =
                        module.values.get(in_chain).map(|v| v.def)
                    else {
                        continue;
                    };
                    let Some(store_op) = module.ops.get(store_id).cloned() else {
                        continue;
                    };
                    if store_op.name.full_name() == "mo.mutable.store"
                        && store_op.operands.len() >= 3
                        && store_op.operands[1] == load_buf
                    {
                        let stored_tensor = store_op.operands[2];
                        let store_out_chain = store_op.results[0];
                        let load_out_tensor = op.results[0];
                        let load_out_chain = op.results[1];
                        let all_ops: Vec<OpId> = module.ops.keys().collect();
                        for oid in all_ops {
                            for opnd in &mut module.ops[oid].operands {
                                if *opnd == load_out_tensor {
                                    *opnd = stored_tensor;
                                } else if *opnd == load_out_chain {
                                    *opnd = store_out_chain;
                                }
                            }
                        }
                        module.blocks[bid].ops.retain(|&id| id != op_id);
                        module.ops.remove(op_id);
                        elided += 1;
                        changed_in_iter = true;
                        break;
                    }
                } else if op.name.full_name() == "mo.mutable.store"
                    && op.operands.len() >= 3
                    && !op.results.is_empty()
                {
                    let curr_in_chain = op.operands[0];
                    let curr_buf = op.operands[1];
                    let curr_out_chain = op.results[0];
                    if use_counts.get(&curr_out_chain).copied().unwrap_or(0) != 1 {
                        continue;
                    }
                    let user_id = module.ops.keys().find(|&oid| {
                        oid != op_id && module.ops[oid].operands.contains(&curr_out_chain)
                    });
                    let Some(next_id) = user_id else {
                        continue;
                    };
                    let Some(next_op) = module.ops.get(next_id).cloned() else {
                        continue;
                    };
                    if next_op.name.full_name() == "mo.mutable.store"
                        && next_op.operands.len() >= 3
                        && next_op.operands[0] == curr_out_chain
                        && next_op.operands[1] == curr_buf
                    {
                        for opnd in &mut module.ops[next_id].operands {
                            if *opnd == curr_out_chain {
                                *opnd = curr_in_chain;
                            }
                        }
                        module.blocks[bid].ops.retain(|&id| id != op_id);
                        module.ops.remove(op_id);
                        elided += 1;
                        changed_in_iter = true;
                        break;
                    }
                }
            }
            if changed_in_iter {
                break;
            }
        }
        if !changed_in_iter {
            break;
        }
    }
    elided
}

pub(super) fn is_absorbable_unbundle_consumer(module: &Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    if op.results.len() != 1 || !op.regions.is_empty() {
        return false;
    }
    let full = op.name.full_name();
    if matches!(
        full.as_str(),
        "func.return"
            | "return"
            | "mo.yield"
            | "mo.tensor.bundle"
            | "mo.tensor.unbundle"
            | "mo.parallel"
    ) {
        return false;
    }
    let Some(res_val) = module.values.get(op.results[0]) else {
        return false;
    };
    if !matches!(res_val.ty, Type::Tensor(_)) {
        return false;
    }
    for &opnd in &op.operands {
        let Some(v) = module.values.get(opnd) else {
            return false;
        };
        if is_chain_type(&v.ty) || is_mutable_operand_type(&v.ty) {
            return false;
        }
    }
    true
}

pub(super) fn unbundle_consumers_match(module: &Module, a_id: OpId, b_id: OpId) -> bool {
    let (Some(a), Some(b)) = (module.ops.get(a_id), module.ops.get(b_id)) else {
        return false;
    };
    if a.name != b.name
        || a.operands.len() != b.operands.len()
        || a.results.len() != b.results.len()
    {
        return false;
    }
    let a_filtered: std::collections::BTreeMap<&String, &Attribute> = a
        .attributes
        .iter()
        .filter(|(k, _)| k.as_str() != "device")
        .collect();
    let b_filtered: std::collections::BTreeMap<&String, &Attribute> = b
        .attributes
        .iter()
        .filter(|(k, _)| k.as_str() != "device")
        .collect();
    a_filtered == b_filtered
}

pub(super) fn can_bundle_per_launch_values(module: &Module, values: &[ValueId]) -> bool {
    let Some(&first_val) = values.first() else {
        return false;
    };
    let Some(Type::Tensor(first_t)) = module.values.get(first_val).map(|v| &v.ty) else {
        return false;
    };
    for &v in &values[1..] {
        let Some(Type::Tensor(t)) = module.values.get(v).map(|val| &val.ty) else {
            return false;
        };
        if t.shape != first_t.shape || t.dtype != first_t.dtype {
            return false;
        }
    }
    true
}

pub(super) fn format_bundle_type(module: &Module, elem_values: &[ValueId]) -> Type {
    let mut parts = Vec::with_capacity(elem_values.len());
    for &v in elem_values {
        if let Some(val) = module.values.get(v) {
            parts.push(format!("{}", val.ty));
        }
    }
    Type::Opaque(format!("!mo.bundle<{}>", parts.join(", ")))
}

/// Wraps structurally identical single-use consumers of a `mo.tensor.unbundle` operation's
/// results into a `mo.parallel` region followed by a new `mo.tensor.unbundle`.
///
/// Iterates to a fixed point so downstream consumer chains of newly created unbundles are also
/// wrapped. Returns the total number of `mo.parallel` operations created.
pub fn create_parallel_from_unbundle(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, Location, OpName, Operation, Region, RegionId, Value, ValueDef};

    let mut wrapped_count = 0usize;
    loop {
        let mut changed = false;
        let use_map = {
            let mut map: std::collections::HashMap<ValueId, Vec<OpId>> =
                std::collections::HashMap::new();
            for (oid, op) in &module.ops {
                for &v in &op.operands {
                    map.entry(v).or_default().push(oid);
                }
            }
            map
        };

        let block_ids: Vec<BlockId> = module.blocks.keys().collect();
        for bid in block_ids {
            let ops_snapshot = module.blocks[bid].ops.clone();
            for &unbundle_id in &ops_snapshot {
                let Some(unbundle_op) = module.ops.get(unbundle_id).cloned() else {
                    continue;
                };
                if unbundle_op.name.full_name() != "mo.tensor.unbundle"
                    || unbundle_op.results.is_empty()
                {
                    continue;
                }
                let num_launches = unbundle_op.results.len();
                let mut consumers = Vec::with_capacity(num_launches);
                let mut unique_consumers = std::collections::HashSet::new();
                let mut valid = true;
                for &res_v in &unbundle_op.results {
                    let Some(users) = use_map.get(&res_v) else {
                        valid = false;
                        break;
                    };
                    if users.len() != 1 {
                        valid = false;
                        break;
                    }
                    let c_id = users[0];
                    if !module.blocks[bid].ops.contains(&c_id) || !unique_consumers.insert(c_id) {
                        valid = false;
                        break;
                    }
                    consumers.push(c_id);
                }
                if !valid {
                    continue;
                }

                let rep_id = consumers[0];
                if !is_absorbable_unbundle_consumer(module, rep_id) {
                    continue;
                }
                if consumers[1..]
                    .iter()
                    .any(|&peer| !unbundle_consumers_match(module, rep_id, peer))
                {
                    continue;
                }

                let rep_operands = module.ops[rep_id].operands.clone();
                let Some(driving_idx) = rep_operands
                    .iter()
                    .position(|&v| v == unbundle_op.results[0])
                else {
                    continue;
                };
                if consumers[1..].iter().enumerate().any(|(i, &peer)| {
                    module.ops[peer].operands[driving_idx] != unbundle_op.results[i + 1]
                }) {
                    continue;
                }

                // Plan per-operand bundles:
                let num_operands = rep_operands.len();
                let mut bundles: Vec<Vec<ValueId>> = vec![unbundle_op.results.to_vec()];
                let mut operand_to_block_arg = vec![0usize; num_operands];
                let mut plan_ok = true;
                for idx in 0..num_operands {
                    if idx == driving_idx {
                        continue;
                    }
                    let outer_vals: Vec<ValueId> = consumers
                        .iter()
                        .map(|&cid| module.ops[cid].operands[idx])
                        .collect();
                    if num_launches > 1 && outer_vals.iter().all(|&v| v == outer_vals[0]) {
                        plan_ok = false;
                        break;
                    }
                    if !can_bundle_per_launch_values(module, &outer_vals) {
                        plan_ok = false;
                        break;
                    }
                    bundles.push(outer_vals);
                    operand_to_block_arg[idx] = bundles.len() - 1;
                }
                if !plan_ok {
                    continue;
                }

                let mut inserted_ops = Vec::new();
                let mut parallel_inputs = smallvec::SmallVec::<[ValueId; 4]>::new();
                for bundle_vals in &bundles {
                    let b_ty = format_bundle_type(module, bundle_vals);
                    let b_res = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: b_ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[b_res].id = b_res;
                    let b_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("mo", "tensor.bundle"),
                        operands: bundle_vals.iter().copied().collect(),
                        results: smallvec::smallvec![b_res],
                        attributes: indexmap::IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[b_op].id = b_op;
                    module.values[b_res].def = ValueDef::OpResult { op: b_op, index: 0 };
                    inserted_ops.push(b_op);
                    parallel_inputs.push(b_res);
                }

                let out_elem_vals: Vec<ValueId> = consumers
                    .iter()
                    .map(|&cid| module.ops[cid].results[0])
                    .collect();
                let out_bundle_ty = format_bundle_type(module, &out_elem_vals);

                let body_block_id = module.blocks.insert(Block {
                    id: BlockId::default(),
                    label: None,
                    arguments: Vec::new(),
                    ops: Vec::new(),
                });
                module.blocks[body_block_id].id = body_block_id;

                let mut block_args = Vec::with_capacity(bundles.len());
                for (arg_i, bundle_vals) in bundles.iter().enumerate() {
                    let elem_ty = module.values[bundle_vals[0]].ty.clone();
                    let arg_val = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: elem_ty,
                        def: ValueDef::BlockArgument {
                            block: body_block_id,
                            index: arg_i as u32,
                        },
                    });
                    module.values[arg_val].id = arg_val;
                    block_args.push(arg_val);
                }
                module.blocks[body_block_id].arguments = block_args.clone();

                let cloned_operands: smallvec::SmallVec<[ValueId; 4]> = (0..num_operands)
                    .map(|i| block_args[operand_to_block_arg[i]])
                    .collect();
                let rep_res_ty = module.values[module.ops[rep_id].results[0]].ty.clone();
                let cloned_res = module.values.insert(Value {
                    id: ValueId::default(),
                    ssa_name: None,
                    ty: rep_res_ty,
                    def: ValueDef::BlockArgument {
                        block: body_block_id,
                        index: 0,
                    },
                });
                module.values[cloned_res].id = cloned_res;
                let mut cloned_attrs = module.ops[rep_id].attributes.clone();
                cloned_attrs.shift_remove("device");
                let cloned_op = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: module.ops[rep_id].name.clone(),
                    operands: cloned_operands,
                    results: smallvec::smallvec![cloned_res],
                    attributes: cloned_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                module.ops[cloned_op].id = cloned_op;
                module.values[cloned_res].def = ValueDef::OpResult {
                    op: cloned_op,
                    index: 0,
                };

                let yield_op = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: OpName::new("mo", "yield"),
                    operands: smallvec::smallvec![cloned_res],
                    results: smallvec::SmallVec::new(),
                    attributes: indexmap::IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                module.ops[yield_op].id = yield_op;
                module.blocks[body_block_id].ops = vec![cloned_op, yield_op];

                let body_region_id = module.regions.insert(Region {
                    id: RegionId::default(),
                    blocks: vec![body_block_id],
                });
                module.regions[body_region_id].id = body_region_id;

                let parallel_res = module.values.insert(Value {
                    id: ValueId::default(),
                    ssa_name: None,
                    ty: out_bundle_ty,
                    def: ValueDef::BlockArgument {
                        block: bid,
                        index: 0,
                    },
                });
                module.values[parallel_res].id = parallel_res;
                let parallel_op = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: OpName::new("mo", "parallel"),
                    operands: parallel_inputs,
                    results: smallvec::smallvec![parallel_res],
                    attributes: indexmap::IndexMap::new(),
                    regions: vec![body_region_id],
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                module.ops[parallel_op].id = parallel_op;
                module.values[parallel_res].def = ValueDef::OpResult {
                    op: parallel_op,
                    index: 0,
                };
                inserted_ops.push(parallel_op);

                let mut new_unbundle_results = smallvec::SmallVec::<[ValueId; 2]>::new();
                for (i, &cid) in consumers.iter().enumerate() {
                    let old_res = module.ops[cid].results[0];
                    let res_ty = module.values[old_res].ty.clone();
                    let new_res = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: res_ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: i as u32,
                        },
                    });
                    module.values[new_res].id = new_res;
                    new_unbundle_results.push(new_res);
                }
                let new_unbundle_op = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: OpName::new("mo", "tensor.unbundle"),
                    operands: smallvec::smallvec![parallel_res],
                    results: new_unbundle_results.clone(),
                    attributes: indexmap::IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                module.ops[new_unbundle_op].id = new_unbundle_op;
                for (i, &r) in new_unbundle_results.iter().enumerate() {
                    module.values[r].def = ValueDef::OpResult {
                        op: new_unbundle_op,
                        index: i as u32,
                    };
                }
                inserted_ops.push(new_unbundle_op);

                // Replace uses of each consumer's output with the new unbundle's result:
                let all_ops: Vec<OpId> = module.ops.keys().collect();
                for (i, &cid) in consumers.iter().enumerate() {
                    let old_res = module.ops[cid].results[0];
                    let replacement = new_unbundle_results[i];
                    for oid in &all_ops {
                        for opnd in &mut module.ops[*oid].operands {
                            if *opnd == old_res {
                                *opnd = replacement;
                            }
                        }
                    }
                }

                // Insert the new ops right after the last consumer in block order
                // and erase the old consumers:
                let consumer_set: std::collections::HashSet<OpId> =
                    consumers.iter().copied().collect();
                let insert_after = module.blocks[bid]
                    .ops
                    .iter()
                    .rposition(|id| consumer_set.contains(id))
                    .unwrap_or(0);
                let mut new_block_ops =
                    Vec::with_capacity(module.blocks[bid].ops.len() + inserted_ops.len());
                for (pos, &oid) in module.blocks[bid].ops.iter().enumerate() {
                    if !consumer_set.contains(&oid) {
                        new_block_ops.push(oid);
                    }
                    if pos == insert_after {
                        new_block_ops.extend(inserted_ops.iter().copied());
                    }
                }
                module.blocks[bid].ops = new_block_ops;
                for cid in consumers {
                    module.ops.remove(cid);
                }

                wrapped_count += 1;
                changed = true;
                break;
            }
            if changed {
                break;
            }
        }
        if !changed {
            break;
        }
    }
    wrapped_count
}

/// Folds `mo.tensor.bundle(mo.tensor.unbundle(X)#0..N-1)` back to `X` and removes dead
/// `mo.tensor.unbundle` operations.
///
/// Returns the number of identity `mo.tensor.bundle` operations folded.
pub fn fold_identity_tensor_bundles(module: &mut Module) -> usize {
    let mut folded = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let ops_snapshot = module.blocks[bid].ops.clone();
        for &op_id in &ops_snapshot {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if op.name.full_name() != "mo.tensor.bundle"
                || op.operands.is_empty()
                || op.results.len() != 1
            {
                continue;
            }
            let Some(crate::ir::mlir::ValueDef::OpResult {
                op: unbundle_id,
                index: 0,
            }) = module.values.get(op.operands[0]).map(|v| v.def)
            else {
                continue;
            };
            let Some(unbundle_op) = module.ops.get(unbundle_id).cloned() else {
                continue;
            };
            if unbundle_op.name.full_name() != "mo.tensor.unbundle"
                || unbundle_op.operands.len() != 1
                || unbundle_op.results.len() != op.operands.len()
            {
                continue;
            }
            if unbundle_op
                .results
                .iter()
                .zip(op.operands.iter())
                .any(|(&u, &b)| u != b)
            {
                continue;
            }
            let source_bundle = unbundle_op.operands[0];
            let bundle_res = op.results[0];
            let all_ops: Vec<OpId> = module.ops.keys().collect();
            for oid in all_ops {
                for opnd in &mut module.ops[oid].operands {
                    if *opnd == bundle_res {
                        *opnd = source_bundle;
                    }
                }
            }
            module.blocks[bid].ops.retain(|&id| id != op_id);
            module.ops.remove(op_id);
            folded += 1;

            // Remove `unbundle_id` if none of its results have remaining uses:
            let still_used = module.ops.values().any(|other| {
                other
                    .operands
                    .iter()
                    .any(|v| unbundle_op.results.contains(v))
            });
            if !still_used {
                for b in module.blocks.values_mut() {
                    b.ops.retain(|&id| id != unbundle_id);
                }
                module.ops.remove(unbundle_id);
            }
        }
    }
    folded
}

pub(super) struct ParallelOperandsLayout {
    bundle_inputs: Vec<ValueId>,
    buffers: Vec<ValueId>,
    in_chain: Option<ValueId>,
    bundle_results: Vec<ValueId>,
    out_chain: Option<ValueId>,
}

pub(super) struct SelectedParallelMerge {
    prod_id: OpId,
    prod_layout: ParallelOperandsLayout,
    shared_inputs: Vec<Option<usize>>,
    buffers: Vec<ValueId>,
    in_chain: Option<ValueId>,
}

pub(super) fn inspect_parallel_op(module: &Module, op_id: OpId) -> Option<ParallelOperandsLayout> {
    let op = module.ops.get(op_id)?;
    if op.name.full_name() != "mo.parallel" || op.regions.len() != 1 {
        return None;
    }
    let reg = module.regions.get(op.regions[0])?;
    if reg.blocks.len() != 1 {
        return None;
    }
    let mut bundle_inputs = Vec::new();
    let mut buffers = Vec::new();
    let mut in_chain = None;
    for &v in &op.operands {
        let ty = &module.values.get(v)?.ty;
        if is_chain_type(ty) {
            in_chain = Some(v);
        } else if is_mutable_operand_type(ty) {
            buffers.push(v);
        } else {
            bundle_inputs.push(v);
        }
    }
    let mut bundle_results = Vec::new();
    let mut out_chain = None;
    for &r in &op.results {
        let ty = &module.values.get(r)?.ty;
        if is_chain_type(ty) {
            out_chain = Some(r);
        } else {
            bundle_results.push(r);
        }
    }
    Some(ParallelOperandsLayout {
        bundle_inputs,
        buffers,
        in_chain,
        bundle_results,
        out_chain,
    })
}

pub(super) fn value_transitively_depends_on_op(
    module: &Module,
    root: ValueId,
    target_op: OpId,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        if !visited.insert(v) {
            continue;
        }
        let Some(crate::ir::mlir::ValueDef::OpResult { op: def_op, .. }) =
            module.values.get(v).map(|val| val.def)
        else {
            continue;
        };
        if def_op == target_op {
            return true;
        }
        if let Some(op) = module.ops.get(def_op) {
            stack.extend(op.operands.iter().copied());
        }
    }
    false
}

/// Fuses adjacent `mo.parallel` producer and consumer operations in the same block when the
/// consumer takes one or more producer bundle results directly as inputs, no dependence cycle
/// exists through non-shared consumer inputs, and buffer/chain constraints match.
///
/// Drops producer bundle outputs used exclusively by the consumer and performs at most one
/// merge per walk iteration until fixed point. Returns the number of parallel merges performed.
pub fn merge_adjacent_parallels(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, Location, OpName, Operation, Region, RegionId, Value, ValueDef};

    let mut merged_count = 0usize;
    loop {
        let mut merged_in_iter = false;
        let use_counts = {
            let mut counts: std::collections::HashMap<ValueId, usize> =
                std::collections::HashMap::new();
            for op in module.ops.values() {
                for &v in &op.operands {
                    *counts.entry(v).or_insert(0) += 1;
                }
            }
            counts
        };

        let block_ids: Vec<BlockId> = module.blocks.keys().collect();
        for bid in block_ids {
            let ops_snapshot = module.blocks[bid].ops.clone();
            for &consumer_id in &ops_snapshot {
                let Some(consumer_layout) = inspect_parallel_op(module, consumer_id) else {
                    continue;
                };
                // Collect candidate producer `mo.parallel` ops in consumer-input order:
                let mut candidates_in_order = Vec::new();
                let mut seen_producers = std::collections::HashSet::new();
                for &inp in &consumer_layout.bundle_inputs {
                    if let Some(ValueDef::OpResult { op: prod_id, .. }) =
                        module.values.get(inp).map(|v| v.def)
                        && prod_id != consumer_id
                        && module.blocks[bid].ops.contains(&prod_id)
                        && inspect_parallel_op(module, prod_id).is_some()
                        && seen_producers.insert(prod_id)
                    {
                        candidates_in_order.push(prod_id);
                    }
                }

                let mut selected: Option<SelectedParallelMerge> = None;

                for prod_id in candidates_in_order {
                    let Some(prod_layout) = inspect_parallel_op(module, prod_id) else {
                        continue;
                    };
                    let mut shared_inputs = Vec::with_capacity(consumer_layout.bundle_inputs.len());
                    for &inp in &consumer_layout.bundle_inputs {
                        let shared_idx = prod_layout.bundle_results.iter().position(|&r| r == inp);
                        shared_inputs.push(shared_idx);
                    }
                    // Reject if any non-shared consumer input transitively depends on `prod_id`:
                    let has_cycle =
                        consumer_layout
                            .bundle_inputs
                            .iter()
                            .enumerate()
                            .any(|(i, &inp)| {
                                shared_inputs[i].is_none()
                                    && value_transitively_depends_on_op(module, inp, prod_id)
                            });
                    if has_cycle {
                        continue;
                    }

                    // Resolve buffer + chain plan:
                    let (buffers, in_chain) = match (
                        prod_layout.in_chain,
                        prod_layout.out_chain,
                        consumer_layout.in_chain,
                    ) {
                        (Some(p_in), Some(p_out), Some(c_in)) => {
                            if prod_layout.buffers != consumer_layout.buffers
                                || c_in != p_out
                                || use_counts.get(&p_out).copied().unwrap_or(0) != 1
                            {
                                continue;
                            }
                            (prod_layout.buffers.clone(), Some(p_in))
                        }
                        (Some(p_in), _, None) => (prod_layout.buffers.clone(), Some(p_in)),
                        (None, _, Some(c_in)) => (consumer_layout.buffers.clone(), Some(c_in)),
                        (None, _, None) => (Vec::new(), None),
                        _ => continue,
                    };

                    selected = Some(SelectedParallelMerge {
                        prod_id,
                        prod_layout,
                        shared_inputs,
                        buffers,
                        in_chain,
                    });
                    break;
                }

                let Some(SelectedParallelMerge {
                    prod_id,
                    prod_layout,
                    shared_inputs,
                    buffers: merged_buffers,
                    in_chain: merged_in_chain,
                }) = selected
                else {
                    continue;
                };

                let prod_body_reg = module.ops[prod_id].regions[0];
                let prod_body_blk = module.regions[prod_body_reg].blocks[0];
                let cons_body_reg = module.ops[consumer_id].regions[0];
                let cons_body_blk = module.regions[cons_body_reg].blocks[0];

                let Some(&prod_yield_id) = module.blocks[prod_body_blk].ops.last() else {
                    continue;
                };
                let Some(&cons_yield_id) = module.blocks[cons_body_blk].ops.last() else {
                    continue;
                };
                if module.ops[prod_yield_id].name.full_name() != "mo.yield"
                    || module.ops[cons_yield_id].name.full_name() != "mo.yield"
                {
                    continue;
                }

                let prod_num_bundles = prod_layout.bundle_results.len();
                let cons_num_bundles = consumer_layout.bundle_results.len();

                let mut drop_prod_yield = vec![false; prod_num_bundles];
                for (i, &res_v) in prod_layout.bundle_results.iter().enumerate() {
                    let has_external_user = module
                        .ops
                        .iter()
                        .any(|(oid, op)| oid != consumer_id && op.operands.contains(&res_v));
                    drop_prod_yield[i] = !has_external_user;
                }

                let mut merged_bundle_inputs = prod_layout.bundle_inputs.clone();
                let mut cons_appended_arg_idx = vec![0usize; consumer_layout.bundle_inputs.len()];
                let mut appended_idx = prod_layout.bundle_inputs.len();
                for (bi, &inp) in consumer_layout.bundle_inputs.iter().enumerate() {
                    if shared_inputs[bi].is_none() {
                        merged_bundle_inputs.push(inp);
                        cons_appended_arg_idx[bi] = appended_idx;
                        appended_idx += 1;
                    }
                }

                let merged_blk = module.blocks.insert(Block {
                    id: BlockId::default(),
                    label: None,
                    arguments: Vec::new(),
                    ops: Vec::new(),
                });
                module.blocks[merged_blk].id = merged_blk;

                let prod_blk_args = module.blocks[prod_body_blk].arguments.clone();
                let cons_blk_args = module.blocks[cons_body_blk].arguments.clone();
                let mut merged_blk_args = Vec::new();
                for &pa in &prod_blk_args[..prod_layout.bundle_inputs.len()] {
                    let ty = module.values[pa].ty.clone();
                    let arg_v = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: merged_blk,
                            index: merged_blk_args.len() as u32,
                        },
                    });
                    module.values[arg_v].id = arg_v;
                    merged_blk_args.push(arg_v);
                }
                for (bi, &ca) in cons_blk_args[..consumer_layout.bundle_inputs.len()]
                    .iter()
                    .enumerate()
                {
                    if shared_inputs[bi].is_none() {
                        let ty = module.values[ca].ty.clone();
                        let arg_v = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty,
                            def: ValueDef::BlockArgument {
                                block: merged_blk,
                                index: merged_blk_args.len() as u32,
                            },
                        });
                        module.values[arg_v].id = arg_v;
                        merged_blk_args.push(arg_v);
                    }
                }
                let merged_chain_arg = if let Some(ch_v) = merged_in_chain {
                    let ty = module.values[ch_v].ty.clone();
                    let arg_v = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: merged_blk,
                            index: merged_blk_args.len() as u32,
                        },
                    });
                    module.values[arg_v].id = arg_v;
                    merged_blk_args.push(arg_v);
                    Some(arg_v)
                } else {
                    None
                };
                module.blocks[merged_blk].arguments = merged_blk_args.clone();

                let mut mapping: std::collections::HashMap<ValueId, ValueId> =
                    std::collections::HashMap::new();
                for i in 0..prod_layout.bundle_inputs.len() {
                    mapping.insert(prod_blk_args[i], merged_blk_args[i]);
                }
                if prod_layout.in_chain.is_some()
                    && let (Some(&p_ch_arg), Some(m_ch_arg)) =
                        (prod_blk_args.last(), merged_chain_arg)
                {
                    mapping.insert(p_ch_arg, m_ch_arg);
                }

                // Clone producer body ops (excluding terminator):
                let prod_body_ops: Vec<OpId> = module.blocks[prod_body_blk]
                    .ops
                    .iter()
                    .copied()
                    .filter(|&id| id != prod_yield_id)
                    .collect();
                for old_op_id in prod_body_ops {
                    let old_op = module.ops[old_op_id].clone();
                    let new_operands: smallvec::SmallVec<[ValueId; 4]> = old_op
                        .operands
                        .iter()
                        .map(|v| mapping.get(v).copied().unwrap_or(*v))
                        .collect();
                    let mut new_results = smallvec::SmallVec::<[ValueId; 2]>::new();
                    for &old_r in &old_op.results {
                        let r_ty = module.values[old_r].ty.clone();
                        let new_r = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: r_ty,
                            def: ValueDef::BlockArgument {
                                block: merged_blk,
                                index: 0,
                            },
                        });
                        module.values[new_r].id = new_r;
                        mapping.insert(old_r, new_r);
                        new_results.push(new_r);
                    }
                    let new_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: old_op.name,
                        operands: new_operands,
                        results: new_results.clone(),
                        attributes: old_op.attributes,
                        regions: Vec::new(),
                        custom_syntax: old_op.custom_syntax,
                        location: old_op.location,
                    });
                    module.ops[new_op].id = new_op;
                    for (idx, &nr) in new_results.iter().enumerate() {
                        module.values[nr].def = ValueDef::OpResult {
                            op: new_op,
                            index: idx as u32,
                        };
                    }
                    module.blocks[merged_blk].ops.push(new_op);
                }

                let prod_yield_operands = module.ops[prod_yield_id].operands.clone();
                for (bi, &ca) in cons_blk_args[..consumer_layout.bundle_inputs.len()]
                    .iter()
                    .enumerate()
                {
                    if let Some(prod_idx) = shared_inputs[bi] {
                        let py = prod_yield_operands[prod_idx];
                        let mapped_py = mapping.get(&py).copied().unwrap_or(py);
                        mapping.insert(ca, mapped_py);
                    } else {
                        mapping.insert(ca, merged_blk_args[cons_appended_arg_idx[bi]]);
                    }
                }
                if consumer_layout.in_chain.is_some()
                    && let Some(&c_ch_arg) = cons_blk_args.last()
                {
                    if prod_layout.in_chain.is_some() {
                        if let Some(&py_last) = prod_yield_operands.last() {
                            let mapped_ch = mapping.get(&py_last).copied().unwrap_or(py_last);
                            mapping.insert(c_ch_arg, mapped_ch);
                        }
                    } else if let Some(m_ch_arg) = merged_chain_arg {
                        mapping.insert(c_ch_arg, m_ch_arg);
                    }
                }

                // Clone consumer body ops (excluding terminator):
                let cons_body_ops: Vec<OpId> = module.blocks[cons_body_blk]
                    .ops
                    .iter()
                    .copied()
                    .filter(|&id| id != cons_yield_id)
                    .collect();
                for old_op_id in cons_body_ops {
                    let old_op = module.ops[old_op_id].clone();
                    let new_operands: smallvec::SmallVec<[ValueId; 4]> = old_op
                        .operands
                        .iter()
                        .map(|v| mapping.get(v).copied().unwrap_or(*v))
                        .collect();
                    let mut new_results = smallvec::SmallVec::<[ValueId; 2]>::new();
                    for &old_r in &old_op.results {
                        let r_ty = module.values[old_r].ty.clone();
                        let new_r = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: r_ty,
                            def: ValueDef::BlockArgument {
                                block: merged_blk,
                                index: 0,
                            },
                        });
                        module.values[new_r].id = new_r;
                        mapping.insert(old_r, new_r);
                        new_results.push(new_r);
                    }
                    let new_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: old_op.name,
                        operands: new_operands,
                        results: new_results.clone(),
                        attributes: old_op.attributes,
                        regions: Vec::new(),
                        custom_syntax: old_op.custom_syntax,
                        location: old_op.location,
                    });
                    module.ops[new_op].id = new_op;
                    for (idx, &nr) in new_results.iter().enumerate() {
                        module.values[nr].def = ValueDef::OpResult {
                            op: new_op,
                            index: idx as u32,
                        };
                    }
                    module.blocks[merged_blk].ops.push(new_op);
                }

                let cons_yield_operands = module.ops[cons_yield_id].operands.clone();
                let mut merged_yield_operands = smallvec::SmallVec::<[ValueId; 4]>::new();
                for (i, &dropped) in drop_prod_yield.iter().enumerate() {
                    if !dropped {
                        let py = prod_yield_operands[i];
                        merged_yield_operands.push(mapping.get(&py).copied().unwrap_or(py));
                    }
                }
                for &cy in &cons_yield_operands[..cons_num_bundles] {
                    merged_yield_operands.push(mapping.get(&cy).copied().unwrap_or(cy));
                }
                if merged_in_chain.is_some() {
                    let ch_src = if consumer_layout.in_chain.is_some() {
                        cons_yield_operands.last().copied()
                    } else {
                        prod_yield_operands.last().copied()
                    };
                    if let Some(ch_v) = ch_src {
                        merged_yield_operands.push(mapping.get(&ch_v).copied().unwrap_or(ch_v));
                    }
                }
                let merged_yield = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: OpName::new("mo", "yield"),
                    operands: merged_yield_operands,
                    results: smallvec::SmallVec::new(),
                    attributes: indexmap::IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                module.ops[merged_yield].id = merged_yield;
                module.blocks[merged_blk].ops.push(merged_yield);

                let merged_reg = module.regions.insert(Region {
                    id: RegionId::default(),
                    blocks: vec![merged_blk],
                });
                module.regions[merged_reg].id = merged_reg;

                let mut merged_operands = smallvec::SmallVec::<[ValueId; 4]>::new();
                merged_operands.extend(merged_bundle_inputs);
                merged_operands.extend(merged_buffers);
                if let Some(ch_in) = merged_in_chain {
                    merged_operands.push(ch_in);
                }

                let mut prod_merged_res = vec![None; prod_num_bundles];
                let mut merged_results = smallvec::SmallVec::<[ValueId; 2]>::new();
                for (i, &dropped) in drop_prod_yield.iter().enumerate() {
                    if !dropped {
                        let ty = module.values[prod_layout.bundle_results[i]].ty.clone();
                        let r = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty,
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 0,
                            },
                        });
                        module.values[r].id = r;
                        prod_merged_res[i] = Some(r);
                        merged_results.push(r);
                    }
                }
                let mut cons_merged_res = Vec::with_capacity(cons_num_bundles);
                for &old_cr in &consumer_layout.bundle_results {
                    let ty = module.values[old_cr].ty.clone();
                    let r = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[r].id = r;
                    cons_merged_res.push(r);
                    merged_results.push(r);
                }
                let merged_out_chain = if let Some(ch_in) = merged_in_chain {
                    let ty = module.values[ch_in].ty.clone();
                    let r = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[r].id = r;
                    merged_results.push(r);
                    Some(r)
                } else {
                    None
                };

                let merged_parallel = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: OpName::new("mo", "parallel"),
                    operands: merged_operands,
                    results: merged_results.clone(),
                    attributes: module.ops[prod_id].attributes.clone(),
                    regions: vec![merged_reg],
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                module.ops[merged_parallel].id = merged_parallel;
                for (idx, &mr) in merged_results.iter().enumerate() {
                    module.values[mr].def = ValueDef::OpResult {
                        op: merged_parallel,
                        index: idx as u32,
                    };
                }

                // Replace uses across module:
                let all_ops: Vec<OpId> = module.ops.keys().collect();
                for (i, maybe_r) in prod_merged_res.iter().enumerate() {
                    if let Some(new_r) = *maybe_r {
                        let old_r = prod_layout.bundle_results[i];
                        for &oid in &all_ops {
                            for opnd in &mut module.ops[oid].operands {
                                if *opnd == old_r {
                                    *opnd = new_r;
                                }
                            }
                        }
                    }
                }
                for (i, &new_cr) in cons_merged_res.iter().enumerate() {
                    let old_cr = consumer_layout.bundle_results[i];
                    for &oid in &all_ops {
                        for opnd in &mut module.ops[oid].operands {
                            if *opnd == old_cr {
                                *opnd = new_cr;
                            }
                        }
                    }
                }
                if let Some(new_ch) = merged_out_chain {
                    for old_ch in [prod_layout.out_chain, consumer_layout.out_chain]
                        .into_iter()
                        .flatten()
                    {
                        for &oid in &all_ops {
                            for opnd in &mut module.ops[oid].operands {
                                if *opnd == old_ch {
                                    *opnd = new_ch;
                                }
                            }
                        }
                    }
                }

                // Replace `consumer_id` in block order with `merged_parallel` and erase `prod_id`:
                for slot in &mut module.blocks[bid].ops {
                    if *slot == consumer_id {
                        *slot = merged_parallel;
                    }
                }
                module.blocks[bid].ops.retain(|&id| id != prod_id);
                module.ops.remove(prod_id);
                module.ops.remove(consumer_id);

                merged_count += 1;
                merged_in_iter = true;
                break;
            }
            if merged_in_iter {
                break;
            }
        }
        if !merged_in_iter {
            break;
        }
    }
    merged_count
}

pub(super) fn is_layout_propagating_op(op: &crate::ir::mlir::Operation) -> bool {
    if matches!(
        op.attributes.get("same_operands_and_result_type"),
        Some(Attribute::Bool(true))
    ) {
        return true;
    }
    matches!(
        op.name.full_name().as_str(),
        "mo.transfer"
            | "mo.add"
            | "mo.sub"
            | "mo.mul"
            | "mo.div"
            | "mo.relu"
            | "mo.exp"
            | "mo.tanh"
            | "mo.sigmoid"
            | "mo.neg"
            | "mo.abs"
            | "mo.sqrt"
            | "mo.rsqrt"
    )
}

pub(super) fn constrain_value_layout(
    inferred: &mut std::collections::HashMap<ValueId, String>,
    value_id: ValueId,
    layout: &str,
) -> Result<(), String> {
    if let Some(existing) = inferred.get(&value_id) {
        if existing != layout {
            return Err(format!(
                "Attempting to constrain value {value_id:?} to layout '{layout}' \
                 when it is already constrained to layout '{existing}'"
            ));
        }
        return Ok(());
    }
    inferred.insert(value_id, layout.to_string());
    Ok(())
}

/// Seeds tensor layout constraints from `mo.conv` (`NHWC`/`FCRS` for 2D, `NDHWC`/`FCQRS` for 3D),
/// `mo.conv_transpose` (`NHWC`/`RSFC` for 2D, `NDHWC`/`QRSFC` for 3D), and `mo.matmul` (`KN` for
/// weights), then propagates them bidirectionally across `mo.guard`, `mo.transfer`, and
/// layout-preserving elementwise operations.
///
/// Annotates defining operations with `"inferred_layout"` attributes and returns the map of
/// constrained `ValueId`s to layout strings, or `Err` on unsupported conv rank or conflicting
/// layout constraints.
pub fn infer_tensor_layouts(
    module: &mut Module,
) -> Result<std::collections::HashMap<ValueId, String>, String> {
    let mut inferred: std::collections::HashMap<ValueId, String> = std::collections::HashMap::new();

    let op_ids: Vec<OpId> = module.ops.keys().collect();
    for oid in op_ids {
        let op = &module.ops[oid];
        let full = op.name.full_name();
        if (full == "mo.conv" || full == "mo.conv_transpose") && op.operands.len() >= 2 {
            let input_v = op.operands[0];
            let filter_v = op.operands[1];
            let input_rank = module
                .values
                .get(input_v)
                .and_then(|v| v.ty.as_tensor())
                .map(|t| t.shape.len())
                .unwrap_or(0);
            let filter_rank = module
                .values
                .get(filter_v)
                .and_then(|v| v.ty.as_tensor())
                .map(|t| t.shape.len())
                .unwrap_or(0);
            let (in_layout, flt_layout) = match (full.as_str(), input_rank, filter_rank) {
                ("mo.conv", 4, 4) => ("NHWC", "FCRS"),
                ("mo.conv", 5, 5) => ("NDHWC", "FCQRS"),
                ("mo.conv_transpose", 4, 4) => ("NHWC", "RSFC"),
                ("mo.conv_transpose", 5, 5) => ("NDHWC", "QRSFC"),
                _ => {
                    return Err(format!(
                        "Unsupported rank pair ({input_rank}, {filter_rank}) for {full}"
                    ));
                }
            };
            constrain_value_layout(&mut inferred, input_v, in_layout)?;
            constrain_value_layout(&mut inferred, filter_v, flt_layout)?;
        } else if full == "mo.matmul" && op.operands.len() >= 2 {
            let weights_v = op.operands[1];
            if module
                .values
                .get(weights_v)
                .and_then(|v| v.ty.as_tensor())
                .is_some()
            {
                constrain_value_layout(&mut inferred, weights_v, "KN")?;
            }
        }
    }

    let mut worklist: Vec<ValueId> = inferred.keys().copied().collect();
    let mut visited: std::collections::HashSet<ValueId> = worklist.iter().copied().collect();

    while let Some(val_id) = worklist.pop() {
        let Some(layout) = inferred.get(&val_id).cloned() else {
            continue;
        };
        let mut affected_ops: Vec<OpId> = module
            .ops
            .iter()
            .filter_map(|(oid, op)| op.operands.contains(&val_id).then_some(oid))
            .collect();
        if let Some(crate::ir::mlir::ValueDef::OpResult { op: def_op, .. }) =
            module.values.get(val_id).map(|v| v.def)
        {
            affected_ops.push(def_op);
        }

        for oid in affected_ops {
            let op = &module.ops[oid];
            let full = op.name.full_name();
            let propagate_vals: Vec<ValueId> = if full == "mo.guard" {
                op.operands
                    .iter()
                    .skip(1)
                    .chain(op.results.iter())
                    .copied()
                    .collect()
            } else if is_layout_propagating_op(op) {
                op.operands
                    .iter()
                    .chain(op.results.iter())
                    .copied()
                    .collect()
            } else {
                Vec::new()
            };

            for pv in propagate_vals {
                if module
                    .values
                    .get(pv)
                    .and_then(|v| v.ty.as_tensor())
                    .is_some()
                {
                    constrain_value_layout(&mut inferred, pv, &layout)?;
                    if visited.insert(pv) {
                        worklist.push(pv);
                    }
                }
            }
        }
    }

    for (&val_id, layout) in &inferred {
        if let Some(crate::ir::mlir::ValueDef::OpResult { op: def_op, .. }) =
            module.values.get(val_id).map(|v| v.def)
            && let Some(op) = module.ops.get_mut(def_op)
        {
            op.attributes.insert(
                "inferred_layout".to_string(),
                Attribute::String(layout.clone()),
            );
        }
    }

    Ok(inferred)
}

/// Inserts `mo.stable.create` wrappers for tensor operands feeding kernel `build_graph`
/// arguments that require stable tensor storage inside device-graph functions
/// (`is_device_graph = true`).
///
/// Looks up `kgen.kernel_decl` declarations (by `mo.custom` `"symbol"` or built-in op name) and
/// reads their `stable_arg_indices` integer array attribute (or per-op `stable_operand_indices`).
/// Returns the number of `mo.stable.create` operations inserted, or `Err` if a required stable
/// operand is not a tensor.
pub fn mark_stable_tensors(module: &mut Module) -> Result<usize, String> {
    use crate::ir::mlir::{Location, OpName, Operation, Value, ValueDef};

    // Collect `kgen.kernel_decl` -> stable operand indices:
    let mut decl_stable_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for op in module.ops.values() {
        if op.name.full_name() == "kgen.kernel_decl" {
            let sym = op
                .attributes
                .get("sym_name")
                .or_else(|| op.attributes.get("kernel_name"))
                .and_then(Attribute::as_str);
            if let Some(sym_str) = sym
                && let Some(Attribute::Array(items)) = op.attributes.get("stable_arg_indices")
            {
                let indices: Vec<usize> = items
                    .iter()
                    .filter_map(|a| match a {
                        Attribute::Int(i) if *i >= 0 => Some(*i as usize),
                        _ => None,
                    })
                    .collect();
                if !indices.is_empty() {
                    decl_stable_indices.insert(sym_str.to_string(), indices);
                }
            }
        }
    }

    let mut inserted_count = 0usize;
    let func_ids = module.functions();
    for fn_id in func_ids {
        let is_device_graph = matches!(
            module.ops[fn_id].attributes.get("is_device_graph"),
            Some(Attribute::Bool(true))
        );
        if !is_device_graph {
            continue;
        }
        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                let ops_snapshot = module.blocks[bid].ops.clone();
                for op_id in ops_snapshot {
                    let op = module.ops[op_id].clone();
                    let full = op.name.full_name();
                    if matches!(
                        full.as_str(),
                        "func.return" | "return" | "mo.output" | "mo.yield" | "mo.stable.create"
                    ) {
                        continue;
                    }
                    let reg_name = if full == "mo.custom" {
                        op.attributes
                            .get("symbol")
                            .and_then(Attribute::as_str)
                            .unwrap_or(full.as_str())
                            .to_string()
                    } else {
                        full
                    };

                    let stable_positions: Vec<usize> = if let Some(Attribute::Array(items)) =
                        op.attributes.get("stable_operand_indices")
                    {
                        items
                            .iter()
                            .filter_map(|a| match a {
                                Attribute::Int(i) if *i >= 0 => Some(*i as usize),
                                _ => None,
                            })
                            .collect()
                    } else if let Some(idxs) = decl_stable_indices.get(&reg_name) {
                        idxs.clone()
                    } else {
                        continue;
                    };

                    let mut pre_ops = Vec::new();
                    for pos in stable_positions {
                        if pos >= module.ops[op_id].operands.len() {
                            continue;
                        }
                        let orig_val = module.ops[op_id].operands[pos];
                        let val_ty = module
                            .values
                            .get(orig_val)
                            .map(|v| v.ty.clone())
                            .ok_or_else(|| format!("Missing SSA value at operand {pos}"))?;
                        if val_ty.as_tensor().is_none() {
                            return Err(format!(
                                "Operand at position {pos} feeds a stable tensor argument of \
                                 kernel '{reg_name}' but is not a tensor"
                            ));
                        }
                        // Skip if already produced by `mo.stable.create`:
                        if let Some(ValueDef::OpResult { op: def_op, .. }) =
                            module.values.get(orig_val).map(|v| v.def)
                            && module
                                .ops
                                .get(def_op)
                                .is_some_and(|d| d.name.full_name() == "mo.stable.create")
                        {
                            continue;
                        }

                        let stable_val = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: val_ty,
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 0,
                            },
                        });
                        module.values[stable_val].id = stable_val;
                        let mut attrs = indexmap::IndexMap::new();
                        attrs.insert("is_stable".to_string(), Attribute::Bool(true));
                        let stable_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("mo", "stable.create"),
                            operands: smallvec::smallvec![orig_val],
                            results: smallvec::smallvec![stable_val],
                            attributes: attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[stable_op].id = stable_op;
                        module.values[stable_val].def = ValueDef::OpResult {
                            op: stable_op,
                            index: 0,
                        };
                        module.ops[op_id].operands[pos] = stable_val;
                        pre_ops.push(stable_op);
                        inserted_count += 1;
                    }

                    if !pre_ops.is_empty()
                        && let Some(slot_pos) =
                            module.blocks[bid].ops.iter().position(|&id| id == op_id)
                    {
                        for (offset, s_op) in pre_ops.into_iter().enumerate() {
                            module.blocks[bid].ops.insert(slot_pos + offset, s_op);
                        }
                    }
                }
            }
        }
    }

    Ok(inserted_count)
}
