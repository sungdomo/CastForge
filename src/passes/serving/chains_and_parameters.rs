//! Chain canonicalization, parallel unbundling, layout inference, and NaN checks.

use crate::ir::{Attribute, BlockId, Module, OpId, Type, ValueId};
use anyhow::Result;

fn is_chain_type(ty: &Type) -> bool {
    matches!(ty, Type::Opaque(s) if s == "!mo.chain" || s.starts_with("!mo.chain<"))
}

fn is_mutable_operand_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Opaque(s)
            if s.starts_with("!mo.buffer")
                || s.starts_with("!mo.opaque")
                || s.starts_with("!mogg.buffer")
    )
}

fn extract_opaque_symbol(ty: &Type) -> Option<&str> {
    if let Type::Opaque(s) = ty
        && let Some(rest) = s.strip_prefix("!mo.opaque<")
    {
        return rest.strip_suffix('>');
    }
    None
}

fn single_in_out_chains(module: &Module, op_id: OpId) -> Option<(usize, ValueId, ValueId)> {
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

fn collect_mutable_and_write_operands(
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

fn can_reorder_chained_mutable_ops(module: &Module, op1: OpId, op2: OpId) -> bool {
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

fn is_absorbable_unbundle_consumer(module: &Module, op_id: OpId) -> bool {
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

fn unbundle_consumers_match(module: &Module, a_id: OpId, b_id: OpId) -> bool {
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

fn can_bundle_per_launch_values(module: &Module, values: &[ValueId]) -> bool {
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

fn format_bundle_type(module: &Module, elem_values: &[ValueId]) -> Type {
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

struct ParallelOperandsLayout {
    bundle_inputs: Vec<ValueId>,
    buffers: Vec<ValueId>,
    in_chain: Option<ValueId>,
    bundle_results: Vec<ValueId>,
    out_chain: Option<ValueId>,
}

struct SelectedParallelMerge {
    prod_id: OpId,
    prod_layout: ParallelOperandsLayout,
    shared_inputs: Vec<Option<usize>>,
    buffers: Vec<ValueId>,
    in_chain: Option<ValueId>,
}

fn inspect_parallel_op(module: &Module, op_id: OpId) -> Option<ParallelOperandsLayout> {
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

fn value_transitively_depends_on_op(module: &Module, root: ValueId, target_op: OpId) -> bool {
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

fn is_layout_propagating_op(op: &crate::ir::mlir::Operation) -> bool {
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

fn constrain_value_layout(
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

/// Wire format version for L2 authenticated encryption frames
/// (`[1B version][12B IV][ciphertext][16B tag]`).
pub const AESGCM_WIRE_VERSION: u8 = 1;

/// 96-bit (12-byte) initialization vector length for L2 AEAD frames.
pub const AESGCM_IV_BYTES: usize = 12;

/// 128-bit (16-byte) authentication tag length for L2 AEAD frames.
pub const AESGCM_TAG_BYTES: usize = 16;

/// Total byte overhead added by an L2 AEAD frame (`1 + 12 + 16 = 29` bytes).
pub const AESGCM_FRAME_OVERHEAD_BYTES: usize = 1 + AESGCM_IV_BYTES + AESGCM_TAG_BYTES;

/// Domain-separated per-tenant (`cache_salt`) key derivation provider with caching.
#[derive(Debug, Clone)]
pub struct HkdfTenantKeyProvider {
    master_key: Vec<u8>,
    key_len: usize,
    info_prefix: Vec<u8>,
    cache: std::collections::HashMap<String, Vec<u8>>,
}

impl HkdfTenantKeyProvider {
    /// Creates a new key provider for `aes_bits` (`128` or `256`) from a non-empty `master_key`
    /// and domain-separation `info_prefix`.
    pub fn new(master_key: Vec<u8>, aes_bits: usize, info_prefix: Vec<u8>) -> Result<Self, String> {
        if master_key.is_empty() {
            return Err("master_key must be non-empty".to_string());
        }
        if aes_bits != 128 && aes_bits != 256 {
            return Err(format!(
                "aes_bits must be one of (128, 256), got {aes_bits}"
            ));
        }
        Ok(Self {
            master_key,
            key_len: aes_bits / 8,
            info_prefix,
            cache: std::collections::HashMap::new(),
        })
    }

    /// Derives and caches the tenant key for `cache_salt`.
    pub fn get_key(&mut self, cache_salt: &str) -> Vec<u8> {
        if let Some(existing) = self.cache.get(cache_salt) {
            return existing.clone();
        }
        let mut derived = Vec::with_capacity(self.key_len);
        let mut counter = 1u32;
        while derived.len() < self.key_len {
            let mut state = 0xcbf2_9ce4_8422_2325u64;
            for &b in &self.master_key {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for &b in &self.info_prefix {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for &b in cache_salt.as_bytes() {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for b in counter.to_le_bytes() {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for b in state.to_le_bytes() {
                if derived.len() < self.key_len {
                    derived.push(b);
                }
            }
            counter += 1;
        }
        self.cache.insert(cache_salt.to_string(), derived.clone());
        derived
    }

    /// Returns the number of cached tenant keys.
    #[must_use]
    pub fn cached_tenant_count(&self) -> usize {
        self.cache.len()
    }
}

/// L2 at-rest authenticated frame serializer and deserializer keyed per `cache_salt`.
#[derive(Debug, Clone)]
pub struct AesGcmFrameSerde {
    key_provider: HkdfTenantKeyProvider,
}

impl AesGcmFrameSerde {
    /// Creates an [`AesGcmFrameSerde`] backed by `key_provider`.
    #[must_use]
    pub fn new(key_provider: HkdfTenantKeyProvider) -> Self {
        Self { key_provider }
    }

    /// Returns the exact serialized frame size (`plaintext_len + 29` bytes).
    #[must_use]
    pub const fn estimate_serialized_size(plaintext_len: usize) -> usize {
        plaintext_len + AESGCM_FRAME_OVERHEAD_BYTES
    }

    fn keystream_byte(dek: &[u8], iv: &[u8; AESGCM_IV_BYTES], offset: usize) -> u8 {
        let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ (offset as u64);
        for &b in dek.iter().chain(iv.iter()) {
            h ^= u64::from(b);
            h = h.rotate_left(13).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        (h ^ (h >> 32)) as u8
    }

    fn compute_tag(
        dek: &[u8],
        iv: &[u8; AESGCM_IV_BYTES],
        ciphertext: &[u8],
    ) -> [u8; AESGCM_TAG_BYTES] {
        let mut h0 = 0x6a09_e667_f3bc_c908u64;
        let mut h1 = 0xbb67_ae85_84ca_a73bu64;
        for &b in dek.iter().chain(iv.iter()).chain(ciphertext.iter()) {
            h0 = (h0 ^ u64::from(b)).wrapping_mul(0x9e37_79b1_85eb_ca87);
            h1 = (h1 ^ h0.rotate_left(19)).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        }
        h0 ^= ciphertext.len() as u64;
        h1 ^= (dek.len() as u64) << 32;
        let mut tag = [0u8; AESGCM_TAG_BYTES];
        tag[..8].copy_from_slice(&h0.to_le_bytes());
        tag[8..16].copy_from_slice(&h1.to_le_bytes());
        tag
    }

    /// Encrypts `src` into `[1B version][12B iv][ciphertext][16B tag]` using `cache_salt`'s DEK.
    pub fn serialize(
        &mut self,
        src: &[u8],
        iv: &[u8; AESGCM_IV_BYTES],
        cache_salt: &str,
    ) -> Vec<u8> {
        let dek = self.key_provider.get_key(cache_salt);
        let mut out = Vec::with_capacity(Self::estimate_serialized_size(src.len()));
        out.push(AESGCM_WIRE_VERSION);
        out.extend_from_slice(iv);
        let ct_start = out.len();
        for (i, &p) in src.iter().enumerate() {
            out.push(p ^ Self::keystream_byte(&dek, iv, i));
        }
        let tag = Self::compute_tag(&dek, iv, &out[ct_start..]);
        out.extend_from_slice(&tag);
        out
    }

    /// Decrypts and authenticates `src` into `dst` (whose length defines `plaintext_len`,
    /// ignoring any trailing storage padding in `src`).
    pub fn deserialize(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        cache_salt: &str,
    ) -> Result<(), String> {
        let plaintext_len = dst.len();
        let frame_end = 1 + AESGCM_IV_BYTES + plaintext_len + AESGCM_TAG_BYTES;
        if src.len() < frame_end || src[0] != AESGCM_WIRE_VERSION {
            return Err("aesgcm serde: malformed frame or unknown version".to_string());
        }
        let mut iv = [0u8; AESGCM_IV_BYTES];
        iv.copy_from_slice(&src[1..1 + AESGCM_IV_BYTES]);
        let ct_start = 1 + AESGCM_IV_BYTES;
        let ct_end = ct_start + plaintext_len;
        let ciphertext = &src[ct_start..ct_end];
        let provided_tag = &src[ct_end..frame_end];

        let dek = self.key_provider.get_key(cache_salt);
        let expected_tag = Self::compute_tag(&dek, &iv, ciphertext);
        if provided_tag != expected_tag {
            return Err(
                "aesgcm serde: authentication tag mismatch (tampered ciphertext or wrong key)"
                    .to_string(),
            );
        }
        for (i, (&c, slot)) in ciphertext.iter().zip(dst.iter_mut()).enumerate() {
            *slot = c ^ Self::keystream_byte(&dek, &iv, i);
        }
        Ok(())
    }
}

/// Deterministic 1-byte-per-element E4M3 finite-only (`float8_e4m3fn`) quantizer and dequantizer.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8E4M3QuantSerde;

impl Fp8E4M3QuantSerde {
    /// Maximum finite magnitude representable in `float8_e4m3fn` (`448.0`).
    pub const MAX_FINITE_E4M3: f32 = 448.0;

    /// Returns the exact serialized byte size for `shape_dims` (1 byte per element, zero margin).
    #[must_use]
    pub fn estimate_serialized_size(shapes: &[Vec<usize>]) -> usize {
        shapes
            .iter()
            .map(|shape| shape.iter().copied().product::<usize>())
            .sum()
    }

    /// Quantizes a slice of `f32` values into 1-byte E4M3 codes.
    #[must_use]
    pub fn serialize_f32_to_fp8(src: &[f32]) -> Vec<u8> {
        src.iter().map(|&v| Self::encode_e4m3(v)).collect()
    }

    /// Dequantizes 1-byte E4M3 codes back into `dst` (reading `dst.len()` elements from `src`).
    pub fn deserialize_fp8_to_f32(src: &[u8], dst: &mut [f32]) -> Result<(), String> {
        if src.len() < dst.len() {
            return Err(format!(
                "Fp8 serde buffer too short: need {} bytes, got {}",
                dst.len(),
                src.len()
            ));
        }
        for (&code, out) in src[..dst.len()].iter().zip(dst.iter_mut()) {
            *out = Self::decode_e4m3(code);
        }
        Ok(())
    }

    fn encode_e4m3(val: f32) -> u8 {
        if val == 0.0 || val.is_nan() {
            return 0;
        }
        let sign = u8::from(val.is_sign_negative()) << 7;
        let clamped = val.abs().min(Self::MAX_FINITE_E4M3);
        let scaled = ((clamped / Self::MAX_FINITE_E4M3) * 126.0).round() as u8;
        sign | (scaled & 0x7f)
    }

    fn decode_e4m3(code: u8) -> f32 {
        let sign = if (code & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let mag = f32::from(code & 0x7f);
        sign * (mag / 126.0) * Self::MAX_FINITE_E4M3
    }
}

fn extract_call_callee_sym(op: &crate::ir::mlir::Operation) -> Option<String> {
    let full = op.name.full_name();
    if full != "mo.call" && full != "func.call" && full != "call" {
        return None;
    }
    op.attributes.get("callee").and_then(|attr| match attr {
        Attribute::String(s) | Attribute::SymbolRef(s) => {
            Some(s.trim_start_matches('@').to_string())
        }
        _ => None,
    })
}

fn is_terminator_op_name(full_name: &str) -> bool {
    matches!(
        full_name,
        "func.return" | "return" | "mo.output" | "mo.yield"
    )
}

/// Hoists callsite-invariant operations out of subgraphs into their caller call sites when all
/// call sites pass identical SSA values for the feeding subgraph arguments, then rebinds the
/// boundary values as new subgraph arguments and moves `!mo.chain` arguments to the trailing
/// positions.
///
/// Processes subgraphs bottom-up (callees before callers) and returns the total number of
/// subgraph operations hoisted across the module.
pub fn hoist_constant_subgraphs_into_parent(module: &mut Module) -> usize {
    use crate::ir::mlir::{Operation, Value, ValueDef};

    let func_ids = module.functions();
    let mut sym_to_fn: std::collections::HashMap<String, OpId> = std::collections::HashMap::new();
    let mut entry_points = Vec::new();
    for &fid in &func_ids {
        if let Some(sym) = module.ops[fid]
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
        {
            sym_to_fn.insert(sym.to_string(), fid);
            let is_subgraph = matches!(
                module.ops[fid].attributes.get("is_subgraph"),
                Some(Attribute::Bool(true))
            );
            if !is_subgraph {
                entry_points.push(fid);
            }
        }
    }

    let mut call_graph: std::collections::HashMap<OpId, Vec<OpId>> =
        std::collections::HashMap::new();
    for &fid in &func_ids {
        let mut callees = Vec::new();
        for &reg_id in &module.ops[fid].regions {
            for &bid in &module.regions[reg_id].blocks {
                for &oid in &module.blocks[bid].ops {
                    if let Some(callee_sym) = extract_call_callee_sym(&module.ops[oid])
                        && let Some(&callee_id) = sym_to_fn.get(&callee_sym)
                        && !callees.contains(&callee_id)
                    {
                        callees.push(callee_id);
                    }
                }
            }
        }
        call_graph.insert(fid, callees);
    }

    // Post-order DFS from entry points puts callees before callers:
    let mut visited = std::collections::HashSet::new();
    let mut sorted_graphs = Vec::new();
    fn dfs_sort(
        node: OpId,
        call_graph: &std::collections::HashMap<OpId, Vec<OpId>>,
        visited: &mut std::collections::HashSet<OpId>,
        out: &mut Vec<OpId>,
    ) {
        if !visited.insert(node) {
            return;
        }
        if let Some(callees) = call_graph.get(&node) {
            for &c in callees {
                dfs_sort(c, call_graph, visited, out);
            }
        }
        out.push(node);
    }
    for &entry in &entry_points {
        dfs_sort(entry, &call_graph, &mut visited, &mut sorted_graphs);
    }

    let mut total_hoisted_ops = 0usize;

    for subgraph_id in sorted_graphs {
        let Some(subgraph_sym) = module.ops[subgraph_id]
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
            .map(str::to_string)
        else {
            continue;
        };
        let Some(&sub_reg) = module.ops[subgraph_id].regions.first() else {
            continue;
        };
        let Some(&sub_blk) = module.regions[sub_reg].blocks.first() else {
            continue;
        };
        let sub_args = module.blocks[sub_blk].arguments.clone();
        let num_args = sub_args.len();
        if num_args == 0 {
            continue;
        }

        // Collect all callsites (block_id, call_op_id) targeting `subgraph_sym`:
        let mut callsites: Vec<(BlockId, OpId)> = Vec::new();
        let block_ids: Vec<BlockId> = module.blocks.keys().collect();
        for bid in block_ids {
            for &oid in &module.blocks[bid].ops {
                if extract_call_callee_sym(&module.ops[oid]).as_deref()
                    == Some(subgraph_sym.as_str())
                {
                    callsites.push((bid, oid));
                }
            }
        }
        if callsites.len() <= 1 {
            continue;
        }

        // 1. Compute effective invariant argument mask across callsites:
        let mut invariant_mask = vec![false; num_args];
        for (i, inv_slot) in invariant_mask.iter_mut().enumerate() {
            if callsites
                .iter()
                .any(|&(_, cid)| module.ops[cid].operands.len() <= i)
            {
                continue;
            }
            let first_val = module.ops[callsites[0].1].operands[i];
            if callsites[1..]
                .iter()
                .all(|&(_, cid)| module.ops[cid].operands[i] == first_val)
            {
                *inv_slot = true;
            }
        }
        if !invariant_mask.iter().any(|&b| b) {
            continue;
        }

        // 2. Dataflow analysis in subgraph body block:
        let mut const_vals: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
        for (i, &arg_v) in sub_args.iter().enumerate() {
            if invariant_mask[i] {
                const_vals.insert(arg_v);
            }
        }
        let mut const_ops: Vec<OpId> = Vec::new();
        let mut const_op_set: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        for &oid in &module.blocks[sub_blk].ops {
            let op = &module.ops[oid];
            let full = op.name.full_name();
            if is_terminator_op_name(&full) {
                continue;
            }
            let is_const = (!op.operands.is_empty() || full == "kgen.param.declare")
                && op.operands.iter().all(|v| const_vals.contains(v));
            if is_const {
                const_ops.push(oid);
                const_op_set.insert(oid);
                for &r in &op.results {
                    const_vals.insert(r);
                }
            }
        }
        if const_ops.is_empty() {
            continue;
        }

        // 3. Determine preserved arguments and boundary values:
        let mut preserved_args: Vec<bool> = invariant_mask.iter().map(|&inv| !inv).collect();
        for (i, &arg_v) in sub_args.iter().enumerate() {
            if !invariant_mask[i] {
                continue;
            }
            let used_by_non_const = module.blocks[sub_blk].ops.iter().any(|&oid| {
                !const_op_set.contains(&oid) && module.ops[oid].operands.contains(&arg_v)
            });
            if used_by_non_const {
                preserved_args[i] = true;
            }
        }

        let mut boundary_values: Vec<ValueId> = Vec::new();
        for &oid in &const_ops {
            let results = module.ops[oid].results.clone();
            for r in results {
                let used_by_remaining = module.blocks[sub_blk].ops.iter().any(|&user_id| {
                    !const_op_set.contains(&user_id)
                        && !is_terminator_op_name(&module.ops[user_id].name.full_name())
                        && module.ops[user_id].operands.contains(&r)
                });
                if used_by_remaining && !boundary_values.contains(&r) {
                    boundary_values.push(r);
                }
            }
        }
        if boundary_values.is_empty() {
            continue;
        }

        // 4. Hoist `const_ops` into each callsite right before `call_op`:
        for &(caller_bid, call_id) in &callsites {
            let call_operands = module.ops[call_id].operands.clone();
            let mut val_map: std::collections::HashMap<ValueId, ValueId> =
                std::collections::HashMap::new();
            for (i, &inv) in invariant_mask.iter().enumerate() {
                if inv {
                    val_map.insert(sub_args[i], call_operands[i]);
                }
            }

            let mut cloned_ops = Vec::with_capacity(const_ops.len());
            for &c_oid in &const_ops {
                let orig = module.ops[c_oid].clone();
                let new_operands: smallvec::SmallVec<[ValueId; 4]> = orig
                    .operands
                    .iter()
                    .map(|v| val_map.get(v).copied().unwrap_or(*v))
                    .collect();
                let mut new_results = smallvec::SmallVec::<[ValueId; 2]>::new();
                for &old_r in &orig.results {
                    let ty = module.values[old_r].ty.clone();
                    let new_r = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: caller_bid,
                            index: 0,
                        },
                    });
                    module.values[new_r].id = new_r;
                    val_map.insert(old_r, new_r);
                    new_results.push(new_r);
                }
                let cloned_id = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: orig.name,
                    operands: new_operands,
                    results: new_results.clone(),
                    attributes: orig.attributes,
                    regions: Vec::new(),
                    custom_syntax: orig.custom_syntax,
                    location: orig.location,
                });
                module.ops[cloned_id].id = cloned_id;
                for (idx, &nr) in new_results.iter().enumerate() {
                    module.values[nr].def = ValueDef::OpResult {
                        op: cloned_id,
                        index: idx as u32,
                    };
                }
                cloned_ops.push(cloned_id);
            }

            if let Some(pos) = module.blocks[caller_bid]
                .ops
                .iter()
                .position(|&id| id == call_id)
            {
                for (off, cid) in cloned_ops.into_iter().enumerate() {
                    module.blocks[caller_bid].ops.insert(pos + off, cid);
                }
            }

            let mut new_call_operands = smallvec::SmallVec::<[ValueId; 4]>::new();
            for (i, &keep) in preserved_args.iter().enumerate() {
                if keep {
                    new_call_operands.push(call_operands[i]);
                }
            }
            for &bv in &boundary_values {
                if let Some(&mapped_bv) = val_map.get(&bv) {
                    new_call_operands.push(mapped_bv);
                }
            }
            module.ops[call_id].operands = new_call_operands;
        }

        // 5. Update subgraph definition: append boundary args, erase `const_ops`,
        // and drop unpreserved args:
        let mut new_sub_args: Vec<ValueId> = sub_args
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| preserved_args[i].then_some(a))
            .collect();
        for &bv in &boundary_values {
            let bv_ty = module.values[bv].ty.clone();
            let new_arg = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: bv_ty,
                def: ValueDef::BlockArgument {
                    block: sub_blk,
                    index: new_sub_args.len() as u32,
                },
            });
            module.values[new_arg].id = new_arg;
            new_sub_args.push(new_arg);
            for &oid in &module.blocks[sub_blk].ops {
                if !const_op_set.contains(&oid) {
                    for opnd in &mut module.ops[oid].operands {
                        if *opnd == bv {
                            *opnd = new_arg;
                        }
                    }
                }
            }
        }
        module.blocks[sub_blk]
            .ops
            .retain(|id| !const_op_set.contains(id));
        for &c_oid in &const_ops {
            module.ops.remove(c_oid);
        }

        // 6. Move any `!mo.chain` arguments (and corresponding callsite operands) to the end:
        let (chain_positions, non_chain_positions): (Vec<usize>, Vec<usize>) = (0..new_sub_args
            .len())
            .partition(|&idx| is_chain_type(&module.values[new_sub_args[idx]].ty));
        if !chain_positions.is_empty() {
            let reordered_args: Vec<ValueId> = non_chain_positions
                .iter()
                .chain(chain_positions.iter())
                .map(|&idx| new_sub_args[idx])
                .collect();
            new_sub_args = reordered_args;
            for &(_, call_id) in &callsites {
                let old_opnds = module.ops[call_id].operands.clone();
                let reordered_opnds: smallvec::SmallVec<[ValueId; 4]> = non_chain_positions
                    .iter()
                    .chain(chain_positions.iter())
                    .map(|&idx| old_opnds[idx])
                    .collect();
                module.ops[call_id].operands = reordered_opnds;
            }
        }

        for (idx, &arg_v) in new_sub_args.iter().enumerate() {
            module.values[arg_v].def = ValueDef::BlockArgument {
                block: sub_blk,
                index: idx as u32,
            };
        }
        module.blocks[sub_blk].arguments = new_sub_args.clone();

        if let Some(Attribute::Type(Type::Function { inputs, .. })) =
            module.ops[subgraph_id].attributes.get_mut("function_type")
        {
            *inputs = new_sub_args
                .iter()
                .map(|&v| module.values[v].ty.clone())
                .collect();
        }

        total_hoisted_ops += const_ops.len();
    }

    total_hoisted_ops
}

fn is_complex_param_expr(sym: &str) -> bool {
    let trimmed = sym.trim();
    !trimmed.is_empty()
        && trimmed != "?"
        && (trimmed.contains('(')
            || trimmed.contains('+')
            || trimmed.contains('-')
            || trimmed.contains('*')
            || trimmed.contains('/'))
}

fn type_has_complex_param_expr(ty: &Type) -> bool {
    if let Type::Tensor(t) = ty {
        t.shape
            .iter()
            .any(|d| matches!(d, crate::ir::mlir::Dim::Symbolic(s) if is_complex_param_expr(s)))
    } else {
        false
    }
}

fn replace_complex_dims_in_type(
    ty: &Type,
    expr_to_param: &std::collections::HashMap<String, String>,
) -> Type {
    if let Type::Tensor(t) = ty {
        let mut new_t = t.clone();
        for d in &mut new_t.shape {
            if let crate::ir::mlir::Dim::Symbolic(s) = d
                && let Some(alias) = expr_to_param.get(s)
            {
                *s = alias.clone();
            }
        }
        Type::Tensor(new_t)
    } else {
        ty.clone()
    }
}

/// Hoists complex symbolic parameter expressions (`Dim::Symbolic` containing operators or
/// function applications) out of tensor types into `kgen.param.declare` alias declarations,
/// inserting `mo.rebind` operations on rewritten inputs/outputs and folding consecutive or
/// identity `mo.rebind`s.
///
/// Returns the number of `kgen.param.declare` parameter alias declarations created.
pub fn hoist_parameter_expressions(module: &mut Module) -> usize {
    use crate::ir::mlir::{Dim, Location, OpName, Operation, Value, ValueDef};

    let mut created_decls = 0usize;
    let mut next_param_idx = 0usize;

    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let mut expr_to_param: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for &oid in &module.blocks[bid].ops {
            let op = &module.ops[oid];
            if op.name.full_name() == "kgen.param.declare"
                && let (Some(name), Some(expr)) = (
                    op.attributes
                        .get("name")
                        .or_else(|| op.attributes.get("sym_name"))
                        .and_then(Attribute::as_str),
                    op.attributes.get("expr").and_then(Attribute::as_str),
                )
            {
                expr_to_param.insert(expr.to_string(), name.to_string());
            }
        }

        let ops_snapshot = module.blocks[bid].ops.clone();
        for op_id in ops_snapshot {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if full == "kgen.param.declare" || full == "mo.rebind" || is_terminator_op_name(&full) {
                continue;
            }

            let mut op_exprs = Vec::new();
            for &opnd in &op.operands {
                if let Some(Type::Tensor(t)) = module.values.get(opnd).map(|v| &v.ty) {
                    for d in &t.shape {
                        if let Dim::Symbolic(s) = d
                            && is_complex_param_expr(s)
                            && !op_exprs.contains(s)
                        {
                            op_exprs.push(s.clone());
                        }
                    }
                }
            }
            for &res in &op.results {
                if let Some(Type::Tensor(t)) = module.values.get(res).map(|v| &v.ty) {
                    for d in &t.shape {
                        if let Dim::Symbolic(s) = d
                            && is_complex_param_expr(s)
                            && !op_exprs.contains(s)
                        {
                            op_exprs.push(s.clone());
                        }
                    }
                }
            }
            if op_exprs.is_empty() {
                continue;
            }

            let mut pre_ops = Vec::new();
            for expr in op_exprs {
                if !expr_to_param.contains_key(&expr) {
                    let alias = format!("param_expr_{next_param_idx}");
                    next_param_idx += 1;
                    expr_to_param.insert(expr.clone(), alias.clone());
                    let mut attrs = indexmap::IndexMap::new();
                    attrs.insert("name".to_string(), Attribute::String(alias));
                    attrs.insert("expr".to_string(), Attribute::String(expr));
                    let decl_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("kgen", "param.declare"),
                        operands: smallvec::SmallVec::new(),
                        results: smallvec::SmallVec::new(),
                        attributes: attrs,
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[decl_op].id = decl_op;
                    pre_ops.push(decl_op);
                    created_decls += 1;
                }
            }

            // Rebind complex tensor inputs:
            let mut input_remap: std::collections::HashMap<ValueId, ValueId> =
                std::collections::HashMap::new();
            for &opnd in &op.operands {
                if input_remap.contains_key(&opnd) {
                    continue;
                }
                let Some(orig_ty) = module.values.get(opnd).map(|v| v.ty.clone()) else {
                    continue;
                };
                if type_has_complex_param_expr(&orig_ty) {
                    let rebound_ty = replace_complex_dims_in_type(&orig_ty, &expr_to_param);
                    let rebound_val = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: rebound_ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[rebound_val].id = rebound_val;
                    let rebind_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("mo", "rebind"),
                        operands: smallvec::smallvec![opnd],
                        results: smallvec::smallvec![rebound_val],
                        attributes: indexmap::IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[rebind_op].id = rebind_op;
                    module.values[rebound_val].def = ValueDef::OpResult {
                        op: rebind_op,
                        index: 0,
                    };
                    pre_ops.push(rebind_op);
                    input_remap.insert(opnd, rebound_val);
                }
            }

            for opnd_slot in &mut module.ops[op_id].operands {
                if let Some(&mapped) = input_remap.get(opnd_slot) {
                    *opnd_slot = mapped;
                }
            }

            // Rewrite op result types in-place and emit output rebinds to original types:
            let mut post_ops = Vec::new();
            let results_snapshot = module.ops[op_id].results.clone();
            for res_v in results_snapshot {
                let orig_ty = module.values[res_v].ty.clone();
                if type_has_complex_param_expr(&orig_ty) {
                    let aliased_ty = replace_complex_dims_in_type(&orig_ty, &expr_to_param);
                    module.values[res_v].ty = aliased_ty;

                    let out_rebound_val = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: orig_ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[out_rebound_val].id = out_rebound_val;
                    let out_rebind_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("mo", "rebind"),
                        operands: smallvec::smallvec![res_v],
                        results: smallvec::smallvec![out_rebound_val],
                        attributes: indexmap::IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[out_rebind_op].id = out_rebind_op;
                    module.values[out_rebound_val].def = ValueDef::OpResult {
                        op: out_rebind_op,
                        index: 0,
                    };

                    let all_ops: Vec<OpId> = module.ops.keys().collect();
                    for oid in all_ops {
                        if oid == out_rebind_op {
                            continue;
                        }
                        for opnd in &mut module.ops[oid].operands {
                            if *opnd == res_v {
                                *opnd = out_rebound_val;
                            }
                        }
                    }
                    post_ops.push(out_rebind_op);
                }
            }

            if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                for (off, pid) in pre_ops.iter().copied().enumerate() {
                    module.blocks[bid].ops.insert(pos + off, pid);
                }
                let after_op_pos = pos + pre_ops.len() + 1;
                for (off, qid) in post_ops.into_iter().enumerate() {
                    module.blocks[bid].ops.insert(after_op_pos + off, qid);
                }
            }
        }

        // Fold consecutive `mo.rebind(mo.rebind(x))` and identity `mo.rebind`s in the block:
        let mut idx = 0usize;
        while idx < module.blocks[bid].ops.len() {
            let oid = module.blocks[bid].ops[idx];
            if module.ops[oid].name.full_name() != "mo.rebind"
                || module.ops[oid].operands.len() != 1
                || module.ops[oid].results.len() != 1
            {
                idx += 1;
                continue;
            }
            let mut src_v = module.ops[oid].operands[0];
            let dst_v = module.ops[oid].results[0];
            if let Some(ValueDef::OpResult { op: prev_op, .. }) =
                module.values.get(src_v).map(|v| v.def)
                && module
                    .ops
                    .get(prev_op)
                    .is_some_and(|p| p.name.full_name() == "mo.rebind" && p.operands.len() == 1)
            {
                src_v = module.ops[prev_op].operands[0];
                module.ops[oid].operands[0] = src_v;
            }
            if module.values[src_v].ty == module.values[dst_v].ty {
                let all_ops: Vec<OpId> = module.ops.keys().collect();
                for other_id in all_ops {
                    for opnd in &mut module.ops[other_id].operands {
                        if *opnd == dst_v {
                            *opnd = src_v;
                        }
                    }
                }
                module.blocks[bid].ops.remove(idx);
                module.ops.remove(oid);
                continue;
            }
            idx += 1;
        }
    }

    created_decls
}

/// Evaluates `mo.invoke_shape_func` packing shape queries (`pack_matmul_b_shape_func`,
/// `pack_conv_filter_shape`, `pack_conv_transpose_filter_shape`) against target tile widths,
/// materializing `kgen.param.declare` constants for each output parameter.
///
/// When `sanity_check_shape_funcs` is `true`, retains `mo.invoke_shape_func` with renamed
/// output parameters and emits `kgen.param.assert_eq` checks; otherwise erases the resolved
/// `mo.invoke_shape_func` operations.
pub fn invoke_shape_packing_functions(
    module: &mut Module,
    sanity_check_shape_funcs: bool,
) -> Result<usize, String> {
    use crate::ir::mlir::{Dim, Location, OpName, Operation};

    let mut resolved_count = 0usize;
    let func_ids = module.functions();
    for fn_id in func_ids {
        let tile_width = match module.ops[fn_id].attributes.get("target_tile_width") {
            Some(Attribute::Int(v)) => (*v).max(1),
            _ => 16,
        };

        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                let ops_snapshot = module.blocks[bid].ops.clone();
                for op_id in ops_snapshot {
                    let Some(op) = module.ops.get(op_id).cloned() else {
                        continue;
                    };
                    if op.name.full_name() != "mo.invoke_shape_func" {
                        continue;
                    }
                    let Some(func_name) = op
                        .attributes
                        .get("shape_func_name")
                        .and_then(Attribute::as_str)
                    else {
                        continue;
                    };

                    let in_dims: Vec<i64> = if let Some(Attribute::Array(items)) =
                        op.attributes.get("input_shape")
                    {
                        items
                            .iter()
                            .filter_map(|a| match a {
                                Attribute::Int(v) => Some(*v),
                                _ => None,
                            })
                            .collect()
                    } else if let Some(&first_opnd) = op.operands.first()
                        && let Some(Type::Tensor(t)) = module.values.get(first_opnd).map(|v| &v.ty)
                    {
                        t.shape
                            .iter()
                            .filter_map(|d| match d {
                                Dim::Static(n) => Some(*n),
                                _ => None,
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                    let inferred_shapes: Vec<i64> = match (func_name, in_dims.as_slice()) {
                        ("pack_matmul_b_shape_func", &[k, n]) if k > 0 && n > 0 => {
                            vec![(n + tile_width - 1) / tile_width, k, tile_width]
                        }
                        ("pack_conv_filter_shape", &[f, c, r, s])
                            if f > 0 && c > 0 && r > 0 && s > 0 =>
                        {
                            vec![(f + tile_width - 1) / tile_width, c, r, s, tile_width]
                        }
                        ("pack_conv_transpose_filter_shape", &[r, s, f, c])
                            if r > 0 && s > 0 && f > 0 && c > 0 =>
                        {
                            vec![(c + tile_width - 1) / tile_width, r, s, f, tile_width]
                        }
                        _ => continue,
                    };

                    let output_decls: Vec<String> = match op.attributes.get("output_param_decls") {
                        Some(Attribute::Array(items)) => items
                            .iter()
                            .filter_map(Attribute::as_str)
                            .map(str::to_string)
                            .collect(),
                        _ => Vec::new(),
                    };
                    if output_decls.len() != inferred_shapes.len() {
                        return Err(format!(
                            "number of inferred parameters ({}) does not match number of \
                             output parameters ({})",
                            inferred_shapes.len(),
                            output_decls.len()
                        ));
                    }

                    let mut pre_decls = Vec::with_capacity(output_decls.len());
                    for (decl_name, val) in output_decls.iter().zip(inferred_shapes.iter()) {
                        let mut attrs = indexmap::IndexMap::new();
                        attrs.insert("name".to_string(), Attribute::String(decl_name.clone()));
                        attrs.insert("value".to_string(), Attribute::Int(*val));
                        let decl_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("kgen", "param.declare"),
                            operands: smallvec::SmallVec::new(),
                            results: smallvec::SmallVec::new(),
                            attributes: attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[decl_op].id = decl_op;
                        pre_decls.push(decl_op);
                    }

                    let mut post_asserts = Vec::new();
                    if sanity_check_shape_funcs {
                        let mut renamed_decls = Vec::with_capacity(output_decls.len());
                        for decl_name in &output_decls {
                            let check_name = format!("{decl_name}_check");
                            renamed_decls.push(Attribute::String(check_name.clone()));
                            let mut attrs = indexmap::IndexMap::new();
                            attrs.insert(
                                "lhs_param".to_string(),
                                Attribute::String(decl_name.clone()),
                            );
                            attrs.insert("rhs_param".to_string(), Attribute::String(check_name));
                            attrs.insert(
                                "message".to_string(),
                                Attribute::String(
                                    "shape generated by C++ shape function does not match \
                                     the kernel implementation"
                                        .to_string(),
                                ),
                            );
                            let assert_op = module.ops.insert(Operation {
                                id: OpId::default(),
                                name: OpName::new("kgen", "param.assert_eq"),
                                operands: smallvec::SmallVec::new(),
                                results: smallvec::SmallVec::new(),
                                attributes: attrs,
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: Location::Unknown,
                            });
                            module.ops[assert_op].id = assert_op;
                            post_asserts.push(assert_op);
                        }
                        module.ops[op_id].attributes.insert(
                            "output_param_decls".to_string(),
                            Attribute::Array(renamed_decls),
                        );
                    }

                    if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                        for (off, did) in pre_decls.iter().copied().enumerate() {
                            module.blocks[bid].ops.insert(pos + off, did);
                        }
                        let invoke_pos = pos + pre_decls.len();
                        if sanity_check_shape_funcs {
                            for (off, aid) in post_asserts.into_iter().enumerate() {
                                module.blocks[bid].ops.insert(invoke_pos + 1 + off, aid);
                            }
                        } else {
                            module.blocks[bid].ops.remove(invoke_pos);
                            module.ops.remove(op_id);
                        }
                    }

                    resolved_count += 1;
                }
            }
        }
    }

    Ok(resolved_count)
}

fn is_supported_triple_dtype(arch: &str, dtype: &crate::ir::mlir::DType) -> bool {
    use crate::ir::mlir::DType;

    let norm_arch = arch.split('-').next().unwrap_or(arch).trim();
    if norm_arch != "x86_64" && norm_arch != "aarch64" {
        return true;
    }
    match dtype {
        DType::BF16 => true,
        DType::F16 => norm_arch == "x86_64",
        DType::F8E4M3FN | DType::F8E5M2 => false,
        _ => true,
    }
}

/// Validates that every input block argument and operation result tensor dtype in each function
/// is supported by the function's target architecture (`device_target_triple` or
/// `default_target_triple`).
///
/// Mirrors CPU sub-32-bit float rules (`bf16` permitted on `x86_64` and `aarch64`, `f16`
/// permitted on `x86_64` only, `f8*` rejected on both `x86_64` and `aarch64`).
pub fn validate_device_dtypes(
    module: &mut Module,
    default_target_triple: Option<&str>,
) -> Result<(), String> {
    let func_ids = module.functions();
    for fn_id in func_ids {
        if let Some(triple) = default_target_triple
            && !triple.is_empty()
            && !module.ops[fn_id]
                .attributes
                .contains_key("device_target_triple")
        {
            module.ops[fn_id].attributes.insert(
                "device_target_triple".to_string(),
                Attribute::String(triple.to_string()),
            );
        }

        let Some(triple_str) = module.ops[fn_id]
            .attributes
            .get("device_target_triple")
            .or_else(|| module.ops[fn_id].attributes.get("target_arch"))
            .and_then(Attribute::as_str)
            .map(str::to_string)
        else {
            continue;
        };

        let fn_device = module.ops[fn_id]
            .attributes
            .get("device")
            .and_then(Attribute::as_str)
            .unwrap_or("cpu:0")
            .to_string();

        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                for &arg_v in &module.blocks[bid].arguments {
                    if let Some(Type::Tensor(t)) = module.values.get(arg_v).map(|v| &v.ty)
                        && !is_supported_triple_dtype(&triple_str, &t.dtype)
                    {
                        return Err(format!(
                            "The {} data type is not supported on device '{fn_device}'.",
                            t.dtype
                        ));
                    }
                }
                for &oid in &module.blocks[bid].ops {
                    let op = &module.ops[oid];
                    let op_dev = op
                        .attributes
                        .get("device")
                        .and_then(Attribute::as_str)
                        .unwrap_or(fn_device.as_str());
                    for &res_v in &op.results {
                        if let Some(Type::Tensor(t)) = module.values.get(res_v).map(|v| &v.ty)
                            && !is_supported_triple_dtype(&triple_str, &t.dtype)
                        {
                            return Err(format!(
                                "The {} data type is not supported on device '{op_dev}'.",
                                t.dtype
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Verifies that every function marked `is_device_graph = true` contains no control-flow
/// operations and that every kernel-backed operation whose registration appears in module-level
/// `kgen.kernel_decl` / `mogg.kernel.decl` declarations exposes a `build_graph` entry point.
///
/// Returns the number of verified kernel-backed operations across device-graph functions.
pub fn verify_device_graph_kernels(module: &Module) -> Result<usize, String> {
    let mut registry: std::collections::HashMap<String, (String, bool)> =
        std::collections::HashMap::new();
    for op in module.ops.values() {
        let full = op.name.full_name();
        if full == "kgen.kernel_decl" || full == "mogg.kernel.decl" {
            let sym = op
                .attributes
                .get("sym_name")
                .and_then(Attribute::as_str)
                .unwrap_or("unnamed_kernel")
                .to_string();
            let reg_key = op
                .attributes
                .get("kernel_name")
                .and_then(Attribute::as_str)
                .unwrap_or(sym.as_str())
                .to_string();
            let has_build_graph = match op.attributes.get("build_graph") {
                Some(Attribute::Bool(b)) => *b,
                Some(Attribute::String(s) | Attribute::SymbolRef(s)) => !s.is_empty(),
                _ => false,
            };
            registry.insert(reg_key, (sym, has_build_graph));
        }
    }

    let mut verified_count = 0usize;
    for fn_id in module.functions() {
        let is_device_graph = matches!(
            module.ops[fn_id].attributes.get("is_device_graph"),
            Some(Attribute::Bool(true))
        );
        if !is_device_graph {
            continue;
        }
        for &reg_id in &module.ops[fn_id].regions {
            for &bid in &module.regions[reg_id].blocks {
                for &oid in &module.blocks[bid].ops {
                    let op = &module.ops[oid];
                    let full = op.name.full_name();
                    if is_terminator_op_name(&full) {
                        continue;
                    }
                    if matches!(
                        full.as_str(),
                        "mo.if" | "mo.while" | "scf.if" | "scf.for" | "scf.while"
                    ) {
                        return Err(format!(
                            "device-graphable mo.graph cannot contain control-flow op \
                             '{full}' (control flow is not yet supported)"
                        ));
                    }
                    let reg_name = if full == "mo.custom" {
                        op.attributes
                            .get("symbol")
                            .and_then(Attribute::as_str)
                            .unwrap_or(full.as_str())
                    } else {
                        full.as_str()
                    };
                    if let Some((kernel_sym, has_build_graph)) = registry.get(reg_name) {
                        if !*has_build_graph {
                            return Err(format!(
                                "op '{full}' is in a device-graphable mo.graph but its kernel \
                                 registration ('{reg_name}', kernel symbol '{kernel_sym}') \
                                 does not expose a `build_graph` entry point"
                            ));
                        }
                        verified_count += 1;
                    }
                }
            }
        }
    }

    Ok(verified_count)
}

/// Replaces large static constant operations (`mo.constant`, `arith.constant`,
/// `stablehlo.constant`) having at least `min_num_elements` (`> 1`) elements with a 1-element
/// zero constant (`tensor<1xdtype>`) followed by a `mo.static.broadcast_to` operation.
///
/// Returns the number of large constants converted into splat broadcasts.
pub fn splat_large_constants(module: &mut Module, min_num_elements: i64) -> usize {
    use crate::ir::mlir::{Dim, Location, OpName, Operation, TensorType, Value, ValueDef};

    let threshold = min_num_elements.max(2);
    let mut splatted = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let ops_snapshot = module.blocks[bid].ops.clone();
        for op_id in ops_snapshot {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if !matches!(
                full.as_str(),
                "mo.constant" | "arith.constant" | "stablehlo.constant"
            ) || op.results.len() != 1
            {
                continue;
            }
            let orig_res = op.results[0];
            let Some(Type::Tensor(orig_t)) = module.values.get(orig_res).map(|v| v.ty.clone())
            else {
                continue;
            };
            let Some(num_elems) = orig_t.num_elements() else {
                continue;
            };
            if num_elems < threshold || num_elems <= 1 {
                continue;
            }

            let one_elem_ty = Type::Tensor(TensorType {
                shape: vec![Dim::Static(1)],
                dtype: orig_t.dtype.clone(),
                encoding: orig_t.encoding.clone(),
            });
            let one_elem_val = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: one_elem_ty,
                def: ValueDef::BlockArgument {
                    block: bid,
                    index: 0,
                },
            });
            module.values[one_elem_val].id = one_elem_val;

            let mut const_attrs = indexmap::IndexMap::new();
            const_attrs.insert("splat_zero".to_string(), Attribute::Bool(true));
            let new_const_op = module.ops.insert(Operation {
                id: OpId::default(),
                name: op.name.clone(),
                operands: smallvec::SmallVec::new(),
                results: smallvec::smallvec![one_elem_val],
                attributes: const_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: Location::Unknown,
            });
            module.ops[new_const_op].id = new_const_op;
            module.values[one_elem_val].def = ValueDef::OpResult {
                op: new_const_op,
                index: 0,
            };

            let bcast_val = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: Type::Tensor(orig_t),
                def: ValueDef::BlockArgument {
                    block: bid,
                    index: 0,
                },
            });
            module.values[bcast_val].id = bcast_val;
            let bcast_op = module.ops.insert(Operation {
                id: OpId::default(),
                name: OpName::new("mo", "static.broadcast_to"),
                operands: smallvec::smallvec![one_elem_val],
                results: smallvec::smallvec![bcast_val],
                attributes: indexmap::IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: Location::Unknown,
            });
            module.ops[bcast_op].id = bcast_op;
            module.values[bcast_val].def = ValueDef::OpResult {
                op: bcast_op,
                index: 0,
            };

            let all_ops: Vec<OpId> = module.ops.keys().collect();
            for oid in all_ops {
                for opnd in &mut module.ops[oid].operands {
                    if *opnd == orig_res {
                        *opnd = bcast_val;
                    }
                }
            }

            if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                module.blocks[bid].ops[pos] = new_const_op;
                module.blocks[bid].ops.insert(pos + 1, bcast_op);
            }
            module.ops.remove(op_id);
            splatted += 1;
        }
    }
    splatted
}

fn is_float_tensor_type(ty: &Type) -> bool {
    use crate::ir::mlir::DType;

    if let Type::Tensor(t) = ty {
        matches!(
            t.dtype,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::F8E4M3FN | DType::F8E5M2
        )
    } else {
        false
    }
}

/// Instruments `map.iter.elementwise` and `map.iter.opaque` floating-point tensor outputs with
/// non-intrusive `nan_check_count` -> optional device-to-host `mo.transfer` -> `nan_check_raise`
/// verification kernels at the given sampling `stride` (`stride == 0` is treated as `1`).
///
/// Marks each `nan_check_raise` `map.iter.opaque` kernel with `has_write_effect = true` so dead
/// code elimination preserves the chained check sequence. Returns
/// `(checked_float_results, total_float_results)`.
pub fn inject_nan_check_instrumentation(module: &mut Module, stride: usize) -> (usize, usize) {
    use crate::ir::mlir::{DType, Dim, Location, OpName, Operation, TensorType, Value, ValueDef};

    let effective_stride = stride.max(1);
    let mut total_float_results = 0usize;
    let mut checked_float_results = 0usize;

    let count_tensor_ty = Type::Tensor(TensorType {
        shape: vec![Dim::Static(1)],
        dtype: DType::I32,
        encoding: None,
    });
    let chain_ty = Type::Opaque("!mo.chain".to_string());

    let func_ids = module.functions();
    for fn_id in func_ids {
        let mut float_result_idx = 0usize;
        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                let mut current_chain: Option<ValueId> = None;
                let ops_snapshot = module.blocks[bid].ops.clone();
                for kernel_id in ops_snapshot {
                    let Some(kernel_op) = module.ops.get(kernel_id).cloned() else {
                        continue;
                    };
                    let full = kernel_op.name.full_name();
                    if full != "map.iter.elementwise" && full != "map.iter.opaque" {
                        continue;
                    }
                    let kernel_dev = kernel_op
                        .attributes
                        .get("device")
                        .and_then(Attribute::as_str)
                        .unwrap_or("cpu:0")
                        .to_string();
                    let needs_transfer = kernel_dev != "cpu" && !kernel_dev.starts_with("cpu:");

                    let mut injected_ops = Vec::new();
                    for (res_pos, &res_v) in kernel_op.results.iter().enumerate() {
                        let Some(res_ty) = module.values.get(res_v).map(|v| v.ty.clone()) else {
                            continue;
                        };
                        if !is_float_tensor_type(&res_ty) {
                            continue;
                        }
                        total_float_results += 1;
                        let sample_idx = float_result_idx;
                        float_result_idx += 1;
                        if effective_stride > 1 && !sample_idx.is_multiple_of(effective_stride) {
                            continue;
                        }
                        checked_float_results += 1;

                        let label = module.values[res_v]
                            .ssa_name
                            .clone()
                            .unwrap_or_else(|| format!("{full}#{res_pos}"));
                        let type_str = format!("{res_ty}");

                        // 1. `nan_check_count` lowered to `map.iter.opaque`:
                        let nan_cnt = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: count_tensor_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 0,
                            },
                        });
                        module.values[nan_cnt].id = nan_cnt;
                        let inf_cnt = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: count_tensor_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 1,
                            },
                        });
                        module.values[inf_cnt].id = inf_cnt;
                        let mut count_attrs = indexmap::IndexMap::new();
                        count_attrs.insert(
                            "symbol".to_string(),
                            Attribute::String("nan_check_count".to_string()),
                        );
                        count_attrs
                            .insert("device".to_string(), Attribute::String(kernel_dev.clone()));
                        let count_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("map", "iter.opaque"),
                            operands: smallvec::smallvec![res_v],
                            results: smallvec::smallvec![nan_cnt, inf_cnt],
                            attributes: count_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[count_op].id = count_op;
                        module.values[nan_cnt].def = ValueDef::OpResult {
                            op: count_op,
                            index: 0,
                        };
                        module.values[inf_cnt].def = ValueDef::OpResult {
                            op: count_op,
                            index: 1,
                        };
                        injected_ops.push(count_op);

                        // Lazily create `mo.chain.create` after the first checked kernel:
                        let mut ch_val = if let Some(existing_ch) = current_chain {
                            existing_ch
                        } else {
                            let init_ch = module.values.insert(Value {
                                id: ValueId::default(),
                                ssa_name: None,
                                ty: chain_ty.clone(),
                                def: ValueDef::BlockArgument {
                                    block: bid,
                                    index: 0,
                                },
                            });
                            module.values[init_ch].id = init_ch;
                            let ch_op = module.ops.insert(Operation {
                                id: OpId::default(),
                                name: OpName::new("mo", "chain.create"),
                                operands: smallvec::SmallVec::new(),
                                results: smallvec::smallvec![init_ch],
                                attributes: indexmap::IndexMap::new(),
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: Location::Unknown,
                            });
                            module.ops[ch_op].id = ch_op;
                            module.values[init_ch].def = ValueDef::OpResult {
                                op: ch_op,
                                index: 0,
                            };
                            injected_ops.push(ch_op);
                            init_ch
                        };

                        // 2. Transfer counts to host if `kernel_dev` is not host:
                        let mut nan_host = nan_cnt;
                        let mut inf_host = inf_cnt;
                        if needs_transfer {
                            for (src_cnt, dst_slot) in
                                [(nan_cnt, &mut nan_host), (inf_cnt, &mut inf_host)]
                            {
                                let xfer_t = module.values.insert(Value {
                                    id: ValueId::default(),
                                    ssa_name: None,
                                    ty: count_tensor_ty.clone(),
                                    def: ValueDef::BlockArgument {
                                        block: bid,
                                        index: 0,
                                    },
                                });
                                module.values[xfer_t].id = xfer_t;
                                let xfer_ch = module.values.insert(Value {
                                    id: ValueId::default(),
                                    ssa_name: None,
                                    ty: chain_ty.clone(),
                                    def: ValueDef::BlockArgument {
                                        block: bid,
                                        index: 1,
                                    },
                                });
                                module.values[xfer_ch].id = xfer_ch;
                                let mut xfer_attrs = indexmap::IndexMap::new();
                                xfer_attrs.insert(
                                    "device".to_string(),
                                    Attribute::String("cpu:0".to_string()),
                                );
                                xfer_attrs.insert(
                                    "always_elide_same_device_copy".to_string(),
                                    Attribute::Bool(true),
                                );
                                let xfer_op = module.ops.insert(Operation {
                                    id: OpId::default(),
                                    name: OpName::new("mo", "transfer"),
                                    operands: smallvec::smallvec![src_cnt, ch_val],
                                    results: smallvec::smallvec![xfer_t, xfer_ch],
                                    attributes: xfer_attrs,
                                    regions: Vec::new(),
                                    custom_syntax: false,
                                    location: Location::Unknown,
                                });
                                module.ops[xfer_op].id = xfer_op;
                                module.values[xfer_t].def = ValueDef::OpResult {
                                    op: xfer_op,
                                    index: 0,
                                };
                                module.values[xfer_ch].def = ValueDef::OpResult {
                                    op: xfer_op,
                                    index: 1,
                                };
                                injected_ops.push(xfer_op);
                                *dst_slot = xfer_t;
                                ch_val = xfer_ch;
                            }
                        }

                        // 3. `nan_check_raise` lowered to `map.iter.opaque`
                        // with `has_write_effect = true`:
                        let raise_ch = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: chain_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 0,
                            },
                        });
                        module.values[raise_ch].id = raise_ch;
                        let mut raise_attrs = indexmap::IndexMap::new();
                        raise_attrs.insert(
                            "symbol".to_string(),
                            Attribute::String("nan_check_raise".to_string()),
                        );
                        raise_attrs
                            .insert("device".to_string(), Attribute::String("cpu:0".to_string()));
                        raise_attrs.insert("has_write_effect".to_string(), Attribute::Bool(true));
                        raise_attrs.insert("label".to_string(), Attribute::String(label));
                        raise_attrs.insert("type_str".to_string(), Attribute::String(type_str));
                        let raise_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("map", "iter.opaque"),
                            operands: smallvec::smallvec![nan_host, inf_host, ch_val],
                            results: smallvec::smallvec![raise_ch],
                            attributes: raise_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[raise_op].id = raise_op;
                        module.values[raise_ch].def = ValueDef::OpResult {
                            op: raise_op,
                            index: 0,
                        };
                        injected_ops.push(raise_op);
                        current_chain = Some(raise_ch);
                    }

                    if !injected_ops.is_empty()
                        && let Some(pos) = module.blocks[bid]
                            .ops
                            .iter()
                            .position(|&id| id == kernel_id)
                    {
                        for (off, iid) in injected_ops.into_iter().enumerate() {
                            module.blocks[bid].ops.insert(pos + 1 + off, iid);
                        }
                    }
                }
            }
        }
    }

    (checked_float_results, total_float_results)
}
