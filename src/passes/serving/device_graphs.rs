//! Device-graph region staging, collective partitioning, and profile scope tracing.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, TensorType, Type, Value,
    ValueDef, ValueId,
};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Device-Graph Guard Lowering & Isolated Region Staging
// ============================================================================

/// Rewrites every `mgp.guard` operation in `module` into a forwarding `mgp.routine.async`
/// operation with `in_out_signature = ["none", ...]`, `dispatch = "immediate"`,
/// `can_fuse_into_other = false`, `can_be_fused_into = false`, and `can_duplicate = false`.
/// Returns the number of lowered guard operations.
pub fn lower_guards_to_async_routines(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut lowered_count = 0usize;

    for blk_id in block_ids {
        let ops = module.blocks[blk_id].ops.clone();
        for &op_id in &ops {
            if module.ops[op_id].name.full_name() != "mgp.guard" {
                continue;
            }
            let num_opds = module.ops[op_id].operands.len();
            let sig: Vec<Attribute> = (0..num_opds)
                .map(|_| Attribute::String("none".to_string()))
                .collect();
            module.ops[op_id].name = OpName::new("mgp", "routine.async");
            module.ops[op_id]
                .attributes
                .insert("in_out_signature".to_string(), Attribute::Array(sig));
            module.ops[op_id].attributes.insert(
                "dispatch".to_string(),
                Attribute::String("immediate".to_string()),
            );
            module.ops[op_id]
                .attributes
                .insert("can_fuse_into_other".to_string(), Attribute::Bool(false));
            module.ops[op_id]
                .attributes
                .insert("can_be_fused_into".to_string(), Attribute::Bool(false));
            module.ops[op_id]
                .attributes
                .insert("can_duplicate".to_string(), Attribute::Bool(false));
            lowered_count += 1;
        }
    }

    lowered_count
}

/// Lowers `mgp.guard` operations into `mgp.routine.async` and stages all `mgp.routine.async` /
/// `mogg.routine.async` and `mgp.device_graph.stable.create` operations in each function entry
/// block into an isolated `mgp.device_graph.staging` region block:
/// - Collects `free_values` (operands defined outside the staged set) and maps them to the
///   staging block's arguments.
/// - Collects `escaping_values` (results of staged ops used by operations outside the staged set)
///   and yields them via `mgp.device_graph.staging.yield`.
/// - Inserts `mgp.device_graph.staging` immediately before the block terminator and rewires
///   external uses of `escaping_values` to the staging operation's results.
///
/// Returns the total number of staged operations moved into `mgp.device_graph.staging`.
pub fn stage_device_graph_regions(module: &mut Module) -> usize {
    lower_guards_to_async_routines(module);
    let funcs = module.functions();
    let mut total_staged = 0usize;

    for func_id in funcs {
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let reg_id = module.ops[func_id].regions[0];
        if module.regions[reg_id].blocks.is_empty() {
            continue;
        }
        let blk_id = module.regions[reg_id].blocks[0];
        let orig_ops = module.blocks[blk_id].ops.clone();

        let mut staged_ops: IndexSet<OpId> = IndexSet::new();
        for &op_id in &orig_ops {
            let full = module.ops[op_id].name.full_name();
            if matches!(
                full.as_str(),
                "mgp.routine.async" | "mogg.routine.async" | "mgp.device_graph.stable.create"
            ) {
                staged_ops.insert(op_id);
            }
        }
        if staged_ops.is_empty() {
            continue;
        }

        // Collect `free_values` (operands of staged ops defined outside `staged_ops`).
        let mut free_values: IndexSet<ValueId> = IndexSet::new();
        for &op_id in &staged_ops {
            for &opd in &module.ops[op_id].operands {
                let is_internal = matches!(
                    module.values[opd].def,
                    ValueDef::OpResult { op: def_op, .. } if staged_ops.contains(&def_op)
                );
                if !is_internal {
                    free_values.insert(opd);
                }
            }
        }

        // Collect unique `escaping_values` (results of staged ops used outside `staged_ops`).
        let mut escaping_values: IndexSet<ValueId> = IndexSet::new();
        for &op_id in &staged_ops {
            for &res in &module.ops[op_id].results {
                let used_outside = orig_ops.iter().any(|&user_id| {
                    !staged_ops.contains(&user_id) && module.ops[user_id].operands.contains(&res)
                });
                if used_outside {
                    escaping_values.insert(res);
                }
            }
        }

        let loc = module.ops[*staged_ops.first().unwrap()].location.clone();
        let staging_blk = module.blocks.insert_with_key(|id| crate::ir::Block {
            id,
            label: None,
            arguments: Vec::new(),
            ops: Vec::new(),
        });

        let mut free_map: IndexMap<ValueId, ValueId> = IndexMap::new();
        for (idx, &fv) in free_values.iter().enumerate() {
            let ty = module.values[fv].ty.clone();
            let arg_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty,
                def: ValueDef::BlockArgument {
                    block: staging_blk,
                    index: idx as u32,
                },
            });
            module.blocks[staging_blk].arguments.push(arg_val);
            free_map.insert(fv, arg_val);
        }

        // Move `staged_ops` into `staging_blk` and remap `free_values` to `staging_blk` arguments.
        for &op_id in &staged_ops {
            for opd in &mut module.ops[op_id].operands {
                if let Some(&mapped) = free_map.get(opd) {
                    *opd = mapped;
                }
            }
            module.blocks[staging_blk].ops.push(op_id);
        }

        let yield_opds: SmallVec<[ValueId; 4]> = escaping_values.iter().copied().collect();
        let yield_op = module.ops.insert_with_key(|id| Operation {
            id,
            name: OpName::new("mgp", "device_graph.staging.yield"),
            operands: yield_opds,
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: loc.clone(),
        });
        module.blocks[staging_blk].ops.push(yield_op);

        let staging_reg = module.regions.insert_with_key(|id| crate::ir::Region {
            id,
            blocks: vec![staging_blk],
        });

        let staging_opds: SmallVec<[ValueId; 4]> = free_values.iter().copied().collect();
        let staging_op = module.ops.insert_with_key(|id| Operation {
            id,
            name: OpName::new("mgp", "device_graph.staging"),
            operands: staging_opds,
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: vec![staging_reg],
            custom_syntax: false,
            location: loc,
        });

        let mut escape_remap: IndexMap<ValueId, ValueId> = IndexMap::new();
        for (idx, &esc_val) in escaping_values.iter().enumerate() {
            let ty = module.values[esc_val].ty.clone();
            let res_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty,
                def: ValueDef::OpResult {
                    op: staging_op,
                    index: idx as u32,
                },
            });
            module.ops[staging_op].results.push(res_val);
            escape_remap.insert(esc_val, res_val);
        }

        // Remove staged ops from outer block, insert `staging_op` before terminator, and rewire
        // external consumers.
        module.blocks[blk_id]
            .ops
            .retain(|id| !staged_ops.contains(id));
        for &ext_op in &module.blocks[blk_id].ops {
            for opd in &mut module.ops[ext_op].operands {
                if let Some(&new_v) = escape_remap.get(opd) {
                    *opd = new_v;
                }
            }
        }
        let insert_pos = module.blocks[blk_id].ops.len().saturating_sub(1);
        module.blocks[blk_id].ops.insert(insert_pos, staging_op);

        total_staged += staged_ops.len();
    }

    total_staged
}

// ============================================================================
// Multi-Device Collective Graph Partitioner
// ============================================================================

/// Returns the device label (`device_ref` attribute, defaulting to `"cpu:0"`) of an operation.
pub(super) fn op_device_ref_str(op: &Operation) -> String {
    match op.attributes.get("device_ref") {
        Some(Attribute::String(s)) => s.clone(),
        _ => "cpu:0".to_string(),
    }
}

/// Returns `true` if `device_ref` represents a host CPU execution context.
pub(super) fn is_host_device_ref(device_ref: &str) -> bool {
    device_ref == "cpu" || device_ref.starts_with("cpu:")
}

/// Partitions each `mgp.device_graph.staging` operation in `module` into an
/// `mgp.device_graph.collective` operation with one region per sorted accelerator device:
/// - Assigns sequential slot indices `0..K` to `mgp.device_graph.stable.create` operations.
/// - Replays each non-terminator operation per device region:
///   - `mgp.device_graph.stable.create` on its owning device lowers to
///     `mgp.device_graph.stable.place` + `mgp.tensor.extract.tensor_spec` +
///     `mgp.device_graph.stable.tensor`.
///   - `mgp.routine.async` / `mogg.routine.async` on its matching accelerator device (or host CPU
///     bookkeeping whose operands are available in the region) is cloned into the region; any
///     operand produced by a `stable.create` on another device is materialized on demand via
///     `mgp.device_graph.stable.get` + `mgp.tensor_spec.create` + `mgp.device_graph.stable.tensor`.
/// - Emits `mgp.device_graph.collective.yield` per device region and rewires the staging
///   operation's results to the collective operation's results.
pub fn partition_device_graph_collective(module: &mut Module) -> Result<usize, String> {
    let funcs = module.functions();
    let mut partitioned_count = 0usize;

    for func_id in funcs {
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let reg_id = module.ops[func_id].regions[0];
        if module.regions[reg_id].blocks.is_empty() {
            continue;
        }
        let blk_id = module.regions[reg_id].blocks[0];
        let orig_ops = module.blocks[blk_id].ops.clone();

        for &staging_op in &orig_ops {
            if !module.ops.contains_key(staging_op)
                || module.ops[staging_op].name.full_name() != "mgp.device_graph.staging"
                || module.ops[staging_op].regions.is_empty()
            {
                continue;
            }
            let staging_reg = module.ops[staging_op].regions[0];
            if module.regions[staging_reg].blocks.is_empty() {
                continue;
            }
            let staging_blk = module.regions[staging_reg].blocks[0];
            let body_ops = module.blocks[staging_blk].ops.clone();
            if body_ops.is_empty() {
                continue;
            }
            let yield_op = *body_ops.last().unwrap();
            let non_term_ops = &body_ops[..body_ops.len() - 1];

            // Collect `stable.create` slot indices and sorted accelerator devices.
            let mut stable_indices: IndexMap<OpId, usize> = IndexMap::new();
            let mut stable_res_to_op: IndexMap<ValueId, OpId> = IndexMap::new();
            let mut devices_set: IndexSet<String> = IndexSet::new();

            for &op_id in non_term_ops {
                let full = module.ops[op_id].name.full_name();
                if full == "mgp.device_graph.stable.create" {
                    let slot = stable_indices.len();
                    stable_indices.insert(op_id, slot);
                    if let Some(&res) = module.ops[op_id].results.first() {
                        stable_res_to_op.insert(res, op_id);
                    }
                } else if matches!(full.as_str(), "mgp.routine.async" | "mogg.routine.async") {
                    let dev = op_device_ref_str(&module.ops[op_id]);
                    if !is_host_device_ref(&dev) {
                        devices_set.insert(dev);
                    }
                } else {
                    return Err(format!(
                        "unexpected operation {full} inside device-graph staging body"
                    ));
                }
            }

            let mut device_order: Vec<String> = devices_set.into_iter().collect();
            device_order.sort();
            if device_order.is_empty() {
                return Err("staging body has no accelerator regions to partition".to_string());
            }

            let loc = module.ops[staging_op].location.clone();
            let i64_scalar_ty = Type::Tensor(crate::ir::mlir::TensorType {
                shape: vec![Dim::Static(1)],
                dtype: DType::I64,
                encoding: None,
            });

            // Map staging block arguments back to outer staging operands.
            let mut base_mapper: IndexMap<ValueId, ValueId> = IndexMap::new();
            for (&blk_arg, &outer_in) in module.blocks[staging_blk]
                .arguments
                .iter()
                .zip(module.ops[staging_op].operands.iter())
            {
                base_mapper.insert(blk_arg, outer_in);
            }

            let orig_yield_opds = module.ops[yield_op].operands.clone();
            let mut collective_regions: Vec<crate::ir::RegionId> =
                Vec::with_capacity(device_order.len());
            let mut result_mapping: IndexMap<ValueId, usize> = IndexMap::new();
            let mut collective_res_types: Vec<Type> = Vec::new();

            for dev in &device_order {
                let dev_blk = module.blocks.insert_with_key(|id| crate::ir::Block {
                    id,
                    label: None,
                    arguments: Vec::new(),
                    ops: Vec::new(),
                });
                let builder_arg = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: i64_scalar_ty.clone(),
                    def: ValueDef::BlockArgument {
                        block: dev_blk,
                        index: 0,
                    },
                });
                module.blocks[dev_blk].arguments.push(builder_arg);

                let mut mapper = base_mapper.clone();
                let mut stable_tensors: IndexMap<OpId, ValueId> = IndexMap::new();

                for &op_id in non_term_ops {
                    let full = module.ops[op_id].name.full_name();
                    let op_dev = op_device_ref_str(&module.ops[op_id]);

                    if full == "mgp.device_graph.stable.create" {
                        if &op_dev != dev {
                            continue;
                        }
                        let src_opd = module.ops[op_id].operands[0];
                        let Some(&mapped_src) = mapper.get(&src_opd) else {
                            return Err(
                                "stable.create source was not materialized on its device".into()
                            );
                        };
                        let slot_idx = stable_indices[&op_id] as i64;
                        let tensor_ty = module.values[mapped_src].ty.clone();

                        // 1. `mgp.device_graph.stable.place`
                        let mut place_attrs = IndexMap::new();
                        place_attrs.insert("slot_index".to_string(), Attribute::Int(slot_idx));
                        let place_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mgp", "device_graph.stable.place"),
                            operands: SmallVec::from_slice(&[builder_arg, mapped_src]),
                            results: SmallVec::new(),
                            attributes: place_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let placed_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: i64_scalar_ty.clone(),
                            def: ValueDef::OpResult {
                                op: place_op,
                                index: 0,
                            },
                        });
                        module.ops[place_op].results.push(placed_val);
                        module.blocks[dev_blk].ops.push(place_op);

                        // 2. `mgp.tensor.extract.tensor_spec`
                        let spec_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mgp", "tensor.extract.tensor_spec"),
                            operands: SmallVec::from_slice(&[mapped_src]),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let spec_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: i64_scalar_ty.clone(),
                            def: ValueDef::OpResult {
                                op: spec_op,
                                index: 0,
                            },
                        });
                        module.ops[spec_op].results.push(spec_val);
                        module.blocks[dev_blk].ops.push(spec_op);

                        // 3. `mgp.device_graph.stable.tensor`
                        let st_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mgp", "device_graph.stable.tensor"),
                            operands: SmallVec::from_slice(&[placed_val, spec_val]),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let st_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: tensor_ty,
                            def: ValueDef::OpResult {
                                op: st_op,
                                index: 0,
                            },
                        });
                        module.ops[st_op].results.push(st_val);
                        module.blocks[dev_blk].ops.push(st_op);
                        stable_tensors.insert(op_id, st_val);
                        continue;
                    }

                    let is_host = is_host_device_ref(&op_dev);
                    if &op_dev != dev && !is_host {
                        continue;
                    }

                    // Check operand availability (or materialize from `stable.create`).
                    let orig_opds = module.ops[op_id].operands.clone();
                    let all_avail = orig_opds
                        .iter()
                        .all(|v| mapper.contains_key(v) || stable_res_to_op.contains_key(v));
                    if !all_avail {
                        if is_host {
                            continue;
                        }
                        return Err(
                            "operand is produced on another device without a stable slot".into(),
                        );
                    }

                    let mut cloned_opds: SmallVec<[ValueId; 4]> = SmallVec::new();
                    for &opd in &orig_opds {
                        if let Some(&st_op_id) = stable_res_to_op.get(&opd) {
                            let resolved = if let Some(&existing) = stable_tensors.get(&st_op_id) {
                                existing
                            } else {
                                let slot_idx = stable_indices[&st_op_id] as i64;
                                let tensor_ty = module.values[opd].ty.clone();
                                let mut get_attrs = IndexMap::new();
                                get_attrs
                                    .insert("slot_index".to_string(), Attribute::Int(slot_idx));
                                let get_op = module.ops.insert_with_key(|id| Operation {
                                    id,
                                    name: OpName::new("mgp", "device_graph.stable.get"),
                                    operands: SmallVec::from_slice(&[builder_arg]),
                                    results: SmallVec::new(),
                                    attributes: get_attrs,
                                    regions: Vec::new(),
                                    custom_syntax: false,
                                    location: loc.clone(),
                                });
                                let get_val = module.values.insert_with_key(|id| Value {
                                    id,
                                    ssa_name: None,
                                    ty: i64_scalar_ty.clone(),
                                    def: ValueDef::OpResult {
                                        op: get_op,
                                        index: 0,
                                    },
                                });
                                module.ops[get_op].results.push(get_val);
                                module.blocks[dev_blk].ops.push(get_op);

                                let spec_op = module.ops.insert_with_key(|id| Operation {
                                    id,
                                    name: OpName::new("mgp", "tensor_spec.create"),
                                    operands: SmallVec::new(),
                                    results: SmallVec::new(),
                                    attributes: IndexMap::new(),
                                    regions: Vec::new(),
                                    custom_syntax: false,
                                    location: loc.clone(),
                                });
                                let spec_val = module.values.insert_with_key(|id| Value {
                                    id,
                                    ssa_name: None,
                                    ty: i64_scalar_ty.clone(),
                                    def: ValueDef::OpResult {
                                        op: spec_op,
                                        index: 0,
                                    },
                                });
                                module.ops[spec_op].results.push(spec_val);
                                module.blocks[dev_blk].ops.push(spec_op);

                                let st_op = module.ops.insert_with_key(|id| Operation {
                                    id,
                                    name: OpName::new("mgp", "device_graph.stable.tensor"),
                                    operands: SmallVec::from_slice(&[get_val, spec_val]),
                                    results: SmallVec::new(),
                                    attributes: IndexMap::new(),
                                    regions: Vec::new(),
                                    custom_syntax: false,
                                    location: loc.clone(),
                                });
                                let st_val = module.values.insert_with_key(|id| Value {
                                    id,
                                    ssa_name: None,
                                    ty: tensor_ty,
                                    def: ValueDef::OpResult {
                                        op: st_op,
                                        index: 0,
                                    },
                                });
                                module.ops[st_op].results.push(st_val);
                                module.blocks[dev_blk].ops.push(st_op);
                                stable_tensors.insert(st_op_id, st_val);
                                st_val
                            };
                            mapper.insert(opd, resolved);
                            cloned_opds.push(resolved);
                        } else {
                            cloned_opds.push(mapper[&opd]);
                        }
                    }

                    let orig_name = module.ops[op_id].name.clone();
                    let orig_attrs = module.ops[op_id].attributes.clone();
                    let orig_res = module.ops[op_id].results.clone();
                    let cloned_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: orig_name,
                        operands: cloned_opds,
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
                        mapper.insert(old_r, new_r);
                    }
                    module.blocks[dev_blk].ops.push(cloned_op);
                }

                let mut reg_yields: SmallVec<[ValueId; 4]> = SmallVec::new();
                for &orig_y in &orig_yield_opds {
                    if let Some(&cloned_y) = mapper.get(&orig_y) {
                        reg_yields.push(cloned_y);
                        let res_idx = collective_res_types.len();
                        collective_res_types.push(module.values[cloned_y].ty.clone());
                        result_mapping.entry(orig_y).or_insert(res_idx);
                    }
                }
                let cy_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mgp", "device_graph.collective.yield"),
                    operands: reg_yields,
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                module.blocks[dev_blk].ops.push(cy_op);

                let dev_reg = module.regions.insert_with_key(|id| crate::ir::Region {
                    id,
                    blocks: vec![dev_blk],
                });
                collective_regions.push(dev_reg);
            }

            let mut coll_attrs = IndexMap::new();
            coll_attrs.insert(
                "devices".to_string(),
                Attribute::Array(
                    device_order
                        .iter()
                        .map(|d| Attribute::String(d.clone()))
                        .collect(),
                ),
            );
            coll_attrs.insert(
                "num_stable_slots".to_string(),
                Attribute::Int(stable_indices.len() as i64),
            );
            let collective_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "device_graph.collective"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: coll_attrs,
                regions: collective_regions,
                custom_syntax: false,
                location: loc,
            });
            for (idx, ty) in collective_res_types.into_iter().enumerate() {
                let v = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty,
                    def: ValueDef::OpResult {
                        op: collective_op,
                        index: idx as u32,
                    },
                });
                module.ops[collective_op].results.push(v);
            }

            // Rewire uses of `staging_op` results to the mapped `collective_op` results.
            let mut replace_map: IndexMap<ValueId, ValueId> = IndexMap::new();
            for (&orig_y, &staging_res) in orig_yield_opds
                .iter()
                .zip(module.ops[staging_op].results.iter())
            {
                let idx = *result_mapping.get(&orig_y).ok_or_else(|| {
                    "staging yield value not materialized in any region".to_string()
                })?;
                replace_map.insert(staging_res, module.ops[collective_op].results[idx]);
            }

            if let Some(pos) = module.blocks[blk_id]
                .ops
                .iter()
                .position(|&id| id == staging_op)
            {
                module.blocks[blk_id].ops[pos] = collective_op;
            }
            for &ext_op in &module.blocks[blk_id].ops {
                for opd in &mut module.ops[ext_op].operands {
                    if let Some(&new_v) = replace_map.get(opd) {
                        *opd = new_v;
                    }
                }
            }
            module.ops.remove(staging_op);
            partitioned_count += 1;
        }
    }

    Ok(partitioned_count)
}

/// Checks whether `val` is used inside `reg_id` exclusively by
/// `mgp.tensor.extract.tensor_spec` operations (and has at least one use inside
/// `reg_id`), enabling shape-only values to cross device-graph region boundaries
/// without materializing device buffer transfers.
pub fn is_shape_only_in_region(module: &Module, reg_id: crate::ir::RegionId, val: ValueId) -> bool {
    let Some(reg) = module.regions.get(reg_id) else {
        return false;
    };
    let mut found_use = false;
    for &blk_id in &reg.blocks {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        for &op_id in &blk.ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.operands.contains(&val) {
                found_use = true;
                if op.name.full_name() != "mgp.tensor.extract.tensor_spec" {
                    return false;
                }
            }
        }
    }
    found_use
}

/// Reconciles inverse `builtin.unrealized_conversion_cast` chains across all
/// blocks in `module` and lowers remaining builder-to-device-context bridge
/// casts (`bridge_kind = "builder_to_device_context"`) into explicit
/// `mgp.device_graph.device_context` operations after auditing that every
/// downstream user is a host-side zero-kernel slice operation
/// (`mgp.buffer.slice`, `mgp.buffer.bulk_slice`, or `mgp.tensor.slice`).
pub fn reconcile_and_resolve_device_graph_builder_casts(
    module: &mut Module,
) -> Result<usize, String> {
    use std::collections::{HashMap, HashSet};

    let mut resolved_count = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    // Phase 1: Reconcile inverse `builtin.unrealized_conversion_cast` pairs
    // where `cast_b(cast_a(x))` restores the exact type of `x`.
    for &blk_id in &block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let ops: Vec<OpId> = blk.ops.clone();
        let mut forward_replacements: HashMap<ValueId, ValueId> = HashMap::new();
        let mut dead_casts: HashSet<OpId> = HashSet::new();

        for &op_id in &ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.name.full_name() != "builtin.unrealized_conversion_cast"
                || op.operands.len() != 1
                || op.results.len() != 1
            {
                continue;
            }
            let mut src_val = op.operands[0];
            while let Some(&mapped) = forward_replacements.get(&src_val) {
                src_val = mapped;
            }
            let dst_val = op.results[0];
            if let ValueDef::OpResult { op: def_op, .. } = module.values[src_val].def
                && let Some(prod_op) = module.ops.get(def_op)
                && prod_op.name.full_name() == "builtin.unrealized_conversion_cast"
                && prod_op.operands.len() == 1
                && prod_op.results.len() == 1
            {
                let orig_val = prod_op.operands[0];
                if module.values[orig_val].ty == module.values[dst_val].ty {
                    forward_replacements.insert(dst_val, orig_val);
                    dead_casts.insert(op_id);
                }
            }
        }

        if !forward_replacements.is_empty() {
            for &op_id in &ops {
                if let Some(op) = module.ops.get_mut(op_id) {
                    for opd in &mut op.operands {
                        while let Some(&mapped) = forward_replacements.get(opd) {
                            *opd = mapped;
                        }
                    }
                }
            }
            // Remove dead inverse casts and any intermediate casts that became unused.
            let mut used_vals: HashSet<ValueId> = HashSet::new();
            for &op_id in &module.blocks[blk_id].ops {
                if !dead_casts.contains(&op_id)
                    && let Some(op) = module.ops.get(op_id)
                {
                    for &opd in &op.operands {
                        used_vals.insert(opd);
                    }
                }
            }
            for &op_id in ops.iter().rev() {
                if dead_casts.contains(&op_id) {
                    module.ops.remove(op_id);
                    resolved_count += 1;
                } else if let Some(op) = module.ops.get(op_id)
                    && op.name.full_name() == "builtin.unrealized_conversion_cast"
                    && op.results.len() == 1
                    && !used_vals.contains(&op.results[0])
                {
                    let prod_operands = op.operands.clone();
                    dead_casts.insert(op_id);
                    module.ops.remove(op_id);
                    resolved_count += 1;
                    let _ = prod_operands;
                }
            }
            module.blocks[blk_id]
                .ops
                .retain(|op_id| !dead_casts.contains(op_id));
        }
    }

    // Phase 2: Audit and rewrite remaining `bridge_kind = "builder_to_device_context"`
    // `builtin.unrealized_conversion_cast` operations into `mgp.device_graph.device_context`.
    for &blk_id in &block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let ops: Vec<OpId> = blk.ops.clone();
        for &op_id in &ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.name.full_name() != "builtin.unrealized_conversion_cast" || op.results.len() != 1
            {
                continue;
            }
            let is_builder_bridge = matches!(
                op.attributes.get("bridge_kind"),
                Some(Attribute::String(s)) if s == "builder_to_device_context"
            );
            if !is_builder_bridge {
                continue;
            }
            let cast_res = op.results[0];
            for &user_id in &ops {
                if user_id == op_id {
                    continue;
                }
                let Some(user_op) = module.ops.get(user_id) else {
                    continue;
                };
                if user_op.operands.contains(&cast_res) {
                    let uname = user_op.name.full_name();
                    if uname != "mgp.buffer.slice"
                        && uname != "mgp.buffer.bulk_slice"
                        && uname != "mgp.tensor.slice"
                    {
                        return Err(format!(
                            "unaudited op '{}' uses backing device context of device_graph_builder",
                            uname
                        ));
                    }
                }
            }
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName {
                    dialect: "mgp".to_string(),
                    name: "device_graph.device_context".to_string(),
                };
                op_mut.attributes.shift_remove("bridge_kind");
                resolved_count += 1;
            }
        }
    }

    Ok(resolved_count)
}

// ============================================================================
// Raw-Block Storage Engine Layout, FDP Placement-Affinity
// Slot Allocator, and Dual-Container CRC32 Checkpoint Recovery
// ============================================================================

/// Validated device layout and checkpoint geometry for a raw-block L2 KV-cache
/// storage engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawBlockLayoutConfig {
    /// Device path identifier.
    pub device_path: String,
    /// Effective device capacity in bytes.
    pub capacity_bytes: u64,
    /// Block alignment in bytes (must be a power of two).
    pub block_align: u64,
    /// Per-slot header bytes (>= 24 and a multiple of `block_align`).
    pub header_bytes: u64,
    /// Total bytes per slot (`>= header_bytes + 1` and a multiple of `block_align`).
    pub slot_bytes: u64,
    /// Total metadata checkpoint region bytes at the start of the device.
    pub meta_total_bytes: u64,
    /// 8-byte ASCII magic identifying metadata checkpoint headers.
    pub meta_magic: [u8; 8],
    /// Metadata format version (`> 0`).
    pub meta_version: u32,
    /// Byte size of each of the two mirrored metadata containers.
    pub meta_container_bytes: u64,
    /// Byte offset where data slots begin.
    pub data_base_offset: u64,
    /// Maximum number of data slots fitting in `capacity_bytes - data_base_offset`.
    pub max_slots: usize,
}

impl RawBlockLayoutConfig {
    /// Validates raw-block layout parameters and computes mirrored metadata
    /// container geometry and data slot capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_path: &str,
        requested_capacity_bytes: u64,
        device_capacity_bytes: u64,
        block_align: u64,
        header_bytes: u64,
        slot_bytes: u64,
        meta_total_bytes: u64,
        meta_magic: [u8; 8],
        meta_version: u32,
    ) -> Result<Self, String> {
        if device_path.is_empty() {
            return Err("raw-block engine requires a non-empty device_path".to_string());
        }
        if block_align == 0 || !block_align.is_power_of_two() {
            return Err(format!(
                "block_align must be a power of 2, got {block_align}"
            ));
        }
        if header_bytes < 24 {
            return Err("header_bytes must be >= 24".to_string());
        }
        if !header_bytes.is_multiple_of(block_align) {
            return Err("header_bytes must be a multiple of block_align".to_string());
        }
        if slot_bytes <= header_bytes {
            return Err("slot_bytes must be >= header_bytes + 1".to_string());
        }
        if !slot_bytes.is_multiple_of(block_align) {
            return Err("slot_bytes must be a multiple of block_align".to_string());
        }
        if meta_total_bytes <= block_align {
            return Err("meta_total_bytes must provide room for metadata header".to_string());
        }
        if !meta_total_bytes.is_multiple_of(block_align) {
            return Err("meta_total_bytes must be a multiple of block_align".to_string());
        }
        if !meta_magic.is_ascii() {
            return Err("meta_magic must be ASCII bytes".to_string());
        }
        if meta_version == 0 {
            return Err("meta_version must be > 0".to_string());
        }
        let meta_container_bytes = ((meta_total_bytes / 2) / block_align) * block_align;
        if meta_container_bytes <= block_align {
            return Err(
                "meta_total_bytes must provide room for at least two metadata copies".to_string(),
            );
        }
        let effective_capacity = if requested_capacity_bytes > 0 {
            requested_capacity_bytes.min(device_capacity_bytes)
        } else {
            device_capacity_bytes
        };
        if meta_total_bytes >= effective_capacity {
            return Err("metadata region exceeds usable device capacity".to_string());
        }
        let data_base_offset = meta_total_bytes;
        let data_bytes = effective_capacity - data_base_offset;
        let max_slots = (data_bytes / slot_bytes) as usize;
        if max_slots == 0 {
            return Err("raw block capacity too small for slot size after metadata".to_string());
        }
        Ok(Self {
            device_path: device_path.to_string(),
            capacity_bytes: effective_capacity,
            block_align,
            header_bytes,
            slot_bytes,
            meta_total_bytes,
            meta_magic,
            meta_version,
            meta_container_bytes,
            data_base_offset,
            max_slots,
        })
    }

    /// Returns the byte offsets of the two mirrored metadata containers.
    #[must_use]
    pub fn metadata_container_offsets(&self) -> [u64; 2] {
        [0, self.meta_container_bytes]
    }

    /// Returns the usable payload capacity inside one metadata checkpoint container.
    #[must_use]
    pub fn meta_payload_capacity(&self) -> usize {
        (self.meta_container_bytes - self.block_align) as usize
    }

    /// Returns whether a logical payload of `payload_len` bytes fits in a single
    /// data slot under optional direct-I/O alignment rounding.
    #[must_use]
    pub fn payload_fits_slot(&self, payload_len: u64, requires_transfer_alignment: bool) -> bool {
        let payload_capacity = self.slot_bytes - self.header_bytes;
        if payload_len > payload_capacity {
            return false;
        }
        if requires_transfer_alignment {
            let aligned = payload_len.div_ceil(self.block_align) * self.block_align;
            return aligned <= payload_capacity;
        }
        true
    }
}

/// Maximum byte size (1 GiB) of a single fixed buffer region registered with
/// kernel zero-copy I/O.
pub const MAX_FIXED_BUFFER_REGION_BYTES: usize = 1 << 30;

/// Splits a contiguous stable memory range `(base_ptr, total_size)` into
/// `(region_ptr, region_size)` slices bounded by [`MAX_FIXED_BUFFER_REGION_BYTES`]
/// for zero-copy kernel buffer registration.
pub fn partition_fixed_buffer_registration_ranges(
    base_ptr: usize,
    total_size: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if base_ptr == 0 {
        return Err("fixed-buffer range pointer must be > 0".to_string());
    }
    if total_size == 0 {
        return Err("fixed-buffer range size must be > 0".to_string());
    }
    let mut ranges = Vec::new();
    let mut next_ptr = base_ptr;
    let mut remaining = total_size;
    while remaining > 0 {
        let region_size = remaining.min(MAX_FIXED_BUFFER_REGION_BYTES);
        ranges.push((next_ptr, region_size));
        next_ptr = next_ptr
            .checked_add(region_size)
            .ok_or_else(|| "fixed-buffer range pointer overflow".to_string())?;
        remaining -= region_size;
    }
    Ok(ranges)
}

/// Partitions `0..item_count` into at most `num_ranges` contiguous `[start, end)`
/// recovery intervals for parallel POSIX slot-header verification.
#[must_use]
pub fn build_recovery_item_ranges(item_count: usize, num_ranges: usize) -> Vec<(usize, usize)> {
    if item_count == 0 {
        return Vec::new();
    }
    let workers = num_ranges.max(1);
    let range_size = item_count.div_ceil(workers).max(1);
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < item_count {
        let end = (start + range_size).min(item_count);
        out.push((start, end));
        start = end;
    }
    out
}

/// Computes the standard IEEE 802.3 CRC32 checksum of `data`.
#[must_use]
pub fn compute_crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// 32-byte binary header for a mirrored raw-block metadata checkpoint container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawBlockMetaHeader {
    /// 8-byte ASCII magic.
    pub magic: [u8; 8],
    /// Metadata schema version.
    pub version: u32,
    /// Monotonically increasing checkpoint sequence number.
    pub seq: u64,
    /// Unpadded payload length in bytes.
    pub payload_len: u64,
    /// IEEE CRC32 checksum of the unpadded payload bytes.
    pub crc32: u32,
}

impl RawBlockMetaHeader {
    /// Serializes the metadata checkpoint header into 32 little-endian bytes.
    #[must_use]
    pub fn encode(&self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..20].copy_from_slice(&self.seq.to_le_bytes());
        buf[20..28].copy_from_slice(&self.payload_len.to_le_bytes());
        buf[28..32].copy_from_slice(&self.crc32.to_le_bytes());
        buf
    }

    /// Decodes and validates a 32-byte metadata checkpoint header against
    /// `expected_magic`, `expected_version`, and `max_payload_bytes`.
    #[must_use]
    pub fn decode(
        buf: &[u8],
        expected_magic: &[u8; 8],
        expected_version: u32,
        max_payload_bytes: usize,
    ) -> Option<Self> {
        if buf.len() < 32 {
            return None;
        }
        let magic: [u8; 8] = buf[0..8].try_into().ok()?;
        let version = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        let seq = u64::from_le_bytes(buf[12..20].try_into().ok()?);
        let payload_len = u64::from_le_bytes(buf[20..28].try_into().ok()?);
        let crc32 = u32::from_le_bytes(buf[28..32].try_into().ok()?);
        if &magic != expected_magic || version != expected_version {
            return None;
        }
        if payload_len == 0 || payload_len > max_payload_bytes as u64 {
            return None;
        }
        Some(Self {
            magic,
            version,
            seq,
            payload_len,
            crc32,
        })
    }
}

/// Selects the valid checkpoint container with the highest sequence number
/// whose payload passes IEEE CRC32 verification.
#[must_use]
pub fn select_latest_valid_checkpoint(
    containers: &[(&[u8], &[u8])],
    expected_magic: &[u8; 8],
    expected_version: u32,
    max_payload_bytes: usize,
) -> Option<(RawBlockMetaHeader, Vec<u8>)> {
    let mut best: Option<(RawBlockMetaHeader, Vec<u8>)> = None;
    for &(header_block, payload_region) in containers {
        let Some(hdr) = RawBlockMetaHeader::decode(
            header_block,
            expected_magic,
            expected_version,
            max_payload_bytes,
        ) else {
            continue;
        };
        let plen = hdr.payload_len as usize;
        if payload_region.len() < plen {
            continue;
        }
        let payload = &payload_region[..plen];
        if compute_crc32_ieee(payload) != hdr.crc32 {
            continue;
        }
        if best.as_ref().is_none_or(|(b_hdr, _)| hdr.seq > b_hdr.seq) {
            best = Some((hdr, payload.to_vec()));
        }
    }
    best
}

/// FDP placement-affinity-aware slot allocator and crash-recovery slot
/// rebuilder for a raw-block storage engine.
#[derive(Debug, Clone)]
pub struct RawBlockSlotAllocator {
    fdp_slot_affinity_enabled: bool,
    data_base_offset: u64,
    slot_bytes: u64,
    max_slots: usize,
    next_slot: usize,
    free_slots: Vec<usize>,
    free_slots_by_placement_id: std::collections::BTreeMap<u32, Vec<usize>>,
    slot_placement_ids: std::collections::BTreeMap<usize, u32>,
    /// Number of allocations satisfied from the matching FDP placement pool.
    pub fdp_slot_affinity_hit_count: usize,
    /// Number of FDP-tagged allocations satisfied by falling back to another free slot.
    pub fdp_slot_affinity_fallback_count: usize,
}

impl RawBlockSlotAllocator {
    /// Creates a new slot allocator for the given raw-block layout.
    #[must_use]
    pub fn new(layout: &RawBlockLayoutConfig, fdp_slot_affinity_enabled: bool) -> Self {
        Self {
            fdp_slot_affinity_enabled,
            data_base_offset: layout.data_base_offset,
            slot_bytes: layout.slot_bytes,
            max_slots: layout.max_slots,
            next_slot: 0,
            free_slots: Vec::new(),
            free_slots_by_placement_id: std::collections::BTreeMap::new(),
            slot_placement_ids: std::collections::BTreeMap::new(),
            fdp_slot_affinity_hit_count: 0,
            fdp_slot_affinity_fallback_count: 0,
        }
    }

    /// Converts a slot index to its device byte offset.
    #[must_use]
    pub fn slot_to_offset(&self, slot: usize) -> u64 {
        self.data_base_offset + (slot as u64) * self.slot_bytes
    }

    /// Converts a device byte offset to its slot index.
    #[must_use]
    pub fn offset_to_slot(&self, offset: u64) -> usize {
        ((offset - self.data_base_offset) / self.slot_bytes) as usize
    }

    /// Returns the number of currently available free slots in the recycle pool.
    #[must_use]
    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }

    /// Allocates a slot byte offset, preferring a recycled slot with matching
    /// `placement_id` when FDP slot affinity is enabled.
    pub fn allocate_slot(&mut self, placement_id: Option<u32>) -> Result<u64, String> {
        if placement_id == Some(0) {
            return Err("explicit placement_id 0 is rejected".to_string());
        }
        if self.fdp_slot_affinity_enabled
            && let Some(pid) = placement_id
            && let Some(pool) = self.free_slots_by_placement_id.get_mut(&pid)
            && let Some(slot) = pool.pop()
        {
            if pool.is_empty() {
                self.free_slots_by_placement_id.remove(&pid);
            }
            self.free_slots.retain(|&s| s != slot);
            self.fdp_slot_affinity_hit_count += 1;
            self.set_slot_placement_id(slot, placement_id);
            return Ok(self.slot_to_offset(slot));
        }

        if let Some(slot) = self.free_slots.pop() {
            self.remove_slot_from_affinity_pool(slot);
            if self.fdp_slot_affinity_enabled && placement_id.is_some() {
                self.fdp_slot_affinity_fallback_count += 1;
            }
            self.set_slot_placement_id(slot, placement_id);
            return Ok(self.slot_to_offset(slot));
        }

        if self.next_slot < self.max_slots {
            let slot = self.next_slot;
            self.next_slot += 1;
            self.set_slot_placement_id(slot, placement_id);
            return Ok(self.slot_to_offset(slot));
        }
        Err("no free slots available".to_string())
    }

    /// Recycles a slot identified by its byte offset back into the free pool.
    pub fn free_slot_by_offset(&mut self, offset: u64) {
        if offset < self.data_base_offset {
            return;
        }
        let slot = self.offset_to_slot(offset);
        self.append_free_slot(slot);
    }

    /// Rebuilds the free slot list from `0..next_slot` excluding `committed_offsets`
    /// so uncommitted crash-time in-flight reservations are reclaimed.
    pub fn rebuild_from_committed_offsets(
        &mut self,
        next_slot: usize,
        committed_offsets: &[u64],
    ) -> Result<(), String> {
        if next_slot > self.max_slots {
            return Err(format!(
                "checkpoint next_slot {next_slot} exceeds max_slots {}",
                self.max_slots
            ));
        }
        self.next_slot = next_slot;
        self.free_slots.clear();
        self.free_slots_by_placement_id.clear();
        self.slot_placement_ids.clear();

        let used: std::collections::BTreeSet<usize> = committed_offsets
            .iter()
            .filter(|&&off| off >= self.data_base_offset)
            .map(|&off| self.offset_to_slot(off))
            .collect();
        for slot in 0..next_slot {
            if !used.contains(&slot) {
                self.free_slots.push(slot);
            }
        }
        Ok(())
    }

    /// Validates recovered checkpoint entries `(key, offset, expected_identity, expected_size)`
    /// against `on_disk_headers` `(slot_identity, payload_len)`, recycling stale
    /// slots and returning the list of dropped keys.
    pub fn validate_recovered_entries(
        &mut self,
        entries: &[(String, u64, u64, u64)],
        on_disk_headers: &[Option<(u64, u64)>],
    ) -> Vec<String> {
        let mut dropped = Vec::new();
        for ((key, offset, expected_id, expected_size), hdr_opt) in
            entries.iter().zip(on_disk_headers.iter())
        {
            let stale = match hdr_opt {
                Some((disk_id, disk_len)) => {
                    *disk_id != *expected_id || *disk_len != *expected_size
                }
                None => true,
            };
            if stale {
                self.free_slot_by_offset(*offset);
                dropped.push(key.clone());
            }
        }
        dropped
    }

    fn append_free_slot(&mut self, slot: usize) {
        if slot >= self.max_slots || self.free_slots.contains(&slot) {
            return;
        }
        self.free_slots.push(slot);
        if self.fdp_slot_affinity_enabled
            && let Some(&pid) = self.slot_placement_ids.get(&slot)
        {
            let pool = self.free_slots_by_placement_id.entry(pid).or_default();
            if !pool.contains(&slot) {
                pool.push(slot);
            }
        }
    }

    fn remove_slot_from_affinity_pool(&mut self, slot: usize) {
        let Some(&pid) = self.slot_placement_ids.get(&slot) else {
            return;
        };
        if let Some(pool) = self.free_slots_by_placement_id.get_mut(&pid) {
            pool.retain(|&s| s != slot);
            if pool.is_empty() {
                self.free_slots_by_placement_id.remove(&pid);
            }
        }
    }

    fn set_slot_placement_id(&mut self, slot: usize, placement_id: Option<u32>) {
        if !self.fdp_slot_affinity_enabled {
            self.slot_placement_ids.remove(&slot);
            return;
        }
        if let Some(pid) = placement_id {
            self.slot_placement_ids.insert(slot, pid);
        } else {
            self.slot_placement_ids.remove(&slot);
        }
    }
}

// ============================================================================
// Device Primitive Specialization & Hierarchical Profile
// Scope Range Tracing
// ============================================================================

pub(super) fn parse_device_ref_spec(spec: &str) -> Option<(&str, i64)> {
    if let Some((label, id_str)) = spec.split_once(':') {
        if label.is_empty() {
            return None;
        }
        let id = id_str.parse::<i64>().ok()?;
        Some((label, id))
    } else if !spec.is_empty() {
        Some((spec, 0))
    } else {
        None
    }
}

/// Specializes device-targeted primitive operations across `module` by
/// appending the target device label (`.<label>`) to each operation's name
/// and rewriting `device_ref` attributes (`"<label>:<id>"`) into integer
/// `<attr>_id` attributes (`Attribute::Int(id)`).
pub fn prepare_device_primitives(module: &mut Module) -> usize {
    let mut rewritten = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let ops: Vec<OpId> = blk.ops.clone();
        for op_id in ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !op.regions.is_empty() {
                continue;
            }
            let Some(Attribute::String(dev_ref)) = op.attributes.get("device_ref") else {
                continue;
            };
            let Some((label, _)) = parse_device_ref_spec(dev_ref) else {
                continue;
            };
            let label_owned = label.to_string();

            let mut updated_attrs = IndexMap::new();
            for (k, v) in &op.attributes {
                if (k == "device_ref" || k.ends_with("_device_ref"))
                    && let Attribute::String(s) = v
                    && let Some((_, dev_id)) = parse_device_ref_spec(s)
                {
                    let new_key = format!("{k}_id");
                    updated_attrs.insert(new_key, Attribute::Int(dev_id));
                } else {
                    updated_attrs.insert(k.clone(), v.clone());
                }
            }

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name.name = format!("{}.{label_owned}", op_mut.name.name);
                op_mut.attributes = updated_attrs;
                rewritten += 1;
            }
        }
    }

    rewritten
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProfileScopeRun {
    first_op: OpId,
    last_op: OpId,
    layers: Vec<Vec<String>>,
}

pub(super) fn op_does_significant_work(op: &Operation) -> bool {
    let full = op.name.full_name();
    full == "func.call"
        || full == "mogg.call"
        || full == "mogg.call_opaque"
        || full == "mgp.routine.async"
        || op.attributes.contains_key("device_ref")
        || op.attributes.contains_key("device_ref_id")
}

pub(super) fn parse_op_profile_scope_layers(op: &Operation) -> Vec<Vec<String>> {
    let Some(Attribute::String(raw)) = op.attributes.get("profile_scope") else {
        return Vec::new();
    };
    raw.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|layer| {
            layer
                .split('/')
                .map(str::trim)
                .filter(|seg| !seg.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<String>>()
        })
        .filter(|v| !v.is_empty())
        .collect()
}

pub(super) fn add_ancestor_profile_scope_runs(
    leaf_runs: &[ProfileScopeRun],
    depth: usize,
    out_runs: &mut Vec<ProfileScopeRun>,
) {
    let mut group_first: Option<OpId> = None;
    let mut group_last: Option<OpId> = None;
    let mut group_prefix: Vec<String> = Vec::new();
    let mut group_size = 0usize;

    let flush = |first: &mut Option<OpId>,
                 last: &mut Option<OpId>,
                 prefix: &mut Vec<String>,
                 size: &mut usize,
                 out: &mut Vec<ProfileScopeRun>| {
        if *size > 1
            && let (Some(f), Some(l)) = (*first, *last)
        {
            out.push(ProfileScopeRun {
                first_op: f,
                last_op: l,
                layers: vec![prefix.clone()],
            });
        }
        *first = None;
        *last = None;
        prefix.clear();
        *size = 0;
    };

    for run in leaf_runs {
        let participates = run.layers.len() == 1 && run.layers[0].len() > depth + 1;
        if participates {
            let prefix = run.layers[0][..=depth].to_vec();
            if group_first.is_some() && prefix == group_prefix {
                group_last = Some(run.last_op);
                group_size += 1;
                continue;
            }
            flush(
                &mut group_first,
                &mut group_last,
                &mut group_prefix,
                &mut group_size,
                out_runs,
            );
            group_first = Some(run.first_op);
            group_last = Some(run.last_op);
            group_prefix = prefix;
            group_size = 1;
        } else {
            flush(
                &mut group_first,
                &mut group_last,
                &mut group_prefix,
                &mut group_size,
                out_runs,
            );
        }
    }
    flush(
        &mut group_first,
        &mut group_last,
        &mut group_prefix,
        &mut group_size,
        out_runs,
    );
}

pub(super) fn format_profile_scope_label(layers: &[Vec<String>], region_name: &str) -> String {
    if layers.is_empty() {
        return region_name.to_string();
    }
    let joined_layers: Vec<String> = layers.iter().map(|l| l.join("/")).collect();
    format!("{}/{region_name}", joined_layers.join("+"))
}

/// Synthesizes hierarchical and leaf profiler scope ranges
/// (`mogg.profile_scope.begin` / `mogg.profile_scope.end`) around contiguous
/// runs of significant operations inside functions in `module`. Returns the
/// total number of profiler scope ranges inserted.
pub fn add_profile_scope_tracing(module: &mut Module, trace_unlabelled: bool) -> usize {
    let mut inserted_ranges = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let region_name = module.ops[func_id]
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
            .unwrap_or("kernel")
            .to_string();
        if let Some(Attribute::String(abi)) = module.ops[func_id].attributes.get("abi")
            && abi != "mgp"
        {
            continue;
        }
        let Some(&reg_id) = module.ops[func_id].regions.first() else {
            continue;
        };
        let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let orig_ops = module.blocks[blk_id].ops.clone();
        if orig_ops.is_empty() {
            continue;
        }

        let terminator_opt = orig_ops.last().copied().filter(|&op_id| {
            let name = module.ops[op_id].name.full_name();
            name == "func.return" || name == "mogg.output" || name.ends_with(".yield")
        });
        let body_slice = if terminator_opt.is_some() {
            &orig_ops[..orig_ops.len() - 1]
        } else {
            &orig_ops[..]
        };

        let traceable: Vec<OpId> = body_slice
            .iter()
            .copied()
            .filter(|&op_id| module.ops[op_id].name.full_name() != "mogg.async.ready")
            .collect();
        if traceable.is_empty() {
            continue;
        }

        let mut leaf_runs: Vec<ProfileScopeRun> = Vec::new();
        for &op_id in &traceable {
            let op = &module.ops[op_id];
            if !op_does_significant_work(op) {
                if let Some(last) = leaf_runs.last_mut() {
                    last.last_op = op_id;
                }
                continue;
            }
            let layers = parse_op_profile_scope_layers(op);
            if layers.is_empty() {
                if let Some(last) = leaf_runs.last_mut() {
                    last.last_op = op_id;
                }
                continue;
            }
            if let Some(last) = leaf_runs.last_mut()
                && last.layers == layers
            {
                last.last_op = op_id;
                continue;
            }
            leaf_runs.push(ProfileScopeRun {
                first_op: op_id,
                last_op: op_id,
                layers,
            });
        }

        if leaf_runs.is_empty()
            && trace_unlabelled
            && let (Some(&first_op), Some(&last_op)) = (traceable.first(), traceable.last())
        {
            leaf_runs.push(ProfileScopeRun {
                first_op,
                last_op,
                layers: Vec::new(),
            });
        }

        let max_layer_len = leaf_runs
            .iter()
            .filter(|r| r.layers.len() == 1)
            .map(|r| r.layers[0].len())
            .max()
            .unwrap_or(0);

        let mut all_runs: Vec<ProfileScopeRun> = Vec::new();
        for depth in 0..max_layer_len.saturating_sub(1) {
            add_ancestor_profile_scope_runs(&leaf_runs, depth, &mut all_runs);
        }
        all_runs.extend(leaf_runs);

        for run in all_runs {
            let cur_ops = module.blocks[blk_id].ops.clone();
            let Some(start_idx) = cur_ops.iter().position(|&id| id == run.first_op) else {
                continue;
            };
            let Some(end_idx) = cur_ops.iter().position(|&id| id == run.last_op) else {
                continue;
            };
            if start_idx > end_idx {
                continue;
            }
            let has_sig = cur_ops[start_idx..=end_idx]
                .iter()
                .any(|&id| op_does_significant_work(&module.ops[id]));
            if !has_sig {
                continue;
            }

            let mut seen_colors: Vec<String> = Vec::new();
            for &id in &cur_ops[start_idx..=end_idx] {
                if let Some(Attribute::String(c)) = module.ops[id].attributes.get("profile_color")
                    && !seen_colors.contains(c)
                {
                    seen_colors.push(c.clone());
                }
            }
            let resolved_color = if seen_colors.len() == 1 {
                Some(seen_colors.remove(0))
            } else {
                None
            };

            let scope_name = format_profile_scope_label(&run.layers, &region_name);
            let range_ty = Type::Tensor(TensorType::new(vec![Dim::Static(1)], DType::I64));
            let loc = module.ops[func_id].location.clone();

            let begin_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg.profile_scope", "begin"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: {
                    let mut attrs = IndexMap::new();
                    attrs.insert("name".to_string(), Attribute::String(scope_name));
                    if let Some(color) = resolved_color {
                        attrs.insert("color".to_string(), Attribute::String(color));
                    }
                    attrs
                },
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let range_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: range_ty,
                def: ValueDef::OpResult {
                    op: begin_op,
                    index: 0,
                },
            });
            module.ops[begin_op].results.push(range_val);

            let end_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg.profile_scope", "end"),
                operands: smallvec::smallvec![range_val],
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });

            module.blocks[blk_id].ops.insert(start_idx, begin_op);
            let new_end_idx = module.blocks[blk_id]
                .ops
                .iter()
                .position(|&id| id == run.last_op)
                .unwrap_or(end_idx + 1);
            module.blocks[blk_id].ops.insert(new_end_idx + 1, end_op);
            inserted_ranges += 1;
        }
    }

    inserted_ranges
}
