//! Device-graph staging, collective partitioning, build stubs, and profile tracing.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, TensorType, Type, Value,
    ValueDef, ValueId,
};
use anyhow::Result;
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Batch 21 Step 3: Device-Graph Guard Lowering & Isolated Region Staging
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
// Batch 21 Step 4: Multi-Device Collective Graph Partitioner
// ============================================================================

/// Returns the device label (`device_ref` attribute, defaulting to `"cpu:0"`) of an operation.
fn op_device_ref_str(op: &Operation) -> String {
    match op.attributes.get("device_ref") {
        Some(Attribute::String(s)) => s.clone(),
        _ => "cpu:0".to_string(),
    }
}

/// Returns `true` if `device_ref` represents a host CPU execution context.
fn is_host_device_ref(device_ref: &str) -> bool {
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
// Batch 22 Step 1: Raw-Block Storage Engine Layout, FDP Placement-Affinity
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
// Batch 22 Step 2: Device Primitive Specialization & Hierarchical Profile
// Scope Range Tracing
// ============================================================================

fn parse_device_ref_spec(spec: &str) -> Option<(&str, i64)> {
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
struct ProfileScopeRun {
    first_op: OpId,
    last_op: OpId,
    layers: Vec<Vec<String>>,
}

fn op_does_significant_work(op: &Operation) -> bool {
    let full = op.name.full_name();
    full == "func.call"
        || full == "mogg.call"
        || full == "mogg.call_opaque"
        || full == "mgp.routine.async"
        || op.attributes.contains_key("device_ref")
        || op.attributes.contains_key("device_ref_id")
}

fn parse_op_profile_scope_layers(op: &Operation) -> Vec<Vec<String>> {
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

fn add_ancestor_profile_scope_runs(
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

fn format_profile_scope_label(layers: &[Vec<String>], region_name: &str) -> String {
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

// ============================================================================
// Batch 22 Step 3: Device-Graph Recording Stub Generator & Builder Pool Alloc
// Lowering
// ============================================================================

#[derive(Debug, Clone)]
struct RecordingStubInfo {
    stub_name: String,
    context_arg_index: usize,
}

fn is_producer_dispatch_op(op: &Operation) -> bool {
    matches!(
        op.name.full_name().as_str(),
        "mogg.call_opaque" | "mogg.call.foreach" | "mogg.call.foreach_tile" | "mogg.call.view_copy"
    )
}

fn create_recording_stub_function(
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
// Batch 22 Step 4: Bottom-Up Model Call-Graph Slot Globalization & Subgraph
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

fn find_model_context_create(module: &Module, func_id: OpId) -> Option<(OpId, i64)> {
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

fn topo_sort_model_call_graph(
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
// Batch 22 Step 5: Rowwise Kernel Legalizer (Primary Input Selection, Monoid
// Binding, Row Store Grouping, and Launch Parameter Synthesis)
// ============================================================================

const CHEAP_ROWWISE_ELEMENT_COST: u32 = 1;
const EXPENSIVE_ROWWISE_ELEMENT_COST: u32 = 16;

fn dtype_bitwidth(ty: &Type) -> u32 {
    match ty {
        Type::Tensor(t) => t.dtype.bit_width() as u32,
        _ => 32,
    }
}

fn rowwise_call_elementwise_cost(op: &Operation) -> u32 {
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
        let row_arg = *rw_args.last().unwrap();

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
// Batch 23 Step 1: Dynamic File-Backed Zero-Copy L2 Storage Agent, L1 Page
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

        let term_id = *block_ops.last().unwrap();
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
