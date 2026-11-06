//! Map index view hoisting, contiguous view raising, and device-graph dependencies.

use super::{count_value_uses_in_block, extract_u64_attr};
use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, TensorType, Type, Value,
    ValueDef, ValueId,
};
use anyhow::Result;
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Loop-Invariant Index-Transform View Hoister
// ============================================================================

/// Returns `true` if `op_name` is a `map.index.*` coordinate transform operation.
#[must_use]
pub fn is_map_index_transform_op(op_name: &str) -> bool {
    matches!(
        op_name,
        "map.index.slice" | "map.index.broadcast" | "map.index.reshape" | "map.index.transpose"
    )
}

/// Walks backward from `index_val` through `map.index.*` transform ops, returning the chain in
/// closest-to-load order.
pub(super) fn collect_index_transform_chain(module: &Module, index_val: ValueId) -> Vec<OpId> {
    let mut chain = Vec::new();
    let mut cur = index_val;
    while let ValueDef::OpResult { op: def_op, .. } = module.values[cur].def {
        let op = &module.ops[def_op];
        if !is_map_index_transform_op(&op.name.full_name()) || op.operands.is_empty() {
            break;
        }
        chain.push(def_op);
        cur = op.operands[0];
    }
    chain
}

/// Returns `true` if `transform_op`'s output index reaches a `map.store`'s index operand
/// (`operands[1]`) directly or through further `map.index.*` transforms in `block_id`.
pub(super) fn transitively_feeds_store_index(
    module: &Module,
    block_id: BlockId,
    transform_op: OpId,
) -> bool {
    let Some(&res0) = module.ops[transform_op].results.first() else {
        return false;
    };
    let mut worklist = vec![res0];
    let mut visited = IndexSet::new();
    visited.insert(res0);

    while let Some(idx_val) = worklist.pop() {
        for &op_id in &module.blocks[block_id].ops {
            let op = &module.ops[op_id];
            let full = op.name.full_name();
            if full == "map.store" {
                // `map.store` operands: [tensor, index, value] (or [index, value] if 2 operands).
                let idx_slot = if op.operands.len() >= 3 { 1 } else { 0 };
                if op.operands.get(idx_slot) == Some(&idx_val) {
                    return true;
                }
            } else if is_map_index_transform_op(&full)
                && op.operands.first() == Some(&idx_val)
                && let Some(&next_idx) = op.results.first()
                && visited.insert(next_idx)
            {
                worklist.push(next_idx);
            }
        }
    }
    false
}

/// Length of the leading prefix of `chain` (closest-to-load first) that does not transitively
/// feed any `map.store` index.
pub(super) fn hoistable_index_prefix_len(
    module: &Module,
    block_id: BlockId,
    chain: &[OpId],
) -> usize {
    chain
        .iter()
        .take_while(|&&op_id| !transitively_feeds_store_index(module, block_id, op_id))
        .count()
}

/// Hoists chains of loop-invariant `map.index.*` coordinate transforms feeding `map.load` (that do
/// not transitively feed a `map.store` index) into `.stride` tensor view operations (e.g.
/// `map.index.slice.stride`), rewires `map.load` and `map.call_elementwise` to use the outermost
/// surviving loop index, and erases dead `map.index.*` ops. Returns the number of hoisted view ops.
pub fn hoist_loop_invariant_index_views(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut hoisted_count = 0usize;

    for block_id in block_ids {
        let orig_ops = module.blocks[block_id].ops.clone();
        let loads: Vec<OpId> = orig_ops
            .iter()
            .copied()
            .filter(|&op_id| {
                module.ops[op_id].name.full_name() == "map.load"
                    && module.ops[op_id].operands.len() >= 2
            })
            .collect();

        for load_op in loads {
            let tensor_val = module.ops[load_op].operands[0];
            let index_val = module.ops[load_op].operands[1];
            let mut chain = collect_index_transform_chain(module, index_val);
            let prefix = hoistable_index_prefix_len(module, block_id, &chain);
            if prefix == 0 {
                continue;
            }
            chain.truncate(prefix);
            let Some(&outermost_op) = chain.last() else {
                continue;
            };
            let outermost_index = module.ops[outermost_op].operands[0];
            let tensor_ty = module.values[tensor_val].ty.clone();

            let mut cur_tensor = tensor_val;
            let mut stride_ops_to_insert = Vec::with_capacity(chain.len());

            for &transform_op in &chain {
                let t_op = &module.ops[transform_op];
                let stride_name = format!("{}.stride", t_op.name.name);
                let mut stride_operands = SmallVec::new();
                stride_operands.push(cur_tensor);
                for &extra_opd in t_op.operands.iter().skip(1) {
                    stride_operands.push(extra_opd);
                }
                let attrs = t_op.attributes.clone();
                let loc = t_op.location.clone();

                let stride_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("map", stride_name),
                    operands: stride_operands,
                    results: SmallVec::new(),
                    attributes: attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc,
                });
                let new_view = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: tensor_ty.clone(),
                    def: ValueDef::OpResult {
                        op: stride_op,
                        index: 0,
                    },
                });
                module.ops[stride_op].results.push(new_view);
                cur_tensor = new_view;
                stride_ops_to_insert.push(stride_op);
                hoisted_count += 1;
            }

            if let Some(pos) = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == load_op)
            {
                for (i, s_op) in stride_ops_to_insert.into_iter().enumerate() {
                    module.blocks[block_id].ops.insert(pos + i, s_op);
                }
            }

            module.ops[load_op].operands[0] = cur_tensor;
            module.ops[load_op].operands[1] = outermost_index;
        }

        // Rewire `map.call_elementwise` ops that share a hoistable index chain.
        let elem_calls: Vec<OpId> = module.blocks[block_id]
            .ops
            .iter()
            .copied()
            .filter(|&op_id| {
                module.ops[op_id].name.full_name() == "map.call_elementwise"
                    && !module.ops[op_id].operands.is_empty()
            })
            .collect();
        for call_op in elem_calls {
            let idx_val = module.ops[call_op].operands[0];
            let chain = collect_index_transform_chain(module, idx_val);
            let prefix = hoistable_index_prefix_len(module, block_id, &chain);
            if prefix > 0 {
                let outermost_idx = module.ops[chain[prefix - 1]].operands[0];
                module.ops[call_op].operands[0] = outermost_idx;
            }
        }

        // Erase unused `map.index.*` transforms in reverse order so outer hops become dead too.
        let transforms: Vec<OpId> = module.blocks[block_id]
            .ops
            .iter()
            .copied()
            .filter(|&op_id| is_map_index_transform_op(&module.ops[op_id].name.full_name()))
            .collect();
        for &t_op in transforms.iter().rev() {
            if let Some(&res0) = module.ops[t_op].results.first()
                && count_value_uses_in_block(module, block_id, res0) == 0
            {
                module.blocks[block_id].ops.retain(|&id| id != t_op);
            }
        }
    }

    hoisted_count
}

// ============================================================================
// Contiguous View Chain Raiser
// ============================================================================

/// Returns the first input dimension index `k` of `in_dims` with extent `!= 1`, or `in_dims.len()`
/// when all dimensions have extent `1`.
#[must_use]
pub fn find_contiguous_slice_dim(in_dims: &[i64]) -> usize {
    in_dims
        .iter()
        .position(|&d| d != 1)
        .unwrap_or(in_dims.len())
}

/// Returns `Some(k)` if a slice over `(in_dims, out_dims, starts, stops, steps)` is a contiguous
/// static unit-step slice along dimension `k = find_contiguous_slice_dim(in_dims)` (all other
/// dimensions `i != k` retain their full input extent `in_dims[i] == out_dims[i]`, all bounds are
/// non-negative/static, and `steps[k] == 1`).
#[must_use]
pub fn match_contiguous_slice_dim(
    in_dims: &[i64],
    out_dims: &[i64],
    starts: &[i64],
    stops: &[i64],
    steps: &[i64],
) -> Option<usize> {
    let rank = in_dims.len();
    if rank == 0
        || out_dims.len() != rank
        || starts.len() != rank
        || stops.len() != rank
        || steps.len() != rank
    {
        return None;
    }
    if in_dims.iter().any(|&d| d < 0) || out_dims.iter().any(|&d| d < 0) {
        return None;
    }
    let k = find_contiguous_slice_dim(in_dims);
    if k >= rank {
        return None;
    }
    for i in 0..rank {
        if i != k && in_dims[i] != out_dims[i] {
            return None;
        }
    }
    if starts[k] < 0 || stops[k] < starts[k] || steps[k] != 1 {
        return None;
    }
    Some(k)
}

#[derive(Debug, Clone)]
pub(super) enum RaisableViewStep {
    Reshape,
    Slice { start_k: i64, stop_k: i64 },
}

pub(super) fn extract_i64_vec_attr(
    attrs: &IndexMap<String, Attribute>,
    key: &str,
) -> Option<Vec<i64>> {
    match attrs.get(key)? {
        Attribute::DenseIntArray(v) => Some(v.clone()),
        Attribute::Array(items) => items
            .iter()
            .map(|a| extract_u64_attr(a).map(|u| u as i64))
            .collect(),
        _ => None,
    }
}

/// Matches a `map.iter.elementwise` op's `"view_chain"` attribute (ordered from output loop index
/// toward input load, e.g. `["slice", "reshape"]`) and validates that any slice step is a
/// contiguous static unit-step slice.
pub(super) fn match_raisable_view_steps(op: &Operation) -> Option<Vec<RaisableViewStep>> {
    if op.name.full_name() != "map.iter.elementwise"
        || op.operands.is_empty()
        || op.results.len() != 1
    {
        return None;
    }
    let Attribute::Array(items) = op.attributes.get("view_chain")? else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let mut steps = Vec::with_capacity(items.len());
    for item in items {
        let kind = item.as_str()?;
        if kind == "reshape" {
            steps.push(RaisableViewStep::Reshape);
        } else if kind == "slice" {
            let in_dims = extract_i64_vec_attr(&op.attributes, "in_dims")?;
            let out_dims = extract_i64_vec_attr(&op.attributes, "out_dims")?;
            let starts = extract_i64_vec_attr(&op.attributes, "starts")?;
            let stops = extract_i64_vec_attr(&op.attributes, "stops")?;
            let step_vals = extract_i64_vec_attr(&op.attributes, "steps")?;
            let k = match_contiguous_slice_dim(&in_dims, &out_dims, &starts, &stops, &step_vals)?;
            steps.push(RaisableViewStep::Slice {
                start_k: starts[k],
                stop_k: stops[k],
            });
        } else {
            return None;
        }
    }
    Some(steps)
}

/// Raises unfused contiguous `map.iter.elementwise` view kernels back to zero-copy data operations
/// (`mo.static.reshape` and `mogg.slice`) in reversed (input-to-output) order. Returns
/// `(num_slices_raised, num_reshapes_raised)`.
pub fn raise_back_contiguous_views(module: &mut Module) -> (usize, usize) {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut slices_raised = 0usize;
    let mut reshapes_raised = 0usize;

    for block_id in block_ids {
        let orig_ops = module.blocks[block_id].ops.clone();
        let mut new_block_ops = Vec::with_capacity(orig_ops.len());

        for op_id in orig_ops {
            let Some(steps) = match_raisable_view_steps(&module.ops[op_id]) else {
                new_block_ops.push(op_id);
                continue;
            };
            let in_val = module.ops[op_id].operands[0];
            let orig_out_val = module.ops[op_id].results[0];
            let out_ty = module.values[orig_out_val].ty.clone();
            let loc = module.ops[op_id].location.clone();

            let mut cur_val = in_val;
            // Body transforms run from output index to input load; data ops run in reverse.
            for step in steps.iter().rev() {
                match *step {
                    RaisableViewStep::Reshape => {
                        let reshape_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mo.static", "reshape"),
                            operands: SmallVec::from_slice(&[cur_val]),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let next_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: out_ty.clone(),
                            def: ValueDef::OpResult {
                                op: reshape_op,
                                index: 0,
                            },
                        });
                        module.ops[reshape_op].results.push(next_val);
                        new_block_ops.push(reshape_op);
                        cur_val = next_val;
                        reshapes_raised += 1;
                    }
                    RaisableViewStep::Slice { start_k, stop_k } => {
                        let mut slice_attrs = IndexMap::new();
                        slice_attrs.insert("start".to_string(), Attribute::Int(start_k));
                        slice_attrs.insert("stop".to_string(), Attribute::Int(stop_k));
                        let slice_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mogg", "slice"),
                            operands: SmallVec::from_slice(&[cur_val]),
                            results: SmallVec::new(),
                            attributes: slice_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let next_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: out_ty.clone(),
                            def: ValueDef::OpResult {
                                op: slice_op,
                                index: 0,
                            },
                        });
                        module.ops[slice_op].results.push(next_val);
                        new_block_ops.push(slice_op);
                        cur_val = next_val;
                        slices_raised += 1;
                    }
                }
            }

            for &cand_op in &module.blocks[block_id].ops {
                if cand_op == op_id {
                    continue;
                }
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == orig_out_val {
                        *opd = cur_val;
                    }
                }
            }
        }

        module.blocks[block_id].ops = new_block_ops;
    }

    (slices_raised, reshapes_raised)
}

// ============================================================================
// Mixed-Precision Accumulating Epilogue Buffer Allocator
// ============================================================================

/// Returns `true` if `kernel_name` names an accumulating opaque operation (`mo.concat`,
/// `mo.matmul`, `mo.batch_matmul`, `mo.conv`, or `mo.conv_transpose`) that requires accumulating
/// in its pre-cast element type when followed by a dtype-changing epilogue.
#[must_use]
pub fn is_accumulating_opaque_kernel_name(kernel_name: &str) -> bool {
    kernel_name.contains("mo.concat")
        || kernel_name.contains("mo.matmul")
        || kernel_name.contains("mo.batch_matmul")
        || kernel_name.contains("mo.conv")
        || kernel_name.contains("mo.conv_transpose")
}

/// Assigns an explicit intermediate store-target tensor to each `map.iter.epilogue` op in `module`.
/// When the feeding `map.iter.opaque` kernel is an accumulating op (`mo.matmul`, `mo.conv`, etc.)
/// and the epilogue casts to a different output element dtype, allocates a pre-cast scratch
/// accumulation tensor result on the kernel (before any trailing chain result). Returns
/// `(scratch_buffers_allocated, direct_targets_bound)`.
pub fn allocate_accumulating_epilogue_buffers(module: &mut Module) -> (usize, usize) {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut scratch_allocated = 0usize;
    let mut direct_bound = 0usize;

    for block_id in block_ids {
        let ops_snapshot = module.blocks[block_id].ops.clone();
        for &epi_op in &ops_snapshot {
            if module.ops[epi_op].name.full_name() != "map.iter.epilogue" {
                continue;
            }
            if matches!(
                module.ops[epi_op].attributes.get("has_intermediate_tensor"),
                Some(Attribute::Bool(true))
            ) {
                continue;
            }
            if module.ops[epi_op].operands.is_empty() || module.ops[epi_op].results.is_empty() {
                continue;
            }
            let in_val = module.ops[epi_op].operands[0];
            let store_target = if module.ops[epi_op].operands.len() >= 2 {
                module.ops[epi_op].operands[1]
            } else {
                module.ops[epi_op].results[0]
            };

            let ValueDef::OpResult { op: kernel_op, .. } = module.values[in_val].def else {
                continue;
            };
            if module.ops[kernel_op].name.full_name() != "map.iter.opaque" {
                continue;
            }
            let kernel_name = module.ops[kernel_op]
                .attributes
                .get("kernel_name")
                .and_then(Attribute::as_str)
                .unwrap_or("")
                .to_string();

            let in_tensor_ty = module.values[in_val].ty.as_tensor().cloned();
            let target_tensor_ty = module.values[store_target].ty.as_tensor().cloned();
            let (Some(in_tty), Some(target_tty)) = (in_tensor_ty, target_tensor_ty) else {
                continue;
            };

            let dtype_changes = in_tty.dtype != target_tty.dtype;
            if dtype_changes && is_accumulating_opaque_kernel_name(&kernel_name) {
                let scratch_ty =
                    Type::Tensor(TensorType::new(target_tty.shape.clone(), in_tty.dtype));
                // Insert scratch result on `kernel_op` before any trailing chain (`tensor<1xi64>`)
                // result.
                let res_len = module.ops[kernel_op].results.len();
                let has_trailing_chain = module.ops[kernel_op]
                    .results
                    .last()
                    .and_then(|&r| module.values[r].ty.as_tensor())
                    .is_some_and(|t| t.shape == [Dim::Static(1)] && t.dtype == DType::I64);
                let insert_idx = if has_trailing_chain && res_len >= 1 {
                    res_len - 1
                } else {
                    res_len
                };

                let scratch_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: scratch_ty,
                    def: ValueDef::OpResult {
                        op: kernel_op,
                        index: insert_idx as u32,
                    },
                });
                module.ops[kernel_op]
                    .results
                    .insert(insert_idx, scratch_val);
                if has_trailing_chain {
                    let last_idx = module.ops[kernel_op].results.len() - 1;
                    let chain_val = module.ops[kernel_op].results[last_idx];
                    module.values[chain_val].def = ValueDef::OpResult {
                        op: kernel_op,
                        index: last_idx as u32,
                    };
                }

                module.ops[epi_op].operands.push(scratch_val);
                module.ops[epi_op]
                    .attributes
                    .insert("has_intermediate_tensor".to_string(), Attribute::Bool(true));
                module.ops[epi_op].attributes.insert(
                    "uses_scratch_accumulator".to_string(),
                    Attribute::Bool(true),
                );
                scratch_allocated += 1;
            } else {
                module.ops[epi_op].operands.push(store_target);
                module.ops[epi_op]
                    .attributes
                    .insert("has_intermediate_tensor".to_string(), Attribute::Bool(true));
                module.ops[epi_op].attributes.insert(
                    "uses_scratch_accumulator".to_string(),
                    Attribute::Bool(false),
                );
                direct_bound += 1;
            }
        }
    }

    (scratch_allocated, direct_bound)
}

// ============================================================================
// Staged Device-Graph Dependency Synthesizer
// ============================================================================

/// Returns the ordering source value for `operand`: if `operand` is produced by
/// `mgp.device_graph.stable.tensor` whose address (`operands[0]`) is produced by
/// `mgp.device_graph.stable.place`, returns the placed tensor (`place_op.operands[0]`) so
/// per-device ordering is preserved from the producer of the placed tensor.
#[must_use]
pub fn resolve_device_graph_ordering_source(module: &Module, operand: ValueId) -> ValueId {
    if let ValueDef::OpResult { op: stable_op, .. } = module.values[operand].def
        && module.ops[stable_op].name.full_name() == "mgp.device_graph.stable.tensor"
        && let Some(&addr_val) = module.ops[stable_op].operands.first()
        && let ValueDef::OpResult { op: place_op, .. } = module.values[addr_val].def
        && module.ops[place_op].name.full_name() == "mgp.device_graph.stable.place"
        && let Some(&placed_tensor) = module.ops[place_op].operands.first()
    {
        return placed_tensor;
    }
    operand
}

/// Converts staged `mogg.routine.async` operations in `module` into
/// `mgp.device_graph.dependencies` operations with explicit producer-consumer DAG edges (resolving
/// `mgp.device_graph.stable.tensor` -> `mgp.device_graph.stable.place` to its placed tensor
/// producer) and aggregates `"constant_resources"` onto the enclosing function. Returns the number
/// of converted `mogg.routine.async` operations.
pub fn compute_device_graph_dependencies(module: &mut Module) -> usize {
    let func_ids = module.functions();
    let mut converted_count = 0usize;

    for f_id in func_ids {
        let mut resources_seen: IndexSet<String> = IndexSet::new();
        let reg_ids = module.ops[f_id].regions.clone();

        for reg_id in reg_ids {
            let block_ids = module.regions[reg_id].blocks.clone();
            for blk_id in block_ids {
                let routine_ops: Vec<OpId> = module.blocks[blk_id]
                    .ops
                    .iter()
                    .copied()
                    .filter(|&op_id| module.ops[op_id].name.full_name() == "mogg.routine.async")
                    .collect();
                if routine_ops.is_empty() {
                    continue;
                }

                let mut op_to_node_info: IndexMap<OpId, (usize, ValueId)> = IndexMap::new();
                for (node_idx, &op_id) in routine_ops.iter().enumerate() {
                    let res_idx = module.ops[op_id].results.len() as u32;
                    let node_token = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: Type::Tensor(TensorType::new(vec![Dim::Static(1)], DType::I64)),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: res_idx,
                        },
                    });
                    module.ops[op_id].results.push(node_token);
                    op_to_node_info.insert(op_id, (node_idx, node_token));
                }

                for &op_id in &routine_ops {
                    let orig_operands = module.ops[op_id].operands.clone();
                    let mut dep_indices: Vec<i64> = Vec::new();
                    let mut dep_tokens: Vec<ValueId> = Vec::new();
                    let mut seen_producers: IndexSet<OpId> = IndexSet::new();

                    for &opd in &orig_operands {
                        let src = resolve_device_graph_ordering_source(module, opd);
                        if let ValueDef::OpResult { op: prod_op, .. } = module.values[src].def
                            && let Some(&(prod_idx, prod_token)) = op_to_node_info.get(&prod_op)
                            && seen_producers.insert(prod_op)
                        {
                            dep_indices.push(prod_idx as i64);
                            dep_tokens.push(prod_token);
                        }
                    }

                    if let Some(Attribute::Array(items)) =
                        module.ops[op_id].attributes.get("constant_resources")
                    {
                        for item in items {
                            if let Some(s) = item.as_str() {
                                resources_seen.insert(s.to_string());
                            }
                        }
                    }

                    module.ops[op_id].name = OpName::new("mgp.device_graph", "dependencies");
                    for tok in dep_tokens {
                        module.ops[op_id].operands.push(tok);
                    }
                    module.ops[op_id].attributes.insert(
                        "dep_node_indices".to_string(),
                        Attribute::DenseIntArray(dep_indices),
                    );
                    converted_count += 1;
                }
            }
        }

        if !resources_seen.is_empty() {
            let agg: Vec<Attribute> = resources_seen.into_iter().map(Attribute::String).collect();
            module.ops[f_id]
                .attributes
                .insert("constant_resources".to_string(), Attribute::Array(agg));
        }
    }

    converted_count
}

// ============================================================================
// Optimal Bloom Filter & Sub-Chunk Sliding-Window Block Downsampler
// ============================================================================

/// Downsamples `block_ids` for a kernel group when `keep_blocks_per_chunk < total_blocks_per_chunk`
/// (such as sub-chunk sliding-window attention), retaining the trailing `keep_blocks_per_chunk`
/// blocks of each `total_blocks_per_chunk` chunk.
pub fn downsample_sliding_window_block_ids(
    block_ids: &[u64],
    total_blocks_per_chunk: usize,
    keep_blocks_per_chunk: usize,
) -> Result<Vec<u64>, String> {
    if total_blocks_per_chunk == 0 || keep_blocks_per_chunk == 0 {
        return Err("blocks per chunk must be non-zero".to_string());
    }
    if keep_blocks_per_chunk > total_blocks_per_chunk {
        return Err("keep_blocks_per_chunk cannot exceed total_blocks_per_chunk".to_string());
    }
    if !block_ids.len().is_multiple_of(total_blocks_per_chunk) {
        return Err(format!(
            "block_ids len {} not a multiple of {total_blocks_per_chunk}",
            block_ids.len()
        ));
    }
    let mut out =
        Vec::with_capacity((block_ids.len() / total_blocks_per_chunk) * keep_blocks_per_chunk);
    for chunk in block_ids.chunks_exact(total_blocks_per_chunk) {
        out.extend_from_slice(&chunk[total_blocks_per_chunk - keep_blocks_per_chunk..]);
    }
    Ok(out)
}

/// Recalculates the number of blocks to skip for a batch of chunks when the sliding-window size
/// (`blocks_per_window`) is smaller than or equal to `blocks_per_chunk`.
#[must_use]
pub fn recalculate_sliding_window_blocks_to_skip(
    blocks_per_chunk: usize,
    blocks_per_window: usize,
    blocks_to_skip: usize,
) -> usize {
    if blocks_per_chunk == 0 || blocks_per_chunk == blocks_per_window {
        return blocks_to_skip;
    }
    let full_windows_to_skip = blocks_to_skip / blocks_per_chunk;
    let tail_blocks = blocks_to_skip % blocks_per_chunk;
    let dropped_prefix = blocks_per_chunk.saturating_sub(blocks_per_window);
    let tail_blocks_to_skip = tail_blocks.saturating_sub(dropped_prefix);
    full_windows_to_skip
        .saturating_mul(blocks_per_window)
        .saturating_add(tail_blocks_to_skip)
}

/// Returns `(batch_start_idx, batch_len)` ranges over `0..total_items` in batches of `batch_size`
/// after skipping the first `skip_count` items.
pub fn batched_ranges_with_skip(
    total_items: usize,
    batch_size: usize,
    skip_count: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if batch_size == 0 {
        return Err("batch_size must be at least 1".to_string());
    }
    let mut ranges = Vec::new();
    let mut start = skip_count;
    while start < total_items {
        let len = (total_items - start).min(batch_size);
        ranges.push((start, len));
        start += len;
    }
    Ok(ranges)
}

/// Memory-efficient Bloom filter sized from `(expected_elements, false_positive_rate)` using
/// $m = \lfloor -(n \ln p) / (\ln 2)^2 \rfloor$ and $k = \max(1, \lfloor (m / n) \ln 2 \rfloor)$.
#[derive(Debug, Clone)]
pub struct OptimalBloomFilter {
    size_bits: usize,
    hash_count: usize,
    words: Vec<u32>,
    item_count: usize,
}

impl OptimalBloomFilter {
    /// Constructs a new [`OptimalBloomFilter`] for `expected_elements` and `false_positive_rate`.
    #[must_use]
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let n = expected_elements.max(1) as f64;
        let p = false_positive_rate.clamp(1e-9, 0.999);
        let ln2 = std::f64::consts::LN_2;
        let size_bits = (-(n * p.ln()) / (ln2 * ln2)).floor().max(32.0) as usize;
        let hash_count = (((size_bits as f64) / n) * ln2).floor().max(1.0) as usize;
        let num_words = size_bits.div_ceil(32);
        Self {
            size_bits,
            hash_count,
            words: vec![0u32; num_words],
            item_count: 0,
        }
    }

    /// Optimal bit-array size $m$ in bits.
    #[must_use]
    pub fn size_bits(&self) -> usize {
        self.size_bits
    }

    /// Optimal number of hash functions $k$.
    #[must_use]
    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    /// Number of unique items inserted.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Computes the $k$ bit positions in `0..self.size_bits` for `key`.
    #[must_use]
    pub fn hash_positions(&self, key: &[u8]) -> Vec<usize> {
        (0..self.hash_count)
            .map(|i| {
                let mut h =
                    0x9E37_79B9_7F4A_7C15u64 ^ ((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                for &b in key {
                    h = (h ^ u64::from(b)).wrapping_mul(0x94D0_49BB_1331_11EB);
                }
                h ^= h >> 31;
                (h as usize) % self.size_bits
            })
            .collect()
    }

    /// Inserts `key` into the Bloom filter.
    pub fn add(&mut self, key: &[u8]) {
        for pos in self.hash_positions(key) {
            self.words[pos >> 5] |= 1u32 << (pos & 31);
        }
        self.item_count += 1;
    }

    /// Inserts a batch of precomputed hash-position lists and returns how many items had at least
    /// one previously unset bit.
    pub fn add_batch_with_positions_and_check(&mut self, positions_list: &[Vec<usize>]) -> usize {
        let mut unique_count = 0usize;
        for positions in positions_list {
            let is_new = positions.iter().any(|&pos| {
                let p = pos % self.size_bits;
                (self.words[p >> 5] & (1u32 << (p & 31))) == 0
            });
            if is_new {
                for &pos in positions {
                    let p = pos % self.size_bits;
                    self.words[p >> 5] |= 1u32 << (p & 31);
                }
                unique_count += 1;
            }
        }
        self.item_count += unique_count;
        unique_count
    }

    /// Returns `true` if all $k$ bit positions for `key` are set.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        self.hash_positions(key)
            .into_iter()
            .all(|pos| (self.words[pos >> 5] & (1u32 << (pos & 31))) != 0)
    }

    /// Number of bits currently set to `1`.
    #[must_use]
    pub fn bits_set(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Fraction of bits currently set (`bits_set / size_bits`).
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        if self.size_bits == 0 {
            0.0
        } else {
            self.bits_set() as f64 / self.size_bits as f64
        }
    }

    /// Resets all bits and item count to zero.
    pub fn clear(&mut self) {
        self.words.fill(0);
        self.item_count = 0;
    }
}

// ============================================================================
// Transitive Device-Graph Dependency Edge Redundancy Eliminator
// ============================================================================

/// Eliminates node indices in `needles` that are transitively reachable from other node indices in
/// `needles` via `edges_by_node` (using wave-based BFS with a `visited` set so even malformed
/// cycles terminate cleanly).
#[must_use]
pub fn eliminate_implied_dependency_indices(
    needles: &[i64],
    edges_by_node: &IndexMap<i64, Vec<i64>>,
) -> Vec<i64> {
    let mut surviving: IndexSet<i64> = needles.iter().copied().collect();
    let mut visited: IndexSet<i64> = IndexSet::new();
    let mut current_gen: IndexSet<i64> = IndexSet::new();

    for &needle in &surviving {
        if let Some(parents) = edges_by_node.get(&needle) {
            for &p in parents {
                current_gen.insert(p);
            }
        }
    }

    while !current_gen.is_empty() {
        for &v in &current_gen {
            visited.insert(v);
            surviving.shift_remove(&v);
        }
        let mut next_gen: IndexSet<i64> = IndexSet::new();
        for &v in &current_gen {
            if let Some(parents) = edges_by_node.get(&v) {
                for &p in parents {
                    if !visited.contains(&p) {
                        next_gen.insert(p);
                    }
                }
            }
        }
        current_gen = next_gen;
    }

    surviving.into_iter().collect()
}

/// Eliminates transitively implied dependency edges across all `mgp.device_graph.dependencies`
/// operations in `module`, pruning both `"dep_node_indices"` and any corresponding dependency
/// token operands. Returns the total number of redundant edges eliminated.
pub fn simplify_device_graph_dependencies(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut eliminated_total = 0usize;

    for blk_id in block_ids {
        let deps_ops: Vec<OpId> = module.blocks[blk_id]
            .ops
            .iter()
            .copied()
            .filter(|&id| module.ops[id].name.full_name() == "mgp.device_graph.dependencies")
            .collect();
        if deps_ops.is_empty() {
            continue;
        }

        let mut edges_by_node: IndexMap<i64, Vec<i64>> = IndexMap::new();
        let mut node_token_by_idx: IndexMap<i64, ValueId> = IndexMap::new();

        for (idx, &op_id) in deps_ops.iter().enumerate() {
            let deps = extract_i64_vec_attr(&module.ops[op_id].attributes, "dep_node_indices")
                .unwrap_or_default();
            edges_by_node.insert(idx as i64, deps);
            if let Some(&last_res) = module.ops[op_id].results.last() {
                node_token_by_idx.insert(idx as i64, last_res);
            }
        }

        for (idx, &op_id) in deps_ops.iter().enumerate() {
            let orig_deps = edges_by_node
                .get(&(idx as i64))
                .cloned()
                .unwrap_or_default();
            if orig_deps.len() <= 1 {
                continue;
            }
            let reduced = eliminate_implied_dependency_indices(&orig_deps, &edges_by_node);
            if reduced.len() < orig_deps.len() {
                let reduced_set: IndexSet<i64> = reduced.iter().copied().collect();
                let mut pruned_tokens: IndexSet<ValueId> = IndexSet::new();
                for &d in &orig_deps {
                    if !reduced_set.contains(&d)
                        && let Some(&tok) = node_token_by_idx.get(&d)
                    {
                        pruned_tokens.insert(tok);
                    }
                }
                if !pruned_tokens.is_empty() {
                    module.ops[op_id]
                        .operands
                        .retain(|v| !pruned_tokens.contains(v));
                }
                eliminated_total += orig_deps.len() - reduced.len();
                module.ops[op_id].attributes.insert(
                    "dep_node_indices".to_string(),
                    Attribute::DenseIntArray(reduced),
                );
            }
        }
    }

    eliminated_total
}

// ============================================================================
// Async Routine Lifetime Extension Absorber
// ============================================================================

/// Absorbs `mgp.extend_lifetime` operations whose `gate` chain (`operands[0]`) is produced by a
/// `mogg.routine.async` operation into that routine's operand list (appending `"none"` entries to
/// the routine's `"in_out_signature"` attribute), replaces uses of the `mgp.extend_lifetime`
/// result with `gate`, and erases the `mgp.extend_lifetime` op. Returns the number of absorbed ops.
pub fn simplify_extend_lifetimes(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut absorbed = 0usize;

    for blk_id in block_ids {
        let orig_ops = module.blocks[blk_id].ops.clone();
        let mut erased: IndexSet<OpId> = IndexSet::new();

        for &op_id in &orig_ops {
            if module.ops[op_id].name.full_name() != "mgp.extend_lifetime"
                || module.ops[op_id].operands.is_empty()
                || module.ops[op_id].results.is_empty()
            {
                continue;
            }
            let gate = module.ops[op_id].operands[0];
            let inputs: Vec<ValueId> = module.ops[op_id].operands[1..].to_vec();
            let ext_res = module.ops[op_id].results[0];

            let ValueDef::OpResult { op: kernel_op, .. } = module.values[gate].def else {
                continue;
            };
            if module.ops[kernel_op].name.full_name() != "mogg.routine.async" {
                continue;
            }

            let mut sig: Vec<Attribute> =
                match module.ops[kernel_op].attributes.get("in_out_signature") {
                    Some(Attribute::Array(items)) => items.clone(),
                    _ => Vec::new(),
                };
            for &inp in &inputs {
                module.ops[kernel_op].operands.push(inp);
                sig.push(Attribute::String("none".to_string()));
            }
            module.ops[kernel_op]
                .attributes
                .insert("in_out_signature".to_string(), Attribute::Array(sig));

            for &cand_op in &module.blocks[blk_id].ops {
                if cand_op == op_id {
                    continue;
                }
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == ext_res {
                        *opd = gate;
                    }
                }
            }
            erased.insert(op_id);
            absorbed += 1;
        }

        if !erased.is_empty() {
            module.blocks[blk_id].ops.retain(|id| !erased.contains(id));
        }
    }

    absorbed
}

// ============================================================================
// Unique-Mutator Buffer Allocation Deferrer
// ============================================================================

/// Returns `true` if `op` mutates buffer operand `buf_val`.
pub(super) fn op_mutates_buffer(op: &Operation, buf_val: ValueId) -> bool {
    if !op.operands.contains(&buf_val) {
        return false;
    }
    let full = op.name.full_name();
    matches!(
        full.as_str(),
        "mo.mutable.store" | "mgp.buffer.write" | "mgp.kernel"
    ) || matches!(
        op.attributes.get("mutates_buffer"),
        Some(Attribute::Bool(true))
    )
}

/// Defers `mgp.buffer.alloc` / `mo.buffer.alloc` operations until immediately before their unique
/// buffer mutator in the block and joins the mutator's input chains into the allocation's input
/// chain (`alsoWaitForChains`) so memory is not reserved while upstream chain work is still
/// running. Returns the number of deferred buffer allocations.
pub fn defer_buffer_allocations_to_mutators(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut deferred_count = 0usize;

    for blk_id in block_ids {
        let orig_ops = module.blocks[blk_id].ops.clone();
        let mut moved_allocs: IndexSet<OpId> = IndexSet::new();

        for &alloc_op in &orig_ops {
            if moved_allocs.contains(&alloc_op) {
                continue;
            }
            let full = module.ops[alloc_op].name.full_name();
            if full != "mgp.buffer.alloc" && full != "mo.buffer.alloc" {
                continue;
            }
            if module.ops[alloc_op].operands.is_empty() || module.ops[alloc_op].results.is_empty() {
                continue;
            }
            let alloc_in_chain = module.ops[alloc_op].operands[0];
            let out_buf = module.ops[alloc_op].results[0];

            // Locate unique mutator and verify no user of `out_buf` precedes that mutator.
            let cur_ops = &module.blocks[blk_id].ops;
            let mut mutators: Vec<(usize, OpId)> = Vec::new();
            let mut first_use_pos: Option<usize> = None;

            for (pos, &cand_op) in cur_ops.iter().enumerate() {
                if cand_op == alloc_op {
                    continue;
                }
                let cand = &module.ops[cand_op];
                if cand.operands.contains(&out_buf) {
                    if first_use_pos.is_none() {
                        first_use_pos = Some(pos);
                    }
                    if op_mutates_buffer(cand, out_buf) {
                        mutators.push((pos, cand_op));
                    }
                }
            }

            if mutators.len() != 1 {
                continue;
            }
            let (mut_pos, mutator_op) = mutators[0];
            if first_use_pos != Some(mut_pos) {
                continue;
            }
            let Some(alloc_pos) = cur_ops.iter().position(|&id| id == alloc_op) else {
                continue;
            };
            if alloc_pos >= mut_pos {
                continue;
            }

            // Collect chain-typed operands (`tensor<1xi64>`) on `mutator_op` that differ from
            // `alloc_in_chain` and any chain result of `alloc_op`.
            let chain_ty = module.values[alloc_in_chain].ty.clone();
            let alloc_results: IndexSet<ValueId> =
                module.ops[alloc_op].results.iter().copied().collect();
            let mut extra_chains: Vec<ValueId> = Vec::new();
            for &opd in &module.ops[mutator_op].operands {
                if opd != out_buf
                    && opd != alloc_in_chain
                    && !alloc_results.contains(&opd)
                    && module.values[opd].ty == chain_ty
                    && !extra_chains.contains(&opd)
                {
                    extra_chains.push(opd);
                }
            }

            // Move `alloc_op` right before `mutator_op`.
            module.blocks[blk_id].ops.remove(alloc_pos);
            let Some(new_mut_pos) = module.blocks[blk_id]
                .ops
                .iter()
                .position(|&id| id == mutator_op)
            else {
                continue;
            };

            if !extra_chains.is_empty() {
                let mut join_opds = SmallVec::new();
                join_opds.push(alloc_in_chain);
                for ch in extra_chains {
                    join_opds.push(ch);
                }
                let loc = module.ops[alloc_op].location.clone();
                let join_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mo", "chain.create"),
                    operands: join_opds,
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc,
                });
                let joined_ch = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: chain_ty,
                    def: ValueDef::OpResult {
                        op: join_op,
                        index: 0,
                    },
                });
                module.ops[join_op].results.push(joined_ch);
                module.ops[alloc_op].operands[0] = joined_ch;
                module.blocks[blk_id].ops.insert(new_mut_pos, join_op);
                module.blocks[blk_id].ops.insert(new_mut_pos + 1, alloc_op);
            } else {
                module.blocks[blk_id].ops.insert(new_mut_pos, alloc_op);
            }

            moved_allocs.insert(alloc_op);
            deferred_count += 1;
        }
    }

    deferred_count
}

// ============================================================================
// Execute-Invariant Code Motion Fixed-Point Partitioner
// ============================================================================

/// Result of the fixed-point execute-invariant code motion analysis over a function block.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExecInvariantPartition {
    /// Operations that must remain in the runtime execute region.
    pub unliftable_ops: Vec<OpId>,
    /// Operations that can be hoisted into the compile-time/init region.
    pub liftable_ops: Vec<OpId>,
    /// Liftable operations that must also be cloned into the execute region because their results
    /// cannot be conveyed through the context cache and are consumed by unliftable/cloned ops.
    pub clone_ops: Vec<OpId>,
}

/// Returns `true` if `op` is considered trivial to recompute (`is_trivial = true` or
/// `index.constant` / `arith.constant`).
pub(super) fn is_trivial_exec_op(op: &Operation) -> bool {
    let full = op.name.full_name();
    matches!(full.as_str(), "index.constant" | "arith.constant")
        || matches!(op.attributes.get("is_trivial"), Some(Attribute::Bool(true)))
}

/// Returns `true` if `op` is recomputable (`recomputable = true` or trivial constant).
pub(super) fn is_recomputable_exec_op(op: &Operation) -> bool {
    is_trivial_exec_op(op)
        || matches!(
            op.attributes.get("recomputable"),
            Some(Attribute::Bool(true))
        )
}

/// Returns `true` if a value produced by `op` can be conveyed across init/execute via context
/// cache (`can_be_conveyed = true`, or defaults to `true` unless `can_be_conveyed = false`).
pub(super) fn can_op_results_be_conveyed(op: &Operation) -> bool {
    !matches!(
        op.attributes.get("can_be_conveyed"),
        Some(Attribute::Bool(false))
    )
}

/// Performs the fixed-point execute-invariant code motion analysis over the entry block of the
/// first function in `module`:
/// 1. **Forwards (`DependsOnArgs`)**: Marks ops that depend on non-chain block arguments, have
///    `unliftable = true` or `has_alias = true`, contain nested regions, are terminators
///    (`func.return`), or consume results of unliftable ops.
/// 2. **Backwards (`SuppressTrivial`)**: Marks trivial ops as unliftable if all of their users
///    are already unliftable.
/// 3. **Backwards (`SuppressMutated`)**: Marks an op as unliftable if any of its results is
///    mutated by an unliftable op (`op_mutates_buffer`).
/// 4. Iterates (1)–(3) to a fixed point, then runs **Backwards (`NeedToClone`)** to identify
///    liftable recomputable ops whose results cannot be conveyed via context cache and are used
///    by unliftable or cloned ops.
pub fn analyze_exec_invariant_code_motion(module: &Module) -> ExecInvariantPartition {
    let funcs = module.functions();
    if funcs.is_empty() {
        return ExecInvariantPartition::default();
    }
    let func_id = funcs[0];
    if module.ops[func_id].regions.is_empty() {
        return ExecInvariantPartition::default();
    }
    let reg_id = module.ops[func_id].regions[0];
    if module.regions[reg_id].blocks.is_empty() {
        return ExecInvariantPartition::default();
    }
    let blk_id = module.regions[reg_id].blocks[0];
    let ops = &module.blocks[blk_id].ops;

    // Collect runtime block arguments that taint consumers in the execute region.
    let runtime_args: IndexSet<ValueId> = module.blocks[blk_id].arguments.iter().copied().collect();

    let mut unliftable: IndexSet<OpId> = IndexSet::new();

    loop {
        let orig_count = unliftable.len();

        // 1. Forwards: DependsOnArgs & dataflow propagation.
        for &op_id in ops {
            if unliftable.contains(&op_id) {
                continue;
            }
            let op = &module.ops[op_id];
            let full = op.name.full_name();
            let forced_unliftable = full == "func.return"
                || !op.regions.is_empty()
                || matches!(op.attributes.get("unliftable"), Some(Attribute::Bool(true)))
                || (full == "mgp.buffer.constant.external"
                    && matches!(op.attributes.get("has_alias"), Some(Attribute::Bool(true))));
            let uses_runtime_arg = op.operands.iter().any(|v| runtime_args.contains(v));
            let uses_unliftable = op.operands.iter().any(|&v| {
                matches!(
                    module.values[v].def,
                    ValueDef::OpResult { op: def_op, .. } if unliftable.contains(&def_op)
                )
            });

            if forced_unliftable || uses_runtime_arg || uses_unliftable {
                unliftable.insert(op_id);
            }
        }

        // 2. Backwards: SuppressTrivial.
        for &op_id in ops.iter().rev() {
            if unliftable.contains(&op_id) {
                continue;
            }
            let op = &module.ops[op_id];
            if !is_trivial_exec_op(op) || op.results.is_empty() {
                continue;
            }
            let mut has_user = false;
            let mut all_users_unliftable = true;
            for &cand_id in ops {
                if cand_id == op_id {
                    continue;
                }
                let cand = &module.ops[cand_id];
                if op.results.iter().any(|r| cand.operands.contains(r)) {
                    has_user = true;
                    if !unliftable.contains(&cand_id) {
                        all_users_unliftable = false;
                        break;
                    }
                }
            }
            if has_user && all_users_unliftable {
                unliftable.insert(op_id);
            }
        }

        // 3. Backwards: SuppressMutated.
        for &op_id in ops.iter().rev() {
            if unliftable.contains(&op_id) {
                continue;
            }
            let op = &module.ops[op_id];
            let mutated_by_unliftable = op.results.iter().any(|&res_val| {
                ops.iter().any(|&user_id| {
                    unliftable.contains(&user_id)
                        && op_mutates_buffer(&module.ops[user_id], res_val)
                })
            });
            if mutated_by_unliftable {
                unliftable.insert(op_id);
            }
        }

        if unliftable.len() == orig_count {
            break;
        }
    }

    // 4. Backwards: NeedToClone for liftable recomputable ops with non-conveyable results.
    let mut clone_set: IndexSet<OpId> = IndexSet::new();
    for &op_id in ops.iter().rev() {
        if unliftable.contains(&op_id) {
            continue;
        }
        let op = &module.ops[op_id];
        if !is_recomputable_exec_op(op) || can_op_results_be_conveyed(op) {
            continue;
        }
        let used_by_exec = op.results.iter().any(|&res_val| {
            ops.iter().any(|&user_id| {
                (unliftable.contains(&user_id) || clone_set.contains(&user_id))
                    && module.ops[user_id].operands.contains(&res_val)
            })
        });
        if used_by_exec {
            clone_set.insert(op_id);
        }
    }

    let mut partition = ExecInvariantPartition::default();
    for &op_id in ops {
        if unliftable.contains(&op_id) {
            partition.unliftable_ops.push(op_id);
        } else {
            partition.liftable_ops.push(op_id);
            if clone_set.contains(&op_id) {
                partition.clone_ops.push(op_id);
            }
        }
    }
    partition
}
