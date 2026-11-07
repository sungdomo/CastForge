//! Multi-group presence bitmaps, buffer concat lowering, and parametric sorting.

use super::{expected_contiguous_shape_from_strides, extract_u64_attr};
use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, Type, Value, ValueDef, ValueId,
};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Multi-Group Fold/Unfold Bitmap & Layer-Group Payload Sharder
// ============================================================================

/// Sentinel window size indicating a full-attention object group (`<= 0`).
pub const FULL_ATTENTION_GROUP_WINDOW: i64 = -1;

/// Computes the half-open chunk range `(lo, hi)` required by a single object group to serve a
/// prefix of `prefix_len` chunks. Full attention (`window <= 0`) requires `(0, prefix_len)`; a
/// sliding window of `w` chunks (`window > 0`) requires `(max(0, prefix_len - w), prefix_len)`.
#[must_use]
pub fn unfold_group_chunk_range(prefix_len: usize, window: i64) -> (usize, usize) {
    if prefix_len == 0 {
        return (0, 0);
    }
    if window <= 0 {
        (0, prefix_len)
    } else {
        let w = window as usize;
        (prefix_len.saturating_sub(w), prefix_len)
    }
}

/// Folds a flat ranked presence bitmap `found` (of length `num_chunks * group_windows.len() *
/// num_ranks`, indexed by `j * (num_groups * num_ranks) + g * num_ranks + r`) into a servable
/// prefix-length bitmap of length `num_chunks + 1`, where index `l` (`0..=num_chunks`) is `true`
/// iff every object group can serve a prefix of length `l` with all `num_ranks` shards present.
pub fn fold_multi_group_presence(
    found: &[bool],
    num_chunks: usize,
    num_ranks: usize,
    group_windows: &[i64],
) -> Result<Vec<bool>, String> {
    if num_ranks == 0 {
        return Err("num_ranks must be >= 1".to_string());
    }
    if group_windows.is_empty() {
        return Err("group_windows must be non-empty".to_string());
    }
    let num_groups = group_windows.len();
    let expected_len = num_chunks * num_groups * num_ranks;
    if found.len() != expected_len {
        return Err(format!(
            "found bitmap length mismatch: expected {expected_len}, got {}",
            found.len()
        ));
    }

    let chunk_stride = num_groups * num_ranks;
    let mut servable = vec![true; num_chunks + 1];
    for (group_idx, &window) in group_windows.iter().enumerate() {
        let gbase = group_idx * num_ranks;
        let effective_window = if window <= 0 {
            num_chunks
        } else {
            window as usize
        };
        let mut run = 0usize;
        for (prefix_len, slot) in servable.iter_mut().enumerate().skip(1) {
            let cbase = (prefix_len - 1) * chunk_stride + gbase;
            let present = (0..num_ranks).all(|r| found[cbase + r]);
            run = if present { run + 1 } else { 0 };
            if *slot && run < effective_window.min(prefix_len) {
                *slot = false;
            }
        }
    }
    Ok(servable)
}

/// Returns the model-wide servable prefix hit length in chunks (the highest index `l` where
/// `servable[l]` is `true`, or `0` if none).
#[must_use]
pub fn highest_servable_prefix_len(servable: &[bool]) -> usize {
    servable.iter().rposition(|&b| b).unwrap_or(0)
}

/// Expands `hit_length` (clamped to `num_chunks`) into the flat ranked retain mask of length
/// `num_chunks * group_windows.len() * num_ranks`.
pub fn unfold_multi_group_retain_mask(
    hit_length: usize,
    num_chunks: usize,
    num_ranks: usize,
    group_windows: &[i64],
) -> Result<Vec<bool>, String> {
    if num_ranks == 0 {
        return Err("num_ranks must be >= 1".to_string());
    }
    if group_windows.is_empty() {
        return Err("group_windows must be non-empty".to_string());
    }
    let clamped = hit_length.min(num_chunks);
    let num_groups = group_windows.len();
    let mut mask = vec![false; num_chunks * num_groups * num_ranks];
    for (g, &win) in group_windows.iter().enumerate() {
        let (lo, hi) = unfold_group_chunk_range(clamped, win);
        for j in lo..hi {
            let base = j * (num_groups * num_ranks) + g * num_ranks;
            for r in 0..num_ranks {
                mask[base + r] = true;
            }
        }
    }
    Ok(mask)
}

/// Folds per-row presence bitmaps `rows` (each of length `num_chunks`) paired with `windows`
/// into a servable prefix-length bitmap of length `num_chunks + 1` using an `O(rows * chunks)`
/// consecutive-run scan.
pub fn fold_grouped_presence(rows: &[Vec<bool>], windows: &[i64]) -> Result<Vec<bool>, String> {
    if windows.is_empty() {
        return Err("windows must be non-empty".to_string());
    }
    if rows.len() != windows.len() {
        return Err(format!(
            "rows and windows length mismatch: {} vs {}",
            rows.len(),
            windows.len()
        ));
    }
    let num_chunks = rows[0].len();
    if rows.iter().any(|r| r.len() != num_chunks) {
        return Err("all rows in fold_grouped_presence must have equal length".to_string());
    }

    let mut servable = vec![true; num_chunks + 1];
    for (row, &window) in rows.iter().zip(windows.iter()) {
        let effective_window = if window <= 0 {
            num_chunks
        } else {
            window as usize
        };
        let mut run = 0usize;
        for (prefix_len, slot) in servable.iter_mut().enumerate().skip(1) {
            run = if row[prefix_len - 1] { run + 1 } else { 0 };
            if *slot && run < effective_window.min(prefix_len) {
                *slot = false;
            }
        }
    }
    Ok(servable)
}

/// Expands `hit_length` (clamped to `num_chunks`) into per-row retain bitmaps of length
/// `num_chunks`, parallel to `windows`.
pub fn unfold_grouped_retain_masks(
    hit_length: usize,
    num_chunks: usize,
    windows: &[i64],
) -> Result<Vec<Vec<bool>>, String> {
    if windows.is_empty() {
        return Err("windows must be non-empty".to_string());
    }
    let clamped = hit_length.min(num_chunks);
    let mut out = Vec::with_capacity(windows.len());
    for &win in windows {
        let mut row = vec![false; num_chunks];
        let (lo, hi) = unfold_group_chunk_range(clamped, win);
        for cell in &mut row[lo..hi] {
            *cell = true;
        }
        out.push(row);
    }
    Ok(out)
}

/// Composes `fold_grouped_presence` -> `highest_servable_prefix_len` ->
/// `unfold_grouped_retain_masks`, returning `(hit_length, retain_rows)`.
pub fn fold_unfold_grouped_presence(
    rows: &[Vec<bool>],
    windows: &[i64],
) -> Result<(usize, Vec<Vec<bool>>), String> {
    let servable = fold_grouped_presence(rows, windows)?;
    let num_chunks = rows.first().map_or(0, Vec::len);
    let hit_len = highest_servable_prefix_len(&servable);
    let retain = unfold_grouped_retain_masks(hit_len, num_chunks, windows)?;
    Ok((hit_len, retain))
}

/// Restricts every row to the columns present across all rows (full-blend intersection policy
/// without prefix or window folding).
pub fn all_grouped_retain_masks(rows: &[Vec<bool>]) -> Result<Vec<Vec<bool>>, String> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let num_chunks = rows[0].len();
    if rows.iter().any(|r| r.len() != num_chunks) {
        return Err("all rows in all_grouped_retain_masks must have equal length".to_string());
    }
    let mut common = rows[0].clone();
    for row in &rows[1..] {
        for (c, &b) in common.iter_mut().zip(row.iter()) {
            *c = *c && b;
        }
    }
    Ok(vec![common; rows.len()])
}

/// Merges multiple presence bitmaps with a bitwise OR into a `num_keys`-sized bitmap.
#[must_use]
pub fn merge_presence_bitmaps(bitmaps: &[Vec<bool>], num_keys: usize) -> Vec<bool> {
    let mut merged = vec![false; num_keys];
    for bm in bitmaps {
        for (dst, &src) in merged.iter_mut().zip(bm.iter()) {
            *dst = *dst || src;
        }
    }
    merged
}

// ============================================================================
// In-Place Buffer Concatenation Lowering with Alignment Check
// ============================================================================

/// Returns `true` if `val` is provably a multiple of `alignment` across `index.constant` /
/// `arith.constant` (`value` attribute), `index.mul` / `arith.muli`, and `index.add` /
/// `arith.addi`.
#[must_use]
pub fn is_provable_multiple_of(module: &Module, val: ValueId, alignment: u64) -> bool {
    if alignment <= 1 {
        return true;
    }
    let ValueDef::OpResult { op: def_op, .. } = module.values[val].def else {
        return false;
    };
    let op = &module.ops[def_op];
    let full = op.name.full_name();
    match full.as_str() {
        "index.constant" | "arith.constant" => op
            .attributes
            .get("value")
            .and_then(extract_u64_attr)
            .is_some_and(|v| v % alignment == 0),
        "index.mul" | "arith.muli" if op.operands.len() >= 2 => {
            is_provable_multiple_of(module, op.operands[0], alignment)
                || is_provable_multiple_of(module, op.operands[1], alignment)
        }
        "index.add" | "arith.addi" if op.operands.len() >= 2 => {
            is_provable_multiple_of(module, op.operands[0], alignment)
                && is_provable_multiple_of(module, op.operands[1], alignment)
        }
        _ => false,
    }
}

/// Lowers `mgp.buffer.concat` operations in-place into `mgp.buffer.slice` views of the output
/// buffer allocation when:
/// 1. Operand 0 is `in_chain`, operand 1 is `out_buffer` defined by `mgp.buffer.alloc`, and
///    operands `2..` are distinct `in_buffers` each defined by `mgp.buffer.alloc` whose first
///    operand is the runtime byte size.
/// 2. Every input buffer size except the last is provably a multiple of `preferred_alignment`.
///
/// Moves the output buffer allocation before the first slice, replaces each input buffer allocation
/// with `mgp.buffer.slice(out_buffer, offset, slice_size)`, forwards `out_chain` to `in_chain`,
/// and erases the `mgp.buffer.concat` op. Returns the number of lowered concatenations.
pub fn lower_buffer_concat_in_place(module: &mut Module, preferred_alignment: u64) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut lowered_count = 0usize;

    for blk_id in block_ids {
        let orig_ops = module.blocks[blk_id].ops.clone();

        for &concat_op in &orig_ops {
            if !module.ops.contains_key(concat_op) {
                continue;
            }
            if module.ops[concat_op].name.full_name() != "mgp.buffer.concat" {
                continue;
            }
            if module.ops[concat_op].operands.len() < 3 || module.ops[concat_op].results.is_empty()
            {
                continue;
            }
            let in_chain = module.ops[concat_op].operands[0];
            let out_buffer = module.ops[concat_op].operands[1];
            let in_buffers: Vec<ValueId> = module.ops[concat_op].operands[2..].to_vec();
            let out_chain = module.ops[concat_op].results[0];

            // Verify `out_buffer` is defined by `mgp.buffer.alloc`.
            let ValueDef::OpResult {
                op: out_alloc_op, ..
            } = module.values[out_buffer].def
            else {
                continue;
            };
            if module.ops[out_alloc_op].name.full_name() != "mgp.buffer.alloc" {
                continue;
            }

            // Verify all `in_buffers` are distinct and defined by `mgp.buffer.alloc` with a size
            // operand.
            let mut distinct_in: IndexSet<ValueId> = IndexSet::new();
            let mut in_alloc_ops: Vec<OpId> = Vec::with_capacity(in_buffers.len());
            let mut slice_sizes: Vec<ValueId> = Vec::with_capacity(in_buffers.len());
            let mut valid = true;
            for &in_buf in &in_buffers {
                if !distinct_in.insert(in_buf) {
                    valid = false;
                    break;
                }
                let ValueDef::OpResult { op: alloc_id, .. } = module.values[in_buf].def else {
                    valid = false;
                    break;
                };
                if module.ops[alloc_id].name.full_name() != "mgp.buffer.alloc"
                    || module.ops[alloc_id].operands.is_empty()
                {
                    valid = false;
                    break;
                }
                in_alloc_ops.push(alloc_id);
                slice_sizes.push(module.ops[alloc_id].operands[0]);
            }
            if !valid {
                continue;
            }

            // Every slice size except the last must be a provable multiple of
            // `preferred_alignment`.
            let aligned = slice_sizes[..slice_sizes.len() - 1]
                .iter()
                .all(|&sz| is_provable_multiple_of(module, sz, preferred_alignment));
            if !aligned {
                continue;
            }

            // Move `out_alloc_op` before the first `in_alloc_ops` in the block if needed.
            if let (Some(out_pos), Some(first_in_pos)) = (
                module.blocks[blk_id]
                    .ops
                    .iter()
                    .position(|&id| id == out_alloc_op),
                module.blocks[blk_id]
                    .ops
                    .iter()
                    .position(|&id| id == in_alloc_ops[0]),
            ) && out_pos > first_in_pos
            {
                module.blocks[blk_id].ops.remove(out_pos);
                if let Some(new_first_pos) = module.blocks[blk_id]
                    .ops
                    .iter()
                    .position(|&id| id == in_alloc_ops[0])
                {
                    module.blocks[blk_id]
                        .ops
                        .insert(new_first_pos, out_alloc_op);
                }
            }

            // Replace each input allocation with `index.constant` + `mgp.buffer.slice`.
            let loc = module.ops[concat_op].location.clone();
            let size_ty = module.values[slice_sizes[0]].ty.clone();

            for (idx, (&in_buf, (&in_alloc_id, &slice_sz))) in in_buffers
                .iter()
                .zip(in_alloc_ops.iter().zip(slice_sizes.iter()))
                .enumerate()
            {
                let mut cst_attrs = IndexMap::new();
                cst_attrs.insert("value".to_string(), Attribute::Int(idx as i64));
                let cst_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("index", "constant"),
                    operands: SmallVec::new(),
                    results: SmallVec::new(),
                    attributes: cst_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let cst_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: size_ty.clone(),
                    def: ValueDef::OpResult {
                        op: cst_op,
                        index: 0,
                    },
                });
                module.ops[cst_op].results.push(cst_val);

                let mut slice_opds = SmallVec::new();
                slice_opds.push(out_buffer);
                slice_opds.push(cst_val);
                slice_opds.push(slice_sz);
                let slice_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mgp", "buffer.slice"),
                    operands: slice_opds,
                    results: SmallVec::from_slice(&[in_buf]),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                module.values[in_buf].def = ValueDef::OpResult {
                    op: slice_op,
                    index: 0,
                };

                if let Some(pos) = module.blocks[blk_id]
                    .ops
                    .iter()
                    .position(|&id| id == in_alloc_id)
                {
                    module.blocks[blk_id].ops[pos] = cst_op;
                    module.blocks[blk_id].ops.insert(pos + 1, slice_op);
                }
                module.ops.remove(in_alloc_id);
            }

            // Replace uses of `out_chain` with `in_chain` across the block and erase `concat_op`.
            for &op_id in &module.blocks[blk_id].ops {
                if op_id == concat_op {
                    continue;
                }
                for opd in &mut module.ops[op_id].operands {
                    if *opd == out_chain {
                        *opd = in_chain;
                    }
                }
            }
            module.blocks[blk_id].ops.retain(|&id| id != concat_op);
            module.ops.remove(concat_op);
            lowered_count += 1;
        }
    }

    lowered_count
}

// ============================================================================
// Hanging Chain Overhang Terminator Joiner
// ============================================================================

/// Returns `true` if `ty` represents a 1-D `i64` execution chain (`tensor<1xi64>`).
pub(super) fn is_chain_tensor_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Tensor(t)
            if t.dtype == DType::I64 && t.shape.as_slice() == [Dim::Static(1)]
    )
}

/// Scans every block in `module` for operations that produce unused chain results
/// (`tensor<1xi64>`), collects at most one hanging chain result per operation, and joins them into
/// the block terminator's (`func.return`) chain operand via `mo.chain.create` so asynchronous
/// chain work is awaited before block exit. Returns the number of hanging chains joined.
pub fn fixup_hanging_chain_overhangs(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut joined_chains = 0usize;

    for blk_id in block_ids {
        let ops = &module.blocks[blk_id].ops;
        let Some(&term_op) = ops.last() else {
            continue;
        };
        if module.ops[term_op].name.full_name() != "func.return" {
            continue;
        }
        let Some(term_chain_idx) = module.ops[term_op]
            .operands
            .iter()
            .position(|&v| is_chain_tensor_type(&module.values[v].ty))
        else {
            continue;
        };

        // Collect all used values in the block.
        let mut used_vals: IndexSet<ValueId> = IndexSet::new();
        for &op_id in ops {
            for &opd in &module.ops[op_id].operands {
                used_vals.insert(opd);
            }
        }

        // Collect at most one unused chain result per non-terminator op.
        let mut hanging_chains: Vec<ValueId> = Vec::new();
        for &op_id in &ops[..ops.len() - 1] {
            for &res in &module.ops[op_id].results {
                if is_chain_tensor_type(&module.values[res].ty) && !used_vals.contains(&res) {
                    hanging_chains.push(res);
                    break;
                }
            }
        }

        if hanging_chains.is_empty() {
            continue;
        }

        let orig_term_chain = module.ops[term_op].operands[term_chain_idx];
        let chain_ty = module.values[orig_term_chain].ty.clone();
        let loc = module.ops[term_op].location.clone();
        let mut join_opds = SmallVec::new();
        join_opds.push(orig_term_chain);
        for &hc in &hanging_chains {
            join_opds.push(hc);
        }

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
        let joined_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: chain_ty,
            def: ValueDef::OpResult {
                op: join_op,
                index: 0,
            },
        });
        module.ops[join_op].results.push(joined_val);
        module.ops[term_op].operands[term_chain_idx] = joined_val;

        let term_pos = module.blocks[blk_id].ops.len() - 1;
        module.blocks[blk_id].ops.insert(term_pos, join_op);
        joined_chains += hanging_chains.len();
    }

    joined_chains
}

// ============================================================================
// Structured Kernel Body Outliner into Opaque Closure Regions
// ============================================================================

/// Returns `true` if `full_name` is a structured `map.iter.*` loop operation eligible for
/// outlining into `map.iter.opaque`.
pub(super) fn is_structured_map_iter_op(full_name: &str) -> bool {
    matches!(
        full_name,
        "map.iter.elementwise" | "map.iter.reduction" | "map.iter.foreach"
    )
}

/// Wraps structured `map.iter.elementwise`, `map.iter.reduction`, and `map.iter.foreach` loops in
/// `map.iter.opaque` operations containing a `map.closure` body so sequential setup can reside
/// outside the structured loop body:
/// - Preserves `device_ref` (defaulting to `"cpu:0"` if absent), `fusion_info`, `has_read_effect`,
///   `has_write_effect`, and `dispatch` attributes on `map.iter.opaque`.
/// - Creates an outer block on `map.iter.opaque` whose block arguments capture all non-chain input
///   operands and non-chain result types.
/// - Moves the structured loop's body block into the inner `map.closure` region and rotates its
///   trailing loop-index block argument (`[inputs..., outputs..., index]`) to position `0`
///   (`[index, captures...]`).
/// - Replaces the structured loop with `map.iter.opaque` and returns the count of outlined loops.
pub fn outline_structured_kernel_bodies(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut outlined_count = 0usize;

    for blk_id in block_ids {
        let orig_ops = module.blocks[blk_id].ops.clone();

        for &loop_op in &orig_ops {
            if !module.ops.contains_key(loop_op) {
                continue;
            }
            let full = module.ops[loop_op].name.full_name();
            if !is_structured_map_iter_op(&full) {
                continue;
            }
            let loc = module.ops[loop_op].location.clone();
            let operands = module.ops[loop_op].operands.clone();
            let results = module.ops[loop_op].results.clone();

            let (loop_reg, inner_blk) = if !module.ops[loop_op].regions.is_empty() {
                let reg = module.ops[loop_op].regions[0];
                if module.regions[reg].blocks.is_empty() {
                    continue;
                }
                let blk = module.regions[reg].blocks[0];
                // Rotate trailing `index` block argument (`[inputs..., outputs..., index]`) to
                // position 0 (`[index, captures...]`).
                if module.blocks[blk].arguments.len() >= 2
                    && let Some(idx_arg) = module.blocks[blk].arguments.pop()
                {
                    module.blocks[blk].arguments.insert(0, idx_arg);
                }
                (reg, blk)
            } else {
                // Synthesize the inner closure region with `[index, captures...]` block arguments.
                let blk = module.blocks.insert_with_key(|id| crate::ir::Block {
                    id,
                    label: None,
                    arguments: Vec::new(),
                    ops: Vec::new(),
                });
                let idx_ty = Type::Tensor(crate::ir::mlir::TensorType {
                    shape: vec![Dim::Static(1)],
                    dtype: DType::I64,
                    encoding: None,
                });
                let idx_arg = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: idx_ty,
                    def: ValueDef::BlockArgument {
                        block: blk,
                        index: 0,
                    },
                });
                module.blocks[blk].arguments.push(idx_arg);
                for &opd in &operands {
                    if !is_chain_tensor_type(&module.values[opd].ty) {
                        let ty = module.values[opd].ty.clone();
                        let arg_idx = module.blocks[blk].arguments.len() as u32;
                        let v = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty,
                            def: ValueDef::BlockArgument {
                                block: blk,
                                index: arg_idx,
                            },
                        });
                        module.blocks[blk].arguments.push(v);
                    }
                }
                for &res in &results {
                    if !is_chain_tensor_type(&module.values[res].ty) {
                        let ty = module.values[res].ty.clone();
                        let arg_idx = module.blocks[blk].arguments.len() as u32;
                        let v = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty,
                            def: ValueDef::BlockArgument {
                                block: blk,
                                index: arg_idx,
                            },
                        });
                        module.blocks[blk].arguments.push(v);
                    }
                }
                let reg = module.regions.insert_with_key(|id| crate::ir::Region {
                    id,
                    blocks: vec![blk],
                });
                (reg, blk)
            };

            // Copy metadata attributes (`device_ref`, `fusion_info`, `has_read_effect`,
            // `has_write_effect`, `dispatch`), defaulting `device_ref` to `"cpu:0"`.
            let mut opaque_attrs: IndexMap<String, Attribute> = IndexMap::new();
            let dev_attr = module.ops[loop_op]
                .attributes
                .get("device_ref")
                .cloned()
                .unwrap_or_else(|| Attribute::String("cpu:0".to_string()));
            opaque_attrs.insert("device_ref".to_string(), dev_attr);
            for key in [
                "fusion_info",
                "has_read_effect",
                "has_write_effect",
                "dispatch",
            ] {
                if let Some(val) = module.ops[loop_op].attributes.get(key).cloned() {
                    opaque_attrs.insert(key.to_string(), val);
                }
            }

            // Create the outer block for `map.iter.opaque` with block arguments for every
            // non-chain input operand and every non-chain result type.
            let mut capture_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
            for &opd in &operands {
                if !is_chain_tensor_type(&module.values[opd].ty) {
                    let ty = module.values[opd].ty.clone();
                    let idx = capture_vals.len() as u32;
                    let arg_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: inner_blk,
                            index: idx,
                        },
                    });
                    capture_vals.push(arg_val);
                }
            }
            for &res in &results {
                if !is_chain_tensor_type(&module.values[res].ty) {
                    let ty = module.values[res].ty.clone();
                    let idx = capture_vals.len() as u32;
                    let arg_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: inner_blk,
                            index: idx,
                        },
                    });
                    capture_vals.push(arg_val);
                }
            }

            // Create `map.closure` taking `capture_vals` and owning `loop_reg`.
            let closure_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("map", "closure"),
                operands: capture_vals.clone(),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: vec![loop_reg],
                custom_syntax: false,
                location: loc.clone(),
            });

            let opaque_blk = module.blocks.insert_with_key(|id| crate::ir::Block {
                id,
                label: None,
                arguments: capture_vals.to_vec(),
                ops: vec![closure_op],
            });
            for (idx, &arg_val) in module.blocks[opaque_blk].arguments.iter().enumerate() {
                module.values[arg_val].def = ValueDef::BlockArgument {
                    block: opaque_blk,
                    index: idx as u32,
                };
            }
            let opaque_reg = module.regions.insert_with_key(|id| crate::ir::Region {
                id,
                blocks: vec![opaque_blk],
            });

            let opaque_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("map", "iter.opaque"),
                operands,
                results: results.clone(),
                attributes: opaque_attrs,
                regions: vec![opaque_reg],
                custom_syntax: false,
                location: loc,
            });
            for (idx, &res) in results.iter().enumerate() {
                module.values[res].def = ValueDef::OpResult {
                    op: opaque_op,
                    index: idx as u32,
                };
            }

            if let Some(pos) = module.blocks[blk_id]
                .ops
                .iter()
                .position(|&id| id == loop_op)
            {
                module.blocks[blk_id].ops[pos] = opaque_op;
            }
            module.ops.remove(loop_op);
            outlined_count += 1;
        }
    }

    outlined_count
}

// ============================================================================
// Parametric Dependency Topological Sorter & Rooted Components
// ============================================================================

/// Tracks combined SSA def-use and symbolic parameter (`decl_param` / `use_params`) dependencies
/// between operations in a block.
#[derive(Debug, Clone, Default)]
pub struct ParametricTopologyGraph {
    /// For each operation, the ordered set of provider operations it depends on.
    pub providers_of: IndexMap<OpId, IndexSet<OpId>>,
    /// For each operation, the ordered set of user operations that depend on it.
    pub users_of: IndexMap<OpId, IndexSet<OpId>>,
}

impl ParametricTopologyGraph {
    /// Builds the combined SSA and parametric dependency graph for `blk_id`.
    #[must_use]
    pub fn from_block(module: &Module, blk_id: BlockId) -> Self {
        let ops = &module.blocks[blk_id].ops;
        let block_set: IndexSet<OpId> = ops.iter().copied().collect();
        let mut graph = Self::default();
        for &op_id in ops {
            graph.providers_of.entry(op_id).or_default();
            graph.users_of.entry(op_id).or_default();
        }

        // Map parameter names declared via `decl_param = "..."` to their declaring OpId.
        let mut param_decls: IndexMap<String, OpId> = IndexMap::new();
        for &op_id in ops {
            if let Some(Attribute::String(name)) = module.ops[op_id].attributes.get("decl_param") {
                param_decls.insert(name.clone(), op_id);
            }
        }

        for &op_id in ops {
            let op = &module.ops[op_id];
            // 1. SSA dependencies within the block.
            for &opd in &op.operands {
                if let ValueDef::OpResult { op: def_op, .. } = module.values[opd].def
                    && def_op != op_id
                    && block_set.contains(&def_op)
                {
                    graph.providers_of.entry(op_id).or_default().insert(def_op);
                    graph.users_of.entry(def_op).or_default().insert(op_id);
                }
            }
            // 2. Parametric dependencies (`use_params = ["..."]`).
            if let Some(Attribute::Array(items)) = op.attributes.get("use_params") {
                for item in items {
                    if let Attribute::String(param_name) = item
                        && let Some(&decl_op) = param_decls.get(param_name)
                        && decl_op != op_id
                    {
                        graph.providers_of.entry(op_id).or_default().insert(decl_op);
                        graph.users_of.entry(decl_op).or_default().insert(op_id);
                    }
                }
            }
        }

        graph
    }

    /// Topologically sorts `ops` in-place using the repeated-scan Kahn blocker-decrement swap
    /// algorithm. Returns `Err(stuck_ops)` in deterministic order if a dependency cycle prevents
    /// progress.
    pub fn sort_op_slice(&self, ops: &mut [OpId]) -> Result<(), Vec<OpId>> {
        let n = ops.len();
        if n <= 1 {
            return Ok(());
        }
        let mut id_of: IndexMap<OpId, usize> = IndexMap::with_capacity(n);
        for (idx, &op_id) in ops.iter().enumerate() {
            id_of.insert(op_id, idx);
        }

        let mut num_blockers = vec![0usize; n];
        let mut waiters_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (idx, &op_id) in ops.iter().enumerate() {
            let mut blockers: IndexSet<usize> = IndexSet::new();
            if let Some(provs) = self.providers_of.get(&op_id) {
                for &prov in provs {
                    if let Some(&prov_idx) = id_of.get(&prov)
                        && prov_idx != idx
                    {
                        blockers.insert(prov_idx);
                    }
                }
            }
            num_blockers[idx] = blockers.len();
            for blocker_idx in blockers {
                waiters_of[blocker_idx].push(idx);
            }
        }

        let mut ids: Vec<usize> = (0..n).collect();
        let mut next_scheduled = 0usize;

        while next_scheduled < n {
            let mut scheduled_any = false;
            let sweep_start = next_scheduled;
            for i in sweep_start..n {
                let orig_id = ids[i];
                if num_blockers[orig_id] != 0 {
                    continue;
                }
                for &waiter in &waiters_of[orig_id] {
                    num_blockers[waiter] -= 1;
                }
                ops.swap(i, next_scheduled);
                ids.swap(i, next_scheduled);
                scheduled_any = true;
                next_scheduled += 1;
            }
            if !scheduled_any {
                return Err(ops[next_scheduled..].to_vec());
            }
        }

        Ok(())
    }
}

/// Topologically sorts all non-terminator operations in `blk_id` using combined SSA and parametric
/// dependencies while keeping any trailing `func.return` terminator at the end of the block.
/// Returns the number of operations whose position changed, or `Err(stuck_ops)` on a cycle.
pub fn topologically_sort_parametric_block(
    module: &mut Module,
    blk_id: BlockId,
) -> Result<usize, Vec<OpId>> {
    let graph = ParametricTopologyGraph::from_block(module, blk_id);
    let orig_ops = module.blocks[blk_id].ops.clone();
    if orig_ops.len() <= 1 {
        return Ok(0);
    }
    let has_terminator = orig_ops
        .last()
        .is_some_and(|&op_id| module.ops[op_id].name.full_name() == "func.return");
    let body_len = if has_terminator {
        orig_ops.len() - 1
    } else {
        orig_ops.len()
    };

    let mut sorted = orig_ops.clone();
    graph.sort_op_slice(&mut sorted[..body_len])?;

    let moved = orig_ops
        .iter()
        .zip(sorted.iter())
        .filter(|(a, b)| a != b)
        .count();
    module.blocks[blk_id].ops = sorted;
    Ok(moved)
}

/// Extracts `root_op` and its connected component of transitive users (plus non-root providers of
/// those users) within `blk_id`, returned in topological order.
pub fn extract_rooted_parametric_components(
    module: &Module,
    blk_id: BlockId,
    root_op: OpId,
) -> Result<Vec<OpId>, Vec<OpId>> {
    let graph = ParametricTopologyGraph::from_block(module, blk_id);
    let mut fringe = vec![root_op];
    let mut enqueued: IndexSet<OpId> = IndexSet::new();
    enqueued.insert(root_op);
    let mut component = Vec::new();

    while let Some(cur) = fringe.pop() {
        component.push(cur);
        if let Some(users) = graph.users_of.get(&cur) {
            for &u in users {
                if enqueued.insert(u) {
                    fringe.push(u);
                }
            }
        }
        if cur == root_op {
            continue;
        }
        if let Some(provs) = graph.providers_of.get(&cur) {
            for &p in provs {
                if enqueued.insert(p) {
                    fringe.push(p);
                }
            }
        }
    }

    graph.sort_op_slice(&mut component)?;
    Ok(component)
}

// ============================================================================
// Strided KV Contiguous View Recovery & Sub-Byte Serde Layout
// ============================================================================

/// Computes the dimension permutation that sorts dimensions by descending stride magnitude,
/// breaking ties by placing size-1 dimensions innermost (`shape[i] != 1` ordered before `false`).
#[must_use]
pub fn compute_stride_permutation(shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let n = shape.len().min(strides.len());
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| {
        let key_a = (strides[a], shape[a] != 1);
        let key_b = (strides[b], shape[b] != 1);
        key_b.cmp(&key_a)
    });
    perm
}

/// Validates that a stride-sorted tensor view satisfies the dim-0-padding-only layout invariant:
/// - `ndim >= 2`
/// - `strides[ndim - 1] == 1` and `strides[ndim - 2] == shape[ndim - 1]`
/// - Every interior dimension `1 <= i < ndim - 1` has `strides[i] == prod(shape[i + 1..])`
/// - `strides[0] >= prod(shape[1..])`
///
/// Returns `padding_per_block_elems = strides[0] - prod(shape[1..])`.
pub fn validate_dim0_padded_strided_layout(
    shape: &[usize],
    strides: &[usize],
) -> Result<usize, String> {
    if shape.len() != strides.len() {
        return Err("shape and strides rank mismatch".to_string());
    }
    let ndim = shape.len();
    if ndim < 2 {
        return Err("ndim < 2".to_string());
    }
    if strides[ndim - 1] != 1 {
        return Err("strides[-1] != 1 (inner dim not contiguous)".to_string());
    }
    if strides[ndim - 2] != shape[ndim - 1] {
        return Err("strides[-2] != shape[-1] (last-two dims not tightly packed)".to_string());
    }
    let mut inner_tight = 1usize;
    for i in (1..ndim).rev() {
        if i < ndim - 1 && strides[i] != inner_tight {
            return Err(format!(
                "dim {i} stride {} != tight {inner_tight} (interior-dim padding not supported)",
                strides[i]
            ));
        }
        inner_tight = inner_tight.saturating_mul(shape[i]);
    }
    if strides[0] < inner_tight {
        return Err(format!(
            "dim-0 stride {} < prod(shape[1..])={inner_tight} (overlapping blocks)",
            strides[0]
        ));
    }
    Ok(strides[0] - inner_tight)
}

/// Result of zero-copy strided KV contiguous/dim-0-padded view recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredStridedKvView {
    /// Dimension permutation sorting strides in descending order.
    pub perm: Vec<usize>,
    /// Canonical shape after permutation (or expected contiguous shape when `padding_elems == 0`).
    pub canonical_shape: Vec<usize>,
    /// Strides after applying `perm`.
    pub permuted_strides: Vec<usize>,
    /// Per-block padding elements on dim 0 (`0` for strictly contiguous views).
    pub padding_per_block_elems: usize,
}

/// Recovers a zero-copy contiguous or dim-0-padded view from `(shape, strides)` by permuting
/// dimensions into descending stride order.
pub fn recover_strided_kv_contiguous_view(
    shape: &[usize],
    strides: &[usize],
) -> Result<RecoveredStridedKvView, String> {
    if shape.is_empty() || shape.len() != strides.len() {
        return Err("shape and strides must be non-empty and of equal rank".to_string());
    }
    let perm = compute_stride_permutation(shape, strides);
    let perm_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    let perm_strides: Vec<usize> = perm.iter().map(|&i| strides[i]).collect();
    let pad = validate_dim0_padded_strided_layout(&perm_shape, &perm_strides)?;
    if pad == 0 {
        let num_elems: usize = perm_shape.iter().product();
        let expected = expected_contiguous_shape_from_strides(&perm_strides, num_elems)
            .ok_or_else(|| "invalid strides for contiguous shape".to_string())?;
        Ok(RecoveredStridedKvView {
            perm,
            canonical_shape: expected,
            permuted_strides: perm_strides,
            padding_per_block_elems: 0,
        })
    } else {
        Ok(RecoveredStridedKvView {
            perm,
            canonical_shape: perm_shape,
            permuted_strides: perm_strides,
            padding_per_block_elems: pad,
        })
    }
}

/// Sub-byte KV quantization preset specifying per-element bit-widths and norm correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubByteKvSerdePreset {
    /// 8-bit FP8 keys, 4-bit values, no norm correction.
    K8V4,
    /// 4-bit keys, 4-bit values, with norm correction.
    K4V4NormCorr,
    /// 3-bit keys, 4-bit values, with norm correction.
    K3V4NormCorr,
    /// 3-bit keys, 3-bit values, with norm correction.
    K3V3NormCorr,
}

impl SubByteKvSerdePreset {
    /// Returns `(key_quant_bits, value_quant_bits, norm_correction)`.
    #[must_use]
    pub const fn params(self) -> (usize, usize, bool) {
        match self {
            Self::K8V4 => (8, 4, false),
            Self::K4V4NormCorr => (4, 4, true),
            Self::K3V4NormCorr => (3, 4, true),
            Self::K3V3NormCorr => (3, 3, true),
        }
    }
}

/// Configuration and byte-layout calculator for sub-byte quantized KV-cache serialization with
/// unquantized leading/trailing layer bypass (`skip_first_layers` / `skip_last_layers`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubByteKvSerdeSpec {
    /// Compression preset.
    pub preset: SubByteKvSerdePreset,
    /// Per-head hidden dimension.
    pub head_dim: usize,
    /// Token block size used by the compressed layout.
    pub block_size: usize,
    /// Number of leading layers stored without quantization.
    pub skip_first_layers: usize,
    /// Number of trailing layers stored without quantization.
    pub skip_last_layers: usize,
}

impl SubByteKvSerdeSpec {
    /// Packed byte size of one head's key vector.
    #[must_use]
    pub const fn key_packed_size(&self) -> usize {
        let (k_bits, _, _) = self.preset.params();
        if k_bits == 8 {
            self.head_dim
        } else {
            (self.head_dim * k_bits).div_ceil(8) + 2
        }
    }

    /// Packed byte size of one head's value vector (including 4-byte fp16 scale + zero-point).
    #[must_use]
    pub const fn value_packed_size(&self) -> usize {
        let (_, v_bits, _) = self.preset.params();
        (self.head_dim * v_bits).div_ceil(8) + 4
    }

    /// 2-byte aligned slot size in bytes (`key_packed_size + value_packed_size` rounded up to
    /// even).
    #[must_use]
    pub const fn slot_size_aligned(&self) -> usize {
        let s = self.key_packed_size() + self.value_packed_size();
        s + (s % 2)
    }

    /// Half-open layer range `(quant_start, quant_end)` compressed with sub-byte quantization.
    #[must_use]
    pub fn quantized_layer_range(&self, num_layers: usize) -> (usize, usize) {
        let start = self.skip_first_layers.min(num_layers);
        let end = num_layers.saturating_sub(self.skip_last_layers).max(start);
        (start, end)
    }

    /// Validates a 4-D KV tensor shape `[2, num_layers, num_tokens, hidden_dim]` and returns
    /// `(num_layers, num_tokens, num_heads, head_dim)`.
    pub fn validate_kv_shape(
        &self,
        shape: &[usize],
    ) -> Result<(usize, usize, usize, usize), String> {
        if shape.len() != 4 || shape[0] != 2 {
            return Err(
                "expected 4D KV tensor shape [2, num_layers, num_tokens, hidden_dim]".into(),
            );
        }
        if self.head_dim == 0 || self.block_size == 0 || !shape[3].is_multiple_of(self.head_dim) {
            return Err("hidden_dim must be divisible by positive head_dim and block_size".into());
        }
        Ok((shape[1], shape[2], shape[3] / self.head_dim, self.head_dim))
    }

    /// Computes the total serialized byte count for `shape` (`[2, L, T, D]`) given `raw_elem_bytes`
    /// for the bypassed unquantized layers.
    pub fn serialized_nbytes(
        &self,
        shape: &[usize],
        raw_elem_bytes: usize,
    ) -> Result<usize, String> {
        let (num_layers, num_tokens, num_heads, _) = self.validate_kv_shape(shape)?;
        let num_blocks = num_tokens.div_ceil(self.block_size);
        let (q_start, q_end) = self.quantized_layer_range(num_layers);
        let quant_layers = q_end - q_start;
        let raw_layers = num_layers - quant_layers;
        let raw_bytes = 2 * raw_layers * num_tokens * shape[3] * raw_elem_bytes;
        let quant_bytes =
            quant_layers * num_blocks * self.block_size * num_heads * self.slot_size_aligned();
        Ok(raw_bytes + quant_bytes)
    }
}

// ============================================================================
// Model Input Argument Static Shape Verifier
// ============================================================================

/// Formats a tensor shape `[Dim]` into a comma-separated diagnostic string where `Dim::Static(d)`
/// prints as `"{d}"` and `Dim::Dynamic(_)` prints as `"any"`.
#[must_use]
pub fn format_expected_tensor_shape(shape: &[Dim]) -> String {
    shape
        .iter()
        .map(|d| match d {
            Dim::Static(v) => v.to_string(),
            _ => "any".to_string(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Inserts runtime shape-verification assertions (`mgp.tensor.extract.tensor_spec`,
/// `index.constant`, `mgp.tensor_spec.get_dim`, `index.cmp`, `pop.and`, and `mgp.assert`) at the
/// start of each function's entry block for every non-chain tensor block argument that has at
/// least one static dimension. Returns the number of verified tensor arguments.
pub fn verify_model_argument_shapes(module: &mut Module) -> usize {
    let funcs = module.functions();
    let mut verified_args = 0usize;

    for func_id in funcs {
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let reg_id = module.ops[func_id].regions[0];
        if module.regions[reg_id].blocks.is_empty() {
            continue;
        }
        let blk_id = module.regions[reg_id].blocks[0];
        let args = module.blocks[blk_id].arguments.clone();
        let loc = module.ops[func_id].location.clone();
        let i64_scalar_ty = Type::Tensor(crate::ir::mlir::TensorType {
            shape: vec![Dim::Static(1)],
            dtype: DType::I64,
            encoding: None,
        });

        let mut prepended_ops: Vec<OpId> = Vec::new();
        let mut explicit_arg_idx = 0usize;

        for &arg_val in &args {
            if is_chain_tensor_type(&module.values[arg_val].ty) {
                continue;
            }
            let cur_arg_idx = explicit_arg_idx;
            explicit_arg_idx += 1;

            let Type::Tensor(ref t_ty) = module.values[arg_val].ty else {
                continue;
            };
            let shape = t_ty.shape.clone();
            if shape.is_empty() || !shape.iter().any(|d| matches!(d, Dim::Static(_))) {
                continue;
            }

            // 1. `mgp.tensor.extract.tensor_spec %arg`
            let spec_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "tensor.extract.tensor_spec"),
                operands: SmallVec::from_slice(&[arg_val]),
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
            prepended_ops.push(spec_op);

            let mut are_equal: Option<ValueId> = None;
            for (dim_idx, dim) in shape.iter().enumerate() {
                let Dim::Static(expected_d) = *dim else {
                    continue;
                };

                let mut cst_attrs = IndexMap::new();
                cst_attrs.insert("value".to_string(), Attribute::Int(expected_d));
                let cst_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("index", "constant"),
                    operands: SmallVec::new(),
                    results: SmallVec::new(),
                    attributes: cst_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let cst_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: i64_scalar_ty.clone(),
                    def: ValueDef::OpResult {
                        op: cst_op,
                        index: 0,
                    },
                });
                module.ops[cst_op].results.push(cst_val);
                prepended_ops.push(cst_op);

                let mut get_dim_attrs = IndexMap::new();
                get_dim_attrs.insert("dim_index".to_string(), Attribute::Int(dim_idx as i64));
                let get_dim_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mgp", "tensor_spec.get_dim"),
                    operands: SmallVec::from_slice(&[spec_val]),
                    results: SmallVec::new(),
                    attributes: get_dim_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let rt_dim_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: i64_scalar_ty.clone(),
                    def: ValueDef::OpResult {
                        op: get_dim_op,
                        index: 0,
                    },
                });
                module.ops[get_dim_op].results.push(rt_dim_val);
                prepended_ops.push(get_dim_op);

                let mut cmp_attrs = IndexMap::new();
                cmp_attrs.insert("predicate".to_string(), Attribute::String("eq".to_string()));
                let cmp_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("index", "cmp"),
                    operands: SmallVec::from_slice(&[rt_dim_val, cst_val]),
                    results: SmallVec::new(),
                    attributes: cmp_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let cmp_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: i64_scalar_ty.clone(),
                    def: ValueDef::OpResult {
                        op: cmp_op,
                        index: 0,
                    },
                });
                module.ops[cmp_op].results.push(cmp_val);
                prepended_ops.push(cmp_op);

                are_equal = Some(match are_equal {
                    None => cmp_val,
                    Some(prev) => {
                        let and_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("pop", "and"),
                            operands: SmallVec::from_slice(&[prev, cmp_val]),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let and_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: i64_scalar_ty.clone(),
                            def: ValueDef::OpResult {
                                op: and_op,
                                index: 0,
                            },
                        });
                        module.ops[and_op].results.push(and_val);
                        prepended_ops.push(and_op);
                        and_val
                    }
                });
            }

            if let Some(cond_val) = are_equal {
                let msg = format!(
                    "Tensor type mismatch for argument {cur_arg_idx}: Expected shape: [{}]",
                    format_expected_tensor_shape(&shape)
                );
                let mut assert_attrs = IndexMap::new();
                assert_attrs.insert("message".to_string(), Attribute::String(msg));
                let assert_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mgp", "assert"),
                    operands: SmallVec::from_slice(&[cond_val]),
                    results: SmallVec::new(),
                    attributes: assert_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                prepended_ops.push(assert_op);
                verified_args += 1;
            }
        }

        if !prepended_ops.is_empty() {
            prepended_ops.extend_from_slice(&module.blocks[blk_id].ops);
            module.blocks[blk_id].ops = prepended_ops;
        }
    }

    verified_args
}
