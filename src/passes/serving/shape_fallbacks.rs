//! Structured loop outlining, traceback attachment, reduction fusion, and shape fallbacks.

use crate::ir::{Attribute, BlockId, Dim, Module, OpId, OpName, Operation, Type, ValueId};

pub(super) fn structured_loop_body_closure_op_name(loop_op_name: &str) -> Option<&'static str> {
    match loop_op_name {
        "map.iter.elementwise" => Some("map.iter.elementwise.body"),
        "map.iter.reduction" => Some("map.iter.reduction.body"),
        _ => None,
    }
}

/// Wraps structured MAP loop operations (`map.iter.elementwise`, `map.iter.reduction`) in a
/// `map.iter.opaque` region whose body contains a matching capturing fusion closure
/// (`map.iter.elementwise.body` or `map.iter.reduction.body`).
///
/// The outer `map.iter.opaque` inherits the loop's device attribute (`"cpu:0"` fallback if
/// unannotated), `fusion_info`, `has_read_effect`, `has_write_effect`, and `dispatch` attributes.
/// Non-chain inputs and non-chain results become block arguments of the `map.iter.opaque` body and
/// are passed as captures to the inner closure, while the loop's trailing index block argument is
/// moved to position `0` of the closure's body block (`[index, captures...]`).
///
/// Returns the number of structured loop operations outlined.
pub fn outline_structured_map_loops(module: &mut Module) -> usize {
    use crate::ir::mlir::{OpName, Operation, RegionId};

    let mut outlined = 0usize;
    let region_ids: Vec<RegionId> = module.regions.iter().map(|(r, _)| r).collect();
    for rid in region_ids {
        let block_ids = module.regions[rid].blocks.clone();
        for bid in block_ids {
            let ops_snapshot = module.blocks[bid].ops.clone();
            for loop_id in ops_snapshot {
                let Some(loop_op) = module.ops.get(loop_id).cloned() else {
                    continue;
                };
                let full_name = loop_op.name.full_name();
                let Some(closure_full_name) = structured_loop_body_closure_op_name(&full_name)
                else {
                    continue;
                };
                let Some(&old_body_reg) = loop_op.regions.first() else {
                    continue;
                };
                let Some(&old_body_blk) = module.regions[old_body_reg].blocks.first() else {
                    continue;
                };
                if module.blocks[old_body_blk].arguments.is_empty() {
                    continue;
                }

                // Move the trailing `index` block argument of `old_body_blk` to index 0:
                let mut inner_args = module.blocks[old_body_blk].arguments.clone();
                if let Some(idx_arg) = inner_args.pop() {
                    inner_args.insert(0, idx_arg);
                }
                for (pos, &arg_v) in inner_args.iter().enumerate() {
                    module.values[arg_v].def = crate::ir::mlir::ValueDef::BlockArgument {
                        block: old_body_blk,
                        index: pos as u32,
                    };
                }
                module.blocks[old_body_blk].arguments = inner_args;

                // Build the new `map.iter.opaque` body block and its non-chain capture args:
                let opaque_blk = module.create_block(None);
                let mut body_captures = smallvec::SmallVec::<[ValueId; 4]>::new();
                for &inp in &loop_op.operands {
                    let ty = module.values[inp].ty.clone();
                    if !matches!(&ty, Type::Opaque(s) if s == "!mo.chain") {
                        let barg = module.add_block_argument(opaque_blk, None, ty);
                        body_captures.push(barg);
                    }
                }
                for &res in &loop_op.results {
                    let ty = module.values[res].ty.clone();
                    if !matches!(&ty, Type::Opaque(s) if s == "!mo.chain") {
                        let barg = module.add_block_argument(opaque_blk, None, ty);
                        body_captures.push(barg);
                    }
                }

                let (closure_dialect, closure_op_short) = closure_full_name
                    .split_once('.')
                    .unwrap_or(("map", closure_full_name));
                let closure_op = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: OpName::new(closure_dialect, closure_op_short),
                    operands: body_captures,
                    results: smallvec::SmallVec::new(),
                    attributes: indexmap::IndexMap::new(),
                    regions: vec![old_body_reg],
                    custom_syntax: false,
                    location: loop_op.location.clone(),
                });
                module.ops[closure_op].id = closure_op;
                module.blocks[opaque_blk].ops.push(closure_op);

                let opaque_reg = module.create_region(vec![opaque_blk]);
                let mut new_attrs = loop_op.attributes.clone();
                if !new_attrs.contains_key("device") {
                    new_attrs.insert("device".to_string(), Attribute::String("cpu:0".to_string()));
                }

                module.ops[loop_id].name = OpName::new("map", "iter.opaque");
                module.ops[loop_id].attributes = new_attrs;
                module.ops[loop_id].regions = vec![opaque_reg];
                outlined += 1;
            }
        }
    }

    outlined
}

/// Extracts a Python stack trace from an operation's attributes or named location.
pub(super) fn extract_op_python_stack_trace(op: &Operation) -> Option<String> {
    for key in ["python_stack_trace", "stack_trace"] {
        if let Some(Attribute::String(s)) = op.attributes.get(key)
            && !s.is_empty()
        {
            return Some(s.clone());
        }
    }
    if let crate::ir::Location::Named(name) = &op.location
        && let Some(rest) = name.strip_prefix("python_frame:")
        && !rest.is_empty()
    {
        return Some(rest.to_string());
    }
    None
}

/// Embeds Python stack traces into `mgp.assert` diagnostic messages and materializes
/// `"source_notes"` metadata across operations while stripping non-serializable frame wrappers.
///
/// For each operation carrying a `python_stack_trace` attribute or a `Location::Named` prefixed
/// with `"python_frame:"`:
/// 1. If the operation is `mgp.assert`, prepends `\nSource Traceback:\n<trace>\nerror: ` to its
///    `"msg"` attribute (if not already attached).
/// 2. Records `"source_notes"` on the operation and unwraps the outer `python_frame:` location to
///    `Location::Unknown`.
///
/// Returns the number of operations whose Python stack traces were materialized.
pub fn attach_python_stack_traces(module: &mut Module) -> usize {
    let op_ids: Vec<OpId> = module.ops.keys().collect();
    let mut materialized = 0usize;

    for op_id in op_ids {
        let Some(trace) = extract_op_python_stack_trace(&module.ops[op_id]) else {
            continue;
        };

        let is_assert = module.ops[op_id].name.full_name() == "mgp.assert";
        if is_assert {
            let orig_msg = match module.ops[op_id].attributes.get("msg") {
                Some(Attribute::String(s)) => s.clone(),
                _ => String::new(),
            };
            if !orig_msg.starts_with("\nSource Traceback:\n") {
                let formatted = format!("\nSource Traceback:\n{trace}\nerror: {orig_msg}");
                module.ops[op_id]
                    .attributes
                    .insert("msg".to_string(), Attribute::String(formatted));
            }
        }

        module.ops[op_id]
            .attributes
            .shift_remove("python_stack_trace");
        module.ops[op_id]
            .attributes
            .insert("source_notes".to_string(), Attribute::String(trace));
        if let crate::ir::Location::Named(name) = &module.ops[op_id].location
            && name.starts_with("python_frame:")
        {
            module.ops[op_id].location = crate::ir::Location::Unknown;
        }
        materialized += 1;
    }

    materialized
}

/// Recursively collects all `BlockId`s inside `region_id` (including nested op regions).
pub(super) fn collect_region_blocks_recursive(
    module: &Module,
    region_id: crate::ir::RegionId,
    out: &mut Vec<BlockId>,
) {
    for &blk_id in &module.regions[region_id].blocks {
        out.push(blk_id);
        for &op_id in &module.blocks[blk_id].ops {
            for &sub_reg in &module.ops[op_id].regions {
                collect_region_blocks_recursive(module, sub_reg, out);
            }
        }
    }
}

/// Rewrites control-flow operations (`grt.if`, `mo.if`, `grt.while`, `mo.while`) so their regions
/// are isolated from above by threading free outer SSA values through explicit operands, block
/// arguments, and (for loop-like ops) region terminators and results.
///
/// For `grt.while` / `mo.while` condition regions (region 0 when two regions exist), newly
/// threaded values are inserted immediately before the final boolean condition operand of the
/// terminator (`mef.output` / `mo.output` / `grt.yield`).
///
/// Returns the number of control-flow operations rewritten.
pub fn fixup_isolated_from_above_regions(module: &mut Module) -> usize {
    let op_ids: Vec<OpId> = module.ops.keys().collect();
    let mut fixed_ops = 0usize;

    for cf_id in op_ids {
        let full_name = module.ops[cf_id].name.full_name();
        let is_loop_like = matches!(full_name.as_str(), "grt.while" | "mo.while");
        let is_branch_like = matches!(full_name.as_str(), "grt.if" | "mo.if");
        if !is_loop_like && !is_branch_like {
            continue;
        }
        let regions = module.ops[cf_id].regions.clone();
        if regions.is_empty() {
            continue;
        }

        // Collect all blocks and values defined inside `cf_id`'s regions:
        let mut all_sub_blocks = Vec::new();
        for &r in &regions {
            collect_region_blocks_recursive(module, r, &mut all_sub_blocks);
        }
        let mut defined_inside = std::collections::HashSet::<ValueId>::new();
        for &b in &all_sub_blocks {
            for &arg in &module.blocks[b].arguments {
                defined_inside.insert(arg);
            }
            for &op in &module.blocks[b].ops {
                for &res in &module.ops[op].results {
                    defined_inside.insert(res);
                }
            }
        }

        // Find free values referenced inside `cf_id`'s regions in deterministic order:
        let mut free_values = Vec::<ValueId>::new();
        for &b in &all_sub_blocks {
            for &op in &module.blocks[b].ops {
                for &operand in &module.ops[op].operands {
                    if !defined_inside.contains(&operand) && !free_values.contains(&operand) {
                        free_values.push(operand);
                    }
                }
            }
        }
        if free_values.is_empty() {
            continue;
        }

        // 1. Append `free_values` to `cf_id` operands:
        for &fv in &free_values {
            module.ops[cf_id].operands.push(fv);
        }

        // 2. If loop-like, append corresponding results to `cf_id`:
        if is_loop_like {
            let next_idx = module.ops[cf_id].results.len();
            for (offset, &fv) in free_values.iter().enumerate() {
                let ty = module.values[fv].ty.clone();
                let res_val = module.values.insert(crate::ir::mlir::Value {
                    id: ValueId::default(),
                    ssa_name: None,
                    ty,
                    def: crate::ir::mlir::ValueDef::OpResult {
                        op: cf_id,
                        index: (next_idx + offset) as u32,
                    },
                });
                module.values[res_val].id = res_val;
                module.ops[cf_id].results.push(res_val);
            }
        }

        // 3. Thread `free_values` into each region's entry block and rewrite uses:
        let num_regions = regions.len();
        for (reg_idx, &reg_id) in regions.iter().enumerate() {
            let Some(&entry_blk) = module.regions[reg_id].blocks.first() else {
                continue;
            };
            let mut reg_blocks = Vec::new();
            collect_region_blocks_recursive(module, reg_id, &mut reg_blocks);

            let mut fv_to_barg = std::collections::HashMap::<ValueId, ValueId>::new();
            let mut new_bargs = Vec::with_capacity(free_values.len());
            for &fv in &free_values {
                let ty = module.values[fv].ty.clone();
                let barg = module.add_block_argument(entry_blk, None, ty);
                fv_to_barg.insert(fv, barg);
                new_bargs.push(barg);
            }

            for &b in &reg_blocks {
                for &op in &module.blocks[b].ops {
                    for operand in &mut module.ops[op].operands {
                        if let Some(&mapped) = fv_to_barg.get(operand) {
                            *operand = mapped;
                        }
                    }
                }
            }

            // If loop-like, also forward `new_bargs` through the region's output terminator:
            if is_loop_like && let Some(&term_op) = module.blocks[entry_blk].ops.last() {
                let term_name = module.ops[term_op].name.full_name();
                if matches!(
                    term_name.as_str(),
                    "mef.output" | "mo.output" | "grt.yield" | "scf.yield"
                ) {
                    let before_last = num_regions == 2 && reg_idx == 0;
                    if before_last && let Some(last_opd) = module.ops[term_op].operands.pop() {
                        for &barg in &new_bargs {
                            module.ops[term_op].operands.push(barg);
                        }
                        module.ops[term_op].operands.push(last_opd);
                    } else {
                        for &barg in &new_bargs {
                            module.ops[term_op].operands.push(barg);
                        }
                    }
                }
            }
        }

        fixed_ops += 1;
    }

    fixed_ops
}

/// Returns the device string of an operation, defaulting to `"cpu:0"`.
pub(super) fn op_device_or_default(op: &Operation) -> &str {
    match op.attributes.get("device") {
        Some(Attribute::String(s)) if !s.is_empty() => s.as_str(),
        _ => "cpu:0",
    }
}

/// Checks whether a `map.iter.elementwise` operation is a plain single-output foreach without
/// input or output `!mo.chain` tokens.
pub(super) fn is_plain_elementwise_foreach(module: &Module, op_id: OpId) -> bool {
    let op = &module.ops[op_id];
    if op.name.full_name() != "map.iter.elementwise" || op.results.len() != 1 {
        return false;
    }
    let no_chain_in = op
        .operands
        .iter()
        .all(|&v| !matches!(&module.values[v].ty, Type::Opaque(s) if s == "!mo.chain"));
    let no_chain_out = op
        .results
        .iter()
        .all(|&v| !matches!(&module.values[v].ty, Type::Opaque(s) if s == "!mo.chain"));
    no_chain_in && no_chain_out
}

/// Checks whether `per_row_ty` has the same rank and non-axis dimensions as `full_ty`, with
/// dimension `axis` equal to static `1`.
pub(super) fn has_matching_per_row_shape(full_ty: &Type, per_row_ty: &Type, axis: usize) -> bool {
    let (Type::Tensor(full_tt), Type::Tensor(row_tt)) = (full_ty, per_row_ty) else {
        return false;
    };
    if full_tt.shape.len() != row_tt.shape.len() || axis >= full_tt.shape.len() {
        return false;
    }
    if row_tt.shape[axis] != Dim::Static(1) {
        return false;
    }
    for d in 0..full_tt.shape.len() {
        if d != axis && full_tt.shape[d] != row_tt.shape[d] {
            return false;
        }
    }
    true
}

/// Checks whether `idx_val` inside `body_blk` is produced by a chain of index views that includes
/// `map.index.reshape`.
pub(super) fn index_value_crosses_reshape(
    module: &Module,
    body_blk: BlockId,
    idx_val: ValueId,
) -> bool {
    let mut worklist = vec![idx_val];
    let mut visited = std::collections::HashSet::new();
    while let Some(curr) = worklist.pop() {
        if !visited.insert(curr) {
            continue;
        }
        if let crate::ir::mlir::ValueDef::OpResult { op: def_op, .. } = module.values[curr].def
            && module.blocks[body_blk].ops.contains(&def_op)
        {
            let name = module.ops[def_op].name.full_name();
            if name == "map.index.reshape" {
                return true;
            }
            if name.starts_with("map.index.") {
                for &opd in &module.ops[def_op].operands {
                    worklist.push(opd);
                }
            }
        }
    }
    false
}

/// Merges boolean effect attributes (`has_read_effect`, `has_write_effect`) from `src_op` into
/// `dst_op`.
pub(super) fn merge_map_iter_effects(module: &mut Module, dst_op: OpId, src_op: OpId) {
    for attr_name in ["has_read_effect", "has_write_effect"] {
        let src_has = matches!(
            module.ops[src_op].attributes.get(attr_name),
            Some(Attribute::Bool(true))
        );
        let dst_has = matches!(
            module.ops[dst_op].attributes.get(attr_name),
            Some(Attribute::Bool(true))
        );
        if src_has || dst_has {
            module.ops[dst_op]
                .attributes
                .insert(attr_name.to_string(), Attribute::Bool(true));
        }
    }
}

/// Fuses sole-use plain `map.iter.elementwise` producers and per-row `map.iter.elementwise`
/// consumers into `map.iter.rowwise` reductions on the same device.
///
/// 1. **Producer fusion**: When a plain `map.iter.elementwise` producer on the same device is only
///    read by a `map.iter.rowwise` via `map.load` (without crossing a `map.index.reshape`
///    barrier), its body computation is spliced into `map.iter.rowwise` at each load site and its
///    inputs are deduplicated into the reduction.
/// 2. **Consumer fusion**: When a plain `map.iter.elementwise` consumer on the same device has the
///    per-row shape of `map.iter.rowwise` (reduced `axis` dimension equal to `1`) and reads the
///    reduction's per-row output at its loop index, its body is appended to the reduction's row
///    epilogue and intermediate per-row stores with no remaining external users are eliminated.
///
/// Returns `(producers_fused, consumers_fused)`.
pub fn fuse_rowwise_reductions(module: &mut Module) -> (usize, usize) {
    let mut producers_fused = 0usize;
    let mut consumers_fused = 0usize;

    for func_id in module.functions() {
        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let rowwise_candidates: Vec<OpId> = module.blocks[blk_id]
                    .ops
                    .iter()
                    .copied()
                    .filter(|&id| module.ops[id].name.full_name() == "map.iter.rowwise")
                    .collect();

                for dst_id in rowwise_candidates {
                    if !module.blocks[blk_id].ops.contains(&dst_id) {
                        continue;
                    }

                    // --- Phase 1: Fuse eligible elementwise producers into `dst_id` ---
                    loop {
                        let dst_dev = op_device_or_default(&module.ops[dst_id]).to_string();
                        let Some(&dst_reg) = module.ops[dst_id].regions.first() else {
                            break;
                        };
                        let Some(&dst_blk) = module.regions[dst_reg].blocks.first() else {
                            break;
                        };
                        let dst_operands = module.ops[dst_id].operands.clone();

                        let mut found_producer: Option<(usize, OpId, ValueId)> = None;
                        for (inp_idx, &inp_val) in dst_operands.iter().enumerate() {
                            let crate::ir::mlir::ValueDef::OpResult { op: src_id, .. } =
                                module.values[inp_val].def
                            else {
                                continue;
                            };
                            if !module.blocks[blk_id].ops.contains(&src_id)
                                || !is_plain_elementwise_foreach(module, src_id)
                                || op_device_or_default(&module.ops[src_id]) != dst_dev
                            {
                                continue;
                            }
                            // Check that `dst_id` is the only user of `inp_val` in `blk_id`:
                            let sole_user = module.blocks[blk_id]
                                .ops
                                .iter()
                                .filter(|&&op| module.ops[op].operands.contains(&inp_val))
                                .all(|&op| op == dst_id);
                            if !sole_user || inp_idx >= module.blocks[dst_blk].arguments.len() {
                                continue;
                            }
                            let dst_in_barg = module.blocks[dst_blk].arguments[inp_idx];
                            // Every use of `dst_in_barg` inside `dst_blk` must be a `map.load`
                            // whose index does not cross `map.index.reshape`:
                            let load_ops: Vec<OpId> = module.blocks[dst_blk]
                                .ops
                                .iter()
                                .copied()
                                .filter(|&op| module.ops[op].operands.contains(&dst_in_barg))
                                .collect();
                            if load_ops.is_empty() {
                                continue;
                            }
                            let valid_loads = load_ops.iter().all(|&lop| {
                                module.ops[lop].name.full_name() == "map.load"
                                    && module.ops[lop].operands.len() == 2
                                    && module.ops[lop].operands[0] == dst_in_barg
                                    && !index_value_crosses_reshape(
                                        module,
                                        dst_blk,
                                        module.ops[lop].operands[1],
                                    )
                            });
                            if valid_loads {
                                found_producer = Some((inp_idx, src_id, dst_in_barg));
                                break;
                            }
                        }

                        let Some((inp_idx, src_id, dst_in_barg)) = found_producer else {
                            break;
                        };
                        let Some(&src_reg) = module.ops[src_id].regions.first() else {
                            break;
                        };
                        let Some(&src_blk) = module.regions[src_reg].blocks.first() else {
                            break;
                        };
                        let src_inputs = module.ops[src_id].operands.clone();
                        let src_bargs = module.blocks[src_blk].arguments.clone();
                        if src_bargs.len() < src_inputs.len() + 2 {
                            break;
                        }
                        let src_loop_idx_barg = src_bargs[src_bargs.len() - 1];

                        // Map `src`'s first input into `inp_idx` and deduplicate/append any
                        // additional `src` inputs:
                        let mut src_barg_map = std::collections::HashMap::<ValueId, ValueId>::new();
                        if !src_inputs.is_empty() {
                            module.ops[dst_id].operands[inp_idx] = src_inputs[0];
                            let ty0 = module.values[src_inputs[0]].ty.clone();
                            module.values[dst_in_barg].ty = ty0;
                            src_barg_map.insert(src_bargs[0], dst_in_barg);

                            for (extra_i, &extra_in) in src_inputs.iter().enumerate().skip(1) {
                                if let Some(existing_pos) = module.ops[dst_id]
                                    .operands
                                    .iter()
                                    .position(|&v| v == extra_in)
                                {
                                    let existing_barg =
                                        module.blocks[dst_blk].arguments[existing_pos];
                                    src_barg_map.insert(src_bargs[extra_i], existing_barg);
                                } else {
                                    let new_pos = module.ops[dst_id].operands.len();
                                    module.ops[dst_id].operands.insert(new_pos, extra_in);
                                    let ty = module.values[extra_in].ty.clone();
                                    let new_barg = module.add_block_argument(dst_blk, None, ty);
                                    // Move `new_barg` to `new_pos` (before output & row args):
                                    if let Some(popped) = module.blocks[dst_blk].arguments.pop() {
                                        module.blocks[dst_blk].arguments.insert(new_pos, popped);
                                    }
                                    for (pos, &b) in
                                        module.blocks[dst_blk].arguments.iter().enumerate()
                                    {
                                        module.values[b].def =
                                            crate::ir::mlir::ValueDef::BlockArgument {
                                                block: dst_blk,
                                                index: pos as u32,
                                            };
                                    }
                                    src_barg_map.insert(src_bargs[extra_i], new_barg);
                                }
                            }
                        }

                        // Splice `src_blk` ops before each `map.load` of `dst_in_barg`:
                        let load_sites: Vec<OpId> = module.blocks[dst_blk]
                            .ops
                            .iter()
                            .copied()
                            .filter(|&op| {
                                module.ops[op].name.full_name() == "map.load"
                                    && module.ops[op].operands.first() == Some(&dst_in_barg)
                            })
                            .collect();
                        let src_body_ops = module.blocks[src_blk].ops.clone();

                        for load_id in load_sites {
                            let load_idx_val = module.ops[load_id].operands[1];
                            let load_res_val = module.ops[load_id].results[0];
                            let mut val_map = src_barg_map.clone();
                            val_map.insert(src_loop_idx_barg, load_idx_val);

                            let mut cloned_ops = Vec::new();
                            let mut stored_replacement = None;
                            for &sop in &src_body_ops {
                                if module.ops[sop].name.full_name() == "map.store" {
                                    if let Some(&val_opd) = module.ops[sop].operands.first() {
                                        stored_replacement =
                                            Some(val_map.get(&val_opd).copied().unwrap_or(val_opd));
                                    }
                                    continue;
                                }
                                let mapped_operands: smallvec::SmallVec<[ValueId; 4]> = module.ops
                                    [sop]
                                    .operands
                                    .iter()
                                    .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                                    .collect();
                                let res_types: Vec<Type> = module.ops[sop]
                                    .results
                                    .iter()
                                    .map(|&r| module.values[r].ty.clone())
                                    .collect();
                                let new_op = module.ops.insert(Operation {
                                    id: OpId::default(),
                                    name: module.ops[sop].name.clone(),
                                    operands: mapped_operands,
                                    results: smallvec::SmallVec::new(),
                                    attributes: module.ops[sop].attributes.clone(),
                                    regions: Vec::new(),
                                    custom_syntax: module.ops[sop].custom_syntax,
                                    location: module.ops[sop].location.clone(),
                                });
                                module.ops[new_op].id = new_op;
                                for (r_idx, r_ty) in res_types.into_iter().enumerate() {
                                    let rv = module.values.insert(crate::ir::mlir::Value {
                                        id: ValueId::default(),
                                        ssa_name: None,
                                        ty: r_ty,
                                        def: crate::ir::mlir::ValueDef::OpResult {
                                            op: new_op,
                                            index: r_idx as u32,
                                        },
                                    });
                                    module.values[rv].id = rv;
                                    module.ops[new_op].results.push(rv);
                                    val_map.insert(module.ops[sop].results[r_idx], rv);
                                }
                                cloned_ops.push(new_op);
                            }

                            if let Some(rep_val) = stored_replacement {
                                // Replace uses of `load_res_val` in `dst_blk` with `rep_val`:
                                for &bop in &module.blocks[dst_blk].ops {
                                    for opd in &mut module.ops[bop].operands {
                                        if *opd == load_res_val {
                                            *opd = rep_val;
                                        }
                                    }
                                }
                                if let Some(load_pos) = module.blocks[dst_blk]
                                    .ops
                                    .iter()
                                    .position(|&x| x == load_id)
                                {
                                    module.blocks[dst_blk]
                                        .ops
                                        .splice(load_pos..=load_pos, cloned_ops);
                                }
                            }
                        }

                        merge_map_iter_effects(module, dst_id, src_id);
                        module.blocks[blk_id].ops.retain(|&op| op != src_id);
                        producers_fused += 1;
                    }

                    // --- Phase 2: Fuse eligible per-row elementwise consumers into `dst_id` ---
                    loop {
                        if module.ops[dst_id].results.len() != 1
                            || module.ops[dst_id].operands.is_empty()
                        {
                            break;
                        }
                        let dst_res = module.ops[dst_id].results[0];
                        let dst_dev = op_device_or_default(&module.ops[dst_id]).to_string();
                        let axis = match module.ops[dst_id].attributes.get("axis") {
                            Some(Attribute::Int(v)) if *v >= 0 => *v as usize,
                            _ => 1usize,
                        };
                        let full_ty = module.values[module.ops[dst_id].operands[0]].ty.clone();
                        let Some(&dst_reg) = module.ops[dst_id].regions.first() else {
                            break;
                        };
                        let Some(&dst_blk) = module.regions[dst_reg].blocks.first() else {
                            break;
                        };
                        let num_dst_inputs = module.ops[dst_id].operands.len();
                        if module.blocks[dst_blk].arguments.len() < num_dst_inputs + 2 {
                            break;
                        }
                        let dst_out_barg = module.blocks[dst_blk].arguments[num_dst_inputs];
                        let dst_bargs_len = module.blocks[dst_blk].arguments.len();
                        let dst_row_barg = module.blocks[dst_blk].arguments[dst_bargs_len - 1];

                        // Verify `dst_out_barg` has a single `map.store(val, dst_out_barg,
                        // dst_row_barg)` in `dst_blk`:
                        let out_stores: Vec<OpId> = module.blocks[dst_blk]
                            .ops
                            .iter()
                            .copied()
                            .filter(|&op| module.ops[op].operands.contains(&dst_out_barg))
                            .collect();
                        if out_stores.len() != 1 {
                            break;
                        }
                        let row_store_id = out_stores[0];
                        if module.ops[row_store_id].name.full_name() != "map.store"
                            || module.ops[row_store_id].operands.len() != 3
                            || module.ops[row_store_id].operands[1] != dst_out_barg
                            || module.ops[row_store_id].operands[2] != dst_row_barg
                        {
                            break;
                        }
                        let per_row_stored_val = module.ops[row_store_id].operands[0];

                        // Find a consumer `map.iter.elementwise` in `blk_id` reading `dst_res`:
                        let candidate_consumer =
                            module.blocks[blk_id].ops.iter().copied().find(|&op| {
                                op != dst_id
                                    && is_plain_elementwise_foreach(module, op)
                                    && op_device_or_default(&module.ops[op]) == dst_dev
                                    && module.ops[op].operands.contains(&dst_res)
                                    && has_matching_per_row_shape(
                                        &full_ty,
                                        &module.values[module.ops[op].results[0]].ty,
                                        axis,
                                    )
                            });
                        let Some(cons_id) = candidate_consumer else {
                            break;
                        };
                        let Some(&cons_reg) = module.ops[cons_id].regions.first() else {
                            break;
                        };
                        let Some(&cons_blk) = module.regions[cons_reg].blocks.first() else {
                            break;
                        };
                        let cons_inputs = module.ops[cons_id].operands.clone();
                        let cons_bargs = module.blocks[cons_blk].arguments.clone();
                        if cons_bargs.len() < cons_inputs.len() + 2 {
                            break;
                        }
                        let cons_out_barg = cons_bargs[cons_inputs.len()];
                        let cons_loop_barg = cons_bargs[cons_bargs.len() - 1];

                        // Every use of the consumer's input block arg for `dst_res` must be a
                        // `map.load` at `cons_loop_barg`:
                        let mut valid_consumer_loads = true;
                        for (ci, &cin) in cons_inputs.iter().enumerate() {
                            if cin == dst_res {
                                let cbarg = cons_bargs[ci];
                                for &cop in &module.blocks[cons_blk].ops {
                                    if module.ops[cop].operands.contains(&cbarg)
                                        && (module.ops[cop].name.full_name() != "map.load"
                                            || module.ops[cop].operands.len() != 2
                                            || module.ops[cop].operands[0] != cbarg
                                            || module.ops[cop].operands[1] != cons_loop_barg)
                                    {
                                        valid_consumer_loads = false;
                                    }
                                }
                            }
                        }
                        if !valid_consumer_loads {
                            break;
                        }

                        // Map `cons_blk` block arguments into `dst_blk`:
                        let mut cons_map = std::collections::HashMap::<ValueId, ValueId>::new();
                        cons_map.insert(cons_out_barg, dst_out_barg);
                        cons_map.insert(cons_loop_barg, dst_row_barg);
                        let mut dst_res_bargs = std::collections::HashSet::<ValueId>::new();
                        for (ci, &cin) in cons_inputs.iter().enumerate() {
                            if cin == dst_res {
                                dst_res_bargs.insert(cons_bargs[ci]);
                            } else if let Some(existing_pos) =
                                module.ops[dst_id].operands.iter().position(|&v| v == cin)
                            {
                                let existing_barg = module.blocks[dst_blk].arguments[existing_pos];
                                cons_map.insert(cons_bargs[ci], existing_barg);
                            } else {
                                let insert_pos = module.ops[dst_id].operands.len();
                                module.ops[dst_id].operands.push(cin);
                                let ty = module.values[cin].ty.clone();
                                let new_barg = module.add_block_argument(dst_blk, None, ty);
                                if let Some(popped) = module.blocks[dst_blk].arguments.pop() {
                                    module.blocks[dst_blk].arguments.insert(insert_pos, popped);
                                }
                                for (pos, &b) in module.blocks[dst_blk].arguments.iter().enumerate()
                                {
                                    module.values[b].def =
                                        crate::ir::mlir::ValueDef::BlockArgument {
                                            block: dst_blk,
                                            index: pos as u32,
                                        };
                                }
                                cons_map.insert(cons_bargs[ci], new_barg);
                            }
                        }

                        // Remove `row_store_id` from `dst_blk` and append `cons_blk` ops:
                        module.blocks[dst_blk].ops.retain(|&op| op != row_store_id);
                        let cons_body_ops = module.blocks[cons_blk].ops.clone();
                        for &cop in &cons_body_ops {
                            if module.ops[cop].name.full_name() == "map.load"
                                && let Some(&t_opd) = module.ops[cop].operands.first()
                                && dst_res_bargs.contains(&t_opd)
                            {
                                cons_map.insert(module.ops[cop].results[0], per_row_stored_val);
                                continue;
                            }
                            let mapped_operands: smallvec::SmallVec<[ValueId; 4]> = module.ops[cop]
                                .operands
                                .iter()
                                .map(|&v| cons_map.get(&v).copied().unwrap_or(v))
                                .collect();
                            let res_types: Vec<Type> = module.ops[cop]
                                .results
                                .iter()
                                .map(|&r| module.values[r].ty.clone())
                                .collect();
                            let new_op = module.ops.insert(Operation {
                                id: OpId::default(),
                                name: module.ops[cop].name.clone(),
                                operands: mapped_operands,
                                results: smallvec::SmallVec::new(),
                                attributes: module.ops[cop].attributes.clone(),
                                regions: Vec::new(),
                                custom_syntax: module.ops[cop].custom_syntax,
                                location: module.ops[cop].location.clone(),
                            });
                            module.ops[new_op].id = new_op;
                            for (r_idx, r_ty) in res_types.into_iter().enumerate() {
                                let rv = module.values.insert(crate::ir::mlir::Value {
                                    id: ValueId::default(),
                                    ssa_name: None,
                                    ty: r_ty,
                                    def: crate::ir::mlir::ValueDef::OpResult {
                                        op: new_op,
                                        index: r_idx as u32,
                                    },
                                });
                                module.values[rv].id = rv;
                                module.ops[new_op].results.push(rv);
                                cons_map.insert(module.ops[cop].results[r_idx], rv);
                            }
                            module.blocks[dst_blk].ops.push(new_op);
                        }

                        // Update `dst_res` type to match `cons`'s result type and replace uses:
                        let cons_res = module.ops[cons_id].results[0];
                        let cons_res_ty = module.values[cons_res].ty.clone();
                        module.values[dst_res].ty = cons_res_ty.clone();
                        module.values[dst_out_barg].ty = cons_res_ty;
                        for &bop in &module.blocks[blk_id].ops {
                            if bop != cons_id {
                                for opd in &mut module.ops[bop].operands {
                                    if *opd == cons_res {
                                        *opd = dst_res;
                                    }
                                }
                            }
                        }

                        merge_map_iter_effects(module, dst_id, cons_id);
                        module.blocks[blk_id].ops.retain(|&op| op != cons_id);
                        consumers_fused += 1;
                    }
                }
            }
        }
    }

    (producers_fused, consumers_fused)
}

/// Parses a list of declared output parameter names from an operation's `"output_param_decls"`
/// attribute.
pub(super) fn parse_output_param_decls(op: &Operation) -> Vec<String> {
    match op.attributes.get("output_param_decls") {
        Some(Attribute::Array(items)) => items
            .iter()
            .filter_map(|a| match a {
                Attribute::String(s) if !s.is_empty() => Some(s.clone()),
                Attribute::Opaque(s) if !s.is_empty() => Some(s.trim().trim_matches('"').into()),
                _ => None,
            })
            .collect(),
        Some(Attribute::String(s)) if !s.is_empty() => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

/// Infers the data-dependent input operand indices for a shape-function fallback on `op`.
pub(super) fn infer_data_dep_input_indices(op: &Operation) -> Vec<i64> {
    if let Some(Attribute::Array(arr)) = op.attributes.get("data_dep_input_idxs") {
        let parsed: Vec<i64> = arr
            .iter()
            .filter_map(|a| match a {
                Attribute::Int(v) => Some(*v),
                _ => None,
            })
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    match op.name.full_name().as_str() {
        "mo.arg_nonzero" => vec![0],
        "mo.broadcast_to" | "mo.reshape" | "mo.arg_max" | "mo.arg_min" | "mo.reduce_mul"
        | "mo.tile" => vec![1],
        "mo.bottom_k" | "mo.top_k" => vec![1, 2],
        "mo.range" => vec![0, 1, 2],
        "mo.split" | "mo.slice" => vec![1, 2, 3],
        "mo.scatter_add" | "mo.scatter_max" | "mo.scatter_min" | "mo.scatter_mul" => vec![3],
        "mo.non_maximum_suppression" => vec![0, 1, 2, 3, 4],
        _ => Vec::new(),
    }
}

/// Checks whether `val_id` resides on a non-host device (`!= "cpu:0"`).
pub(super) fn is_off_host_tensor_value(
    module: &Module,
    val_id: ValueId,
    consumer_op: &Operation,
) -> bool {
    let Type::Tensor(tt) = &module.values[val_id].ty else {
        return false;
    };
    if let Some(enc) = &tt.encoding
        && !enc.is_empty()
    {
        return enc != "cpu:0" && enc != "cpu";
    }
    if let crate::ir::mlir::ValueDef::OpResult { op: def_op, .. } = module.values[val_id].def
        && let Some(Attribute::String(dev)) = module.ops[def_op].attributes.get("device")
        && !dev.is_empty()
    {
        return dev != "cpu:0" && dev != "cpu";
    }
    if let Some(Attribute::String(dev)) = consumer_op.attributes.get("device")
        && !dev.is_empty()
    {
        return dev != "cpu:0" && dev != "cpu";
    }
    false
}

/// Computes the positional parameter mapping for a fallback shape function over `res_ty`, keeping
/// declared symbolic dims in `declared_set` and assigning fresh `"F<n>"` fallback names to static
/// or externally declared dimensions.
pub(super) fn build_fallback_shape_param_mapping(
    res_ty: &Type,
    declared_set: &std::collections::HashSet<String>,
    fallback_counter: &mut usize,
) -> Vec<Attribute> {
    let Type::Tensor(tt) = res_ty else {
        return Vec::new();
    };
    let mut params = Vec::with_capacity(tt.shape.len());
    for dim in &tt.shape {
        match dim {
            Dim::Symbolic(sym) if declared_set.contains(sym) => {
                params.push(Attribute::String(sym.clone()));
            }
            _ => {
                let fb = format!("F{}", *fallback_counter);
                *fallback_counter += 1;
                params.push(Attribute::String(fb));
            }
        }
    }
    params
}

/// Materializes `mo.invoke_shape_func` fallback shape computations for operations with non-empty
/// `"output_param_decls"`, inserting host transfers (`mo.chain.create` + `mo.transfer` to
/// `"cpu:0"`) for off-host data-dependent tensor operands and clearing `"output_param_decls"` on
/// each processed operation.
///
/// For `mo.custom` operations, validates that `"symbol"` is present in any
/// `"registered_shape_funcs"` list on the enclosing function (returning an `Err` with
/// `@extensibility.register_shape_function` guidance if missing).
///
/// Returns `Ok((shape_funcs_emitted, host_transfers_inserted))`.
pub fn add_fallback_shape_functions(module: &mut Module) -> Result<(usize, usize), String> {
    let mut shape_funcs_emitted = 0usize;
    let mut host_transfers_inserted = 0usize;

    for func_id in module.functions() {
        let mut fallback_counter = 0usize;
        let registered_custom: Option<std::collections::HashSet<String>> =
            match module.ops[func_id].attributes.get("registered_shape_funcs") {
                Some(Attribute::Array(items)) => Some(
                    items
                        .iter()
                        .filter_map(|a| match a {
                            Attribute::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            };

        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let op_ids = module.blocks[blk_id].ops.clone();
                for op_id in op_ids {
                    let decls = parse_output_param_decls(&module.ops[op_id]);
                    if decls.is_empty() {
                        continue;
                    }
                    let declared_set: std::collections::HashSet<String> =
                        decls.iter().cloned().collect();
                    let full_name = module.ops[op_id].name.full_name();
                    let loc = module.ops[op_id].location.clone();

                    if full_name == "mo.custom" {
                        let sym = match module.ops[op_id].attributes.get("symbol") {
                            Some(Attribute::String(s)) if !s.is_empty() => s.clone(),
                            _ => {
                                return Err(
                                    "mo.custom with output_param_decls requires a 'symbol' attr"
                                        .to_string(),
                                );
                            }
                        };
                        if let Some(reg_set) = &registered_custom
                            && !reg_set.contains(&sym)
                        {
                            return Err(format!(
                                "custom op '{sym}' with data-dependent output dimensions requires \
                                 a registered shape function; add \
                                 '@extensibility.register_shape_function(\"{sym}\")' to the kernel"
                            ));
                        }
                    }

                    let data_dep_idxs = infer_data_dep_input_indices(&module.ops[op_id]);
                    let mut invoke_operands = module.ops[op_id].operands.clone();
                    let mut pre_ops = Vec::<OpId>::new();
                    let mut chain_val: Option<ValueId> = None;

                    // Copy any off-host data-dependent tensor operands to `"cpu:0"`:
                    for &dep_idx in &data_dep_idxs {
                        if dep_idx < 0 || (dep_idx as usize) >= invoke_operands.len() {
                            continue;
                        }
                        let idx_usize = dep_idx as usize;
                        let opd_val = invoke_operands[idx_usize];
                        if !is_off_host_tensor_value(module, opd_val, &module.ops[op_id]) {
                            continue;
                        }
                        let ch_ty = Type::Opaque("!mo.chain".to_string());
                        let cur_chain = match chain_val {
                            Some(c) => c,
                            None => {
                                let ch_op = module.ops.insert(Operation {
                                    id: OpId::default(),
                                    name: OpName::new("mo", "chain.create"),
                                    operands: smallvec::SmallVec::new(),
                                    results: smallvec::SmallVec::new(),
                                    attributes: indexmap::IndexMap::new(),
                                    regions: Vec::new(),
                                    custom_syntax: false,
                                    location: loc.clone(),
                                });
                                module.ops[ch_op].id = ch_op;
                                let cv = module.values.insert(crate::ir::mlir::Value {
                                    id: ValueId::default(),
                                    ssa_name: None,
                                    ty: ch_ty.clone(),
                                    def: crate::ir::mlir::ValueDef::OpResult {
                                        op: ch_op,
                                        index: 0,
                                    },
                                });
                                module.values[cv].id = cv;
                                module.ops[ch_op].results.push(cv);
                                pre_ops.push(ch_op);
                                cv
                            }
                        };

                        let mut host_tt = match &module.values[opd_val].ty {
                            Type::Tensor(tt) => tt.clone(),
                            _ => unreachable!(),
                        };
                        host_tt.encoding = Some("cpu:0".to_string());
                        let host_tensor_ty = Type::Tensor(host_tt);

                        let mut xfer_attrs = indexmap::IndexMap::new();
                        xfer_attrs
                            .insert("device".to_string(), Attribute::String("cpu:0".to_string()));
                        let xfer_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("mo", "transfer"),
                            operands: smallvec::smallvec![opd_val, cur_chain],
                            results: smallvec::SmallVec::new(),
                            attributes: xfer_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        module.ops[xfer_op].id = xfer_op;
                        let host_v = module.values.insert(crate::ir::mlir::Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: host_tensor_ty,
                            def: crate::ir::mlir::ValueDef::OpResult {
                                op: xfer_op,
                                index: 0,
                            },
                        });
                        module.values[host_v].id = host_v;
                        let out_ch = module.values.insert(crate::ir::mlir::Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: ch_ty,
                            def: crate::ir::mlir::ValueDef::OpResult {
                                op: xfer_op,
                                index: 1,
                            },
                        });
                        module.values[out_ch].id = out_ch;
                        module.ops[xfer_op].results.push(host_v);
                        module.ops[xfer_op].results.push(out_ch);
                        pre_ops.push(xfer_op);

                        invoke_operands[idx_usize] = host_v;
                        chain_val = Some(out_ch);
                        host_transfers_inserted += 1;
                    }

                    let results = module.ops[op_id].results.clone();
                    let target_results: Vec<(usize, ValueId)> =
                        if full_name == "mo.split" || full_name == "mo.custom" {
                            results.iter().copied().enumerate().collect()
                        } else {
                            results.first().copied().into_iter().enumerate().collect()
                        };

                    for (res_idx, res_val) in target_results {
                        let res_ty = module.values[res_val].ty.clone();
                        if !matches!(res_ty, Type::Tensor(_)) {
                            continue;
                        }
                        let shape_params = build_fallback_shape_param_mapping(
                            &res_ty,
                            &declared_set,
                            &mut fallback_counter,
                        );
                        let func_name_str = if full_name == "mo.split" {
                            "split_ith_output_shape".to_string()
                        } else if full_name == "mo.custom" {
                            match module.ops[op_id].attributes.get("symbol") {
                                Some(Attribute::String(s)) => s.clone(),
                                _ => full_name.clone(),
                            }
                        } else {
                            full_name.clone()
                        };

                        let mut inv_attrs = indexmap::IndexMap::new();
                        inv_attrs.insert("func_name".to_string(), Attribute::String(func_name_str));
                        inv_attrs.insert(
                            "output_param_decls".to_string(),
                            Attribute::Array(shape_params),
                        );
                        inv_attrs.insert(
                            "data_dep_input_idxs".to_string(),
                            Attribute::Array(
                                data_dep_idxs.iter().map(|&i| Attribute::Int(i)).collect(),
                            ),
                        );
                        if let Some(axis_attr) = module.ops[op_id].attributes.get("axis").cloned() {
                            inv_attrs.insert("axis".to_string(), axis_attr);
                        }
                        if full_name == "mo.split" || full_name == "mo.custom" {
                            inv_attrs
                                .insert("output_idx".to_string(), Attribute::Int(res_idx as i64));
                        }

                        let inv_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("mo", "invoke_shape_func"),
                            operands: invoke_operands.clone(),
                            results: smallvec::SmallVec::new(),
                            attributes: inv_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        module.ops[inv_op].id = inv_op;
                        pre_ops.push(inv_op);
                        shape_funcs_emitted += 1;
                    }

                    // Clear `output_param_decls` on `op_id` and splice `pre_ops` right before it:
                    module.ops[op_id]
                        .attributes
                        .insert("output_param_decls".to_string(), Attribute::Array(vec![]));
                    if let Some(pos) = module.blocks[blk_id].ops.iter().position(|&x| x == op_id) {
                        module.blocks[blk_id].ops.splice(pos..pos, pre_ops);
                    }
                }
            }
        }
    }

    Ok((shape_funcs_emitted, host_transfers_inserted))
}
