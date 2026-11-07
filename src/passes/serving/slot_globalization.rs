//! Bottom-up call-graph slot globalization, device-graph kernel registration
//! verification, and single-caller index-constant raising.
//!
//! Covers:
//! 1. **Bottom-Up Call-Graph Slot Globalization (`globalize_mgp_call_slot_indices`)**:
//!    Walks the `mgp.model` call graph in bottom-up post-order, reads each
//!    model's `mgp.context.create` `num_slots`, materializes an
//!    `index.constant` for the caller's running slot offset before each
//!    `mgp.call`, appends the offset operand to `mgp.call`, stamps
//!    `is_globalized_index = true`, and accumulates the callee's `slots_used`
//!    into the caller's running slot count.
//! 2. **Device-Graph Kernel Registration Verification
//!    (`verify_device_graph_kernel_registrations`)**:
//!    Validates `is_device_graph = true` `mo.graph` regions: rejects nested
//!    control-flow operations (`mo.if`, `mo.while`, `scf.if`, `scf.while`) and
//!    verifies that every registered kernel-backed operation resolves to a
//!    `mogg.kernel.decl` with `build_graph = true`.
//! 3. **Single-Caller Index-Constant Raising
//!    (`raise_back_single_caller_index_constants`)**:
//!    For `mogg.func` / `func.func` kernel stubs with `abi = "mgp"` called by
//!    **exactly one** `mogg.compiled_call`, raises
//!    `mgp.buffer.to_index(mgp.resource.constant)` back to `index.constant`
//!    when the caller's `constant_resources[slot]` holds an integer scalar or
//!    8-byte little-endian integer payload.

use std::collections::{BTreeMap, HashMap, HashSet};

use indexmap::IndexMap;
use smallvec::SmallVec;

use crate::ir::mlir::{
    Attribute, BlockId, DType, Location, Module, OpId, OpName, Operation, Type, Value, ValueDef,
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

fn is_control_flow_op_name(full: &str) -> bool {
    matches!(
        full,
        "mo.if" | "mo.while" | "rmo.if" | "rmo.while" | "scf.if" | "scf.while"
    )
}

// ---------------------------------------------------------------------------
// 1. Bottom-Up Call-Graph Slot Globalization
// ---------------------------------------------------------------------------

/// Globalizes runtime slot offsets across `mgp.call` edges in bottom-up
/// call-graph order:
/// - Verifies every `mgp.model` has an `mgp.context.create` operation in its
///   `init` region and records its initial `num_slots`.
/// - For each `mgp.call` in a caller's `execute` region, inserts an
///   `index.constant` carrying the caller's running `model_slot_usage` right
///   before the `mgp.call`, appends that constant to the `mgp.call` operands,
///   stamps `is_globalized_index = true`, and increments the caller's running
///   `model_slot_usage` by the callee's total `slots_used`.
///
/// # Errors
///
/// Returns `Err` if a model is missing `mgp.context.create`, a call references
/// an unknown model, or a cycle exists in the call graph.
pub fn globalize_mgp_call_slot_indices(module: &mut Module) -> Result<usize, String> {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();

    let mut models_by_name: BTreeMap<String, OpId> = BTreeMap::new();
    for &oid in &top_ops {
        let Some(op) = module.ops.get(oid) else {
            continue;
        };
        if op.name.full_name() == "mgp.model"
            && let Some(Attribute::String(sym)) = op
                .attributes
                .get("sym_name")
                .or_else(|| op.attributes.get("name"))
        {
            models_by_name.insert(sym.clone(), oid);
        }
    }

    if models_by_name.len() <= 1 {
        return Ok(0);
    }

    let mut base_slots: HashMap<String, u64> = HashMap::new();
    let mut exec_block_of: HashMap<String, BlockId> = HashMap::new();
    let mut call_edges: BTreeMap<String, Vec<(OpId, String)>> = BTreeMap::new();

    for (name, &model_id) in &models_by_name {
        let op = module
            .ops
            .get(model_id)
            .ok_or_else(|| "model op missing".to_string())?;
        if op.regions.len() < 2 {
            return Err(format!("model @{name} requires init and execute regions"));
        }
        let init_blk = module.regions[op.regions[0]]
            .blocks
            .first()
            .copied()
            .ok_or_else(|| format!("model @{name} init region has no block"))?;
        let exec_blk = module.regions[op.regions[1]]
            .blocks
            .first()
            .copied()
            .ok_or_else(|| format!("model @{name} execute region has no block"))?;
        exec_block_of.insert(name.clone(), exec_blk);

        let mut found_slots: Option<u64> = None;
        for &oid in &module.blocks[init_blk].ops {
            if let Some(iop) = module.ops.get(oid)
                && iop.name.full_name() == "mgp.context.create"
            {
                let s = match iop.attributes.get("num_slots") {
                    Some(&Attribute::Int(v)) if v >= 0 => v as u64,
                    _ => 0,
                };
                found_slots = Some(s);
                break;
            }
        }
        let Some(s) = found_slots else {
            return Err(format!("unable to find context in model @{name}"));
        };
        base_slots.insert(name.clone(), s);

        let mut edges = Vec::new();
        for &oid in &module.blocks[exec_blk].ops {
            let Some(eop) = module.ops.get(oid) else {
                continue;
            };
            if eop.name.full_name() == "mgp.call" {
                let callee = match eop
                    .attributes
                    .get("callee")
                    .or_else(|| eop.attributes.get("symbol"))
                {
                    Some(Attribute::String(c)) | Some(Attribute::SymbolRef(c)) => {
                        c.trim_start_matches('@').to_string()
                    }
                    _ => return Err("mgp.call missing 'callee' attribute".to_string()),
                };
                if !models_by_name.contains_key(&callee) {
                    return Err(format!("unknown callee @{callee} in model @{name}"));
                }
                edges.push((oid, callee));
            }
        }
        call_edges.insert(name.clone(), edges);
    }

    // Topological post-order DFS.
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: Vec<String> = Vec::new();
    let mut post_order: Vec<String> = Vec::new();

    fn dfs(
        node: &str,
        call_edges: &BTreeMap<String, Vec<(OpId, String)>>,
        visited: &mut HashSet<String>,
        on_stack: &mut Vec<String>,
        post_order: &mut Vec<String>,
    ) -> Result<(), String> {
        if on_stack.iter().any(|n| n == node) {
            return Err(format!("cycle detected in MGP call graph at @{node}"));
        }
        if visited.contains(node) {
            return Ok(());
        }
        on_stack.push(node.to_string());
        if let Some(edges) = call_edges.get(node) {
            for (_, callee) in edges {
                dfs(callee, call_edges, visited, on_stack, post_order)?;
            }
        }
        on_stack.pop();
        visited.insert(node.to_string());
        post_order.push(node.to_string());
        Ok(())
    }

    for name in models_by_name.keys() {
        dfs(
            name,
            &call_edges,
            &mut visited,
            &mut on_stack,
            &mut post_order,
        )?;
    }

    let mut total_globalized = 0usize;
    let mut total_slots_by_model: HashMap<String, u64> = base_slots;

    for model_name in &post_order {
        let edges = call_edges.get(model_name).cloned().unwrap_or_default();
        if edges.is_empty() {
            continue;
        }
        let exec_blk = exec_block_of[model_name];
        let mut model_slot_usage = total_slots_by_model.get(model_name).copied().unwrap_or(0);

        let orig_ops = module.blocks[exec_blk].ops.clone();
        let mut new_ops = Vec::with_capacity(orig_ops.len() + edges.len());
        let edge_map: HashMap<OpId, String> = edges.into_iter().collect();

        for oid in orig_ops {
            if let Some(callee) = edge_map.get(&oid) {
                let mut cst_attrs = IndexMap::new();
                cst_attrs.insert("value".to_string(), Attribute::Int(model_slot_usage as i64));
                let cst_id = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: make_op_name("index.constant"),
                    operands: SmallVec::new(),
                    results: SmallVec::new(),
                    attributes: cst_attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                let cst_val = module.values.insert_with_key(|vid| Value {
                    id: vid,
                    ssa_name: None,
                    ty: Type::Scalar(DType::Index),
                    def: ValueDef::OpResult {
                        op: cst_id,
                        index: 0,
                    },
                });
                module.ops[cst_id].results.push(cst_val);
                new_ops.push(cst_id);

                if let Some(call_op) = module.ops.get_mut(oid) {
                    call_op.operands.push(cst_val);
                    call_op
                        .attributes
                        .insert("is_globalized_index".to_string(), Attribute::Bool(true));
                }
                let callee_slots = total_slots_by_model.get(callee).copied().unwrap_or(0);
                model_slot_usage += callee_slots;
                total_globalized += 1;
            }
            new_ops.push(oid);
        }

        module.blocks[exec_blk].ops = new_ops;
        total_slots_by_model.insert(model_name.clone(), model_slot_usage);
    }

    Ok(total_globalized)
}

// ---------------------------------------------------------------------------
// 2. Device-Graph Kernel Registration Verification
// ---------------------------------------------------------------------------

/// Verifies all `mo.graph` operations with `is_device_graph = true` in `module`:
/// - Rejects control-flow operations (`mo.if`, `mo.while`, `scf.if`, `scf.while`)
///   inside device-graphable graphs.
/// - Checks every operation whose registration name (or `symbol` for `mo.custom`)
///   matches a top-level `mogg.kernel.decl` and verifies that the resolved
///   declaration exposes `build_graph = true` (or a non-empty `build_graph`
///   symbol string).
///
/// # Errors
///
/// Returns `Err` on the first control-flow operation or kernel lacking a
/// `build_graph` entry point inside an `is_device_graph = true` graph.
pub fn verify_device_graph_kernel_registrations(module: &Module) -> Result<usize, String> {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();

    // Collect registered `mogg.kernel.decl` operations by registration name.
    // Maps registration -> (sym_name, has_build_graph).
    let mut kernel_registry: HashMap<String, (String, bool)> = HashMap::new();
    for &oid in &top_ops {
        let Some(op) = module.ops.get(oid) else {
            continue;
        };
        if op.name.full_name() != "mogg.kernel.decl" {
            continue;
        }
        let sym_name = op
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
            .unwrap_or("unknown_kernel")
            .to_string();
        let registration = op
            .attributes
            .get("registration")
            .or_else(|| op.attributes.get("op_name"))
            .and_then(Attribute::as_str)
            .unwrap_or(&sym_name)
            .to_string();
        let has_build_graph = match op.attributes.get("build_graph") {
            Some(Attribute::Bool(b)) => *b,
            Some(Attribute::String(s)) => !s.is_empty(),
            Some(Attribute::SymbolRef(s)) => !s.is_empty(),
            _ => false,
        };
        kernel_registry.insert(registration, (sym_name, has_build_graph));
    }

    let mut verified_graphs = 0usize;
    for &oid in &top_ops {
        let Some(graph_op) = module.ops.get(oid) else {
            continue;
        };
        if graph_op.name.full_name() != "mo.graph" {
            continue;
        }
        let is_device_graph = matches!(
            graph_op
                .attributes
                .get("is_device_graph")
                .or_else(|| graph_op.attributes.get("isDeviceGraph")),
            Some(Attribute::Bool(true))
        );
        if !is_device_graph {
            continue;
        }

        for &reg_id in &graph_op.regions {
            let Some(reg) = module.regions.get(reg_id) else {
                continue;
            };
            for &blk_id in &reg.blocks {
                let Some(blk) = module.blocks.get(blk_id) else {
                    continue;
                };
                for &child_id in &blk.ops {
                    let Some(child_op) = module.ops.get(child_id) else {
                        continue;
                    };
                    let full = child_op.name.full_name();
                    if is_terminator_name(&full) {
                        continue;
                    }
                    if is_control_flow_op_name(&full) {
                        return Err(format!(
                            "device-graphable mo.graph cannot contain control-flow op \
                             '{full}' (control flow is not yet supported)"
                        ));
                    }
                    let registration = if full == "mo.custom" {
                        child_op
                            .attributes
                            .get("symbol")
                            .and_then(Attribute::as_str)
                            .unwrap_or(&full)
                    } else {
                        &full
                    };
                    if let Some((kernel_sym, has_build_graph)) = kernel_registry.get(registration)
                        && !has_build_graph
                    {
                        return Err(format!(
                            "op '{full}' is in a device-graphable mo.graph but its kernel \
                             registration ('{registration}', kernel symbol '{kernel_sym}') \
                             does not expose a `build_graph` entry point"
                        ));
                    }
                }
            }
        }
        verified_graphs += 1;
    }

    Ok(verified_graphs)
}

// ---------------------------------------------------------------------------
// 3. Single-Caller Index-Constant Raising
// ---------------------------------------------------------------------------

/// Extracts an `i64` constant from a resource entry (`Attribute::Int` or a
/// 1-element `Attribute::DenseIntArray` payload).
fn decode_i64_resource_entry(attr: &Attribute) -> Option<i64> {
    match attr {
        Attribute::Int(v) => Some(*v),
        Attribute::DenseIntArray(vals) if vals.len() == 1 => Some(vals[0]),
        _ => None,
    }
}

/// Raises `mgp.buffer.to_index(mgp.resource.constant)` inside single-caller
/// `abi = "mgp"` kernel stubs (`mogg.func` / `func.func`) back to
/// `index.constant` operations by reading the 64-bit integer value from the
/// sole caller's `constant_resources` array at `resource_index` /
/// `resource_slot`.
///
/// Multi-caller stubs are skipped because their resource arrays may differ per
/// caller.
pub fn raise_back_single_caller_index_constants(module: &mut Module) -> usize {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();

    // Map callee symbol -> Vec<call_op_id> across all `mgp.model` operations.
    let mut symbol_uses: HashMap<String, Vec<OpId>> = HashMap::new();
    for &oid in &module.live_ops() {
        let Some(op) = module.ops.get(oid) else {
            continue;
        };
        let full = op.name.full_name();
        if full != "mogg.compiled_call" && full != "castforge.compiled_call" {
            continue;
        }
        if let Some(callee) = op
            .attributes
            .get("callee_symbol")
            .or_else(|| op.attributes.get("callee"))
            .and_then(Attribute::as_str)
        {
            symbol_uses
                .entry(callee.trim_start_matches('@').to_string())
                .or_default()
                .push(oid);
        }
    }

    let mut raised_count = 0usize;

    for &stub_id in &top_ops {
        let Some(stub_op) = module.ops.get(stub_id) else {
            continue;
        };
        let full = stub_op.name.full_name();
        if full != "mogg.func" && full != "func.func" {
            continue;
        }
        let abi_is_mgp = stub_op
            .attributes
            .get("abi")
            .and_then(Attribute::as_str)
            .is_none_or(|a| a.eq_ignore_ascii_case("mgp"));
        if !abi_is_mgp {
            continue;
        }
        let Some(sym_name) = stub_op
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
            .map(str::to_string)
        else {
            continue;
        };
        let Some(callers) = symbol_uses.get(&sym_name) else {
            continue;
        };
        if callers.len() != 1 {
            continue;
        }
        let caller_id = callers[0];
        let Some(Attribute::Array(resource_arr)) = module
            .ops
            .get(caller_id)
            .and_then(|cop| cop.attributes.get("constant_resources"))
            .cloned()
        else {
            continue;
        };

        let regions = stub_op.regions.clone();
        for reg_id in regions {
            let blocks = module
                .regions
                .get(reg_id)
                .map(|r| r.blocks.clone())
                .unwrap_or_default();
            for blk_id in blocks {
                let child_ops = module
                    .blocks
                    .get(blk_id)
                    .map(|b| b.ops.clone())
                    .unwrap_or_default();
                for child_id in child_ops {
                    let Some(child_op) = module.ops.get(child_id) else {
                        continue;
                    };
                    if child_op.name.full_name() != "mgp.buffer.to_index"
                        || child_op.operands.is_empty()
                    {
                        continue;
                    }
                    let in_buf = child_op.operands[0];
                    let Some(ValueDef::OpResult { op: res_cst_id, .. }) =
                        module.values.get(in_buf).map(|v| v.def)
                    else {
                        continue;
                    };
                    let Some(res_cst_op) = module.ops.get(res_cst_id) else {
                        continue;
                    };
                    if res_cst_op.name.full_name() != "mgp.resource.constant" {
                        continue;
                    }
                    let Some(&Attribute::Int(slot_idx)) = res_cst_op
                        .attributes
                        .get("resource_slot")
                        .or_else(|| res_cst_op.attributes.get("resource_index"))
                    else {
                        continue;
                    };
                    if slot_idx < 0 || (slot_idx as usize) >= resource_arr.len() {
                        continue;
                    }
                    let Some(int_val) = decode_i64_resource_entry(&resource_arr[slot_idx as usize])
                    else {
                        continue;
                    };

                    if let Some(op_mut) = module.ops.get_mut(child_id) {
                        op_mut.name = make_op_name("index.constant");
                        op_mut.operands.clear();
                        op_mut.attributes.clear();
                        op_mut
                            .attributes
                            .insert("value".to_string(), Attribute::Int(int_val));
                        raised_count += 1;
                    }
                }
            }
        }
    }

    raised_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_i64_resource_entry() {
        assert_eq!(decode_i64_resource_entry(&Attribute::Int(128)), Some(128));
        assert_eq!(
            decode_i64_resource_entry(&Attribute::DenseIntArray(vec![256])),
            Some(256)
        );
        assert_eq!(
            decode_i64_resource_entry(&Attribute::DenseIntArray(vec![1, 2, 3])),
            None
        );
    }
}
