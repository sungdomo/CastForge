//! Routine outlining, kernel stub merging, call operand conformance, and module content hashing.

use super::*;
use crate::ir::{Attribute, Location, Module, OpId, Operation, Type, Value, ValueDef, ValueId};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Canonicalizes a device reference string (such as `"gpu:2"` or `"#device_ref<\"gpu\", 3>"`)
/// by preserving the device label and resetting its ordinal ID to `0`.
pub fn canonicalize_device_ref_str(s: &str) -> Option<String> {
    if let Some((label, id_str)) = s.split_once(':')
        && !label.is_empty()
        && id_str.chars().all(|c| c.is_ascii_digit())
    {
        let canonical = format!("{label}:0");
        if canonical != s {
            return Some(canonical);
        }
    }
    None
}

fn canonicalize_type_device_refs(ty: &mut Type) -> usize {
    match ty {
        Type::Tensor(tt) => {
            if let Some(enc) = &tt.encoding
                && let Some(canon) = canonicalize_device_ref_str(enc)
            {
                tt.encoding = Some(canon);
                return 1;
            }
            0
        }
        Type::Tuple(items) => items.iter_mut().map(canonicalize_type_device_refs).sum(),
        Type::Function { inputs, results } => {
            let a: usize = inputs.iter_mut().map(canonicalize_type_device_refs).sum();
            let b: usize = results.iter_mut().map(canonicalize_type_device_refs).sum();
            a + b
        }
        _ => 0,
    }
}

fn canonicalize_attr_for_merge(key: &str, attr: &mut Attribute) -> usize {
    match attr {
        Attribute::String(s) if key == "device_ref" || key == "mo.device_ref" => {
            if let Some(canon) = canonicalize_device_ref_str(s) {
                *s = canon;
                return 1;
            }
            0
        }
        Attribute::Type(ty) => canonicalize_type_device_refs(ty),
        Attribute::Array(items) if key == "shape_dims" || key == "symbolic_shape" => {
            let mut changed = 0usize;
            for item in items.iter_mut() {
                if let Attribute::String(dim_name) = item
                    && !dim_name.starts_with("simd:")
                    && dim_name != "dyn"
                {
                    *dim_name = "dyn".to_string();
                    changed += 1;
                }
            }
            changed
        }
        _ => 0,
    }
}

/// Canonicalizes top-level functions prior to structural kernel deduplication:
/// 1. Normalizes device reference ordinal IDs to `0` (`"gpu:3"` -> `"gpu:0"`) in `"device_ref"` /
///    `"mo.device_ref"` attributes, tensor encodings, and function signature types so kernels that
///    differ only by target device ordinal can merge.
/// 2. Normalizes non-SIMD symbolic dimension names in `"shape_dims"` / `"symbolic_shape"` array
///    attributes to `"dyn"` while preserving `"simd:..."` entries and static integer dimensions.
///
/// Returns the total number of attribute and type sub-elements canonicalized.
pub fn canonicalize_funcs_for_merge(module: &mut Module) -> usize {
    let mut count = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let keys: Vec<String> = module.ops[func_id].attributes.keys().cloned().collect();
        for k in keys {
            if let Some(attr) = module.ops[func_id].attributes.get_mut(&k) {
                count += canonicalize_attr_for_merge(&k, attr);
            }
        }

        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let args = module.blocks[blk_id].arguments.clone();
                for arg_val in args {
                    count += canonicalize_type_device_refs(&mut module.values[arg_val].ty);
                }
                let ops = module.blocks[blk_id].ops.clone();
                for op_id in ops {
                    let op_keys: Vec<String> =
                        module.ops[op_id].attributes.keys().cloned().collect();
                    for k in op_keys {
                        if let Some(attr) = module.ops[op_id].attributes.get_mut(&k) {
                            count += canonicalize_attr_for_merge(&k, attr);
                        }
                    }
                    let res_vals: Vec<ValueId> =
                        module.ops[op_id].results.iter().copied().collect();
                    for rv in res_vals {
                        count += canonicalize_type_device_refs(&mut module.values[rv].ty);
                    }
                }
            }
        }
    }

    count
}

/// Performs bottom-up call-graph kernel stub deduplication across the module:
/// - Compares functions modulo `"sym_name"`, `"fusion_info"`, and device-ref ordinal IDs,
/// - Rewrites intra-kernel call edges (`func.call`, `castforge.call`, `castforge.call.async`,
///   `mogg.call`, `mogg.call.async`) bottom-up so caller hashes see canonicalized callees,
/// - Selects the lexicographically smallest symbol name as the deterministic canonical stub for
///   each equivalence class, fuses source locations, and rewrites external
///   `castforge.compiled_call` / `mogg.compiled_call` ops before erasing duplicate definitions.
pub fn merge_identical_kernel_stubs(module: &mut Module) -> usize {
    use std::collections::{HashMap, HashSet};

    // Fixed-point bottom-up pass: each round canonicalizes leaf/inner callees and rewrites callers
    // until no further equivalence classes collapse.
    let mut total_merged = 0usize;

    loop {
        let fns = module.functions();
        if fns.len() <= 1 {
            break;
        }

        type StubEntry = (OpId, String, Location);
        let mut groups: IndexMap<(String, String, u64), Vec<StubEntry>> = IndexMap::new();

        for &fn_id in &fns {
            let Some(op) = module.ops.get(fn_id) else {
                continue;
            };
            let Some(sym_name) = op
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .map(str::to_string)
            else {
                continue;
            };
            let Some(&reg_id) = op.regions.first() else {
                continue;
            };
            let sig_str = op
                .attributes
                .get("function_type")
                .map(|a| a.to_string())
                .unwrap_or_default();

            // Build discardable attribute signature modulo `sym_name`, `fusion_info`, and device ID
            let mut attr_parts: Vec<String> = Vec::new();
            for (k, v) in &op.attributes {
                if k == "sym_name" || k == "fusion_info" || k == "function_type" {
                    continue;
                }
                if (k == "device_ref" || k == "mo.device_ref")
                    && let Some(s) = v.as_str()
                {
                    let label = s.split_once(':').map(|(l, _)| l).unwrap_or(s);
                    attr_parts.push(format!("{k}={label}"));
                    continue;
                }
                attr_parts.push(format!("{k}={v}"));
            }
            attr_parts.sort();
            let attrs_sig = attr_parts.join(";");

            let body_hash = crate::codegen::compute_region_structural_hash(module, reg_id);
            groups
                .entry((sig_str, attrs_sig, body_hash))
                .or_default()
                .push((fn_id, sym_name, op.location.clone()));
        }

        let mut replacements: HashMap<String, String> = HashMap::new();
        let mut dead_fn_ops: HashSet<OpId> = HashSet::new();

        for (_, mut candidates) in groups {
            if candidates.len() <= 1 {
                continue;
            }
            candidates.sort_by(|a, b| a.1.cmp(&b.1));
            let (canon_id, canon_name, _) = candidates[0].clone();
            let all_locs: Vec<Location> =
                candidates.iter().map(|(_, _, loc)| loc.clone()).collect();
            if let Some(canon_op) = module.ops.get_mut(canon_id) {
                stamp_fused_provenance(canon_op, &all_locs);
            }
            for (dup_id, dup_name, _) in candidates.into_iter().skip(1) {
                replacements.insert(dup_name, canon_name.clone());
                dead_fn_ops.insert(dup_id);
            }
        }

        if dead_fn_ops.is_empty() {
            break;
        }

        // Rewrite all call and compiled_call references across the module.
        for (_, op) in module.ops.iter_mut() {
            let full = op.name.full_name();
            let is_call = matches!(
                full.as_str(),
                "func.call"
                    | "castforge.call"
                    | "castforge.call.async"
                    | "mogg.call"
                    | "mogg.call.async"
                    | "castforge.compiled_call"
                    | "mogg.compiled_call"
            );
            if !is_call {
                continue;
            }
            for attr_key in ["callee", "callee_symbol"] {
                if let Some(callee) = op
                    .attributes
                    .get(attr_key)
                    .and_then(|a| a.as_str())
                    .map(str::to_string)
                    && let Some(new_callee) = replacements.get(&callee)
                {
                    op.attributes.insert(
                        attr_key.to_string(),
                        Attribute::SymbolRef(new_callee.clone()),
                    );
                }
            }
        }

        if let Some(root_reg) = module.regions.get(module.root_region)
            && let Some(&root_blk) = root_reg.blocks.first()
        {
            module.blocks[root_blk]
                .ops
                .retain(|id| !dead_fn_ops.contains(id));
        }
        total_merged += dead_fn_ops.len();
        for dead_id in dead_fn_ops {
            module.ops.remove(dead_id);
        }
    }

    total_merged
}

/// Lowers compiled JIT routines inside top-level functions:
/// 1. Rewrites `castforge.compiled_load` (or `mogg.compiled_load`) into `castforge.module.load`
///    chained after the preceding `castforge.runtime.create` (or `mgp.runtime.create`), assigning
///    a sequential `module_index` per loaded archive and indexing each symbol in its `"symbols"`
///    array as `(module_index, kernel_index)`.
/// 2. Replaces `castforge.module.load.placeholder` (or `mgp.module.load.placeholder`) with the
///    single module-load chain or a newly inserted `castforge.chain.create` over all runtime-create
///    and module-load chains.
/// 3. Rewrites `castforge.compiled_call` (or `mogg.compiled_call`) into
///    `castforge.generic.execute` with resolved `"runtime_slot"`, `"module_index"`,
///    `"kernel_index"`, and `"is_async"` (`"dispatch" == "deferred"`).
/// 4. For any `castforge.generic.execute` in an init function (`"stage" == "init"` or function name
///    ending in `"_init"`) that does not already consume a `!chain` operand, inserts a
///    `castforge.guard(module_load_chain, context)` before the execute so init kernels cannot run
///    before their backing module archive is loaded.
/// 5. Elides identity `castforge.to_graph_from_opaque` (`mogg.to_mgp_from_opaque`) conversions.
pub fn lower_compiled_routines(module: &mut Module) -> Result<usize, String> {
    use crate::ir::mlir::OpName;
    use std::collections::HashMap;

    let mut lowered_count = 0usize;
    let func_ids = module.functions();

    // Shared `(module_index, kernel_index)` table across init and execute functions in the module,
    // plus `runtime_slot` discovered from `runtime.create`.
    let mut kernel_table: HashMap<String, (i64, i64)> = HashMap::new();
    let mut default_runtime_slot: i64 = 0;

    for func_id in func_ids {
        let sym_name = module.ops[func_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("")
            .to_string();
        let is_init_fn = module.ops[func_id]
            .attributes
            .get("stage")
            .and_then(|a| a.as_str())
            .map(|s| s == "init")
            .unwrap_or_else(|| sym_name.ends_with("_init") || sym_name == "init");

        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let reg_id = module.ops[func_id].regions[0];
        if module.regions[reg_id].blocks.is_empty() {
            continue;
        }
        let blk_id = module.regions[reg_id].blocks[0];
        let orig_ops = module.blocks[blk_id].ops.clone();

        let mut module_load_chains: Vec<ValueId> = Vec::new();
        let mut module_load_result_by_index: Vec<ValueId> = Vec::new();
        let mut last_runtime_slot: i64 = default_runtime_slot;
        let mut last_runtime_ctx: Option<ValueId> =
            module.blocks[blk_id].arguments.first().copied();
        let mut last_runtime_chain: Option<ValueId> = None;

        for &op_id in &orig_ops {
            let full = module.ops[op_id].name.full_name();
            if full == "castforge.runtime.create" || full == "mgp.runtime.create" {
                if let Some(Attribute::Int(s)) = module.ops[op_id].attributes.get("slot") {
                    last_runtime_slot = *s;
                    default_runtime_slot = *s;
                }
                if let Some(&ctx_v) = module.ops[op_id].operands.first() {
                    last_runtime_ctx = Some(ctx_v);
                }
                if let Some(&res_v) = module.ops[op_id].results.first() {
                    module_load_chains.push(res_v);
                    last_runtime_chain = Some(res_v);
                }
            }
        }

        let mut device_module_index: i64 = 0;
        for &op_id in &orig_ops {
            let full = module.ops[op_id].name.full_name();
            if full != "castforge.compiled_load" && full != "mogg.compiled_load" {
                continue;
            }
            let symbols = match module.ops[op_id].attributes.get("symbols") {
                Some(Attribute::Array(arr)) => arr.clone(),
                _ => Vec::new(),
            };
            let mut kernel_names: Vec<Attribute> = Vec::new();
            for (k_idx, sym_attr) in symbols.iter().enumerate() {
                if let Some(s) = sym_attr.as_str() {
                    kernel_table.insert(s.to_string(), (device_module_index, k_idx as i64));
                    kernel_names.push(Attribute::String(s.to_string()));
                }
            }

            module.ops[op_id].name = OpName::new("castforge", "module.load");
            let mut new_operands: SmallVec<[ValueId; 4]> = SmallVec::new();
            if let Some(ch) = last_runtime_chain {
                new_operands.push(ch);
            }
            if let Some(ctx) = last_runtime_ctx {
                new_operands.push(ctx);
            }
            if !new_operands.is_empty() {
                module.ops[op_id].operands = new_operands;
            }
            module.ops[op_id]
                .attributes
                .insert("slot".to_string(), Attribute::Int(last_runtime_slot));
            module.ops[op_id]
                .attributes
                .insert("kernel_names".to_string(), Attribute::Array(kernel_names));
            module.ops[op_id].attributes.insert(
                "device_ref".to_string(),
                Attribute::String("cpu:0".to_string()),
            );
            module.ops[op_id].attributes.insert(
                "module_index".to_string(),
                Attribute::Int(device_module_index),
            );

            if let Some(&res_v) = module.ops[op_id].results.first() {
                module_load_chains.push(res_v);
                module_load_result_by_index.push(res_v);
                last_runtime_chain = Some(res_v);
            }
            device_module_index += 1;
            lowered_count += 1;
        }

        // Replace `module.load.placeholder` and rewrite `compiled_call` / `to_graph_from_opaque`.
        let mut new_block_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len() + 4);
        let mut value_replacements: HashMap<ValueId, ValueId> = HashMap::new();

        for &op_id in &orig_ops {
            // Apply any accumulated SSA value replacements to this op's operands first.
            for opd in module.ops[op_id].operands.iter_mut() {
                if let Some(&rep) = value_replacements.get(opd) {
                    *opd = rep;
                }
            }

            let full = module.ops[op_id].name.full_name();
            if (full == "castforge.module.load.placeholder"
                || full == "mgp.module.load.placeholder")
                && !module_load_chains.is_empty()
            {
                let ph_res = module.ops[op_id].results[0];
                if module_load_chains.len() == 1 {
                    value_replacements.insert(ph_res, module_load_chains[0]);
                } else {
                    module.ops[op_id].name = OpName::new("castforge", "chain.create");
                    module.ops[op_id].operands = SmallVec::from_vec(module_load_chains.clone());
                    new_block_ops.push(op_id);
                }
                lowered_count += 1;
                continue;
            }

            if (full == "castforge.to_graph_from_opaque" || full == "mogg.to_mgp_from_opaque")
                && !module.ops[op_id].operands.is_empty()
                && !module.ops[op_id].results.is_empty()
            {
                let inp = module.ops[op_id].operands[0];
                let out = module.ops[op_id].results[0];
                value_replacements.insert(out, inp);
                lowered_count += 1;
                continue;
            }

            if full == "castforge.compiled_call" || full == "mogg.compiled_call" {
                let callee = module.ops[op_id]
                    .attributes
                    .get("callee_symbol")
                    .or_else(|| module.ops[op_id].attributes.get("callee"))
                    .and_then(|a| a.as_str())
                    .ok_or_else(|| "compiled_call missing 'callee_symbol' attribute".to_string())?
                    .to_string();
                let &(mod_idx, kern_idx) = kernel_table.get(&callee).ok_or_else(|| {
                    format!("compiled kernel '{callee}' is not present in any loaded module")
                })?;

                let is_async = module.ops[op_id]
                    .attributes
                    .get("dispatch")
                    .and_then(|a| a.as_str())
                    .map(|d| d == "deferred")
                    .unwrap_or(false);

                // Check if any operand is already a chain before rewriting.
                let has_chain_operand = module.ops[op_id].operands.iter().any(|&v| {
                    matches!(
                        &module.values[v].ty,
                        Type::Opaque(s) if s.contains("chain")
                    )
                });

                let mut ctx_val = last_runtime_ctx;
                if is_init_fn
                    && !has_chain_operand
                    && (mod_idx as usize) < module_load_result_by_index.len()
                    && let Some(raw_ctx) = ctx_val
                {
                    let load_chain = module_load_result_by_index[mod_idx as usize];
                    let ctx_ty = module.values[raw_ctx].ty.clone();
                    let loc = module.ops[op_id].location.clone();
                    let guard_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("castforge", "guard"),
                        operands: SmallVec::from_slice(&[load_chain, raw_ctx]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: loc,
                    });
                    let guarded_ctx = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: ctx_ty,
                        def: ValueDef::OpResult {
                            op: guard_id,
                            index: 0,
                        },
                    });
                    module.ops[guard_id].results.push(guarded_ctx);
                    new_block_ops.push(guard_id);
                    ctx_val = Some(guarded_ctx);
                }

                if let Some(c) = ctx_val
                    && module.ops[op_id].operands.first().copied() != Some(c)
                {
                    module.ops[op_id].operands.insert(0, c);
                }
                module.ops[op_id].name = OpName::new("castforge", "generic.execute");
                module.ops[op_id].attributes.insert(
                    "runtime_slot".to_string(),
                    Attribute::Int(last_runtime_slot),
                );
                module.ops[op_id]
                    .attributes
                    .insert("module_index".to_string(), Attribute::Int(mod_idx));
                module.ops[op_id]
                    .attributes
                    .insert("kernel_index".to_string(), Attribute::Int(kern_idx));
                module.ops[op_id]
                    .attributes
                    .insert("is_async".to_string(), Attribute::Bool(is_async));

                // Convert any `!mo.opaque<...>` result types to `!castforge.opaque<...>`.
                let res_ids: Vec<ValueId> = module.ops[op_id].results.iter().copied().collect();
                for rv in res_ids {
                    if let Type::Opaque(ref mut s) = module.values[rv].ty
                        && let Some(rest) = s.strip_prefix("!mo.opaque")
                    {
                        *s = format!("!castforge.opaque{rest}");
                    }
                }

                new_block_ops.push(op_id);
                lowered_count += 1;
                continue;
            }

            new_block_ops.push(op_id);
        }

        module.blocks[blk_id].ops = new_block_ops;
    }

    Ok(lowered_count)
}

/// Outlines `castforge.model` (or `mgp.model`) ops into pairs of `mef.callable` (`<name>$init` and
/// `<name>$exec`) in post-order over the subgraph call graph:
/// - Rejects cycles in the subgraph call graph with an explicit error.
/// - For subgraph models (`"is_subgraph" == true`), appends a runtime string prefix argument to the
///   init block, inserts a prefix argument into the execute block after `"implicit_args_count"`,
///   concatenates the prefix onto placeholder `mgp.buffer.constant_external` (or
///   `castforge.buffer.constant_external`) weight names via `mgp.string.concat`, and appends the
///   prefix to the init terminator outputs so the execute stage receives it through the forwarded
///   bundle.
/// - Rewrites `mgp.call` (or `castforge.call_subgraph`) ops by synthesizing `mgp.string.create`
///   (and `mgp.string.concat` when nested inside a subgraph), invoking `<callee>$init` via
///   `grt.invoke` in the parent init block, threading the callee's init results into the parent
///   init output and parent execute arguments, and replacing the call with `grt.invoke` to
///   `<callee>$exec`.
/// - Wraps non-subgraph entry models in a `mef.model` containing forwarding `grt.invoke` +
///   `mef.output` blocks for init and execute.
///
/// Returns the number of `model` ops outlined.
pub fn outline_models_to_callables(module: &mut Module) -> Result<usize, String> {
    use crate::ir::mlir::{Block, OpName, Region};
    use std::collections::{HashMap, HashSet};

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return Ok(0);
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return Ok(0);
    };
    let root_ops = module.blocks[root_blk].ops.clone();

    let mut model_by_name: IndexMap<String, OpId> = IndexMap::new();
    let mut entry_models: Vec<String> = Vec::new();

    for &op_id in &root_ops {
        let full = module.ops[op_id].name.full_name();
        if full != "castforge.model" && full != "mgp.model" {
            continue;
        }
        let name = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .ok_or_else(|| "model op is missing 'sym_name' attribute".to_string())?
            .to_string();
        if model_by_name.contains_key(&name) {
            return Err(format!(
                "two models with the same symbol name existed: {name}"
            ));
        }
        let is_subgraph = module.ops[op_id]
            .attributes
            .get("is_subgraph")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);
        if !is_subgraph {
            entry_models.push(name.clone());
        }
        model_by_name.insert(name, op_id);
    }

    if model_by_name.is_empty() {
        return Ok(0);
    }

    // Build call-graph edges (`mgp.call` / `castforge.call_subgraph` inside execute region) and
    // verify acyclic post-order traversal from entry models.
    let mut callees_of: HashMap<String, Vec<(OpId, String, String)>> = HashMap::new();
    for (name, &model_id) in &model_by_name {
        if module.ops[model_id].regions.len() < 2 {
            return Err(format!("model '{name}' must have init and execute regions"));
        }
        let exec_reg = module.ops[model_id].regions[1];
        let exec_blk = module.regions[exec_reg].blocks[0];
        let mut calls = Vec::new();
        for &op_id in &module.blocks[exec_blk].ops {
            let full = module.ops[op_id].name.full_name();
            if full == "mgp.call" || full == "castforge.call_subgraph" {
                let callee = module.ops[op_id]
                    .attributes
                    .get("callee")
                    .and_then(|a| a.as_str())
                    .ok_or_else(|| "subgraph call missing 'callee' attribute".to_string())?
                    .trim_start_matches('@')
                    .to_string();
                if !model_by_name.contains_key(&callee) {
                    return Err(format!(
                        "CallOp's callee '{callee}' did not exist in module"
                    ));
                }
                let prefix = module.ops[op_id]
                    .attributes
                    .get("prefix")
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string();
                calls.push((op_id, callee, prefix));
            }
        }
        callees_of.insert(name.clone(), calls);
    }

    // DFS post-order with cycle detection.
    let mut post_order: Vec<String> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: HashSet<String> = HashSet::new();

    fn dfs_order(
        curr: &str,
        callees_of: &HashMap<String, Vec<(OpId, String, String)>>,
        visited: &mut HashSet<String>,
        on_stack: &mut HashSet<String>,
        post_order: &mut Vec<String>,
    ) -> Result<(), String> {
        if on_stack.contains(curr) {
            return Err(format!("cycle detected in subgraph call graph at '{curr}'"));
        }
        if visited.contains(curr) {
            return Ok(());
        }
        on_stack.insert(curr.to_string());
        if let Some(edges) = callees_of.get(curr) {
            for (_, callee, _) in edges {
                dfs_order(callee, callees_of, visited, on_stack, post_order)?;
            }
        }
        on_stack.remove(curr);
        visited.insert(curr.to_string());
        post_order.push(curr.to_string());
        Ok(())
    }

    for entry in &entry_models {
        dfs_order(
            entry,
            &callees_of,
            &mut visited,
            &mut on_stack,
            &mut post_order,
        )?;
    }
    for name in model_by_name.keys() {
        if !visited.contains(name) {
            dfs_order(
                name,
                &callees_of,
                &mut visited,
                &mut on_stack,
                &mut post_order,
            )?;
        }
    }

    let str_ty = Type::Opaque("!mgp.opaque<\"mojo_string\">".to_string());
    let mut exec_prefix_by_model: HashMap<String, ValueId> = HashMap::new();

    // Step 1: `localPropagatePrefixes` on all subgraphs before sealing.
    for (name, &model_id) in &model_by_name {
        let is_subgraph = module.ops[model_id]
            .attributes
            .get("is_subgraph")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);
        if !is_subgraph {
            continue;
        }
        let implicit_count = match module.ops[model_id].attributes.get("implicit_args_count") {
            Some(Attribute::Int(n)) if *n >= 0 => *n as usize,
            _ => 0,
        };
        let init_reg = module.ops[model_id].regions[0];
        let exec_reg = module.ops[model_id].regions[1];
        let init_blk = module.regions[init_reg].blocks[0];
        let exec_blk = module.regions[exec_reg].blocks[0];
        let loc = module.ops[model_id].location.clone();

        let init_arg_idx = module.blocks[init_blk].arguments.len() as u32;
        let init_prefix = module.values.insert_with_key(|vid| Value {
            id: vid,
            ssa_name: None,
            ty: str_ty.clone(),
            def: ValueDef::BlockArgument {
                block: init_blk,
                index: init_arg_idx,
            },
        });
        module.blocks[init_blk].arguments.push(init_prefix);

        let ins_pos = implicit_count.min(module.blocks[exec_blk].arguments.len());
        let exec_prefix = module.values.insert_with_key(|vid| Value {
            id: vid,
            ssa_name: None,
            ty: str_ty.clone(),
            def: ValueDef::BlockArgument {
                block: exec_blk,
                index: ins_pos as u32,
            },
        });
        module.blocks[exec_blk]
            .arguments
            .insert(ins_pos, exec_prefix);
        exec_prefix_by_model.insert(name.clone(), exec_prefix);

        for (blk_id, p_val) in [(init_blk, init_prefix), (exec_blk, exec_prefix)] {
            let orig_ops = module.blocks[blk_id].ops.clone();
            let mut updated_ops = Vec::with_capacity(orig_ops.len() + 2);
            for op_id in orig_ops {
                let full = module.ops[op_id].name.full_name();
                let is_placeholder = module.ops[op_id]
                    .attributes
                    .get("e_is_placeholder")
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false);
                if (full == "mgp.buffer.constant_external"
                    || full == "castforge.buffer.constant_external")
                    && is_placeholder
                    && !module.ops[op_id].operands.is_empty()
                {
                    let raw_name = module.ops[op_id].operands[0];
                    let concat_loc = loc.clone();
                    let concat_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("mgp", "string.concat"),
                        operands: SmallVec::from_slice(&[p_val, raw_name]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: concat_loc,
                    });
                    let concat_res = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: str_ty.clone(),
                        def: ValueDef::OpResult {
                            op: concat_id,
                            index: 0,
                        },
                    });
                    module.ops[concat_id].results.push(concat_res);
                    module.ops[op_id].operands[0] = concat_res;
                    updated_ops.push(concat_id);
                }
                updated_ops.push(op_id);
            }
            module.blocks[blk_id].ops = updated_ops;
        }

        // Append `init_prefix` to the init terminator's operands so exec receives it.
        if let Some(&term_id) = module.blocks[init_blk].ops.last() {
            module.ops[term_id].operands.push(init_prefix);
        }
    }

    // Step 2: `rewriteAndSeal` in post-order.
    // Maps sealed model name -> (init_result_types, exec_result_types).
    type SealedSig = (Vec<Type>, Vec<Type>);
    let mut sealed_sigs: HashMap<String, SealedSig> = HashMap::new();
    let mut new_top_level_ops: Vec<OpId> = Vec::new();
    let mut erased_models: HashSet<OpId> = HashSet::new();

    for name in &post_order {
        let model_id = model_by_name[name];
        let loc = module.ops[model_id].location.clone();
        let is_subgraph = module.ops[model_id]
            .attributes
            .get("is_subgraph")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);
        let init_reg = module.ops[model_id].regions[0];
        let exec_reg = module.ops[model_id].regions[1];
        let init_blk = module.regions[init_reg].blocks[0];
        let exec_blk = module.regions[exec_reg].blocks[0];

        let calls = callees_of.get(name).cloned().unwrap_or_default();
        if !calls.is_empty() {
            let init_term = *module.blocks[init_blk]
                .ops
                .last()
                .ok_or_else(|| format!("model '{name}' init block has no terminator"))?;
            let init_args_snapshot = module.blocks[init_blk].arguments.clone();
            let mut init_chains: SmallVec<[ValueId; 4]> = SmallVec::new();
            let mut pre_term_ops: Vec<OpId> = Vec::new();

            // Per-call forwarded init results and exec bundle args.
            let mut call_init_results: Vec<Vec<ValueId>> = Vec::new();
            let mut call_init_types: Vec<Vec<Type>> = Vec::new();

            for (_call_op, callee_name, prefix_str) in &calls {
                let (callee_init_res_tys, _) = sealed_sigs
                    .get(callee_name)
                    .ok_or_else(|| format!("callee '{callee_name}' was not sealed before caller"))?
                    .clone();

                let str_create_loc = loc.clone();
                let str_create_id = module.ops.insert_with_key(|oid| Operation {
                    id: oid,
                    name: OpName::new("mgp", "string.create"),
                    operands: SmallVec::new(),
                    results: SmallVec::new(),
                    attributes: {
                        let mut m = IndexMap::new();
                        m.insert("value".to_string(), Attribute::String(prefix_str.clone()));
                        m
                    },
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: str_create_loc,
                });
                let mut prefix_val = module.values.insert_with_key(|vid| Value {
                    id: vid,
                    ssa_name: None,
                    ty: str_ty.clone(),
                    def: ValueDef::OpResult {
                        op: str_create_id,
                        index: 0,
                    },
                });
                module.ops[str_create_id].results.push(prefix_val);
                pre_term_ops.push(str_create_id);

                if let Some(&parent_prefix) = exec_prefix_by_model.get(name) {
                    let concat_loc = loc.clone();
                    let concat_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("mgp", "string.concat"),
                        operands: SmallVec::from_slice(&[parent_prefix, prefix_val]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: concat_loc,
                    });
                    let concat_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: str_ty.clone(),
                        def: ValueDef::OpResult {
                            op: concat_id,
                            index: 0,
                        },
                    });
                    module.ops[concat_id].results.push(concat_val);
                    pre_term_ops.push(concat_id);
                    prefix_val = concat_val;
                }

                let mut invoke_init_args: SmallVec<[ValueId; 4]> =
                    SmallVec::from_slice(&init_args_snapshot);
                invoke_init_args.push(prefix_val);
                if let Some(&first_arg) = init_args_snapshot.first() {
                    init_chains.push(first_arg);
                }

                let invoke_loc = loc.clone();
                let init_callee_sym = format!("{callee_name}$init");
                let invoke_init_id = module.ops.insert_with_key(|oid| Operation {
                    id: oid,
                    name: OpName::new("grt", "invoke"),
                    operands: invoke_init_args,
                    results: SmallVec::new(),
                    attributes: {
                        let mut m = IndexMap::new();
                        m.insert("callee".to_string(), Attribute::SymbolRef(init_callee_sym));
                        m
                    },
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: invoke_loc,
                });
                let mut results_vec = Vec::new();
                for (r_idx, r_ty) in callee_init_res_tys.iter().enumerate() {
                    let rv = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: r_ty.clone(),
                        def: ValueDef::OpResult {
                            op: invoke_init_id,
                            index: r_idx as u32,
                        },
                    });
                    module.ops[invoke_init_id].results.push(rv);
                    results_vec.push(rv);
                }
                pre_term_ops.push(invoke_init_id);
                call_init_results.push(results_vec);
                call_init_types.push(callee_init_res_tys);
            }

            // Stack child init outputs onto parent init terminator and insert corresponding exec
            // block arguments.
            let mut exec_bundles: Vec<Vec<ValueId>> = Vec::new();
            for (res_vals, res_tys) in call_init_results.iter().zip(call_init_types.iter()) {
                let ins_base = module.ops[init_term].operands.len();
                for &rv in res_vals {
                    module.ops[init_term].operands.push(rv);
                }
                let mut bundle = Vec::new();
                for (i, ty) in res_tys.iter().enumerate() {
                    let pos = (ins_base + i).min(module.blocks[exec_blk].arguments.len());
                    let exec_arg = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: ty.clone(),
                        def: ValueDef::BlockArgument {
                            block: exec_blk,
                            index: pos as u32,
                        },
                    });
                    module.blocks[exec_blk].arguments.insert(pos, exec_arg);
                    bundle.push(exec_arg);
                }
                exec_bundles.push(bundle);
            }

            // Replace each `mgp.call` in `exec_blk` with `grt.invoke` to `<callee>$exec`.
            for ((call_op, callee_name, _), mut bundle) in calls.into_iter().zip(exec_bundles) {
                let drop_count = bundle.len().saturating_sub(1);
                let orig_call_opds: Vec<ValueId> =
                    module.ops[call_op].operands.iter().copied().collect();
                let explicit_slice = if orig_call_opds.len() >= drop_count {
                    &orig_call_opds[drop_count..]
                } else {
                    &orig_call_opds[..]
                };
                bundle.extend_from_slice(explicit_slice);
                module.ops[call_op].name = OpName::new("grt", "invoke");
                module.ops[call_op].operands = SmallVec::from_vec(bundle);
                module.ops[call_op].attributes.insert(
                    "callee".to_string(),
                    Attribute::SymbolRef(format!("{callee_name}$exec")),
                );
            }

            // Join child init chains with parent init chain[0] via `grt.chain.create`.
            if !module.ops[init_term].operands.is_empty() {
                let parent_chain = module.ops[init_term].operands[0];
                init_chains.push(parent_chain);
                let chain_ty = module.values[parent_chain].ty.clone();
                let chain_loc = loc.clone();
                let chain_create_id = module.ops.insert_with_key(|oid| Operation {
                    id: oid,
                    name: OpName::new("grt", "chain.create"),
                    operands: init_chains,
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: chain_loc,
                });
                let new_chain = module.values.insert_with_key(|vid| Value {
                    id: vid,
                    ssa_name: None,
                    ty: chain_ty,
                    def: ValueDef::OpResult {
                        op: chain_create_id,
                        index: 0,
                    },
                });
                module.ops[chain_create_id].results.push(new_chain);
                module.ops[init_term].operands[0] = new_chain;
                pre_term_ops.push(chain_create_id);
            }

            let term_pos = module.blocks[init_blk].ops.len() - 1;
            module.blocks[init_blk]
                .ops
                .splice(term_pos..term_pos, pre_term_ops);
        }

        // Compute final init and exec signatures.
        let init_in_tys: Vec<Type> = module.blocks[init_blk]
            .arguments
            .iter()
            .map(|&v| module.values[v].ty.clone())
            .collect();
        let init_out_tys: Vec<Type> = module.blocks[init_blk]
            .ops
            .last()
            .map(|&tid| {
                module.ops[tid]
                    .operands
                    .iter()
                    .map(|&v| module.values[v].ty.clone())
                    .collect()
            })
            .unwrap_or_default();

        let exec_in_tys: Vec<Type> = module.blocks[exec_blk]
            .arguments
            .iter()
            .map(|&v| module.values[v].ty.clone())
            .collect();
        let exec_out_tys: Vec<Type> = module.blocks[exec_blk]
            .ops
            .last()
            .map(|&tid| {
                module.ops[tid]
                    .operands
                    .iter()
                    .map(|&v| module.values[v].ty.clone())
                    .collect()
            })
            .unwrap_or_default();

        sealed_sigs.insert(name.clone(), (init_out_tys.clone(), exec_out_tys.clone()));

        let init_fn_ty = Type::Function {
            inputs: init_in_tys.clone(),
            results: init_out_tys.clone(),
        };
        let exec_fn_ty = Type::Function {
            inputs: exec_in_tys.clone(),
            results: exec_out_tys.clone(),
        };

        let init_callable_loc = loc.clone();
        let init_sym = format!("{name}$init");
        let init_callable_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("mef", "callable"),
            operands: SmallVec::new(),
            results: SmallVec::new(),
            attributes: {
                let mut m = IndexMap::new();
                m.insert("sym_name".to_string(), Attribute::String(init_sym.clone()));
                m.insert(
                    "function_type".to_string(),
                    Attribute::Type(init_fn_ty.clone()),
                );
                m
            },
            regions: vec![init_reg],
            custom_syntax: false,
            location: init_callable_loc,
        });

        let exec_callable_loc = loc.clone();
        let exec_sym = format!("{name}$exec");
        let exec_callable_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("mef", "callable"),
            operands: SmallVec::new(),
            results: SmallVec::new(),
            attributes: {
                let mut m = IndexMap::new();
                m.insert("sym_name".to_string(), Attribute::String(exec_sym.clone()));
                m.insert(
                    "function_type".to_string(),
                    Attribute::Type(exec_fn_ty.clone()),
                );
                m
            },
            regions: vec![exec_reg],
            custom_syntax: false,
            location: exec_callable_loc,
        });

        new_top_level_ops.push(init_callable_id);
        new_top_level_ops.push(exec_callable_id);

        if !is_subgraph {
            // Synthesize wrapper `mef.model` with forwarding `grt.invoke` + `mef.output` regions.
            let mut wrapper_regions = Vec::new();
            for (in_tys, out_tys, callee_sym) in [
                (&init_in_tys, &init_out_tys, &init_sym),
                (&exec_in_tys, &exec_out_tys, &exec_sym),
            ] {
                let w_blk = module.blocks.insert_with_key(|bid| Block {
                    id: bid,
                    label: None,
                    arguments: Vec::new(),
                    ops: Vec::new(),
                });
                let mut w_args: SmallVec<[ValueId; 4]> = SmallVec::new();
                for (idx, ty) in in_tys.iter().enumerate() {
                    let av = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: ty.clone(),
                        def: ValueDef::BlockArgument {
                            block: w_blk,
                            index: idx as u32,
                        },
                    });
                    module.blocks[w_blk].arguments.push(av);
                    w_args.push(av);
                }
                let inv_loc = loc.clone();
                let sym_str = callee_sym.clone();
                let inv_id = module.ops.insert_with_key(|oid| Operation {
                    id: oid,
                    name: OpName::new("grt", "invoke"),
                    operands: w_args,
                    results: SmallVec::new(),
                    attributes: {
                        let mut m = IndexMap::new();
                        m.insert("callee".to_string(), Attribute::SymbolRef(sym_str));
                        m
                    },
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: inv_loc,
                });
                let mut inv_results: SmallVec<[ValueId; 4]> = SmallVec::new();
                for (idx, ty) in out_tys.iter().enumerate() {
                    let rv = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: ty.clone(),
                        def: ValueDef::OpResult {
                            op: inv_id,
                            index: idx as u32,
                        },
                    });
                    module.ops[inv_id].results.push(rv);
                    inv_results.push(rv);
                }
                let out_loc = loc.clone();
                let out_id = module.ops.insert_with_key(|oid| Operation {
                    id: oid,
                    name: OpName::new("mef", "output"),
                    operands: inv_results,
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: out_loc,
                });
                module.blocks[w_blk].ops = vec![inv_id, out_id];
                let w_reg = module.regions.insert_with_key(|rid| Region {
                    id: rid,
                    blocks: vec![w_blk],
                });
                wrapper_regions.push(w_reg);
            }

            let mut mef_attrs = module.ops[model_id].attributes.clone();
            mef_attrs.insert("init_type".to_string(), Attribute::Type(init_fn_ty));
            mef_attrs.insert("exec_type".to_string(), Attribute::Type(exec_fn_ty));
            let mef_model_id = module.ops.insert_with_key(|oid| Operation {
                id: oid,
                name: OpName::new("mef", "model"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: mef_attrs,
                regions: wrapper_regions,
                custom_syntax: false,
                location: loc,
            });
            new_top_level_ops.push(mef_model_id);
        }

        erased_models.insert(model_id);
    }

    let outlined_count = erased_models.len();
    module.blocks[root_blk]
        .ops
        .retain(|id| !erased_models.contains(id));
    module.blocks[root_blk].ops.extend(new_top_level_ops);
    for dead_id in erased_models {
        module.ops.remove(dead_id);
    }

    Ok(outlined_count)
}

/// Magic 4-byte signature (`"AKV8"`) for self-describing asymmetric K16/V8 multi-serde wire blobs.
pub const ASYM_K16_V8_BLOB_MAGIC: [u8; 4] = *b"AKV8";

/// Fixed 32-byte wire header size for [`AsymK16V8MultiSerde`].
pub const ASYM_K16_V8_HEADER_BYTES: usize = 32;

/// Header allowance (`1024` bytes) added by [`AsymK16V8MultiSerde::estimate_serialized_size`].
pub const ASYM_K16_V8_ESTIMATE_ALLOWANCE_BYTES: usize = 1024;

/// Operating mode for [`AsymK16V8MultiSerde`]:
/// - `StorageOnlyDequant`: both `K` (16-bit half precision) and `V` (8-bit quantized + scales)
///   are serialized into a self-contained blob (`k_payload_len > 0`).
/// - `SplitTierVOnly`: `K` remains in L1 host RAM (`slot 0` must be `None` on serialize and is a
///   no-op on deserialize), while only quantized `V` and its scales flow to L2
///   (`k_payload_len == 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsymK16V8SerdeMode {
    StorageOnlyDequant,
    SplitTierVOnly,
}

/// Multi-output asymmetric `K16/V8` serializer and deserializer supporting both self-contained
/// `(K, V)` storage-only-dequant blobs and split-tier `(None, V)` V-only blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsymK16V8MultiSerde {
    mode: AsymK16V8SerdeMode,
    scope: KvScaleScope,
    k_dtype_tag: u8,
}

impl AsymK16V8MultiSerde {
    /// Creates a new [`AsymK16V8MultiSerde`] with the given mode, scale scope, and K dtype tag
    /// (`1` for FP16, `2` for BF16).
    #[must_use]
    pub const fn new(mode: AsymK16V8SerdeMode, scope: KvScaleScope, k_dtype_tag: u8) -> Self {
        Self {
            mode,
            scope,
            k_dtype_tag,
        }
    }

    /// Fixed tuple arity (`group_size == 2`: `slot 0 = K`, `slot 1 = V`).
    #[must_use]
    pub const fn group_size(&self) -> usize {
        2
    }

    /// Returns the configured [`AsymK16V8SerdeMode`].
    #[must_use]
    pub const fn mode(&self) -> AsymK16V8SerdeMode {
        self.mode
    }

    /// Returns the configured [`KvScaleScope`].
    #[must_use]
    pub const fn scale_scope(&self) -> KvScaleScope {
        self.scope
    }

    /// Estimates the upper-bound serialized byte size for `(k_numel, v_numel)` under the active
    /// mode and scale scope. Rejects `PerLayerHead` / `PerPageHead` when only flat element counts
    /// are available (matching the layout-only estimator contract).
    pub fn estimate_serialized_size(
        &self,
        k_numel: Option<usize>,
        v_numel: Option<usize>,
    ) -> Result<usize, String> {
        let scales_bytes = match self.scope {
            KvScaleScope::PerTensor => 4usize,
            KvScaleScope::PerLayerHead | KvScaleScope::PerPageHead => {
                return Err(
                    "estimate_serialized_size requires PerTensor scope when head/page metadata \
                     is absent"
                        .to_string(),
                );
            }
        };

        match self.mode {
            AsymK16V8SerdeMode::StorageOnlyDequant => {
                let (Some(k_n), Some(v_n)) = (k_numel, v_numel) else {
                    return Err(
                        "StorageOnlyDequant estimate_serialized_size requires both K and V layouts"
                            .to_string(),
                    );
                };
                Ok(ASYM_K16_V8_ESTIMATE_ALLOWANCE_BYTES + k_n * 2 + v_n + scales_bytes)
            }
            AsymK16V8SerdeMode::SplitTierVOnly => {
                if k_numel.is_some() {
                    return Err(
                        "SplitTierVOnly estimate_serialized_size requires K layout to be None"
                            .to_string(),
                    );
                }
                let Some(v_n) = v_numel else {
                    return Err(
                        "SplitTierVOnly estimate_serialized_size requires V layout".to_string()
                    );
                };
                Ok(ASYM_K16_V8_ESTIMATE_ALLOWANCE_BYTES + v_n + scales_bytes)
            }
        }
    }

    /// Serializes `(k_opt, v_opt)` of shape `[num_pages, page_tokens, num_kv_heads, head_dim]`
    /// into a self-describing wire byte vector.
    pub fn serialize(
        &self,
        k_opt: Option<&[f32]>,
        v_opt: Option<&[f32]>,
        num_pages: usize,
        page_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<u8>, String> {
        use crate::ir::f32_to_f16_bits;

        let expected = num_pages
            .saturating_mul(page_tokens)
            .saturating_mul(num_kv_heads)
            .saturating_mul(head_dim);
        if expected == 0 {
            return Err("AsymK16V8MultiSerde: tensor shape has 0 elements".to_string());
        }

        let v_slice = v_opt.ok_or_else(|| "AsymK16V8MultiSerde: V slot is required".to_string())?;
        if v_slice.len() != expected {
            return Err(format!(
                "AsymK16V8MultiSerde: V length {} != expected {expected}",
                v_slice.len()
            ));
        }

        let k_bytes: Vec<u8> = match (self.mode, k_opt) {
            (AsymK16V8SerdeMode::StorageOnlyDequant, Some(k_slice)) => {
                if k_slice.len() != expected {
                    return Err(format!(
                        "AsymK16V8MultiSerde: K length {} != expected {expected}",
                        k_slice.len()
                    ));
                }
                let mut out = Vec::with_capacity(expected * 2);
                for &x in k_slice {
                    out.extend_from_slice(&f32_to_f16_bits(x).to_le_bytes());
                }
                out
            }
            (AsymK16V8SerdeMode::StorageOnlyDequant, None) => {
                return Err(
                    "AsymK16V8MultiSerde (StorageOnlyDequant): both K and V must be provided"
                        .to_string(),
                );
            }
            (AsymK16V8SerdeMode::SplitTierVOnly, Some(_)) => {
                return Err("AsymK16V8MultiSerde (SplitTierVOnly): K slot must be None".to_string());
            }
            (AsymK16V8SerdeMode::SplitTierVOnly, None) => Vec::new(),
        };

        let dummy_k = vec![0.0f32; expected];
        let enc = AsymK16V8Codec::encode(
            &dummy_k,
            v_slice,
            num_pages,
            page_tokens,
            num_kv_heads,
            head_dim,
            self.scope,
        )
        .map_err(|e| e.to_string())?;

        let v_payload_len = enc.v_int8_codes.len();
        let scale_payload_len = enc.v_scales.len() * 4;
        let k_payload_len = k_bytes.len();

        let scope_tag: u8 = match self.scope {
            KvScaleScope::PerTensor => 0,
            KvScaleScope::PerLayerHead => 1,
            KvScaleScope::PerPageHead => 2,
        };
        let mode_tag: u8 = match self.mode {
            AsymK16V8SerdeMode::StorageOnlyDequant => 0,
            AsymK16V8SerdeMode::SplitTierVOnly => 1,
        };

        let mut blob = Vec::with_capacity(
            ASYM_K16_V8_HEADER_BYTES + k_payload_len + v_payload_len + scale_payload_len,
        );
        blob.extend_from_slice(&ASYM_K16_V8_BLOB_MAGIC);
        blob.push(mode_tag);
        blob.push(scope_tag);
        blob.push(self.k_dtype_tag);
        blob.push(0u8); // reserved
        blob.extend_from_slice(&(num_pages as u32).to_le_bytes());
        blob.extend_from_slice(&(page_tokens as u32).to_le_bytes());
        blob.extend_from_slice(&(num_kv_heads as u32).to_le_bytes());
        blob.extend_from_slice(&(head_dim as u32).to_le_bytes());
        blob.extend_from_slice(&(k_payload_len as u32).to_le_bytes());
        blob.extend_from_slice(&(v_payload_len as u32).to_le_bytes());
        debug_assert_eq!(blob.len(), ASYM_K16_V8_HEADER_BYTES);

        blob.extend_from_slice(&k_bytes);
        for &code in &enc.v_int8_codes {
            blob.push(code as u8);
        }
        for &sc in &enc.v_scales {
            blob.extend_from_slice(&sc.to_bits().to_le_bytes());
        }
        Ok(blob)
    }

    /// Deserializes `blob` according to `(want_k, want_v)`:
    /// - In `StorageOnlyDequant` mode, either slot may be `false` to skip materializing that
    ///   output (`(true, false)` for K-only read, `(false, true)` for V-only read).
    /// - In `SplitTierVOnly` mode, rejects blobs with `k_payload_len > 0` and always returns `None`
    ///   for K (leaving the caller's L1 K buffer untouched) while dequantizing `V` when `want_v` is
    ///   `true`.
    #[allow(clippy::type_complexity)]
    pub fn deserialize(
        &self,
        blob: &[u8],
        want_k: bool,
        want_v: bool,
    ) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>), String> {
        use crate::ir::f16_bits_to_f32;

        if blob.len() < ASYM_K16_V8_HEADER_BYTES {
            return Err("AsymK16V8MultiSerde: blob shorter than 32-byte header".to_string());
        }
        if blob[0..4] != ASYM_K16_V8_BLOB_MAGIC {
            return Err("AsymK16V8MultiSerde: invalid magic header".to_string());
        }
        let scope = match blob[5] {
            0 => KvScaleScope::PerTensor,
            1 => KvScaleScope::PerLayerHead,
            2 => KvScaleScope::PerPageHead,
            other => return Err(format!("AsymK16V8MultiSerde: unknown scope tag {other}")),
        };
        let num_pages = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
        let page_tokens = u32::from_le_bytes(blob[12..16].try_into().unwrap()) as usize;
        let num_kv_heads = u32::from_le_bytes(blob[16..20].try_into().unwrap()) as usize;
        let head_dim = u32::from_le_bytes(blob[20..24].try_into().unwrap()) as usize;
        let k_payload_len = u32::from_le_bytes(blob[24..28].try_into().unwrap()) as usize;
        let v_payload_len = u32::from_le_bytes(blob[28..32].try_into().unwrap()) as usize;

        if self.mode == AsymK16V8SerdeMode::SplitTierVOnly && k_payload_len != 0 {
            return Err(format!(
                "AsymK16V8MultiSerde (SplitTierVOnly): blob has k_payload_len={k_payload_len} \
                 (>0); use StorageOnlyDequant mode to decode"
            ));
        }
        if self.mode == AsymK16V8SerdeMode::StorageOnlyDequant && k_payload_len == 0 && want_k {
            return Err(
                "AsymK16V8MultiSerde (StorageOnlyDequant): blob has k_payload_len=0".to_string(),
            );
        }

        let num_scales = match scope {
            KvScaleScope::PerTensor => 1,
            KvScaleScope::PerLayerHead => num_kv_heads,
            KvScaleScope::PerPageHead => num_pages * num_kv_heads,
        };
        let scale_payload_len = num_scales * 4;
        let expected_total =
            ASYM_K16_V8_HEADER_BYTES + k_payload_len + v_payload_len + scale_payload_len;
        if blob.len() < expected_total {
            return Err(format!(
                "AsymK16V8MultiSerde: truncated blob ({} < {expected_total})",
                blob.len()
            ));
        }

        let k_off = ASYM_K16_V8_HEADER_BYTES;
        let v_off = k_off + k_payload_len;
        let s_off = v_off + v_payload_len;

        let k_out = if self.mode == AsymK16V8SerdeMode::StorageOnlyDequant && want_k {
            let mut k_vec = Vec::with_capacity(k_payload_len / 2);
            for &chunk in blob[k_off..v_off].as_chunks::<2>().0 {
                let bits = u16::from_le_bytes(chunk);
                k_vec.push(f16_bits_to_f32(bits));
            }
            Some(k_vec)
        } else {
            None
        };

        let v_out = if want_v {
            let v_codes: Vec<i8> = blob[v_off..s_off].iter().map(|&b| b as i8).collect();
            let mut scales = Vec::with_capacity(num_scales);
            for &chunk in blob[s_off..s_off + scale_payload_len].as_chunks::<4>().0 {
                let bits = u32::from_le_bytes(chunk);
                scales.push(f32::from_bits(bits));
            }
            let dummy_page = EncodedAsymKvPage {
                k_fp16_bits: Vec::new(),
                v_int8_codes: v_codes,
                v_scales: scales,
                scope,
                num_pages,
                page_tokens,
                num_kv_heads,
                head_dim,
            };
            let (_, v_f32) = AsymK16V8Codec::decode(&dummy_page);
            Some(v_f32)
        } else {
            None
        };

        Ok((k_out, v_out))
    }
}

/// Conforms `castforge.call.opaque` (`mogg.call.opaque`) and `castforge.call.shape`
/// (`mogg.call.shape`) operands against their declared `castforge.kernel.decl` /
/// `mogg.kernel.decl` or `castforge.shape_func.decl` / `mogg.shape_func.decl` signatures:
/// 1. When a kernel decl sets `"supports_unfused_execute" == true`, the call has
///    `"has_epilogue_fusion" != true`, and every output operand is produced by a single-use,
///    non-recomputing `castforge.tensor.create.epilogue` (`mogg.tensor.create.epilogue`), bypasses
///    the pass-through epilogue to its first capture operand and erases the dead epilogue op.
/// 2. Reconciles mutable `castforge.tensor.create` (`mogg.tensor.create` with `"mut" == true`)
///    producers backing read-only `"tensor"` / `"trait:..."` inputs by cloning an immutable
///    producer (`"mut" = false`).
/// 3. Wraps `"scalar"` operands in `castforge.scalar.create`, non-fused `"trait:<contract>"`
///    operands in `castforge.trait.create` (stamped with `"device"` and `"contract"`), and
///    consecutive `"variadic:<abi>"` (`"plain"`, `"stable"`, `"fused_input"`, `"fused_output"`,
///    `"tensor_tuple"`) runs in `castforge.tuple.create` with `"abi"`.
///
/// Returns the number of calls whose operand lists were conformed.
pub fn conform_call_operands(module: &mut Module) -> Result<usize, String> {
    use crate::ir::mlir::OpName;
    use std::collections::HashMap;

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return Ok(0);
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return Ok(0);
    };

    // Collect top-level kernel declarations and shape function declarations.
    // Decl entry: (param_specs, supports_unfused_execute, is_shape_decl)
    type DeclInfo = (Vec<String>, bool, bool);
    let mut decls: HashMap<String, DeclInfo> = HashMap::new();
    for &op_id in &module.blocks[root_blk].ops {
        let full = module.ops[op_id].name.full_name();
        let is_kernel_decl = full == "castforge.kernel.decl" || full == "mogg.kernel.decl";
        let is_shape_decl = full == "castforge.shape_func.decl" || full == "mogg.shape_func.decl";
        if !is_kernel_decl && !is_shape_decl {
            continue;
        }
        let Some(sym) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        let specs: Vec<String> = match module.ops[op_id].attributes.get("param_specs") {
            Some(Attribute::Array(arr)) => arr
                .iter()
                .filter_map(|a| a.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        let supports_unfused = module.ops[op_id]
            .attributes
            .get("supports_unfused_execute")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);
        decls.insert(sym, (specs, supports_unfused, is_shape_decl));
    }

    let mut conformed_calls = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let device_label = module.ops[func_id]
            .attributes
            .get("device_ref")
            .or_else(|| module.ops[func_id].attributes.get("mo.device_ref"))
            .and_then(|a| a.as_str())
            .map(|s| s.split_once(':').map(|(l, _)| l).unwrap_or(s).to_string())
            .unwrap_or_else(|| "cpu".to_string());

        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let orig_ops = module.blocks[blk_id].ops.clone();
                let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
                for &oid in &orig_ops {
                    for &v in &module.ops[oid].operands {
                        *use_counts.entry(v).or_insert(0) += 1;
                    }
                }

                let mut new_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len() + 8);
                let mut dead_epilogues: std::collections::HashSet<OpId> =
                    std::collections::HashSet::new();

                for &op_id in &orig_ops {
                    let full = module.ops[op_id].name.full_name();
                    let is_opaque_call =
                        full == "castforge.call.opaque" || full == "mogg.call.opaque";
                    let is_shape_call = full == "castforge.call.shape" || full == "mogg.call.shape";
                    if !is_opaque_call && !is_shape_call {
                        new_ops.push(op_id);
                        continue;
                    }

                    let callee = module.ops[op_id]
                        .attributes
                        .get("callee")
                        .and_then(|a| a.as_str())
                        .ok_or_else(|| format!("{full} missing 'callee' attribute"))?
                        .trim_start_matches('@')
                        .to_string();

                    let Some((param_specs, supports_unfused, _)) = decls.get(&callee).cloned()
                    else {
                        return Err(format!("ConformCallOperands: no decl named '{callee}'"));
                    };

                    let mut call_operands: Vec<ValueId> =
                        module.ops[op_id].operands.iter().copied().collect();
                    let mut changed = false;

                    // 1. Check pass-through epilogue bypass for unfused `call.opaque`.
                    let has_epilogue_fusion = module.ops[op_id]
                        .attributes
                        .get("has_epilogue_fusion")
                        .and_then(|a| a.as_bool())
                        .unwrap_or(false);
                    if is_opaque_call && supports_unfused && !has_epilogue_fusion {
                        let mut epilogue_rewrites: Vec<(usize, OpId, ValueId)> = Vec::new();
                        let mut all_valid = true;
                        for (idx, &opd_v) in call_operands.iter().enumerate() {
                            if let ValueDef::OpResult { op: prod_op, .. } = module.values[opd_v].def
                            {
                                let p_name = module.ops[prod_op].name.full_name();
                                if p_name == "castforge.tensor.create.epilogue"
                                    || p_name == "mogg.tensor.create.epilogue"
                                {
                                    let recomputes = module.ops[prod_op]
                                        .attributes
                                        .get("recomputes_value")
                                        .and_then(|a| a.as_bool())
                                        .unwrap_or(false);
                                    let single_use =
                                        use_counts.get(&opd_v).copied().unwrap_or(0) == 1;
                                    if !recomputes
                                        && single_use
                                        && let Some(&cap0) = module.ops[prod_op].operands.first()
                                    {
                                        epilogue_rewrites.push((idx, prod_op, cap0));
                                    } else {
                                        all_valid = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if all_valid && !epilogue_rewrites.is_empty() {
                            for (idx, ep_op, cap0) in epilogue_rewrites {
                                call_operands[idx] = cap0;
                                dead_epilogues.insert(ep_op);
                            }
                            changed = true;
                        }
                    }

                    // 2. Pack operands according to `param_specs`.
                    let loc = module.ops[op_id].location.clone();
                    let mut packed: SmallVec<[ValueId; 4]> = SmallVec::new();
                    let mut i = 0usize;
                    let mut spec_idx = 0usize;

                    while i < call_operands.len() {
                        let spec = param_specs
                            .get(spec_idx)
                            .map(String::as_str)
                            .unwrap_or("tensor");
                        let opd = call_operands[i];

                        // Mut-to-immut reconciliation for read-only kernel inputs:
                        let reconciled_opd = if is_opaque_call && spec != "mut_tensor" {
                            if let ValueDef::OpResult { op: prod_op, .. } = module.values[opd].def {
                                let p_name = module.ops[prod_op].name.full_name();
                                let is_mut = module.ops[prod_op]
                                    .attributes
                                    .get("mut")
                                    .and_then(|a| a.as_bool())
                                    .unwrap_or(false);
                                if (p_name == "castforge.tensor.create"
                                    || p_name == "mogg.tensor.create")
                                    && is_mut
                                {
                                    let p_opds = module.ops[prod_op].operands.clone();
                                    let mut p_attrs = module.ops[prod_op].attributes.clone();
                                    p_attrs.insert("mut".to_string(), Attribute::Bool(false));
                                    let p_ty = module.values[opd].ty.clone();
                                    let immut_loc = loc.clone();
                                    let immut_op = module.ops.insert_with_key(|oid| Operation {
                                        id: oid,
                                        name: OpName::new("castforge", "tensor.create"),
                                        operands: p_opds,
                                        results: SmallVec::new(),
                                        attributes: p_attrs,
                                        regions: Vec::new(),
                                        custom_syntax: false,
                                        location: immut_loc,
                                    });
                                    let immut_val = module.values.insert_with_key(|vid| Value {
                                        id: vid,
                                        ssa_name: None,
                                        ty: p_ty,
                                        def: ValueDef::OpResult {
                                            op: immut_op,
                                            index: 0,
                                        },
                                    });
                                    module.ops[immut_op].results.push(immut_val);
                                    new_ops.push(immut_op);
                                    changed = true;
                                    immut_val
                                } else {
                                    opd
                                }
                            } else {
                                opd
                            }
                        } else {
                            opd
                        };

                        if spec == "scalar" {
                            let s_loc = loc.clone();
                            let s_ty = module.values[reconciled_opd].ty.clone();
                            let sc_op = module.ops.insert_with_key(|oid| Operation {
                                id: oid,
                                name: OpName::new("castforge", "scalar.create"),
                                operands: SmallVec::from_slice(&[reconciled_opd]),
                                results: SmallVec::new(),
                                attributes: IndexMap::new(),
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: s_loc,
                            });
                            let sc_val = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: s_ty,
                                def: ValueDef::OpResult {
                                    op: sc_op,
                                    index: 0,
                                },
                            });
                            module.ops[sc_op].results.push(sc_val);
                            new_ops.push(sc_op);
                            packed.push(sc_val);
                            changed = true;
                            i += 1;
                            spec_idx += 1;
                            continue;
                        }

                        if let Some(contract) = spec.strip_prefix("trait:") {
                            let t_loc = loc.clone();
                            let t_ty = module.values[reconciled_opd].ty.clone();
                            let dev_str = device_label.clone();
                            let contract_str = contract.to_string();
                            let tr_op = module.ops.insert_with_key(|oid| Operation {
                                id: oid,
                                name: OpName::new("castforge", "trait.create"),
                                operands: SmallVec::from_slice(&[reconciled_opd]),
                                results: SmallVec::new(),
                                attributes: {
                                    let mut a = IndexMap::new();
                                    a.insert("device".to_string(), Attribute::String(dev_str));
                                    a.insert(
                                        "contract".to_string(),
                                        Attribute::String(contract_str),
                                    );
                                    a
                                },
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: t_loc,
                            });
                            let tr_val = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: t_ty,
                                def: ValueDef::OpResult {
                                    op: tr_op,
                                    index: 0,
                                },
                            });
                            module.ops[tr_op].results.push(tr_val);
                            new_ops.push(tr_op);
                            packed.push(tr_val);
                            changed = true;
                            i += 1;
                            spec_idx += 1;
                            continue;
                        }

                        if let Some(abi_str) = spec.strip_prefix("variadic:") {
                            // Consume consecutive operands matching this variadic slot.
                            let remaining_specs = param_specs.len().saturating_sub(spec_idx + 1);
                            let run_end = call_operands
                                .len()
                                .saturating_sub(remaining_specs)
                                .max(i + 1);
                            let mut run: SmallVec<[ValueId; 4]> = SmallVec::new();
                            let mut elem_tys: Vec<Type> = Vec::new();
                            while i < run_end && i < call_operands.len() {
                                let mut elem_v = call_operands[i];
                                let elem_ty = module.values[elem_v].ty.clone();
                                if abi_str == "tensor_tuple" {
                                    let tr_loc = loc.clone();
                                    let dev_s = device_label.clone();
                                    let tr_ty = elem_ty.clone();
                                    let tr_op = module.ops.insert_with_key(|oid| Operation {
                                        id: oid,
                                        name: OpName::new("castforge", "trait.create"),
                                        operands: SmallVec::from_slice(&[elem_v]),
                                        results: SmallVec::new(),
                                        attributes: {
                                            let mut a = IndexMap::new();
                                            a.insert(
                                                "device".to_string(),
                                                Attribute::String(dev_s),
                                            );
                                            a.insert(
                                                "contract".to_string(),
                                                Attribute::String("TileTensorable".to_string()),
                                            );
                                            a
                                        },
                                        regions: Vec::new(),
                                        custom_syntax: false,
                                        location: tr_loc,
                                    });
                                    let wrapped_v = module.values.insert_with_key(|vid| Value {
                                        id: vid,
                                        ssa_name: None,
                                        ty: tr_ty,
                                        def: ValueDef::OpResult {
                                            op: tr_op,
                                            index: 0,
                                        },
                                    });
                                    module.ops[tr_op].results.push(wrapped_v);
                                    new_ops.push(tr_op);
                                    elem_v = wrapped_v;
                                }
                                run.push(elem_v);
                                elem_tys.push(elem_ty);
                                i += 1;
                            }
                            let tup_loc = loc.clone();
                            let abi_owned = abi_str.to_string();
                            let tup_op = module.ops.insert_with_key(|oid| Operation {
                                id: oid,
                                name: OpName::new("castforge", "tuple.create"),
                                operands: run,
                                results: SmallVec::new(),
                                attributes: {
                                    let mut a = IndexMap::new();
                                    a.insert("abi".to_string(), Attribute::String(abi_owned));
                                    a
                                },
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: tup_loc,
                            });
                            let tup_val = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: Type::Tuple(elem_tys),
                                def: ValueDef::OpResult {
                                    op: tup_op,
                                    index: 0,
                                },
                            });
                            module.ops[tup_op].results.push(tup_val);
                            new_ops.push(tup_op);
                            packed.push(tup_val);
                            changed = true;
                            spec_idx += 1;
                            continue;
                        }

                        packed.push(reconciled_opd);
                        i += 1;
                        spec_idx += 1;
                    }

                    if changed {
                        module.ops[op_id].operands = packed;
                        module.ops[op_id]
                            .attributes
                            .shift_remove("has_same_variadic_operand_size");
                        conformed_calls += 1;
                    }
                    new_ops.push(op_id);
                }

                if !dead_epilogues.is_empty() {
                    new_ops.retain(|id| !dead_epilogues.contains(id));
                    for dead_id in dead_epilogues {
                        module.ops.remove(dead_id);
                    }
                }
                module.blocks[blk_id].ops = new_ops;
            }
        }
    }

    Ok(conformed_calls)
}

/// Outlines each `castforge.routine` (`mogg.routine`) body into a module-level `func.func` stub
/// (labeled with `"abi" = "runtime"`) while keeping the `routine` container at the call site:
/// - Any `castforge.buffer.constant` (`mogg.buffer.constant`) ops inside the routine body are moved
///   to the rebuilt routine body at the call site, and their uses inside the outlined stub are
///   replaced by trailing `!castforge.pointer` block arguments.
/// - Rebuilds the routine's body as those hoisted buffer constants followed by a single
///   `castforge.call` forwarding fresh block arguments plus the constant pointers to the new stub.
pub fn outline_routines(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, OpName, Region};
    use std::collections::{HashMap, HashSet};

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return 0;
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return 0;
    };

    let mut used_syms: HashSet<String> = HashSet::new();
    for &op_id in &module.blocks[root_blk].ops {
        if let Some(s) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
        {
            used_syms.insert(s.to_string());
        }
    }

    let mut outlined = 0usize;
    let func_ids = module.functions();
    let mut new_stubs: Vec<OpId> = Vec::new();

    for func_id in func_ids {
        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let ops = module.blocks[blk_id].ops.clone();
                for op_id in ops {
                    let full = module.ops[op_id].name.full_name();
                    if full != "castforge.routine" && full != "mogg.routine" {
                        continue;
                    }
                    if module.ops[op_id].regions.is_empty() {
                        continue;
                    }
                    let body_reg = module.ops[op_id].regions[0];
                    if module.regions[body_reg].blocks.is_empty() {
                        continue;
                    }
                    let body_blk = module.regions[body_reg].blocks[0];
                    let body_ops = module.blocks[body_blk].ops.clone();
                    let orig_args = module.blocks[body_blk].arguments.clone();
                    let arg_types: Vec<Type> = orig_args
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();
                    let loc = module.ops[op_id].location.clone();
                    let ptr_ty = Type::Opaque("!castforge.pointer".to_string());

                    // Partition `buffer.constant` ops from the rest of the routine body.
                    let mut const_ops: Vec<OpId> = Vec::new();
                    let mut stub_ops: Vec<OpId> = Vec::new();
                    let mut const_val_to_arg: HashMap<ValueId, ValueId> = HashMap::new();

                    for &b_op in &body_ops {
                        let b_name = module.ops[b_op].name.full_name();
                        if (b_name == "castforge.buffer.constant"
                            || b_name == "mogg.buffer.constant")
                            && let Some(&c_res) = module.ops[b_op].results.first()
                        {
                            let new_idx = module.blocks[body_blk].arguments.len() as u32;
                            let trailing_arg = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: ptr_ty.clone(),
                                def: ValueDef::BlockArgument {
                                    block: body_blk,
                                    index: new_idx,
                                },
                            });
                            module.blocks[body_blk].arguments.push(trailing_arg);
                            const_val_to_arg.insert(c_res, trailing_arg);
                            const_ops.push(b_op);
                        } else {
                            stub_ops.push(b_op);
                        }
                    }

                    // Rewrite uses of hoisted constant values inside `stub_ops` to trailing args.
                    for &s_op in &stub_ops {
                        for opd in module.ops[s_op].operands.iter_mut() {
                            if let Some(&rep) = const_val_to_arg.get(opd) {
                                *opd = rep;
                            }
                        }
                    }

                    // Check if stub_ops ends with a terminator (`mogg.output`, `castforge.output`,
                    // or `func.return`), or cap it with `castforge.output`.
                    let mut stub_res_types: Vec<Type> = Vec::new();
                    let has_terminator = stub_ops.last().is_some_and(|&last_op| {
                        let n = module.ops[last_op].name.full_name();
                        n == "mogg.output" || n == "castforge.output" || n == "func.return"
                    });
                    if has_terminator {
                        let term = *stub_ops.last().unwrap();
                        stub_res_types = module.ops[term]
                            .operands
                            .iter()
                            .map(|&v| module.values[v].ty.clone())
                            .collect();
                    } else {
                        let out_loc = loc.clone();
                        let out_op = module.ops.insert_with_key(|oid| Operation {
                            id: oid,
                            name: OpName::new("castforge", "output"),
                            operands: SmallVec::new(),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: out_loc,
                        });
                        stub_ops.push(out_op);
                    }
                    module.blocks[body_blk].ops = stub_ops;

                    let stub_arg_types: Vec<Type> = module.blocks[body_blk]
                        .arguments
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();

                    let fusion_info = module.ops[op_id]
                        .attributes
                        .get("fusion_info")
                        .and_then(|a| a.as_str())
                        .unwrap_or("routine")
                        .to_string();
                    let base_sym = format!("__{}", fusion_info.trim_start_matches('_'));
                    let mut stub_sym = base_sym.clone();
                    let mut suffix = 0usize;
                    while used_syms.contains(&stub_sym) {
                        suffix += 1;
                        stub_sym = format!("{base_sym}_{suffix}");
                    }
                    used_syms.insert(stub_sym.clone());

                    let device_label = module.ops[op_id]
                        .attributes
                        .get("device_ref")
                        .and_then(|a| a.as_str())
                        .map(|s| s.split_once(':').map(|(l, _)| l).unwrap_or(s).to_string())
                        .unwrap_or_else(|| "cpu".to_string());

                    let stub_loc = loc.clone();
                    let stub_fn_ty = Type::Function {
                        inputs: stub_arg_types,
                        results: stub_res_types.clone(),
                    };
                    let stub_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("func", "func"),
                        operands: SmallVec::new(),
                        results: SmallVec::new(),
                        attributes: {
                            let mut a = IndexMap::new();
                            a.insert("sym_name".to_string(), Attribute::String(stub_sym.clone()));
                            a.insert("function_type".to_string(), Attribute::Type(stub_fn_ty));
                            a.insert("fusion_info".to_string(), Attribute::String(fusion_info));
                            a.insert("abi".to_string(), Attribute::String("runtime".to_string()));
                            a.insert("device".to_string(), Attribute::String(device_label));
                            a.insert(
                                "constant_resources".to_string(),
                                Attribute::Array(Vec::new()),
                            );
                            a
                        },
                        regions: vec![body_reg],
                        custom_syntax: true,
                        location: stub_loc,
                    });
                    new_stubs.push(stub_id);

                    // Rebuild the `routine` body with fresh block args, hoisted constants, and
                    // a single `castforge.call` to `stub_sym`.
                    let call_blk = module.blocks.insert_with_key(|bid| Block {
                        id: bid,
                        label: None,
                        arguments: Vec::new(),
                        ops: Vec::new(),
                    });
                    let mut forwarded: SmallVec<[ValueId; 4]> = SmallVec::new();
                    for (idx, ty) in arg_types.into_iter().enumerate() {
                        let av = module.values.insert_with_key(|vid| Value {
                            id: vid,
                            ssa_name: None,
                            ty,
                            def: ValueDef::BlockArgument {
                                block: call_blk,
                                index: idx as u32,
                            },
                        });
                        module.blocks[call_blk].arguments.push(av);
                        forwarded.push(av);
                    }
                    let mut call_blk_ops: Vec<OpId> = Vec::new();
                    for c_op in const_ops {
                        call_blk_ops.push(c_op);
                        if let Some(&c_res) = module.ops[c_op].results.first() {
                            forwarded.push(c_res);
                        }
                    }
                    let call_loc = loc.clone();
                    let call_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("castforge", "call"),
                        operands: forwarded,
                        results: SmallVec::new(),
                        attributes: {
                            let mut a = IndexMap::new();
                            a.insert("callee".to_string(), Attribute::SymbolRef(stub_sym));
                            a
                        },
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: call_loc,
                    });
                    for (r_idx, r_ty) in stub_res_types.into_iter().enumerate() {
                        let rv = module.values.insert_with_key(|vid| Value {
                            id: vid,
                            ssa_name: None,
                            ty: r_ty,
                            def: ValueDef::OpResult {
                                op: call_id,
                                index: r_idx as u32,
                            },
                        });
                        module.ops[call_id].results.push(rv);
                    }
                    call_blk_ops.push(call_id);
                    module.blocks[call_blk].ops = call_blk_ops;

                    let call_reg = module.regions.insert_with_key(|rid| Region {
                        id: rid,
                        blocks: vec![call_blk],
                    });
                    module.ops[op_id].regions = vec![call_reg];
                    outlined += 1;
                }
            }
        }
    }

    module.blocks[root_blk].ops.extend(new_stubs);
    outlined
}

/// Outlines each `castforge.routine.async` (`mogg.routine.async`) op into a module-level
/// `func.func` stub (`"abi" = "graph"`) named `__<callees>_async` (or `__primitive_<N>_async` when
/// the region contains no inner calls), and replaces the `routine.async` op at the call site with a
/// `castforge.compiled_call` (`"callee_symbol" = @<stub_name>`) forwarding its operands, fusion
/// info, dispatch kind, constant resources, and device reference.
pub fn outline_async_routines(module: &mut Module) -> usize {
    use crate::ir::mlir::OpName;
    use std::collections::HashSet;

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return 0;
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return 0;
    };

    let mut used_syms: HashSet<String> = HashSet::new();
    for &op_id in &module.blocks[root_blk].ops {
        if let Some(s) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
        {
            used_syms.insert(s.to_string());
        }
    }

    let mut stub_number = 0usize;
    let mut outlined = 0usize;
    let func_ids = module.functions();
    let mut new_stubs: Vec<OpId> = Vec::new();

    for func_id in func_ids {
        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let orig_ops = module.blocks[blk_id].ops.clone();
                let mut updated_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len());

                for &op_id in &orig_ops {
                    let full = module.ops[op_id].name.full_name();
                    if full != "castforge.routine.async" && full != "mogg.routine.async" {
                        updated_ops.push(op_id);
                        continue;
                    }
                    if module.ops[op_id].regions.is_empty() {
                        updated_ops.push(op_id);
                        continue;
                    }
                    let body_reg = module.ops[op_id].regions[0];
                    if module.regions[body_reg].blocks.is_empty() {
                        updated_ops.push(op_id);
                        continue;
                    }
                    let body_blk = module.regions[body_reg].blocks[0];

                    // Collect callees inside `body_blk` in call order.
                    let mut callees: Vec<String> = Vec::new();
                    for &b_op in &module.blocks[body_blk].ops {
                        if let Some(c) = module.ops[b_op]
                            .attributes
                            .get("callee")
                            .or_else(|| module.ops[b_op].attributes.get("callee_symbol"))
                            .and_then(|a| a.as_str())
                        {
                            callees.push(
                                c.trim_start_matches('@')
                                    .trim_start_matches('_')
                                    .to_string(),
                            );
                        }
                    }
                    let base_part = if callees.is_empty() {
                        format!("primitive_{stub_number}")
                    } else {
                        callees.join("_")
                    };
                    stub_number += 1;
                    let base_sym = format!("__{base_part}_async");
                    let mut stub_sym = base_sym.clone();
                    let mut suffix = 0usize;
                    while used_syms.contains(&stub_sym) {
                        suffix += 1;
                        stub_sym = format!("{base_sym}_{suffix}");
                    }
                    used_syms.insert(stub_sym.clone());

                    let arg_tys: Vec<Type> = module.blocks[body_blk]
                        .arguments
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();
                    let res_tys: Vec<Type> = module.ops[op_id]
                        .results
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();

                    let fusion_info = module.ops[op_id]
                        .attributes
                        .get("fusion_info")
                        .or_else(|| module.ops[op_id].attributes.get("summary"))
                        .and_then(|a| a.as_str())
                        .unwrap_or(&stub_sym)
                        .to_string();
                    let dispatch_attr = module.ops[op_id].attributes.get("dispatch").cloned();
                    let const_res_attr = module.ops[op_id]
                        .attributes
                        .get("constant_resources")
                        .cloned()
                        .unwrap_or_else(|| Attribute::Array(Vec::new()));
                    let dev_ref = module.ops[op_id]
                        .attributes
                        .get("device_ref")
                        .and_then(|a| a.as_str())
                        .unwrap_or("cpu:0")
                        .to_string();
                    let dev_label = dev_ref
                        .split_once(':')
                        .map(|(l, _)| l)
                        .unwrap_or(&dev_ref)
                        .to_string();
                    let loc = module.ops[op_id].location.clone();

                    let stub_loc = loc.clone();
                    let stub_fn_ty = Type::Function {
                        inputs: arg_tys,
                        results: res_tys.clone(),
                    };
                    let mut stub_attrs = IndexMap::new();
                    stub_attrs.insert("sym_name".to_string(), Attribute::String(stub_sym.clone()));
                    stub_attrs.insert("function_type".to_string(), Attribute::Type(stub_fn_ty));
                    stub_attrs.insert(
                        "fusion_info".to_string(),
                        Attribute::String(fusion_info.clone()),
                    );
                    if let Some(d) = dispatch_attr.clone() {
                        stub_attrs.insert("dispatch".to_string(), d);
                    }
                    stub_attrs.insert("constant_resources".to_string(), const_res_attr.clone());
                    stub_attrs.insert("abi".to_string(), Attribute::String("graph".to_string()));
                    stub_attrs.insert("device".to_string(), Attribute::String(dev_label));

                    let stub_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("func", "func"),
                        operands: SmallVec::new(),
                        results: SmallVec::new(),
                        attributes: stub_attrs,
                        regions: vec![body_reg],
                        custom_syntax: true,
                        location: stub_loc,
                    });
                    new_stubs.push(stub_id);

                    // Replace `routine.async` in-place with `castforge.compiled_call`.
                    module.ops[op_id].name = OpName::new("castforge", "compiled_call");
                    module.ops[op_id].regions.clear();
                    module.ops[op_id]
                        .attributes
                        .insert("callee_symbol".to_string(), Attribute::SymbolRef(stub_sym));
                    module.ops[op_id]
                        .attributes
                        .insert("fusion_info".to_string(), Attribute::String(fusion_info));
                    module.ops[op_id]
                        .attributes
                        .insert("constant_resources".to_string(), const_res_attr);
                    module.ops[op_id]
                        .attributes
                        .insert("device_ref".to_string(), Attribute::String(dev_ref));
                    updated_ops.push(op_id);
                    outlined += 1;
                }

                module.blocks[blk_id].ops = updated_ops;
            }
        }
    }

    // Insert new async stubs at the start of the module root block.
    if !new_stubs.is_empty() {
        let mut combined = new_stubs;
        combined.extend(module.blocks[root_blk].ops.iter().copied());
        module.blocks[root_blk].ops = combined;
    }

    outlined
}

/// Wraps the dispatch suffix of runtime-ABI (`"abi" = "runtime"` or `"mogg"`) functions in a
/// `castforge.with_trace` (`mogg.with_trace`) region:
/// - Keeps setup operations (before the first call-like op) outside the trace region so passes
///   that rebuild dispatch from outside can still reach setup values.
/// - Moves all ops starting at the first call-like op (`castforge.call*`, `mogg.call*`,
///   `func.call`) up to (but excluding) the terminator into `castforge.with_trace`'s body block,
///   appends `castforge.yield` with the terminator's original operands, and forwards the
///   `castforge.with_trace` results into the function terminator.
pub fn wrap_funcs_with_trace(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, OpName, Region};

    let mut wrapped = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let abi = module.ops[func_id]
            .attributes
            .get("abi")
            .and_then(|a| a.as_str())
            .unwrap_or("");
        if abi != "runtime" && abi != "mogg" {
            continue;
        }
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let body_reg = module.ops[func_id].regions[0];
        if module.regions[body_reg].blocks.is_empty() {
            continue;
        }
        let entry_blk = module.regions[body_reg].blocks[0];
        let block_ops = module.blocks[entry_blk].ops.clone();
        if block_ops.len() < 2 {
            continue;
        }

        // Skip if already wrapped in `with_trace`.
        if block_ops.iter().any(|&oid| {
            let f = module.ops[oid].name.full_name();
            f == "castforge.with_trace" || f == "mogg.with_trace"
        }) {
            continue;
        }

        let term_id = *block_ops.last().unwrap();
        let mut setup_ops: Vec<OpId> = Vec::new();
        let mut traced_ops: Vec<OpId> = Vec::new();
        let mut seen_call = false;

        for &oid in &block_ops[..block_ops.len() - 1] {
            let full = module.ops[oid].name.full_name();
            if full == "func.call"
                || full.starts_with("castforge.call")
                || full.starts_with("mogg.call")
            {
                seen_call = true;
            }
            if seen_call {
                traced_ops.push(oid);
            } else {
                setup_ops.push(oid);
            }
        }

        if traced_ops.is_empty() {
            continue;
        }

        let loc = module.ops[func_id].location.clone();
        let term_operands = module.ops[term_id].operands.clone();
        let term_types: Vec<Type> = term_operands
            .iter()
            .map(|&v| module.values[v].ty.clone())
            .collect();

        let trace_blk = module.blocks.insert_with_key(|bid| Block {
            id: bid,
            label: None,
            arguments: Vec::new(),
            ops: Vec::new(),
        });
        let yield_loc = loc.clone();
        let yield_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("castforge", "yield"),
            operands: term_operands,
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: yield_loc,
        });
        traced_ops.push(yield_id);
        module.blocks[trace_blk].ops = traced_ops;

        let trace_reg = module.regions.insert_with_key(|rid| Region {
            id: rid,
            blocks: vec![trace_blk],
        });

        let trace_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("castforge", "with_trace"),
            operands: SmallVec::new(),
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: vec![trace_reg],
            custom_syntax: false,
            location: loc,
        });
        let mut trace_results: SmallVec<[ValueId; 4]> = SmallVec::new();
        for (idx, ty) in term_types.into_iter().enumerate() {
            let rv = module.values.insert_with_key(|vid| Value {
                id: vid,
                ssa_name: None,
                ty,
                def: ValueDef::OpResult {
                    op: trace_id,
                    index: idx as u32,
                },
            });
            module.ops[trace_id].results.push(rv);
            trace_results.push(rv);
        }

        module.ops[term_id].operands = trace_results;
        setup_ops.push(trace_id);
        setup_ops.push(term_id);
        module.blocks[entry_blk].ops = setup_ops;
        wrapped += 1;
    }

    wrapped
}

fn rewrite_single_opaque_type(
    ty: &mut Type,
    mapping: &IndexMap<String, Attribute>,
    kernel_leaves: &std::collections::HashSet<String>,
) -> Result<usize, String> {
    match ty {
        Type::Opaque(raw) => {
            // Match `!mo.opaque<"Symbol"...>` or `!castforge.opaque<"Symbol"...>`
            for prefix in ["!mo.opaque<\"", "!castforge.opaque<\""] {
                if let Some(after) = raw.strip_prefix(prefix)
                    && let Some((sym, tail)) = after.split_once('"')
                {
                    if let Some(qual_attr) = mapping.get(sym)
                        && let Some(qual_str) = qual_attr.as_str()
                    {
                        if qual_str != sym {
                            *raw = format!("{prefix}{qual_str}\"{tail}");
                            return Ok(1);
                        }
                        return Ok(0);
                    }
                    if !sym.contains('.') && kernel_leaves.contains(sym) {
                        return Err(format!(
                            "opaque type '{sym}' refers to a registered kernel struct; kernel \
                             structs are implementations resolved by registration, not opaque \
                             data types"
                        ));
                    }
                }
            }
            Ok(0)
        }
        Type::Tuple(items) => {
            let mut sum = 0usize;
            for item in items {
                sum += rewrite_single_opaque_type(item, mapping, kernel_leaves)?;
            }
            Ok(sum)
        }
        Type::Function { inputs, results } => {
            let mut sum = 0usize;
            for inp in inputs {
                sum += rewrite_single_opaque_type(inp, mapping, kernel_leaves)?;
            }
            for res in results {
                sum += rewrite_single_opaque_type(res, mapping, kernel_leaves)?;
            }
            Ok(sum)
        }
        _ => Ok(0),
    }
}

/// Qualifies short `!mo.opaque<"Symbol">` and `!castforge.opaque<"Symbol">` types across the
/// module using a simple-to-qualified dictionary mapping (`mapping`), while rejecting any
/// unqualified opaque type whose symbol matches the leaf name of a registered `kernel.decl` or
/// `primitive.decl` in the module.
pub fn rewrite_opaque_types(
    module: &mut Module,
    mapping: &IndexMap<String, Attribute>,
) -> Result<usize, String> {
    use std::collections::HashSet;

    let mut kernel_leaves: HashSet<String> = HashSet::new();
    if let Some(root_reg) = module.regions.get(module.root_region)
        && let Some(&root_blk) = root_reg.blocks.first()
    {
        for &op_id in &module.blocks[root_blk].ops {
            let full = module.ops[op_id].name.full_name();
            if matches!(
                full.as_str(),
                "castforge.kernel.decl"
                    | "mogg.kernel.decl"
                    | "castforge.primitive.decl"
                    | "mogg.primitive.decl"
            ) && let Some(sym) = module.ops[op_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
            {
                let leaf = sym.rsplit_once('.').map(|(_, r)| r).unwrap_or(sym);
                kernel_leaves.insert(leaf.to_string());
            }
        }
    }

    if mapping.is_empty() && kernel_leaves.is_empty() {
        return Ok(0);
    }

    let mut rewritten = 0usize;
    let val_ids: Vec<ValueId> = module.values.keys().collect();
    for vid in val_ids {
        rewritten +=
            rewrite_single_opaque_type(&mut module.values[vid].ty, mapping, &kernel_leaves)?;
    }

    let op_ids: Vec<OpId> = module.ops.keys().collect();
    for oid in op_ids {
        let attr_keys: Vec<String> = module.ops[oid].attributes.keys().cloned().collect();
        for k in attr_keys {
            if let Some(Attribute::Type(ty)) = module.ops[oid].attributes.get_mut(&k) {
                rewritten += rewrite_single_opaque_type(ty, mapping, &kernel_leaves)?;
            }
        }
    }

    Ok(rewritten)
}

fn strip_dim_name_in_type(ty: &mut Type) -> bool {
    if let Type::Opaque(raw) = ty {
        for (prefix, bare) in [
            ("!mogg.dim<", "!mogg.dim"),
            ("!castforge.dim<", "!castforge.dim"),
        ] {
            if raw.starts_with(prefix) && raw.ends_with('>') {
                *raw = bare.to_string();
                return true;
            }
        }
    }
    false
}

/// Drops the parameter-reference name from every `!mogg.dim<name>` or `!castforge.dim<name>`
/// block argument of top-level functions and updates their `"function_type"` attributes
/// accordingly. Returns the number of functions whose signatures were updated.
pub fn anonymize_dim_names(module: &mut Module) -> usize {
    let mut updated_funcs = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let reg_id = module.ops[func_id].regions[0];
        if module.regions[reg_id].blocks.is_empty() {
            continue;
        }
        let blk_id = module.regions[reg_id].blocks[0];
        let args = module.blocks[blk_id].arguments.clone();
        let mut changed = false;
        let mut new_arg_types: Vec<Type> = Vec::with_capacity(args.len());

        for arg_v in args {
            if strip_dim_name_in_type(&mut module.values[arg_v].ty) {
                changed = true;
            }
            new_arg_types.push(module.values[arg_v].ty.clone());
        }

        if !changed {
            continue;
        }

        if let Some(Attribute::Type(Type::Function { inputs, .. })) =
            module.ops[func_id].attributes.get_mut("function_type")
        {
            *inputs = new_arg_types;
        }
        updated_funcs += 1;
    }

    updated_funcs
}

/// Renames all top-level symbols in `module` sequentially to `__S0`, `__S1`, ..., rewrites all
/// `Attribute::SymbolRef` references across the module to match, and normalizes scoped
/// parameter declarations (`input_parameters` arrays and `kgen.param.declare` /
/// `castforge.param.declare` ops) to `__P0`, `__P1`, ... within each top-level operation.
pub fn normalize_all_names(module: &mut Module) -> Result<usize, String> {
    use std::collections::HashMap;

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return Ok(0);
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return Ok(0);
    };
    let top_ops = module.blocks[root_blk].ops.clone();

    let mut sym_map: HashMap<String, String> = HashMap::new();
    let mut sym_idx = 0usize;

    for &op_id in &top_ops {
        if let Some(old_name) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        {
            let new_name = format!("__S{sym_idx}");
            sym_idx += 1;
            module.ops[op_id]
                .attributes
                .insert("sym_name".to_string(), Attribute::String(new_name.clone()));
            sym_map.insert(old_name, new_name);
        }
    }

    // Scoped parameter uniquing within each top-level operation.
    for &top_id in &top_ops {
        let mut param_counter = 0usize;
        let mut param_map: HashMap<String, String> = HashMap::new();

        if let Some(Attribute::Array(arr)) = module.ops[top_id].attributes.get("input_parameters") {
            let old_params: Vec<String> = arr
                .iter()
                .filter_map(|a| a.as_str().map(str::to_string))
                .collect();
            let mut new_arr: Vec<Attribute> = Vec::with_capacity(old_params.len());
            for old_p in old_params {
                if param_map.contains_key(&old_p) {
                    return Err(format!("Shadowing of parameter name '{old_p}' detected"));
                }
                let new_p = format!("__P{param_counter}");
                param_counter += 1;
                param_map.insert(old_p, new_p.clone());
                new_arr.push(Attribute::String(new_p));
            }
            module.ops[top_id]
                .attributes
                .insert("input_parameters".to_string(), Attribute::Array(new_arr));
        }

        let regions = module.ops[top_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let mut block_scope = param_map.clone();
                let ops = module.blocks[blk_id].ops.clone();
                for &oid in &ops {
                    let full = module.ops[oid].name.full_name();
                    if (full == "kgen.param.declare" || full == "castforge.param.declare")
                        && let Some(old_p) = module.ops[oid]
                            .attributes
                            .get("param_name")
                            .and_then(|a| a.as_str())
                            .map(str::to_string)
                    {
                        if block_scope.contains_key(&old_p) {
                            return Err(format!("Shadowing of parameter name '{old_p}' detected"));
                        }
                        let new_p = format!("__P{param_counter}");
                        param_counter += 1;
                        block_scope.insert(old_p, new_p.clone());
                        module.ops[oid]
                            .attributes
                            .insert("param_name".to_string(), Attribute::String(new_p));
                    }
                }
                for &oid in &ops {
                    if let Some(old_ref) = module.ops[oid]
                        .attributes
                        .get("param_ref")
                        .and_then(|a| a.as_str())
                        .map(str::to_string)
                        && let Some(new_ref) = block_scope.get(&old_ref)
                    {
                        module.ops[oid]
                            .attributes
                            .insert("param_ref".to_string(), Attribute::String(new_ref.clone()));
                    }
                }
            }
        }
    }

    // Rewrite all SymbolRef attributes across the module.
    let all_ops: Vec<OpId> = module.ops.keys().collect();
    for oid in all_ops {
        let keys: Vec<String> = module.ops[oid].attributes.keys().cloned().collect();
        for k in keys {
            if let Some(Attribute::SymbolRef(s)) = module.ops[oid].attributes.get_mut(&k) {
                let bare = s.trim_start_matches('@');
                if let Some(mapped) = sym_map.get(bare) {
                    *s = mapped.clone();
                }
            }
        }
    }

    Ok(sym_idx)
}

/// Computes call-graph-aware 64-bit semantic hashes (16-char lowercase hex) for every top-level
/// function in `module` and stamps `"semantic_hash"` onto each function's attributes:
/// - A function's generator hash combines its signature, discardable attributes (excluding
///   `"sym_name"`, `"source_name"`, `"linkage_name"`, and `"semantic_hash"`), and its body's
///   structural hash while masking out direct symbol names of external callees.
/// - Incorporates the resolved `"semantic_hash"` of every called function in call order so two
///   callers are semantically equivalent if and only if both their bodies and their callees'
///   semantic hashes match.
pub fn hash_module_contents(module: &mut Module) -> usize {
    use std::collections::HashMap;

    let func_ids = module.functions();
    if func_ids.is_empty() {
        return 0;
    }

    let mut sym_to_func: HashMap<String, OpId> = HashMap::new();
    let mut gen_hashes: HashMap<String, u64> = HashMap::new();
    let mut call_edges: HashMap<String, Vec<String>> = HashMap::new();

    for &fn_id in &func_ids {
        let Some(sym) = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        sym_to_func.insert(sym.clone(), fn_id);

        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let fn_mix = |acc: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *acc ^= u64::from(b);
                *acc = acc.wrapping_mul(0x0100_0000_01b3);
            }
        };

        let mut attr_parts: Vec<String> = Vec::new();
        for (k, v) in &module.ops[fn_id].attributes {
            if matches!(
                k.as_str(),
                "sym_name" | "source_name" | "linkage_name" | "semantic_hash"
            ) {
                continue;
            }
            attr_parts.push(format!("{k}={v}"));
        }
        attr_parts.sort();
        for part in &attr_parts {
            fn_mix(&mut h, part.as_bytes());
        }

        let mut callees: Vec<String> = Vec::new();
        if let Some(&reg_id) = module.ops[fn_id].regions.first() {
            for &blk_id in &module.regions[reg_id].blocks {
                for &oid in &module.blocks[blk_id].ops {
                    fn_mix(&mut h, module.ops[oid].name.full_name().as_bytes());
                    for (ak, av) in &module.ops[oid].attributes {
                        if let Attribute::SymbolRef(target) = av {
                            let bare = target.trim_start_matches('@').to_string();
                            callees.push(bare);
                            fn_mix(&mut h, ak.as_bytes());
                        } else {
                            fn_mix(&mut h, format!("{ak}={av}").as_bytes());
                        }
                    }
                }
            }
        }

        gen_hashes.insert(sym.clone(), h);
        call_edges.insert(sym, callees);
    }

    // Iteratively propagate callee hashes (handles both DAGs and SCCs deterministically).
    let mut total_hashes: HashMap<String, u64> = gen_hashes.clone();
    for _ in 0..func_ids.len().max(1) {
        let prev = total_hashes.clone();
        for (sym, &base_h) in &gen_hashes {
            let mut h = base_h;
            if let Some(callees) = call_edges.get(sym) {
                for c in callees {
                    let ch = prev.get(c).copied().unwrap_or(0);
                    for &b in &ch.to_le_bytes() {
                        h ^= u64::from(b);
                        h = h.wrapping_mul(0x0100_0000_01b3);
                    }
                }
            }
            total_hashes.insert(sym.clone(), h);
        }
    }

    let mut stamped = 0usize;
    for (sym, fn_id) in sym_to_func {
        if let Some(&h) = total_hashes.get(&sym) {
            module.ops[fn_id].attributes.insert(
                "semantic_hash".to_string(),
                Attribute::String(format!("{h:016x}")),
            );
            stamped += 1;
        }
    }
    stamped
}

/// Applies semantic hash-based naming (`<name_prefix><semantic_hash>`) to `symbols_to_rename`,
/// deduplicating any functions that share an identical `"semantic_hash"`, erasing their
/// `"source_name"` and `"linkage_name"` attributes, and rewriting all `SymbolRef` references across
/// `module`. Returns the map from original symbol name to final canonical symbol name.
pub fn apply_semantic_naming(
    module: &mut Module,
    symbols_to_rename: &[String],
    name_prefix: &str,
) -> Result<IndexMap<String, String>, String> {
    use std::collections::{HashMap, HashSet};

    if symbols_to_rename.is_empty() {
        return Ok(IndexMap::new());
    }

    // Index functions by `sym_name` and `linkage_name`.
    let mut name_to_fn: HashMap<String, OpId> = HashMap::new();
    for fn_id in module.functions() {
        if let Some(s) = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
        {
            name_to_fn.insert(s.to_string(), fn_id);
        }
        if let Some(ln) = module.ops[fn_id]
            .attributes
            .get("linkage_name")
            .and_then(|a| a.as_str())
        {
            name_to_fn.insert(ln.to_string(), fn_id);
        }
    }

    for sym in symbols_to_rename {
        let &fn_id = name_to_fn
            .get(sym)
            .ok_or_else(|| format!("symbol '{sym}' not found in module"))?;
        module.ops[fn_id].attributes.shift_remove("source_name");
    }

    hash_module_contents(module);

    let mut visited_hash_to_sym: HashMap<String, String> = HashMap::new();
    let mut old_sym_to_new_sym: HashMap<String, String> = HashMap::new();
    let mut result_map: IndexMap<String, String> = IndexMap::new();
    let mut dead_fns: HashSet<OpId> = HashSet::new();

    for sym in symbols_to_rename {
        let fn_id = name_to_fn[sym];
        let hash = module.ops[fn_id]
            .attributes
            .get("semantic_hash")
            .and_then(|a| a.as_str())
            .ok_or_else(|| "module has not had semantic hashes computed".to_string())?
            .to_string();
        let old_sym = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or(sym)
            .to_string();

        if let Some(existing_sym) = visited_hash_to_sym.get(&hash) {
            old_sym_to_new_sym.insert(old_sym, existing_sym.clone());
            result_map.insert(sym.clone(), existing_sym.clone());
            dead_fns.insert(fn_id);
        } else {
            let new_sym = format!("{name_prefix}{hash}");
            old_sym_to_new_sym.insert(old_sym, new_sym.clone());
            result_map.insert(sym.clone(), new_sym.clone());
            module.ops[fn_id]
                .attributes
                .insert("sym_name".to_string(), Attribute::String(new_sym.clone()));
            module.ops[fn_id].attributes.shift_remove("linkage_name");
            visited_hash_to_sym.insert(hash, new_sym);
        }
    }

    let all_ops: Vec<OpId> = module.ops.keys().collect();
    for oid in all_ops {
        let keys: Vec<String> = module.ops[oid].attributes.keys().cloned().collect();
        for k in keys {
            if let Some(Attribute::SymbolRef(s)) = module.ops[oid].attributes.get_mut(&k) {
                let bare = s.trim_start_matches('@');
                if let Some(mapped) = old_sym_to_new_sym.get(bare) {
                    *s = mapped.clone();
                }
            }
        }
    }

    if !dead_fns.is_empty() {
        if let Some(root_reg) = module.regions.get(module.root_region)
            && let Some(&root_blk) = root_reg.blocks.first()
        {
            module.blocks[root_blk]
                .ops
                .retain(|id| !dead_fns.contains(id));
        }
        for dead_id in dead_fns {
            module.ops.remove(dead_id);
        }
    }

    Ok(result_map)
}

/// Parsed object-store bucket URI location (`hf://buckets/<namespace>/<bucket>[/<prefix>]`
/// or `bucket://<namespace>/<bucket>[/<prefix>]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketLocation {
    /// Bucket owner namespace.
    pub namespace: String,
    /// Bucket identifier.
    pub bucket: String,
    /// Normalized subpath prefix (stripped of leading and trailing `'/'`).
    pub prefix: String,
}

impl BucketLocation {
    /// Parses a bucket URI into a [`BucketLocation`].
    pub fn parse(uri: &str) -> Result<Self, String> {
        let trimmed = uri.trim();
        let rest = if let Some(r) = trimmed.strip_prefix("hf://buckets/") {
            r
        } else if let Some(r) = trimmed.strip_prefix("bucket://") {
            r
        } else {
            return Err(format!("Unsupported bucket URI scheme: '{uri}'"));
        };
        let clean = rest.trim_matches('/');
        let mut parts = clean.splitn(3, '/');
        let namespace = parts.next().unwrap_or("").trim();
        let bucket = parts.next().unwrap_or("").trim();
        if namespace.is_empty() || bucket.is_empty() {
            return Err(format!(
                "Bucket URI must include non-empty namespace and bucket: '{uri}'"
            ));
        }
        let prefix = parts.next().unwrap_or("").trim_matches('/').to_string();
        Ok(Self {
            namespace: namespace.to_string(),
            bucket: bucket.to_string(),
            prefix,
        })
    }

    /// Returns the `<namespace>/<bucket>` handle.
    #[must_use]
    pub fn handle(&self) -> String {
        format!("{}/{}", self.namespace, self.bucket)
    }

    /// Resolves an encoded object filename into a remote bucket path under `prefix`.
    #[must_use]
    pub fn remote_object_path(&self, encoded_name: &str) -> String {
        let name = encoded_name.trim_start_matches('/');
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}/{name}", self.prefix)
        }
    }
}

/// Encodes an arbitrary object key into a reversible URL-safe filename ending with `.bin`.
#[must_use]
pub fn encode_bucket_object_name(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 12);
    for &b in key.as_bytes() {
        let is_unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if is_unreserved {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out.push_str(".bin");
    out
}

/// Decodes a `.bin` bucket object filename produced by [`encode_bucket_object_name`].
pub fn decode_bucket_object_name(encoded: &str) -> Result<String, String> {
    let stem = encoded
        .strip_suffix(".bin")
        .ok_or_else(|| format!("Bucket object name missing '.bin' suffix: '{encoded}'"))?;
    let bytes = stem.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(format!(
                    "Truncated percent-escape in bucket object name: '{encoded}'"
                ));
            }
            let hex_str = std::str::from_utf8(&bytes[i + 1..i + 3])
                .map_err(|e| format!("Invalid UTF-8 in percent-escape: {e}"))?;
            let val = u8::from_str_radix(hex_str, 16)
                .map_err(|_| format!("Invalid hex digits '%{hex_str}' in bucket object name"))?;
            decoded.push(val);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded)
        .map_err(|e| format!("Decoded bucket object key is not valid UTF-8: {e}"))
}

/// Partial failure descriptor when a multi-file batch upload fails partway through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialBatchStoreError {
    /// Object names that were durably uploaded before the failure occurred.
    pub succeeded_names: Vec<String>,
    /// Object name whose upload failed.
    pub failed_name: String,
    /// Diagnostic reason for the failure.
    pub reason: String,
}

impl PartialBatchStoreError {
    /// Builds a per-item completion mask aligned with `requested_names` so L1 callers
    /// can commit succeeded transfers and fail the remainder.
    #[must_use]
    pub fn build_completion_bitmap(&self, requested_names: &[String]) -> Vec<bool> {
        let succeeded: std::collections::HashSet<&str> =
            self.succeeded_names.iter().map(String::as_str).collect();
        requested_names
            .iter()
            .map(|n| succeeded.contains(n.as_str()))
            .collect()
    }
}

/// TTL-bounded L2 bucket object size cache with periodic amortization of expired-entry pruning.
#[derive(Debug, Clone)]
pub struct BucketObjectMetaCache {
    ttl_secs: f64,
    prune_interval: usize,
    writes_since_prune: usize,
    entries: std::collections::HashMap<String, (u64, f64)>,
}

impl BucketObjectMetaCache {
    /// Creates a new metadata cache with the specified `ttl_secs` and `prune_interval`.
    #[must_use]
    pub fn new(ttl_secs: f64, prune_interval: usize) -> Self {
        Self {
            ttl_secs: ttl_secs.max(0.0),
            prune_interval: prune_interval.max(1),
            writes_since_prune: 0,
            entries: std::collections::HashMap::new(),
        }
    }

    /// Inserts or refreshes `name` with `size_bytes` at `now_secs`, triggering a prune
    /// sweep every `prune_interval` writes.
    pub fn put_size(&mut self, name: impl Into<String>, size_bytes: u64, now_secs: f64) {
        self.entries
            .insert(name.into(), (size_bytes, now_secs + self.ttl_secs));
        self.writes_since_prune += 1;
        if self.writes_since_prune >= self.prune_interval {
            self.prune_expired(now_secs);
            self.writes_since_prune = 0;
        }
    }

    /// Looks up the cached object size for `name` at `now_secs`, evicting on expiration.
    pub fn get_size(&mut self, name: &str, now_secs: f64) -> Option<u64> {
        if let Some(&(size, expires_at)) = self.entries.get(name) {
            if now_secs <= expires_at {
                return Some(size);
            }
            self.entries.remove(name);
        }
        None
    }

    /// Removes `name` from the metadata cache.
    pub fn invalidate(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    /// Sweeps all entries whose expiration timestamp is strictly before `now_secs`.
    pub fn prune_expired(&mut self, now_secs: f64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, &mut (_, expires_at)| now_secs <= expires_at);
        before.saturating_sub(self.entries.len())
    }

    /// Returns the current number of cached entries (including not-yet-swept expired entries).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries are currently stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
