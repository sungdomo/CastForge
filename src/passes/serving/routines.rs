//! Device ref canonicalization, kernel stub merging, and model callable outlining.

#[allow(unused_imports)]
use super::*;
use crate::ir::{Attribute, Location, Module, OpId, Operation, Type, Value, ValueDef, ValueId};
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

pub(super) fn canonicalize_type_device_refs(ty: &mut Type) -> usize {
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

pub(super) fn canonicalize_attr_for_merge(key: &str, attr: &mut Attribute) -> usize {
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
