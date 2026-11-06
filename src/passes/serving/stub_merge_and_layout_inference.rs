//! Kernel stub canonicalization and deterministic bottom-up deduplication,
//! constant resource table indexing, compiled routine lowering, and
//! bidirectional tensor layout inference with device architecture validation.
//!
//! Covers:
//! 1. **Kernel Stub Canonicalization & Deduplication
//!    (`canonicalize_kernel_stubs_for_merge`, `deduplicate_kernel_stubs_bottom_up`)**:
//!    Normalizes all `device_ref` device IDs inside kernel stubs to canonical
//!    `id = 0` (`"<label>:0"`) while preserving the device label, canonicalizes
//!    symbolic dimension names in `shape_attr` attributes to `"dyn"`, and
//!    deduplicates structurally identical kernel stubs modulo `sym_name`,
//!    `fusion_info`, locations, and device ordinals, selecting the
//!    lexicographically smallest symbol name for deterministic output and
//!    fusing locations across merged stubs.
//! 2. **Constant Resource Indexing & Compiled Routine Lowering
//!    (`replace_buffer_constants_with_resources`, `lower_compiled_routines_with_init_guards`)**:
//!    Replaces `mgp.buffer.constant` inside kernel stubs with
//!    `mgp.resource.constant` indexed by its position in `constant_resources`
//!    (then replaces array entries with `"unit"` placeholders), and lowers
//!    `mogg.compiled_load` / `mogg.compiled_call` into `mgp.module.load` /
//!    `mgp.execute_generic` with per-module `mgp.guard` sequencing in `init`.
//! 3. **Bidirectional Tensor Layout Inference & Device Architecture Validation
//!    (`propagate_bidirectional_tensor_layouts`, `validate_module_target_arch_dtypes`)**:
//!    Seeds layout constraints from 2D/3D `mo.conv`, `mo.conv_transpose`, and
//!    `mo.matmul`, propagates constraints bidirectionally across `mo.guard`,
//!    `mo.transfer`, and elementwise operations, and validates tensor element
//!    types against target CPU/accelerator architectures (`x86_64`, `aarch64`,
//!    `gpu`).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

use crate::ir::mlir::{
    Attribute, DType, Location, Module, OpId, OpName, Operation, TensorType, Type, ValueDef,
    ValueId,
};
use crate::passes::serving::kv_attention::fuse_locations;

fn make_op_name(full: &str) -> OpName {
    if let Some((dialect, name)) = full.split_once('.') {
        OpName::new(dialect, name)
    } else {
        OpName::new("builtin", full)
    }
}

fn canonicalize_device_ref_str(raw: &str) -> String {
    if let Some((label, _id)) = raw.split_once(':') {
        format!("{label}:0")
    } else {
        format!("{raw}:0")
    }
}

// ---------------------------------------------------------------------------
// 1. Kernel Stub Canonicalization & Deterministic Deduplication
// ---------------------------------------------------------------------------

/// Canonicalizes `func.func` / `mogg.func` kernel stubs in `module` for
/// structural deduplication:
/// - Rewrites `device_ref` attributes and tensor `encoding` device strings to
///   canonical ordinal `0` (`"<label>:0"`).
/// - Replaces non-static symbolic dimension strings in `shape_attr` array
///   attributes on body operations with `"dyn"`.
///
/// Returns the number of kernel stubs canonicalized.
pub fn canonicalize_kernel_stubs_for_merge(module: &mut Module) -> usize {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();
    let mut canonicalized = 0usize;

    for op_id in top_ops {
        let Some(op) = module.ops.get_mut(op_id) else {
            continue;
        };
        let full = op.name.full_name();
        if full != "func.func" && full != "mogg.func" {
            continue;
        }
        for key in ["device_ref", "mo.device_ref"] {
            if let Some(Attribute::String(dev)) = op.attributes.get_mut(key) {
                *dev = canonicalize_device_ref_str(dev);
            }
        }
        let regions = op.regions.clone();
        for reg_id in regions {
            let Some(reg) = module.regions.get(reg_id) else {
                continue;
            };
            let blocks = reg.blocks.clone();
            for blk_id in blocks {
                let args = module
                    .blocks
                    .get(blk_id)
                    .map(|b| b.arguments.clone())
                    .unwrap_or_default();
                for arg_id in args {
                    if let Some(val) = module.values.get_mut(arg_id)
                        && let Type::Tensor(TensorType {
                            encoding: Some(dev),
                            ..
                        }) = &mut val.ty
                        && dev.contains(':')
                    {
                        *dev = canonicalize_device_ref_str(dev);
                    }
                }
                let child_ops = module
                    .blocks
                    .get(blk_id)
                    .map(|b| b.ops.clone())
                    .unwrap_or_default();
                for child_id in child_ops {
                    if let Some(child_op) = module.ops.get_mut(child_id) {
                        for key in ["device_ref", "mo.device_ref"] {
                            if let Some(Attribute::String(dev)) = child_op.attributes.get_mut(key) {
                                *dev = canonicalize_device_ref_str(dev);
                            }
                        }
                        if let Some(Attribute::Array(dims)) =
                            child_op.attributes.get_mut("shape_attr")
                        {
                            for item in dims {
                                if let Attribute::String(sym) = item {
                                    *sym = "dyn".to_string();
                                }
                            }
                        }
                        let results: Vec<ValueId> = child_op.results.iter().copied().collect();
                        for r_id in results {
                            if let Some(val) = module.values.get_mut(r_id)
                                && let Type::Tensor(TensorType {
                                    encoding: Some(dev),
                                    ..
                                }) = &mut val.ty
                                && dev.contains(':')
                            {
                                *dev = canonicalize_device_ref_str(dev);
                            }
                        }
                    }
                }
            }
        }
        canonicalized += 1;
    }

    canonicalized
}

/// Computes a structural equivalence fingerprint for a kernel stub (`func.func`
/// or `mogg.func`) that ignores `sym_name`, `fusion_info`, source locations,
/// and device ordinal numbers on `device_ref`.
fn compute_stub_equivalence_signature(module: &Module, func_id: OpId) -> String {
    let Some(func_op) = module.ops.get(func_id) else {
        return String::new();
    };
    let mut sig = String::new();
    sig.push_str(&func_op.name.full_name());
    sig.push('{');

    // Sort attributes for deterministic comparison, ignoring sym_name and fusion_info.
    let mut sorted_attrs: BTreeMap<&str, String> = BTreeMap::new();
    for (k, v) in &func_op.attributes {
        if k == "sym_name" || k == "fusion_info" || k == "fusionInfo" {
            continue;
        }
        if (k == "device_ref" || k == "mo.device_ref")
            && let Attribute::String(dev) = v
        {
            let label = dev.split_once(':').map_or(dev.as_str(), |(l, _)| l);
            sorted_attrs.insert(k, format!("dev_label({label})"));
            continue;
        }
        sorted_attrs.insert(k, format!("{v:?}"));
    }
    for (k, v) in sorted_attrs {
        sig.push_str(k);
        sig.push('=');
        sig.push_str(&v);
        sig.push(';');
    }
    sig.push('}');

    // Number SSA values canonically inside the function body.
    let mut val_numbering: HashMap<ValueId, usize> = HashMap::new();
    let mut next_val_idx = 0usize;

    for &reg_id in &func_op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            sig.push('(');
            for &arg_id in &blk.arguments {
                let idx = next_val_idx;
                next_val_idx += 1;
                val_numbering.insert(arg_id, idx);
                if let Some(v) = module.values.get(arg_id) {
                    sig.push_str(&format!("%{idx}:{:?},", v.ty));
                }
            }
            sig.push_str("){");
            for &child_id in &blk.ops {
                let Some(child_op) = module.ops.get(child_id) else {
                    continue;
                };
                sig.push_str(&child_op.name.full_name());
                sig.push('[');
                for &opd in &child_op.operands {
                    if let Some(&idx) = val_numbering.get(&opd) {
                        sig.push_str(&format!("%{idx},"));
                    } else {
                        sig.push_str("%ext,");
                    }
                }
                sig.push_str("]{");
                let mut child_attrs: BTreeMap<&str, String> = BTreeMap::new();
                for (k, v) in &child_op.attributes {
                    child_attrs.insert(k, format!("{v:?}"));
                }
                for (k, v) in child_attrs {
                    sig.push_str(k);
                    sig.push('=');
                    sig.push_str(&v);
                    sig.push(';');
                }
                sig.push_str("}->(");
                for &res in &child_op.results {
                    let idx = next_val_idx;
                    next_val_idx += 1;
                    val_numbering.insert(res, idx);
                    if let Some(v) = module.values.get(res) {
                        sig.push_str(&format!("%{idx}:{:?},", v.ty));
                    }
                }
                sig.push_str(");");
            }
            sig.push('}');
        }
    }

    sig
}

/// Deduplicates structurally identical kernel stubs (`func.func` / `mogg.func`)
/// in `module` modulo `sym_name`, `fusion_info`, source locations, and device
/// ordinal numbers:
/// - Canonicalizes callee references in bottom-up call order so caller stubs
///   calling equivalent callees also become identical and merge.
/// - Picks the lexicographically smallest symbol name among equivalent stubs
///   for deterministic output, fuses all merged stubs' locations onto the
///   surviving stub, rewrites all call sites across the module, and erases
///   duplicate stubs.
///
/// Returns the number of duplicate kernel stubs removed.
pub fn deduplicate_kernel_stubs_bottom_up(module: &mut Module) -> usize {
    let root_blk = module.body_block();
    let mut removed_total = 0usize;

    // Iterate to a fixed point so multi-level call-graph chains merge bottom-up.
    loop {
        let top_ops = module
            .blocks
            .get(root_blk)
            .map(|b| b.ops.clone())
            .unwrap_or_default();

        // Group kernel stubs by their structural equivalence signature.
        let mut groups: BTreeMap<String, Vec<(String, OpId)>> = BTreeMap::new();
        for &op_id in &top_ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let full = op.name.full_name();
            if full != "func.func" && full != "mogg.func" {
                continue;
            }
            let Some(Attribute::String(sym_name)) = op.attributes.get("sym_name").cloned() else {
                continue;
            };
            let sig = compute_stub_equivalence_signature(module, op_id);
            groups.entry(sig).or_default().push((sym_name, op_id));
        }

        let mut sym_replacements: HashMap<String, String> = HashMap::new();
        let mut dead_stub_ids: HashSet<OpId> = HashSet::new();

        for (_sig, mut members) in groups {
            if members.len() <= 1 {
                continue;
            }
            // Sort by symbol name and pick the lexicographically smallest candidate.
            members.sort_by(|a, b| a.0.cmp(&b.0));
            let (winner_sym, winner_id) = members[0].clone();

            let mut merged_loc = module
                .ops
                .get(winner_id)
                .map(|op| op.location.clone())
                .unwrap_or(Location::Unknown);

            for (dup_sym, dup_id) in members.into_iter().skip(1) {
                if let Some(dup_op) = module.ops.get(dup_id) {
                    merged_loc = fuse_locations(&[merged_loc, dup_op.location.clone()]);
                }
                sym_replacements.insert(dup_sym, winner_sym.clone());
                dead_stub_ids.insert(dup_id);
            }

            if let Some(winner_op) = module.ops.get_mut(winner_id) {
                winner_op.location = merged_loc;
            }
        }

        if dead_stub_ids.is_empty() {
            break;
        }

        // Rewrite all call-site attributes across the entire module.
        let all_op_ids: Vec<OpId> = module.ops.keys().collect();
        for oid in all_op_ids {
            let Some(op) = module.ops.get_mut(oid) else {
                continue;
            };
            for key in ["callee", "callee_symbol", "symbol"] {
                match op.attributes.get_mut(key) {
                    Some(Attribute::String(s)) => {
                        let bare = s.trim_start_matches('@');
                        if let Some(new_sym) = sym_replacements.get(bare) {
                            *s = new_sym.clone();
                        }
                    }
                    Some(Attribute::SymbolRef(s)) => {
                        let bare = s.trim_start_matches('@');
                        if let Some(new_sym) = sym_replacements.get(bare) {
                            *s = new_sym.clone();
                        }
                    }
                    _ => {}
                }
            }
        }

        for dead_id in dead_stub_ids {
            module.remove_op(root_blk, dead_id);
            removed_total += 1;
        }
    }

    removed_total
}

// ---------------------------------------------------------------------------
// 2. Constant Resource Indexing & Compiled Routine Lowering
// ---------------------------------------------------------------------------

/// Replaces `mgp.buffer.constant` operations inside kernel stubs with
/// `mgp.resource.constant` operations indexed by their position in the stub's
/// `constant_resources` array attribute, then replaces the array elements with
/// `"unit"` placeholders preserving the resource count.
///
/// # Errors
///
/// Returns `Err` if a `mgp.buffer.constant` value inside a kernel stub is not
/// present in that stub's `constant_resources` array.
pub fn replace_buffer_constants_with_resources(module: &mut Module) -> Result<usize, String> {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();
    let mut replaced_count = 0usize;

    for stub_id in top_ops {
        let Some(stub_op) = module.ops.get(stub_id) else {
            continue;
        };
        let full = stub_op.name.full_name();
        if full != "func.func" && full != "mogg.func" {
            continue;
        }
        let Some(Attribute::Array(const_resources)) =
            stub_op.attributes.get("constant_resources").cloned()
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
                    if child_op.name.full_name() != "mgp.buffer.constant" {
                        continue;
                    }
                    let Some(val_attr) = child_op.attributes.get("value").cloned() else {
                        return Err("mgp.buffer.constant is missing 'value' attribute".to_string());
                    };
                    let Some(res_index) = const_resources.iter().position(|item| *item == val_attr)
                    else {
                        return Err(
                            "The constant resource can't be found in the kernel".to_string()
                        );
                    };

                    if let Some(child_mut) = module.ops.get_mut(child_id) {
                        child_mut.name = make_op_name("mgp.resource.constant");
                        child_mut.attributes.clear();
                        child_mut.attributes.insert(
                            "resource_index".to_string(),
                            Attribute::Int(res_index as i64),
                        );
                        replaced_count += 1;
                    }
                }
            }
        }

        // Replace `constant_resources` entries with unit placeholders of the same length.
        let placeholders = vec![Attribute::String("unit".to_string()); const_resources.len()];
        if let Some(stub_mut) = module.ops.get_mut(stub_id) {
            stub_mut.attributes.insert(
                "constant_resources".to_string(),
                Attribute::Array(placeholders),
            );
        }
    }

    Ok(replaced_count)
}

/// Converts `!mo.opaque<...>` types to `!mgp.opaque<...>`.
fn convert_mo_opaque_to_mgp_opaque(ty: &Type) -> Type {
    if let Type::Opaque(raw) = ty
        && let Some(rest) = raw.strip_prefix("!mo.opaque")
    {
        return Type::Opaque(format!("!mgp.opaque{rest}"));
    }
    ty.clone()
}

/// Lowers compiled JIT routines inside `mgp.model` operations:
/// - Rewrites `mogg.compiled_load` in the `init` block into `mgp.module.load`
///   sequenced after the model's `mgp.runtime.create`, assigning `(module_index,
///   kernel_index)` pairs to all loaded kernel symbols.
/// - Replaces `mgp.module.load.placeholder` in `init` with `grt.chain.create`
///   over all runtime/module-load chains.
/// - Rewrites `mogg.compiled_call` / `castforge.compiled_call` in both `init`
///   and `execute` blocks into `mgp.execute_generic` with resolved
///   `(module_index, kernel_index)` and `!mo.opaque -> !mgp.opaque` result
///   types.
/// - Routes any chain-less `init`-block `mgp.execute_generic` context through
///   an `mgp.guard` on the `mgp.module.load` chain of the specific module it
///   calls into.
///
/// # Errors
///
/// Returns `Err` if a `compiled_call` references a kernel symbol not present in
/// any `mogg.compiled_load` archive.
pub fn lower_compiled_routines_with_init_guards(module: &mut Module) -> Result<usize, String> {
    let root_blk = module.body_block();
    let top_ops = module
        .blocks
        .get(root_blk)
        .map(|b| b.ops.clone())
        .unwrap_or_default();
    let mut lowered_calls = 0usize;

    for model_id in top_ops {
        let is_model = module
            .ops
            .get(model_id)
            .is_some_and(|op| op.name.full_name() == "mgp.model" && op.regions.len() >= 2);
        if !is_model {
            continue;
        }
        let init_reg = module.ops[model_id].regions[0];
        let exec_reg = module.ops[model_id].regions[1];
        let Some(&init_blk) = module.regions[init_reg].blocks.first() else {
            continue;
        };
        let Some(&exec_blk) = module.regions[exec_reg].blocks.first() else {
            continue;
        };

        // Find `mgp.runtime.create` in `init_blk`.
        let mut runtime_slot = 0i64;
        let mut runtime_ctx_val: Option<ValueId> = None;
        let mut module_load_chains: SmallVec<[ValueId; 4]> = SmallVec::new();
        for &oid in &module.blocks[init_blk].ops {
            let Some(op) = module.ops.get(oid) else {
                continue;
            };
            if op.name.full_name() == "mgp.runtime.create" {
                if let Some(&Attribute::Int(s)) = op
                    .attributes
                    .get("slot")
                    .or_else(|| op.attributes.get("a_runtime_slot"))
                {
                    runtime_slot = s;
                }
                if let Some(&ctx_v) = op.operands.first() {
                    runtime_ctx_val = Some(ctx_v);
                }
                if let Some(&res_chain) = op.results.first() {
                    module_load_chains.push(res_chain);
                }
            }
        }
        let default_init_ctx = runtime_ctx_val
            .or_else(|| module.blocks[init_blk].arguments.first().copied())
            .ok_or_else(|| "init block has no context value".to_string())?;

        // Rewrite `mogg.compiled_load` -> `mgp.module.load` and index kernel symbols.
        let mut kernel_to_mod_fn: HashMap<String, (usize, usize)> = HashMap::new();
        let mut module_load_chain_by_idx: Vec<ValueId> = Vec::new();
        let mut device_module_idx = 0usize;

        let init_ops = module.blocks[init_blk].ops.clone();
        for oid in init_ops {
            let Some(op) = module.ops.get(oid).cloned() else {
                continue;
            };
            if op.name.full_name() != "mogg.compiled_load" {
                continue;
            }
            if let Some(Attribute::Array(syms)) = op.attributes.get("symbols") {
                for (k_idx, item) in syms.iter().enumerate() {
                    if let Attribute::String(sym) = item {
                        kernel_to_mod_fn.insert(
                            sym.trim_start_matches('@').to_string(),
                            (device_module_idx, k_idx),
                        );
                    }
                }
            }
            if let Some(op_mut) = module.ops.get_mut(oid) {
                op_mut.name = make_op_name("mgp.module.load");
                op_mut
                    .attributes
                    .insert("runtime_slot".to_string(), Attribute::Int(runtime_slot));
                op_mut.attributes.insert(
                    "module_index".to_string(),
                    Attribute::Int(device_module_idx as i64),
                );
                op_mut
                    .attributes
                    .insert("device".to_string(), Attribute::String("cpu:0".to_string()));
                if let Some(&res_chain) = op_mut.results.first() {
                    module_load_chains.push(res_chain);
                    module_load_chain_by_idx.push(res_chain);
                }
            }
            device_module_idx += 1;
        }

        // Replace `mgp.module.load.placeholder` with `grt.chain.create` (or single chain).
        let init_ops_after_load = module.blocks[init_blk].ops.clone();
        for oid in init_ops_after_load {
            let is_ph = module
                .ops
                .get(oid)
                .is_some_and(|op| op.name.full_name() == "mgp.module.load.placeholder");
            if !is_ph {
                continue;
            }
            if module_load_chains.len() == 1 {
                let single_chain = module_load_chains[0];
                let ph_res = module.ops[oid].results.first().copied();
                if let Some(r) = ph_res {
                    module.replace_all_uses(r, single_chain);
                }
                module.remove_op(init_blk, oid);
            } else if let Some(op_mut) = module.ops.get_mut(oid) {
                op_mut.name = make_op_name("grt.chain.create");
                op_mut.operands = module_load_chains.clone();
            }
        }

        // Helper closure to rewrite a compiled_call in `blk_id`.
        let exec_ctx_val = module.blocks[exec_blk]
            .arguments
            .iter()
            .copied()
            .find(|&vid| {
                matches!(
                    &module.values[vid].ty,
                    Type::Opaque(s) if s.contains("context")
                )
            })
            .or_else(|| module.blocks[exec_blk].arguments.first().copied())
            .unwrap_or(default_init_ctx);

        for (blk_id, ctx_val, is_init_block) in [
            (init_blk, default_init_ctx, true),
            (exec_blk, exec_ctx_val, false),
        ] {
            let blk_ops = module.blocks[blk_id].ops.clone();
            let mut new_blk_ops = Vec::with_capacity(blk_ops.len() + 2);

            for oid in blk_ops {
                let Some(op) = module.ops.get(oid).cloned() else {
                    continue;
                };
                let full = op.name.full_name();
                if full != "mogg.compiled_call" && full != "castforge.compiled_call" {
                    new_blk_ops.push(oid);
                    continue;
                }
                let callee = match op
                    .attributes
                    .get("callee_symbol")
                    .or_else(|| op.attributes.get("callee"))
                {
                    Some(Attribute::String(s)) => s.trim_start_matches('@').to_string(),
                    Some(Attribute::SymbolRef(s)) => s.trim_start_matches('@').to_string(),
                    _ => return Err(format!("{full} missing 'callee_symbol' attribute")),
                };
                let Some(&(mod_idx, fn_idx)) = kernel_to_mod_fn.get(&callee) else {
                    return Err(format!(
                        "compiled kernel '{callee}' is not present in any loaded module"
                    ));
                };

                // If in `init` block and the call has no chain operand, gate its context
                // through `mgp.guard` on `module_load_chain_by_idx[mod_idx]`.
                let mut effective_ctx = ctx_val;
                let has_chain_opd = op.operands.iter().any(
                    |&v| matches!(&module.values[v].ty, Type::Opaque(s) if s.contains("chain")),
                );
                if is_init_block
                    && !has_chain_opd
                    && let Some(&load_chain) = module_load_chain_by_idx.get(mod_idx)
                {
                    let ctx_ty = module.values[ctx_val].ty.clone();
                    let guard_id = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: make_op_name("mgp.guard"),
                        operands: smallvec![ctx_val, load_chain],
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: op.location.clone(),
                    });
                    let guarded_ctx = module.values.insert_with_key(|vid| crate::ir::mlir::Value {
                        id: vid,
                        ssa_name: None,
                        ty: ctx_ty,
                        def: ValueDef::OpResult {
                            op: guard_id,
                            index: 0,
                        },
                    });
                    module.ops[guard_id].results.push(guarded_ctx);
                    new_blk_ops.push(guard_id);
                    effective_ctx = guarded_ctx;
                }

                // Convert `!mo.opaque` result types to `!mgp.opaque`.
                for &res_id in &op.results {
                    if let Some(val_mut) = module.values.get_mut(res_id) {
                        val_mut.ty = convert_mo_opaque_to_mgp_opaque(&val_mut.ty);
                    }
                }

                let is_async = op
                    .attributes
                    .get("dispatch")
                    .and_then(Attribute::as_str)
                    .is_some_and(|d| {
                        d.eq_ignore_ascii_case("deferred") || d.eq_ignore_ascii_case("async")
                    });

                if let Some(op_mut) = module.ops.get_mut(oid) {
                    op_mut.name = make_op_name("mgp.execute_generic");
                    op_mut.operands.insert(0, effective_ctx);
                    op_mut
                        .attributes
                        .insert("runtime_slot".to_string(), Attribute::Int(runtime_slot));
                    op_mut
                        .attributes
                        .insert("module_index".to_string(), Attribute::Int(mod_idx as i64));
                    op_mut
                        .attributes
                        .insert("kernel_index".to_string(), Attribute::Int(fn_idx as i64));
                    op_mut
                        .attributes
                        .insert("is_async".to_string(), Attribute::Bool(is_async));
                }
                new_blk_ops.push(oid);
                lowered_calls += 1;
            }

            module.blocks[blk_id].ops = new_blk_ops;
        }
    }

    Ok(lowered_calls)
}

// ---------------------------------------------------------------------------
// 3. Bidirectional Tensor Layout Inference & Architecture DType Validation
// ---------------------------------------------------------------------------

fn is_elementwise_or_same_layout_op(full: &str) -> bool {
    matches!(
        full,
        "mo.transfer"
            | "mo.add"
            | "mo.sub"
            | "mo.mul"
            | "mo.div"
            | "mo.relu"
            | "mo.tanh"
            | "mo.exp"
            | "mo.neg"
            | "mo.abs"
            | "stablehlo.add"
            | "stablehlo.multiply"
    )
}

/// Seeds tensor layout constraints from `mo.conv`, `mo.conv_transpose`, and
/// `mo.matmul` operations in `module`, propagates them bidirectionally across
/// `mo.guard`, `mo.transfer`, and elementwise operations, and assigns the
/// inferred layouts to `TensorType::encoding`.
///
/// Values that already carry a non-empty `encoding` layout are left untouched
/// so the pass is idempotent.
///
/// # Errors
///
/// Returns `Err` if a `mo.conv` or `mo.conv_transpose` has unsupported rank
/// (must be 4D or 5D) or if conflicting layout constraints meet on the same
/// SSA value.
pub fn propagate_bidirectional_tensor_layouts(module: &mut Module) -> Result<usize, String> {
    let mut inferred: HashMap<ValueId, String> = HashMap::new();

    let constrain = |inferred: &mut HashMap<ValueId, String>,
                     module: &Module,
                     vid: ValueId,
                     layout: &str|
     -> Result<bool, String> {
        let Some(val) = module.values.get(vid) else {
            return Ok(false);
        };
        let Type::Tensor(tt) = &val.ty else {
            return Ok(false);
        };
        if tt.encoding.is_some() {
            return Ok(false);
        }
        if let Some(existing) = inferred.get(&vid) {
            if existing != layout {
                return Err(format!(
                    "attempting to constrain value to layout: {layout} when it's already \
                     constrained to layout: {existing}"
                ));
            }
            return Ok(false);
        }
        inferred.insert(vid, layout.to_string());
        Ok(true)
    };

    // Step 1: Seed constraints from `mo.conv`, `mo.conv_transpose`, and `mo.matmul`.
    for &op_id in &module.live_ops() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let full = op.name.full_name();
        match full.as_str() {
            "mo.conv" | "mo.conv_transpose" => {
                if op.operands.len() < 2 {
                    return Err(format!("{full} requires input and filter operands"));
                }
                let input_v = op.operands[0];
                let filter_v = op.operands[1];
                let in_rank = match &module.values[input_v].ty {
                    Type::Tensor(tt) => tt.shape.len(),
                    _ => 0,
                };
                let flt_rank = match &module.values[filter_v].ty {
                    Type::Tensor(tt) => tt.shape.len(),
                    _ => 0,
                };
                let (in_layout, flt_layout) = match (full.as_str(), in_rank, flt_rank) {
                    ("mo.conv", 4, 4) => ("NHWC", "FCRS"),
                    ("mo.conv", 5, 5) => ("NDHWC", "FCQRS"),
                    ("mo.conv_transpose", 4, 4) => ("NHWC", "RSFC"),
                    ("mo.conv_transpose", 5, 5) => ("NDHWC", "QRSFC"),
                    _ => {
                        return Err(format!("Only support 2D and 3D layout for {full}"));
                    }
                };
                constrain(&mut inferred, module, input_v, in_layout)?;
                constrain(&mut inferred, module, filter_v, flt_layout)?;
            }
            "mo.matmul" if op.operands.len() >= 2 => {
                let weights_v = op.operands[1];
                constrain(&mut inferred, module, weights_v, "KN")?;
            }
            _ => {}
        }
    }

    // Build value -> user ops map for bidirectional propagation.
    let mut users_of_val: HashMap<ValueId, Vec<OpId>> = HashMap::new();
    for &oid in &module.live_ops() {
        if let Some(op) = module.ops.get(oid) {
            for &opd in &op.operands {
                users_of_val.entry(opd).or_default().push(oid);
            }
        }
    }

    // Step 2: Propagate layouts through `mo.guard`, `mo.transfer`, and elementwise ops.
    let mut worklist: VecDeque<ValueId> = inferred.keys().copied().collect();
    let mut in_history: HashSet<ValueId> = inferred.keys().copied().collect();

    while let Some(vid) = worklist.pop_front() {
        let Some(layout) = inferred.get(&vid).cloned() else {
            continue;
        };
        let mut affected_ops: Vec<OpId> = users_of_val.get(&vid).cloned().unwrap_or_default();
        if let Some(val) = module.values.get(vid)
            && let ValueDef::OpResult { op: def_op, .. } = val.def
        {
            affected_ops.push(def_op);
        }

        for oid in affected_ops {
            let Some(op) = module.ops.get(oid) else {
                continue;
            };
            let full = op.name.full_name();
            let mut propagate_vals: Vec<ValueId> = Vec::new();
            if full == "mo.guard" {
                propagate_vals.extend(op.operands.iter().skip(1).copied());
                propagate_vals.extend(op.results.iter().copied());
            } else if is_elementwise_or_same_layout_op(&full) {
                propagate_vals.extend(op.operands.iter().copied());
                propagate_vals.extend(op.results.iter().copied());
            }

            for target_v in propagate_vals {
                constrain(&mut inferred, module, target_v, &layout)?;
                if in_history.insert(target_v) {
                    worklist.push_back(target_v);
                }
            }
        }
    }

    // Step 3: Assign inferred layouts to `TensorType::encoding`.
    let count = inferred.len();
    for (vid, layout) in inferred {
        if let Some(val_mut) = module.values.get_mut(vid)
            && let Type::Tensor(tt) = &mut val_mut.ty
        {
            tt.encoding = Some(layout);
        }
    }
    Ok(count)
}

/// Returns `true` if `dtype` is supported for computation/casting on `target_arch`:
/// - `x86_64`: supports `BF16`, `F16`, floats $\ge 32$ bits, and all non-float types;
///   rejects sub-32-bit floats other than `BF16`/`F16` (e.g. `F8E4M3FN`, `F8E5M2`).
/// - `aarch64`: supports `BF16`, floats $\ge 32$ bits, and all non-float types;
///   rejects `F16` and 8-bit floats.
/// - GPU / accelerator architectures (`"nvptx64"`, `"amdgcn"`, `"gpu"`): support all dtypes.
#[must_use]
pub fn is_supported_target_arch_dtype(target_arch: &str, dtype: &DType) -> bool {
    let arch = target_arch
        .split('-')
        .next()
        .unwrap_or(target_arch)
        .trim()
        .to_ascii_lowercase();
    if arch == "x86_64" || arch == "aarch64" {
        if *dtype == DType::BF16 {
            return true;
        }
        if arch == "x86_64" && *dtype == DType::F16 {
            return true;
        }
        if dtype.is_float() && dtype.bit_width() < 32 {
            return false;
        }
    }
    true
}

/// Validates that all tensor operands and results in `module` (excluding
/// operations marked with `dtype_agnostic = true`) use element types supported
/// on `target_arch`.
///
/// # Errors
///
/// Returns `Err` on the first unsupported `(dtype, device)` combination.
pub fn validate_module_target_arch_dtypes(
    module: &Module,
    target_arch: &str,
) -> Result<(), String> {
    if target_arch.trim().is_empty() {
        return Ok(());
    }
    for op_id in module.live_ops() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if matches!(
            op.attributes.get("dtype_agnostic"),
            Some(Attribute::Bool(true))
        ) {
            continue;
        }
        let op_dev = op
            .attributes
            .get("device_ref")
            .or_else(|| op.attributes.get("device"))
            .and_then(Attribute::as_str)
            .unwrap_or("cpu:0");

        for &vid in op.operands.iter().chain(op.results.iter()) {
            let Some(val) = module.values.get(vid) else {
                continue;
            };
            let Type::Tensor(tt) = &val.ty else {
                continue;
            };
            if !is_supported_target_arch_dtype(target_arch, &tt.dtype) {
                let dev = tt.encoding.as_deref().unwrap_or(op_dev);
                return Err(format!(
                    "The {} data type is not supported on device '{dev}'.",
                    tt.dtype
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_dtype_support_rules() {
        assert!(is_supported_target_arch_dtype(
            "x86_64-unknown-linux-gnu",
            &DType::F16
        ));
        assert!(is_supported_target_arch_dtype("x86_64", &DType::BF16));
        assert!(!is_supported_target_arch_dtype("x86_64", &DType::F8E4M3FN));

        assert!(is_supported_target_arch_dtype(
            "aarch64-apple-darwin",
            &DType::BF16
        ));
        assert!(!is_supported_target_arch_dtype(
            "aarch64-apple-darwin",
            &DType::F16
        ));
        assert!(is_supported_target_arch_dtype(
            "nvptx64-nvidia-cuda",
            &DType::F8E4M3FN
        ));
    }
}
