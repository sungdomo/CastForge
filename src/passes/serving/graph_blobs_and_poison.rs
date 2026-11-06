//! Declarative routine graph-blob fusion (`FormGraphBlobs`), relocatable compilation cache-key
//! hashing (`RelocatableCacheKeyHasher`), and dtype-aware debug allocation poisoning
//! (`DebugAllocPoisonTracker`).

use crate::ir::{
    Attribute, Block, BlockId, DType, Location, Module, OpId, OpName, Operation, Region, Type,
    Value, ValueDef, ValueId,
};
use crate::passes::normalize::StripDebugInfoPass;
use crate::passes::registry::Pass;
use crate::passes::serving::kv_attention::fuse_locations;
use crate::passes::serving::module_hashing::normalize_all_names;
use indexmap::{IndexMap, IndexSet};
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

// ============================================================================
// Declarative Routine Graph-Blob Fusion (`FormGraphBlobs`)
// ============================================================================

fn is_effect_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Opaque(s) if s.contains("chain") || s.contains("buffer") || s.contains("token")
    )
}

fn is_terminator_op(name: &str) -> bool {
    matches!(
        name,
        "func.return"
            | "return"
            | "mo.output"
            | "mo.yield"
            | "scf.yield"
            | "mogg.routine.return"
            | "mgp.device_graph.staging.yield"
            | "mgp.device_graph.collective.yield"
    )
}

fn collect_read_values(module: &Module, op_id: OpId) -> IndexSet<ValueId> {
    let mut read_vals: IndexSet<ValueId> = IndexSet::new();
    let Some(op) = module.ops.get(op_id) else {
        return read_vals;
    };
    for &opd in &op.operands {
        read_vals.insert(opd);
    }
    let mut defined_inside: IndexSet<ValueId> = IndexSet::new();
    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &arg in &blk.arguments {
                defined_inside.insert(arg);
            }
            for &inner_id in &blk.ops {
                if let Some(inner_op) = module.ops.get(inner_id) {
                    for &res in &inner_op.results {
                        defined_inside.insert(res);
                    }
                }
            }
        }
    }
    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &inner_id in &blk.ops {
                if let Some(inner_op) = module.ops.get(inner_id) {
                    for &opd in &inner_op.operands {
                        if !defined_inside.contains(&opd) {
                            read_vals.insert(opd);
                        }
                    }
                }
            }
        }
    }
    read_vals
}

/// Identifies SSA values in `blk_id` that are execution-invariant across runs (constants without
/// external mutable aliases, and non-branching/non-parallel operations without effect types whose
/// read values are all execution-invariant).
pub fn find_invariant_values(module: &Module, blk_id: BlockId) -> IndexSet<ValueId> {
    let mut invariant: IndexSet<ValueId> = IndexSet::new();
    let Some(blk) = module.blocks.get(blk_id) else {
        return invariant;
    };
    for &op_id in &blk.ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let full = op.name.full_name();
        let is_const = full.ends_with(".constant")
            || full == "stablehlo.constant"
            || full == "mo.constant"
            || full == "mo.constant.external";
        if is_const {
            let has_alias = matches!(op.attributes.get("has_alias"), Some(Attribute::Bool(true)));
            if full != "mo.constant.external" || !has_alias {
                for &res in &op.results {
                    invariant.insert(res);
                }
            }
            continue;
        }
        let is_branch_or_parallel = matches!(
            full.as_str(),
            "mo.if" | "scf.if" | "mo.while" | "scf.while" | "mo.parallel"
        );
        if is_branch_or_parallel {
            continue;
        }
        let has_effect_opd = op
            .operands
            .iter()
            .any(|&v| is_effect_type(&module.values[v].ty));
        let has_effect_res = op
            .results
            .iter()
            .any(|&v| is_effect_type(&module.values[v].ty));
        if has_effect_opd || has_effect_res {
            continue;
        }
        let reads = collect_read_values(module, op_id);
        if !reads.is_empty() && reads.iter().all(|v| invariant.contains(v)) {
            for &res in &op.results {
                invariant.insert(res);
            }
        }
    }
    invariant
}

/// Collects declarative kernel names (`mogg.kernel_decl` with `execute_device = true`) declared at
/// the module top level.
pub fn collect_declarative_kernel_decls(module: &Module) -> IndexSet<String> {
    let mut decls = IndexSet::new();
    let root_blk = module.body_block();
    if let Some(blk) = module.blocks.get(root_blk) {
        for &op_id in &blk.ops {
            if let Some(op) = module.ops.get(op_id)
                && op.name.full_name() == "mogg.kernel_decl"
                && matches!(
                    op.attributes.get("execute_device"),
                    Some(Attribute::Bool(true))
                )
                && let Some(Attribute::String(sym)) = op
                    .attributes
                    .get("sym_name")
                    .or_else(|| op.attributes.get("symbol"))
            {
                decls.insert(sym.clone());
            }
        }
    }
    decls
}

/// Checks whether an `mogg.routine` operation `op_id` can join a declarative `mogg.graph_blob`:
/// 1. Must be an `mogg.routine` with non-empty tensor inputs and outputs, no effect chain operand
///    or result, and no non-empty `output_param_decls`.
/// 2. At least one output must not be in `invariant` (init-only routines stay out of blobs).
/// 3. Every `!mogg.dim` block argument in its body must have a non-empty `ssa_name`, and its body
///    must contain exactly one call (`mogg.call.opaque` or `mogg.kernel.invoke`) targeting a
///    declarative `execute_device` kernel in `declarative_kernels` (or marked `execute_device =
///    true` on the call).
pub fn can_join_graph_blob(
    module: &Module,
    op_id: OpId,
    declarative_kernels: &IndexSet<String>,
    invariant: &IndexSet<ValueId>,
) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    if op.name.full_name() != "mogg.routine" || op.results.is_empty() || op.regions.is_empty() {
        return false;
    }
    if let Some(Attribute::Array(decls)) = op.attributes.get("output_param_decls")
        && !decls.is_empty()
    {
        return false;
    }
    if op
        .operands
        .iter()
        .any(|&v| !matches!(module.values[v].ty, Type::Tensor(_)))
        || op
            .results
            .iter()
            .any(|&v| !matches!(module.values[v].ty, Type::Tensor(_)))
    {
        return false;
    }
    if op.results.iter().all(|v| invariant.contains(v)) {
        return false;
    }

    let reg_id = op.regions[0];
    let Some(reg) = module.regions.get(reg_id) else {
        return false;
    };
    let Some(&body_blk_id) = reg.blocks.first() else {
        return false;
    };
    let Some(body_blk) = module.blocks.get(body_blk_id) else {
        return false;
    };

    // Every `!mogg.dim` argument must carry a name.
    for &arg_val in &body_blk.arguments {
        if matches!(&module.values[arg_val].ty, Type::Opaque(s) if s == "!mogg.dim")
            && module.values[arg_val]
                .ssa_name
                .as_ref()
                .is_none_or(String::is_empty)
        {
            return false;
        }
    }

    let mut found_call: Option<&Operation> = None;
    for &inner_id in &body_blk.ops {
        let Some(inner_op) = module.ops.get(inner_id) else {
            continue;
        };
        let iname = inner_op.name.full_name();
        if matches!(iname.as_str(), "mogg.call.opaque" | "mogg.kernel.invoke") {
            if found_call.is_some() {
                return false;
            }
            found_call = Some(inner_op);
        }
    }
    let Some(call_op) = found_call else {
        return false;
    };
    if matches!(
        call_op.attributes.get("execute_device"),
        Some(Attribute::Bool(true))
    ) {
        return true;
    }
    let callee = call_op
        .attributes
        .get("callee")
        .or_else(|| call_op.attributes.get("symbol"));
    matches!(callee, Some(Attribute::String(s)) if declarative_kernels.contains(s))
}

fn routine_device_ref(op: &Operation) -> String {
    match op.attributes.get("device_ref") {
        Some(Attribute::String(s)) => s.clone(),
        _ => "cpu:0".to_string(),
    }
}

/// Fuses groups of declarative `mogg.routine` operations (sequences and siblings alike on the same
/// device) into a single `mogg.routine` whose body is an `mogg.graph_blob` containing
/// `mogg.call.execute_device` invocations.
///
/// A blob closes immediately at any member whose result is read by an operation outside the blob
/// (other than the block terminator), ensuring no dataflow path leaves a blob and re-enters it.
/// Returns the number of fused `mogg.graph_blob` routines created.
pub fn form_graph_blobs(module: &mut Module) -> usize {
    let declarative_kernels = collect_declarative_kernel_decls(module);
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut blobs_formed = 0usize;

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        if orig_ops.len() < 2 {
            continue;
        }
        let invariant = find_invariant_values(module, blk_id);

        // Precompute users of every value in `blk_id` and across the module.
        let mut val_users: IndexMap<ValueId, Vec<(BlockId, OpId)>> = IndexMap::new();
        for (b_id, b) in &module.blocks {
            for &o_id in &b.ops {
                if let Some(o) = module.ops.get(o_id) {
                    for &opd in &o.operands {
                        val_users.entry(opd).or_default().push((b_id, o_id));
                    }
                }
            }
        }

        let is_read_outside = |module: &Module, member_id: OpId, dev: &str| -> bool {
            let Some(member_op) = module.ops.get(member_id) else {
                return true;
            };
            for &res in &member_op.results {
                let Some(users) = val_users.get(&res) else {
                    continue;
                };
                for &(user_blk, user_op_id) in users {
                    if user_blk != blk_id {
                        return true;
                    }
                    let Some(user_op) = module.ops.get(user_op_id) else {
                        return true;
                    };
                    if is_terminator_op(&user_op.name.full_name()) {
                        continue;
                    }
                    if !can_join_graph_blob(module, user_op_id, &declarative_kernels, &invariant)
                        || routine_device_ref(user_op) != dev
                    {
                        return true;
                    }
                }
            }
            false
        };

        // Partition `orig_ops` into runs of fusible routines.
        let mut groups_to_fuse: Vec<Vec<OpId>> = Vec::new();
        let mut current_blob: Vec<OpId> = Vec::new();
        let mut current_dev = String::new();

        for &op_id in &orig_ops {
            if can_join_graph_blob(module, op_id, &declarative_kernels, &invariant) {
                let dev = routine_device_ref(&module.ops[op_id]);
                if !current_blob.is_empty() && dev != current_dev {
                    if current_blob.len() > 1 {
                        groups_to_fuse.push(std::mem::take(&mut current_blob));
                    } else {
                        current_blob.clear();
                    }
                }
                current_dev = dev.clone();
                current_blob.push(op_id);
                if is_read_outside(module, op_id, &dev) {
                    if current_blob.len() > 1 {
                        groups_to_fuse.push(std::mem::take(&mut current_blob));
                    } else {
                        current_blob.clear();
                    }
                }
            } else {
                if current_blob.len() > 1 {
                    groups_to_fuse.push(std::mem::take(&mut current_blob));
                } else {
                    current_blob.clear();
                }
            }
        }
        if current_blob.len() > 1 {
            groups_to_fuse.push(current_blob);
        }

        for members in groups_to_fuse {
            fuse_routine_blob_group(module, blk_id, &members);
            blobs_formed += 1;
        }
    }

    blobs_formed
}

fn fuse_routine_blob_group(module: &mut Module, blk_id: BlockId, members: &[OpId]) {
    let Some(&last_id) = members.last() else {
        return;
    };
    let mut locs: Vec<Location> = Vec::with_capacity(members.len());
    let mut external_inputs: Vec<ValueId> = Vec::new();
    let mut input_seen: IndexSet<ValueId> = IndexSet::new();
    let mut all_results: Vec<ValueId> = Vec::new();
    let mut member_result_set: IndexSet<ValueId> = IndexSet::new();
    let mut any_deferred = false;
    let mut shared_dim_names: IndexSet<String> = IndexSet::new();
    let dev_ref = routine_device_ref(&module.ops[last_id]);

    for &m_id in members {
        let Some(m_op) = module.ops.get(m_id) else {
            continue;
        };
        locs.push(m_op.location.clone());
        if matches!(
            m_op.attributes.get("dispatch"),
            Some(Attribute::String(s)) if s == "deferred"
        ) {
            any_deferred = true;
        }
        for &inp in &m_op.operands {
            if !member_result_set.contains(&inp) && input_seen.insert(inp) {
                external_inputs.push(inp);
            }
        }
        for &res in &m_op.results {
            member_result_set.insert(res);
            all_results.push(res);
        }
        if let Some(&r_id) = m_op.regions.first()
            && let Some(reg) = module.regions.get(r_id)
            && let Some(&b_id) = reg.blocks.first()
            && let Some(b) = module.blocks.get(b_id)
        {
            for &arg in &b.arguments {
                if matches!(&module.values[arg].ty, Type::Opaque(s) if s == "!mogg.dim")
                    && let Some(ref name) = module.values[arg].ssa_name
                {
                    shared_dim_names.insert(name.clone());
                }
            }
        }
    }

    let fused_loc = fuse_locations(&locs);
    let routine_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let ptr_ty = Type::Opaque("!mogg.pointer".to_string());
    let dim_ty = Type::Opaque("!mogg.dim".to_string());
    let ctx_ty = Type::Opaque("!mogg.context".to_string());

    for idx in 0..(external_inputs.len() + all_results.len()) {
        let arg_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: ptr_ty.clone(),
            def: ValueDef::BlockArgument {
                block: routine_blk,
                index: idx as u32,
            },
        });
        module.blocks[routine_blk].arguments.push(arg_val);
    }
    for dim_name in &shared_dim_names {
        let idx = module.blocks[routine_blk].arguments.len() as u32;
        let dim_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: Some(dim_name.clone()),
            ty: dim_ty.clone(),
            def: ValueDef::BlockArgument {
                block: routine_blk,
                index: idx,
            },
        });
        module.blocks[routine_blk].arguments.push(dim_val);
    }
    let ctx_idx = module.blocks[routine_blk].arguments.len() as u32;
    let ctx_arg = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: ctx_ty,
        def: ValueDef::BlockArgument {
            block: routine_blk,
            index: ctx_idx,
        },
    });
    module.blocks[routine_blk].arguments.push(ctx_arg);

    // Build the inner `mogg.graph_blob` region containing `mogg.call.execute_device` per member.
    let blob_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    for &m_id in members {
        let Some(m_op) = module.ops.get(m_id).cloned() else {
            continue;
        };
        if let Some(&r_id) = m_op.regions.first()
            && let Some(reg) = module.regions.get(r_id).cloned()
            && let Some(&b_id) = reg.blocks.first()
            && let Some(b) = module.blocks.get(b_id).cloned()
        {
            for &inner_id in &b.ops {
                let Some(inner_op) = module.ops.get(inner_id).cloned() else {
                    continue;
                };
                let iname = inner_op.name.full_name();
                if matches!(iname.as_str(), "mogg.call.opaque" | "mogg.kernel.invoke") {
                    let callee = inner_op
                        .attributes
                        .get("callee")
                        .or_else(|| inner_op.attributes.get("symbol"))
                        .cloned()
                        .unwrap_or_else(|| Attribute::String("kernel".to_string()));
                    let mut exec_attrs = IndexMap::new();
                    exec_attrs.insert("callee".to_string(), callee);
                    let exec_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mogg", "call.execute_device"),
                        operands: SmallVec::from_slice(&[ctx_arg]),
                        results: SmallVec::new(),
                        attributes: exec_attrs,
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: inner_op.location.clone(),
                    });
                    module.blocks[blob_blk].ops.push(exec_op);
                }
            }
        }
    }

    let blob_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![blob_blk],
    });
    let blob_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg", "graph_blob"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: vec![blob_reg],
        custom_syntax: false,
        location: fused_loc.clone(),
    });
    module.blocks[routine_blk].ops.push(blob_op);

    let routine_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![routine_blk],
    });

    let mut fused_attrs = IndexMap::new();
    fused_attrs.insert("device_ref".to_string(), Attribute::String(dev_ref));
    fused_attrs.insert(
        "dispatch".to_string(),
        Attribute::String(
            if any_deferred {
                "deferred"
            } else {
                "immediate"
            }
            .to_string(),
        ),
    );
    fused_attrs.insert(
        "blob_member_count".to_string(),
        Attribute::Int(members.len() as i64),
    );
    fused_attrs.insert(
        "shared_dim_names".to_string(),
        Attribute::Array(
            shared_dim_names
                .iter()
                .map(|s| Attribute::String(s.clone()))
                .collect(),
        ),
    );

    let fused_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg", "routine"),
        operands: SmallVec::from_slice(&external_inputs),
        results: SmallVec::new(),
        attributes: fused_attrs,
        regions: vec![routine_reg],
        custom_syntax: false,
        location: fused_loc,
    });

    for (r_idx, &old_res) in all_results.iter().enumerate() {
        let r_ty = module.values[old_res].ty.clone();
        let new_res = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: r_ty,
            def: ValueDef::OpResult {
                op: fused_op,
                index: r_idx as u32,
            },
        });
        module.ops[fused_op].results.push(new_res);
        module.replace_all_uses(old_res, new_res);
    }

    // Place `fused_op` at `last_id`'s position and remove all `members`.
    let member_set: IndexSet<OpId> = members.iter().copied().collect();
    if let Some(blk_mut) = module.blocks.get_mut(blk_id) {
        let mut new_ops = Vec::with_capacity(blk_mut.ops.len());
        for &id in &blk_mut.ops {
            if id == last_id {
                new_ops.push(fused_op);
            } else if !member_set.contains(&id) {
                new_ops.push(id);
            }
        }
        blk_mut.ops = new_ops;
    }
    for &m_id in members {
        module.ops.remove(m_id);
    }
}

// ============================================================================
// Relocatable Compilation Cache-Key Hasher (`MEFCacheKey`)
// ============================================================================

/// Pipeline configuration options fed into [`RelocatableCacheKeyHasher`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelocatablePipelineKeyOptions {
    /// Textual pipeline specification (e.g., `"canonicalize,cse,fuse-elementwise"`).
    pub textual_pipeline: String,
    /// Host/device target triple or architecture descriptor.
    pub target_descriptor: String,
    /// Compile-time parameter defines (`key=value`).
    pub compile_defines: Vec<(String, String)>,
    /// Optional content-addressed storage cache directory path (stripped to a boolean
    /// `cas_enabled` bit when hashing).
    pub cache_dir: Option<String>,
    /// Machine-local kernel library paths (paths are stripped; only their byte contents are
    /// hashed).
    pub kernel_library_paths: Vec<String>,
    /// Machine-local diagnostic temporary output prefix (stripped before hashing).
    pub save_temp_prefix: String,
    /// Machine-local timing output directory (stripped before hashing).
    pub timing_out_dir: String,
    /// Whether source tracebacks are enabled in debug diagnostics.
    pub source_tracebacks: bool,
}

impl RelocatablePipelineKeyOptions {
    /// Returns a path-stripped copy of these options and whether CAS caching is enabled.
    pub fn stripped_for_relocatable_key(&self) -> (Self, bool) {
        let cas_enabled = self
            .cache_dir
            .as_ref()
            .is_some_and(|d| !d.trim().is_empty());
        let stripped = Self {
            textual_pipeline: self.textual_pipeline.clone(),
            target_descriptor: self.target_descriptor.clone(),
            compile_defines: self.compile_defines.clone(),
            cache_dir: None,
            kernel_library_paths: Vec::new(),
            save_temp_prefix: String::new(),
            timing_out_dir: String::new(),
            source_tracebacks: self.source_tracebacks,
        };
        (stripped, cas_enabled)
    }
}

/// Relocatable compilation cache-key hasher that strips machine-specific install/output paths,
/// hashes kernel package byte payloads rather than file paths, and normalizes debug info and symbol
/// names before hashing the IR module.
#[derive(Debug, Clone)]
pub struct RelocatableCacheKeyHasher {
    prefix_digest: [u8; 32],
}

impl RelocatableCacheKeyHasher {
    /// Builds a prefix hasher from `options` and the in-memory byte contents of any referenced
    /// kernel libraries/packages.
    pub fn build(
        options: &RelocatablePipelineKeyOptions,
        kernel_library_contents: &[&[u8]],
    ) -> Self {
        let (stripped, cas_enabled) = options.stripped_for_relocatable_key();
        let mut hasher = Sha256::new();
        hasher.update(b"castforge-relocatable-cache-key-v1\n");
        hasher.update(format!("pipeline={}\n", stripped.textual_pipeline).as_bytes());
        hasher.update(format!("target={}\n", stripped.target_descriptor).as_bytes());
        hasher.update(format!("cas-enabled={cas_enabled}\n").as_bytes());
        for (k, v) in &stripped.compile_defines {
            hasher.update(format!("define:{k}={v}\n").as_bytes());
        }
        for blob in kernel_library_contents {
            hasher.update((blob.len() as u64).to_le_bytes());
            hasher.update(blob);
        }
        hasher.update(format!("source-tracebacks={}\n", stripped.source_tracebacks).as_bytes());
        let out = hasher.finalize();
        let mut prefix_digest = [0u8; 32];
        prefix_digest.copy_from_slice(&out);
        Self { prefix_digest }
    }

    /// Computes the 64-character hex SHA-256 cache key for `module` by cloning the module,
    /// stripping debug locations (`StripDebugInfoPass`), and normalizing all symbol/SSA names
    /// (`normalize_all_names`).
    pub fn key_for_module(&self, module: &Module) -> Result<String, String> {
        let mut cloned = module.clone();
        let _ = StripDebugInfoPass.run(&mut cloned);
        normalize_all_names(&mut cloned)?;
        let ir_text = crate::ir::print_mlir(&cloned);
        let mut hasher = Sha256::new();
        hasher.update(self.prefix_digest);
        hasher.update((ir_text.len() as u64).to_le_bytes());
        hasher.update(ir_text.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write;
            let _ = write!(&mut hex, "{b:02x}");
        }
        Ok(hex)
    }
}

/// Convenience helper computing a relocatable 64-char hex cache key for `module`.
pub fn compute_relocatable_module_cache_key(
    module: &Module,
    options: &RelocatablePipelineKeyOptions,
    kernel_library_contents: &[&[u8]],
) -> Result<String, String> {
    RelocatableCacheKeyHasher::build(options, kernel_library_contents).key_for_module(module)
}

// ============================================================================
// Debug Allocation Poisoning Engine (`DebugAllocPoison`)
// ============================================================================

/// Bit pattern used to poison or zero-fill newly allocated host/device buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoisonPattern {
    /// Little-endian bit pattern value (`0x7BFF` for `f16`, `0x7F7F_FFFF` for `f32`, `0xCD` for
    /// integers, `0` for zero-fill).
    pub bits: u64,
    /// Width of the repeating pattern in bytes (`1`, `2`, `4`, or `8`).
    pub pattern_bytes: usize,
    /// Human-readable pattern classification (`"largest-finite"`, `"byte-fill"`, or `"zero"`).
    pub pattern_kind: &'static str,
}

/// Zero-fill [`PoisonPattern`] constant (`0x00`, 1 byte, `"zero"`).
pub const ZERO_FILL_PATTERN: PoisonPattern = PoisonPattern {
    bits: 0,
    pattern_bytes: 1,
    pattern_kind: "zero",
};

/// Returns the dtype-specific [`PoisonPattern`] for `dtype`:
/// - Standard IEEE floats (`F16`, `BF16`, `F32`, `F64`) return their `largest-finite` bit pattern.
/// - Sub-byte (`I1`, `I4`) and 8-bit float formats (`F8E4M3FN`, `F8E5M2`) return `None`.
/// - Integer and index types return `0xCD` (`1` byte, `"byte-fill"`).
pub fn get_poison_pattern_for_dtype(dtype: &DType) -> Option<PoisonPattern> {
    match dtype {
        DType::I1 | DType::I4 | DType::F8E4M3FN | DType::F8E5M2 | DType::QuantUniform { .. } => {
            None
        }
        DType::F16 => Some(PoisonPattern {
            bits: 0x7BFF,
            pattern_bytes: 2,
            pattern_kind: "largest-finite",
        }),
        DType::BF16 => Some(PoisonPattern {
            bits: 0x7F7F,
            pattern_bytes: 2,
            pattern_kind: "largest-finite",
        }),
        DType::F32 => Some(PoisonPattern {
            bits: 0x7F7F_FFFF,
            pattern_bytes: 4,
            pattern_kind: "largest-finite",
        }),
        DType::F64 => Some(PoisonPattern {
            bits: 0x7FEF_FFFF_FFFF_FFFF,
            pattern_bytes: 8,
            pattern_kind: "largest-finite",
        }),
        DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::Index => Some(PoisonPattern {
            bits: 0xCD,
            pattern_bytes: 1,
            pattern_kind: "byte-fill",
        }),
    }
}

/// Fills `buf` with `pattern` using unaligned-safe byte copies.
pub fn fill_poison_buffer(buf: &mut [u8], pattern: &PoisonPattern) {
    if pattern.pattern_bytes <= 1 {
        buf.fill(pattern.bits as u8);
        return;
    }
    let pat_len = pattern.pattern_bytes.min(8);
    let pat_bytes = pattern.bits.to_le_bytes();
    let mut off = 0usize;
    while off + pat_len <= buf.len() {
        buf[off..off + pat_len].copy_from_slice(&pat_bytes[..pat_len]);
        off += pat_len;
    }
}

/// Selection mode for [`PoisonSelection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoisonMode {
    /// Poison allocations matching `(dtypes, min_ordinal..=max_ordinal)` and zero-fill the rest.
    Select,
    /// Zero-fill every allocation.
    None,
    /// Leave allocations untouched (`fill = None`), performing synchronization only.
    SyncOnly,
}

/// Parsed allocation poison filter specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoisonSelection {
    /// Poisoning mode.
    pub mode: PoisonMode,
    /// Allowed [`DType`] filter (empty means all dtypes match).
    pub dtypes: Vec<DType>,
    /// Inclusive lower allocation ordinal bound.
    pub min_ordinal: u64,
    /// Inclusive upper allocation ordinal bound.
    pub max_ordinal: u64,
}

impl Default for PoisonSelection {
    fn default() -> Self {
        Self {
            mode: PoisonMode::Select,
            dtypes: Vec::new(),
            min_ordinal: 0,
            max_ordinal: u64::MAX,
        }
    }
}

fn parse_dtype_token(tok: &str) -> Option<DType> {
    match tok.trim().to_ascii_lowercase().as_str() {
        "i1" | "bool" => Some(DType::I1),
        "i8" | "int8" => Some(DType::I8),
        "u8" | "ui8" | "uint8" => Some(DType::U8),
        "i16" | "int16" => Some(DType::I16),
        "u16" | "ui16" | "uint16" => Some(DType::U16),
        "i32" | "int32" | "si32" => Some(DType::I32),
        "u32" | "ui32" | "uint32" => Some(DType::U32),
        "i64" | "int64" | "si64" => Some(DType::I64),
        "u64" | "ui64" | "uint64" => Some(DType::U64),
        "f16" | "float16" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f32" | "float32" => Some(DType::F32),
        "f64" | "float64" => Some(DType::F64),
        "f8e4m3fn" => Some(DType::F8E4M3FN),
        "f8e5m2" => Some(DType::F8E5M2),
        "index" => Some(DType::Index),
        _ => None,
    }
}

fn parse_ordinal_range(val: &str) -> Option<(u64, u64)> {
    let val = val.trim();
    let (lo_s, hi_s) = if let Some((l, r)) = val.split_once('-') {
        (l.trim(), r.trim())
    } else {
        (val, val)
    };
    let lo = lo_s.parse::<u64>().ok()?;
    let hi = hi_s.parse::<u64>().ok()?;
    if lo > hi {
        return None;
    }
    Some((lo, hi))
}

/// Parses an allocation poison selection string (`""`, `"all"`, `"none"`, `"sync-only"`, or `&`-
/// separated clauses `dtype=f32,f16&ord=10-20`).
pub fn parse_poison_selection(spec: &str) -> PoisonSelection {
    let mut sel = PoisonSelection::default();
    let trimmed = spec.trim();
    if trimmed.is_empty() || trimmed == "all" {
        return sel;
    }
    if trimmed == "none" {
        sel.mode = PoisonMode::None;
        return sel;
    }
    if trimmed == "sync-only" {
        sel.mode = PoisonMode::SyncOnly;
        return sel;
    }

    let mut have_dtypes = false;
    let mut have_ordinals = false;
    for clause in trimmed.split('&').filter(|c| !c.trim().is_empty()) {
        let Some((k, v)) = clause.trim().split_once('=') else {
            continue;
        };
        let key = k.trim();
        if key == "dtype" && !have_dtypes {
            let mut parsed = Vec::new();
            let mut all_valid = true;
            for item in v.split(',').filter(|s| !s.trim().is_empty()) {
                if let Some(dt) = parse_dtype_token(item) {
                    parsed.push(dt);
                } else {
                    all_valid = false;
                    break;
                }
            }
            if all_valid && !parsed.is_empty() {
                sel.dtypes = parsed;
                have_dtypes = true;
            }
        } else if key == "ord"
            && !have_ordinals
            && let Some((lo, hi)) = parse_ordinal_range(v)
        {
            sel.min_ordinal = lo;
            sel.max_ordinal = hi;
            have_ordinals = true;
        }
    }
    sel
}

/// Decision returned for a single allocation by [`DebugAllocPoisonTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationPoisonDecision {
    /// Epoch-relative ordinal assigned to this allocation.
    pub ordinal: u64,
    /// Fill pattern to apply (`None` when `SyncOnly` or when the dtype has no poison pattern).
    pub fill: Option<PoisonPattern>,
}

/// Metadata version of a live selection control file (`(unique_id, mtime_ns, size_bytes)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionFileVersion {
    /// Filesystem inode / unique file identifier.
    pub unique_id: u64,
    /// Modification timestamp in nanoseconds.
    pub mtime_ns: u64,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// Stateful allocation poison tracker with file-version epoch resets and ordinal progression.
#[derive(Debug, Clone)]
pub struct DebugAllocPoisonTracker {
    selection: PoisonSelection,
    version: Option<SelectionFileVersion>,
    epoch: u64,
    next_ordinal: u64,
}

impl DebugAllocPoisonTracker {
    /// Creates a tracker initialized with `spec`.
    pub fn new(spec: &str) -> Self {
        Self {
            selection: parse_poison_selection(spec),
            version: None,
            epoch: 0,
            next_ordinal: 0,
        }
    }

    /// Current selection epoch counter.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Next ordinal to be assigned within the current epoch.
    pub fn next_ordinal(&self) -> u64 {
        self.next_ordinal
    }

    /// Updates the selection from a watched file when `new_version != self.version`, advancing
    /// `epoch` and resetting `next_ordinal = 0` even if `new_spec` text is identical.
    pub fn refresh_from_file_version(&mut self, new_version: SelectionFileVersion, new_spec: &str) {
        if self.version != Some(new_version) {
            self.selection = parse_poison_selection(new_spec);
            self.version = Some(new_version);
            self.epoch += 1;
            self.next_ordinal = 0;
        }
    }

    /// Evaluates the poison decision for an allocation of `dtype` and advances `next_ordinal`.
    pub fn decide_for_allocation(&mut self, dtype: &DType) -> AllocationPoisonDecision {
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        let fill = match self.selection.mode {
            PoisonMode::SyncOnly => None,
            PoisonMode::None => Some(ZERO_FILL_PATTERN),
            PoisonMode::Select => {
                let in_ord =
                    ordinal >= self.selection.min_ordinal && ordinal <= self.selection.max_ordinal;
                let in_dtype =
                    self.selection.dtypes.is_empty() || self.selection.dtypes.contains(dtype);
                if in_ord && in_dtype {
                    get_poison_pattern_for_dtype(dtype)
                } else {
                    Some(ZERO_FILL_PATTERN)
                }
            }
        };
        AllocationPoisonDecision { ordinal, fill }
    }
}
