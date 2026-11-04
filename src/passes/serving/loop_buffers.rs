//! KV storage key codecs, loop buffer reuse, memory pool planning, and slice passes.

#[allow(unused_imports)]
use super::*;
use crate::ir::{Attribute, Dim, Location, Module, OpId, Operation, Type, ValueDef, ValueId};
use indexmap::IndexMap;

/// Canonical L2 object key identity shared across storage and filesystem adapters
/// (inspired by LMCache `lmcache/v1/distributed/key_codec.py` `ObjectKey`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvStorageKeySpec {
    /// Model identifier (may contain `/` separators, e.g., `"meta-llama/Llama-3-8B"`).
    pub model_name: String,
    /// KV tensor-parallel / pipeline rank.
    pub kv_rank: u32,
    /// Heterogeneous object group index (`0` for primary attention group, `1` for sliding/Mamba).
    pub object_group_id: u32,
    /// 64-bit prefix chunk hash (`chunk_hash`).
    pub chunk_hash: u64,
    /// Optional tenant isolation `cache_salt` (empty string when unsalted).
    pub cache_salt: String,
}

/// Encodes `key` into the canonical reversible L2 wire key
/// `"{model}@{rank:08x}@{group:x}@{hash:016x}[@{salt}]"` (omitting trailing `@{salt}` when
/// `cache_salt` is empty, matching LMCache `encode_storage_key`).
pub fn encode_kv_storage_key(key: &KvStorageKeySpec) -> String {
    if key.cache_salt.is_empty() {
        format!(
            "{}@{:08x}@{:x}@{:016x}",
            key.model_name, key.kv_rank, key.object_group_id, key.chunk_hash
        )
    } else {
        format!(
            "{}@{:08x}@{:x}@{:016x}@{}",
            key.model_name, key.kv_rank, key.object_group_id, key.chunk_hash, key.cache_salt
        )
    }
}

/// Encodes `key` into the filesystem L2 adapter filename
/// `"{model_sep}@0x{rank:08x}@{group:x}@{hash:016x}[@{salt}].data"` where `'/'` in `model_name` is
/// escaped as `"-SEP-"` (matching LMCache `encode_fs_filename`).
pub fn encode_kv_fs_filename(key: &KvStorageKeySpec) -> String {
    let escaped_model = key.model_name.replace('/', "-SEP-");
    if key.cache_salt.is_empty() {
        format!(
            "{escaped_model}@0x{:08x}@{:x}@{:016x}.data",
            key.kv_rank, key.object_group_id, key.chunk_hash
        )
    } else {
        format!(
            "{escaped_model}@0x{:08x}@{:x}@{:016x}@{}.data",
            key.kv_rank, key.object_group_id, key.chunk_hash, key.cache_salt
        )
    }
}

/// Decodes a canonical L2 wire key (`model@rank@group@hash[@salt]`) back into a
/// [`KvStorageKeySpec`] (matching LMCache `decode_storage_key`).
pub fn decode_kv_storage_key(encoded: &str) -> Option<KvStorageKeySpec> {
    let parts: Vec<&str> = encoded.split('@').collect();
    let (model, rank_str, group_str, hash_str, salt) = match parts.as_slice() {
        [m, r, g, h] => (*m, *r, *g, *h, ""),
        [m, r, g, h, s] => (*m, *r, *g, *h, *s),
        _ => return None,
    };
    if model.is_empty() || hash_str.is_empty() {
        return None;
    }
    let rank_clean = rank_str
        .strip_prefix("0x")
        .or_else(|| rank_str.strip_prefix("0X"))
        .unwrap_or(rank_str);
    let kv_rank = u32::from_str_radix(rank_clean, 16).ok()?;
    let object_group_id = u32::from_str_radix(group_str, 16).ok()?;
    let chunk_hash = u64::from_str_radix(hash_str, 16).ok()?;
    Some(KvStorageKeySpec {
        model_name: model.to_string(),
        kv_rank,
        object_group_id,
        chunk_hash,
        cache_salt: salt.to_string(),
    })
}

/// Decodes a filesystem L2 filename (`*.data`) back into a [`KvStorageKeySpec`], unescaping
/// `"-SEP-"` back to `'/'` (matching LMCache `decode_fs_filename`).
pub fn decode_kv_fs_filename(filename: &str) -> Option<KvStorageKeySpec> {
    let stem = filename.strip_suffix(".data")?;
    let mut spec = decode_kv_storage_key(stem)?;
    spec.model_name = spec.model_name.replace("-SEP-", "/");
    Some(spec)
}

pub(super) const FAULT_INJECT_HASH_DENOM: u64 = 1_000_000;

/// Configuration for deterministic L2 load fault injection
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/fault_inject_l2_adapter.py`
/// `FaultInjectL2AdapterConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct L2FaultInjectConfig {
    /// Deterministic per-key drop probability in `[0.0, 1.0]`.
    pub rate: f64,
    /// Deterministic hash seed so drop decisions reproduce across runs.
    pub seed: u64,
    /// Exact head-relative chunk indices to drop unconditionally.
    pub gap_indices: Vec<usize>,
    /// Workload-agnostic tail-relative ratios in `[0.0, 1.0]`: drops chunk index
    /// `round((1.0 - ratio) * (n - 1))` (`0.0` = last chunk, `0.5` = middle, `1.0` = first).
    pub gap_tail_ratios: Vec<f64>,
}

/// Deterministic L2 load fault-injection policy that faults load results (leaving lookup bitmaps
/// untouched) to exercise segmented gapped prefetch and lock-release trim masks
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/fault_inject_l2_adapter.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct L2FaultInjectPolicy {
    pub config: L2FaultInjectConfig,
}

impl L2FaultInjectPolicy {
    /// Creates a new L2 load fault-injection policy.
    pub fn new(config: L2FaultInjectConfig) -> Self {
        Self { config }
    }

    /// Returns `true` if `key` falls into the deterministic `(seed, key)` rate-based drop bucket.
    pub fn should_drop_key(&self, key: &KvStorageKeySpec) -> bool {
        let rate = self.config.rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return false;
        }
        // Hash stable identity fields directly (`seed`, `chunk_hash`, `model_name`, `kv_rank`,
        // `object_group_id`, `cache_salt`).
        let mut h = self.config.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mix = |acc: u64, val: u64| -> u64 {
            let mut z = acc.wrapping_add(val).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        h = mix(h, key.chunk_hash);
        h = mix(h, key.kv_rank as u64);
        h = mix(h, key.object_group_id as u64);
        for b in key.model_name.bytes().chain(key.cache_salt.bytes()) {
            h = mix(h, b as u64);
        }
        let bucket = h % FAULT_INJECT_HASH_DENOM;
        bucket < ((rate * (FAULT_INJECT_HASH_DENOM as f64)) as u64)
    }

    /// Returns the sorted positions in `keys` that are dropped at load time (combining
    /// `gap_tail_ratios`, `gap_indices`, and `should_drop_key`).
    pub fn drop_positions(&self, keys: &[KvStorageKeySpec]) -> Vec<usize> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }
        let ratio_positions: std::collections::HashSet<usize> = self
            .config
            .gap_tail_ratios
            .iter()
            .filter(|&&r| (0.0..=1.0).contains(&r))
            .map(|&r| ((1.0 - r) * ((n - 1) as f64)).round() as usize)
            .collect();
        let explicit_gaps: std::collections::HashSet<usize> =
            self.config.gap_indices.iter().copied().collect();

        let mut dropped = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if ratio_positions.contains(&i)
                || explicit_gaps.contains(&i)
                || self.should_drop_key(key)
            {
                dropped.push(i);
            }
        }
        dropped
    }

    /// Post-processes a lookup bitmap into `(faulted_load_bitmap, failed_lock_trim_mask)`:
    /// lookup hits at dropped positions are cleared in `faulted_load_bitmap` and set in
    /// `failed_lock_trim_mask` so the prefetch controller releases their L2 locks.
    pub fn fault_load_bitmap(
        &self,
        keys: &[KvStorageKeySpec],
        lookup_bitmap: &[bool],
    ) -> (Vec<bool>, Vec<bool>) {
        let len = keys.len().min(lookup_bitmap.len());
        let mut load_bitmap = lookup_bitmap[..len].to_vec();
        let mut trim_mask = vec![false; len];
        for pos in self.drop_positions(&keys[..len]) {
            if pos < len && load_bitmap[pos] {
                load_bitmap[pos] = false;
                trim_mask[pos] = true;
            }
        }
        (load_bitmap, trim_mask)
    }
}

pub(super) fn replace_ident_in_text(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() || from == to {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let id_len = from.len();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find(from) {
        let pos = cursor + rel;
        let before_ok = pos == 0 || !is_word(bytes[pos - 1]);
        let after_ok = pos + id_len >= bytes.len() || !is_word(bytes[pos + id_len]);
        if before_ok && after_ok {
            out.push_str(&text[cursor..pos]);
            out.push_str(to);
            cursor = pos + id_len;
        } else {
            out.push_str(&text[cursor..pos + 1]);
            cursor = pos + 1;
        }
    }
    out.push_str(&text[cursor..]);
    out
}

pub(super) fn rewrite_stmt_buffer_name(
    stmt: &mut crate::ir::loops::LoopStmt,
    from: &str,
    to: &str,
) {
    use crate::ir::loops::LoopStmt;
    match stmt {
        LoopStmt::Comment(_) | LoopStmt::Alloc { .. } => {}
        LoopStmt::Assign { dst, index, expr } => {
            if dst == from {
                *dst = to.to_string();
            }
            *index = replace_ident_in_text(index, from, to);
            *expr = replace_ident_in_text(expr, from, to);
        }
        LoopStmt::For { body, .. } => {
            for inner in body {
                rewrite_stmt_buffer_name(inner, from, to);
            }
        }
    }
}

/// Aliases non-overlapping `LoopStmt::Alloc` scratchpad buffers of identical length within each
/// [`crate::ir::loops::LoopKernel`], prioritizing buffers with higher use counts first
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`).
///
/// Returns the number of redundant `LoopStmt::Alloc` declarations merged and eliminated.
pub fn reuse_dead_loop_buffers(loop_module: &mut crate::ir::loops::LoopModule) -> usize {
    use crate::ir::loops::LoopStmt;
    let mut reused_count = 0usize;

    for kernel in &mut loop_module.kernels {
        let mut size_groups: IndexMap<usize, Vec<(String, Vec<usize>)>> = IndexMap::new();
        for stmt in &kernel.body {
            if let LoopStmt::Alloc { name, len } = stmt {
                let uses: Vec<usize> = kernel
                    .body
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, s)| stmt_references_buffer(s, name).then_some(idx))
                    .collect();
                if !uses.is_empty() {
                    size_groups
                        .entry(*len)
                        .or_default()
                        .push((name.clone(), uses));
                }
            }
        }

        let mut erased_allocs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut rewrites: Vec<(String, String)> = Vec::new();

        for (_, mut group) in size_groups {
            if group.len() < 2 {
                continue;
            }
            // Allocs with more uses go first (`SimpleBufferReusePass.cpp`).
            group.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

            let mut active = vec![true; group.len()];
            for i in 0..group.len() {
                if !active[i] {
                    continue;
                }
                for j in 0..group.len() {
                    if i == j || !active[j] {
                        continue;
                    }
                    let cur_uses = &group[i].1;
                    let cand_uses = &group[j].1;
                    let shares_op = cur_uses.iter().any(|u| cand_uses.contains(u));
                    if shares_op {
                        continue;
                    }
                    let cur_after_cand = cur_uses
                        .iter()
                        .all(|&cu| cand_uses.iter().all(|&du| cu > du));
                    let cand_after_cur = cand_uses
                        .iter()
                        .all(|&du| cur_uses.iter().all(|&cu| du > cu));
                    if cur_after_cand || cand_after_cur {
                        let primary_name = group[i].0.clone();
                        let cand_name = group[j].0.clone();
                        let mut merged_uses = group[i].1.clone();
                        merged_uses.extend_from_slice(&group[j].1);
                        merged_uses.sort_unstable();
                        group[i].1 = merged_uses;
                        active[j] = false;
                        erased_allocs.insert(cand_name.clone());
                        rewrites.push((cand_name, primary_name));
                        reused_count += 1;
                    }
                }
            }
        }

        if !rewrites.is_empty() {
            for (from, to) in &rewrites {
                for stmt in &mut kernel.body {
                    rewrite_stmt_buffer_name(stmt, from, to);
                }
            }
            // Remove erased candidate `Alloc` statements and ensure surviving `Alloc`s precede
            // their earliest use.
            let mut hoisted_allocs = Vec::new();
            let mut remaining = Vec::with_capacity(kernel.body.len());
            for stmt in kernel.body.drain(..) {
                if let LoopStmt::Alloc { ref name, .. } = stmt {
                    if erased_allocs.contains(name) {
                        continue;
                    }
                    hoisted_allocs.push(stmt);
                } else {
                    remaining.push(stmt);
                }
            }
            hoisted_allocs.append(&mut remaining);
            kernel.body = hoisted_allocs;
        }
    }

    reused_count
}

pub(super) fn is_invariant_root_op(op: &Operation) -> bool {
    op.operands.is_empty()
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.constant" | "castforge.weight_ref" | "stablehlo.iota"
        )
}

pub(super) fn is_trivial_invariant_op(op: &Operation) -> bool {
    matches!(
        op.name.to_string().as_str(),
        "stablehlo.constant"
            | "castforge.weight_ref"
            | "stablehlo.iota"
            | "stablehlo.reshape"
            | "stablehlo.broadcast_in_dim"
            | "stablehlo.bitcast_convert"
    )
}

pub(super) fn is_terminator_or_side_effecting(op: &Operation) -> bool {
    matches!(
        op.name.to_string().as_str(),
        "func.return" | "return" | "stablehlo.return" | "castforge.nan_check"
    )
}

/// Identifies execution-invariant weight/constant subgraphs, filters out standalone trivial ops
/// (`isLiftableTrivialOp`) that do not feed any non-trivial invariant compute op, assigns
/// `castforge.exec_invariant_slot` cache indices to non-trivial invariant ops, and hoists the
/// lifted subgraph to the entry of the block
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp`).
///
/// Returns the number of non-trivial execution-invariant operations assigned cache slots.
pub fn hoist_exec_invariant_ops(module: &mut Module) -> usize {
    let mut cached_nontrivial = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let Some(&reg_id) = module.ops[func_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let op_ids = module.blocks[block_id].ops.clone();

        // Forward pass: find all ops whose operands come strictly from other invariant ops.
        let mut invariant_ops: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        let mut invariant_vals: std::collections::HashSet<ValueId> =
            std::collections::HashSet::new();

        for &op_id in &op_ids {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if is_terminator_or_side_effecting(op) || !op.regions.is_empty() {
                continue;
            }
            let is_inv = if op.operands.is_empty() {
                is_invariant_root_op(op)
            } else {
                op.operands.iter().all(|v| invariant_vals.contains(v))
            };
            if is_inv {
                invariant_ops.insert(op_id);
                for &res in &op.results {
                    invariant_vals.insert(res);
                }
            }
        }

        // Backward pass from non-trivial invariant ops: trivial ops are only lifted if they feed
        // at least one non-trivial invariant op (`isLiftableTrivialOp` in Modular).
        let mut lifted_ops: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        for &op_id in op_ids.iter().rev() {
            if !invariant_ops.contains(&op_id) {
                continue;
            }
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let should_lift = !is_trivial_invariant_op(op) || lifted_ops.contains(&op_id);
            if should_lift {
                lifted_ops.insert(op_id);
                for &in_val in &op.operands {
                    if let Some(v_info) = module.values.get(in_val)
                        && let ValueDef::OpResult { op: prod_id, .. } = v_info.def
                        && invariant_ops.contains(&prod_id)
                    {
                        lifted_ops.insert(prod_id);
                    }
                }
            }
        }

        if lifted_ops.is_empty() {
            continue;
        }

        let mut next_slot = 0i64;
        let mut hoisted = Vec::with_capacity(lifted_ops.len());
        let mut runtime_ops = Vec::with_capacity(op_ids.len());

        for op_id in op_ids {
            if lifted_ops.contains(&op_id) {
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.attributes.insert(
                        "castforge.exec_invariant_init".to_string(),
                        Attribute::Bool(true),
                    );
                    if !is_trivial_invariant_op(op_mut) {
                        op_mut.attributes.insert(
                            "castforge.exec_invariant_slot".to_string(),
                            Attribute::Int(next_slot),
                        );
                        next_slot += 1;
                        cached_nontrivial += 1;
                    }
                }
                hoisted.push(op_id);
            } else {
                runtime_ops.push(op_id);
            }
        }

        hoisted.append(&mut runtime_ops);
        module.blocks[block_id].ops = hoisted;
    }

    cached_nontrivial
}

pub(super) fn is_complex_symbolic_expr(sym: &str) -> bool {
    !sym.is_empty() && !sym.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub(super) fn rewrite_type_symbolic_exprs(
    ty: &mut Type,
    expr_to_alias: &IndexMap<String, String>,
) -> bool {
    match ty {
        Type::Tensor(tt) => {
            let mut changed = false;
            for dim in &mut tt.shape {
                if let Dim::Symbolic(s) = dim
                    && is_complex_symbolic_expr(s)
                {
                    let alias = expr_to_alias
                        .get(s)
                        .cloned()
                        .unwrap_or_else(|| format!("param_expr_{}", expr_to_alias.len()));
                    *s = alias;
                    changed = true;
                }
            }
            changed
        }
        Type::Function { inputs, results } => {
            let mut changed = false;
            for inp in inputs {
                changed |= rewrite_type_symbolic_exprs(inp, expr_to_alias);
            }
            for res in results {
                changed |= rewrite_type_symbolic_exprs(res, expr_to_alias);
            }
            changed
        }
        Type::Tuple(elems) => {
            let mut changed = false;
            for elem in elems {
                changed |= rewrite_type_symbolic_exprs(elem, expr_to_alias);
            }
            changed
        }
        _ => false,
    }
}

pub(super) fn collect_type_symbolic_exprs(ty: &Type, expr_to_alias: &mut IndexMap<String, String>) {
    match ty {
        Type::Tensor(tt) => {
            for dim in &tt.shape {
                if let Dim::Symbolic(s) = dim
                    && is_complex_symbolic_expr(s)
                    && !expr_to_alias.contains_key(s)
                {
                    let alias = format!("param_expr_{}", expr_to_alias.len());
                    expr_to_alias.insert(s.clone(), alias);
                }
            }
        }
        Type::Function { inputs, results } => {
            for inp in inputs {
                collect_type_symbolic_exprs(inp, expr_to_alias);
            }
            for res in results {
                collect_type_symbolic_exprs(res, expr_to_alias);
            }
        }
        Type::Tuple(elems) => {
            for elem in elems {
                collect_type_symbolic_exprs(elem, expr_to_alias);
            }
        }
        _ => {}
    }
}

/// Hoists compound symbolic dimension expressions (e.g., `Dim::Symbolic("batch*seq")`,
/// `Dim::Symbolic("seq+1")`) out of tensor types into deduplicated function-level parameter
/// declarations (`param_expr_0`, `param_expr_1`, ...) stored in `castforge.param_expr_decls`,
/// rewriting all tensor shapes in the function to reference the hoisted parameter aliases
/// (inspired by Modular
/// `GraphCompiler/lib/MODialect/Transforms/HoistParameterExpressionsPass.cpp`).
///
/// Returns the total number of distinct symbolic parameter expressions hoisted across the module.
pub fn hoist_symbolic_dim_expressions(module: &mut Module) -> usize {
    let mut total_hoisted = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let mut expr_to_alias: IndexMap<String, String> = IndexMap::new();

        if let Some(Attribute::Type(ty)) = module.ops[func_id].attributes.get("function_type") {
            collect_type_symbolic_exprs(ty, &mut expr_to_alias);
        }

        let reg_ids = module.ops[func_id].regions.clone();
        let mut val_ids = Vec::new();
        for reg_id in reg_ids {
            let block_ids = module.regions[reg_id].blocks.clone();
            for block_id in block_ids {
                for &arg_val in &module.blocks[block_id].arguments {
                    val_ids.push(arg_val);
                }
                for &op_id in &module.blocks[block_id].ops {
                    if let Some(op) = module.ops.get(op_id) {
                        for &res_val in &op.results {
                            val_ids.push(res_val);
                        }
                    }
                }
            }
        }

        for &val_id in &val_ids {
            if let Some(v_info) = module.values.get(val_id) {
                collect_type_symbolic_exprs(&v_info.ty, &mut expr_to_alias);
            }
        }

        if expr_to_alias.is_empty() {
            continue;
        }
        total_hoisted += expr_to_alias.len();

        for &val_id in &val_ids {
            if let Some(v_info) = module.values.get_mut(val_id) {
                rewrite_type_symbolic_exprs(&mut v_info.ty, &expr_to_alias);
            }
        }

        if let Some(func_op) = module.ops.get_mut(func_id) {
            if let Some(Attribute::Type(ty)) = func_op.attributes.get_mut("function_type") {
                rewrite_type_symbolic_exprs(ty, &expr_to_alias);
            }
            let decls: IndexMap<String, Attribute> = expr_to_alias
                .into_iter()
                .map(|(expr, alias)| (alias, Attribute::String(expr)))
                .collect();
            func_op.attributes.insert(
                "castforge.param_expr_decls".to_string(),
                Attribute::Dictionary(decls),
            );
        }
    }

    total_hoisted
}

/// Downsamples per-group paged KV `block_ids` for sub-chunk sliding-window attention (SWA) groups
/// (inspired by LMCache `lmcache/v1/multiprocess/object_group_transfer.py`
/// `downsample_and_stage_block_ids`).
///
/// For a kernel group with `block_size_tokens`, `chunk_tokens`, and `subchunk_sw_tokens` (`0` or
/// `>= chunk_tokens` for full attention; `< chunk_tokens` for sub-chunk SWA such as DeepSeek-V4):
/// - `total_blocks_per_chunk = chunk_tokens.div_ceil(block_size_tokens)`
/// - `keep_blocks_per_chunk = chunk_tokens.min(subchunk_sw_tokens).div_ceil(block_size_tokens)`
///
/// Keeps only the trailing `keep_blocks_per_chunk` block IDs from each chunk of
/// `total_blocks_per_chunk` blocks. Returns `None` if `group_block_ids.len()` is not a multiple of
/// `total_blocks_per_chunk`.
pub fn downsample_subchunk_swa_block_ids(
    group_block_ids: &[u64],
    chunk_tokens: usize,
    subchunk_sw_tokens: usize,
    block_size_tokens: usize,
) -> Option<Vec<u64>> {
    if chunk_tokens == 0 || block_size_tokens == 0 {
        return None;
    }
    let eff_sw = if subchunk_sw_tokens == 0 {
        chunk_tokens
    } else {
        chunk_tokens.min(subchunk_sw_tokens)
    };
    let total_blocks_per_chunk = chunk_tokens.div_ceil(block_size_tokens);
    let keep_blocks_per_chunk = eff_sw.div_ceil(block_size_tokens);
    if !group_block_ids.len().is_multiple_of(total_blocks_per_chunk) {
        return None;
    }
    if keep_blocks_per_chunk >= total_blocks_per_chunk {
        return Some(group_block_ids.to_vec());
    }
    let mut out = Vec::with_capacity(
        (group_block_ids.len() / total_blocks_per_chunk) * keep_blocks_per_chunk,
    );
    for chunk in group_block_ids.chunks_exact(total_blocks_per_chunk) {
        let start = total_blocks_per_chunk - keep_blocks_per_chunk;
        out.extend_from_slice(&chunk[start..]);
    }
    Some(out)
}

/// Recalculates the number of downsampled blocks to skip when `blocks_per_window <=
/// blocks_per_chunk` (inspired by LMCache `lmcache/v1/multiprocess/object_group_transfer.py`
/// `recalculate_blocks_to_skip`).
pub fn recalculate_subchunk_blocks_to_skip(
    blocks_per_chunk: usize,
    blocks_per_window: usize,
    blocks_to_skip: usize,
) -> usize {
    if blocks_per_chunk == 0 || blocks_per_window >= blocks_per_chunk {
        return blocks_to_skip;
    }
    let full_windows_to_skip = blocks_to_skip / blocks_per_chunk;
    let tail_blocks = blocks_to_skip % blocks_per_chunk;
    let dropped_per_chunk = blocks_per_chunk - blocks_per_window;
    let tail_blocks_to_skip = tail_blocks.saturating_sub(dropped_per_chunk);
    full_windows_to_skip * blocks_per_window + tail_blocks_to_skip
}

/// Bytes per packed wire token ID (big-endian `u32`, matching LMCache `TOKEN_STRIDE = 4`).
pub const TOKEN_WIRE_STRIDE: usize = 4;

/// Packs `token_ids` into a big-endian `u32` wire byte buffer (`TOKEN_WIRE_STRIDE * len(tokens)`)
/// so token sequences cross process boundaries via a single `memcpy` and hash without unpacking
/// (inspired by LMCache `lmcache/v1/multiprocess/token_codec.py` `pack_token_ids`).
pub fn pack_token_ids_be(token_ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(token_ids.len() * TOKEN_WIRE_STRIDE);
    for &tok in token_ids {
        out.extend_from_slice(&tok.to_be_bytes());
    }
    out
}

/// Counts the number of whole packed tokens in `packed` (`packed.len() / TOKEN_WIRE_STRIDE`).
pub fn num_packed_tokens(packed: &[u8]) -> usize {
    packed.len() / TOKEN_WIRE_STRIDE
}

/// Unpacks a big-endian `u32` wire buffer back into token IDs, returning `None` if `packed.len()`
/// is not an exact multiple of [`TOKEN_WIRE_STRIDE`]
/// (matching LMCache `lmcache/v1/multiprocess/token_codec.py` `unpack_token_ids`).
pub fn unpack_token_ids_be(packed: &[u8]) -> Option<Vec<u32>> {
    if !packed.len().is_multiple_of(TOKEN_WIRE_STRIDE) {
        return None;
    }
    let mut out = Vec::with_capacity(packed.len() / TOKEN_WIRE_STRIDE);
    for &chunk in packed.as_chunks::<TOKEN_WIRE_STRIDE>().0 {
        out.push(u32::from_be_bytes(chunk));
    }
    Some(out)
}

/// Computes chained prefix-chunk hashes directly over a big-endian packed token wire buffer
/// without unpacking into `Vec<u32>`, producing hashes identical to hashing `[u32]` chunks
/// (inspired by LMCache `lmcache/v1/multiprocess/token_codec.py` & `token_hasher.py`).
pub fn hash_packed_token_blocks_be(packed: &[u8], chunk_tokens: usize, seed: u64) -> Vec<u64> {
    let stride_bytes = chunk_tokens.saturating_mul(TOKEN_WIRE_STRIDE);
    if stride_bytes == 0 || packed.len() < stride_bytes {
        return Vec::new();
    }
    let mut hashes = Vec::with_capacity(packed.len() / stride_bytes);
    let mut parent_hash = seed;
    for chunk_bytes in packed.chunks_exact(stride_bytes) {
        let mut h = parent_hash ^ 0x9E37_79B9_7F4A_7C15;
        for &word in chunk_bytes.as_chunks::<TOKEN_WIRE_STRIDE>().0 {
            let tok = u32::from_be_bytes(word) as u64;
            h = h.wrapping_add(tok).wrapping_add(0x9E37_79B9_7F4A_7C15);
            h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^= h >> 31;
        }
        hashes.push(h);
        parent_hash = h;
    }
    hashes
}

/// Descriptor for an intermediate buffer allocation input to [`plan_dynamic_memory_pool`]
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
/// `AllocInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicAllocRequest {
    /// Buffer symbol / identifier.
    pub name: String,
    /// Allocation size in bytes.
    pub size_bytes: u64,
    /// Required base-address alignment in bytes.
    pub alignment_bytes: u64,
    /// Earliest topological statement/operation use index.
    pub first_use_idx: usize,
    /// Latest topological statement/operation use index.
    pub last_use_idx: usize,
    /// Source location of the allocation op.
    pub location: Location,
}

/// Bulk-slice memory pool plan produced by [`plan_dynamic_memory_pool`]
/// (inspired by Modular `DynamicMemoryPlanningPass.cpp` `BufferPlanOp` + `BufferBulkSliceOp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicMemoryPoolPlan {
    /// Allocations ordered by ascending `first_use_idx`.
    pub ordered_names: Vec<String>,
    /// Assigned byte offset for each ordered allocation inside the shared memory pool.
    pub bulk_slice_offsets: Vec<u64>,
    /// Byte size for each ordered allocation.
    pub bulk_slice_sizes: Vec<u64>,
    /// Symmetric `n * n` pairwise shareability matrix (`can_share[i * n + j]`).
    pub can_share: Vec<bool>,
    /// Pool alignment in bytes (`LCM` of all constituent buffer alignments).
    pub pool_alignment_bytes: u64,
    /// Peak byte footprint (`high_watermark`) of the consolidated memory pool.
    pub high_watermark_bytes: u64,
    /// Fused source provenance across all consolidated buffer allocations.
    pub fused_location: Location,
}

/// Plans a consolidated LCM-aligned memory pool and single `BufferBulkSliceOp` layout for
/// intermediate buffer allocations
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`).
///
/// 1. Sorts allocations by earliest topological use (`first_use_idx`),
/// 2. Computes `pool_alignment_bytes = LCM(alignments)`,
/// 3. Builds the symmetric `n * n` `can_share` matrix (`true` iff all uses of `i` execute strictly
///    after all uses of `j` or vice versa),
/// 4. Greedily packs buffers into aligned offsets while respecting non-shareable interference, and
/// 5. Fuses all allocation [`Location`]s via [`fuse_locations`].
pub fn plan_dynamic_memory_pool(allocs: &[DynamicAllocRequest]) -> DynamicMemoryPoolPlan {
    let mut sorted = allocs.to_vec();
    sorted.sort_by_key(|a| a.first_use_idx);

    let n = sorted.len();
    let mut pool_alignment_bytes = 1u64;
    let mut locs = Vec::with_capacity(n);
    for a in &sorted {
        pool_alignment_bytes =
            checked_lcm_u64(pool_alignment_bytes, a.alignment_bytes.max(1)).unwrap_or(u64::MAX);
        locs.push(a.location.clone());
    }

    let mut can_share = vec![false; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sharable = sorted[i].last_use_idx < sorted[j].first_use_idx
                || sorted[j].last_use_idx < sorted[i].first_use_idx;
            can_share[i * n + j] = sharable;
            can_share[j * n + i] = sharable;
        }
    }

    let mut offsets = vec![0u64; n];
    let mut high_watermark_bytes = 0u64;

    for i in 0..n {
        let align = sorted[i].alignment_bytes.max(1);
        let size_i = sorted[i].size_bytes;
        let mut cand_offset = 0u64;

        // Advance `cand_offset` past any interfering (`!can_share`) already-placed buffer `j < i`.
        loop {
            cand_offset = cand_offset.div_ceil(align) * align;
            let end_i = cand_offset + size_i;
            let conflict = (0..i).find(|&j| {
                if can_share[i * n + j] {
                    return false;
                }
                let start_j = offsets[j];
                let end_j = start_j + sorted[j].size_bytes;
                cand_offset < end_j && start_j < end_i
            });
            if let Some(j) = conflict {
                cand_offset = offsets[j] + sorted[j].size_bytes;
            } else {
                break;
            }
        }

        offsets[i] = cand_offset;
        high_watermark_bytes = high_watermark_bytes.max(cand_offset + size_i);
    }

    DynamicMemoryPoolPlan {
        ordered_names: sorted.iter().map(|a| a.name.clone()).collect(),
        bulk_slice_offsets: offsets,
        bulk_slice_sizes: sorted.iter().map(|a| a.size_bytes).collect(),
        can_share,
        pool_alignment_bytes,
        high_watermark_bytes,
        fused_location: fuse_locations(&locs),
    }
}

pub(super) fn stmt_has_non_pointwise_access(
    stmt: &crate::ir::loops::LoopStmt,
    buf_name: &str,
    loop_var: &str,
) -> bool {
    use crate::ir::loops::LoopStmt;
    let exact_ref = format!("{buf_name}[{loop_var}]");
    match stmt {
        LoopStmt::Comment(_) | LoopStmt::Alloc { .. } => false,
        LoopStmt::Assign { dst, index, expr } => {
            if dst == buf_name && index != loop_var {
                return true;
            }
            let stripped_idx = index.replace(&exact_ref, "");
            let stripped_expr = expr.replace(&exact_ref, "");
            expr_contains_ident(&stripped_idx, buf_name)
                || expr_contains_ident(&stripped_expr, buf_name)
        }
        LoopStmt::For { body, .. } => body
            .iter()
            .any(|s| stmt_has_non_pointwise_access(s, buf_name, loop_var)),
    }
}

/// Merges adjacent parallel `LoopStmt::For` regions sharing identical `(lower, upper, step)`
/// bounds when all shared producer-to-consumer buffer accesses are strictly pointwise at `[var]`,
/// forwarding sole-use producer yields directly into the consumer body and dropping dead producer
/// intermediate stores and allocations
/// (inspired by Modular `GraphCompiler/lib/MODialect/Transforms/MergeAdjacentParallels.cpp`).
///
/// Returns the number of adjacent parallel loop pairs merged.
pub fn merge_adjacent_parallel_loops(loop_module: &mut crate::ir::loops::LoopModule) -> usize {
    use crate::ir::loops::LoopStmt;
    let mut merged_count = 0usize;

    for kernel in &mut loop_module.kernels {
        loop {
            let mut merged_pair: Option<(usize, LoopStmt, std::collections::HashSet<String>)> =
                None;

            for idx in 0..kernel.body.len().saturating_sub(1) {
                let (
                    LoopStmt::For {
                        var: p_var,
                        lower: p_lo,
                        upper: p_hi,
                        step: p_step,
                        parallel: true,
                        vectorized: p_vec,
                        unroll: p_unroll,
                        body: p_body,
                    },
                    LoopStmt::For {
                        var: c_var,
                        lower: c_lo,
                        upper: c_hi,
                        step: c_step,
                        parallel: true,
                        vectorized: c_vec,
                        unroll: c_unroll,
                        body: c_body,
                    },
                ) = (&kernel.body[idx], &kernel.body[idx + 1])
                else {
                    continue;
                };

                if p_lo != c_lo || p_hi != c_hi || p_step != c_step || p_var != c_var {
                    continue;
                }

                // Collect buffers written pointwisely (`dst[var] = expr`) by the producer loop.
                let mut prod_writes: IndexMap<String, String> = IndexMap::new();
                let mut valid_prod = true;
                for stmt in p_body {
                    match stmt {
                        LoopStmt::Assign { dst, index, expr } if index == p_var => {
                            prod_writes.insert(dst.clone(), expr.clone());
                        }
                        LoopStmt::Comment(_) => {}
                        _ => {
                            valid_prod = false;
                            break;
                        }
                    }
                }
                if !valid_prod || prod_writes.is_empty() {
                    continue;
                }

                // Verify that the consumer loop has no non-pointwise (cross-iteration) access to
                // any buffer written by the producer (`nonSharedInputDependsOnProducer`).
                let has_cross_iter_hazard = prod_writes.keys().any(|buf| {
                    c_body
                        .iter()
                        .any(|s| stmt_has_non_pointwise_access(s, buf, p_var))
                });
                if has_cross_iter_hazard {
                    continue;
                }

                // Identify producer outputs used ONLY by `consumer` (not kernel output, and not
                // referenced by any statement outside `idx` and `idx + 1`).
                let mut dead_after_consumer: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for buf in prod_writes.keys() {
                    if *buf == kernel.output.0 {
                        continue;
                    }
                    let used_outside = kernel.body.iter().enumerate().any(|(k, s)| {
                        k != idx
                            && k != idx + 1
                            && !matches!(s, LoopStmt::Alloc { .. })
                            && stmt_references_buffer(s, buf)
                    });
                    if !used_outside {
                        dead_after_consumer.insert(buf.clone());
                    }
                }

                let mut combined_body = Vec::with_capacity(p_body.len() + c_body.len());
                for stmt in p_body {
                    if let LoopStmt::Assign { dst, .. } = stmt
                        && dead_after_consumer.contains(dst)
                    {
                        continue;
                    }
                    combined_body.push(stmt.clone());
                }

                for stmt in c_body {
                    let mut cloned = stmt.clone();
                    if let LoopStmt::Assign { ref mut expr, .. } = cloned {
                        for buf in &dead_after_consumer {
                            if let Some(prod_expr) = prod_writes.get(buf) {
                                let target = format!("{buf}[{p_var}]");
                                let replacement = format!("({prod_expr})");
                                *expr = expr.replace(&target, &replacement);
                            }
                        }
                    }
                    combined_body.push(cloned);
                }

                let merged_for = LoopStmt::For {
                    var: p_var.clone(),
                    lower: *p_lo,
                    upper: *p_hi,
                    step: *p_step,
                    parallel: true,
                    vectorized: *p_vec && *c_vec,
                    unroll: (*p_unroll).max(*c_unroll),
                    body: combined_body,
                };
                merged_pair = Some((idx, merged_for, dead_after_consumer));
                break;
            }

            if let Some((idx, merged_for, dead_bufs)) = merged_pair {
                kernel.body[idx] = merged_for;
                kernel.body.remove(idx + 1);
                if !dead_bufs.is_empty() {
                    kernel.body.retain(
                        |s| !matches!(s, LoopStmt::Alloc { name, .. } if dead_bufs.contains(name)),
                    );
                }
                merged_count += 1;
            } else {
                break;
            }
        }
    }

    merged_count
}

pub(super) fn is_effectful_chain_producer(op: &Operation) -> bool {
    if matches!(
        op.attributes.get("castforge.async_chain"),
        Some(Attribute::Bool(true))
    ) {
        return true;
    }
    matches!(
        op.name.to_string().as_str(),
        "castforge.kv_offload"
            | "castforge.kv_store"
            | "castforge.async_copy"
            | "castforge.device_transfer"
            | "stablehlo.send"
            | "stablehlo.outfeed"
    )
}

/// Awaits unused ("hanging") asynchronous/effectful operation results before each block's
/// terminator (`func.return` / `return`) by inserting a `castforge.chain_await` barrier operation
/// and recording `castforge.awaited_hanging_chains` on the terminator
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/FixupOverhangsPass.cpp`).
///
/// Matching `FixupOverhangsPass`: at most one unused result per hanging op is collected ("A
/// failing op sets all its results to error, so one per op suffices"). Returns the total number of
/// hanging chains awaited across the module.
pub fn await_hanging_effect_chains(module: &mut Module) -> usize {
    let mut awaited_total = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let Some(&term_id) = op_ids.last() else {
            continue;
        };
        let Some(term_op) = module.ops.get(term_id).cloned() else {
            continue;
        };
        if !matches!(
            term_op.name.to_string().as_str(),
            "func.return" | "return" | "stablehlo.return"
        ) {
            continue;
        }

        let mut hanging_chains: Vec<ValueId> = Vec::new();
        let mut hanging_locs: Vec<Location> = Vec::new();

        for &op_id in &op_ids[..op_ids.len() - 1] {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !is_effectful_chain_producer(op) {
                continue;
            }
            for &res_val in &op.results {
                if module.use_count(res_val) == 0 {
                    hanging_chains.push(res_val);
                    hanging_locs.push(op.location.clone());
                    break;
                }
            }
        }

        if hanging_chains.is_empty() {
            continue;
        }

        let count = hanging_chains.len();
        awaited_total += count;
        let fused_loc = fuse_locations(&hanging_locs);

        let await_op_id = module.ops.insert_with_key(|id| Operation {
            id,
            name: crate::ir::OpName::new("castforge", "chain_await"),
            operands: hanging_chains.into(),
            results: Default::default(),
            attributes: IndexMap::new(),
            regions: Default::default(),
            custom_syntax: false,
            location: fused_loc,
        });

        if let Some(term_mut) = module.ops.get_mut(term_id) {
            term_mut.attributes.insert(
                "castforge.awaited_hanging_chains".to_string(),
                Attribute::Int(count as i64),
            );
        }

        let block_ops = &mut module.blocks[block_id].ops;
        let insert_pos = block_ops.len().saturating_sub(1);
        block_ops.insert(insert_pos, await_op_id);
    }

    awaited_total
}

/// Eliminates direct dependency edges of each node in `dag_edges` (`dag_edges[u]` = list of
/// prerequisite nodes that `u` depends on) that are transitively implied by `u`'s other
/// prerequisites (`u -> w ->+ v`), using a cycle-safe breadth-first generation walk
/// (inspired by Modular
/// `GraphCompiler/lib/MGPDialect/Transforms/SimplifyDeviceGraphDependenciesPass.cpp`
/// `eliminateImpliedDependencies`).
///
/// Returns `(reduced_dag_edges, num_eliminated_edges)`.
pub fn eliminate_implied_dependencies(dag_edges: &[Vec<usize>]) -> (Vec<Vec<usize>>, usize) {
    let n = dag_edges.len();
    let mut reduced = Vec::with_capacity(n);
    let mut eliminated_count = 0usize;

    for direct in dag_edges {
        // Deduplicate while preserving original order (`SmallSetVector<Value, 16> needles`).
        let mut needles: Vec<usize> = Vec::with_capacity(direct.len());
        for &dep in direct {
            if !needles.contains(&dep) {
                needles.push(dep);
            }
        }
        let orig_len = direct.len();

        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut current_gen: Vec<usize> = Vec::new();
        for &needle in &needles {
            if let Some(nexts) = dag_edges.get(needle) {
                for &nx in nexts {
                    if !current_gen.contains(&nx) {
                        current_gen.push(nx);
                    }
                }
            }
        }

        // Track `visited` so even a malformed cycle cannot hang the pass.
        while !current_gen.is_empty() {
            for &v in &current_gen {
                visited.insert(v);
            }
            needles.retain(|v| !current_gen.contains(v));

            let mut next_gen: Vec<usize> = Vec::new();
            for &val in &current_gen {
                if let Some(nexts) = dag_edges.get(val) {
                    for &nx in nexts {
                        if !visited.contains(&nx) && !next_gen.contains(&nx) {
                            next_gen.push(nx);
                        }
                    }
                }
            }
            current_gen = next_gen;
        }

        eliminated_count += orig_len.saturating_sub(needles.len());
        reduced.push(needles);
    }

    (reduced, eliminated_count)
}

pub(super) fn extract_int_list_attr(op: &Operation, key: &str) -> Option<Vec<i64>> {
    match op.attributes.get(key)? {
        Attribute::DenseIntArray(v) => Some(v.clone()),
        Attribute::Array(items) => items
            .iter()
            .map(|a| match a {
                Attribute::Int(i) => Some(*i),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Coalesces groups of `stablehlo.slice` operations that partition a single static axis of a
/// common input tensor with unit stride (`step == 1`) into a unified split specification
/// (inspired by Modular
/// `GraphCompiler/include/GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h`
/// `SliceToSplitPattern`).
///
/// Verifies that:
/// 1. At least 2 `stablehlo.slice` ops consume the same input tensor `X`,
/// 2. All slices slice along the same single static dimension `slice_dim` with unit step,
/// 3. Sorted by `start_indices[slice_dim]`, the slices are non-overlapping and contiguous from `0`
///    to `X.shape[slice_dim]` ("Union of slices must cover all of the input"),
///
/// Stamps `castforge.coalesced_split_axis`, `castforge.coalesced_split_sizes`, and
/// `castforge.coalesced_split_index` onto each slice and fuses their provenance onto the primary
/// slice via [`stamp_fused_provenance`]. Returns the number of partitioning slice groups coalesced.
pub fn coalesce_partitioning_slices_to_split(module: &mut Module) -> usize {
    let mut coalesced_groups = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let mut by_input: IndexMap<ValueId, Vec<OpId>> = IndexMap::new();

        for &op_id in &op_ids {
            if let Some(op) = module.ops.get(op_id)
                && op.is_op("stablehlo", "slice")
                && op.operands.len() == 1
                && op.results.len() == 1
            {
                by_input.entry(op.operands[0]).or_default().push(op_id);
            }
        }

        for (in_val, slice_ops) in by_input {
            if slice_ops.len() < 2 {
                continue;
            }
            let Some(in_info) = module.values.get(in_val) else {
                continue;
            };
            let Type::Tensor(ref in_tt) = in_info.ty else {
                continue;
            };
            let rank = in_tt.shape.len();
            if rank == 0 {
                continue;
            }

            let mut slice_dim_opt: Option<usize> = None;
            // (op_id, start, size, loc)
            let mut intervals: Vec<(OpId, i64, i64, Location)> =
                Vec::with_capacity(slice_ops.len());
            let mut valid = true;

            for &s_id in &slice_ops {
                let Some(s_op) = module.ops.get(s_id) else {
                    valid = false;
                    break;
                };
                if let Some(strides) = extract_int_list_attr(s_op, "strides")
                    && strides.iter().any(|&st| st != 1)
                {
                    valid = false;
                    break;
                }
                let Some(starts) = extract_int_list_attr(s_op, "start_indices") else {
                    valid = false;
                    break;
                };
                if starts.len() != rank {
                    valid = false;
                    break;
                }
                let res_val = s_op.results[0];
                let Some(res_info) = module.values.get(res_val) else {
                    valid = false;
                    break;
                };
                let Type::Tensor(ref res_tt) = res_info.ty else {
                    valid = false;
                    break;
                };
                if res_tt.shape.len() != rank {
                    valid = false;
                    break;
                }

                let mut diff_dims = Vec::new();
                for d in 0..rank {
                    if in_tt.shape[d] != res_tt.shape[d] || starts[d] != 0 {
                        diff_dims.push(d);
                    }
                }
                if diff_dims.len() != 1 {
                    valid = false;
                    break;
                }
                let d = diff_dims[0];
                if let Some(prev_d) = slice_dim_opt
                    && prev_d != d
                {
                    valid = false;
                    break;
                }
                slice_dim_opt = Some(d);

                let Dim::Static(slice_len) = res_tt.shape[d] else {
                    valid = false;
                    break;
                };
                if starts[d] < 0 || slice_len <= 0 {
                    valid = false;
                    break;
                }
                intervals.push((s_id, starts[d], slice_len, s_op.location.clone()));
            }

            let Some(slice_dim) = slice_dim_opt else {
                continue;
            };
            let Dim::Static(input_dim_len) = in_tt.shape[slice_dim] else {
                continue;
            };
            if !valid || input_dim_len <= 0 {
                continue;
            }

            intervals.sort_by_key(|entry| entry.1);

            // Verify contiguous non-overlapping coverage from `0` to `input_dim_len`.
            let mut cursor = 0i64;
            for &(_, start, len, _) in &intervals {
                if start != cursor {
                    valid = false;
                    break;
                }
                cursor += len;
            }
            if !valid || cursor != input_dim_len {
                continue;
            }

            let split_sizes: Vec<i64> = intervals.iter().map(|e| e.2).collect();
            let all_locs: Vec<Location> = intervals.iter().map(|e| e.3.clone()).collect();

            for (idx, &(s_id, _, _, _)) in intervals.iter().enumerate() {
                if let Some(op_mut) = module.ops.get_mut(s_id) {
                    op_mut.attributes.insert(
                        "castforge.coalesced_split_axis".to_string(),
                        Attribute::Int(slice_dim as i64),
                    );
                    op_mut.attributes.insert(
                        "castforge.coalesced_split_sizes".to_string(),
                        Attribute::DenseIntArray(split_sizes.clone()),
                    );
                    op_mut.attributes.insert(
                        "castforge.coalesced_split_index".to_string(),
                        Attribute::Int(idx as i64),
                    );
                    if idx == 0 {
                        stamp_fused_provenance(op_mut, &all_locs);
                    }
                }
            }
            coalesced_groups += 1;
        }
    }

    coalesced_groups
}

/// Returns `Some(axis)` if reshaping from `input_dims` to `output_dims` drops exactly one size-1
/// dimension at `axis` while leaving all other (static, symbolic, or dynamic) dimensions
/// unchanged (inspired by Modular `GraphCompiler/lib/MODialect/Patterns/ShapeOpSinking.cpp`
/// `findSqueezeAxis`).
pub fn find_squeeze_axis(input_dims: &[Dim], output_dims: &[Dim]) -> Option<usize> {
    if input_dims.len() != output_dims.len() + 1 {
        return None;
    }
    let offset = (0..input_dims.len())
        .find(|&i| output_dims.get(i) != Some(&input_dims[i]))
        .unwrap_or(output_dims.len());
    if input_dims[offset] != Dim::Static(1) {
        return None;
    }
    if input_dims[..offset] != output_dims[..offset] {
        return None;
    }
    if input_dims[offset + 1..] != output_dims[offset..] {
        return None;
    }
    Some(offset)
}

/// Returns `Some(axis)` if reshaping from `input_dims` to `output_dims` inserts exactly one size-1
/// dimension at `axis` while leaving all other dimensions unchanged.
pub fn find_unsqueeze_axis(input_dims: &[Dim], output_dims: &[Dim]) -> Option<usize> {
    find_squeeze_axis(output_dims, input_dims)
}

/// Annotates `stablehlo.reshape` operations that drop or insert a single size-1 dimension with
/// `castforge.squeeze_axis` or `castforge.unsqueeze_axis`
/// (inspired by Modular `GraphCompiler/lib/MODialect/Patterns/ShapeOpSinking.cpp`).
///
/// Returns the number of squeeze/unsqueeze reshapes annotated.
pub fn annotate_squeeze_unsqueeze_reshapes(module: &mut Module) -> usize {
    let mut annotated = 0usize;
    let op_ids: Vec<_> = module.ops.keys().collect();

    for op_id in op_ids {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if !op.is_op("stablehlo", "reshape") || op.operands.len() != 1 || op.results.len() != 1 {
            continue;
        }
        let in_val = op.operands[0];
        let out_val = op.results[0];
        let (Some(in_info), Some(out_info)) =
            (module.values.get(in_val), module.values.get(out_val))
        else {
            continue;
        };
        let (Type::Tensor(in_tt), Type::Tensor(out_tt)) = (&in_info.ty, &out_info.ty) else {
            continue;
        };

        let squeeze = find_squeeze_axis(&in_tt.shape, &out_tt.shape);
        let unsqueeze = find_unsqueeze_axis(&in_tt.shape, &out_tt.shape);

        if let Some(axis) = squeeze {
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.squeeze_axis".to_string(),
                    Attribute::Int(axis as i64),
                );
                annotated += 1;
            }
        } else if let Some(axis) = unsqueeze
            && let Some(op_mut) = module.ops.get_mut(op_id)
        {
            op_mut.attributes.insert(
                "castforge.unsqueeze_axis".to_string(),
                Attribute::Int(axis as i64),
            );
            annotated += 1;
        }
    }

    annotated
}
