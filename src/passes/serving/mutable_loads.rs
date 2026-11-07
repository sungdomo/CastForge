//! ARC eviction, FP8 codecs, mutable load fusion, and binary trace stream codecs.

use crate::ir::{
    Attribute, BlockId, DType, Module, OpId, OpName, Operation, Type, Value, ValueDef, ValueId,
};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Polynomial Rolling-Hash Window Matcher & Coordinator Blend Content Index
//
// ============================================================================

/// Fleet-constant 64-bit odd polynomial base for CacheBlend rolling-hash content indexing.
pub const BLEND_POLY_BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Occupancy filter growth factor (`16` slots per live distinct content).
pub const BLEND_INDEX_TABLE_GROWTH: usize = 16;

/// Minimum power-of-two slot count (`1024`) for the occupancy filter.
pub const BLEND_INDEX_MIN_TABLE_SIZE: usize = 1 << 10;

/// Computes sliding-window 64-bit polynomial rolling hashes across `tokens` for window size
/// `window_k` in $O(n)$ time using natural $\bmod 2^{64}$ arithmetic:
/// $$H([x_0, \dots, x_{k-1}]) = \sum_{i=0}^{k-1} x_i \cdot \text{base}^{k - 1 - i} \pmod{2^{64}}$$
#[must_use]
pub fn rolling_hash_windows_u64(tokens: &[u64], window_k: usize, base: u64) -> Vec<u64> {
    if window_k == 0 || tokens.len() < window_k {
        return Vec::new();
    }
    let n = tokens.len();
    let mut out = Vec::with_capacity(n - window_k + 1);

    let mut power = 1u64;
    for _ in 0..(window_k - 1) {
        power = power.wrapping_mul(base);
    }

    let mut h = 0u64;
    for &tok in &tokens[..window_k] {
        h = h.wrapping_mul(base).wrapping_add(tok);
    }
    out.push(h);

    for i in window_k..n {
        let old = tokens[i - window_k];
        let new = tokens[i];
        h = h.wrapping_sub(old.wrapping_mul(power));
        h = h.wrapping_mul(base).wrapping_add(new);
        out.push(h);
    }
    out
}

/// Computes non-overlapping chunk polynomial hashes of length `tokens.len() / window_k`.
#[must_use]
pub fn chunk_hash_windows_u64(tokens: &[u64], window_k: usize, base: u64) -> Vec<u64> {
    if window_k == 0 || tokens.len() < window_k {
        return Vec::new();
    }
    let num_windows = tokens.len() / window_k;
    let mut out = Vec::with_capacity(num_windows);
    for w in 0..num_windows {
        let start = w * window_k;
        let mut h = 0u64;
        for &tok in &tokens[start..start + window_k] {
            h = h.wrapping_mul(base).wrapping_add(tok);
        }
        out.push(h);
    }
    out
}

/// Point-in-time O(1) summary of [`BlendContentFingerprintIndex`] contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendIndexStats {
    /// Distinct chunk token contents indexed.
    pub num_contents: usize,
    /// Total chunk hashes indexed across those contents.
    pub num_chunks: usize,
    /// Total `(chunk_hash, namespace)` claims across those chunks.
    pub num_claims: usize,
    /// Distinct namespaces currently holding indexed content.
    pub num_namespaces: usize,
    /// Number of slots in the power-of-two occupancy filter.
    pub table_size: usize,
}

/// Verified token-exact fragment match returned by [`BlendContentFingerprintIndex::match_tokens`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendFragmentMatch {
    /// Matched chunk's 32-byte `chunk_hash`.
    pub chunk_hash: [u8; 32],
    /// Token offset of the chunk in its originally stored sequence.
    pub old_st: usize,
    /// Starting token index of the matched window in the query sequence.
    pub cur_st: usize,
}

#[derive(Debug, Clone)]
pub(super) struct BlendOccupant {
    token_offset: usize,
    namespaces: IndexSet<String>,
}

#[derive(Debug, Clone)]
pub(super) struct BlendFingerprintEntry {
    token_ids: Vec<u64>,
    occupants: IndexMap<[u8; 32], BlendOccupant>,
}

/// Content fingerprint index over the coordinator key directory's token bindings for CacheBlend
/// fragment lookup.
#[derive(Debug, Clone)]
pub struct BlendContentFingerprintIndex {
    chunk_size: usize,
    probe_stride: usize,
    fingerprint_table: IndexMap<u64, BlendFingerprintEntry>,
    slots: Vec<u8>,
    mask: usize,
    bits_set: usize,
    num_chunks: usize,
    claims_by_namespace: IndexMap<String, usize>,
}

impl BlendContentFingerprintIndex {
    /// Creates an empty [`BlendContentFingerprintIndex`].
    ///
    /// Returns `None` if `chunk_size == 0` or `probe_stride == 0`.
    #[must_use]
    pub fn new(chunk_size: usize, probe_stride: usize) -> Option<Self> {
        if chunk_size == 0 || probe_stride == 0 {
            return None;
        }
        Some(Self {
            chunk_size,
            probe_stride,
            fingerprint_table: IndexMap::new(),
            slots: vec![0u8; BLEND_INDEX_MIN_TABLE_SIZE],
            mask: BLEND_INDEX_MIN_TABLE_SIZE - 1,
            bits_set: 0,
            num_chunks: 0,
            claims_by_namespace: IndexMap::new(),
        })
    }

    fn fingerprint(&self, token_ids: &[u64]) -> u64 {
        chunk_hash_windows_u64(token_ids, self.chunk_size, BLEND_POLY_BASE)[0]
    }

    fn adjust_claim_count(&mut self, namespace: &str, delta: isize) {
        let current = self
            .claims_by_namespace
            .get(namespace)
            .copied()
            .unwrap_or(0) as isize;
        let next = current + delta;
        if next > 0 {
            self.claims_by_namespace
                .insert(namespace.to_string(), next as usize);
        } else {
            self.claims_by_namespace.shift_remove(namespace);
        }
    }

    fn rebuild_table(&mut self) {
        let mut size = BLEND_INDEX_MIN_TABLE_SIZE;
        while size < BLEND_INDEX_TABLE_GROWTH.saturating_mul(self.fingerprint_table.len()) {
            size <<= 1;
        }
        self.slots = vec![0u8; size];
        self.mask = size - 1;
        let mut bits = 0usize;
        for &poly in self.fingerprint_table.keys() {
            let slot = (poly as usize) & self.mask;
            if self.slots[slot] == 0 {
                self.slots[slot] = 1;
                bits += 1;
            }
        }
        self.bits_set = bits;
    }

    /// Records that `namespace` stores `chunk_hash` with `token_ids` at `token_offset`.
    ///
    /// Ignores content whose length does not equal `chunk_size`.
    pub fn add(
        &mut self,
        token_ids: &[u64],
        chunk_hash: [u8; 32],
        token_offset: usize,
        namespace: &str,
    ) {
        if token_ids.len() != self.chunk_size {
            return;
        }
        let poly = self.fingerprint(token_ids);
        let is_new_fp = !self.fingerprint_table.contains_key(&poly);
        if is_new_fp {
            self.fingerprint_table.insert(
                poly,
                BlendFingerprintEntry {
                    token_ids: token_ids.to_vec(),
                    occupants: IndexMap::new(),
                },
            );
            let slot = (poly as usize) & self.mask;
            if self.slots[slot] == 0 {
                self.slots[slot] = 1;
                self.bits_set += 1;
            }
            if BLEND_INDEX_TABLE_GROWTH.saturating_mul(self.fingerprint_table.len())
                > self.slots.len()
            {
                self.rebuild_table();
            }
        }

        let Some(entry) = self.fingerprint_table.get_mut(&poly) else {
            return;
        };
        if let Some(occ) = entry.occupants.get_mut(&chunk_hash) {
            occ.token_offset = token_offset;
            if occ.namespaces.insert(namespace.to_string()) {
                self.adjust_claim_count(namespace, 1);
            }
        } else {
            let mut namespaces = IndexSet::new();
            namespaces.insert(namespace.to_string());
            entry.occupants.insert(
                chunk_hash,
                BlendOccupant {
                    token_offset,
                    namespaces,
                },
            );
            self.num_chunks += 1;
            self.adjust_claim_count(namespace, 1);
        }
    }

    /// Drops `namespace`'s claim on `chunk_hash` under `token_ids`.
    pub fn remove_claim(&mut self, token_ids: &[u64], chunk_hash: &[u8; 32], namespace: &str) {
        if token_ids.len() != self.chunk_size {
            return;
        }
        let poly = self.fingerprint(token_ids);
        let Some(entry) = self.fingerprint_table.get_mut(&poly) else {
            return;
        };
        let Some(occ) = entry.occupants.get_mut(chunk_hash) else {
            return;
        };
        if !occ.namespaces.shift_remove(namespace) {
            return;
        }
        let occ_empty = occ.namespaces.is_empty();
        if occ_empty {
            entry.occupants.shift_remove(chunk_hash);
        }
        let entry_empty = entry.occupants.is_empty();
        self.adjust_claim_count(namespace, -1);
        if !occ_empty {
            return;
        }
        self.num_chunks -= 1;
        if !entry_empty {
            return;
        }
        self.fingerprint_table.shift_remove(&poly);
        if self.bits_set > 2 * self.fingerprint_table.len() {
            self.rebuild_table();
        }
    }

    /// Drops `chunk_hash` and every namespace's claim on it under `token_ids`.
    pub fn remove_chunk(&mut self, token_ids: &[u64], chunk_hash: &[u8; 32]) {
        if token_ids.len() != self.chunk_size {
            return;
        }
        let poly = self.fingerprint(token_ids);
        let Some(entry) = self.fingerprint_table.get_mut(&poly) else {
            return;
        };
        let Some(occ) = entry.occupants.shift_remove(chunk_hash) else {
            return;
        };
        self.num_chunks -= 1;
        let namespaces: Vec<String> = occ.namespaces.into_iter().collect();
        for ns in namespaces {
            self.adjust_claim_count(&ns, -1);
        }
        if self.fingerprint_table[&poly].occupants.is_empty() {
            self.fingerprint_table.shift_remove(&poly);
            if self.bits_set > 2 * self.fingerprint_table.len() {
                self.rebuild_table();
            }
        }
    }

    /// Finds chunks `namespace` can retrieve that are contained in `tokens`, verified token-exact,
    /// returned in ascending `cur_st` order (at most one match per `chunk_hash`).
    #[must_use]
    pub fn match_tokens(&self, tokens: &[u64], namespace: &str) -> Vec<BlendFragmentMatch> {
        if tokens.len() < self.chunk_size {
            return Vec::new();
        }
        let rolling = rolling_hash_windows_u64(tokens, self.chunk_size, BLEND_POLY_BASE);
        let mut matches = Vec::new();
        let mut seen: IndexSet<[u8; 32]> = IndexSet::new();

        for (pos, &fp) in rolling.iter().step_by(self.probe_stride).enumerate() {
            let slot = (fp as usize) & self.mask;
            if self.slots[slot] == 0 {
                continue;
            }
            let Some(entry) = self.fingerprint_table.get(&fp) else {
                continue;
            };
            let cur_st = pos * self.probe_stride;
            for (&chunk_hash, occ) in &entry.occupants {
                if seen.contains(&chunk_hash) || !occ.namespaces.contains(namespace) {
                    continue;
                }
                if tokens[cur_st..cur_st + self.chunk_size] != entry.token_ids[..] {
                    break;
                }
                seen.insert(chunk_hash);
                matches.push(BlendFragmentMatch {
                    chunk_hash,
                    old_st: occ.token_offset,
                    cur_st,
                });
                break;
            }
        }
        matches
    }

    /// Returns a point-in-time O(1) summary of index contents.
    #[must_use]
    pub fn stats(&self) -> BlendIndexStats {
        BlendIndexStats {
            num_contents: self.fingerprint_table.len(),
            num_chunks: self.num_chunks,
            num_claims: self.claims_by_namespace.values().sum(),
            num_namespaces: self.claims_by_namespace.len(),
            table_size: self.slots.len(),
        }
    }
}

// ============================================================================
// FP8 E4M3FN Quantization Codec & Multi-Output Serde Group Planner
// ============================================================================

/// Maximum finite positive value representable in `float8_e4m3fn` (`(1 + 6/8) * 2^8 = 448.0`).
pub const FP8_E4M3FN_MAX_FINITE: f32 = 448.0;

/// Canonical NaN bit-pattern for `float8_e4m3fn` (`s=0, exp=0b1111, mant=0b111`).
pub const FP8_E4M3FN_NAN: u8 = 0x7F;

/// Converts an `f32` scalar to an 8-bit `float8_e4m3fn` byte (1 sign bit, 4-bit exponent with
/// bias 7, 3-bit mantissa, finite-only with `exp=15, mant=7` reserved for NaN and saturating
/// finite magnitudes above `448.0` to `±448.0`).
#[must_use]
pub fn f32_to_fp8_e4m3fn(v: f32) -> u8 {
    if v.is_nan() {
        return FP8_E4M3FN_NAN;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let abs_v = v.abs();
    if abs_v == 0.0 {
        return sign;
    }
    if abs_v >= FP8_E4M3FN_MAX_FINITE {
        return sign | 0x7E;
    }
    // Subnormal range: exp_unbiased = -6, step = 2^-9 = 1/512.
    let min_normal = 1.0f32 / 64.0f32; // 2^-6
    if abs_v < min_normal {
        let scaled = abs_v * 512.0f32;
        let floor = scaled.floor();
        let diff = scaled - floor;
        let mut mant = floor as u8;
        if diff > 0.5 || (diff == 0.5 && (mant & 1) == 1) {
            mant += 1;
        }
        return sign | mant;
    }
    let exp_unbiased = abs_v.log2().floor() as i32;
    let exp_biased = (exp_unbiased + 7) as u8;
    let norm = abs_v / (2.0f32).powi(exp_unbiased) - 1.0f32;
    let scaled_m = norm * 8.0f32;
    let floor_m = scaled_m.floor();
    let diff_m = scaled_m - floor_m;
    let mut mant = floor_m as u8;
    if diff_m > 0.5 || (diff_m == 0.5 && (mant & 1) == 1) {
        mant += 1;
    }
    let (mut e, mut m) = (exp_biased, mant);
    if m == 8 {
        m = 0;
        e += 1;
    }
    if e > 15 || (e == 15 && m >= 7) {
        return sign | 0x7E;
    }
    sign | (e << 3) | m
}

/// Converts an 8-bit `float8_e4m3fn` byte back into an `f32` scalar.
#[must_use]
pub fn fp8_e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        if mant == 0 {
            return if sign < 0.0 { -0.0 } else { 0.0 };
        }
        return sign * (f32::from(mant) / 512.0f32);
    }
    let scale = (2.0f32).powi(i32::from(exp) - 7);
    sign * (1.0f32 + f32::from(mant) / 8.0f32) * scale
}

/// Quantizes an `f32` slice into a contiguous 1-byte-per-element `float8_e4m3fn` buffer.
#[must_use]
pub fn quantize_f32_slice_to_fp8_e4m3fn(src: &[f32]) -> Vec<u8> {
    src.iter().copied().map(f32_to_fp8_e4m3fn).collect()
}

/// Dequantizes a contiguous `float8_e4m3fn` byte slice into `f32` elements.
#[must_use]
pub fn dequantize_fp8_e4m3fn_slice_to_f32(src: &[u8]) -> Vec<f32> {
    src.iter().copied().map(fp8_e4m3fn_to_f32).collect()
}

/// Returns the deterministic serialized byte size for FP8 quantization (`1` byte per element
/// summed across all tensor `shapes`, with no safety-margin inflation).
#[must_use]
pub fn estimate_fp8_serialized_size(shapes: &[Vec<usize>]) -> usize {
    shapes
        .iter()
        .map(|shape| {
            if shape.is_empty() {
                0
            } else {
                shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, d| acc.saturating_mul(d))
            }
        })
        .sum()
}

/// Per-slot specification in a multi-output KV serde group (e.g., `slot 0 = K`, `slot 1 = V`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSerdeSlotSpec {
    /// Human-readable slot role (such as `"K"` or `"V"`).
    pub name: String,
    /// Bytes per element for this slot's serialized format (`1` for FP8, `2` for BF16/FP16).
    pub bytes_per_element: usize,
    /// Whether this slot may be `None` (absent) during tier-split serialization.
    pub allow_absent: bool,
}

/// Schema for a fixed-length multi-output KV serde group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSerdeGroupSchema {
    /// Ordered slot specifications defining `group_size = slots.len()`.
    pub slots: Vec<MultiSerdeSlotSpec>,
}

/// Planned byte-slice layout for a concrete multi-output KV serde group invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSerdeGroupPlan {
    /// Per-slot `(byte_offset, byte_len)` for present slots, or `None` when the slot is absent.
    pub slot_ranges: Vec<Option<(usize, usize)>>,
    /// Total contiguous bytes required in the serialized buffer.
    pub total_bytes: usize,
}

/// Validates that a multi-serde tuple of length `actual_len` matches `expected_len` for `role`
/// (`"src"`, `"dst"`, or `"layout"`).
pub fn validate_multi_serde_group_arity(
    actual_len: usize,
    expected_len: usize,
    role: &str,
) -> Result<(), String> {
    if actual_len != expected_len {
        return Err(format!(
            "MultiSerde {role} group length {actual_len} does not match expected {expected_len}"
        ));
    }
    Ok(())
}

impl MultiSerdeGroupSchema {
    /// Fixed input/output tuple arity (`group_size`) expected by this schema.
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.slots.len()
    }

    /// Validates `layout_shapes` (where `None` denotes an absent slot and `Some(shape)` provides
    /// that slot's tensor dimensions) and plans contiguous `(offset, byte_len)` ranges.
    pub fn plan_group(
        &self,
        layout_shapes: &[Option<Vec<usize>>],
        role: &str,
    ) -> Result<MultiSerdeGroupPlan, String> {
        validate_multi_serde_group_arity(layout_shapes.len(), self.group_size(), role)?;
        let mut slot_ranges = Vec::with_capacity(self.slots.len());
        let mut offset = 0usize;

        for (idx, (spec, maybe_shape)) in self.slots.iter().zip(layout_shapes.iter()).enumerate() {
            match maybe_shape {
                None => {
                    if !spec.allow_absent {
                        return Err(format!(
                            "MultiSerde {role} slot {idx} ('{}') is required and cannot be absent",
                            spec.name
                        ));
                    }
                    slot_ranges.push(None);
                }
                Some(shape) => {
                    let numel = if shape.is_empty() {
                        0
                    } else {
                        shape
                            .iter()
                            .copied()
                            .fold(1usize, |acc, d| acc.saturating_mul(d))
                    };
                    let byte_len = numel.saturating_mul(spec.bytes_per_element);
                    slot_ranges.push(Some((offset, byte_len)));
                    offset = offset.saturating_add(byte_len);
                }
            }
        }

        Ok(MultiSerdeGroupPlan {
            slot_ranges,
            total_bytes: offset,
        })
    }
}

// ============================================================================
// Mutable Load Parallelizer & Exclusive-Chain Kernel Fuser
// ============================================================================

/// Returns `true` if `op_name` is a kernel eligible to fuse an `mo.mutable.load` buffer directly.
#[must_use]
pub fn is_fusible_mutable_load_kernel(op_name: &str) -> bool {
    matches!(op_name, "map.iter.elementwise" | "map.iter.opaque")
}

/// Counts how many operand slots in `block_id` reference `val`.
pub(super) fn count_value_uses_in_block(module: &Module, block_id: BlockId, val: ValueId) -> usize {
    module.blocks[block_id]
        .ops
        .iter()
        .map(|&op_id| {
            module.ops[op_id]
                .operands
                .iter()
                .filter(|&&opd| opd == val)
                .count()
        })
        .sum()
}

/// When `kernel_in_chain` depends on `load_out_chain` exclusively through a tree of single-use
/// `mo.chain.create` operations in `block_id`, rewrites those `mo.chain.create` ops to replace
/// `load_out_chain` with `load_in_chain` so fusing the load into the kernel will not create a
/// chain cycle. Returns `true` if an exclusive path existed and was detached.
pub fn detach_exclusive_chain_path(
    module: &mut Module,
    block_id: BlockId,
    load_out_chain: ValueId,
    load_in_chain: ValueId,
    kernel_in_chain: ValueId,
) -> bool {
    if kernel_in_chain == load_out_chain {
        return false;
    }

    fn analyze_chain_value(
        module: &Module,
        block_id: BlockId,
        v: ValueId,
        load_out_chain: ValueId,
        on_path: &mut IndexSet<OpId>,
        to_rewrite: &mut Vec<OpId>,
    ) -> Option<bool> {
        if v == load_out_chain {
            return Some(true);
        }
        let ValueDef::OpResult { op: def_op, .. } = module.values[v].def else {
            return None;
        };
        if module.ops[def_op].name.full_name() != "mo.chain.create" {
            return None;
        }
        let mut subtree_reaches_load = false;
        let operands = module.ops[def_op].operands.clone();
        for opd in operands {
            if opd == load_out_chain {
                subtree_reaches_load = true;
                continue;
            }
            if let Some(status) =
                analyze_chain_value(module, block_id, opd, load_out_chain, on_path, to_rewrite)
            {
                if !status {
                    return Some(false);
                }
                subtree_reaches_load = true;
            }
        }
        if !subtree_reaches_load {
            return None;
        }
        if count_value_uses_in_block(module, block_id, v) != 1 {
            return Some(false);
        }
        if on_path.insert(def_op) {
            to_rewrite.push(def_op);
        }
        Some(true)
    }

    let mut on_path = IndexSet::new();
    let mut to_rewrite = Vec::new();
    if analyze_chain_value(
        module,
        block_id,
        kernel_in_chain,
        load_out_chain,
        &mut on_path,
        &mut to_rewrite,
    ) != Some(true)
    {
        return false;
    }

    for op_id in to_rewrite {
        for opd in &mut module.ops[op_id].operands {
            if *opd == load_out_chain {
                *opd = load_in_chain;
            }
        }
    }
    true
}

/// Splits each `mo.mutable.load(buf, in_chain) -> (out_tensor, out_chain)` whose `out_tensor` has
/// $n > 1$ uses in the same block into $n$ parallel `mo.mutable.load` ops (each with a single
/// tensor use) joined by an `mo.chain.create` over their output chains, unless a consumer kernel
/// has `hasWriteEffect = true` and also references `buf`. Returns the number of loads split.
pub fn parallelize_multi_use_mutable_loads(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut split_count = 0usize;

    for block_id in block_ids {
        let orig_ops = module.blocks[block_id].ops.clone();
        let mut new_block_ops = Vec::with_capacity(orig_ops.len());

        for op_id in orig_ops {
            let op = &module.ops[op_id];
            if op.name.full_name() != "mo.mutable.load"
                || op.operands.len() < 2
                || op.results.len() < 2
            {
                new_block_ops.push(op_id);
                continue;
            }
            let buf_val = op.operands[0];
            let in_chain = op.operands[1];
            let out_tensor = op.results[0];
            let out_chain = op.results[1];
            let loc = op.location.clone();
            let attrs = op.attributes.clone();

            // Collect all (consumer_op_id, operand_index) uses of `out_tensor` in this block.
            let mut uses: Vec<(OpId, usize)> = Vec::new();
            let mut blocked_by_writer = false;
            for &cand_op in &module.blocks[block_id].ops {
                let cand = &module.ops[cand_op];
                let writes_same_buf = matches!(
                    cand.attributes.get("hasWriteEffect"),
                    Some(Attribute::Bool(true))
                ) && cand.operands.contains(&buf_val);
                for (idx, &opd) in cand.operands.iter().enumerate() {
                    if opd == out_tensor {
                        if writes_same_buf {
                            blocked_by_writer = true;
                        }
                        uses.push((cand_op, idx));
                    }
                }
            }

            if blocked_by_writer || uses.len() <= 1 {
                new_block_ops.push(op_id);
                continue;
            }

            let tensor_ty = module.values[out_tensor].ty.clone();
            let chain_ty = module.values[out_chain].ty.clone();
            let mut new_chains = SmallVec::with_capacity(uses.len());

            for (consumer_op, opd_idx) in uses {
                let new_load_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mo", "mutable.load"),
                    operands: SmallVec::from_slice(&[buf_val, in_chain]),
                    results: SmallVec::new(),
                    attributes: attrs.clone(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let new_t = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: tensor_ty.clone(),
                    def: ValueDef::OpResult {
                        op: new_load_op,
                        index: 0,
                    },
                });
                let new_c = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: chain_ty.clone(),
                    def: ValueDef::OpResult {
                        op: new_load_op,
                        index: 1,
                    },
                });
                module.ops[new_load_op].results = SmallVec::from_slice(&[new_t, new_c]);
                module.ops[consumer_op].operands[opd_idx] = new_t;
                new_block_ops.push(new_load_op);
                new_chains.push(new_c);
            }

            let join_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mo", "chain.create"),
                operands: new_chains,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            let joined_chain = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::OpResult {
                    op: join_op,
                    index: 0,
                },
            });
            module.ops[join_op].results.push(joined_chain);
            new_block_ops.push(join_op);

            for &cand_op in &module.blocks[block_id].ops {
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == out_chain {
                        *opd = joined_chain;
                    }
                }
            }
            split_count += 1;
        }

        module.blocks[block_id].ops = new_block_ops;
    }

    split_count
}

/// Parallelizes multi-use `mo.mutable.load` ops and fuses each single-use `mo.mutable.load` into
/// its compatible `map.iter.elementwise` or `map.iter.opaque` consumer kernel (detaching exclusive
/// `mo.chain.create` paths when needed). Returns the number of fused `mo.mutable.load` operations.
pub fn fuse_mutable_loads_into_kernels(module: &mut Module) -> usize {
    let _ = parallelize_multi_use_mutable_loads(module);
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut fused_count = 0usize;

    for block_id in block_ids {
        let snapshot_ops = module.blocks[block_id].ops.clone();
        let mut erased_loads: IndexSet<OpId> = IndexSet::new();

        for &load_op in &snapshot_ops {
            if module.ops[load_op].name.full_name() != "mo.mutable.load"
                || module.ops[load_op].operands.len() < 2
                || module.ops[load_op].results.len() < 2
            {
                continue;
            }
            let buf_val = module.ops[load_op].operands[0];
            let load_in_chain = module.ops[load_op].operands[1];
            let out_tensor = module.ops[load_op].results[0];
            let load_out_chain = module.ops[load_op].results[1];

            // Find single consumer of `out_tensor` in this block.
            let mut consumer_info: Option<(OpId, usize)> = None;
            let mut multi_use = false;
            for &cand_op in &module.blocks[block_id].ops {
                if erased_loads.contains(&cand_op) {
                    continue;
                }
                for (idx, &opd) in module.ops[cand_op].operands.iter().enumerate() {
                    if opd == out_tensor {
                        if consumer_info.is_some() {
                            multi_use = true;
                        } else {
                            consumer_info = Some((cand_op, idx));
                        }
                    }
                }
            }
            if multi_use {
                continue;
            }
            let Some((kernel_op, arg_idx)) = consumer_info else {
                continue;
            };
            let kernel_name = module.ops[kernel_op].name.full_name();
            if !is_fusible_mutable_load_kernel(&kernel_name) {
                continue;
            }
            // In-place `map.iter.opaque` writing the same buffer cannot safely fuse snapshot loads.
            if kernel_name == "map.iter.opaque" && module.ops[kernel_op].operands.contains(&buf_val)
            {
                continue;
            }

            // Check if the kernel has an existing input chain operand (last operand if chain-typed)
            // that depends on `load_out_chain`.
            let chain_ty = module.values[load_out_chain].ty.clone();
            let existing_chain_opd_idx = module.ops[kernel_op]
                .operands
                .iter()
                .enumerate()
                .rev()
                .find(|&(i, &v)| i != arg_idx && module.values[v].ty == chain_ty)
                .map(|(i, _)| i);

            if let Some(ch_idx) = existing_chain_opd_idx {
                let kernel_in_chain = module.ops[kernel_op].operands[ch_idx];
                if kernel_in_chain != load_out_chain {
                    let _ = detach_exclusive_chain_path(
                        module,
                        block_id,
                        load_out_chain,
                        load_in_chain,
                        kernel_in_chain,
                    );
                } else {
                    module.ops[kernel_op].operands[ch_idx] = load_in_chain;
                }
            } else {
                module.ops[kernel_op].operands.push(load_in_chain);
            }

            // Replace tensor input with direct buffer input.
            module.ops[kernel_op].operands[arg_idx] = buf_val;
            module.ops[kernel_op]
                .attributes
                .insert("hasReadEffect".to_string(), Attribute::Bool(true));
            module.ops[kernel_op].attributes.insert(
                "fusionInfo".to_string(),
                Attribute::String("MutableLoad".to_string()),
            );

            // Ensure kernel has an output chain result and replace `load_out_chain` uses with it.
            let kernel_out_chain = if let Some(&existing_out_ch) = module.ops[kernel_op]
                .results
                .iter()
                .find(|&&r| module.values[r].ty == chain_ty)
            {
                existing_out_ch
            } else {
                let res_idx = module.ops[kernel_op].results.len() as u32;
                let new_out_ch = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: chain_ty,
                    def: ValueDef::OpResult {
                        op: kernel_op,
                        index: res_idx,
                    },
                });
                module.ops[kernel_op].results.push(new_out_ch);
                new_out_ch
            };

            for &cand_op in &module.blocks[block_id].ops {
                if cand_op == kernel_op || cand_op == load_op {
                    continue;
                }
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == load_out_chain {
                        *opd = kernel_out_chain;
                    }
                }
            }

            erased_loads.insert(load_op);
            fused_count += 1;
        }

        if !erased_loads.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|op_id| !erased_loads.contains(op_id));
        }
    }

    fused_count
}

// ============================================================================
// Bidirectional Tensor Layout Inference & Conflict Detector
// ============================================================================

/// Returns the tensor rank of `ty` if it is a ranked `Type::Tensor`.
pub(super) fn tensor_rank_of_type(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor(t) => Some(t.shape.len()),
        _ => None,
    }
}

/// Returns `(input_layout, filter_layout)` for `mo.conv` or `mo.conv_transpose` given `(input_rank,
/// filter_rank)`. Only 2D (rank 4) and 3D (rank 5) convolutions are supported.
pub fn conv_seed_layouts(
    is_transpose: bool,
    input_rank: usize,
    filter_rank: usize,
) -> Result<(&'static str, &'static str), String> {
    if input_rank != filter_rank || (input_rank != 4 && input_rank != 5) {
        return Err(format!(
            "Only 2D/3D (rank 4/5) layouts supported for conv, got ({input_rank}, {filter_rank})"
        ));
    }
    let in_layout = if input_rank == 4 { "NHWC" } else { "NDHWC" };
    let flt_layout = match (is_transpose, filter_rank) {
        (false, 4) => "FCRS",
        (false, _) => "FCQRS",
        (true, 4) => "RSFC",
        (true, _) => "QRSFC",
    };
    Ok((in_layout, flt_layout))
}

/// Returns the list of `ValueId`s across which `op` propagates tensor layout constraints
/// bidirectionally (`mo.guard` drops the leading predicate operand; `mo.transfer` and elementwise
/// ops propagate across all tensor operands and results).
pub(super) fn layout_propagation_values_for_op(op: &Operation) -> Vec<ValueId> {
    let name = op.name.full_name();
    if name == "mo.guard" {
        let mut vals: Vec<ValueId> = op.operands.iter().skip(1).copied().collect();
        vals.extend(op.results.iter().copied());
        vals
    } else if matches!(
        name.as_str(),
        "mo.transfer"
            | "mo.add"
            | "mo.sub"
            | "mo.mul"
            | "mo.div"
            | "mo.relu"
            | "mo.exp"
            | "mo.tanh"
            | "stablehlo.add"
            | "stablehlo.subtract"
            | "stablehlo.multiply"
            | "stablehlo.divide"
            | "stablehlo.abs"
            | "stablehlo.negate"
    ) {
        let mut vals: Vec<ValueId> = op.operands.iter().copied().collect();
        vals.extend(op.results.iter().copied());
        vals
    } else {
        Vec::new()
    }
}

pub(super) fn add_layout_constraint(
    inferred: &mut IndexMap<ValueId, String>,
    op_name: &str,
    val: ValueId,
    layout: &str,
) -> Result<(), String> {
    if let Some(existing) = inferred.get(&val) {
        if existing != layout {
            return Err(format!(
                "{op_name}: layout '{layout}' conflicts with already constrained '{existing}'"
            ));
        }
        return Ok(());
    }
    inferred.insert(val, layout.to_string());
    Ok(())
}

/// Seeds layout constraints from `mo.conv`, `mo.conv_transpose`, and `mo.matmul` ops in `module`,
/// propagates them bidirectionally through layout-preserving ops (`mo.guard`, `mo.transfer`, and
/// elementwise ops), and detects conflicting layout constraints.
pub fn infer_and_propagate_tensor_layouts(
    module: &mut Module,
) -> Result<IndexMap<ValueId, String>, String> {
    let mut inferred: IndexMap<ValueId, String> = IndexMap::new();

    let all_ops: Vec<OpId> = module
        .blocks
        .values()
        .flat_map(|b| b.ops.iter().copied())
        .collect();

    // 1. Seed direct layout constraints from conv, conv_transpose, and matmul ops.
    for &op_id in &all_ops {
        let op = &module.ops[op_id];
        let full = op.name.full_name();
        if (full == "mo.conv" || full == "mo.conv_transpose") && op.operands.len() >= 2 {
            let in_val = op.operands[0];
            let flt_val = op.operands[1];
            let in_rank = tensor_rank_of_type(&module.values[in_val].ty).unwrap_or(0);
            let flt_rank = tensor_rank_of_type(&module.values[flt_val].ty).unwrap_or(0);
            let (in_layout, flt_layout) =
                conv_seed_layouts(full == "mo.conv_transpose", in_rank, flt_rank)?;
            add_layout_constraint(&mut inferred, &full, in_val, in_layout)?;
            add_layout_constraint(&mut inferred, &full, flt_val, flt_layout)?;
        } else if full == "mo.matmul" && op.operands.len() >= 2 {
            let weights = op.operands[1];
            add_layout_constraint(&mut inferred, &full, weights, "KN")?;
        }
    }

    // 2. Bidirectionally propagate layouts across users and defining ops.
    let mut worklist: Vec<ValueId> = inferred.keys().copied().collect();
    let mut history: IndexSet<ValueId> = worklist.iter().copied().collect();

    while let Some(val) = worklist.pop() {
        let layout = inferred[&val].clone();
        let mut affected_ops: Vec<OpId> = Vec::new();
        for &op_id in &all_ops {
            if module.ops[op_id].operands.contains(&val) {
                affected_ops.push(op_id);
            }
        }
        if let ValueDef::OpResult { op: def_op, .. } = module.values[val].def {
            affected_ops.push(def_op);
        }

        for op_id in affected_ops {
            let op = &module.ops[op_id];
            let full = op.name.full_name();
            let prop_vals = layout_propagation_values_for_op(op);
            for target_val in prop_vals {
                if tensor_rank_of_type(&module.values[target_val].ty).is_none() {
                    continue;
                }
                add_layout_constraint(&mut inferred, &full, target_val, &layout)?;
                if history.insert(target_val) {
                    worklist.push(target_val);
                }
            }
        }
    }

    // 3. Annotate ops that touch inferred values with `"inferred_result_layout"` when applicable.
    for &op_id in &all_ops {
        if let Some(&res0) = module.ops[op_id].results.first()
            && let Some(layout) = inferred.get(&res0)
        {
            module.ops[op_id].attributes.insert(
                "inferred_result_layout".to_string(),
                Attribute::String(layout.clone()),
            );
        }
    }

    Ok(inferred)
}

// ============================================================================
// Bottom-Up Call-Graph Slot-Index Globalizer
// ============================================================================

/// Extracts a non-negative integer attribute (handling both `Attribute::Int` and typed integer
/// `Attribute::Opaque` such as `"4 : i64"`).
pub(super) fn extract_u64_attr(attr: &Attribute) -> Option<u64> {
    match attr {
        Attribute::Int(n) if *n >= 0 => Some(*n as u64),
        Attribute::Opaque(s) | Attribute::String(s) => s
            .split(':')
            .next()
            .and_then(|head| head.trim().parse::<u64>().ok()),
        _ => None,
    }
}
/// Extracts a clean symbol name (stripping any leading `'@'`) from an attribute value.
pub(super) fn extract_symbol_attr(attr: &Attribute) -> Option<String> {
    match attr {
        Attribute::String(s) | Attribute::SymbolRef(s) => {
            Some(s.trim_start_matches('@').to_string())
        }
        _ => None,
    }
}

/// Traverses the module's function/model call graph in bottom-up post-order, computing each
/// model's local slot usage from its `mgp.context.create` `"num_slots"` attribute and assigning
/// cumulative subgraph slot offsets to every `mgp.call` / `func.call` site (`"slot_offset"`,
/// `"is_globalized_index" = true`, and a preceding `index.constant` operand). Returns the
/// globalized slot count per function symbol.
pub fn globalize_call_graph_slot_indices(
    module: &mut Module,
) -> Result<IndexMap<String, u64>, String> {
    let func_ids = module.functions();
    if func_ids.len() <= 1 {
        let mut single = IndexMap::new();
        if let Some(&f_id) = func_ids.first()
            && let Some(sym) = module.ops[f_id]
                .attributes
                .get("sym_name")
                .and_then(extract_symbol_attr)
        {
            single.insert(sym, 0);
        }
        return Ok(single);
    }

    let mut sym_to_func: IndexMap<String, OpId> = IndexMap::new();
    let mut local_slots: IndexMap<String, u64> = IndexMap::new();
    let mut ctx_create_op: IndexMap<String, OpId> = IndexMap::new();
    let mut callsites_by_caller: IndexMap<String, Vec<(BlockId, OpId, String)>> = IndexMap::new();

    for &f_id in &func_ids {
        let Some(sym) = module.ops[f_id]
            .attributes
            .get("sym_name")
            .and_then(extract_symbol_attr)
        else {
            continue;
        };
        sym_to_func.insert(sym.clone(), f_id);
        let mut found_ctx: Option<(OpId, u64)> = None;
        let mut calls: Vec<(BlockId, OpId, String)> = Vec::new();

        for &reg_id in &module.ops[f_id].regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                for &op_id in &module.blocks[blk_id].ops {
                    let op = &module.ops[op_id];
                    let full = op.name.full_name();
                    if full == "mgp.context.create" {
                        let slots = op
                            .attributes
                            .get("num_slots")
                            .and_then(extract_u64_attr)
                            .unwrap_or(0);
                        found_ctx = Some((op_id, slots));
                    } else if (full == "mgp.call" || full == "func.call")
                        && let Some(callee) = op
                            .attributes
                            .get("callee")
                            .or_else(|| op.attributes.get("fn"))
                            .and_then(extract_symbol_attr)
                    {
                        calls.push((blk_id, op_id, callee));
                    }
                }
            }
        }

        let Some((c_op, slots)) = found_ctx else {
            return Err(format!(
                "globalize_call_graph_slot_indices: unable to find mgp.context.create in '@{sym}'"
            ));
        };
        local_slots.insert(sym.clone(), slots);
        ctx_create_op.insert(sym.clone(), c_op);
        callsites_by_caller.insert(sym, calls);
    }

    // Build bottom-up post-order over the call graph (detecting cycles).
    fn dfs_postorder(
        sym: &str,
        callsites: &IndexMap<String, Vec<(BlockId, OpId, String)>>,
        state: &mut IndexMap<String, u8>,
        order: &mut Vec<String>,
    ) -> Result<(), String> {
        match state.get(sym).copied() {
            Some(2) => return Ok(()),
            Some(1) => {
                return Err(format!(
                    "globalize_call_graph_slot_indices: cycle in call graph at '@{sym}'"
                ));
            }
            _ => {}
        }
        state.insert(sym.to_string(), 1);
        if let Some(edges) = callsites.get(sym) {
            for (_, _, callee) in edges {
                if callsites.contains_key(callee) {
                    dfs_postorder(callee, callsites, state, order)?;
                }
            }
        }
        state.insert(sym.to_string(), 2);
        order.push(sym.to_string());
        Ok(())
    }

    let mut visit_state: IndexMap<String, u8> = IndexMap::new();
    let mut bottom_up_order: Vec<String> = Vec::new();
    let all_syms: Vec<String> = sym_to_func.keys().cloned().collect();
    for sym in &all_syms {
        dfs_postorder(
            sym,
            &callsites_by_caller,
            &mut visit_state,
            &mut bottom_up_order,
        )?;
    }

    let mut total_slots: IndexMap<String, u64> = local_slots.clone();

    for sym in bottom_up_order {
        let mut running_slots = local_slots[&sym];
        let edges = callsites_by_caller.get(&sym).cloned().unwrap_or_default();
        for (blk_id, call_op_id, callee) in edges {
            let callee_slots = total_slots.get(&callee).copied().unwrap_or(0);
            let offset = running_slots;
            running_slots = running_slots.saturating_add(callee_slots);

            let loc = module.ops[call_op_id].location.clone();
            let mut const_attrs = IndexMap::new();
            const_attrs.insert("value".to_string(), Attribute::Int(offset as i64));
            let const_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("index", "constant"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: const_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            let idx_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: Type::Scalar(DType::I64),
                def: ValueDef::OpResult {
                    op: const_op,
                    index: 0,
                },
            });
            module.ops[const_op].results.push(idx_val);

            if let Some(pos) = module.blocks[blk_id]
                .ops
                .iter()
                .position(|&id| id == call_op_id)
            {
                module.blocks[blk_id].ops.insert(pos, const_op);
            }
            module.ops[call_op_id].operands.push(idx_val);
            module.ops[call_op_id]
                .attributes
                .insert("slot_offset".to_string(), Attribute::Int(offset as i64));
            module.ops[call_op_id]
                .attributes
                .insert("is_globalized_index".to_string(), Attribute::Bool(true));
        }

        total_slots.insert(sym.clone(), running_slots);
        if let Some(&c_op) = ctx_create_op.get(&sym) {
            module.ops[c_op].attributes.insert(
                "globalized_num_slots".to_string(),
                Attribute::Int(running_slots as i64),
            );
        }
    }

    Ok(total_slots)
}

// ============================================================================
// L1/L2 Store Policy Planner & Binary Trace Stream Codec
// ============================================================================

/// Magic bytes at the start of every binary observability trace stream.
pub const TRACE_MAGIC: [u8; 4] = *b"LMCT";

/// On-wire framing layout version for binary observability traces.
pub const TRACE_FORMAT_VERSION: u32 = 1;

/// Storage API trace schema version enforced for `"storage"` level traces.
pub const TRACE_SCHEMA_VERSION: u32 = 2;

/// Descriptor for an available L2 storage adapter in the store controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2AdapterSpec {
    /// Zero-based adapter index.
    pub index: usize,
    /// Human-readable adapter backend label.
    pub name: String,
    /// Whether this adapter accepts new writes.
    pub writable: bool,
}

/// Policy governing which L2 adapters receive newly written L1 keys and whether L1 retains or
/// deletes those keys after a successful L2 store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorePolicyKind {
    /// Replicate all keys to every writable L2 adapter and keep all keys in L1.
    DefaultKeepL1,
    /// Replicate all keys to every writable L2 adapter and immediately delete them from L1
    /// (`"skip_l1"` pure write-buffer mode).
    BufferOnlySkipL1,
}

/// Output plan from [`plan_l1_l2_store_policy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePolicyPlan {
    /// Mapping from L2 adapter index to the ordered list of object keys to persist.
    pub targets_by_adapter: IndexMap<usize, Vec<String>>,
    /// Object keys to delete from L1 once L2 persistence succeeds.
    pub l1_deletions: Vec<String>,
}

/// Computes the L2 adapter target assignment and post-store L1 deletion list for `keys` under
/// `policy`.
#[must_use]
pub fn plan_l1_l2_store_policy(
    policy: StorePolicyKind,
    keys: &[String],
    adapters: &[L2AdapterSpec],
    stored_to_l2_keys: &[String],
) -> StorePolicyPlan {
    let mut targets_by_adapter = IndexMap::new();
    for ad in adapters {
        if ad.writable {
            targets_by_adapter.insert(ad.index, keys.to_vec());
        }
    }
    let l1_deletions = match policy {
        StorePolicyKind::DefaultKeepL1 => Vec::new(),
        StorePolicyKind::BufferOnlySkipL1 => stored_to_l2_keys.to_vec(),
    };
    StorePolicyPlan {
        targets_by_adapter,
        l1_deletions,
    }
}

/// File-level metadata header frame at the start of a binary trace stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceHeader {
    /// Always [`TRACE_MAGIC`] (`b"LMCT"`).
    pub magic: [u8; 4],
    /// Wire framing format version (`1`).
    pub format_version: u32,
    /// Trace level (`"storage"` or `"events"`).
    pub level: String,
    /// Storage trace schema version (`2` for `"storage"` traces).
    pub trace_schema_version: u32,
    /// Monotonic clock start timestamp in nanoseconds.
    pub t_mono_start_ns: u64,
    /// Wall-clock start timestamp in nanoseconds.
    pub t_wall_start_ns: u64,
    /// SHA-256 hex digest of the storage manager configuration.
    pub sm_config_digest: String,
}

/// One captured operation call record in a binary trace stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRecord {
    /// Monotonic nanoseconds elapsed since [`TraceHeader::t_mono_start_ns`].
    pub t_mono_ns: u64,
    /// Wall-clock nanoseconds when the event was published.
    pub t_wall_ns: u64,
    /// Fully qualified call-site name.
    pub qualname: String,
    /// JSON-serialized call arguments.
    pub args_json: String,
}

pub(super) fn push_len_prefixed_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(bytes);
}

pub(super) fn read_be_u32(slice: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        slice[offset],
        slice[offset + 1],
        slice[offset + 2],
        slice[offset + 3],
    ])
}

pub(super) fn read_be_u64(slice: &[u8], offset: usize) -> u64 {
    u64::from_be_bytes([
        slice[offset],
        slice[offset + 1],
        slice[offset + 2],
        slice[offset + 3],
        slice[offset + 4],
        slice[offset + 5],
        slice[offset + 6],
        slice[offset + 7],
    ])
}

pub(super) fn read_len_prefixed_str(frame: &[u8], cursor: &mut usize) -> Result<String, String> {
    if *cursor + 4 > frame.len() {
        return Err("truncated string length in trace frame".to_string());
    }
    let len = read_be_u32(frame, *cursor) as usize;
    *cursor += 4;
    if *cursor + len > frame.len() {
        return Err("truncated string payload in trace frame".to_string());
    }
    let s = std::str::from_utf8(&frame[*cursor..*cursor + len])
        .map_err(|e| format!("invalid UTF-8 in trace frame: {e}"))?
        .to_string();
    *cursor += len;
    Ok(s)
}

pub(super) fn append_length_prefixed_frame(stream: &mut Vec<u8>, payload: &[u8]) {
    stream.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    stream.extend_from_slice(payload);
}

/// Encodes `header` and `records` into a 4-byte big-endian length-prefixed binary trace stream.
#[must_use]
pub fn encode_trace_stream(header: &TraceHeader, records: &[TraceRecord]) -> Vec<u8> {
    let mut stream = Vec::new();
    let mut hdr_buf = Vec::new();
    hdr_buf.extend_from_slice(&header.magic);
    hdr_buf.extend_from_slice(&header.format_version.to_be_bytes());
    push_len_prefixed_str(&mut hdr_buf, &header.level);
    hdr_buf.extend_from_slice(&header.trace_schema_version.to_be_bytes());
    hdr_buf.extend_from_slice(&header.t_mono_start_ns.to_be_bytes());
    hdr_buf.extend_from_slice(&header.t_wall_start_ns.to_be_bytes());
    push_len_prefixed_str(&mut hdr_buf, &header.sm_config_digest);
    append_length_prefixed_frame(&mut stream, &hdr_buf);

    for rec in records {
        let mut rec_buf = Vec::new();
        rec_buf.extend_from_slice(&rec.t_mono_ns.to_be_bytes());
        rec_buf.extend_from_slice(&rec.t_wall_ns.to_be_bytes());
        push_len_prefixed_str(&mut rec_buf, &rec.qualname);
        push_len_prefixed_str(&mut rec_buf, &rec.args_json);
        append_length_prefixed_frame(&mut stream, &rec_buf);
    }
    stream
}

/// Decodes a binary trace stream into `(TraceHeader, Vec<TraceRecord>, had_truncated_tail)`.
/// Enforces strict header validation (`TRACE_MAGIC`, `TRACE_FORMAT_VERSION`, and
/// `TRACE_SCHEMA_VERSION` when `level == "storage"`), and cleanly stops with
/// `had_truncated_tail = true` if a trailing record frame was truncated mid-write.
pub fn decode_trace_stream(stream: &[u8]) -> Result<(TraceHeader, Vec<TraceRecord>, bool), String> {
    if stream.len() < 4 {
        return Err("trace stream is empty or has truncated header length prefix".to_string());
    }
    let hdr_len = read_be_u32(stream, 0) as usize;
    if stream.len() < 4 + hdr_len {
        return Err("trace stream has truncated header frame body".to_string());
    }
    let hdr_frame = &stream[4..4 + hdr_len];
    if hdr_frame.len() < 8 {
        return Err("trace header frame too short".to_string());
    }
    let magic: [u8; 4] = [hdr_frame[0], hdr_frame[1], hdr_frame[2], hdr_frame[3]];
    if magic != TRACE_MAGIC {
        return Err(format!("trace header bad magic: {magic:?}"));
    }
    let format_version = read_be_u32(hdr_frame, 4);
    if format_version != TRACE_FORMAT_VERSION {
        return Err(format!(
            "unsupported trace format_version {format_version} (expected {TRACE_FORMAT_VERSION})"
        ));
    }
    let mut hc = 8usize;
    let level = read_len_prefixed_str(hdr_frame, &mut hc)?;
    if hc + 20 > hdr_frame.len() {
        return Err("trace header truncated numeric fields".to_string());
    }
    let trace_schema_version = read_be_u32(hdr_frame, hc);
    hc += 4;
    if level == "storage" && trace_schema_version != TRACE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported storage trace_schema_version {trace_schema_version}"
        ));
    }
    let t_mono_start_ns = read_be_u64(hdr_frame, hc);
    hc += 8;
    let t_wall_start_ns = read_be_u64(hdr_frame, hc);
    hc += 8;
    let sm_config_digest = read_len_prefixed_str(hdr_frame, &mut hc)?;

    let header = TraceHeader {
        magic,
        format_version,
        level,
        trace_schema_version,
        t_mono_start_ns,
        t_wall_start_ns,
        sm_config_digest,
    };

    let mut pos = 4 + hdr_len;
    let mut records = Vec::new();
    let mut had_truncated_tail = false;

    while pos < stream.len() {
        if pos + 4 > stream.len() {
            had_truncated_tail = true;
            break;
        }
        let rec_len = read_be_u32(stream, pos) as usize;
        pos += 4;
        if pos + rec_len > stream.len() {
            had_truncated_tail = true;
            break;
        }
        let rec_frame = &stream[pos..pos + rec_len];
        pos += rec_len;
        if rec_frame.len() < 16 {
            continue;
        }
        let t_mono_ns = read_be_u64(rec_frame, 0);
        let t_wall_ns = read_be_u64(rec_frame, 8);
        let mut rc = 16usize;
        let Ok(qualname) = read_len_prefixed_str(rec_frame, &mut rc) else {
            continue;
        };
        let Ok(args_json) = read_len_prefixed_str(rec_frame, &mut rc) else {
            continue;
        };
        records.push(TraceRecord {
            t_mono_ns,
            t_wall_ns,
            qualname,
            args_json,
        });
    }

    Ok((header, records, had_truncated_tail))
}
