//! Tiered prefetch, authenticated L2 frames, mutable chain/load passes, and binary trace codecs.

use super::*;
use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, Type, Value, ValueDef, ValueId,
};
use anyhow::Result;
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Multi-Tier Grouped Prefetch Planner, 2D Bitmap Grid & L1 Memory Affinity
// (inspired by LMCache `lmcache/v1/distributed/storage_controllers/prefetch_policy.py`
// and `lmcache/v1/distributed/storage_controllers/utils.py`)
// ============================================================================

/// Memory tag pair identifying an L1 manager's backing segment and residency
/// (e.g., `segment = "cpu"`, `residency = "pinned"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryTagSpec {
    /// Memory segment identifier (e.g., `"cpu"`, `"gpu"`, `"dax"`).
    pub segment: String,
    /// Residency mode (e.g., `"pinned"`, `"pageable"`, `"persistent"`).
    pub residency: String,
}

impl MemoryTagSpec {
    /// Creates a new [`MemoryTagSpec`].
    #[must_use]
    pub fn new(segment: impl Into<String>, residency: impl Into<String>) -> Self {
        Self {
            segment: segment.into(),
            residency: residency.into(),
        }
    }
}

/// Resolves the L1 manager ID whose memory tags match the L2 adapter's highest-priority
/// wanted [`MemoryTagSpec`], breaking ties by L1 write-placement order.
///
/// Pure function: returns `None` if no L1 manager matches any wanted tag.
#[must_use]
pub fn resolve_l1_memory_affinity(
    wanted_l1_tags: &[MemoryTagSpec],
    l1_managers_in_placement_order: &[(usize, MemoryTagSpec)],
) -> Option<usize> {
    for wanted in wanted_l1_tags {
        for (l1_id, tags) in l1_managers_in_placement_order {
            if tags == wanted {
                return Some(*l1_id);
            }
        }
    }
    None
}

/// Rectangular 2D bitmap grid where row `i` represents key group `i` and column `j`
/// represents chunk `j`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmap2D {
    rows: Vec<Vec<bool>>,
}

impl Bitmap2D {
    /// Creates a new `Bitmap2D` from rows of identical width.
    ///
    /// Returns `None` if rows have inconsistent column counts.
    #[must_use]
    pub fn new(rows: Vec<Vec<bool>>) -> Option<Self> {
        if let Some(first) = rows.first() {
            let width = first.len();
            if rows.iter().any(|r| r.len() != width) {
                return None;
            }
        }
        Some(Self { rows })
    }

    /// Creates an all-false `Bitmap2D` of shape `(num_rows, num_cols)`.
    #[must_use]
    pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
        Self {
            rows: vec![vec![false; num_cols]; num_rows],
        }
    }

    /// Returns `(num_rows, num_cols)`, or `(0, 0)` when empty.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        if self.rows.is_empty() {
            (0, 0)
        } else {
            (self.rows.len(), self.rows[0].len())
        }
    }

    /// Returns the total number of set (`true`) bits across all rows.
    #[must_use]
    pub fn popcount(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&b| b)
            .count()
    }

    /// Returns the underlying row bitmaps.
    #[must_use]
    pub fn rows(&self) -> &[Vec<bool>] {
        &self.rows
    }

    /// Row-wise bitwise OR (`self | other`). Returns `None` if shapes differ.
    #[must_use]
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x || y).collect())
            .collect();
        Some(Self { rows })
    }

    /// Row-wise bitwise AND (`self & other`). Returns `None` if shapes differ.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x && y).collect())
            .collect();
        Some(Self { rows })
    }

    /// Row-wise bitwise difference (`self & !other`). Returns `None` if shapes differ.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x && !y).collect())
            .collect();
        Some(Self { rows })
    }
}

/// Maps tier manager/adapter index (`usize`) to a shared-shape [`Bitmap2D`] grid.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TierMapState {
    state: IndexMap<usize, Bitmap2D>,
    shape: Option<(usize, usize)>,
}

impl TierMapState {
    /// Creates an empty [`TierMapState`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a [`Bitmap2D`] for `tier_idx`. Rejects empty grids or shape mismatches.
    pub fn insert(&mut self, tier_idx: usize, grid: Bitmap2D) -> bool {
        let (r, c) = grid.size();
        if r == 0 {
            return false;
        }
        if let Some(expected) = self.shape {
            if expected != (r, c) {
                return false;
            }
        } else {
            self.shape = Some((r, c));
        }
        self.state.insert(tier_idx, grid);
        true
    }

    /// Gets the [`Bitmap2D`] for `tier_idx`.
    #[must_use]
    pub fn get(&self, tier_idx: usize) -> Option<&Bitmap2D> {
        self.state.get(&tier_idx)
    }

    /// Returns the number of populated tier indices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.len()
    }

    /// Returns `true` if no tier indices are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    /// Unions the grids of all stored tier indices, or returns an all-zero `(num_rows, num_cols)`
    /// grid when empty.
    #[must_use]
    pub fn merge_or_zeros(&self, num_rows: usize, num_cols: usize) -> Bitmap2D {
        let mut merged = Bitmap2D::zeros(num_rows, num_cols);
        for grid in self.state.values() {
            if let Some(u) = merged.union(grid) {
                merged = u;
            }
        }
        merged
    }
}

/// Prefetch fetching mode (`"prefix"` vs `"full"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchFetchingPolicy {
    /// Prefix mode: applies `fold -> highest_set_bit -> unfold` with per-group sliding windows.
    Prefix,
    /// Full mode: requires all groups to use full attention (`window <= 0`) and fetches all hits.
    Full,
}

/// Multi-tier prefetch plan partitioning needed `(group, chunk)` cells across L1 managers
/// and L2 adapters so each cell is planned at most once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiTierPrefetchPlan {
    /// Cells served directly from resident L1 managers (`l1_id -> Bitmap2D`).
    pub l1_planned_keys: TierMapState,
    /// Cells loaded from L2 adapters into L1 (`l2_id -> Bitmap2D`).
    pub l2_planned_keys: TierMapState,
}

fn plan_tier_greedy(
    needed: &Bitmap2D,
    locked_keys: &TierMapState,
    sorted_indices: &[usize],
) -> (TierMapState, Bitmap2D) {
    let mut plan = TierMapState::new();
    let mut remaining = needed.clone();
    for &idx in sorted_indices {
        let Some(locked_grid) = locked_keys.get(idx) else {
            continue;
        };
        let Some(local) = remaining.intersect(locked_grid) else {
            continue;
        };
        if local.popcount() == 0 {
            continue;
        }
        if let Some(next_rem) = remaining.difference(&local) {
            remaining = next_rem;
        }
        plan.insert(idx, local);
    }
    (plan, remaining)
}

/// Plans multi-tier grouped prefetch loads across L1 managers and L2 adapters:
/// 1. Validates that `l1_locked_keys` and `l2_locked_keys` match
///    `(group_windows.len(), num_chunks)`.
/// 2. Computes `found = l1_found | l2_found`.
/// 3. Under [`PrefetchFetchingPolicy::Full`], returns an empty plan if any group has a positive
///    sliding window (`w > FULL_ATTENTION_WINDOW`); otherwise `needed = found`.
/// 4. Under [`PrefetchFetchingPolicy::Prefix`], runs [`fold_unfold_hybrid_prefix_bitmaps`] and
///    intersects `retain & found`.
/// 5. Greedily assigns `needed` cells to L1 managers (in ascending index order) first, then to
///    L2 adapters (in ascending index order), ensuring no cell is planned more than once.
#[must_use]
pub fn plan_multi_tier_prefetch_load(
    group_windows: &[i32],
    num_chunks: usize,
    l1_locked_keys: &TierMapState,
    l2_locked_keys: &TierMapState,
    l1_manager_ids: &[usize],
    l2_adapter_ids: &[usize],
    fetching_policy: PrefetchFetchingPolicy,
) -> MultiTierPrefetchPlan {
    let num_groups = group_windows.len();
    if num_groups == 0 || num_chunks == 0 {
        return MultiTierPrefetchPlan::default();
    }
    if l1_locked_keys
        .shape
        .is_some_and(|s| s != (num_groups, num_chunks))
        || l2_locked_keys
            .shape
            .is_some_and(|s| s != (num_groups, num_chunks))
    {
        return MultiTierPrefetchPlan::default();
    }

    let l1_found = l1_locked_keys.merge_or_zeros(num_groups, num_chunks);
    let l2_found = l2_locked_keys.merge_or_zeros(num_groups, num_chunks);
    let Some(found) = l1_found.union(&l2_found) else {
        return MultiTierPrefetchPlan::default();
    };

    let needed = match fetching_policy {
        PrefetchFetchingPolicy::Full => {
            if group_windows.iter().any(|&w| w > FULL_ATTENTION_WINDOW) {
                return MultiTierPrefetchPlan::default();
            }
            found
        }
        PrefetchFetchingPolicy::Prefix => {
            let (_hit_len, _servable, retain_rows) =
                fold_unfold_hybrid_prefix_bitmaps(found.rows(), group_windows);
            let Some(retain_grid) = Bitmap2D::new(retain_rows) else {
                return MultiTierPrefetchPlan::default();
            };
            let Some(masked) = retain_grid.intersect(&found) else {
                return MultiTierPrefetchPlan::default();
            };
            masked
        }
    };

    let mut sorted_l1 = l1_manager_ids.to_vec();
    sorted_l1.sort_unstable();
    sorted_l1.dedup();

    let mut sorted_l2 = l2_adapter_ids.to_vec();
    sorted_l2.sort_unstable();
    sorted_l2.dedup();

    let (l1_plan, rem_after_l1) = plan_tier_greedy(&needed, l1_locked_keys, &sorted_l1);
    let (l2_plan, _rem_after_l2) = plan_tier_greedy(&rem_after_l1, l2_locked_keys, &sorted_l2);

    MultiTierPrefetchPlan {
        l1_planned_keys: l1_plan,
        l2_planned_keys: l2_plan,
    }
}

/// Computes per-key L1 retention flags (`false` = temporary / delete after L1->L0 retrieve,
/// `true` = keep resident in L1 for reuse across requests).
#[must_use]
pub fn plan_l1_prefetch_retention(num_keys: usize, retain_in_l1: bool) -> Vec<bool> {
    vec![retain_in_l1; num_keys]
}

// ============================================================================
// Per-Salt Key Provider & L2 Authenticated KV Frame Codec
// (inspired by LMCache `lmcache/v1/distributed/serde/aesgcm.py` and `key_provider.py`)
// ============================================================================

/// Wire frame version byte for L2-at-rest authenticated KV chunks.
pub const L2_AEAD_VERSION: u8 = 1;

/// 96-bit (12-byte) nonce / initialization vector length per L2 frame.
pub const L2_AEAD_IV_LEN: usize = 12;

/// 128-bit (16-byte) authentication tag length per L2 frame.
pub const L2_AEAD_TAG_LEN: usize = 16;

/// Header length in bytes (`1B version + 12B IV`).
pub const L2_AEAD_HDR_LEN: usize = 1 + L2_AEAD_IV_LEN;

/// Total wire overhead in bytes (`1B version + 12B IV + 16B tag = 29B`).
pub const L2_AEAD_FRAME_OVERHEAD: usize = L2_AEAD_HDR_LEN + L2_AEAD_TAG_LEN;

/// Domain-separation prefix for L2 per-salt key derivation.
pub const L2_HKDF_INFO_PREFIX: &[u8] = b"lmcache-l2-aesgcm-v1";

fn keyed_splitmix_round(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn absorb_bytes_into_state(state: &mut [u64; 4], domain: u64, bytes: &[u8]) {
    state[0] = keyed_splitmix_round(state[0] ^ domain ^ (bytes.len() as u64));
    for (idx, chunk) in bytes.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let v = u64::from_le_bytes(word);
        let lane = idx & 3;
        state[lane] = keyed_splitmix_round(state[lane].wrapping_add(v ^ (idx as u64)));
        let next = (lane + 1) & 3;
        state[next] ^= state[lane].rotate_left(19);
    }
}

/// Memoizing per-`cache_salt` key provider deriving tenant-isolated `16`-byte (`128`-bit) or
/// `32`-byte (`256`-bit) data encryption keys (DEKs) from a master key and domain prefix.
#[derive(Debug, Clone)]
pub struct SaltKeyProvider {
    master_key: Vec<u8>,
    key_len: usize,
    info_prefix: Vec<u8>,
    cache: IndexMap<String, Vec<u8>>,
}

impl SaltKeyProvider {
    /// Creates a new [`SaltKeyProvider`].
    ///
    /// Returns `None` if `master_key` is empty or `aes_bits` is not `128` or `256`.
    #[must_use]
    pub fn new(master_key: Vec<u8>, aes_bits: usize, info_prefix: Vec<u8>) -> Option<Self> {
        if master_key.is_empty() || !matches!(aes_bits, 128 | 256) {
            return None;
        }
        Some(Self {
            master_key,
            key_len: aes_bits / 8,
            info_prefix,
            cache: IndexMap::new(),
        })
    }

    /// Returns the derived key length in bytes (`16` or `32`).
    #[must_use]
    pub fn key_len(&self) -> usize {
        self.key_len
    }

    /// Returns the number of distinct `cache_salt` entries currently memoized.
    #[must_use]
    pub fn cached_salt_count(&self) -> usize {
        self.cache.len()
    }

    /// Derives (or retrieves from the memoization cache) the tenant DEK for `cache_salt`.
    pub fn get_key(&mut self, cache_salt: &str) -> Vec<u8> {
        if let Some(existing) = self.cache.get(cache_salt) {
            return existing.clone();
        }
        let mut state = [
            0x243f_6a88_85a3_08d3,
            0x1319_8a2e_0370_7344,
            0xa409_3822_299f_31d0,
            0x082e_fa98_ec4e_6c89,
        ];
        absorb_bytes_into_state(&mut state, 0x01, &self.master_key);
        absorb_bytes_into_state(&mut state, 0x02, &self.info_prefix);
        absorb_bytes_into_state(&mut state, 0x03, cache_salt.as_bytes());

        let mut derived = Vec::with_capacity(self.key_len);
        let mut counter = 0u64;
        while derived.len() < self.key_len {
            let word = keyed_splitmix_round(
                state[(counter as usize) & 3]
                    .wrapping_add(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            );
            let bytes = word.to_le_bytes();
            let take = (self.key_len - derived.len()).min(8);
            derived.extend_from_slice(&bytes[..take]);
            counter += 1;
        }
        self.cache.insert(cache_salt.to_string(), derived.clone());
        derived
    }
}

/// Computes the exact serialized L2 frame size (`plaintext_bytes + 29`) from a slice of
/// `(num_elements, element_size_bytes)` tensor layout descriptors.
#[must_use]
pub fn estimate_l2_framed_size(layout_shapes_and_elem_bytes: &[(usize, usize)]) -> usize {
    let plaintext_bytes: usize = layout_shapes_and_elem_bytes
        .iter()
        .map(|&(n, elem_sz)| n.saturating_mul(elem_sz))
        .sum();
    plaintext_bytes.saturating_add(L2_AEAD_FRAME_OVERHEAD)
}

fn xor_keystream_in_place(dek: &[u8], iv: &[u8; L2_AEAD_IV_LEN], buf: &mut [u8]) {
    let mut state = [
        0x6a09_e667_f3bc_c908,
        0xbb67_ae85_84ca_a73b,
        0x3c6e_f372_fe94_f82b,
        0xa54f_f53a_5f1d_36f1,
    ];
    absorb_bytes_into_state(&mut state, 0x10, dek);
    absorb_bytes_into_state(&mut state, 0x20, iv);
    for (block_idx, chunk) in buf.chunks_mut(8).enumerate() {
        let ks = keyed_splitmix_round(
            state[block_idx & 3]
                .wrapping_add((block_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95)),
        )
        .to_le_bytes();
        for (b, &k) in chunk.iter_mut().zip(&ks) {
            *b ^= k;
        }
    }
}

fn compute_frame_tag(
    version: u8,
    dek: &[u8],
    iv: &[u8; L2_AEAD_IV_LEN],
    ciphertext: &[u8],
) -> [u8; L2_AEAD_TAG_LEN] {
    let mut state = [
        0x510e_527f_ade6_82d1,
        0x9b05_688c_2b3e_6c1f,
        0x1f83_d9ab_fb41_bd6b,
        0x5be0_cd19_137e_2179,
    ];
    absorb_bytes_into_state(&mut state, 0x30, &[version]);
    absorb_bytes_into_state(&mut state, 0x31, dek);
    absorb_bytes_into_state(&mut state, 0x32, iv);
    absorb_bytes_into_state(&mut state, 0x33, ciphertext);
    let w0 = keyed_splitmix_round(state[0] ^ state[2].rotate_left(17));
    let w1 = keyed_splitmix_round(state[1] ^ state[3].rotate_left(29));
    let mut tag = [0u8; L2_AEAD_TAG_LEN];
    tag[..8].copy_from_slice(&w0.to_le_bytes());
    tag[8..16].copy_from_slice(&w1.to_le_bytes());
    tag
}

/// Seals `plaintext` into an authenticated L2 KV wire frame
/// `[1B version = 1][12B iv][ciphertext || 16B tag]` keyed by `cache_salt`.
pub fn seal_l2_kv_frame(
    provider: &mut SaltKeyProvider,
    cache_salt: &str,
    iv: &[u8; L2_AEAD_IV_LEN],
    plaintext: &[u8],
) -> Vec<u8> {
    let dek = provider.get_key(cache_salt);
    let mut frame = Vec::with_capacity(plaintext.len() + L2_AEAD_FRAME_OVERHEAD);
    frame.push(L2_AEAD_VERSION);
    frame.extend_from_slice(iv);
    frame.extend_from_slice(plaintext);
    xor_keystream_in_place(&dek, iv, &mut frame[L2_AEAD_HDR_LEN..]);
    let tag = compute_frame_tag(L2_AEAD_VERSION, &dek, iv, &frame[L2_AEAD_HDR_LEN..]);
    frame.extend_from_slice(&tag);
    frame
}

/// Opens and authenticates an L2 KV wire frame keyed by `cache_salt`.
///
/// The ciphertext length is taken from `expected_plaintext_len` (allowing `src_blob` to have
/// trailing allocator padding beyond `13 + expected_plaintext_len + 16`). Returns `None` if the
/// header/version is malformed, `src_blob` is shorter than the frame, the wrong `cache_salt` is
/// supplied, or any byte of the header, ciphertext, or tag was tampered with.
pub fn open_l2_kv_frame(
    provider: &mut SaltKeyProvider,
    cache_salt: &str,
    src_blob: &[u8],
    expected_plaintext_len: usize,
) -> Option<Vec<u8>> {
    let frame_end = L2_AEAD_HDR_LEN
        .checked_add(expected_plaintext_len)?
        .checked_add(L2_AEAD_TAG_LEN)?;
    if src_blob.len() < frame_end || src_blob[0] != L2_AEAD_VERSION {
        return None;
    }
    let mut iv = [0u8; L2_AEAD_IV_LEN];
    iv.copy_from_slice(&src_blob[1..L2_AEAD_HDR_LEN]);
    let ct_end = L2_AEAD_HDR_LEN + expected_plaintext_len;
    let ciphertext = &src_blob[L2_AEAD_HDR_LEN..ct_end];
    let stored_tag = &src_blob[ct_end..frame_end];

    let dek = provider.get_key(cache_salt);
    let expected_tag = compute_frame_tag(L2_AEAD_VERSION, &dek, &iv, ciphertext);
    let mut diff = 0u8;
    for (&a, &b) in stored_tag.iter().zip(&expected_tag) {
        diff |= a ^ b;
    }
    if diff != 0 {
        return None;
    }
    let mut plaintext = ciphertext.to_vec();
    xor_keystream_in_place(&dek, &iv, &mut plaintext);
    Some(plaintext)
}

// ============================================================================
// Non-Aliasing Mutable Buffer Chain Sequence Relaxation
// (inspired by Modular `GraphCompiler/lib/MODialect/Transforms/CanonicalizeChainSequences.cpp`)
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct MutableChainOpInfo {
    in_chain: ValueId,
    out_chain: ValueId,
    buffer: ValueId,
    is_write: bool,
    is_multi_device: bool,
}

fn inspect_mutable_chain_op(op: &Operation) -> Option<MutableChainOpInfo> {
    let is_multi_device = matches!(
        op.attributes.get("multi_device"),
        Some(Attribute::Bool(true))
    );
    match op.name.to_string().as_str() {
        "mo.mutable.store" if op.operands.len() >= 3 && op.results.len() == 1 => {
            Some(MutableChainOpInfo {
                in_chain: op.operands[0],
                out_chain: op.results[0],
                buffer: op.operands[2],
                is_write: true,
                is_multi_device,
            })
        }
        "mo.mutable.load" if op.operands.len() >= 2 && op.results.len() == 2 => {
            Some(MutableChainOpInfo {
                in_chain: op.operands[0],
                out_chain: op.results[1],
                buffer: op.operands[1],
                is_write: false,
                is_multi_device,
            })
        }
        _ => None,
    }
}

fn can_safely_reorder_mutable_ops(a: &MutableChainOpInfo, b: &MutableChainOpInfo) -> bool {
    if a.is_multi_device || b.is_multi_device {
        return false;
    }
    if a.buffer != b.buffer {
        return true;
    }
    // Same buffer: safe to reorder iff neither operation writes to the buffer (read-read).
    !a.is_write && !b.is_write
}

fn prune_implied_chains_in_block(
    module: &Module,
    block_id: BlockId,
    chains: &mut SmallVec<[ValueId; 4]>,
) {
    if chains.len() <= 1 {
        return;
    }
    let mut implied: IndexSet<ValueId> = IndexSet::new();
    let mut worklist: Vec<ValueId> = Vec::new();

    for &start in chains.iter() {
        if let Some(ValueDef::OpResult { op: def_id, .. }) = module.values.get(start).map(|v| v.def)
            && module.blocks[block_id].ops.contains(&def_id)
            && let Some(def_op) = module.ops.get(def_id)
        {
            if let Some(info) = inspect_mutable_chain_op(def_op) {
                worklist.push(info.in_chain);
            } else if def_op.is_op("mo", "chain.create") {
                worklist.extend(def_op.operands.iter().copied());
            }
        }
    }

    while let Some(curr) = worklist.pop() {
        if !implied.insert(curr) {
            continue;
        }
        if let Some(ValueDef::OpResult { op: def_id, .. }) = module.values.get(curr).map(|v| v.def)
            && module.blocks[block_id].ops.contains(&def_id)
            && let Some(def_op) = module.ops.get(def_id)
        {
            if let Some(info) = inspect_mutable_chain_op(def_op) {
                worklist.push(info.in_chain);
            } else if def_op.is_op("mo", "chain.create") {
                worklist.extend(def_op.operands.iter().copied());
            }
        }
    }

    chains.retain(|v| !implied.contains(v));
}

/// Relaxes serialized chain dependencies between mutable buffer operations (`mo.mutable.store`,
/// `mo.mutable.load`) when neither operation is marked `multi_device = true` and their buffer
/// accesses do not conflict (distinct buffers, or read-read loads on the same buffer).
///
/// Rewritesboth direct `parent -> child` chain links and upstream `mo.chain.create` joins into
/// parallel fork-join diamonds merged via `mo.chain.create`. Returns the number of chain
/// relaxations performed.
pub fn relax_non_aliasing_mutable_chain_sequences(module: &mut Module) -> usize {
    let mut total_relaxed = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        for _iter in 0..64 {
            let mut changed = false;
            let op_ids = module.blocks[block_id].ops.clone();

            for (pos, &op_id) in op_ids.iter().enumerate() {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(mut_info) = inspect_mutable_chain_op(&op) else {
                    continue;
                };
                let Some(ValueDef::OpResult { op: parent_id, .. }) =
                    module.values.get(mut_info.in_chain).map(|v| v.def)
                else {
                    continue;
                };
                if !module.blocks[block_id].ops.contains(&parent_id) {
                    continue;
                }
                let Some(parent_op) = module.ops.get(parent_id).cloned() else {
                    continue;
                };

                // Case A: Direct mutable parent op in the same block.
                if let Some(parent_info) = inspect_mutable_chain_op(&parent_op)
                    && parent_info.out_chain == mut_info.in_chain
                    && can_safely_reorder_mutable_ops(&mut_info, &parent_info)
                {
                    let chain_ty = module.values[mut_info.out_chain].ty.clone();
                    if let Some(mut_op_mut) = module.ops.get_mut(op_id) {
                        mut_op_mut.operands[0] = parent_info.in_chain;
                    }
                    let merge_op_id = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mo", "chain.create"),
                        operands: SmallVec::from_slice(&[
                            mut_info.out_chain,
                            parent_info.out_chain,
                        ]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: op.location.clone(),
                    });
                    let merged_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: chain_ty,
                        def: ValueDef::OpResult {
                            op: merge_op_id,
                            index: 0,
                        },
                    });
                    module.ops[merge_op_id].results.push(merged_val);
                    module.blocks[block_id].ops.insert(pos + 1, merge_op_id);

                    for &other_id in &module.blocks[block_id].ops {
                        if other_id == merge_op_id {
                            continue;
                        }
                        if let Some(other_op) = module.ops.get_mut(other_id) {
                            for opd in &mut other_op.operands {
                                if *opd == mut_info.out_chain {
                                    *opd = merged_val;
                                }
                            }
                        }
                    }

                    total_relaxed += 1;
                    changed = true;
                    break;
                }

                // Case B: Upstream `mo.chain.create` with one or more non-aliasing mutable inputs.
                if parent_op.is_op("mo", "chain.create") {
                    let mut new_create_operands: SmallVec<[ValueId; 4]> = SmallVec::new();
                    let mut residual_operands: SmallVec<[ValueId; 4]> = SmallVec::new();

                    for &opd in &parent_op.operands {
                        let mut bypassed = false;
                        if let Some(ValueDef::OpResult { op: up_id, .. }) =
                            module.values.get(opd).map(|v| v.def)
                            && module.blocks[block_id].ops.contains(&up_id)
                            && let Some(up_op) = module.ops.get(up_id)
                            && let Some(up_info) = inspect_mutable_chain_op(up_op)
                            && up_info.out_chain == opd
                            && can_safely_reorder_mutable_ops(&mut_info, &up_info)
                        {
                            new_create_operands.push(opd);
                            if !residual_operands.contains(&up_info.in_chain) {
                                residual_operands.push(up_info.in_chain);
                            }
                            bypassed = true;
                        }
                        if !bypassed && !residual_operands.contains(&opd) {
                            residual_operands.push(opd);
                        }
                    }

                    if !new_create_operands.is_empty() && !residual_operands.is_empty() {
                        prune_implied_chains_in_block(module, block_id, &mut residual_operands);
                        let chain_ty = module.values[mut_info.out_chain].ty.clone();
                        let new_in_chain = if residual_operands.len() == 1 {
                            residual_operands[0]
                        } else {
                            let in_create_id = module.ops.insert_with_key(|id| Operation {
                                id,
                                name: OpName::new("mo", "chain.create"),
                                operands: residual_operands,
                                results: SmallVec::new(),
                                attributes: IndexMap::new(),
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: op.location.clone(),
                            });
                            let in_val = module.values.insert_with_key(|id| Value {
                                id,
                                ssa_name: None,
                                ty: chain_ty.clone(),
                                def: ValueDef::OpResult {
                                    op: in_create_id,
                                    index: 0,
                                },
                            });
                            module.ops[in_create_id].results.push(in_val);
                            module.blocks[block_id].ops.insert(pos, in_create_id);
                            in_val
                        };

                        if let Some(mut_op_mut) = module.ops.get_mut(op_id) {
                            mut_op_mut.operands[0] = new_in_chain;
                        }

                        new_create_operands.push(mut_info.out_chain);
                        let merge_op_id = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mo", "chain.create"),
                            operands: new_create_operands,
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: op.location.clone(),
                        });
                        let merged_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: chain_ty,
                            def: ValueDef::OpResult {
                                op: merge_op_id,
                                index: 0,
                            },
                        });
                        module.ops[merge_op_id].results.push(merged_val);
                        let mut_pos = module.blocks[block_id]
                            .ops
                            .iter()
                            .position(|&id| id == op_id)
                            .unwrap_or(pos);
                        module.blocks[block_id].ops.insert(mut_pos + 1, merge_op_id);

                        for &other_id in &module.blocks[block_id].ops {
                            if other_id == merge_op_id {
                                continue;
                            }
                            if let Some(other_op) = module.ops.get_mut(other_id) {
                                for opd in &mut other_op.operands {
                                    if *opd == mut_info.out_chain {
                                        *opd = merged_val;
                                    }
                                }
                            }
                        }

                        total_relaxed += 1;
                        changed = true;
                        break;
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }

    total_relaxed
}

// ============================================================================
// Post-Dominator `runsAfter` Dependency Tree & LCA Explicit Buffer Lifetimes
// (inspired by Modular `GraphCompiler/lib/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.cpp`
// and `GraphCompiler/lib/MGPDialect/Transforms/ExplicitBufferLifetimes.cpp`)
// ============================================================================

/// DAG post-dominator dependency tree for a basic block, rooted at the block terminator.
///
/// Every operation's `runs_after` set contains itself, the block terminator, and all in-block
/// operations transitively consuming its results. Each operation's parent in the tree is its
/// immediate post-dominator: the strict successor `cand != op` in `runs_after[op]` with the
/// largest `runs_after[cand]` cardinality.
#[derive(Debug, Clone)]
pub struct BufferDependencyTree {
    root: OpId,
    parent_map: IndexMap<OpId, OpId>,
    children_map: IndexMap<OpId, Vec<OpId>>,
    depth_map: IndexMap<OpId, usize>,
    pre_index_map: IndexMap<OpId, usize>,
    post_index_map: IndexMap<OpId, usize>,
    op_to_idx: IndexMap<OpId, usize>,
    runs_after: Vec<IndexSet<usize>>,
}

impl BufferDependencyTree {
    /// Builds the [`BufferDependencyTree`] for `block_id`.
    ///
    /// Returns `None` if the block is empty or contains a dataflow cycle.
    #[must_use]
    pub fn build_for_block(module: &Module, block_id: BlockId) -> Option<Self> {
        let block_ops = &module.blocks.get(block_id)?.ops;
        if block_ops.is_empty() {
            return None;
        }
        let num_ops = block_ops.len();
        let terminator = *block_ops.last()?;
        let in_block_set: IndexSet<OpId> = block_ops.iter().copied().collect();

        let mut visited: IndexSet<OpId> = IndexSet::new();
        let mut temp_mark: IndexSet<OpId> = IndexSet::new();
        let mut topo_order: Vec<OpId> = Vec::with_capacity(num_ops);

        fn visit_dag(
            op_id: OpId,
            module: &Module,
            in_block_set: &IndexSet<OpId>,
            visited: &mut IndexSet<OpId>,
            temp_mark: &mut IndexSet<OpId>,
            topo_order: &mut Vec<OpId>,
        ) -> bool {
            if visited.contains(&op_id) {
                return true;
            }
            if !temp_mark.insert(op_id) {
                return false;
            }
            if let Some(op) = module.ops.get(op_id) {
                for &opd in &op.operands {
                    if let Some(ValueDef::OpResult { op: def_id, .. }) =
                        module.values.get(opd).map(|v| v.def)
                        && in_block_set.contains(&def_id)
                        && !visit_dag(def_id, module, in_block_set, visited, temp_mark, topo_order)
                    {
                        return false;
                    }
                }
            }
            temp_mark.swap_remove(&op_id);
            visited.insert(op_id);
            topo_order.push(op_id);
            true
        }

        for &op_id in block_ops.iter().rev() {
            if !visited.contains(&op_id)
                && !visit_dag(
                    op_id,
                    module,
                    &in_block_set,
                    &mut visited,
                    &mut temp_mark,
                    &mut topo_order,
                )
            {
                return None;
            }
        }

        let mut op_to_idx: IndexMap<OpId, usize> = IndexMap::with_capacity(num_ops);
        for (idx, &op_id) in topo_order.iter().enumerate() {
            op_to_idx.insert(op_id, idx);
        }

        let mut users: Vec<Vec<usize>> = vec![Vec::new(); num_ops];
        for (user_idx, &user_id) in topo_order.iter().enumerate() {
            if let Some(user_op) = module.ops.get(user_id) {
                for &opd in &user_op.operands {
                    if let Some(ValueDef::OpResult { op: def_id, .. }) =
                        module.values.get(opd).map(|v| v.def)
                        && let Some(&def_idx) = op_to_idx.get(&def_id)
                        && !users[def_idx].contains(&user_idx)
                    {
                        users[def_idx].push(user_idx);
                    }
                }
            }
        }

        let term_idx = *op_to_idx.get(&terminator)?;
        let mut runs_after: Vec<IndexSet<usize>> = Vec::with_capacity(num_ops);
        for i in 0..num_ops {
            let mut set = IndexSet::new();
            set.insert(i);
            set.insert(term_idx);
            runs_after.push(set);
        }

        for i in (0..num_ops).rev() {
            let user_indices = users[i].clone();
            for u_idx in user_indices {
                let u_set = runs_after[u_idx].clone();
                runs_after[i].extend(u_set);
            }
        }

        let mut parent_map: IndexMap<OpId, OpId> = IndexMap::new();
        let mut children_map: IndexMap<OpId, Vec<OpId>> = IndexMap::new();
        for &op_id in &topo_order {
            children_map.insert(op_id, Vec::new());
        }

        for (i, &op_id) in topo_order.iter().enumerate() {
            if op_id == terminator {
                continue;
            }
            let mut closest: Option<OpId> = None;
            let mut max_sz = 0usize;
            for &cand_idx in &runs_after[i] {
                if cand_idx == i {
                    continue;
                }
                let sz = runs_after[cand_idx].len();
                if sz > max_sz {
                    max_sz = sz;
                    closest = Some(topo_order[cand_idx]);
                }
            }
            let parent_id = closest.unwrap_or(terminator);
            parent_map.insert(op_id, parent_id);
            children_map.entry(parent_id).or_default().push(op_id);
        }

        let mut depth_map: IndexMap<OpId, usize> = IndexMap::new();
        let mut pre_index_map: IndexMap<OpId, usize> = IndexMap::new();
        let mut post_index_map: IndexMap<OpId, usize> = IndexMap::new();
        let mut stack: Vec<(OpId, bool, usize)> = vec![(terminator, true, 0)];
        let mut dfs_counter = 0usize;

        while let Some((node, is_first, depth)) = stack.pop() {
            if is_first {
                depth_map.insert(node, depth);
                pre_index_map.insert(node, dfs_counter);
                dfs_counter += 1;
                stack.push((node, false, depth));
                if let Some(children) = children_map.get(&node) {
                    for &child in children.iter().rev() {
                        stack.push((child, true, depth + 1));
                    }
                }
            } else {
                post_index_map.insert(node, dfs_counter);
                dfs_counter += 1;
            }
        }

        Some(Self {
            root: terminator,
            parent_map,
            children_map,
            depth_map,
            pre_index_map,
            post_index_map,
            op_to_idx,
            runs_after,
        })
    }

    /// Returns the root operation of the dependency tree (the block terminator).
    #[must_use]
    pub fn root(&self) -> OpId {
        self.root
    }

    /// Returns the immediate post-dominator parent of `op`, or `None` for the root.
    #[must_use]
    pub fn parent(&self, op: OpId) -> Option<OpId> {
        self.parent_map.get(&op).copied()
    }

    /// Returns the children of `op` in the dependency tree.
    #[must_use]
    pub fn children(&self, op: OpId) -> &[OpId] {
        self.children_map.get(&op).map_or(&[], Vec::as_slice)
    }

    /// Returns the tree depth of `op` (`0` at the terminator root).
    #[must_use]
    pub fn depth(&self, op: OpId) -> Option<usize> {
        self.depth_map.get(&op).copied()
    }

    /// Returns `(pre_index, post_index)` DFS numbering for `op`.
    #[must_use]
    pub fn dfs_interval(&self, op: OpId) -> Option<(usize, usize)> {
        Some((
            *self.pre_index_map.get(&op)?,
            *self.post_index_map.get(&op)?,
        ))
    }

    /// Returns `true` iff `a` strictly executes after `b` (`a != b` and `a` is in `runs_after[b]`).
    #[must_use]
    pub fn executes_after(&self, a: OpId, b: OpId) -> bool {
        let (Some(&idx_a), Some(&idx_b)) = (self.op_to_idx.get(&a), self.op_to_idx.get(&b)) else {
            return false;
        };
        if idx_a == idx_b {
            return false;
        }
        self.runs_after[idx_b].contains(&idx_a)
    }

    /// Returns `true` iff every operation in `a_ops` strictly executes after every operation in
    /// `b_ops`.
    #[must_use]
    pub fn all_execute_after(&self, a_ops: &[OpId], b_ops: &[OpId]) -> bool {
        a_ops
            .iter()
            .all(|&a| b_ops.iter().all(|&b| self.executes_after(a, b)))
    }

    /// Finds the Lowest Common Ancestor (LCA) of two operations in the post-dominator tree.
    #[must_use]
    pub fn find_lca_pair(&self, mut a: OpId, mut b: OpId) -> Option<OpId> {
        let mut depth_a = *self.depth_map.get(&a)?;
        let mut depth_b = *self.depth_map.get(&b)?;
        while depth_a > depth_b {
            a = *self.parent_map.get(&a)?;
            depth_a -= 1;
        }
        while depth_b > depth_a {
            b = *self.parent_map.get(&b)?;
            depth_b -= 1;
        }
        while a != b {
            a = *self.parent_map.get(&a)?;
            b = *self.parent_map.get(&b)?;
        }
        Some(a)
    }

    /// Finds the Lowest Common Ancestor (LCA) across a slice of operations.
    #[must_use]
    pub fn find_lca(&self, ops: &[OpId]) -> Option<OpId> {
        let mut lca = *ops.first()?;
        for &op in &ops[1..] {
            lca = self.find_lca_pair(lca, op)?;
        }
        Some(lca)
    }
}

fn is_alias_or_lifetime_extending_op(op: &Operation) -> bool {
    matches!(
        op.name.to_string().as_str(),
        "mgp.view"
            | "mgp.tensor.create"
            | "mgp.tensor.slice"
            | "mo.reshape"
            | "mo.slice"
            | "stablehlo.reshape"
            | "stablehlo.slice"
            | "mogg.async.extend_lifetime"
    )
}

fn op_chain_output(module: &Module, op_id: OpId) -> Option<ValueId> {
    let op = module.ops.get(op_id)?;
    for &res in &op.results {
        if let Some(tt) = module.values.get(res).and_then(|v| v.ty.as_tensor())
            && tt.shape == [Dim::Static(1)]
            && tt.dtype == DType::I64
        {
            return Some(res);
        }
    }
    None
}

/// Collects all direct and alias-propagated uses of an `mgp.buffer.alloc` operation within
/// `block_id`, ignoring metadata-only `mgp.tensor.extract_tensor_spec` consumers and returning
/// `None` if the buffer escapes into a control-flow region or `func.call`.
#[must_use]
pub fn collect_alloc_uses_in_block(
    module: &Module,
    block_id: BlockId,
    alloc_id: OpId,
) -> Option<Vec<OpId>> {
    let mut uses: Vec<OpId> = Vec::new();
    let mut stack: Vec<OpId> = vec![alloc_id];
    let mut visited_producers: IndexSet<OpId> = IndexSet::new();

    while let Some(top_id) = stack.pop() {
        if !visited_producers.insert(top_id) {
            continue;
        }
        let top_results = module.ops.get(top_id)?.results.clone();
        for &other_id in &module.blocks.get(block_id)?.ops {
            if other_id == top_id {
                continue;
            }
            let Some(other_op) = module.ops.get(other_id) else {
                continue;
            };
            if !other_op.operands.iter().any(|v| top_results.contains(v)) {
                continue;
            }
            if !other_op.is_op("mgp", "tensor.extract_tensor_spec") && !uses.contains(&other_id) {
                uses.push(other_id);
            }
            if !other_op.regions.is_empty() || other_op.is_op("func", "call") {
                return None;
            }
            if is_alias_or_lifetime_extending_op(other_op) {
                stack.push(other_id);
            }
        }
    }

    if uses.is_empty() { None } else { Some(uses) }
}

/// Inserts explicit `mgp.buffer.free` operations for `mgp.buffer.alloc` operations at the
/// earliest post-dominating ancestor in [`BufferDependencyTree`] that produces a chain output
/// (`tensor<1xi64>`) after all direct and aliasing uses of the allocation have completed.
///
/// Returns the number of `mgp.buffer.free` operations inserted.
pub fn insert_explicit_buffer_frees(module: &mut Module) -> usize {
    let mut frees_added = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let Some(tree) = BufferDependencyTree::build_for_block(module, block_id) else {
            continue;
        };
        let alloc_ids: Vec<OpId> = module.blocks[block_id]
            .ops
            .iter()
            .copied()
            .filter(|&id| {
                module
                    .ops
                    .get(id)
                    .is_some_and(|op| op.is_op("mgp", "buffer.alloc") && !op.results.is_empty())
            })
            .collect();

        for alloc_id in alloc_ids {
            let Some(uses) = collect_alloc_uses_in_block(module, block_id, alloc_id) else {
                continue;
            };
            let Some(mut lca_id) = tree.find_lca(&uses) else {
                continue;
            };
            while op_chain_output(module, lca_id).is_none() {
                let Some(parent_id) = tree.parent(lca_id) else {
                    break;
                };
                lca_id = parent_id;
            }
            let Some(chain_out) = op_chain_output(module, lca_id) else {
                continue;
            };
            let out_buffer = module.ops[alloc_id].results[0];
            let chain_ty = module.values[chain_out].ty.clone();
            let loc = module.ops[lca_id].location.clone();

            let free_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "buffer.free"),
                operands: SmallVec::from_slice(&[chain_out, out_buffer]),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            let free_chain_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::OpResult {
                    op: free_op_id,
                    index: 0,
                },
            });
            module.ops[free_op_id].results.push(free_chain_val);

            let lca_pos = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == lca_id)
                .unwrap_or(0);
            module.blocks[block_id].ops.insert(lca_pos + 1, free_op_id);

            for &other_id in &module.blocks[block_id].ops {
                if other_id == free_op_id {
                    continue;
                }
                if let Some(other_op) = module.ops.get_mut(other_id) {
                    for opd in &mut other_op.operands {
                        if *opd == chain_out {
                            *opd = free_chain_val;
                        }
                    }
                }
            }
            frees_added += 1;
        }
    }

    frees_added
}

// ============================================================================
// Assert-Dependency Chain Predication Pass
// (inspired by Modular
//  `GraphCompiler/lib/MGPDialect/Transforms/PredicateAssertDependenciesPass.cpp`)
// ============================================================================

fn is_chain_value(module: &Module, val_id: ValueId) -> bool {
    module
        .values
        .get(val_id)
        .and_then(|v| v.ty.as_tensor())
        .is_some_and(|tt| tt.shape == [Dim::Static(1)] && tt.dtype == DType::I64)
}

fn is_assert_op(op: &Operation) -> bool {
    matches!(op.name.to_string().as_str(), "mgp.assert" | "mo.assert") && !op.results.is_empty()
}

fn input_chain_operand_index(module: &Module, op: &Operation) -> Option<usize> {
    if op.is_op("func", "return") || op.is_op("mo", "chain.create") {
        return None;
    }
    op.operands.iter().position(|&v| is_chain_value(module, v))
}

/// Predicates every chain-taking operation in a block that consumes (directly or transitively
/// through pure dataflow ops) a value feeding an `mgp.assert` or `mo.assert` operation on that
/// assertion's output chain.
///
/// Stops downward propagation at the first chain-taking operation along each dataflow path so
/// downstream chain-ordered descendants inherit the assertion dependency transitively without
/// redundant barriers. Returns the number of operations predicated on additional assertion chains.
pub fn predicate_assert_dependencies(module: &mut Module) -> usize {
    let mut predicated_ops = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let block_ops = module.blocks[block_id].ops.clone();
        let in_block: IndexSet<OpId> = block_ops.iter().copied().collect();

        let assert_ids: Vec<OpId> = block_ops
            .iter()
            .copied()
            .filter(|&id| module.ops.get(id).is_some_and(is_assert_op))
            .collect();
        if assert_ids.is_empty() {
            continue;
        }

        // 1. Upward propagation: find all non-chain values and defining ops that feed each assert.
        let mut values_that_feed_asserts: IndexMap<ValueId, IndexSet<OpId>> = IndexMap::new();
        for &assert_id in &assert_ids {
            let mut visited_ops: IndexSet<OpId> = IndexSet::new();
            let mut stack: Vec<ValueId> = module.ops[assert_id]
                .operands
                .iter()
                .copied()
                .filter(|&v| !is_chain_value(module, v))
                .collect();

            while let Some(val) = stack.pop() {
                let Some(ValueDef::OpResult { op: def_id, .. }) =
                    module.values.get(val).map(|v| v.def)
                else {
                    values_that_feed_asserts
                        .entry(val)
                        .or_default()
                        .insert(assert_id);
                    continue;
                };
                if !in_block.contains(&def_id) {
                    values_that_feed_asserts
                        .entry(val)
                        .or_default()
                        .insert(assert_id);
                    continue;
                }
                if !visited_ops.insert(def_id) {
                    continue;
                }
                let def_op = &module.ops[def_id];
                for &res in &def_op.results {
                    if !is_chain_value(module, res) {
                        values_that_feed_asserts
                            .entry(res)
                            .or_default()
                            .insert(assert_id);
                    }
                }
                for &opd in &def_op.operands {
                    if !is_chain_value(module, opd) {
                        stack.push(opd);
                    }
                }
            }
        }

        // 2. Downward propagation: propagate assertion sets to consumers, stopping at the first
        //    chain-taking op along each path.
        let mut ops_depending_on_asserts: IndexMap<OpId, IndexSet<OpId>> = IndexMap::new();
        let seed_entries: Vec<(ValueId, IndexSet<OpId>)> = values_that_feed_asserts
            .iter()
            .map(|(&v, s)| (v, s.clone()))
            .collect();

        for (seed_val, asserts) in seed_entries {
            let mut worklist: Vec<ValueId> = vec![seed_val];
            let mut visited_vals: IndexSet<ValueId> = IndexSet::new();

            while let Some(val) = worklist.pop() {
                if is_chain_value(module, val) || !visited_vals.insert(val) {
                    continue;
                }
                for &user_id in &block_ops {
                    if assert_ids.contains(&user_id) {
                        continue;
                    }
                    let Some(user_op) = module.ops.get(user_id) else {
                        continue;
                    };
                    if !user_op.operands.contains(&val) {
                        continue;
                    }
                    let entry = ops_depending_on_asserts.entry(user_id).or_default();
                    let before = entry.len();
                    entry.extend(asserts.iter().copied());
                    if entry.len() == before {
                        continue;
                    }
                    if input_chain_operand_index(module, user_op).is_some() {
                        // Chain-taking op is predicated here; its chain descendants inherit it.
                        continue;
                    }
                    for &res in &user_op.results {
                        if !is_chain_value(module, res) {
                            worklist.push(res);
                        }
                    }
                }
            }
        }

        // 3. Predicate each chain-taking op on `asserts_it_depends_on \ asserts_it_contributed_to`.
        for (op_id, depended_asserts) in ops_depending_on_asserts {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let Some(chain_opd_idx) = input_chain_operand_index(module, &op) else {
                continue;
            };
            let mut diff = depended_asserts;
            if let Some(&first_res) = op.results.first()
                && let Some(contributed) = values_that_feed_asserts.get(&first_res)
            {
                diff.retain(|a| !contributed.contains(a));
            }
            if diff.is_empty() {
                continue;
            }

            let orig_chain = op.operands[chain_opd_idx];
            let mut join_operands: SmallVec<[ValueId; 4]> = SmallVec::from_slice(&[orig_chain]);
            for assert_id in diff {
                let assert_chain = module.ops[assert_id].results[0];
                if !join_operands.contains(&assert_chain) {
                    join_operands.push(assert_chain);
                }
            }
            if join_operands.len() <= 1 {
                continue;
            }

            let chain_ty = module.values[orig_chain].ty.clone();
            let join_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mo", "chain.create"),
                operands: join_operands,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: op.location.clone(),
            });
            let join_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::OpResult {
                    op: join_op_id,
                    index: 0,
                },
            });
            module.ops[join_op_id].results.push(join_val);
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.operands[chain_opd_idx] = join_val;
            }
            let pos = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == op_id)
                .unwrap_or(0);
            module.blocks[block_id].ops.insert(pos, join_op_id);
            predicated_ops += 1;
        }
    }

    predicated_ops
}

// ============================================================================
// Polynomial Rolling-Hash Window Matcher & Coordinator Blend Content Index
// (inspired by LMCache `lmcache/v1/mp_coordinator/blend_index.py` and
//  `lmcache/v1/multiprocess/token_hasher.py`)
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
struct BlendOccupant {
    token_offset: usize,
    namespaces: IndexSet<String>,
}

#[derive(Debug, Clone)]
struct BlendFingerprintEntry {
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

        let entry = self.fingerprint_table.get_mut(&poly).unwrap();
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
// Batch 17 Step 2: FP8 E4M3FN Quantization Codec & Multi-Output Serde Group Planner
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
// Batch 17 Step 3: Mutable Load Parallelizer & Exclusive-Chain Kernel Fuser
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
// Batch 17 Step 4: Bidirectional Tensor Layout Inference & Conflict Detector
// ============================================================================

/// Returns the tensor rank of `ty` if it is a ranked `Type::Tensor`.
fn tensor_rank_of_type(ty: &Type) -> Option<usize> {
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
fn layout_propagation_values_for_op(op: &Operation) -> Vec<ValueId> {
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

fn add_layout_constraint(
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
// Batch 17 Step 5: Bottom-Up Call-Graph Slot-Index Globalizer
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
fn extract_symbol_attr(attr: &Attribute) -> Option<String> {
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
// Batch 18 Step 1: L1/L2 Store Policy Planner & Binary Trace Stream Codec
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

fn push_len_prefixed_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(bytes);
}

fn read_len_prefixed_str(frame: &[u8], cursor: &mut usize) -> Result<String, String> {
    if *cursor + 4 > frame.len() {
        return Err("truncated string length in trace frame".to_string());
    }
    let len = u32::from_be_bytes(frame[*cursor..*cursor + 4].try_into().unwrap()) as usize;
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

fn append_length_prefixed_frame(stream: &mut Vec<u8>, payload: &[u8]) {
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
    let hdr_len = u32::from_be_bytes(stream[0..4].try_into().unwrap()) as usize;
    if stream.len() < 4 + hdr_len {
        return Err("trace stream has truncated header frame body".to_string());
    }
    let hdr_frame = &stream[4..4 + hdr_len];
    if hdr_frame.len() < 8 {
        return Err("trace header frame too short".to_string());
    }
    let magic: [u8; 4] = hdr_frame[0..4].try_into().unwrap();
    if magic != TRACE_MAGIC {
        return Err(format!("trace header bad magic: {magic:?}"));
    }
    let format_version = u32::from_be_bytes(hdr_frame[4..8].try_into().unwrap());
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
    let trace_schema_version = u32::from_be_bytes(hdr_frame[hc..hc + 4].try_into().unwrap());
    hc += 4;
    if level == "storage" && trace_schema_version != TRACE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported storage trace_schema_version {trace_schema_version}"
        ));
    }
    let t_mono_start_ns = u64::from_be_bytes(hdr_frame[hc..hc + 8].try_into().unwrap());
    hc += 8;
    let t_wall_start_ns = u64::from_be_bytes(hdr_frame[hc..hc + 8].try_into().unwrap());
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
        let rec_len = u32::from_be_bytes(stream[pos..pos + 4].try_into().unwrap()) as usize;
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
        let t_mono_ns = u64::from_be_bytes(rec_frame[0..8].try_into().unwrap());
        let t_wall_ns = u64::from_be_bytes(rec_frame[8..16].try_into().unwrap());
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
