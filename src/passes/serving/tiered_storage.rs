//! L1 memory affinity, tiered prefetch, L2 KV frames, and buffer free insertion.

#[allow(unused_imports)]
use super::*;
use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
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

pub(super) fn plan_tier_greedy(
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

pub(super) fn keyed_splitmix_round(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(super) fn absorb_bytes_into_state(state: &mut [u64; 4], domain: u64, bytes: &[u8]) {
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

pub(super) fn xor_keystream_in_place(dek: &[u8], iv: &[u8; L2_AEAD_IV_LEN], buf: &mut [u8]) {
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

pub(super) fn compute_frame_tag(
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
pub(super) struct MutableChainOpInfo {
    in_chain: ValueId,
    out_chain: ValueId,
    buffer: ValueId,
    is_write: bool,
    is_multi_device: bool,
}

pub(super) fn inspect_mutable_chain_op(op: &Operation) -> Option<MutableChainOpInfo> {
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

pub(super) fn can_safely_reorder_mutable_ops(
    a: &MutableChainOpInfo,
    b: &MutableChainOpInfo,
) -> bool {
    if a.is_multi_device || b.is_multi_device {
        return false;
    }
    if a.buffer != b.buffer {
        return true;
    }
    // Same buffer: safe to reorder iff neither operation writes to the buffer (read-read).
    !a.is_write && !b.is_write
}

pub(super) fn prune_implied_chains_in_block(
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

pub(super) fn is_alias_or_lifetime_extending_op(op: &Operation) -> bool {
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

pub(super) fn op_chain_output(module: &Module, op_id: OpId) -> Option<ValueId> {
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

pub(super) fn is_chain_value(module: &Module, val_id: ValueId) -> bool {
    module
        .values
        .get(val_id)
        .and_then(|v| v.ty.as_tensor())
        .is_some_and(|tt| tt.shape == [Dim::Static(1)] && tt.dtype == DType::I64)
}

pub(super) fn is_assert_op(op: &Operation) -> bool {
    matches!(op.name.to_string().as_str(), "mgp.assert" | "mo.assert") && !op.results.is_empty()
}

pub(super) fn input_chain_operand_index(module: &Module, op: &Operation) -> Option<usize> {
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
