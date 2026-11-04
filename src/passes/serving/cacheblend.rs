//! P2P peer discovery, CacheBlend token reordering, and parallel matmul merging.

#[allow(unused_imports)]
use super::*;
use crate::ir::{
    Attribute, Dim, Location, Module, OpId, Operation, TensorType, Type, ValueDef, ValueId,
};
use indexmap::IndexMap;

/// Default consecutive missed coordinator polls before a peer P2P adapter is removed (`3`)
/// (matching LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py` `_MAX_MISSES = 3`).
pub const P2P_DEFAULT_MAX_MISSES: u32 = 3;

/// Peer KV instance connection metadata parsed from coordinator discovery
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py` `_PeerInstance`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2PPeerSpec {
    pub instance_id: String,
    pub ip: String,
    pub p2p_advertised_url: String,
    pub mq_port: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct P2PPeerAdapterEntry {
    adapter_id: u64,
    spec: P2PPeerSpec,
    consecutive_misses: u32,
}

/// P2P peer discovery and adapter lifecycle table with consecutive-miss tolerance
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py` `P2PController`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2PPeerDiscoveryTable {
    self_instance_id: String,
    max_misses: u32,
    next_adapter_id: u64,
    peers: IndexMap<String, P2PPeerAdapterEntry>,
}

impl P2PPeerDiscoveryTable {
    /// Creates a new peer discovery table for `self_instance_id` with `max_misses` tolerance.
    pub fn new(self_instance_id: impl Into<String>, max_misses: u32) -> Self {
        Self {
            self_instance_id: self_instance_id.into(),
            max_misses: max_misses.max(1),
            next_adapter_id: 1,
            peers: IndexMap::new(),
        }
    }

    /// Number of currently active peer adapters.
    pub fn active_peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Reconciles the local peer adapter table against `reported_peers` from the coordinator,
    /// returning `(created_adapter_ids, removed_adapter_ids)`.
    ///
    /// - Ignores `self_instance_id`.
    /// - Resets `consecutive_misses = 0` for present peers with unchanged endpoints.
    /// - Recreates the adapter if a peer's `(ip, p2p_advertised_url, mq_port)` changed.
    /// - Increments `consecutive_misses` for absent peers and evicts them once
    ///   `consecutive_misses >= self.max_misses`.
    pub fn reconcile_peers(&mut self, reported_peers: &[P2PPeerSpec]) -> (Vec<u64>, Vec<u64>) {
        let mut created = Vec::new();
        let mut removed = Vec::new();
        let mut seen_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for peer in reported_peers {
            if peer.instance_id == self.self_instance_id {
                continue;
            }
            seen_ids.insert(peer.instance_id.as_str());
            if let Some(existing) = self.peers.get_mut(&peer.instance_id) {
                if existing.spec == *peer {
                    existing.consecutive_misses = 0;
                } else {
                    removed.push(existing.adapter_id);
                    let new_id = self.next_adapter_id;
                    self.next_adapter_id += 1;
                    *existing = P2PPeerAdapterEntry {
                        adapter_id: new_id,
                        spec: peer.clone(),
                        consecutive_misses: 0,
                    };
                    created.push(new_id);
                }
            } else {
                let new_id = self.next_adapter_id;
                self.next_adapter_id += 1;
                self.peers.insert(
                    peer.instance_id.clone(),
                    P2PPeerAdapterEntry {
                        adapter_id: new_id,
                        spec: peer.clone(),
                        consecutive_misses: 0,
                    },
                );
                created.push(new_id);
            }
        }

        let mut to_evict = Vec::new();
        for (inst_id, entry) in &mut self.peers {
            if !seen_ids.contains(inst_id.as_str()) {
                entry.consecutive_misses += 1;
                if entry.consecutive_misses >= self.max_misses {
                    removed.push(entry.adapter_id);
                    to_evict.push(inst_id.clone());
                }
            }
        }
        for inst_id in to_evict {
            self.peers.shift_remove(&inst_id);
        }

        (created, removed)
    }
}

/// Buckets a flat RPC `keys` list into one `(object_group_id, group_keys)` row per object group in
/// first-seen order, returning an empty `Vec` when `keys` is empty, when object groups receive
/// different numbers of keys, or when any key's `object_group_id` is not in `registered_groups`
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py`
/// `_group_keys_by_object_group`).
pub fn group_p2p_lookup_keys(
    keys: &[KvStorageKeySpec],
    registered_groups: &[u32],
) -> Vec<(u32, Vec<KvStorageKeySpec>)> {
    if keys.is_empty() {
        return Vec::new();
    }
    let reg_set: std::collections::HashSet<u32> = registered_groups.iter().copied().collect();
    let mut by_group: IndexMap<u32, Vec<KvStorageKeySpec>> = IndexMap::new();
    for key in keys {
        if !reg_set.contains(&key.object_group_id) {
            return Vec::new();
        }
        by_group
            .entry(key.object_group_id)
            .or_default()
            .push(key.clone());
    }
    let mut lengths = by_group.values().map(Vec::len);
    let Some(first_len) = lengths.next() else {
        return Vec::new();
    };
    if lengths.any(|l| l != first_len) {
        return Vec::new();
    }
    by_group.into_iter().collect()
}

/// Encodes a [`KvStorageKeySpec`] into a 4-hex-character hash-prefixed wide-column row key
/// (`"{hash_prefix:04x}@{model}[@lg{lg}]@{rank:08x}@{group:x}@{hash:016x}[@{salt}][@layer_{id}]"`)
/// so tablet servers distribute rows uniformly across keyspace prefixes
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/bigtable_key_encoder.py`
/// `BigtableL2KeyEncoder`).
pub fn encode_wide_column_row_key(
    key: &KvStorageKeySpec,
    layer_group_size: usize,
    layer_id: Option<u32>,
) -> String {
    let hash_hex = format!("{:016x}", key.chunk_hash);
    let hash_prefix = &hash_hex[..4];
    let model_with_lg = if layer_group_size > 0 {
        format!("{}@lg{layer_group_size}", key.model_name)
    } else {
        key.model_name.clone()
    };
    let mut row_key = format!(
        "{hash_prefix}@{model_with_lg}@{:08x}@{:x}@{hash_hex}",
        key.kv_rank, key.object_group_id
    );
    if !key.cache_salt.is_empty() {
        row_key.push('@');
        row_key.push_str(&key.cache_salt);
    }
    if let Some(lid) = layer_id {
        row_key.push_str(&format!("@layer_{lid}"));
    }
    row_key
}

/// Fibonacci-hashing polynomial base (`0x9E3779B97F4A7C15`) used for 64-bit chunk and rolling
/// token window fingerprints (inspired by LMCache
/// `lmcache/v1/multiprocess/modules/blend/matcher.py` `BlendTokenRangeMatcher._BASE`).
pub const BLEND_HASH_BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Computes 64-bit polynomial hashes (modulo $2^{64}$ via wrapping arithmetic) over
/// non-overlapping windows of length `chunk_size`, ignoring trailing tokens that do not fill a
/// complete chunk (inspired by LMCache `lmcache/v1/multiprocess/token_hasher.py`
/// `chunk_hash_windows_numba`).
pub fn chunk_poly_hash_windows(token_ids: &[u64], chunk_size: usize, base: u64) -> Vec<u64> {
    if chunk_size == 0 || token_ids.len() < chunk_size {
        return Vec::new();
    }
    let num_windows = token_ids.len() / chunk_size;
    let mut out = Vec::with_capacity(num_windows);
    for w in 0..num_windows {
        let start = w * chunk_size;
        let mut h = 0u64;
        for &tok in &token_ids[start..start + chunk_size] {
            h = h.wrapping_mul(base).wrapping_add(tok);
        }
        out.push(h);
    }
    out
}

/// Computes $O(1)$ sliding-window 64-bit polynomial hashes (modulo $2^{64}$ via wrapping
/// arithmetic) across all `token_ids.len() - window_size + 1` positions (inspired by LMCache
/// `lmcache/v1/multiprocess/token_hasher.py` `rolling_hash_windows_numba`).
pub fn rolling_poly_hash_windows(token_ids: &[u64], window_size: usize, base: u64) -> Vec<u64> {
    if window_size == 0 || token_ids.len() < window_size {
        return Vec::new();
    }
    let num_pos = token_ids.len() - window_size + 1;
    let mut out = Vec::with_capacity(num_pos);
    let mut power = 1u64;
    for _ in 0..window_size.saturating_sub(1) {
        power = power.wrapping_mul(base);
    }
    let mut h = 0u64;
    for &tok in &token_ids[..window_size] {
        h = h.wrapping_mul(base).wrapping_add(tok);
    }
    out.push(h);
    for i in window_size..token_ids.len() {
        let old = token_ids[i - window_size];
        let new = token_ids[i];
        h = h.wrapping_sub(old.wrapping_mul(power));
        h = h.wrapping_mul(base).wrapping_add(new);
        out.push(h);
    }
    out
}

/// A matched sub-sequence chunk span returned by [`BlendTokenRangeMatcher::match_sub_sequence`]
/// (inspired by LMCache `lmcache/v1/multiprocess/custom_types.py` `CBMatchResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlendMatchSpan {
    /// Start token index in the originally indexed sequence (`old_st`).
    pub old_st: usize,
    /// Exclusive end token index in the originally indexed sequence (`old_ed`).
    pub old_ed: usize,
    /// Start token index in the query sequence (`cur_st`).
    pub cur_st: usize,
    /// Exclusive end token index in the query sequence (`cur_ed`).
    pub cur_ed: usize,
    /// Content hash identifier of the matched chunk.
    pub chunk_hash: u64,
}

/// Token-level sliding-window fingerprint matcher over a power-of-two direct-address table of
/// 64-bit chunk polynomial hashes with full-hash collision rejection and content deduplication
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/matcher.py`
/// `BlendTokenRangeMatcher`).
#[derive(Debug, Clone)]
pub struct BlendTokenRangeMatcher {
    chunk_size: usize,
    dedup_content: bool,
    table_size: usize,
    mask: u64,
    table_id: Vec<i64>,
    chunk_token_hash: Vec<Option<u64>>,
    chunk_poly_hash: Vec<u64>,
    token_hash_to_start: std::collections::HashMap<u64, usize>,
    compact_id_to_slot: Vec<i64>,
    token_hash_to_compact_id: std::collections::HashMap<u64, usize>,
}

impl BlendTokenRangeMatcher {
    /// Creates a new matcher with `chunk_size` tokens per chunk and a direct-address table of
    /// size `1 << table_bits` (clamped to `[4, 20]`).
    #[must_use]
    pub fn new(chunk_size: usize, table_bits: u32, dedup_content: bool) -> Self {
        let bits = table_bits.clamp(4, 20);
        let table_size = 1usize << bits;
        Self {
            chunk_size: chunk_size.max(1),
            dedup_content,
            table_size,
            mask: (table_size - 1) as u64,
            table_id: vec![-1; table_size],
            chunk_token_hash: Vec::new(),
            chunk_poly_hash: Vec::new(),
            token_hash_to_start: std::collections::HashMap::new(),
            compact_id_to_slot: vec![-1; table_size],
            token_hash_to_compact_id: std::collections::HashMap::new(),
        }
    }

    fn poly_hash_registered(&self, poly_hash: u64) -> bool {
        let slot = (poly_hash & self.mask) as usize;
        let cid = self.table_id[slot];
        if cid < 0 {
            return false;
        }
        let cid_usize = cid as usize;
        self.chunk_poly_hash.get(cid_usize) == Some(&poly_hash)
            && self
                .chunk_token_hash
                .get(cid_usize)
                .and_then(|h| *h)
                .is_some()
    }

    /// Indexes non-overlapping chunks of `token_ids` with their corresponding `token_hashes` and
    /// `position_offset`, returning the number of newly indexed chunks.
    pub fn on_new_token_hashes(
        &mut self,
        token_ids: &[u64],
        token_hashes: &[u64],
        position_offset: usize,
    ) -> usize {
        let chunk_hashes = chunk_poly_hash_windows(token_ids, self.chunk_size, BLEND_HASH_BASE);
        let n = chunk_hashes.len().min(token_hashes.len());
        if n == 0 {
            return 0;
        }
        let mut new_idxs = Vec::new();
        let mut batch_poly = std::collections::HashSet::new();
        for i in 0..n {
            if self.token_hash_to_compact_id.contains_key(&token_hashes[i]) {
                continue;
            }
            if self.dedup_content {
                let poly_hash = chunk_hashes[i];
                if batch_poly.contains(&poly_hash) || self.poly_hash_registered(poly_hash) {
                    continue;
                }
                batch_poly.insert(poly_hash);
            }
            new_idxs.push(i);
        }
        if new_idxs.is_empty() {
            return 0;
        }
        let base_id = self.chunk_token_hash.len();
        if base_id + new_idxs.len() > self.table_size {
            return 0;
        }
        for (k, &orig_i) in new_idxs.iter().enumerate() {
            let cid = base_id + k;
            let th = token_hashes[orig_i];
            let poly_hash = chunk_hashes[orig_i];
            let slot = (poly_hash & self.mask) as usize;
            self.table_id[slot] = cid as i64;
            self.chunk_token_hash.push(Some(th));
            self.chunk_poly_hash.push(poly_hash);
            self.token_hash_to_start
                .insert(th, position_offset + orig_i * self.chunk_size);
            self.compact_id_to_slot[cid] = slot as i64;
            self.token_hash_to_compact_id.insert(th, cid);
        }
        new_idxs.len()
    }

    /// Probes every sliding-window token position of `token_ids` against the direct-address table,
    /// rejecting bucket collisions via full 64-bit poly-hash verification and deduplicating by
    /// compact chunk ID.
    #[must_use]
    pub fn match_sub_sequence(&self, token_ids: &[u64]) -> Vec<BlendMatchSpan> {
        if token_ids.len() < self.chunk_size || self.chunk_token_hash.is_empty() {
            return Vec::new();
        }
        let rolling = rolling_poly_hash_windows(token_ids, self.chunk_size, BLEND_HASH_BASE);
        let mut seen_cids = std::collections::HashSet::new();
        let mut results = Vec::new();
        for (pos, &poly) in rolling.iter().enumerate() {
            let slot = (poly & self.mask) as usize;
            let cid_i64 = self.table_id[slot];
            if cid_i64 < 0 {
                continue;
            }
            let cid = cid_i64 as usize;
            if seen_cids.contains(&cid) {
                continue;
            }
            if self.chunk_poly_hash.get(cid) != Some(&poly) {
                continue;
            }
            let Some(Some(th)) = self.chunk_token_hash.get(cid).copied() else {
                continue;
            };
            let Some(&old_st) = self.token_hash_to_start.get(&th) else {
                continue;
            };
            seen_cids.insert(cid);
            results.push(BlendMatchSpan {
                old_st,
                old_ed: old_st + self.chunk_size,
                cur_st: pos,
                cur_ed: pos + self.chunk_size,
                chunk_hash: th,
            });
        }
        results
    }

    /// Evicts `token_hashes` from the direct-address table so subsequent probes ignore them.
    pub fn remove_chunks(&mut self, token_hashes: &[u64]) {
        for th in token_hashes {
            let Some(cid) = self.token_hash_to_compact_id.remove(th) else {
                continue;
            };
            let slot = self.compact_id_to_slot[cid];
            if slot >= 0 {
                self.table_id[slot as usize] = -1;
            }
            self.compact_id_to_slot[cid] = -1;
            self.chunk_token_hash[cid] = None;
            self.chunk_poly_hash[cid] = 0;
            self.token_hash_to_start.remove(th);
        }
    }
}

/// Computes the total unique token coverage across `spans` by sorting and merging overlapping
/// `[cur_st, cur_ed)` intervals so overlapping sliding-window matches are not double-counted
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/matcher.py`
/// `_unique_token_coverage`).
#[must_use]
pub fn unique_blend_token_coverage(spans: &[BlendMatchSpan]) -> usize {
    if spans.is_empty() {
        return 0;
    }
    let mut intervals: Vec<(usize, usize)> = spans.iter().map(|s| (s.cur_st, s.cur_ed)).collect();
    intervals.sort_unstable();
    let mut coverage = 0usize;
    let mut cur_end = 0usize;
    let mut first = true;
    for (st, ed) in intervals {
        if first || st >= cur_end {
            coverage += ed.saturating_sub(st);
            first = false;
        } else if ed > cur_end {
            coverage += ed - cur_end;
        }
        cur_end = cur_end.max(ed);
    }
    coverage
}

/// Shortest contiguous run of incoming prompt tokens copied as a piece during CacheBlend prompt
/// reordering; the last `BLEND_REORDER_LMIN` tokens of the prompt are also pinned at the tail
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/reorder.py` `LMIN`).
pub const BLEND_REORDER_LMIN: usize = 16;

/// Maximum short separator run copied between two matched pieces during CacheBlend prompt
/// reordering (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/reorder.py` `GLUE`).
pub const BLEND_REORDER_GLUE: usize = 3;

pub(super) struct BlendReorderCopier<'a> {
    prompt: &'a [u32],
    window_hashes: Vec<u64>,
    sorted_indices: Vec<usize>,
}

impl<'a> BlendReorderCopier<'a> {
    fn new(prompt: &'a [u32]) -> Self {
        let m = prompt.len().saturating_sub(BLEND_REORDER_LMIN - 1);
        let mut window_hashes = Vec::with_capacity(m);
        for i in 0..m {
            let mut h = 0u64;
            for &tok in &prompt[i..i + BLEND_REORDER_LMIN] {
                h = h.wrapping_mul(BLEND_HASH_BASE).wrapping_add(u64::from(tok));
            }
            window_hashes.push(h);
        }
        let mut sorted_indices: Vec<usize> = (0..m).collect();
        sorted_indices.sort_by_key(|&idx| (window_hashes[idx], idx));
        Self {
            prompt,
            window_hashes,
            sorted_indices,
        }
    }

    fn starts(&self, cand: &[u32], k: usize) -> Vec<usize> {
        if k + BLEND_REORDER_LMIN > cand.len() || self.sorted_indices.is_empty() {
            return Vec::new();
        }
        let mut key = 0u64;
        for &tok in &cand[k..k + BLEND_REORDER_LMIN] {
            key = key
                .wrapping_mul(BLEND_HASH_BASE)
                .wrapping_add(u64::from(tok));
        }
        let lo = self
            .sorted_indices
            .partition_point(|&idx| self.window_hashes[idx] < key);
        let hi = self
            .sorted_indices
            .partition_point(|&idx| self.window_hashes[idx] <= key);
        let mut res = self.sorted_indices[lo..hi].to_vec();
        res.sort_unstable();
        res
    }

    fn longest(&self, cand: &[u32], k: usize, used: &[bool]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for i in self.starts(cand, k) {
            if used[i] {
                continue;
            }
            let max_n = (self.prompt.len() - i).min(cand.len() - k);
            if let Some((_, best_len)) = best
                && max_n <= best_len
            {
                break;
            }
            let mut m = max_n;
            for off in 0..max_n {
                if self.prompt[i + off] != cand[k + off] || used[i + off] {
                    m = off;
                    break;
                }
            }
            if m >= BLEND_REORDER_LMIN && best.is_none_or(|(_, bl)| m > bl) {
                best = Some((i, m));
            }
        }
        best
    }

    fn glue(
        &self,
        cand: &[u32],
        k: usize,
        used: &mut [bool],
        pieces: &[(usize, usize)],
    ) -> Option<((usize, usize), (usize, usize))> {
        let prompt_len = self.prompt.len();
        for gl in 1..=BLEND_REORDER_GLUE {
            if k + gl >= cand.len() {
                break;
            }
            if self.starts(cand, k + gl).is_empty() {
                continue;
            }
            let mut cands: Vec<usize> = (0..prompt_len)
                .filter(|&i| {
                    i + gl <= prompt_len
                        && !used[i..i + gl].iter().any(|&u| u)
                        && self.prompt[i..i + gl] == cand[k..k + gl]
                })
                .collect();
            cands.sort_by_key(|&i| {
                let adj = (i > 0 && used[i - 1]) || (i + gl < prompt_len && used[i + gl]);
                (!adj, i)
            });
            for i in cands {
                for u in &mut used[i..i + gl] {
                    *u = true;
                }
                let nxt = self.longest(cand, k + gl, used);
                for u in &mut used[i..i + gl] {
                    *u = false;
                }
                if let Some((nxt_st, nxt_len)) = nxt {
                    let max_end = pieces
                        .iter()
                        .map(|&(_, j)| j)
                        .chain(std::iter::once(nxt_st + nxt_len))
                        .max()
                        .unwrap_or(0);
                    if i + gl <= max_end {
                        return Some(((i, i + gl), (nxt_st, nxt_len)));
                    }
                }
            }
        }
        None
    }

    fn copy(&self, cand: &[u32], keep_prefix: usize) -> Vec<(usize, usize)> {
        let mut used = vec![false; self.prompt.len()];
        let kp = keep_prefix.min(self.prompt.len());
        for u in &mut used[..kp] {
            *u = true;
        }
        let tail_start = self.prompt.len().saturating_sub(BLEND_REORDER_LMIN);
        for u in &mut used[tail_start..] {
            *u = true;
        }
        let mut pieces = if kp > 0 { vec![(0, kp)] } else { Vec::new() };
        let mut k = kp;
        while k < cand.len() {
            let piece = if let Some(p) = self.longest(cand, k, &used) {
                p
            } else if let Some(((gi, gj), nxt)) = self.glue(cand, k, &mut used, &pieces) {
                pieces.push((gi, gj));
                for u in &mut used[gi..gj] {
                    *u = true;
                }
                k += gj - gi;
                nxt
            } else {
                break;
            };
            let (i, m) = piece;
            pieces.push((i, i + m));
            for u in &mut used[i..i + m] {
                *u = true;
            }
            k += m;
        }
        pieces
    }
}

/// Plans a permutation of incoming `prompt` token indices so that `prompt` begins with the
/// longest possible exact-prefix reconstruction of one of `candidates` (followed by all remaining
/// tokens of `prompt` in original order), keeping the first `keep_prefix` tokens in place and
/// pinning the final [`BLEND_REORDER_LMIN`] tokens at the tail (inspired by LMCache
/// `lmcache/v1/multiprocess/modules/blend/reorder.py` `plan`).
#[must_use]
pub fn plan_cacheblend_token_reorder(
    prompt: &[u32],
    candidates: &[Vec<u32>],
    chunk_size: usize,
    baseline_chunks: usize,
    keep_prefix: usize,
) -> Option<Vec<usize>> {
    if chunk_size == 0 || prompt.len() < BLEND_REORDER_LMIN || candidates.is_empty() {
        return None;
    }
    let copier = BlendReorderCopier::new(prompt);
    let kp = keep_prefix.min(prompt.len());
    let mut seen: std::collections::HashSet<&[u32]> = std::collections::HashSet::new();
    type ReorderCandidate<'a> = ((usize, usize, isize), Vec<(usize, usize)>, &'a [u32]);
    let mut best: Option<ReorderCandidate<'_>> = None;

    for cand in candidates {
        if cand.len() < kp || seen.contains(cand.as_slice()) || cand[..kp] != prompt[..kp] {
            continue;
        }
        seen.insert(cand.as_slice());
        let pieces = copier.copy(cand, kp);
        let k: usize = pieces.iter().map(|&(i, j)| j - i).sum();
        let score = (k / chunk_size, k, -(pieces.len() as isize));
        if best.as_ref().is_none_or(|(b_score, _, _)| score > *b_score) {
            best = Some((score, pieces, cand.as_slice()));
        }
    }

    let ((whole_chunks, k, _), pieces, cand) = best?;
    if whole_chunks <= baseline_chunks {
        return None;
    }

    let mut used = vec![false; prompt.len()];
    let mut perm = Vec::with_capacity(prompt.len());
    for &(i, j) in &pieces {
        perm.extend(i..j);
        for u in &mut used[i..j] {
            *u = true;
        }
    }
    if k < cand.len() {
        let max_j = pieces.iter().map(|&(_, j)| j).max().unwrap_or(0);
        for i in 0..max_j.min(prompt.len()) {
            let adj = (i > 0 && used[i - 1]) || (i + 1 < prompt.len() && used[i + 1]);
            if !used[i] && prompt[i] == cand[k] && adj {
                perm.push(i);
                used[i] = true;
                break;
            }
        }
    }
    for (i, &is_used) in used.iter().enumerate() {
        if !is_used {
            perm.push(i);
        }
    }
    let is_identity = perm.iter().enumerate().all(|(idx, &p)| idx == p);
    if is_identity { None } else { Some(perm) }
}

/// Storage tier reported by a cache event batch (inspired by LMCache
/// `lmcache/v1/distributed/api.py` `Tier`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheEventTier {
    /// Volatile host/device L1 cache tier (tied to emitter incarnation).
    L1,
    /// Persistent L2 storage tier (survives emitter restarts).
    L2,
}

/// Metadata header of an emitter's cache event batch offered to [`CoordinatorEventGate`]
/// (inspired by LMCache `lmcache/v1/mp_coordinator/api.py` `CacheEventBatch`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEventBatchSpec {
    /// Unique identifier of the emitting worker instance.
    pub instance_id: String,
    /// Monotonically increasing process incarnation of the emitter.
    pub incarnation: u64,
    /// 1-based sequence number of the batch within `incarnation`.
    pub seq: u64,
    /// Cumulative number of events dropped by the emitter prior to transmission.
    pub dropped_events: u64,
    /// Cache tier affected by this batch.
    pub tier: CacheEventTier,
}

/// Admission outcome returned by [`CoordinatorEventGate::ingest`] (inspired by LMCache
/// `lmcache/v1/mp_coordinator/ingest/event_gate.py` `IngestResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventIngestOutcome {
    /// Batch was admitted and forwarded to downstream state views.
    Admitted,
    /// Batch was dropped because its `seq` was already processed for the current incarnation.
    Duplicate,
    /// Batch was dropped because its `incarnation` was older than the active or departed cursor.
    StaleIncarnation,
}

/// Per-emitter stream cursor state maintained by [`CoordinatorEventGate`] (inspired by LMCache
/// `lmcache/v1/mp_coordinator/ingest/event_gate.py` `_StreamCursor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamCursorState {
    /// Emitter incarnation tracked by this cursor.
    pub incarnation: u64,
    /// Highest `seq` admitted from `incarnation`.
    pub last_seq: u64,
    /// Whether sequence gaps or emitter-side dropped events were detected on this stream.
    pub gap_detected: bool,
    /// Last reported cumulative `dropped_events` (`None` when a stream joined mid-flight at
    /// `seq > 1` before its first baseline report).
    pub last_dropped_events: Option<u64>,
    /// Number of skipped `seq` batches observed since the gate began tracking this stream.
    pub missing_batches: u64,
    /// Total emitter-dropped events observed on this stream.
    pub events_dropped: u64,
}

/// Ordering, sequence-deduplication, gap-detection, and incarnation-fencing gate for distributed
/// cache event streams (inspired by LMCache `lmcache/v1/mp_coordinator/ingest/event_gate.py`
/// `EventGate`).
#[derive(Debug, Clone, Default)]
pub struct CoordinatorEventGate {
    cursors: IndexMap<String, StreamCursorState>,
    departed: IndexMap<String, u64>,
    fenced_instances: Vec<String>,
    /// Total batches admitted across all emitters.
    pub batches_applied: u64,
    /// Total duplicate batches dropped across all emitters.
    pub batches_duplicate: u64,
    /// Total stale-incarnation batches dropped across all emitters.
    pub batches_stale: u64,
    /// Total skipped batch sequence numbers across tracked emitters.
    pub batches_missing: u64,
    /// Total emitter-dropped events reported across all emitters.
    pub events_dropped: u64,
}

impl CoordinatorEventGate {
    /// Creates an empty [`CoordinatorEventGate`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Offers `batch` to the gate, enforcing incarnation fencing, sequence deduplication, and
    /// loss/gap accounting.
    pub fn ingest(&mut self, batch: &CacheEventBatchSpec) -> EventIngestOutcome {
        if let Some(&dep_inc) = self.departed.get(&batch.instance_id) {
            if batch.incarnation > dep_inc {
                self.departed.swap_remove(&batch.instance_id);
            } else if batch.tier == CacheEventTier::L1 {
                self.batches_stale += 1;
                return EventIngestOutcome::StaleIncarnation;
            }
        }

        let tracked = self.cursors.contains_key(&batch.instance_id);
        if let Some(existing) = self.cursors.get(&batch.instance_id) {
            if batch.incarnation < existing.incarnation {
                self.batches_stale += 1;
                return EventIngestOutcome::StaleIncarnation;
            }
            if batch.incarnation > existing.incarnation {
                self.fenced_instances.push(batch.instance_id.clone());
                self.cursors.swap_remove(&batch.instance_id);
            } else if batch.seq <= existing.last_seq {
                self.batches_duplicate += 1;
                return EventIngestOutcome::Duplicate;
            }
        }

        let cursor = self
            .cursors
            .entry(batch.instance_id.clone())
            .or_insert_with(|| StreamCursorState {
                incarnation: batch.incarnation,
                last_seq: 0,
                gap_detected: false,
                last_dropped_events: if tracked || batch.seq == 1 {
                    Some(0)
                } else {
                    None
                },
                missing_batches: 0,
                events_dropped: 0,
            });

        if batch.seq > cursor.last_seq + 1 {
            let skipped = batch.seq - cursor.last_seq - 1;
            cursor.gap_detected = true;
            if tracked {
                cursor.missing_batches += skipped;
                self.batches_missing += skipped;
            }
        }
        if let Some(last_dropped) = cursor.last_dropped_events
            && batch.dropped_events > last_dropped
        {
            let dropped = batch.dropped_events - last_dropped;
            cursor.gap_detected = true;
            cursor.events_dropped += dropped;
            self.events_dropped += dropped;
        }
        cursor.last_dropped_events = Some(batch.dropped_events);
        cursor.last_seq = batch.seq;
        self.batches_applied += 1;
        EventIngestOutcome::Admitted
    }

    /// Fences `instance_id` and records its departed incarnation so stale L1 batches are rejected
    /// while L2 batches or newer incarnations can still proceed.
    pub fn drop_instance(&mut self, instance_id: &str) {
        self.fenced_instances.push(instance_id.to_string());
        if let Some(cursor) = self.cursors.swap_remove(instance_id) {
            self.departed
                .insert(instance_id.to_string(), cursor.incarnation);
        }
    }

    /// Clears the departed fence record for `instance_id` after explicit re-registration.
    pub fn readmit_instance(&mut self, instance_id: &str) {
        self.departed.swap_remove(instance_id);
    }

    /// Returns the active cursor for `instance_id`, if any.
    #[must_use]
    pub fn cursor(&self, instance_id: &str) -> Option<&StreamCursorState> {
        self.cursors.get(instance_id)
    }

    /// Returns the ordered list of instance IDs fenced due to incarnation restarts or drops.
    #[must_use]
    pub fn fenced_instances(&self) -> &[String] {
        &self.fenced_instances
    }
}

pub(super) fn is_mergeable_matmul_op(op: &Operation) -> bool {
    op.is_op("stablehlo", "dot_general")
        || op.is_op("stablehlo", "dot")
        || op.is_op("linalg", "matmul")
        || op.is_op("mo", "matmul")
        || op.is_op("mo", "batch_matmul")
        || op.is_op("tosa", "matmul")
}

/// Merges sibling matrix multiplications (`matmul(A, B1) -> C1`, `matmul(A, B2) -> C2`, ...) that
/// share the same LHS activation tensor `A` into a single weight concatenation
/// (`stablehlo.concatenate`), a single combined matmul, and a multi-result `mo.split`, skipping
/// groups whose RHS weights are produced by a transpose or whose results all feed directly into
/// `func.return` (inspired by Modular `GraphCompiler/lib/MODialect/Patterns/Algebraic.cpp`
/// `MergeParallelMatmulPattern`).
pub fn merge_parallel_matmuls(module: &mut Module) -> usize {
    let mut merged_groups = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let Some(&reg_id) = module.ops[func_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let block_ops = module.blocks[block_id].ops.clone();

        let mut op_pos: std::collections::HashMap<OpId, usize> = std::collections::HashMap::new();
        let mut val_users: std::collections::HashMap<ValueId, Vec<OpId>> =
            std::collections::HashMap::new();
        for (idx, &op_id) in block_ops.iter().enumerate() {
            op_pos.insert(op_id, idx);
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    val_users.entry(v).or_default().push(op_id);
                }
            }
        }

        let mut by_lhs: IndexMap<ValueId, Vec<OpId>> = IndexMap::new();
        for &op_id in &block_ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !is_mergeable_matmul_op(op) || op.operands.len() != 2 || op.results.len() != 1 {
                continue;
            }
            by_lhs.entry(op.operands[0]).or_default().push(op_id);
        }

        let mut to_remove: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        let mut insertions: Vec<(OpId, Vec<OpId>)> = Vec::new();

        for (lhs_val, matmul_ids) in by_lhs {
            if matmul_ids.len() < 2 {
                continue;
            }

            // Skip if every matmul user in this group only feeds directly into `func.return`
            // or `mo.output` (outputs would force the split to materialize, which is slower).
            let all_feed_only_outputs = matmul_ids.iter().all(|&m_id| {
                let Some(m_op) = module.ops.get(m_id) else {
                    return false;
                };
                let res_val = m_op.results[0];
                let Some(users) = val_users.get(&res_val) else {
                    return false;
                };
                !users.is_empty()
                    && users.iter().all(|&u_id| {
                        module
                            .ops
                            .get(u_id)
                            .is_some_and(|u| u.is_op("func", "return") || u.is_op("mo", "output"))
                    })
            });
            if all_feed_only_outputs {
                continue;
            }

            let first_id = matmul_ids[0];
            let first_pos = op_pos[&first_id];
            let first_op = module.ops[first_id].clone();
            let Some(first_res_tt) = module
                .values
                .get(first_op.results[0])
                .and_then(|v| v.ty.as_tensor())
                .cloned()
            else {
                continue;
            };
            let Some(first_rhs_tt) = module
                .values
                .get(first_op.operands[1])
                .and_then(|v| v.ty.as_tensor())
                .cloned()
            else {
                continue;
            };
            let res_rank = first_res_tt.shape.len();
            let rhs_rank = first_rhs_tt.shape.len();
            if res_rank < 2 || rhs_rank < 2 {
                continue;
            }

            let mut valid = true;
            let mut weights_to_concat: smallvec::SmallVec<[ValueId; 4]> = smallvec::SmallVec::new();
            let mut columns_in_each: Vec<i64> = Vec::with_capacity(matmul_ids.len());
            let mut total_columns = 0i64;
            let mut source_locs: Vec<Location> = Vec::with_capacity(matmul_ids.len());
            let mut orig_results: smallvec::SmallVec<[ValueId; 2]> = smallvec::SmallVec::new();

            for &m_id in &matmul_ids {
                let Some(m_op) = module.ops.get(m_id) else {
                    valid = false;
                    break;
                };
                if m_op.name != first_op.name {
                    valid = false;
                    break;
                }
                let rhs_val = m_op.operands[1];
                let res_val = m_op.results[0];

                // Reject if RHS is defined by a transpose (`ForceMatmulTransposeBForGPU` hazard)
                // or defined after `first_pos` in the same block.
                if let Some(ValueDef::OpResult { op: def_op_id, .. }) =
                    module.values.get(rhs_val).map(|v| v.def)
                {
                    if let Some(def_op) = module.ops.get(def_op_id)
                        && (def_op.is_op("stablehlo", "transpose")
                            || def_op.is_op("mo", "transpose"))
                    {
                        valid = false;
                        break;
                    }
                    if let Some(&def_pos) = op_pos.get(&def_op_id)
                        && def_pos > first_pos
                    {
                        valid = false;
                        break;
                    }
                }

                let Some(res_tt) = module.values.get(res_val).and_then(|v| v.ty.as_tensor()) else {
                    valid = false;
                    break;
                };
                let Some(rhs_tt) = module.values.get(rhs_val).and_then(|v| v.ty.as_tensor()) else {
                    valid = false;
                    break;
                };
                if res_tt.dtype != first_res_tt.dtype
                    || rhs_tt.dtype != first_rhs_tt.dtype
                    || res_tt.shape.len() != res_rank
                    || rhs_tt.shape.len() != rhs_rank
                    || res_tt.shape[..res_rank - 1] != first_res_tt.shape[..res_rank - 1]
                    || rhs_tt.shape[..rhs_rank - 1] != first_rhs_tt.shape[..rhs_rank - 1]
                {
                    valid = false;
                    break;
                }
                let (Dim::Static(res_cols), Dim::Static(rhs_cols)) =
                    (&res_tt.shape[res_rank - 1], &rhs_tt.shape[rhs_rank - 1])
                else {
                    valid = false;
                    break;
                };
                if *res_cols <= 0 || res_cols != rhs_cols {
                    valid = false;
                    break;
                }

                weights_to_concat.push(rhs_val);
                columns_in_each.push(*res_cols);
                total_columns += *res_cols;
                source_locs.push(m_op.location.clone());
                orig_results.push(res_val);
            }

            if !valid {
                continue;
            }

            let fused_loc = fuse_locations(&source_locs);
            let mut concat_shape = first_rhs_tt.shape.clone();
            concat_shape[rhs_rank - 1] = Dim::Static(total_columns);
            let concat_ty = Type::Tensor(TensorType {
                shape: concat_shape,
                dtype: first_rhs_tt.dtype.clone(),
                encoding: first_rhs_tt.encoding.clone(),
            });

            let mut concat_attrs = IndexMap::new();
            concat_attrs.insert(
                "dimension".to_string(),
                Attribute::Int((rhs_rank - 1) as i64),
            );
            let concat_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: crate::ir::OpName::new("stablehlo", "concatenate"),
                operands: weights_to_concat,
                results: smallvec::SmallVec::new(),
                attributes: concat_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: fused_loc.clone(),
            });
            let concat_val_id = module.values.insert_with_key(|id| crate::ir::Value {
                id,
                ssa_name: None,
                ty: concat_ty,
                def: ValueDef::OpResult {
                    op: concat_op_id,
                    index: 0,
                },
            });
            module.ops[concat_op_id].results.push(concat_val_id);

            let mut merged_res_shape = first_res_tt.shape.clone();
            merged_res_shape[res_rank - 1] = Dim::Static(total_columns);
            let merged_res_ty = Type::Tensor(TensorType {
                shape: merged_res_shape,
                dtype: first_res_tt.dtype.clone(),
                encoding: first_res_tt.encoding.clone(),
            });

            let merged_matmul_id = module.ops.insert_with_key(|id| {
                let mut op = Operation {
                    id,
                    name: first_op.name.clone(),
                    operands: smallvec::smallvec![lhs_val, concat_val_id],
                    results: smallvec::SmallVec::new(),
                    attributes: first_op.attributes.clone(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: fused_loc.clone(),
                };
                stamp_fused_provenance(&mut op, &source_locs);
                op
            });
            let merged_matmul_val = module.values.insert_with_key(|id| crate::ir::Value {
                id,
                ssa_name: None,
                ty: merged_res_ty,
                def: ValueDef::OpResult {
                    op: merged_matmul_id,
                    index: 0,
                },
            });
            module.ops[merged_matmul_id].results.push(merged_matmul_val);

            let mut split_attrs = IndexMap::new();
            split_attrs.insert("axis".to_string(), Attribute::Int((res_rank - 1) as i64));
            split_attrs.insert(
                "split_sizes".to_string(),
                Attribute::DenseIntArray(columns_in_each),
            );
            let split_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: crate::ir::OpName::new("mo", "split"),
                operands: smallvec::smallvec![merged_matmul_val],
                results: orig_results.clone(),
                attributes: split_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: fused_loc,
            });
            for (idx, &res_v) in orig_results.iter().enumerate() {
                if let Some(v_mut) = module.values.get_mut(res_v) {
                    v_mut.def = ValueDef::OpResult {
                        op: split_op_id,
                        index: idx as u32,
                    };
                }
            }

            insertions.push((first_id, vec![concat_op_id, merged_matmul_id, split_op_id]));
            for &m_id in &matmul_ids {
                to_remove.insert(m_id);
            }
            merged_groups += 1;
        }

        if !insertions.is_empty() {
            let insert_map: std::collections::HashMap<OpId, Vec<OpId>> =
                insertions.into_iter().collect();
            let mut new_block_ops = Vec::with_capacity(block_ops.len() + 4);
            for op_id in block_ops {
                if let Some(new_ops) = insert_map.get(&op_id) {
                    new_block_ops.extend_from_slice(new_ops);
                }
                if to_remove.contains(&op_id) {
                    module.ops.remove(op_id);
                } else {
                    new_block_ops.push(op_id);
                }
            }
            module.blocks[block_id].ops = new_block_ops;
        }
    }

    merged_groups
}

/// Forward and backward dimension correspondence across a reshape operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReshapeDimMapping {
    /// For each input axis `i`, the ordered list of output axes that receive its volume.
    pub forward: Vec<Vec<usize>>,
    /// For each output axis `j`, the ordered list of input axes that contribute to its volume.
    pub backward: Vec<Vec<usize>>,
}

/// Computes the bidirectional dimension correspondence between `before` and `after` reshape
/// shapes using two-pointer cumulative volume matching.
///
/// Positive entries represent statically known dimensions (`> 0`), while negative entries
/// represent symbolic/dynamic dimension identifiers (`< 0`). Zero-sized dimensions, mismatched
/// static volumes, or mixed static/dynamic dimensions return `None`.
pub fn analyze_static_reshape_dim_mapping(
    before: &[i64],
    after: &[i64],
) -> Option<ReshapeDimMapping> {
    if before.is_empty() || after.is_empty() {
        return None;
    }
    if before.contains(&0) || after.contains(&0) {
        return None;
    }

    let calc_static_volume = |dims: &[i64]| -> Option<i64> {
        let mut total = 1i64;
        for &d in dims {
            if d > 0 {
                total = total.checked_mul(d)?;
            }
        }
        Some(total)
    };

    let before_total = calc_static_volume(before)?;
    let after_total = calc_static_volume(after)?;
    if before_total != after_total {
        return None;
    }

    // Verify dynamic dimension sequence compatibility.
    let before_dyn: Vec<i64> = before.iter().copied().filter(|&d| d < 0).collect();
    let after_dyn: Vec<i64> = after.iter().copied().filter(|&d| d < 0).collect();
    if before_dyn != after_dyn {
        return None;
    }

    let mut forward = vec![Vec::new(); before.len()];
    let mut backward = vec![Vec::new(); after.len()];

    let starts_with_static = before[0] > 0;
    let step: isize = if starts_with_static { 1 } else { -1 };
    let mut b_idx: isize = if starts_with_static {
        0
    } else {
        before.len() as isize - 1
    };
    let mut a_idx: isize = if starts_with_static {
        0
    } else {
        after.len() as isize - 1
    };

    let mut b_vol = 1i64;
    let mut a_vol = 1i64;
    let mut b_enabled = true;
    let mut a_enabled = true;

    while b_vol != before_total || a_vol != after_total {
        if b_idx < 0 || b_idx >= before.len() as isize || a_idx < 0 || a_idx >= after.len() as isize
        {
            return None;
        }
        let b_dim = before[b_idx as usize];
        let a_dim = after[a_idx as usize];
        let b_is_static = b_dim > 0;
        let a_is_static = a_dim > 0;

        if b_is_static != a_is_static {
            return None;
        }

        if b_is_static {
            if b_enabled {
                b_vol = b_vol.checked_mul(b_dim)?;
            }
            if a_enabled {
                a_vol = a_vol.checked_mul(a_dim)?;
            }

            b_enabled = b_vol <= a_vol;
            a_enabled = a_vol <= b_vol;

            if b_dim != 1 && a_dim != 1 {
                forward[b_idx as usize].push(a_idx as usize);
                backward[a_idx as usize].push(b_idx as usize);
            }
        } else if b_vol != a_vol || b_dim != a_dim || !b_enabled || !a_enabled {
            return None;
        }

        if b_enabled {
            b_idx += step;
        }
        if a_enabled {
            a_idx += step;
        }
    }

    if !starts_with_static {
        for map in &mut forward {
            map.reverse();
        }
        for map in &mut backward {
            map.reverse();
        }
    }

    Some(ReshapeDimMapping { forward, backward })
}

/// Maps a split or slice axis `axis` through a reshape from `before` to `after`.
///
/// Trailing unit (`1`) dimensions starting at `axis` are skipped until a non-unit static
/// dimension is found. Returns `None` if no non-unit static dimension is found or if that
/// dimension is merged with a left-neighbor dimension in `after`.
pub fn map_axis_through_static_reshape(
    before: &[i64],
    after: &[i64],
    axis: usize,
) -> Option<usize> {
    let mapping = analyze_static_reshape_dim_mapping(before, after)?;
    let mut mapped_axis = axis;
    while mapped_axis < before.len() && before[mapped_axis] == 1 {
        mapped_axis += 1;
    }
    if mapped_axis >= before.len() || before[mapped_axis] <= 0 {
        return None;
    }
    let &new_axis = mapping.forward.get(mapped_axis)?.first()?;
    let backward_sources = mapping.backward.get(new_axis)?;
    let could_mix = backward_sources.len() > 1;
    let axis_is_first = backward_sources.first().copied() == Some(mapped_axis);
    if could_mix && !axis_is_first {
        return None;
    }
    Some(new_axis)
}

pub(super) fn extract_single_int_attr(attr: &Attribute) -> Option<i64> {
    match attr {
        Attribute::Int(v) => Some(*v),
        Attribute::DenseIntArray(v) if v.len() == 1 => Some(v[0]),
        Attribute::Array(v) if v.len() == 1 => extract_single_int_attr(&v[0]),
        Attribute::Opaque(s) | Attribute::String(s) => {
            s.split(':').next()?.trim().parse::<i64>().ok()
        }
        _ => None,
    }
}

pub(super) fn extract_gather_single_index(module: &Module, op: &Operation) -> Option<i64> {
    if let Some(attr) = op
        .attributes
        .get("index")
        .or_else(|| op.attributes.get("indices"))
        .or_else(|| op.attributes.get("start_indices"))
        && let Some(idx) = extract_single_int_attr(attr)
    {
        return Some(idx);
    }
    if let Some(&idx_val) = op.operands.get(1)
        && let Some(v_info) = module.values.get(idx_val)
        && let ValueDef::OpResult { op: def_id, .. } = v_info.def
        && let Some(def_op) = module.ops.get(def_id)
        && matches!(
            def_op.name.to_string().as_str(),
            "stablehlo.constant" | "arith.constant" | "mo.constant"
        )
        && let Some(val_attr) = def_op.attributes.get("value")
    {
        return extract_single_int_attr(val_attr);
    }
    None
}

/// Rewrites exhaustive single-index axis-0 `gather` operations (`mo.gather`, `stablehlo.gather`,
/// `onnx.Gather`) in a basic block that cover every index `0..N` of a static outer dimension `N`
/// into a single `mo.split` inserted before the earliest gather in the group.
///
/// Returns the number of `mo.split` operations created.
pub fn coalesce_exhaustive_gathers_to_split(module: &mut Module) -> usize {
    let mut coalesced = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let block_ops = module.blocks[block_id].ops.clone();
        // Map each input `ValueId` to `(axis0_len, Vec<(block_pos, op_id, gather_idx)>)`.
        type GatherEntry = (usize, OpId, i64);
        let mut by_input: IndexMap<ValueId, (usize, Vec<GatherEntry>)> = IndexMap::new();

        for (pos, &op_id) in block_ops.iter().enumerate() {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !matches!(
                op.name.to_string().as_str(),
                "mo.gather" | "stablehlo.gather" | "onnx.Gather"
            ) {
                continue;
            }
            if op.operands.is_empty() || op.results.len() != 1 {
                continue;
            }
            let axis = op
                .attributes
                .get("axis")
                .or_else(|| op.attributes.get("dimension"))
                .and_then(extract_single_int_attr)
                .unwrap_or(0);
            if axis != 0 {
                continue;
            }
            let Some(idx) = extract_gather_single_index(module, op) else {
                continue;
            };
            let input_val = op.operands[0];
            let Some(in_info) = module.values.get(input_val) else {
                continue;
            };
            let Type::Tensor(ref in_tt) = in_info.ty else {
                continue;
            };
            let Some(Dim::Static(axis0_dim)) = in_tt.shape.first() else {
                continue;
            };
            if *axis0_dim <= 0 || idx < 0 || idx >= *axis0_dim {
                continue;
            }
            let entry = by_input
                .entry(input_val)
                .or_insert_with(|| (*axis0_dim as usize, Vec::new()));
            entry.1.push((pos, op_id, idx));
        }

        let mut insertions: std::collections::HashMap<OpId, OpId> =
            std::collections::HashMap::new();
        let mut to_remove: std::collections::HashSet<OpId> = std::collections::HashSet::new();

        for (input_val, (axis0_len, entries)) in by_input {
            if entries.len() < axis0_len {
                continue;
            }
            // Keep first-seen gather for each index `0..axis0_len` and find the dominant gather.
            let mut idx_to_gather: Vec<Option<(usize, OpId)>> = vec![None; axis0_len];
            for &(pos, op_id, idx) in &entries {
                let slot = &mut idx_to_gather[idx as usize];
                if slot.is_none() {
                    *slot = Some((pos, op_id));
                }
            }
            if idx_to_gather.iter().any(Option::is_none) {
                continue;
            }
            let matched: Vec<(usize, OpId)> =
                idx_to_gather.into_iter().map(Option::unwrap).collect();

            // Verify consistent device placement across all gathers in the group.
            let first_dev = module.ops[matched[0].1]
                .attributes
                .get("castforge.device")
                .cloned();
            let same_device = matched.iter().all(|&(_, g_id)| {
                module.ops[g_id].attributes.get("castforge.device") == first_dev.as_ref()
            });
            if !same_device {
                continue;
            }

            let dominant_op_id = matched
                .iter()
                .min_by_key(|&&(pos, _)| pos)
                .map(|&(_, id)| id)
                .unwrap();

            let mut split_results = Vec::with_capacity(axis0_len);
            let mut locs = Vec::with_capacity(axis0_len);
            for &(_, g_id) in &matched {
                split_results.push(module.ops[g_id].results[0]);
                locs.push(module.ops[g_id].location.clone());
            }

            let mut split_attrs = IndexMap::new();
            split_attrs.insert("axis".to_string(), Attribute::Int(0));
            split_attrs.insert(
                "split_sizes".to_string(),
                Attribute::DenseIntArray(vec![1; axis0_len]),
            );
            if let Some(dev) = first_dev {
                split_attrs.insert("castforge.device".to_string(), dev);
            }

            let split_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: crate::ir::OpName::new("mo", "split"),
                operands: smallvec::smallvec![input_val],
                results: split_results.iter().copied().collect(),
                attributes: split_attrs,
                regions: Default::default(),
                custom_syntax: false,
                location: fuse_locations(&locs),
            });
            if let Some(split_op) = module.ops.get_mut(split_op_id) {
                stamp_fused_provenance(split_op, &locs);
            }

            for (idx, &res_val) in split_results.iter().enumerate() {
                if let Some(v_info) = module.values.get_mut(res_val) {
                    v_info.def = ValueDef::OpResult {
                        op: split_op_id,
                        index: idx as u32,
                    };
                }
            }

            insertions.insert(dominant_op_id, split_op_id);
            for &(_, g_id) in &matched {
                to_remove.insert(g_id);
            }
            coalesced += 1;
        }

        if !insertions.is_empty() {
            let mut new_block_ops = Vec::with_capacity(block_ops.len());
            for op_id in block_ops {
                if let Some(&split_id) = insertions.get(&op_id) {
                    new_block_ops.push(split_id);
                }
                if to_remove.contains(&op_id) {
                    module.ops.remove(op_id);
                } else {
                    new_block_ops.push(op_id);
                }
            }
            module.blocks[block_id].ops = new_block_ops;
        }
    }

    coalesced
}
