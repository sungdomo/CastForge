//! Coordinator key directory, FDP slot management, and algebraic sink/swim passes.

use super::{
    CacheEventTier, analyze_static_reshape_dim_mapping, extract_int_list_attr,
    extract_single_int_attr, map_axis_through_static_reshape,
};
use crate::ir::{Attribute, Dim, Module, OpId, OpName, TensorType, Type, ValueDef, ValueId};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

/// Cache event operation kind applied to [`CoordinatorKeyDirectory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEventOpKind {
    /// Inserts or updates a placement for the key.
    Store,
    /// Removes a placement for the key.
    Delete,
    /// Records a cache hit/access on the key.
    Access,
}

/// Live placement of a cached key in the fleet directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRecord {
    /// Reporting worker instance identifier.
    pub instance_id: String,
    /// Incarnation epoch of the reporting worker.
    pub incarnation: u64,
    /// Storage tier (`L1` volatile worker memory vs `L2` persistent storage).
    pub tier: CacheEventTier,
    /// Backend name within the tier (e.g., `"cpu"`, `"gds"`, `"redis"`).
    pub backend: String,
    /// Reported payload size in bytes.
    pub size_bytes: u64,
    /// Whether the backend is a fleet-shared pool (deduplicated across instances).
    pub shared: bool,
}

/// Single key mutation entry inside a coordinator event batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEventEntrySpec {
    /// Target object key (`chunk_hash`).
    pub key: u64,
    /// Payload size in bytes (used on [`CacheEventOpKind::Store`]).
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct KeyDirectoryRecord {
    placements: Vec<PlacementRecord>,
    last_access_ts: u64,
    access_count: u64,
}

/// Fleet-wide key directory with incremental tier byte accounting and `O(|l1_keys(instance)|)`
/// reverse-indexed L1 instance fencing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoordinatorKeyDirectory {
    directory: IndexMap<u64, KeyDirectoryRecord>,
    l1_keys_by_instance: IndexMap<String, Vec<u64>>,
    l1_count: usize,
    l1_size_bytes: u64,
    l2_count: usize,
    l2_size_bytes: u64,
}

impl CoordinatorKeyDirectory {
    /// Creates an empty fleet key directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keys with at least one active placement.
    pub fn num_keys(&self) -> usize {
        self.directory.len()
    }

    /// Returns `(l1_count, l1_size_bytes, l2_count, l2_size_bytes)`.
    pub fn tier_stats(&self) -> (usize, u64, usize, u64) {
        (
            self.l1_count,
            self.l1_size_bytes,
            self.l2_count,
            self.l2_size_bytes,
        )
    }

    fn adjust_stats(&mut self, tier: CacheEventTier, count_delta: isize, bytes_delta: i64) {
        match tier {
            CacheEventTier::L1 => {
                self.l1_count = (self.l1_count as isize + count_delta).max(0) as usize;
                self.l1_size_bytes = (self.l1_size_bytes as i64 + bytes_delta).max(0) as u64;
            }
            CacheEventTier::L2 => {
                self.l2_count = (self.l2_count as isize + count_delta).max(0) as usize;
                self.l2_size_bytes = (self.l2_size_bytes as i64 + bytes_delta).max(0) as u64;
            }
        }
    }

    fn find_placement_index(
        placements: &[PlacementRecord],
        instance_id: &str,
        tier: CacheEventTier,
        backend: &str,
        shared: bool,
    ) -> Option<usize> {
        placements.iter().position(|p| {
            p.shared == shared
                && (shared || p.instance_id == instance_id)
                && p.tier == tier
                && p.backend == backend
        })
    }

    /// Applies an admitted event batch idempotently (`Store` upserts, `Delete` removes, `Access`
    /// increments access count and updates `last_access_ts`).
    #[allow(clippy::too_many_arguments)]
    pub fn consume_batch(
        &mut self,
        instance_id: &str,
        incarnation: u64,
        tier: CacheEventTier,
        backend: &str,
        shared: bool,
        op_kind: CacheEventOpKind,
        ts: u64,
        entries: &[CacheEventEntrySpec],
    ) {
        for entry in entries {
            match op_kind {
                CacheEventOpKind::Store => {
                    let record = self.directory.entry(entry.key).or_default();
                    let new_p = PlacementRecord {
                        instance_id: instance_id.to_string(),
                        incarnation,
                        tier,
                        backend: backend.to_string(),
                        size_bytes: entry.size_bytes,
                        shared,
                    };
                    if let Some(idx) = Self::find_placement_index(
                        &record.placements,
                        instance_id,
                        tier,
                        backend,
                        shared,
                    ) {
                        let old_bytes = record.placements[idx].size_bytes;
                        record.placements[idx] = new_p;
                        let delta = (entry.size_bytes as i64) - (old_bytes as i64);
                        self.adjust_stats(tier, 0, delta);
                    } else {
                        record.placements.push(new_p);
                        self.adjust_stats(tier, 1, entry.size_bytes as i64);
                    }
                    if let Some(rec) = self.directory.get_mut(&entry.key) {
                        rec.last_access_ts = rec.last_access_ts.max(ts);
                    }
                    if tier == CacheEventTier::L1 {
                        let l1_list = self
                            .l1_keys_by_instance
                            .entry(instance_id.to_string())
                            .or_default();
                        if !l1_list.contains(&entry.key) {
                            l1_list.push(entry.key);
                        }
                    }
                }
                CacheEventOpKind::Delete => {
                    let Some(record) = self.directory.get_mut(&entry.key) else {
                        continue;
                    };
                    if let Some(idx) = Self::find_placement_index(
                        &record.placements,
                        instance_id,
                        tier,
                        backend,
                        shared,
                    ) {
                        let removed = record.placements.remove(idx);
                        let is_empty = record.placements.is_empty();
                        let still_has_inst_l1 = record
                            .placements
                            .iter()
                            .any(|p| p.tier == CacheEventTier::L1 && p.instance_id == instance_id);
                        if is_empty {
                            self.directory.shift_remove(&entry.key);
                        }
                        self.adjust_stats(tier, -1, -(removed.size_bytes as i64));
                        if tier == CacheEventTier::L1
                            && !still_has_inst_l1
                            && let Some(l1_list) = self.l1_keys_by_instance.get_mut(instance_id)
                        {
                            l1_list.retain(|&k| k != entry.key);
                        }
                    }
                }
                CacheEventOpKind::Access => {
                    if let Some(record) = self.directory.get_mut(&entry.key) {
                        record.last_access_ts = record.last_access_ts.max(ts);
                        record.access_count += 1;
                    }
                }
            }
        }
    }

    /// Drops every `L1` placement reported by `instance_id` using the reverse index while
    /// preserving all `L2` placements. Returns the number of `L1` placements removed.
    pub fn fence_instance(&mut self, instance_id: &str) -> usize {
        let Some(l1_keys) = self.l1_keys_by_instance.shift_remove(instance_id) else {
            return 0;
        };
        let mut removed_count = 0usize;
        let mut removed_bytes = 0i64;
        for key in l1_keys {
            let Some(record) = self.directory.get_mut(&key) else {
                continue;
            };
            let mut i = 0;
            while i < record.placements.len() {
                if record.placements[i].tier == CacheEventTier::L1
                    && record.placements[i].instance_id == instance_id
                {
                    let p = record.placements.remove(i);
                    removed_count += 1;
                    removed_bytes += p.size_bytes as i64;
                } else {
                    i += 1;
                }
            }
            if record.placements.is_empty() {
                self.directory.shift_remove(&key);
            }
        }
        self.adjust_stats(
            CacheEventTier::L1,
            -(removed_count as isize),
            -removed_bytes,
        );
        removed_count
    }

    /// Looks up placements for `keys`, sorting each key's placement list by
    /// `(instance_id, tier, backend)`.
    pub fn lookup(&self, keys: &[u64]) -> Vec<Vec<PlacementRecord>> {
        keys.iter()
            .map(|k| {
                let Some(record) = self.directory.get(k) else {
                    return Vec::new();
                };
                let mut list = record.placements.clone();
                list.sort_by(|a, b| {
                    let t_a = matches!(a.tier, CacheEventTier::L2);
                    let t_b = matches!(b.tier, CacheEventTier::L2);
                    a.instance_id
                        .cmp(&b.instance_id)
                        .then_with(|| t_a.cmp(&t_b))
                        .then_with(|| a.backend.cmp(&b.backend))
                });
                list
            })
            .collect()
    }

    /// Returns the number of [`CacheEventOpKind::Access`] events applied to each key in `keys`.
    pub fn get_access_counts(&self, keys: &[u64]) -> Vec<u64> {
        keys.iter()
            .map(|k| self.directory.get(k).map_or(0, |r| r.access_count))
            .collect()
    }
}

/// 8-byte ASCII magic header (`LMCBLK01`) for raw-block per-slot on-disk headers.
pub const RAW_BLOCK_SLOT_MAGIC: &[u8; 8] = b"LMCBLK01";

/// 8-byte ASCII magic header (`LMCIDX01`) for raw-block metadata checkpoint containers.
pub const RAW_BLOCK_META_MAGIC: &[u8; 8] = b"LMCIDX01";

/// Computes the standard IEEE 802.3 CRC-32 checksum over `data` (polynomial `0xEDB88320`).
pub fn compute_ieee_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Encodes a 24-byte raw-block slot header (`LMCBLK01`, `slot_identity`, `payload_len` in
/// little-endian byte order) padded with zeros to `header_bytes`.
///
/// Returns `None` if `header_bytes < 24`.
pub fn encode_raw_block_slot_header(
    slot_identity: u64,
    payload_len: u64,
    header_bytes: usize,
) -> Option<Vec<u8>> {
    if header_bytes < 24 {
        return None;
    }
    let mut out = vec![0u8; header_bytes];
    out[0..8].copy_from_slice(RAW_BLOCK_SLOT_MAGIC);
    out[8..16].copy_from_slice(&slot_identity.to_le_bytes());
    out[16..24].copy_from_slice(&payload_len.to_le_bytes());
    Some(out)
}

/// Decodes a raw-block slot header and returns `(slot_identity, payload_len)` when the
/// buffer begins with [`RAW_BLOCK_SLOT_MAGIC`].
pub fn decode_raw_block_slot_header(buf: &[u8]) -> Option<(u64, u64)> {
    if buf.len() < 24 || &buf[0..8] != RAW_BLOCK_SLOT_MAGIC {
        return None;
    }
    let slot_identity = u64::from_le_bytes(buf[8..16].try_into().ok()?);
    let payload_len = u64::from_le_bytes(buf[16..24].try_into().ok()?);
    Some((slot_identity, payload_len))
}

/// Parsed metadata checkpoint container header for double-buffered raw-block index persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawBlockMetaCheckpointHeader {
    /// Monotonically increasing checkpoint sequence number.
    pub seq: u64,
    /// Unpadded byte length of the serialized metadata payload.
    pub payload_len: u64,
    /// IEEE CRC-32 checksum over the unpadded payload bytes.
    pub crc32: u32,
    /// Byte offset of the selected container on the raw block device.
    pub container_offset: u64,
}

/// Encodes a block-aligned metadata checkpoint container (`LMCIDX01`, `version`, `flags = 0`,
/// `seq`, `payload_len`, `crc32`, `reserved = 0` at offset `0..40`, followed by `payload` at
/// `block_align..block_align + payload.len()`, rounded up to a multiple of `block_align`).
pub fn encode_raw_block_meta_checkpoint(
    version: u32,
    seq: u64,
    payload: &[u8],
    block_align: usize,
) -> Option<Vec<u8>> {
    if block_align < 40 || !block_align.is_power_of_two() {
        return None;
    }
    let payload_padded = if payload.is_empty() {
        block_align
    } else {
        payload.len().div_ceil(block_align) * block_align
    };
    let total_len = block_align.checked_add(payload_padded)?;
    let mut buf = vec![0u8; total_len];
    let crc32 = compute_ieee_crc32(payload);
    buf[0..8].copy_from_slice(RAW_BLOCK_META_MAGIC);
    buf[8..12].copy_from_slice(&version.to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes());
    buf[16..24].copy_from_slice(&seq.to_le_bytes());
    buf[24..32].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    buf[32..36].copy_from_slice(&crc32.to_le_bytes());
    buf[36..40].copy_from_slice(&0u32.to_le_bytes());
    buf[block_align..block_align + payload.len()].copy_from_slice(payload);
    Some(buf)
}

/// Scans double-buffered metadata checkpoint containers `(container_offset, raw_bytes)` and
/// returns the valid container with the highest `seq` number whose magic, `expected_version`,
/// bounds, and IEEE CRC-32 match.
pub fn select_latest_valid_meta_checkpoint(
    containers: &[(u64, Vec<u8>)],
    block_align: usize,
    expected_version: u32,
) -> Option<(RawBlockMetaCheckpointHeader, Vec<u8>)> {
    if block_align < 40 {
        return None;
    }
    let mut best: Option<(RawBlockMetaCheckpointHeader, Vec<u8>)> = None;
    for (container_offset, raw) in containers {
        if raw.len() < block_align || &raw[0..8] != RAW_BLOCK_META_MAGIC {
            continue;
        }
        let version = u32::from_le_bytes(raw[8..12].try_into().ok()?);
        let flags = u32::from_le_bytes(raw[12..16].try_into().ok()?);
        let seq = u64::from_le_bytes(raw[16..24].try_into().ok()?);
        let payload_len = u64::from_le_bytes(raw[24..32].try_into().ok()?);
        let crc32 = u32::from_le_bytes(raw[32..36].try_into().ok()?);
        if version != expected_version || flags != 0 || payload_len == 0 {
            continue;
        }
        let Ok(plen) = usize::try_from(payload_len) else {
            continue;
        };
        let Some(end) = block_align.checked_add(plen) else {
            continue;
        };
        if end > raw.len() {
            continue;
        }
        let payload = &raw[block_align..end];
        if compute_ieee_crc32(payload) != crc32 {
            continue;
        }
        let header = RawBlockMetaCheckpointHeader {
            seq,
            payload_len,
            crc32,
            container_offset: *container_offset,
        };
        if best.as_ref().is_none_or(|(prev, _)| seq >= prev.seq) {
            best = Some((header, payload.to_vec()));
        }
    }
    best
}

/// NVMe Flexible Data Placement (FDP) affinity raw-block slot allocator with crash-safe
/// free-list reconstruction from committed index checkpoints.
#[derive(Debug, Clone)]
pub struct FdpRawBlockSlotManager {
    max_slots: usize,
    next_slot: usize,
    free_slots: Vec<usize>,
    free_slots_by_placement_id: IndexMap<u16, Vec<usize>>,
    slot_placement_ids: IndexMap<usize, u16>,
    fdp_affinity_enabled: bool,
    fdp_affinity_hits: usize,
    fdp_affinity_fallbacks: usize,
}

impl FdpRawBlockSlotManager {
    /// Creates a new [`FdpRawBlockSlotManager`] with capacity `max_slots`.
    pub fn new(max_slots: usize, fdp_affinity_enabled: bool) -> Self {
        Self {
            max_slots,
            next_slot: 0,
            free_slots: Vec::new(),
            free_slots_by_placement_id: IndexMap::new(),
            slot_placement_ids: IndexMap::new(),
            fdp_affinity_enabled,
            fdp_affinity_hits: 0,
            fdp_affinity_fallbacks: 0,
        }
    }

    /// Returns `(next_slot, free_slots_count, fdp_affinity_hits, fdp_affinity_fallbacks)`.
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        (
            self.next_slot,
            self.free_slots.len(),
            self.fdp_affinity_hits,
            self.fdp_affinity_fallbacks,
        )
    }

    /// Returns the last NVMe FDP placement identifier recorded for `slot`, if any.
    pub fn slot_placement_id(&self, slot: usize) -> Option<u16> {
        self.slot_placement_ids.get(&slot).copied()
    }

    /// Allocates a slot, preferring a recycled slot that previously carried the same NVMe FDP
    /// `placement_id` (reclaim unit handle) when `fdp_affinity_enabled` is true.
    pub fn allocate_slot(&mut self, placement_id: Option<u16>) -> Option<usize> {
        if self.fdp_affinity_enabled
            && let Some(pid) = placement_id
        {
            if let Some(bucket) = self.free_slots_by_placement_id.get_mut(&pid)
                && let Some(slot) = bucket.pop()
            {
                if bucket.is_empty() {
                    self.free_slots_by_placement_id.shift_remove(&pid);
                }
                if let Some(pos) = self.free_slots.iter().rposition(|&s| s == slot) {
                    self.free_slots.remove(pos);
                }
                self.slot_placement_ids.insert(slot, pid);
                self.fdp_affinity_hits += 1;
                return Some(slot);
            }
            if !self.free_slots.is_empty() {
                self.fdp_affinity_fallbacks += 1;
            }
        }
        let slot = if let Some(recycled) = self.free_slots.pop() {
            if self.fdp_affinity_enabled
                && let Some(&old_pid) = self.slot_placement_ids.get(&recycled)
                && let Some(bucket) = self.free_slots_by_placement_id.get_mut(&old_pid)
            {
                if let Some(pos) = bucket.iter().rposition(|&s| s == recycled) {
                    bucket.remove(pos);
                }
                if bucket.is_empty() {
                    self.free_slots_by_placement_id.shift_remove(&old_pid);
                }
            }
            recycled
        } else {
            if self.next_slot >= self.max_slots {
                return None;
            }
            let fresh = self.next_slot;
            self.next_slot += 1;
            fresh
        };
        if self.fdp_affinity_enabled {
            if let Some(pid) = placement_id {
                self.slot_placement_ids.insert(slot, pid);
            } else {
                self.slot_placement_ids.shift_remove(&slot);
            }
        }
        Some(slot)
    }

    /// Returns `slot` to the free list and indexes it under its last FDP placement identifier.
    pub fn free_slot(&mut self, slot: usize) {
        if slot >= self.next_slot || self.free_slots.contains(&slot) {
            return;
        }
        self.free_slots.push(slot);
        if self.fdp_affinity_enabled
            && let Some(&pid) = self.slot_placement_ids.get(&slot)
        {
            self.free_slots_by_placement_id
                .entry(pid)
                .or_default()
                .push(slot);
        }
    }

    /// Reconstructs the allocator state from a checkpointed `next_slot` high-water mark and
    /// `committed_slots` set, recovering any uncommitted in-flight slots from `[0, next_slot)`
    /// back into `free_slots`.
    pub fn rebuild_free_slots_from_committed(
        &mut self,
        next_slot: usize,
        committed_slots: &[usize],
    ) -> bool {
        if next_slot > self.max_slots {
            return false;
        }
        let mut occupied = IndexSet::new();
        for &slot in committed_slots {
            if slot >= next_slot || !occupied.insert(slot) {
                return false;
            }
        }
        self.next_slot = next_slot;
        self.free_slots.clear();
        self.free_slots_by_placement_id.clear();
        self.slot_placement_ids.clear();
        for slot in 0..next_slot {
            if !occupied.contains(&slot) {
                self.free_slots.push(slot);
            }
        }
        true
    }
}

pub(super) fn static_i64_dims(tt: &TensorType) -> Option<Vec<i64>> {
    let mut out = Vec::with_capacity(tt.shape.len());
    for d in &tt.shape {
        match d {
            Dim::Static(v) if *v > 0 => out.push(*v),
            _ => return None,
        }
    }
    Some(out)
}

/// Commutes a sole-use `stablehlo.slice` followed by `stablehlo.reshape` into `stablehlo.reshape`
/// followed by `stablehlo.slice` when the sliced dimension maps cleanly through the reshape
/// (either 1-to-1, factored by an integer divisor with unit step, or merged with trailing
/// dimensions with unit step).
///
/// Returns the number of slice-reshape pairs commuted.
pub fn swim_reshape_through_slice(module: &mut Module) -> usize {
    let mut commuted = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    *use_counts.entry(v).or_insert(0) += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        for op_id in op_ids {
            let Some(reshape_op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !matches!(
                reshape_op.name.to_string().as_str(),
                "stablehlo.reshape" | "mo.reshape"
            ) || reshape_op.operands.len() != 1
                || reshape_op.results.len() != 1
            {
                continue;
            }
            let slice_out_val = reshape_op.operands[0];
            if use_counts.get(&slice_out_val).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(slice_out_info) = module.values.get(slice_out_val).cloned() else {
                continue;
            };
            let ValueDef::OpResult { op: slice_id, .. } = slice_out_info.def else {
                continue;
            };
            if !module.blocks[block_id].ops.contains(&slice_id) {
                continue;
            }
            let Some(slice_op) = module.ops.get(slice_id).cloned() else {
                continue;
            };
            if !slice_op.is_op("stablehlo", "slice")
                || slice_op.operands.len() != 1
                || slice_op.results.len() != 1
            {
                continue;
            }
            if slice_op.attributes.get("castforge.device")
                != reshape_op.attributes.get("castforge.device")
            {
                continue;
            }

            let slice_in_val = slice_op.operands[0];
            let reshape_out_val = reshape_op.results[0];
            let (Some(in_info), Some(out_info)) = (
                module.values.get(slice_in_val),
                module.values.get(reshape_out_val),
            ) else {
                continue;
            };
            let (Type::Tensor(in_tt), Type::Tensor(mid_tt), Type::Tensor(out_tt)) =
                (&in_info.ty, &slice_out_info.ty, &out_info.ty)
            else {
                continue;
            };
            let (Some(slice_in_dims), Some(slice_out_dims), Some(reshape_out_dims)) = (
                static_i64_dims(in_tt),
                static_i64_dims(mid_tt),
                static_i64_dims(out_tt),
            ) else {
                continue;
            };
            let in_rank = slice_in_dims.len();
            if in_rank == 0 || slice_out_dims.len() != in_rank {
                continue;
            }

            let Some(starts) = extract_int_list_attr(&slice_op, "start_indices") else {
                continue;
            };
            let limits = extract_int_list_attr(&slice_op, "limit_indices").unwrap_or_else(|| {
                starts
                    .iter()
                    .zip(slice_out_dims.iter())
                    .map(|(&s, &len)| s + len)
                    .collect()
            });
            let strides = extract_int_list_attr(&slice_op, "strides")
                .unwrap_or_else(|| vec![1; slice_in_dims.len()]);
            if starts.len() != in_rank || limits.len() != in_rank || strides.len() != in_rank {
                continue;
            }

            // Identify the single sliced dimension.
            let mut sliced_axes = Vec::new();
            for d in 0..in_rank {
                if starts[d] != 0
                    || limits[d] != slice_in_dims[d]
                    || strides[d] != 1
                    || slice_out_dims[d] != slice_in_dims[d]
                {
                    sliced_axes.push(d);
                }
            }
            if sliced_axes.len() != 1 {
                continue;
            }
            let slice_axis = sliced_axes[0];
            let start = starts[slice_axis];
            let limit = limits[slice_axis];
            let step = strides[slice_axis];
            if start < 0 || limit <= start || step <= 0 {
                continue;
            }

            let Some(new_axis) =
                map_axis_through_static_reshape(&slice_out_dims, &reshape_out_dims, slice_axis)
            else {
                continue;
            };

            let old_sliced_dim = slice_out_dims[slice_axis];
            let new_sliced_dim = reshape_out_dims[new_axis];
            let slice_in_axis_dim = slice_in_dims[slice_axis];

            let (new_reshape_axis_dim, new_start, new_limit, new_step) =
                if old_sliced_dim == new_sliced_dim {
                    (slice_in_axis_dim, start, limit, step)
                } else if old_sliced_dim > new_sliced_dim {
                    let factor = old_sliced_dim / new_sliced_dim;
                    if step != 1
                        || old_sliced_dim % new_sliced_dim != 0
                        || slice_in_axis_dim % factor != 0
                        || start % factor != 0
                        || limit % factor != 0
                    {
                        continue;
                    }
                    (
                        slice_in_axis_dim / factor,
                        start / factor,
                        limit / factor,
                        1,
                    )
                } else {
                    let factor = new_sliced_dim / old_sliced_dim;
                    if step != 1 || new_sliced_dim % old_sliced_dim != 0 {
                        continue;
                    }
                    (
                        slice_in_axis_dim * factor,
                        start * factor,
                        limit * factor,
                        1,
                    )
                };

            let mut new_reshape_dims = reshape_out_dims.clone();
            new_reshape_dims[new_axis] = new_reshape_axis_dim;
            if analyze_static_reshape_dim_mapping(&slice_in_dims, &new_reshape_dims).is_none() {
                continue;
            }

            let out_rank = reshape_out_dims.len();
            let mut new_starts = vec![0i64; out_rank];
            let mut new_limits = new_reshape_dims.clone();
            let mut new_strides = vec![1i64; out_rank];
            new_starts[new_axis] = new_start;
            new_limits[new_axis] = new_limit;
            new_strides[new_axis] = new_step;

            let new_mid_tt = TensorType {
                shape: new_reshape_dims.into_iter().map(Dim::Static).collect(),
                dtype: out_tt.dtype.clone(),
                encoding: out_tt.encoding.clone(),
            };

            // Mutate `slice_id` into `stablehlo.reshape` and `op_id` into `stablehlo.slice`.
            if let Some(mid_val_mut) = module.values.get_mut(slice_out_val) {
                mid_val_mut.ty = Type::Tensor(new_mid_tt);
            }
            if let Some(first_op) = module.ops.get_mut(slice_id) {
                first_op.name = OpName::new("stablehlo", "reshape");
                first_op.attributes.shift_remove("start_indices");
                first_op.attributes.shift_remove("limit_indices");
                first_op.attributes.shift_remove("strides");
            }
            if let Some(second_op) = module.ops.get_mut(op_id) {
                second_op.name = OpName::new("stablehlo", "slice");
                second_op
                    .attributes
                    .insert("start_indices".into(), Attribute::DenseIntArray(new_starts));
                second_op
                    .attributes
                    .insert("limit_indices".into(), Attribute::DenseIntArray(new_limits));
                second_op
                    .attributes
                    .insert("strides".into(), Attribute::DenseIntArray(new_strides));
            }
            commuted += 1;
        }
    }

    commuted
}

/// Sinks a uniform `mo.split` through identical static `stablehlo.reshape` operations on every
/// split output when the split axis maps cleanly through the reshape.
///
/// Rewrites `split -> [reshape_0, ..., reshape_{K-1}]` into `reshape -> split`, replacing the `K`
/// individual reshapes with a single pre-split `stablehlo.reshape`. Returns the number of splits
/// sunk.
pub fn sink_split_through_reshape(module: &mut Module) -> usize {
    let mut sunk = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut single_consumer: IndexMap<ValueId, (OpId, usize)> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    let entry = single_consumer.entry(v).or_insert((op_id, 0));
                    entry.0 = op_id;
                    entry.1 += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for split_id in op_ids {
            if to_remove.contains(&split_id) {
                continue;
            }
            let Some(split_op) = module.ops.get(split_id).cloned() else {
                continue;
            };
            if split_op.name.to_string() != "mo.split"
                || split_op.operands.len() != 1
                || split_op.results.len() < 2
            {
                continue;
            }
            let num_splits = split_op.results.len();
            let split_in_val = split_op.operands[0];
            let Some(in_info) = module.values.get(split_in_val).cloned() else {
                continue;
            };
            let Type::Tensor(ref split_in_tt) = in_info.ty else {
                continue;
            };
            let Some(split_in_dims) = static_i64_dims(split_in_tt) else {
                continue;
            };

            let mut reshape_ids = Vec::with_capacity(num_splits);
            let mut split_out_dims_opt: Option<Vec<i64>> = None;
            let mut reshape_out_tt_opt: Option<TensorType> = None;
            let mut reshape_out_dims_opt: Option<Vec<i64>> = None;
            let mut reshape_results = Vec::with_capacity(num_splits);
            let mut valid = true;

            for &res_v in &split_op.results {
                let Some(&(consumer_id, count)) = single_consumer.get(&res_v) else {
                    valid = false;
                    break;
                };
                if count != 1 || !module.blocks[block_id].ops.contains(&consumer_id) {
                    valid = false;
                    break;
                }
                let Some(res_info) = module.values.get(res_v) else {
                    valid = false;
                    break;
                };
                let Type::Tensor(ref res_tt) = res_info.ty else {
                    valid = false;
                    break;
                };
                let Some(res_dims) = static_i64_dims(res_tt) else {
                    valid = false;
                    break;
                };
                if let Some(ref prev) = split_out_dims_opt {
                    if prev != &res_dims {
                        valid = false;
                        break;
                    }
                } else {
                    split_out_dims_opt = Some(res_dims);
                }

                let Some(r_op) = module.ops.get(consumer_id) else {
                    valid = false;
                    break;
                };
                if !matches!(
                    r_op.name.to_string().as_str(),
                    "stablehlo.reshape" | "mo.reshape"
                ) || r_op.operands.len() != 1
                    || r_op.results.len() != 1
                    || r_op.attributes.get("castforge.device")
                        != split_op.attributes.get("castforge.device")
                {
                    valid = false;
                    break;
                }
                let r_out_v = r_op.results[0];
                let Some(r_out_info) = module.values.get(r_out_v) else {
                    valid = false;
                    break;
                };
                let Type::Tensor(ref r_out_tt) = r_out_info.ty else {
                    valid = false;
                    break;
                };
                let Some(r_out_dims) = static_i64_dims(r_out_tt) else {
                    valid = false;
                    break;
                };
                if let Some(ref prev_out) = reshape_out_dims_opt {
                    if prev_out != &r_out_dims {
                        valid = false;
                        break;
                    }
                } else {
                    reshape_out_dims_opt = Some(r_out_dims);
                    reshape_out_tt_opt = Some(r_out_tt.clone());
                }
                reshape_ids.push(consumer_id);
                reshape_results.push(r_out_v);
            }
            if !valid {
                continue;
            }
            let (Some(split_out_dims), Some(reshape_out_dims), Some(reshape_out_tt)) =
                (split_out_dims_opt, reshape_out_dims_opt, reshape_out_tt_opt)
            else {
                continue;
            };
            if split_in_dims.len() != split_out_dims.len() {
                continue;
            }

            // Determine `split_axis` from attribute or shape difference.
            let attr_axis = split_op
                .attributes
                .get("axis")
                .or_else(|| split_op.attributes.get("split_dim"))
                .and_then(extract_single_int_attr)
                .and_then(|a| usize::try_from(a).ok());
            let inferred_axis = (0..split_in_dims.len()).find(|&d| {
                split_in_dims[d] == split_out_dims[d] * (num_splits as i64)
                    && (0..split_in_dims.len())
                        .all(|k| k == d || split_in_dims[k] == split_out_dims[k])
            });
            let Some(split_axis) = attr_axis.or(inferred_axis) else {
                continue;
            };
            if split_axis >= split_in_dims.len()
                || split_in_dims[split_axis] != split_out_dims[split_axis] * (num_splits as i64)
            {
                continue;
            }

            let Some(new_axis) =
                map_axis_through_static_reshape(&split_out_dims, &reshape_out_dims, split_axis)
            else {
                continue;
            };
            let mut pre_split_reshape_dims = reshape_out_dims.clone();
            pre_split_reshape_dims[new_axis] *= num_splits as i64;
            if analyze_static_reshape_dim_mapping(&split_in_dims, &pre_split_reshape_dims).is_none()
            {
                continue;
            }

            let pre_split_tt = TensorType {
                shape: pre_split_reshape_dims
                    .iter()
                    .copied()
                    .map(Dim::Static)
                    .collect(),
                dtype: reshape_out_tt.dtype.clone(),
                encoding: reshape_out_tt.encoding.clone(),
            };
            let pre_split_val = split_op.results[0];
            if let Some(v_mut) = module.values.get_mut(pre_split_val) {
                v_mut.ty = Type::Tensor(pre_split_tt);
                v_mut.def = ValueDef::OpResult {
                    op: split_id,
                    index: 0,
                };
            }
            for &dead_v in &split_op.results[1..] {
                module.values.remove(dead_v);
            }

            // Rewrite `split_id` in place into `stablehlo.reshape`.
            if let Some(op_mut) = module.ops.get_mut(split_id) {
                op_mut.name = OpName::new("stablehlo", "reshape");
                op_mut.attributes.shift_remove("axis");
                op_mut.attributes.shift_remove("split_dim");
                op_mut.attributes.shift_remove("split_sizes");
                op_mut.results = SmallVec::from_slice(&[pre_split_val]);
            }

            // Reuse `reshape_ids[0]` as the new `mo.split` producing all `reshape_results`.
            let new_split_id = reshape_ids[0];
            for (idx, &res_v) in reshape_results.iter().enumerate() {
                if let Some(v_mut) = module.values.get_mut(res_v) {
                    v_mut.def = ValueDef::OpResult {
                        op: new_split_id,
                        index: idx as u32,
                    };
                }
            }
            let split_sizes = vec![reshape_out_dims[new_axis]; num_splits];
            if let Some(new_split_op) = module.ops.get_mut(new_split_id) {
                new_split_op.name = OpName::new("mo", "split");
                new_split_op.operands = SmallVec::from_slice(&[pre_split_val]);
                new_split_op.results = SmallVec::from_vec(reshape_results);
                new_split_op
                    .attributes
                    .insert("axis".into(), Attribute::Int(new_axis as i64));
                new_split_op
                    .attributes
                    .insert("split_sizes".into(), Attribute::DenseIntArray(split_sizes));
            }
            for &dead_id in &reshape_ids[1..] {
                to_remove.insert(dead_id);
            }
            sunk += 1;
        }

        if !to_remove.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|id| !to_remove.contains(id));
            for dead_id in to_remove {
                module.ops.remove(dead_id);
            }
        }
    }

    sunk
}

/// Sinks or simplifies a sole-use broadcast (`mo.static.broadcast_to` or
/// `stablehlo.broadcast_in_dim`) feeding a single-axis keep-dims reduction (`mo.reduce_add`,
/// `mo.reduce_max`, `mo.reduce_min`, `stablehlo.reduce`).
///
/// - When reducing along a **non-broadcasted** axis, commutes `reduce(broadcast(x))` into
///   `broadcast(reduce(x, mapped_axis))`, reducing the smaller pre-broadcast tensor first.
/// - When reducing along a **broadcast-expanded** axis with an idempotent reduction
///   (`mo.reduce_max` or `mo.reduce_min`), folds `reduce(broadcast(x))` directly into a single
///   `broadcast(x)` to the reduced output shape.
///
/// Returns the number of broadcast-reduction pairs rewritten.
pub fn sink_broadcast_through_reduction(module: &mut Module) -> usize {
    let mut rewritten = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    *use_counts.entry(v).or_insert(0) += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for reduce_id in op_ids {
            if to_remove.contains(&reduce_id) {
                continue;
            }
            let Some(reduce_op) = module.ops.get(reduce_id).cloned() else {
                continue;
            };
            let reduce_name = reduce_op.name.to_string();
            if !matches!(
                reduce_name.as_str(),
                "mo.reduce_add" | "mo.reduce_max" | "mo.reduce_min" | "stablehlo.reduce"
            ) || reduce_op.operands.is_empty()
                || reduce_op.results.len() != 1
            {
                continue;
            }

            let bcast_out_val = reduce_op.operands[0];
            if use_counts.get(&bcast_out_val).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(bcast_out_info) = module.values.get(bcast_out_val).cloned() else {
                continue;
            };
            let ValueDef::OpResult { op: bcast_id, .. } = bcast_out_info.def else {
                continue;
            };
            if !module.blocks[block_id].ops.contains(&bcast_id) {
                continue;
            }
            let Some(bcast_op) = module.ops.get(bcast_id).cloned() else {
                continue;
            };
            let bcast_name = bcast_op.name.to_string();
            if !matches!(
                bcast_name.as_str(),
                "mo.static.broadcast_to" | "stablehlo.broadcast_in_dim"
            ) || bcast_op.operands.is_empty()
                || bcast_op.results.len() != 1
                || bcast_op.attributes.get("castforge.device")
                    != reduce_op.attributes.get("castforge.device")
            {
                continue;
            }

            let bcast_in_val = bcast_op.operands[0];
            let reduce_out_val = reduce_op.results[0];
            let (Some(in_info), Some(out_info)) = (
                module.values.get(bcast_in_val),
                module.values.get(reduce_out_val),
            ) else {
                continue;
            };
            let (Type::Tensor(in_tt), Type::Tensor(bcast_tt), Type::Tensor(out_tt)) = (
                in_info.ty.clone(),
                bcast_out_info.ty.clone(),
                out_info.ty.clone(),
            ) else {
                continue;
            };

            let rank_in = in_tt.shape.len();
            let rank_out = bcast_tt.shape.len();
            if rank_in == 0 || rank_out < rank_in || out_tt.shape.len() != rank_out {
                continue;
            }
            let extra_dims = rank_out - rank_in;
            if let Some(bcast_dims) = extract_int_list_attr(&bcast_op, "broadcast_dimensions") {
                let expected: Vec<i64> = (extra_dims as i64..rank_out as i64).collect();
                if bcast_dims != expected {
                    continue;
                }
            }

            let axis_i64 = reduce_op
                .attributes
                .get("axis")
                .or_else(|| reduce_op.attributes.get("dimensions"))
                .and_then(extract_single_int_attr);
            let Some(axis_raw) = axis_i64 else {
                continue;
            };
            let axis = if axis_raw < 0 {
                let adj = (rank_out as i64) + axis_raw;
                if adj < 0 {
                    continue;
                }
                adj as usize
            } else {
                axis_raw as usize
            };
            if axis >= rank_out || out_tt.shape[axis] != Dim::Static(1) {
                continue;
            }

            let is_broadcast_axis = axis < extra_dims
                || (in_tt.shape[axis - extra_dims] == Dim::Static(1)
                    && bcast_tt.shape[axis] != Dim::Static(1));

            if is_broadcast_axis {
                // Idempotent reductions (`reduce_max` / `reduce_min`) over a broadcast-expanded
                // dimension collapse to `broadcast(x) -> out_tt`.
                if !matches!(reduce_name.as_str(), "mo.reduce_max" | "mo.reduce_min") {
                    continue;
                }
                if let Some(v_mut) = module.values.get_mut(reduce_out_val) {
                    v_mut.def = ValueDef::OpResult {
                        op: bcast_id,
                        index: 0,
                    };
                }
                if let Some(b_mut) = module.ops.get_mut(bcast_id) {
                    b_mut.results[0] = reduce_out_val;
                }
                module.values.remove(bcast_out_val);
                to_remove.insert(reduce_id);
                rewritten += 1;
            } else {
                if rank_in <= 1 {
                    continue;
                }
                let mapped_axis = axis - extra_dims;
                let mut mid_shape = in_tt.shape.clone();
                mid_shape[mapped_axis] = Dim::Static(1);
                let mid_tt = TensorType {
                    shape: mid_shape,
                    dtype: out_tt.dtype.clone(),
                    encoding: out_tt.encoding.clone(),
                };

                if let Some(mid_val_mut) = module.values.get_mut(bcast_out_val) {
                    mid_val_mut.ty = Type::Tensor(mid_tt);
                }
                // Mutate `bcast_id` into the pre-broadcast reduction and `reduce_id` into the
                // post-reduction broadcast!
                let mut new_reduce_attrs = reduce_op.attributes.clone();
                let axis_key = if new_reduce_attrs.contains_key("dimensions")
                    && !new_reduce_attrs.contains_key("axis")
                {
                    "dimensions"
                } else {
                    "axis"
                };
                new_reduce_attrs.insert(axis_key.into(), Attribute::Int(mapped_axis as i64));
                let mut new_bcast_attrs = bcast_op.attributes.clone();
                if new_bcast_attrs.contains_key("shape") {
                    let out_dims: Option<Vec<i64>> = static_i64_dims(&out_tt);
                    if let Some(dims) = out_dims {
                        new_bcast_attrs.insert("shape".into(), Attribute::DenseIntArray(dims));
                    }
                }

                let mut extra_reduce_operands: SmallVec<[ValueId; 4]> =
                    SmallVec::from_slice(&[bcast_in_val]);
                for &extra_opd in &reduce_op.operands[1..] {
                    extra_reduce_operands.push(extra_opd);
                }

                if let Some(first_op) = module.ops.get_mut(bcast_id) {
                    first_op.name = reduce_op.name.clone();
                    first_op.operands = extra_reduce_operands;
                    first_op.attributes = new_reduce_attrs;
                }
                if let Some(second_op) = module.ops.get_mut(reduce_id) {
                    second_op.name = bcast_op.name.clone();
                    second_op.operands = SmallVec::from_slice(&[bcast_out_val]);
                    second_op.attributes = new_bcast_attrs;
                }
                rewritten += 1;
            }
        }

        if !to_remove.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|id| !to_remove.contains(id));
            for dead_id in to_remove {
                module.ops.remove(dead_id);
            }
        }
    }

    rewritten
}

/// Folds `mgp.extend_lifetimes` (or `castforge.extend_lifetimes`) operations into a preceding
/// sole-use asynchronous kernel execution (`mogg.async_exec`, `mgp.async_exec`,
/// `castforge.async_kernel`) or elides no-op lifetime extensions with zero buffer operands.
///
/// When folded into an async kernel launch, the extended buffer operands are deduplicated and
/// appended to the async kernel's operand list, `castforge.extended_lifetime_count` is stamped on
/// the async kernel, and the `extend_lifetimes` result chain is forwarded to the async kernel's
/// chain result. Returns the number of `extend_lifetimes` operations eliminated.
pub fn fold_extend_lifetimes_into_async_kernels(module: &mut Module) -> usize {
    let mut folded = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    *use_counts.entry(v).or_insert(0) += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut replacements: IndexMap<ValueId, ValueId> = IndexMap::new();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for op_id in op_ids {
            let Some(ext_op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !matches!(
                ext_op.name.to_string().as_str(),
                "mgp.extend_lifetimes" | "castforge.extend_lifetimes"
            ) || ext_op.operands.is_empty()
                || ext_op.results.len() != 1
            {
                continue;
            }
            let chain_in = replacements
                .get(&ext_op.operands[0])
                .copied()
                .unwrap_or(ext_op.operands[0]);
            let chain_out = ext_op.results[0];

            // Case 1: No-op `extend_lifetimes` with only the input chain and no buffers.
            if ext_op.operands.len() == 1 {
                replacements.insert(chain_out, chain_in);
                to_remove.insert(op_id);
                folded += 1;
                continue;
            }

            // Case 2: Preceding sole-use async kernel producing `chain_in`.
            if use_counts.get(&chain_in).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(chain_info) = module.values.get(chain_in) else {
                continue;
            };
            let ValueDef::OpResult { op: async_id, .. } = chain_info.def else {
                continue;
            };
            if !module.blocks[block_id].ops.contains(&async_id) || to_remove.contains(&async_id) {
                continue;
            }
            let Some(async_op) = module.ops.get(async_id) else {
                continue;
            };
            if !matches!(
                async_op.name.to_string().as_str(),
                "mogg.async_exec" | "mgp.async_exec" | "castforge.async_kernel"
            ) {
                continue;
            }

            let extra_buffers: Vec<ValueId> = ext_op.operands[1..]
                .iter()
                .map(|v| replacements.get(v).copied().unwrap_or(*v))
                .collect();

            if let Some(async_mut) = module.ops.get_mut(async_id) {
                let mut added = 0i64;
                for buf in extra_buffers {
                    if !async_mut.operands.contains(&buf) {
                        async_mut.operands.push(buf);
                        added += 1;
                    }
                }
                let prev = async_mut
                    .attributes
                    .get("castforge.extended_lifetime_count")
                    .and_then(extract_single_int_attr)
                    .unwrap_or(0);
                async_mut.attributes.insert(
                    "castforge.extended_lifetime_count".into(),
                    Attribute::Int(prev + added),
                );
            }
            replacements.insert(chain_out, chain_in);
            to_remove.insert(op_id);
            folded += 1;
        }

        if !replacements.is_empty() {
            for &op_id in &module.blocks[block_id].ops {
                if to_remove.contains(&op_id) {
                    continue;
                }
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    for opd in &mut op_mut.operands {
                        while let Some(&target) = replacements.get(opd) {
                            *opd = target;
                        }
                    }
                }
            }
        }
        if !to_remove.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|id| !to_remove.contains(id));
            for dead_id in to_remove {
                module.ops.remove(dead_id);
            }
        }
    }

    folded
}

/// Specification of a single execution node in a multi-device collective execution stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveGraphNodeSpec {
    /// Unique identifier of the operation in topological order.
    pub node_id: usize,
    /// Sorted list of device indices (`0..num_devices`) participating in this operation.
    pub devices: Vec<usize>,
    /// True if this operation is a multi-device collective (`all_reduce`, `reduce_scatter`,
    /// `all_gather`, `broadcast`).
    pub is_collective: bool,
    /// True if this operation can be captured inside a hardware device graph (`CUDA Graph` /
    /// `HIP Graph`).
    pub is_graph_capturable: bool,
}

/// A partitioned execution stage produced by [`plan_device_graph_collective_partition`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceGraphStage {
    /// A synchronized hardware device-graph capture stage mapping each device index
    /// `0..num_devices` to the ordered list of `node_id`s captured on that device's stream.
    DeviceGraphCapture {
        /// Per-device captured `node_id` lists of length `num_devices`.
        device_subgraphs: Vec<Vec<usize>>,
    },
    /// A multi-device collective stage executed outside device-graph capture (when
    /// `capture_collectives_in_graph` is `false` or when only a subset of devices participate).
    EagerCollectiveStage {
        /// Ordered `node_id`s of collective operations in this barrier stage.
        node_ids: Vec<usize>,
    },
    /// Host or dynamic-shape operations that cannot be captured in a device graph.
    EagerHostStage {
        /// Ordered `node_id`s of non-capturable operations.
        node_ids: Vec<usize>,
    },
}

/// Partitions a sequence of multi-device operations into synchronized [`DeviceGraphStage`]
/// regions for `mgp.device.graph` capture and collective barrier scheduling.
///
/// - Non-capturable operations flush any active [`DeviceGraphStage::DeviceGraphCapture`] stage and
///   run in [`DeviceGraphStage::EagerHostStage`].
/// - Capturable non-collective operations accumulate into the active per-device subgraphs.
/// - Collective operations are captured inside [`DeviceGraphStage::DeviceGraphCapture`] only when
///   `capture_collectives_in_graph` is `true`, `is_graph_capturable` is `true`, and all
///   `0..num_devices` participate synchronously; otherwise they flush the open graph capture and
///   execute in [`DeviceGraphStage::EagerCollectiveStage`].
pub fn plan_device_graph_collective_partition(
    num_devices: usize,
    nodes: &[CollectiveGraphNodeSpec],
    capture_collectives_in_graph: bool,
) -> Option<Vec<DeviceGraphStage>> {
    if num_devices == 0 {
        return None;
    }
    let mut stages: Vec<DeviceGraphStage> = Vec::new();
    let mut current_capture: Option<Vec<Vec<usize>>> = None;

    let flush_capture = |stages: &mut Vec<DeviceGraphStage>,
                         current_capture: &mut Option<Vec<Vec<usize>>>| {
        if let Some(subgraphs) = current_capture.take()
            && subgraphs.iter().any(|sg| !sg.is_empty())
        {
            stages.push(DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: subgraphs,
            });
        }
    };

    for node in nodes {
        if node.devices.is_empty() || node.devices.iter().any(|&d| d >= num_devices) {
            return None;
        }
        let mut unique_devs = node.devices.clone();
        unique_devs.sort_unstable();
        unique_devs.dedup();
        if unique_devs.len() != node.devices.len() {
            return None;
        }

        if !node.is_graph_capturable {
            flush_capture(&mut stages, &mut current_capture);
            if let Some(DeviceGraphStage::EagerHostStage { node_ids }) = stages.last_mut() {
                node_ids.push(node.node_id);
            } else {
                stages.push(DeviceGraphStage::EagerHostStage {
                    node_ids: vec![node.node_id],
                });
            }
            continue;
        }

        if node.is_collective {
            let all_devices_participate = unique_devs.len() == num_devices;
            if capture_collectives_in_graph && all_devices_participate {
                let subgraphs =
                    current_capture.get_or_insert_with(|| vec![Vec::new(); num_devices]);
                for &d in &unique_devs {
                    subgraphs[d].push(node.node_id);
                }
            } else {
                flush_capture(&mut stages, &mut current_capture);
                if let Some(DeviceGraphStage::EagerCollectiveStage { node_ids }) = stages.last_mut()
                {
                    node_ids.push(node.node_id);
                } else {
                    stages.push(DeviceGraphStage::EagerCollectiveStage {
                        node_ids: vec![node.node_id],
                    });
                }
            }
        } else {
            let subgraphs = current_capture.get_or_insert_with(|| vec![Vec::new(); num_devices]);
            for &d in &unique_devs {
                subgraphs[d].push(node.node_id);
            }
        }
    }

    flush_capture(&mut stages, &mut current_capture);
    Some(stages)
}
