//! Coordinator directory, FDP slot management, and horizontal fusion passes.

use super::*;
use crate::ir::{
    Attribute, BlockId, Dim, Module, OpId, OpName, Operation, TensorType, Type, Value, ValueDef,
    ValueId,
};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

/// Cache event operation kind applied to [`CoordinatorKeyDirectory`]
/// (inspired by LMCache `lmcache/v1/mp_coordinator/api.py` `CacheEventType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEventOpKind {
    /// Inserts or updates a placement for the key.
    Store,
    /// Removes a placement for the key.
    Delete,
    /// Records a cache hit/access on the key.
    Access,
}

/// Live placement of a cached key in the fleet directory
/// (inspired by LMCache `lmcache/v1/mp_coordinator/views/key_directory.py` `Placement`).
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
struct KeyDirectoryRecord {
    placements: Vec<PlacementRecord>,
    last_access_ts: u64,
    access_count: u64,
}

/// Fleet-wide key directory with incremental tier byte accounting and `O(|l1_keys(instance)|)`
/// reverse-indexed L1 instance fencing
/// (inspired by LMCache `lmcache/v1/mp_coordinator/views/key_directory.py` `KeyDirectory`).
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
                    let rec = self.directory.get_mut(&entry.key).unwrap();
                    rec.last_access_ts = rec.last_access_ts.max(ts);
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

fn static_i64_dims(tt: &TensorType) -> Option<Vec<i64>> {
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

/// Compression presets for the TurboQuant KV-cache serializer/deserializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboQuantPreset {
    /// 8-bit FP8 keys, 4-bit quantized values, without norm correction (`turboquant_k8v4`).
    K8V4,
    /// 4-bit MSE-centroid keys and 4-bit values with norm correction (`turboquant_4bit_nc`).
    FourBitNc,
    /// 3-bit MSE-centroid keys and 4-bit values with norm correction (`turboquant_k3v4_nc`).
    K3V4Nc,
    /// 3-bit MSE-centroid keys and 3-bit values with norm correction (`turboquant_3bit_nc`).
    ThreeBitNc,
}

impl TurboQuantPreset {
    /// Parses a preset name string (`turboquant_k8v4`, `turboquant_4bit_nc`,
    /// `turboquant_k3v4_nc`, `turboquant_3bit_nc`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "turboquant_k8v4" => Some(Self::K8V4),
            "turboquant_4bit_nc" => Some(Self::FourBitNc),
            "turboquant_k3v4_nc" => Some(Self::K3V4Nc),
            "turboquant_3bit_nc" => Some(Self::ThreeBitNc),
            _ => None,
        }
    }

    /// Returns `(key_quant_bits, value_quant_bits, norm_correction)` for this preset.
    pub fn params(self) -> (usize, usize, bool) {
        match self {
            Self::K8V4 => (8, 4, false),
            Self::FourBitNc => (4, 4, true),
            Self::K3V4Nc => (3, 4, true),
            Self::ThreeBitNc => (3, 3, true),
        }
    }
}

/// Configuration and slot-geometry planner for TurboQuant KV-cache compression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurboQuantSerdeConfig {
    /// Compression bit-width preset.
    pub preset: TurboQuantPreset,
    /// Per-head hidden dimension (must be a positive power of two for WHT rotation).
    pub head_dim: usize,
    /// Token block size used by the compressed layout.
    pub block_size: usize,
    /// Number of leading transformer layers stored uncompressed.
    pub skip_first_layers: usize,
    /// Number of trailing transformer layers stored uncompressed.
    pub skip_last_layers: usize,
}

impl TurboQuantSerdeConfig {
    /// Creates a new [`TurboQuantSerdeConfig`], returning `None` if `head_dim == 0` or
    /// `block_size == 0`.
    pub fn new(
        preset: TurboQuantPreset,
        head_dim: usize,
        block_size: usize,
        skip_first_layers: usize,
        skip_last_layers: usize,
    ) -> Option<Self> {
        if head_dim == 0 || block_size == 0 {
            return None;
        }
        Some(Self {
            preset,
            head_dim,
            block_size,
            skip_first_layers,
            skip_last_layers,
        })
    }

    /// Returns true when keys are stored in 8-bit FP8 format (`key_quant_bits == 8`).
    pub fn key_fp8(&self) -> bool {
        self.preset.params().0 == 8
    }

    /// Returns the number of MSE Lloyd-Max quantization bits per key coordinate (`0` when FP8).
    pub fn key_mse_bits(&self) -> usize {
        if self.key_fp8() {
            0
        } else {
            self.preset.params().0
        }
    }

    /// Returns the number of bits used for the Lloyd-Max centroid codebook.
    pub fn centroid_bits(&self) -> usize {
        let (k_bits, v_bits, _) = self.preset.params();
        if k_bits == 8 { v_bits } else { k_bits }
    }

    /// Returns the packed byte size of one head's key vector (including 2-byte FP16 norm when
    /// sub-8-bit MSE quantization is used).
    pub fn key_packed_size(&self) -> usize {
        if self.key_fp8() {
            self.head_dim
        } else {
            (self.head_dim * self.key_mse_bits()).div_ceil(8) + 2
        }
    }

    /// Returns the packed byte size of one head's value vector (including 4 bytes for FP16 scale
    /// and FP16 zero-point).
    pub fn value_packed_size(&self) -> usize {
        let v_bits = self.preset.params().1;
        (self.head_dim * v_bits).div_ceil(8) + 4
    }

    /// Returns the unaligned per-head slot size in bytes (`key_packed_size + value_packed_size`).
    pub fn slot_size(&self) -> usize {
        self.key_packed_size() + self.value_packed_size()
    }

    /// Returns the 2-byte aligned per-head slot size in bytes.
    pub fn slot_size_aligned(&self) -> usize {
        let s = self.slot_size();
        s + (s % 2)
    }

    /// Returns the half-open layer index range `(quant_start, quant_end)` compressed with
    /// TurboQuant out of `num_layers`.
    pub fn layer_ranges(&self, num_layers: usize) -> (usize, usize) {
        let quant_start = self.skip_first_layers.min(num_layers);
        let quant_end = num_layers
            .saturating_sub(self.skip_last_layers)
            .max(quant_start);
        (quant_start, quant_end)
    }

    /// Computes total serialized byte size for a 4D KV tensor `[2, num_layers, num_tokens,
    /// hidden_dim]` with `elem_bytes` per uncompressed element.
    pub fn serialized_nbytes_for_shape(
        &self,
        num_layers: usize,
        num_tokens: usize,
        hidden_dim: usize,
        elem_bytes: usize,
    ) -> Option<usize> {
        if hidden_dim == 0 || !hidden_dim.is_multiple_of(self.head_dim) || elem_bytes == 0 {
            return None;
        }
        let num_heads = hidden_dim / self.head_dim;
        let num_blocks = num_tokens.div_ceil(self.block_size);
        let (quant_start, quant_end) = self.layer_ranges(num_layers);
        let quant_layers = quant_end - quant_start;
        let raw_layers = num_layers - quant_layers;
        let raw_bytes = 2usize
            .checked_mul(raw_layers)?
            .checked_mul(num_tokens)?
            .checked_mul(hidden_dim)?
            .checked_mul(elem_bytes)?;
        let quant_bytes = quant_layers
            .checked_mul(num_blocks)?
            .checked_mul(self.block_size)?
            .checked_mul(num_heads)?
            .checked_mul(self.slot_size_aligned())?;
        raw_bytes.checked_add(quant_bytes)
    }
}

/// Solves the 1D Lloyd-Max optimal scalar quantizer for a zero-mean Gaussian distribution
/// $\mathcal{N}(0, 1/\text{head\_dim})$ with `2^bits` reconstruction centroids and `(2^bits) - 1`
/// decision boundaries using composite trapezoidal integration.
pub fn solve_lloyd_max_gaussian(
    head_dim: usize,
    bits: usize,
    max_iter: usize,
    tol: f64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if head_dim == 0 || bits == 0 || bits > 8 {
        return None;
    }
    let n_levels = 1usize << bits;
    let sigma2 = 1.0 / (head_dim as f64);
    let sigma = sigma2.sqrt();
    let inv_norm = 1.0 / (2.0 * std::f64::consts::PI * sigma2).sqrt();
    let pdf = |x: f64| -> f64 { inv_norm * (-x * x / (2.0 * sigma2)).exp() };
    let trapz = |f: &dyn Fn(f64) -> f64, a: f64, b: f64| -> f64 {
        let n = 200usize;
        let h = (b - a) / (n as f64);
        let mut sum = 0.5 * (f(a) + f(b));
        for i in 1..n {
            sum += f(a + (i as f64) * h);
        }
        sum * h
    };

    let lo = -3.5 * sigma;
    let hi = 3.5 * sigma;
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| lo + (hi - lo) * ((i as f64) + 0.5) / (n_levels as f64))
        .collect();

    for _ in 0..max_iter {
        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(lo * 3.0);
        for i in 0..(n_levels - 1) {
            edges.push(0.5 * (centroids[i] + centroids[i + 1]));
        }
        edges.push(hi * 3.0);

        let mut new_centroids = Vec::with_capacity(n_levels);
        let mut max_diff = 0.0f64;
        for i in 0..n_levels {
            let a = edges[i];
            let b = edges[i + 1];
            let num = trapz(&|x| x * pdf(x), a, b);
            let den = trapz(&pdf, a, b);
            let next_c = if den > 1e-15 { num / den } else { centroids[i] };
            max_diff = max_diff.max((next_c - centroids[i]).abs());
            new_centroids.push(next_c);
        }
        centroids = new_centroids;
        if max_diff < tol {
            break;
        }
    }

    let boundaries: Vec<f32> = (0..(n_levels - 1))
        .map(|i| (0.5 * (centroids[i] + centroids[i + 1])) as f32)
        .collect();
    let centroids_f32: Vec<f32> = centroids.into_iter().map(|c| c as f32).collect();
    Some((centroids_f32, boundaries))
}

/// Constructs the row-major `head_dim x head_dim` orthonormal Sylvester Walsh-Hadamard matrix
/// $H_d / \sqrt{d}$ for a power-of-two `head_dim`.
pub fn build_orthonormal_hadamard(head_dim: usize) -> Option<Vec<f32>> {
    if head_dim == 0 || !head_dim.is_power_of_two() {
        return None;
    }
    let mut mat = vec![1.0f32; head_dim * head_dim];
    let mut size = 1usize;
    while size < head_dim {
        for r in 0..size {
            for c in 0..size {
                let v = mat[r * head_dim + c];
                mat[r * head_dim + (c + size)] = v;
                mat[(r + size) * head_dim + c] = v;
                mat[(r + size) * head_dim + (c + size)] = -v;
            }
        }
        size *= 2;
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    for v in &mut mat {
        *v *= scale;
    }
    Some(mat)
}

/// Zero-copy stride-sorted view recovery plan for physical KV-cache layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousViewPlan {
    /// Dimension permutation sorting axes by descending stride (with size-1 dimensions placed
    /// innermost on stride ties).
    pub permutation: Vec<usize>,
    /// Shape after applying `permutation` (or the stride-derived tight shape when strictly
    /// contiguous).
    pub permuted_shape: Vec<usize>,
    /// Strides after applying `permutation`.
    pub permuted_strides: Vec<usize>,
    /// True if the permuted view is strictly contiguous with zero block-row padding.
    pub is_strictly_contiguous: bool,
    /// Number of padding elements per outer (`dim-0`) block row (`0` when strictly contiguous).
    pub padding_per_block_elems: usize,
    /// Stride in elements between consecutive outer (`dim-0`) blocks (`permuted_strides[0]`).
    pub block_stride_elems: usize,
}

/// Derives the tightest contiguous shape `[num_elems / strides[0], strides[0] / strides[1], ...]`
/// from a monotonically non-increasing positive `strides` vector and total element count
/// `num_elems`.
pub fn expected_contiguous_shape_from_strides(
    strides: &[usize],
    num_elems: usize,
) -> Option<Vec<usize>> {
    if strides.is_empty() || strides.contains(&0) {
        return None;
    }
    let mut expected = Vec::with_capacity(strides.len());
    expected.push(num_elems / strides[0]);
    for i in 0..(strides.len() - 1) {
        if strides[i] < strides[i + 1] || !strides[i].is_multiple_of(strides[i + 1]) {
            return None;
        }
        expected.push(strides[i] / strides[i + 1]);
    }
    Some(expected)
}

/// Recovers a zero-copy contiguous or `dim-0`-padded physical view from a logical KV-cache
/// `shape` and `strides`.
///
/// Sorts dimensions by `(strides[i], shape[i] != 1)` in descending order so transposed views
/// (such as logical `[2, NB, BS, NH, HS]` backed by physical `[2, NB, NH, BS, HS]`) are
/// restored without copying and size-1 dimensions tying on stride are placed innermost.
/// Rejects views with interior-dimension padding or overlapping outer blocks (`strides[0] <
/// prod(shape[1..])`).
pub fn recover_contiguous_or_dim0_padded_view(
    shape: &[usize],
    strides: &[usize],
) -> Option<ContiguousViewPlan> {
    let ndim = shape.len();
    if ndim < 2 || strides.len() != ndim || shape.contains(&0) || strides.contains(&0) {
        return None;
    }
    let mut num_elems = 1usize;
    for &d in shape {
        num_elems = num_elems.checked_mul(d)?;
    }

    let mut permutation: Vec<usize> = (0..ndim).collect();
    permutation.sort_by(|&a, &b| {
        let key_a = (strides[a], shape[a] != 1);
        let key_b = (strides[b], shape[b] != 1);
        key_b.cmp(&key_a)
    });

    let mut permuted_shape: Vec<usize> = permutation.iter().map(|&i| shape[i]).collect();
    let permuted_strides: Vec<usize> = permutation.iter().map(|&i| strides[i]).collect();

    if permuted_strides[ndim - 1] != 1 || permuted_strides[ndim - 2] != permuted_shape[ndim - 1] {
        return None;
    }

    let mut inner_tight = 1usize;
    for i in (1..ndim).rev() {
        if i < ndim - 1 && permuted_strides[i] != inner_tight {
            return None;
        }
        inner_tight = inner_tight.checked_mul(permuted_shape[i])?;
    }

    let block_stride_elems = permuted_strides[0];
    if block_stride_elems < inner_tight {
        return None;
    }
    let padding_per_block_elems = block_stride_elems - inner_tight;
    let is_strictly_contiguous = padding_per_block_elems == 0;
    if is_strictly_contiguous
        && let Some(tight_shape) =
            expected_contiguous_shape_from_strides(&permuted_strides, num_elems)
    {
        permuted_shape = tight_shape;
    }

    Some(ContiguousViewPlan {
        permutation,
        permuted_shape,
        permuted_strides,
        is_strictly_contiguous,
        padding_per_block_elems,
        block_stride_elems,
    })
}

/// Multi-tier fleet cache byte-usage view rolled up per key, per `cache_salt`, and per
/// `(instance_id, backend)` with $O(\text{instance placements})$ `L1` fencing.
#[derive(Debug, Clone, Default)]
pub struct FleetCacheUsageView {
    /// `(tier, key, cache_salt, owner, backend) -> size_bytes`
    placement_sizes: IndexMap<(CacheEventTier, u64, String, String, String), u64>,
    /// `(tier, key) -> total_bytes`
    key_bytes: IndexMap<(CacheEventTier, u64), u64>,
    /// `(tier, cache_salt) -> total_bytes`
    salt_bytes: IndexMap<(CacheEventTier, String), u64>,
    /// `(tier, owner, backend) -> total_bytes`
    instance_bytes: IndexMap<(CacheEventTier, String, String), u64>,
    /// Reverse index `owner -> set of (key, cache_salt, backend)` for `L1` placements.
    l1_by_instance: IndexMap<String, IndexSet<(u64, String, String)>>,
}

impl FleetCacheUsageView {
    /// Creates an empty [`FleetCacheUsageView`].
    pub fn new() -> Self {
        Self::default()
    }

    fn adjust_rollups(
        &mut self,
        tier: CacheEventTier,
        key: u64,
        cache_salt: &str,
        owner: &str,
        backend: &str,
        delta: i64,
    ) {
        let update_map = |val: Option<&u64>, d: i64| -> Option<u64> {
            let cur = val.copied().unwrap_or(0) as i64 + d;
            (cur > 0).then_some(cur as u64)
        };
        let k_id = (tier, key);
        if let Some(next) = update_map(self.key_bytes.get(&k_id), delta) {
            self.key_bytes.insert(k_id, next);
        } else {
            self.key_bytes.shift_remove(&k_id);
        }

        let s_id = (tier, cache_salt.to_string());
        if let Some(next) = update_map(self.salt_bytes.get(&s_id), delta) {
            self.salt_bytes.insert(s_id, next);
        } else {
            self.salt_bytes.shift_remove(&s_id);
        }

        let i_id = (tier, owner.to_string(), backend.to_string());
        if let Some(next) = update_map(self.instance_bytes.get(&i_id), delta) {
            self.instance_bytes.insert(i_id, next);
        } else {
            self.instance_bytes.shift_remove(&i_id);
        }
    }

    /// Consumes a gate-admitted cache-event batch for `cache_salt`.
    ///
    /// Shared pools (`shared == true`) are indexed under `owner = ""` so single-instance fencing
    /// never drops fleet-shared storage bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn consume_batch(
        &mut self,
        instance_id: &str,
        tier: CacheEventTier,
        backend: &str,
        shared: bool,
        op: CacheEventOpKind,
        cache_salt: &str,
        entries: &[CacheEventEntrySpec],
    ) {
        if op == CacheEventOpKind::Access {
            return;
        }
        let owner = if shared { "" } else { instance_id };
        for entry in entries {
            let pid = (
                tier,
                entry.key,
                cache_salt.to_string(),
                owner.to_string(),
                backend.to_string(),
            );
            match op {
                CacheEventOpKind::Store => {
                    let old = self
                        .placement_sizes
                        .insert(pid, entry.size_bytes)
                        .unwrap_or(0);
                    if tier == CacheEventTier::L1 && !owner.is_empty() {
                        self.l1_by_instance
                            .entry(owner.to_string())
                            .or_default()
                            .insert((entry.key, cache_salt.to_string(), backend.to_string()));
                    }
                    let delta = (entry.size_bytes as i64) - (old as i64);
                    if delta != 0 {
                        self.adjust_rollups(tier, entry.key, cache_salt, owner, backend, delta);
                    }
                }
                CacheEventOpKind::Delete => {
                    if let Some(old) = self.placement_sizes.shift_remove(&pid) {
                        if tier == CacheEventTier::L1
                            && !owner.is_empty()
                            && let Some(set) = self.l1_by_instance.get_mut(owner)
                        {
                            set.shift_remove(&(
                                entry.key,
                                cache_salt.to_string(),
                                backend.to_string(),
                            ));
                            if set.is_empty() {
                                self.l1_by_instance.shift_remove(owner);
                            }
                        }
                        self.adjust_rollups(
                            tier,
                            entry.key,
                            cache_salt,
                            owner,
                            backend,
                            -(old as i64),
                        );
                    }
                }
                CacheEventOpKind::Access => {}
            }
        }
    }

    /// Drops all `L1` bytes reported by `instance_id` while preserving `L2` and shared-pool bytes.
    /// Returns the total `L1` bytes removed.
    pub fn fence_instance(&mut self, instance_id: &str) -> u64 {
        let Some(items) = self.l1_by_instance.shift_remove(instance_id) else {
            return 0;
        };
        let mut removed_bytes = 0u64;
        for (key, salt, backend) in items {
            let pid = (
                CacheEventTier::L1,
                key,
                salt.clone(),
                instance_id.to_string(),
                backend.clone(),
            );
            if let Some(old) = self.placement_sizes.shift_remove(&pid) {
                removed_bytes += old;
                self.adjust_rollups(
                    CacheEventTier::L1,
                    key,
                    &salt,
                    instance_id,
                    &backend,
                    -(old as i64),
                );
            }
        }
        removed_bytes
    }

    /// Returns total bytes held for `key` in `tier` across all active placements.
    pub fn get_key_bytes(&self, tier: CacheEventTier, key: u64) -> u64 {
        self.key_bytes.get(&(tier, key)).copied().unwrap_or(0)
    }

    /// Returns total bytes held under `cache_salt` in `tier`.
    pub fn get_salt_bytes(&self, tier: CacheEventTier, cache_salt: &str) -> u64 {
        self.salt_bytes
            .get(&(tier, cache_salt.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Returns total bytes held across all tenants in `tier`.
    pub fn get_total_bytes(&self, tier: CacheEventTier) -> u64 {
        self.salt_bytes
            .iter()
            .filter_map(|((t, _), &b)| (*t == tier).then_some(b))
            .sum()
    }
}

/// Fleet-wide per-`cache_salt` `L2` watermark eviction controller with reference-counted key
/// pinning and last-copy-aware LRU accounting.
#[derive(Debug, Clone)]
pub struct FleetWatermarkEvictionController {
    /// Per-tenant byte quota registry.
    pub quota: KvSaltQuotaRegistry,
    /// Per-tenant isolated LRU eviction policy.
    pub policy: IsolatedLruEvictionPolicy,
    pin_counts: IndexMap<u64, usize>,
    eviction_ratio: f64,
    trigger_watermark: f64,
    max_delete_batch: usize,
}

impl FleetWatermarkEvictionController {
    /// Creates a new [`FleetWatermarkEvictionController`].
    pub fn new(eviction_ratio: f64, trigger_watermark: f64, max_delete_batch: usize) -> Self {
        Self {
            quota: KvSaltQuotaRegistry::new(),
            policy: IsolatedLruEvictionPolicy::new(),
            pin_counts: IndexMap::new(),
            eviction_ratio: eviction_ratio.clamp(0.0, 1.0),
            trigger_watermark: trigger_watermark.max(0.0),
            max_delete_batch: max_delete_batch.max(1),
        }
    }

    /// Applies an admitted `L2` batch to the controller's LRU after `usage` has already processed
    /// the same batch.
    ///
    /// A `Delete` event removes a key from the LRU only when its last `L2` placement is gone
    /// (`usage.get_key_bytes(L2, key) == 0`).
    pub fn consume_l2_batch(
        &mut self,
        usage: &FleetCacheUsageView,
        op: CacheEventOpKind,
        cache_salt: &str,
        keys: &[u64],
    ) {
        match op {
            CacheEventOpKind::Store => self.policy.on_keys_created(cache_salt, keys),
            CacheEventOpKind::Access => self.policy.on_keys_touched(cache_salt, keys),
            CacheEventOpKind::Delete => {
                for &k in keys {
                    if usage.get_key_bytes(CacheEventTier::L2, k) == 0 {
                        self.policy.on_keys_removed(cache_salt, &[k]);
                    }
                }
            }
        }
    }

    /// Increments the active `L2` pin count for each key in `keys`, excluding it from eviction.
    pub fn pin(&mut self, keys: &[u64]) {
        for &k in keys {
            *self.pin_counts.entry(k).or_insert(0) += 1;
        }
    }

    /// Decrements the active `L2` pin count for each key in `keys`, removing the entry at `0`.
    pub fn unpin(&mut self, keys: &[u64]) {
        for &k in keys {
            if let Some(c) = self.pin_counts.get_mut(&k) {
                if *c <= 1 {
                    self.pin_counts.shift_remove(&k);
                } else {
                    *c -= 1;
                }
            }
        }
    }

    /// Removes all pins for `keys` (used by force-delete).
    pub fn drop_pins(&mut self, keys: &[u64]) {
        for &k in keys {
            self.pin_counts.shift_remove(&k);
        }
    }

    /// Returns the active pin count for `key`.
    pub fn pin_count(&self, key: u64) -> usize {
        self.pin_counts.get(&key).copied().unwrap_or(0)
    }

    /// Filters `keys` to those with no active `L2` pin, preserving input order.
    pub fn filter_unpinned(&self, keys: &[u64]) -> Vec<u64> {
        keys.iter()
            .copied()
            .filter(|k| !self.pin_counts.contains_key(k))
            .collect()
    }

    /// Computes the per-`cache_salt` `L2` eviction plan against `usage`.
    ///
    /// Salts without an effective quota (`effective_limit_bytes == None`) are exempt; salts with
    /// `current_bytes >= trigger_watermark * limit` select `ceil(len * effective_ratio)` unpinned
    /// LRU keys (`effective_ratio = 1.0` when `limit == 0`).
    pub fn compute_eviction_plan(&self, usage: &FleetCacheUsageView) -> IndexMap<String, Vec<u64>> {
        let mut plan = IndexMap::new();
        for salt in self.policy.tracked_salts() {
            let current_bytes = usage.get_salt_bytes(CacheEventTier::L2, &salt);
            if current_bytes == 0 {
                continue;
            }
            let Some(limit) = self.quota.effective_limit_bytes(&salt) else {
                continue;
            };
            if (current_bytes as f64) < self.trigger_watermark * (limit as f64) {
                continue;
            }
            let effective_ratio = if limit == 0 { 1.0 } else { self.eviction_ratio };
            let victims =
                self.policy
                    .select_eviction_candidates_filtered(&salt, effective_ratio, |k| {
                        !self.pin_counts.contains_key(&k)
                    });
            if !victims.is_empty() {
                plan.insert(salt, victims);
            }
        }
        plan
    }

    /// Splits a computed eviction plan into outbound delete request batches of at most
    /// `max_delete_batch` keys each.
    pub fn chunk_delete_batches(
        &self,
        plan: &IndexMap<String, Vec<u64>>,
    ) -> Vec<(String, Vec<u64>)> {
        let mut batches = Vec::new();
        for (salt, keys) in plan {
            for chunk in keys.chunks(self.max_delete_batch) {
                batches.push((salt.clone(), chunk.to_vec()));
            }
        }
        batches
    }
}

fn is_horizontally_fusible_elementwise(op: &Operation) -> bool {
    is_sinkable_unary_elementwise(op) || is_sinkable_binary_elementwise(op)
}

fn op_transitively_depends_on(
    module: &Module,
    block_ops: &[OpId],
    source_op: OpId,
    target_op: OpId,
) -> bool {
    let mut reachable_vals: IndexSet<ValueId> = IndexSet::new();
    if let Some(src) = module.ops.get(source_op) {
        for &r in &src.results {
            reachable_vals.insert(r);
        }
    }
    for &op_id in block_ops {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if op.operands.iter().any(|v| reachable_vals.contains(v)) {
            if op_id == target_op {
                return true;
            }
            for &r in &op.results {
                reachable_vals.insert(r);
            }
        }
    }
    false
}

/// Fuses pairs of sibling pure elementwise operations in a basic block that share the same output
/// shape, device placement, and at least one common input operand into a multi-result
/// `mo.horizontal_fused` operation so the shared input is loaded only once.
///
/// The merged operation is placed at the later sibling's position (`b`), provided no operation
/// between `a` and `b` reads `a`'s result (`resultsUnusedBefore(a, b)`) and `b` does not depend
/// directly or transitively on `a`. Returns the number of horizontal sibling pairs fused.
pub fn fuse_horizontal_shared_input_ops(module: &mut Module) -> usize {
    let mut fused_pairs = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let mut consumed: IndexSet<OpId> = IndexSet::new();

        for i in 0..op_ids.len() {
            let a_id = op_ids[i];
            if consumed.contains(&a_id) {
                continue;
            }
            let Some(a_op) = module.ops.get(a_id).cloned() else {
                continue;
            };
            if !is_horizontally_fusible_elementwise(&a_op) {
                continue;
            }
            let a_res = a_op.results[0];
            let Some(a_res_tt) = module
                .values
                .get(a_res)
                .and_then(|v| v.ty.as_tensor())
                .cloned()
            else {
                continue;
            };

            for j in (i + 1)..op_ids.len() {
                let b_id = op_ids[j];
                if consumed.contains(&b_id) {
                    continue;
                }
                let Some(b_op) = module.ops.get(b_id).cloned() else {
                    continue;
                };
                if !is_horizontally_fusible_elementwise(&b_op) {
                    continue;
                }
                if a_op.attributes.get("castforge.device")
                    != b_op.attributes.get("castforge.device")
                {
                    continue;
                }
                let b_res = b_op.results[0];
                let Some(b_res_tt) = module
                    .values
                    .get(b_res)
                    .and_then(|v| v.ty.as_tensor())
                    .cloned()
                else {
                    continue;
                };
                if a_res_tt.shape != b_res_tt.shape {
                    continue;
                }

                // Require at least one shared input tensor operand.
                let shares_input = a_op.operands.iter().any(|v| b_op.operands.contains(v));
                if !shares_input {
                    continue;
                }

                // Check `resultsUnusedBefore(a, b)` over `(i + 1)..=j` and absence of transitive
                // dependency from `a` to `b`.
                let used_before_or_at_b = op_ids[(i + 1)..=j].iter().any(|&mid_id| {
                    !consumed.contains(&mid_id)
                        && module
                            .ops
                            .get(mid_id)
                            .is_some_and(|op| op.operands.contains(&a_res))
                });
                if used_before_or_at_b || op_transitively_depends_on(module, &op_ids, a_id, b_id) {
                    continue;
                }

                // Merge `a` and `b` into `mo.horizontal_fused` at `b_id`.
                let mut merged_operands: SmallVec<[ValueId; 4]> = a_op.operands.clone();
                for &opd in &b_op.operands {
                    if !merged_operands.contains(&opd) {
                        merged_operands.push(opd);
                    }
                }
                let a_name = a_op.name.to_string();
                let b_name = b_op.name.to_string();
                let a_loc = a_op.location.clone();
                let b_loc = b_op.location.clone();

                if let Some(v_a) = module.values.get_mut(a_res) {
                    v_a.def = ValueDef::OpResult { op: b_id, index: 0 };
                }
                if let Some(v_b) = module.values.get_mut(b_res) {
                    v_b.def = ValueDef::OpResult { op: b_id, index: 1 };
                }
                if let Some(b_mut) = module.ops.get_mut(b_id) {
                    b_mut.name = OpName::new("mo", "horizontal_fused");
                    b_mut.operands = merged_operands;
                    b_mut.results = SmallVec::from_slice(&[a_res, b_res]);
                    b_mut.attributes.insert(
                        "fused_ops".into(),
                        Attribute::Array(vec![
                            Attribute::String(a_name),
                            Attribute::String(b_name),
                        ]),
                    );
                    stamp_fused_provenance(b_mut, &[a_loc, b_loc]);
                }

                consumed.insert(a_id);
                consumed.insert(b_id);
                module.ops.remove(a_id);
                fused_pairs += 1;
                break;
            }
        }

        if !consumed.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|id| module.ops.contains_key(*id));
        }
    }

    fused_pairs
}

/// Hoists callsite-invariant operations out of non-entry callee subgraphs (`func.func`) into
/// their caller block before the earliest `func.call` call site.
///
/// A subgraph parameter is callsite-invariant when every call site passes the exact same SSA
/// value. Forward `ConstLattice` propagation marks pure operations inside the callee that depend
/// only on callsite-invariant parameters (or constants), moves them before the earliest call site
/// in the caller, passes their boundary values into the callee as new parameters, and prunes dead
/// invariant parameters. Returns the number of operations hoisted out of callees.
pub fn hoist_callsite_invariant_subgraph_ops(module: &mut Module) -> usize {
    let mut total_hoisted = 0usize;
    let fn_ids = module.functions();

    // Map symbol name -> (fn_op_id, entry_block_id) for non-entry functions.
    let mut callees: IndexMap<String, (OpId, BlockId)> = IndexMap::new();
    for &fn_id in &fn_ids {
        let Some(op) = module.ops.get(fn_id) else {
            continue;
        };
        let Some(sym) = op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        if matches!(sym.as_str(), "main" | "forward") {
            continue;
        }
        let Some(&reg_id) = op.regions.first() else {
            continue;
        };
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        if reg.blocks.len() != 1 {
            continue;
        }
        callees.insert(sym, (fn_id, reg.blocks[0]));
    }

    let all_blocks: Vec<BlockId> = module.blocks.keys().collect();
    for (sym, (_fn_id, callee_blk)) in callees {
        // Collect all `func.call` sites targeting `sym`.
        let mut callsites: Vec<(BlockId, OpId)> = Vec::new();
        for &blk_id in &all_blocks {
            if blk_id == callee_blk {
                continue;
            }
            for &op_id in &module.blocks[blk_id].ops {
                if let Some(op) = module.ops.get(op_id)
                    && op.is_op("func", "call")
                    && op
                        .attributes
                        .get("callee")
                        .and_then(|a| a.as_str())
                        .is_some_and(|c| c == sym)
                {
                    callsites.push((blk_id, op_id));
                }
            }
        }
        if callsites.is_empty() {
            continue;
        }
        let caller_blk = callsites[0].0;
        if callsites.iter().any(|&(b, _)| b != caller_blk) {
            continue;
        }

        let callee_args = module.blocks[callee_blk].arguments.clone();
        if callee_args.is_empty() {
            continue;
        }
        if callsites
            .iter()
            .any(|&(_, cid)| module.ops[cid].operands.len() != callee_args.len())
        {
            continue;
        }

        // Identify callsite-invariant parameters: every callsite passes the same caller `ValueId`.
        let mut invariant_arg_to_caller: IndexMap<ValueId, ValueId> = IndexMap::new();
        for (arg_idx, &callee_arg) in callee_args.iter().enumerate() {
            let first_val = module.ops[callsites[0].1].operands[arg_idx];
            if callsites
                .iter()
                .all(|&(_, cid)| module.ops[cid].operands[arg_idx] == first_val)
            {
                invariant_arg_to_caller.insert(callee_arg, first_val);
            }
        }
        if invariant_arg_to_caller.is_empty() {
            continue;
        }

        // Forward `ConstLattice` propagation through `callee_blk`.
        let mut const_vals: IndexSet<ValueId> = invariant_arg_to_caller.keys().copied().collect();
        let mut hoisted_ops: Vec<OpId> = Vec::new();
        let mut hoisted_set: IndexSet<OpId> = IndexSet::new();
        let callee_ops = module.blocks[callee_blk].ops.clone();

        for &op_id in &callee_ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.is_op("func", "return") || op.is_op("func", "call") || !op.regions.is_empty() {
                continue;
            }
            let is_pure_like = op.is_pure() || op.name.dialect == "mo";
            if !is_pure_like {
                continue;
            }
            if op.operands.iter().all(|v| const_vals.contains(v)) {
                hoisted_ops.push(op_id);
                hoisted_set.insert(op_id);
                for &r in &op.results {
                    const_vals.insert(r);
                }
            }
        }
        if hoisted_ops.is_empty() {
            continue;
        }

        // Find boundary values produced by `hoisted_ops` that feed non-hoisted ops in `callee_blk`.
        let mut boundary_vals: Vec<ValueId> = Vec::new();
        for &op_id in &hoisted_ops {
            let results = module.ops[op_id].results.clone();
            for res_v in results {
                let used_by_remaining = callee_ops.iter().any(|&other_id| {
                    !hoisted_set.contains(&other_id)
                        && module.ops[other_id].operands.contains(&res_v)
                });
                if used_by_remaining && !boundary_vals.contains(&res_v) {
                    boundary_vals.push(res_v);
                }
            }
        }

        // Move `hoisted_ops` into `caller_blk` immediately before `callsites[0].1` and rewire
        // their invariant parameter operands to the caller's SSA values.
        for &h_id in &hoisted_ops {
            if let Some(h_op) = module.ops.get_mut(h_id) {
                for opd in &mut h_op.operands {
                    if let Some(&caller_v) = invariant_arg_to_caller.get(opd) {
                        *opd = caller_v;
                    }
                }
            }
        }
        let first_call_id = callsites[0].1;
        if let Some(pos) = module.blocks[caller_blk]
            .ops
            .iter()
            .position(|&id| id == first_call_id)
        {
            for (offset, &h_id) in hoisted_ops.iter().enumerate() {
                module.blocks[caller_blk].ops.insert(pos + offset, h_id);
            }
        }
        module.blocks[callee_blk]
            .ops
            .retain(|id| !hoisted_set.contains(id));

        // Create new callee block arguments for each boundary value and pass them at every call.
        for &b_val in &boundary_vals {
            let b_ty = module.values[b_val].ty.clone();
            let new_arg_idx = module.blocks[callee_blk].arguments.len() as u32;
            let new_arg_val = module.values.insert_with_key(|vid| Value {
                id: vid,
                ssa_name: None,
                ty: b_ty,
                def: ValueDef::BlockArgument {
                    block: callee_blk,
                    index: new_arg_idx,
                },
            });
            module.blocks[callee_blk].arguments.push(new_arg_val);

            for &rem_id in &module.blocks[callee_blk].ops {
                if let Some(rem_op) = module.ops.get_mut(rem_id) {
                    for opd in &mut rem_op.operands {
                        if *opd == b_val {
                            *opd = new_arg_val;
                        }
                    }
                }
            }
            for &(_, call_id) in &callsites {
                if let Some(call_op) = module.ops.get_mut(call_id) {
                    call_op.operands.push(b_val);
                }
            }
        }

        // Prune dead original parameters from `callee_blk` and all its call sites.
        let mut used_in_callee: IndexSet<ValueId> = IndexSet::new();
        for &rem_id in &module.blocks[callee_blk].ops {
            if let Some(rem_op) = module.ops.get(rem_id) {
                for &v in &rem_op.operands {
                    used_in_callee.insert(v);
                }
            }
        }
        let cur_args = module.blocks[callee_blk].arguments.clone();
        let keep_mask: Vec<bool> = cur_args
            .iter()
            .map(|a| used_in_callee.contains(a))
            .collect();
        if keep_mask.iter().any(|&k| !k) {
            let mut new_args = Vec::new();
            for (idx, &arg_v) in cur_args.iter().enumerate() {
                if keep_mask[idx] {
                    let new_idx = new_args.len() as u32;
                    if let Some(v_mut) = module.values.get_mut(arg_v) {
                        v_mut.def = ValueDef::BlockArgument {
                            block: callee_blk,
                            index: new_idx,
                        };
                    }
                    new_args.push(arg_v);
                } else {
                    module.values.remove(arg_v);
                }
            }
            module.blocks[callee_blk].arguments = new_args;
            for &(_, call_id) in &callsites {
                if let Some(call_op) = module.ops.get_mut(call_id) {
                    let old_opds = call_op.operands.clone();
                    call_op.operands.clear();
                    for (idx, opd) in old_opds.into_iter().enumerate() {
                        if keep_mask[idx] {
                            call_op.operands.push(opd);
                        }
                    }
                }
            }
        }

        total_hoisted += hoisted_ops.len();
    }

    total_hoisted
}

/// Sentinel engine-group index (`-1`) assigned to cross-layer KV-sharing layers whose KV state
/// lives in their target owner's paged blocks.
pub const EXCLUDED_ENGINE_GROUP: i32 = -1;

/// Engine-neutral descriptor of one KV-cache group exposed by a hybrid serving engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineGroupInfoSpec {
    /// Dense 0-based engine group identifier (one distinct paged-block address space).
    pub engine_group_id: usize,
    /// Registered KV tensor layer indices assigned to this group.
    pub layer_indices: Vec<usize>,
    /// Logical tokens covered by one paged chunk (`0` = scratch or unreported group).
    pub tokens_per_block: usize,
    /// Sliding-window size in tokens (`-1` = full attention).
    pub sw_size_tokens: i64,
    /// Connector-private extra object-group tag (`0` = regular group, `> 0` = auxiliary group).
    pub extra_object_group_tag: u16,
    /// True if pages hold recurrent state snapshots (Mamba / GDN) rather than attention KV.
    pub recurrent_state: bool,
}

/// Returns the number of distinct engine paged-block address spaces (`max(engine_group_id) + 1`,
/// or `1` when `groups` is empty).
pub fn num_engine_groups(groups: &[EngineGroupInfoSpec]) -> usize {
    groups
        .iter()
        .map(|g| g.engine_group_id + 1)
        .max()
        .unwrap_or(1)
}

/// Unions and sorts `layer_indices` per `engine_group_id` across `groups`.
pub fn engine_group_layer_indices(groups: &[EngineGroupInfoSpec]) -> Vec<Vec<usize>> {
    if groups.is_empty() {
        return Vec::new();
    }
    let n = num_engine_groups(groups);
    let mut per_group = vec![Vec::new(); n];
    for g in groups {
        per_group[g.engine_group_id].extend_from_slice(&g.layer_indices);
    }
    for list in &mut per_group {
        list.sort_unstable();
        list.dedup();
    }
    per_group
}

/// Expands per-engine-group block ID lists `engine_side_block_ids` (indexed by `engine_group_id`)
/// into one block ID list per [`EngineGroupInfoSpec`] in protocol order.
pub fn expand_engine_block_ids(
    groups: &[EngineGroupInfoSpec],
    engine_side_block_ids: &[Vec<usize>],
) -> Option<Vec<Vec<usize>>> {
    if groups.is_empty() {
        return Some(vec![
            engine_side_block_ids.first().cloned().unwrap_or_default(),
        ]);
    }
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let block_ids = engine_side_block_ids.get(g.engine_group_id)?;
        out.push(block_ids.clone());
    }
    Some(out)
}

/// Slices each engine group's `allocated_block_ids` over the token interval
/// `[start_token_idx, end_token_idx)` using `group_tokens_per_block`.
///
/// Groups with `tokens_per_block == 0` (scratch groups) slice to an empty vector. Returns `None`
/// if `end_token_idx < start_token_idx`, if `start_token_idx` or `end_token_idx` is not a multiple
/// of `tokens_per_block`, or if the sliced block range exceeds a group's allocated block list.
pub fn slice_block_ids_per_group(
    allocated_block_ids: &IndexMap<usize, Vec<usize>>,
    group_tokens_per_block: &[usize],
    start_token_idx: usize,
    end_token_idx: usize,
) -> Option<Vec<Vec<usize>>> {
    if end_token_idx < start_token_idx {
        return None;
    }
    let mut sliced = Vec::with_capacity(group_tokens_per_block.len());
    for (group_idx, &tpb) in group_tokens_per_block.iter().enumerate() {
        if tpb == 0 {
            sliced.push(Vec::new());
            continue;
        }
        if !start_token_idx.is_multiple_of(tpb) || !end_token_idx.is_multiple_of(tpb) {
            return None;
        }
        let b_start = start_token_idx / tpb;
        let b_end = end_token_idx / tpb;
        let Some(group_blocks) = allocated_block_ids.get(&group_idx) else {
            if b_start == b_end {
                sliced.push(Vec::new());
                continue;
            }
            return None;
        };
        if b_end > group_blocks.len() {
            return None;
        }
        sliced.push(group_blocks[b_start..b_end].to_vec());
    }
    Some(sliced)
}

/// Maps each registered layer `0..num_registered_layers` to its `engine_group_id`, marking layers
/// not referenced by any group with [`EXCLUDED_ENGINE_GROUP`] (`-1`).
pub fn get_engine_group_indices(
    groups: &[EngineGroupInfoSpec],
    num_registered_layers: usize,
) -> Option<Vec<i32>> {
    if groups.is_empty() || num_registered_layers == 0 {
        return None;
    }
    let mut per_layer = vec![EXCLUDED_ENGINE_GROUP; num_registered_layers];
    for g in groups {
        let gid = i32::try_from(g.engine_group_id).ok()?;
        for &layer_idx in &g.layer_indices {
            if layer_idx >= num_registered_layers {
                return None;
            }
            per_layer[layer_idx] = gid;
        }
    }
    Some(per_layer)
}

/// Deterministic worker-slot router that binds each distinct `affinity_key` on first arrival to
/// `next_slot % num_workers` so all GPU requests from the same client execute sequentially on one
/// worker without per-instance locks.
#[derive(Debug, Clone)]
pub struct AffinitySlotRouter {
    num_workers: usize,
    key_to_slot: IndexMap<u64, usize>,
    next_slot: usize,
    overflow_bindings: usize,
}

impl AffinitySlotRouter {
    /// Creates a new [`AffinitySlotRouter`] with `num_workers` (`>= 1`) worker slots.
    pub fn new(num_workers: usize) -> Option<Self> {
        if num_workers == 0 {
            return None;
        }
        Some(Self {
            num_workers,
            key_to_slot: IndexMap::new(),
            next_slot: 0,
            overflow_bindings: 0,
        })
    }

    /// Returns `(worker_slot, is_newly_bound, is_overflow)` for `affinity_key`.
    pub fn slot_for_key(&mut self, affinity_key: u64) -> (usize, bool, bool) {
        if let Some(&slot) = self.key_to_slot.get(&affinity_key) {
            return (slot, false, false);
        }
        let slot = self.next_slot % self.num_workers;
        let is_overflow = self.next_slot >= self.num_workers;
        self.key_to_slot.insert(affinity_key, slot);
        self.next_slot += 1;
        if is_overflow {
            self.overflow_bindings += 1;
        }
        (slot, true, is_overflow)
    }

    /// Returns `(num_bound_keys, overflow_bindings)`.
    pub fn stats(&self) -> (usize, usize) {
        (self.key_to_slot.len(), self.overflow_bindings)
    }
}

/// Metadata for a registered multiprocess cache server instance in the coordinator fleet view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpInstanceSpec {
    /// Globally unique identifier of the MP server.
    pub instance_id: String,
    /// Reachable IP address of the MP server.
    pub ip: String,
    /// HTTP port of the MP server.
    pub http_port: u16,
    /// Monotonic-clock registration timestamp in milliseconds.
    pub registration_time_ms: u64,
    /// Monotonic-clock timestamp of the most recent heartbeat in milliseconds.
    pub last_heartbeat_time_ms: u64,
    /// Advertised peer-to-peer URL (empty when P2P is disabled).
    pub p2p_advertised_url: String,
    /// ZMQ message-queue port for P2P RPCs (`0` when P2P is disabled).
    pub mq_port: u16,
}

/// Cumulative membership-change counters tracked by [`FleetInstanceRegistry`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FleetMembershipCounters {
    /// New instance registrations.
    pub registered: u64,
    /// Re-registrations of an already-present `instance_id`.
    pub re_registered: u64,
    /// Explicit deregistrations.
    pub deregistered: u64,
    /// Instances removed due to heartbeat expiration.
    pub timed_out: u64,
    /// Heartbeats received from an unregistered `instance_id`.
    pub unknown_heartbeat: u64,
}

/// Monotonic-clock heartbeat registry of live MP cache servers in the fleet.
#[derive(Debug, Clone, Default)]
pub struct FleetInstanceRegistry {
    instances: IndexMap<String, MpInstanceSpec>,
    counters: FleetMembershipCounters,
}

impl FleetInstanceRegistry {
    /// Creates an empty [`FleetInstanceRegistry`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers or replaces `instance`, returning `true` if `instance_id` was already registered
    /// (a re-registration) or `false` on first registration.
    pub fn register(&mut self, instance: MpInstanceSpec) -> bool {
        let existed = self.instances.contains_key(&instance.instance_id);
        self.instances
            .insert(instance.instance_id.clone(), instance);
        if existed {
            self.counters.re_registered += 1;
        } else {
            self.counters.registered += 1;
        }
        existed
    }

    /// Removes `instance_id` and increments `deregistered` when found.
    pub fn deregister(&mut self, instance_id: &str) -> Option<MpInstanceSpec> {
        let removed = self.instances.shift_remove(instance_id);
        if removed.is_some() {
            self.counters.deregistered += 1;
        }
        removed
    }

    /// Updates `last_heartbeat_time_ms` for `instance_id`, returning `true` if found or `false`
    /// (and incrementing `unknown_heartbeat`) if `instance_id` is not registered.
    pub fn update_heartbeat(&mut self, instance_id: &str, timestamp_ms: u64) -> bool {
        let Some(inst) = self.instances.get_mut(instance_id) else {
            self.counters.unknown_heartbeat += 1;
            return false;
        };
        inst.last_heartbeat_time_ms = timestamp_ms;
        true
    }

    /// Removes and returns all `instance_id`s whose `now_ms - last_heartbeat_time_ms > timeout_ms`.
    pub fn remove_stale(&mut self, now_ms: u64, timeout_ms: u64) -> Vec<String> {
        let expired: Vec<String> = self
            .instances
            .iter()
            .filter_map(|(id, inst)| {
                (now_ms.saturating_sub(inst.last_heartbeat_time_ms) > timeout_ms)
                    .then_some(id.clone())
            })
            .collect();
        for id in &expired {
            self.instances.shift_remove(id);
        }
        self.counters.timed_out += expired.len() as u64;
        expired
    }

    /// Returns the number of currently registered servers.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns true if no servers are registered.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Returns the cumulative [`FleetMembershipCounters`].
    pub fn counters(&self) -> FleetMembershipCounters {
        self.counters
    }
}

/// 8-byte ASCII magic header (`LMCKPT\0\0`) for coordinator checkpoint files.
pub const COORDINATOR_CHECKPOINT_MAGIC: &[u8; 8] = b"LMCKPT\0\0";

/// Encodes a coordinator checkpoint binary envelope (`8-byte magic + little-endian u32 version +
/// payload`).
pub fn encode_coordinator_checkpoint_envelope(version: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + payload.len());
    out.extend_from_slice(COORDINATOR_CHECKPOINT_MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Validates and strips the 12-byte coordinator checkpoint header (`LMCKPT\0\0` and
/// `expected_version`), returning the non-empty payload slice.
pub fn decode_coordinator_checkpoint_envelope(buf: &[u8], expected_version: u32) -> Option<&[u8]> {
    if buf.len() <= 12 || &buf[0..8] != COORDINATOR_CHECKPOINT_MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().ok()?);
    if version != expected_version {
        return None;
    }
    Some(&buf[12..])
}

/// Per-`(topic, partition)` stream offset cursor captured atomically inside coordinator
/// checkpoints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KafkaStreamPositionTable {
    offsets: IndexMap<(String, u32), u64>,
}

impl KafkaStreamPositionTable {
    /// Creates an empty [`KafkaStreamPositionTable`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the offset of a batch that the coordinator gate has just admitted.
    pub fn record(&mut self, topic: &str, partition: u32, offset: u64) {
        self.offsets.insert((topic.to_string(), partition), offset);
    }

    /// Returns `Some(last_offset + 1)` to resume `(topic, partition)` from, or `None` if no
    /// offset has been recorded for that partition.
    pub fn next_offset(&self, topic: &str, partition: u32) -> Option<u64> {
        self.offsets
            .get(&(topic.to_string(), partition))
            .map(|&off| off + 1)
    }

    /// Captures all recorded `(topic, partition, offset)` triples for checkpoint serialization.
    pub fn capture_offsets(&self) -> Vec<(String, u32, u64)> {
        self.offsets
            .iter()
            .map(|((t, p), &off)| (t.clone(), *p, off))
            .collect()
    }

    /// Replaces the table with a checkpointed snapshot of `(topic, partition, offset)` triples.
    pub fn restore_offsets(&mut self, entries: &[(String, u32, u64)]) {
        self.offsets.clear();
        for (topic, partition, offset) in entries {
            self.offsets.insert((topic.clone(), *partition), *offset);
        }
    }
}

/// Coordinator ingest quiesce state machine that parks new cache-event batches while waiting for
/// in-flight batches to drain before capturing a consistent multi-view checkpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuiesceStateMachine {
    quiesce_requested: bool,
    in_flight: usize,
}

impl QuiesceStateMachine {
    /// Creates a new idle [`QuiesceStateMachine`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempts to start applying an ingest batch. Returns `false` (parking the caller) when a
    /// checkpoint quiesce is currently requested.
    pub fn try_begin_apply(&mut self) -> bool {
        if self.quiesce_requested {
            return false;
        }
        self.in_flight += 1;
        true
    }

    /// Completes one in-flight ingest batch and returns `true` if a pending quiesce has now
    /// reached `in_flight == 0` (ready for consistent capture).
    pub fn finish_apply(&mut self) -> bool {
        self.in_flight = self.in_flight.saturating_sub(1);
        self.quiesce_requested && self.in_flight == 0
    }

    /// Requests ingest quiescence and returns `true` if `in_flight` is already `0`.
    pub fn request_quiesce(&mut self) -> bool {
        self.quiesce_requested = true;
        self.in_flight == 0
    }

    /// Returns `true` when `quiesce_requested` is active and `in_flight == 0`.
    pub fn is_quiesced(&self) -> bool {
        self.quiesce_requested && self.in_flight == 0
    }

    /// Releases a completed quiesce so parked ingest batches can resume.
    pub fn release_quiesce(&mut self) {
        self.quiesce_requested = false;
    }

    /// Aborts a pending quiesce on timeout without stalling ingest further, returning the number
    /// of batches that were still in flight.
    pub fn abort_quiesce_on_timeout(&mut self) -> usize {
        self.quiesce_requested = false;
        self.in_flight
    }

    /// Returns `(quiesce_requested, in_flight)`.
    pub fn state(&self) -> (bool, usize) {
        (self.quiesce_requested, self.in_flight)
    }
}

/// Elides redundant mutable buffer store-load and store-store pairs on the same buffer and
/// effect chain (`mo.mutable.store` / `mo.mutable.load`).
///
/// - **Redundant load elision:** When `mo.mutable.load(in_chain, buffer) -> (out_tensor,
///   out_chain)` has exclusive access (`hasOneUse`) to `in_chain` produced by
///   `mo.mutable.store(prev_chain, in_tensor, buffer) -> in_chain` on the same `buffer` and
///   device, forwards `out_tensor` to `in_tensor` and `out_chain` to `in_chain`.
/// - **Redundant store elision:** When `curr = mo.mutable.store(in_chain, val1, buffer) ->
///   mid_chain` has `mid_chain` uniquely consumed by `next = mo.mutable.store(mid_chain, val2,
///   buffer)` on the same `buffer` and device, forwards `mid_chain` to `in_chain` and erases the
///   overwritten `curr` store.
///
/// Runs to a fixed point per basic block so cascading `store -> load -> store` chains collapse in
/// a single pass. Returns the total number of mutable operations elided.
pub fn elide_redundant_mutable_buffer_ops(module: &mut Module) -> usize {
    let mut total_elided = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        loop {
            let mut changed = false;
            let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
            let mut unique_user: IndexMap<ValueId, OpId> = IndexMap::new();
            for &op_id in &module.blocks[block_id].ops {
                if let Some(op) = module.ops.get(op_id) {
                    for &v in &op.operands {
                        *use_counts.entry(v).or_insert(0) += 1;
                        unique_user.insert(v, op_id);
                    }
                }
            }

            let op_ids = module.blocks[block_id].ops.clone();
            let mut erased_op: Option<OpId> = None;
            let mut replacements: Vec<(ValueId, ValueId)> = Vec::new();

            for op_id in op_ids {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let name = op.name.to_string();

                if name == "mo.mutable.load" && op.operands.len() == 2 && op.results.len() == 2 {
                    let in_chain = op.operands[0];
                    let in_buf = op.operands[1];
                    if use_counts.get(&in_chain).copied().unwrap_or(0) != 1 {
                        continue;
                    }
                    let Some(chain_info) = module.values.get(in_chain) else {
                        continue;
                    };
                    let ValueDef::OpResult { op: store_id, .. } = chain_info.def else {
                        continue;
                    };
                    if !module.blocks[block_id].ops.contains(&store_id) {
                        continue;
                    }
                    let Some(store_op) = module.ops.get(store_id) else {
                        continue;
                    };
                    if store_op.name.to_string() == "mo.mutable.store"
                        && store_op.operands.len() == 3
                        && store_op.results.len() == 1
                        && store_op.operands[2] == in_buf
                        && store_op.attributes.get("castforge.device")
                            == op.attributes.get("castforge.device")
                    {
                        replacements.push((op.results[0], store_op.operands[1]));
                        replacements.push((op.results[1], store_op.results[0]));
                        erased_op = Some(op_id);
                        break;
                    }
                } else if name == "mo.mutable.store"
                    && op.operands.len() == 3
                    && op.results.len() == 1
                {
                    let out_chain = op.results[0];
                    if use_counts.get(&out_chain).copied().unwrap_or(0) != 1 {
                        continue;
                    }
                    let Some(&next_id) = unique_user.get(&out_chain) else {
                        continue;
                    };
                    if !module.blocks[block_id].ops.contains(&next_id) {
                        continue;
                    }
                    let Some(next_op) = module.ops.get(next_id) else {
                        continue;
                    };
                    if next_op.name.to_string() == "mo.mutable.store"
                        && next_op.operands.len() == 3
                        && next_op.operands[0] == out_chain
                        && next_op.operands[2] == op.operands[2]
                        && next_op.attributes.get("castforge.device")
                            == op.attributes.get("castforge.device")
                    {
                        replacements.push((out_chain, op.operands[0]));
                        erased_op = Some(op_id);
                        break;
                    }
                }
            }

            if let Some(dead_id) = erased_op {
                for &rem_id in &module.blocks[block_id].ops {
                    if rem_id == dead_id {
                        continue;
                    }
                    if let Some(rem_op) = module.ops.get_mut(rem_id) {
                        for opd in &mut rem_op.operands {
                            for &(old_v, new_v) in &replacements {
                                if *opd == old_v {
                                    *opd = new_v;
                                }
                            }
                        }
                    }
                }
                module.blocks[block_id].ops.retain(|&id| id != dead_id);
                module.ops.remove(dead_id);
                for (old_v, _) in replacements {
                    module.values.remove(old_v);
                }
                total_elided += 1;
                changed = true;
            }

            if !changed {
                break;
            }
        }
    }

    total_elided
}

fn ops_match_ignoring_device(a: &Operation, b: &Operation) -> bool {
    if a.name != b.name
        || a.operands.len() != b.operands.len()
        || a.results.len() != b.results.len()
    {
        return false;
    }
    let filtered = |attrs: &IndexMap<String, Attribute>| -> IndexMap<String, Attribute> {
        attrs
            .iter()
            .filter(|(k, _)| k.as_str() != "castforge.device")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    };
    filtered(&a.attributes) == filtered(&b.attributes)
}

/// Coalesces $N \ge 2$ structurally identical, distinct sole-use consumers of an
/// `mo.tensor.unbundle` (or `mo.unbundle`) operation into a single `mo.parallel` launch.
///
/// Verifies that:
/// 1. Every result `unbundle#i` (`i = 0..N`) has exactly 1 use (`consumers[i]`), and all
///    `consumers` are distinct pure tensor operations in the same block.
/// 2. All `consumers` match structurally (`ops_match_ignoring_device`) and reference `unbundle#i`
///    at the exact same driving operand index `driving_idx`.
/// 3. For every non-driving operand index `k != driving_idx`, the per-launch operands have
///    matching tensor shapes and dtypes and are not a single shared outer value across all
///    launches.
///
/// Returns the number of `mo.parallel` launches created.
pub fn coalesce_unbundle_consumers_to_parallel(module: &mut Module) -> usize {
    let mut created = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut single_user: IndexMap<ValueId, (OpId, usize)> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    let entry = single_user.entry(v).or_insert((op_id, 0));
                    entry.0 = op_id;
                    entry.1 += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for unbundle_id in op_ids {
            if to_remove.contains(&unbundle_id) {
                continue;
            }
            let Some(unbundle_op) = module.ops.get(unbundle_id).cloned() else {
                continue;
            };
            if !matches!(
                unbundle_op.name.to_string().as_str(),
                "mo.tensor.unbundle" | "mo.unbundle"
            ) || unbundle_op.operands.len() != 1
                || unbundle_op.results.len() < 2
            {
                continue;
            }
            let n_launches = unbundle_op.results.len();
            let mut consumers: Vec<OpId> = Vec::with_capacity(n_launches);
            let mut unique_set: IndexSet<OpId> = IndexSet::new();
            let mut valid = true;

            for &res_v in &unbundle_op.results {
                let Some(&(cid, count)) = single_user.get(&res_v) else {
                    valid = false;
                    break;
                };
                if count != 1
                    || to_remove.contains(&cid)
                    || !module.blocks[block_id].ops.contains(&cid)
                    || !unique_set.insert(cid)
                {
                    valid = false;
                    break;
                }
                consumers.push(cid);
            }
            if !valid {
                continue;
            }

            let rep_op = module.ops[consumers[0]].clone();
            if rep_op.results.len() != 1
                || !rep_op.regions.is_empty()
                || !(rep_op.is_pure() || rep_op.name.dialect == "mo")
                || rep_op.is_op("func", "return")
                || rep_op.is_op("func", "call")
            {
                continue;
            }
            let Some(driving_idx) = rep_op
                .operands
                .iter()
                .position(|&v| v == unbundle_op.results[0])
            else {
                continue;
            };

            let Some(rep_res_tt) = module
                .values
                .get(rep_op.results[0])
                .and_then(|v| v.ty.as_tensor())
                .cloned()
            else {
                continue;
            };

            for (i, &peer_id) in consumers.iter().enumerate().skip(1) {
                let peer_op = &module.ops[peer_id];
                if !ops_match_ignoring_device(&rep_op, peer_op)
                    || peer_op.operands[driving_idx] != unbundle_op.results[i]
                {
                    valid = false;
                    break;
                }
                let Some(peer_res_tt) = module
                    .values
                    .get(peer_op.results[0])
                    .and_then(|v| v.ty.as_tensor())
                else {
                    valid = false;
                    break;
                };
                if peer_res_tt.shape != rep_res_tt.shape || peer_res_tt.dtype != rep_res_tt.dtype {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }

            // Validate non-driving operand positions across launches.
            let mut merged_operands: SmallVec<[ValueId; 4]> =
                SmallVec::from_slice(&[unbundle_op.operands[0]]);
            for opd_idx in 0..rep_op.operands.len() {
                if opd_idx == driving_idx {
                    continue;
                }
                let first_v = module.ops[consumers[0]].operands[opd_idx];
                let Some(first_tt) = module.values.get(first_v).and_then(|v| v.ty.as_tensor())
                else {
                    valid = false;
                    break;
                };
                let all_equal = consumers
                    .iter()
                    .all(|&cid| module.ops[cid].operands[opd_idx] == first_v);
                if all_equal {
                    valid = false;
                    break;
                }
                for &cid in &consumers {
                    let v = module.ops[cid].operands[opd_idx];
                    let Some(tt) = module.values.get(v).and_then(|val| val.ty.as_tensor()) else {
                        valid = false;
                        break;
                    };
                    if tt.shape != first_tt.shape || tt.dtype != first_tt.dtype {
                        valid = false;
                        break;
                    }
                    merged_operands.push(v);
                }
                if !valid {
                    break;
                }
            }
            if !valid {
                continue;
            }

            let parallel_id = consumers[0];
            let mut parallel_results = SmallVec::with_capacity(n_launches);
            let mut locs = vec![unbundle_op.location.clone()];
            for (idx, &cid) in consumers.iter().enumerate() {
                let res_v = module.ops[cid].results[0];
                locs.push(module.ops[cid].location.clone());
                if let Some(v_mut) = module.values.get_mut(res_v) {
                    v_mut.def = ValueDef::OpResult {
                        op: parallel_id,
                        index: idx as u32,
                    };
                }
                parallel_results.push(res_v);
            }

            if let Some(par_op) = module.ops.get_mut(parallel_id) {
                let wrapped_name = par_op.name.to_string();
                par_op.name = OpName::new("mo", "parallel");
                par_op.operands = merged_operands;
                par_op.results = parallel_results;
                par_op
                    .attributes
                    .insert("wrapped_op".into(), Attribute::String(wrapped_name));
                par_op.attributes.insert(
                    "parallel_launch_count".into(),
                    Attribute::Int(n_launches as i64),
                );
                par_op.attributes.insert(
                    "driving_operand_index".into(),
                    Attribute::Int(driving_idx as i64),
                );
                stamp_fused_provenance(par_op, &locs);
            }

            for &dead_v in &unbundle_op.results {
                module.values.remove(dead_v);
            }
            to_remove.insert(unbundle_id);
            for &dead_cid in &consumers[1..] {
                to_remove.insert(dead_cid);
            }
            created += 1;
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

    created
}
