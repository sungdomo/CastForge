//! RDMA L1 arena window resolution with revoked-region retry, three-phase asynchronous
//! engine-driven KV store coordination, and runtime L2 adapter lifecycle management.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::ir::mlir::DType;

/// Sentinel return code when an RDMA `get_into` misses or returns a partial read.
pub const RDMA_GET_MISS: i64 = -1;

/// Validated window slice inside the registered L1 RDMA memory arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RdmaWindowSpec {
    /// Byte offset from `l1_base`.
    pub arena_offset: usize,
    /// Window length in bytes.
    pub length: usize,
    /// Active arena registration generation when this window was resolved.
    pub registration_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RdmaStoredEntry {
    payload: Vec<u8>,
    expires_at_sec: Option<u64>,
}

/// RDMA L1 arena window resolver and transport pool with single-retry recovery
/// on `"revoked"` memory region errors.
#[derive(Debug, Clone)]
pub struct RdmaArenaWindowPool {
    l1_base: u64,
    l1_size: usize,
    rdma_provider: String,
    rdma_interface: Option<String>,
    registration_epoch: u64,
    remote_store: HashMap<String, RdmaStoredEntry>,
    pending_revoked_faults: usize,
}

impl RdmaArenaWindowPool {
    /// Creates a new RDMA L1 arena window pool after validating the arena base,
    /// size, fabric provider (`"efa-direct"` or `"tcp"`), and optional network
    /// interface name.
    pub fn new(
        l1_base: u64,
        l1_size: usize,
        rdma_provider: &str,
        rdma_interface: Option<String>,
    ) -> Result<Self, String> {
        if l1_base == 0 {
            return Err("RDMA L1 arena base address must be non-zero".to_string());
        }
        if l1_size == 0 {
            return Err("RDMA L1 arena size must be greater than zero".to_string());
        }
        if rdma_provider != "efa-direct" && rdma_provider != "tcp" {
            return Err(format!(
                "unsupported RDMA provider '{rdma_provider}'; expected 'efa-direct' or 'tcp'"
            ));
        }
        if let Some(iface) = &rdma_interface
            && iface.trim().is_empty()
        {
            return Err("RDMA interface name must not be empty when specified".to_string());
        }
        Ok(Self {
            l1_base,
            l1_size,
            rdma_provider: rdma_provider.to_string(),
            rdma_interface,
            registration_epoch: 1,
            remote_store: HashMap::new(),
            pending_revoked_faults: 0,
        })
    }

    /// Returns the active RDMA provider name.
    #[must_use]
    pub fn rdma_provider(&self) -> &str {
        &self.rdma_provider
    }

    /// Returns the optional RDMA network interface name.
    #[must_use]
    pub fn rdma_interface(&self) -> Option<&str> {
        self.rdma_interface.as_deref()
    }

    /// Returns the current L1 arena registration generation counter.
    #[must_use]
    pub const fn registration_epoch(&self) -> u64 {
        self.registration_epoch
    }

    /// Injects `count` simulated `"revoked"` memory-registration faults to test
    /// automatic arena re-registration and single-retry recovery.
    pub fn inject_revoked_faults(&mut self, count: usize) {
        self.pending_revoked_faults += count;
    }

    /// Resolves a buffer `[buf_addr, buf_addr + buf_len)` into an
    /// [`RdmaWindowSpec`] inside `[l1_base, l1_base + l1_size)`.
    pub fn resolve_window(
        &self,
        buf_addr: u64,
        buf_len: usize,
        readonly: bool,
    ) -> Result<RdmaWindowSpec, String> {
        if readonly {
            return Err("RDMA L1 arena window requires a writable buffer".to_string());
        }
        if buf_len == 0 {
            return Err("RDMA buffer length must be non-zero".to_string());
        }
        let arena_end = self
            .l1_base
            .checked_add(self.l1_size as u64)
            .ok_or_else(|| "RDMA arena address overflow".to_string())?;
        let buf_end = buf_addr
            .checked_add(buf_len as u64)
            .ok_or_else(|| "RDMA buffer address overflow".to_string())?;
        if buf_addr < self.l1_base || buf_end > arena_end {
            return Err(format!(
                "buffer [0x{buf_addr:x}, 0x{buf_end:x}) lies outside L1 arena \
                 [0x{:x}, 0x{arena_end:x})",
                self.l1_base
            ));
        }
        Ok(RdmaWindowSpec {
            arena_offset: (buf_addr - self.l1_base) as usize,
            length: buf_len,
            registration_epoch: self.registration_epoch,
        })
    }

    fn run_with_revoked_retry<T, F>(&mut self, mut op: F) -> Result<T, String>
    where
        F: FnMut(&mut Self) -> Result<T, String>,
    {
        match op(self) {
            Ok(val) => Ok(val),
            Err(first_err) if first_err.contains("revoked") => {
                self.registration_epoch += 1;
                op(self)
            }
            Err(other_err) => Err(other_err),
        }
    }

    /// Stores `payload` under `key` via the RDMA worker pool, rejecting empty
    /// 0-byte values and applying an optional `ttl_seconds` relative to `now_sec`.
    pub fn rdma_set(
        &mut self,
        key: &str,
        payload: &[u8],
        now_sec: u64,
        ttl_seconds: Option<u64>,
    ) -> Result<(), String> {
        if payload.is_empty() {
            return Err("RDMA set rejects empty 0-byte payload".to_string());
        }
        let data = payload.to_vec();
        let key_owned = key.to_string();
        self.run_with_revoked_retry(|pool| {
            if pool.pending_revoked_faults > 0 {
                pool.pending_revoked_faults -= 1;
                return Err("RDMA fabric error: local memory key revoked".to_string());
            }
            let expires_at_sec = ttl_seconds.map(|ttl| now_sec.saturating_add(ttl));
            pool.remote_store.insert(
                key_owned.clone(),
                RdmaStoredEntry {
                    payload: data.clone(),
                    expires_at_sec,
                },
            );
            Ok(())
        })
    }

    /// Fetches `key` directly into a resolved L1 arena window
    /// `[buf_addr, buf_addr + expected_size)`.
    pub fn rdma_get_into(
        &mut self,
        key: &str,
        buf_addr: u64,
        expected_size: usize,
        readonly: bool,
        now_sec: u64,
        out_buf: &mut [u8],
    ) -> Result<i64, String> {
        let _window = self.resolve_window(buf_addr, expected_size, readonly)?;
        if out_buf.len() < expected_size {
            return Err("destination slice smaller than expected_size".to_string());
        }
        let key_owned = key.to_string();
        self.run_with_revoked_retry(|pool| {
            if pool.pending_revoked_faults > 0 {
                pool.pending_revoked_faults -= 1;
                return Err("RDMA fabric error: remote access key revoked".to_string());
            }
            let Some(entry) = pool.remote_store.get(&key_owned) else {
                return Ok(RDMA_GET_MISS);
            };
            if let Some(exp) = entry.expires_at_sec
                && now_sec >= exp
            {
                return Ok(RDMA_GET_MISS);
            }
            if entry.payload.len() != expected_size {
                return Ok(RDMA_GET_MISS);
            }
            out_buf[..expected_size].copy_from_slice(&entry.payload);
            Ok(expected_size as i64)
        })
    }
}

/// Key identifying a homogeneous pool of pinned CPU staging buffers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StagingBufferSignature {
    /// Tensor dimensions of the staging buffer.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DType,
}

impl StagingBufferSignature {
    /// Computes the total byte size of a buffer with this signature.
    #[must_use]
    pub fn size_in_bytes(&self) -> usize {
        let numel: usize = self.shape.iter().copied().product();
        let elem_bytes = self.dtype.bit_width().div_ceil(8);
        numel.saturating_mul(elem_bytes)
    }
}

/// Pinned CPU staging buffer handle allocated from [`PinnedStagingBufferPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedStagingBuffer {
    /// Unique buffer handle ID.
    pub buffer_id: u64,
    /// Shape and element type signature.
    pub signature: StagingBufferSignature,
    /// Backing byte capacity.
    pub byte_len: usize,
}

/// Pool of reusable `(shape, dtype)` pinned CPU staging buffers.
#[derive(Debug, Default)]
pub struct PinnedStagingBufferPool {
    free_lists: HashMap<StagingBufferSignature, VecDeque<PinnedStagingBuffer>>,
    next_buffer_id: u64,
    total_allocated: usize,
}

impl PinnedStagingBufferPool {
    /// Creates an empty pinned staging buffer pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates or reuses a pinned CPU staging buffer matching `signature`.
    pub fn alloc_staging(&mut self, signature: StagingBufferSignature) -> PinnedStagingBuffer {
        if let Some(queue) = self.free_lists.get_mut(&signature)
            && let Some(buf) = queue.pop_front()
        {
            return buf;
        }
        self.next_buffer_id += 1;
        self.total_allocated += 1;
        let byte_len = signature.size_in_bytes();
        PinnedStagingBuffer {
            buffer_id: self.next_buffer_id,
            signature,
            byte_len,
        }
    }

    /// Releases a pinned staging buffer back to its `(shape, dtype)` free list.
    pub fn release_staging(&mut self, buffer: PinnedStagingBuffer) {
        self.free_lists
            .entry(buffer.signature.clone())
            .or_default()
            .push_back(buffer);
    }

    /// Returns the total number of distinct host allocations created so far.
    #[must_use]
    pub const fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Returns the number of currently pooled (available) buffers for `signature`.
    #[must_use]
    pub fn pooled_count(&self, signature: &StagingBufferSignature) -> usize {
        self.free_lists.get(signature).map_or(0, VecDeque::len)
    }

    /// Clears all free lists when the coordinator closes.
    pub fn clear(&mut self) {
        self.free_lists.clear();
    }
}

/// Lifecycle phase of an asynchronous engine-driven KV store request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsyncStorePhase {
    /// Phase 1: Submitted on the engine thread; worker has not yet recorded the
    /// GPU gather event (`pending_stores`).
    PendingGatherLaunch {
        /// Staging buffer reserved for this transfer.
        staging_buffer: PinnedStagingBuffer,
    },
    /// Phase 2: GPU gather kernel launched and `gather_event_id` recorded in
    /// `inflight_gather_events`; DMA copy to staging buffer is in progress.
    GatherInflight {
        /// Staging buffer receiving the gathered KV pages.
        staging_buffer: PinnedStagingBuffer,
        /// Device event ID recorded after launching the gather kernel.
        gather_event_id: u64,
    },
    /// Phase 3: Gather event synchronized, L1/L2 commit completed, and staging
    /// buffer returned to the pool.
    Committed,
}

/// Coordinator for three-phase asynchronous engine-driven KV stores that closes
/// the preemption race window between store submission and worker gather launch.
#[derive(Debug, Default)]
pub struct AsyncEngineDrivenStoreCoordinator {
    staging_pool: PinnedStagingBufferPool,
    store_phases: BTreeMap<u64, AsyncStorePhase>,
    pending_stores: BTreeSet<u64>,
    inflight_gather_events: BTreeMap<u64, u64>,
    synchronized_events: Vec<u64>,
    closed: bool,
}

impl AsyncEngineDrivenStoreCoordinator {
    /// Creates a new asynchronous engine-driven store coordinator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Phase 1 (`submit_store`): Allocates a pinned staging buffer and registers
    /// `store_id` in `pending_stores` BEFORE the worker thread launches the GPU
    /// gather kernel.
    pub fn submit_store(
        &mut self,
        store_id: u64,
        signature: StagingBufferSignature,
    ) -> Result<u64, String> {
        if self.closed {
            return Err("cannot submit store on a closed coordinator".to_string());
        }
        if self.store_phases.contains_key(&store_id) {
            return Err(format!("duplicate store_id {store_id}"));
        }
        let buf = self.staging_pool.alloc_staging(signature);
        let buf_id = buf.buffer_id;
        self.pending_stores.insert(store_id);
        self.store_phases.insert(
            store_id,
            AsyncStorePhase::PendingGatherLaunch {
                staging_buffer: buf,
            },
        );
        Ok(buf_id)
    }

    /// Phase 2 (`record_gather_launched`): Records that the worker has launched
    /// the device gather kernel with `gather_event_id`, atomically moving
    /// `store_id` from `pending_stores` into `inflight_gather_events`.
    pub fn record_gather_launched(
        &mut self,
        store_id: u64,
        gather_event_id: u64,
    ) -> Result<(), String> {
        let Some(phase) = self.store_phases.remove(&store_id) else {
            return Err(format!("unknown store_id {store_id}"));
        };
        let AsyncStorePhase::PendingGatherLaunch { staging_buffer } = phase else {
            self.store_phases.insert(store_id, phase);
            return Err(format!(
                "store_id {store_id} is not in PendingGatherLaunch phase"
            ));
        };
        self.pending_stores.remove(&store_id);
        self.inflight_gather_events
            .insert(store_id, gather_event_id);
        self.store_phases.insert(
            store_id,
            AsyncStorePhase::GatherInflight {
                staging_buffer,
                gather_event_id,
            },
        );
        Ok(())
    }

    /// Phase 3 (`complete_commit`): Synchronizes the gather event, commits the
    /// staged KV chunks, and releases the pinned staging buffer back to the pool.
    pub fn complete_commit(&mut self, store_id: u64) -> Result<(), String> {
        let Some(phase) = self.store_phases.remove(&store_id) else {
            return Err(format!("unknown store_id {store_id}"));
        };
        let AsyncStorePhase::GatherInflight {
            staging_buffer,
            gather_event_id,
        } = phase
        else {
            self.store_phases.insert(store_id, phase);
            return Err(format!(
                "store_id {store_id} is not in GatherInflight phase"
            ));
        };
        self.inflight_gather_events.remove(&store_id);
        self.synchronized_events.push(gather_event_id);
        self.staging_pool.release_staging(staging_buffer);
        self.store_phases
            .insert(store_id, AsyncStorePhase::Committed);
        Ok(())
    }

    /// Flushes all in-flight stores before a scheduler preemption or KV page
    /// eviction can overwrite source GPU pages.
    pub fn flush_inflight_stores(
        &mut self,
        mut next_synthetic_event: u64,
    ) -> Result<Vec<u64>, String> {
        let pending_ids: Vec<u64> = self.pending_stores.iter().copied().collect();
        for store_id in pending_ids {
            self.record_gather_launched(store_id, next_synthetic_event)?;
            next_synthetic_event += 1;
        }

        let inflight_ids: Vec<(u64, u64)> = self
            .inflight_gather_events
            .iter()
            .map(|(&sid, &ev)| (sid, ev))
            .collect();
        let mut flushed_events = Vec::with_capacity(inflight_ids.len());
        for (store_id, ev_id) in inflight_ids {
            self.complete_commit(store_id)?;
            flushed_events.push(ev_id);
        }
        Ok(flushed_events)
    }

    /// Closes the coordinator, flushing any remaining stores and clearing the
    /// pinned staging buffer pool.
    pub fn close(&mut self) -> Result<(), String> {
        let _ = self.flush_inflight_stores(1_000_000)?;
        self.staging_pool.clear();
        self.closed = true;
        Ok(())
    }

    /// Accesses the underlying [`PinnedStagingBufferPool`].
    #[must_use]
    pub const fn staging_pool(&self) -> &PinnedStagingBufferPool {
        &self.staging_pool
    }
}

/// Physical memory range owned by an L1/L2 adapter registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalMemoryRange {
    /// Device index (e.g., GPU ordinal or NUMA node).
    pub device_id: u32,
    /// Starting physical/virtual base address.
    pub base_addr: u64,
    /// Byte length of the registered region.
    pub byte_len: u64,
}

impl PhysicalMemoryRange {
    /// Returns `true` if `self` overlaps with `other` on the same `device_id`.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        if self.device_id != other.device_id || self.byte_len == 0 || other.byte_len == 0 {
            return false;
        }
        let self_end = self.base_addr.saturating_add(self.byte_len);
        let other_end = other.base_addr.saturating_add(other.byte_len);
        self.base_addr < other_end && other.base_addr < self_end
    }
}

/// Lifecycle state of an online L2 storage adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterLifecycleState {
    /// Active and accepting new store/lookup/prefetch tasks.
    Active,
    /// Draining in-flight tasks prior to removal; rejects new tasks.
    Draining,
    /// Fully removed after in-flight tasks reached zero.
    Removed,
}

/// Runtime descriptor for a managed L2 storage adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedAdapterDescriptor {
    /// Unique adapter identifier.
    pub adapter_id: String,
    /// Current lifecycle state.
    pub state: AdapterLifecycleState,
    /// Number of in-flight store/lookup tasks currently executing on this adapter.
    pub inflight_tasks: usize,
    /// Physical memory range owned by this adapter.
    pub owned_range: PhysicalMemoryRange,
}

/// Controller managing online L2 adapter addition, graceful drain removal, and
/// non-overlapping physical memory reconfiguration.
#[derive(Debug, Default)]
pub struct AdapterLifecycleController {
    adapters: BTreeMap<String, ManagedAdapterDescriptor>,
}

impl AdapterLifecycleController {
    /// Creates an empty adapter lifecycle controller.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks whether `candidate` overlaps with any active/draining adapter's
    /// owned memory range (excluding `exclude_adapter_id` when reconfiguring an
    /// existing adapter in place).
    #[must_use]
    pub fn has_overlapping_range(
        &self,
        candidate: &PhysicalMemoryRange,
        exclude_adapter_id: Option<&str>,
    ) -> bool {
        self.adapters.values().any(|desc| {
            desc.state != AdapterLifecycleState::Removed
                && exclude_adapter_id != Some(desc.adapter_id.as_str())
                && desc.owned_range.overlaps(candidate)
        })
    }

    /// Registers a new online L2 adapter (`AddAdapter`), validating that its
    /// physical memory range has non-zero length and does not overlap any
    /// existing non-removed adapter.
    pub fn add_adapter(
        &mut self,
        adapter_id: &str,
        owned_range: PhysicalMemoryRange,
    ) -> Result<(), String> {
        if owned_range.byte_len == 0 {
            return Err("adapter owned physical memory range must be non-empty".to_string());
        }
        if let Some(existing) = self.adapters.get(adapter_id)
            && existing.state != AdapterLifecycleState::Removed
        {
            return Err(format!("adapter '{adapter_id}' is already registered"));
        }
        if self.has_overlapping_range(&owned_range, Some(adapter_id)) {
            return Err(format!(
                "adapter '{adapter_id}' physical range overlaps an existing registration"
            ));
        }
        self.adapters.insert(
            adapter_id.to_string(),
            ManagedAdapterDescriptor {
                adapter_id: adapter_id.to_string(),
                state: AdapterLifecycleState::Active,
                inflight_tasks: 0,
                owned_range,
            },
        );
        Ok(())
    }

    /// Reconfigures an existing active adapter's `owned_range` online, verifying
    /// that the new range does not overlap any other registered adapter.
    pub fn reconfigure_adapter(
        &mut self,
        adapter_id: &str,
        new_range: PhysicalMemoryRange,
    ) -> Result<(), String> {
        if new_range.byte_len == 0 {
            return Err("reconfigured physical range must be non-empty".to_string());
        }
        if self.has_overlapping_range(&new_range, Some(adapter_id)) {
            return Err(format!(
                "reconfigured range for '{adapter_id}' overlaps another adapter"
            ));
        }
        let Some(desc) = self.adapters.get_mut(adapter_id) else {
            return Err(format!("unknown adapter '{adapter_id}'"));
        };
        if desc.state != AdapterLifecycleState::Active {
            return Err(format!(
                "cannot reconfigure adapter '{adapter_id}' in state {:?}",
                desc.state
            ));
        }
        desc.owned_range = new_range;
        Ok(())
    }

    /// Starts a new task on `adapter_id` if and only if it is `Active`.
    pub fn acquire_task(&mut self, adapter_id: &str) -> Result<(), String> {
        let Some(desc) = self.adapters.get_mut(adapter_id) else {
            return Err(format!("unknown adapter '{adapter_id}'"));
        };
        if desc.state != AdapterLifecycleState::Active {
            return Err(format!(
                "adapter '{adapter_id}' is {:?} and cannot accept new tasks",
                desc.state
            ));
        }
        desc.inflight_tasks += 1;
        Ok(())
    }

    /// Completes one in-flight task on `adapter_id`; if the adapter is
    /// `Draining` and `inflight_tasks` reaches `0`, transitions it to `Removed`.
    pub fn release_task(&mut self, adapter_id: &str) -> Result<AdapterLifecycleState, String> {
        let Some(desc) = self.adapters.get_mut(adapter_id) else {
            return Err(format!("unknown adapter '{adapter_id}'"));
        };
        if desc.inflight_tasks == 0 {
            return Err(format!("adapter '{adapter_id}' has no in-flight tasks"));
        }
        desc.inflight_tasks -= 1;
        if desc.state == AdapterLifecycleState::Draining && desc.inflight_tasks == 0 {
            desc.state = AdapterLifecycleState::Removed;
        }
        Ok(desc.state)
    }

    /// Initiates graceful removal (`RemoveAdapter`): transitions `adapter_id` to
    /// `Draining` so no new tasks are routed to it, or immediately to `Removed`
    /// if `inflight_tasks == 0`.
    pub fn remove_adapter(&mut self, adapter_id: &str) -> Result<AdapterLifecycleState, String> {
        let Some(desc) = self.adapters.get_mut(adapter_id) else {
            return Err(format!("unknown adapter '{adapter_id}'"));
        };
        if desc.inflight_tasks == 0 {
            desc.state = AdapterLifecycleState::Removed;
        } else {
            desc.state = AdapterLifecycleState::Draining;
        }
        Ok(desc.state)
    }

    /// Returns the descriptor for `adapter_id`.
    #[must_use]
    pub fn get_adapter(&self, adapter_id: &str) -> Option<&ManagedAdapterDescriptor> {
        self.adapters.get(adapter_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdma_window_and_revoked_retry() {
        let mut pool =
            RdmaArenaWindowPool::new(0x1000_0000, 4096, "efa-direct", Some("efa0".to_string()))
                .expect("valid pool");
        assert!(pool.resolve_window(0x1000_0000, 256, true).is_err());
        assert!(pool.resolve_window(0x2000_0000, 256, false).is_err());

        pool.inject_revoked_faults(1);
        assert!(
            pool.rdma_set("chunk_0", &[1, 2, 3, 4], 10, Some(60))
                .is_ok()
        );
        assert_eq!(pool.registration_epoch(), 2);

        let mut out = [0u8; 4];
        let n = pool
            .rdma_get_into("chunk_0", 0x1000_0100, 4, false, 20, &mut out)
            .expect("get_into ok");
        assert_eq!(n, 4);
        assert_eq!(out, [1, 2, 3, 4]);

        let mut out8 = [0u8; 8];
        let miss = pool
            .rdma_get_into("chunk_0", 0x1000_0100, 8, false, 20, &mut out8)
            .expect("mismatch ok");
        assert_eq!(miss, RDMA_GET_MISS);
    }

    #[test]
    fn test_async_store_coordinator_closes_preemption_window() {
        let mut coord = AsyncEngineDrivenStoreCoordinator::new();
        let sig = StagingBufferSignature {
            shape: vec![16, 64],
            dtype: DType::BF16,
        };
        let _b1 = coord.submit_store(1, sig.clone()).expect("submit 1");
        let _b2 = coord.submit_store(2, sig.clone()).expect("submit 2");
        coord.record_gather_launched(1, 501).expect("gather 1");

        let flushed = coord.flush_inflight_stores(900).expect("flush");
        assert_eq!(flushed, vec![501, 900]);
        assert_eq!(coord.staging_pool().pooled_count(&sig), 2);
    }
}
