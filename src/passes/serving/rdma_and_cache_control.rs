//! RDMA L1 arena window resolution with revoked-region retry, three-phase
//! asynchronous engine-driven KV store coordination, and runtime L2 adapter
//! lifecycle, reconfiguration, and cache-control services.
//!
//! Covers:
//! 1. **RDMA L1 Arena Window Resolver & Revoked-Region Retry Pool (`RdmaArenaWindowPool`)**:
//!    Validates L1 arena base/size and fabric provider (`"efa-direct"` or
//!    `"tcp"`), resolves writable in-arena buffer sub-windows, rejects 0-byte
//!    RDMA writes, returns `RDMA_GET_MISS = -1` on missing keys or mismatched
//!    byte counts, and retries once after re-registering the L1 arena if the
//!    fabric error indicates a `"revoked"` memory registration.
//! 2. **Three-Phase Async Engine-Driven Store Coordinator & Pinned Staging Pool
//!    (`PinnedStagingBufferPool`, `AsyncEngineDrivenStoreCoordinator`)**:
//!    Pools reusable `(shape, dtype)` pinned CPU staging buffers and tracks
//!    `pending_stores` (submitted stores whose device gather event has not yet
//!    been recorded) separately from `inflight_gather_events` so
//!    `flush_inflight_stores` closes the preemption race window between
//!    submission and worker gather launch.
//! 3. **Runtime L2 Adapter Lifecycle, Reconfiguration & Cache-Control Services
//!    (`AdapterLifecycleController`, `select_l1_write_targets`,
//!    `resolve_token_chunk_object_keys`, `CacheControlObjectService`)**:
//!    Supports online `AddAdapter` and graceful-drain `RemoveAdapter`,
//!    non-overlapping physical range reconfiguration, multi-L1 write target
//!    routing, bounded token-chunk key resolution, and lock-aware tiered
//!    object deletion (`L1`, `L2`, `All`).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::ir::mlir::DType;

// ---------------------------------------------------------------------------
// 1. RDMA L1 Arena Window Resolver & Revoked-Region Retry Pool
// ---------------------------------------------------------------------------

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

/// Stored RDMA payload entry with optional expiration timestamp.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `l1_base == 0`, `l1_size == 0`, `rdma_provider` is not
    /// `"efa-direct"` or `"tcp"`, or `rdma_interface` is `Some("")`.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `readonly` is `true`, `buf_len == 0`, or the buffer lies
    /// outside `[l1_base, l1_base + l1_size)`.
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

    /// Executes an RDMA operation with single-retry recovery if the transport
    /// fails with an error containing `"revoked"`.
    fn run_with_revoked_retry<T, F>(&mut self, mut op: F) -> Result<T, String>
    where
        F: FnMut(&mut Self) -> Result<T, String>,
    {
        match op(self) {
            Ok(val) => Ok(val),
            Err(first_err) if first_err.contains("revoked") => {
                // Re-register the L1 memory region and retry exactly once.
                self.registration_epoch += 1;
                op(self)
            }
            Err(other_err) => Err(other_err),
        }
    }

    /// Stores `payload` under `key` via the RDMA worker pool, rejecting empty
    /// 0-byte values and applying an optional `ttl_seconds` relative to `now_sec`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `payload` is empty or if a transport fault persists
    /// after re-registering a revoked memory region.
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
    ///
    /// Returns `Ok(expected_size as i64)` on an exact-size hit, or
    /// `Ok(RDMA_GET_MISS)` (`-1`) if the key is missing, expired at `now_sec`,
    /// or its stored byte length does not equal `expected_size`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the target buffer lies outside the L1 arena, is
    /// read-only, or if a revoked-registration fault persists after retry.
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

// ---------------------------------------------------------------------------
// 2. Three-Phase Async Engine-Driven Store Coordinator & Pinned Staging Pool
// ---------------------------------------------------------------------------

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
    ///
    /// # Errors
    ///
    /// Returns `Err` if the coordinator is closed or `store_id` already exists.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `store_id` is not in `PendingGatherLaunch`.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `store_id` is not in `GatherInflight`.
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
    /// eviction can overwrite source GPU pages:
    /// 1. Waits for every store in `pending_stores` to launch its gather event
    ///    (assigning synthetic gather events starting at `next_synthetic_event`
    ///    for any pending store not yet advanced by a background worker).
    /// 2. Synchronizes all `inflight_gather_events` in deterministic order and
    ///    completes their commits, returning the list of synchronized event IDs.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any phase transition fails.
    pub fn flush_inflight_stores(
        &mut self,
        mut next_synthetic_event: u64,
    ) -> Result<Vec<u64>, String> {
        // Step 1: Drain `pending_stores` first so no submitted store can slip
        // through the window between submission and gather-event recording.
        let pending_ids: Vec<u64> = self.pending_stores.iter().copied().collect();
        for store_id in pending_ids {
            self.record_gather_launched(store_id, next_synthetic_event)?;
            next_synthetic_event += 1;
        }

        // Step 2: Synchronize all `inflight_gather_events`.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if flushing remaining stores fails.
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

// ---------------------------------------------------------------------------
// 3. Runtime L2 Adapter Lifecycle, Reconfiguration & Cache-Control Services
// ---------------------------------------------------------------------------

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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `adapter_id` is already active/draining, `byte_len == 0`,
    /// or `owned_range` overlaps another adapter.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `adapter_id` is not `Active`, `new_range.byte_len == 0`,
    /// or `new_range` overlaps another adapter.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if the adapter is unknown, `Draining`, or `Removed`.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `adapter_id` is unknown or has `0` in-flight tasks.
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
    ///
    /// # Errors
    ///
    /// Returns `Err` if `adapter_id` is unknown.
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

/// Write-target selection policy across multiple L1 manager instances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L1WritePolicy {
    /// Routes all writes to the first healthy L1 manager target.
    DefaultFirst,
    /// Routes writes to the L1 manager matching the preferred tag (falling back
    /// to the first healthy target).
    TagAffinity(String),
    /// Routes writes to the healthy L1 manager with the lowest utilization ratio.
    LeastUtilized,
}

/// Target descriptor for an L1 cache manager instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L1ManagerTargetSpec {
    /// Unique manager identifier.
    pub manager_id: String,
    /// Affinity tags associated with this L1 manager (e.g. `"numa0"`, `"nvlink"`).
    pub tags: Vec<String>,
    /// Used bytes in this L1 manager.
    pub used_bytes: u64,
    /// Total capacity bytes in this L1 manager.
    pub capacity_bytes: u64,
    /// Whether the L1 manager is healthy and accepting writes.
    pub healthy: bool,
}

/// Selects the target `manager_id` for a batch of write keys under `policy`.
///
/// # Errors
///
/// Returns `Err` if no healthy L1 manager target with non-zero capacity exists.
pub fn select_l1_write_targets(
    targets: &[L1ManagerTargetSpec],
    policy: &L1WritePolicy,
) -> Result<String, String> {
    let healthy: Vec<&L1ManagerTargetSpec> = targets
        .iter()
        .filter(|t| t.healthy && t.capacity_bytes > 0 && t.used_bytes < t.capacity_bytes)
        .collect();
    if healthy.is_empty() {
        return Err("no healthy L1 manager target with available capacity".to_string());
    }
    match policy {
        L1WritePolicy::DefaultFirst => Ok(healthy[0].manager_id.clone()),
        L1WritePolicy::TagAffinity(tag) => {
            if let Some(matched) = healthy.iter().find(|t| t.tags.iter().any(|tg| tg == tag)) {
                Ok(matched.manager_id.clone())
            } else {
                Ok(healthy[0].manager_id.clone())
            }
        }
        L1WritePolicy::LeastUtilized => {
            let best = healthy
                .iter()
                .min_by_key(|t| (t.used_bytes as u128 * 1_000_000) / (t.capacity_bytes as u128))
                .copied()
                .ok_or_else(|| "no healthy L1 manager target".to_string())?;
            Ok(best.manager_id.clone())
        }
    }
}

/// Maximum number of token IDs accepted in a single cache-control resolution call.
pub const MAX_CACHE_CONTROL_TOKEN_IDS: usize = 1_000_000;

/// Maximum number of object keys accepted in a single cache-control delete call.
pub const MAX_CACHE_CONTROL_DELETE_BATCH: usize = 10_000;

/// Deterministic rolling FNV-1a 64-bit hash used to chain token chunks.
fn hash_token_chunk_with_prefix(prefix_hash: u64, chunk: &[u32]) -> u64 {
    let mut hash = prefix_hash ^ 0xcbf2_9ce4_8422_2325;
    for &tok in chunk {
        for b in tok.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

/// Resolves a sequence of `token_ids` into cumulative prefix-hashed chunk object
/// keys (`"{model_name}@{world_size}:{rank}:chunk_{idx}:{hash:016x}"`).
///
/// Trailing partial chunks smaller than `chunk_size` are omitted (only full
/// `chunk_size` blocks produce cache-control object keys).
///
/// # Errors
///
/// Returns `Err` if `chunk_size == 0` or `token_ids.len() > MAX_CACHE_CONTROL_TOKEN_IDS`.
pub fn resolve_token_chunk_object_keys(
    model_name: &str,
    world_size: u32,
    rank: u32,
    token_ids: &[u32],
    chunk_size: usize,
) -> Result<Vec<String>, String> {
    if chunk_size == 0 {
        return Err("chunk_size must be greater than zero".to_string());
    }
    if token_ids.len() > MAX_CACHE_CONTROL_TOKEN_IDS {
        return Err(format!(
            "token_ids length {} exceeds maximum {}",
            token_ids.len(),
            MAX_CACHE_CONTROL_TOKEN_IDS
        ));
    }
    let num_full_chunks = token_ids.len() / chunk_size;
    let mut keys = Vec::with_capacity(num_full_chunks);
    let mut running_hash = 0u64;
    for chunk_idx in 0..num_full_chunks {
        let slice = &token_ids[chunk_idx * chunk_size..(chunk_idx + 1) * chunk_size];
        running_hash = hash_token_chunk_with_prefix(running_hash, slice);
        keys.push(format!(
            "{model_name}@{world_size}:{rank}:chunk_{chunk_idx}:{running_hash:016x}"
        ));
    }
    Ok(keys)
}

/// Target storage tier for [`CacheControlObjectService::delete_objects`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheDeleteTier {
    /// Delete only from L1 memory.
    L1Only,
    /// Delete only from L2 storage adapters.
    L2Only,
    /// Delete from both L1 and L2 tiers.
    AllTiers,
}

/// Summary of a batch [`CacheControlObjectService::delete_objects`] operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheDeleteSummary {
    /// Number of object keys deleted from L1.
    pub deleted_l1: usize,
    /// Number of object keys deleted from L2.
    pub deleted_l2: usize,
    /// Keys skipped because they had active read/write locks and `force == false`.
    pub skipped_locked: Vec<String>,
}

/// Multi-tier cache-control object management service supporting lock-aware
/// tiered deletion (`L1Only`, `L2Only`, `AllTiers`) and paginated L2 key listing.
#[derive(Debug, Default)]
pub struct CacheControlObjectService {
    l1_objects: BTreeSet<String>,
    l1_locked_keys: HashSet<String>,
    l2_objects: BTreeSet<String>,
}

impl CacheControlObjectService {
    /// Creates a new empty cache-control object service.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an object key into L1 (with optional active lock) and/or L2.
    pub fn insert_object(&mut self, key: &str, in_l1: bool, l1_locked: bool, in_l2: bool) {
        if in_l1 {
            self.l1_objects.insert(key.to_string());
            if l1_locked {
                self.l1_locked_keys.insert(key.to_string());
            }
        }
        if in_l2 {
            self.l2_objects.insert(key.to_string());
        }
    }

    /// Lists L2 keys matching `prefix` using cursor-based pagination (`after_cursor`,
    /// `limit`). Returns `(matching_page, next_cursor)`.
    #[must_use]
    pub fn list_l2_keys_paginated(
        &self,
        prefix: &str,
        after_cursor: Option<&str>,
        limit: usize,
    ) -> (Vec<String>, Option<String>) {
        if limit == 0 {
            return (Vec::new(), None);
        }
        let iter = self
            .l2_objects
            .iter()
            .filter(|k| k.starts_with(prefix))
            .filter(|k| after_cursor.is_none_or(|cur| k.as_str() > cur));

        let mut page = Vec::with_capacity(limit);
        let mut has_more = false;
        for key in iter {
            if page.len() < limit {
                page.push(key.clone());
            } else {
                has_more = true;
                break;
            }
        }
        let next_cursor = if has_more { page.last().cloned() } else { None };
        (page, next_cursor)
    }

    /// Deletes a batch of `keys` from `tier`.
    ///
    /// When `force == false`, keys currently locked in L1 are not deleted from L1
    /// and are reported in `skipped_locked`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `keys.len() > MAX_CACHE_CONTROL_DELETE_BATCH`.
    pub fn delete_objects(
        &mut self,
        keys: &[String],
        tier: CacheDeleteTier,
        force: bool,
    ) -> Result<CacheDeleteSummary, String> {
        if keys.len() > MAX_CACHE_CONTROL_DELETE_BATCH {
            return Err(format!(
                "delete batch size {} exceeds maximum {}",
                keys.len(),
                MAX_CACHE_CONTROL_DELETE_BATCH
            ));
        }

        let mut deleted_l1 = 0usize;
        let mut deleted_l2 = 0usize;
        let mut skipped_locked = Vec::new();

        for key in keys {
            if matches!(tier, CacheDeleteTier::L1Only | CacheDeleteTier::AllTiers)
                && self.l1_objects.contains(key)
            {
                if self.l1_locked_keys.contains(key) && !force {
                    skipped_locked.push(key.clone());
                } else {
                    self.l1_objects.remove(key);
                    self.l1_locked_keys.remove(key);
                    deleted_l1 += 1;
                }
            }
            if matches!(tier, CacheDeleteTier::L2Only | CacheDeleteTier::AllTiers)
                && self.l2_objects.remove(key)
            {
                deleted_l2 += 1;
            }
        }

        Ok(CacheDeleteSummary {
            deleted_l1,
            deleted_l2,
            skipped_locked,
        })
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

        // Mismatched expected size returns RDMA_GET_MISS (-1).
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

        // Store 2 is still in pending_stores; flush must drain pending_stores
        // first and synchronize both events.
        let flushed = coord.flush_inflight_stores(900).expect("flush");
        assert_eq!(flushed, vec![501, 900]);
        assert_eq!(coord.staging_pool().pooled_count(&sig), 2);
    }
}
