//! Shared-memory region lease authority (`SharedRegionState`), idempotent request-digest reply
//! window (`OrchestratorReplyWindow`), pre-prefetch KV load policy (`KvLoadPolicy`), and
//! multi-field object group transfer framing (`ObjectGroupFrame`).

use crate::ir::DType;
use indexmap::IndexMap;
use sha2::{Digest, Sha256};

// ============================================================================
// Shared-Memory Region Lease Authority (`SharedRegionState`)
// ============================================================================

/// Default shared-memory region contract protocol version.
pub const REGION_CONTRACT_VERSION: u32 = 1;

/// Maximum concurrent active read leases permitted on a single key.
pub const MAX_READER_COUNT: u32 = 1024;

/// Maximum completed write tokens retained per client incarnation for idempotent `finish_write`.
pub const MAX_COMPLETED_WRITE_TOKENS: usize = 65_536;

/// Maximum cached replies retained per client incarnation in [`OrchestratorReplyWindow`].
pub const REPLY_WINDOW_SIZE: usize = 1024;

/// Contract parameters governing a shared-memory region authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionContract {
    /// Wire contract version (expected to match [`REGION_CONTRACT_VERSION`]).
    pub version: u32,
    /// Monotonic authority epoch identifier.
    pub epoch: u64,
    /// Total byte capacity of the managed shared-memory arena.
    pub capacity_bytes: u64,
    /// Power-of-two allocation alignment in bytes.
    pub alignment_bytes: u64,
    /// Time-to-live in milliseconds for in-flight write reservations before lazy expiration.
    pub write_ttl_ms: u64,
}

/// Snapshot of shared-memory region capacity and lease occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionUsage {
    /// Total arena capacity in bytes.
    pub capacity_bytes: u64,
    /// Total aligned bytes currently occupied by valid keys and in-flight writes.
    pub used_bytes: u64,
    /// Total aligned bytes currently available for allocation.
    pub free_bytes: u64,
    /// Number of committed valid keys residing in the region.
    pub live_keys: usize,
    /// Number of uncommitted in-flight write reservations.
    pub inflight_writes: usize,
    /// Total number of active read leases across all keys.
    pub active_read_leases: usize,
}

/// Contiguous byte extent within a shared-memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtentSpec {
    /// Aligned starting byte offset within the shared-memory region.
    pub offset: u64,
    /// Logical payload length in bytes requested by the writer.
    pub length: u64,
    /// Physical length in bytes rounded up to `alignment_bytes`.
    pub aligned_length: u64,
}

/// Per-key outcome returned by [`SharedRegionState::reserve_write_batch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteReserveStatus {
    /// New extent reserved for writing under `write_token`.
    Reserved {
        write_token: u64,
        extent: ExtentSpec,
    },
    /// Key already exists in a committed valid state; no write needed.
    ExistsValid { extent: ExtentSpec },
    /// Key is currently being written by an unexpired in-flight reservation.
    BusyWriting,
    /// Arena lacked sufficient free aligned space (triggers all-or-nothing batch rollback).
    OutOfSpace,
}

/// Active read lease granted on a committed key in [`SharedRegionState`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadLease {
    /// Object key covered by the lease.
    pub key: String,
    /// Unique lease identifier used to release the lease via `finish_read`.
    pub lease_id: u64,
    /// Owning client identifier.
    pub client_id: String,
    /// Owning client incarnation identifier.
    pub incarnation_id: u64,
    /// Underlying byte extent of the committed object.
    pub extent: ExtentSpec,
    /// Current reader count on the key after granting this lease.
    pub reader_count: u32,
}

#[derive(Debug, Clone)]
struct InflightWriteRecord {
    write_token: u64,
    key: String,
    client_id: String,
    incarnation_id: u64,
    extent: ExtentSpec,
    reserved_at_ms: u64,
}

#[derive(Debug, Clone)]
struct ValidKeyRecord {
    extent: ExtentSpec,
    reader_count: u32,
}

/// Shared-memory region lease and extent authority with client incarnation tracking, lazy write-TTL
/// expiration, all-or-nothing aligned batch write reservation, reader lease caps, and bounded
/// 65,536-entry idempotent write completion history.
#[derive(Debug, Clone)]
pub struct SharedRegionState {
    contract: RegionContract,
    free_extents: Vec<(u64, u64)>,
    client_incarnations: IndexMap<String, u64>,
    inflight_by_token: IndexMap<u64, InflightWriteRecord>,
    inflight_token_by_key: IndexMap<String, u64>,
    valid_keys: IndexMap<String, ValidKeyRecord>,
    read_leases: IndexMap<u64, ReadLease>,
    completed_tokens_by_client: IndexMap<(String, u64), IndexMap<u64, ExtentSpec>>,
    next_token: u64,
    next_lease_id: u64,
}

impl SharedRegionState {
    /// Creates a new shared-memory region authority governed by `contract`.
    pub fn new(contract: RegionContract) -> Result<Self, String> {
        if contract.version != REGION_CONTRACT_VERSION {
            return Err(format!(
                "unsupported region contract version {} (expected {})",
                contract.version, REGION_CONTRACT_VERSION
            ));
        }
        if contract.alignment_bytes == 0 || !contract.alignment_bytes.is_power_of_two() {
            return Err(format!(
                "alignment_bytes ({}) must be a non-zero power of two",
                contract.alignment_bytes
            ));
        }
        if contract.capacity_bytes < contract.alignment_bytes
            || !contract
                .capacity_bytes
                .is_multiple_of(contract.alignment_bytes)
        {
            return Err(format!(
                "capacity_bytes ({}) must be a positive multiple of alignment_bytes ({})",
                contract.capacity_bytes, contract.alignment_bytes
            ));
        }
        Ok(Self {
            contract,
            free_extents: vec![(0, contract.capacity_bytes)],
            client_incarnations: IndexMap::new(),
            inflight_by_token: IndexMap::new(),
            inflight_token_by_key: IndexMap::new(),
            valid_keys: IndexMap::new(),
            read_leases: IndexMap::new(),
            completed_tokens_by_client: IndexMap::new(),
            next_token: 1,
            next_lease_id: 1,
        })
    }

    /// Returns the governing [`RegionContract`].
    pub fn contract(&self) -> RegionContract {
        self.contract
    }

    /// Verifies that a client's `(version, epoch)` matches the authority's contract.
    pub fn validate_contract(&self, version: u32, epoch: u64) -> Result<(), String> {
        if version != self.contract.version || epoch != self.contract.epoch {
            return Err(format!(
                "contract mismatch: got (v={}, epoch={}), expected (v={}, epoch={})",
                version, epoch, self.contract.version, self.contract.epoch
            ));
        }
        Ok(())
    }

    fn align_up(&self, length: u64) -> u64 {
        let align = self.contract.alignment_bytes;
        let eff = length.max(1);
        let mask = align - 1;
        (eff + mask) & !mask
    }

    fn alloc_aligned(&mut self, length: u64) -> Option<ExtentSpec> {
        let aligned_length = self.align_up(length);
        let pos = self
            .free_extents
            .iter()
            .position(|&(_, cap)| cap >= aligned_length)?;
        let (offset, cap) = self.free_extents[pos];
        if cap == aligned_length {
            self.free_extents.remove(pos);
        } else {
            self.free_extents[pos] = (offset + aligned_length, cap - aligned_length);
        }
        Some(ExtentSpec {
            offset,
            length,
            aligned_length,
        })
    }

    fn free_extent(&mut self, extent: ExtentSpec) {
        if extent.aligned_length == 0 {
            return;
        }
        self.free_extents
            .push((extent.offset, extent.aligned_length));
        self.free_extents.sort_by_key(|&(off, _)| off);
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.free_extents.len());
        for (off, len) in self.free_extents.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.0 + last.1 == off
            {
                last.1 += len;
            } else {
                merged.push((off, len));
            }
        }
        self.free_extents = merged;
    }

    fn check_client_incarnation(&self, client_id: &str, incarnation_id: u64) -> Result<(), String> {
        let Some(&active_inc) = self.client_incarnations.get(client_id) else {
            return Err(format!("unregistered client '{client_id}'"));
        };
        if active_inc != incarnation_id {
            return Err(format!(
                "stale client incarnation {} for '{}' (active: {})",
                incarnation_id, client_id, active_inc
            ));
        }
        Ok(())
    }

    /// Registers `client_id` with `incarnation_id`. If `client_id` was previously registered under
    /// a different incarnation, automatically retires all in-flight write reservations (freeing
    /// their extents) and active read leases belonging to the prior incarnation.
    ///
    /// Returns `(retired_writes, retired_leases)`.
    pub fn register_client(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        now_ms: u64,
    ) -> (usize, usize) {
        self.expire_stale_writes(now_ms);
        let mut retired_writes = 0usize;
        let mut retired_leases = 0usize;

        if let Some(&prev_inc) = self.client_incarnations.get(client_id)
            && prev_inc != incarnation_id
        {
            let stale_tokens: Vec<u64> = self
                .inflight_by_token
                .values()
                .filter(|w| w.client_id == client_id && w.incarnation_id == prev_inc)
                .map(|w| w.write_token)
                .collect();
            for tok in stale_tokens {
                if let Some(rec) = self.inflight_by_token.shift_remove(&tok) {
                    self.inflight_token_by_key.shift_remove(&rec.key);
                    self.free_extent(rec.extent);
                    retired_writes += 1;
                }
            }

            let stale_leases: Vec<u64> = self
                .read_leases
                .values()
                .filter(|l| l.client_id == client_id && l.incarnation_id == prev_inc)
                .map(|l| l.lease_id)
                .collect();
            for lid in stale_leases {
                if let Some(lease) = self.read_leases.shift_remove(&lid) {
                    if let Some(vk) = self.valid_keys.get_mut(&lease.key) {
                        vk.reader_count = vk.reader_count.saturating_sub(1);
                    }
                    retired_leases += 1;
                }
            }
            self.completed_tokens_by_client
                .shift_remove(&(client_id.to_string(), prev_inc));
        }

        self.client_incarnations
            .insert(client_id.to_string(), incarnation_id);
        (retired_writes, retired_leases)
    }

    /// Lazily expires in-flight write reservations whose `now_ms >= reserved_at_ms + write_ttl_ms`
    /// and reclaims their backing extents. Returns the number of expired writes reclaimed.
    pub fn expire_stale_writes(&mut self, now_ms: u64) -> usize {
        if self.contract.write_ttl_ms == 0 {
            return 0;
        }
        let ttl = self.contract.write_ttl_ms;
        let expired_tokens: Vec<u64> = self
            .inflight_by_token
            .values()
            .filter(|w| now_ms >= w.reserved_at_ms.saturating_add(ttl))
            .map(|w| w.write_token)
            .collect();
        let count = expired_tokens.len();
        for tok in expired_tokens {
            if let Some(rec) = self.inflight_by_token.shift_remove(&tok) {
                self.inflight_token_by_key.shift_remove(&rec.key);
                self.free_extent(rec.extent);
            }
        }
        count
    }

    /// Reserves extents for a batch of `(key, byte_length)` write requests using **all-or-nothing**
    /// allocation semantics:
    /// - If a key is already committed, its slot is [`WriteReserveStatus::ExistsValid`].
    /// - If a key has an unexpired in-flight write, its slot is
    ///   [`WriteReserveStatus::BusyWriting`].
    /// - Otherwise an aligned extent is allocated. If **any** new key in `requests` cannot be
    ///   allocated due to insufficient space, all extents newly allocated during this batch call
    ///   are rolled back and freed so no partial reservation leaks.
    pub fn reserve_write_batch(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        requests: &[(String, u64)],
        now_ms: u64,
    ) -> Result<Vec<WriteReserveStatus>, String> {
        self.check_client_incarnation(client_id, incarnation_id)?;
        self.expire_stale_writes(now_ms);

        let mut results: Vec<WriteReserveStatus> = Vec::with_capacity(requests.len());
        let mut newly_reserved_tokens: Vec<u64> = Vec::new();
        let mut out_of_space = false;

        for (key, byte_len) in requests {
            if let Some(vk) = self.valid_keys.get(key) {
                results.push(WriteReserveStatus::ExistsValid { extent: vk.extent });
                continue;
            }
            if self.inflight_token_by_key.contains_key(key) {
                results.push(WriteReserveStatus::BusyWriting);
                continue;
            }
            if let Some(extent) = self.alloc_aligned(*byte_len) {
                let token = self.next_token;
                self.next_token += 1;
                let rec = InflightWriteRecord {
                    write_token: token,
                    key: key.clone(),
                    client_id: client_id.to_string(),
                    incarnation_id,
                    extent,
                    reserved_at_ms: now_ms,
                };
                self.inflight_by_token.insert(token, rec);
                self.inflight_token_by_key.insert(key.clone(), token);
                newly_reserved_tokens.push(token);
                results.push(WriteReserveStatus::Reserved {
                    write_token: token,
                    extent,
                });
            } else {
                out_of_space = true;
                results.push(WriteReserveStatus::OutOfSpace);
                break;
            }
        }

        if out_of_space {
            for tok in newly_reserved_tokens {
                if let Some(rec) = self.inflight_by_token.shift_remove(&tok) {
                    self.inflight_token_by_key.shift_remove(&rec.key);
                    self.free_extent(rec.extent);
                }
            }
            return Ok(vec![WriteReserveStatus::OutOfSpace; requests.len()]);
        }

        Ok(results)
    }

    /// Commits an in-flight write identified by `write_token` into a valid key.
    ///
    /// Idempotent per `(client_id, incarnation_id)` across a bounded history of
    /// [`MAX_COMPLETED_WRITE_TOKENS`] (`65,536`) completed tokens: repeating `finish_write` with an
    /// already-committed token replays the committed [`ExtentSpec`].
    pub fn finish_write(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        write_token: u64,
        now_ms: u64,
    ) -> Result<ExtentSpec, String> {
        self.check_client_incarnation(client_id, incarnation_id)?;
        self.expire_stale_writes(now_ms);

        let client_key = (client_id.to_string(), incarnation_id);
        if let Some(history) = self.completed_tokens_by_client.get(&client_key)
            && let Some(&ext) = history.get(&write_token)
        {
            return Ok(ext);
        }

        let Some(rec) = self.inflight_by_token.shift_remove(&write_token) else {
            return Err(format!(
                "write_token {write_token} is unknown, aborted, or expired"
            ));
        };
        if rec.client_id != client_id || rec.incarnation_id != incarnation_id {
            self.inflight_by_token.insert(write_token, rec);
            return Err(format!(
                "write_token {write_token} does not belong to ({client_id}, {incarnation_id})"
            ));
        }

        self.inflight_token_by_key.shift_remove(&rec.key);
        self.valid_keys.insert(
            rec.key,
            ValidKeyRecord {
                extent: rec.extent,
                reader_count: 0,
            },
        );

        let history = self
            .completed_tokens_by_client
            .entry(client_key)
            .or_default();
        if history.len() >= MAX_COMPLETED_WRITE_TOKENS
            && let Some((&oldest_tok, _)) = history.first()
        {
            history.shift_remove(&oldest_tok);
        }
        history.insert(write_token, rec.extent);

        Ok(rec.extent)
    }

    /// Aborts an in-flight write reservation and returns its extent to the free list.
    pub fn abort_write(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        write_token: u64,
    ) -> Result<bool, String> {
        self.check_client_incarnation(client_id, incarnation_id)?;
        let Some(rec) = self.inflight_by_token.get(&write_token) else {
            return Ok(false);
        };
        if rec.client_id != client_id || rec.incarnation_id != incarnation_id {
            return Err(format!(
                "write_token {write_token} does not belong to ({client_id}, {incarnation_id})"
            ));
        }
        if let Some(removed) = self.inflight_by_token.shift_remove(&write_token) {
            self.inflight_token_by_key.shift_remove(&removed.key);
            self.free_extent(removed.extent);
            return Ok(true);
        }
        Ok(false)
    }

    /// Acquires a shared read lease on a committed `key`, enforcing [`MAX_READER_COUNT`] (`1024`).
    pub fn reserve_read(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        key: &str,
        now_ms: u64,
    ) -> Result<ReadLease, String> {
        self.check_client_incarnation(client_id, incarnation_id)?;
        self.expire_stale_writes(now_ms);

        let Some(vk) = self.valid_keys.get_mut(key) else {
            return Err(format!("key '{key}' is not present in valid state"));
        };
        if vk.reader_count >= MAX_READER_COUNT {
            return Err(format!(
                "key '{key}' reached MAX_READER_COUNT ({MAX_READER_COUNT})"
            ));
        }
        vk.reader_count += 1;
        let lease_id = self.next_lease_id;
        self.next_lease_id += 1;
        let lease = ReadLease {
            key: key.to_string(),
            lease_id,
            client_id: client_id.to_string(),
            incarnation_id,
            extent: vk.extent,
            reader_count: vk.reader_count,
        };
        self.read_leases.insert(lease_id, lease.clone());
        Ok(lease)
    }

    /// Releases an active read lease granted by [`Self::reserve_read`].
    pub fn finish_read(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        lease_id: u64,
    ) -> Result<bool, String> {
        self.check_client_incarnation(client_id, incarnation_id)?;
        let Some(lease) = self.read_leases.get(&lease_id) else {
            return Ok(false);
        };
        if lease.client_id != client_id || lease.incarnation_id != incarnation_id {
            return Err(format!(
                "lease_id {lease_id} does not belong to ({client_id}, {incarnation_id})"
            ));
        }
        if let Some(removed) = self.read_leases.shift_remove(&lease_id) {
            if let Some(vk) = self.valid_keys.get_mut(&removed.key) {
                vk.reader_count = vk.reader_count.saturating_sub(1);
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Evicts a committed key when `reader_count == 0` and returns its reclaimed [`ExtentSpec`].
    pub fn evict_key(&mut self, key: &str, now_ms: u64) -> Result<ExtentSpec, String> {
        self.expire_stale_writes(now_ms);
        let Some(vk) = self.valid_keys.get(key) else {
            return Err(format!("key '{key}' not found"));
        };
        if vk.reader_count > 0 {
            return Err(format!(
                "cannot evict key '{key}' with {} active readers",
                vk.reader_count
            ));
        }
        let removed = self
            .valid_keys
            .shift_remove(key)
            .ok_or_else(|| format!("key '{key}' not found"))?;
        self.free_extent(removed.extent);
        Ok(removed.extent)
    }

    /// Computes a current [`RegionUsage`] snapshot after expiring any stale writes at `now_ms`.
    pub fn usage(&mut self, now_ms: u64) -> RegionUsage {
        self.expire_stale_writes(now_ms);
        let free_bytes: u64 = self.free_extents.iter().map(|&(_, len)| len).sum();
        let used_bytes = self.contract.capacity_bytes.saturating_sub(free_bytes);
        RegionUsage {
            capacity_bytes: self.contract.capacity_bytes,
            used_bytes,
            free_bytes,
            live_keys: self.valid_keys.len(),
            inflight_writes: self.inflight_by_token.len(),
            active_read_leases: self.read_leases.len(),
        }
    }
}

// ============================================================================
// Idempotent Request-Digest Reply Window (`OrchestratorReplyWindow`)
// ============================================================================

/// Lookup outcome in [`OrchestratorReplyWindow`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplyWindowOutcome {
    /// `request_id` is not present in the client incarnation's reply window.
    Miss,
    /// `request_id` matched an existing entry with an identical SHA-256 request payload digest.
    ReplayCached(Vec<u8>),
    /// `request_id` was reused with a different request payload digest (`ALREADY_EXISTS`).
    ConflictAlreadyExists,
}

#[derive(Debug, Clone)]
struct ClientReplyRing {
    incarnation_id: u64,
    entries: IndexMap<u64, ([u8; 32], Vec<u8>)>,
}

/// Per-client incarnation reply window (`REPLY_WINDOW_SIZE = 1024`) indexing `(request_sha256,
/// reply_bytes)` by `request_id` so byte-identical retries replay the cached response while
/// conflicting payloads with a reused `request_id` fail with
/// [`ReplyWindowOutcome::ConflictAlreadyExists`].
#[derive(Debug, Clone, Default)]
pub struct OrchestratorReplyWindow {
    window_size: usize,
    clients: IndexMap<String, ClientReplyRing>,
}

impl OrchestratorReplyWindow {
    /// Creates a reply window with capacity `window_size` (clamped to `>= 1`, defaulting to
    /// [`REPLY_WINDOW_SIZE`]).
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            clients: IndexMap::new(),
        }
    }

    fn sha256_digest(payload: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let out = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&out);
        bytes
    }

    /// Checks whether `(client_id, incarnation_id, request_id)` has a cached reply in the window:
    /// - If `incarnation_id` is newer than the recorded incarnation for `client_id`, resets that
    ///   client's window and returns [`ReplyWindowOutcome::Miss`].
    /// - If `request_id` is found and its SHA-256 digest matches `request_payload`, returns
    ///   [`ReplyWindowOutcome::ReplayCached`].
    /// - If `request_id` is found with a different SHA-256 digest, returns
    ///   [`ReplyWindowOutcome::ConflictAlreadyExists`].
    pub fn check_or_replay(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        request_id: u64,
        request_payload: &[u8],
    ) -> ReplyWindowOutcome {
        let digest = Self::sha256_digest(request_payload);
        let ring = self
            .clients
            .entry(client_id.to_string())
            .or_insert_with(|| ClientReplyRing {
                incarnation_id,
                entries: IndexMap::new(),
            });
        if ring.incarnation_id != incarnation_id {
            ring.incarnation_id = incarnation_id;
            ring.entries.clear();
            return ReplyWindowOutcome::Miss;
        }
        let Some((saved_digest, reply)) = ring.entries.get(&request_id) else {
            return ReplyWindowOutcome::Miss;
        };
        if saved_digest == &digest {
            ReplyWindowOutcome::ReplayCached(reply.clone())
        } else {
            ReplyWindowOutcome::ConflictAlreadyExists
        }
    }

    /// Records `reply_bytes` for `(client_id, incarnation_id, request_id, request_payload)`,
    /// evicting the oldest entry if the client's window exceeds `window_size`. Returns `Err` if a
    /// conflicting payload digest already exists for `request_id`.
    pub fn record_reply(
        &mut self,
        client_id: &str,
        incarnation_id: u64,
        request_id: u64,
        request_payload: &[u8],
        reply_bytes: Vec<u8>,
    ) -> Result<(), String> {
        let digest = Self::sha256_digest(request_payload);
        let ring = self
            .clients
            .entry(client_id.to_string())
            .or_insert_with(|| ClientReplyRing {
                incarnation_id,
                entries: IndexMap::new(),
            });
        if ring.incarnation_id != incarnation_id {
            ring.incarnation_id = incarnation_id;
            ring.entries.clear();
        }
        if let Some((saved_digest, _)) = ring.entries.get(&request_id)
            && saved_digest != &digest
        {
            return Err(format!(
                "conflicting payload digest for request_id {request_id} on client '{client_id}'"
            ));
        }
        if !ring.entries.contains_key(&request_id)
            && ring.entries.len() >= self.window_size
            && let Some((&oldest_req, _)) = ring.entries.first()
        {
            ring.entries.shift_remove(&oldest_req);
        }
        ring.entries.insert(request_id, (digest, reply_bytes));
        Ok(())
    }
}

// ============================================================================
// Pre-Prefetch KV Load Policy (`KvLoadPolicy`) & Object Group Framing
// ============================================================================

/// Per-request context passed to [`evaluate_kv_load_policy`] prior to L1/L2 KV-cache lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLoadContext {
    /// Request identifier.
    pub request_id: String,
    /// Total number of prompt tokens eligible for KV lookup.
    pub num_lookup_tokens: usize,
    /// KV cache chunk size in tokens.
    pub chunk_size: usize,
    /// Explicit caller override requesting that KV load be skipped (`skip_load`).
    pub skip_load: bool,
    /// Optional hint reporting how many prefix tokens the inference engine already has in GPU HBM.
    pub engine_computed_tokens: Option<usize>,
    /// Minimum novel (uncomputed) tokens required for an external KV load to amortize transfer
    /// overhead over local recomputation.
    pub min_novel_tokens_threshold: usize,
}

/// Decision returned by [`evaluate_kv_load_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLoadDecision {
    /// Whether the serving runtime should proceed with L1/L2 KV lookup and load.
    pub should_load: bool,
    /// Number of tokens that would actually be saved over engine-resident tokens.
    pub effective_novel_tokens: usize,
    /// Diagnostic reason code explaining the decision.
    pub reason: &'static str,
}

/// Evaluates whether a request should perform L1/L2 KV-cache lookup and load based on `skip_load`
/// overrides, `chunk_size` alignment, `engine_computed_tokens` hints, and minimum novelty
/// thresholds.
pub fn evaluate_kv_load_policy(ctx: &KvLoadContext) -> KvLoadDecision {
    if ctx.skip_load {
        return KvLoadDecision {
            should_load: false,
            effective_novel_tokens: 0,
            reason: "skip_load_override",
        };
    }
    if ctx.chunk_size == 0 || ctx.num_lookup_tokens < ctx.chunk_size {
        return KvLoadDecision {
            should_load: false,
            effective_novel_tokens: 0,
            reason: "below_chunk_size",
        };
    }
    let already_computed = ctx
        .engine_computed_tokens
        .unwrap_or(0)
        .min(ctx.num_lookup_tokens);
    let novel_tokens = ctx.num_lookup_tokens - already_computed;
    let min_required = ctx.min_novel_tokens_threshold.max(ctx.chunk_size);
    if novel_tokens < min_required {
        return KvLoadDecision {
            should_load: false,
            effective_novel_tokens: novel_tokens,
            reason: "recompute_cheaper_than_load",
        };
    }
    KvLoadDecision {
        should_load: true,
        effective_novel_tokens: novel_tokens,
        reason: "load_approved",
    }
}

/// 4-byte wire magic (`"CFOG"`) for [`ObjectGroupFrame`] multi-field object group transfers.
pub const OBJECT_GROUP_FRAME_MAGIC: [u8; 4] = *b"CFOG";

/// Typed tensor field within a multi-field KV object (`ObjectGroupFrame`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectFieldSpec {
    /// Field name (e.g., `"k"`, `"v"`, `"scales"`).
    pub name: String,
    /// Element data type.
    pub dtype: DType,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Raw little-endian payload bytes (`shape.product() * dtype.size_in_bytes()`).
    pub data: Vec<u8>,
}

impl ObjectFieldSpec {
    /// Expected byte length from `shape` and `dtype.bit_width()`.
    pub fn expected_byte_len(&self) -> Option<usize> {
        let elem_bytes = self.dtype.bit_width().div_ceil(8).max(1);
        let mut total = elem_bytes;
        for &d in &self.shape {
            total = total.checked_mul(d)?;
        }
        Some(total)
    }
}

/// Homogeneous batch of multi-field KV objects transferred as a single framed payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectGroupFrame {
    /// Group identifier (e.g., layer-group or attention-group label).
    pub group_id: String,
    /// Per-object list of [`ObjectFieldSpec`] fields. All objects in a non-empty group must share
    /// the same field count, field names, and field dtypes.
    pub objects: Vec<Vec<ObjectFieldSpec>>,
}

fn dtype_to_wire_code(dt: &DType) -> u8 {
    match dt {
        DType::U8 => 1,
        DType::I8 => 2,
        DType::U16 => 3,
        DType::I16 => 4,
        DType::U32 => 5,
        DType::I32 => 6,
        DType::U64 => 7,
        DType::I64 => 8,
        DType::F16 => 9,
        DType::BF16 => 10,
        DType::F32 => 11,
        DType::F64 => 12,
        DType::F8E4M3FN => 13,
        DType::F8E5M2 => 14,
        DType::I1 => 15,
        _ => 255,
    }
}

fn wire_code_to_dtype(code: u8) -> Option<DType> {
    match code {
        1 => Some(DType::U8),
        2 => Some(DType::I8),
        3 => Some(DType::U16),
        4 => Some(DType::I16),
        5 => Some(DType::U32),
        6 => Some(DType::I32),
        7 => Some(DType::U64),
        8 => Some(DType::I64),
        9 => Some(DType::F16),
        10 => Some(DType::BF16),
        11 => Some(DType::F32),
        12 => Some(DType::F64),
        13 => Some(DType::F8E4M3FN),
        14 => Some(DType::F8E5M2),
        15 => Some(DType::I1),
        _ => None,
    }
}

impl ObjectGroupFrame {
    /// Validates that every field's byte length matches its `(dtype, shape)` and that all objects
    /// in the group share an identical field schema `(name, dtype)`.
    pub fn validate(&self) -> Result<(), String> {
        let Some(first_obj) = self.objects.first() else {
            return Ok(());
        };
        if first_obj.is_empty() {
            return Err("object in ObjectGroupFrame must have at least one field".to_string());
        }
        for (obj_idx, obj) in self.objects.iter().enumerate() {
            if obj.len() != first_obj.len() {
                return Err(format!(
                    "object {obj_idx} has {} fields, expected {}",
                    obj.len(),
                    first_obj.len()
                ));
            }
            for (f_idx, (field, schema)) in obj.iter().zip(first_obj.iter()).enumerate() {
                if field.name != schema.name || field.dtype != schema.dtype {
                    return Err(format!(
                        "object {obj_idx} field {f_idx} schema ({}, {:?}) mismatches ({}, {:?})",
                        field.name, field.dtype, schema.name, schema.dtype
                    ));
                }
                if dtype_to_wire_code(&field.dtype) == 255 {
                    return Err(format!("unsupported wire dtype {:?}", field.dtype));
                }
                let expected = field
                    .expected_byte_len()
                    .ok_or_else(|| format!("shape product overflow on field '{}'", field.name))?;
                if field.data.len() != expected {
                    return Err(format!(
                        "object {obj_idx} field '{}' data length {} != expected {}",
                        field.name,
                        field.data.len(),
                        expected
                    ));
                }
            }
        }
        Ok(())
    }

    /// Serializes this [`ObjectGroupFrame`] into a self-describing binary frame.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut buf = Vec::new();
        buf.extend_from_slice(&OBJECT_GROUP_FRAME_MAGIC);
        let gid_bytes = self.group_id.as_bytes();
        buf.extend_from_slice(&(gid_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(gid_bytes);
        buf.extend_from_slice(&(self.objects.len() as u32).to_le_bytes());
        let num_fields = self.objects.first().map_or(0, Vec::len);
        buf.extend_from_slice(&(num_fields as u32).to_le_bytes());

        for obj in &self.objects {
            for field in obj {
                let name_bytes = field.name.as_bytes();
                buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(name_bytes);
                buf.push(dtype_to_wire_code(&field.dtype));
                buf.extend_from_slice(&(field.shape.len() as u32).to_le_bytes());
                for &dim in &field.shape {
                    buf.extend_from_slice(&(dim as u64).to_le_bytes());
                }
                buf.extend_from_slice(&(field.data.len() as u64).to_le_bytes());
                buf.extend_from_slice(&field.data);
            }
        }
        Ok(buf)
    }

    /// Deserializes and validates an [`ObjectGroupFrame`] from `bytes`.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = 0usize;
        let read_slice = |cur: &mut usize, len: usize| -> Result<&[u8], String> {
            let end = cur
                .checked_add(len)
                .ok_or_else(|| "cursor overflow".to_string())?;
            if end > bytes.len() {
                return Err("truncated ObjectGroupFrame payload".to_string());
            }
            let s = &bytes[*cur..end];
            *cur = end;
            Ok(s)
        };
        let read_u32 = |cur: &mut usize| -> Result<u32, String> {
            let s = read_slice(cur, 4)?;
            let mut arr = [0u8; 4];
            arr.copy_from_slice(s);
            Ok(u32::from_le_bytes(arr))
        };
        let read_u64 = |cur: &mut usize| -> Result<u64, String> {
            let s = read_slice(cur, 8)?;
            let mut arr = [0u8; 8];
            arr.copy_from_slice(s);
            Ok(u64::from_le_bytes(arr))
        };

        let magic = read_slice(&mut cursor, 4)?;
        if magic != OBJECT_GROUP_FRAME_MAGIC {
            return Err("invalid ObjectGroupFrame magic".to_string());
        }
        let gid_len = read_u32(&mut cursor)? as usize;
        let gid_bytes = read_slice(&mut cursor, gid_len)?;
        let group_id = std::str::from_utf8(gid_bytes)
            .map_err(|e| format!("invalid utf8 group_id: {e}"))?
            .to_string();
        let num_objects = read_u32(&mut cursor)? as usize;
        let num_fields = read_u32(&mut cursor)? as usize;

        let mut objects = Vec::with_capacity(num_objects);
        for _ in 0..num_objects {
            let mut fields = Vec::with_capacity(num_fields);
            for _ in 0..num_fields {
                let name_len = read_u32(&mut cursor)? as usize;
                let name_bytes = read_slice(&mut cursor, name_len)?;
                let name = std::str::from_utf8(name_bytes)
                    .map_err(|e| format!("invalid utf8 field name: {e}"))?
                    .to_string();
                let dt_code = read_slice(&mut cursor, 1)?[0];
                let dtype = wire_code_to_dtype(dt_code)
                    .ok_or_else(|| format!("unknown wire dtype code {dt_code}"))?;
                let rank = read_u32(&mut cursor)? as usize;
                let mut shape = Vec::with_capacity(rank);
                for _ in 0..rank {
                    shape.push(read_u64(&mut cursor)? as usize);
                }
                let data_len = read_u64(&mut cursor)? as usize;
                let data = read_slice(&mut cursor, data_len)?.to_vec();
                fields.push(ObjectFieldSpec {
                    name,
                    dtype,
                    shape,
                    data,
                });
            }
            objects.push(fields);
        }
        if cursor != bytes.len() {
            return Err("trailing bytes in ObjectGroupFrame".to_string());
        }
        let frame = Self { group_id, objects };
        frame.validate()?;
        Ok(frame)
    }
}
