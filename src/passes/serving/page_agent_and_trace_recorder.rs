//! Dynamic page-aligned L1 storage agent, two-level chunk-hash-sharded object
//! paths, exactly-once no-lock warm prefetch service, and length-prefixed
//! binary trace recording with transfer phase timing sampling.
//!
//! Covers:
//! 1. **Dynamic Page-Aligned L1 Storage Agent & Sharded Paths
//!    (`storage_object_key_to_sharded_relpath`, `DynamicPageStorageAgent`)**:
//!    Derives deterministic two-level chunk-hash-sharded relative paths
//!    (`<hex[0..2]>/<hex[2..4]>/<model>--<rank>_<group>_<hash>[@<salt>].bin`)
//!    where the directory shard depends only on `chunk_hash` while `cache_salt`
//!    suffixes the filename; validates page-aligned L1 ranges, classifies
//!    memory as `"DRAM"` (`cpu`) vs `"VRAM"`, and executes the two-stage
//!    non-blocking transfer loop (up to 20 fast spin-checks before sleep-first
//!    backoff polling).
//! 2. **Exactly-Once No-Lock Warm Prefetch Job Table & Service
//!    (`WarmPrefetchStateTable`, `WarmPrefetchService`)**:
//!    Enforces `L2 -> L1` warm-prefetch direction, validates registered model
//!    layout descriptors, tracks in-flight `NO_LOCK` retain prefetch handles,
//!    and implements **exactly-once** completion consumption on `poll`
//!    (`Pending` -> `Completed { found_keys, total_keys }` on first completed
//!    poll -> `Unknown` on subsequent polls).
//! 3. **Length-Prefixed Binary Trace Recorder, Tagged Argument Codec & Phase
//!    Timing Sampler (`TraceValue`, `encode_trace_value`, `decode_trace_value`,
//!    `BinaryTraceRecorder`, `TransferPhaseTimingSampler`)**:
//!    Encodes/decodes primitive values, tuples, and typed domain objects via
//!    `{"__t__": tag, "v": payload}` envelopes, lazily writes the 4-byte
//!    big-endian length-prefixed header with SHA-256 config digest on the first
//!    of `attach_storage_config`, `write_record`, or `close()`, and pops GPU
//!    gather/DMA phase timing samples on `MP_STORE_END` / `MP_RETRIEVE_END`
//!    (emitting `MP_TRANSFER_PHASE_SAMPLES` even when the sample batch is empty).

use std::collections::{BTreeMap, HashMap, VecDeque};

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// 1. Dynamic Page-Aligned L1 Storage Agent & Two-Level Sharded Paths
// ---------------------------------------------------------------------------

/// Key identifying a stored KV cache object in dynamic L2 storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StorageObjectKey {
    /// Model identifier (`/` replaced by `--` in filenames).
    pub model_name: String,
    /// Tensor/pipeline parallel KV rank.
    pub kv_rank: u32,
    /// Layer/object group index.
    pub object_group_id: u32,
    /// Lowercase hex chunk hash (at least 4 hex chars).
    pub chunk_hash_hex: String,
    /// Optional tenant/session cache salt.
    pub cache_salt: String,
}

/// Formats `key` into a deterministic storage filename
/// (`"<model>--<rank>_<group>_<hash>[@<salt>].bin"`).
#[must_use]
pub fn object_key_to_filename(key: &StorageObjectKey) -> String {
    let safe_model = key.model_name.replace('/', "--");
    let suffix = if key.cache_salt.is_empty() {
        String::new()
    } else {
        format!("@{}", key.cache_salt)
    };
    format!(
        "{safe_model}_{}_{}_{}{suffix}.bin",
        key.kv_rank, key.object_group_id, key.chunk_hash_hex
    )
}

/// Formats `key` into a two-level chunk-hash-sharded relative path
/// (`"<hex[0..2]>/<hex[2..4]>/<filename>"`).
///
/// The directory shard depends only on `chunk_hash_hex`; salted keys with the
/// same chunk hash reside in the same shard with distinct filenames.
///
/// # Errors
///
/// Returns `Err` if `chunk_hash_hex` has fewer than 4 ASCII hex characters.
pub fn storage_object_key_to_sharded_relpath(key: &StorageObjectKey) -> Result<String, String> {
    if key.chunk_hash_hex.len() < 4 || !key.chunk_hash_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "chunk_hash_hex '{}' must contain at least 4 hex characters",
            key.chunk_hash_hex
        ));
    }
    let l1 = &key.chunk_hash_hex[0..2];
    let l2 = &key.chunk_hash_hex[2..4];
    let filename = object_key_to_filename(key);
    Ok(format!("{l1}/{l2}/{filename}"))
}

/// Poll state of a non-blocking NIXL-style page transfer handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageXferState {
    /// Transfer is still in progress (`"PROC"`).
    InProgress,
    /// Transfer completed (`"DONE"`).
    Done,
    /// Transfer failed (`"ERR"`).
    Error,
}

/// Telemetry returned by [`DynamicPageStorageAgent::post_non_blocking`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonBlockingXferReport {
    /// Number of fast spin-checks performed (at most 20).
    pub spin_checks: usize,
    /// Number of 1 ms backoff sleeps performed before subsequent checks.
    pub backoff_sleeps: usize,
}

/// Dynamic page-aligned L1 storage agent managing page registration, `"DRAM"` vs
/// `"VRAM"` classification, and two-stage spin-then-backoff transfer polling.
#[derive(Debug, Clone)]
pub struct DynamicPageStorageAgent {
    device: String,
    backend: String,
    mem_type: &'static str,
    l1_base_ptr: u64,
    l1_size_bytes: u64,
    l1_align_bytes: u64,
    registered_pages: usize,
    objects: HashMap<String, Vec<Vec<u8>>>,
    closed: bool,
}

impl DynamicPageStorageAgent {
    /// Initializes a dynamic page-aligned storage agent and registers L1 memory
    /// descriptors (`"DRAM"` when `device == "cpu"`, otherwise `"VRAM"`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `l1_align_bytes == 0`, `l1_size_bytes == 0`, or
    /// `l1_base_ptr` / `l1_size_bytes` is not aligned to `l1_align_bytes`.
    pub fn new(
        device: &str,
        backend: &str,
        l1_base_ptr: u64,
        l1_size_bytes: u64,
        l1_align_bytes: u64,
    ) -> Result<Self, String> {
        if l1_align_bytes == 0 {
            return Err("l1_align_bytes must be greater than zero".to_string());
        }
        if l1_size_bytes == 0 || !l1_size_bytes.is_multiple_of(l1_align_bytes) {
            return Err(format!(
                "L1 buffer size {l1_size_bytes} is not a positive multiple of page \
                 size {l1_align_bytes}"
            ));
        }
        if !l1_base_ptr.is_multiple_of(l1_align_bytes) {
            return Err(format!(
                "L1 base address {l1_base_ptr} is not aligned to page size {l1_align_bytes}"
            ));
        }
        let mem_type = if device == "cpu" { "DRAM" } else { "VRAM" };
        let registered_pages = (l1_size_bytes / l1_align_bytes) as usize;
        Ok(Self {
            device: device.to_string(),
            backend: backend.to_string(),
            mem_type,
            l1_base_ptr,
            l1_size_bytes,
            l1_align_bytes,
            registered_pages,
            objects: HashMap::new(),
            closed: false,
        })
    }

    /// Returns `"DRAM"` for CPU agents and `"VRAM"` for accelerator agents.
    #[must_use]
    pub const fn mem_type(&self) -> &'static str {
        self.mem_type
    }

    /// Returns the device name.
    #[must_use]
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Returns the storage backend name.
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Returns the number of registered L1 pages.
    #[must_use]
    pub const fn registered_pages(&self) -> usize {
        self.registered_pages
    }

    /// Computes the registered page indices covering `[raw_addr, raw_addr + mem_size)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `raw_addr` or `mem_size` is not aligned to
    /// `l1_align_bytes`, or if `mem_size == 0`.
    pub fn get_memory_indices(&self, raw_addr: u64, mem_size: u64) -> Result<Vec<u64>, String> {
        if !raw_addr.is_multiple_of(self.l1_align_bytes) {
            return Err(format!(
                "Raw address {raw_addr} is not aligned to page size {}",
                self.l1_align_bytes
            ));
        }
        if mem_size == 0 || !mem_size.is_multiple_of(self.l1_align_bytes) {
            return Err(format!(
                "Memory size {mem_size} is not a multiple of page size {}",
                self.l1_align_bytes
            ));
        }
        let arena_end = self.l1_base_ptr.saturating_add(self.l1_size_bytes);
        let range_end = raw_addr.saturating_add(mem_size);
        if raw_addr < self.l1_base_ptr || range_end > arena_end {
            return Err("Memory range lies outside registered L1 arena".to_string());
        }
        let num_pages = mem_size / self.l1_align_bytes;
        let base_page = raw_addr / self.l1_align_bytes;
        Ok((0..num_pages).map(|i| base_page + i).collect())
    }

    /// Executes the two-stage non-blocking completion protocol given an
    /// `initial_state` and a sequence of subsequent poll states:
    /// 1. Fast path: up to 20 spin-checks without sleeping.
    /// 2. Back-off loop: sleeps *before* each `check_xfer_state` call so a
    ///    transfer completing on the first poll exits without an extra sleep.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the transfer enters [`PageXferState::Error`] or exhausts
    /// the provided state stream before reaching `Done`.
    pub fn post_non_blocking(
        &self,
        initial_state: PageXferState,
        poll_sequence: &[PageXferState],
    ) -> Result<NonBlockingXferReport, String> {
        let mut state = initial_state;
        let mut idx = 0usize;
        let mut spin_checks = 0usize;
        let mut backoff_sleeps = 0usize;

        if state != PageXferState::Done && state != PageXferState::Error {
            for _ in 0..20 {
                let Some(&next_state) = poll_sequence.get(idx) else {
                    return Err("transfer state stream exhausted during spin-check".to_string());
                };
                idx += 1;
                spin_checks += 1;
                state = next_state;
                if state == PageXferState::Done || state == PageXferState::Error {
                    break;
                }
            }
        }

        while state != PageXferState::Done && state != PageXferState::Error {
            backoff_sleeps += 1;
            let Some(&next_state) = poll_sequence.get(idx) else {
                return Err("transfer state stream exhausted during backoff".to_string());
            };
            idx += 1;
            state = next_state;
        }

        if state == PageXferState::Error {
            return Err("NIXL transfer failed".to_string());
        }
        Ok(NonBlockingXferReport {
            spin_checks,
            backoff_sleeps,
        })
    }

    /// Stores `pages` under the sharded relative path of `key`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the agent is closed or `key` has an invalid chunk hash.
    pub fn dynamic_store(
        &mut self,
        key: &StorageObjectKey,
        pages: Vec<Vec<u8>>,
    ) -> Result<String, String> {
        if self.closed {
            return Err("storage agent is closed".to_string());
        }
        let relpath = storage_object_key_to_sharded_relpath(key)?;
        self.objects.insert(relpath.clone(), pages);
        Ok(relpath)
    }

    /// Returns the total stored size in bytes for `key`, or `None` if absent.
    #[must_use]
    pub fn get_stored_size(&self, key: &StorageObjectKey) -> Option<usize> {
        let relpath = storage_object_key_to_sharded_relpath(key).ok()?;
        let pages = self.objects.get(&relpath)?;
        Some(pages.iter().map(Vec::len).sum())
    }

    /// Deletes the stored object for `key`.
    pub fn dynamic_delete(&mut self, key: &StorageObjectKey) -> bool {
        let Ok(relpath) = storage_object_key_to_sharded_relpath(key) else {
            return false;
        };
        self.objects.remove(&relpath).is_some()
    }

    /// Releases descriptor lists and deregisters L1 memory.
    pub fn close(&mut self) {
        self.objects.clear();
        self.closed = true;
    }
}

// ---------------------------------------------------------------------------
// 2. Exactly-Once No-Lock Warm Prefetch Job Table & Service
// ---------------------------------------------------------------------------

/// Outcome of polling a warm-prefetch request in [`WarmPrefetchStateTable`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarmPrefetchOnceStatus {
    /// Load is still running in the background.
    Pending,
    /// Load finished; the job is atomically consumed and dropped on this first
    /// completed poll.
    Completed {
        /// Number of keys loaded into L1.
        found_keys: usize,
        /// Total keys originally requested.
        total_keys: usize,
    },
    /// Unknown request ID (already completed-and-consumed on a prior poll, or
    /// never submitted).
    Unknown,
}

#[derive(Debug, Clone)]
struct WarmPrefetchEntry {
    total_requested_keys: usize,
    completed_hit_popcounts: Option<Vec<usize>>,
}

/// Table of in-flight `NO_LOCK` retain warm-prefetch jobs with **exactly-once**
/// completion observation.
#[derive(Debug, Default)]
pub struct WarmPrefetchStateTable {
    jobs: HashMap<String, WarmPrefetchEntry>,
    next_id: u64,
}

impl WarmPrefetchStateTable {
    /// Creates an empty warm-prefetch job table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submits a `NO_LOCK` warm-prefetch job for `total_requested_keys` and
    /// returns an opaque `request_id`.
    pub fn submit(&mut self, total_requested_keys: usize) -> String {
        self.next_id += 1;
        let request_id = format!("warm_req_{:08x}", self.next_id);
        self.jobs.insert(
            request_id.clone(),
            WarmPrefetchEntry {
                total_requested_keys,
                completed_hit_popcounts: None,
            },
        );
        request_id
    }

    /// Marks `request_id` as completed on the storage controller with per-group
    /// hit popcounts.
    pub fn record_completion(&mut self, request_id: &str, hit_popcounts: Vec<usize>) -> bool {
        if let Some(entry) = self.jobs.get_mut(request_id) {
            entry.completed_hit_popcounts = Some(hit_popcounts);
            true
        } else {
            false
        }
    }

    /// Polls `request_id`. The first poll that observes completion drops the
    /// job and returns [`WarmPrefetchOnceStatus::Completed`]; any subsequent
    /// poll for the same ID returns [`WarmPrefetchOnceStatus::Unknown`].
    pub fn poll(&mut self, request_id: &str) -> WarmPrefetchOnceStatus {
        let Some(entry) = self.jobs.get(request_id) else {
            return WarmPrefetchOnceStatus::Unknown;
        };
        let Some(popcounts) = &entry.completed_hit_popcounts else {
            return WarmPrefetchOnceStatus::Pending;
        };
        let found_keys: usize = popcounts.iter().sum();
        let total_keys = entry.total_requested_keys;
        self.jobs.remove(request_id);
        WarmPrefetchOnceStatus::Completed {
            found_keys,
            total_keys,
        }
    }
}

/// Response returned by [`WarmPrefetchService::submit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarmPrefetchSubmitResult {
    /// Sub-chunk token sequence (`chunks == 0`); no prefetch job was needed.
    Noop {
        /// Always `0`.
        chunks: usize,
    },
    /// Prefetch job submitted (`status == "submitted"`).
    Submitted {
        /// Assigned request identifier.
        request_id: String,
        /// Number of full token chunks submitted across registered object groups.
        chunks: usize,
    },
}

/// Node-local warm-prefetch service enforcing `L2 -> L1` direction and
/// registered model object-group layouts.
#[derive(Debug, Default)]
pub struct WarmPrefetchService {
    /// Registered `(model_name, world_size) -> (chunk_size, num_object_groups)`.
    registered_models: HashMap<(String, u32), (usize, usize)>,
    jobs: WarmPrefetchStateTable,
}

impl WarmPrefetchService {
    /// Creates a new warm-prefetch service.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers layout/attention/replication descriptors for `(model_name, world_size)`.
    pub fn register_model_layout(
        &mut self,
        model_name: &str,
        world_size: u32,
        chunk_size: usize,
        num_object_groups: usize,
    ) {
        self.registered_models.insert(
            (model_name.to_string(), world_size),
            (chunk_size.max(1), num_object_groups.max(1)),
        );
    }

    /// Submits a warm prefetch of `token_ids` from `source_tier` (`"l2"`) into
    /// `target_tier` (`"l1"`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the direction is not `"l2" -> "l1"`, if no layout is
    /// registered for `(model_name, world_size)`, or if `token_ids.len() > 1_000_000`.
    pub fn submit(
        &mut self,
        model_name: &str,
        world_size: u32,
        token_ids: &[u32],
        source_tier: &str,
        target_tier: &str,
    ) -> Result<WarmPrefetchSubmitResult, String> {
        if !source_tier.eq_ignore_ascii_case("l2") || !target_tier.eq_ignore_ascii_case("l1") {
            return Err(format!(
                "unsupported prefetch direction '{source_tier}'->'{target_tier}'; only 'l2'->'l1'"
            ));
        }
        if token_ids.len() > 1_000_000 {
            return Err("token_ids length exceeds maximum 1000000".to_string());
        }
        let Some(&(chunk_size, num_groups)) = self
            .registered_models
            .get(&(model_name.to_string(), world_size))
        else {
            return Err(format!(
                "no layout registered for model_name='{model_name}' world_size={world_size}; \
                 the model has not allocated KV cache on this node yet"
            ));
        };
        let chunks = token_ids.len() / chunk_size;
        if chunks == 0 {
            return Ok(WarmPrefetchSubmitResult::Noop { chunks: 0 });
        }
        let total_keys = chunks * (world_size as usize) * num_groups;
        let request_id = self.jobs.submit(total_keys);
        Ok(WarmPrefetchSubmitResult::Submitted { request_id, chunks })
    }

    /// Completes an in-flight warm-prefetch request with `hit_popcounts`.
    pub fn complete_job(&mut self, request_id: &str, hit_popcounts: Vec<usize>) -> bool {
        self.jobs.record_completion(request_id, hit_popcounts)
    }

    /// Polls `request_id` and returns its current status, or `Err` (`NotFound`)
    /// if the request is unknown or was already consumed on an earlier poll.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `poll` reports [`WarmPrefetchOnceStatus::Unknown`].
    pub fn status(&mut self, request_id: &str) -> Result<WarmPrefetchOnceStatus, String> {
        match self.jobs.poll(request_id) {
            WarmPrefetchOnceStatus::Unknown => Err(format!(
                "unknown prefetch request_id='{request_id}' (already completed or never submitted)"
            )),
            other => Ok(other),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Length-Prefixed Binary Trace Recorder & Transfer Phase Timing Sampler
// ---------------------------------------------------------------------------

/// Magic header prefix for binary trace files.
pub const TRACE_FILE_MAGIC: &[u8; 4] = b"CFTR";

/// Trace value representation supporting native primitives, tuples, dicts, and
/// tagged custom domain codecs (`{"__t__": tag, "v": payload}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceValue {
    /// Null/None value.
    Null,
    /// Boolean scalar.
    Bool(bool),
    /// Signed 64-bit integer.
    Int(i64),
    /// UTF-8 string.
    Str(String),
    /// Ordered list of trace values.
    List(Vec<TraceValue>),
    /// Ordered tuple preserved across serialization via `{"__t__": "tuple", "v": [...]}`.
    Tuple(Vec<TraceValue>),
    /// String-keyed dictionary.
    Dict(BTreeMap<String, TraceValue>),
    /// Typed domain object encoded as `{"__t__": tag, "v": fields}`.
    TaggedObject {
        /// Registered type tag (e.g., `"ObjectKey"`, `"MemoryLayoutDesc"`,
        /// `"PrefetchHandle"`, `"AttnWindowDesc"`).
        tag: String,
        /// Encoded payload fields.
        payload: BTreeMap<String, TraceValue>,
    },
}

/// Encodes a [`TraceValue`] into its canonical wire representation where
/// [`TraceValue::Tuple`] and [`TraceValue::TaggedObject`] are wrapped in
/// `{"__t__": tag, "v": ...}` dictionaries.
#[must_use]
pub fn encode_trace_value(val: &TraceValue) -> TraceValue {
    match val {
        TraceValue::Null | TraceValue::Bool(_) | TraceValue::Int(_) | TraceValue::Str(_) => {
            val.clone()
        }
        TraceValue::List(items) => TraceValue::List(items.iter().map(encode_trace_value).collect()),
        TraceValue::Tuple(items) => {
            let mut wrapper = BTreeMap::new();
            wrapper.insert("__t__".to_string(), TraceValue::Str("tuple".to_string()));
            wrapper.insert(
                "v".to_string(),
                TraceValue::List(items.iter().map(encode_trace_value).collect()),
            );
            TraceValue::Dict(wrapper)
        }
        TraceValue::Dict(map) => {
            let mut out = BTreeMap::new();
            for (k, v) in map {
                out.insert(k.clone(), encode_trace_value(v));
            }
            TraceValue::Dict(out)
        }
        TraceValue::TaggedObject { tag, payload } => {
            let mut inner = BTreeMap::new();
            for (k, v) in payload {
                inner.insert(k.clone(), encode_trace_value(v));
            }
            let mut wrapper = BTreeMap::new();
            wrapper.insert("__t__".to_string(), TraceValue::Str(tag.clone()));
            wrapper.insert("v".to_string(), TraceValue::Dict(inner));
            TraceValue::Dict(wrapper)
        }
    }
}

/// Decodes a wire-encoded [`TraceValue`] back into its native [`TraceValue`]
/// form, recognizing `{"__t__": "tuple", "v": [...]}` and registered domain
/// tags (`"ObjectKey"`, `"MemoryLayoutDesc"`, `"PrefetchHandle"`,
/// `"AttnWindowDesc"`).
///
/// # Errors
///
/// Returns `Err` if a wrapped dictionary carries an unknown `__t__` tag.
pub fn decode_trace_value(val: &TraceValue) -> Result<TraceValue, String> {
    match val {
        TraceValue::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(decode_trace_value(item)?);
            }
            Ok(TraceValue::List(out))
        }
        TraceValue::Dict(map) => {
            if let Some(TraceValue::Str(tag)) = map.get("__t__") {
                let Some(raw_v) = map.get("v") else {
                    return Err(format!("trace.codecs: missing 'v' field for tag '{tag}'"));
                };
                if tag == "tuple" {
                    let TraceValue::List(items) = raw_v else {
                        return Err("trace.codecs: tuple payload must be a list".to_string());
                    };
                    let mut decoded = Vec::with_capacity(items.len());
                    for item in items {
                        decoded.push(decode_trace_value(item)?);
                    }
                    return Ok(TraceValue::Tuple(decoded));
                }
                let is_known_tag = matches!(
                    tag.as_str(),
                    "ObjectKey" | "MemoryLayoutDesc" | "PrefetchHandle" | "AttnWindowDesc"
                );
                if !is_known_tag {
                    return Err(format!("trace.codecs: unknown tag '{tag}'"));
                }
                let TraceValue::Dict(inner_map) = raw_v else {
                    return Err(format!(
                        "trace.codecs: payload for tag '{tag}' must be a dict"
                    ));
                };
                let mut decoded_inner = BTreeMap::new();
                for (k, v) in inner_map {
                    decoded_inner.insert(k.clone(), decode_trace_value(v)?);
                }
                return Ok(TraceValue::TaggedObject {
                    tag: tag.clone(),
                    payload: decoded_inner,
                });
            }
            let mut out = BTreeMap::new();
            for (k, v) in map {
                out.insert(k.clone(), decode_trace_value(v)?);
            }
            Ok(TraceValue::Dict(out))
        }
        other => Ok(other.clone()),
    }
}

/// Binary trace recorder that lazily writes a 4-byte big-endian length-prefixed
/// header (with SHA-256 config digest) on the first of `attach_storage_config`,
/// `write_record`, or `close()`, and frames every call record with relative
/// monotonic timestamps.
#[derive(Debug, Clone)]
pub struct BinaryTraceRecorder {
    level: String,
    captures_calls: bool,
    tracing_gate_enabled: bool,
    header_written: bool,
    closed: bool,
    t_mono_start_ns: u64,
    sm_config_json: String,
    sm_config_digest: String,
    dropped_count: usize,
    stream_bytes: Vec<u8>,
}

impl BinaryTraceRecorder {
    /// Creates a new binary trace recorder starting at `t_mono_start_ns`.
    /// When `captures_calls` is `true`, flips `tracing_gate_enabled` on.
    #[must_use]
    pub fn new(level: &str, captures_calls: bool, t_mono_start_ns: u64) -> Self {
        Self {
            level: level.to_string(),
            captures_calls,
            tracing_gate_enabled: captures_calls,
            header_written: false,
            closed: false,
            t_mono_start_ns,
            sm_config_json: String::new(),
            sm_config_digest: String::new(),
            dropped_count: 0,
            stream_bytes: Vec::new(),
        }
    }

    /// Returns whether the global call-tracing gate is currently enabled.
    #[must_use]
    pub const fn tracing_gate_enabled(&self) -> bool {
        self.tracing_gate_enabled
    }

    /// Returns whether the header frame has been written.
    #[must_use]
    pub const fn header_written(&self) -> bool {
        self.header_written
    }

    /// Returns the SHA-256 hex digest of the attached storage configuration.
    #[must_use]
    pub fn sm_config_digest(&self) -> &str {
        &self.sm_config_digest
    }

    /// Returns the number of dropped records.
    #[must_use]
    pub const fn dropped_count(&self) -> usize {
        self.dropped_count
    }

    /// Attaches a sorted-key storage configuration map, computing its SHA-256
    /// digest and writing the header frame once. Subsequent calls are ignored.
    pub fn attach_storage_config(&mut self, config_kv: &BTreeMap<String, String>) {
        if self.closed || self.header_written {
            return;
        }
        let mut json = String::from("{");
        for (idx, (k, v)) in config_kv.iter().enumerate() {
            if idx > 0 {
                json.push(',');
            }
            json.push_str(&format!("\"{k}\":\"{v}\""));
        }
        json.push('}');

        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        let digest_bytes = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest_bytes {
            use std::fmt::Write;
            let _ = write!(&mut hex, "{b:02x}");
        }
        self.sm_config_json = json;
        self.sm_config_digest = hex;
        self.emit_header_frame();
    }

    fn emit_header_frame(&mut self) {
        if self.header_written {
            return;
        }
        let payload = format!(
            "HDR|magic=CFTR|level={}|digest={}|cfg={}",
            self.level, self.sm_config_digest, self.sm_config_json
        );
        self.write_len_prefixed_frame(payload.as_bytes());
        self.header_written = true;
    }

    fn write_len_prefixed_frame(&mut self, frame: &[u8]) {
        let len_be = (frame.len() as u32).to_be_bytes();
        self.stream_bytes.extend_from_slice(&len_be);
        self.stream_bytes.extend_from_slice(frame);
    }

    /// Appends a `TRACE_CALL` record with monotonic timestamp `t_mono_ns`
    /// stored relative to `t_mono_start_ns`. If no header has been written yet,
    /// writes the fallback header first.
    pub fn write_record(
        &mut self,
        qualname: &str,
        args: &BTreeMap<String, TraceValue>,
        t_mono_ns: u64,
    ) {
        if self.closed {
            return;
        }
        if qualname.trim().is_empty() {
            self.dropped_count += 1;
            return;
        }
        if !self.header_written {
            self.emit_header_frame();
        }
        let rel_mono_ns = t_mono_ns.saturating_sub(self.t_mono_start_ns);
        let mut encoded_args = BTreeMap::new();
        for (k, v) in args {
            encoded_args.insert(k.clone(), encode_trace_value(v));
        }
        let frame_str = format!("REC|t_rel={rel_mono_ns}|fn={qualname}|args={encoded_args:?}");
        self.write_len_prefixed_frame(frame_str.as_bytes());
    }

    /// Closes the trace recorder idempotently, flipping the tracing gate off
    /// and writing a fallback empty-config header if neither
    /// `attach_storage_config` nor `write_record` was called.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if self.captures_calls {
            self.tracing_gate_enabled = false;
        }
        if !self.header_written {
            self.emit_header_frame();
        }
    }

    /// Parses the recorded byte stream into individual frame strings by reading
    /// each 4-byte big-endian length prefix, tolerating a truncated tail frame.
    #[must_use]
    pub fn read_frames(&self) -> Vec<String> {
        let mut frames = Vec::new();
        let mut offset = 0usize;
        while offset + 4 <= self.stream_bytes.len() {
            let mut len_bytes = [0u8; 4];
            len_bytes.copy_from_slice(&self.stream_bytes[offset..offset + 4]);
            let frame_len = u32::from_be_bytes(len_bytes) as usize;
            offset += 4;
            if offset + frame_len > self.stream_bytes.len() {
                // Gracefully tolerate partial-write tail truncation.
                break;
            }
            if let Ok(s) = std::str::from_utf8(&self.stream_bytes[offset..offset + frame_len]) {
                frames.push(s.to_string());
            }
            offset += frame_len;
        }
        frames
    }
}

/// GPU gather/DMA transfer phase timing sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPhaseSample {
    /// Phase name (e.g., `"gpu_gather"`, `"host_dma"`).
    pub phase_name: String,
    /// Duration in microseconds.
    pub duration_us: u64,
}

/// Published `MP_TRANSFER_PHASE_SAMPLES` event emitted by [`TransferPhaseTimingSampler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferPhaseSamplesEvent {
    /// Transfer key that just finished (`ended_transfer_key`).
    pub ended_transfer_key: String,
    /// Drained phase timing samples (may be empty, which still signals transfer completion).
    pub samples: Vec<TransferPhaseSample>,
}

/// Bridges the native GPU transfer phase-timing queue to the observability
/// event bus on `MP_STORE_END` and `MP_RETRIEVE_END`.
#[derive(Debug, Default)]
pub struct TransferPhaseTimingSampler {
    native_timing_enabled: bool,
    completed_queue: VecDeque<TransferPhaseSample>,
    published_events: Vec<TransferPhaseSamplesEvent>,
}

impl TransferPhaseTimingSampler {
    /// Creates a new phase-timing sampler (`native_timing_enabled` controls
    /// whether `pop_completed_phase_timings` is active).
    #[must_use]
    pub fn new(native_timing_enabled: bool) -> Self {
        Self {
            native_timing_enabled,
            completed_queue: VecDeque::new(),
            published_events: Vec::new(),
        }
    }

    /// Pushes a completed GPU phase timing sample into the native recorder queue.
    pub fn record_native_sample(&mut self, phase_name: &str, duration_us: u64) {
        self.completed_queue.push_back(TransferPhaseSample {
            phase_name: phase_name.to_string(),
            duration_us,
        });
    }

    /// Handles an `MP_STORE_END` or `MP_RETRIEVE_END` event for `transfer_key`,
    /// popping all finished samples and publishing an `MP_TRANSFER_PHASE_SAMPLES`
    /// event even when the popped sample list is empty.
    pub fn on_transfer_end(&mut self, transfer_key: &str) {
        if !self.native_timing_enabled {
            return;
        }
        let samples: Vec<TransferPhaseSample> = self.completed_queue.drain(..).collect();
        self.published_events.push(TransferPhaseSamplesEvent {
            ended_transfer_key: transfer_key.to_string(),
            samples,
        });
    }

    /// Returns all published `MP_TRANSFER_PHASE_SAMPLES` events.
    #[must_use]
    pub fn published_events(&self) -> &[TransferPhaseSamplesEvent] {
        &self.published_events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharded_relpath_and_page_agent() {
        let key_unsalted = StorageObjectKey {
            model_name: "meta/llama-3-70b".to_string(),
            kv_rank: 0,
            object_group_id: 1,
            chunk_hash_hex: "a1b2c3d4".to_string(),
            cache_salt: String::new(),
        };
        let key_salted = StorageObjectKey {
            cache_salt: "tenant_a".to_string(),
            ..key_unsalted.clone()
        };
        let p1 = storage_object_key_to_sharded_relpath(&key_unsalted).expect("relpath 1");
        let p2 = storage_object_key_to_sharded_relpath(&key_salted).expect("relpath 2");
        assert_eq!(p1, "a1/b2/meta--llama-3-70b_0_1_a1b2c3d4.bin");
        assert_eq!(p2, "a1/b2/meta--llama-3-70b_0_1_a1b2c3d4@tenant_a.bin");

        let agent =
            DynamicPageStorageAgent::new("cuda:0", "GDS", 0x1000, 0x4000, 0x1000).expect("agent");
        assert_eq!(agent.mem_type(), "VRAM");
        assert_eq!(agent.get_memory_indices(0x2000, 0x2000), Ok(vec![2, 3]));
        assert!(agent.get_memory_indices(0x2001, 0x1000).is_err());

        // Fast-path spin completion (3 checks, 0 sleeps).
        let rep = agent
            .post_non_blocking(
                PageXferState::InProgress,
                &[
                    PageXferState::InProgress,
                    PageXferState::InProgress,
                    PageXferState::Done,
                ],
            )
            .expect("spin done");
        assert_eq!(
            rep,
            NonBlockingXferReport {
                spin_checks: 3,
                backoff_sleeps: 0
            }
        );
    }

    #[test]
    fn test_warm_prefetch_exactly_once_and_trace_recorder() {
        let mut svc = WarmPrefetchService::new();
        svc.register_model_layout("qwen-72b", 4, 4, 2);

        // Sub-chunk sequence -> Noop
        assert_eq!(
            svc.submit("qwen-72b", 4, &[1, 2, 3], "l2", "l1"),
            Ok(WarmPrefetchSubmitResult::Noop { chunks: 0 })
        );

        let WarmPrefetchSubmitResult::Submitted { request_id, chunks } = svc
            .submit("qwen-72b", 4, &[1, 2, 3, 4, 5, 6, 7, 8], "l2", "l1")
            .expect("submit")
        else {
            panic!("expected Submitted");
        };
        assert_eq!(chunks, 2);
        assert_eq!(svc.status(&request_id), Ok(WarmPrefetchOnceStatus::Pending));

        svc.complete_job(&request_id, vec![6, 8]);
        assert_eq!(
            svc.status(&request_id),
            Ok(WarmPrefetchOnceStatus::Completed {
                found_keys: 14,
                total_keys: 16,
            })
        );
        // Second poll after completion returns Err (Unknown/NotFound).
        assert!(svc.status(&request_id).is_err());

        // Trace codec & recorder test
        let val = TraceValue::Tuple(vec![
            TraceValue::Int(42),
            TraceValue::TaggedObject {
                tag: "ObjectKey".to_string(),
                payload: BTreeMap::from([(
                    "model_name".to_string(),
                    TraceValue::Str("qwen".to_string()),
                )]),
            },
        ]);
        let wire = encode_trace_value(&val);
        assert_eq!(decode_trace_value(&wire), Ok(val));

        let mut rec = BinaryTraceRecorder::new("storage", true, 1_000);
        assert!(rec.tracing_gate_enabled());
        assert!(!rec.header_written());
        let mut cfg = BTreeMap::new();
        cfg.insert("l1_gb".to_string(), "64".to_string());
        rec.attach_storage_config(&cfg);
        assert!(rec.header_written());
        rec.write_record("storage.put", &BTreeMap::new(), 1_500);
        rec.close();
        assert!(!rec.tracing_gate_enabled());
        assert_eq!(rec.read_frames().len(), 2);
    }
}
