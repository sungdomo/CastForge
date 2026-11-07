//! Eviction-aware lazy KV cache offload policy, request generation registry,
//! and contiguous store metadata coalescing.
//!
//! Covers:
//! 1. **Request Generation & Orphaned Store-Batch Registry
//!    (`RequestPhase`, `SubmittedStoreBatch`, `LazyOffloadRequestRegistry`)**:
//!    Tracks reusable `request_id` slots across multiple generations
//!    (`Active`, `Finished`), separating preemption resets (`reset`) and
//!    finished-ID reuse (`arrive` on a `Finished` slot) from active execution
//!    by marking any in-flight `SubmittedStoreBatch` as `orphaned = true` so a
//!    late worker failure on an orphaned batch still releases GPU block pins
//!    without breaking the current generation's prefix chain.
//! 2. **Contiguous Store Metadata Coalescing (`coalesce_contiguous_store_ops`)**:
//!    Merges a request's prefix-ordered `STORE` operations into a single
//!    coalesced range `[first.start, last.end)` with per-cache-group
//!    concatenated `block_ids`, rejecting gaps (`meta.start != expected_start`)
//!    or mid-batch cache-group count changes.
//! 3. **Eviction-Aware Lazy Store Queue
//!    (`EvictionAwarePolicyConfig`, `LazyOffloadCounters`,
//!    `EvictionAwareStoreQueue`)**:
//!    Smooths per-step GPU block consumption with an EMA ($\alpha = 0.3$) and
//!    computes the free-queue *danger depth*
//!    ($\lceil \max(\text{EMA}, \text{est\_next}) \times \text{horizon\_steps} \rceil$,
//!    or `0` when $< 0.5$ so an idle engine never drains). Validates
//!    admission-time block-hash snapshots, drops any evicted suffix and marks
//!    the request's prefix chain broken on hash mismatch, prioritizes imminent
//!    free-queue evictions ahead of deferral-deadline (`OVERDUE_RANK = 1 << 62`)
//!    backlogs up to `max_drain_per_step`, and maintains a closed accounting
//!    ledger (`admitted == pending + emitted + all dropped/rejected counters`).

use std::collections::{BTreeMap, HashMap, HashSet};

use indexmap::IndexMap;

/// Default scheduler-step horizon for imminent free-queue eviction (`2.5`).
pub const DEFAULT_HORIZON_STEPS: f64 = 2.5;

/// Exponential moving average smoothing factor for per-step block consumption (`0.3`).
pub const BLOCK_CONSUMPTION_EMA_ALPHA: f64 = 0.3;

/// Rank assigned to requests released solely by the deferral deadline (`1 << 62`),
/// sorting behind every real free-queue rank so deadline backlogs never starve
/// blocks about to be recycled.
pub const OVERDUE_RANK: usize = 1usize << 62;

// ---------------------------------------------------------------------------
// 1. Request Generation & Orphaned Store-Batch Registry
// ---------------------------------------------------------------------------

/// Lifecycle phase of the request generation currently holding a `request_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Request generation is actively running.
    Active,
    /// Request generation has finished on the scheduler.
    Finished,
}

/// GPU blocks pinned by one submitted store batch awaiting worker receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmittedStoreBatch {
    /// GPU block IDs pinned at submission time and unpinned upon completion.
    pub block_ids: Vec<u64>,
    /// Whether a preemption reset or ID reuse detached this batch from the
    /// generation now holding the `request_id`.
    pub orphaned: bool,
}

#[derive(Debug, Clone)]
struct RequestSlot {
    phase: RequestPhase,
    in_flight: Option<SubmittedStoreBatch>,
}

/// Registry owning request generation lifecycle state and the single in-flight
/// store batch per `request_id`.
#[derive(Debug, Default, Clone)]
pub struct LazyOffloadRequestRegistry {
    slots: HashMap<String, RequestSlot>,
}

impl LazyOffloadRequestRegistry {
    /// Creates an empty request registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a tracker was created for `request_id`:
    /// - First arrival: creates an `Active` slot.
    /// - Re-arrival after preemption (`reset` already called): stays `Active`.
    /// - Reuse of a `Finished` ID: orphans any outstanding `in_flight` batch
    ///   from the predecessor and transitions to `Active`.
    pub fn arrive(&mut self, request_id: &str) {
        let Some(slot) = self.slots.get_mut(request_id) else {
            self.slots.insert(
                request_id.to_string(),
                RequestSlot {
                    phase: RequestPhase::Active,
                    in_flight: None,
                },
            );
            return;
        };
        if slot.phase == RequestPhase::Finished
            && let Some(batch) = &mut slot.in_flight
        {
            batch.orphaned = true;
        }
        slot.phase = RequestPhase::Active;
    }

    /// Detaches submitted state before a preempted request tracker is recreated,
    /// marking any open `in_flight` batch as `orphaned = true` while keeping
    /// the slot `Active`.
    pub fn reset(&mut self, request_id: &str) {
        let slot = self
            .slots
            .entry(request_id.to_string())
            .or_insert(RequestSlot {
                phase: RequestPhase::Active,
                in_flight: None,
            });
        if let Some(batch) = &mut slot.in_flight {
            batch.orphaned = true;
        }
        slot.phase = RequestPhase::Active;
    }

    /// Marks the generation currently holding `request_id` as `Finished`.
    pub fn finish(&mut self, request_id: &str) {
        let slot = self
            .slots
            .entry(request_id.to_string())
            .or_insert(RequestSlot {
                phase: RequestPhase::Active,
                in_flight: None,
            });
        slot.phase = RequestPhase::Finished;
    }

    /// Records an in-flight submitted store batch and its pinned `block_ids`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `request_id` already has an open in-flight batch.
    pub fn register_batch(&mut self, request_id: &str, block_ids: Vec<u64>) -> Result<(), String> {
        let slot = self
            .slots
            .entry(request_id.to_string())
            .or_insert(RequestSlot {
                phase: RequestPhase::Active,
                in_flight: None,
            });
        if slot.in_flight.is_some() {
            return Err(format!(
                "request '{request_id}' already has an in-flight store batch"
            ));
        }
        slot.in_flight = Some(SubmittedStoreBatch {
            block_ids,
            orphaned: false,
        });
        Ok(())
    }

    /// Clears and returns the in-flight batch once all worker receipts arrive.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `request_id` has no slot or no open in-flight batch.
    pub fn complete_batch(&mut self, request_id: &str) -> Result<SubmittedStoreBatch, String> {
        let slot = self
            .slots
            .get_mut(request_id)
            .ok_or_else(|| format!("unknown request '{request_id}'"))?;
        slot.in_flight
            .take()
            .ok_or_else(|| format!("request '{request_id}' has no in-flight store batch"))
    }

    /// Drops the slot of a finished, settled request after session teardown.
    pub fn session_ended(&mut self, request_id: &str) {
        if self.can_end_session(request_id) {
            self.slots.remove(request_id);
        }
    }

    /// Returns `true` if `request_id` is currently in `Finished` phase.
    #[must_use]
    pub fn is_finished(&self, request_id: &str) -> bool {
        self.slots
            .get(request_id)
            .is_some_and(|s| s.phase == RequestPhase::Finished)
    }

    /// Returns `true` while a batch is open for `request_id` (orphaned or not).
    #[must_use]
    pub fn has_in_flight(&self, request_id: &str) -> bool {
        self.slots
            .get(request_id)
            .is_some_and(|s| s.in_flight.is_some())
    }

    /// Returns `true` if `request_id` has an open batch that was orphaned by a
    /// preemption reset or ID reuse.
    #[must_use]
    pub fn in_flight_is_orphaned(&self, request_id: &str) -> bool {
        self.slots
            .get(request_id)
            .and_then(|s| s.in_flight.as_ref())
            .is_some_and(|b| b.orphaned)
    }

    /// Returns all `request_id`s that currently have an open in-flight batch.
    #[must_use]
    pub fn in_flight_request_ids(&self) -> HashSet<String> {
        self.slots
            .iter()
            .filter_map(|(k, v)| v.in_flight.as_ref().map(|_| k.clone()))
            .collect()
    }

    /// Returns `true` when the generation has finished and no batch is open.
    #[must_use]
    pub fn can_end_session(&self, request_id: &str) -> bool {
        self.slots
            .get(request_id)
            .is_some_and(|s| s.phase == RequestPhase::Finished && s.in_flight.is_none())
    }
}

// ---------------------------------------------------------------------------
// 2. Contiguous Store Metadata Coalescing
// ---------------------------------------------------------------------------

/// Metadata for a single buffered `STORE` slice of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LazyStoreSliceOp {
    /// Request identifier.
    pub request_id: String,
    /// Inclusive start token index.
    pub start: usize,
    /// Exclusive end token index.
    pub end: usize,
    /// Per-cache-group GPU block IDs covering `[start, end)`.
    pub block_ids: Vec<Vec<u64>>,
    /// Serialized token ledger bytes up to `end`.
    pub token_bytes: Vec<u8>,
    /// Optional tenant/session cache salt.
    pub cache_salt: String,
}

/// Coalesces a prefix-ordered slice of `LazyStoreSliceOp` items for a single
/// request into one contiguous `STORE` operation spanning `[first.start, last.end)`
/// with concatenated per-group `block_ids`.
///
/// # Errors
///
/// Returns `Err` if `ops` is empty, has a gap (`op.start != expected_start`),
/// or changes cache-group count mid-batch.
pub fn coalesce_contiguous_store_ops(ops: &[LazyStoreSliceOp]) -> Result<LazyStoreSliceOp, String> {
    let Some(first) = ops.first() else {
        return Err("cannot coalesce an empty store batch".to_string());
    };
    if ops.len() == 1 {
        return Ok(first.clone());
    }
    let last = &ops[ops.len() - 1];
    let mut merged_block_ids: Vec<Vec<u64>> = first.block_ids.clone();
    let mut expected_start = first.end;

    for meta in &ops[1..] {
        if meta.start != expected_start {
            return Err(format!(
                "non-contiguous store ops for request {}: expected start {expected_start}, \
                 got {}",
                first.request_id, meta.start
            ));
        }
        if meta.block_ids.len() != merged_block_ids.len() {
            return Err(format!(
                "cache-group count changed within store batch for request {}",
                first.request_id
            ));
        }
        expected_start = meta.end;
        for (group_idx, group_ids) in meta.block_ids.iter().enumerate() {
            merged_block_ids[group_idx].extend_from_slice(group_ids);
        }
    }

    Ok(LazyStoreSliceOp {
        request_id: first.request_id.clone(),
        start: first.start,
        end: last.end,
        block_ids: merged_block_ids,
        token_bytes: last.token_bytes.clone(),
        cache_salt: first.cache_salt.clone(),
    })
}

// ---------------------------------------------------------------------------
// 3. Eviction-Aware Lazy Store Queue
// ---------------------------------------------------------------------------

/// Validated tunables for [`EvictionAwareStoreQueue`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvictionAwarePolicyConfig {
    /// Scheduler steps of block consumption treated as imminent eviction (`> 0.0`).
    pub horizon_steps: f64,
    /// Maximum buffered store operations emitted in a single `drain` call (`>= 1`).
    pub max_drain_per_step: usize,
    /// Maximum seconds an operation may stay buffered before deadline emission (`>= 0.0`;
    /// `0.0` disables deadline emission).
    pub max_deferral_seconds: f64,
}

impl EvictionAwarePolicyConfig {
    /// Validates and constructs an [`EvictionAwarePolicyConfig`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `horizon_steps <= 0.0`, `max_drain_per_step < 1`, or
    /// `max_deferral_seconds < 0.0`.
    pub fn new(
        horizon_steps: f64,
        max_drain_per_step: usize,
        max_deferral_seconds: f64,
    ) -> Result<Self, String> {
        if !horizon_steps.is_finite() || horizon_steps <= 0.0 {
            return Err(format!("horizon_steps must be > 0, got {horizon_steps}"));
        }
        if max_drain_per_step < 1 {
            return Err(format!(
                "max_drain_per_step must be >= 1, got {max_drain_per_step}"
            ));
        }
        if max_deferral_seconds < 0.0 {
            return Err(format!(
                "max_deferral_seconds must be >= 0.0, got {max_deferral_seconds}"
            ));
        }
        Ok(Self {
            horizon_steps,
            max_drain_per_step,
            max_deferral_seconds,
        })
    }
}

impl Default for EvictionAwarePolicyConfig {
    fn default() -> Self {
        Self {
            horizon_steps: DEFAULT_HORIZON_STEPS,
            max_drain_per_step: 64,
            max_deferral_seconds: 0.0,
        }
    }
}

/// Closed accounting ledger for [`EvictionAwareStoreQueue`]:
/// `admitted == pending + emitted + dropped_evicted + dropped_on_request_drop +
///  dropped_failed_store + dropped_id_reuse`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LazyOffloadCounters {
    /// Operations admitted into the pending buffer.
    pub admitted: usize,
    /// Operations released by `drain`.
    pub emitted: usize,
    /// Subset of `emitted` released by the deferral deadline rather than eviction pressure.
    pub emitted_overdue: usize,
    /// Operations dropped because a covered GPU block was recycled before emission.
    pub dropped_evicted: usize,
    /// Operations refused at admission because a covered block carried `None` hash.
    pub rejected_unhashed: usize,
    /// Operations refused at admission because the request's prefix chain was already broken.
    pub rejected_prefix_broken: usize,
    /// Buffered operations discarded on preemption (`drop_request`).
    pub dropped_on_request_drop: usize,
    /// Buffered operations discarded when an in-flight store failed (`mark_store_failed`).
    pub dropped_failed_store: usize,
    /// Buffered operations discarded when a new request reused a finished ID (`discard_for_reuse`).
    pub dropped_id_reuse: usize,
}

/// Per-step scheduler signals driving [`EvictionAwareStoreQueue::drain`].
#[derive(Debug, Clone)]
pub struct DrainSignals {
    /// Current monotonic timestamp in seconds.
    pub now_secs: f64,
    /// Gross GPU blocks allocated during this scheduler step.
    pub new_blocks_allocated: usize,
    /// Estimated GPU blocks the next scheduler step will allocate.
    pub est_next_step_blocks: usize,
    /// Request IDs that already have an in-flight store batch and must not be drained.
    pub blocked_request_ids: HashSet<String>,
    /// Ordered GPU block IDs from the head of the block pool's free queue.
    pub free_queue_order: Vec<u64>,
    /// Current prefix-cache hash of each GPU block in the block pool (`None` if unhashed/recycled).
    pub live_block_hashes: HashMap<u64, Option<u64>>,
}

/// Drained operations emitted for one request during [`EvictionAwareStoreQueue::drain`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingStoreItem {
    /// Request identifier.
    pub request_id: String,
    /// Emitted operations in prefix order along with their admission-time block hashes.
    pub ops: Vec<(LazyStoreSliceOp, BTreeMap<u64, u64>)>,
}

/// Output of a single [`EvictionAwareStoreQueue::drain`] step.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LazyOffloadDrain {
    /// Emitted per-request store items ordered by eviction imminence.
    pub items: Vec<PendingStoreItem>,
    /// Request IDs whose pending buffer became empty during this drain.
    pub emptied_request_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct BufferedStoreEntry {
    meta: LazyStoreSliceOp,
    block_hashes: BTreeMap<u64, u64>,
    admitted_at_secs: f64,
}

/// Eviction-aware lazy offload store queue.
#[derive(Debug, Clone)]
pub struct EvictionAwareStoreQueue {
    config: EvictionAwarePolicyConfig,
    /// Insertion-ordered map from `request_id` to buffered operations.
    pending: IndexMap<String, Vec<BufferedStoreEntry>>,
    broken_prefixes: HashSet<String>,
    blocks_per_step_ema: f64,
    ema_initialized: bool,
    next_step_estimate: usize,
    counters: LazyOffloadCounters,
}

impl EvictionAwareStoreQueue {
    /// Creates an empty eviction-aware store queue.
    #[must_use]
    pub fn new(config: EvictionAwarePolicyConfig) -> Self {
        Self {
            config,
            pending: IndexMap::new(),
            broken_prefixes: HashSet::new(),
            blocks_per_step_ema: 0.0,
            ema_initialized: false,
            next_step_estimate: 0,
            counters: LazyOffloadCounters::default(),
        }
    }

    /// Returns the cumulative policy counters.
    #[must_use]
    pub const fn counters(&self) -> LazyOffloadCounters {
        self.counters
    }

    /// Returns the total number of currently buffered operations across all requests.
    #[must_use]
    pub fn pending_op_count(&self) -> usize {
        self.pending.values().map(Vec::len).sum()
    }

    /// Renders the closed accounting ledger line (`key=value ... pending=N`).
    #[must_use]
    pub fn ledger_line(&self) -> String {
        let c = self.counters;
        format!(
            "admitted={} emitted={} emitted_overdue={} dropped_evicted={} \
             rejected_unhashed={} rejected_prefix_broken={} dropped_on_request_drop={} \
             dropped_failed_store={} dropped_id_reuse={} pending={}",
            c.admitted,
            c.emitted,
            c.emitted_overdue,
            c.dropped_evicted,
            c.rejected_unhashed,
            c.rejected_prefix_broken,
            c.dropped_on_request_drop,
            c.dropped_failed_store,
            c.dropped_id_reuse,
            self.pending_op_count()
        )
    }

    /// Buffers one store operation at `admitted_at_secs`.
    ///
    /// If `request_id` already has a broken prefix chain, increments
    /// `rejected_prefix_broken`. If any covered block in `block_hashes` carries
    /// `None`, marks the request's prefix chain broken and increments
    /// `rejected_unhashed`.
    pub fn add(
        &mut self,
        meta: LazyStoreSliceOp,
        block_hashes: &BTreeMap<u64, Option<u64>>,
        admitted_at_secs: f64,
    ) {
        if self.broken_prefixes.contains(&meta.request_id) {
            self.counters.rejected_prefix_broken += 1;
            return;
        }
        let mut verified_hashes = BTreeMap::new();
        for (&blk_id, &maybe_hash) in block_hashes {
            let Some(h) = maybe_hash else {
                self.broken_prefixes.insert(meta.request_id);
                self.counters.rejected_unhashed += 1;
                return;
            };
            verified_hashes.insert(blk_id, h);
        }
        self.pending
            .entry(meta.request_id.clone())
            .or_default()
            .push(BufferedStoreEntry {
                meta,
                block_hashes: verified_hashes,
                admitted_at_secs,
            });
        self.counters.admitted += 1;
    }

    /// Returns `true` if `request_id` currently owns buffered operations.
    #[must_use]
    pub fn has_pending_request(&self, request_id: &str) -> bool {
        self.pending.contains_key(request_id)
    }

    /// Discards buffered operations invalidated by preemption (`drop_request`)
    /// and clears the broken-chain marker so the restarted request begins clean.
    pub fn drop_request(&mut self, request_id: &str) -> usize {
        let dropped = self.pending.shift_remove(request_id).map_or(0, |v| v.len());
        self.broken_prefixes.remove(request_id);
        self.counters.dropped_on_request_drop += dropped;
        dropped
    }

    /// Discards buffered state of a reused `request_id`'s previous holder.
    pub fn discard_for_reuse(&mut self, request_id: &str) -> usize {
        let dropped = self.pending.shift_remove(request_id).map_or(0, |v| v.len());
        self.broken_prefixes.remove(request_id);
        self.counters.dropped_id_reuse += dropped;
        dropped
    }

    /// Releases non-pending policy state (`broken_prefixes`) after session teardown.
    pub fn release_request(&mut self, request_id: &str) {
        self.broken_prefixes.remove(request_id);
    }

    /// Marks `request_id`'s prefix chain broken after a submitted store failed
    /// and drops any held-back suffix operations.
    pub fn mark_store_failed(&mut self, request_id: &str) -> usize {
        let dropped = self.pending.shift_remove(request_id).map_or(0, |v| v.len());
        self.counters.dropped_failed_store += dropped;
        self.broken_prefixes.insert(request_id.to_string());
        dropped
    }

    /// Computes the free-queue danger depth for the current consumption estimate.
    /// Below `0.5` blocks over the horizon counts as idle (`0`).
    #[must_use]
    pub fn danger_depth(&self) -> usize {
        let per_step = self.blocks_per_step_ema.max(self.next_step_estimate as f64);
        let horizon_blocks = per_step * self.config.horizon_steps;
        if horizon_blocks < 0.5 {
            0
        } else {
            horizon_blocks.ceil() as usize
        }
    }

    /// Records the step's allocation pressure and drains due store operations
    /// in eviction-imminence order up to `max_drain_per_step`.
    pub fn drain(&mut self, signals: &DrainSignals) -> LazyOffloadDrain {
        if self.ema_initialized {
            self.blocks_per_step_ema = BLOCK_CONSUMPTION_EMA_ALPHA
                * (signals.new_blocks_allocated as f64)
                + (1.0 - BLOCK_CONSUMPTION_EMA_ALPHA) * self.blocks_per_step_ema;
        } else {
            self.blocks_per_step_ema = signals.new_blocks_allocated as f64;
            self.ema_initialized = true;
        }
        self.next_step_estimate = signals.est_next_step_blocks;

        if self.pending.is_empty() {
            return LazyOffloadDrain::default();
        }

        let depth = self.danger_depth();
        let mut ranks: HashMap<u64, usize> = HashMap::new();
        for (rank, &blk_id) in signals.free_queue_order.iter().take(depth).enumerate() {
            ranks.entry(blk_id).or_insert(rank);
        }

        let overdue_cutoff = signals.now_secs - self.config.max_deferral_seconds;
        let mut out = LazyOffloadDrain::default();
        let mut candidates: Vec<(usize, usize, String)> = Vec::new();
        let mut overdue_ids: HashSet<String> = HashSet::new();

        let request_keys: Vec<String> = self.pending.keys().cloned().collect();
        for (order, request_id) in request_keys.into_iter().enumerate() {
            if signals.blocked_request_ids.contains(&request_id) {
                continue;
            }
            let Some(ops) = self.pending.get(&request_id).cloned() else {
                continue;
            };
            let first_lost = ops
                .iter()
                .position(|op| {
                    !op.block_hashes.iter().all(|(&blk_id, &snap)| {
                        signals
                            .live_block_hashes
                            .get(&blk_id)
                            .and_then(|&v| v)
                            .is_some_and(|live| live == snap)
                    })
                })
                .unwrap_or(ops.len());

            if first_lost < ops.len() {
                self.counters.dropped_evicted += ops.len() - first_lost;
                self.broken_prefixes.insert(request_id.clone());
                if first_lost == 0 {
                    self.pending.shift_remove(&request_id);
                    out.emptied_request_ids.push(request_id);
                    continue;
                }
                if let Some(slot) = self.pending.get_mut(&request_id) {
                    slot.truncate(first_lost);
                }
            }

            let surviving = &self.pending[&request_id];
            let overdue = self.config.max_deferral_seconds > 0.0
                && surviving[0].admitted_at_secs <= overdue_cutoff;
            let min_window_rank = surviving
                .iter()
                .flat_map(|op| op.block_hashes.keys())
                .filter_map(|blk_id| ranks.get(blk_id).copied())
                .min();

            if let Some(best_rank) = min_window_rank {
                candidates.push((best_rank, order, request_id));
            } else if overdue {
                candidates.push((OVERDUE_RANK, order, request_id.clone()));
                overdue_ids.insert(request_id);
            }
        }

        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut ops_left = self.config.max_drain_per_step;

        for (_rank, _order, request_id) in candidates {
            if ops_left == 0 {
                break;
            }
            let Some(ops) = self.pending.get(&request_id).cloned() else {
                continue;
            };
            let is_overdue = overdue_ids.contains(&request_id);
            let due_len = if is_overdue {
                ops.len()
            } else {
                let mut last_due = 0usize;
                for (idx, op) in ops.iter().enumerate() {
                    if op.block_hashes.keys().any(|b| ranks.contains_key(b)) {
                        last_due = idx + 1;
                    }
                }
                last_due
            };
            let emit_count = due_len.min(ops_left);
            if emit_count == 0 {
                continue;
            }
            ops_left -= emit_count;
            self.counters.emitted += emit_count;
            if is_overdue {
                self.counters.emitted_overdue += emit_count;
            }

            let emitted_ops: Vec<(LazyStoreSliceOp, BTreeMap<u64, u64>)> = ops[..emit_count]
                .iter()
                .map(|e| (e.meta.clone(), e.block_hashes.clone()))
                .collect();
            out.items.push(PendingStoreItem {
                request_id: request_id.clone(),
                ops: emitted_ops,
            });

            if emit_count == ops.len() {
                self.pending.shift_remove(&request_id);
                out.emptied_request_ids.push(request_id);
            } else if let Some(slot) = self.pending.get_mut(&request_id) {
                slot.drain(..emit_count);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eviction_aware_queue_danger_depth_and_closed_ledger() {
        let cfg = EvictionAwarePolicyConfig::new(2.5, 4, 10.0).expect("valid config");
        let mut q = EvictionAwareStoreQueue::new(cfg);

        // Request 1: block 10 (hash 100), Request 2: block 20 (hash 200).
        let op1 = LazyStoreSliceOp {
            request_id: "req1".to_string(),
            start: 0,
            end: 16,
            block_ids: vec![vec![10]],
            token_bytes: vec![1, 2],
            cache_salt: String::new(),
        };
        let op2 = LazyStoreSliceOp {
            request_id: "req2".to_string(),
            start: 0,
            end: 16,
            block_ids: vec![vec![20]],
            token_bytes: vec![3, 4],
            cache_salt: String::new(),
        };
        q.add(op1, &BTreeMap::from([(10, Some(100))]), 1.0);
        q.add(op2, &BTreeMap::from([(20, Some(200))]), 2.0);
        assert_eq!(q.counters().admitted, 2);

        // Idle step (0 allocated, 0 next) -> danger_depth == 0 -> nothing drained.
        let idle_signals = DrainSignals {
            now_secs: 5.0,
            new_blocks_allocated: 0,
            est_next_step_blocks: 0,
            blocked_request_ids: HashSet::new(),
            free_queue_order: vec![20, 10],
            live_block_hashes: HashMap::from([(10, Some(100)), (20, Some(200))]),
        };
        let d0 = q.drain(&idle_signals);
        assert!(d0.items.is_empty());

        // Active step: block 20 is at head of free queue (rank 0), block 10 at rank 1.
        let active_signals = DrainSignals {
            now_secs: 6.0,
            new_blocks_allocated: 2,
            est_next_step_blocks: 1,
            blocked_request_ids: HashSet::new(),
            free_queue_order: vec![20, 10],
            live_block_hashes: HashMap::from([(10, Some(100)), (20, Some(200))]),
        };
        let d1 = q.drain(&active_signals);
        assert_eq!(d1.items.len(), 2);
        // req2 (rank 0) is served before req1 (rank 1).
        assert_eq!(d1.items[0].request_id, "req2");
        assert_eq!(d1.items[1].request_id, "req1");
        assert_eq!(q.counters().emitted, 2);
        assert_eq!(q.pending_op_count(), 0);
    }
}
