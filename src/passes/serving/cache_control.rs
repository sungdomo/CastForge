//! Multi-L1 write target routing, bounded token-chunk object key resolution, and lock-aware
//! multi-tier cache-control deletion services.

use std::collections::{BTreeSet, HashSet};

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
