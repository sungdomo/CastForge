//! TurboQuant compression, GDS slab allocation, and horizontal fusion passes.

#[allow(unused_imports)]
use super::*;
use crate::ir::{Attribute, BlockId, Module, OpId, OpName, Operation, Value, ValueDef, ValueId};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

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

pub(super) fn is_horizontally_fusible_elementwise(op: &Operation) -> bool {
    is_sinkable_unary_elementwise(op) || is_sinkable_binary_elementwise(op)
}

pub(super) fn op_transitively_depends_on(
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

pub(super) fn ops_match_ignoring_device(a: &Operation, b: &Operation) -> bool {
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
