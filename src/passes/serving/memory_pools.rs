//! Elementwise chain flattening, state transfer elision, and rowwise launch passes.

#[allow(unused_imports)]
use super::*;
use crate::ir::{
    Attribute, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type, ValueDef,
    ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

pub(super) fn is_flattenable_elemwise_op(op: &Operation) -> bool {
    if op.name.dialect != "stablehlo" || op.results.len() != 1 {
        return false;
    }
    let name = op.name.name.as_str();
    (op.operands.len() == 1
        && matches!(
            name,
            "tanh"
                | "abs"
                | "negate"
                | "exponential"
                | "log"
                | "logistic"
                | "sqrt"
                | "rsqrt"
                | "sine"
                | "cosine"
                | "ceil"
                | "floor"
        ))
        || (op.operands.len() == 2
            && matches!(
                name,
                "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum"
            ))
}

/// Flattens connected chains of high-rank (`rank >= 2`) elementwise operations and intervening
/// `stablehlo.reshape` views to 1D (`tensor<Nxdtype>`), eliminating interior reshapes so
/// downstream loop fusion can merge the entire elementwise subgraph into a single 1D kernel
/// (inspired by Modular `FlattenElementwise.cpp`).
///
/// Returns the number of high-rank elementwise operations flattened to rank 1.
pub fn flatten_elementwise_chains(module: &mut Module) -> usize {
    let mut flattened_count = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let orig_ops = module.blocks[block_id].ops.clone();

        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !is_flattenable_elemwise_op(&op) {
                continue;
            }
            let res_val = op.results[0];
            let orig_ty = module.values[res_val].ty.clone();
            let Some(res_tt) = orig_ty.as_tensor() else {
                continue;
            };
            if res_tt.shape.len() <= 1 {
                continue;
            }
            let Some(num_elems) = res_tt.num_elements() else {
                continue;
            };
            if num_elems <= 0 {
                continue;
            }

            // Check that all operands share the same element count `num_elems`
            let all_same_vol = op.operands.iter().all(|&opnd| {
                module.values[opnd]
                    .ty
                    .as_tensor()
                    .and_then(|t| t.num_elements())
                    == Some(num_elems)
            });
            if !all_same_vol {
                continue;
            }

            // Flatten each operand to `tensor<num_elems x dtype>` right before `op_id`
            let mut new_operands: SmallVec<[ValueId; 4]> = SmallVec::new();
            let mut pre_reshapes: Vec<OpId> = Vec::new();
            for &opnd in &op.operands {
                let opnd_dt = module.values[opnd]
                    .ty
                    .as_tensor()
                    .map(|t| t.dtype.clone())
                    .unwrap_or(DType::F32);
                let flat_in_ty =
                    Type::Tensor(TensorType::new(vec![Dim::Static(num_elems)], opnd_dt));
                if module.values[opnd].ty == flat_in_ty {
                    new_operands.push(opnd);
                } else {
                    let r_op = module.add_operation(
                        block_id,
                        OpName::new("stablehlo", "reshape"),
                        smallvec::smallvec![opnd],
                        vec![(None, flat_in_ty)],
                        IndexMap::new(),
                        Vec::new(),
                        true,
                        op.location.clone(),
                    );
                    // `add_operation` appends to `module.blocks[block_id].ops`; pop and splice
                    module.blocks[block_id].ops.pop();
                    pre_reshapes.push(r_op);
                    new_operands.push(module.ops[r_op].results[0]);
                }
            }

            let flat_res_ty = Type::Tensor(TensorType::new(
                vec![Dim::Static(num_elems)],
                res_tt.dtype.clone(),
            ));
            module.values[res_val].ty = flat_res_ty;
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.operands = new_operands;
            }

            // Create exit reshape restoring `orig_ty` and replace downstream uses of `res_val`
            let exit_r_op = module.add_operation(
                block_id,
                OpName::new("stablehlo", "reshape"),
                smallvec::smallvec![res_val],
                vec![(None, orig_ty)],
                IndexMap::new(),
                Vec::new(),
                true,
                op.location.clone(),
            );
            module.blocks[block_id].ops.pop();
            let restored_val = module.ops[exit_r_op].results[0];

            for &other_id in &module.blocks[block_id].ops {
                if other_id == op_id || other_id == exit_r_op {
                    continue;
                }
                if let Some(other_op) = module.ops.get_mut(other_id) {
                    for opnd_slot in &mut other_op.operands {
                        if *opnd_slot == res_val {
                            *opnd_slot = restored_val;
                        }
                    }
                }
            }

            // Splice `pre_reshapes` before `op_id` and `exit_r_op` immediately after `op_id`
            if let Some(pos) = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == op_id)
            {
                module.blocks[block_id].ops.insert(pos + 1, exit_r_op);
                for (offset, r_id) in pre_reshapes.into_iter().enumerate() {
                    module.blocks[block_id].ops.insert(pos + offset, r_id);
                }
            }
            flattened_count += 1;
        }

        if flattened_count > 0 {
            // Fold `reshape(reshape(x))` chains and eliminate identity `reshape(x : T) -> T`
            loop {
                let mut changed = false;
                let cur_ops = module.blocks[block_id].ops.clone();
                for &r_id in &cur_ops {
                    let Some(r_op) = module.ops.get(r_id).cloned() else {
                        continue;
                    };
                    if !r_op.is_op("stablehlo", "reshape") || r_op.operands.len() != 1 {
                        continue;
                    }
                    let inp = r_op.operands[0];
                    let out = r_op.results[0];
                    if let ValueDef::OpResult { op: prev_id, .. } = module.values[inp].def
                        && let Some(prev_op) = module.ops.get(prev_id).cloned()
                        && prev_op.is_op("stablehlo", "reshape")
                        && prev_op.operands.len() == 1
                    {
                        let root_inp = prev_op.operands[0];
                        if let Some(r_mut) = module.ops.get_mut(r_id) {
                            r_mut.operands[0] = root_inp;
                        }
                        if module.use_count(inp) == 0 {
                            module.remove_op(block_id, prev_id);
                        }
                        changed = true;
                        break;
                    }
                    if module.values[inp].ty == module.values[out].ty {
                        module.replace_all_uses(out, inp);
                        module.remove_op(block_id, r_id);
                        changed = true;
                        break;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
    }
    flattened_count
}

pub(super) fn is_named_state_op(op: &Operation, short_name: &str) -> bool {
    op.name.name == short_name
        || (op.is_op("stablehlo", "composite")
            && op
                .attributes
                .get("name")
                .and_then(|a| a.as_str())
                .is_some_and(|n| n.ends_with(short_name)))
}

/// Elides redundant store-to-load roundtrips (`kv_offload -> kv_reload` and exact
/// `dynamic_update_slice -> dynamic_slice` read-after-write pairs) and dead overwritten stores
/// (`dynamic_update_slice -> dynamic_update_slice` writing the exact same slice when the first
/// store has a single use) (inspired by Modular `ElideRedundantMutableOps.cpp`).
///
/// Returns the number of redundant state transfer / slice operations elided.
pub fn elide_redundant_state_transfers(module: &mut Module) -> usize {
    let mut elided = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };

        loop {
            let mut changed = false;
            let ops = module.blocks[block_id].ops.clone();

            for &op_id in &ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.results.len() != 1 || op.operands.is_empty() {
                    continue;
                }
                let out_val = op.results[0];

                // 1. `kv_offload -> kv_reload` store-to-load forwarding
                if is_named_state_op(&op, "kv_reload") {
                    let in_val = op.operands[0];
                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[in_val].def
                        && let Some(prod_op) = module.ops.get(prod_id).cloned()
                        && is_named_state_op(&prod_op, "kv_offload")
                        && !prod_op.operands.is_empty()
                    {
                        let orig_tensor = prod_op.operands[0];
                        module.replace_all_uses(out_val, orig_tensor);
                        module.remove_op(block_id, op_id);
                        elided += 1;
                        if module.use_count(in_val) == 0 {
                            module.remove_op(block_id, prod_id);
                            elided += 1;
                        }
                        changed = true;
                        break;
                    }
                }

                // 2. `dynamic_update_slice -> dynamic_slice` exact read-after-write forwarding
                if op.is_op("stablehlo", "dynamic_slice") && op.operands.len() >= 2 {
                    let src_buf = op.operands[0];
                    let slice_indices = &op.operands[1..];
                    if let ValueDef::OpResult { op: upd_id, .. } = module.values[src_buf].def
                        && let Some(upd_op) = module.ops.get(upd_id).cloned()
                        && upd_op.is_op("stablehlo", "dynamic_update_slice")
                        && upd_op.operands.len() == op.operands.len() + 1
                    {
                        let patch_val = upd_op.operands[1];
                        let upd_indices = &upd_op.operands[2..];
                        if slice_indices == upd_indices
                            && module.values[out_val].ty == module.values[patch_val].ty
                        {
                            module.replace_all_uses(out_val, patch_val);
                            module.remove_op(block_id, op_id);
                            elided += 1;
                            if module.use_count(src_buf) == 0 {
                                module.remove_op(block_id, upd_id);
                                elided += 1;
                            }
                            changed = true;
                            break;
                        }
                    }
                }

                // 3. `dynamic_update_slice -> dynamic_update_slice` dead overwritten store elision
                if op.is_op("stablehlo", "dynamic_update_slice") && op.operands.len() >= 3 {
                    let prev_buf = op.operands[0];
                    let cur_patch = op.operands[1];
                    let cur_indices = &op.operands[2..];
                    if module.use_count(prev_buf) == 1
                        && let ValueDef::OpResult { op: prev_id, .. } = module.values[prev_buf].def
                        && let Some(prev_op) = module.ops.get(prev_id).cloned()
                        && prev_op.is_op("stablehlo", "dynamic_update_slice")
                        && prev_op.operands.len() == op.operands.len()
                    {
                        let base_buf = prev_op.operands[0];
                        let prev_patch = prev_op.operands[1];
                        let prev_indices = &prev_op.operands[2..];
                        if cur_indices == prev_indices
                            && module.values[cur_patch].ty == module.values[prev_patch].ty
                        {
                            if let Some(op_mut) = module.ops.get_mut(op_id) {
                                op_mut.operands[0] = base_buf;
                            }
                            module.remove_op(block_id, prev_id);
                            elided += 1;
                            changed = true;
                            break;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }
    elided
}

/// Target hardware architecture profile for device/dtype validation
/// (inspired by Modular `ValidateDevicesPass.cpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTargetArch {
    /// x86_64 CPU: supports `f32`/`f64`, integers, `bf16`, and `f16`; rejects sub-16-bit floats
    /// (`f8E4M3FN`, `f8E5M2`).
    X86_64,
    /// AArch64 CPU: supports `f32`/`f64`, integers, and `bf16`; rejects `f16` and sub-16-bit
    /// floats (`f8E4M3FN`, `f8E5M2`) per Modular `isSupportedArchDTypeCombination`.
    AArch64,
    /// WebGPU WGSL: supports `f32`, `f16`, `i32`, `ui32`; rejects `bf16`, `f8`, and `f64`.
    Wgsl,
    /// Modern GPU (`sm_90` / `gfx942`): supports all floating-point and integer dtypes including
    /// `f8E4M3FN` and `f8E5M2`.
    GpuSm90,
}

impl DeviceTargetArch {
    pub fn default_device_label(self) -> &'static str {
        match self {
            Self::X86_64 => "cpu:x86_64",
            Self::AArch64 => "cpu:aarch64",
            Self::Wgsl => "gpu:wgsl",
            Self::GpuSm90 => "gpu:0",
        }
    }
}

/// Returns `true` if `dtype` is natively supported on `arch`
/// (inspired by Modular `ValidateDevicesPass.cpp` `isSupportedArchDTypeCombination`).
pub fn is_supported_arch_dtype(arch: DeviceTargetArch, dtype: &DType) -> bool {
    match arch {
        DeviceTargetArch::X86_64 => {
            if matches!(dtype, DType::BF16 | DType::F16) {
                return true;
            }
            !(dtype.is_float() && dtype.bit_width() < 32)
        }
        DeviceTargetArch::AArch64 => {
            if matches!(dtype, DType::BF16) {
                return true;
            }
            !(dtype.is_float() && dtype.bit_width() < 32)
        }
        DeviceTargetArch::Wgsl => matches!(
            dtype,
            DType::F32 | DType::F16 | DType::I32 | DType::U32 | DType::I1 | DType::Index
        ),
        DeviceTargetArch::GpuSm90 => true,
    }
}

/// Diagnostic emitted when an operation or block argument uses a [`DType`] unsupported by the
/// target device architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceDTypeDiagnostic {
    pub op_name: String,
    pub device: String,
    pub dtype: DType,
    pub location: Location,
    pub message: String,
}

/// Validates that all function arguments and operation result tensor [`DType`]s in `module` are
/// supported on `arch` (inspired by Modular `ValidateDevicesPass.cpp`).
pub fn validate_target_device_dtypes(
    module: &Module,
    arch: DeviceTargetArch,
) -> Vec<DeviceDTypeDiagnostic> {
    let mut diags = Vec::new();
    let default_dev = arch.default_device_label();

    for fn_op_id in module.functions() {
        let fn_op = &module.ops[fn_op_id];
        let Some(&reg_id) = fn_op.regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let fn_dev = fn_op
            .attributes
            .get("castforge.device")
            .and_then(|a| a.as_str())
            .unwrap_or(default_dev);

        for &arg_val in &module.blocks[block_id].arguments {
            if let Some(tt) = module.values[arg_val].ty.as_tensor()
                && !is_supported_arch_dtype(arch, &tt.dtype)
            {
                let dt_str = tt.dtype.to_string();
                diags.push(DeviceDTypeDiagnostic {
                    op_name: "func.func".to_string(),
                    device: fn_dev.to_string(),
                    dtype: tt.dtype.clone(),
                    location: fn_op.location.clone(),
                    message: format!(
                        "The {dt_str} data type is not supported on device '{fn_dev}'."
                    ),
                });
            }
        }

        for &op_id in &module.blocks[block_id].ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let dev = op
                .attributes
                .get("castforge.device")
                .and_then(|a| a.as_str())
                .unwrap_or(fn_dev);
            for &res_val in &op.results {
                if let Some(tt) = module.values[res_val].ty.as_tensor()
                    && !is_supported_arch_dtype(arch, &tt.dtype)
                {
                    let dt_str = tt.dtype.to_string();
                    diags.push(DeviceDTypeDiagnostic {
                        op_name: op.name.to_string(),
                        device: dev.to_string(),
                        dtype: tt.dtype.clone(),
                        location: op.location.clone(),
                        message: format!(
                            "The {dt_str} data type is not supported on device '{dev}'."
                        ),
                    });
                }
            }
        }
    }
    diags
}

/// Sentinel window value (`-1`) marking a full-attention KV object group that requires the
/// entire `[0, prefix_len)` chunk prefix (inspired by LMCache `bitmap_ops/fold.py`).
pub const FULL_ATTENTION_WINDOW: i32 = -1;

/// Half-open chunk range `[lo, hi)` that an object group with `window_chunks` (`<= 0` for full
/// attention, `w > 0` for sliding window / Mamba) needs to serve a prefix of `prefix_len` chunks.
pub fn unfold_chunk_range(prefix_len: usize, window_chunks: i32) -> (usize, usize) {
    if prefix_len == 0 {
        return (0, 0);
    }
    let lo = if window_chunks <= 0 {
        0
    } else {
        prefix_len.saturating_sub(window_chunks as usize)
    };
    (lo, prefix_len)
}

/// Three-stage `fold -> highest_set_bit -> unfold` multi-object-group prefix-cache hit planner
/// (inspired by LMCache `lmcache/v1/distributed/bitmap_ops/fold.py` `fold_unfold_grouped`).
///
/// Given `group_presence` (`[num_groups][num_chunks]` booleans) and `group_windows` (`<= 0` for
/// full attention, `w > 0` for sliding window of `w` chunks):
/// 1. `fold`: computes `servable[j]` (`true` iff every group can serve prefix length `j + 1`),
/// 2. `highest_set_bit`: finds the longest jointly servable prefix `hit_length` (recovering
///    longer hits even when earlier chunks outside a sliding window were evicted), and
/// 3. `unfold`: expands `hit_length` into per-group retain masks so only required chunks are
///    locked and transferred.
///
/// Returns `(hit_length, servable_bitmap, retain_masks)`.
pub fn fold_unfold_hybrid_prefix_bitmaps(
    group_presence: &[Vec<bool>],
    group_windows: &[i32],
) -> (usize, Vec<bool>, Vec<Vec<bool>>) {
    let num_groups = group_presence.len().min(group_windows.len());
    if num_groups == 0 {
        return (0, Vec::new(), Vec::new());
    }
    let num_chunks = group_presence[..num_groups]
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    if num_chunks == 0 {
        return (0, Vec::new(), vec![Vec::new(); num_groups]);
    }

    let mut servable = vec![false; num_chunks];
    let mut hit_length = 0usize;
    for (j, slot) in servable.iter_mut().enumerate() {
        let cand_len = j + 1;
        let all_ok = (0..num_groups).all(|g| {
            let (lo, hi) = unfold_chunk_range(cand_len, group_windows[g]);
            group_presence[g][lo..hi].iter().all(|&present| present)
        });
        *slot = all_ok;
        if all_ok {
            hit_length = cand_len;
        }
    }

    let mut retain_masks = Vec::with_capacity(num_groups);
    for &win in &group_windows[..num_groups] {
        let mut mask = vec![false; num_chunks];
        let (lo, hi) = unfold_chunk_range(hit_length, win);
        for bit in &mut mask[lo..hi] {
            *bit = true;
        }
        retain_masks.push(mask);
    }
    (hit_length, servable, retain_masks)
}

/// Per-tenant isolated LRU eviction policy partitioned by `cache_salt`
/// (inspired by LMCache `lmcache/v1/distributed/eviction_policy/isolated_lru.py`).
///
/// Maintains an independent ordered LRU bucket per tenant `cache_salt`. Keys created within a
/// single request batch are inserted in **reverse order** so that later sequence chunks (which are
/// less valuable for future prefix reuse) sit closer to the LRU eviction head than earlier prefix
/// chunks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IsolatedLruEvictionPolicy {
    buckets: IndexMap<String, IndexMap<u64, ()>>,
}

impl IsolatedLruEvictionPolicy {
    /// Creates an empty per-salt isolated LRU eviction policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of resident keys currently tracked for `cache_salt`.
    pub fn resident_len_for_salt(&self, cache_salt: &str) -> usize {
        self.buckets.get(cache_salt).map_or(0, IndexMap::len)
    }

    /// Total number of resident keys across all tenant buckets.
    pub fn total_resident_len(&self) -> usize {
        self.buckets.values().map(IndexMap::len).sum()
    }

    /// Records newly created `keys` under `cache_salt` in reverse request order so later chunks
    /// are evicted before earlier prefix chunks.
    pub fn on_keys_created(&mut self, cache_salt: &str, keys: &[u64]) {
        let bucket = self.buckets.entry(cache_salt.to_string()).or_default();
        for &key in keys.iter().rev() {
            bucket.shift_remove(&key);
            bucket.insert(key, ());
        }
    }

    /// Promotes accessed `keys` in `cache_salt` to the MRU tail in reverse order.
    pub fn on_keys_touched(&mut self, cache_salt: &str, keys: &[u64]) {
        if let Some(bucket) = self.buckets.get_mut(cache_salt) {
            for &key in keys.iter().rev() {
                if bucket.shift_remove(&key).is_some() {
                    bucket.insert(key, ());
                }
            }
        }
    }

    /// Removes `keys` from `cache_salt` when deleted externally.
    pub fn on_keys_removed(&mut self, cache_salt: &str, keys: &[u64]) {
        if let Some(bucket) = self.buckets.get_mut(cache_salt) {
            for &key in keys {
                bucket.shift_remove(&key);
            }
        }
    }

    /// Evicts `ceil(len * expected_ratio)` LRU keys strictly from `cache_salt`'s bucket without
    /// disturbing any other tenant's resident keys.
    pub fn evict_for_salt(&mut self, cache_salt: &str, expected_ratio: f64) -> Vec<u64> {
        let ratio = expected_ratio.clamp(0.0, 1.0);
        if ratio <= 0.0 {
            return Vec::new();
        }
        let Some(bucket) = self.buckets.get_mut(cache_salt) else {
            return Vec::new();
        };
        let num_to_evict = ((bucket.len() as f64) * ratio).ceil() as usize;
        let count = num_to_evict.min(bucket.len());
        let victims: Vec<u64> = bucket.keys().take(count).copied().collect();
        for &v in &victims {
            bucket.shift_remove(&v);
        }
        victims
    }

    /// Returns all `cache_salt` identifiers that currently hold at least one tracked key.
    pub fn tracked_salts(&self) -> Vec<String> {
        self.buckets
            .iter()
            .filter_map(|(salt, b)| (!b.is_empty()).then_some(salt.clone()))
            .collect()
    }

    /// Selects up to `ceil(len * expected_ratio)` LRU keys from `cache_salt` that satisfy
    /// `is_eligible` without mutating the LRU bucket (used by asynchronous fleet eviction where
    /// keys leave the LRU only after the `DELETE` event is confirmed).
    pub fn select_eviction_candidates_filtered<F>(
        &self,
        cache_salt: &str,
        expected_ratio: f64,
        is_eligible: F,
    ) -> Vec<u64>
    where
        F: Fn(u64) -> bool,
    {
        let ratio = expected_ratio.clamp(0.0, 1.0);
        if ratio <= 0.0 {
            return Vec::new();
        }
        let Some(bucket) = self.buckets.get(cache_salt) else {
            return Vec::new();
        };
        let target = ((bucket.len() as f64) * ratio).ceil() as usize;
        bucket
            .keys()
            .copied()
            .filter(|&k| is_eligible(k))
            .take(target)
            .collect()
    }
}

/// Default GPUDirect Storage (`cuFile` / `O_DIRECT`) filesystem block alignment (`4096` bytes).
pub const GDS_DEFAULT_ALIGN_BYTES: u64 = 4096;

/// Aligned physical slab region in a GPUDirect Storage (GDS) L1 backing file
/// (inspired by LMCache `lmcache/v1/distributed/memory_manager/gds_l1_memory_manager.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdsSlabRegion {
    /// Aligned starting byte offset in the backing file.
    pub offset: u64,
    /// Aligned physical span length in bytes (`ceil(logical_bytes / align_bytes) * align_bytes`).
    pub physical_size: u64,
}

/// GPUDirect Storage (GDS) 4 KiB-aligned all-or-nothing slab address allocator with sorted
/// free-interval coalescing
/// (inspired by LMCache `lmcache/v1/distributed/memory_manager/gds_l1_memory_manager.py`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdsSlabAddressManager {
    total_size_bytes: u64,
    align_bytes: u64,
    free_regions: Vec<GdsSlabRegion>,
}

impl GdsSlabAddressManager {
    /// Creates a new GDS slab allocator over `total_size_bytes` rounded down to a multiple of
    /// `align_bytes` (defaults to [`GDS_DEFAULT_ALIGN_BYTES`] when `align_bytes == 0`).
    pub fn new(total_size_bytes: u64, align_bytes: u64) -> Self {
        let align = if align_bytes == 0 {
            GDS_DEFAULT_ALIGN_BYTES
        } else {
            align_bytes
        };
        let aligned_total = (total_size_bytes / align) * align;
        let free_regions = if aligned_total > 0 {
            vec![GdsSlabRegion {
                offset: 0,
                physical_size: aligned_total,
            }]
        } else {
            Vec::new()
        };
        Self {
            total_size_bytes: aligned_total,
            align_bytes: align,
            free_regions,
        }
    }

    /// Rounds `logical_bytes` up to the nearest non-zero multiple of `self.align_bytes`.
    pub fn align_up(&self, logical_bytes: u64) -> u64 {
        let n = logical_bytes.max(1);
        n.div_ceil(self.align_bytes) * self.align_bytes
    }

    /// Total free physical bytes across all coalesced free regions.
    pub fn free_size_bytes(&self) -> u64 {
        self.free_regions.iter().map(|r| r.physical_size).sum()
    }

    /// Number of disjoint free fragments currently tracked.
    pub fn free_fragment_count(&self) -> usize {
        self.free_regions.len()
    }

    fn allocate_single(&mut self, physical_size: u64) -> Option<GdsSlabRegion> {
        let idx = self
            .free_regions
            .iter()
            .position(|r| r.physical_size >= physical_size)?;
        let region = &mut self.free_regions[idx];
        let allocated = GdsSlabRegion {
            offset: region.offset,
            physical_size,
        };
        region.offset += physical_size;
        region.physical_size -= physical_size;
        if region.physical_size == 0 {
            self.free_regions.remove(idx);
        }
        Some(allocated)
    }

    /// Allocates `count` aligned slabs of `logical_chunk_bytes` with **all-or-nothing** semantics:
    /// if any slab cannot be satisfied, all slabs allocated earlier in the batch are rolled back
    /// and coalesced before returning `None`.
    pub fn allocate_batch(
        &mut self,
        logical_chunk_bytes: u64,
        count: usize,
    ) -> Option<Vec<GdsSlabRegion>> {
        if count == 0 {
            return Some(Vec::new());
        }
        let phys = self.align_up(logical_chunk_bytes);
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(slab) = self.allocate_single(phys) {
                allocated.push(slab);
            } else {
                for prev in allocated {
                    self.free_region(prev);
                }
                return None;
            }
        }
        Some(allocated)
    }

    /// Frees `region` and coalesces adjacent free intervals in sorted offset order.
    pub fn free_region(&mut self, region: GdsSlabRegion) {
        if region.physical_size == 0 {
            return;
        }
        self.free_regions.push(region);
        self.free_regions.sort_by_key(|r| r.offset);
        let mut merged: Vec<GdsSlabRegion> = Vec::with_capacity(self.free_regions.len());
        for r in self.free_regions.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.offset + last.physical_size == r.offset
            {
                last.physical_size += r.physical_size;
            } else {
                merged.push(r);
            }
        }
        self.free_regions = merged;
    }
}

pub(super) fn expr_contains_ident(text: &str, ident: &str) -> bool {
    if ident.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    let id_len = ident.len();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = 0usize;
    while let Some(rel) = text[start..].find(ident) {
        let pos = start + rel;
        let before_ok = pos == 0 || !is_word(bytes[pos - 1]);
        let after_ok = pos + id_len >= bytes.len() || !is_word(bytes[pos + id_len]);
        if before_ok && after_ok {
            return true;
        }
        start = pos + 1;
    }
    false
}

pub(super) fn stmt_references_buffer(stmt: &crate::ir::loops::LoopStmt, buf_name: &str) -> bool {
    use crate::ir::loops::LoopStmt;
    match stmt {
        LoopStmt::Comment(_) | LoopStmt::Alloc { .. } => false,
        LoopStmt::Assign { dst, index, expr } => {
            dst == buf_name
                || expr_contains_ident(index, buf_name)
                || expr_contains_ident(expr, buf_name)
        }
        LoopStmt::For { body, .. } => body.iter().any(|s| stmt_references_buffer(s, buf_name)),
    }
}

/// Defers `LoopStmt::Alloc` buffer declarations in each [`crate::ir::loops::LoopKernel`] to
/// immediately precede their first use (and eliminates dead unused allocations), shrinking buffer
/// live ranges to reduce peak scratchpad memory
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/DeferAllocsPass.cpp`).
///
/// Returns the number of `Alloc` statements that were deferred across computational statements or
/// eliminated as dead.
pub fn defer_allocs_to_first_use(loop_module: &mut crate::ir::loops::LoopModule) -> usize {
    use crate::ir::loops::LoopStmt;
    let mut moved_or_elided = 0usize;
    for kernel in &mut loop_module.kernels {
        let mut allocs: Vec<(String, LoopStmt)> = Vec::new();
        let mut other_stmts: Vec<(usize, LoopStmt)> = Vec::new();
        let mut alloc_orig_pos: IndexMap<String, usize> = IndexMap::new();

        for (orig_idx, stmt) in kernel.body.drain(..).enumerate() {
            if let LoopStmt::Alloc { ref name, .. } = stmt {
                alloc_orig_pos.insert(name.clone(), orig_idx);
                allocs.push((name.clone(), stmt));
            } else {
                other_stmts.push((orig_idx, stmt));
            }
        }

        let mut insertions: Vec<Vec<LoopStmt>> = vec![Vec::new(); other_stmts.len()];
        for (name, alloc_stmt) in allocs {
            if let Some(first_use_slot) = other_stmts
                .iter()
                .position(|(_, s)| stmt_references_buffer(s, &name))
            {
                let orig_pos = alloc_orig_pos.get(&name).copied().unwrap_or(0);
                // Count as deferred if at least one computational statement (`For` or `Assign`)
                // originally appeared after `orig_pos` and now precedes `first_use_slot`.
                let crossed_compute = other_stmts[..first_use_slot].iter().any(|(pos, s)| {
                    *pos > orig_pos && matches!(s, LoopStmt::For { .. } | LoopStmt::Assign { .. })
                });
                if crossed_compute {
                    moved_or_elided += 1;
                }
                insertions[first_use_slot].push(alloc_stmt);
            } else {
                // Dead allocation with no references in the kernel body
                moved_or_elided += 1;
            }
        }

        let mut new_body = Vec::with_capacity(other_stmts.len() + alloc_orig_pos.len());
        for (slot, (_, stmt)) in other_stmts.into_iter().enumerate() {
            new_body.append(&mut insertions[slot]);
            new_body.push(stmt);
        }
        kernel.body = new_body;
    }
    moved_or_elided
}

pub(super) fn is_shape_materialization_op(name: &str) -> bool {
    matches!(
        name,
        "stablehlo.get_dimension_size"
            | "stablehlo.iota"
            | "stablehlo.broadcast_in_dim"
            | "stablehlo.reshape"
            | "shape.shape_of"
            | "tensor.dim"
    )
}

pub(super) type ShapeOpSignature = (String, Vec<ValueId>, Vec<Type>, String);

/// Merges duplicate shape-materialization and broadcast/metadata operations
/// (`stablehlo.get_dimension_size`, `stablehlo.iota`, `stablehlo.broadcast_in_dim`,
/// `stablehlo.reshape`, `shape.shape_of`, `tensor.dim`) within each block while fusing their
/// [`Location`] provenance onto the surviving primary op via [`stamp_fused_provenance`]
/// (inspired by Modular
/// `GraphCompiler/lib/MODialect/Transforms/MergeDuplicateShapeMaterializations.cpp`).
///
/// Returns the number of duplicate shape/metadata operations merged and eliminated.
pub fn merge_duplicate_shape_ops(module: &mut Module) -> usize {
    let mut merged_count = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let mut seen: IndexMap<ShapeOpSignature, (OpId, ValueId)> = IndexMap::new();

        for op_id in op_ids {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let op_name_str = op.name.to_string();
            if !is_shape_materialization_op(&op_name_str) || op.results.len() != 1 {
                continue;
            }
            let res_val = op.results[0];
            let res_tys: Vec<Type> = op
                .results
                .iter()
                .filter_map(|&v| module.values.get(v).map(|val| val.ty.clone()))
                .collect();
            let mut attr_pairs: Vec<(String, String)> = op
                .attributes
                .iter()
                .filter(|(k, _)| !k.starts_with("castforge.fused_"))
                .map(|(k, v)| (k.clone(), format!("{v:?}")))
                .collect();
            attr_pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let attr_sig = format!("{attr_pairs:?}");
            let sig: ShapeOpSignature = (op_name_str, op.operands.to_vec(), res_tys, attr_sig);

            if let Some(&(primary_op_id, primary_val)) = seen.get(&sig) {
                let dup_loc = op.location.clone();
                if let Some(primary_op) = module.ops.get_mut(primary_op_id) {
                    let locs = [primary_op.location.clone(), dup_loc];
                    stamp_fused_provenance(primary_op, &locs);
                }
                module.replace_all_uses(res_val, primary_val);
                module.remove_op(block_id, op_id);
                merged_count += 1;
            } else {
                seen.insert(sig, (op_id, res_val));
            }
        }
    }

    merged_count
}

/// Memory-efficient optimal-parameter Bloom filter for fast KV prefix-chunk membership testing
/// and unique-chunk counting (inspired by LMCache `lmcache/v1/utils/bloom_filter.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct KvPrefixBloomFilter {
    bit_size: usize,
    hash_count: usize,
    words: Vec<u32>,
    item_count: usize,
    expected_elements: usize,
    false_positive_rate: f64,
}

impl KvPrefixBloomFilter {
    /// Optimal bit-array size `m = floor(-(n * ln(p)) / (ln(2)^2))`.
    pub fn optimal_bit_size(expected_elements: usize, false_positive_rate: f64) -> usize {
        let n = expected_elements.max(1) as f64;
        let p = false_positive_rate.clamp(1e-9, 0.999_999);
        let ln2 = std::f64::consts::LN_2;
        ((-(n * p.ln()) / (ln2 * ln2)).floor() as usize).max(32)
    }

    /// Optimal hash function count `k = max(1, floor((m / n) * ln(2)))`.
    pub fn optimal_hash_count(bit_size: usize, expected_elements: usize) -> usize {
        let n = expected_elements.max(1) as f64;
        let m = bit_size.max(1) as f64;
        (((m / n) * std::f64::consts::LN_2).floor() as usize).max(1)
    }

    /// Creates a new Bloom filter sized for `expected_elements` and `false_positive_rate`.
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let bit_size = Self::optimal_bit_size(expected_elements, false_positive_rate);
        let hash_count = Self::optimal_hash_count(bit_size, expected_elements);
        let word_count = bit_size.div_ceil(32);
        Self {
            bit_size,
            hash_count,
            words: vec![0u32; word_count],
            item_count: 0,
            expected_elements: expected_elements.max(1),
            false_positive_rate: false_positive_rate.clamp(1e-9, 0.999_999),
        }
    }

    /// Bit capacity `m` of the filter.
    pub fn bit_size(&self) -> usize {
        self.bit_size
    }

    /// Number of independent hash probes `k` per key.
    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    /// Number of unique/inserted items recorded.
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    fn hash_positions(&self, key: u64) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.hash_count);
        for i in 0..self.hash_count {
            // SplitMix64-style double hashing seeded by `(key, probe_index)`
            let mut z = key
                .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            out.push((z as usize) % self.bit_size);
        }
        out
    }

    /// Inserts `key` into the Bloom filter.
    pub fn add(&mut self, key: u64) {
        for pos in self.hash_positions(key) {
            self.words[pos >> 5] |= 1u32 << (pos & 31);
        }
        self.item_count += 1;
    }

    /// Inserts a batch of `keys` and returns the count of newly observed unique keys (keys for
    /// which at least one bit position was previously unset).
    pub fn add_batch_and_count_new(&mut self, keys: &[u64]) -> usize {
        let mut unique_new = 0usize;
        for &key in keys {
            let positions = self.hash_positions(key);
            let is_new = positions
                .iter()
                .any(|&pos| (self.words[pos >> 5] & (1u32 << (pos & 31))) == 0);
            if is_new {
                for pos in positions {
                    self.words[pos >> 5] |= 1u32 << (pos & 31);
                }
                unique_new += 1;
            }
        }
        self.item_count += unique_new;
        unique_new
    }

    /// Returns `true` if all probe bits for `key` are set.
    pub fn contains(&self, key: u64) -> bool {
        self.hash_positions(key)
            .into_iter()
            .all(|pos| (self.words[pos >> 5] & (1u32 << (pos & 31))) != 0)
    }

    /// Total number of set bits across the bit array.
    pub fn bits_set(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Fraction of bits currently set in `[0.0, 1.0]`.
    pub fn fill_rate(&self) -> f64 {
        if self.bit_size == 0 {
            0.0
        } else {
            (self.bits_set() as f64) / (self.bit_size as f64)
        }
    }

    /// Resets all bits and item counters to zero.
    pub fn clear(&mut self) {
        self.words.fill(0);
        self.item_count = 0;
    }
}

/// Status of an asynchronous L2-to-L1 no-lock warm prefetch job
/// (inspired by LMCache `lmcache/v1/multiprocess/warm_prefetch.py` `WarmStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmPrefetchStatus {
    /// Load is still in flight.
    Pending,
    /// Load finished; reports `(found_keys, total_keys)` and consumes the job entry.
    Completed {
        found_keys: usize,
        total_keys: usize,
    },
    /// Unknown request ID (never submitted, or already completed and consumed by an earlier poll).
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WarmPrefetchEntry {
    total_keys: usize,
    completed_found_keys: Option<usize>,
}

/// Table of in-flight no-lock L2-to-L1 warm-prefetch jobs with **exactly-once** completion
/// consumption on `poll`
/// (inspired by LMCache `lmcache/v1/multiprocess/warm_prefetch.py` `WarmPrefetchJobs`).
///
/// A warm prefetch loads keys from L2 into L1 without holding L1 read pins. While in flight,
/// `poll(request_id)` returns [`WarmPrefetchStatus::Pending`]. Once `record_completion` marks the
/// job finished, the **first** `poll(request_id)` returns [`WarmPrefetchStatus::Completed`] and
/// removes the job entry so subsequent polls return [`WarmPrefetchStatus::Unknown`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WarmPrefetchJobTable {
    jobs: IndexMap<String, WarmPrefetchEntry>,
}

impl WarmPrefetchJobTable {
    /// Creates an empty warm-prefetch job table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently tracked jobs (pending or unconsumed completed).
    pub fn active_job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Submits a no-lock retain prefetch job across `key_groups` (`[num_groups][num_keys]`).
    pub fn submit(&mut self, request_id: impl Into<String>, key_groups: &[Vec<u64>]) {
        let total_keys: usize = key_groups.iter().map(Vec::len).sum();
        self.jobs.insert(
            request_id.into(),
            WarmPrefetchEntry {
                total_keys,
                completed_found_keys: None,
            },
        );
    }

    /// Marks `request_id` as completed given per-group hit bitmaps (`[num_groups][num_keys]`),
    /// counting total loaded keys via popcount across all groups.
    pub fn record_completion(&mut self, request_id: &str, hit_bitmaps: &[Vec<bool>]) -> bool {
        let Some(entry) = self.jobs.get_mut(request_id) else {
            return false;
        };
        let found: usize = hit_bitmaps
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&hit| hit)
            .count();
        entry.completed_found_keys = Some(found.min(entry.total_keys));
        true
    }

    /// Polls `request_id`: returns [`WarmPrefetchStatus::Pending`] while in flight, pops and
    /// returns [`WarmPrefetchStatus::Completed`] on the first poll after completion, and returns
    /// [`WarmPrefetchStatus::Unknown`] thereafter.
    pub fn poll(&mut self, request_id: &str) -> WarmPrefetchStatus {
        let Some(entry) = self.jobs.get(request_id) else {
            return WarmPrefetchStatus::Unknown;
        };
        if let Some(found_keys) = entry.completed_found_keys {
            let total_keys = entry.total_keys;
            self.jobs.shift_remove(request_id);
            WarmPrefetchStatus::Completed {
                found_keys,
                total_keys,
            }
        } else {
            WarmPrefetchStatus::Pending
        }
    }
}

/// Layer-group KV payload column sharder and reassembler for wide-column L2 storage
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/bigtable_sharder.py`
/// `BigtablePayloadSharder`).
///
/// Slices a contiguous `[kv_size, num_layers, layer_bytes]` byte payload into bounded layer-group
/// column shards keyed by `layers_{start}_{end}` (where each shard packs all `kv_size` components
/// for `start..=end` contiguously) and losslessly reassembles them back into contiguous
/// `[kv_size, num_layers, layer_bytes]` layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerGroupPayloadSharder {
    pub num_layers: usize,
    pub layer_group_size: usize,
    pub kv_size: usize,
}

impl LayerGroupPayloadSharder {
    /// Creates a new payload sharder for `num_layers`, `layer_group_size`, and `kv_size` (typically
    /// `2` for K and V).
    pub fn new(num_layers: usize, layer_group_size: usize, kv_size: usize) -> Self {
        Self {
            num_layers: num_layers.max(1),
            layer_group_size: layer_group_size.max(1),
            kv_size: kv_size.max(1),
        }
    }

    /// Number of layer groups `ceil(num_layers / layer_group_size)`.
    pub fn num_groups(&self) -> usize {
        self.num_layers.div_ceil(self.layer_group_size)
    }

    /// Column qualifier name (`layers_{start}_{end}`) for `group_idx`.
    pub fn group_qualifier(&self, group_idx: usize) -> String {
        let start = group_idx * self.layer_group_size;
        let end = ((group_idx + 1) * self.layer_group_size)
            .min(self.num_layers)
            .saturating_sub(1);
        format!("layers_{start}_{end}")
    }

    /// Shards contiguous `[kv_size, num_layers, layer_bytes]` `payload` into an ordered map from
    /// column qualifier (`layers_{start}_{end}`) to sharded bytes. Returns `None` if
    /// `payload.len()` is not a multiple of `kv_size * num_layers`.
    pub fn shard(&self, payload: &[u8]) -> Option<IndexMap<String, Vec<u8>>> {
        let denom = self.kv_size * self.num_layers;
        if denom == 0 || !payload.len().is_multiple_of(denom) {
            return None;
        }
        let layer_size = payload.len() / denom;
        let num_groups = self.num_groups();
        let mut shards = IndexMap::with_capacity(num_groups);

        for g in 0..num_groups {
            let start_layer = g * self.layer_group_size;
            let end_layer = ((g + 1) * self.layer_group_size).min(self.num_layers);
            let group_layers = end_layer - start_layer;
            let mut group_bytes = Vec::with_capacity(self.kv_size * group_layers * layer_size);
            for kv_idx in 0..self.kv_size {
                let kv_offset = kv_idx * self.num_layers * layer_size;
                let start = kv_offset + start_layer * layer_size;
                let end = kv_offset + end_layer * layer_size;
                group_bytes.extend_from_slice(&payload[start..end]);
            }
            shards.insert(self.group_qualifier(g), group_bytes);
        }
        Some(shards)
    }

    /// Reassembles `shards` back into contiguous `[kv_size, num_layers, layer_bytes]` bytes.
    /// Returns `None` if any expected group qualifier is missing or has an inconsistent byte size.
    pub fn reassemble(&self, shards: &IndexMap<String, Vec<u8>>) -> Option<Vec<u8>> {
        if shards.is_empty() {
            return Some(Vec::new());
        }
        let num_groups = self.num_groups();
        let first_qual = self.group_qualifier(0);
        let first_data = shards.get(&first_qual)?;
        let first_layers = self.layer_group_size.min(self.num_layers);
        let first_denom = self.kv_size * first_layers;
        if first_denom == 0 || !first_data.len().is_multiple_of(first_denom) {
            return None;
        }
        let layer_size = first_data.len() / first_denom;

        let mut kv_parts: Vec<Vec<u8>> = (0..self.kv_size)
            .map(|_| Vec::with_capacity(self.num_layers * layer_size))
            .collect();

        for g in 0..num_groups {
            let qual = self.group_qualifier(g);
            let shard_data = shards.get(&qual)?;
            let start_layer = g * self.layer_group_size;
            let end_layer = ((g + 1) * self.layer_group_size).min(self.num_layers);
            let group_layers = end_layer - start_layer;
            let expected_size = self.kv_size * group_layers * layer_size;
            if shard_data.len() != expected_size {
                return None;
            }
            let group_part_size = group_layers * layer_size;
            for (kv_idx, part) in kv_parts.iter_mut().enumerate() {
                let start = kv_idx * group_part_size;
                let end = (kv_idx + 1) * group_part_size;
                part.extend_from_slice(&shard_data[start..end]);
            }
        }

        let mut out = Vec::with_capacity(self.kv_size * self.num_layers * layer_size);
        for part in kv_parts {
            out.extend_from_slice(&part);
        }
        Some(out)
    }
}

/// Returns `(provable_factor_bytes, static_total_bytes)` for a tensor type `ty`:
/// - `provable_factor_bytes` is the product of all static dimension extents times the element byte
///   width (matching Modular `isProvableMultipleOf`: a dynamic extent times a static row byte
///   stride is provably a multiple of `alignment` whenever the static factor is a multiple of
///   `alignment`),
/// - `static_total_bytes` is `Some(total_bytes)` when all dimensions are static.
pub(super) fn tensor_provable_byte_factor(ty: &Type) -> Option<(u64, Option<u64>)> {
    let Type::Tensor(tt) = ty else {
        return None;
    };
    let elem_bytes = (tt.dtype.bit_width() as u64).div_ceil(8).max(1);
    let mut static_factor = elem_bytes;
    let mut all_static = true;
    for dim in &tt.shape {
        match dim {
            Dim::Static(d) if *d >= 0 => {
                static_factor = static_factor.saturating_mul(*d as u64);
            }
            _ => {
                all_static = false;
            }
        }
    }
    let total = if all_static {
        Some(static_factor)
    } else {
        None
    };
    Some((static_factor, total))
}

/// Plans alignment-aware in-place buffer concatenation (`stablehlo.concatenate`)
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/ConcatInPlacePass.cpp`).
///
/// A `stablehlo.concatenate` along outer dimension `0` can alias its distinct producer input
/// buffers directly into subviews of the output buffer iff:
/// 1. Every input operand is a distinct `ValueDef::OpResult` (not a block argument or duplicate
///    operand), and
/// 2. Every preceding input slice (`inputs[..inputs.len() - 1]`) has a byte size provably divisible
///    by `alignment_bytes` (either via static total bytes or via a dynamic extent multiplied by an
///    aligned static row byte stride).
///
/// Stamps `castforge.concat_in_place = true`, `castforge.concat_uniform_stride`, and (when slice
/// byte sizes are statically known) `castforge.concat_slice_byte_offsets` onto eligible
/// `stablehlo.concatenate` ops. Returns the number of concatenations planned in-place.
pub fn plan_concat_in_place(module: &mut Module, alignment_bytes: u64) -> usize {
    let align = alignment_bytes.max(1);
    let mut planned = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        for op_id in op_ids {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "concatenate") || op.operands.len() < 2 {
                continue;
            }
            let axis = match op.attributes.get("dimension") {
                Some(Attribute::Int(d)) => *d,
                _ => 0,
            };
            if axis != 0 {
                continue;
            }

            // Inputs must be distinct and produced by explicit ops (`ValueDef::OpResult`).
            let mut seen_inputs = std::collections::HashSet::with_capacity(op.operands.len());
            let mut all_distinct_allocs = true;
            let mut factors_and_totals = Vec::with_capacity(op.operands.len());

            for &in_val in &op.operands {
                if !seen_inputs.insert(in_val) {
                    all_distinct_allocs = false;
                    break;
                }
                let Some(val_info) = module.values.get(in_val) else {
                    all_distinct_allocs = false;
                    break;
                };
                if !matches!(val_info.def, ValueDef::OpResult { .. }) {
                    all_distinct_allocs = false;
                    break;
                }
                let Some(ft) = tensor_provable_byte_factor(&val_info.ty) else {
                    all_distinct_allocs = false;
                    break;
                };
                factors_and_totals.push(ft);
            }
            if !all_distinct_allocs {
                continue;
            }

            // Every slice except the last feeds a subsequent slice's base address and must be a
            // provable multiple of `align`.
            let preceding_aligned = factors_and_totals[..factors_and_totals.len() - 1]
                .iter()
                .all(|&(factor, _)| align <= 1 || factor.is_multiple_of(align));
            if !preceding_aligned {
                continue;
            }

            let static_sizes: Option<Vec<u64>> =
                factors_and_totals.iter().map(|&(_, tot)| tot).collect();
            let uniform_stride = static_sizes
                .as_ref()
                .is_some_and(|sizes| sizes.windows(2).all(|w| w[0] == w[1]));

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.concat_in_place".to_string(),
                    Attribute::Bool(true),
                );
                op_mut.attributes.insert(
                    "castforge.concat_uniform_stride".to_string(),
                    Attribute::Bool(uniform_stride),
                );
                if let Some(sizes) = static_sizes {
                    let offsets: Vec<i64> = if uniform_stride {
                        let slice_sz = sizes[0] as i64;
                        (0..sizes.len()).map(|i| (i as i64) * slice_sz).collect()
                    } else {
                        let mut acc = 0i64;
                        sizes
                            .iter()
                            .map(|&sz| {
                                let cur = acc;
                                acc += sz as i64;
                                cur
                            })
                            .collect()
                    };
                    op_mut.attributes.insert(
                        "castforge.concat_slice_byte_offsets".to_string(),
                        Attribute::DenseIntArray(offsets),
                    );
                }
                planned += 1;
            }
        }
    }

    planned
}

/// Cost weight of a cheap arithmetic/comparison op in a rowwise reduction body (`1`).
pub const ROWWISE_CHEAP_OP_COST: u32 = 1;

/// Cost weight of a division or transcendental op in a rowwise reduction body (`16`)
/// (matching Modular `LegalizeRowwiseKernels.cpp`: `kExpensiveOpCost = 16`).
pub const ROWWISE_EXPENSIVE_OP_COST: u32 = 16;

/// Legalized launch parameters for a rowwise reduction or normalization kernel
/// (inspired by Modular `GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowwiseLaunchPlan {
    /// Operation arena ID in the [`Module`].
    pub op_id: OpId,
    /// Full operation name (e.g., `"castforge.softmax"`, `"stablehlo.reduce"`).
    pub op_name: String,
    /// Per-element weighted instruction cost across the rowwise fold body and prologue.
    pub element_cost: u32,
    /// Whether `element_cost >= expensive_cost_threshold`.
    pub computationally_expensive: bool,
    /// Number of row passes (`1` for pure reductions, `2` for normalize-shaped bodies).
    pub num_phases: u32,
    /// Reduction accumulator bit-width (`max(32, element_dtype_bits)`).
    pub accum_bits: u32,
    /// Static reduction axis extent (`cols`) when statically known.
    pub cols: Option<i64>,
}

pub(super) fn elementwise_op_cost(op: &Operation) -> Option<u32> {
    if op.name.dialect != "stablehlo" {
        return None;
    }
    match op.name.name.as_str() {
        "exponential" | "tanh" | "rsqrt" | "sqrt" | "log" | "logistic" | "divide" | "sine"
        | "cosine" => Some(ROWWISE_EXPENSIVE_OP_COST),
        "add" | "subtract" | "multiply" | "maximum" | "minimum" | "abs" | "negate" | "clamp"
        | "select" => Some(ROWWISE_CHEAP_OP_COST),
        _ => None,
    }
}

/// Scores per-element arithmetic vs. transcendental cost for rowwise reduction and normalization
/// kernels (`castforge.softmax`, `castforge.log_softmax`, `castforge.rmsnorm`,
/// `castforge.layernorm`, `stablehlo.reduce`) and stamps launch parameters onto each op
/// (inspired by Modular `GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp`
/// `setLaunchParams` and `bindMonoids`).
pub fn legalize_rowwise_reduction_launch(
    module: &mut Module,
    expensive_cost_threshold: u32,
) -> Vec<RowwiseLaunchPlan> {
    let mut plans = Vec::new();
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        for op_id in op_ids {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let op_name = op.name.to_string();
            let (base_cost, num_phases) = match op_name.as_str() {
                "castforge.softmax" | "castforge.log_softmax" => {
                    (2 * ROWWISE_EXPENSIVE_OP_COST + 3 * ROWWISE_CHEAP_OP_COST, 2)
                }
                "castforge.rmsnorm" => (ROWWISE_EXPENSIVE_OP_COST + 3 * ROWWISE_CHEAP_OP_COST, 2),
                "castforge.layernorm" => (ROWWISE_EXPENSIVE_OP_COST + 6 * ROWWISE_CHEAP_OP_COST, 2),
                "stablehlo.reduce" => (ROWWISE_CHEAP_OP_COST, 1),
                _ => continue,
            };

            if op.operands.is_empty() {
                continue;
            }
            let primary_in = op.operands[0];

            // Walk backward up sole-use elementwise producers feeding `primary_in` to accumulate
            // fused per-element prologue cost.
            let mut prologue_cost = 0u32;
            let mut cur_val = primary_in;
            while let Some(v_info) = module.values.get(cur_val) {
                let ValueDef::OpResult { op: prod_id, .. } = v_info.def else {
                    break;
                };
                let Some(prod_op) = module.ops.get(prod_id) else {
                    break;
                };
                let Some(c) = elementwise_op_cost(prod_op) else {
                    break;
                };
                prologue_cost += c;
                if let Some(&first_arg) = prod_op.operands.first() {
                    cur_val = first_arg;
                } else {
                    break;
                }
            }

            let element_cost = base_cost + prologue_cost;
            let computationally_expensive = element_cost >= expensive_cost_threshold;

            let (accum_bits, cols) = if let Some(v_info) = module.values.get(primary_in)
                && let Type::Tensor(ref tt) = v_info.ty
            {
                let bits = (tt.dtype.bit_width() as u32).max(32);
                let last_col = match tt.shape.last() {
                    Some(Dim::Static(d)) if *d > 0 => Some(*d),
                    _ => None,
                };
                (bits, last_col)
            } else {
                (32, None)
            };

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.rowwise_element_cost".to_string(),
                    Attribute::Int(element_cost as i64),
                );
                op_mut.attributes.insert(
                    "castforge.rowwise_expensive".to_string(),
                    Attribute::Bool(computationally_expensive),
                );
                op_mut.attributes.insert(
                    "castforge.rowwise_num_phases".to_string(),
                    Attribute::Int(num_phases as i64),
                );
                op_mut.attributes.insert(
                    "castforge.rowwise_accum_bits".to_string(),
                    Attribute::Int(accum_bits as i64),
                );
                if let Some(c) = cols {
                    op_mut
                        .attributes
                        .insert("castforge.rowwise_cols".to_string(), Attribute::Int(c));
                }
            }

            plans.push(RowwiseLaunchPlan {
                op_id,
                op_name,
                element_cost,
                computationally_expensive,
                num_phases,
                accum_bits,
                cols,
            });
        }
    }

    plans
}
