//! Elementwise chain flattening, state transfer elision, and memory pool passes.

use super::*;
use crate::ir::{
    Attribute, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type, ValueDef,
    ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

fn is_flattenable_elemwise_op(op: &Operation) -> bool {
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

fn is_named_state_op(op: &Operation, short_name: &str) -> bool {
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

fn expr_contains_ident(text: &str, ident: &str) -> bool {
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

fn stmt_references_buffer(stmt: &crate::ir::loops::LoopStmt, buf_name: &str) -> bool {
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

fn is_shape_materialization_op(name: &str) -> bool {
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

type ShapeOpSignature = (String, Vec<ValueId>, Vec<Type>, String);

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
struct WarmPrefetchEntry {
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
fn tensor_provable_byte_factor(ty: &Type) -> Option<(u64, Option<u64>)> {
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

fn elementwise_op_cost(op: &Operation) -> Option<u32> {
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

/// Canonical L2 object key identity shared across storage and filesystem adapters
/// (inspired by LMCache `lmcache/v1/distributed/key_codec.py` `ObjectKey`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvStorageKeySpec {
    /// Model identifier (may contain `/` separators, e.g., `"meta-llama/Llama-3-8B"`).
    pub model_name: String,
    /// KV tensor-parallel / pipeline rank.
    pub kv_rank: u32,
    /// Heterogeneous object group index (`0` for primary attention group, `1` for sliding/Mamba).
    pub object_group_id: u32,
    /// 64-bit prefix chunk hash (`chunk_hash`).
    pub chunk_hash: u64,
    /// Optional tenant isolation `cache_salt` (empty string when unsalted).
    pub cache_salt: String,
}

/// Encodes `key` into the canonical reversible L2 wire key
/// `"{model}@{rank:08x}@{group:x}@{hash:016x}[@{salt}]"` (omitting trailing `@{salt}` when
/// `cache_salt` is empty, matching LMCache `encode_storage_key`).
pub fn encode_kv_storage_key(key: &KvStorageKeySpec) -> String {
    if key.cache_salt.is_empty() {
        format!(
            "{}@{:08x}@{:x}@{:016x}",
            key.model_name, key.kv_rank, key.object_group_id, key.chunk_hash
        )
    } else {
        format!(
            "{}@{:08x}@{:x}@{:016x}@{}",
            key.model_name, key.kv_rank, key.object_group_id, key.chunk_hash, key.cache_salt
        )
    }
}

/// Encodes `key` into the filesystem L2 adapter filename
/// `"{model_sep}@0x{rank:08x}@{group:x}@{hash:016x}[@{salt}].data"` where `'/'` in `model_name` is
/// escaped as `"-SEP-"` (matching LMCache `encode_fs_filename`).
pub fn encode_kv_fs_filename(key: &KvStorageKeySpec) -> String {
    let escaped_model = key.model_name.replace('/', "-SEP-");
    if key.cache_salt.is_empty() {
        format!(
            "{escaped_model}@0x{:08x}@{:x}@{:016x}.data",
            key.kv_rank, key.object_group_id, key.chunk_hash
        )
    } else {
        format!(
            "{escaped_model}@0x{:08x}@{:x}@{:016x}@{}.data",
            key.kv_rank, key.object_group_id, key.chunk_hash, key.cache_salt
        )
    }
}

/// Decodes a canonical L2 wire key (`model@rank@group@hash[@salt]`) back into a
/// [`KvStorageKeySpec`] (matching LMCache `decode_storage_key`).
pub fn decode_kv_storage_key(encoded: &str) -> Option<KvStorageKeySpec> {
    let parts: Vec<&str> = encoded.split('@').collect();
    let (model, rank_str, group_str, hash_str, salt) = match parts.as_slice() {
        [m, r, g, h] => (*m, *r, *g, *h, ""),
        [m, r, g, h, s] => (*m, *r, *g, *h, *s),
        _ => return None,
    };
    if model.is_empty() || hash_str.is_empty() {
        return None;
    }
    let rank_clean = rank_str
        .strip_prefix("0x")
        .or_else(|| rank_str.strip_prefix("0X"))
        .unwrap_or(rank_str);
    let kv_rank = u32::from_str_radix(rank_clean, 16).ok()?;
    let object_group_id = u32::from_str_radix(group_str, 16).ok()?;
    let chunk_hash = u64::from_str_radix(hash_str, 16).ok()?;
    Some(KvStorageKeySpec {
        model_name: model.to_string(),
        kv_rank,
        object_group_id,
        chunk_hash,
        cache_salt: salt.to_string(),
    })
}

/// Decodes a filesystem L2 filename (`*.data`) back into a [`KvStorageKeySpec`], unescaping
/// `"-SEP-"` back to `'/'` (matching LMCache `decode_fs_filename`).
pub fn decode_kv_fs_filename(filename: &str) -> Option<KvStorageKeySpec> {
    let stem = filename.strip_suffix(".data")?;
    let mut spec = decode_kv_storage_key(stem)?;
    spec.model_name = spec.model_name.replace("-SEP-", "/");
    Some(spec)
}

const FAULT_INJECT_HASH_DENOM: u64 = 1_000_000;

/// Configuration for deterministic L2 load fault injection
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/fault_inject_l2_adapter.py`
/// `FaultInjectL2AdapterConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct L2FaultInjectConfig {
    /// Deterministic per-key drop probability in `[0.0, 1.0]`.
    pub rate: f64,
    /// Deterministic hash seed so drop decisions reproduce across runs.
    pub seed: u64,
    /// Exact head-relative chunk indices to drop unconditionally.
    pub gap_indices: Vec<usize>,
    /// Workload-agnostic tail-relative ratios in `[0.0, 1.0]`: drops chunk index
    /// `round((1.0 - ratio) * (n - 1))` (`0.0` = last chunk, `0.5` = middle, `1.0` = first).
    pub gap_tail_ratios: Vec<f64>,
}

/// Deterministic L2 load fault-injection policy that faults load results (leaving lookup bitmaps
/// untouched) to exercise segmented gapped prefetch and lock-release trim masks
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/fault_inject_l2_adapter.py`).
#[derive(Debug, Clone, PartialEq)]
pub struct L2FaultInjectPolicy {
    pub config: L2FaultInjectConfig,
}

impl L2FaultInjectPolicy {
    /// Creates a new L2 load fault-injection policy.
    pub fn new(config: L2FaultInjectConfig) -> Self {
        Self { config }
    }

    /// Returns `true` if `key` falls into the deterministic `(seed, key)` rate-based drop bucket.
    pub fn should_drop_key(&self, key: &KvStorageKeySpec) -> bool {
        let rate = self.config.rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return false;
        }
        // Hash stable identity fields directly (`seed`, `chunk_hash`, `model_name`, `kv_rank`,
        // `object_group_id`, `cache_salt`).
        let mut h = self.config.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mix = |acc: u64, val: u64| -> u64 {
            let mut z = acc.wrapping_add(val).wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        h = mix(h, key.chunk_hash);
        h = mix(h, key.kv_rank as u64);
        h = mix(h, key.object_group_id as u64);
        for b in key.model_name.bytes().chain(key.cache_salt.bytes()) {
            h = mix(h, b as u64);
        }
        let bucket = h % FAULT_INJECT_HASH_DENOM;
        bucket < ((rate * (FAULT_INJECT_HASH_DENOM as f64)) as u64)
    }

    /// Returns the sorted positions in `keys` that are dropped at load time (combining
    /// `gap_tail_ratios`, `gap_indices`, and `should_drop_key`).
    pub fn drop_positions(&self, keys: &[KvStorageKeySpec]) -> Vec<usize> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }
        let ratio_positions: std::collections::HashSet<usize> = self
            .config
            .gap_tail_ratios
            .iter()
            .filter(|&&r| (0.0..=1.0).contains(&r))
            .map(|&r| ((1.0 - r) * ((n - 1) as f64)).round() as usize)
            .collect();
        let explicit_gaps: std::collections::HashSet<usize> =
            self.config.gap_indices.iter().copied().collect();

        let mut dropped = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            if ratio_positions.contains(&i)
                || explicit_gaps.contains(&i)
                || self.should_drop_key(key)
            {
                dropped.push(i);
            }
        }
        dropped
    }

    /// Post-processes a lookup bitmap into `(faulted_load_bitmap, failed_lock_trim_mask)`:
    /// lookup hits at dropped positions are cleared in `faulted_load_bitmap` and set in
    /// `failed_lock_trim_mask` so the prefetch controller releases their L2 locks.
    pub fn fault_load_bitmap(
        &self,
        keys: &[KvStorageKeySpec],
        lookup_bitmap: &[bool],
    ) -> (Vec<bool>, Vec<bool>) {
        let len = keys.len().min(lookup_bitmap.len());
        let mut load_bitmap = lookup_bitmap[..len].to_vec();
        let mut trim_mask = vec![false; len];
        for pos in self.drop_positions(&keys[..len]) {
            if pos < len && load_bitmap[pos] {
                load_bitmap[pos] = false;
                trim_mask[pos] = true;
            }
        }
        (load_bitmap, trim_mask)
    }
}

fn replace_ident_in_text(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() || from == to {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let id_len = from.len();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find(from) {
        let pos = cursor + rel;
        let before_ok = pos == 0 || !is_word(bytes[pos - 1]);
        let after_ok = pos + id_len >= bytes.len() || !is_word(bytes[pos + id_len]);
        if before_ok && after_ok {
            out.push_str(&text[cursor..pos]);
            out.push_str(to);
            cursor = pos + id_len;
        } else {
            out.push_str(&text[cursor..pos + 1]);
            cursor = pos + 1;
        }
    }
    out.push_str(&text[cursor..]);
    out
}

fn rewrite_stmt_buffer_name(stmt: &mut crate::ir::loops::LoopStmt, from: &str, to: &str) {
    use crate::ir::loops::LoopStmt;
    match stmt {
        LoopStmt::Comment(_) | LoopStmt::Alloc { .. } => {}
        LoopStmt::Assign { dst, index, expr } => {
            if dst == from {
                *dst = to.to_string();
            }
            *index = replace_ident_in_text(index, from, to);
            *expr = replace_ident_in_text(expr, from, to);
        }
        LoopStmt::For { body, .. } => {
            for inner in body {
                rewrite_stmt_buffer_name(inner, from, to);
            }
        }
    }
}

/// Aliases non-overlapping `LoopStmt::Alloc` scratchpad buffers of identical length within each
/// [`crate::ir::loops::LoopKernel`], prioritizing buffers with higher use counts first
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`).
///
/// Returns the number of redundant `LoopStmt::Alloc` declarations merged and eliminated.
pub fn reuse_dead_loop_buffers(loop_module: &mut crate::ir::loops::LoopModule) -> usize {
    use crate::ir::loops::LoopStmt;
    let mut reused_count = 0usize;

    for kernel in &mut loop_module.kernels {
        let mut size_groups: IndexMap<usize, Vec<(String, Vec<usize>)>> = IndexMap::new();
        for stmt in &kernel.body {
            if let LoopStmt::Alloc { name, len } = stmt {
                let uses: Vec<usize> = kernel
                    .body
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, s)| stmt_references_buffer(s, name).then_some(idx))
                    .collect();
                if !uses.is_empty() {
                    size_groups
                        .entry(*len)
                        .or_default()
                        .push((name.clone(), uses));
                }
            }
        }

        let mut erased_allocs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut rewrites: Vec<(String, String)> = Vec::new();

        for (_, mut group) in size_groups {
            if group.len() < 2 {
                continue;
            }
            // Allocs with more uses go first (`SimpleBufferReusePass.cpp`).
            group.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

            let mut active = vec![true; group.len()];
            for i in 0..group.len() {
                if !active[i] {
                    continue;
                }
                for j in 0..group.len() {
                    if i == j || !active[j] {
                        continue;
                    }
                    let cur_uses = &group[i].1;
                    let cand_uses = &group[j].1;
                    let shares_op = cur_uses.iter().any(|u| cand_uses.contains(u));
                    if shares_op {
                        continue;
                    }
                    let cur_after_cand = cur_uses
                        .iter()
                        .all(|&cu| cand_uses.iter().all(|&du| cu > du));
                    let cand_after_cur = cand_uses
                        .iter()
                        .all(|&du| cur_uses.iter().all(|&cu| du > cu));
                    if cur_after_cand || cand_after_cur {
                        let primary_name = group[i].0.clone();
                        let cand_name = group[j].0.clone();
                        let mut merged_uses = group[i].1.clone();
                        merged_uses.extend_from_slice(&group[j].1);
                        merged_uses.sort_unstable();
                        group[i].1 = merged_uses;
                        active[j] = false;
                        erased_allocs.insert(cand_name.clone());
                        rewrites.push((cand_name, primary_name));
                        reused_count += 1;
                    }
                }
            }
        }

        if !rewrites.is_empty() {
            for (from, to) in &rewrites {
                for stmt in &mut kernel.body {
                    rewrite_stmt_buffer_name(stmt, from, to);
                }
            }
            // Remove erased candidate `Alloc` statements and ensure surviving `Alloc`s precede
            // their earliest use.
            let mut hoisted_allocs = Vec::new();
            let mut remaining = Vec::with_capacity(kernel.body.len());
            for stmt in kernel.body.drain(..) {
                if let LoopStmt::Alloc { ref name, .. } = stmt {
                    if erased_allocs.contains(name) {
                        continue;
                    }
                    hoisted_allocs.push(stmt);
                } else {
                    remaining.push(stmt);
                }
            }
            hoisted_allocs.append(&mut remaining);
            kernel.body = hoisted_allocs;
        }
    }

    reused_count
}

fn is_invariant_root_op(op: &Operation) -> bool {
    op.operands.is_empty()
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.constant" | "castforge.weight_ref" | "stablehlo.iota"
        )
}

fn is_trivial_invariant_op(op: &Operation) -> bool {
    matches!(
        op.name.to_string().as_str(),
        "stablehlo.constant"
            | "castforge.weight_ref"
            | "stablehlo.iota"
            | "stablehlo.reshape"
            | "stablehlo.broadcast_in_dim"
            | "stablehlo.bitcast_convert"
    )
}

fn is_terminator_or_side_effecting(op: &Operation) -> bool {
    matches!(
        op.name.to_string().as_str(),
        "func.return" | "return" | "stablehlo.return" | "castforge.nan_check"
    )
}

/// Identifies execution-invariant weight/constant subgraphs, filters out standalone trivial ops
/// (`isLiftableTrivialOp`) that do not feed any non-trivial invariant compute op, assigns
/// `castforge.exec_invariant_slot` cache indices to non-trivial invariant ops, and hoists the
/// lifted subgraph to the entry of the block
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp`).
///
/// Returns the number of non-trivial execution-invariant operations assigned cache slots.
pub fn hoist_exec_invariant_ops(module: &mut Module) -> usize {
    let mut cached_nontrivial = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let Some(&reg_id) = module.ops[func_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let op_ids = module.blocks[block_id].ops.clone();

        // Forward pass: find all ops whose operands come strictly from other invariant ops.
        let mut invariant_ops: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        let mut invariant_vals: std::collections::HashSet<ValueId> =
            std::collections::HashSet::new();

        for &op_id in &op_ids {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if is_terminator_or_side_effecting(op) || !op.regions.is_empty() {
                continue;
            }
            let is_inv = if op.operands.is_empty() {
                is_invariant_root_op(op)
            } else {
                op.operands.iter().all(|v| invariant_vals.contains(v))
            };
            if is_inv {
                invariant_ops.insert(op_id);
                for &res in &op.results {
                    invariant_vals.insert(res);
                }
            }
        }

        // Backward pass from non-trivial invariant ops: trivial ops are only lifted if they feed
        // at least one non-trivial invariant op (`isLiftableTrivialOp` in Modular).
        let mut lifted_ops: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        for &op_id in op_ids.iter().rev() {
            if !invariant_ops.contains(&op_id) {
                continue;
            }
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let should_lift = !is_trivial_invariant_op(op) || lifted_ops.contains(&op_id);
            if should_lift {
                lifted_ops.insert(op_id);
                for &in_val in &op.operands {
                    if let Some(v_info) = module.values.get(in_val)
                        && let ValueDef::OpResult { op: prod_id, .. } = v_info.def
                        && invariant_ops.contains(&prod_id)
                    {
                        lifted_ops.insert(prod_id);
                    }
                }
            }
        }

        if lifted_ops.is_empty() {
            continue;
        }

        let mut next_slot = 0i64;
        let mut hoisted = Vec::with_capacity(lifted_ops.len());
        let mut runtime_ops = Vec::with_capacity(op_ids.len());

        for op_id in op_ids {
            if lifted_ops.contains(&op_id) {
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.attributes.insert(
                        "castforge.exec_invariant_init".to_string(),
                        Attribute::Bool(true),
                    );
                    if !is_trivial_invariant_op(op_mut) {
                        op_mut.attributes.insert(
                            "castforge.exec_invariant_slot".to_string(),
                            Attribute::Int(next_slot),
                        );
                        next_slot += 1;
                        cached_nontrivial += 1;
                    }
                }
                hoisted.push(op_id);
            } else {
                runtime_ops.push(op_id);
            }
        }

        hoisted.append(&mut runtime_ops);
        module.blocks[block_id].ops = hoisted;
    }

    cached_nontrivial
}

fn is_complex_symbolic_expr(sym: &str) -> bool {
    !sym.is_empty() && !sym.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn rewrite_type_symbolic_exprs(ty: &mut Type, expr_to_alias: &IndexMap<String, String>) -> bool {
    match ty {
        Type::Tensor(tt) => {
            let mut changed = false;
            for dim in &mut tt.shape {
                if let Dim::Symbolic(s) = dim
                    && is_complex_symbolic_expr(s)
                {
                    let alias = expr_to_alias
                        .get(s)
                        .cloned()
                        .unwrap_or_else(|| format!("param_expr_{}", expr_to_alias.len()));
                    *s = alias;
                    changed = true;
                }
            }
            changed
        }
        Type::Function { inputs, results } => {
            let mut changed = false;
            for inp in inputs {
                changed |= rewrite_type_symbolic_exprs(inp, expr_to_alias);
            }
            for res in results {
                changed |= rewrite_type_symbolic_exprs(res, expr_to_alias);
            }
            changed
        }
        Type::Tuple(elems) => {
            let mut changed = false;
            for elem in elems {
                changed |= rewrite_type_symbolic_exprs(elem, expr_to_alias);
            }
            changed
        }
        _ => false,
    }
}

fn collect_type_symbolic_exprs(ty: &Type, expr_to_alias: &mut IndexMap<String, String>) {
    match ty {
        Type::Tensor(tt) => {
            for dim in &tt.shape {
                if let Dim::Symbolic(s) = dim
                    && is_complex_symbolic_expr(s)
                    && !expr_to_alias.contains_key(s)
                {
                    let alias = format!("param_expr_{}", expr_to_alias.len());
                    expr_to_alias.insert(s.clone(), alias);
                }
            }
        }
        Type::Function { inputs, results } => {
            for inp in inputs {
                collect_type_symbolic_exprs(inp, expr_to_alias);
            }
            for res in results {
                collect_type_symbolic_exprs(res, expr_to_alias);
            }
        }
        Type::Tuple(elems) => {
            for elem in elems {
                collect_type_symbolic_exprs(elem, expr_to_alias);
            }
        }
        _ => {}
    }
}

/// Hoists compound symbolic dimension expressions (e.g., `Dim::Symbolic("batch*seq")`,
/// `Dim::Symbolic("seq+1")`) out of tensor types into deduplicated function-level parameter
/// declarations (`param_expr_0`, `param_expr_1`, ...) stored in `castforge.param_expr_decls`,
/// rewriting all tensor shapes in the function to reference the hoisted parameter aliases
/// (inspired by Modular
/// `GraphCompiler/lib/MODialect/Transforms/HoistParameterExpressionsPass.cpp`).
///
/// Returns the total number of distinct symbolic parameter expressions hoisted across the module.
pub fn hoist_symbolic_dim_expressions(module: &mut Module) -> usize {
    let mut total_hoisted = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let mut expr_to_alias: IndexMap<String, String> = IndexMap::new();

        if let Some(Attribute::Type(ty)) = module.ops[func_id].attributes.get("function_type") {
            collect_type_symbolic_exprs(ty, &mut expr_to_alias);
        }

        let reg_ids = module.ops[func_id].regions.clone();
        let mut val_ids = Vec::new();
        for reg_id in reg_ids {
            let block_ids = module.regions[reg_id].blocks.clone();
            for block_id in block_ids {
                for &arg_val in &module.blocks[block_id].arguments {
                    val_ids.push(arg_val);
                }
                for &op_id in &module.blocks[block_id].ops {
                    if let Some(op) = module.ops.get(op_id) {
                        for &res_val in &op.results {
                            val_ids.push(res_val);
                        }
                    }
                }
            }
        }

        for &val_id in &val_ids {
            if let Some(v_info) = module.values.get(val_id) {
                collect_type_symbolic_exprs(&v_info.ty, &mut expr_to_alias);
            }
        }

        if expr_to_alias.is_empty() {
            continue;
        }
        total_hoisted += expr_to_alias.len();

        for &val_id in &val_ids {
            if let Some(v_info) = module.values.get_mut(val_id) {
                rewrite_type_symbolic_exprs(&mut v_info.ty, &expr_to_alias);
            }
        }

        if let Some(func_op) = module.ops.get_mut(func_id) {
            if let Some(Attribute::Type(ty)) = func_op.attributes.get_mut("function_type") {
                rewrite_type_symbolic_exprs(ty, &expr_to_alias);
            }
            let decls: IndexMap<String, Attribute> = expr_to_alias
                .into_iter()
                .map(|(expr, alias)| (alias, Attribute::String(expr)))
                .collect();
            func_op.attributes.insert(
                "castforge.param_expr_decls".to_string(),
                Attribute::Dictionary(decls),
            );
        }
    }

    total_hoisted
}

/// Downsamples per-group paged KV `block_ids` for sub-chunk sliding-window attention (SWA) groups
/// (inspired by LMCache `lmcache/v1/multiprocess/object_group_transfer.py`
/// `downsample_and_stage_block_ids`).
///
/// For a kernel group with `block_size_tokens`, `chunk_tokens`, and `subchunk_sw_tokens` (`0` or
/// `>= chunk_tokens` for full attention; `< chunk_tokens` for sub-chunk SWA such as DeepSeek-V4):
/// - `total_blocks_per_chunk = chunk_tokens.div_ceil(block_size_tokens)`
/// - `keep_blocks_per_chunk = chunk_tokens.min(subchunk_sw_tokens).div_ceil(block_size_tokens)`
///
/// Keeps only the trailing `keep_blocks_per_chunk` block IDs from each chunk of
/// `total_blocks_per_chunk` blocks. Returns `None` if `group_block_ids.len()` is not a multiple of
/// `total_blocks_per_chunk`.
pub fn downsample_subchunk_swa_block_ids(
    group_block_ids: &[u64],
    chunk_tokens: usize,
    subchunk_sw_tokens: usize,
    block_size_tokens: usize,
) -> Option<Vec<u64>> {
    if chunk_tokens == 0 || block_size_tokens == 0 {
        return None;
    }
    let eff_sw = if subchunk_sw_tokens == 0 {
        chunk_tokens
    } else {
        chunk_tokens.min(subchunk_sw_tokens)
    };
    let total_blocks_per_chunk = chunk_tokens.div_ceil(block_size_tokens);
    let keep_blocks_per_chunk = eff_sw.div_ceil(block_size_tokens);
    if !group_block_ids.len().is_multiple_of(total_blocks_per_chunk) {
        return None;
    }
    if keep_blocks_per_chunk >= total_blocks_per_chunk {
        return Some(group_block_ids.to_vec());
    }
    let mut out = Vec::with_capacity(
        (group_block_ids.len() / total_blocks_per_chunk) * keep_blocks_per_chunk,
    );
    for chunk in group_block_ids.chunks_exact(total_blocks_per_chunk) {
        let start = total_blocks_per_chunk - keep_blocks_per_chunk;
        out.extend_from_slice(&chunk[start..]);
    }
    Some(out)
}

/// Recalculates the number of downsampled blocks to skip when `blocks_per_window <=
/// blocks_per_chunk` (inspired by LMCache `lmcache/v1/multiprocess/object_group_transfer.py`
/// `recalculate_blocks_to_skip`).
pub fn recalculate_subchunk_blocks_to_skip(
    blocks_per_chunk: usize,
    blocks_per_window: usize,
    blocks_to_skip: usize,
) -> usize {
    if blocks_per_chunk == 0 || blocks_per_window >= blocks_per_chunk {
        return blocks_to_skip;
    }
    let full_windows_to_skip = blocks_to_skip / blocks_per_chunk;
    let tail_blocks = blocks_to_skip % blocks_per_chunk;
    let dropped_per_chunk = blocks_per_chunk - blocks_per_window;
    let tail_blocks_to_skip = tail_blocks.saturating_sub(dropped_per_chunk);
    full_windows_to_skip * blocks_per_window + tail_blocks_to_skip
}

/// Bytes per packed wire token ID (big-endian `u32`, matching LMCache `TOKEN_STRIDE = 4`).
pub const TOKEN_WIRE_STRIDE: usize = 4;

/// Packs `token_ids` into a big-endian `u32` wire byte buffer (`TOKEN_WIRE_STRIDE * len(tokens)`)
/// so token sequences cross process boundaries via a single `memcpy` and hash without unpacking
/// (inspired by LMCache `lmcache/v1/multiprocess/token_codec.py` `pack_token_ids`).
pub fn pack_token_ids_be(token_ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(token_ids.len() * TOKEN_WIRE_STRIDE);
    for &tok in token_ids {
        out.extend_from_slice(&tok.to_be_bytes());
    }
    out
}

/// Counts the number of whole packed tokens in `packed` (`packed.len() / TOKEN_WIRE_STRIDE`).
pub fn num_packed_tokens(packed: &[u8]) -> usize {
    packed.len() / TOKEN_WIRE_STRIDE
}

/// Unpacks a big-endian `u32` wire buffer back into token IDs, returning `None` if `packed.len()`
/// is not an exact multiple of [`TOKEN_WIRE_STRIDE`]
/// (matching LMCache `lmcache/v1/multiprocess/token_codec.py` `unpack_token_ids`).
pub fn unpack_token_ids_be(packed: &[u8]) -> Option<Vec<u32>> {
    if !packed.len().is_multiple_of(TOKEN_WIRE_STRIDE) {
        return None;
    }
    let mut out = Vec::with_capacity(packed.len() / TOKEN_WIRE_STRIDE);
    for &chunk in packed.as_chunks::<TOKEN_WIRE_STRIDE>().0 {
        out.push(u32::from_be_bytes(chunk));
    }
    Some(out)
}

/// Computes chained prefix-chunk hashes directly over a big-endian packed token wire buffer
/// without unpacking into `Vec<u32>`, producing hashes identical to hashing `[u32]` chunks
/// (inspired by LMCache `lmcache/v1/multiprocess/token_codec.py` & `token_hasher.py`).
pub fn hash_packed_token_blocks_be(packed: &[u8], chunk_tokens: usize, seed: u64) -> Vec<u64> {
    let stride_bytes = chunk_tokens.saturating_mul(TOKEN_WIRE_STRIDE);
    if stride_bytes == 0 || packed.len() < stride_bytes {
        return Vec::new();
    }
    let mut hashes = Vec::with_capacity(packed.len() / stride_bytes);
    let mut parent_hash = seed;
    for chunk_bytes in packed.chunks_exact(stride_bytes) {
        let mut h = parent_hash ^ 0x9E37_79B9_7F4A_7C15;
        for &word in chunk_bytes.as_chunks::<TOKEN_WIRE_STRIDE>().0 {
            let tok = u32::from_be_bytes(word) as u64;
            h = h.wrapping_add(tok).wrapping_add(0x9E37_79B9_7F4A_7C15);
            h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h ^= h >> 31;
        }
        hashes.push(h);
        parent_hash = h;
    }
    hashes
}

/// Descriptor for an intermediate buffer allocation input to [`plan_dynamic_memory_pool`]
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
/// `AllocInfo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicAllocRequest {
    /// Buffer symbol / identifier.
    pub name: String,
    /// Allocation size in bytes.
    pub size_bytes: u64,
    /// Required base-address alignment in bytes.
    pub alignment_bytes: u64,
    /// Earliest topological statement/operation use index.
    pub first_use_idx: usize,
    /// Latest topological statement/operation use index.
    pub last_use_idx: usize,
    /// Source location of the allocation op.
    pub location: Location,
}

/// Bulk-slice memory pool plan produced by [`plan_dynamic_memory_pool`]
/// (inspired by Modular `DynamicMemoryPlanningPass.cpp` `BufferPlanOp` + `BufferBulkSliceOp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicMemoryPoolPlan {
    /// Allocations ordered by ascending `first_use_idx`.
    pub ordered_names: Vec<String>,
    /// Assigned byte offset for each ordered allocation inside the shared memory pool.
    pub bulk_slice_offsets: Vec<u64>,
    /// Byte size for each ordered allocation.
    pub bulk_slice_sizes: Vec<u64>,
    /// Symmetric `n * n` pairwise shareability matrix (`can_share[i * n + j]`).
    pub can_share: Vec<bool>,
    /// Pool alignment in bytes (`LCM` of all constituent buffer alignments).
    pub pool_alignment_bytes: u64,
    /// Peak byte footprint (`high_watermark`) of the consolidated memory pool.
    pub high_watermark_bytes: u64,
    /// Fused source provenance across all consolidated buffer allocations.
    pub fused_location: Location,
}

/// Plans a consolidated LCM-aligned memory pool and single `BufferBulkSliceOp` layout for
/// intermediate buffer allocations
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`).
///
/// 1. Sorts allocations by earliest topological use (`first_use_idx`),
/// 2. Computes `pool_alignment_bytes = LCM(alignments)`,
/// 3. Builds the symmetric `n * n` `can_share` matrix (`true` iff all uses of `i` execute strictly
///    after all uses of `j` or vice versa),
/// 4. Greedily packs buffers into aligned offsets while respecting non-shareable interference, and
/// 5. Fuses all allocation [`Location`]s via [`fuse_locations`].
pub fn plan_dynamic_memory_pool(allocs: &[DynamicAllocRequest]) -> DynamicMemoryPoolPlan {
    let mut sorted = allocs.to_vec();
    sorted.sort_by_key(|a| a.first_use_idx);

    let n = sorted.len();
    let mut pool_alignment_bytes = 1u64;
    let mut locs = Vec::with_capacity(n);
    for a in &sorted {
        pool_alignment_bytes =
            checked_lcm_u64(pool_alignment_bytes, a.alignment_bytes.max(1)).unwrap_or(u64::MAX);
        locs.push(a.location.clone());
    }

    let mut can_share = vec![false; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sharable = sorted[i].last_use_idx < sorted[j].first_use_idx
                || sorted[j].last_use_idx < sorted[i].first_use_idx;
            can_share[i * n + j] = sharable;
            can_share[j * n + i] = sharable;
        }
    }

    let mut offsets = vec![0u64; n];
    let mut high_watermark_bytes = 0u64;

    for i in 0..n {
        let align = sorted[i].alignment_bytes.max(1);
        let size_i = sorted[i].size_bytes;
        let mut cand_offset = 0u64;

        // Advance `cand_offset` past any interfering (`!can_share`) already-placed buffer `j < i`.
        loop {
            cand_offset = cand_offset.div_ceil(align) * align;
            let end_i = cand_offset + size_i;
            let conflict = (0..i).find(|&j| {
                if can_share[i * n + j] {
                    return false;
                }
                let start_j = offsets[j];
                let end_j = start_j + sorted[j].size_bytes;
                cand_offset < end_j && start_j < end_i
            });
            if let Some(j) = conflict {
                cand_offset = offsets[j] + sorted[j].size_bytes;
            } else {
                break;
            }
        }

        offsets[i] = cand_offset;
        high_watermark_bytes = high_watermark_bytes.max(cand_offset + size_i);
    }

    DynamicMemoryPoolPlan {
        ordered_names: sorted.iter().map(|a| a.name.clone()).collect(),
        bulk_slice_offsets: offsets,
        bulk_slice_sizes: sorted.iter().map(|a| a.size_bytes).collect(),
        can_share,
        pool_alignment_bytes,
        high_watermark_bytes,
        fused_location: fuse_locations(&locs),
    }
}

fn stmt_has_non_pointwise_access(
    stmt: &crate::ir::loops::LoopStmt,
    buf_name: &str,
    loop_var: &str,
) -> bool {
    use crate::ir::loops::LoopStmt;
    let exact_ref = format!("{buf_name}[{loop_var}]");
    match stmt {
        LoopStmt::Comment(_) | LoopStmt::Alloc { .. } => false,
        LoopStmt::Assign { dst, index, expr } => {
            if dst == buf_name && index != loop_var {
                return true;
            }
            let stripped_idx = index.replace(&exact_ref, "");
            let stripped_expr = expr.replace(&exact_ref, "");
            expr_contains_ident(&stripped_idx, buf_name)
                || expr_contains_ident(&stripped_expr, buf_name)
        }
        LoopStmt::For { body, .. } => body
            .iter()
            .any(|s| stmt_has_non_pointwise_access(s, buf_name, loop_var)),
    }
}

/// Merges adjacent parallel `LoopStmt::For` regions sharing identical `(lower, upper, step)`
/// bounds when all shared producer-to-consumer buffer accesses are strictly pointwise at `[var]`,
/// forwarding sole-use producer yields directly into the consumer body and dropping dead producer
/// intermediate stores and allocations
/// (inspired by Modular `GraphCompiler/lib/MODialect/Transforms/MergeAdjacentParallels.cpp`).
///
/// Returns the number of adjacent parallel loop pairs merged.
pub fn merge_adjacent_parallel_loops(loop_module: &mut crate::ir::loops::LoopModule) -> usize {
    use crate::ir::loops::LoopStmt;
    let mut merged_count = 0usize;

    for kernel in &mut loop_module.kernels {
        loop {
            let mut merged_pair: Option<(usize, LoopStmt, std::collections::HashSet<String>)> =
                None;

            for idx in 0..kernel.body.len().saturating_sub(1) {
                let (
                    LoopStmt::For {
                        var: p_var,
                        lower: p_lo,
                        upper: p_hi,
                        step: p_step,
                        parallel: true,
                        vectorized: p_vec,
                        unroll: p_unroll,
                        body: p_body,
                    },
                    LoopStmt::For {
                        var: c_var,
                        lower: c_lo,
                        upper: c_hi,
                        step: c_step,
                        parallel: true,
                        vectorized: c_vec,
                        unroll: c_unroll,
                        body: c_body,
                    },
                ) = (&kernel.body[idx], &kernel.body[idx + 1])
                else {
                    continue;
                };

                if p_lo != c_lo || p_hi != c_hi || p_step != c_step || p_var != c_var {
                    continue;
                }

                // Collect buffers written pointwisely (`dst[var] = expr`) by the producer loop.
                let mut prod_writes: IndexMap<String, String> = IndexMap::new();
                let mut valid_prod = true;
                for stmt in p_body {
                    match stmt {
                        LoopStmt::Assign { dst, index, expr } if index == p_var => {
                            prod_writes.insert(dst.clone(), expr.clone());
                        }
                        LoopStmt::Comment(_) => {}
                        _ => {
                            valid_prod = false;
                            break;
                        }
                    }
                }
                if !valid_prod || prod_writes.is_empty() {
                    continue;
                }

                // Verify that the consumer loop has no non-pointwise (cross-iteration) access to
                // any buffer written by the producer (`nonSharedInputDependsOnProducer`).
                let has_cross_iter_hazard = prod_writes.keys().any(|buf| {
                    c_body
                        .iter()
                        .any(|s| stmt_has_non_pointwise_access(s, buf, p_var))
                });
                if has_cross_iter_hazard {
                    continue;
                }

                // Identify producer outputs used ONLY by `consumer` (not kernel output, and not
                // referenced by any statement outside `idx` and `idx + 1`).
                let mut dead_after_consumer: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for buf in prod_writes.keys() {
                    if *buf == kernel.output.0 {
                        continue;
                    }
                    let used_outside = kernel.body.iter().enumerate().any(|(k, s)| {
                        k != idx
                            && k != idx + 1
                            && !matches!(s, LoopStmt::Alloc { .. })
                            && stmt_references_buffer(s, buf)
                    });
                    if !used_outside {
                        dead_after_consumer.insert(buf.clone());
                    }
                }

                let mut combined_body = Vec::with_capacity(p_body.len() + c_body.len());
                for stmt in p_body {
                    if let LoopStmt::Assign { dst, .. } = stmt
                        && dead_after_consumer.contains(dst)
                    {
                        continue;
                    }
                    combined_body.push(stmt.clone());
                }

                for stmt in c_body {
                    let mut cloned = stmt.clone();
                    if let LoopStmt::Assign { ref mut expr, .. } = cloned {
                        for buf in &dead_after_consumer {
                            if let Some(prod_expr) = prod_writes.get(buf) {
                                let target = format!("{buf}[{p_var}]");
                                let replacement = format!("({prod_expr})");
                                *expr = expr.replace(&target, &replacement);
                            }
                        }
                    }
                    combined_body.push(cloned);
                }

                let merged_for = LoopStmt::For {
                    var: p_var.clone(),
                    lower: *p_lo,
                    upper: *p_hi,
                    step: *p_step,
                    parallel: true,
                    vectorized: *p_vec && *c_vec,
                    unroll: (*p_unroll).max(*c_unroll),
                    body: combined_body,
                };
                merged_pair = Some((idx, merged_for, dead_after_consumer));
                break;
            }

            if let Some((idx, merged_for, dead_bufs)) = merged_pair {
                kernel.body[idx] = merged_for;
                kernel.body.remove(idx + 1);
                if !dead_bufs.is_empty() {
                    kernel.body.retain(
                        |s| !matches!(s, LoopStmt::Alloc { name, .. } if dead_bufs.contains(name)),
                    );
                }
                merged_count += 1;
            } else {
                break;
            }
        }
    }

    merged_count
}

fn is_effectful_chain_producer(op: &Operation) -> bool {
    if matches!(
        op.attributes.get("castforge.async_chain"),
        Some(Attribute::Bool(true))
    ) {
        return true;
    }
    matches!(
        op.name.to_string().as_str(),
        "castforge.kv_offload"
            | "castforge.kv_store"
            | "castforge.async_copy"
            | "castforge.device_transfer"
            | "stablehlo.send"
            | "stablehlo.outfeed"
    )
}

/// Awaits unused ("hanging") asynchronous/effectful operation results before each block's
/// terminator (`func.return` / `return`) by inserting a `castforge.chain_await` barrier operation
/// and recording `castforge.awaited_hanging_chains` on the terminator
/// (inspired by Modular `GraphCompiler/lib/MGPDialect/Transforms/FixupOverhangsPass.cpp`).
///
/// Matching `FixupOverhangsPass`: at most one unused result per hanging op is collected ("A
/// failing op sets all its results to error, so one per op suffices"). Returns the total number of
/// hanging chains awaited across the module.
pub fn await_hanging_effect_chains(module: &mut Module) -> usize {
    let mut awaited_total = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let Some(&term_id) = op_ids.last() else {
            continue;
        };
        let Some(term_op) = module.ops.get(term_id).cloned() else {
            continue;
        };
        if !matches!(
            term_op.name.to_string().as_str(),
            "func.return" | "return" | "stablehlo.return"
        ) {
            continue;
        }

        let mut hanging_chains: Vec<ValueId> = Vec::new();
        let mut hanging_locs: Vec<Location> = Vec::new();

        for &op_id in &op_ids[..op_ids.len() - 1] {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !is_effectful_chain_producer(op) {
                continue;
            }
            for &res_val in &op.results {
                if module.use_count(res_val) == 0 {
                    hanging_chains.push(res_val);
                    hanging_locs.push(op.location.clone());
                    break;
                }
            }
        }

        if hanging_chains.is_empty() {
            continue;
        }

        let count = hanging_chains.len();
        awaited_total += count;
        let fused_loc = fuse_locations(&hanging_locs);

        let await_op_id = module.ops.insert_with_key(|id| Operation {
            id,
            name: crate::ir::OpName::new("castforge", "chain_await"),
            operands: hanging_chains.into(),
            results: Default::default(),
            attributes: IndexMap::new(),
            regions: Default::default(),
            custom_syntax: false,
            location: fused_loc,
        });

        if let Some(term_mut) = module.ops.get_mut(term_id) {
            term_mut.attributes.insert(
                "castforge.awaited_hanging_chains".to_string(),
                Attribute::Int(count as i64),
            );
        }

        let block_ops = &mut module.blocks[block_id].ops;
        let insert_pos = block_ops.len().saturating_sub(1);
        block_ops.insert(insert_pos, await_op_id);
    }

    awaited_total
}

/// Eliminates direct dependency edges of each node in `dag_edges` (`dag_edges[u]` = list of
/// prerequisite nodes that `u` depends on) that are transitively implied by `u`'s other
/// prerequisites (`u -> w ->+ v`), using a cycle-safe breadth-first generation walk
/// (inspired by Modular
/// `GraphCompiler/lib/MGPDialect/Transforms/SimplifyDeviceGraphDependenciesPass.cpp`
/// `eliminateImpliedDependencies`).
///
/// Returns `(reduced_dag_edges, num_eliminated_edges)`.
pub fn eliminate_implied_dependencies(dag_edges: &[Vec<usize>]) -> (Vec<Vec<usize>>, usize) {
    let n = dag_edges.len();
    let mut reduced = Vec::with_capacity(n);
    let mut eliminated_count = 0usize;

    for direct in dag_edges {
        // Deduplicate while preserving original order (`SmallSetVector<Value, 16> needles`).
        let mut needles: Vec<usize> = Vec::with_capacity(direct.len());
        for &dep in direct {
            if !needles.contains(&dep) {
                needles.push(dep);
            }
        }
        let orig_len = direct.len();

        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut current_gen: Vec<usize> = Vec::new();
        for &needle in &needles {
            if let Some(nexts) = dag_edges.get(needle) {
                for &nx in nexts {
                    if !current_gen.contains(&nx) {
                        current_gen.push(nx);
                    }
                }
            }
        }

        // Track `visited` so even a malformed cycle cannot hang the pass.
        while !current_gen.is_empty() {
            for &v in &current_gen {
                visited.insert(v);
            }
            needles.retain(|v| !current_gen.contains(v));

            let mut next_gen: Vec<usize> = Vec::new();
            for &val in &current_gen {
                if let Some(nexts) = dag_edges.get(val) {
                    for &nx in nexts {
                        if !visited.contains(&nx) && !next_gen.contains(&nx) {
                            next_gen.push(nx);
                        }
                    }
                }
            }
            current_gen = next_gen;
        }

        eliminated_count += orig_len.saturating_sub(needles.len());
        reduced.push(needles);
    }

    (reduced, eliminated_count)
}

pub(super) fn extract_int_list_attr(op: &Operation, key: &str) -> Option<Vec<i64>> {
    match op.attributes.get(key)? {
        Attribute::DenseIntArray(v) => Some(v.clone()),
        Attribute::Array(items) => items
            .iter()
            .map(|a| match a {
                Attribute::Int(i) => Some(*i),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Coalesces groups of `stablehlo.slice` operations that partition a single static axis of a
/// common input tensor with unit stride (`step == 1`) into a unified split specification
/// (inspired by Modular
/// `GraphCompiler/include/GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h`
/// `SliceToSplitPattern`).
///
/// Verifies that:
/// 1. At least 2 `stablehlo.slice` ops consume the same input tensor `X`,
/// 2. All slices slice along the same single static dimension `slice_dim` with unit step,
/// 3. Sorted by `start_indices[slice_dim]`, the slices are non-overlapping and contiguous from `0`
///    to `X.shape[slice_dim]` ("Union of slices must cover all of the input"),
///
/// Stamps `castforge.coalesced_split_axis`, `castforge.coalesced_split_sizes`, and
/// `castforge.coalesced_split_index` onto each slice and fuses their provenance onto the primary
/// slice via [`stamp_fused_provenance`]. Returns the number of partitioning slice groups coalesced.
pub fn coalesce_partitioning_slices_to_split(module: &mut Module) -> usize {
    let mut coalesced_groups = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let mut by_input: IndexMap<ValueId, Vec<OpId>> = IndexMap::new();

        for &op_id in &op_ids {
            if let Some(op) = module.ops.get(op_id)
                && op.is_op("stablehlo", "slice")
                && op.operands.len() == 1
                && op.results.len() == 1
            {
                by_input.entry(op.operands[0]).or_default().push(op_id);
            }
        }

        for (in_val, slice_ops) in by_input {
            if slice_ops.len() < 2 {
                continue;
            }
            let Some(in_info) = module.values.get(in_val) else {
                continue;
            };
            let Type::Tensor(ref in_tt) = in_info.ty else {
                continue;
            };
            let rank = in_tt.shape.len();
            if rank == 0 {
                continue;
            }

            let mut slice_dim_opt: Option<usize> = None;
            // (op_id, start, size, loc)
            let mut intervals: Vec<(OpId, i64, i64, Location)> =
                Vec::with_capacity(slice_ops.len());
            let mut valid = true;

            for &s_id in &slice_ops {
                let Some(s_op) = module.ops.get(s_id) else {
                    valid = false;
                    break;
                };
                if let Some(strides) = extract_int_list_attr(s_op, "strides")
                    && strides.iter().any(|&st| st != 1)
                {
                    valid = false;
                    break;
                }
                let Some(starts) = extract_int_list_attr(s_op, "start_indices") else {
                    valid = false;
                    break;
                };
                if starts.len() != rank {
                    valid = false;
                    break;
                }
                let res_val = s_op.results[0];
                let Some(res_info) = module.values.get(res_val) else {
                    valid = false;
                    break;
                };
                let Type::Tensor(ref res_tt) = res_info.ty else {
                    valid = false;
                    break;
                };
                if res_tt.shape.len() != rank {
                    valid = false;
                    break;
                }

                let mut diff_dims = Vec::new();
                for d in 0..rank {
                    if in_tt.shape[d] != res_tt.shape[d] || starts[d] != 0 {
                        diff_dims.push(d);
                    }
                }
                if diff_dims.len() != 1 {
                    valid = false;
                    break;
                }
                let d = diff_dims[0];
                if let Some(prev_d) = slice_dim_opt
                    && prev_d != d
                {
                    valid = false;
                    break;
                }
                slice_dim_opt = Some(d);

                let Dim::Static(slice_len) = res_tt.shape[d] else {
                    valid = false;
                    break;
                };
                if starts[d] < 0 || slice_len <= 0 {
                    valid = false;
                    break;
                }
                intervals.push((s_id, starts[d], slice_len, s_op.location.clone()));
            }

            let Some(slice_dim) = slice_dim_opt else {
                continue;
            };
            let Dim::Static(input_dim_len) = in_tt.shape[slice_dim] else {
                continue;
            };
            if !valid || input_dim_len <= 0 {
                continue;
            }

            intervals.sort_by_key(|entry| entry.1);

            // Verify contiguous non-overlapping coverage from `0` to `input_dim_len`.
            let mut cursor = 0i64;
            for &(_, start, len, _) in &intervals {
                if start != cursor {
                    valid = false;
                    break;
                }
                cursor += len;
            }
            if !valid || cursor != input_dim_len {
                continue;
            }

            let split_sizes: Vec<i64> = intervals.iter().map(|e| e.2).collect();
            let all_locs: Vec<Location> = intervals.iter().map(|e| e.3.clone()).collect();

            for (idx, &(s_id, _, _, _)) in intervals.iter().enumerate() {
                if let Some(op_mut) = module.ops.get_mut(s_id) {
                    op_mut.attributes.insert(
                        "castforge.coalesced_split_axis".to_string(),
                        Attribute::Int(slice_dim as i64),
                    );
                    op_mut.attributes.insert(
                        "castforge.coalesced_split_sizes".to_string(),
                        Attribute::DenseIntArray(split_sizes.clone()),
                    );
                    op_mut.attributes.insert(
                        "castforge.coalesced_split_index".to_string(),
                        Attribute::Int(idx as i64),
                    );
                    if idx == 0 {
                        stamp_fused_provenance(op_mut, &all_locs);
                    }
                }
            }
            coalesced_groups += 1;
        }
    }

    coalesced_groups
}

/// Returns `Some(axis)` if reshaping from `input_dims` to `output_dims` drops exactly one size-1
/// dimension at `axis` while leaving all other (static, symbolic, or dynamic) dimensions
/// unchanged (inspired by Modular `GraphCompiler/lib/MODialect/Patterns/ShapeOpSinking.cpp`
/// `findSqueezeAxis`).
pub fn find_squeeze_axis(input_dims: &[Dim], output_dims: &[Dim]) -> Option<usize> {
    if input_dims.len() != output_dims.len() + 1 {
        return None;
    }
    let offset = (0..input_dims.len())
        .find(|&i| output_dims.get(i) != Some(&input_dims[i]))
        .unwrap_or(output_dims.len());
    if input_dims[offset] != Dim::Static(1) {
        return None;
    }
    if input_dims[..offset] != output_dims[..offset] {
        return None;
    }
    if input_dims[offset + 1..] != output_dims[offset..] {
        return None;
    }
    Some(offset)
}

/// Returns `Some(axis)` if reshaping from `input_dims` to `output_dims` inserts exactly one size-1
/// dimension at `axis` while leaving all other dimensions unchanged.
pub fn find_unsqueeze_axis(input_dims: &[Dim], output_dims: &[Dim]) -> Option<usize> {
    find_squeeze_axis(output_dims, input_dims)
}

/// Annotates `stablehlo.reshape` operations that drop or insert a single size-1 dimension with
/// `castforge.squeeze_axis` or `castforge.unsqueeze_axis`
/// (inspired by Modular `GraphCompiler/lib/MODialect/Patterns/ShapeOpSinking.cpp`).
///
/// Returns the number of squeeze/unsqueeze reshapes annotated.
pub fn annotate_squeeze_unsqueeze_reshapes(module: &mut Module) -> usize {
    let mut annotated = 0usize;
    let op_ids: Vec<_> = module.ops.keys().collect();

    for op_id in op_ids {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if !op.is_op("stablehlo", "reshape") || op.operands.len() != 1 || op.results.len() != 1 {
            continue;
        }
        let in_val = op.operands[0];
        let out_val = op.results[0];
        let (Some(in_info), Some(out_info)) =
            (module.values.get(in_val), module.values.get(out_val))
        else {
            continue;
        };
        let (Type::Tensor(in_tt), Type::Tensor(out_tt)) = (&in_info.ty, &out_info.ty) else {
            continue;
        };

        let squeeze = find_squeeze_axis(&in_tt.shape, &out_tt.shape);
        let unsqueeze = find_unsqueeze_axis(&in_tt.shape, &out_tt.shape);

        if let Some(axis) = squeeze {
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.squeeze_axis".to_string(),
                    Attribute::Int(axis as i64),
                );
                annotated += 1;
            }
        } else if let Some(axis) = unsqueeze
            && let Some(op_mut) = module.ops.get_mut(op_id)
        {
            op_mut.attributes.insert(
                "castforge.unsqueeze_axis".to_string(),
                Attribute::Int(axis as i64),
            );
            annotated += 1;
        }
    }

    annotated
}
