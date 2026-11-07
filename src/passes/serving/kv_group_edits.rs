//! Hybrid Mamba / sub-paged attention KV cache group view edits and in-place
//! token-drop KV sequence compaction.
//!
//! Covers:
//! 1. **KV Cache Group Spec Validation (`validate_kv_cache_group_specs`)**:
//!    Rejects `CrossAttention` groups and `Mamba` groups whose `mamba_cache_mode`
//!    is not in `{"align", "all"}` with a single aggregated error, while
//!    exempting slot-compressed groups (`tokens_per_state > 1`,
//!    `compress_ratio > 1`, or `tq_slot_size > 0`).
//! 2. **Zero-Copy Hybrid KV Group View Edits**:
//!    - [`apply_mamba_page_view_edit`]: reinterprets `[conv_state, ssm_state]`
//!      sharing a padded page (`stride[0] * elem_bytes == page_size_bytes`) as a
//!      synthetic 5D page view `(num_blocks, 2, block_size, 1, head_size)`.
//!    - [`apply_mamba_slice_plane_edit`]: re-views contiguous slice-strided
//!      Mamba states into `uint8` planes `(num_blocks, block_size, 1, width)`
//!      with `width = nbytes_per_block / block_size`.
//!    - [`apply_subpaged_attention_view_edit`]: re-views kernel-paged rank-4 or
//!      rank-5 attention KV tensors (`kernel_block_size < logical_block_size`,
//!      e.g. `17 * 32 = 544`) into `(num_logical_blocks, 2, logical_block_size,
//!      1, head_size)` after verifying exact page-byte tiling and memory/logical
//!      contiguity.
//!    - [`apply_mamba_unified_view_edit`] & [`compute_compressed_mamba_slots`]:
//!      computes vector-aligned `(num_slots, head_size)` (trying `16, 8, 4, 2`
//!      byte alignments within `page_size_bytes`, falling back to compressed
//!      slot tiling when unpadded) for `NHD`/`BLNHC` and `HND`/`BLHNC` layouts.
//! 3. **In-Place Paged KV Token-Drop Compaction (`PagedKvSequenceView`)**:
//!    Reads `[tokens, kv_heads, head_dim]` K/V slices in request token order and
//!    compacts retained per-head `[kept_tokens, kv_heads]` positions in place
//!    into the leading slots `0..kept_len`.

/// Supported attention / state-space KV cache spec kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheSpecKind {
    /// Standard full causal attention.
    FullAttention,
    /// Sliding-window local attention.
    SlidingWindow,
    /// Chunked local attention.
    ChunkedLocalAttention,
    /// Attention with sink tokens.
    SinkFullAttention,
    /// Multi-head latent attention (MLA).
    Mla,
    /// Mamba / linear state-space layer (`[conv_state, ssm_state]` or unified row).
    Mamba,
    /// Encoder-decoder cross attention (rejected for KV transfer).
    CrossAttention,
}

impl KvCacheSpecKind {
    /// Returns `true` if this spec kind is a standard-paged (non-MLA) attention
    /// eligible for the sub-paged attention view edit.
    #[must_use]
    pub const fn is_subpageable_attention(self) -> bool {
        matches!(
            self,
            Self::FullAttention
                | Self::SlidingWindow
                | Self::ChunkedLocalAttention
                | Self::SinkFullAttention
        )
    }
}

/// Descriptor for a single KV cache group specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheGroupSpec {
    /// Spec kind (`FullAttention`, `Mamba`, `CrossAttention`, etc.).
    pub kind: KvCacheSpecKind,
    /// Logical scheduler block size in tokens.
    pub block_size: usize,
    /// Physical page size in bytes for one block.
    pub page_size_bytes: usize,
    /// Mamba state caching mode (`"align"`, `"all"`, or `"none"`).
    pub mamba_cache_mode: String,
    /// Tokens packed per state slot (`> 1` indicates slot compression).
    pub tokens_per_state: usize,
    /// Legacy compression ratio (`> 1` indicates slot compression).
    pub compress_ratio: usize,
    /// TurboQuant slot byte size (`> 0` indicates TurboQuant slot compression).
    pub tq_slot_size: usize,
}

/// Returns `true` if `spec` declares slot compression (`tokens_per_state > 1`,
/// `compress_ratio > 1`, or `tq_slot_size > 0`) and must not be edited by
/// sub-paged attention rules.
#[must_use]
pub const fn declares_slot_compression(spec: &KvCacheGroupSpec) -> bool {
    spec.tokens_per_state > 1 || spec.compress_ratio > 1 || spec.tq_slot_size > 0
}

/// Validates a list of KV cache group specs at startup, rejecting all
/// unsupported groups (`CrossAttention` or `Mamba` with `mamba_cache_mode`
/// outside `{"align", "all"}`) in a single aggregated error message.
///
/// # Errors
///
/// Returns `Err` listing every unsupported group index and reason.
pub fn validate_kv_cache_group_specs(groups: &[KvCacheGroupSpec]) -> Result<(), String> {
    let mut unsupported: Vec<String> = Vec::new();
    for (group_idx, spec) in groups.iter().enumerate() {
        match spec.kind {
            KvCacheSpecKind::CrossAttention => {
                unsupported.push(format!("group {group_idx}: CrossAttentionSpec"));
            }
            KvCacheSpecKind::Mamba => {
                let mode = if spec.mamba_cache_mode.is_empty() {
                    "none"
                } else {
                    spec.mamba_cache_mode.as_str()
                };
                if mode != "align" && mode != "all" {
                    unsupported.push(format!(
                        "group {group_idx}: MambaSpec with mamba_cache_mode='{mode}' \
                         (only 'align' and 'all' keep reusable state snapshots)"
                    ));
                }
            }
            _ => {}
        }
    }
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "cannot serve this model's KV cache groups: {}",
            unsupported.join("; ")
        ))
    }
}

/// Factors `elems_per_page` into `(num_heads = 1, head_size)` such that
/// `2 * block_size * 1 * head_size == elems_per_page`.
///
/// # Errors
///
/// Returns `Err` if `block_size == 0` or `elems_per_page` is not divisible by
/// `2 * block_size`.
pub fn synthetic_attention_shape(
    elems_per_page: usize,
    block_size: usize,
) -> Result<(usize, usize), String> {
    let denom = 2usize.saturating_mul(block_size);
    if denom == 0 || !elems_per_page.is_multiple_of(denom) {
        return Err(format!(
            "page ({elems_per_page} elems) does not factor into \
             (2, block_size={block_size}, num_heads=1, head_size)"
        ));
    }
    Ok((1, elems_per_page / denom))
}

/// Strided tensor view metadata (shape, element strides, element byte size, and
/// storage offset) used by zero-copy KV cache group view edits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedTensorMeta {
    /// Logical dimensions.
    pub shape: Vec<usize>,
    /// Per-dimension strides in elements.
    pub strides: Vec<usize>,
    /// Element size in bytes (`1`, `2`, `4`, etc.).
    pub element_size: usize,
    /// Storage offset in elements.
    pub storage_offset: usize,
}

impl StridedTensorMeta {
    /// Total number of elements in `shape`.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Returns `true` if `strides` match standard row-major C-contiguous layout
    /// for `shape`.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.strides.len() {
            return false;
        }
        let mut expected = 1usize;
        for (&dim, &st) in self.shape.iter().zip(self.strides.iter()).rev() {
            if dim > 1 && st != expected {
                return false;
            }
            expected = expected.saturating_mul(dim);
        }
        true
    }
}

/// Re-views a Mamba `[conv_state, ssm_state]` list sharing one padded page per
/// block (`conv.strides[0] * conv.element_size == spec.page_size_bytes`) as a
/// synthetic 5D attention page view `(num_blocks, 2, block_size, 1, head_size)`.
///
/// # Errors
///
/// Returns `Err` if `states` is empty, `conv_state` per-block stride does not
/// equal `spec.page_size_bytes`, or the page does not factor into the synthetic
/// 5D shape.
pub fn apply_mamba_page_view_edit(
    spec: &KvCacheGroupSpec,
    states: &[StridedTensorMeta],
) -> Result<StridedTensorMeta, String> {
    let Some(conv) = states.first() else {
        return Err("expected a non-empty Mamba [conv_state, ssm_state] tensor list".to_string());
    };
    if conv.shape.is_empty() || conv.strides.is_empty() || conv.element_size == 0 {
        return Err("invalid Mamba conv_state shape or element_size".to_string());
    }
    let block_stride_bytes = conv.strides[0].saturating_mul(conv.element_size);
    if block_stride_bytes != spec.page_size_bytes {
        return Err(format!(
            "Mamba conv state per-block stride ({block_stride_bytes} bytes) does not equal \
             the page size ({} bytes)",
            spec.page_size_bytes
        ));
    }
    let num_blocks = conv.shape[0];
    let elems_per_page = spec.page_size_bytes / conv.element_size;
    let (num_heads, head_size) = synthetic_attention_shape(elems_per_page, spec.block_size)?;
    let s3 = head_size;
    let s2 = num_heads * s3;
    let s1 = spec.block_size * s2;
    let s0 = 2 * s1;
    Ok(StridedTensorMeta {
        shape: vec![num_blocks, 2, spec.block_size, num_heads, head_size],
        strides: vec![s0, s1, s2, s3, 1],
        element_size: conv.element_size,
        storage_offset: conv.storage_offset,
    })
}

/// Re-views slice-strided contiguous Mamba `[conv_state, ssm_state]` tensors as
/// `uint8` planes `(num_blocks, block_size, 1, width)` where
/// `width = nbytes_per_block / block_size`.
///
/// # Errors
///
/// Returns `Err` if `states` is empty, states have differing `num_blocks`, any
/// state is non-contiguous, or `spec.block_size` does not divide `nbytes_per_block`.
pub fn apply_mamba_slice_plane_edit(
    spec: &KvCacheGroupSpec,
    states: &[StridedTensorMeta],
) -> Result<Vec<StridedTensorMeta>, String> {
    if states.is_empty() {
        return Err("expected a non-empty Mamba [conv_state, ssm_state] tensor list".to_string());
    }
    if spec.block_size == 0 {
        return Err("block_size must be greater than zero".to_string());
    }
    let expected_blocks = states[0].shape.first().copied().unwrap_or(0);
    if expected_blocks == 0 {
        return Err("Mamba state num_blocks must be > 0".to_string());
    }
    let mut planes = Vec::with_capacity(states.len());
    for state in states {
        let nb = state.shape.first().copied().unwrap_or(0);
        if nb != expected_blocks {
            return Err(format!(
                "Mamba states cover different block counts ({expected_blocks} vs {nb})"
            ));
        }
        if !state.is_contiguous() {
            return Err("Mamba state must be a non-empty contiguous slice".to_string());
        }
        let nbytes = (state.numel() * state.element_size) / nb;
        if !nbytes.is_multiple_of(spec.block_size) {
            return Err(format!(
                "scheduler block size {} does not divide the {nbytes}-byte state block",
                spec.block_size
            ));
        }
        let width = nbytes / spec.block_size;
        planes.push(StridedTensorMeta {
            shape: vec![nb, spec.block_size, 1, width],
            strides: vec![nbytes, width, width, 1],
            element_size: 1,
            storage_offset: state.storage_offset * state.element_size,
        });
    }
    Ok(planes)
}

/// Re-views a kernel-paged rank-4 or rank-5 attention KV tensor
/// (`kv_cache.shape[2] == kernel_block_size < spec.block_size`) into logical
/// pages `(num_logical_blocks, 2, spec.block_size, 1, head_size)`.
///
/// # Errors
///
/// Returns `Err` if `kv_cache` is not rank 4/5, `spec.block_size` is not a
/// multiple of `kernel_block_size`, `num_kernel_pages` is not a multiple of the
/// ratio, or `ratio * kernel_page_bytes != spec.page_size_bytes`.
pub fn apply_subpaged_attention_view_edit(
    spec: &KvCacheGroupSpec,
    kv_cache: &StridedTensorMeta,
) -> Result<StridedTensorMeta, String> {
    let ndim = kv_cache.shape.len();
    if (ndim != 4 && ndim != 5) || (ndim == 5 && kv_cache.shape[1] != 2) {
        return Err(format!(
            "expected a rank-5 (num_blocks, 2, block_size, num_heads, head_size) or \
             rank-4 (num_blocks, num_heads, block_size, 2 * head_size) attention KV \
             tensor, got {:?}",
            kv_cache.shape
        ));
    }
    let logical_block_size = spec.block_size;
    let kernel_block_size = kv_cache.shape[2];
    if kernel_block_size == 0 || !logical_block_size.is_multiple_of(kernel_block_size) {
        return Err(format!(
            "logical block size {logical_block_size} is not a multiple of kernel block \
             size {kernel_block_size}"
        ));
    }
    let ratio = logical_block_size / kernel_block_size;
    let num_kernel_pages = kv_cache.shape[0];
    if !num_kernel_pages.is_multiple_of(ratio) {
        return Err(format!(
            "kernel page count {num_kernel_pages} is not a multiple of the \
             logical/kernel block ratio {ratio}"
        ));
    }
    let inner_elems: usize = kv_cache.shape[1..].iter().copied().product();
    let kernel_page_bytes = inner_elems * kv_cache.element_size;
    if kernel_page_bytes * ratio != spec.page_size_bytes {
        return Err(format!(
            "{ratio} kernel pages ({} bytes) do not tile the logical page ({} bytes)",
            kernel_page_bytes * ratio,
            spec.page_size_bytes
        ));
    }
    if ndim == 5 && !kv_cache.is_contiguous() {
        return Err(
            "kernel-paged rank-5 attention KV tensor must be contiguous to re-view as \
             logical pages"
                .to_string(),
        );
    }

    let elems_per_page = spec.page_size_bytes / kv_cache.element_size;
    let (num_heads, head_size) = synthetic_attention_shape(elems_per_page, logical_block_size)?;
    let s3 = head_size;
    let s2 = num_heads * s3;
    let s1 = logical_block_size * s2;
    let s0 = 2 * s1;
    Ok(StridedTensorMeta {
        shape: vec![
            num_kernel_pages / ratio,
            2,
            logical_block_size,
            num_heads,
            head_size,
        ],
        strides: vec![s0, s1, s2, s3, 1],
        element_size: kv_cache.element_size,
        storage_offset: kv_cache.storage_offset,
    })
}

/// Picks `(num_slots, head_size)` with `num_slots * head_size == row`,
/// `block_size % num_slots == 0`, and the widest vector alignment in
/// `(16, 8, 4, 2)` bytes (falling back to `(1, row)` when `elem >= 2`),
/// preferring more slots within one alignment tier.
///
/// # Errors
///
/// Returns `Err` if no aligned slot count tiles `row`.
pub fn compute_compressed_mamba_slots(
    row: usize,
    block_size: usize,
    elem_bytes: usize,
    block_step_elems: usize,
) -> Result<(usize, usize), String> {
    if row == 0 || block_size == 0 || elem_bytes == 0 {
        return Err("row, block_size, and elem_bytes must be > 0".to_string());
    }
    for align_bytes in [16usize, 8, 4, 2] {
        if !align_bytes.is_multiple_of(elem_bytes)
            || !(block_step_elems * elem_bytes).is_multiple_of(align_bytes)
        {
            continue;
        }
        let step = align_bytes / elem_bytes;
        for slots in (1..=block_size).rev() {
            if block_size.is_multiple_of(slots) && row.is_multiple_of(slots * step) {
                return Ok((slots, row / slots));
            }
        }
    }
    if elem_bytes >= 2 {
        return Ok((1, row));
    }
    Err(format!(
        "Mamba state row ({row} x {elem_bytes}B, block_size={block_size}, \
         stride(0)={block_step_elems}) has no 2-byte-aligned slot tiling"
    ))
}

/// Re-views a unified Mamba rank-4 state `[num_blocks, 1, 1, row]` as a
/// per-token or slot-compressed 4D view according to `kv_layout`
/// (`"NHD"`, `"BLNHC"`, `"HND"`, or `"BLHNC"`).
///
/// # Errors
///
/// Returns `Err` if `kv_layout` is unsupported or no aligned slot tiling exists.
pub fn apply_mamba_unified_view_edit(
    spec: &KvCacheGroupSpec,
    kv_cache: &StridedTensorMeta,
    kv_layout: &str,
) -> Result<StridedTensorMeta, String> {
    if !matches!(kv_layout, "NHD" | "BLNHC" | "HND" | "BLHNC") {
        return Err(format!("Unsupported kv_layout: {kv_layout}"));
    }
    if kv_cache.shape.len() != 4 || kv_cache.strides.is_empty() || kv_cache.element_size == 0 {
        return Err("expected rank-4 unified Mamba tensor [num_blocks, 1, 1, row]".to_string());
    }
    let num_blocks = kv_cache.shape[0];
    let row: usize = kv_cache.shape[1..].iter().copied().product();
    let block_size = spec.block_size;
    let elem = kv_cache.element_size;
    let block_step = kv_cache.strides[0];
    let base = row.div_ceil(block_size);
    let page_bytes = spec.page_size_bytes;

    let mut head_size = 0usize;
    for align_bytes in [16usize, 8, 4, 2] {
        if !align_bytes.is_multiple_of(elem) || !(block_step * elem).is_multiple_of(align_bytes) {
            continue;
        }
        let step = align_bytes / elem;
        let candidate = base.div_ceil(step) * step;
        if candidate * block_size * elem <= page_bytes {
            head_size = candidate;
            break;
        }
    }

    let mut num_slots = block_size;
    if head_size == 0 {
        let (s, h) = compute_compressed_mamba_slots(row, block_size, elem, block_step)?;
        num_slots = s;
        head_size = h;
    }

    let (shape, strides) = if matches!(kv_layout, "NHD" | "BLNHC") {
        (
            vec![num_blocks, num_slots, 1, head_size],
            vec![block_step, head_size, head_size, 1],
        )
    } else {
        (
            vec![num_blocks, 1, num_slots, head_size],
            vec![block_step, num_slots * head_size, head_size, 1],
        )
    };

    Ok(StridedTensorMeta {
        shape,
        strides,
        element_size: elem,
        storage_offset: kv_cache.storage_offset,
    })
}

// ---------------------------------------------------------------------------
// 3. In-Place Paged KV Token-Drop Compaction
// ---------------------------------------------------------------------------

/// Paged K/V sequence view over a `[num_blocks, 2, block_size, kv_heads, head_dim]`
/// buffer supporting token-order reads and in-place per-head token-drop compaction.
#[derive(Debug, Clone)]
pub struct PagedKvSequenceView {
    num_blocks: usize,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    /// Flat storage of shape `[num_blocks, 2, block_size, kv_heads, head_dim]`.
    cache: Vec<f32>,
    /// Logical slot indices (`block_idx * block_size + offset`) for the request's tokens.
    slots: Vec<usize>,
}

impl PagedKvSequenceView {
    /// Creates a new paged KV sequence view.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `cache.len() != num_blocks * 2 * block_size * kv_heads * head_dim`
    /// or any slot index in `slots` is out of bounds.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        head_dim: usize,
        cache: Vec<f32>,
        slots: Vec<usize>,
    ) -> Result<Self, String> {
        let expected_len = num_blocks * 2 * block_size * kv_heads * head_dim;
        if block_size == 0 || kv_heads == 0 || head_dim == 0 || cache.len() != expected_len {
            return Err("invalid paged KV cache shape or buffer length".to_string());
        }
        let max_slot = num_blocks * block_size;
        if slots.iter().any(|&s| s >= max_slot) {
            return Err("slot index out of bounds for paged KV cache".to_string());
        }
        Ok(Self {
            num_blocks,
            block_size,
            kv_heads,
            head_dim,
            cache,
            slots,
        })
    }

    fn flat_index(
        &self,
        block: usize,
        kv_idx: usize,
        offset: usize,
        head: usize,
        dim: usize,
    ) -> usize {
        ((((block * 2 + kv_idx) * self.block_size + offset) * self.kv_heads + head) * self.head_dim)
            + dim
    }

    fn read_plane(&self, kv_idx: usize, token_limit: usize) -> Vec<f32> {
        let n_tok = token_limit.min(self.slots.len());
        let mut out = Vec::with_capacity(n_tok * self.kv_heads * self.head_dim);
        for &slot in &self.slots[..n_tok] {
            let blk = slot / self.block_size;
            let off = slot % self.block_size;
            for h in 0..self.kv_heads {
                for d in 0..self.head_dim {
                    out.push(self.cache[self.flat_index(blk, kv_idx, off, h, d)]);
                }
            }
        }
        out
    }

    /// Reads the request's keys `[tokens, kv_heads, head_dim]` up to `token_limit`.
    #[must_use]
    pub fn get_keys(&self, token_limit: usize) -> Vec<f32> {
        self.read_plane(0, token_limit)
    }

    /// Reads the request's values `[tokens, kv_heads, head_dim]` up to `token_limit`.
    #[must_use]
    pub fn get_values(&self, token_limit: usize) -> Vec<f32> {
        self.read_plane(1, token_limit)
    }

    /// Compacts per-head K/V in place into this request's leading `kept_positions.len()`
    /// slots (`kept_positions` has shape `[kept_tokens][kv_heads]`).
    ///
    /// Example: `[A, B, C, D]`, keep `[1, 3]` -> `[B, D, _, _]`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `kept_positions` is empty, exceeds `slots.len()`, has a
    /// row with length `!= kv_heads`, or contains an out-of-range token position.
    pub fn compact_kv_in_place(&mut self, kept_positions: &[Vec<usize>]) -> Result<usize, String> {
        let kept_len = kept_positions.len();
        if kept_len == 0 || kept_len > self.slots.len() {
            return Err("Invalid retained token count".to_string());
        }
        for row in kept_positions {
            if row.len() != self.kv_heads {
                return Err("Expected [kept_tokens, kv_heads] positions".to_string());
            }
            if row.iter().any(|&pos| pos >= self.slots.len()) {
                return Err("Retained token position out of range".to_string());
            }
        }

        // Gather selected [kept_len, 2, kv_heads, head_dim] before writing in place.
        let mut selected = vec![0.0f32; kept_len * 2 * self.kv_heads * self.head_dim];
        for (t_out, row) in kept_positions.iter().enumerate() {
            for (h, &t_src) in row.iter().enumerate() {
                let src_slot = self.slots[t_src];
                let src_blk = src_slot / self.block_size;
                let src_off = src_slot % self.block_size;
                for kv_idx in 0..2 {
                    for d in 0..self.head_dim {
                        let dst_idx =
                            (((t_out * 2 + kv_idx) * self.kv_heads + h) * self.head_dim) + d;
                        selected[dst_idx] =
                            self.cache[self.flat_index(src_blk, kv_idx, src_off, h, d)];
                    }
                }
            }
        }

        // Write compacted values into the leading `0..kept_len` slots.
        for t_out in 0..kept_len {
            let dst_slot = self.slots[t_out];
            let dst_blk = dst_slot / self.block_size;
            let dst_off = dst_slot % self.block_size;
            for kv_idx in 0..2 {
                for h in 0..self.kv_heads {
                    for d in 0..self.head_dim {
                        let src_idx =
                            (((t_out * 2 + kv_idx) * self.kv_heads + h) * self.head_dim) + d;
                        let flat_dst = self.flat_index(dst_blk, kv_idx, dst_off, h, d);
                        self.cache[flat_dst] = selected[src_idx];
                    }
                }
            }
        }

        let _ = self.num_blocks;
        Ok(kept_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_kv_sequence_view_compact_in_place() {
        // 1 block, block_size = 4, kv_heads = 2, head_dim = 1
        // Tokens 0..4 have keys [10, 11], [20, 21], [30, 31], [40, 41]
        let mut cache = vec![0.0f32; 2 * 4 * 2];
        let mut view = PagedKvSequenceView::new(1, 4, 2, 1, cache.clone(), vec![0, 1, 2, 3])
            .expect("valid view");
        for t in 0..4 {
            for h in 0..2 {
                let k_idx = view.flat_index(0, 0, t, h, 0);
                let v_idx = view.flat_index(0, 1, t, h, 0);
                cache[k_idx] = ((t + 1) * 10 + h) as f32;
                cache[v_idx] = ((t + 1) * 100 + h) as f32;
            }
        }
        view.cache = cache;

        // Head 0 keeps tokens [1, 3]; Head 1 keeps tokens [0, 2].
        let kept = view
            .compact_kv_in_place(&[vec![1, 0], vec![3, 2]])
            .expect("compact");
        assert_eq!(kept, 2);
        assert_eq!(view.get_keys(2), vec![20.0, 11.0, 40.0, 31.0]);
        assert_eq!(view.get_values(2), vec![200.0, 101.0, 400.0, 301.0]);
    }
}
