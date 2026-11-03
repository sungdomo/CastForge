//! Autoregressive KV-cache, PagedAttention, Flash-Decoding, and Tree-Attention passes.

use super::attention::*;
use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

/// Autoregressive KV-Cache Injection Pass (`inject-kv-cache`).
pub struct InjectKvCachePass {
    pub max_seq_len: usize,
}

impl Default for InjectKvCachePass {
    fn default() -> Self {
        Self { max_seq_len: 2048 }
    }
}

impl Pass for InjectKvCachePass {
    fn name(&self) -> &'static str {
        "inject-kv-cache"
    }

    fn description(&self) -> &'static str {
        "Injects autoregressive KV-cache state updates (castforge.kv_cache_update) before attention"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut res = FuseAttentionPass {
            kind: "kv_cache".to_string(),
            max_seq_len: self.max_seq_len,
            ..Default::default()
        }
        .run(module)?;
        res.pass_name = self.name().to_string();
        Ok(res)
    }
}

pub(super) fn insert_kv_cache_op(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    src_val: ValueId,
    role: &str,
    max_seq: i64,
) -> ValueId {
    let out_ty = module.values[src_val].ty.clone();
    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.kv_cache_update".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("kv_cache_update_impl".to_string()),
    );
    attrs.insert(
        "cache_role".to_string(),
        Attribute::String(role.to_string()),
    );
    attrs.insert("max_seq_len".to_string(), Attribute::Int(max_seq));

    let kv_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "composite"),
        operands: SmallVec::from_slice(&[src_val]),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let kv_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: kv_op_id,
            index: 0,
        },
    });
    module.ops[kv_op_id].results.push(kv_val);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, kv_op_id);
    } else {
        block_ops.push(kv_op_id);
    }
    kv_val
}

/// Physical KV-cache memory layout handed to or emitted for a serving engine (vLLM, SGLang,
/// TensorRT-LLM, LMCache).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineKvFormat {
    /// `[2, num_blocks, block_size, num_heads, head_size]` (vLLM FlashAttention default NHD).
    Nhd,
    /// `[2, num_blocks, num_heads, block_size, head_size]` (vLLM FlashInfer / TRT-LLM HND).
    Hnd,
    /// `[num_blocks, num_layers, 2, block_size, num_heads, head_size]` (Cross-Layer bulk DMA).
    CrossLayer,
    /// `[num_blocks, block_size, kv_lora_rank]` (DeepSeek MLA compressed single-head latent KV).
    MlaLatent,
}

impl EngineKvFormat {
    /// Parses a layout identifier (`"nhd"`, `"hnd"`, `"cross_layer"`, `"mla"`).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "hnd" | "flashinfer" => Self::Hnd,
            "cross_layer" | "cross-layer" | "cross" => Self::CrossLayer,
            "mla" | "mla_latent" | "latent" => Self::MlaLatent,
            _ => Self::Nhd,
        }
    }

    /// Canonical lowercase identifier for IR attributes and codegen.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nhd => "nhd",
            Self::Hnd => "hnd",
            Self::CrossLayer => "cross_layer",
            Self::MlaLatent => "mla_latent",
        }
    }

    /// Returns `true` if a single KV pool block spans all layers (`CrossLayer`).
    pub fn is_cross_layer(&self) -> bool {
        matches!(self, Self::CrossLayer)
    }

    /// Returns `true` if `num_heads` precedes `block_size` (`Hnd`).
    pub fn is_hnd(&self) -> bool {
        matches!(self, Self::Hnd)
    }

    /// Returns `true` if K and V are compressed into a single latent vector (`MlaLatent`).
    pub fn is_mla(&self) -> bool {
        matches!(self, Self::MlaLatent)
    }

    /// Computes the physical tensor shape for this KV-cache layout.
    pub fn physical_shape(
        &self,
        num_layers: i64,
        num_blocks: i64,
        block_size: i64,
        num_heads: i64,
        head_size: i64,
    ) -> Vec<i64> {
        let nl = num_layers.max(1);
        let nb = num_blocks.max(1);
        let bs = block_size.max(1);
        let nh = num_heads.max(1);
        let hs = head_size.max(1);
        match self {
            Self::Nhd => vec![2, nb, bs, nh, hs],
            Self::Hnd => vec![2, nb, nh, bs, hs],
            Self::CrossLayer => vec![nb, nl, 2, bs, nh, hs],
            Self::MlaLatent => vec![nb, bs, hs],
        }
    }

    /// Synthesizes a linear index expression for addressing `(blk, layer, kv, tok, head, dim)`.
    pub fn index_expr(
        &self,
        blk: &str,
        layer: &str,
        kv: &str,
        tok: &str,
        head: &str,
        dim: &str,
    ) -> String {
        match self {
            Self::Nhd => format!("((({kv} * NB + {blk}) * BS + {tok}) * NH + {head}) * HS + {dim}"),
            Self::Hnd => format!("((({kv} * NB + {blk}) * NH + {head}) * BS + {tok}) * HS + {dim}"),
            Self::CrossLayer => format!(
                "(((({blk} * NL + {layer}) * 2 + {kv}) * BS + {tok}) * NH + {head}) * HS + {dim}"
            ),
            Self::MlaLatent => format!("({blk} * BS + {tok}) * HS + {dim}"),
        }
    }
}

/// Homogeneous group of attention/SSM layers sharing identical physical KV-cache geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayerGroup {
    pub group_id: usize,
    pub format: EngineKvFormat,
    pub page_block_size: i64,
    pub head_size: i64,
    pub sliding_window: Option<i64>,
    pub layer_indices: Vec<usize>,
    pub bytes_per_page: u64,
}

/// Partitions all `castforge.fused_attention` operations in `module` into homogeneous
/// [`KvLayerGroup`]s by `(format, page_block_size, head_size, sliding_window)`.
pub fn analyze_kv_layer_groups(module: &Module) -> Vec<KvLayerGroup> {
    use crate::ir::Dim;
    let mut groups: Vec<KvLayerGroup> = Vec::new();
    let mut layer_idx = 0usize;

    for op_id in module.live_ops() {
        let op = &module.ops[op_id];
        let is_attn = op.is_op("stablehlo", "composite")
            && op
                .attributes
                .get("name")
                .and_then(|a| a.as_str())
                .is_some_and(|n| n == "castforge.fused_attention");
        if !is_attn {
            continue;
        }
        let fmt_str = op
            .attributes
            .get("castforge.kv_format")
            .and_then(|a| a.as_str())
            .unwrap_or("nhd");
        let format = EngineKvFormat::parse(fmt_str);
        let page_block_size = match op.attributes.get("castforge.page_block_size") {
            Some(Attribute::Int(v)) => (*v).max(1),
            _ => 16,
        };
        let sliding_window = match op.attributes.get("castforge.sliding_window") {
            Some(Attribute::Int(v)) => Some(*v),
            _ => None,
        };
        let head_size = op
            .operands
            .first()
            .and_then(|&v| module.values[v].ty.as_tensor())
            .and_then(|t| match t.shape.last() {
                Some(Dim::Static(d)) if *d > 0 => Some(*d),
                _ => None,
            })
            .unwrap_or(64);
        let kv_factor: u64 = if format.is_mla() { 1 } else { 2 };
        let bytes_per_page = kv_factor * (page_block_size as u64) * (head_size as u64) * 2;

        if let Some(existing) = groups.iter_mut().find(|g| {
            g.format == format
                && g.page_block_size == page_block_size
                && g.head_size == head_size
                && g.sliding_window == sliding_window
        }) {
            existing.layer_indices.push(layer_idx);
        } else {
            let group_id = groups.len();
            groups.push(KvLayerGroup {
                group_id,
                format,
                page_block_size,
                head_size,
                sliding_window,
                layer_indices: vec![layer_idx],
                bytes_per_page,
            });
        }
        layer_idx += 1;
    }
    groups
}

/// PagedAttention Block-Table Lowering Pass (`lower-paged-attention`).
pub struct LowerPagedAttentionPass {
    pub page_block_size: usize,
    pub max_blocks: usize,
    pub kv_format: String,
}

impl Default for LowerPagedAttentionPass {
    fn default() -> Self {
        Self {
            page_block_size: 16,
            max_blocks: 64,
            kv_format: "nhd".to_string(),
        }
    }
}

impl Pass for LowerPagedAttentionPass {
    fn name(&self) -> &'static str {
        "lower-paged-attention"
    }

    fn description(&self) -> &'static str {
        "Lowers fused_attention to vLLM/LMCache PagedAttention with physical KV layout descriptors"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        use crate::ir::Dim;
        let mut modified = 0usize;
        let block_size = self.page_block_size.max(4) as i64;
        let max_blocks = self.max_blocks.max(1) as i64;
        let fmt = EngineKvFormat::parse(&self.kv_format);

        let attn_ops: Vec<OpId> = module
            .live_ops()
            .into_iter()
            .filter(|&op_id| {
                let op = &module.ops[op_id];
                op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n == "castforge.fused_attention")
            })
            .collect();
        let num_layers = attn_ops.len().max(1) as i64;

        for &op_id in &attn_ops {
            let head_size = module.ops[op_id]
                .operands
                .first()
                .and_then(|&v| module.values[v].ty.as_tensor())
                .and_then(|t| match t.shape.last() {
                    Some(Dim::Static(d)) if *d > 0 => Some(*d),
                    _ => None,
                })
                .unwrap_or(64);
            let phys_shape = fmt.physical_shape(num_layers, max_blocks, block_size, 1, head_size);
            let Some(op) = module.ops.get_mut(op_id) else {
                continue;
            };
            let mut ch = false;
            ch |= op.attributes.insert(
                "castforge.paged_attention".to_string(),
                Attribute::Bool(true),
            ) != Some(Attribute::Bool(true));
            ch |= op.attributes.insert(
                "castforge.page_block_size".to_string(),
                Attribute::Int(block_size),
            ) != Some(Attribute::Int(block_size));
            ch |= op.attributes.insert(
                "castforge.max_blocks".to_string(),
                Attribute::Int(max_blocks),
            ) != Some(Attribute::Int(max_blocks));
            ch |= op.attributes.insert(
                "castforge.kv_format".to_string(),
                Attribute::String(fmt.as_str().to_string()),
            ) != Some(Attribute::String(fmt.as_str().to_string()));
            ch |= op.attributes.insert(
                "castforge.kv_physical_shape".to_string(),
                Attribute::DenseIntArray(phys_shape.clone()),
            ) != Some(Attribute::DenseIntArray(phys_shape));
            if ch {
                modified += 1;
            }
        }

        // Stamp homogeneous KV layer group IDs and unified Jenga huge-block pool geometry
        let groups = analyze_kv_layer_groups(module);
        let page_specs: Vec<KvPageSpec> = groups
            .iter()
            .map(|g| KvPageSpec {
                data_bytes: g.bytes_per_page,
                stride_alignment: Some(256),
            })
            .collect();
        let jenga = plan_jenga_kv_geometry(&page_specs, 0.125, 128 * 1024 * 1024);
        for (g_idx, group) in groups.iter().enumerate() {
            for &layer_idx in &group.layer_indices {
                if let Some(&op_id) = attn_ops.get(layer_idx)
                    && let Some(op) = module.ops.get_mut(op_id)
                {
                    op.attributes.insert(
                        "castforge.kv_layer_group".to_string(),
                        Attribute::Int(group.group_id as i64),
                    );
                    if let Some(ref pool) = jenga {
                        op.attributes.insert(
                            "castforge.jenga_huge_block_bytes".to_string(),
                            Attribute::Int(pool.huge_block_bytes as i64),
                        );
                        op.attributes.insert(
                            "castforge.jenga_pages_per_huge_block".to_string(),
                            Attribute::Int(pool.pages_per_huge_block[g_idx] as i64),
                        );
                    }
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// Physical page specification for a single KV-cache layer group in a unified Jenga block pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPageSpec {
    /// Non-zero payload size in bytes for one KV page in this layer group.
    pub data_bytes: u64,
    /// Optional hardware stride alignment in bytes (e.g., 256B for Tensor Core / TMA transfers).
    pub stride_alignment: Option<u64>,
}

/// Unified "Jenga" huge-block pool geometry across heterogeneous KV-cache layer groups.
///
/// Every huge block of `huge_block_bytes` is an exact integer multiple of each group's
/// `padded_page_bytes[i]`, allowing full-attention, sliding-window, MLA, and SSM layers to
/// dynamically share a single physical memory pool without external fragmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JengaPoolGeometry {
    /// Size in bytes of one unified huge block (`lcm` of all `padded_page_bytes`).
    pub huge_block_bytes: u64,
    /// Padded byte stride of one page for each layer group.
    pub padded_page_bytes: Vec<u64>,
    /// Number of pages of group `i` that pack into one huge block.
    pub pages_per_huge_block: Vec<u64>,
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn checked_lcm_u64(a: u64, b: u64) -> Option<u64> {
    if a == 0 || b == 0 {
        return None;
    }
    (a / gcd_u64(a, b)).checked_mul(b)
}

/// Plans a unified "Jenga" huge-block pool geometry across heterogeneous [`KvPageSpec`]s.
///
/// Each page may be padded up to `1.0 + max_padding_fraction` (respecting `stride_alignment`)
/// to minimize the least common multiple (`huge_block_bytes <= max_huge_block_bytes`).
pub fn plan_jenga_kv_geometry(
    pages: &[KvPageSpec],
    max_padding_fraction: f64,
    max_huge_block_bytes: u64,
) -> Option<JengaPoolGeometry> {
    if pages.is_empty() || !(0.0..1.0).contains(&max_padding_fraction) || max_huge_block_bytes == 0
    {
        return None;
    }

    let mut candidates_per_group: Vec<Vec<u64>> = Vec::with_capacity(pages.len());
    for spec in pages {
        if spec.data_bytes == 0 {
            return None;
        }
        let align = spec.stride_alignment.unwrap_or(1).max(1);
        let min_stride = spec.data_bytes.div_ceil(align) * align;
        let max_allowed = ((spec.data_bytes as f64) * (1.0 + max_padding_fraction)).floor() as u64;
        let max_stride = max_allowed.max(min_stride);

        let mut strides = Vec::new();
        let mut s = min_stride;
        while s <= max_stride && strides.len() < 32 {
            strides.push(s);
            s = match s.checked_add(align) {
                Some(next) => next,
                None => break,
            };
        }
        if strides.is_empty() {
            return None;
        }
        candidates_per_group.push(strides);
    }

    let mut best_lcm: Option<u64> = None;
    let mut best_strides: Vec<u64> = Vec::new();
    let mut current = vec![0u64; pages.len()];

    fn search(
        group_idx: usize,
        running_lcm: u64,
        candidates: &[Vec<u64>],
        max_huge: u64,
        current: &mut [u64],
        best_lcm: &mut Option<u64>,
        best_strides: &mut Vec<u64>,
    ) {
        if let Some(best) = *best_lcm
            && running_lcm >= best
        {
            return;
        }
        if group_idx == candidates.len() {
            *best_lcm = Some(running_lcm);
            *best_strides = current.to_vec();
            return;
        }
        for &s in &candidates[group_idx] {
            if let Some(next_lcm) = checked_lcm_u64(running_lcm, s)
                && next_lcm <= max_huge
            {
                current[group_idx] = s;
                search(
                    group_idx + 1,
                    next_lcm,
                    candidates,
                    max_huge,
                    current,
                    best_lcm,
                    best_strides,
                );
            }
        }
    }

    search(
        0,
        1,
        &candidates_per_group,
        max_huge_block_bytes,
        &mut current,
        &mut best_lcm,
        &mut best_strides,
    );

    let huge_block_bytes = best_lcm?;
    let pages_per_huge_block = best_strides.iter().map(|&s| huge_block_bytes / s).collect();
    Some(JengaPoolGeometry {
        huge_block_bytes,
        padded_page_bytes: best_strides,
        pages_per_huge_block,
    })
}

/// Attention mask regime for a [`KvLayerGroup`] during prefix-cache hit planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAttentionMaskKind {
    /// Standard causal full attention: requires a contiguous cached prefix `[0..h)`.
    FullAttention,
    /// Sliding-window causal attention: requires at most `blocks_in_window` consecutive cached
    /// blocks immediately preceding the stopping point `h`.
    SlidingWindow { blocks_in_window: usize },
}

/// Computes the longest valid prefix-cache hit `h` (`0 <= h <= num_blocks`) for a full-attention
/// layer group where `is_cached(block_idx)` reports whether block `block_idx` is resident.
pub fn longest_full_attention_hit(num_blocks: usize, is_cached: impl Fn(usize) -> bool) -> usize {
    (0..num_blocks).take_while(|&i| is_cached(i)).count()
}

/// Computes the longest valid prefix-cache hit `h` (`0 <= h <= num_blocks`) for a sliding-window
/// attention layer group with window size `blocks_in_window` (`>= 1`).
///
/// A stopping point `h` is valid iff all blocks in `h.saturating_sub(blocks_in_window)..h` are
/// cached (i.e. the `blocks_in_window` blocks immediately before `h`, or `0..h` when `h <= W`).
pub fn longest_sliding_window_hit(
    num_blocks: usize,
    blocks_in_window: usize,
    is_cached: impl Fn(usize) -> bool,
) -> usize {
    let w = blocks_in_window.max(1);
    for h in (1..=num_blocks).rev() {
        let start = h.saturating_sub(w);
        if (start..h).all(&is_cached) {
            return h;
        }
    }
    0
}

/// Computes the longest joint prefix-cache hit across hybrid full-attention and sliding-window
/// [`KvLayerGroup`]s by iteratively narrowing the candidate stopping point until convergence.
pub fn longest_joint_prefix_hit(
    num_blocks: usize,
    groups: &[(KvAttentionMaskKind, &dyn Fn(usize) -> bool)],
) -> usize {
    if groups.is_empty() || num_blocks == 0 {
        return 0;
    }
    let mut upper = num_blocks;
    loop {
        let prev = upper;
        for &(mask_kind, is_cached) in groups {
            upper = match mask_kind {
                KvAttentionMaskKind::FullAttention => longest_full_attention_hit(upper, is_cached),
                KvAttentionMaskKind::SlidingWindow { blocks_in_window } => {
                    longest_sliding_window_hit(upper, blocks_in_window, is_cached)
                }
            };
            if upper == 0 {
                return 0;
            }
        }
        if upper == prev {
            return upper;
        }
    }
}

/// FlashDecoding / Split-KV Attention Lowering Pass (`lower-flash-decoding`).
pub struct LowerFlashDecodingPass {
    pub split_kv: usize,
}

impl Default for LowerFlashDecodingPass {
    fn default() -> Self {
        Self { split_kv: 4 }
    }
}

impl Pass for LowerFlashDecodingPass {
    fn name(&self) -> &'static str {
        "lower-flash-decoding"
    }

    fn description(&self) -> &'static str {
        "Partitions KV sequence across parallel splits with LSE reduction for LLM decoding"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let split_val = self.split_kv.max(2) as i64;
        for op_id in module.live_ops() {
            let Some(op) = module.ops.get_mut(op_id) else {
                continue;
            };
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                let mut ch = false;
                ch |= op.attributes.insert(
                    "castforge.flash_decoding".to_string(),
                    Attribute::Bool(true),
                ) != Some(Attribute::Bool(true));
                ch |= op
                    .attributes
                    .insert("castforge.split_kv".to_string(), Attribute::Int(split_val))
                    != Some(Attribute::Int(split_val));
                ch |= op.attributes.insert(
                    "castforge.partial_lse_reduce".to_string(),
                    Attribute::Bool(true),
                ) != Some(Attribute::Bool(true));
                if ch {
                    modified += 1;
                }
            }
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// Speculative Decoding Tree Attention Lowering Pass (`lower-tree-attention`).
pub struct LowerTreeAttentionPass {
    pub draft_tokens: usize,
    pub tree_depth: usize,
}

impl Default for LowerTreeAttentionPass {
    fn default() -> Self {
        Self {
            draft_tokens: 4,
            tree_depth: 2,
        }
    }
}

impl Pass for LowerTreeAttentionPass {
    fn name(&self) -> &'static str {
        "lower-tree-attention"
    }

    fn description(&self) -> &'static str {
        "Configures Medusa/EAGLE speculative decoding tree-attention verification masks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let draft_val = self.draft_tokens.max(1) as i64;
        let depth_val = self.tree_depth.max(1) as i64;
        for op_id in module.live_ops() {
            let Some(op) = module.ops.get_mut(op_id) else {
                continue;
            };
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                let mut ch = false;
                ch |= op.attributes.insert(
                    "castforge.speculative_tree".to_string(),
                    Attribute::Bool(true),
                ) != Some(Attribute::Bool(true));
                ch |= op.attributes.insert(
                    "castforge.draft_tokens".to_string(),
                    Attribute::Int(draft_val),
                ) != Some(Attribute::Int(draft_val));
                ch |= op.attributes.insert(
                    "castforge.tree_depth".to_string(),
                    Attribute::Int(depth_val),
                ) != Some(Attribute::Int(depth_val));
                if ch {
                    modified += 1;
                }
            }
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// Sentinel scale used when a V tensor, head, or page is identically zero, keeping
/// `dequant(quant(0)) == 0.0` exact and avoiding `0.0 / 0.0` NaNs.
pub const KV_ZERO_SCALE_SENTINEL: f32 = 1.0;

/// Quantization scale granularity for the V plane in [`AsymK16V8Codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvScaleScope {
    /// Single FP32 scale across the entire V buffer (`1` scale).
    PerTensor,
    /// One FP32 scale per KV head across all pages (`num_kv_heads` scales).
    PerLayerHead,
    /// One FP32 scale per `(page, kv_head)` pair (`num_pages * num_kv_heads` scales).
    PerPageHead,
}

/// Asymmetric K16/V8 encoded KV-cache payload (inspired by LMCache `asym_k16_v8`).
///
/// Keys (`K`) are preserved in 16-bit IEEE-754 half precision (`u16` bits) to protect RoPE phase
/// and exponential $\text{softmax}(Q K^T / \sqrt{d_k})$ logits, while Values (`V`) are quantized
/// to 8-bit signed codes (`i8`) with scoped FP32 scales.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedAsymKvPage {
    pub k_fp16_bits: Vec<u16>,
    pub v_int8_codes: Vec<i8>,
    pub v_scales: Vec<f32>,
    pub scope: KvScaleScope,
    pub num_pages: usize,
    pub page_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl EncodedAsymKvPage {
    /// Total compressed byte footprint (`K` in 2B + `V` in 1B + FP32 scale metadata).
    pub fn payload_bytes(&self) -> usize {
        self.k_fp16_bits.len() * 2 + self.v_int8_codes.len() + self.v_scales.len() * 4
    }

    /// Baseline 16-bit (`FP16`/`BF16`) byte footprint for both `K` and `V` (`2B + 2B` per elem).
    pub fn uncompressed_fp16_bytes(&self) -> usize {
        (self.k_fp16_bits.len() + self.v_int8_codes.len()) * 2
    }
}

/// Asymmetric `K16/V8` KV-cache encoder/decoder with `PerTensor`, `PerLayerHead`, and
/// `PerPageHead` scale scopes.
pub struct AsymK16V8Codec;

impl AsymK16V8Codec {
    const QMAX: f32 = 127.0;

    /// Encodes `k` and `v` buffers of shape `[num_pages, page_tokens, num_kv_heads, head_dim]`
    /// into an [`EncodedAsymKvPage`].
    pub fn encode(
        k: &[f32],
        v: &[f32],
        num_pages: usize,
        page_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scope: KvScaleScope,
    ) -> Result<EncodedAsymKvPage> {
        use crate::ir::f32_to_f16_bits;
        let expected = num_pages
            .saturating_mul(page_tokens)
            .saturating_mul(num_kv_heads)
            .saturating_mul(head_dim);
        if expected == 0 || k.len() != expected || v.len() != expected {
            anyhow::bail!(
                "AsymK16V8Codec::encode shape mismatch: expected {expected} elems, \
                 got k={} v={}",
                k.len(),
                v.len()
            );
        }

        let k_fp16_bits: Vec<u16> = k.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let v_scales =
            Self::compute_v_scales(v, num_pages, page_tokens, num_kv_heads, head_dim, scope);
        let mut v_int8_codes = vec![0i8; expected];

        let head_stride = head_dim;
        let tok_stride = num_kv_heads * head_stride;
        let page_stride = page_tokens * tok_stride;

        for p in 0..num_pages {
            for t in 0..page_tokens {
                for h in 0..num_kv_heads {
                    let scale = match scope {
                        KvScaleScope::PerTensor => v_scales[0],
                        KvScaleScope::PerLayerHead => v_scales[h],
                        KvScaleScope::PerPageHead => v_scales[p * num_kv_heads + h],
                    };
                    let base = p * page_stride + t * tok_stride + h * head_stride;
                    for d in 0..head_dim {
                        let val = v[base + d];
                        let q = (val / scale).round().clamp(-Self::QMAX, Self::QMAX) as i8;
                        v_int8_codes[base + d] = q;
                    }
                }
            }
        }

        Ok(EncodedAsymKvPage {
            k_fp16_bits,
            v_int8_codes,
            v_scales,
            scope,
            num_pages,
            page_tokens,
            num_kv_heads,
            head_dim,
        })
    }

    /// Computes scoped FP32 quantization scales for `v`, substituting [`KV_ZERO_SCALE_SENTINEL`]
    /// whenever the maximum absolute value in a scope is `0.0`.
    pub fn compute_v_scales(
        v: &[f32],
        num_pages: usize,
        page_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scope: KvScaleScope,
    ) -> Vec<f32> {
        let head_stride = head_dim;
        let tok_stride = num_kv_heads * head_stride;
        let page_stride = page_tokens * tok_stride;

        match scope {
            KvScaleScope::PerTensor => {
                let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let s = if amax > 0.0 {
                    amax / Self::QMAX
                } else {
                    KV_ZERO_SCALE_SENTINEL
                };
                vec![s]
            }
            KvScaleScope::PerLayerHead => {
                let mut scales = vec![KV_ZERO_SCALE_SENTINEL; num_kv_heads];
                for (h, slot) in scales.iter_mut().enumerate() {
                    let mut amax = 0.0f32;
                    for p in 0..num_pages {
                        for t in 0..page_tokens {
                            let base = p * page_stride + t * tok_stride + h * head_stride;
                            for d in 0..head_dim {
                                amax = amax.max(v[base + d].abs());
                            }
                        }
                    }
                    *slot = if amax > 0.0 {
                        amax / Self::QMAX
                    } else {
                        KV_ZERO_SCALE_SENTINEL
                    };
                }
                scales
            }
            KvScaleScope::PerPageHead => {
                let mut scales = vec![KV_ZERO_SCALE_SENTINEL; num_pages * num_kv_heads];
                for p in 0..num_pages {
                    for h in 0..num_kv_heads {
                        let mut amax = 0.0f32;
                        for t in 0..page_tokens {
                            let base = p * page_stride + t * tok_stride + h * head_stride;
                            for d in 0..head_dim {
                                amax = amax.max(v[base + d].abs());
                            }
                        }
                        scales[p * num_kv_heads + h] = if amax > 0.0 {
                            amax / Self::QMAX
                        } else {
                            KV_ZERO_SCALE_SENTINEL
                        };
                    }
                }
                scales
            }
        }
    }

    /// Decodes an [`EncodedAsymKvPage`] back into `(k_f32, v_f32)`.
    pub fn decode(encoded: &EncodedAsymKvPage) -> (Vec<f32>, Vec<f32>) {
        use crate::ir::f16_bits_to_f32;
        let k_f32: Vec<f32> = encoded
            .k_fp16_bits
            .iter()
            .map(|&bits| f16_bits_to_f32(bits))
            .collect();
        let total = encoded.v_int8_codes.len();
        let mut v_f32 = vec![0.0f32; total];

        let head_stride = encoded.head_dim;
        let tok_stride = encoded.num_kv_heads * head_stride;
        let page_stride = encoded.page_tokens * tok_stride;

        for p in 0..encoded.num_pages {
            for t in 0..encoded.page_tokens {
                for h in 0..encoded.num_kv_heads {
                    let scale = match encoded.scope {
                        KvScaleScope::PerTensor => encoded.v_scales[0],
                        KvScaleScope::PerLayerHead => encoded.v_scales[h],
                        KvScaleScope::PerPageHead => encoded.v_scales[p * encoded.num_kv_heads + h],
                    };
                    let base = p * page_stride + t * tok_stride + h * head_stride;
                    for d in 0..encoded.head_dim {
                        v_f32[base + d] = (encoded.v_int8_codes[base + d] as f32) * scale;
                    }
                }
            }
        }
        (k_f32, v_f32)
    }
}

/// Geometry and rotary embedding parameters for CacheBlend in-place delta re-RoPE
/// (inspired by LMCache `csrc/cuda/pos_kernels.cuh` and `blend_kernels.cuh`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheBlendRopeSpec {
    /// Number of KV heads per token.
    pub num_kv_heads: usize,
    /// Element stride between consecutive heads (`== head_size` for contiguous K, or
    /// `2 * head_size` for packed K/V buffers so only the K half is rotated).
    pub head_stride: usize,
    /// Element offset inside each head where RoPE dimensions start (`0` for standard GQA/MHA,
    /// or `kv_lora_rank` for DeepSeek MLA where decoupled RoPE keys trail the latent vector).
    pub rope_base_offset: usize,
    /// Even number of rotary embedding dimensions (`rot_dim <= head_stride - rope_base_offset`).
    pub rot_dim: usize,
    /// Base RoPE frequency (`10000.0` for standard LLaMA, `500000.0` for LLaMA-3).
    pub rope_theta: f32,
    /// `true` for GPT-NeoX / LLaMA half-split pairing `(i, i + rot_dim/2)`; `false` for
    /// GPT-J interleaved pairing `(2*i, 2*i + 1)`.
    pub is_neox: bool,
}

/// Rotates cached Key vectors `key_buf` in-place from `old_positions` to `new_positions`
/// using the abelian group property $\mathcal{R}(p_{\text{new}}) = \mathcal{R}(p_{\text{new}} -
/// p_{\text{old}}) \cdot \mathcal{R}(p_{\text{old}})$.
///
/// This enables non-prefix KV-cache reuse (e.g. RAG document chunks spliced at arbitrary prompt
/// offsets) without recomputing QKV projections.
pub fn re_rope_key_in_place(
    key_buf: &mut [f32],
    old_positions: &[i64],
    new_positions: &[i64],
    spec: &CacheBlendRopeSpec,
) -> Result<()> {
    let num_tokens = old_positions.len();
    if new_positions.len() != num_tokens {
        anyhow::bail!(
            "re_rope_key_in_place position length mismatch: old={} new={}",
            old_positions.len(),
            new_positions.len()
        );
    }
    if spec.rot_dim == 0 || !spec.rot_dim.is_multiple_of(2) {
        anyhow::bail!(
            "re_rope_key_in_place requires positive even rot_dim, got {}",
            spec.rot_dim
        );
    }
    if spec.rope_base_offset + spec.rot_dim > spec.head_stride {
        anyhow::bail!(
            "re_rope_key_in_place rope_base_offset ({}) + rot_dim ({}) exceeds head_stride ({})",
            spec.rope_base_offset,
            spec.rot_dim,
            spec.head_stride
        );
    }
    let tok_stride = spec.num_kv_heads.saturating_mul(spec.head_stride);
    let required = num_tokens.saturating_mul(tok_stride);
    if key_buf.len() < required {
        anyhow::bail!(
            "re_rope_key_in_place buffer too small: len={} < required={required}",
            key_buf.len()
        );
    }

    let half = spec.rot_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / spec.rope_theta.powf((2 * i) as f32 / spec.rot_dim as f32))
        .collect();

    for t in 0..num_tokens {
        let delta_pos = (new_positions[t] - old_positions[t]) as f32;
        if delta_pos == 0.0 {
            continue;
        }
        for h in 0..spec.num_kv_heads {
            let base = t * tok_stride + h * spec.head_stride + spec.rope_base_offset;
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = delta_pos * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let (idx0, idx1) = if spec.is_neox {
                    (base + i, base + i + half)
                } else {
                    (base + 2 * i, base + 2 * i + 1)
                };
                let k0 = key_buf[idx0];
                let k1 = key_buf[idx1];
                key_buf[idx0] = k0 * cos_a - k1 * sin_a;
                key_buf[idx1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
    Ok(())
}

/// Configuration for chained prefix-cache token-block hashing
/// (inspired by Modular `rust_kv/kv-core/src/hash.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainedBlockHashConfig<'a> {
    /// Number of tokens per KV-cache page block (`> 0`). Trailing ragged tokens
    /// (`tokens.len() % block_size`) are dropped so partial blocks never pollute the cache.
    pub block_size: usize,
    /// 64-bit domain seed for the hash chain.
    pub seed: u64,
    /// Optional model/LoRA-adapter/quantization salt mixed into the chain root.
    pub salt: &'a [u8],
    /// Optional parent block hash when continuing a multi-turn or chunked prefix.
    pub parent_hash: Option<u64>,
}

impl<'a> Default for ChainedBlockHashConfig<'a> {
    fn default() -> Self {
        Self {
            block_size: 16,
            seed: 0,
            salt: b"",
            parent_hash: None,
        }
    }
}

fn mix64(mut x: u64, k: u64) -> u64 {
    x ^= k;
    let wide = (x as u128).wrapping_mul(0x9e3779b97f4a7c15u128);
    let folded = (wide as u64) ^ ((wide >> 64) as u64);
    folded.rotate_left(27).wrapping_mul(0x94d049bb133111ebu64)
}

fn hash_bytes_64(mut state: u64, bytes: &[u8]) -> u64 {
    let (chunks, rem) = bytes.as_chunks::<8>();
    for &chunk in chunks {
        state = mix64(state, u64::from_le_bytes(chunk));
    }
    if !rem.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rem.len()].copy_from_slice(rem);
        state = mix64(state, u64::from_le_bytes(tail) ^ (rem.len() as u64));
    }
    mix64(state, bytes.len() as u64)
}

/// Slices `tokens` into full blocks of `cfg.block_size` (dropping any trailing ragged tokens)
/// and computes a chained 64-bit prefix hash per block:
///
/// $$H_k = \text{Hash}(\text{root},\; H_{k-1} \parallel \text{tokens}_k)$$
///
/// Each returned hash $H_k$ uniquely identifies the entire token prefix `[0 .. (k + 1) * B)`
/// under `(seed, salt)`.
pub fn hash_prefix_token_blocks(
    tokens: &[i32],
    cfg: &ChainedBlockHashConfig<'_>,
) -> Result<Vec<u64>> {
    if cfg.block_size == 0 {
        anyhow::bail!("hash_prefix_token_blocks: block_size must be > 0");
    }
    let root_key = if cfg.salt.is_empty() {
        mix64(cfg.seed, 0x517cc1b727220a95)
    } else {
        hash_bytes_64(mix64(cfg.seed, 0x517cc1b727220a95), cfg.salt)
    };
    let mut prev_hash = cfg.parent_hash.unwrap_or(root_key);
    let num_full_blocks = tokens.len() / cfg.block_size;
    let mut out = Vec::with_capacity(num_full_blocks);

    for blk_idx in 0..num_full_blocks {
        let slice = &tokens[blk_idx * cfg.block_size..(blk_idx + 1) * cfg.block_size];
        let mut block_state = root_key;
        for &tok in slice {
            block_state = mix64(block_state, (tok as u32) as u64);
        }
        block_state = mix64(block_state, cfg.block_size as u64);
        let chained = mix64(mix64(root_key, prev_hash), block_state);
        out.push(chained);
        prev_hash = chained;
    }
    Ok(out)
}

/// Compresses large uniform `stablehlo.constant` tensors with `>= min_elements` (`>= 2`) into a
/// 1-element scalar constant (`tensor<1xdtype>`) followed by `stablehlo.broadcast_in_dim`
/// (inspired by Modular `SplatLargeConstants.cpp`).
///
/// Also evicts any unreferenced uniform tensor entries from `module.tensor_store`, returning the
/// number of large constants compressed.
pub fn compress_splat_constants(module: &mut Module, min_elements: usize) -> usize {
    use crate::ir::{Dim, TensorType, Type, parse_dense_f32_literal};
    use smallvec::smallvec;

    let threshold = min_elements.max(2) as i64;
    let mut compressed = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
                continue;
            }
            let orig_res = op.results[0];
            let Some(tensor_ty) = module.values[orig_res].ty.as_tensor().cloned() else {
                continue;
            };
            let Some(num_elems) = tensor_ty.num_elements() else {
                continue;
            };
            if num_elems < threshold {
                continue;
            }

            // Extract uniform scalar value if all elements are identical
            let mut evicted_key: Option<String> = None;
            let uniform_scalar: Option<f32> = match op.attributes.get("value") {
                Some(Attribute::DenseElements { raw_literal, .. }) => {
                    parse_dense_f32_literal(raw_literal, num_elems as usize).and_then(|vals| {
                        let &v0 = vals.first()?;
                        vals.iter()
                            .all(|&x| (x - v0).abs() <= f32::EPSILON)
                            .then_some(v0)
                    })
                }
                Some(Attribute::TensorRef { key, .. }) => {
                    if let Some(entry) = module.tensor_store.get(key)
                        && let Some(vals) = entry.as_f32_slice()
                        && let Some(&v0) = vals.first()
                        && vals.iter().all(|&x| (x - v0).abs() <= f32::EPSILON)
                    {
                        evicted_key = Some(key.clone());
                        Some(v0)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let Some(v0) = uniform_scalar else {
                continue;
            };

            let scalar_tensor_ty = TensorType::new(vec![Dim::Static(1)], tensor_ty.dtype);
            let scalar_ty = Type::Tensor(scalar_tensor_ty.clone());
            let lit = format!("{v0:.6}");
            let mut const_attrs = IndexMap::new();
            const_attrs.insert(
                "value".to_string(),
                Attribute::DenseElements {
                    ty: scalar_tensor_ty,
                    raw_literal: lit,
                },
            );

            // Rewrite `op_id` in place to produce the 1-element scalar constant
            let scalar_val_id = module.values.insert_with_key(|vid| Value {
                id: vid,
                ssa_name: Some(format!("splat_scalar_{compressed}")),
                ty: scalar_ty,
                def: ValueDef::OpResult {
                    op: op_id,
                    index: 0,
                },
            });
            module.ops[op_id].results[0] = scalar_val_id;
            module.ops[op_id].attributes = const_attrs;

            // Insert `stablehlo.broadcast_in_dim` producing the original `orig_res` ValueId
            let mut bcast_attrs = IndexMap::new();
            bcast_attrs.insert(
                "broadcast_dimensions".to_string(),
                Attribute::DenseIntArray(vec![0]),
            );
            let bcast_op_id = module.ops.insert_with_key(|oid| Operation {
                id: oid,
                name: OpName::new("stablehlo", "broadcast_in_dim"),
                operands: smallvec![scalar_val_id],
                results: smallvec![orig_res],
                attributes: bcast_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: op.location.clone(),
            });
            module.values[orig_res].def = ValueDef::OpResult {
                op: bcast_op_id,
                index: 0,
            };

            if let Some(pos) = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == op_id)
            {
                module.blocks[block_id].ops.insert(pos + 1, bcast_op_id);
            }
            if let Some(k) = evicted_key {
                module.tensor_store.entries.shift_remove(&k);
            }
            compressed += 1;
        }
    }
    compressed
}

/// Summary report of CUDA Graph / HIP Graph capture stability analysis
/// (inspired by Modular `MarkStableTensorsPass.cpp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaGraphCaptureReport {
    /// `true` if every operation in the module has static shapes and no host-blocking sync ops.
    pub is_capture_safe: bool,
    /// Number of SSA tensor values backed by address-stable storage (arena slots, in-place
    /// aliases, or constant weights).
    pub stable_value_count: usize,
    /// Number of function input arguments requiring stable device pointers or graph-exec update.
    pub dynamic_input_count: usize,
    /// Operations that prevent full CUDA/HIP Graph capture (e.g. dynamic shapes or host sync).
    pub blocking_ops: Vec<String>,
}

/// Analyzes and stamps `castforge.cuda_graph_stable = true` onto operations whose tensor operands
/// and results reside in address-stable arena slots or constant weights, and stamps
/// `castforge.cuda_graph_capturable` on each `func.func` operation.
pub fn mark_cuda_graph_stable_tensors(module: &mut Module) -> CudaGraphCaptureReport {
    use crate::codegen::build_memory_plan;
    use crate::ir::Dim;

    let plan = build_memory_plan(module);
    let mut stable_value_count = 0usize;
    let mut dynamic_input_count = 0usize;
    let mut blocking_ops = Vec::new();

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        dynamic_input_count += module.blocks[block_id].arguments.len();
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if op.is_op("func", "return") {
                continue;
            }
            let has_dynamic_dim = op
                .results
                .iter()
                .chain(op.operands.iter())
                .filter_map(|&v| module.values[v].ty.as_tensor())
                .any(|t| !t.shape.iter().all(|d| matches!(d, Dim::Static(_))));
            let is_host_sync = op.is_op("castforge", "offload_to_host")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n == "castforge.offload_sync"));
            if has_dynamic_dim || is_host_sync {
                blocking_ops.push(op.name.to_string());
                continue;
            }

            let all_results_stable = op.results.iter().all(|&res| {
                op.is_op("stablehlo", "constant")
                    || plan
                        .allocation_for(res)
                        .is_some_and(|a| a.arena_slot.is_some() || a.inplace_from.is_some())
            });
            if all_results_stable && !op.results.is_empty() {
                stable_value_count += op.results.len();
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.attributes.insert(
                        "castforge.cuda_graph_stable".to_string(),
                        Attribute::Bool(true),
                    );
                }
            }
        }
    }

    let is_capture_safe = blocking_ops.is_empty();
    for fn_op_id in module.functions() {
        if let Some(fn_op) = module.ops.get_mut(fn_op_id) {
            fn_op.attributes.insert(
                "castforge.cuda_graph_capturable".to_string(),
                Attribute::Bool(is_capture_safe),
            );
        }
    }

    CudaGraphCaptureReport {
        is_capture_safe,
        stable_value_count,
        dynamic_input_count,
        blocking_ops,
    }
}

/// Intrusive doubly-linked LRU free-list links stored inline inside each [`KvPoolBlock`]
/// (inspired by Modular `rust_kv/kv-core/src/pool/free_list.rs`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvFreeLinks {
    pub prev: Option<usize>,
    pub next: Option<usize>,
    pub in_free_list: bool,
}

/// Metadata for a single physical block in [`KvBlockPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPoolBlock {
    pub block_id: usize,
    pub ref_count: usize,
    pub prefix_hash: Option<u64>,
    pub links: KvFreeLinks,
}

/// $O(1)$ Intrusive LRU KV-Cache Block Pool with prefix-hit revival, uncommitted front-recycle,
/// and reverse-chain eviction ordering (inspired by Modular `rust_kv/kv-core/src/pool`).
#[derive(Debug, Clone)]
pub struct KvBlockPool {
    blocks: Vec<KvPoolBlock>,
    free_head: Option<usize>,
    free_tail: Option<usize>,
    free_len: usize,
    hash_to_block: std::collections::HashMap<u64, usize>,
}

impl KvBlockPool {
    /// Creates a new [`KvBlockPool`] with `num_blocks` physical blocks (`0..num_blocks`), all
    /// initially on the free list in ascending ID order.
    pub fn new(num_blocks: usize) -> Self {
        let mut pool = Self {
            blocks: (0..num_blocks)
                .map(|id| KvPoolBlock {
                    block_id: id,
                    ref_count: 0,
                    prefix_hash: None,
                    links: KvFreeLinks::default(),
                })
                .collect(),
            free_head: None,
            free_tail: None,
            free_len: 0,
            hash_to_block: std::collections::HashMap::new(),
        };
        for id in 0..num_blocks {
            pool.push_free_back(id);
        }
        pool
    }

    /// Number of currently unreferenced blocks on the LRU free list.
    pub fn free_count(&self) -> usize {
        self.free_len
    }

    /// Total capacity in blocks.
    pub fn capacity(&self) -> usize {
        self.blocks.len()
    }

    /// Inspects metadata for `block_id`.
    pub fn block(&self, block_id: usize) -> Option<&KvPoolBlock> {
        self.blocks.get(block_id)
    }

    fn push_free_back(&mut self, id: usize) {
        debug_assert!(!self.blocks[id].links.in_free_list);
        if let Some(tail) = self.free_tail {
            self.blocks[tail].links.next = Some(id);
        } else {
            self.free_head = Some(id);
        }
        self.blocks[id].links = KvFreeLinks {
            prev: self.free_tail,
            next: None,
            in_free_list: true,
        };
        self.free_tail = Some(id);
        self.free_len += 1;
    }

    fn push_free_front(&mut self, id: usize) {
        debug_assert!(!self.blocks[id].links.in_free_list);
        if let Some(head) = self.free_head {
            self.blocks[head].links.prev = Some(id);
        } else {
            self.free_tail = Some(id);
        }
        self.blocks[id].links = KvFreeLinks {
            prev: None,
            next: self.free_head,
            in_free_list: true,
        };
        self.free_head = Some(id);
        self.free_len += 1;
    }

    fn unlink_free(&mut self, id: usize) {
        let links = self.blocks[id].links;
        debug_assert!(links.in_free_list);
        if let Some(prev) = links.prev {
            self.blocks[prev].links.next = links.next;
        } else {
            self.free_head = links.next;
        }
        if let Some(next) = links.next {
            self.blocks[next].links.prev = links.prev;
        } else {
            self.free_tail = links.prev;
        }
        self.blocks[id].links = KvFreeLinks::default();
        self.free_len -= 1;
    }

    /// Allocates a block from the front of the LRU free list in $O(1)$, evicting its old
    /// `prefix_hash` mapping if it was previously committed.
    pub fn allocate(&mut self) -> Option<usize> {
        let id = self.free_head?;
        self.unlink_free(id);
        if let Some(old_hash) = self.blocks[id].prefix_hash.take()
            && self.hash_to_block.get(&old_hash) == Some(&id)
        {
            self.hash_to_block.remove(&old_hash);
        }
        self.blocks[id].ref_count = 1;
        Some(id)
    }

    /// Commits a full block `block_id` under `prefix_hash` so future requests can match it.
    pub fn commit(&mut self, block_id: usize, prefix_hash: u64) {
        if let Some(blk) = self.blocks.get_mut(block_id) {
            blk.prefix_hash = Some(prefix_hash);
            self.hash_to_block.insert(prefix_hash, block_id);
        }
    }

    /// Looks up `prefix_hash` in the pool; if the block is currently sitting on the free list
    /// awaiting eviction, revives it in $O(1)$ by unlinking it from the middle of the free list!
    pub fn touch_or_revive_prefix(&mut self, prefix_hash: u64) -> Option<usize> {
        let &id = self.hash_to_block.get(&prefix_hash)?;
        if self.blocks[id].prefix_hash != Some(prefix_hash) {
            return None;
        }
        if self.blocks[id].links.in_free_list {
            self.unlink_free(id);
        }
        self.blocks[id].ref_count += 1;
        Some(id)
    }

    /// Releases a single block:
    /// - If `ref_count` drops to `0` and the block was **committed** (`prefix_hash.is_some()`),
    ///   appends it to the **back** (MRU tail) of the free list so its cached prefix stays alive
    ///   as long as possible.
    /// - If `ref_count` drops to `0` and the block was **uncommitted** (`prefix_hash.is_none()`),
    ///   prepends it to the **front** (LRU head) of the free list so it is recycled immediately
    ///   before evicting any committed prefix block!
    pub fn release_block(&mut self, block_id: usize) {
        let Some(blk) = self.blocks.get_mut(block_id) else {
            return;
        };
        if blk.ref_count == 0 {
            return;
        }
        blk.ref_count -= 1;
        if blk.ref_count == 0 {
            let committed = blk.prefix_hash.is_some();
            if committed {
                self.push_free_back(block_id);
            } else {
                self.push_free_front(block_id);
            }
        }
    }

    /// Releases a request's chain of blocks `[b_0, b_1, ..., b_{k-1}]` in **reverse order**
    /// (`b_{k-1}` first, `b_0` last) so the tail of the chain sits closer to the LRU eviction
    /// head than the root prefix block `b_0`.
    pub fn release_request_chain(&mut self, chain: &[usize]) {
        for &block_id in chain.iter().rev() {
            self.release_block(block_id);
        }
    }
}

/// Hardware transfer execution path for moving KV-cache pages between host memory and GPU VRAM
/// (inspired by LMCache `csrc/cuda/mp_mem_kernels.cuh` and `transfer_plan_types.cuh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDmaTransferMode {
    /// Direct 1D Copy-Engine batch DMA (`cudaMemcpyBatchAsync`), bypassing SM kernels.
    DirectCopy1D,
    /// Direct 2D Pitched Copy-Engine DMA (`cudaMemcpy3DBatchAsync`) collapsing all `num_layers`
    /// of a block into a single pitched transfer (`EngineKvFormat::CrossLayer`).
    DirectCopy2DPitched,
    /// Staged H2D/D2H copy + SM scatter/gather kernel for non-contiguous layouts (`Hnd`).
    StagedSmKernel,
}

/// Returns `true` if `format` stores each page block as a contiguous `[block_size, nh, hs]` run
/// eligible for direct Copy-Engine DMA without SM gather/scatter kernels.
pub fn direct_copy_format_supported(format: EngineKvFormat) -> bool {
    matches!(
        format,
        EngineKvFormat::Nhd | EngineKvFormat::CrossLayer | EngineKvFormat::MlaLatent
    )
}

/// A single DMA copy descriptor, guaranteed not to straddle a `host_buffer_alignment` pin-chunk
/// boundary (`cudaHostRegister` region boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvDmaCopyEntry {
    pub block_id: usize,
    pub host_offset: u64,
    pub device_offset: u64,
    pub width_bytes: u64,
    pub height_layers: usize,
}

/// Planned KV-cache transfer batch across a sequence of page blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvDmaTransferPlan {
    pub mode: KvDmaTransferMode,
    pub entries: Vec<KvDmaCopyEntry>,
    pub pin_boundary_splits: usize,
}

/// Plans a KV-cache H2D/D2H transfer for `block_ids` starting at `initial_host_offset`,
/// selecting Direct 1D/2D Copy-Engine DMA vs. Staged SM kernels and splitting any copy that
/// crosses a `host_buffer_alignment` (`cudaHostRegister`) boundary.
pub fn plan_kv_dma_transfer(
    format: EngineKvFormat,
    num_layers: usize,
    bytes_per_layer_block: u64,
    block_ids: &[usize],
    initial_host_offset: u64,
    host_buffer_alignment: u64,
) -> KvDmaTransferPlan {
    let layers = num_layers.max(1);
    let align = host_buffer_alignment.max(1);
    let mode = if !direct_copy_format_supported(format) {
        KvDmaTransferMode::StagedSmKernel
    } else if format == EngineKvFormat::CrossLayer && layers > 1 {
        KvDmaTransferMode::DirectCopy2DPitched
    } else {
        KvDmaTransferMode::DirectCopy1D
    };

    let mut entries = Vec::new();
    let mut pin_boundary_splits = 0usize;
    let mut cur_host = initial_host_offset;

    for &blk_id in block_ids {
        if mode == KvDmaTransferMode::DirectCopy2DPitched {
            let total_block_bytes = bytes_per_layer_block * (layers as u64);
            let mut rem = total_block_bytes;
            let mut dev_off = (blk_id as u64) * total_block_bytes;
            while rem > 0 {
                let chunk_rem = align - (cur_host % align);
                let step = rem.min(chunk_rem);
                if step < rem {
                    pin_boundary_splits += 1;
                }
                entries.push(KvDmaCopyEntry {
                    block_id: blk_id,
                    host_offset: cur_host,
                    device_offset: dev_off,
                    width_bytes: step,
                    height_layers: 1,
                });
                cur_host += step;
                dev_off += step;
                rem -= step;
            }
        } else {
            for layer_idx in 0..layers {
                let mut rem = bytes_per_layer_block;
                let mut dev_off =
                    ((layer_idx * 1024 + blk_id) as u64).saturating_mul(bytes_per_layer_block);
                while rem > 0 {
                    let chunk_rem = align - (cur_host % align);
                    let step = rem.min(chunk_rem);
                    if step < rem {
                        pin_boundary_splits += 1;
                    }
                    entries.push(KvDmaCopyEntry {
                        block_id: blk_id,
                        host_offset: cur_host,
                        device_offset: dev_off,
                        width_bytes: step,
                        height_layers: 1,
                    });
                    cur_host += step;
                    dev_off += step;
                    rem -= step;
                }
            }
        }
    }

    KvDmaTransferPlan {
        mode,
        entries,
        pin_boundary_splits,
    }
}

/// Fuses sole-use unary elementwise producers (`negate`, `abs`, `tanh`, `exponential`, `log`,
/// `sqrt`, `rsqrt`, `logistic`, `convert`) into the data-operand prologue of a `dot_general`,
/// `gather`, or `reduce` consumer (inspired by Modular `PrologueFuser.cpp` & `InputFusion.cpp`).
///
/// Eliminates the intermediate producer buffer and evaluates the unary transform on-the-fly
/// during the consumer's load. Returns the number of prologues fused.
pub fn fuse_sole_use_prologues(module: &mut Module) -> usize {
    let mut fused_count = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        let mut use_counts: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        for &op_id in &ops {
            if let Some(op) = module.ops.get(op_id) {
                for &opnd in &op.operands {
                    *use_counts.entry(opnd).or_insert(0) += 1;
                }
            }
        }

        let mut to_remove: std::collections::HashSet<OpId> = std::collections::HashSet::new();

        for &cons_id in &ops {
            if to_remove.contains(&cons_id) {
                continue;
            }
            let Some(cons_op) = module.ops.get(cons_id).cloned() else {
                continue;
            };
            let is_eligible_consumer = cons_op.is_op("stablehlo", "dot_general")
                || cons_op.is_op("stablehlo", "gather")
                || cons_op.is_op("stablehlo", "reduce");
            if !is_eligible_consumer
                || cons_op.operands.is_empty()
                || cons_op.attributes.contains_key("castforge.prologue_op")
            {
                continue;
            }
            let first_opnd = cons_op.operands[0];
            if use_counts.get(&first_opnd).copied().unwrap_or(0) != 1 {
                continue;
            }
            let ValueDef::OpResult { op: prod_id, .. } = module.values[first_opnd].def else {
                continue;
            };
            if to_remove.contains(&prod_id) {
                continue;
            }
            let Some(prod_op) = module.ops.get(prod_id).cloned() else {
                continue;
            };
            let is_unary_elementwise = prod_op.name.dialect == "stablehlo"
                && prod_op.operands.len() == 1
                && matches!(
                    prod_op.name.name.as_str(),
                    "negate"
                        | "abs"
                        | "tanh"
                        | "exponential"
                        | "log"
                        | "sqrt"
                        | "rsqrt"
                        | "logistic"
                        | "convert"
                );
            if !is_unary_elementwise {
                continue;
            }

            let raw_input = prod_op.operands[0];
            let prod_name = prod_op.name.name.clone();
            let merged_locs = [prod_op.location.clone(), cons_op.location.clone()];
            if let Some(cons_mut) = module.ops.get_mut(cons_id) {
                cons_mut.operands[0] = raw_input;
                cons_mut.attributes.insert(
                    "castforge.prologue_op".to_string(),
                    Attribute::String(prod_name),
                );
                stamp_fused_provenance(cons_mut, &merged_locs);
            }
            to_remove.insert(prod_id);
            fused_count += 1;
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
    fused_count
}

/// Default maximum number of kernels allowed in a single fused launch sequence
/// (inspired by Modular `SequenceFusion.cpp` `kMaxKernelsInFusionSet = 16`, `GEX-2986`).
pub const DEFAULT_MAX_KERNELS_IN_FUSION_SET: usize = 16;

/// A bounded group of consecutive device operations sharing the same execution context
/// (`device_context` and `stream_id`) that can be dispatched as a single fused sequence
/// without overflowing kernel constant-parameter memory or deadlocking across streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceFusionSet {
    pub set_id: usize,
    pub device_context: String,
    pub stream_id: i64,
    pub op_ids: Vec<OpId>,
    pub op_names: Vec<String>,
}

/// Partitions consecutive device kernels into bounded [`SequenceFusionSet`]s (inspired by
/// Modular `SequenceFusion.cpp`: `useSameContexts` (`GEX-2590`) and `kMaxKernelsInFusionSet`
/// (`GEX-2986`)).
///
/// Breaks the active fusion set whenever:
/// 1. A collective, host-offload, or explicit barrier operation is encountered (`all_reduce`,
///    `all_gather`, `reduce_scatter`, `collective_permute`, `kv_offload`, `kv_reload`, or
///    `castforge.sequence_barrier = true`),
/// 2. The operation's `(castforge.device, castforge.stream_id)` context differs from the active
///    set's context (`useSameContexts`), or
/// 3. The active set reaches `max_kernels_per_set.max(1)` operations.
///
/// Each grouped operation is stamped with `castforge.sequence_fusion_set = <set_id>`.
pub fn plan_kernel_sequence_fusion(
    module: &mut Module,
    max_kernels_per_set: usize,
) -> Vec<SequenceFusionSet> {
    let cap = max_kernels_per_set.max(1);
    let mut sets: Vec<SequenceFusionSet> = Vec::new();
    let mut active: Option<SequenceFusionSet> = None;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            // Skip non-kernel metadata/constants and function terminators
            if op.is_op("func", "return") || op.is_op("stablehlo", "constant") {
                continue;
            }
            let op_short = op.name.name.as_str();
            let is_barrier = matches!(
                op_short,
                "all_reduce"
                    | "all_gather"
                    | "reduce_scatter"
                    | "collective_permute"
                    | "kv_offload"
                    | "kv_reload"
            ) || matches!(
                op.attributes.get("castforge.sequence_barrier"),
                Some(Attribute::Bool(true))
            );
            if is_barrier {
                if let Some(finished) = active.take() {
                    sets.push(finished);
                }
                continue;
            }

            let dev_ctx = match op.attributes.get("castforge.device") {
                Some(Attribute::String(s)) => s.clone(),
                _ => "gpu:0".to_string(),
            };
            let stream_id = match op.attributes.get("castforge.stream_id") {
                Some(Attribute::Int(id)) => *id,
                _ => 0,
            };
            let full_name = format!("{}.{}", op.name.dialect, op.name.name);

            let needs_new_set = match &active {
                None => true,
                Some(curr) => {
                    curr.device_context != dev_ctx
                        || curr.stream_id != stream_id
                        || curr.op_ids.len() >= cap
                }
            };
            if needs_new_set {
                if let Some(finished) = active.take() {
                    sets.push(finished);
                }
                let next_id = sets.len();
                active = Some(SequenceFusionSet {
                    set_id: next_id,
                    device_context: dev_ctx,
                    stream_id,
                    op_ids: Vec::new(),
                    op_names: Vec::new(),
                });
            }

            let set_id = active.as_ref().unwrap().set_id;
            if let Some(curr) = active.as_mut() {
                curr.op_ids.push(op_id);
                curr.op_names.push(full_name);
            }
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.sequence_fusion_set".to_string(),
                    Attribute::Int(set_id as i64),
                );
            }
        }
        if let Some(finished) = active.take() {
            sets.push(finished);
        }
    }
    sets
}

/// Combines multiple constituent [`Location`]s into a single deduplicated fused location
/// (`Location::Named("op1+op2+...")`), preserving source provenance across operator fusions
/// (inspired by Modular `GraphCompiler/AGENTS.md` `QA-C1` `getFusedLoc` & `FuseMutableLoads.cpp`).
pub fn fuse_locations(locs: &[Location]) -> Location {
    let mut parts: Vec<String> = Vec::new();
    let mut single_flc: Option<Location> = None;
    let mut non_unknown_count = 0usize;

    for loc in locs {
        match loc {
            Location::Unknown => {}
            Location::Named(name) => {
                non_unknown_count += 1;
                for token in name.split('+') {
                    let trimmed = token.trim();
                    if !trimmed.is_empty() && !parts.iter().any(|p| p == trimmed) {
                        parts.push(trimmed.to_string());
                    }
                }
            }
            Location::FileLineCol { file, line, col } => {
                non_unknown_count += 1;
                if non_unknown_count == 1 {
                    single_flc = Some(loc.clone());
                } else {
                    single_flc = None;
                }
                let formatted = format!("{file}:{line}:{col}");
                if !parts.iter().any(|p| p == &formatted) {
                    parts.push(formatted);
                }
            }
        }
    }

    if parts.is_empty() {
        Location::Unknown
    } else if non_unknown_count == 1
        && parts.len() == 1
        && let Some(flc) = single_flc
    {
        flc
    } else {
        Location::Named(parts.join("+"))
    }
}

/// Stamps `target_op` with the combined [`Location`] from `source_locs` via [`fuse_locations`]
/// and records a human-readable `castforge.provenance` string attribute when known so lowered
/// Loop IR and target codegen retain fused operator provenance comments.
pub fn stamp_fused_provenance(target_op: &mut Operation, source_locs: &[Location]) {
    let fused = fuse_locations(source_locs);
    if fused != Location::Unknown {
        let prov_str = match &fused {
            Location::Named(s) => s.clone(),
            Location::FileLineCol { file, line, col } => format!("{file}:{line}:{col}"),
            Location::Unknown => String::new(),
        };
        target_op.location = fused;
        if !prov_str.is_empty() {
            target_op.attributes.insert(
                "castforge.provenance".to_string(),
                Attribute::String(prov_str),
            );
        }
    }
}

fn anonymize_type_dims(ty: &mut Type) -> usize {
    match ty {
        Type::Tensor(t) => {
            let mut count = 0usize;
            for dim in &mut t.shape {
                if matches!(dim, Dim::Symbolic(_)) {
                    *dim = Dim::Dynamic;
                    count += 1;
                }
            }
            count
        }
        Type::Tuple(items) => items.iter_mut().map(anonymize_type_dims).sum(),
        Type::Function { inputs, results } => {
            let a: usize = inputs.iter_mut().map(anonymize_type_dims).sum();
            let b: usize = results.iter_mut().map(anonymize_type_dims).sum();
            a + b
        }
        _ => 0,
    }
}

/// Strips symbolic dimension names (`Dim::Symbolic(name)` -> `Dim::Dynamic`) across all SSA
/// values and function signatures in `module` (inspired by Modular `AnonymizeDimNames.cpp`
/// `stripStubDimNames`).
///
/// Idempotent: returns the number of symbolic dimension slots anonymized.
pub fn anonymize_symbolic_dims(module: &mut Module) -> usize {
    let mut count = 0usize;
    for (_, val) in module.values.iter_mut() {
        count += anonymize_type_dims(&mut val.ty);
    }
    for (_, op) in module.ops.iter_mut() {
        for (_, attr) in op.attributes.iter_mut() {
            if let Attribute::Type(ty) = attr {
                count += anonymize_type_dims(ty);
            }
        }
    }
    count
}

/// Deduplicates structurally identical `func.func` definitions in `module` modulo `@sym_name`,
/// SSA value numbering, and source locations (inspired by Modular `MergeIdenticalFuncs.cpp`).
///
/// For each equivalence class of 2 or more functions:
/// 1. Selects a deterministic canonical symbol (`@main` / `@forward` if present, otherwise the
///    lexicographically smallest symbol name),
/// 2. Fuses all duplicate functions' [`Location`]s into the canonical function via
///    [`stamp_fused_provenance`],
/// 3. Rewrites all `func.call` callee attributes across `module` to reference the canonical
///    symbol, and
/// 4. Removes the duplicate `func.func` operations from `module`.
///
/// Returns the number of duplicate functions merged and eliminated.
pub fn merge_identical_functions(module: &mut Module) -> usize {
    let fns = module.functions();
    if fns.len() <= 1 {
        return 0;
    }

    // Group functions by (signature_string, body_structural_hash)
    type FuncStubEntry = (OpId, String, Location);
    let mut groups: IndexMap<(String, u64), Vec<FuncStubEntry>> = IndexMap::new();
    for &fn_id in &fns {
        let Some(op) = module.ops.get(fn_id) else {
            continue;
        };
        let Some(sym_name) = op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        let Some(&reg_id) = op.regions.first() else {
            continue;
        };
        let sig_str = op
            .attributes
            .get("function_type")
            .map(|a| a.to_string())
            .unwrap_or_default();
        let body_hash = crate::codegen::compute_region_structural_hash(module, reg_id);
        groups.entry((sig_str, body_hash)).or_default().push((
            fn_id,
            sym_name,
            op.location.clone(),
        ));
    }

    let mut replacements: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut dead_fn_ops: std::collections::HashSet<OpId> = std::collections::HashSet::new();

    for (_, mut candidates) in groups {
        if candidates.len() <= 1 {
            continue;
        }
        // Sort deterministically: prefer entry points (`main`/`forward`), then lexicographical name
        candidates.sort_by(|a, b| {
            let a_entry = matches!(a.1.as_str(), "main" | "forward");
            let b_entry = matches!(b.1.as_str(), "main" | "forward");
            b_entry.cmp(&a_entry).then_with(|| a.1.cmp(&b.1))
        });

        let (canon_id, canon_name, _) = candidates[0].clone();
        let all_locs: Vec<Location> = candidates.iter().map(|(_, _, loc)| loc.clone()).collect();
        if let Some(canon_op) = module.ops.get_mut(canon_id) {
            stamp_fused_provenance(canon_op, &all_locs);
        }

        for (dup_id, dup_name, _) in candidates.into_iter().skip(1) {
            replacements.insert(dup_name, canon_name.clone());
            dead_fn_ops.insert(dup_id);
        }
    }

    if dead_fn_ops.is_empty() {
        return 0;
    }

    // Rewrite `func.call` callee attributes across all remaining operations
    for (_, op) in module.ops.iter_mut() {
        if op.is_op("func", "call")
            && let Some(callee) = op.attributes.get("callee").and_then(|a| a.as_str())
            && let Some(new_callee) = replacements.get(callee)
        {
            op.attributes.insert(
                "callee".to_string(),
                Attribute::SymbolRef(new_callee.clone()),
            );
        }
    }

    // Remove duplicate functions from the module root block and ops arena
    if let Some(root_reg) = module.regions.get(module.root_region)
        && let Some(&root_blk) = root_reg.blocks.first()
    {
        module.blocks[root_blk]
            .ops
            .retain(|id| !dead_fn_ops.contains(id));
    }
    let merged_count = dead_fn_ops.len();
    for dead_id in dead_fn_ops {
        module.ops.remove(dead_id);
    }
    merged_count
}

/// Fuses elementwise epilogue consumers across an intervening `stablehlo.reshape` into a
/// preceding `dot_general` or `fused_matmul_epilogue` (inspired by Modular
/// `ReshapeEpilogueFuser.cpp`).
///
/// Moves the bijective `reshape` past sole-use unary (`tanh`, `abs`, `negate`, `exponential`,
/// `log`, `logistic`, `sqrt`, `rsqrt`) or binary (`add`, `multiply`, `maximum`) elementwise
/// operations and invokes [`crate::passes::FuseMatmulEpiloguePass`] so `matmul -> reshape ->
/// bias -> activation` collapses into `fused_matmul_epilogue -> reshape`.
///
/// Returns the number of reshape-epilogue commutations performed.
pub fn fuse_reshape_epilogue(module: &mut Module) -> usize {
    let mut total_commuted = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };

        loop {
            let mut commuted_in_iter = false;
            let ops = module.blocks[block_id].ops.clone();

            for &cons_id in &ops {
                let Some(cons_op) = module.ops.get(cons_id).cloned() else {
                    continue;
                };
                if cons_op.name.dialect != "stablehlo" || cons_op.results.len() != 1 {
                    continue;
                }
                let op_short = cons_op.name.name.as_str();
                let is_unary = cons_op.operands.len() == 1
                    && matches!(
                        op_short,
                        "tanh"
                            | "abs"
                            | "negate"
                            | "exponential"
                            | "log"
                            | "logistic"
                            | "sqrt"
                            | "rsqrt"
                    );
                let is_binary = cons_op.operands.len() == 2
                    && matches!(op_short, "add" | "multiply" | "maximum");
                if !is_unary && !is_binary {
                    continue;
                }

                // Check if operand 0 is a sole-use `stablehlo.reshape` fed by GEMM/epilogue/add
                let r_val = cons_op.operands[0];
                if module.use_count(r_val) != 1 {
                    continue;
                }
                let ValueDef::OpResult { op: resh_id, .. } = module.values[r_val].def else {
                    continue;
                };
                let Some(resh_op) = module.ops.get(resh_id).cloned() else {
                    continue;
                };
                if !resh_op.is_op("stablehlo", "reshape") || resh_op.operands.len() != 1 {
                    continue;
                }
                let src_val = resh_op.operands[0];
                let src_ty = module.values[src_val].ty.clone();
                let ValueDef::OpResult { op: prod_id, .. } = module.values[src_val].def else {
                    continue;
                };
                let Some(prod_op) = module.ops.get(prod_id) else {
                    continue;
                };
                let prod_eligible = prod_op.is_op("stablehlo", "dot_general")
                    || prod_op.is_op("stablehlo", "add")
                    || prod_op.is_op("stablehlo", "composite");
                if !prod_eligible {
                    continue;
                }

                // For binary ops, resolve operand 1 in pre-reshape `src_ty` space
                let mut new_operands: SmallVec<[ValueId; 4]> = smallvec::smallvec![src_val];
                let mut dead_other_resh: Option<OpId> = None;
                if is_binary {
                    let other_val = cons_op.operands[1];
                    let ValueDef::OpResult { op: other_def, .. } = module.values[other_val].def
                    else {
                        continue;
                    };
                    let Some(other_op) = module.ops.get(other_def).cloned() else {
                        continue;
                    };
                    if other_op.is_op("stablehlo", "reshape")
                        && other_op.operands.len() == 1
                        && module.values[other_op.operands[0]].ty == src_ty
                    {
                        new_operands.push(other_op.operands[0]);
                        if module.use_count(other_val) == 1 {
                            dead_other_resh = Some(other_def);
                        }
                    } else if other_op.is_op("stablehlo", "constant")
                        && module.use_count(other_val) == 1
                    {
                        // Retype sole-use splat/dense constant to `src_ty`
                        module.values[other_val].ty = src_ty.clone();
                        new_operands.push(other_val);
                    } else {
                        continue;
                    }
                }

                // Commute in-place:
                // - `resh_id` becomes the elementwise op (`cons_op.name`) on `src_ty` -> `r_val`
                // - `cons_id` becomes `stablehlo.reshape` on `[r_val]` -> `cons_op.results[0]`
                module.values[r_val].ty = src_ty;
                let merged_locs = [resh_op.location.clone(), cons_op.location.clone()];
                if let Some(elem_mut) = module.ops.get_mut(resh_id) {
                    elem_mut.name = cons_op.name.clone();
                    elem_mut.operands = new_operands;
                    elem_mut.attributes = cons_op.attributes.clone();
                    stamp_fused_provenance(elem_mut, &merged_locs);
                }
                if let Some(resh_mut) = module.ops.get_mut(cons_id) {
                    resh_mut.name = OpName::new("stablehlo", "reshape");
                    resh_mut.operands = smallvec::smallvec![r_val];
                    resh_mut.attributes = resh_op.attributes.clone();
                    stamp_fused_provenance(resh_mut, &merged_locs);
                }
                if let Some(dead_id) = dead_other_resh {
                    module.remove_op(block_id, dead_id);
                }
                total_commuted += 1;
                commuted_in_iter = true;
                break;
            }

            if !commuted_in_iter {
                break;
            }
        }
    }

    if total_commuted > 0 {
        let _ = crate::passes::FuseMatmulEpiloguePass.run(module);
    }
    total_commuted
}

/// CacheGen 16-bit strictly monotonic Cumulative Distribution Function (CDF) and lossless
/// arithmetic range codec for quantized KV-cache token streams (inspired by LMCache
/// `csrc/cuda/cal_cdf.cu`, `ac_enc.cu`, and `ac_dec.cu`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheGenCdfCodec {
    pub max_bins: usize,
}

impl CacheGenCdfCodec {
    pub const MAX_BINS_SUPPORTED: usize = 64;
    pub const PRECISION_BITS: u32 = 16;

    pub fn new(max_bins: usize) -> Self {
        Self {
            max_bins: max_bins.clamp(2, Self::MAX_BINS_SUPPORTED),
        }
    }

    /// Computes the normalized 16-bit strictly monotonic CDF of length `max_bins + 1` across
    /// `symbols` in a single `(layer, channel)` stream using LMCache's `normalize_cdf_value`
    /// (`floor((0xFFFF - max_bins) * cum / ntokens) + i`), guaranteeing every bin has width
    /// `>= 1`.
    pub fn compute_channel_cdf(&self, symbols: &[u8]) -> Vec<u16> {
        let bins = self.max_bins;
        let mut hist = vec![0u32; bins + 1];
        for &sym in symbols {
            let idx = (sym as usize).min(bins - 1);
            hist[idx + 1] += 1;
        }
        let mut local_sum = 0u32;
        for count in hist.iter_mut().skip(1) {
            let v = *count;
            *count += local_sum;
            local_sum += v;
        }
        let max_u16_val = (0xFFFFusize).saturating_sub(bins) as u32;
        let mut cdf = vec![0u16; bins + 1];
        for (i, &cum) in hist.iter().enumerate() {
            let norm = (max_u16_val * cum).checked_div(local_sum).unwrap_or(0) as u16;
            cdf[i] = norm + (i as u16);
        }
        cdf
    }

    /// Encodes a channel's quantized `symbols` (`0..max_bins`) into a compact byte stream using
    /// 16-bit precision integer arithmetic coding (`ac_enc.cu`).
    pub fn encode_channel(&self, cdf: &[u16], symbols: &[u8]) -> Vec<u8> {
        struct BitWriter {
            out_bytes: Vec<u8>,
            cur_byte: u8,
            bit_count: u8,
        }
        impl BitWriter {
            fn push_bit(&mut self, bit: u8) {
                self.cur_byte = (self.cur_byte << 1) | (bit & 1);
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.out_bytes.push(self.cur_byte);
                    self.cur_byte = 0;
                    self.bit_count = 0;
                }
            }
            fn emit_with_pending(&mut self, bit: u8, pending: &mut u64) {
                self.push_bit(bit);
                let inv = 1 - (bit & 1);
                while *pending > 0 {
                    self.push_bit(inv);
                    *pending -= 1;
                }
            }
        }

        let max_sym = cdf.len().saturating_sub(2);
        let mut low: u32 = 0;
        let mut high: u32 = 0xFFFF_FFFF;
        let mut pending_bits: u64 = 0;
        let mut writer = BitWriter {
            out_bytes: Vec::new(),
            cur_byte: 0,
            bit_count: 0,
        };

        for &raw_sym in symbols {
            let sym = (raw_sym as usize).min(max_sym);
            let span = (high as u64) - (low as u64) + 1;
            let c_low = cdf[sym] as u64;
            let c_high = if sym == max_sym {
                0x10000u64
            } else {
                cdf[sym + 1] as u64
            };
            high = low
                .wrapping_sub(1)
                .wrapping_add(((span * c_high) >> Self::PRECISION_BITS) as u32);
            low = low.wrapping_add(((span * c_low) >> Self::PRECISION_BITS) as u32);

            loop {
                if high < 0x8000_0000 {
                    writer.emit_with_pending(0, &mut pending_bits);
                    low <<= 1;
                    high = (high << 1) | 1;
                } else if low >= 0x8000_0000 {
                    writer.emit_with_pending(1, &mut pending_bits);
                    low <<= 1;
                    high = (high << 1) | 1;
                } else if low >= 0x4000_0000 && high < 0xC000_0000 {
                    pending_bits += 1;
                    low = (low << 1) & 0x7FFF_FFFF;
                    high = (high << 1) | 0x8000_0001;
                } else {
                    break;
                }
            }
        }

        pending_bits += 1;
        if low < 0x4000_0000 {
            writer.emit_with_pending(0, &mut pending_bits);
        } else {
            writer.emit_with_pending(1, &mut pending_bits);
        }
        if writer.bit_count > 0 {
            writer.cur_byte <<= 8 - writer.bit_count;
            writer.out_bytes.push(writer.cur_byte);
        }
        writer.out_bytes
    }

    /// Decodes `num_tokens` quantized symbols from a CacheGen arithmetic-coded `bitstream` using
    /// binary search over the 16-bit `cdf` (`ac_dec.cu`).
    pub fn decode_channel(&self, cdf: &[u16], bitstream: &[u8], num_tokens: usize) -> Vec<u8> {
        let max_sym = cdf.len().saturating_sub(2);
        let mut low: u32 = 0;
        let mut high: u32 = 0xFFFF_FFFF;
        let mut value: u32 = 0;
        let mut bit_cursor: usize = 0;

        let read_bit = |cursor: &mut usize| -> u32 {
            let byte_idx = *cursor / 8;
            let bit_idx = 7 - (*cursor % 8);
            *cursor += 1;
            if byte_idx < bitstream.len() {
                ((bitstream[byte_idx] >> bit_idx) & 1) as u32
            } else {
                0
            }
        };

        for _ in 0..32 {
            value = (value << 1) | read_bit(&mut bit_cursor);
        }

        let mut decoded = Vec::with_capacity(num_tokens);
        for tok_idx in 0..num_tokens {
            let span = (high as u64) - (low as u64) + 1;
            let count = ((((value as u64) - (low as u64) + 1) * 0x10000u64 - 1) / span) as u16;

            // Binary search over `cdf[0..=max_sym]` (`binsearch` in `ac_dec.cu`)
            let mut left = 0usize;
            let mut right = max_sym + 1;
            while left + 1 < right {
                let mid = (left + right) / 2;
                let v = cdf[mid];
                if v < count {
                    left = mid;
                } else if v > count {
                    right = mid;
                } else {
                    left = mid;
                    break;
                }
            }
            let sym = left;
            decoded.push(sym as u8);
            if tok_idx + 1 == num_tokens {
                break;
            }

            let c_low = cdf[sym] as u64;
            let c_high = if sym == max_sym {
                0x10000u64
            } else {
                cdf[sym + 1] as u64
            };
            high = low
                .wrapping_sub(1)
                .wrapping_add(((span * c_high) >> Self::PRECISION_BITS) as u32);
            low = low.wrapping_add(((span * c_low) >> Self::PRECISION_BITS) as u32);

            loop {
                if high < 0x8000_0000 || low >= 0x8000_0000 {
                    low <<= 1;
                    high = (high << 1) | 1;
                    value = (value << 1) | read_bit(&mut bit_cursor);
                } else if low >= 0x4000_0000 && high < 0xC000_0000 {
                    low = (low << 1) & 0x7FFF_FFFF;
                    high = (high << 1) | 0x8000_0001;
                    value = (value & 0x8000_0000)
                        | ((value << 1) & 0x7FFF_FFFE)
                        | read_bit(&mut bit_cursor);
                } else {
                    break;
                }
            }
        }
        decoded
    }
}

/// Per-layer-group invariants for CacheBlend non-block-aligned KV retrieval
/// (inspired by LMCache `csrc/cuda/blend_kernels.cuh` `CBGroupSpec`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBGroupSpec {
    pub group_idx: usize,
    pub num_layers: usize,
    pub slot_tokens: usize,
    pub num_temp_slots: usize,
    pub bytes_per_token_per_layer: usize,
    pub rope_enabled: bool,
}

/// Descriptor for a single non-block-aligned RAG chunk hit in host KV memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBChunkMatch {
    pub host_offset: usize,
    pub num_tokens: usize,
    pub old_start_pos: i64,
    pub cur_start_pos: i64,
    pub slot_mapping_offset: usize,
}

/// K-only in-place delta re-RoPE launch descriptor for one temporary staging slot
/// (inspired by LMCache `csrc/cuda/blend_kernels.cuh` `CBRopeVar`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBRopeVar {
    pub group_idx: usize,
    pub slot_idx: usize,
    pub old_st: i64,
    pub cur_st: i64,
}

/// Per-token scatter launch descriptor writing `n_tok` tokens from temporary staging slot
/// `slot_idx` into paged KV slots starting at `slot_mapping_offset`
/// (inspired by LMCache `csrc/cuda/blend_kernels.cuh` `CBScatterVar`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBScatterVar {
    pub group_idx: usize,
    pub slot_idx: usize,
    pub slot_mapping_offset: usize,
    pub n_tok: usize,
}

/// One batched wave of CacheBlend temporary staging slots: H2D staging copies, then K-only
/// delta re-RoPE launches, then per-token paged KV scatters
/// (inspired by LMCache `csrc/cuda/blend_kernels.cuh` `CBRetrieveStep`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBRetrieveStep {
    pub step_idx: usize,
    pub staging_copies: Vec<KvDmaCopyEntry>,
    pub ropes: Vec<CBRopeVar>,
    pub scatters: Vec<CBScatterVar>,
}

/// Builds a batched plan-then-execute CacheBlend retrieve schedule (`Vec<CBRetrieveStep>`)
/// over `groups` and `chunks`, partitioning chunks into waves sized by the available temporary
/// GPU staging slots (`num_temp_slots`) and splitting H2D staging copies at `host_buffer_alignment`
/// pin-chunk boundaries (inspired by LMCache `blend_kernels.cuh` `execute_cb_retrieve_plan`).
pub fn plan_cacheblend_retrieve(
    groups: &[CBGroupSpec],
    chunks: &[CBChunkMatch],
    host_buffer_alignment: usize,
) -> Vec<CBRetrieveStep> {
    if groups.is_empty() || chunks.is_empty() {
        return Vec::new();
    }
    let wave_cap = groups
        .iter()
        .map(|g| g.num_temp_slots.max(1))
        .min()
        .unwrap_or(1);
    let align = host_buffer_alignment.max(1);
    let mut steps = Vec::new();

    for (step_idx, wave) in chunks.chunks(wave_cap).enumerate() {
        let mut staging_copies = Vec::new();
        let mut ropes = Vec::new();
        let mut scatters = Vec::new();

        for (slot_idx, chunk) in wave.iter().enumerate() {
            let n_tok = chunk.num_tokens;
            if n_tok == 0 {
                continue;
            }
            let mut group_host_cursor = chunk.host_offset;
            for grp in groups {
                let total_bytes = grp
                    .num_layers
                    .saturating_mul(n_tok)
                    .saturating_mul(grp.bytes_per_token_per_layer);
                let slot_cap_bytes = grp
                    .num_layers
                    .saturating_mul(grp.slot_tokens.max(n_tok))
                    .saturating_mul(grp.bytes_per_token_per_layer);
                let dev_base = slot_idx.saturating_mul(slot_cap_bytes);

                let mut rem = total_bytes;
                let mut h_cur = group_host_cursor;
                let mut d_cur = dev_base;
                while rem > 0 {
                    let next_boundary = ((h_cur / align) + 1) * align;
                    let step_bytes = rem.min(next_boundary - h_cur);
                    staging_copies.push(KvDmaCopyEntry {
                        block_id: slot_idx,
                        host_offset: h_cur as u64,
                        device_offset: d_cur as u64,
                        width_bytes: step_bytes as u64,
                        height_layers: grp.num_layers,
                    });
                    h_cur += step_bytes;
                    d_cur += step_bytes;
                    rem -= step_bytes;
                }
                group_host_cursor += total_bytes;

                if grp.rope_enabled && chunk.old_start_pos != chunk.cur_start_pos {
                    ropes.push(CBRopeVar {
                        group_idx: grp.group_idx,
                        slot_idx,
                        old_st: chunk.old_start_pos,
                        cur_st: chunk.cur_start_pos,
                    });
                }
                scatters.push(CBScatterVar {
                    group_idx: grp.group_idx,
                    slot_idx,
                    slot_mapping_offset: chunk.slot_mapping_offset,
                    n_tok,
                });
            }
        }

        steps.push(CBRetrieveStep {
            step_idx,
            staging_copies,
            ropes,
            scatters,
        });
    }
    steps
}

fn is_narrow_float_dtype(dt: &DType) -> bool {
    matches!(
        dt,
        DType::F16 | DType::BF16 | DType::F8E4M3FN | DType::F8E5M2
    )
}

/// Ensures accumulating operations (`dot_general`, `convolution`, `reduce`, and fused GEMM
/// composites) with narrow floating-point operands or outputs (`f16`, `bf16`, `f8E4M3FN`,
/// `f8E5M2`) accumulate in `f32` before downcasting on epilogue store (inspired by Modular
/// `AllocateAccumulatingBuffers.cpp`).
///
/// Stamps `castforge.accumulator_dtype = "f32"` (and `castforge.epilogue_downcast = "<dtype>"`
/// when the output tensor is narrow) and returns the number of accumulating ops promoted.
pub fn promote_narrow_accumulators(module: &mut Module) -> usize {
    let mut promoted = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.attributes.contains_key("castforge.accumulator_dtype") {
                continue;
            }
            let is_accum_op = op.is_op("stablehlo", "dot_general")
                || op.is_op("stablehlo", "convolution")
                || op.is_op("stablehlo", "reduce")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| {
                            matches!(
                                n,
                                "castforge.fused_matmul_epilogue"
                                    | "castforge.fused_dequant_matmul"
                                    | "castforge.fused_qmatmul"
                            )
                        }));
            if !is_accum_op {
                continue;
            }

            let has_narrow_input = op.operands.iter().any(|&v| {
                module
                    .values
                    .get(v)
                    .and_then(|val| val.ty.as_tensor())
                    .is_some_and(|t| is_narrow_float_dtype(&t.dtype))
            });
            let narrow_output_dt = op.results.first().and_then(|&r| {
                module
                    .values
                    .get(r)
                    .and_then(|val| val.ty.as_tensor())
                    .and_then(|t| {
                        if is_narrow_float_dtype(&t.dtype) {
                            Some(t.dtype.to_string())
                        } else {
                            None
                        }
                    })
            });

            if !has_narrow_input && narrow_output_dt.is_none() {
                continue;
            }

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.accumulator_dtype".to_string(),
                    Attribute::String("f32".to_string()),
                );
                if let Some(down_dt) = narrow_output_dt {
                    op_mut.attributes.insert(
                        "castforge.epilogue_downcast".to_string(),
                        Attribute::String(down_dt),
                    );
                }
                promoted += 1;
            }
        }
    }
    promoted
}

/// Adaptive Replacement Cache (ARC) KV eviction policy with `T1` (recent), `T2` (frequent),
/// and `B1`/`B2` ghost-history feedback (inspired by LMCache
/// `lmcache/v1/distributed/eviction_policy/arc.py`).
#[derive(Debug, Clone, Default)]
pub struct ArcKvEvictionPolicy {
    t1: IndexMap<u64, ()>,
    t2: IndexMap<u64, ()>,
    b1: IndexMap<u64, ()>,
    b2: IndexMap<u64, ()>,
    target_t1_size: f64,
    capacity: usize,
}

impl ArcKvEvictionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resident_len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    pub fn t1_len(&self) -> usize {
        self.t1.len()
    }

    pub fn t2_len(&self) -> usize {
        self.t2.len()
    }

    pub fn b1_len(&self) -> usize {
        self.b1.len()
    }

    pub fn b2_len(&self) -> usize {
        self.b2.len()
    }

    pub fn target_t1_size(&self) -> f64 {
        self.target_t1_size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn adaptation_delta(hit_len: usize, other_len: usize) -> f64 {
        (other_len as f64 / hit_len.max(1) as f64).max(1.0)
    }

    fn trim_ghost_lists(&mut self) {
        if self.capacity == 0 {
            return;
        }
        while self.b1.len() > self.capacity {
            self.b1.shift_remove_index(0);
        }
        while self.b2.len() > self.capacity {
            self.b2.shift_remove_index(0);
        }
        while self.b1.len() + self.b2.len() > 2 * self.capacity {
            if !self.b2.is_empty() {
                self.b2.shift_remove_index(0);
            } else if !self.b1.is_empty() {
                self.b1.shift_remove_index(0);
            } else {
                break;
            }
        }
    }

    /// Admits newly created KV block keys in reverse request order (so later suffix chunks stay
    /// closer to the LRU head) and applies `B1`/`B2` ghost-history feedback to `target_t1_size`.
    pub fn on_keys_created(&mut self, keys: &[u64]) {
        for &key in keys.iter().rev() {
            if self.t1.shift_remove(&key).is_some() || self.t2.shift_remove(&key).is_some() {
                self.t2.insert(key, ());
            } else if self.b1.contains_key(&key) {
                let delta = Self::adaptation_delta(self.b1.len(), self.b2.len());
                self.target_t1_size = (self.target_t1_size + delta).min(self.capacity as f64);
                self.b1.shift_remove(&key);
                self.t2.insert(key, ());
            } else if self.b2.contains_key(&key) {
                let delta = Self::adaptation_delta(self.b2.len(), self.b1.len());
                self.target_t1_size = (self.target_t1_size - delta).max(0.0);
                self.b2.shift_remove(&key);
                self.t2.insert(key, ());
            } else {
                self.t1.insert(key, ());
            }
            self.capacity = self.capacity.max(self.resident_len());
            self.trim_ghost_lists();
        }
    }

    /// Promotes touched resident keys from `T1` to the MRU tail of `T2`, or refreshes their
    /// recency in `T2`.
    pub fn on_keys_touched(&mut self, keys: &[u64]) {
        for &key in keys.iter().rev() {
            if self.t1.shift_remove(&key).is_some() || self.t2.shift_remove(&key).is_some() {
                self.t2.insert(key, ());
            }
        }
    }

    /// Selects and evicts `expected_ratio` of resident keys into ghost history (`B1` or `B2`)
    /// using LMCache's request-independent adaptive boundary rule (`|T1| >= int(p)`).
    pub fn select_and_evict_victims(&mut self, expected_ratio: f64) -> Vec<u64> {
        let resident = self.resident_len();
        if resident == 0 {
            return Vec::new();
        }
        let ratio = expected_ratio.clamp(0.0, 1.0);
        if ratio <= 0.0 {
            return Vec::new();
        }
        let target_count = ((resident as f64 * ratio) as usize).max(1);
        self.capacity = self.capacity.max(resident);
        self.target_t1_size = self.target_t1_size.min(self.capacity as f64);

        let mut victims = Vec::with_capacity(target_count);
        while victims.len() < target_count {
            let prefer_t1 = self.t1.len() >= (self.target_t1_size as usize);
            if prefer_t1 && let Some((k, _)) = self.t1.shift_remove_index(0) {
                self.b1.insert(k, ());
                victims.push(k);
                continue;
            }
            if let Some((k, _)) = self.t2.shift_remove_index(0) {
                self.b2.insert(k, ());
                victims.push(k);
                continue;
            }
            if let Some((k, _)) = self.t1.shift_remove_index(0) {
                self.b1.insert(k, ());
                victims.push(k);
                continue;
            }
            break;
        }
        self.trim_ghost_lists();
        victims
    }
}

/// Multi-tenant `cache_salt` byte-budget quota registry supporting both strict allowlist
/// (`get_limit_bytes` -> `0` for unknown tenants) and coordinator bootstrap-exempt
/// (`effective_limit_bytes` -> `None` until armed) resolution lenses (inspired by LMCache
/// `lmcache/v1/distributed/quota_manager.py`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvSaltQuotaRegistry {
    limits: IndexMap<String, u64>,
    default_limit_bytes: Option<u64>,
}

impl KvSaltQuotaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_quota(&mut self, cache_salt: impl Into<String>, limit_bytes: u64) {
        self.limits.insert(cache_salt.into(), limit_bytes);
    }

    pub fn delete_quota(&mut self, cache_salt: &str) -> bool {
        self.limits.shift_remove(cache_salt).is_some()
    }

    pub fn has_quota(&self, cache_salt: &str) -> bool {
        self.limits.contains_key(cache_salt)
    }

    /// Strict allowlist lens: unregistered salts resolve to `0` bytes so unknown tenant data is
    /// immediately eligible for eviction on local storage workers.
    pub fn get_limit_bytes(&self, cache_salt: &str) -> u64 {
        self.limits.get(cache_salt).copied().unwrap_or(0)
    }

    pub fn set_default_limit_bytes(&mut self, limit_bytes: Option<u64>) {
        self.default_limit_bytes = limit_bytes;
    }

    pub fn get_default_limit_bytes(&self) -> Option<u64> {
        self.default_limit_bytes
    }

    /// Coordinator bootstrap-safe lens: returns `Some(limit)` for registered salts, or
    /// `default_limit_bytes` (`None` = exempt from eviction until the quota controller syncs
    /// tenant tables and arms enforcement).
    pub fn effective_limit_bytes(&self, cache_salt: &str) -> Option<u64> {
        self.limits
            .get(cache_salt)
            .copied()
            .or(self.default_limit_bytes)
    }

    /// Computes how many bytes `used_bytes` exceeds `cache_salt`'s quota under either the
    /// strict allowlist lens (`allowlist_lens = true`) or the bootstrap-safe coordinator lens
    /// (`allowlist_lens = false`).
    pub fn over_quota_bytes(&self, cache_salt: &str, used_bytes: u64, allowlist_lens: bool) -> u64 {
        let limit = if allowlist_lens {
            Some(self.get_limit_bytes(cache_salt))
        } else {
            self.effective_limit_bytes(cache_salt)
        };
        match limit {
            None => 0,
            Some(lim) => used_bytes.saturating_sub(lim),
        }
    }
}

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

fn extract_int_list_attr(op: &Operation, key: &str) -> Option<Vec<i64>> {
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

/// Default consecutive missed coordinator polls before a peer P2P adapter is removed (`3`)
/// (matching LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py` `_MAX_MISSES = 3`).
pub const P2P_DEFAULT_MAX_MISSES: u32 = 3;

/// Peer KV instance connection metadata parsed from coordinator discovery
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py` `_PeerInstance`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2PPeerSpec {
    pub instance_id: String,
    pub ip: String,
    pub p2p_advertised_url: String,
    pub mq_port: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct P2PPeerAdapterEntry {
    adapter_id: u64,
    spec: P2PPeerSpec,
    consecutive_misses: u32,
}

/// P2P peer discovery and adapter lifecycle table with consecutive-miss tolerance
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py` `P2PController`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P2PPeerDiscoveryTable {
    self_instance_id: String,
    max_misses: u32,
    next_adapter_id: u64,
    peers: IndexMap<String, P2PPeerAdapterEntry>,
}

impl P2PPeerDiscoveryTable {
    /// Creates a new peer discovery table for `self_instance_id` with `max_misses` tolerance.
    pub fn new(self_instance_id: impl Into<String>, max_misses: u32) -> Self {
        Self {
            self_instance_id: self_instance_id.into(),
            max_misses: max_misses.max(1),
            next_adapter_id: 1,
            peers: IndexMap::new(),
        }
    }

    /// Number of currently active peer adapters.
    pub fn active_peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Reconciles the local peer adapter table against `reported_peers` from the coordinator,
    /// returning `(created_adapter_ids, removed_adapter_ids)`.
    ///
    /// - Ignores `self_instance_id`.
    /// - Resets `consecutive_misses = 0` for present peers with unchanged endpoints.
    /// - Recreates the adapter if a peer's `(ip, p2p_advertised_url, mq_port)` changed.
    /// - Increments `consecutive_misses` for absent peers and evicts them once
    ///   `consecutive_misses >= self.max_misses`.
    pub fn reconcile_peers(&mut self, reported_peers: &[P2PPeerSpec]) -> (Vec<u64>, Vec<u64>) {
        let mut created = Vec::new();
        let mut removed = Vec::new();
        let mut seen_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for peer in reported_peers {
            if peer.instance_id == self.self_instance_id {
                continue;
            }
            seen_ids.insert(peer.instance_id.as_str());
            if let Some(existing) = self.peers.get_mut(&peer.instance_id) {
                if existing.spec == *peer {
                    existing.consecutive_misses = 0;
                } else {
                    removed.push(existing.adapter_id);
                    let new_id = self.next_adapter_id;
                    self.next_adapter_id += 1;
                    *existing = P2PPeerAdapterEntry {
                        adapter_id: new_id,
                        spec: peer.clone(),
                        consecutive_misses: 0,
                    };
                    created.push(new_id);
                }
            } else {
                let new_id = self.next_adapter_id;
                self.next_adapter_id += 1;
                self.peers.insert(
                    peer.instance_id.clone(),
                    P2PPeerAdapterEntry {
                        adapter_id: new_id,
                        spec: peer.clone(),
                        consecutive_misses: 0,
                    },
                );
                created.push(new_id);
            }
        }

        let mut to_evict = Vec::new();
        for (inst_id, entry) in &mut self.peers {
            if !seen_ids.contains(inst_id.as_str()) {
                entry.consecutive_misses += 1;
                if entry.consecutive_misses >= self.max_misses {
                    removed.push(entry.adapter_id);
                    to_evict.push(inst_id.clone());
                }
            }
        }
        for inst_id in to_evict {
            self.peers.shift_remove(&inst_id);
        }

        (created, removed)
    }
}

/// Buckets a flat RPC `keys` list into one `(object_group_id, group_keys)` row per object group in
/// first-seen order, returning an empty `Vec` when `keys` is empty, when object groups receive
/// different numbers of keys, or when any key's `object_group_id` is not in `registered_groups`
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/p2p_controller.py`
/// `_group_keys_by_object_group`).
pub fn group_p2p_lookup_keys(
    keys: &[KvStorageKeySpec],
    registered_groups: &[u32],
) -> Vec<(u32, Vec<KvStorageKeySpec>)> {
    if keys.is_empty() {
        return Vec::new();
    }
    let reg_set: std::collections::HashSet<u32> = registered_groups.iter().copied().collect();
    let mut by_group: IndexMap<u32, Vec<KvStorageKeySpec>> = IndexMap::new();
    for key in keys {
        if !reg_set.contains(&key.object_group_id) {
            return Vec::new();
        }
        by_group
            .entry(key.object_group_id)
            .or_default()
            .push(key.clone());
    }
    let mut lengths = by_group.values().map(Vec::len);
    let Some(first_len) = lengths.next() else {
        return Vec::new();
    };
    if lengths.any(|l| l != first_len) {
        return Vec::new();
    }
    by_group.into_iter().collect()
}

/// Encodes a [`KvStorageKeySpec`] into a 4-hex-character hash-prefixed wide-column row key
/// (`"{hash_prefix:04x}@{model}[@lg{lg}]@{rank:08x}@{group:x}@{hash:016x}[@{salt}][@layer_{id}]"`)
/// so tablet servers distribute rows uniformly across keyspace prefixes
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/bigtable_key_encoder.py`
/// `BigtableL2KeyEncoder`).
pub fn encode_wide_column_row_key(
    key: &KvStorageKeySpec,
    layer_group_size: usize,
    layer_id: Option<u32>,
) -> String {
    let hash_hex = format!("{:016x}", key.chunk_hash);
    let hash_prefix = &hash_hex[..4];
    let model_with_lg = if layer_group_size > 0 {
        format!("{}@lg{layer_group_size}", key.model_name)
    } else {
        key.model_name.clone()
    };
    let mut row_key = format!(
        "{hash_prefix}@{model_with_lg}@{:08x}@{:x}@{hash_hex}",
        key.kv_rank, key.object_group_id
    );
    if !key.cache_salt.is_empty() {
        row_key.push('@');
        row_key.push_str(&key.cache_salt);
    }
    if let Some(lid) = layer_id {
        row_key.push_str(&format!("@layer_{lid}"));
    }
    row_key
}

/// Fibonacci-hashing polynomial base (`0x9E3779B97F4A7C15`) used for 64-bit chunk and rolling
/// token window fingerprints (inspired by LMCache
/// `lmcache/v1/multiprocess/modules/blend/matcher.py` `BlendTokenRangeMatcher._BASE`).
pub const BLEND_HASH_BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Computes 64-bit polynomial hashes (modulo $2^{64}$ via wrapping arithmetic) over
/// non-overlapping windows of length `chunk_size`, ignoring trailing tokens that do not fill a
/// complete chunk (inspired by LMCache `lmcache/v1/multiprocess/token_hasher.py`
/// `chunk_hash_windows_numba`).
pub fn chunk_poly_hash_windows(token_ids: &[u64], chunk_size: usize, base: u64) -> Vec<u64> {
    if chunk_size == 0 || token_ids.len() < chunk_size {
        return Vec::new();
    }
    let num_windows = token_ids.len() / chunk_size;
    let mut out = Vec::with_capacity(num_windows);
    for w in 0..num_windows {
        let start = w * chunk_size;
        let mut h = 0u64;
        for &tok in &token_ids[start..start + chunk_size] {
            h = h.wrapping_mul(base).wrapping_add(tok);
        }
        out.push(h);
    }
    out
}

/// Computes $O(1)$ sliding-window 64-bit polynomial hashes (modulo $2^{64}$ via wrapping
/// arithmetic) across all `token_ids.len() - window_size + 1` positions (inspired by LMCache
/// `lmcache/v1/multiprocess/token_hasher.py` `rolling_hash_windows_numba`).
pub fn rolling_poly_hash_windows(token_ids: &[u64], window_size: usize, base: u64) -> Vec<u64> {
    if window_size == 0 || token_ids.len() < window_size {
        return Vec::new();
    }
    let num_pos = token_ids.len() - window_size + 1;
    let mut out = Vec::with_capacity(num_pos);
    let mut power = 1u64;
    for _ in 0..window_size.saturating_sub(1) {
        power = power.wrapping_mul(base);
    }
    let mut h = 0u64;
    for &tok in &token_ids[..window_size] {
        h = h.wrapping_mul(base).wrapping_add(tok);
    }
    out.push(h);
    for i in window_size..token_ids.len() {
        let old = token_ids[i - window_size];
        let new = token_ids[i];
        h = h.wrapping_sub(old.wrapping_mul(power));
        h = h.wrapping_mul(base).wrapping_add(new);
        out.push(h);
    }
    out
}

/// A matched sub-sequence chunk span returned by [`BlendTokenRangeMatcher::match_sub_sequence`]
/// (inspired by LMCache `lmcache/v1/multiprocess/custom_types.py` `CBMatchResult`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlendMatchSpan {
    /// Start token index in the originally indexed sequence (`old_st`).
    pub old_st: usize,
    /// Exclusive end token index in the originally indexed sequence (`old_ed`).
    pub old_ed: usize,
    /// Start token index in the query sequence (`cur_st`).
    pub cur_st: usize,
    /// Exclusive end token index in the query sequence (`cur_ed`).
    pub cur_ed: usize,
    /// Content hash identifier of the matched chunk.
    pub chunk_hash: u64,
}

/// Token-level sliding-window fingerprint matcher over a power-of-two direct-address table of
/// 64-bit chunk polynomial hashes with full-hash collision rejection and content deduplication
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/matcher.py`
/// `BlendTokenRangeMatcher`).
#[derive(Debug, Clone)]
pub struct BlendTokenRangeMatcher {
    chunk_size: usize,
    dedup_content: bool,
    table_size: usize,
    mask: u64,
    table_id: Vec<i64>,
    chunk_token_hash: Vec<Option<u64>>,
    chunk_poly_hash: Vec<u64>,
    token_hash_to_start: std::collections::HashMap<u64, usize>,
    compact_id_to_slot: Vec<i64>,
    token_hash_to_compact_id: std::collections::HashMap<u64, usize>,
}

impl BlendTokenRangeMatcher {
    /// Creates a new matcher with `chunk_size` tokens per chunk and a direct-address table of
    /// size `1 << table_bits` (clamped to `[4, 20]`).
    #[must_use]
    pub fn new(chunk_size: usize, table_bits: u32, dedup_content: bool) -> Self {
        let bits = table_bits.clamp(4, 20);
        let table_size = 1usize << bits;
        Self {
            chunk_size: chunk_size.max(1),
            dedup_content,
            table_size,
            mask: (table_size - 1) as u64,
            table_id: vec![-1; table_size],
            chunk_token_hash: Vec::new(),
            chunk_poly_hash: Vec::new(),
            token_hash_to_start: std::collections::HashMap::new(),
            compact_id_to_slot: vec![-1; table_size],
            token_hash_to_compact_id: std::collections::HashMap::new(),
        }
    }

    fn poly_hash_registered(&self, poly_hash: u64) -> bool {
        let slot = (poly_hash & self.mask) as usize;
        let cid = self.table_id[slot];
        if cid < 0 {
            return false;
        }
        let cid_usize = cid as usize;
        self.chunk_poly_hash.get(cid_usize) == Some(&poly_hash)
            && self
                .chunk_token_hash
                .get(cid_usize)
                .and_then(|h| *h)
                .is_some()
    }

    /// Indexes non-overlapping chunks of `token_ids` with their corresponding `token_hashes` and
    /// `position_offset`, returning the number of newly indexed chunks.
    pub fn on_new_token_hashes(
        &mut self,
        token_ids: &[u64],
        token_hashes: &[u64],
        position_offset: usize,
    ) -> usize {
        let chunk_hashes = chunk_poly_hash_windows(token_ids, self.chunk_size, BLEND_HASH_BASE);
        let n = chunk_hashes.len().min(token_hashes.len());
        if n == 0 {
            return 0;
        }
        let mut new_idxs = Vec::new();
        let mut batch_poly = std::collections::HashSet::new();
        for i in 0..n {
            if self.token_hash_to_compact_id.contains_key(&token_hashes[i]) {
                continue;
            }
            if self.dedup_content {
                let poly_hash = chunk_hashes[i];
                if batch_poly.contains(&poly_hash) || self.poly_hash_registered(poly_hash) {
                    continue;
                }
                batch_poly.insert(poly_hash);
            }
            new_idxs.push(i);
        }
        if new_idxs.is_empty() {
            return 0;
        }
        let base_id = self.chunk_token_hash.len();
        if base_id + new_idxs.len() > self.table_size {
            return 0;
        }
        for (k, &orig_i) in new_idxs.iter().enumerate() {
            let cid = base_id + k;
            let th = token_hashes[orig_i];
            let poly_hash = chunk_hashes[orig_i];
            let slot = (poly_hash & self.mask) as usize;
            self.table_id[slot] = cid as i64;
            self.chunk_token_hash.push(Some(th));
            self.chunk_poly_hash.push(poly_hash);
            self.token_hash_to_start
                .insert(th, position_offset + orig_i * self.chunk_size);
            self.compact_id_to_slot[cid] = slot as i64;
            self.token_hash_to_compact_id.insert(th, cid);
        }
        new_idxs.len()
    }

    /// Probes every sliding-window token position of `token_ids` against the direct-address table,
    /// rejecting bucket collisions via full 64-bit poly-hash verification and deduplicating by
    /// compact chunk ID.
    #[must_use]
    pub fn match_sub_sequence(&self, token_ids: &[u64]) -> Vec<BlendMatchSpan> {
        if token_ids.len() < self.chunk_size || self.chunk_token_hash.is_empty() {
            return Vec::new();
        }
        let rolling = rolling_poly_hash_windows(token_ids, self.chunk_size, BLEND_HASH_BASE);
        let mut seen_cids = std::collections::HashSet::new();
        let mut results = Vec::new();
        for (pos, &poly) in rolling.iter().enumerate() {
            let slot = (poly & self.mask) as usize;
            let cid_i64 = self.table_id[slot];
            if cid_i64 < 0 {
                continue;
            }
            let cid = cid_i64 as usize;
            if seen_cids.contains(&cid) {
                continue;
            }
            if self.chunk_poly_hash.get(cid) != Some(&poly) {
                continue;
            }
            let Some(Some(th)) = self.chunk_token_hash.get(cid).copied() else {
                continue;
            };
            let Some(&old_st) = self.token_hash_to_start.get(&th) else {
                continue;
            };
            seen_cids.insert(cid);
            results.push(BlendMatchSpan {
                old_st,
                old_ed: old_st + self.chunk_size,
                cur_st: pos,
                cur_ed: pos + self.chunk_size,
                chunk_hash: th,
            });
        }
        results
    }

    /// Evicts `token_hashes` from the direct-address table so subsequent probes ignore them.
    pub fn remove_chunks(&mut self, token_hashes: &[u64]) {
        for th in token_hashes {
            let Some(cid) = self.token_hash_to_compact_id.remove(th) else {
                continue;
            };
            let slot = self.compact_id_to_slot[cid];
            if slot >= 0 {
                self.table_id[slot as usize] = -1;
            }
            self.compact_id_to_slot[cid] = -1;
            self.chunk_token_hash[cid] = None;
            self.chunk_poly_hash[cid] = 0;
            self.token_hash_to_start.remove(th);
        }
    }
}

/// Computes the total unique token coverage across `spans` by sorting and merging overlapping
/// `[cur_st, cur_ed)` intervals so overlapping sliding-window matches are not double-counted
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/matcher.py`
/// `_unique_token_coverage`).
#[must_use]
pub fn unique_blend_token_coverage(spans: &[BlendMatchSpan]) -> usize {
    if spans.is_empty() {
        return 0;
    }
    let mut intervals: Vec<(usize, usize)> = spans.iter().map(|s| (s.cur_st, s.cur_ed)).collect();
    intervals.sort_unstable();
    let mut coverage = 0usize;
    let mut cur_end = 0usize;
    let mut first = true;
    for (st, ed) in intervals {
        if first || st >= cur_end {
            coverage += ed.saturating_sub(st);
            first = false;
        } else if ed > cur_end {
            coverage += ed - cur_end;
        }
        cur_end = cur_end.max(ed);
    }
    coverage
}

/// Shortest contiguous run of incoming prompt tokens copied as a piece during CacheBlend prompt
/// reordering; the last `BLEND_REORDER_LMIN` tokens of the prompt are also pinned at the tail
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/reorder.py` `LMIN`).
pub const BLEND_REORDER_LMIN: usize = 16;

/// Maximum short separator run copied between two matched pieces during CacheBlend prompt
/// reordering (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/reorder.py` `GLUE`).
pub const BLEND_REORDER_GLUE: usize = 3;

struct BlendReorderCopier<'a> {
    prompt: &'a [u32],
    window_hashes: Vec<u64>,
    sorted_indices: Vec<usize>,
}

impl<'a> BlendReorderCopier<'a> {
    fn new(prompt: &'a [u32]) -> Self {
        let m = prompt.len().saturating_sub(BLEND_REORDER_LMIN - 1);
        let mut window_hashes = Vec::with_capacity(m);
        for i in 0..m {
            let mut h = 0u64;
            for &tok in &prompt[i..i + BLEND_REORDER_LMIN] {
                h = h.wrapping_mul(BLEND_HASH_BASE).wrapping_add(u64::from(tok));
            }
            window_hashes.push(h);
        }
        let mut sorted_indices: Vec<usize> = (0..m).collect();
        sorted_indices.sort_by_key(|&idx| (window_hashes[idx], idx));
        Self {
            prompt,
            window_hashes,
            sorted_indices,
        }
    }

    fn starts(&self, cand: &[u32], k: usize) -> Vec<usize> {
        if k + BLEND_REORDER_LMIN > cand.len() || self.sorted_indices.is_empty() {
            return Vec::new();
        }
        let mut key = 0u64;
        for &tok in &cand[k..k + BLEND_REORDER_LMIN] {
            key = key
                .wrapping_mul(BLEND_HASH_BASE)
                .wrapping_add(u64::from(tok));
        }
        let lo = self
            .sorted_indices
            .partition_point(|&idx| self.window_hashes[idx] < key);
        let hi = self
            .sorted_indices
            .partition_point(|&idx| self.window_hashes[idx] <= key);
        let mut res = self.sorted_indices[lo..hi].to_vec();
        res.sort_unstable();
        res
    }

    fn longest(&self, cand: &[u32], k: usize, used: &[bool]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for i in self.starts(cand, k) {
            if used[i] {
                continue;
            }
            let max_n = (self.prompt.len() - i).min(cand.len() - k);
            if let Some((_, best_len)) = best
                && max_n <= best_len
            {
                break;
            }
            let mut m = max_n;
            for off in 0..max_n {
                if self.prompt[i + off] != cand[k + off] || used[i + off] {
                    m = off;
                    break;
                }
            }
            if m >= BLEND_REORDER_LMIN && best.is_none_or(|(_, bl)| m > bl) {
                best = Some((i, m));
            }
        }
        best
    }

    fn glue(
        &self,
        cand: &[u32],
        k: usize,
        used: &mut [bool],
        pieces: &[(usize, usize)],
    ) -> Option<((usize, usize), (usize, usize))> {
        let prompt_len = self.prompt.len();
        for gl in 1..=BLEND_REORDER_GLUE {
            if k + gl >= cand.len() {
                break;
            }
            if self.starts(cand, k + gl).is_empty() {
                continue;
            }
            let mut cands: Vec<usize> = (0..prompt_len)
                .filter(|&i| {
                    i + gl <= prompt_len
                        && !used[i..i + gl].iter().any(|&u| u)
                        && self.prompt[i..i + gl] == cand[k..k + gl]
                })
                .collect();
            cands.sort_by_key(|&i| {
                let adj = (i > 0 && used[i - 1]) || (i + gl < prompt_len && used[i + gl]);
                (!adj, i)
            });
            for i in cands {
                for u in &mut used[i..i + gl] {
                    *u = true;
                }
                let nxt = self.longest(cand, k + gl, used);
                for u in &mut used[i..i + gl] {
                    *u = false;
                }
                if let Some((nxt_st, nxt_len)) = nxt {
                    let max_end = pieces
                        .iter()
                        .map(|&(_, j)| j)
                        .chain(std::iter::once(nxt_st + nxt_len))
                        .max()
                        .unwrap_or(0);
                    if i + gl <= max_end {
                        return Some(((i, i + gl), (nxt_st, nxt_len)));
                    }
                }
            }
        }
        None
    }

    fn copy(&self, cand: &[u32], keep_prefix: usize) -> Vec<(usize, usize)> {
        let mut used = vec![false; self.prompt.len()];
        let kp = keep_prefix.min(self.prompt.len());
        for u in &mut used[..kp] {
            *u = true;
        }
        let tail_start = self.prompt.len().saturating_sub(BLEND_REORDER_LMIN);
        for u in &mut used[tail_start..] {
            *u = true;
        }
        let mut pieces = if kp > 0 { vec![(0, kp)] } else { Vec::new() };
        let mut k = kp;
        while k < cand.len() {
            let piece = if let Some(p) = self.longest(cand, k, &used) {
                p
            } else if let Some(((gi, gj), nxt)) = self.glue(cand, k, &mut used, &pieces) {
                pieces.push((gi, gj));
                for u in &mut used[gi..gj] {
                    *u = true;
                }
                k += gj - gi;
                nxt
            } else {
                break;
            };
            let (i, m) = piece;
            pieces.push((i, i + m));
            for u in &mut used[i..i + m] {
                *u = true;
            }
            k += m;
        }
        pieces
    }
}

/// Plans a permutation of incoming `prompt` token indices so that `prompt` begins with the
/// longest possible exact-prefix reconstruction of one of `candidates` (followed by all remaining
/// tokens of `prompt` in original order), keeping the first `keep_prefix` tokens in place and
/// pinning the final [`BLEND_REORDER_LMIN`] tokens at the tail (inspired by LMCache
/// `lmcache/v1/multiprocess/modules/blend/reorder.py` `plan`).
#[must_use]
pub fn plan_cacheblend_token_reorder(
    prompt: &[u32],
    candidates: &[Vec<u32>],
    chunk_size: usize,
    baseline_chunks: usize,
    keep_prefix: usize,
) -> Option<Vec<usize>> {
    if chunk_size == 0 || prompt.len() < BLEND_REORDER_LMIN || candidates.is_empty() {
        return None;
    }
    let copier = BlendReorderCopier::new(prompt);
    let kp = keep_prefix.min(prompt.len());
    let mut seen: std::collections::HashSet<&[u32]> = std::collections::HashSet::new();
    type ReorderCandidate<'a> = ((usize, usize, isize), Vec<(usize, usize)>, &'a [u32]);
    let mut best: Option<ReorderCandidate<'_>> = None;

    for cand in candidates {
        if cand.len() < kp || seen.contains(cand.as_slice()) || cand[..kp] != prompt[..kp] {
            continue;
        }
        seen.insert(cand.as_slice());
        let pieces = copier.copy(cand, kp);
        let k: usize = pieces.iter().map(|&(i, j)| j - i).sum();
        let score = (k / chunk_size, k, -(pieces.len() as isize));
        if best.as_ref().is_none_or(|(b_score, _, _)| score > *b_score) {
            best = Some((score, pieces, cand.as_slice()));
        }
    }

    let ((whole_chunks, k, _), pieces, cand) = best?;
    if whole_chunks <= baseline_chunks {
        return None;
    }

    let mut used = vec![false; prompt.len()];
    let mut perm = Vec::with_capacity(prompt.len());
    for &(i, j) in &pieces {
        perm.extend(i..j);
        for u in &mut used[i..j] {
            *u = true;
        }
    }
    if k < cand.len() {
        let max_j = pieces.iter().map(|&(_, j)| j).max().unwrap_or(0);
        for i in 0..max_j.min(prompt.len()) {
            let adj = (i > 0 && used[i - 1]) || (i + 1 < prompt.len() && used[i + 1]);
            if !used[i] && prompt[i] == cand[k] && adj {
                perm.push(i);
                used[i] = true;
                break;
            }
        }
    }
    for (i, &is_used) in used.iter().enumerate() {
        if !is_used {
            perm.push(i);
        }
    }
    let is_identity = perm.iter().enumerate().all(|(idx, &p)| idx == p);
    if is_identity { None } else { Some(perm) }
}

/// Storage tier reported by a cache event batch (inspired by LMCache
/// `lmcache/v1/distributed/api.py` `Tier`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheEventTier {
    /// Volatile host/device L1 cache tier (tied to emitter incarnation).
    L1,
    /// Persistent L2 storage tier (survives emitter restarts).
    L2,
}

/// Metadata header of an emitter's cache event batch offered to [`CoordinatorEventGate`]
/// (inspired by LMCache `lmcache/v1/mp_coordinator/api.py` `CacheEventBatch`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEventBatchSpec {
    /// Unique identifier of the emitting worker instance.
    pub instance_id: String,
    /// Monotonically increasing process incarnation of the emitter.
    pub incarnation: u64,
    /// 1-based sequence number of the batch within `incarnation`.
    pub seq: u64,
    /// Cumulative number of events dropped by the emitter prior to transmission.
    pub dropped_events: u64,
    /// Cache tier affected by this batch.
    pub tier: CacheEventTier,
}

/// Admission outcome returned by [`CoordinatorEventGate::ingest`] (inspired by LMCache
/// `lmcache/v1/mp_coordinator/ingest/event_gate.py` `IngestResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventIngestOutcome {
    /// Batch was admitted and forwarded to downstream state views.
    Admitted,
    /// Batch was dropped because its `seq` was already processed for the current incarnation.
    Duplicate,
    /// Batch was dropped because its `incarnation` was older than the active or departed cursor.
    StaleIncarnation,
}

/// Per-emitter stream cursor state maintained by [`CoordinatorEventGate`] (inspired by LMCache
/// `lmcache/v1/mp_coordinator/ingest/event_gate.py` `_StreamCursor`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamCursorState {
    /// Emitter incarnation tracked by this cursor.
    pub incarnation: u64,
    /// Highest `seq` admitted from `incarnation`.
    pub last_seq: u64,
    /// Whether sequence gaps or emitter-side dropped events were detected on this stream.
    pub gap_detected: bool,
    /// Last reported cumulative `dropped_events` (`None` when a stream joined mid-flight at
    /// `seq > 1` before its first baseline report).
    pub last_dropped_events: Option<u64>,
    /// Number of skipped `seq` batches observed since the gate began tracking this stream.
    pub missing_batches: u64,
    /// Total emitter-dropped events observed on this stream.
    pub events_dropped: u64,
}

/// Ordering, sequence-deduplication, gap-detection, and incarnation-fencing gate for distributed
/// cache event streams (inspired by LMCache `lmcache/v1/mp_coordinator/ingest/event_gate.py`
/// `EventGate`).
#[derive(Debug, Clone, Default)]
pub struct CoordinatorEventGate {
    cursors: IndexMap<String, StreamCursorState>,
    departed: IndexMap<String, u64>,
    fenced_instances: Vec<String>,
    /// Total batches admitted across all emitters.
    pub batches_applied: u64,
    /// Total duplicate batches dropped across all emitters.
    pub batches_duplicate: u64,
    /// Total stale-incarnation batches dropped across all emitters.
    pub batches_stale: u64,
    /// Total skipped batch sequence numbers across tracked emitters.
    pub batches_missing: u64,
    /// Total emitter-dropped events reported across all emitters.
    pub events_dropped: u64,
}

impl CoordinatorEventGate {
    /// Creates an empty [`CoordinatorEventGate`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Offers `batch` to the gate, enforcing incarnation fencing, sequence deduplication, and
    /// loss/gap accounting.
    pub fn ingest(&mut self, batch: &CacheEventBatchSpec) -> EventIngestOutcome {
        if let Some(&dep_inc) = self.departed.get(&batch.instance_id) {
            if batch.incarnation > dep_inc {
                self.departed.swap_remove(&batch.instance_id);
            } else if batch.tier == CacheEventTier::L1 {
                self.batches_stale += 1;
                return EventIngestOutcome::StaleIncarnation;
            }
        }

        let tracked = self.cursors.contains_key(&batch.instance_id);
        if let Some(existing) = self.cursors.get(&batch.instance_id) {
            if batch.incarnation < existing.incarnation {
                self.batches_stale += 1;
                return EventIngestOutcome::StaleIncarnation;
            }
            if batch.incarnation > existing.incarnation {
                self.fenced_instances.push(batch.instance_id.clone());
                self.cursors.swap_remove(&batch.instance_id);
            } else if batch.seq <= existing.last_seq {
                self.batches_duplicate += 1;
                return EventIngestOutcome::Duplicate;
            }
        }

        let cursor = self
            .cursors
            .entry(batch.instance_id.clone())
            .or_insert_with(|| StreamCursorState {
                incarnation: batch.incarnation,
                last_seq: 0,
                gap_detected: false,
                last_dropped_events: if tracked || batch.seq == 1 {
                    Some(0)
                } else {
                    None
                },
                missing_batches: 0,
                events_dropped: 0,
            });

        if batch.seq > cursor.last_seq + 1 {
            let skipped = batch.seq - cursor.last_seq - 1;
            cursor.gap_detected = true;
            if tracked {
                cursor.missing_batches += skipped;
                self.batches_missing += skipped;
            }
        }
        if let Some(last_dropped) = cursor.last_dropped_events
            && batch.dropped_events > last_dropped
        {
            let dropped = batch.dropped_events - last_dropped;
            cursor.gap_detected = true;
            cursor.events_dropped += dropped;
            self.events_dropped += dropped;
        }
        cursor.last_dropped_events = Some(batch.dropped_events);
        cursor.last_seq = batch.seq;
        self.batches_applied += 1;
        EventIngestOutcome::Admitted
    }

    /// Fences `instance_id` and records its departed incarnation so stale L1 batches are rejected
    /// while L2 batches or newer incarnations can still proceed.
    pub fn drop_instance(&mut self, instance_id: &str) {
        self.fenced_instances.push(instance_id.to_string());
        if let Some(cursor) = self.cursors.swap_remove(instance_id) {
            self.departed
                .insert(instance_id.to_string(), cursor.incarnation);
        }
    }

    /// Clears the departed fence record for `instance_id` after explicit re-registration.
    pub fn readmit_instance(&mut self, instance_id: &str) {
        self.departed.swap_remove(instance_id);
    }

    /// Returns the active cursor for `instance_id`, if any.
    #[must_use]
    pub fn cursor(&self, instance_id: &str) -> Option<&StreamCursorState> {
        self.cursors.get(instance_id)
    }

    /// Returns the ordered list of instance IDs fenced due to incarnation restarts or drops.
    #[must_use]
    pub fn fenced_instances(&self) -> &[String] {
        &self.fenced_instances
    }
}

fn is_mergeable_matmul_op(op: &Operation) -> bool {
    op.is_op("stablehlo", "dot_general")
        || op.is_op("stablehlo", "dot")
        || op.is_op("linalg", "matmul")
        || op.is_op("mo", "matmul")
        || op.is_op("mo", "batch_matmul")
        || op.is_op("tosa", "matmul")
}

/// Merges sibling matrix multiplications (`matmul(A, B1) -> C1`, `matmul(A, B2) -> C2`, ...) that
/// share the same LHS activation tensor `A` into a single weight concatenation
/// (`stablehlo.concatenate`), a single combined matmul, and a multi-result `mo.split`, skipping
/// groups whose RHS weights are produced by a transpose or whose results all feed directly into
/// `func.return` (inspired by Modular `GraphCompiler/lib/MODialect/Patterns/Algebraic.cpp`
/// `MergeParallelMatmulPattern`).
pub fn merge_parallel_matmuls(module: &mut Module) -> usize {
    let mut merged_groups = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let Some(&reg_id) = module.ops[func_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let block_ops = module.blocks[block_id].ops.clone();

        let mut op_pos: std::collections::HashMap<OpId, usize> = std::collections::HashMap::new();
        let mut val_users: std::collections::HashMap<ValueId, Vec<OpId>> =
            std::collections::HashMap::new();
        for (idx, &op_id) in block_ops.iter().enumerate() {
            op_pos.insert(op_id, idx);
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    val_users.entry(v).or_default().push(op_id);
                }
            }
        }

        let mut by_lhs: IndexMap<ValueId, Vec<OpId>> = IndexMap::new();
        for &op_id in &block_ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !is_mergeable_matmul_op(op) || op.operands.len() != 2 || op.results.len() != 1 {
                continue;
            }
            by_lhs.entry(op.operands[0]).or_default().push(op_id);
        }

        let mut to_remove: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        let mut insertions: Vec<(OpId, Vec<OpId>)> = Vec::new();

        for (lhs_val, matmul_ids) in by_lhs {
            if matmul_ids.len() < 2 {
                continue;
            }

            // Skip if every matmul user in this group only feeds directly into `func.return`
            // or `mo.output` (outputs would force the split to materialize, which is slower).
            let all_feed_only_outputs = matmul_ids.iter().all(|&m_id| {
                let Some(m_op) = module.ops.get(m_id) else {
                    return false;
                };
                let res_val = m_op.results[0];
                let Some(users) = val_users.get(&res_val) else {
                    return false;
                };
                !users.is_empty()
                    && users.iter().all(|&u_id| {
                        module
                            .ops
                            .get(u_id)
                            .is_some_and(|u| u.is_op("func", "return") || u.is_op("mo", "output"))
                    })
            });
            if all_feed_only_outputs {
                continue;
            }

            let first_id = matmul_ids[0];
            let first_pos = op_pos[&first_id];
            let first_op = module.ops[first_id].clone();
            let Some(first_res_tt) = module
                .values
                .get(first_op.results[0])
                .and_then(|v| v.ty.as_tensor())
                .cloned()
            else {
                continue;
            };
            let Some(first_rhs_tt) = module
                .values
                .get(first_op.operands[1])
                .and_then(|v| v.ty.as_tensor())
                .cloned()
            else {
                continue;
            };
            let res_rank = first_res_tt.shape.len();
            let rhs_rank = first_rhs_tt.shape.len();
            if res_rank < 2 || rhs_rank < 2 {
                continue;
            }

            let mut valid = true;
            let mut weights_to_concat: smallvec::SmallVec<[ValueId; 4]> = smallvec::SmallVec::new();
            let mut columns_in_each: Vec<i64> = Vec::with_capacity(matmul_ids.len());
            let mut total_columns = 0i64;
            let mut source_locs: Vec<Location> = Vec::with_capacity(matmul_ids.len());
            let mut orig_results: smallvec::SmallVec<[ValueId; 2]> = smallvec::SmallVec::new();

            for &m_id in &matmul_ids {
                let Some(m_op) = module.ops.get(m_id) else {
                    valid = false;
                    break;
                };
                if m_op.name != first_op.name {
                    valid = false;
                    break;
                }
                let rhs_val = m_op.operands[1];
                let res_val = m_op.results[0];

                // Reject if RHS is defined by a transpose (`ForceMatmulTransposeBForGPU` hazard)
                // or defined after `first_pos` in the same block.
                if let Some(ValueDef::OpResult { op: def_op_id, .. }) =
                    module.values.get(rhs_val).map(|v| v.def)
                {
                    if let Some(def_op) = module.ops.get(def_op_id)
                        && (def_op.is_op("stablehlo", "transpose")
                            || def_op.is_op("mo", "transpose"))
                    {
                        valid = false;
                        break;
                    }
                    if let Some(&def_pos) = op_pos.get(&def_op_id)
                        && def_pos > first_pos
                    {
                        valid = false;
                        break;
                    }
                }

                let Some(res_tt) = module.values.get(res_val).and_then(|v| v.ty.as_tensor()) else {
                    valid = false;
                    break;
                };
                let Some(rhs_tt) = module.values.get(rhs_val).and_then(|v| v.ty.as_tensor()) else {
                    valid = false;
                    break;
                };
                if res_tt.dtype != first_res_tt.dtype
                    || rhs_tt.dtype != first_rhs_tt.dtype
                    || res_tt.shape.len() != res_rank
                    || rhs_tt.shape.len() != rhs_rank
                    || res_tt.shape[..res_rank - 1] != first_res_tt.shape[..res_rank - 1]
                    || rhs_tt.shape[..rhs_rank - 1] != first_rhs_tt.shape[..rhs_rank - 1]
                {
                    valid = false;
                    break;
                }
                let (Dim::Static(res_cols), Dim::Static(rhs_cols)) =
                    (&res_tt.shape[res_rank - 1], &rhs_tt.shape[rhs_rank - 1])
                else {
                    valid = false;
                    break;
                };
                if *res_cols <= 0 || res_cols != rhs_cols {
                    valid = false;
                    break;
                }

                weights_to_concat.push(rhs_val);
                columns_in_each.push(*res_cols);
                total_columns += *res_cols;
                source_locs.push(m_op.location.clone());
                orig_results.push(res_val);
            }

            if !valid {
                continue;
            }

            let fused_loc = fuse_locations(&source_locs);
            let mut concat_shape = first_rhs_tt.shape.clone();
            concat_shape[rhs_rank - 1] = Dim::Static(total_columns);
            let concat_ty = Type::Tensor(TensorType {
                shape: concat_shape,
                dtype: first_rhs_tt.dtype.clone(),
                encoding: first_rhs_tt.encoding.clone(),
            });

            let mut concat_attrs = IndexMap::new();
            concat_attrs.insert(
                "dimension".to_string(),
                Attribute::Int((rhs_rank - 1) as i64),
            );
            let concat_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: crate::ir::OpName::new("stablehlo", "concatenate"),
                operands: weights_to_concat,
                results: smallvec::SmallVec::new(),
                attributes: concat_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: fused_loc.clone(),
            });
            let concat_val_id = module.values.insert_with_key(|id| crate::ir::Value {
                id,
                ssa_name: None,
                ty: concat_ty,
                def: ValueDef::OpResult {
                    op: concat_op_id,
                    index: 0,
                },
            });
            module.ops[concat_op_id].results.push(concat_val_id);

            let mut merged_res_shape = first_res_tt.shape.clone();
            merged_res_shape[res_rank - 1] = Dim::Static(total_columns);
            let merged_res_ty = Type::Tensor(TensorType {
                shape: merged_res_shape,
                dtype: first_res_tt.dtype.clone(),
                encoding: first_res_tt.encoding.clone(),
            });

            let merged_matmul_id = module.ops.insert_with_key(|id| {
                let mut op = Operation {
                    id,
                    name: first_op.name.clone(),
                    operands: smallvec::smallvec![lhs_val, concat_val_id],
                    results: smallvec::SmallVec::new(),
                    attributes: first_op.attributes.clone(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: fused_loc.clone(),
                };
                stamp_fused_provenance(&mut op, &source_locs);
                op
            });
            let merged_matmul_val = module.values.insert_with_key(|id| crate::ir::Value {
                id,
                ssa_name: None,
                ty: merged_res_ty,
                def: ValueDef::OpResult {
                    op: merged_matmul_id,
                    index: 0,
                },
            });
            module.ops[merged_matmul_id].results.push(merged_matmul_val);

            let mut split_attrs = IndexMap::new();
            split_attrs.insert("axis".to_string(), Attribute::Int((res_rank - 1) as i64));
            split_attrs.insert(
                "split_sizes".to_string(),
                Attribute::DenseIntArray(columns_in_each),
            );
            let split_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: crate::ir::OpName::new("mo", "split"),
                operands: smallvec::smallvec![merged_matmul_val],
                results: orig_results.clone(),
                attributes: split_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: fused_loc,
            });
            for (idx, &res_v) in orig_results.iter().enumerate() {
                if let Some(v_mut) = module.values.get_mut(res_v) {
                    v_mut.def = ValueDef::OpResult {
                        op: split_op_id,
                        index: idx as u32,
                    };
                }
            }

            insertions.push((first_id, vec![concat_op_id, merged_matmul_id, split_op_id]));
            for &m_id in &matmul_ids {
                to_remove.insert(m_id);
            }
            merged_groups += 1;
        }

        if !insertions.is_empty() {
            let insert_map: std::collections::HashMap<OpId, Vec<OpId>> =
                insertions.into_iter().collect();
            let mut new_block_ops = Vec::with_capacity(block_ops.len() + 4);
            for op_id in block_ops {
                if let Some(new_ops) = insert_map.get(&op_id) {
                    new_block_ops.extend_from_slice(new_ops);
                }
                if to_remove.contains(&op_id) {
                    module.ops.remove(op_id);
                } else {
                    new_block_ops.push(op_id);
                }
            }
            module.blocks[block_id].ops = new_block_ops;
        }
    }

    merged_groups
}

/// Forward and backward dimension correspondence across a reshape operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReshapeDimMapping {
    /// For each input axis `i`, the ordered list of output axes that receive its volume.
    pub forward: Vec<Vec<usize>>,
    /// For each output axis `j`, the ordered list of input axes that contribute to its volume.
    pub backward: Vec<Vec<usize>>,
}

/// Computes the bidirectional dimension correspondence between `before` and `after` reshape
/// shapes using two-pointer cumulative volume matching.
///
/// Positive entries represent statically known dimensions (`> 0`), while negative entries
/// represent symbolic/dynamic dimension identifiers (`< 0`). Zero-sized dimensions, mismatched
/// static volumes, or mixed static/dynamic dimensions return `None`.
pub fn analyze_static_reshape_dim_mapping(
    before: &[i64],
    after: &[i64],
) -> Option<ReshapeDimMapping> {
    if before.is_empty() || after.is_empty() {
        return None;
    }
    if before.contains(&0) || after.contains(&0) {
        return None;
    }

    let calc_static_volume = |dims: &[i64]| -> Option<i64> {
        let mut total = 1i64;
        for &d in dims {
            if d > 0 {
                total = total.checked_mul(d)?;
            }
        }
        Some(total)
    };

    let before_total = calc_static_volume(before)?;
    let after_total = calc_static_volume(after)?;
    if before_total != after_total {
        return None;
    }

    // Verify dynamic dimension sequence compatibility.
    let before_dyn: Vec<i64> = before.iter().copied().filter(|&d| d < 0).collect();
    let after_dyn: Vec<i64> = after.iter().copied().filter(|&d| d < 0).collect();
    if before_dyn != after_dyn {
        return None;
    }

    let mut forward = vec![Vec::new(); before.len()];
    let mut backward = vec![Vec::new(); after.len()];

    let starts_with_static = before[0] > 0;
    let step: isize = if starts_with_static { 1 } else { -1 };
    let mut b_idx: isize = if starts_with_static {
        0
    } else {
        before.len() as isize - 1
    };
    let mut a_idx: isize = if starts_with_static {
        0
    } else {
        after.len() as isize - 1
    };

    let mut b_vol = 1i64;
    let mut a_vol = 1i64;
    let mut b_enabled = true;
    let mut a_enabled = true;

    while b_vol != before_total || a_vol != after_total {
        if b_idx < 0 || b_idx >= before.len() as isize || a_idx < 0 || a_idx >= after.len() as isize
        {
            return None;
        }
        let b_dim = before[b_idx as usize];
        let a_dim = after[a_idx as usize];
        let b_is_static = b_dim > 0;
        let a_is_static = a_dim > 0;

        if b_is_static != a_is_static {
            return None;
        }

        if b_is_static {
            if b_enabled {
                b_vol = b_vol.checked_mul(b_dim)?;
            }
            if a_enabled {
                a_vol = a_vol.checked_mul(a_dim)?;
            }

            b_enabled = b_vol <= a_vol;
            a_enabled = a_vol <= b_vol;

            if b_dim != 1 && a_dim != 1 {
                forward[b_idx as usize].push(a_idx as usize);
                backward[a_idx as usize].push(b_idx as usize);
            }
        } else if b_vol != a_vol || b_dim != a_dim || !b_enabled || !a_enabled {
            return None;
        }

        if b_enabled {
            b_idx += step;
        }
        if a_enabled {
            a_idx += step;
        }
    }

    if !starts_with_static {
        for map in &mut forward {
            map.reverse();
        }
        for map in &mut backward {
            map.reverse();
        }
    }

    Some(ReshapeDimMapping { forward, backward })
}

/// Maps a split or slice axis `axis` through a reshape from `before` to `after`.
///
/// Trailing unit (`1`) dimensions starting at `axis` are skipped until a non-unit static
/// dimension is found. Returns `None` if no non-unit static dimension is found or if that
/// dimension is merged with a left-neighbor dimension in `after`.
pub fn map_axis_through_static_reshape(
    before: &[i64],
    after: &[i64],
    axis: usize,
) -> Option<usize> {
    let mapping = analyze_static_reshape_dim_mapping(before, after)?;
    let mut mapped_axis = axis;
    while mapped_axis < before.len() && before[mapped_axis] == 1 {
        mapped_axis += 1;
    }
    if mapped_axis >= before.len() || before[mapped_axis] <= 0 {
        return None;
    }
    let &new_axis = mapping.forward.get(mapped_axis)?.first()?;
    let backward_sources = mapping.backward.get(new_axis)?;
    let could_mix = backward_sources.len() > 1;
    let axis_is_first = backward_sources.first().copied() == Some(mapped_axis);
    if could_mix && !axis_is_first {
        return None;
    }
    Some(new_axis)
}

fn extract_single_int_attr(attr: &Attribute) -> Option<i64> {
    match attr {
        Attribute::Int(v) => Some(*v),
        Attribute::DenseIntArray(v) if v.len() == 1 => Some(v[0]),
        Attribute::Array(v) if v.len() == 1 => extract_single_int_attr(&v[0]),
        Attribute::Opaque(s) | Attribute::String(s) => {
            s.split(':').next()?.trim().parse::<i64>().ok()
        }
        _ => None,
    }
}

fn extract_gather_single_index(module: &Module, op: &Operation) -> Option<i64> {
    if let Some(attr) = op
        .attributes
        .get("index")
        .or_else(|| op.attributes.get("indices"))
        .or_else(|| op.attributes.get("start_indices"))
        && let Some(idx) = extract_single_int_attr(attr)
    {
        return Some(idx);
    }
    if let Some(&idx_val) = op.operands.get(1)
        && let Some(v_info) = module.values.get(idx_val)
        && let ValueDef::OpResult { op: def_id, .. } = v_info.def
        && let Some(def_op) = module.ops.get(def_id)
        && matches!(
            def_op.name.to_string().as_str(),
            "stablehlo.constant" | "arith.constant" | "mo.constant"
        )
        && let Some(val_attr) = def_op.attributes.get("value")
    {
        return extract_single_int_attr(val_attr);
    }
    None
}

/// Rewrites exhaustive single-index axis-0 `gather` operations (`mo.gather`, `stablehlo.gather`,
/// `onnx.Gather`) in a basic block that cover every index `0..N` of a static outer dimension `N`
/// into a single `mo.split` inserted before the earliest gather in the group.
///
/// Returns the number of `mo.split` operations created.
pub fn coalesce_exhaustive_gathers_to_split(module: &mut Module) -> usize {
    let mut coalesced = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let block_ops = module.blocks[block_id].ops.clone();
        // Map each input `ValueId` to `(axis0_len, Vec<(block_pos, op_id, gather_idx)>)`.
        type GatherEntry = (usize, OpId, i64);
        let mut by_input: IndexMap<ValueId, (usize, Vec<GatherEntry>)> = IndexMap::new();

        for (pos, &op_id) in block_ops.iter().enumerate() {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if !matches!(
                op.name.to_string().as_str(),
                "mo.gather" | "stablehlo.gather" | "onnx.Gather"
            ) {
                continue;
            }
            if op.operands.is_empty() || op.results.len() != 1 {
                continue;
            }
            let axis = op
                .attributes
                .get("axis")
                .or_else(|| op.attributes.get("dimension"))
                .and_then(extract_single_int_attr)
                .unwrap_or(0);
            if axis != 0 {
                continue;
            }
            let Some(idx) = extract_gather_single_index(module, op) else {
                continue;
            };
            let input_val = op.operands[0];
            let Some(in_info) = module.values.get(input_val) else {
                continue;
            };
            let Type::Tensor(ref in_tt) = in_info.ty else {
                continue;
            };
            let Some(Dim::Static(axis0_dim)) = in_tt.shape.first() else {
                continue;
            };
            if *axis0_dim <= 0 || idx < 0 || idx >= *axis0_dim {
                continue;
            }
            let entry = by_input
                .entry(input_val)
                .or_insert_with(|| (*axis0_dim as usize, Vec::new()));
            entry.1.push((pos, op_id, idx));
        }

        let mut insertions: std::collections::HashMap<OpId, OpId> =
            std::collections::HashMap::new();
        let mut to_remove: std::collections::HashSet<OpId> = std::collections::HashSet::new();

        for (input_val, (axis0_len, entries)) in by_input {
            if entries.len() < axis0_len {
                continue;
            }
            // Keep first-seen gather for each index `0..axis0_len` and find the dominant gather.
            let mut idx_to_gather: Vec<Option<(usize, OpId)>> = vec![None; axis0_len];
            for &(pos, op_id, idx) in &entries {
                let slot = &mut idx_to_gather[idx as usize];
                if slot.is_none() {
                    *slot = Some((pos, op_id));
                }
            }
            if idx_to_gather.iter().any(Option::is_none) {
                continue;
            }
            let matched: Vec<(usize, OpId)> =
                idx_to_gather.into_iter().map(Option::unwrap).collect();

            // Verify consistent device placement across all gathers in the group.
            let first_dev = module.ops[matched[0].1]
                .attributes
                .get("castforge.device")
                .cloned();
            let same_device = matched.iter().all(|&(_, g_id)| {
                module.ops[g_id].attributes.get("castforge.device") == first_dev.as_ref()
            });
            if !same_device {
                continue;
            }

            let dominant_op_id = matched
                .iter()
                .min_by_key(|&&(pos, _)| pos)
                .map(|&(_, id)| id)
                .unwrap();

            let mut split_results = Vec::with_capacity(axis0_len);
            let mut locs = Vec::with_capacity(axis0_len);
            for &(_, g_id) in &matched {
                split_results.push(module.ops[g_id].results[0]);
                locs.push(module.ops[g_id].location.clone());
            }

            let mut split_attrs = IndexMap::new();
            split_attrs.insert("axis".to_string(), Attribute::Int(0));
            split_attrs.insert(
                "split_sizes".to_string(),
                Attribute::DenseIntArray(vec![1; axis0_len]),
            );
            if let Some(dev) = first_dev {
                split_attrs.insert("castforge.device".to_string(), dev);
            }

            let split_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: crate::ir::OpName::new("mo", "split"),
                operands: smallvec::smallvec![input_val],
                results: split_results.iter().copied().collect(),
                attributes: split_attrs,
                regions: Default::default(),
                custom_syntax: false,
                location: fuse_locations(&locs),
            });
            if let Some(split_op) = module.ops.get_mut(split_op_id) {
                stamp_fused_provenance(split_op, &locs);
            }

            for (idx, &res_val) in split_results.iter().enumerate() {
                if let Some(v_info) = module.values.get_mut(res_val) {
                    v_info.def = ValueDef::OpResult {
                        op: split_op_id,
                        index: idx as u32,
                    };
                }
            }

            insertions.insert(dominant_op_id, split_op_id);
            for &(_, g_id) in &matched {
                to_remove.insert(g_id);
            }
            coalesced += 1;
        }

        if !insertions.is_empty() {
            let mut new_block_ops = Vec::with_capacity(block_ops.len());
            for op_id in block_ops {
                if let Some(&split_id) = insertions.get(&op_id) {
                    new_block_ops.push(split_id);
                }
                if to_remove.contains(&op_id) {
                    module.ops.remove(op_id);
                } else {
                    new_block_ops.push(op_id);
                }
            }
            module.blocks[block_id].ops = new_block_ops;
        }
    }

    coalesced
}

fn def_op_of(module: &Module, val: ValueId) -> Option<&Operation> {
    let v_info = module.values.get(val)?;
    let ValueDef::OpResult { op, .. } = v_info.def else {
        return None;
    };
    module.ops.get(op)
}

fn match_splat_f64(module: &Module, mut val: ValueId) -> Option<f64> {
    for _ in 0..3 {
        let op = def_op_of(module, val)?;
        if matches!(
            op.name.to_string().as_str(),
            "stablehlo.broadcast_in_dim" | "mo.static.broadcast_to" | "mo.broadcast_to"
        ) && !op.operands.is_empty()
        {
            val = op.operands[0];
            continue;
        }
        if matches!(
            op.name.to_string().as_str(),
            "stablehlo.constant" | "arith.constant" | "mo.constant"
        ) {
            return op.attributes.get("value")?.as_splat_f64();
        }
        break;
    }
    None
}

fn is_splat_approx(module: &Module, val: ValueId, target: f64, tol: f64) -> bool {
    match_splat_f64(module, val).is_some_and(|v| (v - target).abs() <= tol)
}

fn is_binary_op(module: &Module, val: ValueId, names: &[&str]) -> Option<(ValueId, ValueId, OpId)> {
    let op = def_op_of(module, val)?;
    if op.operands.len() == 2 && names.contains(&op.name.to_string().as_str()) {
        Some((op.operands[0], op.operands[1], op.id))
    } else {
        None
    }
}

fn is_unary_op(module: &Module, val: ValueId, names: &[&str]) -> Option<(ValueId, OpId)> {
    let op = def_op_of(module, val)?;
    if op.operands.len() == 1 && names.contains(&op.name.to_string().as_str()) {
        Some((op.operands[0], op.id))
    } else {
        None
    }
}

fn match_comm_const_other(
    module: &Module,
    lhs: ValueId,
    rhs: ValueId,
    target: f64,
    tol: f64,
) -> Option<(ValueId, ValueId)> {
    if is_splat_approx(module, lhs, target, tol) {
        Some((rhs, lhs))
    } else if is_splat_approx(module, rhs, target, tol) {
        Some((lhs, rhs))
    } else {
        None
    }
}

const ADD_OPS: &[&str] = &["stablehlo.add", "mo.add", "arith.addf"];
const MUL_OPS: &[&str] = &["stablehlo.multiply", "mo.mul", "arith.mulf"];
const DIV_OPS: &[&str] = &["stablehlo.divide", "mo.div", "arith.divf"];
const NEG_OPS: &[&str] = &["stablehlo.negate", "mo.negative", "arith.negf"];
const EXP_OPS: &[&str] = &["stablehlo.exponential", "mo.exp", "math.exp"];
const SIGMOID_OPS: &[&str] = &["mo.sigmoid", "stablehlo.logistic"];
const ERF_OPS: &[&str] = &["stablehlo.erf", "mo.erf", "onnx.Erf", "math.erf"];
const TANH_OPS: &[&str] = &["stablehlo.tanh", "mo.tanh", "math.tanh"];
const POW_OPS: &[&str] = &["stablehlo.power", "mo.pow", "math.powf"];

fn match_exp_to_sigmoid(module: &Module, op: &Operation) -> Option<(ValueId, Vec<OpId>)> {
    if op.operands.len() != 2 || !DIV_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    let num = op.operands[0];
    let den = op.operands[1];
    if !is_splat_approx(module, num, 1.0, 1e-5) {
        return None;
    }
    let (add_l, add_r, add_id) = is_binary_op(module, den, ADD_OPS)?;
    let (exp_val, _) = match_comm_const_other(module, add_l, add_r, 1.0, 1e-5)?;
    let (neg_val, exp_id) = is_unary_op(module, exp_val, EXP_OPS)?;
    let (x_val, neg_id) = is_unary_op(module, neg_val, NEG_OPS)?;
    Some((x_val, vec![add_id, exp_id, neg_id]))
}

fn match_sigmoid_mul_to_silu(module: &Module, op: &Operation) -> Option<(ValueId, Vec<OpId>)> {
    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (cand_sig, cand_x) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        if let Some((sig_in, sig_id)) = is_unary_op(module, cand_sig, SIGMOID_OPS)
            && sig_in == cand_x
        {
            return Some((cand_x, vec![sig_id]));
        }
    }
    None
}

fn match_sigmoid_mul_to_gelu_quick(
    module: &Module,
    op: &Operation,
) -> Option<(ValueId, Vec<OpId>)> {
    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (cand_sig, cand_x) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        if let Some((sig_in, sig_id)) = is_unary_op(module, cand_sig, SIGMOID_OPS)
            && let Some((mul_l, mul_r, inner_mul_id)) = is_binary_op(module, sig_in, MUL_OPS)
            && let Some((inner_x, _)) = match_comm_const_other(module, mul_l, mul_r, 1.702, 1e-3)
            && inner_x == cand_x
        {
            return Some((cand_x, vec![sig_id, inner_mul_id]));
        }
    }
    None
}

fn match_erf_to_gelu(module: &Module, op: &Operation) -> Option<(ValueId, Vec<OpId>)> {
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (half_side, erf_side) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        let Some((hl, hr, half_mul_id)) = is_binary_op(module, half_side, MUL_OPS) else {
            continue;
        };
        let Some((x_half, _)) = match_comm_const_other(module, hl, hr, 0.5, 1e-5) else {
            continue;
        };
        let Some((al, ar, add_id)) = is_binary_op(module, erf_side, ADD_OPS) else {
            continue;
        };
        let Some((erf_val, _)) = match_comm_const_other(module, al, ar, 1.0, 1e-5) else {
            continue;
        };
        let Some((erf_arg, erf_id)) = is_unary_op(module, erf_val, ERF_OPS) else {
            continue;
        };

        // Match either `x / sqrt(2)` or `x * (1 / sqrt(2))`.
        if let Some((dl, dr, scale_id)) = is_binary_op(module, erf_arg, DIV_OPS)
            && dl == x_half
            && is_splat_approx(module, dr, SQRT2, 1e-4)
        {
            return Some((x_half, vec![half_mul_id, add_id, erf_id, scale_id]));
        }
        if let Some((ml, mr, scale_id)) = is_binary_op(module, erf_arg, MUL_OPS)
            && let Some((x_erf, _)) = match_comm_const_other(module, ml, mr, INV_SQRT2, 1e-4)
            && x_erf == x_half
        {
            return Some((x_half, vec![half_mul_id, add_id, erf_id, scale_id]));
        }
    }
    None
}

fn match_cube_of(module: &Module, val: ValueId, x: ValueId) -> Option<Vec<OpId>> {
    if let Some((pl, pr, pow_id)) = is_binary_op(module, val, POW_OPS)
        && pl == x
        && is_splat_approx(module, pr, 3.0, 1e-5)
    {
        return Some(vec![pow_id]);
    }
    let (cl, cr, cube_id) = is_binary_op(module, val, MUL_OPS)?;
    for (sq_cand, other) in [(cl, cr), (cr, cl)] {
        if other == x
            && let Some((sl, sr, sq_id)) = is_binary_op(module, sq_cand, MUL_OPS)
            && sl == x
            && sr == x
        {
            return Some(vec![cube_id, sq_id]);
        }
    }
    None
}

fn match_tanh_to_gelu_tanh(module: &Module, op: &Operation) -> Option<(ValueId, Vec<OpId>)> {
    // sqrt(2 / pi) = 0.7978845608028654
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    const COEFF: f64 = 0.044715;

    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (half_side, tanh_side) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        let Some((hl, hr, half_mul_id)) = is_binary_op(module, half_side, MUL_OPS) else {
            continue;
        };
        let Some((x_half, _)) = match_comm_const_other(module, hl, hr, 0.5, 1e-5) else {
            continue;
        };
        let Some((al, ar, outer_add_id)) = is_binary_op(module, tanh_side, ADD_OPS) else {
            continue;
        };
        let Some((tanh_val, _)) = match_comm_const_other(module, al, ar, 1.0, 1e-5) else {
            continue;
        };
        let Some((tanh_arg, tanh_id)) = is_unary_op(module, tanh_val, TANH_OPS) else {
            continue;
        };
        let Some((sl, sr, scale_mul_id)) = is_binary_op(module, tanh_arg, MUL_OPS) else {
            continue;
        };
        let Some((poly_val, _)) = match_comm_const_other(module, sl, sr, SQRT_2_OVER_PI, 1e-4)
        else {
            continue;
        };
        let Some((pl, pr, poly_add_id)) = is_binary_op(module, poly_val, ADD_OPS) else {
            continue;
        };
        for (cand_x, cand_term) in [(pl, pr), (pr, pl)] {
            if cand_x != x_half {
                continue;
            }
            let Some((tl, tr, term_mul_id)) = is_binary_op(module, cand_term, MUL_OPS) else {
                continue;
            };
            let Some((cube_val, _)) = match_comm_const_other(module, tl, tr, COEFF, 1e-5) else {
                continue;
            };
            if let Some(mut cube_ids) = match_cube_of(module, cube_val, x_half) {
                let mut dead = vec![
                    half_mul_id,
                    outer_add_id,
                    tanh_id,
                    scale_mul_id,
                    poly_add_id,
                    term_mul_id,
                ];
                dead.append(&mut cube_ids);
                return Some((x_half, dead));
            }
        }
    }
    None
}

/// Recognizes decomposed arithmetic subgraphs for `sigmoid`, `silu`, `gelu_quick`, `gelu`, and
/// `gelu_tanh` and folds each root operation into its canonical `mo.<activation>` operator while
/// pruning dead intermediate arithmetic nodes.
///
/// Patterns recognized:
/// - `1 / (1 + exp(-x))` -> `mo.sigmoid(x)`
/// - `x * sigmoid(x)` -> `mo.silu(x)`
/// - `x * sigmoid(1.702 * x)` -> `mo.gelu_quick(x)`
/// - `0.5 * x * (1 + erf(x / sqrt(2)))` -> `mo.gelu(x)`
/// - `0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))` -> `mo.gelu_tanh(x)`
///
/// Returns the number of composite activations recognized and folded.
pub fn recognize_composite_activations(module: &mut Module) -> usize {
    let mut folded = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let mut candidate_dead: std::collections::HashSet<OpId> = std::collections::HashSet::new();

        for &op_id in &op_ids {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let matched: Option<(&str, ValueId, Vec<OpId>)> =
                if let Some((x, dead)) = match_exp_to_sigmoid(module, &op) {
                    Some(("sigmoid", x, dead))
                } else if let Some((x, dead)) = match_sigmoid_mul_to_silu(module, &op) {
                    Some(("silu", x, dead))
                } else if let Some((x, dead)) = match_sigmoid_mul_to_gelu_quick(module, &op) {
                    Some(("gelu_quick", x, dead))
                } else if let Some((x, dead)) = match_erf_to_gelu(module, &op) {
                    Some(("gelu", x, dead))
                } else if let Some((x, dead)) = match_tanh_to_gelu_tanh(module, &op) {
                    Some(("gelu_tanh", x, dead))
                } else {
                    None
                };

            let Some((act_name, x_val, dead_ids)) = matched else {
                continue;
            };

            let mut locs = vec![op.location.clone()];
            for &d_id in &dead_ids {
                if let Some(d_op) = module.ops.get(d_id) {
                    locs.push(d_op.location.clone());
                }
                candidate_dead.insert(d_id);
            }

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName::new("mo", act_name);
                op_mut.operands = smallvec::smallvec![x_val];
                op_mut.attributes.insert(
                    "castforge.composite_activation".to_string(),
                    Attribute::String(act_name.to_string()),
                );
                stamp_fused_provenance(op_mut, &locs);
            }
            folded += 1;
        }

        if !candidate_dead.is_empty() {
            // Sweep dead intermediate ops in reverse topological order.
            for &op_id in op_ids.iter().rev() {
                if !candidate_dead.contains(&op_id) {
                    continue;
                }
                let is_unused = module
                    .ops
                    .get(op_id)
                    .is_some_and(|op| op.results.iter().all(|&r| module.use_count(r) == 0));
                if is_unused {
                    module.remove_op(block_id, op_id);
                }
            }
        }
    }

    folded
}

/// Applies the inverse of `perm` to `items` (`result[perm[i]] = items[i]`).
///
/// Returns `None` if `items.len() != perm.len()` or if `perm` is not a valid permutation of
/// `0..items.len()`.
pub fn unpermute_slice<T: Clone>(items: &[T], perm: &[usize]) -> Option<Vec<T>> {
    let n = items.len();
    if perm.len() != n {
        return None;
    }
    let mut out: Vec<Option<T>> = vec![None; n];
    for (idx, &perm_idx) in perm.iter().enumerate() {
        if perm_idx >= n || out[perm_idx].is_some() {
            return None;
        }
        out[perm_idx] = Some(items[idx].clone());
    }
    out.into_iter().collect()
}

fn extract_transpose_perm(op: &Operation) -> Option<Vec<usize>> {
    if !matches!(
        op.name.to_string().as_str(),
        "stablehlo.transpose" | "mo.transpose" | "onnx.Transpose"
    ) || op.operands.is_empty()
        || op.results.len() != 1
    {
        return None;
    }
    let raw =
        extract_int_list_attr(op, "permutation").or_else(|| extract_int_list_attr(op, "perm"))?;
    let mut perm = Vec::with_capacity(raw.len());
    for v in raw {
        if v < 0 {
            return None;
        }
        perm.push(v as usize);
    }
    let n = perm.len();
    let mut seen = vec![false; n];
    for &p in &perm {
        if p >= n || seen[p] {
            return None;
        }
        seen[p] = true;
    }
    Some(perm)
}

fn is_sinkable_unary_elementwise(op: &Operation) -> bool {
    op.operands.len() == 1
        && op.results.len() == 1
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.tanh"
                | "stablehlo.exponential"
                | "stablehlo.negate"
                | "stablehlo.abs"
                | "stablehlo.log"
                | "stablehlo.sqrt"
                | "stablehlo.rsqrt"
                | "stablehlo.logistic"
                | "stablehlo.erf"
                | "stablehlo.convert"
                | "mo.relu"
                | "mo.sigmoid"
                | "mo.silu"
                | "mo.gelu"
                | "mo.gelu_quick"
                | "mo.gelu_tanh"
                | "mo.tanh"
                | "mo.exp"
                | "mo.negative"
        )
}

fn is_sinkable_binary_elementwise(op: &Operation) -> bool {
    op.operands.len() == 2
        && op.results.len() == 1
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.add"
                | "stablehlo.subtract"
                | "stablehlo.multiply"
                | "stablehlo.divide"
                | "stablehlo.maximum"
                | "stablehlo.minimum"
                | "stablehlo.power"
                | "mo.add"
                | "mo.sub"
                | "mo.mul"
                | "mo.div"
                | "mo.max"
                | "mo.min"
                | "mo.pow"
        )
}

fn is_sinkable_broadcast_op(op: &Operation) -> bool {
    op.operands.len() == 1
        && op.results.len() == 1
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.broadcast_in_dim" | "mo.static.broadcast_to" | "mo.broadcast_to"
        )
}

/// Sinks sole-use `transpose` operations past unary elementwise ops, matching-permutation binary
/// elementwise ops (coalescing two input transposes into one output transpose), and static
/// broadcasts (padding the permutation for leading broadcast dimensions and unpermuting the
/// intermediate broadcast shape).
///
/// Returns the number of transpose-sinking rewrites applied.
pub fn sink_transpose_through_elementwise_and_broadcast(module: &mut Module) -> usize {
    let mut sunk_count = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        loop {
            let mut changed_in_pass = false;
            let op_ids = module.blocks[block_id].ops.clone();

            for &op_id in &op_ids {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.results.len() != 1 {
                    continue;
                }
                let orig_res_val = op.results[0];
                let Some(orig_res_tt) = module
                    .values
                    .get(orig_res_val)
                    .and_then(|v| v.ty.as_tensor())
                    .cloned()
                else {
                    continue;
                };

                // Case 1: Unary `unary(transpose(x, perm))` -> `transpose(unary(x), perm)`
                if is_sinkable_unary_elementwise(&op) {
                    let trans_val = op.operands[0];
                    if module.use_count(trans_val) != 1 {
                        continue;
                    }
                    let Some(trans_op) = def_op_of(module, trans_val).cloned() else {
                        continue;
                    };
                    if extract_transpose_perm(&trans_op).is_none() {
                        continue;
                    }
                    if trans_op.attributes.get("castforge.device")
                        != op.attributes.get("castforge.device")
                    {
                        continue;
                    }
                    let raw_in = trans_op.operands[0];
                    let Some(raw_in_tt) = module
                        .values
                        .get(raw_in)
                        .and_then(|v| v.ty.as_tensor())
                        .cloned()
                    else {
                        continue;
                    };

                    let inter_tt = TensorType {
                        shape: raw_in_tt.shape,
                        dtype: orig_res_tt.dtype.clone(),
                        encoding: orig_res_tt.encoding.clone(),
                    };
                    let inter_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: Type::Tensor(inter_tt),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: 0,
                        },
                    });

                    if let Some(u_mut) = module.ops.get_mut(op_id) {
                        u_mut.operands[0] = raw_in;
                        u_mut.results[0] = inter_val;
                    }
                    if let Some(t_mut) = module.ops.get_mut(trans_op.id) {
                        t_mut.operands[0] = inter_val;
                        t_mut.results[0] = orig_res_val;
                    }
                    if let Some(v_mut) = module.values.get_mut(orig_res_val) {
                        v_mut.def = ValueDef::OpResult {
                            op: trans_op.id,
                            index: 0,
                        };
                    }
                    module.values.remove(trans_val);

                    let blk = &mut module.blocks[block_id].ops;
                    blk.retain(|&id| id != trans_op.id);
                    if let Some(pos) = blk.iter().position(|&id| id == op_id) {
                        blk.insert(pos + 1, trans_op.id);
                    }

                    sunk_count += 1;
                    changed_in_pass = true;
                    break;
                }

                // Case 2: Binary elementwise `bin(transpose(x, p), transpose(y, p))`
                // -> `transpose(bin(x, y), p)`
                if is_sinkable_binary_elementwise(&op) {
                    let lhs_val = op.operands[0];
                    let rhs_val = op.operands[1];
                    if lhs_val == rhs_val
                        || module.use_count(lhs_val) != 1
                        || module.use_count(rhs_val) != 1
                    {
                        continue;
                    }
                    let (Some(t1_op), Some(t2_op)) = (
                        def_op_of(module, lhs_val).cloned(),
                        def_op_of(module, rhs_val).cloned(),
                    ) else {
                        continue;
                    };
                    let (Some(perm1), Some(perm2)) = (
                        extract_transpose_perm(&t1_op),
                        extract_transpose_perm(&t2_op),
                    ) else {
                        continue;
                    };
                    if perm1 != perm2 {
                        continue;
                    }
                    let dev = op.attributes.get("castforge.device");
                    if t1_op.attributes.get("castforge.device") != dev
                        || t2_op.attributes.get("castforge.device") != dev
                    {
                        continue;
                    }
                    let in1 = t1_op.operands[0];
                    let in2 = t2_op.operands[0];
                    let (Some(in1_tt), Some(in2_tt)) = (
                        module
                            .values
                            .get(in1)
                            .and_then(|v| v.ty.as_tensor())
                            .cloned(),
                        module
                            .values
                            .get(in2)
                            .and_then(|v| v.ty.as_tensor())
                            .cloned(),
                    ) else {
                        continue;
                    };
                    if in1_tt.shape != in2_tt.shape {
                        continue;
                    }

                    let inter_tt = TensorType {
                        shape: in1_tt.shape,
                        dtype: orig_res_tt.dtype.clone(),
                        encoding: orig_res_tt.encoding.clone(),
                    };
                    let inter_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: Type::Tensor(inter_tt),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: 0,
                        },
                    });

                    if let Some(b_mut) = module.ops.get_mut(op_id) {
                        b_mut.operands[0] = in1;
                        b_mut.operands[1] = in2;
                        b_mut.results[0] = inter_val;
                    }
                    let locs = vec![t1_op.location.clone(), t2_op.location.clone()];
                    if let Some(t1_mut) = module.ops.get_mut(t1_op.id) {
                        t1_mut.operands[0] = inter_val;
                        t1_mut.results[0] = orig_res_val;
                        stamp_fused_provenance(t1_mut, &locs);
                    }
                    if let Some(v_mut) = module.values.get_mut(orig_res_val) {
                        v_mut.def = ValueDef::OpResult {
                            op: t1_op.id,
                            index: 0,
                        };
                    }
                    module.values.remove(lhs_val);
                    module.remove_op(block_id, t2_op.id);

                    let blk = &mut module.blocks[block_id].ops;
                    blk.retain(|&id| id != t1_op.id);
                    if let Some(pos) = blk.iter().position(|&id| id == op_id) {
                        blk.insert(pos + 1, t1_op.id);
                    }

                    sunk_count += 1;
                    changed_in_pass = true;
                    break;
                }

                // Case 3: Static broadcast `broadcast(transpose(x, perm))`
                // -> `transpose(broadcast(x), padded_perm)`
                if is_sinkable_broadcast_op(&op) {
                    let trans_val = op.operands[0];
                    if module.use_count(trans_val) != 1 {
                        continue;
                    }
                    let Some(trans_op) = def_op_of(module, trans_val).cloned() else {
                        continue;
                    };
                    let Some(perm) = extract_transpose_perm(&trans_op) else {
                        continue;
                    };
                    if trans_op.attributes.get("castforge.device")
                        != op.attributes.get("castforge.device")
                    {
                        continue;
                    }
                    let raw_in = trans_op.operands[0];
                    let Some(trans_tt) = module
                        .values
                        .get(trans_val)
                        .and_then(|v| v.ty.as_tensor())
                        .cloned()
                    else {
                        continue;
                    };
                    if orig_res_tt.shape.len() < trans_tt.shape.len() {
                        continue;
                    }
                    // If `broadcast_dimensions` is present on `stablehlo.broadcast_in_dim`,
                    // verify it represents standard right-aligned broadcasting `[extra..out_rank]`.
                    let extra_dims = orig_res_tt.shape.len() - trans_tt.shape.len();
                    if let Some(bcast_dims) = extract_int_list_attr(&op, "broadcast_dimensions") {
                        let expected: Vec<i64> =
                            (extra_dims as i64..orig_res_tt.shape.len() as i64).collect();
                        if bcast_dims != expected {
                            continue;
                        }
                    }

                    let mut padded_perm: Vec<usize> = (0..extra_dims).collect();
                    for p in perm {
                        padded_perm.push(p + extra_dims);
                    }
                    let Some(new_bcast_shape) = unpermute_slice(&orig_res_tt.shape, &padded_perm)
                    else {
                        continue;
                    };

                    let inter_tt = TensorType {
                        shape: new_bcast_shape,
                        dtype: orig_res_tt.dtype.clone(),
                        encoding: orig_res_tt.encoding.clone(),
                    };
                    let inter_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: Type::Tensor(inter_tt),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: 0,
                        },
                    });

                    if let Some(b_mut) = module.ops.get_mut(op_id) {
                        b_mut.operands[0] = raw_in;
                        b_mut.results[0] = inter_val;
                    }
                    let new_perm_i64: Vec<i64> = padded_perm.iter().map(|&p| p as i64).collect();
                    if let Some(t_mut) = module.ops.get_mut(trans_op.id) {
                        t_mut.operands[0] = inter_val;
                        t_mut.results[0] = orig_res_val;
                        let key = if t_mut.attributes.contains_key("perm") {
                            "perm"
                        } else {
                            "permutation"
                        };
                        t_mut
                            .attributes
                            .insert(key.to_string(), Attribute::DenseIntArray(new_perm_i64));
                    }
                    if let Some(v_mut) = module.values.get_mut(orig_res_val) {
                        v_mut.def = ValueDef::OpResult {
                            op: trans_op.id,
                            index: 0,
                        };
                    }
                    module.values.remove(trans_val);

                    let blk = &mut module.blocks[block_id].ops;
                    blk.retain(|&id| id != trans_op.id);
                    if let Some(pos) = blk.iter().position(|&id| id == op_id) {
                        blk.insert(pos + 1, trans_op.id);
                    }

                    sunk_count += 1;
                    changed_in_pass = true;
                    break;
                }
            }

            if !changed_in_pass {
                break;
            }
        }
    }

    sunk_count
}

/// Object-group role in a hybrid CacheBlend registration
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/read_set.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendObjectGroupKind {
    /// Full-attention KV object group read by both the prefix and non-prefix blend legs.
    Attention,
    /// Connector-private auxiliary object group read alongside attention on the blend leg.
    Aux,
    /// Recurrent/SSM state object group read only on the contiguous prefix leg.
    Recurrent,
}

/// Per-leg object-group read routing for CacheBlend hybrid lookups
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/read_set.py` `_BlendReadGroups`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlendReadGroupSet {
    /// Ascending group IDs read by the relocatable blend leg (`attention + aux`).
    pub blend_gids: Vec<usize>,
    /// Ascending group IDs read by the contiguous prefix leg (`attention + recurrent`).
    pub prefix_gids: Vec<usize>,
    /// Ascending group IDs holding recurrent state alone.
    pub recurrent_gids: Vec<usize>,
    /// The single full-attention object group ID.
    pub attn_gid: usize,
}

/// Classifies a registration's `num_object_groups` and `group_kinds` into per-leg read sets
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/read_set.py`
/// `_classify_cb_read_groups`).
///
/// Single-group (`<= 1`) fused layouts map both legs to group `0`. Multi-group layouts must
/// contain exactly one [`BlendObjectGroupKind::Attention`] group and at most one
/// [`BlendObjectGroupKind::Aux`] group.
pub fn classify_blend_read_groups(
    num_object_groups: usize,
    group_kinds: &[BlendObjectGroupKind],
) -> Option<BlendReadGroupSet> {
    if num_object_groups <= 1 {
        return Some(BlendReadGroupSet {
            blend_gids: vec![0],
            prefix_gids: vec![0],
            recurrent_gids: Vec::new(),
            attn_gid: 0,
        });
    }
    if group_kinds.len() != num_object_groups {
        return None;
    }
    let mut attn = Vec::new();
    let mut aux = Vec::new();
    let mut recurrent = Vec::new();
    for (idx, &kind) in group_kinds.iter().enumerate() {
        match kind {
            BlendObjectGroupKind::Attention => attn.push(idx),
            BlendObjectGroupKind::Aux => aux.push(idx),
            BlendObjectGroupKind::Recurrent => recurrent.push(idx),
        }
    }
    if attn.len() != 1 || aux.len() > 1 {
        return None;
    }
    let attn_gid = attn[0];
    let mut blend_gids = attn.clone();
    blend_gids.extend_from_slice(&aux);
    blend_gids.sort_unstable();

    let mut prefix_gids = attn;
    prefix_gids.extend_from_slice(&recurrent);
    prefix_gids.sort_unstable();

    Some(BlendReadGroupSet {
        blend_gids,
        prefix_gids,
        recurrent_gids: recurrent,
        attn_gid,
    })
}

/// Interleaves per-group expanded object keys into a chunk-major key sequence with uniform stride
/// `per_group_keys.len() * expansion` per chunk so prefix bitmaps stay leading-ones-aligned
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/read_set.py`
/// `_cb_chunk_major_object_keys`).
pub fn interleave_chunk_major_object_keys<T: Clone>(
    per_group_keys: &[Vec<T>],
    num_chunks: usize,
) -> Option<Vec<T>> {
    if per_group_keys.is_empty() || num_chunks == 0 {
        return Some(Vec::new());
    }
    let first_len = per_group_keys[0].len();
    if first_len == 0 || !first_len.is_multiple_of(num_chunks) {
        return None;
    }
    if per_group_keys.iter().any(|g| g.len() != first_len) {
        return None;
    }
    let expansion = first_len / num_chunks;
    let mut out = Vec::with_capacity(per_group_keys.len() * first_len);
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * expansion;
        let end = start + expansion;
        for group_keys in per_group_keys {
            out.extend_from_slice(&group_keys[start..end]);
        }
    }
    Some(out)
}

/// Resolved per-group RoPE geometry for CacheBlend in-place key rotation
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/rope.py`
/// `_cb_group_rope_geometry`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CBGroupRopeGeometry {
    /// Whether K and V are fused-packed into a doubled `2 * head_size` per-head stride.
    pub fused_packed: bool,
    /// Element stride per head (`2 * head_size` when `fused_packed`, `hidden_dim` for MLA, else
    /// `head_size`).
    pub per_head_elems: usize,
    /// Number of heads rotated across the row (`1` for MLA latents).
    pub num_heads: usize,
    /// Starting element offset of the rotated slice within each head (`> 0` for MLA).
    pub rot_offset: usize,
}

/// Computes the per-group CacheBlend RoPE geometry across split K/V, fused-packed K/V, and
/// Multi-head Latent Attention (MLA `[content | rope]`) layouts
/// (inspired by LMCache `lmcache/v1/multiprocess/modules/blend/rope.py`
/// `_cb_group_rope_geometry`).
#[allow(clippy::too_many_arguments)]
pub fn compute_cb_group_rope_geometry(
    tokens_per_block: usize,
    slots_per_block: usize,
    kv_size: usize,
    hidden_dim: usize,
    head_size: usize,
    is_fused_format: bool,
    declared_hs: Option<usize>,
    rot_window: Option<(usize, usize)>,
) -> Option<CBGroupRopeGeometry> {
    if tokens_per_block == 0
        || tokens_per_block != slots_per_block
        || hidden_dim == 0
        || head_size == 0
    {
        return None;
    }
    if let Some((rot_offset, rot_width)) = rot_window
        && rot_offset > 0
    {
        if kv_size != 1 || rot_width == 0 || rot_offset + rot_width != hidden_dim {
            return None;
        }
        return Some(CBGroupRopeGeometry {
            fused_packed: false,
            per_head_elems: hidden_dim,
            num_heads: 1,
            rot_offset,
        });
    }
    if !matches!(kv_size, 1 | 2) {
        return None;
    }
    let fused_packed = is_fused_format && declared_hs.is_none_or(|hs| hs == 2 * head_size);
    let per_head_elems = head_size * if fused_packed { 2 } else { 1 };
    let num_heads = hidden_dim / per_head_elems;
    if num_heads == 0 || num_heads * per_head_elems != hidden_dim {
        return None;
    }
    Some(CBGroupRopeGeometry {
        fused_packed,
        per_head_elems,
        num_heads,
        rot_offset: 0,
    })
}

/// Outcome of reserving or committing a payload in [`DaxGenerationSlotArena`]
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/dax/core.py` `DaxPutFromPtrResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaxPutOutcome {
    /// Payload was newly reserved/committed into a slot.
    Inserted,
    /// Key is already committed in the index.
    AlreadyExists,
    /// Key already has an active uncommitted write reservation.
    Inflight,
    /// Arena has no free slots available.
    NoSpace,
    /// Payload size is zero or exceeds `slot_bytes`.
    InvalidSize,
}

/// Read reservation pinning a committed DAX slot generation against concurrent recycling
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/dax/core.py` `DaxReadReservation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DaxReadReservation {
    /// Position of the key in the caller's request slice.
    pub result_index: usize,
    /// Reserved key identifier.
    pub key: u64,
    /// Byte offset of the slot within the arena (`slot_id * slot_bytes`).
    pub offset_bytes: u64,
    /// Payload size in bytes stored in the slot.
    pub size_bytes: u64,
    /// Physical slot index.
    pub slot_id: usize,
    /// Monotonic slot generation counter at reservation time.
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DaxIndexEntry {
    offset_bytes: u64,
    size_bytes: u64,
    slot_id: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DaxInflightEntry {
    offset_bytes: u64,
    size_bytes: u64,
    slot_id: usize,
    generation: u64,
    canceled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DaxSlotState {
    generation: u64,
    committed: bool,
    borrow_count: u32,
    pending_free: bool,
}

/// Generational fixed-slot persistent memory (DAX) arena manager with read-borrow reference
/// counting and deferred `pending_free` slot reclamation
/// (inspired by LMCache `lmcache/v1/distributed/l2_adapters/dax/core.py` `DaxCore`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaxGenerationSlotArena {
    slot_bytes: u64,
    max_slots: usize,
    next_slot: usize,
    free_slots: Vec<usize>,
    index: IndexMap<u64, DaxIndexEntry>,
    inflight: IndexMap<u64, DaxInflightEntry>,
    slot_states: IndexMap<usize, DaxSlotState>,
    external_lock_counts: IndexMap<u64, u32>,
    active_reads: usize,
    active_writes: usize,
}

impl DaxGenerationSlotArena {
    /// Creates a new arena with `max_dax_size_bytes` capacity partitioned into fixed `slot_bytes`
    /// slots. Returns `None` if `slot_bytes == 0` or the arena cannot fit at least one slot.
    pub fn new(max_dax_size_bytes: u64, slot_bytes: u64) -> Option<Self> {
        if slot_bytes == 0 {
            return None;
        }
        let max_slots = (max_dax_size_bytes / slot_bytes) as usize;
        if max_slots == 0 {
            return None;
        }
        Some(Self {
            slot_bytes,
            max_slots,
            next_slot: 0,
            free_slots: Vec::new(),
            index: IndexMap::new(),
            inflight: IndexMap::new(),
            slot_states: IndexMap::new(),
            external_lock_counts: IndexMap::new(),
            active_reads: 0,
            active_writes: 0,
        })
    }

    /// Maximum number of fixed-size slots in the arena.
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Fixed byte capacity of each slot.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    fn allocate_slot(&mut self) -> Option<usize> {
        if let Some(slot_id) = self.free_slots.pop() {
            return Some(slot_id);
        }
        if self.next_slot < self.max_slots {
            let slot_id = self.next_slot;
            self.next_slot += 1;
            return Some(slot_id);
        }
        None
    }

    fn reserve_slot_state(&mut self, slot_id: usize) -> u64 {
        let next_gen = self
            .slot_states
            .get(&slot_id)
            .map_or(1, |s| s.generation + 1);
        self.slot_states.insert(
            slot_id,
            DaxSlotState {
                generation: next_gen,
                committed: false,
                borrow_count: 0,
                pending_free: false,
            },
        );
        next_gen
    }

    fn schedule_slot_reclaim(&mut self, slot_id: usize, generation: u64) {
        let Some(state) = self.slot_states.get_mut(&slot_id) else {
            return;
        };
        if state.generation != generation {
            return;
        }
        state.committed = false;
        if state.borrow_count == 0 {
            state.pending_free = false;
            if !self.free_slots.contains(&slot_id) {
                self.free_slots.push(slot_id);
            }
        } else {
            state.pending_free = true;
        }
    }

    fn is_entry_committed(&self, entry: &DaxIndexEntry) -> bool {
        self.slot_states
            .get(&entry.slot_id)
            .is_some_and(|s| s.generation == entry.generation && s.committed)
    }

    /// Reserves an in-flight write slot for `key` of `size_bytes`, returning the outcome and
    /// `Some((slot_id, generation, offset_bytes))` when a new slot was allocated.
    pub fn begin_put(
        &mut self,
        key: u64,
        size_bytes: u64,
    ) -> (DaxPutOutcome, Option<(usize, u64, u64)>) {
        if size_bytes == 0 || size_bytes > self.slot_bytes {
            return (DaxPutOutcome::InvalidSize, None);
        }
        if self.index.contains_key(&key) {
            return (DaxPutOutcome::AlreadyExists, None);
        }
        if self.inflight.contains_key(&key) {
            return (DaxPutOutcome::Inflight, None);
        }
        let Some(slot_id) = self.allocate_slot() else {
            return (DaxPutOutcome::NoSpace, None);
        };
        let offset_bytes = (slot_id as u64) * self.slot_bytes;
        let generation = self.reserve_slot_state(slot_id);
        self.inflight.insert(
            key,
            DaxInflightEntry {
                offset_bytes,
                size_bytes,
                slot_id,
                generation,
                canceled: false,
            },
        );
        self.active_writes += 1;
        (
            DaxPutOutcome::Inserted,
            Some((slot_id, generation, offset_bytes)),
        )
    }

    /// Finalizes an in-flight write for `key`. Commits the slot if `!write_failed` and the write
    /// was not force-canceled; otherwise reclaims the slot.
    pub fn finalize_put(&mut self, key: u64, write_failed: bool) -> bool {
        self.active_writes = self.active_writes.saturating_sub(1);
        let Some(inflight) = self.inflight.shift_remove(&key) else {
            return false;
        };
        if inflight.canceled || write_failed {
            self.schedule_slot_reclaim(inflight.slot_id, inflight.generation);
            return false;
        }
        if let Some(state) = self.slot_states.get_mut(&inflight.slot_id)
            && state.generation == inflight.generation
        {
            state.committed = true;
            state.pending_free = false;
        }
        self.index.insert(
            key,
            DaxIndexEntry {
                offset_bytes: inflight.offset_bytes,
                size_bytes: inflight.size_bytes,
                slot_id: inflight.slot_id,
                generation: inflight.generation,
            },
        );
        true
    }

    /// Synchronously allocates and commits a slot for `key` (`begin_put` + `finalize_put`).
    pub fn put(&mut self, key: u64, size_bytes: u64) -> DaxPutOutcome {
        let (outcome, reserved) = self.begin_put(key, size_bytes);
        if reserved.is_some() {
            self.finalize_put(key, false);
        }
        outcome
    }

    /// Checks whether `keys` are committed in the arena, optionally incrementing external lock
    /// reference counts for hits.
    pub fn exists_many(&mut self, keys: &[u64], lock: bool) -> Vec<bool> {
        let mut results = vec![false; keys.len()];
        for (i, &key) in keys.iter().enumerate() {
            let Some(&entry) = self.index.get(&key) else {
                continue;
            };
            if !self.is_entry_committed(&entry) {
                continue;
            }
            results[i] = true;
            if lock {
                *self.external_lock_counts.entry(key).or_insert(0) += 1;
            }
        }
        results
    }

    /// Decrements external lock reference counts for `keys`.
    pub fn unlock_many(&mut self, keys: &[u64]) {
        for &key in keys {
            if let Some(count) = self.external_lock_counts.get_mut(&key) {
                if *count <= 1 {
                    self.external_lock_counts.shift_remove(&key);
                } else {
                    *count -= 1;
                }
            }
        }
    }

    /// Reserves committed entries for a read pipeline, incrementing each hit slot's
    /// `borrow_count` so concurrent eviction cannot recycle the physical slot until
    /// [`Self::finalize_reads`] is called.
    pub fn reserve_reads(
        &mut self,
        keys: &[u64],
        prefix_only: bool,
    ) -> (Vec<DaxReadReservation>, Vec<bool>) {
        let mut reservations = Vec::new();
        let mut hits = vec![false; keys.len()];

        for (i, &key) in keys.iter().enumerate() {
            let Some(&entry) = self.index.get(&key) else {
                if prefix_only {
                    break;
                }
                continue;
            };
            let Some(state) = self.slot_states.get_mut(&entry.slot_id) else {
                if prefix_only {
                    break;
                }
                continue;
            };
            if state.generation != entry.generation || !state.committed {
                if prefix_only {
                    break;
                }
                continue;
            }
            state.borrow_count += 1;
            reservations.push(DaxReadReservation {
                result_index: i,
                key,
                offset_bytes: entry.offset_bytes,
                size_bytes: entry.size_bytes,
                slot_id: entry.slot_id,
                generation: entry.generation,
            });
            hits[i] = true;
        }
        if !reservations.is_empty() {
            self.active_reads += 1;
        }
        (reservations, hits)
    }

    /// Releases read reservations and recycles any `pending_free` slots whose `borrow_count`
    /// reaches zero.
    pub fn finalize_reads(&mut self, reservations: &[DaxReadReservation]) {
        if reservations.is_empty() {
            return;
        }
        self.active_reads = self.active_reads.saturating_sub(1);
        for res in reservations {
            let Some(state) = self.slot_states.get_mut(&res.slot_id) else {
                continue;
            };
            if state.generation != res.generation {
                continue;
            };
            state.borrow_count = state.borrow_count.saturating_sub(1);
            if state.pending_free && state.borrow_count == 0 {
                state.pending_free = false;
                if !self.free_slots.contains(&res.slot_id) {
                    self.free_slots.push(res.slot_id);
                }
            }
        }
    }

    /// Deletes `keys` from the index. When `!force`, externally locked and in-flight keys are
    /// skipped. Slots with active read borrows enter `pending_free` until borrows drain.
    pub fn delete_many(&mut self, keys: &[u64], force: bool) -> Vec<bool> {
        let mut results = vec![false; keys.len()];
        for (i, &key) in keys.iter().enumerate() {
            if let Some(inflight) = self.inflight.get_mut(&key) {
                if !force {
                    continue;
                }
                inflight.canceled = true;
                self.external_lock_counts.shift_remove(&key);
                results[i] = true;
                continue;
            }
            let Some(&entry) = self.index.get(&key) else {
                continue;
            };
            if !force && self.external_lock_counts.get(&key).copied().unwrap_or(0) > 0 {
                continue;
            }
            self.index.shift_remove(&key);
            self.external_lock_counts.shift_remove(&key);
            self.schedule_slot_reclaim(entry.slot_id, entry.generation);
            results[i] = true;
        }
        results
    }

    /// Returns `(current_usage, usage_after_ongoing_eviction)` as fractions of `max_slots`.
    /// The second value subtracts `pending_free` slots that have been evicted from the index but
    /// are waiting for active read borrows to drain.
    pub fn usage(&self) -> (f64, f64) {
        if self.max_slots == 0 {
            return (0.0, 0.0);
        }
        let live_slot_count = self.next_slot.saturating_sub(self.free_slots.len());
        let pending_reclaim_count = self.slot_states.values().filter(|s| s.pending_free).count();
        let current_usage = (live_slot_count as f64) / (self.max_slots as f64);
        let post_eviction_slots = live_slot_count.saturating_sub(pending_reclaim_count);
        let usage_after_eviction = (post_eviction_slots as f64) / (self.max_slots as f64);
        (current_usage, usage_after_eviction)
    }

    /// Returns `true` iff every committed, non-canceled in-flight, or borrowed slot has
    /// `slot_id < new_max_slots`.
    pub fn can_shrink_to(&self, new_max_slots: usize) -> bool {
        if new_max_slots == 0 {
            return false;
        }
        for entry in self.index.values() {
            if self.is_entry_committed(entry) && entry.slot_id >= new_max_slots {
                return false;
            }
        }
        for inflight in self.inflight.values() {
            if !inflight.canceled && inflight.slot_id >= new_max_slots {
                return false;
            }
        }
        for (&slot_id, state) in &self.slot_states {
            if slot_id >= new_max_slots && state.borrow_count > 0 {
                return false;
            }
        }
        true
    }
}

/// Default chunk-count histogram upper bounds (`0, 1, 2, 4, ..., 2048`)
/// (matching LMCache `lmcache/v1/mp_coordinator/utils/histogram.py` `CHUNK_COUNT_BOUNDS`).
pub const CHUNK_COUNT_HISTOGRAM_BOUNDS: &[f64] = &[
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
];

/// Fixed-bound bucket histogram with Prometheus-compatible cumulative bucket export
/// (inspired by LMCache `lmcache/v1/mp_coordinator/utils/histogram.py` `Histogram`).
#[derive(Debug, Clone, PartialEq)]
pub struct BucketHistogram {
    bounds: Vec<f64>,
    bucket_counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl BucketHistogram {
    /// Creates an empty histogram with strictly ascending `bounds`. Returns `None` if `bounds` is
    /// empty or not strictly ascending.
    pub fn with_bounds(bounds: &[f64]) -> Option<Self> {
        if bounds.is_empty() || bounds.windows(2).any(|w| w[0] >= w[1] || w[0].is_nan()) {
            return None;
        }
        if bounds.last().is_some_and(|b| b.is_nan()) {
            return None;
        }
        Some(Self {
            bounds: bounds.to_vec(),
            bucket_counts: vec![0; bounds.len() + 1],
            sum: 0.0,
            count: 0,
        })
    }

    /// Records one observation of `value` into the first bucket with `bound >= value` (or the
    /// overflow `+Inf` bucket).
    pub fn observe(&mut self, value: f64) {
        let idx = self.bounds.partition_point(|&b| b < value);
        self.bucket_counts[idx] += 1;
        self.sum += value;
        self.count += 1;
    }

    /// Total number of recorded observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sum of all recorded observation values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns cumulative `(le_label, cumulative_count)` pairs followed by `("+Inf", count)`.
    pub fn cumulative_buckets(&self) -> Vec<(String, u64)> {
        let mut out = Vec::with_capacity(self.bounds.len() + 1);
        let mut acc = 0u64;
        for (&bound, &cnt) in self.bounds.iter().zip(self.bucket_counts.iter()) {
            acc += cnt;
            out.push((format!("{bound}"), acc));
        }
        out.push(("+Inf".to_string(), self.count));
        out
    }
}

/// Cache event operation kind applied to [`CoordinatorKeyDirectory`]
/// (inspired by LMCache `lmcache/v1/mp_coordinator/api.py` `CacheEventType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEventOpKind {
    /// Inserts or updates a placement for the key.
    Store,
    /// Removes a placement for the key.
    Delete,
    /// Records a cache hit/access on the key.
    Access,
}

/// Live placement of a cached key in the fleet directory
/// (inspired by LMCache `lmcache/v1/mp_coordinator/views/key_directory.py` `Placement`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRecord {
    /// Reporting worker instance identifier.
    pub instance_id: String,
    /// Incarnation epoch of the reporting worker.
    pub incarnation: u64,
    /// Storage tier (`L1` volatile worker memory vs `L2` persistent storage).
    pub tier: CacheEventTier,
    /// Backend name within the tier (e.g., `"cpu"`, `"gds"`, `"redis"`).
    pub backend: String,
    /// Reported payload size in bytes.
    pub size_bytes: u64,
    /// Whether the backend is a fleet-shared pool (deduplicated across instances).
    pub shared: bool,
}

/// Single key mutation entry inside a coordinator event batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEventEntrySpec {
    /// Target object key (`chunk_hash`).
    pub key: u64,
    /// Payload size in bytes (used on [`CacheEventOpKind::Store`]).
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct KeyDirectoryRecord {
    placements: Vec<PlacementRecord>,
    last_access_ts: u64,
    access_count: u64,
}

/// Fleet-wide key directory with incremental tier byte accounting and `O(|l1_keys(instance)|)`
/// reverse-indexed L1 instance fencing
/// (inspired by LMCache `lmcache/v1/mp_coordinator/views/key_directory.py` `KeyDirectory`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoordinatorKeyDirectory {
    directory: IndexMap<u64, KeyDirectoryRecord>,
    l1_keys_by_instance: IndexMap<String, Vec<u64>>,
    l1_count: usize,
    l1_size_bytes: u64,
    l2_count: usize,
    l2_size_bytes: u64,
}

impl CoordinatorKeyDirectory {
    /// Creates an empty fleet key directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of keys with at least one active placement.
    pub fn num_keys(&self) -> usize {
        self.directory.len()
    }

    /// Returns `(l1_count, l1_size_bytes, l2_count, l2_size_bytes)`.
    pub fn tier_stats(&self) -> (usize, u64, usize, u64) {
        (
            self.l1_count,
            self.l1_size_bytes,
            self.l2_count,
            self.l2_size_bytes,
        )
    }

    fn adjust_stats(&mut self, tier: CacheEventTier, count_delta: isize, bytes_delta: i64) {
        match tier {
            CacheEventTier::L1 => {
                self.l1_count = (self.l1_count as isize + count_delta).max(0) as usize;
                self.l1_size_bytes = (self.l1_size_bytes as i64 + bytes_delta).max(0) as u64;
            }
            CacheEventTier::L2 => {
                self.l2_count = (self.l2_count as isize + count_delta).max(0) as usize;
                self.l2_size_bytes = (self.l2_size_bytes as i64 + bytes_delta).max(0) as u64;
            }
        }
    }

    fn find_placement_index(
        placements: &[PlacementRecord],
        instance_id: &str,
        tier: CacheEventTier,
        backend: &str,
        shared: bool,
    ) -> Option<usize> {
        placements.iter().position(|p| {
            p.shared == shared
                && (shared || p.instance_id == instance_id)
                && p.tier == tier
                && p.backend == backend
        })
    }

    /// Applies an admitted event batch idempotently (`Store` upserts, `Delete` removes, `Access`
    /// increments access count and updates `last_access_ts`).
    #[allow(clippy::too_many_arguments)]
    pub fn consume_batch(
        &mut self,
        instance_id: &str,
        incarnation: u64,
        tier: CacheEventTier,
        backend: &str,
        shared: bool,
        op_kind: CacheEventOpKind,
        ts: u64,
        entries: &[CacheEventEntrySpec],
    ) {
        for entry in entries {
            match op_kind {
                CacheEventOpKind::Store => {
                    let record = self.directory.entry(entry.key).or_default();
                    let new_p = PlacementRecord {
                        instance_id: instance_id.to_string(),
                        incarnation,
                        tier,
                        backend: backend.to_string(),
                        size_bytes: entry.size_bytes,
                        shared,
                    };
                    if let Some(idx) = Self::find_placement_index(
                        &record.placements,
                        instance_id,
                        tier,
                        backend,
                        shared,
                    ) {
                        let old_bytes = record.placements[idx].size_bytes;
                        record.placements[idx] = new_p;
                        let delta = (entry.size_bytes as i64) - (old_bytes as i64);
                        self.adjust_stats(tier, 0, delta);
                    } else {
                        record.placements.push(new_p);
                        self.adjust_stats(tier, 1, entry.size_bytes as i64);
                    }
                    let rec = self.directory.get_mut(&entry.key).unwrap();
                    rec.last_access_ts = rec.last_access_ts.max(ts);
                    if tier == CacheEventTier::L1 {
                        let l1_list = self
                            .l1_keys_by_instance
                            .entry(instance_id.to_string())
                            .or_default();
                        if !l1_list.contains(&entry.key) {
                            l1_list.push(entry.key);
                        }
                    }
                }
                CacheEventOpKind::Delete => {
                    let Some(record) = self.directory.get_mut(&entry.key) else {
                        continue;
                    };
                    if let Some(idx) = Self::find_placement_index(
                        &record.placements,
                        instance_id,
                        tier,
                        backend,
                        shared,
                    ) {
                        let removed = record.placements.remove(idx);
                        let is_empty = record.placements.is_empty();
                        let still_has_inst_l1 = record
                            .placements
                            .iter()
                            .any(|p| p.tier == CacheEventTier::L1 && p.instance_id == instance_id);
                        if is_empty {
                            self.directory.shift_remove(&entry.key);
                        }
                        self.adjust_stats(tier, -1, -(removed.size_bytes as i64));
                        if tier == CacheEventTier::L1
                            && !still_has_inst_l1
                            && let Some(l1_list) = self.l1_keys_by_instance.get_mut(instance_id)
                        {
                            l1_list.retain(|&k| k != entry.key);
                        }
                    }
                }
                CacheEventOpKind::Access => {
                    if let Some(record) = self.directory.get_mut(&entry.key) {
                        record.last_access_ts = record.last_access_ts.max(ts);
                        record.access_count += 1;
                    }
                }
            }
        }
    }

    /// Drops every `L1` placement reported by `instance_id` using the reverse index while
    /// preserving all `L2` placements. Returns the number of `L1` placements removed.
    pub fn fence_instance(&mut self, instance_id: &str) -> usize {
        let Some(l1_keys) = self.l1_keys_by_instance.shift_remove(instance_id) else {
            return 0;
        };
        let mut removed_count = 0usize;
        let mut removed_bytes = 0i64;
        for key in l1_keys {
            let Some(record) = self.directory.get_mut(&key) else {
                continue;
            };
            let mut i = 0;
            while i < record.placements.len() {
                if record.placements[i].tier == CacheEventTier::L1
                    && record.placements[i].instance_id == instance_id
                {
                    let p = record.placements.remove(i);
                    removed_count += 1;
                    removed_bytes += p.size_bytes as i64;
                } else {
                    i += 1;
                }
            }
            if record.placements.is_empty() {
                self.directory.shift_remove(&key);
            }
        }
        self.adjust_stats(
            CacheEventTier::L1,
            -(removed_count as isize),
            -removed_bytes,
        );
        removed_count
    }

    /// Looks up placements for `keys`, sorting each key's placement list by
    /// `(instance_id, tier, backend)`.
    pub fn lookup(&self, keys: &[u64]) -> Vec<Vec<PlacementRecord>> {
        keys.iter()
            .map(|k| {
                let Some(record) = self.directory.get(k) else {
                    return Vec::new();
                };
                let mut list = record.placements.clone();
                list.sort_by(|a, b| {
                    let t_a = matches!(a.tier, CacheEventTier::L2);
                    let t_b = matches!(b.tier, CacheEventTier::L2);
                    a.instance_id
                        .cmp(&b.instance_id)
                        .then_with(|| t_a.cmp(&t_b))
                        .then_with(|| a.backend.cmp(&b.backend))
                });
                list
            })
            .collect()
    }

    /// Returns the number of [`CacheEventOpKind::Access`] events applied to each key in `keys`.
    pub fn get_access_counts(&self, keys: &[u64]) -> Vec<u64> {
        keys.iter()
            .map(|k| self.directory.get(k).map_or(0, |r| r.access_count))
            .collect()
    }
}

/// 8-byte ASCII magic header (`LMCBLK01`) for raw-block per-slot on-disk headers.
pub const RAW_BLOCK_SLOT_MAGIC: &[u8; 8] = b"LMCBLK01";

/// 8-byte ASCII magic header (`LMCIDX01`) for raw-block metadata checkpoint containers.
pub const RAW_BLOCK_META_MAGIC: &[u8; 8] = b"LMCIDX01";

/// Computes the standard IEEE 802.3 CRC-32 checksum over `data` (polynomial `0xEDB88320`).
pub fn compute_ieee_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Encodes a 24-byte raw-block slot header (`LMCBLK01`, `slot_identity`, `payload_len` in
/// little-endian byte order) padded with zeros to `header_bytes`.
///
/// Returns `None` if `header_bytes < 24`.
pub fn encode_raw_block_slot_header(
    slot_identity: u64,
    payload_len: u64,
    header_bytes: usize,
) -> Option<Vec<u8>> {
    if header_bytes < 24 {
        return None;
    }
    let mut out = vec![0u8; header_bytes];
    out[0..8].copy_from_slice(RAW_BLOCK_SLOT_MAGIC);
    out[8..16].copy_from_slice(&slot_identity.to_le_bytes());
    out[16..24].copy_from_slice(&payload_len.to_le_bytes());
    Some(out)
}

/// Decodes a raw-block slot header and returns `(slot_identity, payload_len)` when the
/// buffer begins with [`RAW_BLOCK_SLOT_MAGIC`].
pub fn decode_raw_block_slot_header(buf: &[u8]) -> Option<(u64, u64)> {
    if buf.len() < 24 || &buf[0..8] != RAW_BLOCK_SLOT_MAGIC {
        return None;
    }
    let slot_identity = u64::from_le_bytes(buf[8..16].try_into().ok()?);
    let payload_len = u64::from_le_bytes(buf[16..24].try_into().ok()?);
    Some((slot_identity, payload_len))
}

/// Parsed metadata checkpoint container header for double-buffered raw-block index persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawBlockMetaCheckpointHeader {
    /// Monotonically increasing checkpoint sequence number.
    pub seq: u64,
    /// Unpadded byte length of the serialized metadata payload.
    pub payload_len: u64,
    /// IEEE CRC-32 checksum over the unpadded payload bytes.
    pub crc32: u32,
    /// Byte offset of the selected container on the raw block device.
    pub container_offset: u64,
}

/// Encodes a block-aligned metadata checkpoint container (`LMCIDX01`, `version`, `flags = 0`,
/// `seq`, `payload_len`, `crc32`, `reserved = 0` at offset `0..40`, followed by `payload` at
/// `block_align..block_align + payload.len()`, rounded up to a multiple of `block_align`).
pub fn encode_raw_block_meta_checkpoint(
    version: u32,
    seq: u64,
    payload: &[u8],
    block_align: usize,
) -> Option<Vec<u8>> {
    if block_align < 40 || !block_align.is_power_of_two() {
        return None;
    }
    let payload_padded = if payload.is_empty() {
        block_align
    } else {
        payload.len().div_ceil(block_align) * block_align
    };
    let total_len = block_align.checked_add(payload_padded)?;
    let mut buf = vec![0u8; total_len];
    let crc32 = compute_ieee_crc32(payload);
    buf[0..8].copy_from_slice(RAW_BLOCK_META_MAGIC);
    buf[8..12].copy_from_slice(&version.to_le_bytes());
    buf[12..16].copy_from_slice(&0u32.to_le_bytes());
    buf[16..24].copy_from_slice(&seq.to_le_bytes());
    buf[24..32].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    buf[32..36].copy_from_slice(&crc32.to_le_bytes());
    buf[36..40].copy_from_slice(&0u32.to_le_bytes());
    buf[block_align..block_align + payload.len()].copy_from_slice(payload);
    Some(buf)
}

/// Scans double-buffered metadata checkpoint containers `(container_offset, raw_bytes)` and
/// returns the valid container with the highest `seq` number whose magic, `expected_version`,
/// bounds, and IEEE CRC-32 match.
pub fn select_latest_valid_meta_checkpoint(
    containers: &[(u64, Vec<u8>)],
    block_align: usize,
    expected_version: u32,
) -> Option<(RawBlockMetaCheckpointHeader, Vec<u8>)> {
    if block_align < 40 {
        return None;
    }
    let mut best: Option<(RawBlockMetaCheckpointHeader, Vec<u8>)> = None;
    for (container_offset, raw) in containers {
        if raw.len() < block_align || &raw[0..8] != RAW_BLOCK_META_MAGIC {
            continue;
        }
        let version = u32::from_le_bytes(raw[8..12].try_into().ok()?);
        let flags = u32::from_le_bytes(raw[12..16].try_into().ok()?);
        let seq = u64::from_le_bytes(raw[16..24].try_into().ok()?);
        let payload_len = u64::from_le_bytes(raw[24..32].try_into().ok()?);
        let crc32 = u32::from_le_bytes(raw[32..36].try_into().ok()?);
        if version != expected_version || flags != 0 || payload_len == 0 {
            continue;
        }
        let Ok(plen) = usize::try_from(payload_len) else {
            continue;
        };
        let Some(end) = block_align.checked_add(plen) else {
            continue;
        };
        if end > raw.len() {
            continue;
        }
        let payload = &raw[block_align..end];
        if compute_ieee_crc32(payload) != crc32 {
            continue;
        }
        let header = RawBlockMetaCheckpointHeader {
            seq,
            payload_len,
            crc32,
            container_offset: *container_offset,
        };
        if best.as_ref().is_none_or(|(prev, _)| seq >= prev.seq) {
            best = Some((header, payload.to_vec()));
        }
    }
    best
}

/// NVMe Flexible Data Placement (FDP) affinity raw-block slot allocator with crash-safe
/// free-list reconstruction from committed index checkpoints.
#[derive(Debug, Clone)]
pub struct FdpRawBlockSlotManager {
    max_slots: usize,
    next_slot: usize,
    free_slots: Vec<usize>,
    free_slots_by_placement_id: IndexMap<u16, Vec<usize>>,
    slot_placement_ids: IndexMap<usize, u16>,
    fdp_affinity_enabled: bool,
    fdp_affinity_hits: usize,
    fdp_affinity_fallbacks: usize,
}

impl FdpRawBlockSlotManager {
    /// Creates a new [`FdpRawBlockSlotManager`] with capacity `max_slots`.
    pub fn new(max_slots: usize, fdp_affinity_enabled: bool) -> Self {
        Self {
            max_slots,
            next_slot: 0,
            free_slots: Vec::new(),
            free_slots_by_placement_id: IndexMap::new(),
            slot_placement_ids: IndexMap::new(),
            fdp_affinity_enabled,
            fdp_affinity_hits: 0,
            fdp_affinity_fallbacks: 0,
        }
    }

    /// Returns `(next_slot, free_slots_count, fdp_affinity_hits, fdp_affinity_fallbacks)`.
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        (
            self.next_slot,
            self.free_slots.len(),
            self.fdp_affinity_hits,
            self.fdp_affinity_fallbacks,
        )
    }

    /// Returns the last NVMe FDP placement identifier recorded for `slot`, if any.
    pub fn slot_placement_id(&self, slot: usize) -> Option<u16> {
        self.slot_placement_ids.get(&slot).copied()
    }

    /// Allocates a slot, preferring a recycled slot that previously carried the same NVMe FDP
    /// `placement_id` (reclaim unit handle) when `fdp_affinity_enabled` is true.
    pub fn allocate_slot(&mut self, placement_id: Option<u16>) -> Option<usize> {
        if self.fdp_affinity_enabled
            && let Some(pid) = placement_id
        {
            if let Some(bucket) = self.free_slots_by_placement_id.get_mut(&pid)
                && let Some(slot) = bucket.pop()
            {
                if bucket.is_empty() {
                    self.free_slots_by_placement_id.shift_remove(&pid);
                }
                if let Some(pos) = self.free_slots.iter().rposition(|&s| s == slot) {
                    self.free_slots.remove(pos);
                }
                self.slot_placement_ids.insert(slot, pid);
                self.fdp_affinity_hits += 1;
                return Some(slot);
            }
            if !self.free_slots.is_empty() {
                self.fdp_affinity_fallbacks += 1;
            }
        }
        let slot = if let Some(recycled) = self.free_slots.pop() {
            if self.fdp_affinity_enabled
                && let Some(&old_pid) = self.slot_placement_ids.get(&recycled)
                && let Some(bucket) = self.free_slots_by_placement_id.get_mut(&old_pid)
            {
                if let Some(pos) = bucket.iter().rposition(|&s| s == recycled) {
                    bucket.remove(pos);
                }
                if bucket.is_empty() {
                    self.free_slots_by_placement_id.shift_remove(&old_pid);
                }
            }
            recycled
        } else {
            if self.next_slot >= self.max_slots {
                return None;
            }
            let fresh = self.next_slot;
            self.next_slot += 1;
            fresh
        };
        if self.fdp_affinity_enabled {
            if let Some(pid) = placement_id {
                self.slot_placement_ids.insert(slot, pid);
            } else {
                self.slot_placement_ids.shift_remove(&slot);
            }
        }
        Some(slot)
    }

    /// Returns `slot` to the free list and indexes it under its last FDP placement identifier.
    pub fn free_slot(&mut self, slot: usize) {
        if slot >= self.next_slot || self.free_slots.contains(&slot) {
            return;
        }
        self.free_slots.push(slot);
        if self.fdp_affinity_enabled
            && let Some(&pid) = self.slot_placement_ids.get(&slot)
        {
            self.free_slots_by_placement_id
                .entry(pid)
                .or_default()
                .push(slot);
        }
    }

    /// Reconstructs the allocator state from a checkpointed `next_slot` high-water mark and
    /// `committed_slots` set, recovering any uncommitted in-flight slots from `[0, next_slot)`
    /// back into `free_slots`.
    pub fn rebuild_free_slots_from_committed(
        &mut self,
        next_slot: usize,
        committed_slots: &[usize],
    ) -> bool {
        if next_slot > self.max_slots {
            return false;
        }
        let mut occupied = IndexSet::new();
        for &slot in committed_slots {
            if slot >= next_slot || !occupied.insert(slot) {
                return false;
            }
        }
        self.next_slot = next_slot;
        self.free_slots.clear();
        self.free_slots_by_placement_id.clear();
        self.slot_placement_ids.clear();
        for slot in 0..next_slot {
            if !occupied.contains(&slot) {
                self.free_slots.push(slot);
            }
        }
        true
    }
}

fn static_i64_dims(tt: &TensorType) -> Option<Vec<i64>> {
    let mut out = Vec::with_capacity(tt.shape.len());
    for d in &tt.shape {
        match d {
            Dim::Static(v) if *v > 0 => out.push(*v),
            _ => return None,
        }
    }
    Some(out)
}

/// Commutes a sole-use `stablehlo.slice` followed by `stablehlo.reshape` into `stablehlo.reshape`
/// followed by `stablehlo.slice` when the sliced dimension maps cleanly through the reshape
/// (either 1-to-1, factored by an integer divisor with unit step, or merged with trailing
/// dimensions with unit step).
///
/// Returns the number of slice-reshape pairs commuted.
pub fn swim_reshape_through_slice(module: &mut Module) -> usize {
    let mut commuted = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    *use_counts.entry(v).or_insert(0) += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        for op_id in op_ids {
            let Some(reshape_op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !matches!(
                reshape_op.name.to_string().as_str(),
                "stablehlo.reshape" | "mo.reshape"
            ) || reshape_op.operands.len() != 1
                || reshape_op.results.len() != 1
            {
                continue;
            }
            let slice_out_val = reshape_op.operands[0];
            if use_counts.get(&slice_out_val).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(slice_out_info) = module.values.get(slice_out_val).cloned() else {
                continue;
            };
            let ValueDef::OpResult { op: slice_id, .. } = slice_out_info.def else {
                continue;
            };
            if !module.blocks[block_id].ops.contains(&slice_id) {
                continue;
            }
            let Some(slice_op) = module.ops.get(slice_id).cloned() else {
                continue;
            };
            if !slice_op.is_op("stablehlo", "slice")
                || slice_op.operands.len() != 1
                || slice_op.results.len() != 1
            {
                continue;
            }
            if slice_op.attributes.get("castforge.device")
                != reshape_op.attributes.get("castforge.device")
            {
                continue;
            }

            let slice_in_val = slice_op.operands[0];
            let reshape_out_val = reshape_op.results[0];
            let (Some(in_info), Some(out_info)) = (
                module.values.get(slice_in_val),
                module.values.get(reshape_out_val),
            ) else {
                continue;
            };
            let (Type::Tensor(in_tt), Type::Tensor(mid_tt), Type::Tensor(out_tt)) =
                (&in_info.ty, &slice_out_info.ty, &out_info.ty)
            else {
                continue;
            };
            let (Some(slice_in_dims), Some(slice_out_dims), Some(reshape_out_dims)) = (
                static_i64_dims(in_tt),
                static_i64_dims(mid_tt),
                static_i64_dims(out_tt),
            ) else {
                continue;
            };
            let in_rank = slice_in_dims.len();
            if in_rank == 0 || slice_out_dims.len() != in_rank {
                continue;
            }

            let Some(starts) = extract_int_list_attr(&slice_op, "start_indices") else {
                continue;
            };
            let limits = extract_int_list_attr(&slice_op, "limit_indices").unwrap_or_else(|| {
                starts
                    .iter()
                    .zip(slice_out_dims.iter())
                    .map(|(&s, &len)| s + len)
                    .collect()
            });
            let strides = extract_int_list_attr(&slice_op, "strides")
                .unwrap_or_else(|| vec![1; slice_in_dims.len()]);
            if starts.len() != in_rank || limits.len() != in_rank || strides.len() != in_rank {
                continue;
            }

            // Identify the single sliced dimension.
            let mut sliced_axes = Vec::new();
            for d in 0..in_rank {
                if starts[d] != 0
                    || limits[d] != slice_in_dims[d]
                    || strides[d] != 1
                    || slice_out_dims[d] != slice_in_dims[d]
                {
                    sliced_axes.push(d);
                }
            }
            if sliced_axes.len() != 1 {
                continue;
            }
            let slice_axis = sliced_axes[0];
            let start = starts[slice_axis];
            let limit = limits[slice_axis];
            let step = strides[slice_axis];
            if start < 0 || limit <= start || step <= 0 {
                continue;
            }

            let Some(new_axis) =
                map_axis_through_static_reshape(&slice_out_dims, &reshape_out_dims, slice_axis)
            else {
                continue;
            };

            let old_sliced_dim = slice_out_dims[slice_axis];
            let new_sliced_dim = reshape_out_dims[new_axis];
            let slice_in_axis_dim = slice_in_dims[slice_axis];

            let (new_reshape_axis_dim, new_start, new_limit, new_step) =
                if old_sliced_dim == new_sliced_dim {
                    (slice_in_axis_dim, start, limit, step)
                } else if old_sliced_dim > new_sliced_dim {
                    let factor = old_sliced_dim / new_sliced_dim;
                    if step != 1
                        || old_sliced_dim % new_sliced_dim != 0
                        || slice_in_axis_dim % factor != 0
                        || start % factor != 0
                        || limit % factor != 0
                    {
                        continue;
                    }
                    (
                        slice_in_axis_dim / factor,
                        start / factor,
                        limit / factor,
                        1,
                    )
                } else {
                    let factor = new_sliced_dim / old_sliced_dim;
                    if step != 1 || new_sliced_dim % old_sliced_dim != 0 {
                        continue;
                    }
                    (
                        slice_in_axis_dim * factor,
                        start * factor,
                        limit * factor,
                        1,
                    )
                };

            let mut new_reshape_dims = reshape_out_dims.clone();
            new_reshape_dims[new_axis] = new_reshape_axis_dim;
            if analyze_static_reshape_dim_mapping(&slice_in_dims, &new_reshape_dims).is_none() {
                continue;
            }

            let out_rank = reshape_out_dims.len();
            let mut new_starts = vec![0i64; out_rank];
            let mut new_limits = new_reshape_dims.clone();
            let mut new_strides = vec![1i64; out_rank];
            new_starts[new_axis] = new_start;
            new_limits[new_axis] = new_limit;
            new_strides[new_axis] = new_step;

            let new_mid_tt = TensorType {
                shape: new_reshape_dims.into_iter().map(Dim::Static).collect(),
                dtype: out_tt.dtype.clone(),
                encoding: out_tt.encoding.clone(),
            };

            // Mutate `slice_id` into `stablehlo.reshape` and `op_id` into `stablehlo.slice`.
            if let Some(mid_val_mut) = module.values.get_mut(slice_out_val) {
                mid_val_mut.ty = Type::Tensor(new_mid_tt);
            }
            if let Some(first_op) = module.ops.get_mut(slice_id) {
                first_op.name = OpName::new("stablehlo", "reshape");
                first_op.attributes.shift_remove("start_indices");
                first_op.attributes.shift_remove("limit_indices");
                first_op.attributes.shift_remove("strides");
            }
            if let Some(second_op) = module.ops.get_mut(op_id) {
                second_op.name = OpName::new("stablehlo", "slice");
                second_op
                    .attributes
                    .insert("start_indices".into(), Attribute::DenseIntArray(new_starts));
                second_op
                    .attributes
                    .insert("limit_indices".into(), Attribute::DenseIntArray(new_limits));
                second_op
                    .attributes
                    .insert("strides".into(), Attribute::DenseIntArray(new_strides));
            }
            commuted += 1;
        }
    }

    commuted
}

/// Sinks a uniform `mo.split` through identical static `stablehlo.reshape` operations on every
/// split output when the split axis maps cleanly through the reshape.
///
/// Rewrites `split -> [reshape_0, ..., reshape_{K-1}]` into `reshape -> split`, replacing the `K`
/// individual reshapes with a single pre-split `stablehlo.reshape`. Returns the number of splits
/// sunk.
pub fn sink_split_through_reshape(module: &mut Module) -> usize {
    let mut sunk = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut single_consumer: IndexMap<ValueId, (OpId, usize)> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    let entry = single_consumer.entry(v).or_insert((op_id, 0));
                    entry.0 = op_id;
                    entry.1 += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for split_id in op_ids {
            if to_remove.contains(&split_id) {
                continue;
            }
            let Some(split_op) = module.ops.get(split_id).cloned() else {
                continue;
            };
            if split_op.name.to_string() != "mo.split"
                || split_op.operands.len() != 1
                || split_op.results.len() < 2
            {
                continue;
            }
            let num_splits = split_op.results.len();
            let split_in_val = split_op.operands[0];
            let Some(in_info) = module.values.get(split_in_val).cloned() else {
                continue;
            };
            let Type::Tensor(ref split_in_tt) = in_info.ty else {
                continue;
            };
            let Some(split_in_dims) = static_i64_dims(split_in_tt) else {
                continue;
            };

            let mut reshape_ids = Vec::with_capacity(num_splits);
            let mut split_out_dims_opt: Option<Vec<i64>> = None;
            let mut reshape_out_tt_opt: Option<TensorType> = None;
            let mut reshape_out_dims_opt: Option<Vec<i64>> = None;
            let mut reshape_results = Vec::with_capacity(num_splits);
            let mut valid = true;

            for &res_v in &split_op.results {
                let Some(&(consumer_id, count)) = single_consumer.get(&res_v) else {
                    valid = false;
                    break;
                };
                if count != 1 || !module.blocks[block_id].ops.contains(&consumer_id) {
                    valid = false;
                    break;
                }
                let Some(res_info) = module.values.get(res_v) else {
                    valid = false;
                    break;
                };
                let Type::Tensor(ref res_tt) = res_info.ty else {
                    valid = false;
                    break;
                };
                let Some(res_dims) = static_i64_dims(res_tt) else {
                    valid = false;
                    break;
                };
                if let Some(ref prev) = split_out_dims_opt {
                    if prev != &res_dims {
                        valid = false;
                        break;
                    }
                } else {
                    split_out_dims_opt = Some(res_dims);
                }

                let Some(r_op) = module.ops.get(consumer_id) else {
                    valid = false;
                    break;
                };
                if !matches!(
                    r_op.name.to_string().as_str(),
                    "stablehlo.reshape" | "mo.reshape"
                ) || r_op.operands.len() != 1
                    || r_op.results.len() != 1
                    || r_op.attributes.get("castforge.device")
                        != split_op.attributes.get("castforge.device")
                {
                    valid = false;
                    break;
                }
                let r_out_v = r_op.results[0];
                let Some(r_out_info) = module.values.get(r_out_v) else {
                    valid = false;
                    break;
                };
                let Type::Tensor(ref r_out_tt) = r_out_info.ty else {
                    valid = false;
                    break;
                };
                let Some(r_out_dims) = static_i64_dims(r_out_tt) else {
                    valid = false;
                    break;
                };
                if let Some(ref prev_out) = reshape_out_dims_opt {
                    if prev_out != &r_out_dims {
                        valid = false;
                        break;
                    }
                } else {
                    reshape_out_dims_opt = Some(r_out_dims);
                    reshape_out_tt_opt = Some(r_out_tt.clone());
                }
                reshape_ids.push(consumer_id);
                reshape_results.push(r_out_v);
            }
            if !valid {
                continue;
            }
            let (Some(split_out_dims), Some(reshape_out_dims), Some(reshape_out_tt)) =
                (split_out_dims_opt, reshape_out_dims_opt, reshape_out_tt_opt)
            else {
                continue;
            };
            if split_in_dims.len() != split_out_dims.len() {
                continue;
            }

            // Determine `split_axis` from attribute or shape difference.
            let attr_axis = split_op
                .attributes
                .get("axis")
                .or_else(|| split_op.attributes.get("split_dim"))
                .and_then(extract_single_int_attr)
                .and_then(|a| usize::try_from(a).ok());
            let inferred_axis = (0..split_in_dims.len()).find(|&d| {
                split_in_dims[d] == split_out_dims[d] * (num_splits as i64)
                    && (0..split_in_dims.len())
                        .all(|k| k == d || split_in_dims[k] == split_out_dims[k])
            });
            let Some(split_axis) = attr_axis.or(inferred_axis) else {
                continue;
            };
            if split_axis >= split_in_dims.len()
                || split_in_dims[split_axis] != split_out_dims[split_axis] * (num_splits as i64)
            {
                continue;
            }

            let Some(new_axis) =
                map_axis_through_static_reshape(&split_out_dims, &reshape_out_dims, split_axis)
            else {
                continue;
            };
            let mut pre_split_reshape_dims = reshape_out_dims.clone();
            pre_split_reshape_dims[new_axis] *= num_splits as i64;
            if analyze_static_reshape_dim_mapping(&split_in_dims, &pre_split_reshape_dims).is_none()
            {
                continue;
            }

            let pre_split_tt = TensorType {
                shape: pre_split_reshape_dims
                    .iter()
                    .copied()
                    .map(Dim::Static)
                    .collect(),
                dtype: reshape_out_tt.dtype.clone(),
                encoding: reshape_out_tt.encoding.clone(),
            };
            let pre_split_val = split_op.results[0];
            if let Some(v_mut) = module.values.get_mut(pre_split_val) {
                v_mut.ty = Type::Tensor(pre_split_tt);
                v_mut.def = ValueDef::OpResult {
                    op: split_id,
                    index: 0,
                };
            }
            for &dead_v in &split_op.results[1..] {
                module.values.remove(dead_v);
            }

            // Rewrite `split_id` in place into `stablehlo.reshape`.
            if let Some(op_mut) = module.ops.get_mut(split_id) {
                op_mut.name = OpName::new("stablehlo", "reshape");
                op_mut.attributes.shift_remove("axis");
                op_mut.attributes.shift_remove("split_dim");
                op_mut.attributes.shift_remove("split_sizes");
                op_mut.results = SmallVec::from_slice(&[pre_split_val]);
            }

            // Reuse `reshape_ids[0]` as the new `mo.split` producing all `reshape_results`.
            let new_split_id = reshape_ids[0];
            for (idx, &res_v) in reshape_results.iter().enumerate() {
                if let Some(v_mut) = module.values.get_mut(res_v) {
                    v_mut.def = ValueDef::OpResult {
                        op: new_split_id,
                        index: idx as u32,
                    };
                }
            }
            let split_sizes = vec![reshape_out_dims[new_axis]; num_splits];
            if let Some(new_split_op) = module.ops.get_mut(new_split_id) {
                new_split_op.name = OpName::new("mo", "split");
                new_split_op.operands = SmallVec::from_slice(&[pre_split_val]);
                new_split_op.results = SmallVec::from_vec(reshape_results);
                new_split_op
                    .attributes
                    .insert("axis".into(), Attribute::Int(new_axis as i64));
                new_split_op
                    .attributes
                    .insert("split_sizes".into(), Attribute::DenseIntArray(split_sizes));
            }
            for &dead_id in &reshape_ids[1..] {
                to_remove.insert(dead_id);
            }
            sunk += 1;
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

    sunk
}

/// Sinks or simplifies a sole-use broadcast (`mo.static.broadcast_to` or
/// `stablehlo.broadcast_in_dim`) feeding a single-axis keep-dims reduction (`mo.reduce_add`,
/// `mo.reduce_max`, `mo.reduce_min`, `stablehlo.reduce`).
///
/// - When reducing along a **non-broadcasted** axis, commutes `reduce(broadcast(x))` into
///   `broadcast(reduce(x, mapped_axis))`, reducing the smaller pre-broadcast tensor first.
/// - When reducing along a **broadcast-expanded** axis with an idempotent reduction
///   (`mo.reduce_max` or `mo.reduce_min`), folds `reduce(broadcast(x))` directly into a single
///   `broadcast(x)` to the reduced output shape.
///
/// Returns the number of broadcast-reduction pairs rewritten.
pub fn sink_broadcast_through_reduction(module: &mut Module) -> usize {
    let mut rewritten = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    *use_counts.entry(v).or_insert(0) += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for reduce_id in op_ids {
            if to_remove.contains(&reduce_id) {
                continue;
            }
            let Some(reduce_op) = module.ops.get(reduce_id).cloned() else {
                continue;
            };
            let reduce_name = reduce_op.name.to_string();
            if !matches!(
                reduce_name.as_str(),
                "mo.reduce_add" | "mo.reduce_max" | "mo.reduce_min" | "stablehlo.reduce"
            ) || reduce_op.operands.is_empty()
                || reduce_op.results.len() != 1
            {
                continue;
            }

            let bcast_out_val = reduce_op.operands[0];
            if use_counts.get(&bcast_out_val).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(bcast_out_info) = module.values.get(bcast_out_val).cloned() else {
                continue;
            };
            let ValueDef::OpResult { op: bcast_id, .. } = bcast_out_info.def else {
                continue;
            };
            if !module.blocks[block_id].ops.contains(&bcast_id) {
                continue;
            }
            let Some(bcast_op) = module.ops.get(bcast_id).cloned() else {
                continue;
            };
            let bcast_name = bcast_op.name.to_string();
            if !matches!(
                bcast_name.as_str(),
                "mo.static.broadcast_to" | "stablehlo.broadcast_in_dim"
            ) || bcast_op.operands.is_empty()
                || bcast_op.results.len() != 1
                || bcast_op.attributes.get("castforge.device")
                    != reduce_op.attributes.get("castforge.device")
            {
                continue;
            }

            let bcast_in_val = bcast_op.operands[0];
            let reduce_out_val = reduce_op.results[0];
            let (Some(in_info), Some(out_info)) = (
                module.values.get(bcast_in_val),
                module.values.get(reduce_out_val),
            ) else {
                continue;
            };
            let (Type::Tensor(in_tt), Type::Tensor(bcast_tt), Type::Tensor(out_tt)) = (
                in_info.ty.clone(),
                bcast_out_info.ty.clone(),
                out_info.ty.clone(),
            ) else {
                continue;
            };

            let rank_in = in_tt.shape.len();
            let rank_out = bcast_tt.shape.len();
            if rank_in == 0 || rank_out < rank_in || out_tt.shape.len() != rank_out {
                continue;
            }
            let extra_dims = rank_out - rank_in;
            if let Some(bcast_dims) = extract_int_list_attr(&bcast_op, "broadcast_dimensions") {
                let expected: Vec<i64> = (extra_dims as i64..rank_out as i64).collect();
                if bcast_dims != expected {
                    continue;
                }
            }

            let axis_i64 = reduce_op
                .attributes
                .get("axis")
                .or_else(|| reduce_op.attributes.get("dimensions"))
                .and_then(extract_single_int_attr);
            let Some(axis_raw) = axis_i64 else {
                continue;
            };
            let axis = if axis_raw < 0 {
                let adj = (rank_out as i64) + axis_raw;
                if adj < 0 {
                    continue;
                }
                adj as usize
            } else {
                axis_raw as usize
            };
            if axis >= rank_out || out_tt.shape[axis] != Dim::Static(1) {
                continue;
            }

            let is_broadcast_axis = axis < extra_dims
                || (in_tt.shape[axis - extra_dims] == Dim::Static(1)
                    && bcast_tt.shape[axis] != Dim::Static(1));

            if is_broadcast_axis {
                // Idempotent reductions (`reduce_max` / `reduce_min`) over a broadcast-expanded
                // dimension collapse to `broadcast(x) -> out_tt`.
                if !matches!(reduce_name.as_str(), "mo.reduce_max" | "mo.reduce_min") {
                    continue;
                }
                if let Some(v_mut) = module.values.get_mut(reduce_out_val) {
                    v_mut.def = ValueDef::OpResult {
                        op: bcast_id,
                        index: 0,
                    };
                }
                if let Some(b_mut) = module.ops.get_mut(bcast_id) {
                    b_mut.results[0] = reduce_out_val;
                }
                module.values.remove(bcast_out_val);
                to_remove.insert(reduce_id);
                rewritten += 1;
            } else {
                if rank_in <= 1 {
                    continue;
                }
                let mapped_axis = axis - extra_dims;
                let mut mid_shape = in_tt.shape.clone();
                mid_shape[mapped_axis] = Dim::Static(1);
                let mid_tt = TensorType {
                    shape: mid_shape,
                    dtype: out_tt.dtype.clone(),
                    encoding: out_tt.encoding.clone(),
                };

                if let Some(mid_val_mut) = module.values.get_mut(bcast_out_val) {
                    mid_val_mut.ty = Type::Tensor(mid_tt);
                }
                // Mutate `bcast_id` into the pre-broadcast reduction and `reduce_id` into the
                // post-reduction broadcast!
                let mut new_reduce_attrs = reduce_op.attributes.clone();
                let axis_key = if new_reduce_attrs.contains_key("dimensions")
                    && !new_reduce_attrs.contains_key("axis")
                {
                    "dimensions"
                } else {
                    "axis"
                };
                new_reduce_attrs.insert(axis_key.into(), Attribute::Int(mapped_axis as i64));
                let mut new_bcast_attrs = bcast_op.attributes.clone();
                if new_bcast_attrs.contains_key("shape") {
                    let out_dims: Option<Vec<i64>> = static_i64_dims(&out_tt);
                    if let Some(dims) = out_dims {
                        new_bcast_attrs.insert("shape".into(), Attribute::DenseIntArray(dims));
                    }
                }

                let mut extra_reduce_operands: SmallVec<[ValueId; 4]> =
                    SmallVec::from_slice(&[bcast_in_val]);
                for &extra_opd in &reduce_op.operands[1..] {
                    extra_reduce_operands.push(extra_opd);
                }

                if let Some(first_op) = module.ops.get_mut(bcast_id) {
                    first_op.name = reduce_op.name.clone();
                    first_op.operands = extra_reduce_operands;
                    first_op.attributes = new_reduce_attrs;
                }
                if let Some(second_op) = module.ops.get_mut(reduce_id) {
                    second_op.name = bcast_op.name.clone();
                    second_op.operands = SmallVec::from_slice(&[bcast_out_val]);
                    second_op.attributes = new_bcast_attrs;
                }
                rewritten += 1;
            }
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

    rewritten
}

/// Folds `mgp.extend_lifetimes` (or `castforge.extend_lifetimes`) operations into a preceding
/// sole-use asynchronous kernel execution (`mogg.async_exec`, `mgp.async_exec`,
/// `castforge.async_kernel`) or elides no-op lifetime extensions with zero buffer operands.
///
/// When folded into an async kernel launch, the extended buffer operands are deduplicated and
/// appended to the async kernel's operand list, `castforge.extended_lifetime_count` is stamped on
/// the async kernel, and the `extend_lifetimes` result chain is forwarded to the async kernel's
/// chain result. Returns the number of `extend_lifetimes` operations eliminated.
pub fn fold_extend_lifetimes_into_async_kernels(module: &mut Module) -> usize {
    let mut folded = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let mut use_counts: IndexMap<ValueId, usize> = IndexMap::new();
        for &op_id in &module.blocks[block_id].ops {
            if let Some(op) = module.ops.get(op_id) {
                for &v in &op.operands {
                    *use_counts.entry(v).or_insert(0) += 1;
                }
            }
        }

        let op_ids = module.blocks[block_id].ops.clone();
        let mut replacements: IndexMap<ValueId, ValueId> = IndexMap::new();
        let mut to_remove: IndexSet<OpId> = IndexSet::new();

        for op_id in op_ids {
            let Some(ext_op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !matches!(
                ext_op.name.to_string().as_str(),
                "mgp.extend_lifetimes" | "castforge.extend_lifetimes"
            ) || ext_op.operands.is_empty()
                || ext_op.results.len() != 1
            {
                continue;
            }
            let chain_in = replacements
                .get(&ext_op.operands[0])
                .copied()
                .unwrap_or(ext_op.operands[0]);
            let chain_out = ext_op.results[0];

            // Case 1: No-op `extend_lifetimes` with only the input chain and no buffers.
            if ext_op.operands.len() == 1 {
                replacements.insert(chain_out, chain_in);
                to_remove.insert(op_id);
                folded += 1;
                continue;
            }

            // Case 2: Preceding sole-use async kernel producing `chain_in`.
            if use_counts.get(&chain_in).copied().unwrap_or(0) != 1 {
                continue;
            }
            let Some(chain_info) = module.values.get(chain_in) else {
                continue;
            };
            let ValueDef::OpResult { op: async_id, .. } = chain_info.def else {
                continue;
            };
            if !module.blocks[block_id].ops.contains(&async_id) || to_remove.contains(&async_id) {
                continue;
            }
            let Some(async_op) = module.ops.get(async_id) else {
                continue;
            };
            if !matches!(
                async_op.name.to_string().as_str(),
                "mogg.async_exec" | "mgp.async_exec" | "castforge.async_kernel"
            ) {
                continue;
            }

            let extra_buffers: Vec<ValueId> = ext_op.operands[1..]
                .iter()
                .map(|v| replacements.get(v).copied().unwrap_or(*v))
                .collect();

            if let Some(async_mut) = module.ops.get_mut(async_id) {
                let mut added = 0i64;
                for buf in extra_buffers {
                    if !async_mut.operands.contains(&buf) {
                        async_mut.operands.push(buf);
                        added += 1;
                    }
                }
                let prev = async_mut
                    .attributes
                    .get("castforge.extended_lifetime_count")
                    .and_then(extract_single_int_attr)
                    .unwrap_or(0);
                async_mut.attributes.insert(
                    "castforge.extended_lifetime_count".into(),
                    Attribute::Int(prev + added),
                );
            }
            replacements.insert(chain_out, chain_in);
            to_remove.insert(op_id);
            folded += 1;
        }

        if !replacements.is_empty() {
            for &op_id in &module.blocks[block_id].ops {
                if to_remove.contains(&op_id) {
                    continue;
                }
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    for opd in &mut op_mut.operands {
                        while let Some(&target) = replacements.get(opd) {
                            *opd = target;
                        }
                    }
                }
            }
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

    folded
}

/// Specification of a single execution node in a multi-device collective execution stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveGraphNodeSpec {
    /// Unique identifier of the operation in topological order.
    pub node_id: usize,
    /// Sorted list of device indices (`0..num_devices`) participating in this operation.
    pub devices: Vec<usize>,
    /// True if this operation is a multi-device collective (`all_reduce`, `reduce_scatter`,
    /// `all_gather`, `broadcast`).
    pub is_collective: bool,
    /// True if this operation can be captured inside a hardware device graph (`CUDA Graph` /
    /// `HIP Graph`).
    pub is_graph_capturable: bool,
}

/// A partitioned execution stage produced by [`plan_device_graph_collective_partition`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceGraphStage {
    /// A synchronized hardware device-graph capture stage mapping each device index
    /// `0..num_devices` to the ordered list of `node_id`s captured on that device's stream.
    DeviceGraphCapture {
        /// Per-device captured `node_id` lists of length `num_devices`.
        device_subgraphs: Vec<Vec<usize>>,
    },
    /// A multi-device collective stage executed outside device-graph capture (when
    /// `capture_collectives_in_graph` is `false` or when only a subset of devices participate).
    EagerCollectiveStage {
        /// Ordered `node_id`s of collective operations in this barrier stage.
        node_ids: Vec<usize>,
    },
    /// Host or dynamic-shape operations that cannot be captured in a device graph.
    EagerHostStage {
        /// Ordered `node_id`s of non-capturable operations.
        node_ids: Vec<usize>,
    },
}

/// Partitions a sequence of multi-device operations into synchronized [`DeviceGraphStage`]
/// regions for `mgp.device.graph` capture and collective barrier scheduling.
///
/// - Non-capturable operations flush any active [`DeviceGraphStage::DeviceGraphCapture`] stage and
///   run in [`DeviceGraphStage::EagerHostStage`].
/// - Capturable non-collective operations accumulate into the active per-device subgraphs.
/// - Collective operations are captured inside [`DeviceGraphStage::DeviceGraphCapture`] only when
///   `capture_collectives_in_graph` is `true`, `is_graph_capturable` is `true`, and all
///   `0..num_devices` participate synchronously; otherwise they flush the open graph capture and
///   execute in [`DeviceGraphStage::EagerCollectiveStage`].
pub fn plan_device_graph_collective_partition(
    num_devices: usize,
    nodes: &[CollectiveGraphNodeSpec],
    capture_collectives_in_graph: bool,
) -> Option<Vec<DeviceGraphStage>> {
    if num_devices == 0 {
        return None;
    }
    let mut stages: Vec<DeviceGraphStage> = Vec::new();
    let mut current_capture: Option<Vec<Vec<usize>>> = None;

    let flush_capture = |stages: &mut Vec<DeviceGraphStage>,
                         current_capture: &mut Option<Vec<Vec<usize>>>| {
        if let Some(subgraphs) = current_capture.take()
            && subgraphs.iter().any(|sg| !sg.is_empty())
        {
            stages.push(DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: subgraphs,
            });
        }
    };

    for node in nodes {
        if node.devices.is_empty() || node.devices.iter().any(|&d| d >= num_devices) {
            return None;
        }
        let mut unique_devs = node.devices.clone();
        unique_devs.sort_unstable();
        unique_devs.dedup();
        if unique_devs.len() != node.devices.len() {
            return None;
        }

        if !node.is_graph_capturable {
            flush_capture(&mut stages, &mut current_capture);
            if let Some(DeviceGraphStage::EagerHostStage { node_ids }) = stages.last_mut() {
                node_ids.push(node.node_id);
            } else {
                stages.push(DeviceGraphStage::EagerHostStage {
                    node_ids: vec![node.node_id],
                });
            }
            continue;
        }

        if node.is_collective {
            let all_devices_participate = unique_devs.len() == num_devices;
            if capture_collectives_in_graph && all_devices_participate {
                let subgraphs =
                    current_capture.get_or_insert_with(|| vec![Vec::new(); num_devices]);
                for &d in &unique_devs {
                    subgraphs[d].push(node.node_id);
                }
            } else {
                flush_capture(&mut stages, &mut current_capture);
                if let Some(DeviceGraphStage::EagerCollectiveStage { node_ids }) = stages.last_mut()
                {
                    node_ids.push(node.node_id);
                } else {
                    stages.push(DeviceGraphStage::EagerCollectiveStage {
                        node_ids: vec![node.node_id],
                    });
                }
            }
        } else {
            let subgraphs = current_capture.get_or_insert_with(|| vec![Vec::new(); num_devices]);
            for &d in &unique_devs {
                subgraphs[d].push(node.node_id);
            }
        }
    }

    flush_capture(&mut stages, &mut current_capture);
    Some(stages)
}

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

fn is_horizontally_fusible_elementwise(op: &Operation) -> bool {
    is_sinkable_unary_elementwise(op) || is_sinkable_binary_elementwise(op)
}

fn op_transitively_depends_on(
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

fn ops_match_ignoring_device(a: &Operation, b: &Operation) -> bool {
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

// ============================================================================
// Multi-Tier Grouped Prefetch Planner, 2D Bitmap Grid & L1 Memory Affinity
// (inspired by LMCache `lmcache/v1/distributed/storage_controllers/prefetch_policy.py`
// and `lmcache/v1/distributed/storage_controllers/utils.py`)
// ============================================================================

/// Memory tag pair identifying an L1 manager's backing segment and residency
/// (e.g., `segment = "cpu"`, `residency = "pinned"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryTagSpec {
    /// Memory segment identifier (e.g., `"cpu"`, `"gpu"`, `"dax"`).
    pub segment: String,
    /// Residency mode (e.g., `"pinned"`, `"pageable"`, `"persistent"`).
    pub residency: String,
}

impl MemoryTagSpec {
    /// Creates a new [`MemoryTagSpec`].
    #[must_use]
    pub fn new(segment: impl Into<String>, residency: impl Into<String>) -> Self {
        Self {
            segment: segment.into(),
            residency: residency.into(),
        }
    }
}

/// Resolves the L1 manager ID whose memory tags match the L2 adapter's highest-priority
/// wanted [`MemoryTagSpec`], breaking ties by L1 write-placement order.
///
/// Pure function: returns `None` if no L1 manager matches any wanted tag.
#[must_use]
pub fn resolve_l1_memory_affinity(
    wanted_l1_tags: &[MemoryTagSpec],
    l1_managers_in_placement_order: &[(usize, MemoryTagSpec)],
) -> Option<usize> {
    for wanted in wanted_l1_tags {
        for (l1_id, tags) in l1_managers_in_placement_order {
            if tags == wanted {
                return Some(*l1_id);
            }
        }
    }
    None
}

/// Rectangular 2D bitmap grid where row `i` represents key group `i` and column `j`
/// represents chunk `j`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmap2D {
    rows: Vec<Vec<bool>>,
}

impl Bitmap2D {
    /// Creates a new `Bitmap2D` from rows of identical width.
    ///
    /// Returns `None` if rows have inconsistent column counts.
    #[must_use]
    pub fn new(rows: Vec<Vec<bool>>) -> Option<Self> {
        if let Some(first) = rows.first() {
            let width = first.len();
            if rows.iter().any(|r| r.len() != width) {
                return None;
            }
        }
        Some(Self { rows })
    }

    /// Creates an all-false `Bitmap2D` of shape `(num_rows, num_cols)`.
    #[must_use]
    pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
        Self {
            rows: vec![vec![false; num_cols]; num_rows],
        }
    }

    /// Returns `(num_rows, num_cols)`, or `(0, 0)` when empty.
    #[must_use]
    pub fn size(&self) -> (usize, usize) {
        if self.rows.is_empty() {
            (0, 0)
        } else {
            (self.rows.len(), self.rows[0].len())
        }
    }

    /// Returns the total number of set (`true`) bits across all rows.
    #[must_use]
    pub fn popcount(&self) -> usize {
        self.rows
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&b| b)
            .count()
    }

    /// Returns the underlying row bitmaps.
    #[must_use]
    pub fn rows(&self) -> &[Vec<bool>] {
        &self.rows
    }

    /// Row-wise bitwise OR (`self | other`). Returns `None` if shapes differ.
    #[must_use]
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x || y).collect())
            .collect();
        Some(Self { rows })
    }

    /// Row-wise bitwise AND (`self & other`). Returns `None` if shapes differ.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x && y).collect())
            .collect();
        Some(Self { rows })
    }

    /// Row-wise bitwise difference (`self & !other`). Returns `None` if shapes differ.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Option<Self> {
        if self.size() != other.size() {
            return None;
        }
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(a, b)| a.iter().zip(b).map(|(&x, &y)| x && !y).collect())
            .collect();
        Some(Self { rows })
    }
}

/// Maps tier manager/adapter index (`usize`) to a shared-shape [`Bitmap2D`] grid.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TierMapState {
    state: IndexMap<usize, Bitmap2D>,
    shape: Option<(usize, usize)>,
}

impl TierMapState {
    /// Creates an empty [`TierMapState`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a [`Bitmap2D`] for `tier_idx`. Rejects empty grids or shape mismatches.
    pub fn insert(&mut self, tier_idx: usize, grid: Bitmap2D) -> bool {
        let (r, c) = grid.size();
        if r == 0 {
            return false;
        }
        if let Some(expected) = self.shape {
            if expected != (r, c) {
                return false;
            }
        } else {
            self.shape = Some((r, c));
        }
        self.state.insert(tier_idx, grid);
        true
    }

    /// Gets the [`Bitmap2D`] for `tier_idx`.
    #[must_use]
    pub fn get(&self, tier_idx: usize) -> Option<&Bitmap2D> {
        self.state.get(&tier_idx)
    }

    /// Returns the number of populated tier indices.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.len()
    }

    /// Returns `true` if no tier indices are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    /// Unions the grids of all stored tier indices, or returns an all-zero `(num_rows, num_cols)`
    /// grid when empty.
    #[must_use]
    pub fn merge_or_zeros(&self, num_rows: usize, num_cols: usize) -> Bitmap2D {
        let mut merged = Bitmap2D::zeros(num_rows, num_cols);
        for grid in self.state.values() {
            if let Some(u) = merged.union(grid) {
                merged = u;
            }
        }
        merged
    }
}

/// Prefetch fetching mode (`"prefix"` vs `"full"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchFetchingPolicy {
    /// Prefix mode: applies `fold -> highest_set_bit -> unfold` with per-group sliding windows.
    Prefix,
    /// Full mode: requires all groups to use full attention (`window <= 0`) and fetches all hits.
    Full,
}

/// Multi-tier prefetch plan partitioning needed `(group, chunk)` cells across L1 managers
/// and L2 adapters so each cell is planned at most once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiTierPrefetchPlan {
    /// Cells served directly from resident L1 managers (`l1_id -> Bitmap2D`).
    pub l1_planned_keys: TierMapState,
    /// Cells loaded from L2 adapters into L1 (`l2_id -> Bitmap2D`).
    pub l2_planned_keys: TierMapState,
}

fn plan_tier_greedy(
    needed: &Bitmap2D,
    locked_keys: &TierMapState,
    sorted_indices: &[usize],
) -> (TierMapState, Bitmap2D) {
    let mut plan = TierMapState::new();
    let mut remaining = needed.clone();
    for &idx in sorted_indices {
        let Some(locked_grid) = locked_keys.get(idx) else {
            continue;
        };
        let Some(local) = remaining.intersect(locked_grid) else {
            continue;
        };
        if local.popcount() == 0 {
            continue;
        }
        if let Some(next_rem) = remaining.difference(&local) {
            remaining = next_rem;
        }
        plan.insert(idx, local);
    }
    (plan, remaining)
}

/// Plans multi-tier grouped prefetch loads across L1 managers and L2 adapters:
/// 1. Validates that `l1_locked_keys` and `l2_locked_keys` match
///    `(group_windows.len(), num_chunks)`.
/// 2. Computes `found = l1_found | l2_found`.
/// 3. Under [`PrefetchFetchingPolicy::Full`], returns an empty plan if any group has a positive
///    sliding window (`w > FULL_ATTENTION_WINDOW`); otherwise `needed = found`.
/// 4. Under [`PrefetchFetchingPolicy::Prefix`], runs [`fold_unfold_hybrid_prefix_bitmaps`] and
///    intersects `retain & found`.
/// 5. Greedily assigns `needed` cells to L1 managers (in ascending index order) first, then to
///    L2 adapters (in ascending index order), ensuring no cell is planned more than once.
#[must_use]
pub fn plan_multi_tier_prefetch_load(
    group_windows: &[i32],
    num_chunks: usize,
    l1_locked_keys: &TierMapState,
    l2_locked_keys: &TierMapState,
    l1_manager_ids: &[usize],
    l2_adapter_ids: &[usize],
    fetching_policy: PrefetchFetchingPolicy,
) -> MultiTierPrefetchPlan {
    let num_groups = group_windows.len();
    if num_groups == 0 || num_chunks == 0 {
        return MultiTierPrefetchPlan::default();
    }
    if l1_locked_keys
        .shape
        .is_some_and(|s| s != (num_groups, num_chunks))
        || l2_locked_keys
            .shape
            .is_some_and(|s| s != (num_groups, num_chunks))
    {
        return MultiTierPrefetchPlan::default();
    }

    let l1_found = l1_locked_keys.merge_or_zeros(num_groups, num_chunks);
    let l2_found = l2_locked_keys.merge_or_zeros(num_groups, num_chunks);
    let Some(found) = l1_found.union(&l2_found) else {
        return MultiTierPrefetchPlan::default();
    };

    let needed = match fetching_policy {
        PrefetchFetchingPolicy::Full => {
            if group_windows.iter().any(|&w| w > FULL_ATTENTION_WINDOW) {
                return MultiTierPrefetchPlan::default();
            }
            found
        }
        PrefetchFetchingPolicy::Prefix => {
            let (_hit_len, _servable, retain_rows) =
                fold_unfold_hybrid_prefix_bitmaps(found.rows(), group_windows);
            let Some(retain_grid) = Bitmap2D::new(retain_rows) else {
                return MultiTierPrefetchPlan::default();
            };
            let Some(masked) = retain_grid.intersect(&found) else {
                return MultiTierPrefetchPlan::default();
            };
            masked
        }
    };

    let mut sorted_l1 = l1_manager_ids.to_vec();
    sorted_l1.sort_unstable();
    sorted_l1.dedup();

    let mut sorted_l2 = l2_adapter_ids.to_vec();
    sorted_l2.sort_unstable();
    sorted_l2.dedup();

    let (l1_plan, rem_after_l1) = plan_tier_greedy(&needed, l1_locked_keys, &sorted_l1);
    let (l2_plan, _rem_after_l2) = plan_tier_greedy(&rem_after_l1, l2_locked_keys, &sorted_l2);

    MultiTierPrefetchPlan {
        l1_planned_keys: l1_plan,
        l2_planned_keys: l2_plan,
    }
}

/// Computes per-key L1 retention flags (`false` = temporary / delete after L1->L0 retrieve,
/// `true` = keep resident in L1 for reuse across requests).
#[must_use]
pub fn plan_l1_prefetch_retention(num_keys: usize, retain_in_l1: bool) -> Vec<bool> {
    vec![retain_in_l1; num_keys]
}

// ============================================================================
// Per-Salt Key Provider & L2 Authenticated KV Frame Codec
// (inspired by LMCache `lmcache/v1/distributed/serde/aesgcm.py` and `key_provider.py`)
// ============================================================================

/// Wire frame version byte for L2-at-rest authenticated KV chunks.
pub const L2_AEAD_VERSION: u8 = 1;

/// 96-bit (12-byte) nonce / initialization vector length per L2 frame.
pub const L2_AEAD_IV_LEN: usize = 12;

/// 128-bit (16-byte) authentication tag length per L2 frame.
pub const L2_AEAD_TAG_LEN: usize = 16;

/// Header length in bytes (`1B version + 12B IV`).
pub const L2_AEAD_HDR_LEN: usize = 1 + L2_AEAD_IV_LEN;

/// Total wire overhead in bytes (`1B version + 12B IV + 16B tag = 29B`).
pub const L2_AEAD_FRAME_OVERHEAD: usize = L2_AEAD_HDR_LEN + L2_AEAD_TAG_LEN;

/// Domain-separation prefix for L2 per-salt key derivation.
pub const L2_HKDF_INFO_PREFIX: &[u8] = b"lmcache-l2-aesgcm-v1";

fn keyed_splitmix_round(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn absorb_bytes_into_state(state: &mut [u64; 4], domain: u64, bytes: &[u8]) {
    state[0] = keyed_splitmix_round(state[0] ^ domain ^ (bytes.len() as u64));
    for (idx, chunk) in bytes.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let v = u64::from_le_bytes(word);
        let lane = idx & 3;
        state[lane] = keyed_splitmix_round(state[lane].wrapping_add(v ^ (idx as u64)));
        let next = (lane + 1) & 3;
        state[next] ^= state[lane].rotate_left(19);
    }
}

/// Memoizing per-`cache_salt` key provider deriving tenant-isolated `16`-byte (`128`-bit) or
/// `32`-byte (`256`-bit) data encryption keys (DEKs) from a master key and domain prefix.
#[derive(Debug, Clone)]
pub struct SaltKeyProvider {
    master_key: Vec<u8>,
    key_len: usize,
    info_prefix: Vec<u8>,
    cache: IndexMap<String, Vec<u8>>,
}

impl SaltKeyProvider {
    /// Creates a new [`SaltKeyProvider`].
    ///
    /// Returns `None` if `master_key` is empty or `aes_bits` is not `128` or `256`.
    #[must_use]
    pub fn new(master_key: Vec<u8>, aes_bits: usize, info_prefix: Vec<u8>) -> Option<Self> {
        if master_key.is_empty() || !matches!(aes_bits, 128 | 256) {
            return None;
        }
        Some(Self {
            master_key,
            key_len: aes_bits / 8,
            info_prefix,
            cache: IndexMap::new(),
        })
    }

    /// Returns the derived key length in bytes (`16` or `32`).
    #[must_use]
    pub fn key_len(&self) -> usize {
        self.key_len
    }

    /// Returns the number of distinct `cache_salt` entries currently memoized.
    #[must_use]
    pub fn cached_salt_count(&self) -> usize {
        self.cache.len()
    }

    /// Derives (or retrieves from the memoization cache) the tenant DEK for `cache_salt`.
    pub fn get_key(&mut self, cache_salt: &str) -> Vec<u8> {
        if let Some(existing) = self.cache.get(cache_salt) {
            return existing.clone();
        }
        let mut state = [
            0x243f_6a88_85a3_08d3,
            0x1319_8a2e_0370_7344,
            0xa409_3822_299f_31d0,
            0x082e_fa98_ec4e_6c89,
        ];
        absorb_bytes_into_state(&mut state, 0x01, &self.master_key);
        absorb_bytes_into_state(&mut state, 0x02, &self.info_prefix);
        absorb_bytes_into_state(&mut state, 0x03, cache_salt.as_bytes());

        let mut derived = Vec::with_capacity(self.key_len);
        let mut counter = 0u64;
        while derived.len() < self.key_len {
            let word = keyed_splitmix_round(
                state[(counter as usize) & 3]
                    .wrapping_add(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            );
            let bytes = word.to_le_bytes();
            let take = (self.key_len - derived.len()).min(8);
            derived.extend_from_slice(&bytes[..take]);
            counter += 1;
        }
        self.cache.insert(cache_salt.to_string(), derived.clone());
        derived
    }
}

/// Computes the exact serialized L2 frame size (`plaintext_bytes + 29`) from a slice of
/// `(num_elements, element_size_bytes)` tensor layout descriptors.
#[must_use]
pub fn estimate_l2_framed_size(layout_shapes_and_elem_bytes: &[(usize, usize)]) -> usize {
    let plaintext_bytes: usize = layout_shapes_and_elem_bytes
        .iter()
        .map(|&(n, elem_sz)| n.saturating_mul(elem_sz))
        .sum();
    plaintext_bytes.saturating_add(L2_AEAD_FRAME_OVERHEAD)
}

fn xor_keystream_in_place(dek: &[u8], iv: &[u8; L2_AEAD_IV_LEN], buf: &mut [u8]) {
    let mut state = [
        0x6a09_e667_f3bc_c908,
        0xbb67_ae85_84ca_a73b,
        0x3c6e_f372_fe94_f82b,
        0xa54f_f53a_5f1d_36f1,
    ];
    absorb_bytes_into_state(&mut state, 0x10, dek);
    absorb_bytes_into_state(&mut state, 0x20, iv);
    for (block_idx, chunk) in buf.chunks_mut(8).enumerate() {
        let ks = keyed_splitmix_round(
            state[block_idx & 3]
                .wrapping_add((block_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95)),
        )
        .to_le_bytes();
        for (b, &k) in chunk.iter_mut().zip(&ks) {
            *b ^= k;
        }
    }
}

fn compute_frame_tag(
    version: u8,
    dek: &[u8],
    iv: &[u8; L2_AEAD_IV_LEN],
    ciphertext: &[u8],
) -> [u8; L2_AEAD_TAG_LEN] {
    let mut state = [
        0x510e_527f_ade6_82d1,
        0x9b05_688c_2b3e_6c1f,
        0x1f83_d9ab_fb41_bd6b,
        0x5be0_cd19_137e_2179,
    ];
    absorb_bytes_into_state(&mut state, 0x30, &[version]);
    absorb_bytes_into_state(&mut state, 0x31, dek);
    absorb_bytes_into_state(&mut state, 0x32, iv);
    absorb_bytes_into_state(&mut state, 0x33, ciphertext);
    let w0 = keyed_splitmix_round(state[0] ^ state[2].rotate_left(17));
    let w1 = keyed_splitmix_round(state[1] ^ state[3].rotate_left(29));
    let mut tag = [0u8; L2_AEAD_TAG_LEN];
    tag[..8].copy_from_slice(&w0.to_le_bytes());
    tag[8..16].copy_from_slice(&w1.to_le_bytes());
    tag
}

/// Seals `plaintext` into an authenticated L2 KV wire frame
/// `[1B version = 1][12B iv][ciphertext || 16B tag]` keyed by `cache_salt`.
pub fn seal_l2_kv_frame(
    provider: &mut SaltKeyProvider,
    cache_salt: &str,
    iv: &[u8; L2_AEAD_IV_LEN],
    plaintext: &[u8],
) -> Vec<u8> {
    let dek = provider.get_key(cache_salt);
    let mut frame = Vec::with_capacity(plaintext.len() + L2_AEAD_FRAME_OVERHEAD);
    frame.push(L2_AEAD_VERSION);
    frame.extend_from_slice(iv);
    frame.extend_from_slice(plaintext);
    xor_keystream_in_place(&dek, iv, &mut frame[L2_AEAD_HDR_LEN..]);
    let tag = compute_frame_tag(L2_AEAD_VERSION, &dek, iv, &frame[L2_AEAD_HDR_LEN..]);
    frame.extend_from_slice(&tag);
    frame
}

/// Opens and authenticates an L2 KV wire frame keyed by `cache_salt`.
///
/// The ciphertext length is taken from `expected_plaintext_len` (allowing `src_blob` to have
/// trailing allocator padding beyond `13 + expected_plaintext_len + 16`). Returns `None` if the
/// header/version is malformed, `src_blob` is shorter than the frame, the wrong `cache_salt` is
/// supplied, or any byte of the header, ciphertext, or tag was tampered with.
pub fn open_l2_kv_frame(
    provider: &mut SaltKeyProvider,
    cache_salt: &str,
    src_blob: &[u8],
    expected_plaintext_len: usize,
) -> Option<Vec<u8>> {
    let frame_end = L2_AEAD_HDR_LEN
        .checked_add(expected_plaintext_len)?
        .checked_add(L2_AEAD_TAG_LEN)?;
    if src_blob.len() < frame_end || src_blob[0] != L2_AEAD_VERSION {
        return None;
    }
    let mut iv = [0u8; L2_AEAD_IV_LEN];
    iv.copy_from_slice(&src_blob[1..L2_AEAD_HDR_LEN]);
    let ct_end = L2_AEAD_HDR_LEN + expected_plaintext_len;
    let ciphertext = &src_blob[L2_AEAD_HDR_LEN..ct_end];
    let stored_tag = &src_blob[ct_end..frame_end];

    let dek = provider.get_key(cache_salt);
    let expected_tag = compute_frame_tag(L2_AEAD_VERSION, &dek, &iv, ciphertext);
    let mut diff = 0u8;
    for (&a, &b) in stored_tag.iter().zip(&expected_tag) {
        diff |= a ^ b;
    }
    if diff != 0 {
        return None;
    }
    let mut plaintext = ciphertext.to_vec();
    xor_keystream_in_place(&dek, &iv, &mut plaintext);
    Some(plaintext)
}

// ============================================================================
// Non-Aliasing Mutable Buffer Chain Sequence Relaxation
// (inspired by Modular `GraphCompiler/lib/MODialect/Transforms/CanonicalizeChainSequences.cpp`)
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct MutableChainOpInfo {
    in_chain: ValueId,
    out_chain: ValueId,
    buffer: ValueId,
    is_write: bool,
    is_multi_device: bool,
}

fn inspect_mutable_chain_op(op: &Operation) -> Option<MutableChainOpInfo> {
    let is_multi_device = matches!(
        op.attributes.get("multi_device"),
        Some(Attribute::Bool(true))
    );
    match op.name.to_string().as_str() {
        "mo.mutable.store" if op.operands.len() >= 3 && op.results.len() == 1 => {
            Some(MutableChainOpInfo {
                in_chain: op.operands[0],
                out_chain: op.results[0],
                buffer: op.operands[2],
                is_write: true,
                is_multi_device,
            })
        }
        "mo.mutable.load" if op.operands.len() >= 2 && op.results.len() == 2 => {
            Some(MutableChainOpInfo {
                in_chain: op.operands[0],
                out_chain: op.results[1],
                buffer: op.operands[1],
                is_write: false,
                is_multi_device,
            })
        }
        _ => None,
    }
}

fn can_safely_reorder_mutable_ops(a: &MutableChainOpInfo, b: &MutableChainOpInfo) -> bool {
    if a.is_multi_device || b.is_multi_device {
        return false;
    }
    if a.buffer != b.buffer {
        return true;
    }
    // Same buffer: safe to reorder iff neither operation writes to the buffer (read-read).
    !a.is_write && !b.is_write
}

fn prune_implied_chains_in_block(
    module: &Module,
    block_id: BlockId,
    chains: &mut SmallVec<[ValueId; 4]>,
) {
    if chains.len() <= 1 {
        return;
    }
    let mut implied: IndexSet<ValueId> = IndexSet::new();
    let mut worklist: Vec<ValueId> = Vec::new();

    for &start in chains.iter() {
        if let Some(ValueDef::OpResult { op: def_id, .. }) = module.values.get(start).map(|v| v.def)
            && module.blocks[block_id].ops.contains(&def_id)
            && let Some(def_op) = module.ops.get(def_id)
        {
            if let Some(info) = inspect_mutable_chain_op(def_op) {
                worklist.push(info.in_chain);
            } else if def_op.is_op("mo", "chain.create") {
                worklist.extend(def_op.operands.iter().copied());
            }
        }
    }

    while let Some(curr) = worklist.pop() {
        if !implied.insert(curr) {
            continue;
        }
        if let Some(ValueDef::OpResult { op: def_id, .. }) = module.values.get(curr).map(|v| v.def)
            && module.blocks[block_id].ops.contains(&def_id)
            && let Some(def_op) = module.ops.get(def_id)
        {
            if let Some(info) = inspect_mutable_chain_op(def_op) {
                worklist.push(info.in_chain);
            } else if def_op.is_op("mo", "chain.create") {
                worklist.extend(def_op.operands.iter().copied());
            }
        }
    }

    chains.retain(|v| !implied.contains(v));
}

/// Relaxes serialized chain dependencies between mutable buffer operations (`mo.mutable.store`,
/// `mo.mutable.load`) when neither operation is marked `multi_device = true` and their buffer
/// accesses do not conflict (distinct buffers, or read-read loads on the same buffer).
///
/// Rewritesboth direct `parent -> child` chain links and upstream `mo.chain.create` joins into
/// parallel fork-join diamonds merged via `mo.chain.create`. Returns the number of chain
/// relaxations performed.
pub fn relax_non_aliasing_mutable_chain_sequences(module: &mut Module) -> usize {
    let mut total_relaxed = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        for _iter in 0..64 {
            let mut changed = false;
            let op_ids = module.blocks[block_id].ops.clone();

            for (pos, &op_id) in op_ids.iter().enumerate() {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(mut_info) = inspect_mutable_chain_op(&op) else {
                    continue;
                };
                let Some(ValueDef::OpResult { op: parent_id, .. }) =
                    module.values.get(mut_info.in_chain).map(|v| v.def)
                else {
                    continue;
                };
                if !module.blocks[block_id].ops.contains(&parent_id) {
                    continue;
                }
                let Some(parent_op) = module.ops.get(parent_id).cloned() else {
                    continue;
                };

                // Case A: Direct mutable parent op in the same block.
                if let Some(parent_info) = inspect_mutable_chain_op(&parent_op)
                    && parent_info.out_chain == mut_info.in_chain
                    && can_safely_reorder_mutable_ops(&mut_info, &parent_info)
                {
                    let chain_ty = module.values[mut_info.out_chain].ty.clone();
                    if let Some(mut_op_mut) = module.ops.get_mut(op_id) {
                        mut_op_mut.operands[0] = parent_info.in_chain;
                    }
                    let merge_op_id = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mo", "chain.create"),
                        operands: SmallVec::from_slice(&[
                            mut_info.out_chain,
                            parent_info.out_chain,
                        ]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: op.location.clone(),
                    });
                    let merged_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: chain_ty,
                        def: ValueDef::OpResult {
                            op: merge_op_id,
                            index: 0,
                        },
                    });
                    module.ops[merge_op_id].results.push(merged_val);
                    module.blocks[block_id].ops.insert(pos + 1, merge_op_id);

                    for &other_id in &module.blocks[block_id].ops {
                        if other_id == merge_op_id {
                            continue;
                        }
                        if let Some(other_op) = module.ops.get_mut(other_id) {
                            for opd in &mut other_op.operands {
                                if *opd == mut_info.out_chain {
                                    *opd = merged_val;
                                }
                            }
                        }
                    }

                    total_relaxed += 1;
                    changed = true;
                    break;
                }

                // Case B: Upstream `mo.chain.create` with one or more non-aliasing mutable inputs.
                if parent_op.is_op("mo", "chain.create") {
                    let mut new_create_operands: SmallVec<[ValueId; 4]> = SmallVec::new();
                    let mut residual_operands: SmallVec<[ValueId; 4]> = SmallVec::new();

                    for &opd in &parent_op.operands {
                        let mut bypassed = false;
                        if let Some(ValueDef::OpResult { op: up_id, .. }) =
                            module.values.get(opd).map(|v| v.def)
                            && module.blocks[block_id].ops.contains(&up_id)
                            && let Some(up_op) = module.ops.get(up_id)
                            && let Some(up_info) = inspect_mutable_chain_op(up_op)
                            && up_info.out_chain == opd
                            && can_safely_reorder_mutable_ops(&mut_info, &up_info)
                        {
                            new_create_operands.push(opd);
                            if !residual_operands.contains(&up_info.in_chain) {
                                residual_operands.push(up_info.in_chain);
                            }
                            bypassed = true;
                        }
                        if !bypassed && !residual_operands.contains(&opd) {
                            residual_operands.push(opd);
                        }
                    }

                    if !new_create_operands.is_empty() && !residual_operands.is_empty() {
                        prune_implied_chains_in_block(module, block_id, &mut residual_operands);
                        let chain_ty = module.values[mut_info.out_chain].ty.clone();
                        let new_in_chain = if residual_operands.len() == 1 {
                            residual_operands[0]
                        } else {
                            let in_create_id = module.ops.insert_with_key(|id| Operation {
                                id,
                                name: OpName::new("mo", "chain.create"),
                                operands: residual_operands,
                                results: SmallVec::new(),
                                attributes: IndexMap::new(),
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: op.location.clone(),
                            });
                            let in_val = module.values.insert_with_key(|id| Value {
                                id,
                                ssa_name: None,
                                ty: chain_ty.clone(),
                                def: ValueDef::OpResult {
                                    op: in_create_id,
                                    index: 0,
                                },
                            });
                            module.ops[in_create_id].results.push(in_val);
                            module.blocks[block_id].ops.insert(pos, in_create_id);
                            in_val
                        };

                        if let Some(mut_op_mut) = module.ops.get_mut(op_id) {
                            mut_op_mut.operands[0] = new_in_chain;
                        }

                        new_create_operands.push(mut_info.out_chain);
                        let merge_op_id = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mo", "chain.create"),
                            operands: new_create_operands,
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: op.location.clone(),
                        });
                        let merged_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: chain_ty,
                            def: ValueDef::OpResult {
                                op: merge_op_id,
                                index: 0,
                            },
                        });
                        module.ops[merge_op_id].results.push(merged_val);
                        let mut_pos = module.blocks[block_id]
                            .ops
                            .iter()
                            .position(|&id| id == op_id)
                            .unwrap_or(pos);
                        module.blocks[block_id].ops.insert(mut_pos + 1, merge_op_id);

                        for &other_id in &module.blocks[block_id].ops {
                            if other_id == merge_op_id {
                                continue;
                            }
                            if let Some(other_op) = module.ops.get_mut(other_id) {
                                for opd in &mut other_op.operands {
                                    if *opd == mut_info.out_chain {
                                        *opd = merged_val;
                                    }
                                }
                            }
                        }

                        total_relaxed += 1;
                        changed = true;
                        break;
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }

    total_relaxed
}

// ============================================================================
// Post-Dominator `runsAfter` Dependency Tree & LCA Explicit Buffer Lifetimes
// (inspired by Modular `GraphCompiler/lib/MGPDialect/Dialect/MGPBufferLifetimeAnalysis.cpp`
// and `GraphCompiler/lib/MGPDialect/Transforms/ExplicitBufferLifetimes.cpp`)
// ============================================================================

/// DAG post-dominator dependency tree for a basic block, rooted at the block terminator.
///
/// Every operation's `runs_after` set contains itself, the block terminator, and all in-block
/// operations transitively consuming its results. Each operation's parent in the tree is its
/// immediate post-dominator: the strict successor `cand != op` in `runs_after[op]` with the
/// largest `runs_after[cand]` cardinality.
#[derive(Debug, Clone)]
pub struct BufferDependencyTree {
    root: OpId,
    parent_map: IndexMap<OpId, OpId>,
    children_map: IndexMap<OpId, Vec<OpId>>,
    depth_map: IndexMap<OpId, usize>,
    pre_index_map: IndexMap<OpId, usize>,
    post_index_map: IndexMap<OpId, usize>,
    op_to_idx: IndexMap<OpId, usize>,
    runs_after: Vec<IndexSet<usize>>,
}

impl BufferDependencyTree {
    /// Builds the [`BufferDependencyTree`] for `block_id`.
    ///
    /// Returns `None` if the block is empty or contains a dataflow cycle.
    #[must_use]
    pub fn build_for_block(module: &Module, block_id: BlockId) -> Option<Self> {
        let block_ops = &module.blocks.get(block_id)?.ops;
        if block_ops.is_empty() {
            return None;
        }
        let num_ops = block_ops.len();
        let terminator = *block_ops.last()?;
        let in_block_set: IndexSet<OpId> = block_ops.iter().copied().collect();

        let mut visited: IndexSet<OpId> = IndexSet::new();
        let mut temp_mark: IndexSet<OpId> = IndexSet::new();
        let mut topo_order: Vec<OpId> = Vec::with_capacity(num_ops);

        fn visit_dag(
            op_id: OpId,
            module: &Module,
            in_block_set: &IndexSet<OpId>,
            visited: &mut IndexSet<OpId>,
            temp_mark: &mut IndexSet<OpId>,
            topo_order: &mut Vec<OpId>,
        ) -> bool {
            if visited.contains(&op_id) {
                return true;
            }
            if !temp_mark.insert(op_id) {
                return false;
            }
            if let Some(op) = module.ops.get(op_id) {
                for &opd in &op.operands {
                    if let Some(ValueDef::OpResult { op: def_id, .. }) =
                        module.values.get(opd).map(|v| v.def)
                        && in_block_set.contains(&def_id)
                        && !visit_dag(def_id, module, in_block_set, visited, temp_mark, topo_order)
                    {
                        return false;
                    }
                }
            }
            temp_mark.swap_remove(&op_id);
            visited.insert(op_id);
            topo_order.push(op_id);
            true
        }

        for &op_id in block_ops.iter().rev() {
            if !visited.contains(&op_id)
                && !visit_dag(
                    op_id,
                    module,
                    &in_block_set,
                    &mut visited,
                    &mut temp_mark,
                    &mut topo_order,
                )
            {
                return None;
            }
        }

        let mut op_to_idx: IndexMap<OpId, usize> = IndexMap::with_capacity(num_ops);
        for (idx, &op_id) in topo_order.iter().enumerate() {
            op_to_idx.insert(op_id, idx);
        }

        let mut users: Vec<Vec<usize>> = vec![Vec::new(); num_ops];
        for (user_idx, &user_id) in topo_order.iter().enumerate() {
            if let Some(user_op) = module.ops.get(user_id) {
                for &opd in &user_op.operands {
                    if let Some(ValueDef::OpResult { op: def_id, .. }) =
                        module.values.get(opd).map(|v| v.def)
                        && let Some(&def_idx) = op_to_idx.get(&def_id)
                        && !users[def_idx].contains(&user_idx)
                    {
                        users[def_idx].push(user_idx);
                    }
                }
            }
        }

        let term_idx = *op_to_idx.get(&terminator)?;
        let mut runs_after: Vec<IndexSet<usize>> = Vec::with_capacity(num_ops);
        for i in 0..num_ops {
            let mut set = IndexSet::new();
            set.insert(i);
            set.insert(term_idx);
            runs_after.push(set);
        }

        for i in (0..num_ops).rev() {
            let user_indices = users[i].clone();
            for u_idx in user_indices {
                let u_set = runs_after[u_idx].clone();
                runs_after[i].extend(u_set);
            }
        }

        let mut parent_map: IndexMap<OpId, OpId> = IndexMap::new();
        let mut children_map: IndexMap<OpId, Vec<OpId>> = IndexMap::new();
        for &op_id in &topo_order {
            children_map.insert(op_id, Vec::new());
        }

        for (i, &op_id) in topo_order.iter().enumerate() {
            if op_id == terminator {
                continue;
            }
            let mut closest: Option<OpId> = None;
            let mut max_sz = 0usize;
            for &cand_idx in &runs_after[i] {
                if cand_idx == i {
                    continue;
                }
                let sz = runs_after[cand_idx].len();
                if sz > max_sz {
                    max_sz = sz;
                    closest = Some(topo_order[cand_idx]);
                }
            }
            let parent_id = closest.unwrap_or(terminator);
            parent_map.insert(op_id, parent_id);
            children_map.entry(parent_id).or_default().push(op_id);
        }

        let mut depth_map: IndexMap<OpId, usize> = IndexMap::new();
        let mut pre_index_map: IndexMap<OpId, usize> = IndexMap::new();
        let mut post_index_map: IndexMap<OpId, usize> = IndexMap::new();
        let mut stack: Vec<(OpId, bool, usize)> = vec![(terminator, true, 0)];
        let mut dfs_counter = 0usize;

        while let Some((node, is_first, depth)) = stack.pop() {
            if is_first {
                depth_map.insert(node, depth);
                pre_index_map.insert(node, dfs_counter);
                dfs_counter += 1;
                stack.push((node, false, depth));
                if let Some(children) = children_map.get(&node) {
                    for &child in children.iter().rev() {
                        stack.push((child, true, depth + 1));
                    }
                }
            } else {
                post_index_map.insert(node, dfs_counter);
                dfs_counter += 1;
            }
        }

        Some(Self {
            root: terminator,
            parent_map,
            children_map,
            depth_map,
            pre_index_map,
            post_index_map,
            op_to_idx,
            runs_after,
        })
    }

    /// Returns the root operation of the dependency tree (the block terminator).
    #[must_use]
    pub fn root(&self) -> OpId {
        self.root
    }

    /// Returns the immediate post-dominator parent of `op`, or `None` for the root.
    #[must_use]
    pub fn parent(&self, op: OpId) -> Option<OpId> {
        self.parent_map.get(&op).copied()
    }

    /// Returns the children of `op` in the dependency tree.
    #[must_use]
    pub fn children(&self, op: OpId) -> &[OpId] {
        self.children_map.get(&op).map_or(&[], Vec::as_slice)
    }

    /// Returns the tree depth of `op` (`0` at the terminator root).
    #[must_use]
    pub fn depth(&self, op: OpId) -> Option<usize> {
        self.depth_map.get(&op).copied()
    }

    /// Returns `(pre_index, post_index)` DFS numbering for `op`.
    #[must_use]
    pub fn dfs_interval(&self, op: OpId) -> Option<(usize, usize)> {
        Some((
            *self.pre_index_map.get(&op)?,
            *self.post_index_map.get(&op)?,
        ))
    }

    /// Returns `true` iff `a` strictly executes after `b` (`a != b` and `a` is in `runs_after[b]`).
    #[must_use]
    pub fn executes_after(&self, a: OpId, b: OpId) -> bool {
        let (Some(&idx_a), Some(&idx_b)) = (self.op_to_idx.get(&a), self.op_to_idx.get(&b)) else {
            return false;
        };
        if idx_a == idx_b {
            return false;
        }
        self.runs_after[idx_b].contains(&idx_a)
    }

    /// Returns `true` iff every operation in `a_ops` strictly executes after every operation in
    /// `b_ops`.
    #[must_use]
    pub fn all_execute_after(&self, a_ops: &[OpId], b_ops: &[OpId]) -> bool {
        a_ops
            .iter()
            .all(|&a| b_ops.iter().all(|&b| self.executes_after(a, b)))
    }

    /// Finds the Lowest Common Ancestor (LCA) of two operations in the post-dominator tree.
    #[must_use]
    pub fn find_lca_pair(&self, mut a: OpId, mut b: OpId) -> Option<OpId> {
        let mut depth_a = *self.depth_map.get(&a)?;
        let mut depth_b = *self.depth_map.get(&b)?;
        while depth_a > depth_b {
            a = *self.parent_map.get(&a)?;
            depth_a -= 1;
        }
        while depth_b > depth_a {
            b = *self.parent_map.get(&b)?;
            depth_b -= 1;
        }
        while a != b {
            a = *self.parent_map.get(&a)?;
            b = *self.parent_map.get(&b)?;
        }
        Some(a)
    }

    /// Finds the Lowest Common Ancestor (LCA) across a slice of operations.
    #[must_use]
    pub fn find_lca(&self, ops: &[OpId]) -> Option<OpId> {
        let mut lca = *ops.first()?;
        for &op in &ops[1..] {
            lca = self.find_lca_pair(lca, op)?;
        }
        Some(lca)
    }
}

fn is_alias_or_lifetime_extending_op(op: &Operation) -> bool {
    matches!(
        op.name.to_string().as_str(),
        "mgp.view"
            | "mgp.tensor.create"
            | "mgp.tensor.slice"
            | "mo.reshape"
            | "mo.slice"
            | "stablehlo.reshape"
            | "stablehlo.slice"
            | "mogg.async.extend_lifetime"
    )
}

fn op_chain_output(module: &Module, op_id: OpId) -> Option<ValueId> {
    let op = module.ops.get(op_id)?;
    for &res in &op.results {
        if let Some(tt) = module.values.get(res).and_then(|v| v.ty.as_tensor())
            && tt.shape == [Dim::Static(1)]
            && tt.dtype == DType::I64
        {
            return Some(res);
        }
    }
    None
}

/// Collects all direct and alias-propagated uses of an `mgp.buffer.alloc` operation within
/// `block_id`, ignoring metadata-only `mgp.tensor.extract_tensor_spec` consumers and returning
/// `None` if the buffer escapes into a control-flow region or `func.call`.
#[must_use]
pub fn collect_alloc_uses_in_block(
    module: &Module,
    block_id: BlockId,
    alloc_id: OpId,
) -> Option<Vec<OpId>> {
    let mut uses: Vec<OpId> = Vec::new();
    let mut stack: Vec<OpId> = vec![alloc_id];
    let mut visited_producers: IndexSet<OpId> = IndexSet::new();

    while let Some(top_id) = stack.pop() {
        if !visited_producers.insert(top_id) {
            continue;
        }
        let top_results = module.ops.get(top_id)?.results.clone();
        for &other_id in &module.blocks.get(block_id)?.ops {
            if other_id == top_id {
                continue;
            }
            let Some(other_op) = module.ops.get(other_id) else {
                continue;
            };
            if !other_op.operands.iter().any(|v| top_results.contains(v)) {
                continue;
            }
            if !other_op.is_op("mgp", "tensor.extract_tensor_spec") && !uses.contains(&other_id) {
                uses.push(other_id);
            }
            if !other_op.regions.is_empty() || other_op.is_op("func", "call") {
                return None;
            }
            if is_alias_or_lifetime_extending_op(other_op) {
                stack.push(other_id);
            }
        }
    }

    if uses.is_empty() { None } else { Some(uses) }
}

/// Inserts explicit `mgp.buffer.free` operations for `mgp.buffer.alloc` operations at the
/// earliest post-dominating ancestor in [`BufferDependencyTree`] that produces a chain output
/// (`tensor<1xi64>`) after all direct and aliasing uses of the allocation have completed.
///
/// Returns the number of `mgp.buffer.free` operations inserted.
pub fn insert_explicit_buffer_frees(module: &mut Module) -> usize {
    let mut frees_added = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let Some(tree) = BufferDependencyTree::build_for_block(module, block_id) else {
            continue;
        };
        let alloc_ids: Vec<OpId> = module.blocks[block_id]
            .ops
            .iter()
            .copied()
            .filter(|&id| {
                module
                    .ops
                    .get(id)
                    .is_some_and(|op| op.is_op("mgp", "buffer.alloc") && !op.results.is_empty())
            })
            .collect();

        for alloc_id in alloc_ids {
            let Some(uses) = collect_alloc_uses_in_block(module, block_id, alloc_id) else {
                continue;
            };
            let Some(mut lca_id) = tree.find_lca(&uses) else {
                continue;
            };
            while op_chain_output(module, lca_id).is_none() {
                let Some(parent_id) = tree.parent(lca_id) else {
                    break;
                };
                lca_id = parent_id;
            }
            let Some(chain_out) = op_chain_output(module, lca_id) else {
                continue;
            };
            let out_buffer = module.ops[alloc_id].results[0];
            let chain_ty = module.values[chain_out].ty.clone();
            let loc = module.ops[lca_id].location.clone();

            let free_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "buffer.free"),
                operands: SmallVec::from_slice(&[chain_out, out_buffer]),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            let free_chain_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::OpResult {
                    op: free_op_id,
                    index: 0,
                },
            });
            module.ops[free_op_id].results.push(free_chain_val);

            let lca_pos = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == lca_id)
                .unwrap_or(0);
            module.blocks[block_id].ops.insert(lca_pos + 1, free_op_id);

            for &other_id in &module.blocks[block_id].ops {
                if other_id == free_op_id {
                    continue;
                }
                if let Some(other_op) = module.ops.get_mut(other_id) {
                    for opd in &mut other_op.operands {
                        if *opd == chain_out {
                            *opd = free_chain_val;
                        }
                    }
                }
            }
            frees_added += 1;
        }
    }

    frees_added
}

// ============================================================================
// Assert-Dependency Chain Predication Pass
// (inspired by Modular
//  `GraphCompiler/lib/MGPDialect/Transforms/PredicateAssertDependenciesPass.cpp`)
// ============================================================================

fn is_chain_value(module: &Module, val_id: ValueId) -> bool {
    module
        .values
        .get(val_id)
        .and_then(|v| v.ty.as_tensor())
        .is_some_and(|tt| tt.shape == [Dim::Static(1)] && tt.dtype == DType::I64)
}

fn is_assert_op(op: &Operation) -> bool {
    matches!(op.name.to_string().as_str(), "mgp.assert" | "mo.assert") && !op.results.is_empty()
}

fn input_chain_operand_index(module: &Module, op: &Operation) -> Option<usize> {
    if op.is_op("func", "return") || op.is_op("mo", "chain.create") {
        return None;
    }
    op.operands.iter().position(|&v| is_chain_value(module, v))
}

/// Predicates every chain-taking operation in a block that consumes (directly or transitively
/// through pure dataflow ops) a value feeding an `mgp.assert` or `mo.assert` operation on that
/// assertion's output chain.
///
/// Stops downward propagation at the first chain-taking operation along each dataflow path so
/// downstream chain-ordered descendants inherit the assertion dependency transitively without
/// redundant barriers. Returns the number of operations predicated on additional assertion chains.
pub fn predicate_assert_dependencies(module: &mut Module) -> usize {
    let mut predicated_ops = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let block_ops = module.blocks[block_id].ops.clone();
        let in_block: IndexSet<OpId> = block_ops.iter().copied().collect();

        let assert_ids: Vec<OpId> = block_ops
            .iter()
            .copied()
            .filter(|&id| module.ops.get(id).is_some_and(is_assert_op))
            .collect();
        if assert_ids.is_empty() {
            continue;
        }

        // 1. Upward propagation: find all non-chain values and defining ops that feed each assert.
        let mut values_that_feed_asserts: IndexMap<ValueId, IndexSet<OpId>> = IndexMap::new();
        for &assert_id in &assert_ids {
            let mut visited_ops: IndexSet<OpId> = IndexSet::new();
            let mut stack: Vec<ValueId> = module.ops[assert_id]
                .operands
                .iter()
                .copied()
                .filter(|&v| !is_chain_value(module, v))
                .collect();

            while let Some(val) = stack.pop() {
                let Some(ValueDef::OpResult { op: def_id, .. }) =
                    module.values.get(val).map(|v| v.def)
                else {
                    values_that_feed_asserts
                        .entry(val)
                        .or_default()
                        .insert(assert_id);
                    continue;
                };
                if !in_block.contains(&def_id) {
                    values_that_feed_asserts
                        .entry(val)
                        .or_default()
                        .insert(assert_id);
                    continue;
                }
                if !visited_ops.insert(def_id) {
                    continue;
                }
                let def_op = &module.ops[def_id];
                for &res in &def_op.results {
                    if !is_chain_value(module, res) {
                        values_that_feed_asserts
                            .entry(res)
                            .or_default()
                            .insert(assert_id);
                    }
                }
                for &opd in &def_op.operands {
                    if !is_chain_value(module, opd) {
                        stack.push(opd);
                    }
                }
            }
        }

        // 2. Downward propagation: propagate assertion sets to consumers, stopping at the first
        //    chain-taking op along each path.
        let mut ops_depending_on_asserts: IndexMap<OpId, IndexSet<OpId>> = IndexMap::new();
        let seed_entries: Vec<(ValueId, IndexSet<OpId>)> = values_that_feed_asserts
            .iter()
            .map(|(&v, s)| (v, s.clone()))
            .collect();

        for (seed_val, asserts) in seed_entries {
            let mut worklist: Vec<ValueId> = vec![seed_val];
            let mut visited_vals: IndexSet<ValueId> = IndexSet::new();

            while let Some(val) = worklist.pop() {
                if is_chain_value(module, val) || !visited_vals.insert(val) {
                    continue;
                }
                for &user_id in &block_ops {
                    if assert_ids.contains(&user_id) {
                        continue;
                    }
                    let Some(user_op) = module.ops.get(user_id) else {
                        continue;
                    };
                    if !user_op.operands.contains(&val) {
                        continue;
                    }
                    let entry = ops_depending_on_asserts.entry(user_id).or_default();
                    let before = entry.len();
                    entry.extend(asserts.iter().copied());
                    if entry.len() == before {
                        continue;
                    }
                    if input_chain_operand_index(module, user_op).is_some() {
                        // Chain-taking op is predicated here; its chain descendants inherit it.
                        continue;
                    }
                    for &res in &user_op.results {
                        if !is_chain_value(module, res) {
                            worklist.push(res);
                        }
                    }
                }
            }
        }

        // 3. Predicate each chain-taking op on `asserts_it_depends_on \ asserts_it_contributed_to`.
        for (op_id, depended_asserts) in ops_depending_on_asserts {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let Some(chain_opd_idx) = input_chain_operand_index(module, &op) else {
                continue;
            };
            let mut diff = depended_asserts;
            if let Some(&first_res) = op.results.first()
                && let Some(contributed) = values_that_feed_asserts.get(&first_res)
            {
                diff.retain(|a| !contributed.contains(a));
            }
            if diff.is_empty() {
                continue;
            }

            let orig_chain = op.operands[chain_opd_idx];
            let mut join_operands: SmallVec<[ValueId; 4]> = SmallVec::from_slice(&[orig_chain]);
            for assert_id in diff {
                let assert_chain = module.ops[assert_id].results[0];
                if !join_operands.contains(&assert_chain) {
                    join_operands.push(assert_chain);
                }
            }
            if join_operands.len() <= 1 {
                continue;
            }

            let chain_ty = module.values[orig_chain].ty.clone();
            let join_op_id = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mo", "chain.create"),
                operands: join_operands,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: op.location.clone(),
            });
            let join_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::OpResult {
                    op: join_op_id,
                    index: 0,
                },
            });
            module.ops[join_op_id].results.push(join_val);
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.operands[chain_opd_idx] = join_val;
            }
            let pos = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == op_id)
                .unwrap_or(0);
            module.blocks[block_id].ops.insert(pos, join_op_id);
            predicated_ops += 1;
        }
    }

    predicated_ops
}

// ============================================================================
// Polynomial Rolling-Hash Window Matcher & Coordinator Blend Content Index
// (inspired by LMCache `lmcache/v1/mp_coordinator/blend_index.py` and
//  `lmcache/v1/multiprocess/token_hasher.py`)
// ============================================================================

/// Fleet-constant 64-bit odd polynomial base for CacheBlend rolling-hash content indexing.
pub const BLEND_POLY_BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Occupancy filter growth factor (`16` slots per live distinct content).
pub const BLEND_INDEX_TABLE_GROWTH: usize = 16;

/// Minimum power-of-two slot count (`1024`) for the occupancy filter.
pub const BLEND_INDEX_MIN_TABLE_SIZE: usize = 1 << 10;

/// Computes sliding-window 64-bit polynomial rolling hashes across `tokens` for window size
/// `window_k` in $O(n)$ time using natural $\bmod 2^{64}$ arithmetic:
/// $$H([x_0, \dots, x_{k-1}]) = \sum_{i=0}^{k-1} x_i \cdot \text{base}^{k - 1 - i} \pmod{2^{64}}$$
#[must_use]
pub fn rolling_hash_windows_u64(tokens: &[u64], window_k: usize, base: u64) -> Vec<u64> {
    if window_k == 0 || tokens.len() < window_k {
        return Vec::new();
    }
    let n = tokens.len();
    let mut out = Vec::with_capacity(n - window_k + 1);

    let mut power = 1u64;
    for _ in 0..(window_k - 1) {
        power = power.wrapping_mul(base);
    }

    let mut h = 0u64;
    for &tok in &tokens[..window_k] {
        h = h.wrapping_mul(base).wrapping_add(tok);
    }
    out.push(h);

    for i in window_k..n {
        let old = tokens[i - window_k];
        let new = tokens[i];
        h = h.wrapping_sub(old.wrapping_mul(power));
        h = h.wrapping_mul(base).wrapping_add(new);
        out.push(h);
    }
    out
}

/// Computes non-overlapping chunk polynomial hashes of length `tokens.len() / window_k`.
#[must_use]
pub fn chunk_hash_windows_u64(tokens: &[u64], window_k: usize, base: u64) -> Vec<u64> {
    if window_k == 0 || tokens.len() < window_k {
        return Vec::new();
    }
    let num_windows = tokens.len() / window_k;
    let mut out = Vec::with_capacity(num_windows);
    for w in 0..num_windows {
        let start = w * window_k;
        let mut h = 0u64;
        for &tok in &tokens[start..start + window_k] {
            h = h.wrapping_mul(base).wrapping_add(tok);
        }
        out.push(h);
    }
    out
}

/// Point-in-time O(1) summary of [`BlendContentFingerprintIndex`] contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendIndexStats {
    /// Distinct chunk token contents indexed.
    pub num_contents: usize,
    /// Total chunk hashes indexed across those contents.
    pub num_chunks: usize,
    /// Total `(chunk_hash, namespace)` claims across those chunks.
    pub num_claims: usize,
    /// Distinct namespaces currently holding indexed content.
    pub num_namespaces: usize,
    /// Number of slots in the power-of-two occupancy filter.
    pub table_size: usize,
}

/// Verified token-exact fragment match returned by [`BlendContentFingerprintIndex::match_tokens`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendFragmentMatch {
    /// Matched chunk's 32-byte `chunk_hash`.
    pub chunk_hash: [u8; 32],
    /// Token offset of the chunk in its originally stored sequence.
    pub old_st: usize,
    /// Starting token index of the matched window in the query sequence.
    pub cur_st: usize,
}

#[derive(Debug, Clone)]
struct BlendOccupant {
    token_offset: usize,
    namespaces: IndexSet<String>,
}

#[derive(Debug, Clone)]
struct BlendFingerprintEntry {
    token_ids: Vec<u64>,
    occupants: IndexMap<[u8; 32], BlendOccupant>,
}

/// Content fingerprint index over the coordinator key directory's token bindings for CacheBlend
/// fragment lookup.
#[derive(Debug, Clone)]
pub struct BlendContentFingerprintIndex {
    chunk_size: usize,
    probe_stride: usize,
    fingerprint_table: IndexMap<u64, BlendFingerprintEntry>,
    slots: Vec<u8>,
    mask: usize,
    bits_set: usize,
    num_chunks: usize,
    claims_by_namespace: IndexMap<String, usize>,
}

impl BlendContentFingerprintIndex {
    /// Creates an empty [`BlendContentFingerprintIndex`].
    ///
    /// Returns `None` if `chunk_size == 0` or `probe_stride == 0`.
    #[must_use]
    pub fn new(chunk_size: usize, probe_stride: usize) -> Option<Self> {
        if chunk_size == 0 || probe_stride == 0 {
            return None;
        }
        Some(Self {
            chunk_size,
            probe_stride,
            fingerprint_table: IndexMap::new(),
            slots: vec![0u8; BLEND_INDEX_MIN_TABLE_SIZE],
            mask: BLEND_INDEX_MIN_TABLE_SIZE - 1,
            bits_set: 0,
            num_chunks: 0,
            claims_by_namespace: IndexMap::new(),
        })
    }

    fn fingerprint(&self, token_ids: &[u64]) -> u64 {
        chunk_hash_windows_u64(token_ids, self.chunk_size, BLEND_POLY_BASE)[0]
    }

    fn adjust_claim_count(&mut self, namespace: &str, delta: isize) {
        let current = self
            .claims_by_namespace
            .get(namespace)
            .copied()
            .unwrap_or(0) as isize;
        let next = current + delta;
        if next > 0 {
            self.claims_by_namespace
                .insert(namespace.to_string(), next as usize);
        } else {
            self.claims_by_namespace.shift_remove(namespace);
        }
    }

    fn rebuild_table(&mut self) {
        let mut size = BLEND_INDEX_MIN_TABLE_SIZE;
        while size < BLEND_INDEX_TABLE_GROWTH.saturating_mul(self.fingerprint_table.len()) {
            size <<= 1;
        }
        self.slots = vec![0u8; size];
        self.mask = size - 1;
        let mut bits = 0usize;
        for &poly in self.fingerprint_table.keys() {
            let slot = (poly as usize) & self.mask;
            if self.slots[slot] == 0 {
                self.slots[slot] = 1;
                bits += 1;
            }
        }
        self.bits_set = bits;
    }

    /// Records that `namespace` stores `chunk_hash` with `token_ids` at `token_offset`.
    ///
    /// Ignores content whose length does not equal `chunk_size`.
    pub fn add(
        &mut self,
        token_ids: &[u64],
        chunk_hash: [u8; 32],
        token_offset: usize,
        namespace: &str,
    ) {
        if token_ids.len() != self.chunk_size {
            return;
        }
        let poly = self.fingerprint(token_ids);
        let is_new_fp = !self.fingerprint_table.contains_key(&poly);
        if is_new_fp {
            self.fingerprint_table.insert(
                poly,
                BlendFingerprintEntry {
                    token_ids: token_ids.to_vec(),
                    occupants: IndexMap::new(),
                },
            );
            let slot = (poly as usize) & self.mask;
            if self.slots[slot] == 0 {
                self.slots[slot] = 1;
                self.bits_set += 1;
            }
            if BLEND_INDEX_TABLE_GROWTH.saturating_mul(self.fingerprint_table.len())
                > self.slots.len()
            {
                self.rebuild_table();
            }
        }

        let entry = self.fingerprint_table.get_mut(&poly).unwrap();
        if let Some(occ) = entry.occupants.get_mut(&chunk_hash) {
            occ.token_offset = token_offset;
            if occ.namespaces.insert(namespace.to_string()) {
                self.adjust_claim_count(namespace, 1);
            }
        } else {
            let mut namespaces = IndexSet::new();
            namespaces.insert(namespace.to_string());
            entry.occupants.insert(
                chunk_hash,
                BlendOccupant {
                    token_offset,
                    namespaces,
                },
            );
            self.num_chunks += 1;
            self.adjust_claim_count(namespace, 1);
        }
    }

    /// Drops `namespace`'s claim on `chunk_hash` under `token_ids`.
    pub fn remove_claim(&mut self, token_ids: &[u64], chunk_hash: &[u8; 32], namespace: &str) {
        if token_ids.len() != self.chunk_size {
            return;
        }
        let poly = self.fingerprint(token_ids);
        let Some(entry) = self.fingerprint_table.get_mut(&poly) else {
            return;
        };
        let Some(occ) = entry.occupants.get_mut(chunk_hash) else {
            return;
        };
        if !occ.namespaces.shift_remove(namespace) {
            return;
        }
        let occ_empty = occ.namespaces.is_empty();
        if occ_empty {
            entry.occupants.shift_remove(chunk_hash);
        }
        let entry_empty = entry.occupants.is_empty();
        self.adjust_claim_count(namespace, -1);
        if !occ_empty {
            return;
        }
        self.num_chunks -= 1;
        if !entry_empty {
            return;
        }
        self.fingerprint_table.shift_remove(&poly);
        if self.bits_set > 2 * self.fingerprint_table.len() {
            self.rebuild_table();
        }
    }

    /// Drops `chunk_hash` and every namespace's claim on it under `token_ids`.
    pub fn remove_chunk(&mut self, token_ids: &[u64], chunk_hash: &[u8; 32]) {
        if token_ids.len() != self.chunk_size {
            return;
        }
        let poly = self.fingerprint(token_ids);
        let Some(entry) = self.fingerprint_table.get_mut(&poly) else {
            return;
        };
        let Some(occ) = entry.occupants.shift_remove(chunk_hash) else {
            return;
        };
        self.num_chunks -= 1;
        let namespaces: Vec<String> = occ.namespaces.into_iter().collect();
        for ns in namespaces {
            self.adjust_claim_count(&ns, -1);
        }
        if self.fingerprint_table[&poly].occupants.is_empty() {
            self.fingerprint_table.shift_remove(&poly);
            if self.bits_set > 2 * self.fingerprint_table.len() {
                self.rebuild_table();
            }
        }
    }

    /// Finds chunks `namespace` can retrieve that are contained in `tokens`, verified token-exact,
    /// returned in ascending `cur_st` order (at most one match per `chunk_hash`).
    #[must_use]
    pub fn match_tokens(&self, tokens: &[u64], namespace: &str) -> Vec<BlendFragmentMatch> {
        if tokens.len() < self.chunk_size {
            return Vec::new();
        }
        let rolling = rolling_hash_windows_u64(tokens, self.chunk_size, BLEND_POLY_BASE);
        let mut matches = Vec::new();
        let mut seen: IndexSet<[u8; 32]> = IndexSet::new();

        for (pos, &fp) in rolling.iter().step_by(self.probe_stride).enumerate() {
            let slot = (fp as usize) & self.mask;
            if self.slots[slot] == 0 {
                continue;
            }
            let Some(entry) = self.fingerprint_table.get(&fp) else {
                continue;
            };
            let cur_st = pos * self.probe_stride;
            for (&chunk_hash, occ) in &entry.occupants {
                if seen.contains(&chunk_hash) || !occ.namespaces.contains(namespace) {
                    continue;
                }
                if tokens[cur_st..cur_st + self.chunk_size] != entry.token_ids[..] {
                    break;
                }
                seen.insert(chunk_hash);
                matches.push(BlendFragmentMatch {
                    chunk_hash,
                    old_st: occ.token_offset,
                    cur_st,
                });
                break;
            }
        }
        matches
    }

    /// Returns a point-in-time O(1) summary of index contents.
    #[must_use]
    pub fn stats(&self) -> BlendIndexStats {
        BlendIndexStats {
            num_contents: self.fingerprint_table.len(),
            num_chunks: self.num_chunks,
            num_claims: self.claims_by_namespace.values().sum(),
            num_namespaces: self.claims_by_namespace.len(),
            table_size: self.slots.len(),
        }
    }
}

// ============================================================================
// Batch 17 Step 2: FP8 E4M3FN Quantization Codec & Multi-Output Serde Group Planner
// ============================================================================

/// Maximum finite positive value representable in `float8_e4m3fn` (`(1 + 6/8) * 2^8 = 448.0`).
pub const FP8_E4M3FN_MAX_FINITE: f32 = 448.0;

/// Canonical NaN bit-pattern for `float8_e4m3fn` (`s=0, exp=0b1111, mant=0b111`).
pub const FP8_E4M3FN_NAN: u8 = 0x7F;

/// Converts an `f32` scalar to an 8-bit `float8_e4m3fn` byte (1 sign bit, 4-bit exponent with
/// bias 7, 3-bit mantissa, finite-only with `exp=15, mant=7` reserved for NaN and saturating
/// finite magnitudes above `448.0` to `±448.0`).
#[must_use]
pub fn f32_to_fp8_e4m3fn(v: f32) -> u8 {
    if v.is_nan() {
        return FP8_E4M3FN_NAN;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let abs_v = v.abs();
    if abs_v == 0.0 {
        return sign;
    }
    if abs_v >= FP8_E4M3FN_MAX_FINITE {
        return sign | 0x7E;
    }
    // Subnormal range: exp_unbiased = -6, step = 2^-9 = 1/512.
    let min_normal = 1.0f32 / 64.0f32; // 2^-6
    if abs_v < min_normal {
        let scaled = abs_v * 512.0f32;
        let floor = scaled.floor();
        let diff = scaled - floor;
        let mut mant = floor as u8;
        if diff > 0.5 || (diff == 0.5 && (mant & 1) == 1) {
            mant += 1;
        }
        return sign | mant;
    }
    let exp_unbiased = abs_v.log2().floor() as i32;
    let exp_biased = (exp_unbiased + 7) as u8;
    let norm = abs_v / (2.0f32).powi(exp_unbiased) - 1.0f32;
    let scaled_m = norm * 8.0f32;
    let floor_m = scaled_m.floor();
    let diff_m = scaled_m - floor_m;
    let mut mant = floor_m as u8;
    if diff_m > 0.5 || (diff_m == 0.5 && (mant & 1) == 1) {
        mant += 1;
    }
    let (mut e, mut m) = (exp_biased, mant);
    if m == 8 {
        m = 0;
        e += 1;
    }
    if e > 15 || (e == 15 && m >= 7) {
        return sign | 0x7E;
    }
    sign | (e << 3) | m
}

/// Converts an 8-bit `float8_e4m3fn` byte back into an `f32` scalar.
#[must_use]
pub fn fp8_e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }
    if exp == 0 {
        if mant == 0 {
            return if sign < 0.0 { -0.0 } else { 0.0 };
        }
        return sign * (f32::from(mant) / 512.0f32);
    }
    let scale = (2.0f32).powi(i32::from(exp) - 7);
    sign * (1.0f32 + f32::from(mant) / 8.0f32) * scale
}

/// Quantizes an `f32` slice into a contiguous 1-byte-per-element `float8_e4m3fn` buffer.
#[must_use]
pub fn quantize_f32_slice_to_fp8_e4m3fn(src: &[f32]) -> Vec<u8> {
    src.iter().copied().map(f32_to_fp8_e4m3fn).collect()
}

/// Dequantizes a contiguous `float8_e4m3fn` byte slice into `f32` elements.
#[must_use]
pub fn dequantize_fp8_e4m3fn_slice_to_f32(src: &[u8]) -> Vec<f32> {
    src.iter().copied().map(fp8_e4m3fn_to_f32).collect()
}

/// Returns the deterministic serialized byte size for FP8 quantization (`1` byte per element
/// summed across all tensor `shapes`, with no safety-margin inflation).
#[must_use]
pub fn estimate_fp8_serialized_size(shapes: &[Vec<usize>]) -> usize {
    shapes
        .iter()
        .map(|shape| {
            if shape.is_empty() {
                0
            } else {
                shape
                    .iter()
                    .copied()
                    .fold(1usize, |acc, d| acc.saturating_mul(d))
            }
        })
        .sum()
}

/// Per-slot specification in a multi-output KV serde group (e.g., `slot 0 = K`, `slot 1 = V`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSerdeSlotSpec {
    /// Human-readable slot role (such as `"K"` or `"V"`).
    pub name: String,
    /// Bytes per element for this slot's serialized format (`1` for FP8, `2` for BF16/FP16).
    pub bytes_per_element: usize,
    /// Whether this slot may be `None` (absent) during tier-split serialization.
    pub allow_absent: bool,
}

/// Schema for a fixed-length multi-output KV serde group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSerdeGroupSchema {
    /// Ordered slot specifications defining `group_size = slots.len()`.
    pub slots: Vec<MultiSerdeSlotSpec>,
}

/// Planned byte-slice layout for a concrete multi-output KV serde group invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiSerdeGroupPlan {
    /// Per-slot `(byte_offset, byte_len)` for present slots, or `None` when the slot is absent.
    pub slot_ranges: Vec<Option<(usize, usize)>>,
    /// Total contiguous bytes required in the serialized buffer.
    pub total_bytes: usize,
}

/// Validates that a multi-serde tuple of length `actual_len` matches `expected_len` for `role`
/// (`"src"`, `"dst"`, or `"layout"`).
pub fn validate_multi_serde_group_arity(
    actual_len: usize,
    expected_len: usize,
    role: &str,
) -> Result<(), String> {
    if actual_len != expected_len {
        return Err(format!(
            "MultiSerde {role} group length {actual_len} does not match expected {expected_len}"
        ));
    }
    Ok(())
}

impl MultiSerdeGroupSchema {
    /// Fixed input/output tuple arity (`group_size`) expected by this schema.
    #[must_use]
    pub fn group_size(&self) -> usize {
        self.slots.len()
    }

    /// Validates `layout_shapes` (where `None` denotes an absent slot and `Some(shape)` provides
    /// that slot's tensor dimensions) and plans contiguous `(offset, byte_len)` ranges.
    pub fn plan_group(
        &self,
        layout_shapes: &[Option<Vec<usize>>],
        role: &str,
    ) -> Result<MultiSerdeGroupPlan, String> {
        validate_multi_serde_group_arity(layout_shapes.len(), self.group_size(), role)?;
        let mut slot_ranges = Vec::with_capacity(self.slots.len());
        let mut offset = 0usize;

        for (idx, (spec, maybe_shape)) in self.slots.iter().zip(layout_shapes.iter()).enumerate() {
            match maybe_shape {
                None => {
                    if !spec.allow_absent {
                        return Err(format!(
                            "MultiSerde {role} slot {idx} ('{}') is required and cannot be absent",
                            spec.name
                        ));
                    }
                    slot_ranges.push(None);
                }
                Some(shape) => {
                    let numel = if shape.is_empty() {
                        0
                    } else {
                        shape
                            .iter()
                            .copied()
                            .fold(1usize, |acc, d| acc.saturating_mul(d))
                    };
                    let byte_len = numel.saturating_mul(spec.bytes_per_element);
                    slot_ranges.push(Some((offset, byte_len)));
                    offset = offset.saturating_add(byte_len);
                }
            }
        }

        Ok(MultiSerdeGroupPlan {
            slot_ranges,
            total_bytes: offset,
        })
    }
}

// ============================================================================
// Batch 17 Step 3: Mutable Load Parallelizer & Exclusive-Chain Kernel Fuser
// ============================================================================

/// Returns `true` if `op_name` is a kernel eligible to fuse an `mo.mutable.load` buffer directly.
#[must_use]
pub fn is_fusible_mutable_load_kernel(op_name: &str) -> bool {
    matches!(op_name, "map.iter.elementwise" | "map.iter.opaque")
}

/// Counts how many operand slots in `block_id` reference `val`.
fn count_value_uses_in_block(module: &Module, block_id: BlockId, val: ValueId) -> usize {
    module.blocks[block_id]
        .ops
        .iter()
        .map(|&op_id| {
            module.ops[op_id]
                .operands
                .iter()
                .filter(|&&opd| opd == val)
                .count()
        })
        .sum()
}

/// When `kernel_in_chain` depends on `load_out_chain` exclusively through a tree of single-use
/// `mo.chain.create` operations in `block_id`, rewrites those `mo.chain.create` ops to replace
/// `load_out_chain` with `load_in_chain` so fusing the load into the kernel will not create a
/// chain cycle. Returns `true` if an exclusive path existed and was detached.
pub fn detach_exclusive_chain_path(
    module: &mut Module,
    block_id: BlockId,
    load_out_chain: ValueId,
    load_in_chain: ValueId,
    kernel_in_chain: ValueId,
) -> bool {
    if kernel_in_chain == load_out_chain {
        return false;
    }

    fn analyze_chain_value(
        module: &Module,
        block_id: BlockId,
        v: ValueId,
        load_out_chain: ValueId,
        on_path: &mut IndexSet<OpId>,
        to_rewrite: &mut Vec<OpId>,
    ) -> Option<bool> {
        if v == load_out_chain {
            return Some(true);
        }
        let ValueDef::OpResult { op: def_op, .. } = module.values[v].def else {
            return None;
        };
        if module.ops[def_op].name.full_name() != "mo.chain.create" {
            return None;
        }
        let mut subtree_reaches_load = false;
        let operands = module.ops[def_op].operands.clone();
        for opd in operands {
            if opd == load_out_chain {
                subtree_reaches_load = true;
                continue;
            }
            if let Some(status) =
                analyze_chain_value(module, block_id, opd, load_out_chain, on_path, to_rewrite)
            {
                if !status {
                    return Some(false);
                }
                subtree_reaches_load = true;
            }
        }
        if !subtree_reaches_load {
            return None;
        }
        if count_value_uses_in_block(module, block_id, v) != 1 {
            return Some(false);
        }
        if on_path.insert(def_op) {
            to_rewrite.push(def_op);
        }
        Some(true)
    }

    let mut on_path = IndexSet::new();
    let mut to_rewrite = Vec::new();
    if analyze_chain_value(
        module,
        block_id,
        kernel_in_chain,
        load_out_chain,
        &mut on_path,
        &mut to_rewrite,
    ) != Some(true)
    {
        return false;
    }

    for op_id in to_rewrite {
        for opd in &mut module.ops[op_id].operands {
            if *opd == load_out_chain {
                *opd = load_in_chain;
            }
        }
    }
    true
}

/// Splits each `mo.mutable.load(buf, in_chain) -> (out_tensor, out_chain)` whose `out_tensor` has
/// $n > 1$ uses in the same block into $n$ parallel `mo.mutable.load` ops (each with a single
/// tensor use) joined by an `mo.chain.create` over their output chains, unless a consumer kernel
/// has `hasWriteEffect = true` and also references `buf`. Returns the number of loads split.
pub fn parallelize_multi_use_mutable_loads(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut split_count = 0usize;

    for block_id in block_ids {
        let orig_ops = module.blocks[block_id].ops.clone();
        let mut new_block_ops = Vec::with_capacity(orig_ops.len());

        for op_id in orig_ops {
            let op = &module.ops[op_id];
            if op.name.full_name() != "mo.mutable.load"
                || op.operands.len() < 2
                || op.results.len() < 2
            {
                new_block_ops.push(op_id);
                continue;
            }
            let buf_val = op.operands[0];
            let in_chain = op.operands[1];
            let out_tensor = op.results[0];
            let out_chain = op.results[1];
            let loc = op.location.clone();
            let attrs = op.attributes.clone();

            // Collect all (consumer_op_id, operand_index) uses of `out_tensor` in this block.
            let mut uses: Vec<(OpId, usize)> = Vec::new();
            let mut blocked_by_writer = false;
            for &cand_op in &module.blocks[block_id].ops {
                let cand = &module.ops[cand_op];
                let writes_same_buf = matches!(
                    cand.attributes.get("hasWriteEffect"),
                    Some(Attribute::Bool(true))
                ) && cand.operands.contains(&buf_val);
                for (idx, &opd) in cand.operands.iter().enumerate() {
                    if opd == out_tensor {
                        if writes_same_buf {
                            blocked_by_writer = true;
                        }
                        uses.push((cand_op, idx));
                    }
                }
            }

            if blocked_by_writer || uses.len() <= 1 {
                new_block_ops.push(op_id);
                continue;
            }

            let tensor_ty = module.values[out_tensor].ty.clone();
            let chain_ty = module.values[out_chain].ty.clone();
            let mut new_chains = SmallVec::with_capacity(uses.len());

            for (consumer_op, opd_idx) in uses {
                let new_load_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mo", "mutable.load"),
                    operands: SmallVec::from_slice(&[buf_val, in_chain]),
                    results: SmallVec::new(),
                    attributes: attrs.clone(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let new_t = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: tensor_ty.clone(),
                    def: ValueDef::OpResult {
                        op: new_load_op,
                        index: 0,
                    },
                });
                let new_c = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: chain_ty.clone(),
                    def: ValueDef::OpResult {
                        op: new_load_op,
                        index: 1,
                    },
                });
                module.ops[new_load_op].results = SmallVec::from_slice(&[new_t, new_c]);
                module.ops[consumer_op].operands[opd_idx] = new_t;
                new_block_ops.push(new_load_op);
                new_chains.push(new_c);
            }

            let join_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mo", "chain.create"),
                operands: new_chains,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            let joined_chain = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: chain_ty,
                def: ValueDef::OpResult {
                    op: join_op,
                    index: 0,
                },
            });
            module.ops[join_op].results.push(joined_chain);
            new_block_ops.push(join_op);

            for &cand_op in &module.blocks[block_id].ops {
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == out_chain {
                        *opd = joined_chain;
                    }
                }
            }
            split_count += 1;
        }

        module.blocks[block_id].ops = new_block_ops;
    }

    split_count
}

/// Parallelizes multi-use `mo.mutable.load` ops and fuses each single-use `mo.mutable.load` into
/// its compatible `map.iter.elementwise` or `map.iter.opaque` consumer kernel (detaching exclusive
/// `mo.chain.create` paths when needed). Returns the number of fused `mo.mutable.load` operations.
pub fn fuse_mutable_loads_into_kernels(module: &mut Module) -> usize {
    let _ = parallelize_multi_use_mutable_loads(module);
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut fused_count = 0usize;

    for block_id in block_ids {
        let snapshot_ops = module.blocks[block_id].ops.clone();
        let mut erased_loads: IndexSet<OpId> = IndexSet::new();

        for &load_op in &snapshot_ops {
            if module.ops[load_op].name.full_name() != "mo.mutable.load"
                || module.ops[load_op].operands.len() < 2
                || module.ops[load_op].results.len() < 2
            {
                continue;
            }
            let buf_val = module.ops[load_op].operands[0];
            let load_in_chain = module.ops[load_op].operands[1];
            let out_tensor = module.ops[load_op].results[0];
            let load_out_chain = module.ops[load_op].results[1];

            // Find single consumer of `out_tensor` in this block.
            let mut consumer_info: Option<(OpId, usize)> = None;
            let mut multi_use = false;
            for &cand_op in &module.blocks[block_id].ops {
                if erased_loads.contains(&cand_op) {
                    continue;
                }
                for (idx, &opd) in module.ops[cand_op].operands.iter().enumerate() {
                    if opd == out_tensor {
                        if consumer_info.is_some() {
                            multi_use = true;
                        } else {
                            consumer_info = Some((cand_op, idx));
                        }
                    }
                }
            }
            if multi_use {
                continue;
            }
            let Some((kernel_op, arg_idx)) = consumer_info else {
                continue;
            };
            let kernel_name = module.ops[kernel_op].name.full_name();
            if !is_fusible_mutable_load_kernel(&kernel_name) {
                continue;
            }
            // In-place `map.iter.opaque` writing the same buffer cannot safely fuse snapshot loads.
            if kernel_name == "map.iter.opaque" && module.ops[kernel_op].operands.contains(&buf_val)
            {
                continue;
            }

            // Check if the kernel has an existing input chain operand (last operand if chain-typed)
            // that depends on `load_out_chain`.
            let chain_ty = module.values[load_out_chain].ty.clone();
            let existing_chain_opd_idx = module.ops[kernel_op]
                .operands
                .iter()
                .enumerate()
                .rev()
                .find(|&(i, &v)| i != arg_idx && module.values[v].ty == chain_ty)
                .map(|(i, _)| i);

            if let Some(ch_idx) = existing_chain_opd_idx {
                let kernel_in_chain = module.ops[kernel_op].operands[ch_idx];
                if kernel_in_chain != load_out_chain {
                    let _ = detach_exclusive_chain_path(
                        module,
                        block_id,
                        load_out_chain,
                        load_in_chain,
                        kernel_in_chain,
                    );
                } else {
                    module.ops[kernel_op].operands[ch_idx] = load_in_chain;
                }
            } else {
                module.ops[kernel_op].operands.push(load_in_chain);
            }

            // Replace tensor input with direct buffer input.
            module.ops[kernel_op].operands[arg_idx] = buf_val;
            module.ops[kernel_op]
                .attributes
                .insert("hasReadEffect".to_string(), Attribute::Bool(true));
            module.ops[kernel_op].attributes.insert(
                "fusionInfo".to_string(),
                Attribute::String("MutableLoad".to_string()),
            );

            // Ensure kernel has an output chain result and replace `load_out_chain` uses with it.
            let kernel_out_chain = if let Some(&existing_out_ch) = module.ops[kernel_op]
                .results
                .iter()
                .find(|&&r| module.values[r].ty == chain_ty)
            {
                existing_out_ch
            } else {
                let res_idx = module.ops[kernel_op].results.len() as u32;
                let new_out_ch = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: chain_ty,
                    def: ValueDef::OpResult {
                        op: kernel_op,
                        index: res_idx,
                    },
                });
                module.ops[kernel_op].results.push(new_out_ch);
                new_out_ch
            };

            for &cand_op in &module.blocks[block_id].ops {
                if cand_op == kernel_op || cand_op == load_op {
                    continue;
                }
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == load_out_chain {
                        *opd = kernel_out_chain;
                    }
                }
            }

            erased_loads.insert(load_op);
            fused_count += 1;
        }

        if !erased_loads.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|op_id| !erased_loads.contains(op_id));
        }
    }

    fused_count
}

// ============================================================================
// Batch 17 Step 4: Bidirectional Tensor Layout Inference & Conflict Detector
// ============================================================================

/// Returns the tensor rank of `ty` if it is a ranked `Type::Tensor`.
fn tensor_rank_of_type(ty: &Type) -> Option<usize> {
    match ty {
        Type::Tensor(t) => Some(t.shape.len()),
        _ => None,
    }
}

/// Returns `(input_layout, filter_layout)` for `mo.conv` or `mo.conv_transpose` given `(input_rank,
/// filter_rank)`. Only 2D (rank 4) and 3D (rank 5) convolutions are supported.
pub fn conv_seed_layouts(
    is_transpose: bool,
    input_rank: usize,
    filter_rank: usize,
) -> Result<(&'static str, &'static str), String> {
    if input_rank != filter_rank || (input_rank != 4 && input_rank != 5) {
        return Err(format!(
            "Only 2D/3D (rank 4/5) layouts supported for conv, got ({input_rank}, {filter_rank})"
        ));
    }
    let in_layout = if input_rank == 4 { "NHWC" } else { "NDHWC" };
    let flt_layout = match (is_transpose, filter_rank) {
        (false, 4) => "FCRS",
        (false, _) => "FCQRS",
        (true, 4) => "RSFC",
        (true, _) => "QRSFC",
    };
    Ok((in_layout, flt_layout))
}

/// Returns the list of `ValueId`s across which `op` propagates tensor layout constraints
/// bidirectionally (`mo.guard` drops the leading predicate operand; `mo.transfer` and elementwise
/// ops propagate across all tensor operands and results).
fn layout_propagation_values_for_op(op: &Operation) -> Vec<ValueId> {
    let name = op.name.full_name();
    if name == "mo.guard" {
        let mut vals: Vec<ValueId> = op.operands.iter().skip(1).copied().collect();
        vals.extend(op.results.iter().copied());
        vals
    } else if matches!(
        name.as_str(),
        "mo.transfer"
            | "mo.add"
            | "mo.sub"
            | "mo.mul"
            | "mo.div"
            | "mo.relu"
            | "mo.exp"
            | "mo.tanh"
            | "stablehlo.add"
            | "stablehlo.subtract"
            | "stablehlo.multiply"
            | "stablehlo.divide"
            | "stablehlo.abs"
            | "stablehlo.negate"
    ) {
        let mut vals: Vec<ValueId> = op.operands.iter().copied().collect();
        vals.extend(op.results.iter().copied());
        vals
    } else {
        Vec::new()
    }
}

fn add_layout_constraint(
    inferred: &mut IndexMap<ValueId, String>,
    op_name: &str,
    val: ValueId,
    layout: &str,
) -> Result<(), String> {
    if let Some(existing) = inferred.get(&val) {
        if existing != layout {
            return Err(format!(
                "{op_name}: layout '{layout}' conflicts with already constrained '{existing}'"
            ));
        }
        return Ok(());
    }
    inferred.insert(val, layout.to_string());
    Ok(())
}

/// Seeds layout constraints from `mo.conv`, `mo.conv_transpose`, and `mo.matmul` ops in `module`,
/// propagates them bidirectionally through layout-preserving ops (`mo.guard`, `mo.transfer`, and
/// elementwise ops), and detects conflicting layout constraints.
pub fn infer_and_propagate_tensor_layouts(
    module: &mut Module,
) -> Result<IndexMap<ValueId, String>, String> {
    let mut inferred: IndexMap<ValueId, String> = IndexMap::new();

    let all_ops: Vec<OpId> = module
        .blocks
        .values()
        .flat_map(|b| b.ops.iter().copied())
        .collect();

    // 1. Seed direct layout constraints from conv, conv_transpose, and matmul ops.
    for &op_id in &all_ops {
        let op = &module.ops[op_id];
        let full = op.name.full_name();
        if (full == "mo.conv" || full == "mo.conv_transpose") && op.operands.len() >= 2 {
            let in_val = op.operands[0];
            let flt_val = op.operands[1];
            let in_rank = tensor_rank_of_type(&module.values[in_val].ty).unwrap_or(0);
            let flt_rank = tensor_rank_of_type(&module.values[flt_val].ty).unwrap_or(0);
            let (in_layout, flt_layout) =
                conv_seed_layouts(full == "mo.conv_transpose", in_rank, flt_rank)?;
            add_layout_constraint(&mut inferred, &full, in_val, in_layout)?;
            add_layout_constraint(&mut inferred, &full, flt_val, flt_layout)?;
        } else if full == "mo.matmul" && op.operands.len() >= 2 {
            let weights = op.operands[1];
            add_layout_constraint(&mut inferred, &full, weights, "KN")?;
        }
    }

    // 2. Bidirectionally propagate layouts across users and defining ops.
    let mut worklist: Vec<ValueId> = inferred.keys().copied().collect();
    let mut history: IndexSet<ValueId> = worklist.iter().copied().collect();

    while let Some(val) = worklist.pop() {
        let layout = inferred[&val].clone();
        let mut affected_ops: Vec<OpId> = Vec::new();
        for &op_id in &all_ops {
            if module.ops[op_id].operands.contains(&val) {
                affected_ops.push(op_id);
            }
        }
        if let ValueDef::OpResult { op: def_op, .. } = module.values[val].def {
            affected_ops.push(def_op);
        }

        for op_id in affected_ops {
            let op = &module.ops[op_id];
            let full = op.name.full_name();
            let prop_vals = layout_propagation_values_for_op(op);
            for target_val in prop_vals {
                if tensor_rank_of_type(&module.values[target_val].ty).is_none() {
                    continue;
                }
                add_layout_constraint(&mut inferred, &full, target_val, &layout)?;
                if history.insert(target_val) {
                    worklist.push(target_val);
                }
            }
        }
    }

    // 3. Annotate ops that touch inferred values with `"inferred_result_layout"` when applicable.
    for &op_id in &all_ops {
        if let Some(&res0) = module.ops[op_id].results.first()
            && let Some(layout) = inferred.get(&res0)
        {
            module.ops[op_id].attributes.insert(
                "inferred_result_layout".to_string(),
                Attribute::String(layout.clone()),
            );
        }
    }

    Ok(inferred)
}

// ============================================================================
// Batch 17 Step 5: Bottom-Up Call-Graph Slot-Index Globalizer
// ============================================================================

/// Extracts a non-negative integer attribute (handling both `Attribute::Int` and typed integer
/// `Attribute::Opaque` such as `"4 : i64"`).
fn extract_u64_attr(attr: &Attribute) -> Option<u64> {
    match attr {
        Attribute::Int(n) if *n >= 0 => Some(*n as u64),
        Attribute::Opaque(s) | Attribute::String(s) => s
            .split(':')
            .next()
            .and_then(|head| head.trim().parse::<u64>().ok()),
        _ => None,
    }
}
/// Extracts a clean symbol name (stripping any leading `'@'`) from an attribute value.
fn extract_symbol_attr(attr: &Attribute) -> Option<String> {
    match attr {
        Attribute::String(s) | Attribute::SymbolRef(s) => {
            Some(s.trim_start_matches('@').to_string())
        }
        _ => None,
    }
}

/// Traverses the module's function/model call graph in bottom-up post-order, computing each
/// model's local slot usage from its `mgp.context.create` `"num_slots"` attribute and assigning
/// cumulative subgraph slot offsets to every `mgp.call` / `func.call` site (`"slot_offset"`,
/// `"is_globalized_index" = true`, and a preceding `index.constant` operand). Returns the
/// globalized slot count per function symbol.
pub fn globalize_call_graph_slot_indices(
    module: &mut Module,
) -> Result<IndexMap<String, u64>, String> {
    let func_ids = module.functions();
    if func_ids.len() <= 1 {
        let mut single = IndexMap::new();
        if let Some(&f_id) = func_ids.first()
            && let Some(sym) = module.ops[f_id]
                .attributes
                .get("sym_name")
                .and_then(extract_symbol_attr)
        {
            single.insert(sym, 0);
        }
        return Ok(single);
    }

    let mut sym_to_func: IndexMap<String, OpId> = IndexMap::new();
    let mut local_slots: IndexMap<String, u64> = IndexMap::new();
    let mut ctx_create_op: IndexMap<String, OpId> = IndexMap::new();
    let mut callsites_by_caller: IndexMap<String, Vec<(BlockId, OpId, String)>> = IndexMap::new();

    for &f_id in &func_ids {
        let Some(sym) = module.ops[f_id]
            .attributes
            .get("sym_name")
            .and_then(extract_symbol_attr)
        else {
            continue;
        };
        sym_to_func.insert(sym.clone(), f_id);
        let mut found_ctx: Option<(OpId, u64)> = None;
        let mut calls: Vec<(BlockId, OpId, String)> = Vec::new();

        for &reg_id in &module.ops[f_id].regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                for &op_id in &module.blocks[blk_id].ops {
                    let op = &module.ops[op_id];
                    let full = op.name.full_name();
                    if full == "mgp.context.create" {
                        let slots = op
                            .attributes
                            .get("num_slots")
                            .and_then(extract_u64_attr)
                            .unwrap_or(0);
                        found_ctx = Some((op_id, slots));
                    } else if (full == "mgp.call" || full == "func.call")
                        && let Some(callee) = op
                            .attributes
                            .get("callee")
                            .or_else(|| op.attributes.get("fn"))
                            .and_then(extract_symbol_attr)
                    {
                        calls.push((blk_id, op_id, callee));
                    }
                }
            }
        }

        let Some((c_op, slots)) = found_ctx else {
            return Err(format!(
                "globalize_call_graph_slot_indices: unable to find mgp.context.create in '@{sym}'"
            ));
        };
        local_slots.insert(sym.clone(), slots);
        ctx_create_op.insert(sym.clone(), c_op);
        callsites_by_caller.insert(sym, calls);
    }

    // Build bottom-up post-order over the call graph (detecting cycles).
    fn dfs_postorder(
        sym: &str,
        callsites: &IndexMap<String, Vec<(BlockId, OpId, String)>>,
        state: &mut IndexMap<String, u8>,
        order: &mut Vec<String>,
    ) -> Result<(), String> {
        match state.get(sym).copied() {
            Some(2) => return Ok(()),
            Some(1) => {
                return Err(format!(
                    "globalize_call_graph_slot_indices: cycle in call graph at '@{sym}'"
                ));
            }
            _ => {}
        }
        state.insert(sym.to_string(), 1);
        if let Some(edges) = callsites.get(sym) {
            for (_, _, callee) in edges {
                if callsites.contains_key(callee) {
                    dfs_postorder(callee, callsites, state, order)?;
                }
            }
        }
        state.insert(sym.to_string(), 2);
        order.push(sym.to_string());
        Ok(())
    }

    let mut visit_state: IndexMap<String, u8> = IndexMap::new();
    let mut bottom_up_order: Vec<String> = Vec::new();
    let all_syms: Vec<String> = sym_to_func.keys().cloned().collect();
    for sym in &all_syms {
        dfs_postorder(
            sym,
            &callsites_by_caller,
            &mut visit_state,
            &mut bottom_up_order,
        )?;
    }

    let mut total_slots: IndexMap<String, u64> = local_slots.clone();

    for sym in bottom_up_order {
        let mut running_slots = local_slots[&sym];
        let edges = callsites_by_caller.get(&sym).cloned().unwrap_or_default();
        for (blk_id, call_op_id, callee) in edges {
            let callee_slots = total_slots.get(&callee).copied().unwrap_or(0);
            let offset = running_slots;
            running_slots = running_slots.saturating_add(callee_slots);

            let loc = module.ops[call_op_id].location.clone();
            let mut const_attrs = IndexMap::new();
            const_attrs.insert("value".to_string(), Attribute::Int(offset as i64));
            let const_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("index", "constant"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: const_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            let idx_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: Type::Scalar(DType::I64),
                def: ValueDef::OpResult {
                    op: const_op,
                    index: 0,
                },
            });
            module.ops[const_op].results.push(idx_val);

            if let Some(pos) = module.blocks[blk_id]
                .ops
                .iter()
                .position(|&id| id == call_op_id)
            {
                module.blocks[blk_id].ops.insert(pos, const_op);
            }
            module.ops[call_op_id].operands.push(idx_val);
            module.ops[call_op_id]
                .attributes
                .insert("slot_offset".to_string(), Attribute::Int(offset as i64));
            module.ops[call_op_id]
                .attributes
                .insert("is_globalized_index".to_string(), Attribute::Bool(true));
        }

        total_slots.insert(sym.clone(), running_slots);
        if let Some(&c_op) = ctx_create_op.get(&sym) {
            module.ops[c_op].attributes.insert(
                "globalized_num_slots".to_string(),
                Attribute::Int(running_slots as i64),
            );
        }
    }

    Ok(total_slots)
}

// ============================================================================
// Batch 18 Step 1: L1/L2 Store Policy Planner & Binary Trace Stream Codec
// ============================================================================

/// Magic bytes at the start of every binary observability trace stream.
pub const TRACE_MAGIC: [u8; 4] = *b"LMCT";

/// On-wire framing layout version for binary observability traces.
pub const TRACE_FORMAT_VERSION: u32 = 1;

/// Storage API trace schema version enforced for `"storage"` level traces.
pub const TRACE_SCHEMA_VERSION: u32 = 2;

/// Descriptor for an available L2 storage adapter in the store controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L2AdapterSpec {
    /// Zero-based adapter index.
    pub index: usize,
    /// Human-readable adapter backend label.
    pub name: String,
    /// Whether this adapter accepts new writes.
    pub writable: bool,
}

/// Policy governing which L2 adapters receive newly written L1 keys and whether L1 retains or
/// deletes those keys after a successful L2 store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorePolicyKind {
    /// Replicate all keys to every writable L2 adapter and keep all keys in L1.
    DefaultKeepL1,
    /// Replicate all keys to every writable L2 adapter and immediately delete them from L1
    /// (`"skip_l1"` pure write-buffer mode).
    BufferOnlySkipL1,
}

/// Output plan from [`plan_l1_l2_store_policy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePolicyPlan {
    /// Mapping from L2 adapter index to the ordered list of object keys to persist.
    pub targets_by_adapter: IndexMap<usize, Vec<String>>,
    /// Object keys to delete from L1 once L2 persistence succeeds.
    pub l1_deletions: Vec<String>,
}

/// Computes the L2 adapter target assignment and post-store L1 deletion list for `keys` under
/// `policy`.
#[must_use]
pub fn plan_l1_l2_store_policy(
    policy: StorePolicyKind,
    keys: &[String],
    adapters: &[L2AdapterSpec],
    stored_to_l2_keys: &[String],
) -> StorePolicyPlan {
    let mut targets_by_adapter = IndexMap::new();
    for ad in adapters {
        if ad.writable {
            targets_by_adapter.insert(ad.index, keys.to_vec());
        }
    }
    let l1_deletions = match policy {
        StorePolicyKind::DefaultKeepL1 => Vec::new(),
        StorePolicyKind::BufferOnlySkipL1 => stored_to_l2_keys.to_vec(),
    };
    StorePolicyPlan {
        targets_by_adapter,
        l1_deletions,
    }
}

/// File-level metadata header frame at the start of a binary trace stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceHeader {
    /// Always [`TRACE_MAGIC`] (`b"LMCT"`).
    pub magic: [u8; 4],
    /// Wire framing format version (`1`).
    pub format_version: u32,
    /// Trace level (`"storage"` or `"events"`).
    pub level: String,
    /// Storage trace schema version (`2` for `"storage"` traces).
    pub trace_schema_version: u32,
    /// Monotonic clock start timestamp in nanoseconds.
    pub t_mono_start_ns: u64,
    /// Wall-clock start timestamp in nanoseconds.
    pub t_wall_start_ns: u64,
    /// SHA-256 hex digest of the storage manager configuration.
    pub sm_config_digest: String,
}

/// One captured operation call record in a binary trace stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRecord {
    /// Monotonic nanoseconds elapsed since [`TraceHeader::t_mono_start_ns`].
    pub t_mono_ns: u64,
    /// Wall-clock nanoseconds when the event was published.
    pub t_wall_ns: u64,
    /// Fully qualified call-site name.
    pub qualname: String,
    /// JSON-serialized call arguments.
    pub args_json: String,
}

fn push_len_prefixed_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(bytes);
}

fn read_len_prefixed_str(frame: &[u8], cursor: &mut usize) -> Result<String, String> {
    if *cursor + 4 > frame.len() {
        return Err("truncated string length in trace frame".to_string());
    }
    let len = u32::from_be_bytes(frame[*cursor..*cursor + 4].try_into().unwrap()) as usize;
    *cursor += 4;
    if *cursor + len > frame.len() {
        return Err("truncated string payload in trace frame".to_string());
    }
    let s = std::str::from_utf8(&frame[*cursor..*cursor + len])
        .map_err(|e| format!("invalid UTF-8 in trace frame: {e}"))?
        .to_string();
    *cursor += len;
    Ok(s)
}

fn append_length_prefixed_frame(stream: &mut Vec<u8>, payload: &[u8]) {
    stream.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    stream.extend_from_slice(payload);
}

/// Encodes `header` and `records` into a 4-byte big-endian length-prefixed binary trace stream.
#[must_use]
pub fn encode_trace_stream(header: &TraceHeader, records: &[TraceRecord]) -> Vec<u8> {
    let mut stream = Vec::new();
    let mut hdr_buf = Vec::new();
    hdr_buf.extend_from_slice(&header.magic);
    hdr_buf.extend_from_slice(&header.format_version.to_be_bytes());
    push_len_prefixed_str(&mut hdr_buf, &header.level);
    hdr_buf.extend_from_slice(&header.trace_schema_version.to_be_bytes());
    hdr_buf.extend_from_slice(&header.t_mono_start_ns.to_be_bytes());
    hdr_buf.extend_from_slice(&header.t_wall_start_ns.to_be_bytes());
    push_len_prefixed_str(&mut hdr_buf, &header.sm_config_digest);
    append_length_prefixed_frame(&mut stream, &hdr_buf);

    for rec in records {
        let mut rec_buf = Vec::new();
        rec_buf.extend_from_slice(&rec.t_mono_ns.to_be_bytes());
        rec_buf.extend_from_slice(&rec.t_wall_ns.to_be_bytes());
        push_len_prefixed_str(&mut rec_buf, &rec.qualname);
        push_len_prefixed_str(&mut rec_buf, &rec.args_json);
        append_length_prefixed_frame(&mut stream, &rec_buf);
    }
    stream
}

/// Decodes a binary trace stream into `(TraceHeader, Vec<TraceRecord>, had_truncated_tail)`.
/// Enforces strict header validation (`TRACE_MAGIC`, `TRACE_FORMAT_VERSION`, and
/// `TRACE_SCHEMA_VERSION` when `level == "storage"`), and cleanly stops with
/// `had_truncated_tail = true` if a trailing record frame was truncated mid-write.
pub fn decode_trace_stream(stream: &[u8]) -> Result<(TraceHeader, Vec<TraceRecord>, bool), String> {
    if stream.len() < 4 {
        return Err("trace stream is empty or has truncated header length prefix".to_string());
    }
    let hdr_len = u32::from_be_bytes(stream[0..4].try_into().unwrap()) as usize;
    if stream.len() < 4 + hdr_len {
        return Err("trace stream has truncated header frame body".to_string());
    }
    let hdr_frame = &stream[4..4 + hdr_len];
    if hdr_frame.len() < 8 {
        return Err("trace header frame too short".to_string());
    }
    let magic: [u8; 4] = hdr_frame[0..4].try_into().unwrap();
    if magic != TRACE_MAGIC {
        return Err(format!("trace header bad magic: {magic:?}"));
    }
    let format_version = u32::from_be_bytes(hdr_frame[4..8].try_into().unwrap());
    if format_version != TRACE_FORMAT_VERSION {
        return Err(format!(
            "unsupported trace format_version {format_version} (expected {TRACE_FORMAT_VERSION})"
        ));
    }
    let mut hc = 8usize;
    let level = read_len_prefixed_str(hdr_frame, &mut hc)?;
    if hc + 20 > hdr_frame.len() {
        return Err("trace header truncated numeric fields".to_string());
    }
    let trace_schema_version = u32::from_be_bytes(hdr_frame[hc..hc + 4].try_into().unwrap());
    hc += 4;
    if level == "storage" && trace_schema_version != TRACE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported storage trace_schema_version {trace_schema_version}"
        ));
    }
    let t_mono_start_ns = u64::from_be_bytes(hdr_frame[hc..hc + 8].try_into().unwrap());
    hc += 8;
    let t_wall_start_ns = u64::from_be_bytes(hdr_frame[hc..hc + 8].try_into().unwrap());
    hc += 8;
    let sm_config_digest = read_len_prefixed_str(hdr_frame, &mut hc)?;

    let header = TraceHeader {
        magic,
        format_version,
        level,
        trace_schema_version,
        t_mono_start_ns,
        t_wall_start_ns,
        sm_config_digest,
    };

    let mut pos = 4 + hdr_len;
    let mut records = Vec::new();
    let mut had_truncated_tail = false;

    while pos < stream.len() {
        if pos + 4 > stream.len() {
            had_truncated_tail = true;
            break;
        }
        let rec_len = u32::from_be_bytes(stream[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + rec_len > stream.len() {
            had_truncated_tail = true;
            break;
        }
        let rec_frame = &stream[pos..pos + rec_len];
        pos += rec_len;
        if rec_frame.len() < 16 {
            continue;
        }
        let t_mono_ns = u64::from_be_bytes(rec_frame[0..8].try_into().unwrap());
        let t_wall_ns = u64::from_be_bytes(rec_frame[8..16].try_into().unwrap());
        let mut rc = 16usize;
        let Ok(qualname) = read_len_prefixed_str(rec_frame, &mut rc) else {
            continue;
        };
        let Ok(args_json) = read_len_prefixed_str(rec_frame, &mut rc) else {
            continue;
        };
        records.push(TraceRecord {
            t_mono_ns,
            t_wall_ns,
            qualname,
            args_json,
        });
    }

    Ok((header, records, had_truncated_tail))
}

// ============================================================================
// Batch 18 Step 2: Loop-Invariant Index-Transform View Hoister
// ============================================================================

/// Returns `true` if `op_name` is a `map.index.*` coordinate transform operation.
#[must_use]
pub fn is_map_index_transform_op(op_name: &str) -> bool {
    matches!(
        op_name,
        "map.index.slice" | "map.index.broadcast" | "map.index.reshape" | "map.index.transpose"
    )
}

/// Walks backward from `index_val` through `map.index.*` transform ops, returning the chain in
/// closest-to-load order.
fn collect_index_transform_chain(module: &Module, index_val: ValueId) -> Vec<OpId> {
    let mut chain = Vec::new();
    let mut cur = index_val;
    while let ValueDef::OpResult { op: def_op, .. } = module.values[cur].def {
        let op = &module.ops[def_op];
        if !is_map_index_transform_op(&op.name.full_name()) || op.operands.is_empty() {
            break;
        }
        chain.push(def_op);
        cur = op.operands[0];
    }
    chain
}

/// Returns `true` if `transform_op`'s output index reaches a `map.store`'s index operand
/// (`operands[1]`) directly or through further `map.index.*` transforms in `block_id`.
fn transitively_feeds_store_index(module: &Module, block_id: BlockId, transform_op: OpId) -> bool {
    let Some(&res0) = module.ops[transform_op].results.first() else {
        return false;
    };
    let mut worklist = vec![res0];
    let mut visited = IndexSet::new();
    visited.insert(res0);

    while let Some(idx_val) = worklist.pop() {
        for &op_id in &module.blocks[block_id].ops {
            let op = &module.ops[op_id];
            let full = op.name.full_name();
            if full == "map.store" {
                // `map.store` operands: [tensor, index, value] (or [index, value] if 2 operands).
                let idx_slot = if op.operands.len() >= 3 { 1 } else { 0 };
                if op.operands.get(idx_slot) == Some(&idx_val) {
                    return true;
                }
            } else if is_map_index_transform_op(&full)
                && op.operands.first() == Some(&idx_val)
                && let Some(&next_idx) = op.results.first()
                && visited.insert(next_idx)
            {
                worklist.push(next_idx);
            }
        }
    }
    false
}

/// Length of the leading prefix of `chain` (closest-to-load first) that does not transitively
/// feed any `map.store` index.
fn hoistable_index_prefix_len(module: &Module, block_id: BlockId, chain: &[OpId]) -> usize {
    chain
        .iter()
        .take_while(|&&op_id| !transitively_feeds_store_index(module, block_id, op_id))
        .count()
}

/// Hoists chains of loop-invariant `map.index.*` coordinate transforms feeding `map.load` (that do
/// not transitively feed a `map.store` index) into `.stride` tensor view operations (e.g.
/// `map.index.slice.stride`), rewires `map.load` and `map.call_elementwise` to use the outermost
/// surviving loop index, and erases dead `map.index.*` ops. Returns the number of hoisted view ops.
pub fn hoist_loop_invariant_index_views(module: &mut Module) -> usize {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut hoisted_count = 0usize;

    for block_id in block_ids {
        let orig_ops = module.blocks[block_id].ops.clone();
        let loads: Vec<OpId> = orig_ops
            .iter()
            .copied()
            .filter(|&op_id| {
                module.ops[op_id].name.full_name() == "map.load"
                    && module.ops[op_id].operands.len() >= 2
            })
            .collect();

        for load_op in loads {
            let tensor_val = module.ops[load_op].operands[0];
            let index_val = module.ops[load_op].operands[1];
            let mut chain = collect_index_transform_chain(module, index_val);
            let prefix = hoistable_index_prefix_len(module, block_id, &chain);
            if prefix == 0 {
                continue;
            }
            chain.truncate(prefix);

            let outermost_op = *chain.last().unwrap();
            let outermost_index = module.ops[outermost_op].operands[0];
            let tensor_ty = module.values[tensor_val].ty.clone();

            let mut cur_tensor = tensor_val;
            let mut stride_ops_to_insert = Vec::with_capacity(chain.len());

            for &transform_op in &chain {
                let t_op = &module.ops[transform_op];
                let stride_name = format!("{}.stride", t_op.name.name);
                let mut stride_operands = SmallVec::new();
                stride_operands.push(cur_tensor);
                for &extra_opd in t_op.operands.iter().skip(1) {
                    stride_operands.push(extra_opd);
                }
                let attrs = t_op.attributes.clone();
                let loc = t_op.location.clone();

                let stride_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("map", stride_name),
                    operands: stride_operands,
                    results: SmallVec::new(),
                    attributes: attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc,
                });
                let new_view = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: tensor_ty.clone(),
                    def: ValueDef::OpResult {
                        op: stride_op,
                        index: 0,
                    },
                });
                module.ops[stride_op].results.push(new_view);
                cur_tensor = new_view;
                stride_ops_to_insert.push(stride_op);
                hoisted_count += 1;
            }

            if let Some(pos) = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == load_op)
            {
                for (i, s_op) in stride_ops_to_insert.into_iter().enumerate() {
                    module.blocks[block_id].ops.insert(pos + i, s_op);
                }
            }

            module.ops[load_op].operands[0] = cur_tensor;
            module.ops[load_op].operands[1] = outermost_index;
        }

        // Rewire `map.call_elementwise` ops that share a hoistable index chain.
        let elem_calls: Vec<OpId> = module.blocks[block_id]
            .ops
            .iter()
            .copied()
            .filter(|&op_id| {
                module.ops[op_id].name.full_name() == "map.call_elementwise"
                    && !module.ops[op_id].operands.is_empty()
            })
            .collect();
        for call_op in elem_calls {
            let idx_val = module.ops[call_op].operands[0];
            let chain = collect_index_transform_chain(module, idx_val);
            let prefix = hoistable_index_prefix_len(module, block_id, &chain);
            if prefix > 0 {
                let outermost_idx = module.ops[chain[prefix - 1]].operands[0];
                module.ops[call_op].operands[0] = outermost_idx;
            }
        }

        // Erase unused `map.index.*` transforms in reverse order so outer hops become dead too.
        let transforms: Vec<OpId> = module.blocks[block_id]
            .ops
            .iter()
            .copied()
            .filter(|&op_id| is_map_index_transform_op(&module.ops[op_id].name.full_name()))
            .collect();
        for &t_op in transforms.iter().rev() {
            if let Some(&res0) = module.ops[t_op].results.first()
                && count_value_uses_in_block(module, block_id, res0) == 0
            {
                module.blocks[block_id].ops.retain(|&id| id != t_op);
            }
        }
    }

    hoisted_count
}

// ============================================================================
// Batch 18 Step 3: Contiguous View Chain Raiser
// ============================================================================

/// Returns the first input dimension index `k` of `in_dims` with extent `!= 1`, or `in_dims.len()`
/// when all dimensions have extent `1`.
#[must_use]
pub fn find_contiguous_slice_dim(in_dims: &[i64]) -> usize {
    in_dims
        .iter()
        .position(|&d| d != 1)
        .unwrap_or(in_dims.len())
}

/// Returns `Some(k)` if a slice over `(in_dims, out_dims, starts, stops, steps)` is a contiguous
/// static unit-step slice along dimension `k = find_contiguous_slice_dim(in_dims)` (all other
/// dimensions `i != k` retain their full input extent `in_dims[i] == out_dims[i]`, all bounds are
/// non-negative/static, and `steps[k] == 1`).
#[must_use]
pub fn match_contiguous_slice_dim(
    in_dims: &[i64],
    out_dims: &[i64],
    starts: &[i64],
    stops: &[i64],
    steps: &[i64],
) -> Option<usize> {
    let rank = in_dims.len();
    if rank == 0
        || out_dims.len() != rank
        || starts.len() != rank
        || stops.len() != rank
        || steps.len() != rank
    {
        return None;
    }
    if in_dims.iter().any(|&d| d < 0) || out_dims.iter().any(|&d| d < 0) {
        return None;
    }
    let k = find_contiguous_slice_dim(in_dims);
    if k >= rank {
        return None;
    }
    for i in 0..rank {
        if i != k && in_dims[i] != out_dims[i] {
            return None;
        }
    }
    if starts[k] < 0 || stops[k] < starts[k] || steps[k] != 1 {
        return None;
    }
    Some(k)
}

#[derive(Debug, Clone)]
enum RaisableViewStep {
    Reshape,
    Slice { start_k: i64, stop_k: i64 },
}

fn extract_i64_vec_attr(attrs: &IndexMap<String, Attribute>, key: &str) -> Option<Vec<i64>> {
    match attrs.get(key)? {
        Attribute::DenseIntArray(v) => Some(v.clone()),
        Attribute::Array(items) => items
            .iter()
            .map(|a| extract_u64_attr(a).map(|u| u as i64))
            .collect(),
        _ => None,
    }
}

/// Matches a `map.iter.elementwise` op's `"view_chain"` attribute (ordered from output loop index
/// toward input load, e.g. `["slice", "reshape"]`) and validates that any slice step is a
/// contiguous static unit-step slice.
fn match_raisable_view_steps(op: &Operation) -> Option<Vec<RaisableViewStep>> {
    if op.name.full_name() != "map.iter.elementwise"
        || op.operands.is_empty()
        || op.results.len() != 1
    {
        return None;
    }
    let Attribute::Array(items) = op.attributes.get("view_chain")? else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let mut steps = Vec::with_capacity(items.len());
    for item in items {
        let kind = item.as_str()?;
        if kind == "reshape" {
            steps.push(RaisableViewStep::Reshape);
        } else if kind == "slice" {
            let in_dims = extract_i64_vec_attr(&op.attributes, "in_dims")?;
            let out_dims = extract_i64_vec_attr(&op.attributes, "out_dims")?;
            let starts = extract_i64_vec_attr(&op.attributes, "starts")?;
            let stops = extract_i64_vec_attr(&op.attributes, "stops")?;
            let step_vals = extract_i64_vec_attr(&op.attributes, "steps")?;
            let k = match_contiguous_slice_dim(&in_dims, &out_dims, &starts, &stops, &step_vals)?;
            steps.push(RaisableViewStep::Slice {
                start_k: starts[k],
                stop_k: stops[k],
            });
        } else {
            return None;
        }
    }
    Some(steps)
}

/// Raises unfused contiguous `map.iter.elementwise` view kernels back to zero-copy data operations
/// (`mo.static.reshape` and `mogg.slice`) in reversed (input-to-output) order. Returns
/// `(num_slices_raised, num_reshapes_raised)`.
pub fn raise_back_contiguous_views(module: &mut Module) -> (usize, usize) {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut slices_raised = 0usize;
    let mut reshapes_raised = 0usize;

    for block_id in block_ids {
        let orig_ops = module.blocks[block_id].ops.clone();
        let mut new_block_ops = Vec::with_capacity(orig_ops.len());

        for op_id in orig_ops {
            let Some(steps) = match_raisable_view_steps(&module.ops[op_id]) else {
                new_block_ops.push(op_id);
                continue;
            };
            let in_val = module.ops[op_id].operands[0];
            let orig_out_val = module.ops[op_id].results[0];
            let out_ty = module.values[orig_out_val].ty.clone();
            let loc = module.ops[op_id].location.clone();

            let mut cur_val = in_val;
            // Body transforms run from output index to input load; data ops run in reverse.
            for step in steps.iter().rev() {
                match *step {
                    RaisableViewStep::Reshape => {
                        let reshape_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mo.static", "reshape"),
                            operands: SmallVec::from_slice(&[cur_val]),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let next_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: out_ty.clone(),
                            def: ValueDef::OpResult {
                                op: reshape_op,
                                index: 0,
                            },
                        });
                        module.ops[reshape_op].results.push(next_val);
                        new_block_ops.push(reshape_op);
                        cur_val = next_val;
                        reshapes_raised += 1;
                    }
                    RaisableViewStep::Slice { start_k, stop_k } => {
                        let mut slice_attrs = IndexMap::new();
                        slice_attrs.insert("start".to_string(), Attribute::Int(start_k));
                        slice_attrs.insert("stop".to_string(), Attribute::Int(stop_k));
                        let slice_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mogg", "slice"),
                            operands: SmallVec::from_slice(&[cur_val]),
                            results: SmallVec::new(),
                            attributes: slice_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let next_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: out_ty.clone(),
                            def: ValueDef::OpResult {
                                op: slice_op,
                                index: 0,
                            },
                        });
                        module.ops[slice_op].results.push(next_val);
                        new_block_ops.push(slice_op);
                        cur_val = next_val;
                        slices_raised += 1;
                    }
                }
            }

            for &cand_op in &module.blocks[block_id].ops {
                if cand_op == op_id {
                    continue;
                }
                for opd in &mut module.ops[cand_op].operands {
                    if *opd == orig_out_val {
                        *opd = cur_val;
                    }
                }
            }
        }

        module.blocks[block_id].ops = new_block_ops;
    }

    (slices_raised, reshapes_raised)
}

// ============================================================================
// Batch 18 Step 4: Mixed-Precision Accumulating Epilogue Buffer Allocator
// ============================================================================

/// Returns `true` if `kernel_name` names an accumulating opaque operation (`mo.concat`,
/// `mo.matmul`, `mo.batch_matmul`, `mo.conv`, or `mo.conv_transpose`) that requires accumulating
/// in its pre-cast element type when followed by a dtype-changing epilogue.
#[must_use]
pub fn is_accumulating_opaque_kernel_name(kernel_name: &str) -> bool {
    kernel_name.contains("mo.concat")
        || kernel_name.contains("mo.matmul")
        || kernel_name.contains("mo.batch_matmul")
        || kernel_name.contains("mo.conv")
        || kernel_name.contains("mo.conv_transpose")
}

/// Assigns an explicit intermediate store-target tensor to each `map.iter.epilogue` op in `module`.
/// When the feeding `map.iter.opaque` kernel is an accumulating op (`mo.matmul`, `mo.conv`, etc.)
/// and the epilogue casts to a different output element dtype, allocates a pre-cast scratch
/// accumulation tensor result on the kernel (before any trailing chain result). Returns
/// `(scratch_buffers_allocated, direct_targets_bound)`.
pub fn allocate_accumulating_epilogue_buffers(module: &mut Module) -> (usize, usize) {
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    let mut scratch_allocated = 0usize;
    let mut direct_bound = 0usize;

    for block_id in block_ids {
        let ops_snapshot = module.blocks[block_id].ops.clone();
        for &epi_op in &ops_snapshot {
            if module.ops[epi_op].name.full_name() != "map.iter.epilogue" {
                continue;
            }
            if matches!(
                module.ops[epi_op].attributes.get("has_intermediate_tensor"),
                Some(Attribute::Bool(true))
            ) {
                continue;
            }
            if module.ops[epi_op].operands.is_empty() || module.ops[epi_op].results.is_empty() {
                continue;
            }
            let in_val = module.ops[epi_op].operands[0];
            let store_target = if module.ops[epi_op].operands.len() >= 2 {
                module.ops[epi_op].operands[1]
            } else {
                module.ops[epi_op].results[0]
            };

            let ValueDef::OpResult { op: kernel_op, .. } = module.values[in_val].def else {
                continue;
            };
            if module.ops[kernel_op].name.full_name() != "map.iter.opaque" {
                continue;
            }
            let kernel_name = module.ops[kernel_op]
                .attributes
                .get("kernel_name")
                .and_then(Attribute::as_str)
                .unwrap_or("")
                .to_string();

            let in_tensor_ty = module.values[in_val].ty.as_tensor().cloned();
            let target_tensor_ty = module.values[store_target].ty.as_tensor().cloned();
            let (Some(in_tty), Some(target_tty)) = (in_tensor_ty, target_tensor_ty) else {
                continue;
            };

            let dtype_changes = in_tty.dtype != target_tty.dtype;
            if dtype_changes && is_accumulating_opaque_kernel_name(&kernel_name) {
                let scratch_ty =
                    Type::Tensor(TensorType::new(target_tty.shape.clone(), in_tty.dtype));
                // Insert scratch result on `kernel_op` before any trailing chain (`tensor<1xi64>`)
                // result.
                let res_len = module.ops[kernel_op].results.len();
                let has_trailing_chain = module.ops[kernel_op]
                    .results
                    .last()
                    .and_then(|&r| module.values[r].ty.as_tensor())
                    .is_some_and(|t| t.shape == [Dim::Static(1)] && t.dtype == DType::I64);
                let insert_idx = if has_trailing_chain && res_len >= 1 {
                    res_len - 1
                } else {
                    res_len
                };

                let scratch_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: scratch_ty,
                    def: ValueDef::OpResult {
                        op: kernel_op,
                        index: insert_idx as u32,
                    },
                });
                module.ops[kernel_op]
                    .results
                    .insert(insert_idx, scratch_val);
                if has_trailing_chain {
                    let last_idx = module.ops[kernel_op].results.len() - 1;
                    let chain_val = module.ops[kernel_op].results[last_idx];
                    module.values[chain_val].def = ValueDef::OpResult {
                        op: kernel_op,
                        index: last_idx as u32,
                    };
                }

                module.ops[epi_op].operands.push(scratch_val);
                module.ops[epi_op]
                    .attributes
                    .insert("has_intermediate_tensor".to_string(), Attribute::Bool(true));
                module.ops[epi_op].attributes.insert(
                    "uses_scratch_accumulator".to_string(),
                    Attribute::Bool(true),
                );
                scratch_allocated += 1;
            } else {
                module.ops[epi_op].operands.push(store_target);
                module.ops[epi_op]
                    .attributes
                    .insert("has_intermediate_tensor".to_string(), Attribute::Bool(true));
                module.ops[epi_op].attributes.insert(
                    "uses_scratch_accumulator".to_string(),
                    Attribute::Bool(false),
                );
                direct_bound += 1;
            }
        }
    }

    (scratch_allocated, direct_bound)
}

// ============================================================================
// Batch 18 Step 5: Staged Device-Graph Dependency Synthesizer
// ============================================================================

/// Returns the ordering source value for `operand`: if `operand` is produced by
/// `mgp.device_graph.stable.tensor` whose address (`operands[0]`) is produced by
/// `mgp.device_graph.stable.place`, returns the placed tensor (`place_op.operands[0]`) so
/// per-device ordering is preserved from the producer of the placed tensor.
#[must_use]
pub fn resolve_device_graph_ordering_source(module: &Module, operand: ValueId) -> ValueId {
    if let ValueDef::OpResult { op: stable_op, .. } = module.values[operand].def
        && module.ops[stable_op].name.full_name() == "mgp.device_graph.stable.tensor"
        && let Some(&addr_val) = module.ops[stable_op].operands.first()
        && let ValueDef::OpResult { op: place_op, .. } = module.values[addr_val].def
        && module.ops[place_op].name.full_name() == "mgp.device_graph.stable.place"
        && let Some(&placed_tensor) = module.ops[place_op].operands.first()
    {
        return placed_tensor;
    }
    operand
}

/// Converts staged `mogg.routine.async` operations in `module` into
/// `mgp.device_graph.dependencies` operations with explicit producer-consumer DAG edges (resolving
/// `mgp.device_graph.stable.tensor` -> `mgp.device_graph.stable.place` to its placed tensor
/// producer) and aggregates `"constant_resources"` onto the enclosing function. Returns the number
/// of converted `mogg.routine.async` operations.
pub fn compute_device_graph_dependencies(module: &mut Module) -> usize {
    let func_ids = module.functions();
    let mut converted_count = 0usize;

    for f_id in func_ids {
        let mut resources_seen: IndexSet<String> = IndexSet::new();
        let reg_ids = module.ops[f_id].regions.clone();

        for reg_id in reg_ids {
            let block_ids = module.regions[reg_id].blocks.clone();
            for blk_id in block_ids {
                let routine_ops: Vec<OpId> = module.blocks[blk_id]
                    .ops
                    .iter()
                    .copied()
                    .filter(|&op_id| module.ops[op_id].name.full_name() == "mogg.routine.async")
                    .collect();
                if routine_ops.is_empty() {
                    continue;
                }

                let mut op_to_node_info: IndexMap<OpId, (usize, ValueId)> = IndexMap::new();
                for (node_idx, &op_id) in routine_ops.iter().enumerate() {
                    let res_idx = module.ops[op_id].results.len() as u32;
                    let node_token = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: Type::Tensor(TensorType::new(vec![Dim::Static(1)], DType::I64)),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: res_idx,
                        },
                    });
                    module.ops[op_id].results.push(node_token);
                    op_to_node_info.insert(op_id, (node_idx, node_token));
                }

                for &op_id in &routine_ops {
                    let orig_operands = module.ops[op_id].operands.clone();
                    let mut dep_indices: Vec<i64> = Vec::new();
                    let mut dep_tokens: Vec<ValueId> = Vec::new();
                    let mut seen_producers: IndexSet<OpId> = IndexSet::new();

                    for &opd in &orig_operands {
                        let src = resolve_device_graph_ordering_source(module, opd);
                        if let ValueDef::OpResult { op: prod_op, .. } = module.values[src].def
                            && let Some(&(prod_idx, prod_token)) = op_to_node_info.get(&prod_op)
                            && seen_producers.insert(prod_op)
                        {
                            dep_indices.push(prod_idx as i64);
                            dep_tokens.push(prod_token);
                        }
                    }

                    if let Some(Attribute::Array(items)) =
                        module.ops[op_id].attributes.get("constant_resources")
                    {
                        for item in items {
                            if let Some(s) = item.as_str() {
                                resources_seen.insert(s.to_string());
                            }
                        }
                    }

                    module.ops[op_id].name = OpName::new("mgp.device_graph", "dependencies");
                    for tok in dep_tokens {
                        module.ops[op_id].operands.push(tok);
                    }
                    module.ops[op_id].attributes.insert(
                        "dep_node_indices".to_string(),
                        Attribute::DenseIntArray(dep_indices),
                    );
                    converted_count += 1;
                }
            }
        }

        if !resources_seen.is_empty() {
            let agg: Vec<Attribute> = resources_seen.into_iter().map(Attribute::String).collect();
            module.ops[f_id]
                .attributes
                .insert("constant_resources".to_string(), Attribute::Array(agg));
        }
    }

    converted_count
}
