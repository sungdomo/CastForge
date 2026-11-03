//! Autoregressive KV-cache, PagedAttention, Flash-Decoding, and Tree-Attention passes.

use super::attention::*;
use crate::ir::{
    Attribute, BlockId, Dim, Location, Module, OpId, OpName, Operation, Type, Value, ValueDef,
    ValueId,
};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
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
