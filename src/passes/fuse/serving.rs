//! Autoregressive KV-cache, PagedAttention, Flash-Decoding, and Tree-Attention passes.

use super::attention::*;
use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
