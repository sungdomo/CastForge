//! Autoregressive KV-cache, PagedAttention, Flash-Decoding, and Tree-Attention passes.

use super::attention::*;
use crate::ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
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

        // Stamp homogeneous KV layer group IDs onto each attention op
        let groups = analyze_kv_layer_groups(module);
        for group in &groups {
            for &layer_idx in &group.layer_indices {
                if let Some(&op_id) = attn_ops.get(layer_idx)
                    && let Some(op) = module.ops.get_mut(op_id)
                {
                    op.attributes.insert(
                        "castforge.kv_layer_group".to_string(),
                        Attribute::Int(group.group_id as i64),
                    );
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
