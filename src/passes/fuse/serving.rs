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

/// PagedAttention Block-Table Lowering Pass (`lower-paged-attention`).
pub struct LowerPagedAttentionPass {
    pub page_block_size: usize,
    pub max_blocks: usize,
}

impl Default for LowerPagedAttentionPass {
    fn default() -> Self {
        Self {
            page_block_size: 16,
            max_blocks: 64,
        }
    }
}

impl Pass for LowerPagedAttentionPass {
    fn name(&self) -> &'static str {
        "lower-paged-attention"
    }

    fn description(&self) -> &'static str {
        "Lowers fused_attention to vLLM-style PagedAttention with block-table KV addressing"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let block_size = self.page_block_size.max(4) as i64;
        let max_blocks = self.max_blocks.max(1) as i64;

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
