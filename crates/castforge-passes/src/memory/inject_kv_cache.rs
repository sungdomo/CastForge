use crate::{FuseAttentionPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Autoregressive KV-Cache Injection Pass (`inject-kv-cache`).
///
/// Transforms `castforge.fused_attention` operations (running `fuse-attention` first if SDPA is
/// still decomposed) by inserting explicit `"castforge.kv_cache_update"` composite operations for
/// the Key (`K`) and Value (`V`) tensors (`max_seq_len = N`, `cache_role = "key" | "value"`),
/// modeling paged/rolling KV-cache state management for autoregressive LLM token decoding.
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
        let attn_res = FuseAttentionPass.run(module)?;
        let mut created = attn_res.ops_created;
        let mut modified = attn_res.ops_modified;
        let removed = attn_res.ops_removed;
        let max_seq = self.max_seq_len.max(1) as i64;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.attributes.contains_key("kv_cached") {
                    continue;
                }
                let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if name != "castforge.fused_attention" || inputs.len() < 3 {
                    continue;
                }

                let k_val = inputs[1];
                let v_val = inputs[2];
                let k_cached = insert_kv_cache_op(module, block_id, op_id, k_val, "key", max_seq);
                let v_cached = insert_kv_cache_op(module, block_id, op_id, v_val, "value", max_seq);

                let op_mut = &mut module.ops[op_id];
                op_mut.operands[1] = k_cached;
                op_mut.operands[2] = v_cached;
                op_mut
                    .attributes
                    .insert("kv_cached".to_string(), Attribute::Bool(true));
                created += 2;
                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn insert_kv_cache_op(
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
