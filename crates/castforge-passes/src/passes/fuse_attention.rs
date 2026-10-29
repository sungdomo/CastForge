//! Scaled Dot-Product Attention (SDPA) and Grouped-Query Attention (GQA) fusion (`fuse-attention`).

use crate::{DeadCodeEliminationPass, FuseSoftmaxPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// AI Optimization Pass: Scaled Dot-Product & Grouped-Query Attention Fusion (`fuse-attention`).
///
/// Fuses `dot_general(softmax(multiply(dot_general(q, k), scale)), v)` (including GQA
/// `broadcast_in_dim` / `reshape` head-expansion patterns on `k` and `v`) into a single
/// `stablehlo.composite` operation (`"castforge.fused_attention"`).
pub struct FuseAttentionPass;

struct AttentionMatch {
    q: ValueId,
    k: ValueId,
    v: ValueId,
    mask: Option<ValueId>,
    scale: f64,
    k_transposed: bool,
    is_gqa: bool,
}

impl Pass for FuseAttentionPass {
    fn name(&self) -> &'static str {
        "fuse-attention"
    }

    fn description(&self) -> &'static str {
        "Fuses Q*K^T + scale + softmax + P*V (with GQA support) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        // First normalize any raw softmax subgraphs into `castforge.fused_softmax`.
        let softmax_res = FuseSoftmaxPass.run(module)?;
        let mut created = 0usize;
        let mut removed = softmax_res.ops_removed;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    lhs: probs_val,
                    rhs: v_val,
                    ..
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let Some(AttentionMatch {
                    q,
                    k,
                    v,
                    mask,
                    scale,
                    k_transposed,
                    is_gqa,
                }) = match_sdpa_chain(module, probs_val, v_val)
                else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_attention".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_attention_impl".to_string()),
                );
                attrs.insert("scale".to_string(), Attribute::Float(scale));
                if k_transposed {
                    attrs.insert("k_transposed".to_string(), Attribute::Bool(true));
                }
                if is_gqa {
                    attrs.insert("gqa".to_string(), Attribute::Bool(true));
                }
                if mask.is_some() {
                    attrs.insert("masked".to_string(), Attribute::Bool(true));
                }

                let mut operands = SmallVec::from_slice(&[q, k, v]);
                if let Some(m_val) = mask {
                    operands.push(m_val);
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = operands;
                op_mut.attributes = attrs;
                created += 1;
            }
        }

        if created > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

fn match_sdpa_chain(module: &Module, probs_val: ValueId, v_val: ValueId) -> Option<AttentionMatch> {
    let ValueDef::OpResult { op: sm_op_id, .. } = module.values[probs_val].def else {
        return None;
    };
    let sm_op = module.ops.get(sm_op_id)?;
    let StableHloOp::Composite { name, inputs, .. } = StableHloOp::from_op(sm_op)? else {
        return None;
    };
    if name != "castforge.fused_softmax" || inputs.is_empty() {
        return None;
    }

    let scores_val = inputs[0];
    let (pre_mask_val, mask) = unwrap_optional_mask(module, scores_val);
    let (qk_val, scale) = unwrap_optional_scale(module, pre_mask_val);

    let ValueDef::OpResult { op: qk_op_id, .. } = module.values[qk_val].def else {
        return None;
    };
    let qk_op = module.ops.get(qk_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: q,
        rhs: k_raw,
        dimension_numbers: dims,
        ..
    } = StableHloOp::from_op(qk_op)?
    else {
        return None;
    };

    let (k_unwrapped, k_from_transpose) = unwrap_optional_transpose(module, k_raw);
    let k_transposed = k_from_transpose || dims.rhs_contracting_dimensions == [1];
    let is_gqa = is_gqa_expanded(module, k_unwrapped) || is_gqa_expanded(module, v_val);

    Some(AttentionMatch {
        q,
        k: k_unwrapped,
        v: v_val,
        mask,
        scale,
        k_transposed,
        is_gqa,
    })
}

fn unwrap_optional_mask(module: &Module, val: ValueId) -> (ValueId, Option<ValueId>) {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return (val, None);
    };
    let Some(op) = module.ops.get(op_id) else {
        return (val, None);
    };
    if let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(op) {
        let (qk_cand, _) = unwrap_optional_scale(module, lhs);
        if let ValueDef::OpResult { op: qk_id, .. } = module.values[qk_cand].def
            && module
                .ops
                .get(qk_id)
                .and_then(StableHloOp::from_op)
                .is_some_and(|h| matches!(h, StableHloOp::DotGeneral { .. }))
        {
            return (lhs, Some(rhs));
        }
        let (qk_cand_r, _) = unwrap_optional_scale(module, rhs);
        if let ValueDef::OpResult { op: qk_id, .. } = module.values[qk_cand_r].def
            && module
                .ops
                .get(qk_id)
                .and_then(StableHloOp::from_op)
                .is_some_and(|h| matches!(h, StableHloOp::DotGeneral { .. }))
        {
            return (rhs, Some(lhs));
        }
    }
    (val, None)
}

fn unwrap_optional_scale(module: &Module, val: ValueId) -> (ValueId, f64) {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return (val, 1.0);
    };
    let Some(op) = module.ops.get(op_id) else {
        return (val, 1.0);
    };
    if let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) {
        if let Some(s) = extract_scalar_constant(module, rhs) {
            return (lhs, s);
        }
        if let Some(s) = extract_scalar_constant(module, lhs) {
            return (rhs, s);
        }
    }
    (val, 1.0)
}

fn extract_scalar_constant(module: &Module, val: ValueId) -> Option<f64> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    let StableHloOp::Constant { value, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    match value {
        Attribute::Float(f) => Some(f),
        Attribute::DenseElements { raw_literal, .. } => raw_literal.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn unwrap_optional_transpose(module: &Module, val: ValueId) -> (ValueId, bool) {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return (val, false);
    };
    let Some(op) = module.ops.get(op_id) else {
        return (val, false);
    };
    if let Some(StableHloOp::Transpose {
        operand,
        permutation,
        ..
    }) = StableHloOp::from_op(op)
        && permutation == [1, 0]
    {
        return (operand, true);
    }
    (val, false)
}

fn is_gqa_expanded(module: &Module, val: ValueId) -> bool {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val].def else {
        return false;
    };
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    matches!(
        StableHloOp::from_op(op),
        Some(StableHloOp::BroadcastInDim { .. } | StableHloOp::Reshape { .. })
    )
}
