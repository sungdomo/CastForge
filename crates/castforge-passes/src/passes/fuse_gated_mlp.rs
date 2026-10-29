use crate::{
    DeadCodeEliminationPass, FuseGeluPass, FuseHorizontalDotPass, FuseSwiGluPass, Pass, PassResult,
};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Gated MLP Macro-Fusion Pass (`fuse-gated-mlp`).
///
/// Fuses horizontally-concatenated dual projections (`dot_general(x, W_gate_up)`) +
/// column `slice`s + gated activation (`castforge.fused_swiglu` or `castforge.fused_geglu`)
/// into a single `stablehlo.composite` operation (`"castforge.fused_gated_mlp"`), avoiding
/// materialization of the `[M, 2 * N]` intermediate projection buffer in memory.
pub struct FuseGatedMlpPass;

impl Pass for FuseGatedMlpPass {
    fn name(&self) -> &'static str {
        "fuse-gated-mlp"
    }

    fn description(&self) -> &'static str {
        "Fuses dual gate/up dot_general + slices + SwiGLU/GeGLU into fused_gated_mlp"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let h_res = FuseHorizontalDotPass.run(module)?;
        let sw_res = FuseSwiGluPass.run(module)?;
        let ge_res = FuseGeluPass.run(module)?;

        let mut created = h_res.ops_created + sw_res.ops_created + ge_res.ops_created;
        let mut removed = h_res.ops_removed + sw_res.ops_removed + ge_res.ops_removed;
        let mut fused_count = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                let act_kind = if name == "castforge.fused_swiglu" {
                    "swiglu"
                } else if name == "castforge.fused_geglu" {
                    "geglu"
                } else {
                    continue;
                };
                if inputs.len() < 2 {
                    continue;
                }

                let Some((dot_val_0, start_col_0)) = match_2d_col_slice(module, inputs[0]) else {
                    continue;
                };
                let Some((dot_val_1, start_col_1)) = match_2d_col_slice(module, inputs[1]) else {
                    continue;
                };
                if dot_val_0 != dot_val_1 || start_col_0 != 0 || start_col_1 <= 0 {
                    continue;
                }

                let ValueDef::OpResult { op: dot_op_id, .. } = module.values[dot_val_0].def else {
                    continue;
                };
                let Some(dot_op) = module.ops.get(dot_op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    lhs: x_val,
                    rhs: w_gate_up,
                    dimension_numbers: dims,
                    ..
                }) = StableHloOp::from_op(dot_op)
                else {
                    continue;
                };
                let packed = dims.rhs_contracting_dimensions == [1];

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_gated_mlp".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_gated_mlp_impl".to_string()),
                );
                attrs.insert(
                    "activation".to_string(),
                    Attribute::String(act_kind.to_string()),
                );
                if packed {
                    attrs.insert("packed_rhs".to_string(), Attribute::Bool(true));
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x_val, w_gate_up]);
                op_mut.attributes = attrs;
                fused_count += 1;
            }
        }

        if fused_count > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
            created += fused_count;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || fused_count > 0,
            ops_removed: removed,
            ops_modified: fused_count,
            ops_created: created,
        })
    }
}

fn match_2d_col_slice(
    module: &Module,
    val_id: castforge_ir::ValueId,
) -> Option<(castforge_ir::ValueId, i64)> {
    let ValueDef::OpResult { op: s_id, .. } = module.values[val_id].def else {
        return None;
    };
    let s_op = module.ops.get(s_id)?;
    let StableHloOp::Slice {
        operand,
        start_indices,
        ..
    } = StableHloOp::from_op(s_op)?
    else {
        return None;
    };
    if start_indices.len() == 2 && start_indices[0] == 0 {
        Some((operand, start_indices[1]))
    } else {
        None
    }
}
