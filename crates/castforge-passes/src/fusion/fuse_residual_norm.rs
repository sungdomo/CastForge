use crate::{DeadCodeEliminationPass, FuseRmsNormPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, ValueDef};
use smallvec::SmallVec;

/// Residual-Add + RMSNorm Fusion Pass (`fuse-residual-norm`).
///
/// Recognizes `h = stablehlo.add(x, residual)` feeding `castforge.fused_rmsnorm(h, weight)`
/// (automatically running `fuse-rmsnorm` first if RMSNorm is still decomposed) and fuses the
/// residual addition directly into `"castforge.fused_residual_rmsnorm"` taking
/// `(x, residual, weight)` (or `(x, residual)`), eliminating a global-memory read/write pass over
/// the residual stream.
pub struct FuseResidualNormPass;

impl Pass for FuseResidualNormPass {
    fn name(&self) -> &'static str {
        "fuse-residual-norm"
    }

    fn description(&self) -> &'static str {
        "Fuses add(x, residual) + RMSNorm into stablehlo.composite (fused_residual_rmsnorm)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let rms_res = FuseRmsNormPass.run(module)?;
        let mut created = rms_res.ops_created;
        let mut removed = rms_res.ops_removed;
        let mut modified = 0usize;
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
                if name != "castforge.fused_rmsnorm" || inputs.is_empty() {
                    continue;
                }

                let h_val = inputs[0];
                let ValueDef::OpResult { op: add_op_id, .. } = module.values[h_val].def else {
                    continue;
                };
                let Some(add_op) = module.ops.get(add_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add {
                    lhs: x, rhs: res, ..
                }) = StableHloOp::from_op(add_op)
                else {
                    continue;
                };

                let mut new_operands = SmallVec::from_slice(&[x, res]);
                for &extra in inputs.iter().skip(1) {
                    new_operands.push(extra);
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.operands = new_operands;
                op_mut.attributes.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_residual_rmsnorm".to_string()),
                );
                op_mut.attributes.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_residual_rmsnorm_impl".to_string()),
                );
                modified += 1;
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
            created += modified;
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
