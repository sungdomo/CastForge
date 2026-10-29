use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, DType, Module, OpName, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Quantized MatMul Fusion Pass (`fuse-dequant-matmul`).
///
/// Fuses `stablehlo.uniform_dequantize(w_q) -> stablehlo.dot_general(x, w)` (as well as
/// `castforge.fused_matmul_epilogue` with a dequantized RHS weight) into a single
/// `stablehlo.composite` operation (`"castforge.fused_qmatmul"`) that dequantizes `i8`/`i4`
/// weights on the fly in registers inside the GEMM inner loop (`(w_q[k, j] - zp) * scale`),
/// eliminating intermediate full-precision weight materialization and reducing weight memory
/// bandwidth by 4x–8x.
pub struct FuseDequantMatmulPass;

impl Pass for FuseDequantMatmulPass {
    fn name(&self) -> &'static str {
        "fuse-dequant-matmul"
    }

    fn description(&self) -> &'static str {
        "Fuses uniform_dequantize + dot_general into on-the-fly quantized GEMM (fused_qmatmul)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };

                match StableHloOp::from_op(&op) {
                    Some(StableHloOp::DotGeneral {
                        lhs,
                        rhs,
                        dimension_numbers: dims,
                        ..
                    }) => {
                        let Some((w_q, scale, zp)) = match_uniform_dequantize(module, rhs) else {
                            continue;
                        };
                        let packed = dims.rhs_contracting_dimensions == [1];
                        let mut attrs = IndexMap::new();
                        attrs.insert(
                            "name".to_string(),
                            Attribute::String("castforge.fused_qmatmul".to_string()),
                        );
                        attrs.insert(
                            "decomposition".to_string(),
                            Attribute::SymbolRef("fused_qmatmul_impl".to_string()),
                        );
                        attrs.insert("scale".to_string(), Attribute::Float(scale));
                        attrs.insert("zero_point".to_string(), Attribute::Int(zp));
                        if packed {
                            attrs.insert("packed_rhs".to_string(), Attribute::Bool(true));
                        }

                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "composite");
                        op_mut.operands = SmallVec::from_slice(&[lhs, w_q]);
                        op_mut.attributes = attrs;
                        created += 1;
                    }
                    Some(StableHloOp::Composite { name, inputs, .. })
                        if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 =>
                    {
                        let Some((w_q, scale, zp)) = match_uniform_dequantize(module, inputs[1])
                        else {
                            continue;
                        };
                        let mut new_operands = inputs.clone();
                        new_operands[1] = w_q;

                        let op_mut = &mut module.ops[op_id];
                        op_mut.operands = SmallVec::from_vec(new_operands);
                        op_mut.attributes.insert(
                            "name".to_string(),
                            Attribute::String("castforge.fused_qmatmul".to_string()),
                        );
                        op_mut.attributes.insert(
                            "decomposition".to_string(),
                            Attribute::SymbolRef("fused_qmatmul_impl".to_string()),
                        );
                        op_mut
                            .attributes
                            .insert("scale".to_string(), Attribute::Float(scale));
                        op_mut
                            .attributes
                            .insert("zero_point".to_string(), Attribute::Int(zp));
                        created += 1;
                    }
                    _ => {}
                }
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

fn match_uniform_dequantize(module: &Module, val_id: ValueId) -> Option<(ValueId, f64, i64)> {
    let ValueDef::OpResult { op: deq_op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let deq_op = module.ops.get(deq_op_id)?;
    if !deq_op.is_op("stablehlo", "uniform_dequantize") || deq_op.operands.len() != 1 {
        return None;
    }
    let w_q = deq_op.operands[0];
    let (scale, zp) = match module.values[w_q].ty.as_tensor().map(|t| &t.dtype) {
        Some(DType::QuantUniform {
            scale, zero_point, ..
        }) => (scale.parse::<f64>().unwrap_or(1.0), *zero_point),
        _ => (1.0, 0),
    };
    Some((w_q, scale, zp))
}
