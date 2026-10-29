use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, DType, Module, OpName, ValueDef};

/// Fold Convert Pass (`fold-convert`).
///
/// Simplifies `stablehlo.convert` operations by:
/// - Eliminating identity conversions (`convert(x) : T -> T`).
/// - Collapsing chained conversions `convert(convert(x : T0) : T1) : T2 -> convert(x : T0) : T2`
///   when the intermediate type `T1` is a lossless widening of `T0` (or `T2 == T0` after widening,
///   such as `f16 -> f32 -> f16` or `bf16 -> f32 -> bf16`, which collapses directly to `x`).
/// - Folding `convert(constant)` on splat constants directly into a target-typed constant.
pub struct FoldConvertPass;

fn is_float_dtype(dt: &DType) -> bool {
    matches!(
        dt,
        DType::F8E4M3FN | DType::F8E5M2 | DType::F16 | DType::BF16 | DType::F32 | DType::F64
    )
}

fn is_lossless_intermediate(t0: &DType, t1: &DType, t2: &DType) -> bool {
    if t0 == t1 || t1 == t2 {
        return true;
    }
    // Widening float cast (e.g. f16 -> f32 -> f16, bf16 -> f32 -> bf16, f16 -> f64 -> f32)
    if is_float_dtype(t0)
        && is_float_dtype(t1)
        && is_float_dtype(t2)
        && t1.bit_width() >= t0.bit_width()
        && t1.bit_width() >= t2.bit_width()
    {
        return true;
    }
    // Widening signed integer cast (e.g. i8 -> i32 -> i16)
    let is_sint = |d: &DType| {
        matches!(
            d,
            DType::I4 | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    };
    is_sint(t0)
        && is_sint(t1)
        && is_sint(t2)
        && t1.bit_width() >= t0.bit_width()
        && t1.bit_width() >= t2.bit_width()
}

impl Pass for FoldConvertPass {
    fn name(&self) -> &'static str {
        "fold-convert"
    }

    fn description(&self) -> &'static str {
        "Eliminates redundant or round-trip stablehlo.convert casts and folds constant casts"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Convert { operand, result }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut cur_operand = operand;

                // 1. Check for chained `convert(convert(x : T0) : T1) : T2`
                if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                    && let Some(prod_op) = module.ops.get(prod_id)
                    && let Some(StableHloOp::Convert {
                        operand: base_in, ..
                    }) = StableHloOp::from_op(prod_op)
                    && let (Some(t0), Some(t1), Some(t2)) = (
                        module.values[base_in].ty.as_tensor().map(|t| &t.dtype),
                        module.values[operand].ty.as_tensor().map(|t| &t.dtype),
                        module.values[result].ty.as_tensor().map(|t| &t.dtype),
                    )
                    && is_lossless_intermediate(t0, t1, t2)
                {
                    cur_operand = base_in;
                    module.ops[op_id].operands[0] = cur_operand;
                    modified += 1;
                }

                // 2. Identity conversion elimination (`T -> T`)
                if module.values[cur_operand].ty == module.values[result].ty {
                    module.replace_all_uses(result, cur_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                    continue;
                }

                // 3. Fold `convert(splat_constant)` into a target-typed constant
                if let ValueDef::OpResult { op: prod_id, .. } = module.values[cur_operand].def
                    && let Some(prod_op) = module.ops.get(prod_id)
                    && prod_op.is_op("stablehlo", "constant")
                    && let Some(Attribute::DenseElements { raw_literal, .. }) =
                        prod_op.attributes.get("value")
                    && !raw_literal.trim().starts_with('[')
                    && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
                {
                    let lit = raw_literal.clone();
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "constant");
                    op_mut.operands.clear();
                    op_mut.attributes.clear();
                    op_mut.attributes.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty: res_ty,
                            raw_literal: lit,
                        },
                    );
                    modified += 1;
                }
            }
        }

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
