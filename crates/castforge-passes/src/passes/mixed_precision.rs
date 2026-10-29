use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Location, Module, OpId, OpName, Operation, TensorType, Type, Value,
    ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Automatic Mixed Precision Pass (`mixed-precision`).
///
/// Inserts `stablehlo.convert` downcasts (`f32 -> f16` or `bf16`) and upcasts on the operands of
/// compute-intensive `dot_general` and `convolution` operations while keeping numerically
/// sensitive reductions and normalizations (`reduce`, `fused_rmsnorm`, `fused_layernorm`,
/// `fused_softmax`, `fused_log_softmax`) in `f32` accumulation precision.
pub struct MixedPrecisionPass {
    pub target_dtype: DType,
}

impl Default for MixedPrecisionPass {
    fn default() -> Self {
        Self {
            target_dtype: DType::F16,
        }
    }
}

impl Pass for MixedPrecisionPass {
    fn name(&self) -> &'static str {
        "mixed-precision"
    }

    fn description(&self) -> &'static str {
        "Applies Automatic Mixed Precision (f16/bf16) casts to GEMM/Conv ops, keeping norms in f32"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let low_dtype = match &self.target_dtype {
            DType::BF16 => DType::BF16,
            _ => DType::F16,
        };
        let prec_label = match &low_dtype {
            DType::BF16 => "bf16",
            _ => "f16",
        };
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.attributes.contains_key("compute_precision") {
                    continue;
                }
                let is_compute_heavy = matches!(
                    StableHloOp::from_op(&op),
                    Some(StableHloOp::DotGeneral { .. } | StableHloOp::Convolution { .. })
                );
                if !is_compute_heavy || op.operands.len() < 2 {
                    continue;
                }

                let mut new_operands = op.operands.clone();
                for opnd_slot in new_operands.iter_mut().take(2) {
                    let orig_val = *opnd_slot;
                    let Some(t_ty) = module.values[orig_val].ty.as_tensor().cloned() else {
                        continue;
                    };
                    if t_ty.dtype != DType::F32 {
                        continue;
                    }
                    let low_ty = Type::Tensor(TensorType {
                        shape: t_ty.shape.clone(),
                        dtype: low_dtype.clone(),
                        encoding: None,
                    });
                    let f32_ty = Type::Tensor(t_ty);

                    let down_val = insert_convert_before(module, block_id, op_id, orig_val, low_ty);
                    let up_val = insert_convert_before(module, block_id, op_id, down_val, f32_ty);
                    *opnd_slot = up_val;
                    created += 2;
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.operands = new_operands;
                op_mut.attributes.insert(
                    "compute_precision".to_string(),
                    Attribute::String(prec_label.to_string()),
                );
                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn insert_convert_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input_val: ValueId,
    out_ty: Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert("amp_cast".to_string(), Attribute::Bool(true));
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "convert"),
        operands: SmallVec::from_slice(&[input_val]),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(val_id);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, new_op_id);
    } else {
        block_ops.push(new_op_id);
    }
    val_id
}
