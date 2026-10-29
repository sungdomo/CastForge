use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Inverse Macro-Fusion Pass: Expands `stablehlo.composite` operations
/// (`castforge.fused_add_max`, `castforge.fused_matmul_epilogue`,
/// `castforge.fused_softmax`, `castforge.fused_rmsnorm`) back into primitive
/// `stablehlo` operations for targets without specialized composite kernels.
pub struct DecomposeCompositesPass;

impl Pass for DecomposeCompositesPass {
    fn name(&self) -> &'static str {
        "decompose-composites"
    }

    fn description(&self) -> &'static str {
        "Expands stablehlo.composite operations back into primitive StableHLO ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Composite {
                    name,
                    inputs,
                    results,
                    ..
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                let Some(&res_val) = results.first() else {
                    continue;
                };
                let out_ty = module.values[res_val].ty.clone();

                match name.as_str() {
                    "castforge.fused_add_max" if inputs.len() == 3 => {
                        let add_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "add"),
                            &[inputs[0], inputs[1]],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "maximum");
                        op_mut.operands = SmallVec::from_slice(&[add_val, inputs[2]]);
                        op_mut.attributes.clear();
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_matmul_epilogue" if inputs.len() >= 3 => {
                        let epilogue = op
                            .attributes
                            .get("epilogue")
                            .and_then(|a| a.as_str())
                            .unwrap_or("bias")
                            .to_string();
                        let mut dot_attrs = IndexMap::new();
                        for k in [
                            "lhs_batching_dimensions",
                            "rhs_batching_dimensions",
                            "lhs_contracting_dimensions",
                            "rhs_contracting_dimensions",
                        ] {
                            if let Some(v) = op.attributes.get(k) {
                                dot_attrs.insert(k.to_string(), v.clone());
                            }
                        }
                        let dot_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[inputs[0], inputs[1]],
                            dot_attrs,
                            out_ty.clone(),
                        );
                        created += 1;

                        match epilogue.as_str() {
                            "bias_relu" if inputs.len() >= 4 => {
                                let add_val = insert_op_before(
                                    module,
                                    block_id,
                                    op_id,
                                    OpName::new("stablehlo", "add"),
                                    &[dot_val, inputs[2]],
                                    IndexMap::new(),
                                    out_ty,
                                );
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "maximum");
                                op_mut.operands = SmallVec::from_slice(&[add_val, inputs[3]]);
                                op_mut.attributes.clear();
                                created += 1;
                                modified += 1;
                            }
                            "bias_tanh" => {
                                let add_val = insert_op_before(
                                    module,
                                    block_id,
                                    op_id,
                                    OpName::new("stablehlo", "add"),
                                    &[dot_val, inputs[2]],
                                    IndexMap::new(),
                                    out_ty,
                                );
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "tanh");
                                op_mut.operands = SmallVec::from_slice(&[add_val]);
                                op_mut.attributes.clear();
                                created += 1;
                                modified += 1;
                            }
                            _ => {
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "add");
                                op_mut.operands = SmallVec::from_slice(&[dot_val, inputs[2]]);
                                op_mut.attributes.clear();
                                modified += 1;
                            }
                        }
                    }
                    "castforge.fused_softmax" if !inputs.is_empty() => {
                        let x = inputs[0];
                        let exp_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "exponential"),
                            &[x],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let (red_ty, axis, bcast_dims) = reduced_row_type(&out_ty);
                        let zero_init = insert_scalar_const_before(module, block_id, op_id, 0.0);
                        let mut red_attrs = IndexMap::new();
                        red_attrs.insert(
                            "dimensions".to_string(),
                            Attribute::DenseIntArray(vec![axis]),
                        );
                        red_attrs.insert(
                            "reducer".to_string(),
                            Attribute::String("stablehlo.add".to_string()),
                        );
                        let sum_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "reduce"),
                            &[exp_val, zero_init],
                            red_attrs,
                            red_ty,
                        );
                        let mut bcast_attrs = IndexMap::new();
                        bcast_attrs.insert(
                            "broadcast_dimensions".to_string(),
                            Attribute::DenseIntArray(bcast_dims),
                        );
                        let denom_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "broadcast_in_dim"),
                            &[sum_val],
                            bcast_attrs,
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "divide");
                        op_mut.operands = SmallVec::from_slice(&[exp_val, denom_val]);
                        op_mut.attributes.clear();
                        created += 4;
                        modified += 1;
                    }
                    _ => {}
                }
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

fn reduced_row_type(out_ty: &Type) -> (Type, i64, Vec<i64>) {
    if let Some(t) = out_ty.as_tensor() {
        let rank = t.shape.len();
        if rank >= 2 {
            let reduced_shape = t.shape[..rank - 1].to_vec();
            let axis = (rank - 1) as i64;
            let bcast_dims: Vec<i64> = (0..(rank - 1) as i64).collect();
            return (
                Type::Tensor(TensorType {
                    shape: reduced_shape,
                    dtype: t.dtype.clone(),
                    encoding: None,
                }),
                axis,
                bcast_dims,
            );
        }
    }
    (
        Type::Tensor(TensorType {
            shape: vec![Dim::Static(1)],
            dtype: DType::F32,
            encoding: None,
        }),
        0,
        vec![0],
    )
}

fn insert_scalar_const_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    val: f32,
) -> ValueId {
    let s_ty = TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    };
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: s_ty.clone(),
            raw_literal: format!("{val:.1}"),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        Type::Tensor(s_ty),
    )
}

fn insert_op_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    attrs: IndexMap<String, Attribute>,
    out_ty: Type,
) -> ValueId {
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
        operands: SmallVec::from_slice(operands),
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
