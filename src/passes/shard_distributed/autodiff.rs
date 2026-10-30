//! Submodule `autodiff` of `shard_distributed`.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId, format_dense_f32_literal,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Reverse-Mode Automatic Differentiation Pass (`autodiff`).
pub struct AutodiffPass {
    pub learning_rate: f32,
}

impl Default for AutodiffPass {
    fn default() -> Self {
        Self { learning_rate: 0.0 }
    }
}

impl Pass for AutodiffPass {
    fn name(&self) -> &'static str {
        "autodiff"
    }

    fn description(&self) -> &'static str {
        "Synthesizes reverse-mode VJP gradient ops in StableHLO (optional SGD step via lr=...)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;

        for fn_id in module.functions() {
            let Some(reg_id) = module.ops[fn_id].regions.first().copied() else {
                continue;
            };
            let Some(block_id) = module.regions[reg_id].blocks.first().copied() else {
                continue;
            };
            let forward_ops = module.block_ops(block_id);
            let Some(&ret_op_id) = forward_ops
                .iter()
                .find(|&&id| module.ops[id].is_op("func", "return"))
            else {
                continue;
            };
            let Some(&ret_val) = module.ops[ret_op_id].operands.first() else {
                continue;
            };
            let ret_ty = module.values[ret_val].ty.clone();

            let seed_grad = insert_splat_const_before(module, block_id, ret_op_id, &ret_ty, 1.0);
            created += 1;

            let mut grads: HashMap<ValueId, ValueId> = HashMap::new();
            grads.insert(ret_val, seed_grad);

            for &op_id in forward_ops.iter().rev() {
                if op_id == ret_op_id {
                    continue;
                }
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(&out_val) = op.results.first() else {
                    continue;
                };
                let Some(&dy) = grads.get(&out_val) else {
                    continue;
                };
                let out_ty = module.values[out_val].ty.clone();

                let Some(hlo) = StableHloOp::from_op(&op) else {
                    continue;
                };

                match hlo {
                    StableHloOp::Add { lhs, rhs, .. } => {
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dy,
                            &mut created,
                        );
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            rhs,
                            dy,
                            &mut created,
                        );
                    }
                    StableHloOp::Subtract { lhs, rhs, .. } => {
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dy,
                            &mut created,
                        );
                        let zero =
                            insert_splat_const_before(module, block_id, ret_op_id, &out_ty, 0.0);
                        let neg_dy = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "subtract"),
                            &[zero, dy],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 2;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            rhs,
                            neg_dy,
                            &mut created,
                        );
                    }
                    StableHloOp::Multiply { lhs, rhs, .. } => {
                        let dlhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, rhs],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let drhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, lhs],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 2;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dlhs,
                            &mut created,
                        );
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            rhs,
                            drhs,
                            &mut created,
                        );
                    }
                    StableHloOp::Divide { lhs, rhs, .. } => {
                        let dlhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "divide"),
                            &[dy, rhs],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dlhs,
                            &mut created,
                        );
                    }
                    StableHloOp::Tanh { operand, .. } => {
                        let one =
                            insert_splat_const_before(module, block_id, ret_op_id, &out_ty, 1.0);
                        let y_sq = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[out_val, out_val],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let dtanh = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "subtract"),
                            &[one, y_sq],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, dtanh],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 4;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::Exp { operand, .. } => {
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, out_val],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::Maximum { lhs, rhs, .. } => {
                        let pred_dims = module.values[out_val]
                            .ty
                            .as_tensor()
                            .map(|t| t.shape.clone())
                            .unwrap_or_default();
                        let pred_ty = Type::Tensor(TensorType {
                            shape: pred_dims,
                            dtype: DType::I1,
                            encoding: None,
                        });
                        let mut cmp_attrs = IndexMap::new();
                        cmp_attrs.insert(
                            "comparison_direction".to_string(),
                            Attribute::String("GT".to_string()),
                        );
                        let mask = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "compare"),
                            &[lhs, rhs],
                            cmp_attrs,
                            pred_ty,
                        );
                        let zero =
                            insert_splat_const_before(module, block_id, ret_op_id, &out_ty, 0.0);
                        let dlhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "select"),
                            &[mask, dy, zero],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 3;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dlhs,
                            &mut created,
                        );
                    }
                    StableHloOp::Reshape { operand, .. } => {
                        let in_ty = module.values[operand].ty.clone();
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "reshape"),
                            &[dy],
                            IndexMap::new(),
                            in_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::Transpose {
                        operand,
                        permutation,
                        ..
                    } => {
                        let in_ty = module.values[operand].ty.clone();
                        let mut inv_perm = vec![0i64; permutation.len()];
                        for (dst, &src) in permutation.iter().enumerate() {
                            if (src as usize) < inv_perm.len() {
                                inv_perm[src as usize] = dst as i64;
                            }
                        }
                        let mut t_attrs = IndexMap::new();
                        t_attrs.insert(
                            "permutation".to_string(),
                            Attribute::DenseIntArray(inv_perm),
                        );
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "transpose"),
                            &[dy],
                            t_attrs,
                            in_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::DotGeneral { lhs, rhs, .. } => {
                        let lhs_ty = module.values[lhs].ty.clone();
                        let rhs_ty = module.values[rhs].ty.clone();
                        let lhs_dims = tensor_dims(&lhs_ty);
                        let rhs_dims = tensor_dims(&rhs_ty);
                        if lhs_dims.len() == 2 && rhs_dims.len() == 2 {
                            let bt_ty = Type::Tensor(TensorType {
                                shape: vec![Dim::Static(rhs_dims[1]), Dim::Static(rhs_dims[0])],
                                dtype: DType::F32,
                                encoding: None,
                            });
                            let mut t_attrs = IndexMap::new();
                            t_attrs.insert(
                                "permutation".to_string(),
                                Attribute::DenseIntArray(vec![1, 0]),
                            );
                            let bt = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "transpose"),
                                &[rhs],
                                t_attrs.clone(),
                                bt_ty,
                            );
                            let dlhs = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "dot_general"),
                                &[dy, bt],
                                dot_2d_attrs(),
                                lhs_ty,
                            );

                            let at_ty = Type::Tensor(TensorType {
                                shape: vec![Dim::Static(lhs_dims[1]), Dim::Static(lhs_dims[0])],
                                dtype: DType::F32,
                                encoding: None,
                            });
                            let at = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "transpose"),
                                &[lhs],
                                t_attrs,
                                at_ty,
                            );
                            let drhs = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "dot_general"),
                                &[at, dy],
                                dot_2d_attrs(),
                                rhs_ty,
                            );
                            created += 4;
                            accumulate_grad(
                                module,
                                block_id,
                                ret_op_id,
                                &mut grads,
                                lhs,
                                dlhs,
                                &mut created,
                            );
                            accumulate_grad(
                                module,
                                block_id,
                                ret_op_id,
                                &mut grads,
                                rhs,
                                drhs,
                                &mut created,
                            );
                        }
                    }
                    _ => {}
                }
            }

            let block_args = module.blocks[block_id].arguments.clone();
            let target_arg = block_args
                .iter()
                .rev()
                .copied()
                .find(|&arg| grads.contains_key(&arg) && module.values[arg].ty == ret_ty);

            if let Some(arg_val) = target_arg
                && let Some(&grad_val) = grads.get(&arg_val)
            {
                let new_ret = if self.learning_rate > 0.0 {
                    let lr_const = insert_splat_const_before(
                        module,
                        block_id,
                        ret_op_id,
                        &ret_ty,
                        self.learning_rate,
                    );
                    let step = insert_vjp_op(
                        module,
                        block_id,
                        ret_op_id,
                        OpName::new("stablehlo", "multiply"),
                        &[grad_val, lr_const],
                        IndexMap::new(),
                        ret_ty.clone(),
                    );
                    let updated = insert_vjp_op(
                        module,
                        block_id,
                        ret_op_id,
                        OpName::new("stablehlo", "subtract"),
                        &[arg_val, step],
                        IndexMap::new(),
                        ret_ty,
                    );
                    created += 3;
                    updated
                } else {
                    grad_val
                };
                module.ops[ret_op_id].operands[0] = new_ret;
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

fn dot_2d_attrs() -> IndexMap<String, Attribute> {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "lhs_contracting_dimensions".to_string(),
        Attribute::DenseIntArray(vec![1]),
    );
    attrs.insert(
        "rhs_contracting_dimensions".to_string(),
        Attribute::DenseIntArray(vec![0]),
    );
    attrs
}

fn tensor_dims(ty: &Type) -> Vec<i64> {
    ty.as_tensor()
        .map(|t| {
            t.shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n,
                    _ => 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn accumulate_grad(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    grads: &mut HashMap<ValueId, ValueId>,
    target: ValueId,
    contrib: ValueId,
    created: &mut usize,
) {
    if module.values[target].ty != module.values[contrib].ty {
        return;
    }
    if let Some(&existing) = grads.get(&target) {
        let ty = module.values[target].ty.clone();
        let sum = insert_vjp_op(
            module,
            block_id,
            before_op,
            OpName::new("stablehlo", "add"),
            &[existing, contrib],
            IndexMap::new(),
            ty,
        );
        *created += 1;
        grads.insert(target, sum);
    } else {
        grads.insert(target, contrib);
    }
}

fn insert_splat_const_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    ty: &Type,
    val: f32,
) -> ValueId {
    let dims = tensor_dims(ty);
    let dim_objs: Vec<Dim> = dims.iter().map(|&d| Dim::Static(d)).collect();
    let num_elems: usize = dims.iter().map(|&d| d.max(1) as usize).product::<usize>();
    let lit = format_dense_f32_literal(&vec![val; num_elems.max(1)], &dim_objs);
    let tensor_ty = ty
        .as_tensor()
        .cloned()
        .unwrap_or_else(|| TensorType::new(dim_objs, DType::F32));
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: tensor_ty,
            raw_literal: lit,
        },
    );
    insert_vjp_op(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        ty.clone(),
    )
}

fn insert_vjp_op(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    mut attrs: IndexMap<String, Attribute>,
    out_ty: Type,
) -> ValueId {
    if name.name != "constant" {
        attrs.insert(
            "autodiff_role".to_string(),
            Attribute::String("vjp_grad".to_string()),
        );
    }
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
