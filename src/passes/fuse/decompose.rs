//! Inverse Macro-Fusion (`DecomposeCompositesPass`): expands `stablehlo.composite` operations back
//! into primitive `stablehlo` operations for targets without specialized composite kernels.

use super::decompose_helpers::{
    half_col_type, insert_gelu_before, insert_layernorm_core_before, insert_op_before,
    insert_rmsnorm_core_before, insert_scalar_const_before, insert_silu_denom_before,
    insert_slice_2d_before, insert_splat_tensor_before, insert_zero_tensor_before,
    reduced_row_type,
};
use crate::ir::{Attribute, BlockId, DType, Dim, Module, OpName, TensorType, Type};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Inverse Macro-Fusion Pass: Expands `stablehlo.composite` operations
/// (`castforge.fused_add_max`, `castforge.fused_matmul_epilogue`,
/// `castforge.fused_softmax`, `castforge.fused_rmsnorm`, etc.) back into primitive
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
                        let (base_ep, has_residual) =
                            if let Some(base) = epilogue.strip_suffix("_residual") {
                                (base, true)
                            } else {
                                (epilogue.as_str(), false)
                            };
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

                        if !has_residual {
                            match base_ep {
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
                                "bias_silu" => {
                                    let add_val = insert_op_before(
                                        module,
                                        block_id,
                                        op_id,
                                        OpName::new("stablehlo", "add"),
                                        &[dot_val, inputs[2]],
                                        IndexMap::new(),
                                        out_ty.clone(),
                                    );
                                    let denom_val = insert_silu_denom_before(
                                        module, block_id, op_id, add_val, &out_ty,
                                    );
                                    let op_mut = &mut module.ops[op_id];
                                    op_mut.name = OpName::new("stablehlo", "divide");
                                    op_mut.operands = SmallVec::from_slice(&[add_val, denom_val]);
                                    op_mut.attributes.clear();
                                    created += 6;
                                    modified += 1;
                                }
                                "bias_gelu" => {
                                    let add_val = insert_op_before(
                                        module,
                                        block_id,
                                        op_id,
                                        OpName::new("stablehlo", "add"),
                                        &[dot_val, inputs[2]],
                                        IndexMap::new(),
                                        out_ty.clone(),
                                    );
                                    let gelu_val = insert_gelu_before(
                                        module, block_id, op_id, add_val, &out_ty,
                                    );
                                    let zero_t =
                                        insert_zero_tensor_before(module, block_id, op_id, &out_ty);
                                    let op_mut = &mut module.ops[op_id];
                                    op_mut.name = OpName::new("stablehlo", "add");
                                    op_mut.operands = SmallVec::from_slice(&[gelu_val, zero_t]);
                                    op_mut.attributes.clear();
                                    created += 10;
                                    modified += 1;
                                }
                                "bias_scale" if inputs.len() >= 4 => {
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
                                    op_mut.name = OpName::new("stablehlo", "multiply");
                                    op_mut.operands = SmallVec::from_slice(&[add_val, inputs[3]]);
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
                        } else if let Some(&res_in) = inputs.last() {
                            let add_val = insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", "add"),
                                &[dot_val, inputs[2]],
                                IndexMap::new(),
                                out_ty.clone(),
                            );
                            created += 1;
                            let act_val = match base_ep {
                                "bias_relu" if inputs.len() >= 5 => {
                                    created += 1;
                                    insert_op_before(
                                        module,
                                        block_id,
                                        op_id,
                                        OpName::new("stablehlo", "maximum"),
                                        &[add_val, inputs[3]],
                                        IndexMap::new(),
                                        out_ty.clone(),
                                    )
                                }
                                "bias_tanh" => {
                                    created += 1;
                                    insert_op_before(
                                        module,
                                        block_id,
                                        op_id,
                                        OpName::new("stablehlo", "tanh"),
                                        &[add_val],
                                        IndexMap::new(),
                                        out_ty.clone(),
                                    )
                                }
                                "bias_silu" => {
                                    let denom_val = insert_silu_denom_before(
                                        module, block_id, op_id, add_val, &out_ty,
                                    );
                                    created += 6;
                                    insert_op_before(
                                        module,
                                        block_id,
                                        op_id,
                                        OpName::new("stablehlo", "divide"),
                                        &[add_val, denom_val],
                                        IndexMap::new(),
                                        out_ty.clone(),
                                    )
                                }
                                "bias_gelu" => {
                                    created += 8;
                                    insert_gelu_before(module, block_id, op_id, add_val, &out_ty)
                                }
                                "bias_scale" if inputs.len() >= 5 => {
                                    created += 1;
                                    insert_op_before(
                                        module,
                                        block_id,
                                        op_id,
                                        OpName::new("stablehlo", "multiply"),
                                        &[add_val, inputs[3]],
                                        IndexMap::new(),
                                        out_ty.clone(),
                                    )
                                }
                                _ => add_val,
                            };
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "add");
                            op_mut.operands = SmallVec::from_slice(&[act_val, res_in]);
                            op_mut.attributes.clear();
                            modified += 1;
                        }
                    }
                    "castforge.fused_swiglu" if inputs.len() >= 2 => {
                        let gate = inputs[0];
                        let up = inputs[1];
                        let denom_val =
                            insert_silu_denom_before(module, block_id, op_id, gate, &out_ty);
                        let silu_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "divide"),
                            &[gate, denom_val],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[silu_val, up]);
                        op_mut.attributes.clear();
                        created += 6;
                        modified += 1;
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
                    "castforge.fused_rope" if inputs.len() >= 3 => {
                        let (x, cos_v, sin_v) = (inputs[0], inputs[1], inputs[2]);
                        let (half_ty, rows, cols) = half_col_type(&out_ty);
                        let half = cols / 2;
                        let x_lo = insert_slice_2d_before(
                            module,
                            block_id,
                            op_id,
                            x,
                            [0, 0],
                            [rows, half],
                            half_ty.clone(),
                        );
                        let x_hi = insert_slice_2d_before(
                            module,
                            block_id,
                            op_id,
                            x,
                            [0, half],
                            [rows, cols],
                            half_ty.clone(),
                        );
                        let zero_hi = insert_zero_tensor_before(module, block_id, op_id, &half_ty);
                        let neg_hi = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "subtract"),
                            &[zero_hi, x_hi],
                            IndexMap::new(),
                            half_ty,
                        );
                        let mut cat_attrs = IndexMap::new();
                        cat_attrs.insert("dimension".to_string(), Attribute::Int(1));
                        let rot_x = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "concatenate"),
                            &[neg_hi, x_lo],
                            cat_attrs,
                            out_ty.clone(),
                        );
                        let mul_cos = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[x, cos_v],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let mul_sin = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[rot_x, sin_v],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[mul_cos, mul_sin]);
                        op_mut.attributes.clear();
                        created += 7;
                        modified += 1;
                    }
                    "castforge.fused_rmsnorm" if !inputs.is_empty() => {
                        let eps = match op.attributes.get("eps") {
                            Some(Attribute::Float(f)) => *f as f32,
                            _ => 1e-5,
                        };
                        let (norm_val, c_cnt) = insert_rmsnorm_core_before(
                            module, block_id, op_id, inputs[0], eps, &out_ty,
                        );
                        created += c_cnt;
                        if let Some(&w_val) = inputs.get(1) {
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "multiply");
                            op_mut.operands = SmallVec::from_slice(&[norm_val, w_val]);
                            op_mut.attributes.clear();
                            modified += 1;
                        } else {
                            let one_t =
                                insert_splat_tensor_before(module, block_id, op_id, &out_ty, 1.0);
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "multiply");
                            op_mut.operands = SmallVec::from_slice(&[norm_val, one_t]);
                            op_mut.attributes.clear();
                            created += 1;
                            modified += 1;
                        }
                    }
                    "castforge.fused_residual_rmsnorm" if inputs.len() >= 2 => {
                        let eps = match op.attributes.get("eps") {
                            Some(Attribute::Float(f)) => *f as f32,
                            _ => 1e-5,
                        };
                        let h_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "add"),
                            &[inputs[0], inputs[1]],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let (norm_val, c_cnt) = insert_rmsnorm_core_before(
                            module, block_id, op_id, h_val, eps, &out_ty,
                        );
                        created += 1 + c_cnt;
                        let w_val = inputs.get(2).copied().unwrap_or_else(|| {
                            insert_splat_tensor_before(module, block_id, op_id, &out_ty, 1.0)
                        });
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[norm_val, w_val]);
                        op_mut.attributes.clear();
                        modified += 1;
                    }
                    "castforge.fused_layernorm" if !inputs.is_empty() => {
                        let eps = match op.attributes.get("eps") {
                            Some(Attribute::Float(f)) => *f as f32,
                            _ => 1e-5,
                        };
                        let (norm_val, c_cnt) = insert_layernorm_core_before(
                            module, block_id, op_id, inputs[0], eps, &out_ty,
                        );
                        created += c_cnt;
                        let w_val = inputs.get(1).copied().unwrap_or_else(|| {
                            created += 1;
                            insert_splat_tensor_before(module, block_id, op_id, &out_ty, 1.0)
                        });
                        if let Some(&b_val) = inputs.get(2) {
                            let scaled = insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", "multiply"),
                                &[norm_val, w_val],
                                IndexMap::new(),
                                out_ty,
                            );
                            created += 1;
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "add");
                            op_mut.operands = SmallVec::from_slice(&[scaled, b_val]);
                            op_mut.attributes.clear();
                            modified += 1;
                        } else {
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "multiply");
                            op_mut.operands = SmallVec::from_slice(&[norm_val, w_val]);
                            op_mut.attributes.clear();
                            modified += 1;
                        }
                    }
                    "castforge.fused_gated_mlp" if inputs.len() == 2 => {
                        let (x, w_gu) = (inputs[0], inputs[1]);
                        let is_gelu = op
                            .attributes
                            .get("activation")
                            .and_then(|a| a.as_str())
                            .is_some_and(|s| {
                                s.eq_ignore_ascii_case("gelu") || s.eq_ignore_ascii_case("geglu")
                            });
                        let packed = op
                            .attributes
                            .get("packed_rhs")
                            .is_some_and(|a| matches!(a, Attribute::Bool(true)));
                        let (rows, cols) = if let Some(ot) = out_ty.as_tensor()
                            && ot.shape.len() == 2
                        {
                            let r = match ot.shape[0] {
                                Dim::Static(v) => v,
                                _ => 1,
                            };
                            let c = match ot.shape[1] {
                                Dim::Static(v) => v,
                                _ => 1,
                            };
                            (r, c)
                        } else {
                            (1, 1)
                        };
                        let gu_ty = Type::Tensor(TensorType {
                            shape: vec![Dim::Static(rows), Dim::Static(cols * 2)],
                            dtype: DType::F32,
                            encoding: None,
                        });
                        let mut dot_attrs = IndexMap::new();
                        dot_attrs.insert(
                            "lhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![1]),
                        );
                        dot_attrs.insert(
                            "rhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![if packed { 1 } else { 0 }]),
                        );
                        let gu_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w_gu],
                            dot_attrs,
                            gu_ty,
                        );
                        let gate = insert_slice_2d_before(
                            module,
                            block_id,
                            op_id,
                            gu_val,
                            [0, 0],
                            [rows, cols],
                            out_ty.clone(),
                        );
                        let up = insert_slice_2d_before(
                            module,
                            block_id,
                            op_id,
                            gu_val,
                            [0, cols],
                            [rows, cols * 2],
                            out_ty.clone(),
                        );
                        created += 3;
                        let act_val = if is_gelu {
                            created += 8;
                            insert_gelu_before(module, block_id, op_id, gate, &out_ty)
                        } else {
                            let denom =
                                insert_silu_denom_before(module, block_id, op_id, gate, &out_ty);
                            created += 6;
                            insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", "divide"),
                                &[gate, denom],
                                IndexMap::new(),
                                out_ty.clone(),
                            )
                        };
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[act_val, up]);
                        op_mut.attributes.clear();
                        modified += 1;
                    }
                    "castforge.fused_gated_mlp" if inputs.len() >= 4 => {
                        let (x, w_gate, w_up, w_down) =
                            (inputs[0], inputs[1], inputs[2], inputs[3]);
                        let is_gelu = op
                            .attributes
                            .get("activation")
                            .and_then(|a| a.as_str())
                            .is_some_and(|s| s.eq_ignore_ascii_case("gelu"));
                        let hid_ty = if let (Some(xt), Some(wt)) = (
                            module.values[x].ty.as_tensor(),
                            module.values[w_gate].ty.as_tensor(),
                        ) && xt.shape.len() == 2
                            && wt.shape.len() == 2
                        {
                            Type::Tensor(TensorType {
                                shape: vec![xt.shape[0].clone(), wt.shape[1].clone()],
                                dtype: xt.dtype.clone(),
                                encoding: None,
                            })
                        } else {
                            out_ty.clone()
                        };
                        let mut dot_attrs = IndexMap::new();
                        dot_attrs.insert(
                            "lhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![1]),
                        );
                        dot_attrs.insert(
                            "rhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![0]),
                        );
                        let gate = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w_gate],
                            dot_attrs.clone(),
                            hid_ty.clone(),
                        );
                        let up = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w_up],
                            dot_attrs.clone(),
                            hid_ty.clone(),
                        );
                        created += 2;
                        let act_val = if is_gelu {
                            created += 8;
                            insert_gelu_before(module, block_id, op_id, gate, &hid_ty)
                        } else {
                            let denom =
                                insert_silu_denom_before(module, block_id, op_id, gate, &hid_ty);
                            created += 6;
                            insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", "divide"),
                                &[gate, denom],
                                IndexMap::new(),
                                hid_ty.clone(),
                            )
                        };
                        let hidden = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[act_val, up],
                            IndexMap::new(),
                            hid_ty,
                        );
                        created += 1;
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "dot_general");
                        op_mut.operands = SmallVec::from_slice(&[hidden, w_down]);
                        op_mut.attributes = dot_attrs;
                        modified += 1;
                    }
                    "castforge.fused_gelu" if !inputs.is_empty() => {
                        let gelu_val =
                            insert_gelu_before(module, block_id, op_id, inputs[0], &out_ty);
                        let zero_t = insert_zero_tensor_before(module, block_id, op_id, &out_ty);
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[gelu_val, zero_t]);
                        op_mut.attributes.clear();
                        created += 9;
                        modified += 1;
                    }
                    "castforge.fused_geglu" if inputs.len() >= 2 => {
                        let gelu_val =
                            insert_gelu_before(module, block_id, op_id, inputs[0], &out_ty);
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[gelu_val, inputs[1]]);
                        op_mut.attributes.clear();
                        created += 8;
                        modified += 1;
                    }
                    "castforge.fused_qmatmul" if inputs.len() >= 2 => {
                        let w_q = inputs[1];
                        let w_f32_ty = if let Some(t) = module.values[w_q].ty.as_tensor() {
                            Type::Tensor(TensorType {
                                shape: t.shape.clone(),
                                dtype: DType::F32,
                                encoding: None,
                            })
                        } else {
                            out_ty.clone()
                        };
                        let deq_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "uniform_dequantize"),
                            &[w_q],
                            IndexMap::new(),
                            w_f32_ty,
                        );
                        let packed = op
                            .attributes
                            .get("packed_rhs")
                            .is_some_and(|a| matches!(a, Attribute::Bool(true)));
                        let mut dot_attrs = IndexMap::new();
                        dot_attrs.insert(
                            "lhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![1]),
                        );
                        dot_attrs.insert(
                            "rhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![if packed { 1 } else { 0 }]),
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "dot_general");
                        op_mut.operands = SmallVec::from_slice(&[inputs[0], deq_val]);
                        op_mut.attributes = dot_attrs;
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_map_reduce" if inputs.len() >= 2 => {
                        let map_op = op
                            .attributes
                            .get("map_op")
                            .and_then(|a| a.as_str())
                            .unwrap_or("exponential")
                            .to_string();
                        let x_ty = module.values[inputs[0]].ty.clone();
                        let mapped_val = if map_op == "square" {
                            insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", "multiply"),
                                &[inputs[0], inputs[0]],
                                IndexMap::new(),
                                x_ty,
                            )
                        } else {
                            insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", &map_op),
                                &[inputs[0]],
                                IndexMap::new(),
                                x_ty,
                            )
                        };
                        let mut red_attrs = IndexMap::new();
                        if let Some(d) = op.attributes.get("dimensions") {
                            red_attrs.insert("dimensions".to_string(), d.clone());
                        }
                        if let Some(r) = op.attributes.get("reducer") {
                            red_attrs.insert("reducer".to_string(), r.clone());
                        }
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "reduce");
                        op_mut.operands = SmallVec::from_slice(&[mapped_val, inputs[1]]);
                        op_mut.attributes = red_attrs;
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_log_softmax" if !inputs.is_empty() => {
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
                            red_attrs.clone(),
                            red_ty.clone(),
                        );
                        let log_sum = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "log"),
                            &[sum_val],
                            IndexMap::new(),
                            red_ty,
                        );
                        let mut bcast_attrs = IndexMap::new();
                        bcast_attrs.insert(
                            "broadcast_dimensions".to_string(),
                            Attribute::DenseIntArray(bcast_dims),
                        );
                        let lse_bcast = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "broadcast_in_dim"),
                            &[log_sum],
                            bcast_attrs,
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "subtract");
                        op_mut.operands = SmallVec::from_slice(&[x, lse_bcast]);
                        op_mut.attributes.clear();
                        created += 5;
                        modified += 1;
                    }
                    "castforge.fma" if inputs.len() >= 3 => {
                        let (a, b, c) = (inputs[0], inputs[1], inputs[2]);
                        let prod = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[a, b],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[prod, c]);
                        op_mut.attributes.clear();
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_moe" if inputs.len() >= 5 => {
                        let (x, w0, w1, g0, g1) =
                            (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
                        let mut dot_attrs = IndexMap::new();
                        dot_attrs.insert(
                            "lhs_contracting_dimensions".to_string(),
                            Attribute::Array(vec![Attribute::Int(1)]),
                        );
                        dot_attrs.insert(
                            "rhs_contracting_dimensions".to_string(),
                            Attribute::Array(vec![Attribute::Int(0)]),
                        );
                        let e0 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w0],
                            dot_attrs.clone(),
                            out_ty.clone(),
                        );
                        let e1 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w1],
                            dot_attrs,
                            out_ty.clone(),
                        );
                        let m0 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[e0, g0],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let m1 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[e1, g1],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[m0, m1]);
                        op_mut.attributes.clear();
                        created += 4;
                        modified += 1;
                    }
                    "castforge.stream_wait"
                    | "castforge.static_quantize"
                    | "castforge.nan_guard"
                    | "castforge.host_offload"
                    | "castforge.profile_region"
                        if !inputs.is_empty() =>
                    {
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "convert");
                        op_mut.operands = SmallVec::from_slice(&[inputs[0]]);
                        op_mut.attributes.clear();
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
