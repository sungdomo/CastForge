//! Quantized Dequant-MatMul and Generalized MatMul Epilogue Fusion patterns.

use crate::ir::{Attribute, BlockId, DType, Module, OpId, OpName, ValueDef, ValueId};
use crate::passes::fuse_elementwise::{match_gelu_op, match_silu_val};
use crate::stablehlo::StableHloOp;
use indexmap::IndexMap;
use smallvec::SmallVec;

pub(super) fn try_fuse_dequant_at_op(
    module: &mut Module,
    op_id: OpId,
    op: &crate::ir::Operation,
) -> usize {
    match StableHloOp::from_op(op) {
        Some(StableHloOp::DotGeneral {
            lhs,
            rhs,
            dimension_numbers: dims,
            ..
        }) => {
            let Some((w_q, scale, zp)) = match_uniform_dequantize(module, rhs) else {
                return 0;
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
            1
        }
        Some(StableHloOp::Composite { name, inputs, .. })
            if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 =>
        {
            let Some((w_q, scale, zp)) = match_uniform_dequantize(module, inputs[1]) else {
                return 0;
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
            1
        }
        _ => 0,
    }
}

fn match_uniform_dequantize(module: &Module, val_id: ValueId) -> Option<(ValueId, f64, i64)> {
    let ValueDef::OpResult { op: deq_op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let deq_op = module.ops.get(deq_op_id)?;
    if !deq_op.is_op("stablehlo", "uniform_dequantize") || deq_op.operands.len() != 1 {
        return None;
    };
    let w_q = deq_op.operands[0];
    let (scale, zp) = match module.values[w_q].ty.as_tensor().map(|t| &t.dtype) {
        Some(DType::QuantUniform {
            scale, zero_point, ..
        }) => (scale.parse::<f64>().unwrap_or(1.0), *zero_point),
        _ => (1.0, 0),
    };
    Some((w_q, scale, zp))
}

pub(super) fn try_fuse_epilogue_at_op(
    module: &mut Module,
    block_id: BlockId,
    op_id: OpId,
    op: &crate::ir::Operation,
) -> (usize, usize) {
    let Some(hlo) = StableHloOp::from_op(op) else {
        return (0, 0);
    };

    // 1. SiLU epilogue (`divide(add_val, 1 + exp(-add_val))`)
    if let StableHloOp::Divide { result, .. } = hlo
        && let Some(add_val) = match_silu_val(module, result)
        && let Some(MatmulAddMatch {
            lhs,
            rhs,
            bias,
            dot_attrs,
            ..
        }) = match_add_of_dot_ignoring_uses(module, add_val)
    {
        let mut attrs = dot_attrs;
        attrs.insert(
            "name".to_string(),
            Attribute::String("castforge.fused_matmul_epilogue".to_string()),
        );
        attrs.insert(
            "decomposition".to_string(),
            Attribute::SymbolRef("fused_matmul_bias_silu_impl".to_string()),
        );
        attrs.insert(
            "epilogue".to_string(),
            Attribute::String("bias_silu".to_string()),
        );

        let op_mut = &mut module.ops[op_id];
        op_mut.name = OpName::new("stablehlo", "composite");
        op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
        op_mut.attributes = attrs;
        return (1, 0);
    }

    // 2. GELU epilogue (`0.5 * add_val * (1 + tanh(...))`)
    if let Some(add_val) = match_gelu_op(module, op)
        && let Some(MatmulAddMatch {
            lhs,
            rhs,
            bias,
            dot_attrs,
            ..
        }) = match_add_of_dot_ignoring_uses(module, add_val)
    {
        let mut attrs = dot_attrs;
        attrs.insert(
            "name".to_string(),
            Attribute::String("castforge.fused_matmul_epilogue".to_string()),
        );
        attrs.insert(
            "decomposition".to_string(),
            Attribute::SymbolRef("fused_matmul_bias_gelu_impl".to_string()),
        );
        attrs.insert(
            "epilogue".to_string(),
            Attribute::String("bias_gelu".to_string()),
        );

        let op_mut = &mut module.ops[op_id];
        op_mut.name = OpName::new("stablehlo", "composite");
        op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
        op_mut.attributes = attrs;
        return (1, 0);
    }

    match hlo {
        StableHloOp::Maximum {
            lhs: add_val,
            rhs: zero_val,
            ..
        } => {
            let Some(MatmulAddMatch {
                add_op_id,
                dot_op_id,
                lhs,
                rhs,
                bias,
                dot_attrs,
            }) = match_add_of_dot(module, add_val)
            else {
                return (0, 0);
            };
            let mut attrs = dot_attrs;
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_matmul_epilogue".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_matmul_bias_relu_impl".to_string()),
            );
            attrs.insert(
                "epilogue".to_string(),
                Attribute::String("bias_relu".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias, zero_val]);
            op_mut.attributes = attrs;
            module.remove_op(block_id, add_op_id);
            if let Some(d_id) = dot_op_id {
                module.remove_op(block_id, d_id);
                (0, 2)
            } else {
                (0, 1)
            }
        }
        StableHloOp::Tanh {
            operand: add_val, ..
        } => {
            let Some(MatmulAddMatch {
                add_op_id,
                dot_op_id,
                lhs,
                rhs,
                bias,
                dot_attrs,
            }) = match_add_of_dot(module, add_val)
            else {
                return (0, 0);
            };
            let mut attrs = dot_attrs;
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_matmul_epilogue".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_matmul_bias_tanh_impl".to_string()),
            );
            attrs.insert(
                "epilogue".to_string(),
                Attribute::String("bias_tanh".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
            op_mut.attributes = attrs;
            module.remove_op(block_id, add_op_id);
            if let Some(d_id) = dot_op_id {
                module.remove_op(block_id, d_id);
                (0, 2)
            } else {
                (0, 1)
            }
        }
        StableHloOp::Multiply {
            lhs: m_lhs,
            rhs: m_rhs,
            ..
        } => {
            let matched = match_add_of_dot(module, m_lhs)
                .map(|m| (m, m_rhs))
                .or_else(|| match_add_of_dot(module, m_rhs).map(|m| (m, m_lhs)));
            let Some((
                MatmulAddMatch {
                    add_op_id,
                    dot_op_id,
                    lhs,
                    rhs,
                    bias,
                    dot_attrs,
                },
                scale_val,
            )) = matched
            else {
                return (0, 0);
            };
            let mut attrs = dot_attrs;
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_matmul_epilogue".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_matmul_bias_scale_impl".to_string()),
            );
            attrs.insert(
                "epilogue".to_string(),
                Attribute::String("bias_scale".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias, scale_val]);
            op_mut.attributes = attrs;
            module.remove_op(block_id, add_op_id);
            if let Some(d_id) = dot_op_id {
                module.remove_op(block_id, d_id);
                (0, 2)
            } else {
                (0, 1)
            }
        }
        StableHloOp::Add { lhs, rhs, result } => {
            // Check chained residual epilogue first: `add(fused_matmul_epilogue(ep != bias), res)`
            for (cand_epi, cand_res) in [(lhs, rhs), (rhs, lhs)] {
                if module.use_count(cand_epi) != 1 {
                    continue;
                }
                if let ValueDef::OpResult { op: epi_id, .. } = module.values[cand_epi].def
                    && let Some(epi_op) = module.ops.get(epi_id).cloned()
                    && epi_op.is_op("stablehlo", "composite")
                    && epi_op.attributes.get("name").and_then(|a| a.as_str())
                        == Some("castforge.fused_matmul_epilogue")
                    && !epi_op.attributes.contains_key("castforge.has_residual")
                {
                    let prev_ep = epi_op
                        .attributes
                        .get("epilogue")
                        .and_then(|a| a.as_str())
                        .unwrap_or("bias")
                        .to_string();
                    if prev_ep != "bias" {
                        let mut new_attrs = epi_op.attributes.clone();
                        new_attrs.insert(
                            "epilogue".to_string(),
                            Attribute::String(format!("{prev_ep}_residual")),
                        );
                        new_attrs
                            .insert("castforge.has_residual".to_string(), Attribute::Bool(true));
                        let mut new_operands = epi_op.operands.clone();
                        new_operands.push(cand_res);

                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "composite");
                        op_mut.operands = new_operands;
                        op_mut.attributes = new_attrs;
                        module.remove_op(block_id, epi_id);
                        return (0, 1);
                    }
                }
            }

            let Some(MatmulAddMatch {
                dot_op_id: Some(dot_op_id),
                lhs: dot_lhs,
                rhs: dot_rhs,
                bias,
                dot_attrs,
                ..
            }) = match_add_of_dot_ignoring_uses(module, result)
            else {
                return (0, 0);
            };

            let mut attrs = dot_attrs;
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_matmul_epilogue".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_matmul_bias_impl".to_string()),
            );
            attrs.insert(
                "epilogue".to_string(),
                Attribute::String("bias".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[dot_lhs, dot_rhs, bias]);
            op_mut.attributes = attrs;
            module.remove_op(block_id, dot_op_id);
            (1, 1)
        }
        _ => (0, 0),
    }
}

struct MatmulAddMatch {
    add_op_id: OpId,
    dot_op_id: Option<OpId>,
    lhs: ValueId,
    rhs: ValueId,
    bias: ValueId,
    dot_attrs: IndexMap<String, Attribute>,
}

fn match_add_of_dot(module: &Module, add_val: ValueId) -> Option<MatmulAddMatch> {
    if module.use_count(add_val) != 1 {
        return None;
    }
    match_add_of_dot_ignoring_uses(module, add_val)
}

fn match_add_of_dot_ignoring_uses(module: &Module, add_val: ValueId) -> Option<MatmulAddMatch> {
    let ValueDef::OpResult { op: add_op_id, .. } = module.values[add_val].def else {
        return None;
    };
    let add_op = module.ops.get(add_op_id)?;

    if add_op.is_op("stablehlo", "composite")
        && add_op.attributes.get("name").and_then(|a| a.as_str())
            == Some("castforge.fused_matmul_epilogue")
        && add_op.attributes.get("epilogue").and_then(|a| a.as_str()) == Some("bias")
        && add_op.operands.len() == 3
    {
        let mut dot_attrs = IndexMap::new();
        for k in [
            "lhs_batching_dimensions",
            "rhs_batching_dimensions",
            "lhs_contracting_dimensions",
            "rhs_contracting_dimensions",
        ] {
            if let Some(v) = add_op.attributes.get(k) {
                dot_attrs.insert(k.to_string(), v.clone());
            }
        }
        return Some(MatmulAddMatch {
            add_op_id,
            dot_op_id: None,
            lhs: add_op.operands[0],
            rhs: add_op.operands[1],
            bias: add_op.operands[2],
            dot_attrs,
        });
    }

    let StableHloOp::Add { lhs: a, rhs: b, .. } = StableHloOp::from_op(add_op)? else {
        return None;
    };

    for (dot_cand, bias_cand) in [(a, b), (b, a)] {
        if module.use_count(dot_cand) != 1 {
            continue;
        }
        let ValueDef::OpResult { op: dot_op_id, .. } = module.values[dot_cand].def else {
            continue;
        };
        let Some(dot_op) = module.ops.get(dot_op_id) else {
            continue;
        };
        if let Some(StableHloOp::DotGeneral { lhs, rhs, .. }) = StableHloOp::from_op(dot_op) {
            return Some(MatmulAddMatch {
                add_op_id,
                dot_op_id: Some(dot_op_id),
                lhs,
                rhs,
                bias: bias_cand,
                dot_attrs: dot_op.attributes.clone(),
            });
        }
    }
    None
}
