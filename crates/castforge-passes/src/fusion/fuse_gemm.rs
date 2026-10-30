//! Unified GEMM Fusion (`fuse-gemm`): Horizontal Dot, Quantized Dequant-MatMul, and MatMul
//! Epilogue Fusion.

use crate::fusion::fuse_activations::{match_gelu_op, match_silu_val};
use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, TensorType, Type, ValueDef,
    ValueId, format_dense_f32_literal,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified GEMM Fusion Pass (`fuse-gemm`).
pub struct FuseGemmPass {
    pub kind: String,
}

impl Default for FuseGemmPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
        }
    }
}

impl Pass for FuseGemmPass {
    fn name(&self) -> &'static str {
        "fuse-gemm"
    }

    fn description(&self) -> &'static str {
        "Unified GEMM fusion (horizontal QKV/MLP dot, dequant+GEMM, and GEMM+bias/activation)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        if k == "all" || k == "horizontal" {
            steps.push(FuseHorizontalDotPass.run(module)?);
        }
        if k == "all" || k == "dequant" {
            steps.push(FuseDequantMatmulPass.run(module)?);
        }
        if k == "all" || k == "epilogue" {
            steps.push(FuseMatmulEpiloguePass.run(module)?);
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Horizontal GEMM / QKV Projection Fusion Pass (`fuse-horizontal-dot`).
pub struct FuseHorizontalDotPass;

struct DotCandidate {
    dot_op_id: OpId,
    result_val: ValueId,
    n_cols: usize,
    w_vals: Vec<f32>,
}

impl Pass for FuseHorizontalDotPass {
    fn name(&self) -> &'static str {
        "fuse-horizontal-dot"
    }

    fn description(&self) -> &'static str {
        "Horizontally fuses parallel dot_general ops sharing an LHS input (e.g. Q/K/V projections)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut groups: IndexMap<(ValueId, usize, usize), Vec<DotCandidate>> = IndexMap::new();

            for &op_id in &ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    lhs,
                    rhs,
                    dimension_numbers: dims,
                    result,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                if !dims.lhs_batching_dimensions.is_empty()
                    || !dims.rhs_batching_dimensions.is_empty()
                    || dims.lhs_contracting_dimensions != [1]
                    || dims.rhs_contracting_dimensions != [0]
                {
                    continue;
                }

                let (Some(lhs_ty), Some(rhs_ty)) = (
                    module.values[lhs].ty.as_tensor(),
                    module.values[rhs].ty.as_tensor(),
                ) else {
                    continue;
                };
                if lhs_ty.dtype != DType::F32
                    || rhs_ty.dtype != DType::F32
                    || lhs_ty.shape.len() != 2
                    || rhs_ty.shape.len() != 2
                {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k_lhs), Dim::Static(k_rhs), Dim::Static(n)) = (
                    &lhs_ty.shape[0],
                    &lhs_ty.shape[1],
                    &rhs_ty.shape[0],
                    &rhs_ty.shape[1],
                ) else {
                    continue;
                };
                if k_lhs != k_rhs {
                    continue;
                }
                let (m_usize, k_usize, n_usize) = (*m as usize, *k_lhs as usize, *n as usize);

                let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs].def else {
                    continue;
                };
                let Some(w_op) = module.ops.get(w_op_id) else {
                    continue;
                };
                if !w_op.is_op("stablehlo", "constant") {
                    continue;
                }
                let Some(w_attr) = w_op.attributes.get("value") else {
                    continue;
                };
                let Some(w_vals) = module
                    .tensor_store
                    .resolve_f32_attr(w_attr, k_usize * n_usize)
                else {
                    continue;
                };

                groups
                    .entry((lhs, m_usize, k_usize))
                    .or_default()
                    .push(DotCandidate {
                        dot_op_id: op_id,
                        result_val: result,
                        n_cols: n_usize,
                        w_vals,
                    });
            }

            for ((lhs_val, m, k), cands) in groups {
                if cands.len() < 2 {
                    continue;
                }
                let total_n: usize = cands.iter().map(|c| c.n_cols).sum();
                let mut fused_w = vec![0.0_f32; k * total_n];

                for row in 0..k {
                    let mut col_off = 0usize;
                    for cand in &cands {
                        let src_start = row * cand.n_cols;
                        let dst_start = row * total_n + col_off;
                        fused_w[dst_start..dst_start + cand.n_cols]
                            .copy_from_slice(&cand.w_vals[src_start..src_start + cand.n_cols]);
                        col_off += cand.n_cols;
                    }
                }

                let w_fused_ty = TensorType {
                    shape: vec![Dim::Static(k as i64), Dim::Static(total_n as i64)],
                    dtype: DType::F32,
                    encoding: None,
                };
                let dot_fused_ty = TensorType {
                    shape: vec![Dim::Static(m as i64), Dim::Static(total_n as i64)],
                    dtype: DType::F32,
                    encoding: None,
                };

                let first_dot_id = cands[0].dot_op_id;
                let dot_attrs = module.ops[first_dot_id].attributes.clone();

                let mut w_attrs = IndexMap::new();
                w_attrs.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: w_fused_ty.clone(),
                        raw_literal: format_dense_f32_literal(&fused_w, &w_fused_ty.shape),
                    },
                );
                let w_fused_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "constant"),
                    SmallVec::new(),
                    vec![(None, Type::Tensor(w_fused_ty))],
                    w_attrs,
                    Vec::new(),
                    false,
                    Location::Unknown,
                );
                let w_fused_val = module.ops[w_fused_op].results[0];

                let dot_fused_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    SmallVec::from_slice(&[lhs_val, w_fused_val]),
                    vec![(None, Type::Tensor(dot_fused_ty))],
                    dot_attrs,
                    Vec::new(),
                    false,
                    Location::Unknown,
                );
                let dot_fused_val = module.ops[dot_fused_op].results[0];

                let block_ops = &mut module.blocks[block_id].ops;
                block_ops.retain(|&id| id != w_fused_op && id != dot_fused_op);
                let insert_pos = block_ops
                    .iter()
                    .position(|&id| id == first_dot_id)
                    .unwrap_or(0);
                block_ops.insert(insert_pos, w_fused_op);
                block_ops.insert(insert_pos + 1, dot_fused_op);
                created += 2;

                let mut col_off = 0i64;
                for cand in cands {
                    let n_i64 = cand.n_cols as i64;
                    let mut slice_attrs = IndexMap::new();
                    slice_attrs.insert(
                        "start_indices".to_string(),
                        Attribute::DenseIntArray(vec![0, col_off]),
                    );
                    slice_attrs.insert(
                        "limit_indices".to_string(),
                        Attribute::DenseIntArray(vec![m as i64, col_off + n_i64]),
                    );
                    slice_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));

                    let op_mut = &mut module.ops[cand.dot_op_id];
                    op_mut.name = OpName::new("stablehlo", "slice");
                    op_mut.operands = SmallVec::from_slice(&[dot_fused_val]);
                    op_mut.attributes = slice_attrs;
                    let _ = cand.result_val;
                    col_off += n_i64;
                    modified += 1;
                }
            }
        }

        if created > 0 || modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
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

/// Quantized MatMul Fusion Pass (`fuse-dequant-matmul`).
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

/// Generalized MatMul Epilogue Fusion (`fuse-matmul-epilogue`).
pub struct FuseMatmulEpiloguePass;

impl Pass for FuseMatmulEpiloguePass {
    fn name(&self) -> &'static str {
        "fuse-matmul-epilogue"
    }

    fn description(&self) -> &'static str {
        "Fuses dot_general + bias add (+ relu/tanh/silu/scale epilogue) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

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
                    created += 1;
                    continue;
                }

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
                    created += 1;
                    continue;
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
                            continue;
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
                        created += 1;

                        module.remove_op(block_id, add_op_id);
                        module.remove_op(block_id, dot_op_id);
                        removed += 2;
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
                            continue;
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
                        created += 1;

                        module.remove_op(block_id, add_op_id);
                        module.remove_op(block_id, dot_op_id);
                        removed += 2;
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
                            continue;
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
                        created += 1;

                        module.remove_op(block_id, add_op_id);
                        module.remove_op(block_id, dot_op_id);
                        removed += 2;
                    }
                    _ => {}
                }
            }

            if created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
            }

            let ops2 = module.block_ops(block_id);
            for op_id in ops2 {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add { result, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };
                let Some(MatmulAddMatch {
                    dot_op_id,
                    lhs,
                    rhs,
                    bias,
                    dot_attrs,
                    ..
                }) = match_add_of_dot_ignoring_uses(module, result)
                else {
                    continue;
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
                op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
                op_mut.attributes = attrs;
                created += 1;

                module.remove_op(block_id, dot_op_id);
                removed += 1;
            }

            // Chained Residual Epilogue Fusion: `add(fused_matmul_epilogue(A,B,bias,..), res)`
            let ops3 = module.block_ops(block_id);
            for op_id in ops3 {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op) else {
                    continue;
                };
                let mut matched_epi: Option<(OpId, ValueId)> = None;
                for (cand_epi, cand_res) in [(lhs, rhs), (rhs, lhs)] {
                    if module.use_count(cand_epi) != 1 {
                        continue;
                    }
                    if let ValueDef::OpResult { op: epi_id, .. } = module.values[cand_epi].def
                        && let Some(epi_op) = module.ops.get(epi_id)
                        && epi_op.is_op("stablehlo", "composite")
                        && epi_op.attributes.get("name").and_then(|a| a.as_str())
                            == Some("castforge.fused_matmul_epilogue")
                        && !epi_op.attributes.contains_key("castforge.has_residual")
                    {
                        let prev_ep = epi_op
                            .attributes
                            .get("epilogue")
                            .and_then(|a| a.as_str())
                            .unwrap_or("bias");
                        if prev_ep != "bias" {
                            matched_epi = Some((epi_id, cand_res));
                            break;
                        }
                    }
                }
                let Some((epi_id, residual_val)) = matched_epi else {
                    continue;
                };
                let epi_op = module.ops[epi_id].clone();
                let prev_ep = epi_op
                    .attributes
                    .get("epilogue")
                    .and_then(|a| a.as_str())
                    .unwrap_or("bias")
                    .to_string();
                let mut new_attrs = epi_op.attributes.clone();
                new_attrs.insert(
                    "epilogue".to_string(),
                    Attribute::String(format!("{prev_ep}_residual")),
                );
                new_attrs.insert("castforge.has_residual".to_string(), Attribute::Bool(true));
                let mut new_operands = epi_op.operands.clone();
                new_operands.push(residual_val);

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = new_operands;
                op_mut.attributes = new_attrs;
                module.remove_op(block_id, epi_id);
                removed += 1;
            }
        }

        if created > 0 || removed > 0 {
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

struct MatmulAddMatch {
    add_op_id: OpId,
    dot_op_id: OpId,
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
                dot_op_id,
                lhs,
                rhs,
                bias: bias_cand,
                dot_attrs: dot_op.attributes.clone(),
            });
        }
    }
    None
}
