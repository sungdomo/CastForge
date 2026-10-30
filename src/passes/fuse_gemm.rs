//! Unified Single-Traversal GEMM Fusion (`fuse-gemm`): Horizontal Dot, Quantized Dequant-MatMul,
//! and MatMul Epilogue Fusion.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, TensorType, Type, ValueDef,
    ValueId, format_dense_f32_literal,
};
use crate::passes::fuse_elementwise::{match_gelu_op, match_silu_val};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Single-Traversal GEMM Fusion Pass (`fuse-gemm`).
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
        "Unified GEMM fusion (horizontal QKV/MLP dot, dequant+GEMM, and GEMM+epilogue)"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "fuse-horizontal-dot",
            "fuse-dequant-matmul",
            "fuse-matmul-epilogue",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        match k.as_str() {
            "horizontal" => Ok(FuseHorizontalDotPass.run(module)?.with_name(self.name())),
            "dequant" => Ok(FuseDequantMatmulPass.run(module)?.with_name(self.name())),
            "epilogue" => Ok(FuseMatmulEpiloguePass.run(module)?.with_name(self.name())),
            _ => PassResult::run_sequence(
                self.name(),
                module,
                &mut [
                    &mut FuseHorizontalDotPass,
                    &mut FuseDequantMatmulPass,
                    &mut FuseMatmulEpiloguePass,
                ],
            ),
        }
    }
}

/// Horizontal GEMM / QKV Projection Fusion Pass (`fuse-horizontal-dot`).
pub struct FuseHorizontalDotPass;

pub(crate) struct DotCandidate {
    pub(crate) dot_op_id: OpId,
    pub(crate) result_val: ValueId,
    pub(crate) n_cols: usize,
    pub(crate) w_vals: Vec<f32>,
}

impl Pass for FuseHorizontalDotPass {
    fn name(&self) -> &'static str {
        "fuse-horizontal-dot"
    }

    fn description(&self) -> &'static str {
        "Horizontally fuses parallel dot_general ops sharing an LHS input (e.g. Q/K/V projections)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let (created, modified, mut removed) =
            fuse_gemm_single_traversal(module, true, false, false)?;
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
        let (created, _, mut removed) = fuse_gemm_single_traversal(module, false, true, false)?;
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
        let (created, _, mut removed) = fuse_gemm_single_traversal(module, false, false, true)?;
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

/// Executes horizontal dot collection, quantized dequant+GEMM fusion, and top-down GEMM epilogue
/// fusion (`bias`, `bias_relu`, `bias_tanh`, `bias_scale`, `bias_silu`, `bias_gelu`, and chained
/// `*_residual`) in a **single forward traversal** per block.
pub(crate) fn fuse_gemm_single_traversal(
    module: &mut Module,
    do_horiz: bool,
    do_deq: bool,
    do_epi: bool,
) -> Result<(usize, usize, usize)> {
    let mut created = 0usize;
    let mut modified = 0usize;
    let mut removed = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        let mut groups: IndexMap<(ValueId, usize, usize), Vec<DotCandidate>> = IndexMap::new();

        for &op_id in &ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };

            // 1. Collect horizontal constant-weight 2D dot_general candidates during the walk.
            if do_horiz && let Some(cand_entry) = try_collect_horizontal_candidate(module, &op) {
                let (key, cand) = cand_entry;
                groups.entry(key).or_default().push(cand);
            }

            // 2. Top-down GEMM epilogue fusion (`bias`, activations, and chained `*_residual`).
            if do_epi {
                let (epi_c, epi_r) = try_fuse_epilogue_at_op(module, block_id, op_id, &op);
                created += epi_c;
                removed += epi_r;
            }

            // 3. On-the-fly weight dequantization (`uniform_dequantize` -> `fused_qmatmul`).
            if do_deq && let Some(cur_op) = module.ops.get(op_id).cloned() {
                created += try_fuse_dequant_at_op(module, op_id, &cur_op);
            }
        }

        // Emit horizontal concat for any `(lhs, m, k)` group with >= 2 live dot_general candidates.
        if do_horiz {
            for ((lhs_val, m, k), cands) in groups {
                let live_cands: Vec<DotCandidate> = cands
                    .into_iter()
                    .filter(|c| {
                        module
                            .ops
                            .get(c.dot_op_id)
                            .is_some_and(|op| op.is_op("stablehlo", "dot_general"))
                    })
                    .collect();
                if live_cands.len() < 2 {
                    continue;
                }
                let (hc, hm) =
                    emit_horizontal_dot_group(module, block_id, lhs_val, m, k, &live_cands);
                created += hc;
                modified += hm;
            }
        }
    }

    Ok((created, modified, removed))
}

pub(crate) fn try_collect_horizontal_candidate(
    module: &Module,
    op: &crate::ir::Operation,
) -> Option<((ValueId, usize, usize), DotCandidate)> {
    let StableHloOp::DotGeneral {
        lhs,
        rhs,
        dimension_numbers: dims,
        result,
    } = StableHloOp::from_op(op)?
    else {
        return None;
    };
    if !dims.lhs_batching_dimensions.is_empty()
        || !dims.rhs_batching_dimensions.is_empty()
        || dims.lhs_contracting_dimensions != [1]
        || dims.rhs_contracting_dimensions != [0]
    {
        return None;
    }
    let (lhs_ty, rhs_ty) = (
        module.values[lhs].ty.as_tensor()?,
        module.values[rhs].ty.as_tensor()?,
    );
    if lhs_ty.dtype != DType::F32
        || rhs_ty.dtype != DType::F32
        || lhs_ty.shape.len() != 2
        || rhs_ty.shape.len() != 2
    {
        return None;
    }
    let (Dim::Static(m), Dim::Static(k_lhs), Dim::Static(k_rhs), Dim::Static(n)) = (
        &lhs_ty.shape[0],
        &lhs_ty.shape[1],
        &rhs_ty.shape[0],
        &rhs_ty.shape[1],
    ) else {
        return None;
    };
    if k_lhs != k_rhs || *m <= 0 || *k_lhs <= 0 || *n <= 0 {
        return None;
    }
    let (m_usize, k_usize, n_usize) = (*m as usize, *k_lhs as usize, *n as usize);
    let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs].def else {
        return None;
    };
    let w_op = module.ops.get(w_op_id)?;
    let StableHloOp::Constant { value, .. } = StableHloOp::from_op(w_op)? else {
        return None;
    };
    let w_vals = module
        .tensor_store
        .resolve_f32_attr(&value, k_usize * n_usize)?;
    if w_vals.len() != k_usize * n_usize {
        return None;
    }
    Some((
        (lhs, m_usize, k_usize),
        DotCandidate {
            dot_op_id: op.id,
            result_val: result,
            n_cols: n_usize,
            w_vals,
        },
    ))
}

pub(crate) fn emit_horizontal_dot_group(
    module: &mut Module,
    block_id: BlockId,
    lhs_val: ValueId,
    m: usize,
    k: usize,
    cands: &[DotCandidate],
) -> (usize, usize) {
    let total_n: usize = cands.iter().map(|c| c.n_cols).sum();
    let mut fused_w = vec![0.0f32; k * total_n];
    for row in 0..k {
        let mut col_off = 0usize;
        for cand in cands {
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

    let mut w_attrs = IndexMap::new();
    let raw_literal = format_dense_f32_literal(&fused_w, &w_fused_ty.shape);
    w_attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: w_fused_ty.clone(),
            raw_literal,
        },
    );

    let first_dot_id = cands[0].dot_op_id;
    let mut dot_attrs = module.ops[first_dot_id].attributes.clone();
    dot_attrs.insert(
        "castforge.horizontal_weights_folded".to_string(),
        Attribute::Bool(true),
    );
    dot_attrs.insert(
        "castforge.horizontal_fused_count".to_string(),
        Attribute::Int(cands.len() as i64),
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

    let mut col_off = 0i64;
    let mut modified = 0usize;
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

    (2, modified)
}

fn try_fuse_dequant_at_op(module: &mut Module, op_id: OpId, op: &crate::ir::Operation) -> usize {
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

fn try_fuse_epilogue_at_op(
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

            // Otherwise check `add(dot_general, bias)`:
            // Only fuse immediately into `epilogue="bias"` if `result` is not used multiple times
            // by an un-fused multi-use activation (`silu`/`gelu`), or let
            // `match_add_of_dot_ignoring_uses`
            // handle both `add` and `fused_matmul_epilogue(bias)`.
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

    // Also match if `add_op` was already fused into
    // `castforge.fused_matmul_epilogue(epilogue="bias")`
    // earlier in the same top-down traversal!
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
