use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, TensorType, Type, ValueDef,
    ValueId, format_dense_f32_literal,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Horizontal GEMM / QKV Projection Fusion Pass (`fuse-horizontal-dot`).
///
/// Finds multiple `stablehlo.dot_general` operations in the same block that share the same
/// 2D LHS activation tensor `x` (`tensor<M x K x f32>`) and multiply it by constant 2D RHS
/// weight matrices `W_0, W_1, ...` (`tensor<K x N_i x f32>`).
/// Concatenates `[W_0, W_1, ...]` horizontally along columns at compile time into a single
/// combined weight matrix `W_fused` (`tensor<K x sum(N_i) x f32>`), executes a single fused
/// `stablehlo.dot_general(x, W_fused)`, and extracts each output via `stablehlo.slice`.
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

                // RHS must be a constant weight tensor
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

                // Create `w_fused` constant and `dot_fused` op before the first candidate dot
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

                // Move `w_fused_op` and `dot_fused_op` right before `first_dot_id` in the block
                let block_ops = &mut module.blocks[block_id].ops;
                block_ops.retain(|&id| id != w_fused_op && id != dot_fused_op);
                let insert_pos = block_ops
                    .iter()
                    .position(|&id| id == first_dot_id)
                    .unwrap_or(0);
                block_ops.insert(insert_pos, w_fused_op);
                block_ops.insert(insert_pos + 1, dot_fused_op);
                created += 2;

                // Rewrite each candidate `dot_general` in-place into a `stablehlo.slice`
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
