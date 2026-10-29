//! Tile-Alignment Padding (`pad-for-tiling`) and Large GEMM Splitter (`split-large-gemm`)
//! Passes (`Ir -> Ir`).
//!
//! - [`PadForTilingPass`] (`pad-for-tiling`) rounds 2D GEMM dimensions (`M, K, N`) up to the
//!   nearest multiple of `align` (default `8`) and stamps `padded_m`, `padded_k`, `padded_n`, and
//!   `pad_align` attributes onto `stablehlo.dot_general` ops (plus optional `castforge.pad_to_tile`
//!   composite markers when `wrap_inputs=true`).
//! - [`SplitLargeGemmPass`] (`split-large-gemm`) decomposes `stablehlo.dot_general` operations
//!   whose output column dimension `N > max_n` into two column-sliced sub-GEMMs (`stablehlo.slice`
//!   + `stablehlo.dot_general` + `stablehlo.concatenate`), preserving exact numerical equivalence.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Dim, Module, OpName, TensorType, Type};
use indexmap::IndexMap;
use smallvec::smallvec;

/// Static Tile-Alignment Padding Pass (`pad-for-tiling`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct PadForTilingPass {
    /// Tile alignment multiple (default `8`).
    pub align: i64,
}

impl Default for PadForTilingPass {
    fn default() -> Self {
        Self { align: 8 }
    }
}

impl Pass for PadForTilingPass {
    fn name(&self) -> &'static str {
        "pad-for-tiling"
    }

    fn description(&self) -> &'static str {
        "Computes tile-aligned (M, K, N) multiples for GEMM ops to eliminate remainder tails"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let align = self.align.max(1);
        let round_up = |x: i64| ((x + align - 1) / align) * align;
        let mut modified = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "dot_general") || op.operands.len() < 2 {
                    continue;
                }
                let lhs_ty = module.values[op.operands[0]].ty.as_tensor().cloned();
                let rhs_ty = module.values[op.operands[1]].ty.as_tensor().cloned();
                let (Some(lt), Some(rt)) = (lhs_ty, rhs_ty) else {
                    continue;
                };
                if lt.shape.len() != 2 || rt.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k), Dim::Static(n)) =
                    (&lt.shape[0], &lt.shape[1], &rt.shape[1])
                else {
                    continue;
                };
                let (pm, pk, pn) = (round_up(*m), round_up(*k), round_up(*n));
                let dst = &mut module.ops[op_id].attributes;
                dst.insert("pad_align".to_string(), Attribute::Int(align));
                dst.insert(
                    "padded_mnk".to_string(),
                    Attribute::DenseIntArray(vec![pm, pk, pn]),
                );
                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// Large GEMM Column-Splitting Pass (`split-large-gemm`: `Ir -> Ir`).
///
/// Decomposes 2D `stablehlo.dot_general` ops with output columns `N > max_n` into two smaller
/// column-tiled GEMMs (`slice(W, 0..N/2)` and `slice(W, N/2..N)`) followed by
/// `stablehlo.concatenate` along dimension `1`.
#[derive(Debug, Clone)]
pub struct SplitLargeGemmPass {
    /// Maximum output columns `N` before splitting a GEMM in half (default `2`).
    pub max_n: i64,
}

impl Default for SplitLargeGemmPass {
    fn default() -> Self {
        Self { max_n: 2 }
    }
}

impl Pass for SplitLargeGemmPass {
    fn name(&self) -> &'static str {
        "split-large-gemm"
    }

    fn description(&self) -> &'static str {
        "Splits wide dot_general GEMMs (N > max_n) into sliced sub-GEMMs + concatenate"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let max_n = self.max_n.max(1);
        let mut created = 0usize;
        let mut removed = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "dot_general")
                    || op.operands.len() < 2
                    || op.results.len() != 1
                {
                    continue;
                }
                let lhs_id = op.operands[0];
                let rhs_id = op.operands[1];
                let orig_res = op.results[0];
                let Some(lt) = module.values[lhs_id].ty.as_tensor().cloned() else {
                    continue;
                };
                let Some(rt) = module.values[rhs_id].ty.as_tensor().cloned() else {
                    continue;
                };
                if lt.shape.len() != 2 || rt.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k), Dim::Static(k2), Dim::Static(n)) =
                    (&lt.shape[0], &lt.shape[1], &rt.shape[0], &rt.shape[1])
                else {
                    continue;
                };
                if *k != *k2 || *n <= max_n || *n < 2 {
                    continue;
                }

                let n_left = *n / 2;
                let n_right = *n - n_left;
                let dt = lt.dtype.clone();

                // 1. Slice RHS left: [0..K, 0..n_left]
                let mut sl_l_attrs = IndexMap::new();
                sl_l_attrs.insert(
                    "start_indices".to_string(),
                    Attribute::DenseIntArray(vec![0, 0]),
                );
                sl_l_attrs.insert(
                    "limit_indices".to_string(),
                    Attribute::DenseIntArray(vec![*k, n_left]),
                );
                sl_l_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
                let w_l_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*k), Dim::Static(n_left)],
                    dt.clone(),
                ));
                let sl_l_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "slice"),
                    smallvec![rhs_id],
                    vec![(Some("w_split_l".to_string()), w_l_ty)],
                    sl_l_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let w_l_val = module.ops[sl_l_op].results[0];

                // 2. Slice RHS right: [0..K, n_left..N]
                let mut sl_r_attrs = IndexMap::new();
                sl_r_attrs.insert(
                    "start_indices".to_string(),
                    Attribute::DenseIntArray(vec![0, n_left]),
                );
                sl_r_attrs.insert(
                    "limit_indices".to_string(),
                    Attribute::DenseIntArray(vec![*k, *n]),
                );
                sl_r_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
                let w_r_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*k), Dim::Static(n_right)],
                    dt.clone(),
                ));
                let sl_r_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "slice"),
                    smallvec![rhs_id],
                    vec![(Some("w_split_r".to_string()), w_r_ty)],
                    sl_r_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let w_r_val = module.ops[sl_r_op].results[0];

                // 3. Left & Right sub-GEMMs
                let out_l_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*m), Dim::Static(n_left)],
                    dt.clone(),
                ));
                let dot_l_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    smallvec![lhs_id, w_l_val],
                    vec![(Some("gemm_l".to_string()), out_l_ty)],
                    op.attributes.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let dot_l_val = module.ops[dot_l_op].results[0];

                let out_r_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*m), Dim::Static(n_right)],
                    dt.clone(),
                ));
                let dot_r_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    smallvec![lhs_id, w_r_val],
                    vec![(Some("gemm_r".to_string()), out_r_ty)],
                    op.attributes.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let dot_r_val = module.ops[dot_r_op].results[0];

                // 4. Concatenate along dimension 1
                let mut cat_attrs = IndexMap::new();
                cat_attrs.insert("dimension".to_string(), Attribute::Int(1));
                let full_out_ty = module.values[orig_res].ty.clone();
                let cat_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "concatenate"),
                    smallvec![dot_l_val, dot_r_val],
                    vec![(Some("gemm_cat".to_string()), full_out_ty)],
                    cat_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let cat_val = module.ops[cat_op].results[0];

                module.replace_all_uses(orig_res, cat_val);
                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id]
                        .ops
                        .splice(pos..=pos, [sl_l_op, sl_r_op, dot_l_op, dot_r_op, cat_op]);
                }
                created += 5;
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: removed,
            ops_created: created,
        })
    }
}
