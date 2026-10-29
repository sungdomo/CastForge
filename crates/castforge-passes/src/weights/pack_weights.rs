use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Module, OpId, TensorBacking, TensorType, Type, ValueDef,
    format_dense_f32_literal,
};
use std::collections::HashSet;

/// Compile-Time Weight Pre-Packing & Layout Transformation Pass (`pack-weights`).
///
/// Inspired by Modular MAX Engine's compile-time weight layout pre-packing:
/// In a standard 2D `dot_general(x, W)` where `x` is `[M, K]` and `W` is `[K, N]`
/// (`rhs_contracting_dimensions = [0]`), reading `W[k * N + n]` in the inner `k` reduction loop
/// strides across memory by `N` elements per step (cache-hostile column access).
///
/// This pass transposes constant 2D RHS weight tensors `W` at compile time into `[N, K]` layout
/// (`W_packed[n * K + k] = W[k * N + n]`) in `TensorStore` / `DenseElements` and updates
/// `rhs_contracting_dimensions = [1]` (`packed_rhs = true`) on `stablehlo.dot_general` and
/// `castforge.fused_matmul_epilogue` ops so both `x` and `W_packed` are accessed with unit stride
/// (`stride = 1`), enabling contiguous cache lines and SIMD auto-vectorization.
pub struct PackWeightsPass;

impl Pass for PackWeightsPass {
    fn name(&self) -> &'static str {
        "pack-weights"
    }

    fn description(&self) -> &'static str {
        "Pre-packs 2D constant RHS weights from [K, N] to [N, K] for unit-stride SIMD MatMul"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut packed_const_ops: HashSet<OpId> = HashSet::new();
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };

                // Match either `stablehlo.dot_general` or `castforge.fused_matmul_epilogue`
                let is_cand = match StableHloOp::from_op(op) {
                    Some(StableHloOp::DotGeneral {
                        dimension_numbers: dims,
                        ..
                    }) => {
                        dims.lhs_batching_dimensions.is_empty()
                            && dims.rhs_batching_dimensions.is_empty()
                            && dims.lhs_contracting_dimensions == [1]
                            && dims.rhs_contracting_dimensions == [0]
                    }
                    Some(StableHloOp::Composite { name, inputs, .. })
                        if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 =>
                    {
                        let rhs_c = op
                            .attributes
                            .get("rhs_contracting_dimensions")
                            .and_then(|a| match a {
                                Attribute::DenseIntArray(v) => Some(v.as_slice()),
                                _ => None,
                            })
                            .unwrap_or(&[0]);
                        rhs_c == [0]
                    }
                    _ => false,
                };
                if !is_cand {
                    continue;
                }

                let Some(&rhs_val) = module.ops[op_id].operands.get(1) else {
                    continue;
                };
                let ValueDef::OpResult { op: w_op_id, .. } = module.values[rhs_val].def else {
                    continue;
                };

                // If `w_op_id` was not yet packed, transpose its constant tensor `[K, N] -> [N, K]`
                if !packed_const_ops.contains(&w_op_id) {
                    let Some(w_op) = module.ops.get(w_op_id) else {
                        continue;
                    };
                    if !w_op.is_op("stablehlo", "constant") {
                        continue;
                    }
                    let Some(w_ty) = module.values[rhs_val].ty.as_tensor().cloned() else {
                        continue;
                    };
                    if w_ty.dtype != DType::F32 || w_ty.shape.len() != 2 {
                        continue;
                    }
                    let (Dim::Static(k_i64), Dim::Static(n_i64)) = (&w_ty.shape[0], &w_ty.shape[1])
                    else {
                        continue;
                    };
                    let (k, n) = (*k_i64 as usize, *n_i64 as usize);
                    let Some(w_attr) = w_op.attributes.get("value").cloned() else {
                        continue;
                    };
                    let Some(w_vals) = module.tensor_store.resolve_f32_attr(&w_attr, k * n) else {
                        continue;
                    };

                    let mut packed_vals = vec![0.0_f32; n * k];
                    for row_k in 0..k {
                        for col_n in 0..n {
                            packed_vals[col_n * k + row_k] = w_vals[row_k * n + col_n];
                        }
                    }

                    let packed_ty = TensorType {
                        shape: vec![Dim::Static(n as i64), Dim::Static(k as i64)],
                        dtype: DType::F32,
                        encoding: w_ty.encoding.clone(),
                    };

                    match &w_attr {
                        Attribute::TensorRef { key, .. } => {
                            if let Some(entry) = module.tensor_store.get_mut(key) {
                                entry.ty = packed_ty.clone();
                                entry.backing = TensorBacking::F32Data(packed_vals);
                            }
                            module.ops[w_op_id].attributes.insert(
                                "value".to_string(),
                                Attribute::TensorRef {
                                    key: key.clone(),
                                    ty: packed_ty.clone(),
                                },
                            );
                        }
                        _ => {
                            let raw_literal =
                                format_dense_f32_literal(&packed_vals, &packed_ty.shape);
                            module.ops[w_op_id].attributes.insert(
                                "value".to_string(),
                                Attribute::DenseElements {
                                    ty: packed_ty.clone(),
                                    raw_literal,
                                },
                            );
                        }
                    }
                    module.values[rhs_val].ty = Type::Tensor(packed_ty);
                    packed_const_ops.insert(w_op_id);
                    modified += 1;
                }

                // Update consuming op to contract along RHS dimension 1 (`[N, K]`)
                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert(
                    "lhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![1]),
                );
                op_mut.attributes.insert(
                    "rhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![1]),
                );
                op_mut
                    .attributes
                    .insert("packed_rhs".to_string(), Attribute::Bool(true));
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
