//! Weight Pruning & Structured Sparsity (`prune-weights`).
//!
//! Consolidates `prune-structured` (`mode=structured`) and `sparsify-2to4` (`mode=2to4`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Module, TensorEntry, TensorType, Type, ValueDef, ValueId,
    format_dense_f32_literal,
};
use std::collections::HashSet;

/// Unified Weight Pruning & Structured Sparsity Pass (`prune-weights`).
pub struct PruneWeightsPass {
    pub mode: String,
    pub sparsity: f32,
}

impl Default for PruneWeightsPass {
    fn default() -> Self {
        Self {
            mode: "structured".to_string(),
            sparsity: 0.5,
        }
    }
}

impl Pass for PruneWeightsPass {
    fn name(&self) -> &'static str {
        "prune-weights"
    }

    fn description(&self) -> &'static str {
        "Unified weight & activation sparsity (mode=structured|2to4|activation_sparse)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut res = if m == "2to4" {
            Sparsify2To4Pass.run(module)?
        } else if matches!(m.as_str(), "activation_sparse" | "activation" | "teal") {
            run_activation_sparsity(module, self.sparsity)?
        } else {
            StructuredPruningPass {
                sparsity: self.sparsity,
            }
            .run(module)?
        };
        res.pass_name = self.name().to_string();
        Ok(res)
    }
}

fn run_activation_sparsity(module: &mut Module, sparsity: f32) -> Result<PassResult> {
    let ratio = sparsity.clamp(0.05, 0.95);
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();
    for block_id in blocks {
        for op_id in module.block_ops(block_id) {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let is_gemm = matches!(
                StableHloOp::from_op(op),
                Some(StableHloOp::DotGeneral { .. })
            ) || (op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n.starts_with("castforge.fused_")));
            if !is_gemm {
                continue;
            }
            let attrs = &mut module.ops[op_id].attributes;
            attrs.insert(
                "castforge.activation_sparsity".to_string(),
                Attribute::Float(ratio as f64),
            );
            attrs.insert(
                "castforge.sparse_activation_mode".to_string(),
                Attribute::String("top_k_magnitude".to_string()),
            );
            modified += 1;
        }
    }
    Ok(PassResult {
        pass_name: "prune-weights".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

/// Structured Channel Pruning Pass (`prune-structured`).
pub struct StructuredPruningPass {
    pub sparsity: f32,
}

impl Default for StructuredPruningPass {
    fn default() -> Self {
        Self { sparsity: 0.5 }
    }
}

impl Pass for StructuredPruningPass {
    fn name(&self) -> &'static str {
        "prune-structured"
    }

    fn description(&self) -> &'static str {
        "Structured L1 channel pruning on 2D weight matrices in dot_general (sparsity=0.5)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    rhs,
                    result: dot_res,
                    ..
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let ValueDef::OpResult {
                    op: weight_op_id, ..
                } = module.values[rhs].def
                else {
                    continue;
                };
                let Some(weight_op) = module.ops.get(weight_op_id) else {
                    continue;
                };
                if !weight_op.is_op("stablehlo", "constant") {
                    continue;
                }

                let Some(w_ty) = module.values[rhs].ty.as_tensor().cloned() else {
                    continue;
                };
                if w_ty.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(rows_i64), Dim::Static(cols_i64)) =
                    (&w_ty.shape[0], &w_ty.shape[1])
                else {
                    continue;
                };
                let rows = *rows_i64 as usize;
                let cols = *cols_i64 as usize;
                if cols <= 1 {
                    continue;
                }

                let keep_cols =
                    ((cols as f32) * (1.0 - self.sparsity.clamp(0.0, 0.9))).round() as usize;
                let keep_cols = keep_cols.clamp(1, cols);
                if keep_cols == cols {
                    continue;
                }

                let Some(val_attr) = weight_op.attributes.get("value").cloned() else {
                    continue;
                };
                let resource_key = match &val_attr {
                    Attribute::TensorRef { key, .. } => Some(key.clone()),
                    _ => None,
                };
                let Some(f32_vals) = module.tensor_store.resolve_f32_attr(&val_attr, rows * cols)
                else {
                    continue;
                };

                let (pruned_vals, _kept_cols) =
                    prune_2d_columns_by_l1(&f32_vals, rows, cols, keep_cols);

                let new_w_ty = TensorType {
                    shape: vec![Dim::Static(rows as i64), Dim::Static(keep_cols as i64)],
                    dtype: w_ty.dtype.clone(),
                    encoding: None,
                };
                let new_literal = format_dense_f32_literal(&pruned_vals, &new_w_ty.shape);

                let store_key = resource_key.clone().unwrap_or_else(|| {
                    module.values[rhs]
                        .ssa_name
                        .clone()
                        .unwrap_or_else(|| "pruned_weight".to_string())
                });

                module.values[rhs].ty = Type::Tensor(new_w_ty.clone());
                let updated_attr = if let Some(key) = resource_key {
                    Attribute::TensorRef {
                        key,
                        ty: new_w_ty.clone(),
                    }
                } else {
                    Attribute::DenseElements {
                        ty: new_w_ty.clone(),
                        raw_literal: new_literal,
                    }
                };
                module.ops[weight_op_id]
                    .attributes
                    .insert("value".to_string(), updated_attr);
                module
                    .tensor_store
                    .insert(TensorEntry::from_f32(store_key, new_w_ty, pruned_vals));

                if let Some(mut dot_out_ty) = module.values[dot_res].ty.as_tensor().cloned() {
                    if let Some(last_dim) = dot_out_ty.shape.last_mut() {
                        *last_dim = Dim::Static(keep_cols as i64);
                    }
                    let updated_out_ty = Type::Tensor(dot_out_ty);
                    module.values[dot_res].ty = updated_out_ty.clone();

                    for fn_op_id in module.functions() {
                        if let Some(Attribute::Type(Type::Function { inputs, results })) = module
                            .ops[fn_op_id]
                            .attributes
                            .get("function_type")
                            .cloned()
                            && results.len() == 1
                        {
                            module.ops[fn_op_id].attributes.insert(
                                "function_type".to_string(),
                                Attribute::Type(Type::Function {
                                    inputs,
                                    results: vec![updated_out_ty.clone()],
                                }),
                            );
                        }
                    }
                }

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

/// NVIDIA 2:4 Semi-Structured Weight Sparsifier Pass (`sparsify-2to4`).
#[derive(Debug, Clone, Default)]
pub struct Sparsify2To4Pass;

impl Pass for Sparsify2To4Pass {
    fn name(&self) -> &'static str {
        "sparsify-2to4"
    }

    fn description(&self) -> &'static str {
        "Enforces 2:4 semi-structured sparsity on weight tensors and annotates dot_general ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut rhs_weights: HashSet<ValueId> = HashSet::new();

        for (_, op) in module.ops.iter_mut() {
            if op.is_op("stablehlo", "dot_general") && op.operands.len() >= 2 {
                rhs_weights.insert(op.operands[1]);
                if op.attributes.get("sparsity").and_then(|a| a.as_str()) != Some("2:4") {
                    op.attributes
                        .insert("sparsity".to_string(), Attribute::String("2:4".to_string()));
                    modified += 1;
                }
            }
        }

        let mut const_ops = Vec::new();
        for &vid in &rhs_weights {
            if let ValueDef::OpResult { op: prod_op, .. } = module.values[vid].def
                && module.ops[prod_op].is_op("stablehlo", "constant")
            {
                const_ops.push((prod_op, vid));
            }
        }

        for (op_id, vid) in const_ops {
            let Some(t_ty) = module.values[vid].ty.as_tensor().cloned() else {
                continue;
            };
            let n = t_ty.num_elements().unwrap_or(1).max(1) as usize;
            let Some(val_attr) = module.ops[op_id].attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut vals) = module.tensor_store.resolve_f32_attr(&val_attr, n) else {
                continue;
            };
            apply_2to4_mask(&mut vals);

            match val_attr {
                Attribute::TensorRef { key, ty } => {
                    module.tensor_store.insert(TensorEntry::from_f32(
                        key.clone(),
                        TensorType::new(ty.shape.clone(), DType::F32),
                        vals,
                    ));
                    modified += 1;
                }
                Attribute::DenseElements { ty, .. } => {
                    let lit = format_dense_f32_literal(&vals, &ty.shape);
                    module.ops[op_id].attributes.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty,
                            raw_literal: lit,
                        },
                    );
                    modified += 1;
                }
                _ => {}
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

/// Zeroes out the 2 smallest-magnitude elements in every contiguous 4-element chunk of `vals`.
pub fn apply_2to4_mask(vals: &mut [f32]) {
    for chunk in vals.chunks_mut(4) {
        if chunk.len() < 4 {
            continue;
        }
        let mut indexed: [(usize, f32); 4] = [
            (0, chunk[0].abs()),
            (1, chunk[1].abs()),
            (2, chunk[2].abs()),
            (3, chunk[3].abs()),
        ];
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        chunk[indexed[0].0] = 0.0;
        chunk[indexed[1].0] = 0.0;
    }
}

/// Structured pruning of a 2D weight matrix `[rows, cols]` along its output columns (`dim = 1`),
/// keeping the top `keep_cols` columns with highest L1 importance norm.
/// Returns `(pruned_f32_data, kept_column_indices)`.
pub fn prune_2d_columns_by_l1(
    values: &[f32],
    rows: usize,
    cols: usize,
    keep_cols: usize,
) -> (Vec<f32>, Vec<usize>) {
    if rows == 0 || cols == 0 || values.len() < rows * cols {
        return (values.to_vec(), (0..cols).collect());
    }
    let keep_cols = keep_cols.clamp(1, cols);
    let mut col_scores: Vec<(usize, f32)> = (0..cols)
        .map(|c| {
            let mut l1 = 0.0f32;
            for r in 0..rows {
                l1 += values[r * cols + c].abs();
            }
            (c, l1)
        })
        .collect();

    col_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept_indices: Vec<usize> = col_scores
        .into_iter()
        .take(keep_cols)
        .map(|(idx, _)| idx)
        .collect();
    kept_indices.sort_unstable();

    let mut pruned = Vec::with_capacity(rows * keep_cols);
    for r in 0..rows {
        for &c in &kept_indices {
            pruned.push(values[r * cols + c]);
        }
    }
    (pruned, kept_indices)
}
