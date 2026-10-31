//! Weight externalization (`externalize-weights`) and structured/2:4 sparsity pruning
//! (`prune-2to4`, `prune-weights`).

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, TensorEntry, TensorType, Type, ValueDef, ValueId,
    format_dense_f32_literal, format_dense_i8_literal, parse_dense_f32_literal,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use std::collections::HashSet;

/// Externalize / Inline Weights Pass (`externalize-weights`).
pub struct ExternalizeWeightsPass {
    pub min_elements: usize,
    pub inline: bool,
}

impl Default for ExternalizeWeightsPass {
    fn default() -> Self {
        Self {
            min_elements: 2,
            inline: false,
        }
    }
}

impl Pass for ExternalizeWeightsPass {
    fn name(&self) -> &'static str {
        "externalize-weights"
    }

    fn description(&self) -> &'static str {
        "Converts inline dense<...> weights to dense_resource<key> in TensorStore (or inline=true)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut claimed_keys: HashSet<String> = HashSet::new();
        for op_id in module.live_ops() {
            if let Some(Attribute::TensorRef { key, .. }) = module
                .ops
                .get(op_id)
                .and_then(|o| o.attributes.get("value"))
            {
                claimed_keys.insert(key.clone());
            }
        }
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_op("stablehlo", "constant") || op.results.len() != 1 {
                    continue;
                }
                let val_id = op.results[0];
                let Some(tensor_ty) = module.values[val_id].ty.as_tensor().cloned() else {
                    continue;
                };
                let num_elems = tensor_ty.num_elements().unwrap_or(1) as usize;
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };

                if self.inline {
                    if let Attribute::TensorRef { .. } = &val_attr {
                        let raw_literal = if matches!(tensor_ty.dtype, DType::QuantUniform { .. }) {
                            let Some(q_vals) =
                                module.tensor_store.resolve_i8_attr(&val_attr, num_elems)
                            else {
                                continue;
                            };
                            format_dense_i8_literal(&q_vals, &tensor_ty.shape)
                        } else {
                            let Some(f_vals) =
                                module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                            else {
                                continue;
                            };
                            format_dense_f32_literal(&f_vals, &tensor_ty.shape)
                        };
                        module.ops[op_id].attributes.insert(
                            "value".to_string(),
                            Attribute::DenseElements {
                                ty: tensor_ty,
                                raw_literal,
                            },
                        );
                        modified += 1;
                    }
                } else if let Attribute::DenseElements { raw_literal, .. } = &val_attr {
                    if num_elems < self.min_elements {
                        continue;
                    }
                    let Some(vals) = parse_dense_f32_literal(raw_literal, num_elems) else {
                        continue;
                    };
                    let base_key = module.values[val_id]
                        .ssa_name
                        .clone()
                        .unwrap_or_else(|| format!("weight_{modified}"));
                    let mut key = base_key.clone();
                    let mut suffix = 1usize;
                    while claimed_keys.contains(&key) {
                        key = format!("{base_key}_{suffix}");
                        suffix += 1;
                    }
                    claimed_keys.insert(key.clone());
                    module.tensor_store.insert(TensorEntry::from_f32(
                        key.clone(),
                        tensor_ty.clone(),
                        vals,
                    ));
                    module.ops[op_id].attributes.insert(
                        "value".to_string(),
                        Attribute::TensorRef { key, ty: tensor_ty },
                    );
                    modified += 1;
                }
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

    fn child_passes(&self) -> &'static [&'static str] {
        &["prune-structured", "prune-2to4", "prune-activations"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let res = if m == "2to4" {
            Prune2To4Pass.run(module)?
        } else if matches!(m.as_str(), "activation_sparse" | "activation" | "teal") {
            PruneActivationsPass {
                sparsity: self.sparsity,
            }
            .run(module)?
        } else {
            PruneStructuredPass {
                sparsity: self.sparsity,
            }
            .run(module)?
        };
        Ok(res.with_name(self.name()))
    }
}

/// Dynamic Top-K Activation Sparsity Pass (`prune-activations`).
pub struct PruneActivationsPass {
    pub sparsity: f32,
}

impl Default for PruneActivationsPass {
    fn default() -> Self {
        Self { sparsity: 0.5 }
    }
}

impl Pass for PruneActivationsPass {
    fn name(&self) -> &'static str {
        "prune-activations"
    }

    fn description(&self) -> &'static str {
        "Applies TEAL/CATS dynamic top-k magnitude activation sparsity to GEMM inputs"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(run_activation_sparsity(module, self.sparsity)?.with_name(self.name()))
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
            if attrs
                .get("castforge.activation_sparsity")
                .and_then(|a| a.as_float())
                == Some(ratio as f64)
                && attrs
                    .get("castforge.sparse_activation_mode")
                    .and_then(|a| a.as_str())
                    == Some("top_k_magnitude")
            {
                continue;
            }
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
pub struct PruneStructuredPass {
    pub sparsity: f32,
}

impl Default for PruneStructuredPass {
    fn default() -> Self {
        Self { sparsity: 0.5 }
    }
}

impl Pass for PruneStructuredPass {
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
                if op
                    .attributes
                    .get("castforge.structured_pruned")
                    .and_then(|a| a.as_bool())
                    == Some(true)
                {
                    continue;
                }
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
                };
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
                module.ops[op_id].attributes.insert(
                    "castforge.structured_pruned".to_string(),
                    Attribute::Bool(true),
                );

                if let Some(mut dot_out_ty) = module.values[dot_res].ty.as_tensor().cloned() {
                    if let Some(last_dim) = dot_out_ty.shape.last_mut() {
                        *last_dim = Dim::Static(keep_cols as i64);
                    }
                    let updated_out_ty = Type::Tensor(dot_out_ty);
                    module.values[dot_res].ty = updated_out_ty.clone();

                    for fn_op_id in module.functions() {
                        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
                            continue;
                        };
                        let Some(&fn_blk_id) = module.regions[reg_id].blocks.first() else {
                            continue;
                        };
                        let Some(&ret_op_id) = module.blocks[fn_blk_id].ops.last() else {
                            continue;
                        };
                        let Some(ret_op) = module.ops.get(ret_op_id) else {
                            continue;
                        };
                        if !ret_op.is_op("func", "return") {
                            continue;
                        }
                        let ret_indices: Vec<usize> = ret_op
                            .operands
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, &opnd)| (opnd == dot_res).then_some(idx))
                            .collect();
                        if ret_indices.is_empty() {
                            continue;
                        }
                        if let Some(Attribute::Type(Type::Function {
                            inputs,
                            mut results,
                        })) = module.ops[fn_op_id]
                            .attributes
                            .get("function_type")
                            .cloned()
                        {
                            for idx in ret_indices {
                                if idx < results.len() {
                                    results[idx] = updated_out_ty.clone();
                                }
                            }
                            module.ops[fn_op_id].attributes.insert(
                                "function_type".to_string(),
                                Attribute::Type(Type::Function { inputs, results }),
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

/// NVIDIA 2:4 Semi-Structured Weight Pruning Pass (`prune-2to4`).
#[derive(Debug, Clone, Default)]
pub struct Prune2To4Pass;

impl Pass for Prune2To4Pass {
    fn name(&self) -> &'static str {
        "prune-2to4"
    }

    fn description(&self) -> &'static str {
        "Enforces 2:4 semi-structured sparsity on weight tensors and annotates dot_general ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut rhs_weights: HashSet<ValueId> = HashSet::new();

        for op_id in module.live_ops() {
            let Some(op) = module.ops.get_mut(op_id) else {
                continue;
            };
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
                && module.ops[prod_op]
                    .attributes
                    .get("castforge.sparsity")
                    .and_then(|a| a.as_str())
                    != Some("2:4")
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
            let ssa_name = module.values[vid].ssa_name.clone();
            let new_val_attr =
                module
                    .tensor_store
                    .update_f32_constant(ssa_name.as_deref(), &val_attr, t_ty, vals);
            module.ops[op_id]
                .attributes
                .insert("value".to_string(), new_val_attr);
            module.ops[op_id].attributes.insert(
                "castforge.sparsity".to_string(),
                Attribute::String("2:4".to_string()),
            );
            modified += 1;
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
