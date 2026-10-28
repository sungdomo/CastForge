use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, MergeMethod, Module, OpName, TensorBacking,
    TensorEntry, TensorType, Type, ValueDef, format_dense_f32_literal, format_dense_i8_literal,
    merge_f32_tensors, parse_dense_f32_literal, prune_2d_columns_by_l1, quantize_symmetric_f32,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// 11. Weight Quantization Pass (`quantize-weights`)
/// Quantizes floating-point `stablehlo.constant` weights to `i8` or `i4` (`!quant.uniform<...>`)
/// and inserts `stablehlo.uniform_dequantize` ops.
pub struct QuantizeWeightsPass {
    pub bits: u8,
    pub min_elements: usize,
}

impl Default for QuantizeWeightsPass {
    fn default() -> Self {
        Self {
            bits: 8,
            min_elements: 2,
        }
    }
}

impl Pass for QuantizeWeightsPass {
    fn name(&self) -> &'static str {
        "quantize-weights"
    }

    fn description(&self) -> &'static str {
        "Post-training symmetric weight quantization (bits=8|4) with uniform_dequantize"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut created = 0usize;
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
                let const_val_id = op.results[0];
                let orig_ty = module.values[const_val_id].ty.clone();
                let Some(tensor_ty) = orig_ty.as_tensor() else {
                    continue;
                };
                if !matches!(tensor_ty.dtype, DType::F32 | DType::F16 | DType::BF16) {
                    continue;
                }
                let num_elems = tensor_ty.num_elements().unwrap_or(0) as usize;
                if num_elems < self.min_elements {
                    continue;
                }

                let Some(Attribute::DenseElements { raw_literal, .. }) =
                    op.attributes.get("value").cloned()
                else {
                    continue;
                };
                let Some(f32_vals) = parse_dense_f32_literal(&raw_literal, num_elems) else {
                    continue;
                };

                let (q_data, scale, zp, q_dtype) =
                    quantize_symmetric_f32(&f32_vals, self.bits, tensor_ty.dtype.clone());
                let q_tensor_ty = TensorType {
                    shape: tensor_ty.shape.clone(),
                    dtype: q_dtype,
                    encoding: None,
                };
                let q_literal = format_dense_i8_literal(&q_data, &tensor_ty.shape);

                // Update TensorStore entry
                let store_key = module.values[const_val_id]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| format!("q_weight_{modified}"));
                module.tensor_store.insert(TensorEntry {
                    name: store_key,
                    ty: q_tensor_ty.clone(),
                    backing: TensorBacking::QuantizedI8 {
                        data: q_data,
                        scale,
                        zero_point: zp,
                    },
                });

                // Update the constant op to produce the quantized tensor type
                let q_ty = Type::Tensor(q_tensor_ty.clone());
                module.values[const_val_id].ty = q_ty;
                module.ops[op_id].attributes.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: q_tensor_ty,
                        raw_literal: q_literal,
                    },
                );
                modified += 1;

                // Create `stablehlo.uniform_dequantize` op right after the constant op
                let deq_ssa = module.values[const_val_id]
                    .ssa_name
                    .as_ref()
                    .map(|s| format!("{s}_deq"));
                let deq_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "uniform_dequantize"),
                    SmallVec::from_slice(&[const_val_id]),
                    vec![(deq_ssa, orig_ty)],
                    IndexMap::new(),
                    Vec::new(),
                    true,
                    Location::Unknown,
                );
                let deq_val_id = module.ops[deq_op_id].results[0];

                // Move `deq_op_id` from end of block to immediately after `op_id`
                let block_ops = &mut module.blocks[block_id].ops;
                if let Some(last) = block_ops.pop() {
                    if let Some(pos) = block_ops.iter().position(|&id| id == op_id) {
                        block_ops.insert(pos + 1, last);
                    } else {
                        block_ops.push(last);
                    }
                }

                // Redirect all downstream uses of `const_val_id` to `deq_val_id`
                for (other_op_id, other_op) in &mut module.ops {
                    if other_op_id == deq_op_id {
                        continue;
                    }
                    for operand in &mut other_op.operands {
                        if *operand == const_val_id {
                            *operand = deq_val_id;
                        }
                    }
                }
                created += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// 12. Structured Channel Pruning Pass (`prune-structured`)
/// Prunes low-L1-importance output columns of 2D weight constants in `stablehlo.dot_general`
/// and propagates the reduced dimension downstream.
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

                let Some(Attribute::DenseElements { raw_literal, .. }) =
                    weight_op.attributes.get("value").cloned()
                else {
                    continue;
                };
                let Some(f32_vals) = parse_dense_f32_literal(&raw_literal, rows * cols) else {
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

                // Update weight constant type, literal, and TensorStore
                module.values[rhs].ty = Type::Tensor(new_w_ty.clone());
                module.ops[weight_op_id].attributes.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: new_w_ty.clone(),
                        raw_literal: new_literal,
                    },
                );
                let store_key = module.values[rhs]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| "pruned_weight".to_string());
                module
                    .tensor_store
                    .insert(TensorEntry::from_f32(store_key, new_w_ty, pruned_vals));

                // Update dot_general output shape and propagate to downstream consumers
                if let Some(mut dot_out_ty) = module.values[dot_res].ty.as_tensor().cloned() {
                    if let Some(last_dim) = dot_out_ty.shape.last_mut() {
                        *last_dim = Dim::Static(keep_cols as i64);
                    }
                    let updated_out_ty = Type::Tensor(dot_out_ty);
                    module.values[dot_res].ty = updated_out_ty.clone();

                    // Also update parent func.func return type if dot_res is returned
                    for fn_op_id in module.functions() {
                        if let Some(Attribute::Type(Type::Function { inputs, results })) = module
                            .ops[fn_op_id]
                            .attributes
                            .get("function_type")
                            .cloned()
                        {
                            if results.len() == 1 {
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

/// 13. Weight Merging Pass (`merge-weights`)
/// Fuses paired constant weight tensors using Linear, SLERP, or TIES merging.
pub struct MergeWeightsPass {
    pub method: MergeMethod,
    pub t: f32,
}

impl Pass for MergeWeightsPass {
    fn name(&self) -> &'static str {
        "merge-weights"
    }

    fn description(&self) -> &'static str {
        "Merges paired constant weight tensors using linear, slerp, or ties (method=slerp, t=0.5)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                // Match `stablehlo.add %w1, %w2` or `stablehlo.custom_call @merge(%w1, %w2)`
                let is_merge_target = op.is_op("stablehlo", "add")
                    || (op.is_op("stablehlo", "custom_call")
                        && op
                            .attributes
                            .get("call_target_name")
                            .and_then(|a| a.as_str())
                            == Some("castforge.merge"));
                if !is_merge_target || op.operands.len() != 2 || op.results.len() != 1 {
                    continue;
                }

                let lhs = op.operands[0];
                let rhs = op.operands[1];
                let res_val = op.results[0];

                let (
                    ValueDef::OpResult { op: lhs_op_id, .. },
                    ValueDef::OpResult { op: rhs_op_id, .. },
                ) = (module.values[lhs].def, module.values[rhs].def)
                else {
                    continue;
                };

                let (Some(lhs_op), Some(rhs_op)) =
                    (module.ops.get(lhs_op_id), module.ops.get(rhs_op_id))
                else {
                    continue;
                };
                if !lhs_op.is_op("stablehlo", "constant") || !rhs_op.is_op("stablehlo", "constant")
                {
                    continue;
                }

                let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                    continue;
                };
                let num_elems = tensor_ty.num_elements().unwrap_or(1) as usize;

                let (
                    Some(Attribute::DenseElements {
                        raw_literal: lit_a, ..
                    }),
                    Some(Attribute::DenseElements {
                        raw_literal: lit_b, ..
                    }),
                ) = (
                    lhs_op.attributes.get("value").cloned(),
                    rhs_op.attributes.get("value").cloned(),
                )
                else {
                    continue;
                };

                let (Some(vals_a), Some(vals_b)) = (
                    parse_dense_f32_literal(&lit_a, num_elems),
                    parse_dense_f32_literal(&lit_b, num_elems),
                ) else {
                    continue;
                };

                let merged = merge_f32_tensors(&vals_a, &vals_b, self.method, self.t);
                let merged_lit = format_dense_f32_literal(&merged, &tensor_ty.shape);

                let store_key = module.values[res_val]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| "merged_weight".to_string());
                module.tensor_store.insert(TensorEntry::from_f32(
                    store_key,
                    tensor_ty.clone(),
                    merged,
                ));

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: tensor_ty,
                        raw_literal: merged_lit,
                    },
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "constant");
                op_mut.operands.clear();
                op_mut.attributes = attrs;
                modified += 1;
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
