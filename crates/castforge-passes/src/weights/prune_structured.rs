use crate::{Pass, PassResult, prune_2d_columns_by_l1};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Dim, Module, TensorEntry, TensorType, Type, ValueDef,
    format_dense_f32_literal,
};

/// Structured Channel Pruning Pass (`prune-structured`).
///
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

                // Update weight constant type, literal/resource, and TensorStore
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
