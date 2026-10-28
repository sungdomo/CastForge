use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, BlockId, DType, Module, TensorEntry, format_dense_f32_literal,
    format_dense_i8_literal, parse_dense_f32_literal,
};

/// 20. Externalize / Inline Weights Pass (`externalize-weights`)
/// Converts inline `dense<...>` constants into external `dense_resource<key>` references backed by
/// `Module::tensor_store` (or inlines `dense_resource<key>` back into `dense<...>` when
/// `inline=true`).
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
                    let key = module.values[val_id]
                        .ssa_name
                        .clone()
                        .unwrap_or_else(|| format!("weight_{modified}"));
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
