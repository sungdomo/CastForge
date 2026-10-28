use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, BlockId, MergeMethod, Module, OpName, TensorEntry, ValueDef,
    format_dense_f32_literal, merge_f32_tensors,
};
use indexmap::IndexMap;

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

                let (Some(attr_a), Some(attr_b)) = (
                    lhs_op.attributes.get("value").cloned(),
                    rhs_op.attributes.get("value").cloned(),
                ) else {
                    continue;
                };

                let is_resource = matches!(attr_a, Attribute::TensorRef { .. })
                    || matches!(attr_b, Attribute::TensorRef { .. });

                let (Some(vals_a), Some(vals_b)) = (
                    module.tensor_store.resolve_f32_attr(&attr_a, num_elems),
                    module.tensor_store.resolve_f32_attr(&attr_b, num_elems),
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
                    store_key.clone(),
                    tensor_ty.clone(),
                    merged,
                ));

                let mut attrs = IndexMap::new();
                let new_val_attr = if is_resource {
                    Attribute::TensorRef {
                        key: store_key,
                        ty: tensor_ty,
                    }
                } else {
                    Attribute::DenseElements {
                        ty: tensor_ty,
                        raw_literal: merged_lit,
                    }
                };
                attrs.insert("value".to_string(), new_val_attr);

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
