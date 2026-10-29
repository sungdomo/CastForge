use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, BlockId, Module, OpName, TensorEntry, ValueDef, ValueId, format_dense_f32_literal,
};
use indexmap::IndexMap;

/// Weight Surgery Pass: Multi-Checkpoint Task Arithmetic & DARE Delta Pruning (`task-arithmetic`).
/// Evaluates `stablehlo.custom_call %w_base, %w_1, ...` (`"castforge.task_arithmetic"`)
/// by computing `W_new = W_base + sum_i(lambda * DARE(W_i - W_base, drop_rate))`.
pub struct TaskArithmeticPass {
    pub lambda: f32,
    pub drop_rate: f32,
}

impl Default for TaskArithmeticPass {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            drop_rate: 0.0,
        }
    }
}

impl Pass for TaskArithmeticPass {
    fn name(&self) -> &'static str {
        "task-arithmetic"
    }

    fn description(&self) -> &'static str {
        "Applies multi-checkpoint task vector arithmetic and DARE pruning (lambda, drop_rate)"
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
                let is_target = op.is_op("stablehlo", "custom_call")
                    && op
                        .attributes
                        .get("call_target_name")
                        .and_then(|a| a.as_str())
                        == Some("castforge.task_arithmetic");
                if !is_target || op.operands.len() < 2 || op.results.len() != 1 {
                    continue;
                }

                let res_val = op.results[0];
                let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                    continue;
                };
                let num_elems = tensor_ty.num_elements().unwrap_or(1) as usize;
                let operands: Vec<ValueId> = op.operands.iter().copied().collect();

                let Some((mut base_vals, mut any_resource)) =
                    resolve_const_operand(module, operands[0], num_elems)
                else {
                    continue;
                };

                let mut task_vectors = Vec::with_capacity(operands.len() - 1);
                let mut all_valid = true;
                for &ft_id in &operands[1..] {
                    if let Some((ft_vals, is_res)) = resolve_const_operand(module, ft_id, num_elems)
                    {
                        any_resource |= is_res;
                        let delta: Vec<f32> = ft_vals
                            .iter()
                            .zip(base_vals.iter())
                            .map(|(w, b)| w - b)
                            .collect();
                        task_vectors.push(apply_dare_pruning(&delta, self.drop_rate));
                    } else {
                        all_valid = false;
                        break;
                    }
                }
                if !all_valid {
                    continue;
                }

                for tv in &task_vectors {
                    for (b, &d) in base_vals.iter_mut().zip(tv.iter()) {
                        *b += self.lambda * d;
                    }
                }

                let merged_lit = format_dense_f32_literal(&base_vals, &tensor_ty.shape);
                let store_key = module.values[res_val]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| "task_weight".to_string());
                module.tensor_store.insert(TensorEntry::from_f32(
                    store_key.clone(),
                    tensor_ty.clone(),
                    base_vals,
                ));

                let mut attrs = IndexMap::new();
                let new_val_attr = if any_resource {
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

fn resolve_const_operand(
    module: &Module,
    val_id: ValueId,
    num_elems: usize,
) -> Option<(Vec<f32>, bool)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if !op.is_op("stablehlo", "constant") {
        return None;
    }
    let attr = op.attributes.get("value")?;
    let is_res = matches!(attr, Attribute::TensorRef { .. });
    let vals = module.tensor_store.resolve_f32_attr(attr, num_elems)?;
    Some((vals, is_res))
}

/// DARE-style deterministic magnitude drop-and-rescale: drops the smallest `drop_rate` fraction
/// of delta entries and rescales survivors by `1 / (1 - drop_rate)`.
fn apply_dare_pruning(delta: &[f32], drop_rate: f32) -> Vec<f32> {
    let p = drop_rate.clamp(0.0, 0.99);
    if p <= 1e-6 || delta.is_empty() {
        return delta.to_vec();
    }
    let drop_count = ((delta.len() as f32) * p).floor() as usize;
    if drop_count == 0 {
        return delta.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = delta
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let rescale = 1.0_f32 / (1.0_f32 - p);
    let mut out = delta.to_vec();
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        if rank < drop_count {
            out[idx] = 0.0;
        } else {
            out[idx] *= rescale;
        }
    }
    out
}
