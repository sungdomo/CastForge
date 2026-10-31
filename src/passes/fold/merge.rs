//! Unified Weight Merging, Task Arithmetic/DARE, and LoRA Baking (`merge-weights`).

use super::{BakeDoraPass, BakeLoraPass};
use crate::ir::{
    Attribute, BlockId, Module, OpName, TensorEntry, ValueDef, ValueId, format_dense_f32_literal,
};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;

/// Unified Weight Merging, Task Arithmetic, DARE & LoRA Baking Pass (`merge-weights`).
pub struct MergeWeightsPass {
    pub method: String,
    pub t: f32,
    pub lambda: f32,
    pub drop_rate: f32,
    pub alpha: f32,
}

impl Default for MergeWeightsPass {
    fn default() -> Self {
        Self {
            method: "linear".to_string(),
            t: 0.5,
            lambda: 1.0,
            drop_rate: 0.0,
            alpha: 1.0,
        }
    }
}

impl Pass for MergeWeightsPass {
    fn name(&self) -> &'static str {
        "merge-weights"
    }

    fn description(&self) -> &'static str {
        "Unified weight merging, task arithmetic/DARE, and LoRA baking (slerp|ties|dare|lora)"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &["bake-lora", "bake-dora", "merge-task-vectors"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.method.to_ascii_lowercase();
        let res = match m.as_str() {
            "lora" | "bake_lora" => BakeLoraPass { alpha: self.alpha }.run(module)?,
            "dora" | "bake_dora" => BakeDoraPass { alpha: self.alpha }.run(module)?,
            "task_arithmetic" | "task" | "dare" => MergeTaskVectorsPass {
                lambda: self.lambda,
                drop_rate: self.drop_rate,
            }
            .run(module)?,
            "slerp" => PairwiseMergeWeightsPass {
                method: MergeMethod::Slerp,
                t: self.t,
            }
            .run(module)?,
            "ties" => PairwiseMergeWeightsPass {
                method: MergeMethod::Ties,
                t: self.t,
            }
            .run(module)?,
            _ => PairwiseMergeWeightsPass {
                method: MergeMethod::Linear,
                t: self.t,
            }
            .run(module)?,
        };
        Ok(res.with_name(self.name()))
    }
}

/// Pairwise Weight Merging helper (`merge-weights`).
pub struct PairwiseMergeWeightsPass {
    pub method: MergeMethod,
    pub t: f32,
}

impl Pass for PairwiseMergeWeightsPass {
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

/// Multi-Checkpoint Task Vector Merging & DARE Delta Pruning (`merge-task-vectors`).
pub struct MergeTaskVectorsPass {
    pub lambda: f32,
    pub drop_rate: f32,
}

impl Default for MergeTaskVectorsPass {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            drop_rate: 0.0,
        }
    }
}

impl Pass for MergeTaskVectorsPass {
    fn name(&self) -> &'static str {
        "merge-task-vectors"
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

/// Model Merging algorithms on two equal-length `f32` weight slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMethod {
    /// `(1 - t) * a + t * b`
    Linear,
    /// Spherical Linear Interpolation (SLERP)
    Slerp,
    /// TIES-Merging (Trim, Elect Sign, Disjoint Merge)
    Ties,
}

pub fn merge_f32_tensors(a: &[f32], b: &[f32], method: MergeMethod, t: f32) -> Vec<f32> {
    let len = a.len().min(b.len());
    match method {
        MergeMethod::Linear => (0..len).map(|i| (1.0 - t) * a[i] + t * b[i]).collect(),
        MergeMethod::Slerp => {
            let norm_a: f32 = a.iter().take(len).map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().take(len).map(|x| x * x).sum::<f32>().sqrt();
            if norm_a < 1e-8 || norm_b < 1e-8 {
                return merge_f32_tensors(a, b, MergeMethod::Linear, t);
            }
            let dot: f32 = (0..len).map(|i| (a[i] / norm_a) * (b[i] / norm_b)).sum();
            let dot_clamped = dot.clamp(-0.9995, 0.9995);
            let omega = dot_clamped.acos();
            let sin_omega = omega.sin();
            if sin_omega.abs() < 1e-6 {
                return merge_f32_tensors(a, b, MergeMethod::Linear, t);
            }
            let coeff_a = ((1.0 - t) * omega).sin() / sin_omega;
            let coeff_b = (t * omega).sin() / sin_omega;
            (0..len).map(|i| coeff_a * a[i] + coeff_b * b[i]).collect()
        }
        MergeMethod::Ties => (0..len)
            .map(|i| {
                let va = a[i];
                let vb = b[i];
                let elected_sign = ((1.0 - t) * va + t * vb).signum();
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                if va.signum() == elected_sign || va == 0.0 {
                    sum += (1.0 - t) * va;
                    weight_sum += 1.0 - t;
                }
                if vb.signum() == elected_sign || vb == 0.0 {
                    sum += t * vb;
                    weight_sum += t;
                }
                if weight_sum > 0.0 {
                    sum / weight_sum
                } else {
                    (1.0 - t) * va + t * vb
                }
            })
            .collect(),
    }
}
