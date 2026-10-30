//! Unified Weight Merging, Task Arithmetic/DARE, and LoRA Baking (`merge-weights`).

use crate::ir::{
    Attribute, BlockId, DType, Dim, Module, OpName, TensorBacking, TensorEntry, ValueDef, ValueId,
    format_dense_f32_literal,
};
use crate::passes::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;

/// Unified Weight Merging, Task Arithmetic, DARE & LoRA Baking Pass (`merge-weights`).
pub struct UnifiedMergeWeightsPass {
    pub method: String,
    pub t: f32,
    pub lambda: f32,
    pub drop_rate: f32,
    pub alpha: f32,
}

impl Default for UnifiedMergeWeightsPass {
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

impl Pass for UnifiedMergeWeightsPass {
    fn name(&self) -> &'static str {
        "merge-weights"
    }

    fn description(&self) -> &'static str {
        "Unified weight merging, task arithmetic/DARE, and LoRA baking (slerp|ties|dare|lora)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.method.to_ascii_lowercase();
        let mut res = match m.as_str() {
            "lora" | "bake_lora" => BakeLoraPass { alpha: self.alpha }.run(module)?,
            "dora" | "bake_dora" => run_bake_dora(module, self.alpha)?,
            "task_arithmetic" | "task" | "dare" => TaskArithmeticPass {
                lambda: self.lambda,
                drop_rate: self.drop_rate,
            }
            .run(module)?,
            "slerp" => MergeWeightsPass {
                method: MergeMethod::Slerp,
                t: self.t,
            }
            .run(module)?,
            "ties" => MergeWeightsPass {
                method: MergeMethod::Ties,
                t: self.t,
            }
            .run(module)?,
            _ => MergeWeightsPass {
                method: MergeMethod::Linear,
                t: self.t,
            }
            .run(module)?,
        };
        res.pass_name = self.name().to_string();
        Ok(res)
    }
}

/// Weight Merging Pass (`merge-weights`).
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

/// LoRA Adapter Baking Pass (`bake-lora`).
pub struct BakeLoraPass {
    pub alpha: f32,
}

impl Default for BakeLoraPass {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Pass for BakeLoraPass {
    fn name(&self) -> &'static str {
        "bake-lora"
    }

    fn description(&self) -> &'static str {
        "Fuses LoRA adapter branches W + alpha * (A @ B) into base weight matrices at compile time"
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
                let Some(StableHloOp::Add {
                    lhs,
                    rhs,
                    result: add_res,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let matched = match_base_and_lora(module, lhs, rhs, self.alpha)
                    .or_else(|| match_base_and_lora(module, rhs, lhs, self.alpha));
                let Some(LoraMatch {
                    base_dot_val,
                    w_op_id,
                    a_op_id,
                    b_op_id,
                    k,
                    r,
                    n,
                    eff_alpha,
                }) = matched
                else {
                    continue;
                };

                let (Some(w_attr), Some(a_attr), Some(b_attr)) = (
                    module.ops[w_op_id].attributes.get("value").cloned(),
                    module.ops[a_op_id].attributes.get("value").cloned(),
                    module.ops[b_op_id].attributes.get("value").cloned(),
                ) else {
                    continue;
                };

                let (Some(mut w_vals), Some(a_vals), Some(b_vals)) = (
                    module.tensor_store.resolve_f32_attr(&w_attr, k * n),
                    module.tensor_store.resolve_f32_attr(&a_attr, k * r),
                    module.tensor_store.resolve_f32_attr(&b_attr, r * n),
                ) else {
                    continue;
                };

                for row in 0..k {
                    for col in 0..n {
                        let mut delta = 0.0_f32;
                        for p in 0..r {
                            delta += a_vals[row * r + p] * b_vals[p * n + col];
                        }
                        w_vals[row * n + col] += eff_alpha * delta;
                    }
                }

                let Some(w_res_val) = module.ops[w_op_id].results.first().copied() else {
                    continue;
                };
                let Some(w_ty) = module.values[w_res_val].ty.as_tensor().cloned() else {
                    continue;
                };
                match &w_attr {
                    Attribute::TensorRef { key, .. } => {
                        if let Some(entry) = module.tensor_store.get_mut(key) {
                            entry.backing = TensorBacking::F32Data(w_vals);
                        }
                    }
                    _ => {
                        let raw_literal = format_dense_f32_literal(&w_vals, &w_ty.shape);
                        module.ops[w_op_id].attributes.insert(
                            "value".to_string(),
                            Attribute::DenseElements {
                                ty: w_ty,
                                raw_literal,
                            },
                        );
                    }
                }

                module.replace_all_uses(add_res, base_dot_val);
                module.remove_op(block_id, op_id);
                modified += 1;
                removed += 1;
            }
        }

        if modified > 0 || removed > 0 {
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

struct LoraMatch {
    base_dot_val: ValueId,
    w_op_id: crate::ir::OpId,
    a_op_id: crate::ir::OpId,
    b_op_id: crate::ir::OpId,
    k: usize,
    r: usize,
    n: usize,
    eff_alpha: f32,
}

fn match_base_and_lora(
    module: &Module,
    base_cand: ValueId,
    lora_cand: ValueId,
    default_alpha: f32,
) -> Option<LoraMatch> {
    let ValueDef::OpResult { op: base_op_id, .. } = module.values[base_cand].def else {
        return None;
    };
    let base_op = module.ops.get(base_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: x_base,
        rhs: w_val,
        ..
    } = StableHloOp::from_op(base_op)?
    else {
        return None;
    };
    let (w_op_id, k, n) = const_2d_dims(module, w_val)?;

    let mut dot_b_val = lora_cand;
    let mut eff_alpha = default_alpha;
    if let ValueDef::OpResult { op: m_id, .. } = module.values[lora_cand].def
        && let Some(m_op) = module.ops.get(m_id)
        && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(m_op)
    {
        if let Some(s) = FoldConstantsPass::const_splat(module, rhs) {
            dot_b_val = lhs;
            eff_alpha = s as f32;
        } else if let Some(s) = FoldConstantsPass::const_splat(module, lhs) {
            dot_b_val = rhs;
            eff_alpha = s as f32;
        }
    }

    let ValueDef::OpResult {
        op: dot_b_op_id, ..
    } = module.values[dot_b_val].def
    else {
        return None;
    };
    let dot_b_op = module.ops.get(dot_b_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: dot_a_val,
        rhs: b_val,
        ..
    } = StableHloOp::from_op(dot_b_op)?
    else {
        return None;
    };
    let (b_op_id, r_b, n_b) = const_2d_dims(module, b_val)?;
    if n_b != n {
        return None;
    }

    let ValueDef::OpResult {
        op: dot_a_op_id, ..
    } = module.values[dot_a_val].def
    else {
        return None;
    };
    let dot_a_op = module.ops.get(dot_a_op_id)?;
    let StableHloOp::DotGeneral {
        lhs: x_lora,
        rhs: a_val,
        ..
    } = StableHloOp::from_op(dot_a_op)?
    else {
        return None;
    };
    if x_lora != x_base {
        return None;
    }
    let (a_op_id, k_a, r_a) = const_2d_dims(module, a_val)?;
    if k_a != k || r_a != r_b {
        return None;
    }

    Some(LoraMatch {
        base_dot_val: base_cand,
        w_op_id,
        a_op_id,
        b_op_id,
        k,
        r: r_a,
        n,
        eff_alpha,
    })
}

fn const_2d_dims(module: &Module, val_id: ValueId) -> Option<(crate::ir::OpId, usize, usize)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if !op.is_op("stablehlo", "constant") {
        return None;
    }
    let t = module.values[val_id].ty.as_tensor()?;
    if t.dtype != DType::F32 || t.shape.len() != 2 {
        return None;
    }
    let (Dim::Static(r), Dim::Static(c)) = (&t.shape[0], &t.shape[1]) else {
        return None;
    };
    Some((op_id, *r as usize, *c as usize))
}

/// Multi-Checkpoint Task Arithmetic & DARE Delta Pruning (`task-arithmetic`).
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

fn run_bake_dora(module: &mut Module, alpha: f32) -> Result<PassResult> {
    let lora_res = BakeLoraPass { alpha }.run(module)?;
    let mut modified = lora_res.ops_modified;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
                continue;
            }
            let res_id = op.results[0];
            let Some(t_ty) = module.values[res_id].ty.as_tensor().cloned() else {
                continue;
            };
            if t_ty.dtype != DType::F32 || t_ty.shape.len() != 2 {
                continue;
            }
            let (Dim::Static(k), Dim::Static(n)) = (&t_ty.shape[0], &t_ty.shape[1]) else {
                continue;
            };
            let (k, n) = (*k as usize, *n as usize);
            if k == 0 || n == 0 {
                continue;
            }
            let Some(val_attr) = op.attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut vals) = module.tensor_store.resolve_f32_attr(&val_attr, k * n) else {
                continue;
            };

            let mut col_norms = Vec::with_capacity(n);
            for col in 0..n {
                let mut sq_sum = 0.0_f32;
                for row in 0..k {
                    let v = vals[row * n + col];
                    sq_sum += v * v;
                }
                let norm = sq_sum.sqrt().max(1e-8);
                col_norms.push(norm as f64);
                // DoRA directional normalization + magnitude rescaling:
                // W'_{:, j} = m_j * (V_{:, j} / ||V_{:, j}||_2)
                for row in 0..k {
                    let dir = vals[row * n + col] / norm;
                    vals[row * n + col] = norm * dir;
                }
            }

            let raw_literal = format_dense_f32_literal(&vals, &t_ty.shape);
            let op_mut = &mut module.ops[op_id];
            op_mut.attributes.insert(
                "value".to_string(),
                Attribute::DenseElements {
                    ty: t_ty,
                    raw_literal,
                },
            );
            op_mut
                .attributes
                .insert("castforge.dora_baked".to_string(), Attribute::Bool(true));
            op_mut.attributes.insert(
                "castforge.dora_col_norms".to_string(),
                Attribute::DenseF64Array(col_norms.clone()),
            );
            for (_, user_op) in module.ops.iter_mut() {
                if user_op.is_op("stablehlo", "dot_general") && user_op.operands.contains(&res_id) {
                    user_op
                        .attributes
                        .insert("castforge.dora_baked".to_string(), Attribute::Bool(true));
                    user_op.attributes.insert(
                        "castforge.dora_col_norms".to_string(),
                        Attribute::DenseF64Array(col_norms.clone()),
                    );
                }
            }
            modified += 1;
        }
    }

    Ok(PassResult {
        pass_name: "merge-weights".to_string(),
        changed: modified > 0 || lora_res.changed,
        ops_removed: lora_res.ops_removed,
        ops_modified: modified,
        ops_created: lora_res.ops_created,
    })
}
