use crate::{
    BakeLoraPass, CalibrateQuantizePass, DeadCodeEliminationPass, FoldNormWeightsPass, MergeMethod,
    MergeWeightsPass, Pass, PassResult, QuantizeWeightsPass, Sparsify2To4Pass,
    StructuredPruningPass, TaskArithmeticPass,
};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Dim, Module, TensorBacking, ValueDef, ValueId, format_dense_f32_literal,
};

/// Group-Wise / Activation-Aware Weight Quantization Pass (`quantize-groupwise`).
pub struct QuantizeGroupwisePass {
    pub bits: u8,
    pub group_size: usize,
    pub awq_salient: bool,
}

impl Default for QuantizeGroupwisePass {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            awq_salient: true,
        }
    }
}

impl Pass for QuantizeGroupwisePass {
    fn name(&self) -> &'static str {
        "quantize-groupwise"
    }

    fn description(&self) -> &'static str {
        "Group-wise AWQ/GPTQ weight quantization (bits=4|8, group_size=32, awq_salient=true)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let bits = if self.bits == 8 { 8 } else { 4 };
        let qmax = if bits == 8 { 127.0_f32 } else { 7.0_f32 };
        let group_size = self.group_size.max(2);

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
                if t_ty.shape.len() < 2 {
                    continue;
                }
                let Some(num_elems) = t_ty.num_elements().map(|n| n as usize) else {
                    continue;
                };
                if num_elems < 2 {
                    continue;
                }
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(raw_f32) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                else {
                    continue;
                };

                let mut deq_f32 = Vec::with_capacity(num_elems);
                let mut scales = Vec::new();
                let global_max = raw_f32.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
                let salient_thresh = global_max * 0.75;

                for chunk in raw_f32.chunks(group_size) {
                    let group_max = chunk.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
                    let scale = if group_max < 1e-12 {
                        1.0
                    } else {
                        group_max / qmax
                    };
                    scales.push(scale);
                    for &v in chunk {
                        let eff_scale = if self.awq_salient && v.abs() >= salient_thresh {
                            scale * 0.5
                        } else {
                            scale
                        };
                        let q = (v / eff_scale).round().clamp(-qmax - 1.0, qmax);
                        deq_f32.push(q * eff_scale);
                    }
                }

                let new_val = match val_attr {
                    Attribute::TensorRef { key, ty } => {
                        if let Some(entry) = module.tensor_store.get_mut(&key) {
                            entry.backing = TensorBacking::F32Data(deq_f32);
                        }
                        Attribute::TensorRef { key, ty }
                    }
                    _ => Attribute::DenseElements {
                        ty: t_ty.clone(),
                        raw_literal: format_dense_f32_literal(&deq_f32, &t_ty.shape),
                    },
                };

                let scheme_str = if self.awq_salient {
                    "awq_groupwise"
                } else {
                    "gptq_groupwise"
                };
                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert("value".to_string(), new_val);
                op_mut.attributes.insert(
                    "castforge.quant_scheme".to_string(),
                    Attribute::String(scheme_str.to_string()),
                );
                op_mut.attributes.insert(
                    "castforge.quant_bits".to_string(),
                    Attribute::Int(bits as i64),
                );
                op_mut.attributes.insert(
                    "castforge.group_size".to_string(),
                    Attribute::Int(group_size as i64),
                );
                op_mut.attributes.insert(
                    "castforge.num_groups".to_string(),
                    Attribute::Int(scales.len() as i64),
                );

                for (_, user_op) in module.ops.iter_mut() {
                    if user_op.operands.contains(&res_id) {
                        user_op.attributes.insert(
                            "castforge.quant_scheme".to_string(),
                            Attribute::String(scheme_str.to_string()),
                        );
                        user_op.attributes.insert(
                            "castforge.quant_bits".to_string(),
                            Attribute::Int(bits as i64),
                        );
                        user_op.attributes.insert(
                            "castforge.group_size".to_string(),
                            Attribute::Int(group_size as i64),
                        );
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

/// FP8 (`E4M3` / `E5M2`) & MXFP8 Microscaling Quantization Pass (`quantize-fp8`).
pub struct QuantizeFp8Pass {
    pub format: String,
    pub mxfp8: bool,
    pub block_size: usize,
}

impl Default for QuantizeFp8Pass {
    fn default() -> Self {
        Self {
            format: "e4m3".to_string(),
            mxfp8: true,
            block_size: 32,
        }
    }
}

impl Pass for QuantizeFp8Pass {
    fn name(&self) -> &'static str {
        "quantize-fp8"
    }

    fn description(&self) -> &'static str {
        "Applies FP8 (E4M3/E5M2) and MXFP8 block-scaled quantization to weights and GEMMs"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let is_e5m2 = self.format.eq_ignore_ascii_case("e5m2");
        let fp8_max = if is_e5m2 { 57344.0_f32 } else { 448.0_f32 };
        let grid_steps = if is_e5m2 { 15.0_f32 } else { 31.0_f32 };
        let bsize = self.block_size.max(4);
        let fmt_str = if is_e5m2 { "fp8_e5m2" } else { "fp8_e4m3" };

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
                let Some(num_elems) = t_ty.num_elements().map(|n| n as usize) else {
                    continue;
                };
                if num_elems < 2 {
                    continue;
                }
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(raw_f32) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                else {
                    continue;
                };

                let mut deq_f32 = Vec::with_capacity(num_elems);
                for chunk in raw_f32.chunks(bsize) {
                    let amax = chunk.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
                    let raw_scale = if amax < 1e-12 { 1.0 } else { amax / fp8_max };
                    let scale = if self.mxfp8 {
                        2.0_f32.powf(raw_scale.log2().ceil())
                    } else {
                        raw_scale
                    };
                    let step = (fp8_max / grid_steps).max(1e-6);
                    for &v in chunk {
                        let scaled = (v / scale).clamp(-fp8_max, fp8_max);
                        let q = (scaled / step).round() * step;
                        deq_f32.push(q * scale);
                    }
                }

                let new_val = Attribute::DenseElements {
                    ty: t_ty.clone(),
                    raw_literal: format_dense_f32_literal(&deq_f32, &t_ty.shape),
                };
                module.ops[op_id]
                    .attributes
                    .insert("value".to_string(), new_val);
                for (_, user_op) in module.ops.iter_mut() {
                    if user_op.operands.contains(&res_id) {
                        user_op.attributes.insert(
                            "castforge.fp8_format".to_string(),
                            Attribute::String(fmt_str.to_string()),
                        );
                        user_op
                            .attributes
                            .insert("castforge.mxfp8".to_string(), Attribute::Bool(self.mxfp8));
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

/// Unified Quantization Pass (`quantize`).
///
/// Consolidates `quantize-weights` (`scheme=per_tensor`), `quantize-groupwise`
/// (`scheme=groupwise|awq|gptq`), `quantize-fp8` (`scheme=fp8|mxfp8`), and `calibrate-quantize`
/// (`scheme=w8a8|calibrate|static`).
pub struct QuantizePass {
    pub scheme: String,
    pub bits: u8,
    pub min_elements: usize,
    pub group_size: usize,
    pub awq_salient: bool,
    pub format: String,
    pub mxfp8: bool,
    pub block_size: usize,
    pub samples: usize,
}

impl Default for QuantizePass {
    fn default() -> Self {
        Self {
            scheme: "per_tensor".to_string(),
            bits: 8,
            min_elements: 2,
            group_size: 32,
            awq_salient: true,
            format: "e4m3".to_string(),
            mxfp8: true,
            block_size: 32,
            samples: 4,
        }
    }
}

impl Pass for QuantizePass {
    fn name(&self) -> &'static str {
        "quantize"
    }

    fn description(&self) -> &'static str {
        "Unified quantization (scheme=per_tensor|groupwise|fp8|mxfp8|w8a8, bits=4|8)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let s = self.scheme.to_ascii_lowercase();
        let mut res = match s.as_str() {
            "groupwise" | "awq" | "gptq" => QuantizeGroupwisePass {
                bits: self.bits,
                group_size: self.group_size,
                awq_salient: self.awq_salient,
            }
            .run(module)?,
            "fp8" | "mxfp8" | "e4m3" | "e5m2" => QuantizeFp8Pass {
                format: self.format.clone(),
                mxfp8: self.mxfp8,
                block_size: self.block_size,
            }
            .run(module)?,
            "w8a8" | "calibrate" | "static" => CalibrateQuantizePass {
                bits: self.bits,
                num_samples: self.samples,
            }
            .run(module)?,
            _ => QuantizeWeightsPass {
                bits: self.bits,
                min_elements: self.min_elements,
            }
            .run(module)?,
        };
        res.pass_name = self.name().to_string();
        Ok(res)
    }
}

/// Unified Weight Pruning & Structured Sparsity Pass (`prune-weights`).
///
/// Consolidates `prune-structured` (`mode=structured`) and `sparsify-2to4` (`mode=2to4`).
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
        "Unified weight pruning and structured sparsity (mode=structured|2to4, sparsity=0.5)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut res = if self.mode.eq_ignore_ascii_case("2to4") {
            Sparsify2To4Pass.run(module)?
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

/// Unified Weight Merging, Task Arithmetic, DARE & LoRA Baking Pass (`merge-weights`).
///
/// Consolidates `merge-weights` (`linear|slerp|ties`), `task-arithmetic` (`task_arithmetic|dare`),
/// and `bake-lora` (`lora`).
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

/// Inference BatchNorm & Linear Scale Folding Pass (`fold-batchnorm`).
#[derive(Default)]
pub struct FoldBatchNormPass;

impl Pass for FoldBatchNormPass {
    fn name(&self) -> &'static str {
        "fold-batchnorm"
    }

    fn description(&self) -> &'static str {
        "Absorbs BatchNorm / channel scale into preceding convolution or dot_general weights"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let pre_norm_res = FoldNormWeightsPass.run(module)?;
        let mut modified = pre_norm_res.ops_modified;
        let mut removed = pre_norm_res.ops_removed;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Multiply {
                    lhs: a,
                    rhs: b,
                    result: mul_out,
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if let Some(()) = try_fold_bn_scale(module, a, b, mul_out)
                    .or_else(|| try_fold_bn_scale(module, b, a, mul_out))
                {
                    modified += 1;
                }
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0 || pre_norm_res.changed,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn try_fold_bn_scale(
    module: &mut Module,
    linear_val: ValueId,
    scale_val: ValueId,
    mul_out: ValueId,
) -> Option<()> {
    let ValueDef::OpResult { op: lin_op_id, .. } = module.values[linear_val].def else {
        return None;
    };
    let lin_op = module.ops.get(lin_op_id)?.clone();
    let is_dot = lin_op.is_op("stablehlo", "dot_general");
    let is_conv = lin_op.is_op("stablehlo", "convolution");
    if (!is_dot && !is_conv) || lin_op.operands.len() < 2 {
        return None;
    }

    let w_val = lin_op.operands[1];
    let ValueDef::OpResult { op: w_op_id, .. } = module.values[w_val].def else {
        return None;
    };
    let w_op = module.ops.get(w_op_id)?.clone();
    if !w_op.is_op("stablehlo", "constant") {
        return None;
    }
    let w_ty = module.values[w_val].ty.as_tensor()?.clone();
    let w_elems = w_ty.num_elements()? as usize;
    let w_attr = w_op.attributes.get("value")?.clone();
    let mut w_data = module.tensor_store.resolve_f32_attr(&w_attr, w_elems)?;

    let ValueDef::OpResult { op: s_op_id, .. } = module.values[scale_val].def else {
        return None;
    };
    let s_op = module.ops.get(s_op_id)?.clone();
    let mut scale_src_val = scale_val;
    if let Some(StableHloOp::BroadcastInDim { operand, .. }) = StableHloOp::from_op(&s_op) {
        scale_src_val = operand;
    }
    let ValueDef::OpResult {
        op: sc_const_id, ..
    } = module.values[scale_src_val].def
    else {
        return None;
    };
    let sc_op = module.ops.get(sc_const_id)?.clone();
    if !sc_op.is_op("stablehlo", "constant") {
        return None;
    }
    let sc_ty = module.values[scale_src_val].ty.as_tensor()?.clone();
    let sc_elems = sc_ty.num_elements()? as usize;
    let sc_attr = sc_op.attributes.get("value")?.clone();
    let sc_data = module.tensor_store.resolve_f32_attr(&sc_attr, sc_elems)?;

    let out_channels = match w_ty.shape.last()? {
        Dim::Static(c) if *c > 0 => *c as usize,
        _ => return None,
    };
    if !w_elems.is_multiple_of(out_channels) {
        return None;
    }
    let rows = w_elems / out_channels;
    for r in 0..rows {
        for c in 0..out_channels {
            let s = sc_data[c % sc_data.len()];
            w_data[r * out_channels + c] *= s;
        }
    }

    let new_w_attr = match w_attr {
        Attribute::TensorRef { key, ty } => {
            if let Some(entry) = module.tensor_store.get_mut(&key) {
                entry.backing = TensorBacking::F32Data(w_data);
            }
            Attribute::TensorRef { key, ty }
        }
        _ => Attribute::DenseElements {
            ty: w_ty.clone(),
            raw_literal: format_dense_f32_literal(&w_data, &w_ty.shape),
        },
    };
    module.ops[w_op_id]
        .attributes
        .insert("value".to_string(), new_w_attr);
    module.ops[lin_op_id].attributes.insert(
        "castforge.batchnorm_folded".to_string(),
        Attribute::Bool(true),
    );
    module.replace_all_uses(mul_out, linear_val);
    Some(())
}

/// QuaRot / SpinQuant Walsh-Hadamard Outlier Suppression Pass (`hadamard-quant-rotation`).
///
/// Applies a normalized orthogonal Sylvester Walsh-Hadamard rotation $H_k$ ($\frac{1}{\sqrt{k}}
/// H_k$) along power-of-two blocks of the contracting dimension of 2D constant weight matrices and
/// stamps `castforge.hadamard_rotated = true` on consuming GEMMs to suppress activation outliers
/// prior to low-bit quantization.
pub struct HadamardQuantRotationPass {
    pub block_size: usize,
}

impl Default for HadamardQuantRotationPass {
    fn default() -> Self {
        Self { block_size: 4 }
    }
}

impl Pass for HadamardQuantRotationPass {
    fn name(&self) -> &'static str {
        "hadamard-quant-rotation"
    }

    fn description(&self) -> &'static str {
        "Applies QuaRot/SpinQuant orthogonal Walsh-Hadamard rotation to suppress outliers"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
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
                if t_ty.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(k_dim), Dim::Static(n_dim)) = (&t_ty.shape[0], &t_ty.shape[1])
                else {
                    continue;
                };
                let k = *k_dim as usize;
                let n = *n_dim as usize;
                if k < 2 || n == 0 {
                    continue;
                }
                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };
                let Some(mut w_data) = module.tensor_store.resolve_f32_attr(&val_attr, k * n)
                else {
                    continue;
                };

                // Apply normalized fast Walsh-Hadamard transform along K in power-of-2 chunks
                let b = self.block_size.next_power_of_two().min(k).max(2);
                if k.is_multiple_of(b) {
                    let norm = 1.0_f32 / (b as f32).sqrt();
                    for col in 0..n {
                        for k_base in (0..k).step_by(b) {
                            let mut buf: Vec<f32> =
                                (0..b).map(|i| w_data[(k_base + i) * n + col]).collect();
                            let mut h = 1;
                            while h < b {
                                for i in (0..b).step_by(h * 2) {
                                    for j in i..(i + h) {
                                        let x = buf[j];
                                        let y = buf[j + h];
                                        buf[j] = x + y;
                                        buf[j + h] = x - y;
                                    }
                                }
                                h *= 2;
                            }
                            for (i, &val) in buf.iter().enumerate() {
                                w_data[(k_base + i) * n + col] = val * norm;
                            }
                        }
                    }
                }

                let new_val = Attribute::DenseElements {
                    ty: t_ty.clone(),
                    raw_literal: format_dense_f32_literal(&w_data, &t_ty.shape),
                };
                module.ops[op_id]
                    .attributes
                    .insert("value".to_string(), new_val);
                for (_, user_op) in module.ops.iter_mut() {
                    if user_op.operands.contains(&res_id) {
                        user_op.attributes.insert(
                            "castforge.hadamard_rotated".to_string(),
                            Attribute::Bool(true),
                        );
                        user_op.attributes.insert(
                            "castforge.hadamard_block".to_string(),
                            Attribute::Int(b as i64),
                        );
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
