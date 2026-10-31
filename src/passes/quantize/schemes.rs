//! Unified Quantization & Outlier Suppression Passes (`quantize`, `rotate-hadamard`).
//!
//! Consolidates `quantize-weights`, `quantize-groupwise`, `quantize-fp8`, `calibrate-quantize`,
//! **SmoothQuant** (`scheme=smoothquant`), and `rotate-hadamard`.

use super::formats::*;
use crate::ir::{
    Attribute, BlockId, DType, Location, Module, OpName, TensorBacking, TensorEntry, TensorType,
    Type, format_dense_f32_literal, format_dense_i8_literal,
};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Quantization, Precision Lowering & Outlier Rotation Pass (`quantize`).
///
/// Consolidates:
/// - `scheme=per_tensor` (`QuantizeWeightsPass`)
/// - `scheme=groupwise|awq|gptq` (`QuantizeGroupwisePass` with AWQ salient-channel protection)
/// - `scheme=fp8|mxfp8` (`QuantizeFp8Pass`)
/// - `scheme=fp4|mxfp4|nvfp4` (Blackwell NVFP4 / OCP MXFP4 E2M1 4-bit float microscaling)
/// - `scheme=f16|bf16|mixed_precision` (`LowerMixedPrecisionPass`)
/// - `scheme=w8a8|calibrate|static` (`CalibrateQuantizePass`)
/// - `scheme=smoothquant` (per-channel activation-to-weight migration)
/// - `scheme=hadamard|quarot|spinquant` or `hadamard=true` (`RotateHadamardPass`)
pub struct QuantizePass {
    pub scheme: String,
    pub bits: u8,
    pub min_elements: usize,
    pub group_size: usize,
    pub awq_salient: bool,
    pub salient_ratio: f32,
    pub format: String,
    pub mxfp8: bool,
    pub block_size: usize,
    pub samples: usize,
    pub alpha: f32,
    pub hadamard: bool,
}

impl Default for QuantizePass {
    fn default() -> Self {
        Self {
            scheme: "per_tensor".to_string(),
            bits: 8,
            min_elements: 2,
            group_size: 32,
            awq_salient: true,
            salient_ratio: 0.01,
            format: "e4m3".to_string(),
            mxfp8: true,
            block_size: 32,
            samples: 4,
            alpha: 0.5,
            hadamard: false,
        }
    }
}

impl Pass for QuantizePass {
    fn name(&self) -> &'static str {
        "quantize"
    }

    fn description(&self) -> &'static str {
        "Unified quantization & precision (weight|awq|fp8|nvfp4|f16|bf16|w8a8|smoothquant|hadamard)"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "rotate-hadamard",
            "quantize-weights",
            "quantize-groupwise",
            "quantize-fp8",
            "quantize-fp4",
            "calibrate-quantize",
            "smooth-quantize",
            "insert-qat-ste",
            "lower-mixed-precision",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let s = self.scheme.to_ascii_lowercase();
        if matches!(s.as_str(), "f16" | "bf16" | "mixed_precision" | "amp") {
            let target_dtype = if s == "bf16" || self.format.eq_ignore_ascii_case("bf16") {
                DType::BF16
            } else {
                DType::F16
            };
            return Ok(LowerMixedPrecisionPass { target_dtype }
                .run(module)?
                .with_name(self.name()));
        }
        if matches!(s.as_str(), "hadamard" | "quarot" | "spinquant") {
            return Ok(RotateHadamardPass {
                block_size: self.block_size.max(2),
            }
            .run(module)?
            .with_name(self.name()));
        }

        let mut steps = Vec::new();
        if self.hadamard {
            steps.push(
                RotateHadamardPass {
                    block_size: self.block_size.max(2),
                }
                .run(module)?,
            );
        }

        let main_step = match s.as_str() {
            "groupwise" | "awq" | "gptq" => QuantizeGroupwisePass {
                bits: self.bits,
                group_size: self.group_size,
                awq_salient: self.awq_salient || s == "awq",
                salient_ratio: self.salient_ratio,
            }
            .run(module)?,
            "fp8" | "mxfp8" | "e4m3" | "e5m2" => QuantizeFp8Pass {
                format: self.format.clone(),
                mxfp8: self.mxfp8,
                block_size: self.block_size,
            }
            .run(module)?,
            "fp4" | "mxfp4" | "nvfp4" | "e2m1" => QuantizeFp4Pass {
                scheme: s,
                block_size: self.block_size,
            }
            .run(module)?,
            "w8a8" | "calibrate" | "static" => CalibrateQuantizePass {
                bits: self.bits,
                num_samples: self.samples,
            }
            .run(module)?,
            "smoothquant" | "smooth" => SmoothQuantizePass {
                bits: self.bits,
                alpha: self.alpha,
                num_samples: self.samples,
            }
            .run(module)?,
            "qat" | "qat_ste" | "ste" => InsertQatStePass { bits: self.bits }.run(module)?,
            _ => QuantizeWeightsPass {
                bits: self.bits,
                min_elements: self.min_elements,
            }
            .run(module)?,
        };
        steps.push(main_step);
        Ok(PassResult::aggregate(self.name(), &steps))
    }
}

/// SmoothQuant Activation-to-Weight Outlier Migration Pass (`smooth-quantize`).
pub struct SmoothQuantizePass {
    pub bits: u8,
    pub alpha: f32,
    pub num_samples: usize,
}

impl Default for SmoothQuantizePass {
    fn default() -> Self {
        Self {
            bits: 8,
            alpha: 0.5,
            num_samples: 4,
        }
    }
}

impl Pass for SmoothQuantizePass {
    fn name(&self) -> &'static str {
        "smooth-quantize"
    }

    fn description(&self) -> &'static str {
        "Migrates activation outlier difficulty into weights via SmoothQuant alpha scaling"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(
            run_smoothquant(module, self.bits, self.alpha, self.num_samples)?
                .with_name(self.name()),
        )
    }
}

fn run_smoothquant(
    module: &mut Module,
    bits: u8,
    alpha: f32,
    num_samples: usize,
) -> Result<PassResult> {
    let alpha = alpha.clamp(0.05, 0.95);
    let qmax = if bits == 4 { 7.0_f32 } else { 127.0_f32 };
    let ranges = calibrate_value_ranges(module, num_samples.max(1));
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
            let Some(num_elems) = t_ty.num_elements().map(|n| n as usize) else {
                continue;
            };
            if num_elems < 2 {
                continue;
            }
            let Some(val_attr) = op.attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut raw_f32) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
            else {
                continue;
            };

            let act_max = ranges.get(&res_id).copied().unwrap_or(1.0) as f32;
            let w_max = raw_f32
                .iter()
                .fold(0.0_f32, |acc, &v| acc.max(v.abs()))
                .max(1e-5);
            let smooth_scale = (act_max.max(1e-4).powf(alpha) / w_max.powf(1.0 - alpha)).max(1e-4);
            let eff_qscale = (w_max * smooth_scale / qmax).max(1e-6);

            for v in &mut raw_f32 {
                let migrated = *v * smooth_scale;
                let q = (migrated / eff_qscale).round().clamp(-qmax - 1.0, qmax);
                *v = (q * eff_qscale) / smooth_scale;
            }

            let new_val = match val_attr {
                Attribute::TensorRef { key, ty } => {
                    if let Some(entry) = module.tensor_store.get_mut(&key) {
                        entry.backing = TensorBacking::F32Data(raw_f32);
                    }
                    Attribute::TensorRef { key, ty }
                }
                _ => Attribute::DenseElements {
                    ty: t_ty.clone(),
                    raw_literal: format_dense_f32_literal(&raw_f32, &t_ty.shape),
                },
            };

            let op_mut = &mut module.ops[op_id];
            op_mut.attributes.insert("value".to_string(), new_val);
            op_mut.attributes.insert(
                "castforge.quant_scheme".to_string(),
                Attribute::String("smoothquant".to_string()),
            );
            op_mut.attributes.insert(
                "castforge.smoothquant_alpha".to_string(),
                Attribute::Float(alpha as f64),
            );
            op_mut.attributes.insert(
                "castforge.quant_bits".to_string(),
                Attribute::Int(bits as i64),
            );

            for (_, user_op) in module.ops.iter_mut() {
                if user_op.operands.contains(&res_id) {
                    user_op.attributes.insert(
                        "castforge.quant_scheme".to_string(),
                        Attribute::String("smoothquant".to_string()),
                    );
                    user_op.attributes.insert(
                        "castforge.smoothquant_alpha".to_string(),
                        Attribute::Float(alpha as f64),
                    );
                    user_op.attributes.insert(
                        "castforge.quant_bits".to_string(),
                        Attribute::Int(bits as i64),
                    );
                }
            }
            modified += 1;
        }
    }

    Ok(PassResult {
        pass_name: "quantize".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

/// Weight Quantization Pass (`quantize-weights`).
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

                let Some(val_attr) = op.attributes.get("value").cloned() else {
                    continue;
                };
                let resource_key = match &val_attr {
                    Attribute::TensorRef { key, .. } => Some(key.clone()),
                    _ => None,
                };
                let Some(f32_vals) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems)
                else {
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

                let store_key = resource_key.clone().unwrap_or_else(|| {
                    module.values[const_val_id]
                        .ssa_name
                        .clone()
                        .unwrap_or_else(|| format!("q_weight_{modified}"))
                });
                module.tensor_store.insert(TensorEntry {
                    name: store_key,
                    ty: q_tensor_ty.clone(),
                    backing: TensorBacking::QuantizedI8 {
                        data: q_data,
                        scale,
                        zero_point: zp,
                    },
                });

                let q_ty = Type::Tensor(q_tensor_ty.clone());
                module.values[const_val_id].ty = q_ty;
                let updated_attr = if let Some(key) = resource_key {
                    Attribute::TensorRef {
                        key,
                        ty: q_tensor_ty,
                    }
                } else {
                    Attribute::DenseElements {
                        ty: q_tensor_ty,
                        raw_literal: q_literal,
                    }
                };
                module.ops[op_id]
                    .attributes
                    .insert("value".to_string(), updated_attr);
                modified += 1;

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

                let block_ops = &mut module.blocks[block_id].ops;
                if let Some(last) = block_ops.pop() {
                    if let Some(pos) = block_ops.iter().position(|&id| id == op_id) {
                        block_ops.insert(pos + 1, last);
                    } else {
                        block_ops.push(last);
                    }
                }

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

/// Group-Wise / Activation-Aware Weight Quantization Pass (`quantize-groupwise`).
pub struct QuantizeGroupwisePass {
    pub bits: u8,
    pub group_size: usize,
    pub awq_salient: bool,
    pub salient_ratio: f32,
}

impl Default for QuantizeGroupwisePass {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            awq_salient: true,
            salient_ratio: 0.01,
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
        let ratio = self.salient_ratio.clamp(0.001, 0.5);

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
                let mut sorted_abs: Vec<f32> = raw_f32.iter().map(|v| v.abs()).collect();
                sorted_abs.sort_by(|a, b| b.total_cmp(a));
                let salient_count = ((num_elems as f32 * ratio).ceil() as usize)
                    .max(1)
                    .min(num_elems);
                let salient_thresh = sorted_abs
                    .get(salient_count.saturating_sub(1))
                    .copied()
                    .unwrap_or(0.0);

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

                let scheme_str = if self.awq_salient {
                    "awq_groupwise"
                } else {
                    "gptq_groupwise"
                };
                if op
                    .attributes
                    .get("castforge.quant_scheme")
                    .and_then(|a| a.as_str())
                    == Some(scheme_str)
                {
                    continue;
                }
                let ssa = module.values[res_id].ssa_name.clone();
                let new_val = module.tensor_store.update_f32_constant(
                    ssa.as_deref(),
                    &val_attr,
                    t_ty,
                    deq_f32,
                );
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
                if self.awq_salient {
                    op_mut.attributes.insert(
                        "castforge.awq_salient_channels".to_string(),
                        Attribute::Int(salient_count as i64),
                    );
                }

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
                        if self.awq_salient {
                            user_op.attributes.insert(
                                "castforge.awq_salient_channels".to_string(),
                                Attribute::Int(salient_count as i64),
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
                if !op.is_op("stablehlo", "constant")
                    || op.results.is_empty()
                    || op
                        .attributes
                        .get("castforge.fp8_format")
                        .and_then(|a| a.as_str())
                        == Some(fmt_str)
                {
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

                let ssa = module.values[res_id].ssa_name.clone();
                let new_val = module.tensor_store.update_f32_constant(
                    ssa.as_deref(),
                    &val_attr,
                    t_ty,
                    deq_f32,
                );
                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert("value".to_string(), new_val);
                op_mut.attributes.insert(
                    "castforge.fp8_format".to_string(),
                    Attribute::String(fmt_str.to_string()),
                );
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
