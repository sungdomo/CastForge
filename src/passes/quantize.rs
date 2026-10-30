//! Unified Quantization & Outlier Suppression Passes (`quantize`, `hadamard-quant-rotation`).
//!
//! Consolidates `quantize-weights`, `quantize-groupwise`, `quantize-fp8`, `calibrate-quantize`,
//! **SmoothQuant** (`scheme=smoothquant`), and `hadamard-quant-rotation`.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorBacking,
    TensorEntry, TensorType, Type, Value, ValueDef, ValueId, format_dense_f32_literal,
    format_dense_i8_literal,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Unified Quantization, Precision Lowering & Outlier Rotation Pass (`quantize`).
///
/// Consolidates:
/// - `scheme=per_tensor` (`QuantizeWeightsPass`)
/// - `scheme=groupwise|awq|gptq` (`QuantizeGroupwisePass` with AWQ salient-channel protection)
/// - `scheme=fp8|mxfp8` (`QuantizeFp8Pass`)
/// - `scheme=fp4|mxfp4|nvfp4` (Blackwell NVFP4 / OCP MXFP4 E2M1 4-bit float microscaling)
/// - `scheme=f16|bf16|mixed_precision` (`MixedPrecisionPass`)
/// - `scheme=w8a8|calibrate|static` (`CalibrateQuantizePass`)
/// - `scheme=smoothquant` (per-channel activation-to-weight migration)
/// - `scheme=hadamard|quarot|spinquant` or `hadamard=true` (`HadamardQuantRotationPass`)
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

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let s = self.scheme.to_ascii_lowercase();
        if matches!(s.as_str(), "f16" | "bf16" | "mixed_precision" | "amp") {
            let target_dtype = if s == "bf16" || self.format.eq_ignore_ascii_case("bf16") {
                DType::BF16
            } else {
                DType::F16
            };
            let mut res = MixedPrecisionPass { target_dtype }.run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }
        if matches!(s.as_str(), "hadamard" | "quarot" | "spinquant") {
            let mut res = HadamardQuantRotationPass {
                block_size: self.block_size.max(2),
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }

        let mut rot_modified = 0usize;
        if self.hadamard {
            let rot = HadamardQuantRotationPass {
                block_size: self.block_size.max(2),
            }
            .run(module)?;
            rot_modified = rot.ops_modified;
        }

        let mut res = match s.as_str() {
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
            "fp4" | "mxfp4" | "nvfp4" | "e2m1" => run_fp4_quantize(module, &s, self.block_size)?,
            "w8a8" | "calibrate" | "static" => CalibrateQuantizePass {
                bits: self.bits,
                num_samples: self.samples,
            }
            .run(module)?,
            "smoothquant" | "smooth" => {
                run_smoothquant(module, self.bits, self.alpha, self.samples)?
            }
            "qat" | "qat_ste" | "ste" => run_qat_ste(module, self.bits)?,
            _ => QuantizeWeightsPass {
                bits: self.bits,
                min_elements: self.min_elements,
            }
            .run(module)?,
        };
        res.pass_name = self.name().to_string();
        res.changed |= rot_modified > 0;
        res.ops_modified += rot_modified;
        Ok(res)
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

/// Post-Training Static Activation Calibration & W8A8 Quantization Pass (`calibrate-quantize`).
pub struct CalibrateQuantizePass {
    pub bits: u8,
    pub num_samples: usize,
}

impl Default for CalibrateQuantizePass {
    fn default() -> Self {
        Self {
            bits: 8,
            num_samples: 4,
        }
    }
}

impl Pass for CalibrateQuantizePass {
    fn name(&self) -> &'static str {
        "calibrate-quantize"
    }

    fn description(&self) -> &'static str {
        "Calibrates activation dynamic ranges and inserts static W8A8 quantization scales"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let qmax = if self.bits == 4 { 7.0_f64 } else { 127.0_f64 };
        let ranges = calibrate_value_ranges(module, self.num_samples.max(1));
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.attributes.contains_key("quant_mode") {
                    continue;
                }
                let is_target = matches!(
                    StableHloOp::from_op(&op),
                    Some(StableHloOp::DotGeneral { .. } | StableHloOp::Convolution { .. })
                );
                if !is_target || op.operands.len() < 2 {
                    continue;
                }
                let lhs = op.operands[0];
                let rhs = op.operands[1];
                let Some(lhs_t) = module.values[lhs].ty.as_tensor() else {
                    continue;
                };
                if !matches!(lhs_t.dtype, DType::F32 | DType::F16 | DType::BF16) {
                    continue;
                }

                let lhs_max = ranges.get(&lhs).copied().unwrap_or(1.0).max(1e-5);
                let rhs_max = ranges.get(&rhs).copied().unwrap_or(1.0).max(1e-5);
                let lhs_scale = lhs_max / qmax;
                let rhs_scale = rhs_max / qmax;

                let lhs_q = insert_static_quantize_before(
                    module,
                    block_id,
                    op_id,
                    lhs,
                    lhs_scale,
                    self.bits as i64,
                );
                let op_mut = &mut module.ops[op_id];
                op_mut.operands[0] = lhs_q;
                op_mut.attributes.insert(
                    "quant_mode".to_string(),
                    Attribute::String(format!("w{}a{}_static", self.bits, self.bits)),
                );
                op_mut
                    .attributes
                    .insert("lhs_scale".to_string(), Attribute::Float(lhs_scale));
                op_mut
                    .attributes
                    .insert("rhs_scale".to_string(), Attribute::Float(rhs_scale));
                op_mut.attributes.insert(
                    "calibration_samples".to_string(),
                    Attribute::Int(self.num_samples as i64),
                );
                created += 1;
                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn calibrate_value_ranges(module: &Module, num_samples: usize) -> HashMap<ValueId, f64> {
    let mut max_abs: HashMap<ValueId, f64> = HashMap::new();
    let sample_scales: Vec<f64> = (0..num_samples)
        .map(|s| 0.5 + (s as f64) / (num_samples.max(1) as f64))
        .collect();

    for (vid, v) in &module.values {
        if matches!(v.def, ValueDef::BlockArgument { .. }) {
            let peak = sample_scales.iter().copied().fold(0.8_f64, f64::max);
            max_abs.insert(vid, peak);
        }
    }

    for (_, block) in &module.blocks {
        for &op_id in &block.ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let Some(&res) = op.results.first() else {
                continue;
            };
            if op.is_op("stablehlo", "constant") {
                let n = module.values[res]
                    .ty
                    .as_tensor()
                    .and_then(|t| t.num_elements())
                    .unwrap_or(1) as usize;
                let peak = op
                    .attributes
                    .get("value")
                    .and_then(|a| module.tensor_store.resolve_f32_attr(a, n))
                    .map(|vals| {
                        vals.into_iter()
                            .map(|x| (x as f64).abs())
                            .fold(1e-5_f64, f64::max)
                    })
                    .unwrap_or(1.0);
                max_abs.insert(res, peak);
                continue;
            }
            let in0 = op
                .operands
                .first()
                .and_then(|v| max_abs.get(v).copied())
                .unwrap_or(1.0);
            let in1 = op
                .operands
                .get(1)
                .and_then(|v| max_abs.get(v).copied())
                .unwrap_or(1.0);
            let est = match StableHloOp::from_op(op) {
                Some(StableHloOp::Add { .. } | StableHloOp::Subtract { .. }) => in0 + in1,
                Some(StableHloOp::Multiply { .. }) => in0 * in1,
                Some(StableHloOp::DotGeneral { lhs, .. }) => {
                    let k = module.values[lhs]
                        .ty
                        .as_tensor()
                        .and_then(|t| match t.shape.last() {
                            Some(Dim::Static(d)) => Some((*d).max(1) as f64),
                            _ => None,
                        })
                        .unwrap_or(4.0);
                    in0 * in1 * k.sqrt()
                }
                Some(StableHloOp::Tanh { .. }) => 1.0,
                _ => in0,
            };
            max_abs.insert(res, est.max(1e-5));
        }
    }

    max_abs
}

fn insert_static_quantize_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input: ValueId,
    scale: f64,
    bits: i64,
) -> ValueId {
    let out_ty: Type = module.values[input].ty.clone();
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "composite"),
        operands: SmallVec::from_slice(&[input]),
        results: SmallVec::new(),
        attributes: {
            let mut m = IndexMap::new();
            m.insert(
                "name".to_string(),
                Attribute::String("castforge.static_quantize".to_string()),
            );
            m.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("static_quantize_impl".to_string()),
            );
            m.insert("scale".to_string(), Attribute::Float(scale));
            m.insert("bits".to_string(), Attribute::Int(bits));
            m
        },
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let res_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(res_val);
    let block_ops = &mut module.blocks[block_id].ops;
    let pos = block_ops
        .iter()
        .position(|&x| x == before_op)
        .unwrap_or(block_ops.len());
    block_ops.insert(pos, new_op_id);
    res_val
}

/// QuaRot / SpinQuant Walsh-Hadamard Outlier Suppression Pass (`hadamard-quant-rotation`).
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

/// Snap a normalized value in `[-6.0, 6.0]` to the exact E2M1 4-bit float grid:
/// `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.
fn snap_e2m1(val: f32) -> f32 {
    const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if val < 0.0 { -1.0 } else { 1.0 };
    let abs_v = val.abs();
    let mut best = 0.0_f32;
    let mut best_err = f32::MAX;
    for &lvl in &E2M1_LEVELS {
        let err = (abs_v - lvl).abs();
        if err < best_err {
            best_err = err;
            best = lvl;
        }
    }
    sign * best
}

fn run_fp4_quantize(module: &mut Module, scheme: &str, block_size: usize) -> Result<PassResult> {
    let blk = if scheme == "nvfp4" && block_size == 32 {
        16
    } else {
        block_size.max(4)
    };
    let scheme_label = if scheme == "nvfp4" { "nvfp4" } else { "mxfp4" };
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "constant") {
                continue;
            }
            let Some(&res_id) = op.results.first() else {
                continue;
            };
            let Some(t_ty) = module.values[res_id].ty.as_tensor().cloned() else {
                continue;
            };
            if t_ty.dtype != DType::F32 {
                continue;
            }
            let Some(numel) = t_ty.num_elements().filter(|&n| n >= 2) else {
                continue;
            };
            let Some(val_attr) = op.attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut data) = module
                .tensor_store
                .resolve_f32_attr(&val_attr, numel as usize)
            else {
                continue;
            };

            let mut num_blocks = 0i64;
            for chunk in data.chunks_mut(blk) {
                let amax = chunk.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
                let scale = if amax > 1e-8 { amax / 6.0 } else { 1.0 };
                num_blocks += 1;
                for v in chunk.iter_mut() {
                    *v = snap_e2m1(*v / scale) * scale;
                }
            }

            let new_val = Attribute::DenseElements {
                ty: t_ty.clone(),
                raw_literal: format_dense_f32_literal(&data, &t_ty.shape),
            };
            let op_mut = &mut module.ops[op_id];
            op_mut.attributes.insert("value".to_string(), new_val);
            op_mut.attributes.insert(
                "castforge.quant_scheme".to_string(),
                Attribute::String(scheme_label.to_string()),
            );
            op_mut.attributes.insert(
                "castforge.fp4_block_size".to_string(),
                Attribute::Int(blk as i64),
            );
            op_mut.attributes.insert(
                "castforge.fp4_num_blocks".to_string(),
                Attribute::Int(num_blocks),
            );
            for (_, user_op) in module.ops.iter_mut() {
                if user_op.operands.contains(&res_id) {
                    user_op.attributes.insert(
                        "castforge.quant_scheme".to_string(),
                        Attribute::String(scheme_label.to_string()),
                    );
                    user_op.attributes.insert(
                        "castforge.fp4_block_size".to_string(),
                        Attribute::Int(blk as i64),
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

/// Automatic Mixed Precision Pass (`mixed-precision`).
pub struct MixedPrecisionPass {
    pub target_dtype: DType,
}

impl Default for MixedPrecisionPass {
    fn default() -> Self {
        Self {
            target_dtype: DType::F16,
        }
    }
}

impl Pass for MixedPrecisionPass {
    fn name(&self) -> &'static str {
        "mixed-precision"
    }

    fn description(&self) -> &'static str {
        "Applies Automatic Mixed Precision (f16/bf16) casts to GEMM/Conv ops, keeping norms in f32"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let low_dtype = match &self.target_dtype {
            DType::BF16 => DType::BF16,
            _ => DType::F16,
        };
        let prec_label = match &low_dtype {
            DType::BF16 => "bf16",
            _ => "f16",
        };
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.attributes.contains_key("compute_precision") {
                    continue;
                }
                let is_compute_heavy = matches!(
                    StableHloOp::from_op(&op),
                    Some(StableHloOp::DotGeneral { .. } | StableHloOp::Convolution { .. })
                );
                if !is_compute_heavy || op.operands.len() < 2 {
                    continue;
                }

                let mut new_operands = op.operands.clone();
                for opnd_slot in new_operands.iter_mut().take(2) {
                    let orig_val = *opnd_slot;
                    let Some(t_ty) = module.values[orig_val].ty.as_tensor().cloned() else {
                        continue;
                    };
                    if t_ty.dtype != DType::F32 {
                        continue;
                    }
                    let low_ty = Type::Tensor(TensorType {
                        shape: t_ty.shape.clone(),
                        dtype: low_dtype.clone(),
                        encoding: None,
                    });
                    let f32_ty = Type::Tensor(t_ty);

                    let down_val = insert_convert_before(module, block_id, op_id, orig_val, low_ty);
                    let up_val = insert_convert_before(module, block_id, op_id, down_val, f32_ty);
                    *opnd_slot = up_val;
                    created += 2;
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.operands = new_operands;
                op_mut.attributes.insert(
                    "compute_precision".to_string(),
                    Attribute::String(prec_label.to_string()),
                );
                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn insert_convert_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input_val: ValueId,
    out_ty: Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert("amp_cast".to_string(), Attribute::Bool(true));
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "convert"),
        operands: SmallVec::from_slice(&[input_val]),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(val_id);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, new_op_id);
    } else {
        block_ops.push(new_op_id);
    }
    val_id
}

/// Symmetric per-tensor quantization of `f32` weights to `i8` or `i4`.
/// Returns `(quantized_values, scale, zero_point, quantized_dtype)`.
pub fn quantize_symmetric_f32(
    values: &[f32],
    bits: u8,
    expressed_dtype: DType,
) -> (Vec<i8>, f32, i64, DType) {
    let qmax = if bits == 4 { 7.0f32 } else { 127.0f32 };
    let qmin = if bits == 4 { -8.0f32 } else { -128.0f32 };

    let max_abs = values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / qmax };

    let q_data: Vec<i8> = values
        .iter()
        .map(|&v| (v / scale).round().clamp(qmin, qmax) as i8)
        .collect();

    let storage_type = if bits == 4 { DType::I4 } else { DType::I8 };
    let scale_str = format!("{:.6}", scale)
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string();
    let scale_str = if scale_str.contains('.') {
        scale_str
    } else {
        format!("{scale_str}.0")
    };

    let q_dtype = DType::QuantUniform {
        storage_type: Box::new(storage_type),
        expressed_type: Box::new(expressed_dtype),
        scale: scale_str,
        zero_point: 0,
    };

    (q_data, scale, 0, q_dtype)
}

fn run_qat_ste(module: &mut Module, bits: u8) -> Result<PassResult> {
    let eff_bits = if bits == 4 { 4 } else { 8 };
    let qmax = if eff_bits == 4 { 7.0_f32 } else { 127.0_f32 };
    let qmin = if eff_bits == 4 { -8.0_f32 } else { -128.0_f32 };
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
            if t_ty.dtype != DType::F32 || num_elems < 2 {
                continue;
            }
            let Some(val_attr) = op.attributes.get("value").cloned() else {
                continue;
            };
            let Some(vals) = module.tensor_store.resolve_f32_attr(&val_attr, num_elems) else {
                continue;
            };

            let max_abs = vals.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
            let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / qmax };
            let fake_quant: Vec<f32> = vals
                .iter()
                .map(|&v| (v / scale).round().clamp(qmin, qmax) * scale)
                .collect();

            let raw_literal = format_dense_f32_literal(&fake_quant, &t_ty.shape);
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
                .insert("castforge.qat_ste".to_string(), Attribute::Bool(true));
            op_mut.attributes.insert(
                "castforge.qat_bits".to_string(),
                Attribute::Int(eff_bits as i64),
            );
            op_mut.attributes.insert(
                "castforge.qat_scale".to_string(),
                Attribute::Float(scale as f64),
            );
            for (_, user_op) in module.ops.iter_mut() {
                if user_op.is_op("stablehlo", "dot_general") && user_op.operands.contains(&res_id) {
                    user_op
                        .attributes
                        .insert("castforge.qat_ste".to_string(), Attribute::Bool(true));
                    user_op.attributes.insert(
                        "castforge.qat_bits".to_string(),
                        Attribute::Int(eff_bits as i64),
                    );
                    user_op.attributes.insert(
                        "castforge.qat_scale".to_string(),
                        Attribute::Float(scale as f64),
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
