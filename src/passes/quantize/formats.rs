//! Submodule `formats` of `quantize`.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;

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

pub(super) fn calibrate_value_ranges(module: &Module, num_samples: usize) -> HashMap<ValueId, f64> {
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

/// QuaRot / SpinQuant Walsh-Hadamard Outlier Suppression Pass (`rotate-hadamard`).
pub struct RotateHadamardPass {
    pub block_size: usize,
}

impl Default for RotateHadamardPass {
    fn default() -> Self {
        Self { block_size: 4 }
    }
}

impl Pass for RotateHadamardPass {
    fn name(&self) -> &'static str {
        "rotate-hadamard"
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
                if !op.is_op("stablehlo", "constant")
                    || op.results.is_empty()
                    || op
                        .attributes
                        .get("castforge.hadamard_rotated")
                        .and_then(|a| a.as_bool())
                        == Some(true)
                {
                    continue;
                }
                let res_id = op.results[0];
                let Some(t_ty) = module.values[res_id].ty.as_tensor().cloned() else {
                    continue;
                };
                if t_ty.shape.len() != 2 {
                    continue;
                };
                let (Dim::Static(k_dim), Dim::Static(n_dim)) = (&t_ty.shape[0], &t_ty.shape[1])
                else {
                    continue;
                };
                let k = *k_dim as usize;
                let n = *n_dim as usize;
                if k < 2 || n == 0 {
                    continue;
                };
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

                let ssa = module.values[res_id].ssa_name.clone();
                let new_val = module.tensor_store.update_f32_constant(
                    ssa.as_deref(),
                    &val_attr,
                    t_ty,
                    w_data,
                );
                let op_mut = &mut module.ops[op_id];
                op_mut.attributes.insert("value".to_string(), new_val);
                op_mut.attributes.insert(
                    "castforge.hadamard_rotated".to_string(),
                    Attribute::Bool(true),
                );
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

/// Blackwell NVFP4 / OCP MXFP4 4-Bit Float Microscaling Quantization Pass (`quantize-fp4`).
pub struct QuantizeFp4Pass {
    pub scheme: String,
    pub block_size: usize,
}

impl Default for QuantizeFp4Pass {
    fn default() -> Self {
        Self {
            scheme: "nvfp4".to_string(),
            block_size: 16,
        }
    }
}

impl Pass for QuantizeFp4Pass {
    fn name(&self) -> &'static str {
        "quantize-fp4"
    }

    fn description(&self) -> &'static str {
        "Quantizes weights to Blackwell NVFP4 / OCP MXFP4 E2M1 4-bit float microscaling blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(run_fp4_quantize(module, &self.scheme, self.block_size)?.with_name(self.name()))
    }
}

pub(super) fn run_fp4_quantize(
    module: &mut Module,
    scheme: &str,
    block_size: usize,
) -> Result<PassResult> {
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
            if !op.is_op("stablehlo", "constant")
                || op
                    .attributes
                    .get("castforge.quant_scheme")
                    .and_then(|a| a.as_str())
                    == Some(scheme_label)
            {
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

            let ssa = module.values[res_id].ssa_name.clone();
            let new_val =
                module
                    .tensor_store
                    .update_f32_constant(ssa.as_deref(), &val_attr, t_ty, data);
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

/// Automatic Mixed Precision Lowering Pass (`lower-mixed-precision`).
pub struct LowerMixedPrecisionPass {
    pub target_dtype: DType,
}

impl Default for LowerMixedPrecisionPass {
    fn default() -> Self {
        Self {
            target_dtype: DType::F16,
        }
    }
}

impl Pass for LowerMixedPrecisionPass {
    fn name(&self) -> &'static str {
        "lower-mixed-precision"
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

/// QAT Straight-Through Estimator Fake-Quantization Pass (`insert-qat-ste`).
pub struct InsertQatStePass {
    pub bits: u8,
}

impl Default for InsertQatStePass {
    fn default() -> Self {
        Self { bits: 8 }
    }
}

impl Pass for InsertQatStePass {
    fn name(&self) -> &'static str {
        "insert-qat-ste"
    }

    fn description(&self) -> &'static str {
        "Applies fake-quantization with Straight-Through Estimator (QAT-STE) for training"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        Ok(run_qat_ste(module, self.bits)?.with_name(self.name()))
    }
}

pub(super) fn run_qat_ste(module: &mut Module, bits: u8) -> Result<PassResult> {
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
            if !op.is_op("stablehlo", "constant")
                || op.results.is_empty()
                || op
                    .attributes
                    .get("castforge.qat_ste")
                    .and_then(|a| a.as_bool())
                    == Some(true)
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

            let ssa = module.values[res_id].ssa_name.clone();
            let new_val = module.tensor_store.update_f32_constant(
                ssa.as_deref(),
                &val_attr,
                t_ty,
                fake_quant,
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.attributes.insert("value".to_string(), new_val);
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
