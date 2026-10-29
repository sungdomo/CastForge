use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, Type, Value,
    ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Post-Training Static Activation Calibration & W8A8 Quantization Pass (`calibrate-quantize`).
///
/// Runs a synthetic calibration sweep across the module to estimate the dynamic range
/// `max_abs` of every SSA tensor, derives symmetric INT8 (`bits = 8`, `qmax = 127.0`) or
/// INT4 (`bits = 4`, `qmax = 7.0`) activation scales `scale = max_abs / qmax`, and inserts
/// `stablehlo.composite` (`"castforge.static_quantize"`) fake-quantization / clamp-scale nodes
/// on `dot_general` and `convolution` inputs with `quant_mode = "w8a8_static"`.
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
