//! Strongly-typed `stablehlo` and `func` dialect views and shape verifiers
//! (`castforge::ir::stablehlo`).

use crate::ir::{Attribute, Dim, Module, OpId, Operation, Type, ValueId};
use serde::{Deserialize, Serialize};

/// Dimension numbers for `stablehlo.dot_general`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DotDimensionNumbers {
    pub lhs_batching_dimensions: Vec<i64>,
    pub rhs_batching_dimensions: Vec<i64>,
    pub lhs_contracting_dimensions: Vec<i64>,
    pub rhs_contracting_dimensions: Vec<i64>,
}

/// Strongly-typed view of `stablehlo` operations for ergonomic pattern matching in passes.
#[derive(Debug, Clone, PartialEq)]
pub enum StableHloOp {
    Constant {
        value: Attribute,
        result: ValueId,
    },
    Iota {
        iota_dimension: i64,
        result: ValueId,
    },
    Add {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Subtract {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Multiply {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Divide {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Maximum {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Minimum {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Exp {
        operand: ValueId,
        result: ValueId,
    },
    Tanh {
        operand: ValueId,
        result: ValueId,
    },
    Rsqrt {
        operand: ValueId,
        result: ValueId,
    },
    Sqrt {
        operand: ValueId,
        result: ValueId,
    },
    Log {
        operand: ValueId,
        result: ValueId,
    },
    Sine {
        operand: ValueId,
        result: ValueId,
    },
    Cosine {
        operand: ValueId,
        result: ValueId,
    },
    Abs {
        operand: ValueId,
        result: ValueId,
    },
    Negate {
        operand: ValueId,
        result: ValueId,
    },
    Logistic {
        operand: ValueId,
        result: ValueId,
    },
    Power {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Clamp {
        min: ValueId,
        operand: ValueId,
        max: ValueId,
        result: ValueId,
    },
    Pad {
        operand: ValueId,
        padding_value: ValueId,
        edge_padding_low: Vec<i64>,
        edge_padding_high: Vec<i64>,
        interior_padding: Vec<i64>,
        result: ValueId,
    },
    Gather {
        operand: ValueId,
        start_indices: ValueId,
        slice_sizes: Vec<i64>,
        result: ValueId,
    },
    Scatter {
        operand: ValueId,
        scatter_indices: ValueId,
        updates: ValueId,
        result: ValueId,
    },
    DynamicSlice {
        operand: ValueId,
        start_indices: Vec<ValueId>,
        slice_sizes: Vec<i64>,
        result: ValueId,
    },
    DynamicUpdateSlice {
        operand: ValueId,
        update: ValueId,
        start_indices: Vec<ValueId>,
        result: ValueId,
    },
    While {
        operands: Vec<ValueId>,
        results: Vec<ValueId>,
    },
    Case {
        index: ValueId,
        results: Vec<ValueId>,
    },
    Convert {
        operand: ValueId,
        result: ValueId,
    },
    Reshape {
        operand: ValueId,
        result: ValueId,
    },
    Transpose {
        operand: ValueId,
        permutation: Vec<i64>,
        result: ValueId,
    },
    BroadcastInDim {
        operand: ValueId,
        broadcast_dimensions: Vec<i64>,
        result: ValueId,
    },
    Slice {
        operand: ValueId,
        start_indices: Vec<i64>,
        limit_indices: Vec<i64>,
        strides: Vec<i64>,
        result: ValueId,
    },
    Concatenate {
        inputs: Vec<ValueId>,
        dimension: i64,
        result: ValueId,
    },
    Compare {
        lhs: ValueId,
        rhs: ValueId,
        direction: String,
        result: ValueId,
    },
    Select {
        pred: ValueId,
        on_true: ValueId,
        on_false: ValueId,
        result: ValueId,
    },
    DotGeneral {
        lhs: ValueId,
        rhs: ValueId,
        dimension_numbers: DotDimensionNumbers,
        result: ValueId,
    },
    Convolution {
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
    },
    Reduce {
        inputs: Vec<ValueId>,
        init_values: Vec<ValueId>,
        dimensions: Vec<i64>,
        reducer_op: Option<String>,
        results: Vec<ValueId>,
    },
    Composite {
        name: String,
        decomposition: String,
        inputs: Vec<ValueId>,
        results: Vec<ValueId>,
    },
    CustomCall {
        call_target_name: String,
        inputs: Vec<ValueId>,
        results: Vec<ValueId>,
    },
    Return {
        operands: Vec<ValueId>,
    },
    Other {
        op_name: String,
    },
}

impl StableHloOp {
    /// Attempt to classify a generic `Operation` into a strongly-typed `StableHloOp`.
    pub fn from_op(op: &Operation) -> Option<Self> {
        if op.name.dialect != "stablehlo" {
            return None;
        }
        let first_res = op.results.first().copied();
        match op.name.name.as_str() {
            "constant" => Some(StableHloOp::Constant {
                value: op.attributes.get("value")?.clone(),
                result: first_res?,
            }),
            "iota" => {
                let dim = match op.attributes.get("iota_dimension") {
                    Some(Attribute::Int(d)) => *d,
                    _ => 0,
                };
                Some(StableHloOp::Iota {
                    iota_dimension: dim,
                    result: first_res?,
                })
            }
            "add" => Some(StableHloOp::Add {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "subtract" => Some(StableHloOp::Subtract {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "multiply" => Some(StableHloOp::Multiply {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "divide" => Some(StableHloOp::Divide {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "maximum" => Some(StableHloOp::Maximum {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "minimum" => Some(StableHloOp::Minimum {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "exponential" => Some(StableHloOp::Exp {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "tanh" => Some(StableHloOp::Tanh {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "rsqrt" => Some(StableHloOp::Rsqrt {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "sqrt" => Some(StableHloOp::Sqrt {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "log" => Some(StableHloOp::Log {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "sine" => Some(StableHloOp::Sine {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "cosine" => Some(StableHloOp::Cosine {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "abs" => Some(StableHloOp::Abs {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "negate" => Some(StableHloOp::Negate {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "logistic" => Some(StableHloOp::Logistic {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "power" => Some(StableHloOp::Power {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "clamp" => Some(StableHloOp::Clamp {
                min: *op.operands.first()?,
                operand: *op.operands.get(1)?,
                max: *op.operands.get(2)?,
                result: first_res?,
            }),
            "pad" => {
                let edge_padding_low = op
                    .attributes
                    .get("edge_padding_low")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let edge_padding_high = op
                    .attributes
                    .get("edge_padding_high")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let interior_padding = op
                    .attributes
                    .get("interior_padding")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                Some(StableHloOp::Pad {
                    operand: *op.operands.first()?,
                    padding_value: *op.operands.get(1)?,
                    edge_padding_low,
                    edge_padding_high,
                    interior_padding,
                    result: first_res?,
                })
            }
            "gather" => {
                let slice_sizes = op
                    .attributes
                    .get("slice_sizes")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                Some(StableHloOp::Gather {
                    operand: *op.operands.first()?,
                    start_indices: *op.operands.get(1)?,
                    slice_sizes,
                    result: first_res?,
                })
            }
            "scatter" => Some(StableHloOp::Scatter {
                operand: *op.operands.first()?,
                scatter_indices: *op.operands.get(1)?,
                updates: *op.operands.get(2)?,
                result: first_res?,
            }),
            "dynamic_slice" => {
                let slice_sizes = op
                    .attributes
                    .get("slice_sizes")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                Some(StableHloOp::DynamicSlice {
                    operand: *op.operands.first()?,
                    start_indices: op.operands.iter().skip(1).copied().collect(),
                    slice_sizes,
                    result: first_res?,
                })
            }
            "dynamic_update_slice" => Some(StableHloOp::DynamicUpdateSlice {
                operand: *op.operands.first()?,
                update: *op.operands.get(1)?,
                start_indices: op.operands.iter().skip(2).copied().collect(),
                result: first_res?,
            }),
            "while" => Some(StableHloOp::While {
                operands: op.operands.to_vec(),
                results: op.results.to_vec(),
            }),
            "case" => Some(StableHloOp::Case {
                index: *op.operands.first()?,
                results: op.results.to_vec(),
            }),
            "convert" => Some(StableHloOp::Convert {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "reshape" => Some(StableHloOp::Reshape {
                operand: *op.operands.first()?,
                result: first_res?,
            }),
            "transpose" => {
                let perm = op
                    .attributes
                    .get("permutation")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                Some(StableHloOp::Transpose {
                    operand: *op.operands.first()?,
                    permutation: perm,
                    result: first_res?,
                })
            }
            "broadcast_in_dim" => {
                let dims = op
                    .attributes
                    .get("broadcast_dimensions")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                Some(StableHloOp::BroadcastInDim {
                    operand: *op.operands.first()?,
                    broadcast_dimensions: dims,
                    result: first_res?,
                })
            }
            "slice" => {
                let start_indices = op
                    .attributes
                    .get("start_indices")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let limit_indices = op
                    .attributes
                    .get("limit_indices")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let strides = op
                    .attributes
                    .get("strides")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![1; start_indices.len()]);
                Some(StableHloOp::Slice {
                    operand: *op.operands.first()?,
                    start_indices,
                    limit_indices,
                    strides,
                    result: first_res?,
                })
            }
            "concatenate" => {
                let dimension = match op.attributes.get("dimension") {
                    Some(Attribute::Int(d)) => *d,
                    _ => 0,
                };
                Some(StableHloOp::Concatenate {
                    inputs: op.operands.to_vec(),
                    dimension,
                    result: first_res?,
                })
            }
            "compare" => {
                let direction = op
                    .attributes
                    .get("comparison_direction")
                    .and_then(|a| a.as_str())
                    .unwrap_or("EQ")
                    .to_string();
                Some(StableHloOp::Compare {
                    lhs: *op.operands.first()?,
                    rhs: *op.operands.get(1)?,
                    direction,
                    result: first_res?,
                })
            }
            "select" => Some(StableHloOp::Select {
                pred: *op.operands.first()?,
                on_true: *op.operands.get(1)?,
                on_false: *op.operands.get(2)?,
                result: first_res?,
            }),
            "dot_general" => {
                let lhs_batching_dimensions = op
                    .attributes
                    .get("lhs_batching_dimensions")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let rhs_batching_dimensions = op
                    .attributes
                    .get("rhs_batching_dimensions")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let lhs_contracting_dimensions = op
                    .attributes
                    .get("lhs_contracting_dimensions")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![1]);
                let rhs_contracting_dimensions = op
                    .attributes
                    .get("rhs_contracting_dimensions")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![0]);

                Some(StableHloOp::DotGeneral {
                    lhs: *op.operands.first()?,
                    rhs: *op.operands.get(1)?,
                    dimension_numbers: DotDimensionNumbers {
                        lhs_batching_dimensions,
                        rhs_batching_dimensions,
                        lhs_contracting_dimensions,
                        rhs_contracting_dimensions,
                    },
                    result: first_res?,
                })
            }
            "convolution" => Some(StableHloOp::Convolution {
                lhs: *op.operands.first()?,
                rhs: *op.operands.get(1)?,
                result: first_res?,
            }),
            "reduce" => {
                let half = op.operands.len() / 2;
                let inputs = op.operands[..half].to_vec();
                let init_values = op.operands[half..].to_vec();
                let dimensions = op
                    .attributes
                    .get("dimensions")
                    .and_then(|a| a.as_int_slice())
                    .map(|s| s.to_vec())
                    .unwrap_or_default();
                let reducer_op = op
                    .attributes
                    .get("reducer")
                    .and_then(|a| a.as_str())
                    .map(|s| s.to_string());
                Some(StableHloOp::Reduce {
                    inputs,
                    init_values,
                    dimensions,
                    reducer_op,
                    results: op.results.to_vec(),
                })
            }
            "composite" => Some(StableHloOp::Composite {
                name: op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .unwrap_or("composite")
                    .to_string(),
                decomposition: op
                    .attributes
                    .get("decomposition")
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string(),
                inputs: op.operands.to_vec(),
                results: op.results.to_vec(),
            }),
            "custom_call" => Some(StableHloOp::CustomCall {
                call_target_name: op
                    .attributes
                    .get("call_target_name")
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string(),
                inputs: op.operands.to_vec(),
                results: op.results.to_vec(),
            }),
            "return" => Some(StableHloOp::Return {
                operands: op.operands.to_vec(),
            }),
            other => Some(StableHloOp::Other {
                op_name: other.to_string(),
            }),
        }
    }
}

/// Validates StableHLO tensor shape rules for `op_id` inside `module`.
pub fn verify_stablehlo_shapes(module: &Module, op_id: OpId) -> Result<(), String> {
    let Some(op) = module.ops.get(op_id) else {
        return Ok(());
    };
    let Some(hlo) = StableHloOp::from_op(op) else {
        return Ok(());
    };

    match hlo {
        StableHloOp::Transpose {
            operand,
            permutation,
            result,
        } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            if in_ty.rank() != permutation.len() || out_ty.rank() != permutation.len() {
                return Err(format!(
                    "stablehlo.transpose rank mismatch: input rank {}, perm {:?}, output rank {}",
                    in_ty.rank(),
                    permutation,
                    out_ty.rank()
                ));
            }
            for (out_idx, &in_idx) in permutation.iter().enumerate() {
                let Some(in_dim) = in_ty.shape.get(in_idx as usize) else {
                    return Err(format!(
                        "stablehlo.transpose permutation index {in_idx} out of bounds"
                    ));
                };
                let out_dim = &out_ty.shape[out_idx];
                if let (Dim::Static(a), Dim::Static(b)) = (in_dim, out_dim)
                    && a != b
                {
                    return Err(format!(
                        "stablehlo.transpose dim mismatch at axis {out_idx}: {a} vs {b}"
                    ));
                }
            }
        }
        StableHloOp::Reshape { operand, result } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            if let (Some(in_elems), Some(out_elems)) = (in_ty.num_elements(), out_ty.num_elements())
                && in_elems != out_elems
            {
                return Err(format!(
                    "stablehlo.reshape count mismatch: {in_ty} ({in_elems}) vs {out_ty}"
                ));
            }
        }
        StableHloOp::Slice {
            operand,
            start_indices,
            limit_indices,
            strides,
            result,
        } => {
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            let _ = operand;
            for (i, &start) in start_indices.iter().enumerate() {
                let limit = *limit_indices.get(i).unwrap_or(&start);
                let stride = (*strides.get(i).unwrap_or(&1)).max(1);
                let expected = (limit - start + stride - 1) / stride;
                if let Some(Dim::Static(actual)) = out_ty.shape.get(i)
                    && *actual != expected
                {
                    return Err(format!(
                        "stablehlo.slice axis {i} expected dim {expected}, got {actual}"
                    ));
                }
            }
        }
        StableHloOp::DynamicUpdateSlice {
            operand, result, ..
        } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            if in_ty.shape != out_ty.shape {
                return Err(format!(
                    "stablehlo.dynamic_update_slice result shape {:?} must match operand {:?}",
                    out_ty.shape, in_ty.shape
                ));
            }
        }
        StableHloOp::Clamp {
            operand, result, ..
        } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            if in_ty.shape != out_ty.shape {
                return Err(format!(
                    "stablehlo.clamp result shape {:?} must match operand {:?}",
                    out_ty.shape, in_ty.shape
                ));
            }
        }
        StableHloOp::DotGeneral {
            lhs,
            rhs,
            dimension_numbers,
            ..
        } => {
            let Some(lhs_ty) = module.values[lhs].ty.as_tensor() else {
                return Ok(());
            };
            let Some(rhs_ty) = module.values[rhs].ty.as_tensor() else {
                return Ok(());
            };
            if dimension_numbers.lhs_batching_dimensions.len()
                != dimension_numbers.rhs_batching_dimensions.len()
            {
                return Err("stablehlo.dot_general batching dimension count mismatch".to_string());
            }
            for (&lb, &rb) in dimension_numbers
                .lhs_batching_dimensions
                .iter()
                .zip(dimension_numbers.rhs_batching_dimensions.iter())
            {
                if let (Some(Dim::Static(a)), Some(Dim::Static(b))) =
                    (lhs_ty.shape.get(lb as usize), rhs_ty.shape.get(rb as usize))
                    && a != b
                {
                    return Err(format!(
                        "stablehlo.dot_general batching dim mismatch: \
                         lhs[{lb}]={a} vs rhs[{rb}]={b}"
                    ));
                }
            }
            for (&lc, &rc) in dimension_numbers
                .lhs_contracting_dimensions
                .iter()
                .zip(dimension_numbers.rhs_contracting_dimensions.iter())
            {
                if let (Some(Dim::Static(a)), Some(Dim::Static(b))) =
                    (lhs_ty.shape.get(lc as usize), rhs_ty.shape.get(rc as usize))
                    && a != b
                {
                    return Err(format!(
                        "stablehlo.dot_general contracting dim mismatch: \
                         lhs[{lc}]={a} vs rhs[{rc}]={b}"
                    ));
                }
            }
        }
        StableHloOp::Concatenate {
            inputs,
            dimension,
            result,
        } => {
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            let axis = dimension as usize;
            let mut sum_axis = 0i64;
            let mut all_static = true;
            for &inp in &inputs {
                let Some(in_ty) = module.values[inp].ty.as_tensor() else {
                    continue;
                };
                if in_ty.rank() != out_ty.rank() {
                    return Err(format!(
                        "stablehlo.concatenate input rank {} != result rank {}",
                        in_ty.rank(),
                        out_ty.rank()
                    ));
                }
                match in_ty.shape.get(axis) {
                    Some(Dim::Static(n)) => sum_axis += *n,
                    _ => all_static = false,
                }
            }
            if all_static
                && let Some(Dim::Static(out_n)) = out_ty.shape.get(axis)
                && *out_n != sum_axis
            {
                return Err(format!(
                    "stablehlo.concatenate axis {axis} sum {sum_axis} != result dim {out_n}"
                ));
            }
        }
        StableHloOp::BroadcastInDim {
            operand,
            broadcast_dimensions,
            result,
        } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            for &bdim in &broadcast_dimensions {
                if bdim < 0 || (bdim as usize) >= out_ty.rank() {
                    return Err(format!(
                        "stablehlo.broadcast_in_dim axis {bdim} out of bounds for rank {}",
                        out_ty.rank()
                    ));
                }
            }
            if broadcast_dimensions.len() == in_ty.rank() {
                for (in_axis, &bdim) in broadcast_dimensions.iter().enumerate() {
                    if let (Some(Dim::Static(in_d)), Some(Dim::Static(out_d))) =
                        (in_ty.shape.get(in_axis), out_ty.shape.get(bdim as usize))
                        && *in_d != 1
                        && in_d != out_d
                    {
                        return Err(format!(
                            "stablehlo.broadcast_in_dim dim mismatch at axis {in_axis} -> {bdim}: \
                             {in_d} vs {out_d}"
                        ));
                    }
                }
            }
        }
        StableHloOp::Iota {
            iota_dimension,
            result,
        } => {
            if let Some(out_ty) = module.values[result].ty.as_tensor()
                && out_ty.rank() > 0
                && (iota_dimension < 0 || (iota_dimension as usize) >= out_ty.rank())
            {
                return Err(format!(
                    "stablehlo.iota dim {iota_dimension} out of bounds for rank {}",
                    out_ty.rank()
                ));
            }
        }
        StableHloOp::Reduce {
            inputs, dimensions, ..
        } => {
            if let Some(&inp) = inputs.first()
                && let Some(in_ty) = module.values[inp].ty.as_tensor()
            {
                for &dim in &dimensions {
                    if dim < 0 || (dim as usize) >= in_ty.rank() {
                        return Err(format!(
                            "stablehlo.reduce dimension {dim} out of bounds for rank {}",
                            in_ty.rank()
                        ));
                    }
                }
            }
        }
        StableHloOp::Exp { operand, result }
        | StableHloOp::Tanh { operand, result }
        | StableHloOp::Rsqrt { operand, result }
        | StableHloOp::Sqrt { operand, result }
        | StableHloOp::Log { operand, result }
        | StableHloOp::Sine { operand, result }
        | StableHloOp::Cosine { operand, result }
        | StableHloOp::Abs { operand, result }
        | StableHloOp::Negate { operand, result }
        | StableHloOp::Logistic { operand, result } => {
            if let (Some(in_ty), Some(out_ty)) = (
                module.values[operand].ty.as_tensor(),
                module.values[result].ty.as_tensor(),
            ) {
                if in_ty.rank() != out_ty.rank() {
                    return Err(format!(
                        "{} rank mismatch: {} vs {}",
                        op.name,
                        in_ty.rank(),
                        out_ty.rank()
                    ));
                }
                for (i, (a, b)) in in_ty.shape.iter().zip(out_ty.shape.iter()).enumerate() {
                    if let (Dim::Static(da), Dim::Static(db)) = (a, b)
                        && da != db
                    {
                        return Err(format!(
                            "{} dim mismatch at axis {i}: {da} vs {db}",
                            op.name
                        ));
                    }
                }
            }
        }
        StableHloOp::Add { lhs, rhs, result }
        | StableHloOp::Subtract { lhs, rhs, result }
        | StableHloOp::Multiply { lhs, rhs, result }
        | StableHloOp::Divide { lhs, rhs, result }
        | StableHloOp::Maximum { lhs, rhs, result }
        | StableHloOp::Minimum { lhs, rhs, result }
        | StableHloOp::Power { lhs, rhs, result }
        | StableHloOp::Compare {
            lhs, rhs, result, ..
        } => {
            if let (Some(l_ty), Some(r_ty), Some(out_ty)) = (
                module.values[lhs].ty.as_tensor(),
                module.values[rhs].ty.as_tensor(),
                module.values[result].ty.as_tensor(),
            ) && l_ty.rank() == r_ty.rank()
                && l_ty.rank() == out_ty.rank()
            {
                for i in 0..out_ty.rank() {
                    if let (Dim::Static(dl), Dim::Static(dr), Dim::Static(do_)) =
                        (&l_ty.shape[i], &r_ty.shape[i], &out_ty.shape[i])
                        && ((*dl != *do_ && *dl != 1) || (*dr != *do_ && *dr != 1))
                    {
                        return Err(format!(
                            "{} incompatible dims at axis {i}: lhs={dl}, rhs={dr}, out={do_}",
                            op.name
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    Ok(())
}

/// Helper to query function metadata from a `func.func` operation.
pub struct FuncView<'a> {
    pub op_id: OpId,
    pub sym_name: &'a str,
    pub function_type: Option<&'a Type>,
}

impl<'a> FuncView<'a> {
    pub fn from_op(module: &'a Module, op_id: OpId) -> Option<Self> {
        let op = module.ops.get(op_id)?;
        if !op.is_op("func", "func") {
            return None;
        }
        let sym_name = op.attributes.get("sym_name")?.as_str()?;
        let function_type = match op.attributes.get("function_type") {
            Some(Attribute::Type(ty)) => Some(ty),
            _ => None,
        };
        Some(Self {
            op_id,
            sym_name,
            function_type,
        })
    }
}
