//! Strongly-typed `stablehlo` and `func` dialect views and shape verifiers
//! (`castforge-dialect-stablehlo`).

pub mod verify;
pub use verify::{FuncView, verify_stablehlo_shapes};

use castforge_ir::{Attribute, Operation, ValueId};
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
