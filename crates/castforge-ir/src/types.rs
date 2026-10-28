use serde::{Deserialize, Serialize};
use std::fmt;

/// Element data types supported in tensor and scalar types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    I1,
    I4,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F8E4M3FN,
    F8E5M2,
    F16,
    BF16,
    F32,
    F64,
    Index,
    QuantUniform {
        storage_type: Box<DType>,
        expressed_type: Box<DType>,
        scale: String,
        zero_point: i64,
    },
}

impl DType {
    pub fn bit_width(&self) -> usize {
        match self {
            DType::I1 => 1,
            DType::I4 => 4,
            DType::I8 | DType::U8 | DType::F8E4M3FN | DType::F8E5M2 => 8,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 16,
            DType::I32 | DType::U32 | DType::F32 => 32,
            DType::I64 | DType::U64 | DType::F64 | DType::Index => 64,
            DType::QuantUniform { storage_type, .. } => storage_type.bit_width(),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::I1 => write!(f, "i1"),
            DType::I4 => write!(f, "i4"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "ui8"),
            DType::U16 => write!(f, "ui16"),
            DType::U32 => write!(f, "ui32"),
            DType::U64 => write!(f, "ui64"),
            DType::F8E4M3FN => write!(f, "f8E4M3FN"),
            DType::F8E5M2 => write!(f, "f8E5M2"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::Index => write!(f, "index"),
            DType::QuantUniform {
                storage_type,
                expressed_type,
                scale,
                zero_point,
            } => write!(
                f,
                "!quant.uniform<{}:{},{}:{}>",
                storage_type, expressed_type, scale, zero_point
            ),
        }
    }
}

/// Tensor dimension (static, dynamic `?`, or symbolic).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Static(i64),
    Dynamic,
    Symbolic(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(d) => write!(f, "{d}"),
            Dim::Dynamic => write!(f, "?"),
            Dim::Symbolic(s) => write!(f, "{s}"),
        }
    }
}

/// Ranked or unranked tensor type (`tensor<4x8xf32>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub dtype: DType,
    pub encoding: Option<String>,
}

impl TensorType {
    pub fn new(shape: Vec<Dim>, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            encoding: None,
        }
    }

    pub fn static_shape(dims: &[i64], dtype: DType) -> Self {
        Self {
            shape: dims.iter().copied().map(Dim::Static).collect(),
            dtype,
            encoding: None,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> Option<i64> {
        let mut prod = 1i64;
        for d in &self.shape {
            match d {
                Dim::Static(n) => prod = prod.checked_mul(*n)?,
                _ => return None,
            }
        }
        Some(prod)
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor<")?;
        for d in &self.shape {
            write!(f, "{d}x")?;
        }
        write!(f, "{}", self.dtype)?;
        if let Some(enc) = &self.encoding {
            write!(f, ", {enc}")?;
        }
        write!(f, ">")
    }
}

/// IR Type representation (compatible with MLIR type system).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    Tensor(TensorType),
    Scalar(DType),
    Tuple(Vec<Type>),
    Function {
        inputs: Vec<Type>,
        results: Vec<Type>,
    },
    Opaque(String),
}

impl Type {
    pub fn as_tensor(&self) -> Option<&TensorType> {
        match self {
            Type::Tensor(t) => Some(t),
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Tensor(t) => write!(f, "{t}"),
            Type::Scalar(d) => write!(f, "{d}"),
            Type::Tuple(items) => {
                write!(f, "tuple<")?;
                for (i, ty) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ">")
            }
            Type::Function { inputs, results } => {
                write!(f, "(")?;
                for (i, inp) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{inp}")?;
                }
                write!(f, ") -> ")?;
                if results.len() == 1 {
                    write!(f, "{}", results[0])
                } else {
                    write!(f, "(")?;
                    for (i, res) in results.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{res}")?;
                    }
                    write!(f, ")")
                }
            }
            Type::Opaque(raw) => write!(f, "{raw}"),
        }
    }
}
