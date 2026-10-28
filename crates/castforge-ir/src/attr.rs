use crate::types::{TensorType, Type};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Compile-time attributes attached to MLIR/StableHLO operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    SymbolRef(String),
    Type(Type),
    Array(Vec<Attribute>),
    DenseIntArray(Vec<i64>),
    DenseF64Array(Vec<f64>),
    DenseElements {
        ty: TensorType,
        raw_literal: String,
    },
    /// Reference to a weight tensor stored in the Module's `TensorStore`
    TensorRef {
        key: String,
        ty: TensorType,
    },
    Dictionary(IndexMap<String, Attribute>),
    Opaque(String),
}

impl Attribute {
    pub fn as_int_slice(&self) -> Option<&[i64]> {
        match self {
            Attribute::DenseIntArray(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attribute::String(s) | Attribute::SymbolRef(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// If this attribute is a scalar/splat `DenseElements` constant (e.g., `dense<0.0>`),
    /// parses and returns its scalar `f64` value.
    pub fn as_splat_f64(&self) -> Option<f64> {
        match self {
            Attribute::DenseElements { raw_literal, .. } => {
                let s = raw_literal.trim();
                if s.starts_with('[') {
                    return None;
                }
                s.parse::<f64>().ok()
            }
            Attribute::Float(f) => Some(*f),
            Attribute::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn is_splat_zero(&self) -> bool {
        self.as_splat_f64().map(|v| v == 0.0).unwrap_or(false)
    }

    pub fn is_splat_one(&self) -> bool {
        self.as_splat_f64()
            .map(|v| (v - 1.0).abs() < f64::EPSILON)
            .unwrap_or(false)
    }
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribute::Bool(b) => write!(f, "{b}"),
            Attribute::Int(i) => write!(f, "{i}"),
            Attribute::Float(v) => write!(f, "{v:?}"),
            Attribute::String(s) => write!(f, "\"{}\"", s.escape_default()),
            Attribute::SymbolRef(s) => write!(f, "@{s}"),
            Attribute::Type(ty) => write!(f, "{ty}"),
            Attribute::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Attribute::DenseIntArray(vals) => {
                write!(f, "array<i64: ")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ">")
            }
            Attribute::DenseF64Array(vals) => {
                write!(f, "array<f64: ")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v:?}")?;
                }
                write!(f, ">")
            }
            Attribute::DenseElements { ty, raw_literal } => {
                write!(f, "dense<{raw_literal}> : {ty}")
            }
            Attribute::TensorRef { key, ty } => {
                write!(f, "dense_resource<{key}> : {ty}")
            }
            Attribute::Dictionary(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k} = {v}")?;
                }
                write!(f, "}}")
            }
            Attribute::Opaque(raw) => write!(f, "{raw}"),
        }
    }
}
