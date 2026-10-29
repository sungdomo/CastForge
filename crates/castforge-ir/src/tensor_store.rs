use crate::types::{Dim, TensorType};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Backing storage for a weight/constant tensor in the Weight Plane (`TensorStore`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorBacking {
    /// In-memory raw byte buffer.
    InMemory(Vec<u8>),
    /// In-memory typed F32 values (convenient for pruning, merging, and PTQ quantization).
    F32Data(Vec<f32>),
    /// In-memory quantized I8 values with scale and zero_point.
    QuantizedI8 {
        data: Vec<i8>,
        scale: f32,
        zero_point: i64,
    },
    /// External file slice (e.g., Safetensors, GGUF, or MLIR external resource).
    ExternalFile {
        path: PathBuf,
        offset: u64,
        length: u64,
    },
    /// Unallocated symbolic placeholder (graph-only mode).
    Symbolic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub ty: TensorType,
    pub backing: TensorBacking,
}

impl TensorEntry {
    pub fn from_f32(name: impl Into<String>, ty: TensorType, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            ty,
            backing: TensorBacking::F32Data(values),
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.backing {
            TensorBacking::F32Data(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Parse a flat or nested MLIR dense literal (e.g., `[[1.0, 2.0], [3.0, 4.0]]` or `0.5`)
/// into a vector of `f32` of length `expected_len` (expanding splats automatically).
pub fn parse_dense_f32_literal(raw_literal: &str, expected_len: usize) -> Option<Vec<f32>> {
    let trimmed = raw_literal.trim();
    if !trimmed.starts_with('[') {
        let val = trimmed.parse::<f32>().ok()?;
        return Some(vec![val; expected_len.max(1)]);
    }
    let mut vals = Vec::new();
    let cleaned: String = trimmed
        .chars()
        .map(|c| if c == '[' || c == ']' { ' ' } else { c })
        .collect();
    for tok in cleaned.split(',') {
        let tok = tok.trim();
        if !tok.is_empty() {
            vals.push(tok.parse::<f32>().ok()?);
        }
    }
    if vals.len() == 1 && expected_len > 1 {
        return Some(vec![vals[0]; expected_len]);
    }
    Some(vals)
}

/// Format a slice of `f32` or `i8` values as an MLIR `dense<...>` literal matching `shape`.
pub fn format_dense_f32_literal(values: &[f32], shape: &[Dim]) -> String {
    if values.is_empty() {
        return "0.0".to_string();
    }
    // Check if all elements are identical (splat)
    let first = values[0];
    if values.len() > 1 && values.iter().all(|&v| (v - first).abs() < 1e-7) {
        return format!("{first:?}");
    }
    let static_dims: Vec<usize> = shape
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) if *n > 0 => Some(*n as usize),
            _ => None,
        })
        .collect();
    if static_dims.len() == 2 && static_dims[0] * static_dims[1] == values.len() {
        let rows = static_dims[0];
        let cols = static_dims[1];
        let mut row_strs = Vec::with_capacity(rows);
        for r in 0..rows {
            let slice = &values[r * cols..(r + 1) * cols];
            let elems: Vec<String> = slice.iter().map(|v| format!("{v:?}")).collect();
            row_strs.push(format!("[{}]", elems.join(", ")));
        }
        format!("[{}]", row_strs.join(", "))
    } else {
        let elems: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
        format!("[{}]", elems.join(", "))
    }
}

pub fn format_dense_i8_literal(values: &[i8], shape: &[Dim]) -> String {
    if values.is_empty() {
        return "0".to_string();
    }
    let first = values[0];
    if values.len() > 1 && values.iter().all(|&v| v == first) {
        return format!("{first}");
    }
    let static_dims: Vec<usize> = shape
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) if *n > 0 => Some(*n as usize),
            _ => None,
        })
        .collect();
    if static_dims.len() == 2 && static_dims[0] * static_dims[1] == values.len() {
        let rows = static_dims[0];
        let cols = static_dims[1];
        let mut row_strs = Vec::with_capacity(rows);
        for r in 0..rows {
            let slice = &values[r * cols..(r + 1) * cols];
            let elems: Vec<String> = slice.iter().map(|v| v.to_string()).collect();
            row_strs.push(format!("[{}]", elems.join(", ")));
        }
        format!("[{}]", row_strs.join(", "))
    } else {
        let elems: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        format!("[{}]", elems.join(", "))
    }
}

/// The Weight Plane of CastForge's Dual-Plane IR.
/// Holds weight buffers or lazy external file references alongside the SSA Compute Graph.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStore {
    pub(crate) entries: IndexMap<String, TensorEntry>,
}

impl TensorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entry: TensorEntry) {
        self.entries.insert(entry.name.clone(), entry);
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut TensorEntry> {
        self.entries.get_mut(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorEntry)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
