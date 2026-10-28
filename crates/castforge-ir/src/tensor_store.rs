use crate::types::{DType, Dim, TensorType};
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

/// Symmetric per-tensor quantization of `f32` weights to `i8` or `i4`.
/// Returns `(quantized_values, scale, zero_point, quantized_dtype)`.
pub fn quantize_symmetric_f32(
    values: &[f32],
    bits: u8,
    expressed_dtype: DType,
) -> (Vec<i8>, f32, i64, DType) {
    let qmax = if bits == 4 { 7.0f32 } else { 127.0f32 };
    let qmin = if bits == 4 { -8.0f32 } else { -128.0f32 };

    let max_abs = values
        .iter()
        .fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs < 1e-12 {
        1.0
    } else {
        max_abs / qmax
    };

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

/// Structured pruning of a 2D weight matrix `[rows, cols]` along its output columns (`dim = 1`),
/// keeping the top `keep_cols` columns with highest L1 importance norm.
/// Returns `(pruned_f32_data, kept_column_indices)`.
pub fn prune_2d_columns_by_l1(
    values: &[f32],
    rows: usize,
    cols: usize,
    keep_cols: usize,
) -> (Vec<f32>, Vec<usize>) {
    let keep_cols = keep_cols.clamp(1, cols);
    let mut col_scores: Vec<(usize, f32)> = (0..cols)
        .map(|c| {
            let mut l1 = 0.0f32;
            for r in 0..rows {
                l1 += values[r * cols + c].abs();
            }
            (c, l1)
        })
        .collect();

    // Sort descending by L1 score, then preserve ascending column index order
    col_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept_indices: Vec<usize> = col_scores
        .into_iter()
        .take(keep_cols)
        .map(|(idx, _)| idx)
        .collect();
    kept_indices.sort_unstable();

    let mut pruned = Vec::with_capacity(rows * keep_cols);
    for r in 0..rows {
        for &c in &kept_indices {
            pruned.push(values[r * cols + c]);
        }
    }
    (pruned, kept_indices)
}

/// Model Merging algorithms on two equal-length `f32` weight slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMethod {
    /// `(1 - t) * a + t * b`
    Linear,
    /// Spherical Linear Interpolation (SLERP)
    Slerp,
    /// TIES-Merging (Trim, Elect Sign, Disjoint Merge)
    Ties,
}

pub fn merge_f32_tensors(a: &[f32], b: &[f32], method: MergeMethod, t: f32) -> Vec<f32> {
    let len = a.len().min(b.len());
    match method {
        MergeMethod::Linear => (0..len)
            .map(|i| (1.0 - t) * a[i] + t * b[i])
            .collect(),
        MergeMethod::Slerp => {
            let norm_a: f32 = a.iter().take(len).map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().take(len).map(|x| x * x).sum::<f32>().sqrt();
            if norm_a < 1e-8 || norm_b < 1e-8 {
                return merge_f32_tensors(a, b, MergeMethod::Linear, t);
            }
            let dot: f32 = (0..len).map(|i| (a[i] / norm_a) * (b[i] / norm_b)).sum();
            let dot_clamped = dot.clamp(-0.9995, 0.9995);
            let omega = dot_clamped.acos();
            let sin_omega = omega.sin();
            if sin_omega.abs() < 1e-6 {
                return merge_f32_tensors(a, b, MergeMethod::Linear, t);
            }
            let coeff_a = ((1.0 - t) * omega).sin() / sin_omega;
            let coeff_b = (t * omega).sin() / sin_omega;
            (0..len).map(|i| coeff_a * a[i] + coeff_b * b[i]).collect()
        }
        MergeMethod::Ties => {
            // Elect dominant sign weighted by (1-t) and t, then average matching-sign values
            (0..len)
                .map(|i| {
                    let va = a[i];
                    let vb = b[i];
                    let elected_sign = ((1.0 - t) * va + t * vb).signum();
                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;
                    if va.signum() == elected_sign || va == 0.0 {
                        sum += (1.0 - t) * va;
                        weight_sum += 1.0 - t;
                    }
                    if vb.signum() == elected_sign || vb == 0.0 {
                        sum += t * vb;
                        weight_sum += t;
                    }
                    if weight_sum > 0.0 {
                        sum / weight_sum
                    } else {
                        (1.0 - t) * va + t * vb
                    }
                })
                .collect()
        }
    }
}

/// The Weight Plane of CastForge's Dual-Plane IR.
/// Holds weight buffers or lazy external file references alongside the SSA Compute Graph.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStore {
    entries: IndexMap<String, TensorEntry>,
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
