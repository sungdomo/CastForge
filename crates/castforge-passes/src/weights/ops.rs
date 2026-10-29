use castforge_ir::DType;

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

/// Structured pruning of a 2D weight matrix `[rows, cols]` along its output columns (`dim = 1`),
/// keeping the top `keep_cols` columns with highest L1 importance norm.
/// Returns `(pruned_f32_data, kept_column_indices)`.
pub fn prune_2d_columns_by_l1(
    values: &[f32],
    rows: usize,
    cols: usize,
    keep_cols: usize,
) -> (Vec<f32>, Vec<usize>) {
    if rows == 0 || cols == 0 || values.len() < rows * cols {
        return (values.to_vec(), (0..cols).collect());
    }
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
        MergeMethod::Linear => (0..len).map(|i| (1.0 - t) * a[i] + t * b[i]).collect(),
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
        MergeMethod::Ties => (0..len)
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
            .collect(),
    }
}
