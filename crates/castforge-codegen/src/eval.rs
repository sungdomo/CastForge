use crate::{tensor_len, tensor_shape_dims};
use anyhow::{Result, anyhow};
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{DType, Module, OpId, ValueId};
use std::collections::HashMap;

/// Numerical comparison report between two IR evaluations (e.g. before vs. after optimization).
#[derive(Debug, Clone)]
pub struct NumericVerificationReport {
    pub passed: bool,
    pub max_abs_diff: f32,
    pub tolerance: f32,
    pub num_outputs_checked: usize,
}

/// Generates deterministic synthetic inputs (`[-0.5, 0.5]` ramp) for a function's parameters.
pub fn generate_synthetic_inputs(module: &Module, fn_op_id: Option<OpId>) -> Result<Vec<Vec<f32>>> {
    let fns = module.functions();
    let target_fn = fn_op_id
        .or_else(|| fns.first().copied())
        .ok_or_else(|| anyhow!("Module contains no func.func to evaluate"))?;
    let fn_op = &module.ops[target_fn];
    let reg_id = *fn_op
        .regions
        .first()
        .ok_or_else(|| anyhow!("Function has no body region"))?;
    let blk_id = *module.regions[reg_id]
        .blocks
        .first()
        .ok_or_else(|| anyhow!("Function region has no entry block"))?;
    let block = &module.blocks[blk_id];

    let mut inputs = Vec::with_capacity(block.arguments.len());
    for (arg_idx, &arg_id) in block.arguments.iter().enumerate() {
        let len = tensor_len(module, arg_id);
        let vec: Vec<f32> = (0..len)
            .map(|i| {
                let raw = ((i * 7 + arg_idx * 13 + 3) % 19) as f32;
                (raw - 9.0) * 0.1
            })
            .collect();
        inputs.push(vec);
    }
    Ok(inputs)
}

/// Verifies that `before` and `after` modules produce numerically equivalent outputs
/// within `tolerance` on deterministic synthetic inputs.
pub fn verify_module_numerics(
    before: &Module,
    after: &Module,
    tolerance: f32,
) -> Result<NumericVerificationReport> {
    let inputs = generate_synthetic_inputs(before, None)?;
    let out_before = evaluate_module(before, &inputs)?;
    let out_after = evaluate_module(after, &inputs)?;

    if out_before.len() != out_after.len() {
        return Err(anyhow!(
            "Output count mismatch: before={} vs after={}",
            out_before.len(),
            out_after.len()
        ));
    }

    let mut max_abs_diff = 0.0_f32;
    let mut checked = 0usize;
    for (a, b) in out_before.iter().zip(out_after.iter()) {
        if a.len() != b.len() {
            return Err(anyhow!(
                "Output tensor shape mismatch: before len={} vs after len={}",
                a.len(),
                b.len()
            ));
        }
        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            checked += 1;
        }
    }

    Ok(NumericVerificationReport {
        passed: max_abs_diff <= tolerance,
        max_abs_diff,
        tolerance,
        num_outputs_checked: checked,
    })
}

/// Evaluates the entry function of `module` with the given `inputs` and returns its output tensors.
pub fn evaluate_module(module: &Module, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let fns = module.functions();
    let fn_op_id = *fns
        .first()
        .ok_or_else(|| anyhow!("No func.func in module"))?;
    evaluate_function(module, fn_op_id, inputs)
}

/// Evaluates a specific `func.func` operation in `module`.
pub fn evaluate_function(
    module: &Module,
    fn_op_id: OpId,
    inputs: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>> {
    let fn_op = &module.ops[fn_op_id];
    let reg_id = fn_op.regions[0];
    let blk_id = module.regions[reg_id].blocks[0];
    let block = &module.blocks[blk_id];

    if inputs.len() != block.arguments.len() {
        return Err(anyhow!(
            "Expected {} inputs, got {}",
            block.arguments.len(),
            inputs.len()
        ));
    }

    let mut env: HashMap<ValueId, Vec<f32>> = HashMap::new();
    let mut q_env: HashMap<ValueId, Vec<i8>> = HashMap::new();

    for (&arg_id, inp) in block.arguments.iter().zip(inputs.iter()) {
        env.insert(arg_id, inp.clone());
    }

    for &op_id in &block.ops {
        let op = &module.ops[op_id];
        if op.is_op("func", "return") || op.is_op("stablehlo", "return") {
            let mut outs = Vec::with_capacity(op.operands.len());
            for &ret_val in &op.operands {
                let v = env
                    .get(&ret_val)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; tensor_len(module, ret_val)]);
                outs.push(v);
            }
            return Ok(outs);
        }

        if op.is_op("func", "call") {
            let callee_sym = op
                .attributes
                .get("callee")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            if let Some(callee_id) = module.functions().into_iter().find(|&fid| {
                module.ops[fid]
                    .attributes
                    .get("sym_name")
                    .and_then(|a| a.as_str())
                    == Some(callee_sym)
            }) {
                let call_args: Vec<Vec<f32>> = op
                    .operands
                    .iter()
                    .map(|&v| env.get(&v).cloned().unwrap_or_default())
                    .collect();
                let call_outs = evaluate_function(module, callee_id, &call_args)?;
                for (&res_id, out_vec) in op.results.iter().zip(call_outs) {
                    env.insert(res_id, out_vec);
                }
                continue;
            }
        }

        let Some(&res_val) = op.results.first() else {
            continue;
        };
        let res_len = tensor_len(module, res_val);

        if op.is_op("stablehlo", "constant") {
            let t_ty = module.values[res_val]
                .ty
                .as_tensor()
                .ok_or_else(|| anyhow!("constant without tensor type"))?;
            let val_attr = op.attributes.get("value");
            if let DType::QuantUniform { .. } = &t_ty.dtype {
                let q_vals = val_attr
                    .and_then(|a| module.tensor_store.resolve_i8_attr(a, res_len))
                    .unwrap_or_else(|| vec![0; res_len]);
                q_env.insert(res_val, q_vals);
            } else {
                let vals = val_attr
                    .and_then(|a| module.tensor_store.resolve_f32_attr(a, res_len))
                    .unwrap_or_else(|| vec![0.0; res_len]);
                env.insert(res_val, vals);
            }
            continue;
        }

        if op.is_op("stablehlo", "uniform_dequantize") {
            let src_val = op.operands[0];
            let (scale, zp) = match module.values[src_val].ty.as_tensor().map(|t| &t.dtype) {
                Some(DType::QuantUniform {
                    scale, zero_point, ..
                }) => (scale.parse::<f32>().unwrap_or(1.0), *zero_point),
                _ => (1.0, 0),
            };
            let q_src = q_env
                .get(&src_val)
                .cloned()
                .unwrap_or_else(|| vec![0; res_len]);
            let deq: Vec<f32> = q_src
                .into_iter()
                .map(|q| (q as i32 - zp as i32) as f32 * scale)
                .collect();
            env.insert(res_val, deq);
            continue;
        }

        eval_stablehlo_op(module, op_id, res_val, res_len, &mut env)?;
    }

    Ok(Vec::new())
}

fn eval_stablehlo_op(
    module: &Module,
    op_id: OpId,
    res_val: ValueId,
    res_len: usize,
    env: &mut HashMap<ValueId, Vec<f32>>,
) -> Result<()> {
    let op = &module.ops[op_id];
    let get_v = |vid: ValueId| -> Vec<f32> {
        env.get(&vid)
            .cloned()
            .unwrap_or_else(|| vec![0.0; tensor_len(module, vid)])
    };

    let Some(hlo) = StableHloOp::from_op(op) else {
        env.insert(res_val, vec![0.0; res_len]);
        return Ok(());
    };

    let out: Vec<f32> = match hlo {
        StableHloOp::Add { lhs, rhs, .. } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| a + b)
            .collect(),
        StableHloOp::Subtract { lhs, rhs, .. } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| a - b)
            .collect(),
        StableHloOp::Multiply { lhs, rhs, .. } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| a * b)
            .collect(),
        StableHloOp::Divide { lhs, rhs, .. } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| if b.abs() > 1e-12 { a / b } else { 0.0 })
            .collect(),
        StableHloOp::Maximum { lhs, rhs, .. } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| a.max(*b))
            .collect(),
        StableHloOp::Minimum { lhs, rhs, .. } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| a.min(*b))
            .collect(),
        StableHloOp::Exp { operand, .. } => get_v(operand).into_iter().map(|x| x.exp()).collect(),
        StableHloOp::Tanh { operand, .. } => get_v(operand).into_iter().map(|x| x.tanh()).collect(),
        StableHloOp::Rsqrt { operand, .. } => {
            get_v(operand).into_iter().map(|x| 1.0 / x.sqrt()).collect()
        }
        StableHloOp::Convert { operand, .. } | StableHloOp::Reshape { operand, .. } => {
            get_v(operand)
        }
        StableHloOp::Transpose {
            operand,
            permutation,
            ..
        } => {
            let x = get_v(operand);
            let in_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            if in_dims.len() == 2 && permutation == vec![1, 0] {
                let (r, c) = (in_dims[0], in_dims[1]);
                let mut t = vec![0.0_f32; res_len];
                for row in 0..r {
                    for col in 0..c {
                        t[col * r + row] = x[row * c + col];
                    }
                }
                t
            } else {
                x
            }
        }
        StableHloOp::BroadcastInDim { operand, .. } => {
            let x = get_v(operand);
            let in_len = x.len().max(1);
            (0..res_len)
                .map(|i| x[i / (res_len / in_len).max(1) % in_len])
                .collect()
        }
        StableHloOp::Slice {
            operand,
            start_indices,
            limit_indices,
            strides,
            ..
        } => {
            let x = get_v(operand);
            let in_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![x.len()]);
            if in_dims.len() == 2 && start_indices.len() == 2 {
                let (s0, s1) = (start_indices[0] as usize, start_indices[1] as usize);
                let (l0, l1) = (limit_indices[0] as usize, limit_indices[1] as usize);
                let st0 = (strides.first().copied().unwrap_or(1) as usize).max(1);
                let st1 = (strides.get(1).copied().unwrap_or(1) as usize).max(1);
                let in_cols = in_dims[1];
                let mut out_v = Vec::with_capacity(res_len);
                for r in (s0..l0).step_by(st0) {
                    for c in (s1..l1).step_by(st1) {
                        out_v.push(*x.get(r * in_cols + c).unwrap_or(&0.0));
                    }
                }
                out_v
            } else {
                let s0 = start_indices.first().copied().unwrap_or(0) as usize;
                let st0 = (strides.first().copied().unwrap_or(1) as usize).max(1);
                (0..res_len)
                    .map(|i| *x.get(s0 + i * st0).unwrap_or(&0.0))
                    .collect()
            }
        }
        StableHloOp::Concatenate {
            inputs, dimension, ..
        } => {
            let out_dims = module.values[res_val]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            if out_dims.len() == 2 && dimension == 1 {
                let rows = out_dims[0];
                let in_vecs: Vec<(Vec<f32>, usize)> = inputs
                    .iter()
                    .map(|&inp| {
                        let v = get_v(inp);
                        let c = module.values[inp]
                            .ty
                            .as_tensor()
                            .and_then(|t| tensor_shape_dims(t).get(1).copied())
                            .unwrap_or(1);
                        (v, c)
                    })
                    .collect();
                let mut out_v = Vec::with_capacity(res_len);
                for r in 0..rows {
                    for (v, cols) in &in_vecs {
                        let start = r * cols;
                        out_v.extend_from_slice(&v[start..start + cols]);
                    }
                }
                out_v
            } else {
                let mut out_v = Vec::with_capacity(res_len);
                for inp in inputs {
                    out_v.extend(get_v(inp));
                }
                out_v.truncate(res_len);
                out_v
            }
        }
        StableHloOp::Compare {
            lhs,
            rhs,
            direction,
            ..
        } => get_v(lhs)
            .iter()
            .zip(get_v(rhs).iter())
            .map(|(a, b)| {
                let cond = match direction.as_str() {
                    "EQ" => (a - b).abs() < 1e-7,
                    "NE" => (a - b).abs() >= 1e-7,
                    "LT" => a < b,
                    "LE" => a <= b,
                    "GT" => a > b,
                    "GE" => a >= b,
                    _ => false,
                };
                if cond { 1.0 } else { 0.0 }
            })
            .collect(),
        StableHloOp::Select {
            pred,
            on_true,
            on_false,
            ..
        } => {
            let p = get_v(pred);
            let t = get_v(on_true);
            let f = get_v(on_false);
            (0..res_len)
                .map(|i| {
                    let pv = p.get(i).or_else(|| p.first()).copied().unwrap_or(0.0);
                    if pv != 0.0 { t[i] } else { f[i] }
                })
                .collect()
        }
        StableHloOp::Reduce {
            inputs,
            init_values,
            ..
        } => {
            let inp = get_v(inputs[0]);
            let init = get_v(init_values[0]).first().copied().unwrap_or(0.0);
            let inner = (inp.len() / res_len.max(1)).max(1);
            (0..res_len)
                .map(|o| {
                    let mut acc = init;
                    for k in 0..inner {
                        acc += inp[o * inner + k];
                    }
                    acc
                })
                .collect()
        }
        StableHloOp::DotGeneral {
            lhs,
            rhs,
            dimension_numbers: dims,
            ..
        } => {
            let packed = dims.rhs_contracting_dimensions == [1];
            eval_matmul_2d(module, lhs, rhs, packed, &get_v)
        }
        StableHloOp::Composite { name, inputs, .. } => {
            if name == "castforge.fused_add_max" && inputs.len() == 3 {
                let (a, b, c) = (get_v(inputs[0]), get_v(inputs[1]), get_v(inputs[2]));
                (0..res_len).map(|i| (a[i] + b[i]).max(c[i])).collect()
            } else if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 {
                let packed = op
                    .attributes
                    .get("packed_rhs")
                    .is_some_and(|a| matches!(a, castforge_ir::Attribute::Bool(true)));
                let mm = eval_matmul_2d(module, inputs[0], inputs[1], packed, &get_v);
                let bias = get_v(inputs[2]);
                let epilogue = op
                    .attributes
                    .get("epilogue")
                    .and_then(|a| a.as_str())
                    .unwrap_or("bias");
                let z = inputs
                    .get(3)
                    .map(|&v| get_v(v))
                    .unwrap_or_else(|| vec![0.0; res_len]);
                (0..res_len)
                    .map(|i| {
                        let val = mm[i] + bias[i];
                        match epilogue {
                            "bias_relu" => val.max(z[i]),
                            "bias_tanh" => val.tanh(),
                            "bias_silu" => val / (1.0 + (-val).exp()),
                            "bias_scale" => val * z[i],
                            _ => val,
                        }
                    })
                    .collect()
            } else if name == "castforge.fused_softmax" && !inputs.is_empty() {
                let x = get_v(inputs[0]);
                let dims = module.values[res_val]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![1, res_len]);
                let cols = *dims.last().unwrap_or(&res_len).max(&1);
                let rows = (res_len / cols).max(1);
                let mut out = vec![0.0_f32; res_len];
                for r in 0..rows {
                    let row_slice = &x[r * cols..(r + 1) * cols];
                    let max_v = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0_f32;
                    for c in 0..cols {
                        let e = (row_slice[c] - max_v).exp();
                        out[r * cols + c] = e;
                        sum += e;
                    }
                    for c in 0..cols {
                        out[r * cols + c] /= sum;
                    }
                }
                out
            } else if name == "castforge.fused_rmsnorm" && !inputs.is_empty() {
                let x = get_v(inputs[0]);
                let w_opt = inputs.get(1).map(|&v| get_v(v));
                let eps = match op.attributes.get("eps") {
                    Some(castforge_ir::Attribute::Float(f)) => *f as f32,
                    _ => 1e-5_f32,
                };
                let dims = module.values[res_val]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![1, res_len]);
                let cols = *dims.last().unwrap_or(&res_len).max(&1);
                let rows = (res_len / cols).max(1);
                let mut out = vec![0.0_f32; res_len];
                for r in 0..rows {
                    let mut sq_sum = 0.0_f32;
                    for c in 0..cols {
                        let v = x[r * cols + c];
                        sq_sum += v * v;
                    }
                    let inv_rms = 1.0_f32 / (sq_sum / (cols as f32) + eps).sqrt();
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let w_val = w_opt
                            .as_ref()
                            .and_then(|w| w.get(idx).or_else(|| w.get(c)).copied())
                            .unwrap_or(1.0);
                        out[idx] = x[idx] * inv_rms * w_val;
                    }
                }
                out
            } else if name == "castforge.fused_swiglu" && inputs.len() >= 2 {
                let g = get_v(inputs[0]);
                let u = get_v(inputs[1]);
                (0..res_len)
                    .map(|i| (g[i] / (1.0 + (-g[i]).exp())) * u[i])
                    .collect()
            } else if name == "castforge.fused_rope" && inputs.len() >= 3 {
                let (x, cos_v, sin_v) = (get_v(inputs[0]), get_v(inputs[1]), get_v(inputs[2]));
                let dims = module.values[res_val]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![1, res_len]);
                let cols = *dims.last().unwrap_or(&res_len).max(&2);
                let rows = (res_len / cols).max(1);
                let half = cols / 2;
                let mut out = vec![0.0_f32; res_len];
                for r in 0..rows {
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let rot = if c < half {
                            -x[r * cols + c + half]
                        } else {
                            x[r * cols + c - half]
                        };
                        out[idx] = x[idx] * cos_v[idx] + rot * sin_v[idx];
                    }
                }
                out
            } else if name == "castforge.fused_attention" && inputs.len() >= 3 {
                let scale = match op.attributes.get("scale") {
                    Some(castforge_ir::Attribute::Float(f)) => *f as f32,
                    _ => 1.0_f32,
                };
                let k_transposed = op
                    .attributes
                    .get("k_transposed")
                    .is_some_and(|a| matches!(a, castforge_ir::Attribute::Bool(true)));
                eval_fused_attention_2d(
                    module,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    scale,
                    k_transposed,
                    &get_v,
                )
            } else if name == "castforge.fused_loop_chain" {
                let steps_str = op
                    .attributes
                    .get("steps")
                    .and_then(|a| a.as_str())
                    .unwrap_or("");
                eval_fused_loop_chain(res_len, &inputs, steps_str, &get_v)
            } else {
                get_v(inputs[0])
            }
        }
        _ => vec![0.0; res_len],
    };

    env.insert(res_val, out);
    Ok(())
}

fn eval_matmul_2d(
    module: &Module,
    lhs: ValueId,
    rhs: ValueId,
    packed_rhs: bool,
    get_v: &dyn Fn(ValueId) -> Vec<f32>,
) -> Vec<f32> {
    let a = get_v(lhs);
    let b = get_v(rhs);
    let lhs_dims = module.values[lhs]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, 1]);
    let rhs_dims = module.values[rhs]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, 1]);
    let m = *lhs_dims.first().unwrap_or(&1);
    let k = *lhs_dims.get(1).unwrap_or(&1);
    let n = if packed_rhs {
        *rhs_dims.first().unwrap_or(&1)
    } else {
        *rhs_dims.get(1).unwrap_or(&1)
    };
    let mut out = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                let b_val = if packed_rhs {
                    b[col * k + kk]
                } else {
                    b[kk * n + col]
                };
                acc += a[row * k + kk] * b_val;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn eval_fused_attention_2d(
    module: &Module,
    q_id: ValueId,
    k_id: ValueId,
    v_id: ValueId,
    scale: f32,
    k_transposed: bool,
    get_v: &dyn Fn(ValueId) -> Vec<f32>,
) -> Vec<f32> {
    let q_dims = module.values[q_id]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, 1]);
    let v_dims = module.values[v_id]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, 1]);
    let seq_q = *q_dims.first().unwrap_or(&1);
    let seq_kv = *v_dims.first().unwrap_or(&1);
    let d_v = *v_dims.get(1).unwrap_or(&1);

    let qk = eval_matmul_2d(module, q_id, k_id, k_transposed, get_v);
    let v = get_v(v_id);
    let mut out = vec![0.0_f32; seq_q * d_v];

    for r in 0..seq_q {
        let row_scores = &qk[r * seq_kv..(r + 1) * seq_kv];
        let max_s = row_scores
            .iter()
            .map(|&s| s * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0_f32; seq_kv];
        let mut sum = 0.0_f32;
        for (c, &s) in row_scores.iter().enumerate() {
            let e = (s * scale - max_s).exp();
            probs[c] = e;
            sum += e;
        }
        for c in 0..seq_kv {
            let p = probs[c] / sum;
            for d in 0..d_v {
                out[r * d_v + d] += p * v[c * d_v + d];
            }
        }
    }
    out
}

fn eval_fused_loop_chain(
    res_len: usize,
    inputs: &[ValueId],
    steps_str: &str,
    get_v: &dyn Fn(ValueId) -> Vec<f32>,
) -> Vec<f32> {
    let ext: Vec<Vec<f32>> = inputs.iter().map(|&v| get_v(v)).collect();
    let specs: Vec<&str> = steps_str.split(';').filter(|s| !s.is_empty()).collect();
    let mut out = vec![0.0_f32; res_len];

    for (i, out_elem) in out.iter_mut().enumerate() {
        let mut regs: Vec<f32> = Vec::with_capacity(specs.len());
        for spec in &specs {
            let Some((op_name, rest)) = spec.split_once('(') else {
                continue;
            };
            let args_part = rest.trim_end_matches(')');
            let args: Vec<f32> = args_part
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|tok| {
                    if let Some(step_idx) = tok.strip_prefix('s') {
                        step_idx
                            .parse::<usize>()
                            .ok()
                            .and_then(|k| regs.get(k).copied())
                            .unwrap_or(0.0)
                    } else {
                        tok.parse::<usize>()
                            .ok()
                            .and_then(|k| ext.get(k))
                            .and_then(|v| v.get(i).copied())
                            .unwrap_or(0.0)
                    }
                })
                .collect();
            let a0 = args.first().copied().unwrap_or(0.0);
            let a1 = args.get(1).copied().unwrap_or(0.0);
            let val = match op_name.trim() {
                "add" => a0 + a1,
                "subtract" => a0 - a1,
                "multiply" => a0 * a1,
                "divide" => {
                    if a1.abs() > 1e-12 {
                        a0 / a1
                    } else {
                        0.0
                    }
                }
                "maximum" => a0.max(a1),
                "minimum" => a0.min(a1),
                "exponential" => a0.exp(),
                "tanh" => a0.tanh(),
                "rsqrt" => 1.0 / a0.sqrt(),
                _ => a0,
            };
            regs.push(val);
        }
        *out_elem = regs.last().copied().unwrap_or(0.0);
    }
    out
}
