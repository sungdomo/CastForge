//! Reference IR interpreter (`evaluate_module`, `evaluate_function`) and numerical verification /
//! pass bisection engine (`verify_module_numerics`, `bisect_module_snapshots`).

use crate::codegen::{tensor_len, tensor_shape_dims};
use crate::ir::{DType, Module, OpId, ValueId};
use crate::stablehlo::StableHloOp;
use anyhow::{Result, anyhow};
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
    compare_output_tensors(&out_before, &out_after, tolerance)
}

/// Compares two sets of evaluated output tensors and returns a [`NumericVerificationReport`].
pub fn compare_output_tensors(
    out_before: &[Vec<f32>],
    out_after: &[Vec<f32>],
    tolerance: f32,
) -> Result<NumericVerificationReport> {
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

/// Numerical verification result for a single step in [`bisect_module_snapshots`].
#[derive(Debug, Clone)]
pub struct BisectStep {
    /// Step index (`0..`).
    pub step_index: usize,
    /// Pass name executed at this step.
    pub pass_name: String,
    /// Maximum absolute numerical difference against the baseline `Module`.
    pub max_abs_diff: f32,
    /// Whether this step stayed within `tolerance`.
    pub passed: bool,
}

/// Complete bisection report across a sequence of pass snapshots.
#[derive(Debug, Clone)]
pub struct BisectReport {
    /// Per-step numerical comparison results.
    pub steps: Vec<BisectStep>,
    /// Name of the first pass that exceeded `tolerance`, if any.
    pub first_failing_pass: Option<String>,
}

/// Evaluates a sequence of `(pass_name, module_snapshot)` pairs against `baseline` to pinpoint
/// the exact pass where numerical drift exceeds `tolerance`.
///
/// Evaluates `baseline` once up front and reuses its reference output across all snapshots.
pub fn bisect_module_snapshots(
    baseline: &Module,
    snapshots: &[(String, Module)],
    tolerance: f32,
) -> Result<BisectReport> {
    let inputs = generate_synthetic_inputs(baseline, None)?;
    let out_base = evaluate_module(baseline, &inputs)?;
    let mut steps = Vec::with_capacity(snapshots.len());
    let mut first_failing_pass = None;
    for (idx, (pass_name, snap)) in snapshots.iter().enumerate() {
        let out_snap = evaluate_module(snap, &inputs)?;
        let rep = compare_output_tensors(&out_base, &out_snap, tolerance)?;
        if !rep.passed && first_failing_pass.is_none() {
            first_failing_pass = Some(pass_name.clone());
        }
        steps.push(BisectStep {
            step_index: idx,
            pass_name: pass_name.clone(),
            max_abs_diff: rep.max_abs_diff,
            passed: rep.passed,
        });
    }
    Ok(BisectReport {
        steps,
        first_failing_pass,
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
    evaluate_region(module, reg_id, inputs, &mut env, &mut q_env)
}

fn evaluate_region(
    module: &Module,
    reg_id: crate::ir::RegionId,
    inputs: &[Vec<f32>],
    env: &mut HashMap<ValueId, Vec<f32>>,
    q_env: &mut HashMap<ValueId, Vec<i8>>,
) -> Result<Vec<Vec<f32>>> {
    let blk_id = module.regions[reg_id].blocks[0];
    let block = &module.blocks[blk_id];

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

        if op.is_op("stablehlo", "while") && op.regions.len() >= 2 {
            let cond_reg = op.regions[0];
            let body_reg = op.regions[1];
            let mut state: Vec<Vec<f32>> = op
                .operands
                .iter()
                .map(|&v| {
                    env.get(&v)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; tensor_len(module, v)])
                })
                .collect();
            for _step in 0..64 {
                let cond_out = evaluate_region(module, cond_reg, &state, env, q_env)?;
                let keep_going = cond_out
                    .first()
                    .and_then(|v| v.first())
                    .is_some_and(|&c| c > 0.5);
                if !keep_going {
                    break;
                }
                state = evaluate_region(module, body_reg, &state, env, q_env)?;
            }
            for (&res_id, vec_val) in op.results.iter().zip(state) {
                env.insert(res_id, vec_val);
            }
            continue;
        }

        if op.is_op("stablehlo", "case") && !op.regions.is_empty() && !op.operands.is_empty() {
            let idx_val = env
                .get(&op.operands[0])
                .and_then(|v| v.first().copied())
                .unwrap_or(0.0);
            let branch_idx = if idx_val < 0.0 {
                op.regions.len() - 1
            } else {
                (idx_val.round() as usize).min(op.regions.len() - 1)
            };
            let branch_outs = evaluate_region(module, op.regions[branch_idx], &[], env, q_env)?;
            for (&res_id, vec_val) in op.results.iter().zip(branch_outs) {
                env.insert(res_id, vec_val);
            }
            continue;
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
            if env.contains_key(&res_val) {
                continue;
            }
            let t_ty = module.values[res_val]
                .ty
                .as_tensor()
                .ok_or_else(|| anyhow!("constant without tensor type"))?;
            let val_attr = op.attributes.get("value");
            if let DType::QuantUniform {
                scale, zero_point, ..
            } = &t_ty.dtype
            {
                let q_vals = val_attr
                    .and_then(|a| module.tensor_store.resolve_i8_attr(a, res_len))
                    .unwrap_or_else(|| vec![0; res_len]);
                let sc = scale.parse::<f32>().unwrap_or(1.0);
                let zp = *zero_point;
                let deq: Vec<f32> = q_vals
                    .iter()
                    .map(|&q| (q as i32 - zp as i32) as f32 * sc)
                    .collect();
                q_env.insert(res_val, q_vals);
                env.insert(res_val, deq);
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

        eval_stablehlo_op(module, op_id, res_val, res_len, env)?;
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

    if op.is_op("stablehlo", "log") && !op.operands.is_empty() {
        let out: Vec<f32> = get_v(op.operands[0]).into_iter().map(|x| x.ln()).collect();
        env.insert(res_val, out);
        return Ok(());
    }
    let hlo_opt = StableHloOp::from_op(op).or_else(|| {
        if op.name.dialect == "castforge" {
            Some(StableHloOp::Composite {
                name: format!("castforge.{}", op.name.name),
                decomposition: String::new(),
                inputs: op.operands.to_vec(),
                results: op.results.to_vec(),
            })
        } else {
            None
        }
    });
    let Some(hlo) = hlo_opt else {
        env.insert(res_val, vec![0.0; res_len]);
        return Ok(());
    };

    let eval_bin = |lhs: ValueId, rhs: ValueId, f: fn(f32, f32) -> f32| -> Vec<f32> {
        let va = get_v(lhs);
        let vb = get_v(rhs);
        let la = va.len().max(1);
        let lb = vb.len().max(1);
        (0..res_len)
            .map(|i| {
                let a = va.get(i % la).copied().unwrap_or(0.0);
                let b = vb.get(i % lb).copied().unwrap_or(0.0);
                f(a, b)
            })
            .collect()
    };

    let out: Vec<f32> = match hlo {
        StableHloOp::Add { lhs, rhs, .. } => eval_bin(lhs, rhs, |a, b| a + b),
        StableHloOp::Subtract { lhs, rhs, .. } => eval_bin(lhs, rhs, |a, b| a - b),
        StableHloOp::Multiply { lhs, rhs, .. } => eval_bin(lhs, rhs, |a, b| a * b),
        StableHloOp::Divide { lhs, rhs, .. } => {
            eval_bin(lhs, rhs, |a, b| if b.abs() > 1e-12 { a / b } else { 0.0 })
        }
        StableHloOp::Maximum { lhs, rhs, .. } => eval_bin(lhs, rhs, f32::max),
        StableHloOp::Minimum { lhs, rhs, .. } => eval_bin(lhs, rhs, f32::min),
        StableHloOp::Exp { operand, .. } => get_v(operand).into_iter().map(|x| x.exp()).collect(),
        StableHloOp::Tanh { operand, .. } => get_v(operand).into_iter().map(|x| x.tanh()).collect(),
        StableHloOp::Rsqrt { operand, .. } => {
            get_v(operand).into_iter().map(|x| 1.0 / x.sqrt()).collect()
        }
        StableHloOp::Sqrt { operand, .. } => get_v(operand).into_iter().map(|x| x.sqrt()).collect(),
        StableHloOp::Log { operand, .. } => get_v(operand).into_iter().map(|x| x.ln()).collect(),
        StableHloOp::Sine { operand, .. } => get_v(operand).into_iter().map(|x| x.sin()).collect(),
        StableHloOp::Cosine { operand, .. } => {
            get_v(operand).into_iter().map(|x| x.cos()).collect()
        }
        StableHloOp::Abs { operand, .. } => get_v(operand).into_iter().map(|x| x.abs()).collect(),
        StableHloOp::Negate { operand, .. } => get_v(operand).into_iter().map(|x| -x).collect(),
        StableHloOp::Logistic { operand, .. } => get_v(operand)
            .into_iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect(),
        StableHloOp::Power { lhs, rhs, .. } => eval_bin(lhs, rhs, f32::powf),
        StableHloOp::Clamp {
            min, operand, max, ..
        } => {
            let mn = get_v(min);
            let x = get_v(operand);
            let mx = get_v(max);
            (0..res_len)
                .map(|i| {
                    let lo = mn.get(i).or_else(|| mn.first()).copied().unwrap_or(0.0);
                    let hi = mx.get(i).or_else(|| mx.first()).copied().unwrap_or(0.0);
                    x.get(i).copied().unwrap_or(0.0).clamp(lo, hi)
                })
                .collect()
        }
        StableHloOp::Pad {
            operand,
            padding_value,
            edge_padding_low,
            edge_padding_high,
            interior_padding,
            ..
        } => {
            let x = get_v(operand);
            let pad_v = get_v(padding_value).first().copied().unwrap_or(0.0);
            let in_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![x.len()]);
            let out_dims = module.values[res_val]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            let mut out_v = vec![pad_v; res_len];
            if in_dims.len() == 2 && out_dims.len() == 2 {
                let (in_r, in_c) = (in_dims[0], in_dims[1]);
                let out_c = out_dims[1];
                let lo0 = edge_padding_low.first().copied().unwrap_or(0) as isize;
                let lo1 = edge_padding_low.get(1).copied().unwrap_or(0) as isize;
                let ip0 = interior_padding.first().copied().unwrap_or(0).max(0) as isize + 1;
                let ip1 = interior_padding.get(1).copied().unwrap_or(0).max(0) as isize + 1;
                for r in 0..in_r {
                    for c in 0..in_c {
                        let dr = lo0 + (r as isize) * ip0;
                        let dc = lo1 + (c as isize) * ip1;
                        if dr >= 0
                            && (dr as usize) < out_dims[0]
                            && dc >= 0
                            && (dc as usize) < out_c
                        {
                            out_v[(dr as usize) * out_c + (dc as usize)] = x[r * in_c + c];
                        }
                    }
                }
            } else {
                let _ = edge_padding_high;
                let lo0 = edge_padding_low.first().copied().unwrap_or(0).max(0) as usize;
                let ip0 = interior_padding.first().copied().unwrap_or(0).max(0) as usize + 1;
                for (i, &val) in x.iter().enumerate() {
                    let dst_i = lo0 + i * ip0;
                    if dst_i < out_v.len() {
                        out_v[dst_i] = val;
                    }
                }
            }
            out_v
        }
        StableHloOp::Gather {
            operand,
            start_indices,
            ..
        } => {
            let table = get_v(operand);
            let indices = get_v(start_indices);
            let tbl_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![table.len(), 1]);
            let num_rows = (*tbl_dims.first().unwrap_or(&1)).max(1);
            let row_width = (table.len() / num_rows).max(1);
            let mut out_v = Vec::with_capacity(res_len);
            for &raw_idx in &indices {
                let row = (raw_idx.round() as isize).clamp(0, (num_rows as isize) - 1) as usize;
                let start = row * row_width;
                for c in 0..row_width {
                    if out_v.len() < res_len {
                        out_v.push(*table.get(start + c).unwrap_or(&0.0));
                    }
                }
            }
            out_v.resize(res_len, 0.0);
            out_v
        }
        StableHloOp::Scatter {
            operand,
            scatter_indices,
            updates,
            ..
        } => {
            let mut out_v = get_v(operand);
            out_v.resize(res_len, 0.0);
            let indices = get_v(scatter_indices);
            let upd = get_v(updates);
            let dst_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len, 1]);
            let num_rows = (*dst_dims.first().unwrap_or(&1)).max(1);
            let row_width = (res_len / num_rows).max(1);
            for (k, &raw_idx) in indices.iter().enumerate() {
                let row = (raw_idx.round() as isize).clamp(0, (num_rows as isize) - 1) as usize;
                let dst_base = row * row_width;
                let src_base = k * row_width;
                for c in 0..row_width {
                    if let (Some(dst_elem), Some(&u_val)) =
                        (out_v.get_mut(dst_base + c), upd.get(src_base + c))
                    {
                        *dst_elem = u_val;
                    }
                }
            }
            out_v
        }
        StableHloOp::DynamicSlice {
            operand,
            start_indices,
            slice_sizes,
            ..
        } => {
            let x = get_v(operand);
            let in_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![x.len()]);
            let starts: Vec<usize> = start_indices
                .iter()
                .enumerate()
                .map(|(axis, &vid)| {
                    let raw = get_v(vid).first().copied().unwrap_or(0.0).round() as isize;
                    let dim_len = in_dims.get(axis).copied().unwrap_or(1) as isize;
                    let sz = slice_sizes.get(axis).copied().unwrap_or(1) as isize;
                    raw.clamp(0, (dim_len - sz).max(0)) as usize
                })
                .collect();
            if in_dims.len() == 2 && starts.len() >= 2 && slice_sizes.len() >= 2 {
                let (s0, s1) = (starts[0], starts[1]);
                let (sz0, sz1) = (slice_sizes[0] as usize, slice_sizes[1] as usize);
                let in_cols = in_dims[1];
                let mut out_v = Vec::with_capacity(res_len);
                for r in s0..(s0 + sz0) {
                    for c in s1..(s1 + sz1) {
                        out_v.push(*x.get(r * in_cols + c).unwrap_or(&0.0));
                    }
                }
                out_v.resize(res_len, 0.0);
                out_v
            } else {
                let s0 = starts.first().copied().unwrap_or(0);
                (0..res_len)
                    .map(|i| *x.get(s0 + i).unwrap_or(&0.0))
                    .collect()
            }
        }
        StableHloOp::DynamicUpdateSlice {
            operand,
            update,
            start_indices,
            ..
        } => {
            let mut out_v = get_v(operand);
            out_v.resize(res_len, 0.0);
            let upd = get_v(update);
            let in_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            let upd_dims = module.values[update]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![upd.len()]);
            let starts: Vec<usize> = start_indices
                .iter()
                .enumerate()
                .map(|(axis, &vid)| {
                    let raw = get_v(vid).first().copied().unwrap_or(0.0).round() as isize;
                    let dim_len = in_dims.get(axis).copied().unwrap_or(1) as isize;
                    let sz = upd_dims.get(axis).copied().unwrap_or(1) as isize;
                    raw.clamp(0, (dim_len - sz).max(0)) as usize
                })
                .collect();
            if in_dims.len() == 2 && upd_dims.len() == 2 && starts.len() >= 2 {
                let (s0, s1) = (starts[0], starts[1]);
                let (u_r, u_c) = (upd_dims[0], upd_dims[1]);
                let in_cols = in_dims[1];
                for r in 0..u_r {
                    for c in 0..u_c {
                        let dst_idx = (s0 + r) * in_cols + (s1 + c);
                        if let (Some(d), Some(&u)) = (out_v.get_mut(dst_idx), upd.get(r * u_c + c))
                        {
                            *d = u;
                        }
                    }
                }
            } else {
                let s0 = starts.first().copied().unwrap_or(0);
                for (i, &u) in upd.iter().enumerate() {
                    if let Some(d) = out_v.get_mut(s0 + i) {
                        *d = u;
                    }
                }
            }
            out_v
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
            let out_dims = module.values[res_val]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            let rank = in_dims.len();
            if rank >= 2 && permutation.len() == rank && out_dims.len() == rank {
                let mut in_strides = vec![1usize; rank];
                let mut out_strides = vec![1usize; rank];
                for d in (0..rank - 1).rev() {
                    in_strides[d] = in_strides[d + 1] * in_dims[d + 1].max(1);
                    out_strides[d] = out_strides[d + 1] * out_dims[d + 1].max(1);
                }
                let mut t = vec![0.0_f32; res_len];
                for (out_idx, dst) in t.iter_mut().enumerate() {
                    let mut rem = out_idx;
                    let mut in_idx = 0usize;
                    for (out_axis, &in_axis) in permutation.iter().enumerate() {
                        let coord = rem / out_strides[out_axis];
                        rem %= out_strides[out_axis];
                        if let Some(&st) = in_strides.get(in_axis as usize) {
                            in_idx += coord * st;
                        }
                    }
                    *dst = *x.get(in_idx).unwrap_or(&0.0);
                }
                t
            } else {
                x
            }
        }
        StableHloOp::BroadcastInDim {
            operand,
            broadcast_dimensions,
            ..
        } => {
            let x = get_v(operand);
            let in_dims = module.values[operand]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![x.len().max(1)]);
            let out_dims = module.values[res_val]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            if !broadcast_dimensions.is_empty()
                && broadcast_dimensions.len() == in_dims.len()
                && out_dims.len() >= in_dims.len()
            {
                let out_rank = out_dims.len();
                let in_rank = in_dims.len();
                let mut out_strides = vec![1usize; out_rank];
                for d in (0..out_rank.saturating_sub(1)).rev() {
                    out_strides[d] = out_strides[d + 1] * out_dims[d + 1].max(1);
                }
                let mut in_strides = vec![1usize; in_rank];
                for d in (0..in_rank.saturating_sub(1)).rev() {
                    in_strides[d] = in_strides[d + 1] * in_dims[d + 1].max(1);
                }
                (0..res_len)
                    .map(|out_idx| {
                        let mut rem = out_idx;
                        let mut out_coords = vec![0usize; out_rank];
                        for d in 0..out_rank {
                            out_coords[d] = rem / out_strides[d];
                            rem %= out_strides[d];
                        }
                        let mut in_idx = 0usize;
                        for (in_axis, &out_axis) in broadcast_dimensions.iter().enumerate() {
                            let c = out_coords.get(out_axis as usize).copied().unwrap_or(0);
                            let in_c = if in_dims[in_axis] <= 1 { 0 } else { c };
                            in_idx += in_c * in_strides[in_axis];
                        }
                        *x.get(in_idx).unwrap_or(&0.0)
                    })
                    .collect()
            } else {
                let in_len = x.len().max(1);
                (0..res_len)
                    .map(|i| x[i / (res_len / in_len).max(1) % in_len])
                    .collect()
            }
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
            let dim = (dimension.max(0) as usize).min(out_dims.len().saturating_sub(1));
            let outer: usize = out_dims.iter().take(dim).product::<usize>().max(1);
            let out_cat_stride: usize = out_dims.iter().skip(dim).product::<usize>().max(1);
            let inner: usize = out_dims.iter().skip(dim + 1).product::<usize>().max(1);
            let mut out_v = vec![0.0_f32; res_len];
            let mut cat_offset = 0usize;
            for &inp in &inputs {
                let v = get_v(inp);
                let in_dims = module.values[inp]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![v.len().max(1)]);
                let in_cat = (*in_dims.get(dim).unwrap_or(&1)).max(1);
                let chunk = in_cat * inner;
                for o in 0..outer {
                    for c in 0..chunk {
                        let dst_idx = o * out_cat_stride + cat_offset + c;
                        if let Some(slot) = out_v.get_mut(dst_idx) {
                            *slot = *v.get(o * chunk + c).unwrap_or(&0.0);
                        }
                    }
                }
                cat_offset += chunk;
            }
            out_v
        }
        StableHloOp::Compare {
            lhs,
            rhs,
            direction,
            ..
        } => {
            let lv = get_v(lhs);
            let rv = get_v(rhs);
            let l_len = lv.len().max(1);
            let r_len = rv.len().max(1);
            (0..res_len)
                .map(|i| {
                    let a = lv.get(i % l_len).copied().unwrap_or(0.0);
                    let b = rv.get(i % r_len).copied().unwrap_or(0.0);
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
                .collect()
        }
        StableHloOp::Select {
            pred,
            on_true,
            on_false,
            ..
        } => {
            let p = get_v(pred);
            let t = get_v(on_true);
            let f = get_v(on_false);
            let p_len = p.len().max(1);
            let t_len = t.len().max(1);
            let f_len = f.len().max(1);
            (0..res_len)
                .map(|i| {
                    let pv = p.get(i % p_len).copied().unwrap_or(0.0);
                    let tv = t.get(i % t_len).copied().unwrap_or(0.0);
                    let fv = f.get(i % f_len).copied().unwrap_or(0.0);
                    if pv != 0.0 { tv } else { fv }
                })
                .collect()
        }
        StableHloOp::Reduce {
            inputs,
            init_values,
            dimensions,
            reducer_op,
            ..
        } => {
            let inp = get_v(inputs[0]);
            let in_dims = module.values[inputs[0]]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![inp.len().max(1)]);
            let init = init_values
                .first()
                .and_then(|&v| get_v(v).first().copied())
                .unwrap_or(0.0);
            let inner = (inp.len() / res_len.max(1)).max(1);
            let reduce_dim0 = in_dims.len() == 2 && dimensions.as_slice() == [0];
            let red_op = reducer_op.as_deref().unwrap_or("add");
            (0..res_len)
                .map(|o| {
                    let mut acc = init;
                    for k in 0..inner {
                        let idx = if reduce_dim0 {
                            k * res_len + o
                        } else {
                            o * inner + k
                        };
                        let v = *inp.get(idx).unwrap_or(&0.0);
                        if red_op.contains("maximum") || red_op.contains("max") {
                            acc = acc.max(v);
                        } else if red_op.contains("minimum") || red_op.contains("min") {
                            acc = acc.min(v);
                        } else if red_op.contains("multiply") || red_op.contains("mul") {
                            acc *= v;
                        } else {
                            acc += v;
                        }
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
            let packed = dims
                .rhs_contracting_dimensions
                .last()
                .copied()
                .is_some_and(|d| {
                    let rank = module.values[rhs]
                        .ty
                        .as_tensor()
                        .map(|t| t.rank())
                        .unwrap_or(2);
                    d as usize == rank.saturating_sub(1) && rank == 2
                });
            eval_dot_general(module, lhs, rhs, res_val, &dims, packed, &get_v)
        }
        StableHloOp::Convolution { lhs, rhs, .. } => {
            eval_conv_2d(module, lhs, rhs, res_len, &get_v)
        }
        StableHloOp::Composite { name, inputs, .. } => {
            if name == "castforge.fused_add_max" && inputs.len() == 3 {
                let (a, b, c) = (get_v(inputs[0]), get_v(inputs[1]), get_v(inputs[2]));
                (0..res_len).map(|i| (a[i] + b[i]).max(c[i])).collect()
            } else if name == "castforge.fused_matmul_epilogue" && inputs.len() >= 3 {
                let packed = op
                    .attributes
                    .get("packed_rhs")
                    .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
                let mm = eval_matmul_2d(module, inputs[0], inputs[1], packed, &get_v);
                let bias = get_v(inputs[2]);
                let epilogue = op
                    .attributes
                    .get("epilogue")
                    .and_then(|a| a.as_str())
                    .unwrap_or("bias");
                let (base_ep, has_residual) = if let Some(base) = epilogue.strip_suffix("_residual")
                {
                    (base, true)
                } else {
                    (epilogue, false)
                };
                let z = inputs
                    .get(3)
                    .map(|&v| get_v(v))
                    .unwrap_or_else(|| vec![0.0; res_len]);
                let res_vec = if has_residual {
                    inputs.last().map(|&v| get_v(v))
                } else {
                    None
                };
                let b_len = bias.len().max(1);
                let z_len = z.len().max(1);
                (0..res_len)
                    .map(|i| {
                        let val = mm[i] + bias[i % b_len];
                        let act = match base_ep {
                            "bias_relu" | "relu" => val.max(z[i % z_len]),
                            "bias_tanh" | "tanh" => val.tanh(),
                            "bias_silu" | "silu" => val / (1.0 + (-val).exp()),
                            "bias_gelu" | "gelu" => eval_gelu_scalar(val),
                            "bias_scale" => val * z[i % z_len],
                            _ => val,
                        };
                        if let Some(rv) = &res_vec {
                            act + rv[i % rv.len().max(1)]
                        } else {
                            act
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
                    Some(crate::ir::Attribute::Float(f)) => *f as f32,
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
                    Some(crate::ir::Attribute::Float(f)) => *f as f32,
                    _ => 1.0_f32,
                };
                let k_transposed = op
                    .attributes
                    .get("k_transposed")
                    .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
                eval_fused_attention_2d(
                    module,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs.get(3).copied(),
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
            } else if name == "castforge.fused_qmatmul" && inputs.len() >= 2 {
                let packed = op
                    .attributes
                    .get("packed_rhs")
                    .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
                let mm = eval_matmul_2d(module, inputs[0], inputs[1], packed, &get_v);
                if inputs.len() >= 3 {
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
                    let b_len = bias.len().max(1);
                    let z_len = z.len().max(1);
                    (0..res_len)
                        .map(|i| {
                            let val = mm[i] + bias[i % b_len];
                            match epilogue {
                                "bias_relu" | "relu" => val.max(z[i % z_len]),
                                "bias_tanh" | "tanh" => val.tanh(),
                                "bias_silu" | "silu" => val / (1.0 + (-val).exp()),
                                "bias_gelu" | "gelu" => eval_gelu_scalar(val),
                                "bias_scale" => val * z[i % z_len],
                                _ => val,
                            }
                        })
                        .collect()
                } else {
                    mm
                }
            } else if name == "castforge.fused_residual_rmsnorm" && inputs.len() >= 2 {
                let (x, res_in) = (get_v(inputs[0]), get_v(inputs[1]));
                let w_opt = inputs.get(2).map(|&v| get_v(v));
                let eps = match op.attributes.get("eps") {
                    Some(crate::ir::Attribute::Float(f)) => *f as f32,
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
                        let idx = r * cols + c;
                        let h = x[idx] + res_in[idx];
                        sq_sum += h * h;
                    }
                    let inv_rms = 1.0_f32 / (sq_sum / (cols as f32) + eps).sqrt();
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let h = x[idx] + res_in[idx];
                        let w_val = w_opt
                            .as_ref()
                            .and_then(|w| w.get(idx).or_else(|| w.get(c)).copied())
                            .unwrap_or(1.0);
                        out[idx] = h * inv_rms * w_val;
                    }
                }
                out
            } else if name == "castforge.fused_layernorm" && !inputs.is_empty() {
                let x = get_v(inputs[0]);
                let g_opt = inputs.get(1).map(|&v| get_v(v));
                let b_opt = inputs.get(2).map(|&v| get_v(v));
                let eps = match op.attributes.get("eps") {
                    Some(crate::ir::Attribute::Float(f)) => *f as f32,
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
                    let mut sum = 0.0_f32;
                    for c in 0..cols {
                        sum += x[r * cols + c];
                    }
                    let mean = sum / (cols as f32);
                    let mut var_sum = 0.0_f32;
                    for c in 0..cols {
                        let d = x[r * cols + c] - mean;
                        var_sum += d * d;
                    }
                    let inv_std = 1.0_f32 / (var_sum / (cols as f32) + eps).sqrt();
                    for c in 0..cols {
                        let idx = r * cols + c;
                        let g_v = g_opt
                            .as_ref()
                            .and_then(|g| g.get(idx).or_else(|| g.get(c)).copied())
                            .unwrap_or(1.0);
                        let b_v = b_opt
                            .as_ref()
                            .and_then(|b| b.get(idx).or_else(|| b.get(c)).copied())
                            .unwrap_or(0.0);
                        out[idx] = (x[idx] - mean) * inv_std * g_v + b_v;
                    }
                }
                out
            } else if name == "castforge.fused_gelu" && !inputs.is_empty() {
                let x = get_v(inputs[0]);
                x.into_iter().map(eval_gelu_scalar).collect()
            } else if name == "castforge.fused_geglu" && inputs.len() >= 2 {
                let (g, u) = (get_v(inputs[0]), get_v(inputs[1]));
                (0..res_len)
                    .map(|i| eval_gelu_scalar(g[i]) * u[i])
                    .collect()
            } else if name == "castforge.fused_gated_mlp" && inputs.len() >= 2 {
                let packed = op
                    .attributes
                    .get("packed_rhs")
                    .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
                let act = op
                    .attributes
                    .get("activation")
                    .and_then(|a| a.as_str())
                    .unwrap_or("swiglu");
                let wide = eval_matmul_2d(module, inputs[0], inputs[1], packed, &get_v);
                let out_dims = module.values[res_val]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![1, res_len]);
                let n = *out_dims.last().unwrap_or(&res_len).max(&1);
                let m = (res_len / n).max(1);
                let mut out = vec![0.0_f32; res_len];
                for r in 0..m {
                    for c in 0..n {
                        let gate = wide[r * (2 * n) + c];
                        let up = wide[r * (2 * n) + n + c];
                        let g_act = if act == "geglu" {
                            eval_gelu_scalar(gate)
                        } else {
                            gate / (1.0 + (-gate).exp())
                        };
                        out[r * n + c] = g_act * up;
                    }
                }
                out
            } else if name == "castforge.fused_map_reduce" && inputs.len() >= 2 {
                let x = get_v(inputs[0]);
                let init = get_v(inputs[1]).first().copied().unwrap_or(0.0);
                let map_op = op
                    .attributes
                    .get("map_op")
                    .and_then(|a| a.as_str())
                    .unwrap_or("exponential");
                let inner = (x.len() / res_len.max(1)).max(1);
                (0..res_len)
                    .map(|o| {
                        let mut acc = init;
                        for k in 0..inner {
                            let v = x[o * inner + k];
                            let mv = match map_op {
                                "square" => v * v,
                                "exponential" => v.exp(),
                                "tanh" => v.tanh(),
                                "rsqrt" => 1.0 / v.sqrt(),
                                _ => v,
                            };
                            acc += mv;
                        }
                        acc
                    })
                    .collect()
            } else if name == "castforge.fused_log_softmax" && !inputs.is_empty() {
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
                    for &v in row_slice {
                        sum += (v - max_v).exp();
                    }
                    let lse = max_v + sum.ln();
                    for c in 0..cols {
                        out[r * cols + c] = row_slice[c] - lse;
                    }
                }
                out
            } else if name == "castforge.im2col" && !inputs.is_empty() {
                let get_i = |k: &str, def: usize| -> usize {
                    match op.attributes.get(k) {
                        Some(crate::ir::Attribute::Int(v)) if *v > 0 => *v as usize,
                        _ => def,
                    }
                };
                let h_in = get_i("h_in", 1);
                let w_in = get_i("w_in", 1);
                let c_in = get_i("c_in", 1);
                let kh = get_i("kh", 1);
                let kw = get_i("kw", 1);
                eval_im2col_raw(&get_v(inputs[0]), h_in, w_in, c_in, kh, kw, res_len)
            } else if name == "castforge.fma" && inputs.len() >= 3 {
                let (a, b, c) = (get_v(inputs[0]), get_v(inputs[1]), get_v(inputs[2]));
                (0..res_len).map(|i| a[i].mul_add(b[i], c[i])).collect()
            } else if name == "castforge.fused_moe" && inputs.len() >= 5 {
                let e0 = eval_matmul_2d(module, inputs[0], inputs[1], false, &get_v);
                let e1 = eval_matmul_2d(module, inputs[0], inputs[2], false, &get_v);
                let g0 = get_v(inputs[3]);
                let g1 = get_v(inputs[4]);
                (0..res_len)
                    .map(|i| e0[i] * g0[i % g0.len()] + e1[i] * g1[i % g1.len()])
                    .collect()
            } else if name == "castforge.static_quantize" && !inputs.is_empty() {
                let sc = match op.attributes.get("scale") {
                    Some(crate::ir::Attribute::Float(f)) if *f > 0.0 => *f as f32,
                    _ => 1.0 / 127.0,
                };
                let bits = match op.attributes.get("bits") {
                    Some(crate::ir::Attribute::Int(b)) => *b,
                    _ => 8,
                };
                let qmax = if bits == 4 { 7.0_f32 } else { 127.0_f32 };
                get_v(inputs[0])
                    .into_iter()
                    .map(|x| (x / sc).round().clamp(-qmax, qmax) * sc)
                    .collect()
            } else if name == "castforge.nan_guard" && !inputs.is_empty() {
                let clamp_max = match op.attributes.get("clamp_max") {
                    Some(crate::ir::Attribute::Float(f)) if *f > 0.0 => *f as f32,
                    _ => 65504.0_f32,
                };
                get_v(inputs[0])
                    .into_iter()
                    .map(|x| {
                        if x.is_nan() {
                            0.0_f32
                        } else {
                            x.clamp(-clamp_max, clamp_max)
                        }
                    })
                    .collect()
            } else {
                get_v(inputs[0])
            }
        }
        StableHloOp::Iota { iota_dimension, .. } => {
            let dims = module.values[res_val]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![res_len]);
            let d = (iota_dimension.max(0) as usize).min(dims.len().saturating_sub(1));
            let dim_size = (*dims.get(d).unwrap_or(&res_len)).max(1);
            let inner: usize = dims.iter().skip(d + 1).product::<usize>().max(1);
            (0..res_len)
                .map(|i| ((i / inner) % dim_size) as f32)
                .collect()
        }
        StableHloOp::CustomCall { inputs, .. } => inputs
            .first()
            .map(|&v| get_v(v))
            .unwrap_or_else(|| vec![0.0; res_len]),
        StableHloOp::Other { op_name } => match op_name.as_str() {
            "negate" if !op.operands.is_empty() => {
                get_v(op.operands[0]).into_iter().map(|x| -x).collect()
            }
            "abs" if !op.operands.is_empty() => {
                get_v(op.operands[0]).into_iter().map(|x| x.abs()).collect()
            }
            "cosine" if !op.operands.is_empty() => {
                get_v(op.operands[0]).into_iter().map(|x| x.cos()).collect()
            }
            "sine" if !op.operands.is_empty() => {
                get_v(op.operands[0]).into_iter().map(|x| x.sin()).collect()
            }
            "clamp" if op.operands.len() >= 3 => {
                let mn = get_v(op.operands[0]);
                let x = get_v(op.operands[1]);
                let mx = get_v(op.operands[2]);
                (0..res_len)
                    .map(|i| {
                        let lo = mn.get(i).or_else(|| mn.first()).copied().unwrap_or(0.0);
                        let hi = mx.get(i).or_else(|| mx.first()).copied().unwrap_or(0.0);
                        x[i].clamp(lo, hi)
                    })
                    .collect()
            }
            "reverse" if !op.operands.is_empty() => {
                let mut v = get_v(op.operands[0]);
                v.reverse();
                v
            }
            "all_reduce" if !op.operands.is_empty() => get_v(op.operands[0]),
            _ => op
                .operands
                .first()
                .map(|&v| {
                    let mut out = get_v(v);
                    out.resize(res_len, 0.0);
                    out
                })
                .unwrap_or_else(|| vec![0.0; res_len]),
        },
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

fn eval_dot_general(
    module: &Module,
    lhs: ValueId,
    rhs: ValueId,
    res_val: ValueId,
    dims: &crate::stablehlo::DotDimensionNumbers,
    packed_rhs_2d: bool,
    get_v: &dyn Fn(ValueId) -> Vec<f32>,
) -> Vec<f32> {
    if dims.lhs_batching_dimensions.is_empty() && dims.rhs_batching_dimensions.is_empty() {
        let lhs_rank = module.values[lhs]
            .ty
            .as_tensor()
            .map(|t| t.rank())
            .unwrap_or(2);
        let rhs_rank = module.values[rhs]
            .ty
            .as_tensor()
            .map(|t| t.rank())
            .unwrap_or(2);
        if lhs_rank <= 2 && rhs_rank <= 2 {
            return eval_matmul_2d(module, lhs, rhs, packed_rhs_2d, get_v);
        }
    }

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
    let out_dims = module.values[res_val]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, 1]);

    let batch_count: usize = dims
        .lhs_batching_dimensions
        .iter()
        .filter_map(|&ax| lhs_dims.get(ax as usize).copied())
        .product::<usize>()
        .max(1);
    let k: usize = dims
        .lhs_contracting_dimensions
        .iter()
        .filter_map(|&ax| lhs_dims.get(ax as usize).copied())
        .product::<usize>()
        .max(1);
    let lhs_total: usize = lhs_dims.iter().product::<usize>().max(1);
    let rhs_total: usize = rhs_dims.iter().product::<usize>().max(1);
    let m = (lhs_total / (batch_count * k).max(1)).max(1);
    let n = (rhs_total / (batch_count * k).max(1)).max(1);
    let rhs_contract_last = dims
        .rhs_contracting_dimensions
        .last()
        .copied()
        .is_some_and(|d| (d as usize) + 1 == rhs_dims.len());

    let out_len: usize = out_dims.iter().product::<usize>().max(batch_count * m * n);
    let mut out = vec![0.0_f32; out_len];
    let a_batch_stride = m * k;
    let b_batch_stride = k * n;
    let c_batch_stride = m * n;

    for batch in 0..batch_count {
        let a_base = batch * a_batch_stride;
        let b_base = batch * b_batch_stride;
        let c_base = batch * c_batch_stride;
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    let a_val = *a.get(a_base + row * k + kk).unwrap_or(&0.0);
                    let b_idx = if rhs_contract_last {
                        b_base + col * k + kk
                    } else {
                        b_base + kk * n + col
                    };
                    let b_val = *b.get(b_idx).unwrap_or(&0.0);
                    acc += a_val * b_val;
                }
                if let Some(dst) = out.get_mut(c_base + row * n + col) {
                    *dst = acc;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn eval_fused_attention_2d(
    module: &Module,
    q_id: ValueId,
    k_id: ValueId,
    v_id: ValueId,
    mask_opt: Option<ValueId>,
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
    let mask_v = mask_opt.map(get_v);
    let v = get_v(v_id);
    let mut out = vec![0.0_f32; seq_q * d_v];

    for r in 0..seq_q {
        let row_scores = &qk[r * seq_kv..(r + 1) * seq_kv];
        let max_s = row_scores
            .iter()
            .enumerate()
            .map(|(c, &s)| {
                let m_add = mask_v
                    .as_ref()
                    .and_then(|m| m.get(r * seq_kv + c).copied())
                    .unwrap_or(0.0);
                s * scale + m_add
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0_f32; seq_kv];
        let mut sum = 0.0_f32;
        for (c, &s) in row_scores.iter().enumerate() {
            let m_add = mask_v
                .as_ref()
                .and_then(|m| m.get(r * seq_kv + c).copied())
                .unwrap_or(0.0);
            let e = (s * scale + m_add - max_s).exp();
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
                            .and_then(|v| v.get(i % v.len().max(1)).copied())
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

fn eval_gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044715_f32 * x * x * x)).tanh())
}

fn eval_im2col_raw(
    x: &[f32],
    h_in: usize,
    w_in: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
    res_len: usize,
) -> Vec<f32> {
    let h_out = h_in.saturating_sub(kh).saturating_add(1).max(1);
    let w_out = w_in.saturating_sub(kw).saturating_add(1).max(1);
    let k_total = (kh * kw * c_in).max(1);
    let mut out = vec![0.0_f32; res_len];
    for oh in 0..h_out {
        for ow in 0..w_out {
            let p = oh * w_out + ow;
            for r in 0..kh {
                for s in 0..kw {
                    for c in 0..c_in {
                        let in_idx = ((oh + r) * w_in + (ow + s)) * c_in + c;
                        let out_idx = p * k_total + (r * kw + s) * c_in + c;
                        if out_idx < out.len() && in_idx < x.len() {
                            out[out_idx] = x[in_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

fn eval_conv_2d(
    module: &Module,
    lhs: ValueId,
    rhs: ValueId,
    res_len: usize,
    get_v: &dyn Fn(ValueId) -> Vec<f32>,
) -> Vec<f32> {
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
    if lhs_dims.len() == 2 && rhs_dims.len() == 2 && lhs_dims[1] == rhs_dims[0] {
        return eval_matmul_2d(module, lhs, rhs, false, get_v);
    }
    let (h_in, w_in, c_in) = match lhs_dims.as_slice() {
        [h, w, c] => (*h, *w, *c),
        [_, h, w, c] => (*h, *w, *c),
        [hw, c] => (*hw, 1, *c),
        _ => (1, 1, 1),
    };
    let (kh, kw, k_total, c_out) = match rhs_dims.as_slice() {
        [kh, kw, cin, cout] => (*kh, *kw, kh * kw * cin, *cout),
        [kt, cout] => {
            let spatial = (kt / c_in.max(1)).max(1);
            let k_side = (spatial as f64).sqrt().round() as usize;
            (k_side.max(1), (spatial / k_side.max(1)).max(1), *kt, *cout)
        }
        _ => (1, 1, c_in, 1),
    };
    let m_patches = (res_len / c_out.max(1)).max(1);
    let patches = eval_im2col_raw(&get_v(lhs), h_in, w_in, c_in, kh, kw, m_patches * k_total);
    let w = get_v(rhs);
    let mut out = vec![0.0_f32; res_len];
    for p in 0..m_patches {
        for co in 0..c_out {
            let mut acc = 0.0_f32;
            for k in 0..k_total {
                acc += patches[p * k_total + k] * w[k * c_out + co];
            }
            if p * c_out + co < out.len() {
                out[p * c_out + co] = acc;
            }
        }
    }
    out
}
