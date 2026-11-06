//! Composite operation emitters and Loop-IR codegen for the Rust backend.

use super::rust::{r_bcast_idx, r_idx};
use crate::codegen::{
    LoopModule, LoopStmt, MemoryPlan, matmul_dims_ex, tensor_len, tensor_shape_dims,
};
use crate::ir::{Module, ValueId};
use anyhow::Result;
use std::collections::HashMap;
use std::fmt::Write;

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_rust_fused_attention(
    module: &Module,
    plan: &MemoryPlan,
    op: &crate::ir::Operation,
    q_id: ValueId,
    k_id: ValueId,
    v_id: ValueId,
    res_val: ValueId,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
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
    let d_k = *q_dims.get(1).unwrap_or(&1);
    let seq_kv = *v_dims.first().unwrap_or(&1);
    let d_v = *v_dims.get(1).unwrap_or(&1);
    let scale = match op.attributes.get("scale") {
        Some(crate::ir::Attribute::Float(f)) => *f as f32,
        _ => 1.0_f32,
    };
    let k_transposed = op
        .attributes
        .get("k_transposed")
        .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
    let q_idx = r_idx(
        module,
        plan,
        q_id,
        &format!("r * {d_k} + kd"),
        names,
        counter,
    );
    let k_idx_str = if k_transposed {
        format!("c * {d_k} + kd")
    } else {
        format!("kd * {seq_kv} + c")
    };
    let k_idx = r_idx(module, plan, k_id, &k_idx_str, names, counter);
    let v_idx = r_idx(
        module,
        plan,
        v_id,
        &format!("c * {d_v} + d"),
        names,
        counter,
    );
    let dst = r_idx(
        module,
        plan,
        res_val,
        &format!("r * {d_v} + d"),
        names,
        counter,
    );
    writeln!(
        out,
        "    // [attention:flash_online_softmax(O(1)_workspace, d_v={d_v})]"
    )?;
    writeln!(out, "    for r in 0..{seq_q} {{")?;
    writeln!(out, "        let mut o_acc = [0.0_f32; {d_v}];")?;
    writeln!(out, "        let mut m_prev = f32::NEG_INFINITY;")?;
    writeln!(out, "        let mut l_prev = 0.0_f32;")?;
    writeln!(out, "        for c in 0..{seq_kv} {{")?;
    writeln!(out, "            let mut acc = 0.0_f32;")?;
    writeln!(
        out,
        "            for kd in 0..{d_k} {{ acc += {q_idx} * {k_idx}; }}"
    )?;
    writeln!(out, "            let s = acc * {scale:?}_f32;")?;
    writeln!(out, "            let m_new = m_prev.max(s);")?;
    writeln!(out, "            let alpha = (m_prev - m_new).exp();")?;
    writeln!(out, "            let beta = (s - m_new).exp();")?;
    writeln!(out, "            l_prev = l_prev * alpha + beta;")?;
    writeln!(
        out,
        "            for d in 0..{d_v} {{ o_acc[d] = o_acc[d] * alpha + beta * {v_idx}; }}"
    )?;
    writeln!(out, "            m_prev = m_new;")?;
    writeln!(out, "        }}")?;
    writeln!(
        out,
        "        for d in 0..{d_v} {{ {dst} = o_acc[d] / l_prev; }}"
    )?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_rust_fused_loop_chain(
    module: &Module,
    plan: &MemoryPlan,
    inputs: &[ValueId],
    res_val: ValueId,
    res_len: usize,
    steps_str: &str,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let ext_exprs: Vec<String> = inputs
        .iter()
        .map(|&v| r_bcast_idx(module, plan, v, "i", res_len, names, counter))
        .collect();
    let dst = r_idx(module, plan, res_val, "i", names, counter);
    let specs: Vec<&str> = steps_str.split(';').filter(|s| !s.is_empty()).collect();
    writeln!(out, "    for i in 0..{res_len} {{")?;
    for (idx, spec) in specs.iter().enumerate() {
        let Some((op_name, rest)) = spec.split_once('(') else {
            continue;
        };
        let args: Vec<String> = rest
            .trim_end_matches(')')
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|tok| {
                let tok = tok.trim();
                if let Some(step_idx) = tok
                    .strip_prefix('s')
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&k| k < idx)
                {
                    format!("s{step_idx}")
                } else {
                    tok.parse::<usize>()
                        .ok()
                        .and_then(|k| ext_exprs.get(k).cloned())
                        .unwrap_or_else(|| "0.0_f32".to_string())
                }
            })
            .collect();
        let a0 = args.first().map(String::as_str).unwrap_or("0.0_f32");
        let a1 = args.get(1).map(String::as_str).unwrap_or("0.0_f32");
        let expr = match op_name.trim() {
            "add" => format!("({a0}) + ({a1})"),
            "subtract" => format!("({a0}) - ({a1})"),
            "multiply" => format!("({a0}) * ({a1})"),
            "divide" => format!("({a0}) / ({a1})"),
            "maximum" => format!("({a0}).max({a1})"),
            "minimum" => format!("({a0}).min({a1})"),
            "exponential" => format!("({a0}).exp()"),
            "tanh" => format!("({a0}).tanh()"),
            "rsqrt" => format!("1.0_f32 / ({a0}).sqrt()"),
            _ => a0.to_string(),
        };
        writeln!(out, "        let s{idx} = {expr};")?;
    }
    let last_idx = specs.len().saturating_sub(1);
    writeln!(out, "        {dst} = s{last_idx};")?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_rust_qmatmul(
    module: &Module,
    plan: &MemoryPlan,
    op: &crate::ir::Operation,
    inputs: &[ValueId],
    res_val: ValueId,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let packed = op
        .attributes
        .get("packed_rhs")
        .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
    let scale = match op.attributes.get("scale") {
        Some(crate::ir::Attribute::Float(f)) => *f as f32,
        _ => 1.0_f32,
    };
    let zp = match op.attributes.get("zero_point") {
        Some(crate::ir::Attribute::Int(z)) => *z as i32,
        _ => 0,
    };
    let (m, k, n) = matmul_dims_ex(module, inputs[0], inputs[1], packed);
    let a = r_idx(
        module,
        plan,
        inputs[0],
        &format!("m * {k} + k"),
        names,
        counter,
    );
    let b_idx = if packed {
        format!("n * {k} + k")
    } else {
        format!("k * {n} + n")
    };
    let w_q = r_idx(module, plan, inputs[1], &b_idx, names, counter);
    let dst = r_idx(module, plan, res_val, "idx", names, counter);
    writeln!(
        out,
        "    // [qmatmul:on-the-fly dequant(scale={scale:?}, zp={zp})]"
    )?;
    writeln!(out, "    for m in 0..{m} {{")?;
    writeln!(out, "        for n in 0..{n} {{")?;
    writeln!(out, "            let mut acc = 0.0_f32;")?;
    writeln!(out, "            for k in 0..{k} {{")?;
    writeln!(
        out,
        "                let w_val = (({w_q} as i32 - {zp}) as f32) * {scale:?}_f32;"
    )?;
    writeln!(out, "                acc += {a} * w_val;")?;
    writeln!(out, "            }}")?;
    writeln!(out, "            let idx = m * {n} + n;")?;
    if let Some(&bias_id) = inputs.get(2) {
        let bias_len = tensor_len(module, bias_id).max(1);
        let bias_idx_s = if bias_len <= n {
            format!("n % {bias_len}")
        } else {
            "idx".to_string()
        };
        let bias = r_idx(module, plan, bias_id, &bias_idx_s, names, counter);
        writeln!(out, "            {dst} = acc + {bias};")?;
    } else {
        writeln!(out, "            {dst} = acc;")?;
    }
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_rust_residual_rmsnorm(
    module: &Module,
    plan: &MemoryPlan,
    op: &crate::ir::Operation,
    inputs: &[ValueId],
    res_val: ValueId,
    res_len: usize,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let dims = module.values[res_val]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, res_len]);
    let cols = *dims.last().unwrap_or(&res_len).max(&1);
    let rows = (res_len / cols).max(1);
    let eps = match op.attributes.get("eps") {
        Some(crate::ir::Attribute::Float(f)) => *f as f32,
        _ => 1e-5_f32,
    };
    let idx_s = format!("r * {cols} + c");
    let x = r_idx(module, plan, inputs[0], &idx_s, names, counter);
    let res_in = r_idx(module, plan, inputs[1], &idx_s, names, counter);
    let dst = r_idx(module, plan, res_val, &idx_s, names, counter);
    writeln!(out, "    for r in 0..{rows} {{")?;
    writeln!(out, "        let mut sq = 0.0_f32;")?;
    writeln!(out, "        for c in 0..{cols} {{")?;
    writeln!(out, "            let h = ({x}) + ({res_in});")?;
    writeln!(out, "            sq += h * h;")?;
    writeln!(out, "        }}")?;
    writeln!(
        out,
        "        let inv = 1.0_f32 / (sq / {cols}.0_f32 + {eps:?}_f32).sqrt();"
    )?;
    writeln!(out, "        for c in 0..{cols} {{")?;
    writeln!(out, "            let h = ({x}) + ({res_in});")?;
    if let Some(&w_id) = inputs.get(2) {
        let w_len = tensor_len(module, w_id).max(1);
        let w_idx_s = if w_len <= cols {
            format!("c % {w_len}")
        } else {
            idx_s.clone()
        };
        let w = r_idx(module, plan, w_id, &w_idx_s, names, counter);
        writeln!(out, "            {dst} = h * inv * ({w});")?;
    } else {
        writeln!(out, "            {dst} = h * inv;")?;
    }
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_rust_layernorm(
    module: &Module,
    plan: &MemoryPlan,
    op: &crate::ir::Operation,
    inputs: &[ValueId],
    res_val: ValueId,
    res_len: usize,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let dims = module.values[res_val]
        .ty
        .as_tensor()
        .map(tensor_shape_dims)
        .unwrap_or_else(|| vec![1, res_len]);
    let cols = *dims.last().unwrap_or(&res_len).max(&1);
    let rows = (res_len / cols).max(1);
    let eps = match op.attributes.get("eps") {
        Some(crate::ir::Attribute::Float(f)) => *f as f32,
        _ => 1e-5_f32,
    };
    let idx_s = format!("r * {cols} + c");
    let x = r_idx(module, plan, inputs[0], &idx_s, names, counter);
    let dst = r_idx(module, plan, res_val, &idx_s, names, counter);
    writeln!(out, "    for r in 0..{rows} {{")?;
    writeln!(out, "        let mut sum = 0.0_f32;")?;
    writeln!(out, "        for c in 0..{cols} {{ sum += {x}; }}")?;
    writeln!(out, "        let mean = sum / {cols}.0_f32;")?;
    writeln!(out, "        let mut var_s = 0.0_f32;")?;
    writeln!(
        out,
        "        for c in 0..{cols} {{ let d = ({x}) - mean; var_s += d * d; }}"
    )?;
    writeln!(
        out,
        "        let inv = 1.0_f32 / (var_s / {cols}.0_f32 + {eps:?}_f32).sqrt();"
    )?;
    writeln!(out, "        for c in 0..{cols} {{")?;
    let mut expr = format!("(({x}) - mean) * inv");
    if let Some(&g_id) = inputs.get(1) {
        let g_len = tensor_len(module, g_id).max(1);
        let g_idx_s = if g_len <= cols {
            format!("c % {g_len}")
        } else {
            idx_s.clone()
        };
        let g = r_idx(module, plan, g_id, &g_idx_s, names, counter);
        expr = format!("{expr} * ({g})");
    }
    if let Some(&b_id) = inputs.get(2) {
        let b_len = tensor_len(module, b_id).max(1);
        let b_idx_s = if b_len <= cols {
            format!("c % {b_len}")
        } else {
            idx_s.clone()
        };
        let b = r_idx(module, plan, b_id, &b_idx_s, names, counter);
        expr = format!("{expr} + ({b})");
    }
    writeln!(out, "            {dst} = {expr};")?;
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_rust_gated_mlp(
    module: &Module,
    plan: &MemoryPlan,
    op: &crate::ir::Operation,
    x_id: ValueId,
    w_id: ValueId,
    res_val: ValueId,
    res_len: usize,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let packed = op
        .attributes
        .get("packed_rhs")
        .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)));
    let act = op
        .attributes
        .get("activation")
        .and_then(|a| a.as_str())
        .unwrap_or("swiglu");
    let (m, k, wide_n) = matmul_dims_ex(module, x_id, w_id, packed);
    let n = (wide_n / 2).max(1);
    let _ = res_len;
    let a = r_idx(module, plan, x_id, &format!("m * {k} + k"), names, counter);
    let w_gate_idx = if packed {
        format!("n * {k} + k")
    } else {
        format!("k * {wide_n} + n")
    };
    let w_up_idx = if packed {
        format!("(n + {n}) * {k} + k")
    } else {
        format!("k * {wide_n} + n + {n}")
    };
    let wg = r_idx(module, plan, w_id, &w_gate_idx, names, counter);
    let wu = r_idx(module, plan, w_id, &w_up_idx, names, counter);
    let dst = r_idx(
        module,
        plan,
        res_val,
        &format!("m * {n} + n"),
        names,
        counter,
    );
    writeln!(out, "    for m in 0..{m} {{")?;
    writeln!(out, "        for n in 0..{n} {{")?;
    writeln!(out, "            let mut gate = 0.0_f32;")?;
    writeln!(out, "            let mut up = 0.0_f32;")?;
    writeln!(out, "            for k in 0..{k} {{")?;
    writeln!(out, "                let xv = {a};")?;
    writeln!(out, "                gate += xv * {wg};")?;
    writeln!(out, "                up += xv * {wu};")?;
    writeln!(out, "            }}")?;
    if act == "geglu" {
        writeln!(
            out,
            "            let g_act = 0.5_f32 * gate * \
             (1.0_f32 + (0.79788456_f32 * (gate + 0.044715_f32 * gate * gate * gate)).tanh());"
        )?;
    } else {
        writeln!(
            out,
            "            let g_act = gate / (1.0_f32 + (-gate).exp());"
        )?;
    }
    writeln!(out, "            {dst} = g_act * up;")?;
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    Ok(())
}

/// Emits standalone Rust source code from a lowered [`LoopModule`].
pub fn emit_rust_from_loop_ir(loop_mod: &LoopModule) -> Result<String> {
    let mut out = String::new();
    writeln!(
        out,
        "// Auto-generated by CastForge Loop-IR Codegen (Target: Rust)"
    )?;
    writeln!(
        out,
        "#![allow(clippy::excessive_precision, unused_variables, clippy::identity_op, dead_code)]\n"
    )?;
    for kernel in &loop_mod.kernels {
        let params: Vec<String> = kernel
            .params
            .iter()
            .map(|(n, len)| format!("{n}: &[f32; {len}]"))
            .collect();
        let (out_name, out_len) = (&kernel.output.0, kernel.output.1);
        writeln!(
            out,
            "pub fn {}({}) -> [f32; {out_len}] {{",
            kernel.name,
            params.join(", ")
        )?;
        writeln!(out, "    let mut {out_name} = [0.0_f32; {out_len}];")?;
        for stmt in &kernel.body {
            emit_rust_loop_stmt(stmt, &kernel.output.0, 1, &mut out)?;
        }
        writeln!(out, "    {out_name}")?;
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

fn emit_rust_loop_stmt(
    stmt: &LoopStmt,
    ret_name: &str,
    indent: usize,
    out: &mut String,
) -> Result<()> {
    let pad = "    ".repeat(indent);
    match stmt {
        LoopStmt::Comment(c) => writeln!(out, "{pad}// {c}")?,
        LoopStmt::Alloc { name, len } if name != ret_name => {
            writeln!(out, "{pad}let mut {name} = [0.0_f32; {len}];")?;
        }
        LoopStmt::Alloc { .. } => {}
        LoopStmt::For {
            var,
            lower,
            upper,
            step,
            parallel,
            vectorized,
            unroll,
            body,
        } => {
            if *parallel {
                writeln!(out, "{pad}// [parallel]")?;
            }
            if *vectorized {
                writeln!(out, "{pad}// [vectorize]")?;
            }
            if *unroll > 1 {
                writeln!(out, "{pad}// [unroll({unroll})]")?;
            }
            if *step <= 1 {
                writeln!(out, "{pad}for {var} in {lower}..{upper} {{")?;
            } else {
                writeln!(
                    out,
                    "{pad}for {var} in ({lower}..{upper}).step_by({step}) {{"
                )?;
            }
            for inner in body {
                emit_rust_loop_stmt(inner, ret_name, indent + 1, out)?;
            }
            writeln!(out, "{pad}}}")?;
        }
        LoopStmt::Assign { dst, index, expr } => {
            writeln!(out, "{pad}{dst}[{index}] = {expr};")?;
        }
    }
    Ok(())
}
