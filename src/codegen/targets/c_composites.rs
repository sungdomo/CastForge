//! Composite operation emitters, C/C++ header generator, shared-library compiler, and Loop-IR
//! C99 backend.

use super::c::c_idx;
use crate::codegen::{
    CodegenTarget, TargetConfig, generate_code_with_config, lower_to_loop_ir, matmul_dims_ex,
    sanitize_ident, tensor_len, tensor_shape_dims, value_ident,
};
use crate::ir::{LoopModule, Module, Payload, Translator, ValueId};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static SHLIB_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_c_fused_attention(
    module: &Module,
    op: &crate::ir::Operation,
    q_id: ValueId,
    k_id: ValueId,
    v_id: ValueId,
    res_name: &str,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let q = value_ident(module, q_id, names, counter);
    let k = value_ident(module, k_id, names, counter);
    let v = value_ident(module, v_id, names, counter);
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
    let k_idx = if k_transposed {
        format!("c * {d_k} + kd")
    } else {
        format!("kd * {seq_kv} + c")
    };
    writeln!(
        out,
        "    /* [attention:flash_online_softmax(O(1)_workspace, d_v={d_v})] */"
    )?;
    writeln!(out, "    for (int r = 0; r < {seq_q}; ++r) {{")?;
    writeln!(out, "        float o_acc[{d_v}] = {{0}};")?;
    writeln!(out, "        float m_prev = -1e30f;")?;
    writeln!(out, "        float l_prev = 0.0f;")?;
    writeln!(out, "        for (int c = 0; c < {seq_kv}; ++c) {{")?;
    writeln!(out, "            float acc = 0.0f;")?;
    writeln!(
        out,
        "            for (int kd = 0; kd < {d_k}; ++kd) acc += {q}[r * {d_k} + kd] * {k}[{k_idx}];"
    )?;
    writeln!(out, "            float s = acc * {scale:.6}f;")?;
    writeln!(out, "            float m_new = fmaxf(m_prev, s);")?;
    writeln!(out, "            float alpha = expf(m_prev - m_new);")?;
    writeln!(out, "            float beta = expf(s - m_new);")?;
    writeln!(out, "            l_prev = l_prev * alpha + beta;")?;
    writeln!(
        out,
        "            for (int d = 0; d < {d_v}; ++d) \
         o_acc[d] = o_acc[d] * alpha + beta * {v}[c * {d_v} + d];"
    )?;
    writeln!(out, "            m_prev = m_new;")?;
    writeln!(out, "        }}")?;
    writeln!(
        out,
        "        for (int d = 0; d < {d_v}; ++d) {res_name}[r * {d_v} + d] = o_acc[d] / l_prev;"
    )?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_c_fused_loop_chain(
    module: &Module,
    inputs: &[ValueId],
    res_name: &str,
    res_len: usize,
    steps_str: &str,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let ext_exprs: Vec<String> = inputs
        .iter()
        .map(|&v| {
            let n = value_ident(module, v, names, counter);
            c_idx(module, v, res_len, &n, "i")
        })
        .collect();
    let specs: Vec<&str> = steps_str.split(';').filter(|s| !s.is_empty()).collect();
    writeln!(out, "    for (int i = 0; i < {res_len}; ++i) {{")?;
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
                        .unwrap_or_else(|| "0.0f".to_string())
                }
            })
            .collect();
        let a0 = args.first().map(String::as_str).unwrap_or("0.0f");
        let a1 = args.get(1).map(String::as_str).unwrap_or("0.0f");
        let expr = match op_name.trim() {
            "add" => format!("({a0}) + ({a1})"),
            "subtract" => format!("({a0}) - ({a1})"),
            "multiply" => format!("({a0}) * ({a1})"),
            "divide" => format!("({a0}) / ({a1})"),
            "maximum" => format!("fmaxf({a0}, {a1})"),
            "minimum" => format!("fminf({a0}, {a1})"),
            "exponential" => format!("expf({a0})"),
            "tanh" => format!("tanhf({a0})"),
            "rsqrt" => format!("1.0f / sqrtf({a0})"),
            _ => a0.to_string(),
        };
        writeln!(out, "        float s{idx} = {expr};")?;
    }
    let last_idx = specs.len().saturating_sub(1);
    writeln!(out, "        {res_name}[i] = s{last_idx};")?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_c_qmatmul(
    module: &Module,
    op: &crate::ir::Operation,
    inputs: &[ValueId],
    res_name: &str,
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
    let a = value_ident(module, inputs[0], names, counter);
    let w_q = value_ident(module, inputs[1], names, counter);
    let b_idx = if packed {
        format!("n * {k} + k")
    } else {
        format!("k * {n} + n")
    };
    writeln!(
        out,
        "    /* [qmatmul:on-the-fly dequant(scale={scale:.6}, zp={zp})] */"
    )?;
    writeln!(out, "    for (int m = 0; m < {m}; ++m) {{")?;
    writeln!(out, "        for (int n = 0; n < {n}; ++n) {{")?;
    writeln!(out, "            float acc = 0.0f;")?;
    writeln!(out, "            for (int k = 0; k < {k}; ++k) {{")?;
    writeln!(
        out,
        "                float w_val = ((float)((int){w_q}[{b_idx}] - {zp})) * {scale:.6}f;"
    )?;
    writeln!(out, "                acc += {a}[m * {k} + k] * w_val;")?;
    writeln!(out, "            }}")?;
    if let Some(&bias_id) = inputs.get(2) {
        let bias = value_ident(module, bias_id, names, counter);
        let bias_len = tensor_len(module, bias_id).max(1);
        let bias_idx = if bias_len <= n {
            format!("n % {bias_len}")
        } else {
            format!("m * {n} + n")
        };
        writeln!(
            out,
            "            {res_name}[m * {n} + n] = acc + {bias}[{bias_idx}];"
        )?;
    } else {
        writeln!(out, "            {res_name}[m * {n} + n] = acc;")?;
    }
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_c_residual_rmsnorm(
    module: &Module,
    op: &crate::ir::Operation,
    inputs: &[ValueId],
    res_name: &str,
    res_len: usize,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let x = value_ident(module, inputs[0], names, counter);
    let res_in = value_ident(module, inputs[1], names, counter);
    let dims = module.values[inputs[0]]
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
    writeln!(out, "    for (int r = 0; r < {rows}; ++r) {{")?;
    writeln!(out, "        float sq = 0.0f;")?;
    writeln!(out, "        for (int c = 0; c < {cols}; ++c) {{")?;
    writeln!(
        out,
        "            float h = {x}[r * {cols} + c] + {res_in}[r * {cols} + c];"
    )?;
    writeln!(out, "            sq += h * h;")?;
    writeln!(out, "        }}")?;
    writeln!(
        out,
        "        float inv = 1.0f / sqrtf(sq / {cols}.0f + {eps:.6}f);"
    )?;
    if let Some(&w_id) = inputs.get(2) {
        let w = value_ident(module, w_id, names, counter);
        let w_idx = if tensor_len(module, w_id) <= cols {
            "c".to_string()
        } else {
            format!("r * {cols} + c")
        };
        writeln!(
            out,
            "        for (int c = 0; c < {cols}; ++c) \
             {res_name}[r * {cols} + c] = \
             ({x}[r * {cols} + c] + {res_in}[r * {cols} + c]) * inv * {w}[{w_idx}];"
        )?;
    } else {
        writeln!(
            out,
            "        for (int c = 0; c < {cols}; ++c) \
             {res_name}[r * {cols} + c] = ({x}[r * {cols} + c] + {res_in}[r * {cols} + c]) * inv;"
        )?;
    }
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_c_layernorm(
    module: &Module,
    op: &crate::ir::Operation,
    inputs: &[ValueId],
    res_name: &str,
    res_len: usize,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
    out: &mut String,
) -> Result<()> {
    let x = value_ident(module, inputs[0], names, counter);
    let dims = module.values[inputs[0]]
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
    writeln!(out, "    for (int r = 0; r < {rows}; ++r) {{")?;
    writeln!(out, "        float sum = 0.0f;")?;
    writeln!(
        out,
        "        for (int c = 0; c < {cols}; ++c) sum += {x}[r * {cols} + c];"
    )?;
    writeln!(out, "        float mean = sum / {cols}.0f;")?;
    writeln!(out, "        float var_s = 0.0f;")?;
    writeln!(
        out,
        "        for (int c = 0; c < {cols}; ++c) \
         {{ float d = {x}[r * {cols} + c] - mean; var_s += d * d; }}"
    )?;
    writeln!(
        out,
        "        float inv = 1.0f / sqrtf(var_s / {cols}.0f + {eps:.6}f);"
    )?;
    let mut expr = format!("({x}[r * {cols} + c] - mean) * inv");
    if let Some(&g_id) = inputs.get(1) {
        let g = value_ident(module, g_id, names, counter);
        let g_idx = if tensor_len(module, g_id) <= cols {
            "c".to_string()
        } else {
            format!("r * {cols} + c")
        };
        expr = format!("{expr} * {g}[{g_idx}]");
    }
    if let Some(&b_id) = inputs.get(2) {
        let b = value_ident(module, b_id, names, counter);
        let b_idx = if tensor_len(module, b_id) <= cols {
            "c".to_string()
        } else {
            format!("r * {cols} + c")
        };
        expr = format!("{expr} + {b}[{b_idx}]");
    }
    writeln!(
        out,
        "        for (int c = 0; c < {cols}; ++c) {res_name}[r * {cols} + c] = {expr};"
    )?;
    writeln!(out, "    }}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_c_gated_mlp(
    module: &Module,
    op: &crate::ir::Operation,
    x_id: ValueId,
    w_id: ValueId,
    res_name: &str,
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
    let a = value_ident(module, x_id, names, counter);
    let w = value_ident(module, w_id, names, counter);
    let wg_idx = if packed {
        format!("n * {k} + k")
    } else {
        format!("k * {wide_n} + n")
    };
    let wu_idx = if packed {
        format!("(n + {n}) * {k} + k")
    } else {
        format!("k * {wide_n} + n + {n}")
    };
    writeln!(out, "    for (int m = 0; m < {m}; ++m) {{")?;
    writeln!(out, "        for (int n = 0; n < {n}; ++n) {{")?;
    writeln!(out, "            float gate = 0.0f;")?;
    writeln!(out, "            float up = 0.0f;")?;
    writeln!(out, "            for (int k = 0; k < {k}; ++k) {{")?;
    writeln!(out, "                float xv = {a}[m * {k} + k];")?;
    writeln!(out, "                gate += xv * {w}[{wg_idx}];")?;
    writeln!(out, "                up += xv * {w}[{wu_idx}];")?;
    writeln!(out, "            }}")?;
    if act == "geglu" {
        writeln!(
            out,
            "            float g_act = 0.5f * gate * \
             (1.0f + tanhf(0.79788456f * (gate + 0.044715f * gate * gate * gate)));"
        )?;
    } else {
        writeln!(
            out,
            "            float g_act = gate / (1.0f + expf(-gate));"
        )?;
    }
    writeln!(out, "            {res_name}[m * {n} + n] = g_act * up;")?;
    writeln!(out, "        }}")?;
    writeln!(out, "    }}")?;
    Ok(())
}

/// Generates a C/C++ FFI header (`.h`) matching the C99 symbols emitted by `codegen-c`.
pub fn emit_c_header(module: &Module) -> Result<String> {
    let loop_mod = lower_to_loop_ir(module, &TargetConfig::default());
    let guard = format!(
        "CASTFORGE_{}_H",
        sanitize_ident(&loop_mod.name).to_uppercase()
    );
    let mut out = String::new();
    writeln!(out, "/* Auto-generated C/C++ FFI Header by CastForge */")?;
    writeln!(out, "#ifndef {guard}")?;
    writeln!(out, "#define {guard}\n")?;
    writeln!(out, "#include <stddef.h>\n")?;
    writeln!(out, "#ifdef __cplusplus")?;
    writeln!(out, "extern \"C\" {{")?;
    writeln!(out, "#endif\n")?;

    for kernel in &loop_mod.kernels {
        let mut args: Vec<String> = kernel
            .params
            .iter()
            .map(|(n, len)| format!("const float {n}[{len}]"))
            .collect();
        args.push(format!("float {}[{}]", kernel.output.0, kernel.output.1));
        writeln!(out, "void {}({});", kernel.name, args.join(", "))?;
    }

    writeln!(out, "\n#ifdef __cplusplus")?;
    writeln!(out, "}}")?;
    writeln!(out, "#endif\n")?;
    writeln!(out, "#endif /* {guard} */")?;
    Ok(out)
}

/// Compiles `module` into a native shared library (`.so` / `.dylib`) byte vector.
pub fn compile_shared_lib(module: &Module, config: &TargetConfig) -> Result<Vec<u8>> {
    let c_src = generate_code_with_config(module, CodegenTarget::C, config)?;
    let seq = SHLIB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_dir = std::env::temp_dir().join(format!("castforge_shlib_{pid}_{seq}"));
    fs::create_dir_all(&tmp_dir)?;
    struct TempDirGuard(std::path::PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _tmp_guard = TempDirGuard(tmp_dir.clone());
    let src_path = tmp_dir.join("kernel.c");
    let lib_path = tmp_dir.join("libcastforge_kernel.so");
    fs::write(&src_path, &c_src)?;

    let status = Command::new("cc")
        .arg("-O3")
        .arg("-shared")
        .arg("-fPIC")
        .arg(&src_path)
        .arg("-lm")
        .arg("-o")
        .arg(&lib_path)
        .status()
        .context("Failed to invoke `cc` for shared library compilation")?;
    if !status.success() {
        bail!("`cc -shared` exited with non-zero status: {status}");
    }
    let bytes = fs::read(&lib_path).context("Failed to read compiled shared library")?;
    Ok(bytes)
}

/// Emits portable C99 source code from a lowered [`LoopModule`].
pub fn emit_c_from_loop_ir(loop_mod: &LoopModule) -> Result<String> {
    use crate::codegen::backend::emit_c_like_loop_stmt;
    let mut out = String::new();
    writeln!(out, "/* Auto-generated by CastForge C99 LoopIR Backend */")?;
    writeln!(out, "#include <math.h>\n")?;
    for kernel in &loop_mod.kernels {
        let mut sig: Vec<String> = kernel
            .params
            .iter()
            .map(|(n, len)| format!("const float {n}[{len}]"))
            .collect();
        sig.push(format!("float {}[{}]", kernel.output.0, kernel.output.1));
        writeln!(out, "void {}({}) {{", kernel.name, sig.join(", "))?;
        for stmt in &kernel.body {
            emit_c_like_loop_stmt(&mut out, stmt, &kernel.output.0, "0", 1, false)?;
        }
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

/// C/C++ Header Translator (`Ir(Module) -> Text(C Header)`).
#[derive(Default)]
pub struct CHeaderTranslator;

impl Translator for CHeaderTranslator {
    fn name(&self) -> &str {
        "codegen-header"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        Ok(Payload::Text(emit_c_header(&module)?))
    }
}

/// Native Shared Library Translator (`Ir(Module) -> Bytes(.so/.dylib)`).
#[derive(Default)]
pub struct SharedLibTranslator {
    pub config: TargetConfig,
}

impl Translator for SharedLibTranslator {
    fn name(&self) -> &str {
        "compile-lib"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        Ok(Payload::Bytes(compile_shared_lib(&module, &self.config)?))
    }
}
