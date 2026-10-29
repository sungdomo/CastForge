//! Native JIT compilation, execution, numerical verification, and hardware benchmarking
//! (`jit_execute_and_benchmark`, `autotune_module_jit`) for Rust and C99 codegen targets.

use crate::{
    AutotuneReport, CodegenTarget, HardwareRoofline, TargetConfig,
    analyze_module_cost_with_roofline, autotune_module, evaluate_module, generate_code_with_config,
    generate_synthetic_inputs, sanitize_ident, tensor_len,
};
use anyhow::{Context, Result, anyhow, bail};
use castforge_ir::Module;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static JIT_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Result of compiling and benchmarking a generated kernel on the native host CPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitBenchmarkResult {
    /// Codegen target compiled (`Rust` or `C`).
    pub target: CodegenTarget,
    /// Loop schedule configuration used.
    pub config: TargetConfig,
    /// Average wall-clock execution latency in nanoseconds across `iterations`.
    pub avg_latency_ns: f64,
    /// Minimum wall-clock execution latency in nanoseconds across `iterations`.
    pub min_latency_ns: f64,
    /// Number of timed benchmark iterations executed.
    pub iterations: usize,
    /// Output tensor elements produced by the compiled native binary.
    pub outputs: Vec<f32>,
    /// Whether the native binary outputs matched the reference IR interpreter within tolerance.
    pub numerics_verified: bool,
    /// Maximum absolute difference between native binary output and reference interpreter.
    pub max_abs_diff: f32,
}

/// Compiles the generated Rust or C99 code for `module` into a native binary, executes it with
/// synthetic inputs, verifies numerical equivalence against [`evaluate_module`], and measures
/// real hardware execution latency.
pub fn jit_execute_and_benchmark(
    module: &Module,
    target: CodegenTarget,
    config: &TargetConfig,
    warmup: usize,
    iterations: usize,
) -> Result<JitBenchmarkResult> {
    let funcs = module.functions();
    let fn_op_id = *funcs
        .first()
        .ok_or_else(|| anyhow!("Module contains no functions for JIT execution"))?;
    let fn_op = &module.ops[fn_op_id];
    let fn_name = sanitize_ident(
        fn_op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("forward"),
    );
    let region_id = fn_op.regions[0];
    let block_id = module.regions[region_id].blocks[0];
    let block = &module.blocks[block_id];

    let mut ret_len = 1usize;
    for &op_id in &block.ops {
        let op = &module.ops[op_id];
        if op.is_op("func", "return")
            && let Some(&ret_val) = op.operands.first()
        {
            ret_len = tensor_len(module, ret_val);
        }
    }

    let synth_inputs = generate_synthetic_inputs(module, None)?;
    let ref_outputs_all = evaluate_module(module, &synth_inputs)?;
    let ref_out = ref_outputs_all.first().cloned().unwrap_or_default();

    let mut code = generate_code_with_config(module, target, config)?;
    let call_fn = if fn_name == "main" {
        code = code
            .replace("pub fn main(", "pub fn castforge_entry(")
            .replace("void main(", "void castforge_entry(");
        "castforge_entry".to_string()
    } else {
        fn_name
    };
    let iters = iterations.max(1);
    let warm = warmup.max(1);

    let seq = JIT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_dir = std::env::temp_dir().join(format!("castforge_jit_{pid}_{seq}"));
    fs::create_dir_all(&tmp_dir)?;
    let bin_path = tmp_dir.join("kernel_bench");

    match target {
        CodegenTarget::Rust => {
            writeln!(code, "\npub fn main() {{")?;
            let mut call_args = Vec::new();
            for (idx, &arg_id) in block.arguments.iter().enumerate() {
                let len = tensor_len(module, arg_id);
                let vals = synth_inputs
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| vec![0.1; len]);
                let lit = vals
                    .iter()
                    .take(len)
                    .map(|v| format!("{v:.6}_f32"))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(code, "    let arg_{idx}: [f32; {len}] = [{lit}];")?;
                call_args.push(format!("std::hint::black_box(&arg_{idx})"));
            }
            let call_expr = format!("{call_fn}({})", call_args.join(", "));
            writeln!(code, "    for _ in 0..{warm} {{")?;
            writeln!(code, "        std::hint::black_box({call_expr});")?;
            writeln!(code, "    }}")?;
            writeln!(code, "    let mut min_ns = f64::MAX;")?;
            writeln!(code, "    let mut total_ns = 0.0_f64;")?;
            writeln!(code, "    let mut last_out = [0.0_f32; {ret_len}];")?;
            writeln!(code, "    for _ in 0..{iters} {{")?;
            writeln!(code, "        let t0 = std::time::Instant::now();")?;
            writeln!(
                code,
                "        last_out = std::hint::black_box({call_expr});"
            )?;
            writeln!(
                code,
                "        let dt = t0.elapsed().as_secs_f64() * 1e9_f64;"
            )?;
            writeln!(code, "        if dt < min_ns {{ min_ns = dt; }}")?;
            writeln!(code, "        total_ns += dt;")?;
            writeln!(code, "    }}")?;
            writeln!(
                code,
                "    println!(\"LATENCY_NS:{{:.3}}:{{:.3}}\", total_ns / {iters}.0, min_ns);"
            )?;
            writeln!(code, "    print!(\"OUTPUTS:\");")?;
            writeln!(code, "    for (i, v) in last_out.iter().enumerate() {{")?;
            writeln!(code, "        if i > 0 {{ print!(\",\"); }}")?;
            writeln!(code, "        print!(\"{{v:.6}}\");")?;
            writeln!(code, "    }}")?;
            writeln!(code, "    println!();")?;
            writeln!(code, "}}")?;

            let src_path = tmp_dir.join("kernel_bench.rs");
            fs::write(&src_path, &code)?;
            let status = Command::new("rustc")
                .arg("-C")
                .arg("opt-level=3")
                .arg("-A")
                .arg("warnings")
                .arg(&src_path)
                .arg("-o")
                .arg(&bin_path)
                .output()
                .context("Failed to invoke `rustc` for JIT compilation")?;
            if !status.status.success() {
                let stderr = String::from_utf8_lossy(&status.stderr);
                let _ = fs::remove_dir_all(&tmp_dir);
                bail!("rustc JIT compilation failed: {stderr}");
            }
        }
        CodegenTarget::C => {
            let mut full_c = String::from("#define _POSIX_C_SOURCE 199309L\n#include <stdio.h>\n");
            full_c.push_str("#include <time.h>\n");
            full_c.push_str(&code);
            writeln!(full_c, "\nint main(void) {{")?;
            let mut call_args = Vec::new();
            for (idx, &arg_id) in block.arguments.iter().enumerate() {
                let len = tensor_len(module, arg_id);
                let vals = synth_inputs
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| vec![0.1; len]);
                let lit = vals
                    .iter()
                    .take(len)
                    .map(|v| format!("{v:.6}f"))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(full_c, "    float arg_{idx}[{len}] = {{{lit}}};")?;
                call_args.push(format!("arg_{idx}"));
            }
            writeln!(full_c, "    float out_buf[{ret_len}] = {{0}};")?;
            call_args.push("out_buf".to_string());
            let call_stmt = format!("{call_fn}({});", call_args.join(", "));
            writeln!(full_c, "    for (int w = 0; w < {warm}; ++w) {{")?;
            writeln!(full_c, "        {call_stmt}")?;
            writeln!(full_c, "    }}")?;
            writeln!(full_c, "    double min_ns = 1e18, total_ns = 0.0;")?;
            writeln!(full_c, "    for (int it = 0; it < {iters}; ++it) {{")?;
            writeln!(full_c, "        struct timespec t0, t1;")?;
            writeln!(full_c, "        clock_gettime(CLOCK_MONOTONIC, &t0);")?;
            writeln!(full_c, "        {call_stmt}")?;
            writeln!(full_c, "        clock_gettime(CLOCK_MONOTONIC, &t1);")?;
            writeln!(
                full_c,
                "        double dt = (double)(t1.tv_sec - t0.tv_sec) * 1e9 + \
                 (double)(t1.tv_nsec - t0.tv_nsec);"
            )?;
            writeln!(full_c, "        if (dt < min_ns) min_ns = dt;")?;
            writeln!(full_c, "        total_ns += dt;")?;
            writeln!(full_c, "    }}")?;
            writeln!(
                full_c,
                "    printf(\"LATENCY_NS:%.3f:%.3f\\n\", total_ns / {iters}.0, min_ns);"
            )?;
            writeln!(full_c, "    printf(\"OUTPUTS:\");")?;
            writeln!(full_c, "    for (int i = 0; i < {ret_len}; ++i) {{")?;
            writeln!(full_c, "        if (i > 0) printf(\",\");")?;
            writeln!(full_c, "        printf(\"%.6f\", out_buf[i]);")?;
            writeln!(full_c, "    }}")?;
            writeln!(full_c, "    printf(\"\\n\");")?;
            writeln!(full_c, "    return 0;")?;
            writeln!(full_c, "}}")?;

            let src_path = tmp_dir.join("kernel_bench.c");
            fs::write(&src_path, &full_c)?;
            let status = Command::new("cc")
                .arg("-std=c99")
                .arg("-O3")
                .arg(&src_path)
                .arg("-o")
                .arg(&bin_path)
                .arg("-lm")
                .output()
                .context("Failed to invoke `cc` for C99 JIT compilation")?;
            if !status.status.success() {
                let stderr = String::from_utf8_lossy(&status.stderr);
                let _ = fs::remove_dir_all(&tmp_dir);
                bail!("cc JIT compilation failed: {stderr}");
            }
        }
        CodegenTarget::Wgsl
        | CodegenTarget::Cuda
        | CodegenTarget::Msl
        | CodegenTarget::Hip
        | CodegenTarget::OpenCl => {
            let _ = fs::remove_dir_all(&tmp_dir);
            bail!("Native CPU JIT execution supports Rust and C99 targets");
        }
    }

    let run_out = Command::new(&bin_path)
        .output()
        .context("Failed to execute compiled JIT binary")?;
    let _ = fs::remove_dir_all(&tmp_dir);
    if !run_out.status.success() {
        bail!("Compiled JIT binary exited with non-zero status");
    }

    let stdout = String::from_utf8_lossy(&run_out.stdout);
    let mut avg_latency_ns = 1.0_f64;
    let mut min_latency_ns = 1.0_f64;
    let mut outputs = Vec::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("LATENCY_NS:") {
            let parts: Vec<&str> = rest.split(':').collect();
            if parts.len() >= 2 {
                avg_latency_ns = parts[0].parse::<f64>().unwrap_or(1.0).max(0.1);
                min_latency_ns = parts[1].parse::<f64>().unwrap_or(1.0).max(0.1);
            }
        } else if let Some(rest) = line.strip_prefix("OUTPUTS:") {
            outputs = rest
                .split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse::<f32>().ok())
                .collect();
        }
    }

    let mut max_abs_diff = 0.0_f32;
    for (&a, &b) in outputs.iter().zip(ref_out.iter()) {
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    let numerics_verified = outputs.len() == ref_out.len() && max_abs_diff <= 1e-3;

    Ok(JitBenchmarkResult {
        target,
        config: *config,
        avg_latency_ns,
        min_latency_ns,
        iterations: iters,
        outputs,
        numerics_verified,
        max_abs_diff,
    })
}

/// Runs Roofline analytical schedule search followed by hardware-in-the-loop native JIT
/// compilation and empirical timing of the top `top_k_empirically` candidate schedules.
pub fn autotune_module_jit(
    module: &Module,
    roofline: &HardwareRoofline,
    top_k_empirically: usize,
) -> Result<AutotuneReport> {
    let mut report = autotune_module(module, roofline);
    let base_cost = analyze_module_cost_with_roofline(module, roofline);
    let flops = base_cost.total_flops.max(1) as f64;

    let base_jit =
        jit_execute_and_benchmark(module, CodegenTarget::Rust, &TargetConfig::default(), 3, 15)?;
    let base_ns = base_jit.min_latency_ns.max(0.1);
    report.baseline_latency_ns = base_ns;

    let k_count = top_k_empirically.clamp(1, report.candidates.len().max(1));
    for cand in report.candidates.iter_mut().take(k_count) {
        if let Ok(bench) =
            jit_execute_and_benchmark(module, CodegenTarget::Rust, &cand.config, 3, 15)
        {
            let measured_ns = bench.min_latency_ns.max(0.1);
            cand.estimated_latency_ns = measured_ns;
            cand.effective_gflops = flops / measured_ns;
            cand.speedup_over_baseline = (base_ns / measured_ns).max(1.0);
        }
    }

    report.candidates[..k_count].sort_by(|a, b| {
        a.estimated_latency_ns
            .partial_cmp(&b.estimated_latency_ns)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = report.candidates.first().cloned() {
        report.best = first;
    }
    Ok(report)
}
