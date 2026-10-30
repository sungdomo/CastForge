//! Roofline-guided polyhedral & loop schedule auto-tuner (`autotune_module`).
//!
//! Searches over candidate [`TargetConfig`] schedules (loop order `MNK`/`MKN`/`NKM`, L1/L2 cache
//! tile sizes, `(mr x nr)` Unroll-and-Jam register micro-kernels, inner-loop unroll factors,
//! Split-K reduction factors, SIMD vectorization, OpenMP parallelism, and software prefetching)
//! and scores each candidate against a [`HardwareRoofline`] and the module's tensor shapes.

use crate::codegen::cost_model::{
    GemmLoopOrder, HardwareRoofline, TargetConfig, analyze_module_cost_with_roofline,
};
use crate::codegen::{
    CodegenTarget, evaluate_module, generate_code_with_config, generate_synthetic_inputs,
    sanitize_ident, tensor_len,
};
use crate::ir::{Dim, Module};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A single evaluated schedule candidate from the auto-tuner search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotuneCandidate {
    /// Human-readable schedule summary label.
    pub label: String,
    /// Candidate hardware & loop-transformation configuration.
    pub config: TargetConfig,
    /// Estimated effective memory traffic in bytes after cache/register blocking reuse.
    pub effective_memory_bytes: u64,
    /// Estimated effective compute throughput in GFLOP/s.
    pub effective_gflops: f64,
    /// Estimated execution latency in nanoseconds.
    pub estimated_latency_ns: f64,
    /// Speedup factor relative to the naive baseline schedule (`TargetConfig::default()`).
    pub speedup_over_baseline: f64,
}

/// Complete auto-tuning report over a module for a target [`HardwareRoofline`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutotuneReport {
    /// Target hardware Roofline model used for scoring.
    pub roofline: HardwareRoofline,
    /// Baseline (naive `TargetConfig::default()`) latency in nanoseconds.
    pub baseline_latency_ns: f64,
    /// Best-performing candidate schedule discovered.
    pub best: AutotuneCandidate,
    /// Top ranked candidate schedules sorted by ascending `estimated_latency_ns`.
    pub candidates: Vec<AutotuneCandidate>,
    /// Total number of schedule candidates evaluated.
    pub total_evaluated: usize,
}

impl AutotuneReport {
    /// Formats a human-readable table of the top auto-tuned schedules.
    pub fn summary_table(&self, top_k: usize) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "=== CastForge Roofline-Guided Loop Schedule Auto-Tuner \
             ({:.0} GFLOP/s, {:.0} GB/s) ===",
            self.roofline.peak_gflops, self.roofline.peak_bandwidth_gbps
        );
        let _ = writeln!(
            out,
            "Evaluated Schedules: {} | Baseline Latency: {:.2} ns | Best Speedup: {:.2}x",
            self.total_evaluated, self.baseline_latency_ns, self.best.speedup_over_baseline
        );
        let _ = writeln!(
            out,
            "Recommended Schedule: {} (--loop-order {:?} --tile-size {} --reg-tile {})",
            self.best.label,
            self.best.config.loop_order,
            self.best.config.tile_m,
            self.best.config.reg_tile_m
        );
        let _ = writeln!(
            out,
            "{:<4} | {:<44} | {:>10} | {:>12} | {:>10} | {:>8}",
            "Rank", "Schedule Configuration", "Eff. Bytes", "Eff. GFLOP/s", "Est. (ns)", "Speedup"
        );
        let _ = writeln!(
            out,
            "{}",
            "-".repeat(102).chars().take(98).collect::<String>()
        );
        for (idx, cand) in self.candidates.iter().take(top_k).enumerate() {
            let _ = writeln!(
                out,
                "#{:<3} | {:<44} | {:>10} | {:>12.2} | {:>10.2} | {:>7.2}x",
                idx + 1,
                cand.label,
                cand.effective_memory_bytes,
                cand.effective_gflops,
                cand.estimated_latency_ns,
                cand.speedup_over_baseline
            );
        }
        out
    }
}

/// Searches the polyhedral/HPC loop schedule space for `module` under `roofline` and returns
/// the ranked [`AutotuneReport`].
pub fn autotune_module(module: &Module, roofline: &HardwareRoofline) -> AutotuneReport {
    let base_cost = analyze_module_cost_with_roofline(module, roofline);
    let (max_m, max_n, max_k) = dominant_gemm_dims(module);

    let baseline_cfg = TargetConfig::default();
    let baseline_cand = score_candidate(
        &baseline_cfg,
        &base_cost.total_flops,
        &base_cost.total_memory_traffic_bytes,
        (max_m, max_n, max_k),
        roofline,
        None,
    );
    let baseline_ns = baseline_cand.estimated_latency_ns.max(1e-6);

    let mut candidates = Vec::new();
    let orders = [GemmLoopOrder::Mnk, GemmLoopOrder::Mkn, GemmLoopOrder::Nkm];
    let l1_tiles = [0usize, 8, 16, 32];
    let l2_tiles = [0usize, 64];
    let reg_tiles = [(0usize, 0usize), (2, 2), (2, 4), (4, 4)];
    let unrolls = [1usize, 2, 4];
    let split_ks = [1usize, 2, 4];

    for &order in &orders {
        for &l1 in &l1_tiles {
            for &l2 in &l2_tiles {
                if l2 > 0 && l1 == 0 {
                    continue;
                }
                for &(mr, nr) in &reg_tiles {
                    for &uf in &unrolls {
                        for &sk in &split_ks {
                            for &vec_en in &[false, true] {
                                for &par_en in &[false, true] {
                                    let pf = if l1 >= 16 || mr >= 2 { 4 } else { 0 };
                                    let cfg = TargetConfig {
                                        tile_m: l1,
                                        tile_n: l1,
                                        tile_k: l1,
                                        simd_width: 8,
                                        wgsl_workgroup_size: 64,
                                        loop_order: order,
                                        unroll_factor: uf,
                                        vectorize: vec_en,
                                        reg_tile_m: mr,
                                        reg_tile_n: nr,
                                        collapse_loops: par_en,
                                        parallel: par_en,
                                        l2_tile_size: l2,
                                        prefetch_distance: pf,
                                        split_k: sk,
                                        loop_skew: false,
                                        double_buffer: l1 >= 16,
                                        simd_isa: crate::codegen::SimdIsa::Scalar,
                                    };
                                    let cand = score_candidate(
                                        &cfg,
                                        &base_cost.total_flops,
                                        &base_cost.total_memory_traffic_bytes,
                                        (max_m, max_n, max_k),
                                        roofline,
                                        Some(baseline_ns),
                                    );
                                    candidates.push(cand);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    candidates.sort_by(|a, b| {
        a.estimated_latency_ns
            .partial_cmp(&b.estimated_latency_ns)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_evaluated = candidates.len();
    let best = candidates.first().cloned().unwrap_or(baseline_cand);
    candidates.truncate(15);

    AutotuneReport {
        roofline: *roofline,
        baseline_latency_ns: baseline_ns,
        best,
        candidates,
        total_evaluated,
    }
}

fn score_candidate(
    cfg: &TargetConfig,
    total_flops: &u64,
    base_bytes: &u64,
    (m, n, k): (usize, usize, usize),
    roofline: &HardwareRoofline,
    baseline_ns: Option<f64>,
) -> AutotuneCandidate {
    let flops = (*total_flops).max(1) as f64;
    let raw_bytes = (*base_bytes).max(1) as f64;

    // 1. Model memory traffic reduction from register blocking (mr x nr), L1 tiling, and L2 tiling
    let mut mem_factor = 1.0_f64;
    if cfg.is_register_tiled() {
        let mr = cfg.reg_tile_m.min(m.max(2)) as f64;
        let nr = cfg.reg_tile_n.min(n.max(2)) as f64;
        // Register blocking reduces A+B reloads from 2*mr*nr to (mr + nr)
        let reg_reuse = (mr + nr) / (2.0 * mr * nr);
        mem_factor *= reg_reuse.clamp(0.25, 1.0);
    }
    if cfg.is_tiled() {
        let l1_fit = if cfg.tile_m <= 32 { 0.72 } else { 0.85 };
        mem_factor *= l1_fit;
    }
    if cfg.l2_tile_size > 0 {
        mem_factor *= 0.88;
    }
    if cfg.prefetch_distance > 0 {
        mem_factor *= 0.93;
    }

    // 2. Model compute throughput utilization (SIMD, loop order stride, unrolling, parallelism)
    let mut compute_util = 0.22_f64;
    match cfg.loop_order {
        GemmLoopOrder::Mkn => compute_util *= 1.35, // Unit-stride row-streaming
        GemmLoopOrder::Mnk => compute_util *= 1.0,
        GemmLoopOrder::Nkm => compute_util *= 0.95,
    }
    if cfg.vectorize {
        compute_util *= 1.85;
    }
    if cfg.unroll_factor > 1 {
        let uf_gain = match cfg.unroll_factor {
            2 => 1.18,
            4 => 1.28,
            _ => 1.22,
        };
        compute_util *= uf_gain;
    }
    if cfg.is_register_tiled() {
        compute_util *= 1.32;
    }
    if cfg.parallel {
        let par_ways = m.max(n).clamp(1, 4) as f64;
        compute_util *= 1.0 + 0.35 * (par_ways - 1.0);
    }
    if cfg.split_k > 1 {
        if m <= 2 && k >= cfg.split_k * 2 {
            compute_util *= 1.25;
        } else if k < cfg.split_k * 2 {
            compute_util *= 0.85;
        }
    }
    compute_util = compute_util.clamp(0.10, 1.0);

    let effective_memory_bytes = ((raw_bytes * mem_factor).round() as u64).max(1);
    let eff_peak_gflops = (roofline.peak_gflops * compute_util).max(1.0);
    let compute_ns = flops / eff_peak_gflops;
    let mem_ns = (effective_memory_bytes as f64) / roofline.peak_bandwidth_gbps.max(1.0);
    let estimated_latency_ns = compute_ns.max(mem_ns).max(0.01);
    let effective_gflops = flops / estimated_latency_ns;
    let speedup_over_baseline = match baseline_ns {
        Some(b) if b > 0.0 => b / estimated_latency_ns,
        _ => 1.0,
    };

    let order_str = match cfg.loop_order {
        GemmLoopOrder::Mnk => "MNK",
        GemmLoopOrder::Mkn => "MKN",
        GemmLoopOrder::Nkm => "NKM",
    };
    let label = format!(
        "ord={order_str} L1={} L2={} reg={}x{} uf={} sk={} vec={} par={}",
        cfg.tile_m,
        cfg.l2_tile_size,
        cfg.reg_tile_m,
        cfg.reg_tile_n,
        cfg.unroll_factor,
        cfg.split_k,
        u8::from(cfg.vectorize),
        u8::from(cfg.parallel)
    );

    AutotuneCandidate {
        label,
        config: *cfg,
        effective_memory_bytes,
        effective_gflops,
        estimated_latency_ns,
        speedup_over_baseline,
    }
}

fn dominant_gemm_dims(module: &Module) -> (usize, usize, usize) {
    let mut best = (4usize, 4usize, 4usize);
    let mut best_vol = 0usize;
    for op in module.ops.values() {
        if op.operands.len() >= 2 && !op.results.is_empty() {
            let lhs_d = static_shape(module, op.operands[0]);
            let out_d = static_shape(module, op.results[0]);
            if lhs_d.len() >= 2 && out_d.len() >= 2 {
                let m = out_d[0];
                let n = out_d[1];
                let k = *lhs_d.last().unwrap_or(&1);
                let vol = m * n * k;
                if vol > best_vol {
                    best_vol = vol;
                    best = (m, n, k);
                }
            }
        }
    }
    best
}

fn static_shape(module: &Module, val_id: crate::ir::ValueId) -> Vec<usize> {
    module.values[val_id]
        .ty
        .as_tensor()
        .map(|t| {
            t.shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n as usize,
                    _ => 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Stamps the schedule parameters of `config` (`loop_tile_m`, `loop_order`, `loop_unroll`,
/// `simd_vectorize`, etc.) directly onto all compute operations in `module`.
pub fn apply_schedule_to_module(module: &mut Module, config: &TargetConfig) {
    use crate::ir::Attribute;
    for (_, op) in module.ops.iter_mut() {
        if op.is_op("builtin", "module")
            || op.is_op("func", "func")
            || op.is_op("func", "return")
            || op.is_op("stablehlo", "constant")
        {
            continue;
        }
        if config.tile_m > 0 {
            op.attributes.insert(
                "loop_tile_m".to_string(),
                Attribute::Int(config.tile_m as i64),
            );
            op.attributes.insert(
                "loop_tile_n".to_string(),
                Attribute::Int(config.tile_n as i64),
            );
            op.attributes.insert(
                "loop_tile_k".to_string(),
                Attribute::Int(config.tile_k as i64),
            );
        }
        if config.l2_tile_size > 0 {
            op.attributes.insert(
                "loop_l2_tile".to_string(),
                Attribute::Int(config.l2_tile_size as i64),
            );
        }
        if config.reg_tile_m > 1 && config.reg_tile_n > 1 {
            op.attributes.insert(
                "loop_reg_m".to_string(),
                Attribute::Int(config.reg_tile_m as i64),
            );
            op.attributes.insert(
                "loop_reg_n".to_string(),
                Attribute::Int(config.reg_tile_n as i64),
            );
        }
        op.attributes.insert(
            "loop_order".to_string(),
            Attribute::String(config.loop_order.as_str().to_string()),
        );
        if config.unroll_factor > 1 {
            op.attributes.insert(
                "loop_unroll".to_string(),
                Attribute::Int(config.unroll_factor as i64),
            );
        }
        if config.prefetch_distance > 0 {
            op.attributes.insert(
                "loop_prefetch".to_string(),
                Attribute::Int(config.prefetch_distance as i64),
            );
        }
        if config.split_k > 1 {
            op.attributes.insert(
                "loop_split_k".to_string(),
                Attribute::Int(config.split_k as i64),
            );
        }
        if config.double_buffer {
            op.attributes
                .insert("loop_double_buffer".to_string(), Attribute::Bool(true));
        }
        if config.vectorize {
            op.attributes
                .insert("simd_vectorize".to_string(), Attribute::Bool(true));
            op.attributes.insert(
                "simd_width".to_string(),
                Attribute::Int(config.simd_width as i64),
            );
        }
        if config.parallel {
            op.attributes
                .insert("loop_parallel".to_string(), Attribute::Bool(true));
        }
    }
}

/// Pipeable Auto-Tuning Schedule Translator (`Ir(Module) -> Ir(Module)`).
///
/// Evaluates the Roofline-guided polyhedral schedule search space (`autotune_module`) and
/// stamps the winning [`TargetConfig`] attributes directly onto the IR so downstream
/// `codegen-*` translators automatically emit the optimal schedule.
#[derive(Default)]
pub struct AutotuneScheduleTranslator {
    /// Target hardware Roofline model used for schedule scoring.
    pub roofline: HardwareRoofline,
}

impl crate::ir::Translator for AutotuneScheduleTranslator {
    fn name(&self) -> &str {
        "autotune-schedule"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> anyhow::Result<crate::ir::Payload> {
        let mut module = input.into_ir()?;
        let report = autotune_module(&module, &self.roofline);
        apply_schedule_to_module(&mut module, &report.best.config);
        Ok(crate::ir::Payload::from_ir(module))
    }
}

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
