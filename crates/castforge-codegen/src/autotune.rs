//! Roofline-guided polyhedral & loop schedule auto-tuner (`autotune_module`).
//!
//! Searches over candidate [`TargetConfig`] schedules (loop order `MNK`/`MKN`/`NKM`, L1/L2 cache
//! tile sizes, `(mr x nr)` Unroll-and-Jam register micro-kernels, inner-loop unroll factors,
//! Split-K reduction factors, SIMD vectorization, OpenMP parallelism, and software prefetching)
//! and scores each candidate against a [`HardwareRoofline`] and the module's tensor shapes.

use crate::cost_model::{HardwareRoofline, analyze_module_cost_with_roofline};
use crate::target_config::{GemmLoopOrder, TargetConfig};
use castforge_ir::{Dim, Module};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

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
                                        simd_isa: crate::SimdIsa::Scalar,
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

fn static_shape(module: &Module, val_id: castforge_ir::ValueId) -> Vec<usize> {
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
