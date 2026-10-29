//! Roofline, FLOPs, and memory bandwidth cost model for CastForge IR modules.

use crate::memory_plan::build_memory_plan;
use castforge_ir::{Attribute, Dim, Module, OpId, ValueId};
use serde::Serialize;
use std::fmt::Write;

/// Hardware Roofline parameters used to classify operations as compute- or memory-bound.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HardwareRoofline {
    /// Peak compute throughput in GFLOP/s (10^9 FLOP/s, equivalent to FLOP/ns).
    pub peak_gflops: f64,
    /// Peak memory bandwidth in GB/s (10^9 Bytes/s, equivalent to Bytes/ns).
    pub peak_bandwidth_gbps: f64,
}

impl Default for HardwareRoofline {
    fn default() -> Self {
        Self {
            peak_gflops: 1000.0,
            peak_bandwidth_gbps: 100.0,
        }
    }
}

impl HardwareRoofline {
    /// Creates a custom hardware roofline configuration.
    pub fn new(peak_gflops: f64, peak_bandwidth_gbps: f64) -> Self {
        Self {
            peak_gflops: peak_gflops.max(1e-3),
            peak_bandwidth_gbps: peak_bandwidth_gbps.max(1e-3),
        }
    }

    /// Returns the Roofline ridge point in FLOPs/Byte (`peak_gflops / peak_bandwidth_gbps`).
    pub fn ridge_point(&self) -> f64 {
        self.peak_gflops / self.peak_bandwidth_gbps.max(1e-9)
    }
}

/// Roofline bottleneck classification for an operation or module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BottleneckKind {
    /// Arithmetic intensity is at or above the hardware ridge point.
    ComputeBound,
    /// Arithmetic intensity is below the hardware ridge point.
    MemoryBound,
    /// Compile-time metadata or terminator with zero runtime traffic.
    MetadataOnly,
}

impl std::fmt::Display for BottleneckKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComputeBound => write!(f, "Compute-Bound"),
            Self::MemoryBound => write!(f, "Memory-Bound"),
            Self::MetadataOnly => write!(f, "Metadata-Only"),
        }
    }
}

/// Cost and Roofline metrics for a single IR operation.
#[derive(Debug, Clone, Serialize)]
pub struct OpCostProfile {
    pub op_name: String,
    pub result_name: String,
    pub flops: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub arithmetic_intensity: f64,
    pub bottleneck: BottleneckKind,
    pub estimated_ns: f64,
}

impl OpCostProfile {
    /// Total memory traffic (`bytes_read + bytes_written`) for this operation.
    pub fn total_bytes(&self) -> u64 {
        self.bytes_read + self.bytes_written
    }
}

/// Whole-module FLOPs, memory traffic, arithmetic intensity, and Roofline summary.
#[derive(Debug, Clone, Serialize)]
pub struct ModuleCostReport {
    pub roofline: HardwareRoofline,
    pub ops: Vec<OpCostProfile>,
    pub total_flops: u64,
    pub total_bytes_read: u64,
    pub total_bytes_written: u64,
    pub total_memory_traffic_bytes: u64,
    pub arithmetic_intensity: f64,
    pub overall_bottleneck: BottleneckKind,
    pub estimated_latency_ns: f64,
    pub peak_activation_bytes: usize,
    pub naive_activation_bytes: usize,
    pub total_constant_bytes: usize,
    pub compute_bound_ops: usize,
    pub memory_bound_ops: usize,
}

impl ModuleCostReport {
    /// Renders a human-readable Roofline & cost summary table.
    pub fn summary_table(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Roofline Target: {:.1} GFLOP/s | {:.1} GB/s (Ridge: {:.2} FLOP/Byte)",
            self.roofline.peak_gflops,
            self.roofline.peak_bandwidth_gbps,
            self.roofline.ridge_point()
        );
        let _ = writeln!(
            out,
            "{:<34} {:>10} {:>12} {:>10} {:>15}",
            "Operation", "FLOPs", "Traffic (B)", "AI (F/B)", "Bottleneck"
        );
        let _ = writeln!(out, "{}", "-".repeat(85));
        for op in &self.ops {
            if op.bottleneck == BottleneckKind::MetadataOnly {
                continue;
            }
            let _ = writeln!(
                out,
                "{:<34} {:>10} {:>12} {:>10.2} {:>15}",
                op.op_name,
                op.flops,
                op.total_bytes(),
                op.arithmetic_intensity,
                op.bottleneck.to_string()
            );
        }
        let _ = writeln!(out, "{}", "-".repeat(85));
        let _ = writeln!(
            out,
            "Total FLOPs: {} | Traffic: {} B | AI: {:.2} FLOP/B | {} | Est: {:.2} ns",
            self.total_flops,
            self.total_memory_traffic_bytes,
            self.arithmetic_intensity,
            self.overall_bottleneck,
            self.estimated_latency_ns
        );
        let _ = writeln!(
            out,
            "Arena Workspace: {} B (Naive: {} B, Constants: {} B)",
            self.peak_activation_bytes, self.naive_activation_bytes, self.total_constant_bytes
        );
        out
    }
}

/// Analyzes `module` using the default [`HardwareRoofline`].
pub fn analyze_module_cost(module: &Module) -> ModuleCostReport {
    analyze_module_cost_with_roofline(module, &HardwareRoofline::default())
}

/// Analyzes `module` FLOPs, memory traffic, and Roofline bottlenecks for `roofline`.
pub fn analyze_module_cost_with_roofline(
    module: &Module,
    roofline: &HardwareRoofline,
) -> ModuleCostReport {
    let mem_plan = build_memory_plan(module);
    let mut ops = Vec::new();
    let mut total_flops = 0u64;
    let mut total_bytes_read = 0u64;
    let mut total_bytes_written = 0u64;
    let mut estimated_latency_ns = 0.0f64;
    let mut compute_bound_ops = 0usize;
    let mut memory_bound_ops = 0usize;

    for func_id in module.functions() {
        let Some(body_reg) = module.ops[func_id].regions.first().copied() else {
            continue;
        };
        let Some(entry_blk) = module.regions[body_reg].blocks.first().copied() else {
            continue;
        };
        let op_ids = module.blocks[entry_blk].ops.clone();
        for op_id in op_ids {
            let profile = profile_single_op(module, op_id, roofline);
            total_flops += profile.flops;
            total_bytes_read += profile.bytes_read;
            total_bytes_written += profile.bytes_written;
            estimated_latency_ns += profile.estimated_ns;
            match profile.bottleneck {
                BottleneckKind::ComputeBound => compute_bound_ops += 1,
                BottleneckKind::MemoryBound => memory_bound_ops += 1,
                BottleneckKind::MetadataOnly => {}
            }
            ops.push(profile);
        }
    }

    let total_memory_traffic_bytes = total_bytes_read + total_bytes_written;
    let arithmetic_intensity = if total_memory_traffic_bytes > 0 {
        total_flops as f64 / total_memory_traffic_bytes as f64
    } else {
        0.0
    };
    let overall_bottleneck = if total_flops == 0 && total_memory_traffic_bytes == 0 {
        BottleneckKind::MetadataOnly
    } else if arithmetic_intensity >= roofline.ridge_point() {
        BottleneckKind::ComputeBound
    } else {
        BottleneckKind::MemoryBound
    };

    ModuleCostReport {
        roofline: *roofline,
        ops,
        total_flops,
        total_bytes_read,
        total_bytes_written,
        total_memory_traffic_bytes,
        arithmetic_intensity,
        overall_bottleneck,
        estimated_latency_ns,
        peak_activation_bytes: mem_plan.total_activation_bytes,
        naive_activation_bytes: mem_plan.naive_activation_bytes,
        total_constant_bytes: mem_plan.total_constant_bytes,
        compute_bound_ops,
        memory_bound_ops,
    }
}

fn value_bytes(module: &Module, val_id: ValueId) -> u64 {
    let Some(t) = module.values[val_id].ty.as_tensor() else {
        return 0;
    };
    let elems = t.num_elements().unwrap_or(1).max(1) as u64;
    (elems * t.dtype.bit_width() as u64).div_ceil(8)
}

fn value_elems(module: &Module, val_id: ValueId) -> u64 {
    module.values[val_id]
        .ty
        .as_tensor()
        .and_then(|t| t.num_elements())
        .unwrap_or(1)
        .max(1) as u64
}

fn static_dims(module: &Module, val_id: ValueId) -> Vec<u64> {
    module.values[val_id]
        .ty
        .as_tensor()
        .map(|t| {
            t.shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n as u64,
                    _ => 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn profile_single_op(module: &Module, op_id: OpId, roofline: &HardwareRoofline) -> OpCostProfile {
    let op = &module.ops[op_id];
    let (full_name, short_str) = if op.name.name == "composite" {
        if let Some(Attribute::String(comp_name)) = op.attributes.get("name") {
            let short = comp_name
                .rsplit_once('.')
                .map(|(_, s)| s.to_string())
                .unwrap_or_else(|| comp_name.clone());
            (comp_name.clone(), short)
        } else {
            (op.name.full_name(), op.name.name.clone())
        }
    } else {
        (op.name.full_name(), op.name.name.clone())
    };
    let short = short_str.as_str();
    let result_name = op
        .results
        .first()
        .and_then(|&r| module.values[r].ssa_name.clone())
        .unwrap_or_else(|| "()".to_string());

    if short == "constant" || short == "return" {
        return OpCostProfile {
            op_name: full_name,
            result_name,
            flops: 0,
            bytes_read: 0,
            bytes_written: 0,
            arithmetic_intensity: 0.0,
            bottleneck: BottleneckKind::MetadataOnly,
            estimated_ns: 0.0,
        };
    }

    let active_operands: &[ValueId] = if short == "fused_matmul_epilogue" {
        let epilogue = match op.attributes.get("epilogue") {
            Some(Attribute::String(s)) => s.as_str(),
            _ => "bias",
        };
        if epilogue == "bias_relu" && op.operands.len() > 3 {
            &op.operands[..3]
        } else {
            &op.operands
        }
    } else {
        &op.operands
    };
    let bytes_read: u64 = active_operands
        .iter()
        .map(|&v| value_bytes(module, v))
        .sum();
    let bytes_written: u64 = op.results.iter().map(|&v| value_bytes(module, v)).sum();
    let out_elems = op
        .results
        .first()
        .map(|&r| value_elems(module, r))
        .unwrap_or(0);

    let flops = match short {
        "dot_general" | "fused_matmul_bias" | "fused_matmul_epilogue" | "fused_qmatmul" => {
            let lhs_dims = op
                .operands
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let out_dims = op
                .results
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let m = out_dims.first().copied().unwrap_or(1);
            let n = out_dims.get(1).copied().unwrap_or(1);
            let k = lhs_dims.last().copied().unwrap_or(1);
            let mut gemm_flops = 2 * m * n * k;
            if short == "fused_matmul_bias" {
                gemm_flops += m * n;
            } else if short == "fused_matmul_epilogue" || short == "fused_qmatmul" {
                if op.operands.len() >= 3 {
                    gemm_flops += m * n;
                }
                let epilogue = match op.attributes.get("epilogue") {
                    Some(Attribute::String(s)) => s.as_str(),
                    _ => "bias",
                };
                gemm_flops += match epilogue {
                    "bias_relu" | "bias_scale" => m * n,
                    "bias_tanh" | "bias_silu" => 4 * m * n,
                    "bias_gelu" => 8 * m * n,
                    _ => 0,
                };
            }
            gemm_flops
        }
        "fused_gated_mlp" => {
            let lhs_dims = op
                .operands
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let out_dims = op
                .results
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let m = out_dims.first().copied().unwrap_or(1);
            let n = out_dims.get(1).copied().unwrap_or(1);
            let k = lhs_dims.last().copied().unwrap_or(1);
            4 * m * n * k + 5 * m * n
        }
        "lora_linear" => {
            let x_dims = op
                .operands
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let a_dims = op
                .operands
                .get(2)
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let out_dims = op
                .results
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let m = x_dims.first().copied().unwrap_or(1);
            let k = x_dims.last().copied().unwrap_or(1);
            let r = a_dims.last().copied().unwrap_or(1);
            let n = out_dims.last().copied().unwrap_or(1);
            2 * m * n * k + 2 * m * k * r + 2 * m * r * n + m * n
        }
        "scaled_dot_product_attention" | "fused_attention" => {
            let q_dims = op
                .operands
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let (s, d) = match q_dims.as_slice() {
                [s, d] => (*s, *d),
                [b, s, d] => (b * *s, *d),
                _ => (out_elems, 1),
            };
            4 * s * s * d + 5 * s * s
        }
        "swiglu" | "fused_swiglu" | "fused_geglu" | "layer_norm" | "fused_layernorm" => {
            6 * out_elems
        }
        "rope" | "fused_rope" => 3 * out_elems,
        "gelu"
        | "fused_gelu"
        | "silu"
        | "sigmoid"
        | "logistic"
        | "tanh"
        | "exp"
        | "log"
        | "rsqrt"
        | "sqrt"
        | "rms_norm"
        | "fused_rmsnorm"
        | "fused_residual_rmsnorm"
        | "fused_softmax" => 4 * out_elems,
        "fma" => 2 * out_elems,
        "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum" | "power" | "negate"
        | "abs" | "relu" | "clamp" | "dequantize_int8" => out_elems,
        "reduce" | "reduce_sum" | "reduce_max" => op
            .operands
            .first()
            .map(|&v| value_elems(module, v))
            .unwrap_or(out_elems),
        _ => 0,
    };

    let total_bytes = bytes_read + bytes_written;
    let arithmetic_intensity = if total_bytes > 0 {
        flops as f64 / total_bytes as f64
    } else {
        0.0
    };
    let bottleneck = if arithmetic_intensity >= roofline.ridge_point() {
        BottleneckKind::ComputeBound
    } else {
        BottleneckKind::MemoryBound
    };
    let compute_ns = flops as f64 / roofline.peak_gflops;
    let mem_ns = total_bytes as f64 / roofline.peak_bandwidth_gbps;
    let estimated_ns = compute_ns.max(mem_ns);

    OpCostProfile {
        op_name: full_name,
        result_name,
        flops,
        bytes_read,
        bytes_written,
        arithmetic_intensity,
        bottleneck,
        estimated_ns,
    }
}
