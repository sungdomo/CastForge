//! Roofline, FLOPs, and memory bandwidth cost model for CastForge IR modules.

use crate::codegen::memory_plan::build_memory_plan;
use crate::ir::{Attribute, Dim, Module, OpId, ValueId};
use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Hardware Roofline parameters used to classify operations as compute- or memory-bound.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

    /// Creates a pre-configured [`HardwareRoofline`] for a named CPU or GPU architecture profile.
    pub fn for_arch(arch: &str) -> Self {
        match arch.trim().to_ascii_lowercase().as_str() {
            "x86_64-avx512" | "avx512" => Self::new(2048.0, 200.0),
            "x86_64-avx2" | "x86_64" | "avx2" => Self::new(1024.0, 100.0),
            "aarch64-sve" | "sve" => Self::new(1536.0, 160.0),
            "aarch64-neon" | "aarch64" | "arm64" | "neon" => Self::new(768.0, 100.0),
            "metal-apple" | "apple-m-series" | "metal" | "msl" => Self::new(14000.0, 400.0),
            "cuda-sm80" | "nvidia-ampere" | "cuda" => Self::new(19500.0, 1555.0),
            "nvidia-hopper" | "hopper" | "sm90" => Self::new(67000.0, 3350.0),
            "nvidia-blackwell" | "blackwell" | "sm100" => Self::new(140000.0, 8000.0),
            "rocm-gfx90a" | "amd-cdna3" | "hip" | "rocm" => Self::new(45000.0, 3200.0),
            "riscv64-rvv" | "riscv64" | "rvv" => Self::new(512.0, 64.0),
            "wasm32-simd128" | "wasm32" | "wasm" => Self::new(256.0, 32.0),
            _ => Self::default(),
        }
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
    } else if op.is_op("func", "call") {
        let callee = match op.attributes.get("callee") {
            Some(Attribute::SymbolRef(s)) | Some(Attribute::String(s)) => {
                s.trim_start_matches('@').to_string()
            }
            _ => "callee".to_string(),
        };
        (format!("func.call(@{callee})"), "call".to_string())
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
        "call" => {
            let callee_sym = match op.attributes.get("callee") {
                Some(Attribute::SymbolRef(s)) | Some(Attribute::String(s)) => {
                    s.trim_start_matches('@')
                }
                _ => "",
            };
            let mut callee_flops = 0u64;
            for fn_id in module.functions() {
                let sym = match module.ops[fn_id].attributes.get("sym_name") {
                    Some(Attribute::SymbolRef(s)) | Some(Attribute::String(s)) => {
                        s.trim_start_matches('@')
                    }
                    _ => "",
                };
                if sym == callee_sym
                    && let Some(&reg) = module.ops[fn_id].regions.first()
                    && let Some(&blk) = module.regions[reg].blocks.first()
                {
                    for &inner_id in &module.blocks[blk].ops {
                        if inner_id != op_id && !module.ops[inner_id].is_op("func", "call") {
                            callee_flops += profile_single_op(module, inner_id, roofline).flops;
                        }
                    }
                }
            }
            callee_flops.max(out_elems)
        }
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
        | "exponential"
        | "log"
        | "sine"
        | "cosine"
        | "rsqrt"
        | "sqrt"
        | "rms_norm"
        | "fused_rmsnorm"
        | "fused_residual_rmsnorm"
        | "fused_softmax"
        | "fused_log_softmax" => 4 * out_elems,
        "fma" | "fused_add_max" => 2 * out_elems,
        "fused_moe" => {
            let lhs_dims = op
                .operands
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let k = lhs_dims.last().copied().unwrap_or(1);
            4 * out_elems * k + 3 * out_elems
        }
        "fused_map_reduce" => {
            2 * op
                .operands
                .first()
                .map(|&v| value_elems(module, v))
                .unwrap_or(out_elems)
        }
        "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum" | "power" | "negate"
        | "abs" | "relu" | "clamp" | "dequantize_int8" | "uniform_dequantize" => out_elems,
        "reduce" | "reduce_sum" | "reduce_max" => op
            .operands
            .first()
            .map(|&v| value_elems(module, v))
            .unwrap_or(out_elems),
        "convolution" => {
            let rhs_dims = op
                .operands
                .get(1)
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let out_dims = op
                .results
                .first()
                .map(|&v| static_dims(module, v))
                .unwrap_or_default();
            let rhs_total: u64 = rhs_dims.iter().copied().product::<u64>().max(1);
            let c_out = out_dims
                .last()
                .copied()
                .filter(|&c| rhs_dims.last() == Some(&c))
                .or_else(|| rhs_dims.first().copied())
                .unwrap_or(1)
                .max(1);
            let k_total = (rhs_total / c_out).max(1);
            2 * out_elems * k_total
        }
        "im2col" => 0,
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

/// Loop permutation order for 2D Matrix Multiplication (`M`, `N`, `K`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GemmLoopOrder {
    /// Classic inner-product reduction order (`for m { for n { for k } }`).
    #[default]
    Mnk,
    /// Broadcast-A / row-streaming outer-product order (`for m { for k { for n } }`),
    /// providing unit-stride `n` access across unpacked `B[k, n]` and `C[m, n]`.
    Mkn,
    /// Column-stationary order (`for n { for k { for m } }`).
    Nkm,
}

impl GemmLoopOrder {
    /// Parses a loop order string (`"mnk"`, `"mkn"`, `"nkm"`).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "mkn" | "ikj" => Self::Mkn,
            "nkm" | "jki" => Self::Nkm,
            _ => Self::Mnk,
        }
    }

    /// Returns the canonical lowercase string (`"mnk"`, `"mkn"`, `"nkm"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mnk => "mnk",
            Self::Mkn => "mkn",
            Self::Nkm => "nkm",
        }
    }
}

/// Hardware SIMD instruction set architecture for explicit multi-arch intrinsic emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SimdIsa {
    /// Portable scalar / auto-vectorized loops (`#pragma omp simd`).
    #[default]
    Scalar,
    /// x86_64 AVX2 + FMA 256-bit (`8 x f32`) hardware intrinsics (`_mm256_fmadd_ps`).
    Avx2,
    /// x86_64 AVX-512F 512-bit (`16 x f32`) hardware intrinsics (`_mm512_fmadd_ps`).
    Avx512,
    /// ARM64 NEON 128-bit (`4 x f32`) hardware intrinsics (`vfmaq_f32`).
    Neon,
    /// ARM64 SVE/SVE2 scalable vector (`8 x f32`) hardware intrinsics (`svmla_f32_x`).
    Sve,
    /// RISC-V Vector 1.0 (`rv64gcv`, `8 x f32`) hardware intrinsics (`__riscv_vfmacc_vv_f32m1`).
    Rvv,
    /// WebAssembly SIMD128 (`wasm32`, `4 x f32`) hardware intrinsics (`f32x4_mul` + `f32x4_add`).
    WasmSimd128,
}

impl SimdIsa {
    /// Parses a multi-architecture SIMD ISA identifier (`"scalar"`, `"avx2"`, `"avx512"`,
    /// `"neon"`, `"sve"`, `"rvv"`, `"wasm"`).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "avx512" | "avx-512" | "avx512f" => Self::Avx512,
            "avx2" | "fma" | "x86_64" => Self::Avx2,
            "sve" | "sve2" | "armv9" => Self::Sve,
            "neon" | "aarch64" | "arm64" => Self::Neon,
            "rvv" | "riscv" | "riscv64" => Self::Rvv,
            "wasm" | "wasm32" | "simd128" | "wasm-simd128" => Self::WasmSimd128,
            _ => Self::Scalar,
        }
    }

    /// Canonical lowercase ISA identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::Sve => "sve",
            Self::Rvv => "rvv",
            Self::WasmSimd128 => "wasm_simd128",
        }
    }

    /// Default `f32` vector lane width for this architecture.
    pub fn default_lanes(&self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Avx512 => 16,
            Self::Avx2 | Self::Sve | Self::Rvv => 8,
            Self::Neon | Self::WasmSimd128 => 4,
        }
    }

    /// Primary FMA/MAC hardware intrinsic symbol for this ISA.
    pub fn intrinsic_name(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar_fma",
            Self::Avx2 => "_mm256_fmadd_ps",
            Self::Avx512 => "_mm512_fmadd_ps",
            Self::Neon => "vfmaq_f32",
            Self::Sve => "svmla_f32_x",
            Self::Rvv => "__riscv_vfmacc_vv_f32m1",
            Self::WasmSimd128 => "f32x4_add_mul",
        }
    }
}

/// Hardware target descriptor parameterizing cache tiling, loop permutation, unrolling,
/// SIMD vectorization, and GPU workgroup dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Cache / register tile size along the M (row) dimension (`0` disables loop tiling).
    pub tile_m: usize,
    /// Cache / register tile size along the N (column) dimension (`0` disables loop tiling).
    pub tile_n: usize,
    /// Cache tile size along the K (reduction) dimension (`0` disables loop tiling).
    pub tile_k: usize,
    /// Target SIMD vector width in `f32` lanes (e.g., `8` for AVX2, `4` for NEON).
    pub simd_width: usize,
    /// WebGPU `@workgroup_size(X)` thread count per compute workgroup.
    pub wgsl_workgroup_size: u32,
    /// GEMM loop permutation order (`Mnk`, `Mkn`, `Nkm`).
    pub loop_order: GemmLoopOrder,
    /// Inner-loop unroll factor (`1` disables unrolling; `2`, `4`, `8` unroll with tail peeling).
    pub unroll_factor: usize,
    /// Emit explicit SIMD vectorization loops/pragmas with tail-loop peeling.
    pub vectorize: bool,
    /// Register micro-kernel row blocking factor (`mr` for Unroll-and-Jam; `0` or `1` = disabled).
    pub reg_tile_m: usize,
    /// Register micro-kernel col blocking factor (`nr` for Unroll-and-Jam; `0` or `1` = disabled).
    pub reg_tile_n: usize,
    /// Collapse/coalesce perfectly nested 2D loops into a single 1D loop.
    pub collapse_loops: bool,
    /// Emit multicore parallel outer-loop work-sharing (`#pragma omp parallel for`).
    pub parallel: bool,
    /// Outer L2/L3 cache tile size (`0` disables hierarchical 2-level L2->L1 cache blocking).
    pub l2_tile_size: usize,
    /// Software pipelining / hardware prefetch distance in iterations (`0` disables prefetching).
    pub prefetch_distance: usize,
    /// Split-K reduction partitioning factor (`1` disables Split-K; `> 1` partitions `k`).
    pub split_k: usize,
    /// Enable polyhedral wavefront loop skewing (`w = i + j` diagonal wavefront traversal).
    pub loop_skew: bool,
    /// Enable 2-stage software-pipelined ping-pong tile double-buffering across K tiles.
    pub double_buffer: bool,
    /// Hardware SIMD ISA for explicit intrinsic codegen (`Scalar`, `Avx2`, `Neon`).
    pub simd_isa: SimdIsa,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            tile_m: 0,
            tile_n: 0,
            tile_k: 0,
            simd_width: 8,
            wgsl_workgroup_size: 64,
            loop_order: GemmLoopOrder::Mnk,
            unroll_factor: 1,
            vectorize: false,
            reg_tile_m: 0,
            reg_tile_n: 0,
            collapse_loops: false,
            parallel: false,
            l2_tile_size: 0,
            prefetch_distance: 0,
            split_k: 1,
            loop_skew: false,
            double_buffer: false,
            simd_isa: SimdIsa::Scalar,
        }
    }
}

impl TargetConfig {
    /// Create a symmetric `(tile x tile x tile)` cache-blocked CPU configuration.
    pub fn cpu_tiled(tile: usize) -> Self {
        Self {
            tile_m: tile,
            tile_n: tile,
            tile_k: tile,
            ..Self::default()
        }
    }

    /// Create a WebGPU workgroup-tiled configuration with shared `var<workgroup>` tile memory.
    pub fn gpu_workgroup(tile: usize, workgroup_size: u32) -> Self {
        Self {
            tile_m: tile,
            tile_n: tile,
            tile_k: tile,
            wgsl_workgroup_size: workgroup_size.max(1),
            ..Self::default()
        }
    }

    /// Configure the GEMM loop permutation order (`Mnk`, `Mkn`, `Nkm`).
    pub fn with_loop_order(mut self, order: GemmLoopOrder) -> Self {
        self.loop_order = order;
        self
    }

    /// Configure the inner-loop unroll factor (`1` = disabled, `2`/`4`/`8` = unrolled + peeled).
    pub fn with_unroll(mut self, factor: usize) -> Self {
        self.unroll_factor = factor.max(1);
        self
    }

    /// Enable or disable explicit SIMD vectorization and tail peeling.
    pub fn with_vectorize(mut self, enabled: bool) -> Self {
        self.vectorize = enabled;
        self
    }

    /// Configure `(mr x nr)` outer-loop Unroll-and-Jam register micro-kernel blocking.
    pub fn with_register_tile(mut self, mr: usize, nr: usize) -> Self {
        self.reg_tile_m = mr;
        self.reg_tile_n = nr;
        self
    }

    /// Enable or disable 2D -> 1D loop coalescing / flattening.
    pub fn with_collapse_loops(mut self, enabled: bool) -> Self {
        self.collapse_loops = enabled;
        self
    }

    /// Enable or disable multicore parallel outer-loop work-sharing.
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Configure hierarchical L2 cache tile size wrapping L1 `tile_m`/`tile_n`/`tile_k` blocks.
    pub fn with_l2_tile_size(mut self, l2_tile: usize) -> Self {
        self.l2_tile_size = l2_tile;
        self
    }

    /// Configure software pipelining / hardware prefetch lookahead distance along `k`.
    pub fn with_prefetch(mut self, distance: usize) -> Self {
        self.prefetch_distance = distance;
        self
    }

    /// Configure Split-K reduction partitioning across `splits` partial accumulators.
    pub fn with_split_k(mut self, splits: usize) -> Self {
        self.split_k = splits.max(1);
        self
    }

    /// Enable or disable polyhedral wavefront loop skewing (`w = i + j`).
    pub fn with_loop_skew(mut self, enabled: bool) -> Self {
        self.loop_skew = enabled;
        self
    }

    /// Enable or disable 2-stage software-pipelined ping-pong tile double-buffering.
    pub fn with_double_buffer(mut self, enabled: bool) -> Self {
        self.double_buffer = enabled;
        self
    }

    /// Select the target hardware SIMD instruction set (`Scalar`, `Avx2`, `Avx512`, `Neon`, `Sve`,
    /// `Rvv`, `WasmSimd128`).
    pub fn with_simd_isa(mut self, isa: SimdIsa) -> Self {
        self.simd_isa = isa;
        if isa != SimdIsa::Scalar {
            self.vectorize = true;
            self.simd_width = isa.default_lanes();
        }
        self
    }

    /// Creates a pre-tuned hardware [`TargetConfig`] for a named CPU or GPU architecture profile
    /// (`"x86_64-avx2"`, `"x86_64-avx512"`, `"aarch64-neon"`, `"aarch64-sve"`, `"riscv64-rvv"`,
    /// `"wasm32-simd128"`, `"cuda-sm80"`, `"rocm-gfx90a"`, `"metal-apple"`, `"webgpu"`).
    pub fn for_arch(arch: &str) -> Self {
        match arch.trim().to_ascii_lowercase().as_str() {
            "x86_64-avx512" | "avx512" => Self::cpu_tiled(32)
                .with_simd_isa(SimdIsa::Avx512)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "x86_64-avx2" | "x86_64" | "avx2" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Avx2)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "aarch64-sve" | "sve" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Sve)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "aarch64-neon" | "aarch64" | "arm64" | "neon" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Neon)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "riscv64-rvv" | "riscv64" | "rvv" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Rvv)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(2),
            "wasm32-simd128" | "wasm32" | "wasm" => Self::cpu_tiled(8)
                .with_simd_isa(SimdIsa::WasmSimd128)
                .with_loop_order(GemmLoopOrder::Mkn),
            "cuda-sm80" | "cuda" | "rocm-gfx90a" | "hip" => Self::gpu_workgroup(16, 256),
            "metal-apple" | "metal" | "msl" | "opencl" | "webgpu" | "wgsl" => {
                Self::gpu_workgroup(16, 64)
            }
            _ => Self::default(),
        }
    }

    /// Returns `true` when 2D/3D loop tiling is enabled.
    pub fn is_tiled(&self) -> bool {
        self.tile_m > 0 && self.tile_n > 0 && self.tile_k > 0
    }

    /// Returns `true` when `(mr x nr)` register micro-kernel Unroll-and-Jam is enabled.
    pub fn is_register_tiled(&self) -> bool {
        self.reg_tile_m > 1 && self.reg_tile_n > 1
    }

    /// Merges per-operation schedule attributes stamped by IR-to-IR loop passes
    /// (`schedule-loops`) into a copy of this [`TargetConfig`].
    pub fn with_op_attributes(&self, op: &crate::ir::Operation) -> Self {
        let attrs = &op.attributes;
        let mut cfg = *self;
        if let Some(Attribute::Int(v)) = attrs.get("loop_tile_m") {
            cfg.tile_m = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_tile_n") {
            cfg.tile_n = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_tile_k") {
            cfg.tile_k = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_l2_tile") {
            cfg.l2_tile_size = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_reg_m") {
            cfg.reg_tile_m = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_reg_n") {
            cfg.reg_tile_n = (*v).max(0) as usize;
        }
        if let Some(Attribute::String(s)) = attrs.get("loop_order") {
            cfg.loop_order = GemmLoopOrder::parse(s);
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_collapse") {
            cfg.collapse_loops = *b;
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_skew") {
            cfg.loop_skew = *b;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_unroll") {
            cfg.unroll_factor = (*v).max(1) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_prefetch") {
            cfg.prefetch_distance = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_split_k") {
            cfg.split_k = (*v).max(1) as usize;
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_double_buffer") {
            cfg.double_buffer = *b;
        }
        if let Some(Attribute::Bool(b)) = attrs.get("simd_vectorize") {
            cfg.vectorize = *b;
        }
        if let Some(Attribute::Int(v)) = attrs.get("simd_width") {
            cfg.simd_width = (*v).max(1) as usize;
        }
        if let Some(Attribute::String(s)) = attrs.get("simd_isa") {
            cfg = cfg.with_simd_isa(SimdIsa::parse(s));
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_parallel") {
            cfg.parallel = *b;
        }
        cfg
    }

    /// Scans `module` for any IR schedule attributes and returns an updated module-level
    /// [`TargetConfig`] (used for emitting top-level SIMD headers or module banners).
    pub fn with_module_attributes(&self, module: &crate::ir::Module) -> Self {
        let mut cfg = *self;
        for (_, op) in &module.ops {
            cfg = cfg.with_op_attributes(op);
        }
        cfg
    }
}
