use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Dim, Module, Operation, ValueId};
use indexmap::IndexMap;

/// Roofline & Operation Statistics Summary for a `Module`.
#[derive(Debug, Clone, Default)]
pub struct RooflineSummary {
    pub total_ops: usize,
    pub total_params: i64,
    pub total_flops: i64,
    pub total_bytes_transferred: i64,
    pub arithmetic_intensity: f64,
    pub bottleneck: &'static str,
}

/// Computes FLOPs, bytes transferred, and Roofline Arithmetic Intensity (`FLOPs / Byte`)
/// for `module` against a hardware ridge point (`ridge_flops_per_byte`).
pub fn analyze_roofline(module: &Module, ridge_flops_per_byte: f64) -> RooflineSummary {
    let mut total_params = 0i64;
    let mut total_flops = 0i64;
    let mut total_bytes = 0i64;

    for (_, op) in &module.ops {
        if op.is_op("func", "func") || op.is_op("func", "return") {
            continue;
        }
        if op.is_op("stablehlo", "constant") {
            if let Some(&res) = op.results.first()
                && let Some(t) = module.values[res].ty.as_tensor()
            {
                total_params += t.num_elements().unwrap_or(0);
            }
            continue;
        }
        let (flops, bytes) = op_flops_and_bytes(module, op);
        total_flops += flops;
        total_bytes += bytes;
    }

    let ai = if total_bytes > 0 {
        (total_flops as f64) / (total_bytes as f64)
    } else {
        0.0
    };
    let bottleneck = if ai >= ridge_flops_per_byte {
        "Compute-Bound"
    } else {
        "Memory-Bandwidth-Bound"
    };

    RooflineSummary {
        total_ops: module.ops.len(),
        total_params,
        total_flops,
        total_bytes_transferred: total_bytes,
        arithmetic_intensity: ai,
        bottleneck,
    }
}

fn val_bytes(module: &Module, vid: ValueId) -> i64 {
    module.values[vid]
        .ty
        .as_tensor()
        .map(|t| {
            let elems = t.num_elements().unwrap_or(1).max(1);
            (elems * (t.dtype.bit_width() as i64) + 7) / 8
        })
        .unwrap_or(0)
}

fn op_flops_and_bytes(module: &Module, op: &Operation) -> (i64, i64) {
    let in_bytes: i64 = op.operands.iter().map(|&v| val_bytes(module, v)).sum();
    let out_bytes: i64 = op.results.iter().map(|&v| val_bytes(module, v)).sum();
    let bytes = in_bytes + out_bytes;

    let out_elems = op
        .results
        .first()
        .and_then(|&v| module.values[v].ty.as_tensor())
        .and_then(|t| t.num_elements())
        .unwrap_or(0);

    if op.is_op("stablehlo", "dot_general") && op.operands.len() >= 2 {
        let lhs_ty = module.values[op.operands[0]].ty.as_tensor();
        let rhs_ty = module.values[op.operands[1]].ty.as_tensor();
        if let (Some(l), Some(r)) = (lhs_ty, rhs_ty)
            && let (Some(Dim::Static(m)), Some(Dim::Static(k)), Some(Dim::Static(n))) =
                (l.shape.first(), l.shape.get(1), r.shape.get(1))
        {
            return (2 * m * k * n, bytes);
        }
    }

    if op.is_op("stablehlo", "composite") {
        let name = op
            .attributes
            .get("name")
            .and_then(|a| a.as_str())
            .unwrap_or("");
        if name == "castforge.fused_matmul_epilogue" && op.operands.len() >= 2 {
            let lhs_ty = module.values[op.operands[0]].ty.as_tensor();
            let rhs_ty = module.values[op.operands[1]].ty.as_tensor();
            if let (Some(l), Some(r)) = (lhs_ty, rhs_ty)
                && let (Some(Dim::Static(m)), Some(Dim::Static(k)), Some(Dim::Static(n))) =
                    (l.shape.first(), l.shape.get(1), r.shape.get(1))
            {
                return (2 * m * k * n + 2 * m * n, bytes);
            }
        } else if name == "castforge.fused_softmax" || name == "castforge.fused_rmsnorm" {
            return (out_elems * 5, bytes);
        }
    }

    (out_elems, bytes)
}

/// Print Op Stats & Roofline Pass (`print-op-stats`).
pub struct PrintOpStatsPass {
    pub ridge_flops_per_byte: f64,
}

impl Default for PrintOpStatsPass {
    fn default() -> Self {
        Self {
            ridge_flops_per_byte: 10.0,
        }
    }
}

impl Pass for PrintOpStatsPass {
    fn name(&self) -> &'static str {
        "print-op-stats"
    }

    fn description(&self) -> &'static str {
        "Computes operation counts, parameter sizes, FLOPs, and Roofline arithmetic intensity"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut counts: IndexMap<String, usize> = IndexMap::new();
        for (_, op) in &module.ops {
            *counts.entry(op.name.to_string()).or_insert(0) += 1;
        }
        let roof = analyze_roofline(module, self.ridge_flops_per_byte);

        eprintln!(
            "[castforge:stats] Total Ops: {} | Params: {} | FLOPs: {} | \
             Bytes: {} B | AI: {:.2} FLOP/B ({})",
            roof.total_ops,
            roof.total_params,
            roof.total_flops,
            roof.total_bytes_transferred,
            roof.arithmetic_intensity,
            roof.bottleneck
        );
        for (op_name, count) in &counts {
            eprintln!("  - {:<28} : {}", op_name, count);
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: false,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}
