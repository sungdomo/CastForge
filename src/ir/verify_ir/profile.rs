//! Submodule `profile` of `verify_ir`.

use crate::ir::{Attribute, Dim, Module, OpName, Operation, ValueId};
use crate::ir::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::smallvec;

/// Numerical Sanitizer Pass (`sanitize-numerics`).
#[derive(Debug, Clone)]
pub struct SanitizeNumericsPass {
    /// Maximum finite magnitude clamp bound (default: `65504.0`).
    pub clamp_max: f64,
    /// Epsilon floor for denominators and `rsqrt` inputs (default: `1e-6`).
    pub eps: f64,
}

impl Default for SanitizeNumericsPass {
    fn default() -> Self {
        Self {
            clamp_max: 65504.0,
            eps: 1e-6,
        }
    }
}

impl Pass for SanitizeNumericsPass {
    fn name(&self) -> &'static str {
        "sanitize-numerics"
    }

    fn description(&self) -> &'static str {
        "Injects castforge.nan_guard clamps on numerically sensitive ops and function outputs"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let clamp = self.clamp_max.abs().max(1.0);
        let eps = self.eps.abs().max(1e-12);

        for fn_op_id in module.functions() {
            let reg_id = module.ops[fn_op_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.is_op("stablehlo", "composite")
                    && op.attributes.get("name").and_then(|a| a.as_str())
                        == Some("castforge.nan_guard")
                {
                    continue;
                }

                let is_sensitive = matches!(
                    op.name.name.as_str(),
                    "divide" | "rsqrt" | "exponential" | "convert" | "dot_general"
                );
                if !is_sensitive || op.results.is_empty() {
                    continue;
                }

                let orig_res = op.results[0];
                let res_ty = module.values[orig_res].ty.clone();
                let base_ssa = module.values[orig_res]
                    .ssa_name
                    .as_deref()
                    .unwrap_or("v")
                    .to_string();

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.nan_guard".to_string()),
                );
                attrs.insert("clamp_max".to_string(), Attribute::Float(clamp));
                attrs.insert("eps".to_string(), Attribute::Float(eps));

                let guard_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "composite"),
                    smallvec![orig_res],
                    vec![(Some(format!("{base_ssa}_san")), res_ty)],
                    attrs,
                    Vec::new(),
                    false,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let guard_res: ValueId = module.ops[guard_op_id].results[0];

                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == op_id || other_id == guard_op_id {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = guard_res;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id].ops.insert(pos + 1, guard_op_id);
                }
                created += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: 0,
            ops_modified: created,
            ops_created: created,
        })
    }
}

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
    }
    .max(0.0);
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

pub(super) fn op_flops_and_bytes(module: &Module, op: &Operation) -> (i64, i64) {
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

/// Per-Op Profiling Region Injector Pass (`instrument-profiling`).
#[derive(Debug, Clone)]
pub struct InstrumentProfilingPass {
    pub filter: String,
}

impl Default for InstrumentProfilingPass {
    fn default() -> Self {
        Self {
            filter: "gemm".to_string(),
        }
    }
}

impl Pass for InstrumentProfilingPass {
    fn name(&self) -> &'static str {
        "instrument-profiling"
    }

    fn description(&self) -> &'static str {
        "Injects castforge.profile_region timing markers around target compute operations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let all_ops = self.filter.eq_ignore_ascii_case("all");
        let mut created = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.is_op("func", "return")
                    || op.is_op("stablehlo", "constant")
                    || (op.is_op("stablehlo", "composite")
                        && op.attributes.get("name").and_then(|a| a.as_str())
                            == Some("castforge.profile_region"))
                {
                    continue;
                }
                let is_target = if all_ops {
                    !op.results.is_empty()
                } else {
                    op.is_op("stablehlo", "dot_general")
                        || op.is_op("stablehlo", "convolution")
                        || op.is_op("stablehlo", "reduce")
                };
                if !is_target || op.results.is_empty() {
                    continue;
                }

                let orig_res = op.results[0];
                let res_ty = module.values[orig_res].ty.clone();
                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.profile_region".to_string()),
                );
                attrs.insert("profile_id".to_string(), Attribute::Int(created as i64));
                attrs.insert(
                    "profile_label".to_string(),
                    Attribute::String(op.name.to_string()),
                );

                let prof_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "composite"),
                    smallvec![orig_res],
                    vec![(Some(format!("prof_{created}")), res_ty)],
                    attrs,
                    Vec::new(),
                    false,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let prof_res = module.ops[prof_op_id].results[0];

                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == op_id || other_id == prof_op_id {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = prof_res;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id].ops.insert(pos + 1, prof_op_id);
                }
                created += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: 0,
            ops_modified: created,
            ops_created: created,
        })
    }
}
