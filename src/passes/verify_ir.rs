//! Unified IR Verification, Numerical Sanitizer, Structural Hashing, Bisection & Roofline
//! Profiling (`verify-ir`).

use crate::ir::{Attribute, Dim, Module, OpName, Operation, ValueId};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::verify_stablehlo_shapes;
use anyhow::{Result, anyhow, bail};
use indexmap::IndexMap;
use smallvec::smallvec;
use std::collections::HashMap;

/// Shape Verification Pass (`verify-shapes`).
pub struct VerifyShapesPass;

impl Pass for VerifyShapesPass {
    fn name(&self) -> &'static str {
        "verify-shapes"
    }

    fn description(&self) -> &'static str {
        "Verifies StableHLO tensor shapes and rank consistency across all ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        for (op_id, _) in &module.ops {
            if let Err(msg) = verify_stablehlo_shapes(module, op_id) {
                return Err(anyhow!(msg));
            }
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

/// Unified IR Verification, Equivalence, Sanitizer, Structural Hashing, Bisection & Roofline
/// Profiling Pass (`verify-ir`).
pub struct UnifiedVerifyIrPass {
    pub shapes: bool,
    pub equiv: bool,
    pub hash: bool,
    pub bisect_limit: Option<usize>,
    pub tolerance: f64,
    pub sanitize: bool,
    pub clamp_max: f64,
    pub eps: f64,
    pub profile: bool,
    pub instrument: bool,
    pub filter: String,
    pub ridge_flops_per_byte: f64,
}

impl Default for UnifiedVerifyIrPass {
    fn default() -> Self {
        Self {
            shapes: true,
            equiv: false,
            hash: false,
            bisect_limit: None,
            tolerance: 1e-4,
            sanitize: false,
            clamp_max: 65504.0,
            eps: 1e-6,
            profile: false,
            instrument: false,
            filter: "gemm".to_string(),
            ridge_flops_per_byte: 10.0,
        }
    }
}

impl Pass for UnifiedVerifyIrPass {
    fn name(&self) -> &'static str {
        "verify-ir"
    }

    fn description(&self) -> &'static str {
        "Unified shape verification, structural hashing, equivalence, sanitizer, bisection & stats"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.sanitize {
            steps.push(
                SanitizeNumericsPass {
                    clamp_max: self.clamp_max,
                    eps: self.eps,
                }
                .run(module)?,
            );
        }
        if self.instrument {
            steps.push(
                InstrumentProfilingPass {
                    filter: self.filter.clone(),
                }
                .run(module)?,
            );
        }
        if self.profile {
            steps.push(
                PrintOpStatsPass {
                    ridge_flops_per_byte: self.ridge_flops_per_byte,
                }
                .run(module)?,
            );
        }
        if self.shapes {
            steps.push(VerifyShapesPass.run(module)?);
        }
        if self.hash {
            steps.push(HashIrPass.run(module)?);
        }
        if self.equiv {
            steps.push(
                VerifyEquivPass {
                    tolerance: self.tolerance,
                }
                .run(module)?,
            );
        }
        if let Some(limit) = self.bisect_limit {
            steps.push(BisectPassesPass { limit }.run(module)?);
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Computes a deterministic 64-bit FNV-1a structural hash of `module` that is invariant to
/// arbitrary SSA variable names.
pub fn compute_structural_hash(module: &Module) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    for fn_id in module.functions() {
        let op = &module.ops[fn_id];
        let Some(&reg_id) = op.regions.first() else {
            continue;
        };
        let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let block = &module.blocks[blk_id];

        let mut val_slots: HashMap<ValueId, u32> = HashMap::new();
        let mut next_slot = 0u32;
        for &arg_id in &block.arguments {
            mix(module.values[arg_id].ty.to_string().as_bytes());
            val_slots.insert(arg_id, next_slot);
            next_slot += 1;
        }

        for &child_id in &block.ops {
            let child = &module.ops[child_id];
            mix(child.name.to_string().as_bytes());
            for &opnd in &child.operands {
                let slot = val_slots.get(&opnd).copied().unwrap_or(u32::MAX);
                mix(&slot.to_le_bytes());
            }
            for &res in &child.results {
                mix(module.values[res].ty.to_string().as_bytes());
                val_slots.insert(res, next_slot);
                next_slot += 1;
            }
            let mut attr_keys: Vec<&String> = child.attributes.keys().collect();
            attr_keys.sort();
            for k in attr_keys {
                if k.starts_with("castforge.") {
                    continue;
                }
                mix(k.as_bytes());
                mix(child.attributes[k].to_string().as_bytes());
            }
        }
    }

    for (name, entry) in module.tensor_store.iter() {
        mix(name.as_bytes());
        if let Some(vals) = entry.as_f32_slice() {
            for &v in vals {
                mix(&v.to_bits().to_le_bytes());
            }
        }
    }

    hash
}

/// Structural Graph Hasher Pass (`hash-ir`: `Ir -> Ir`).
#[derive(Debug, Clone, Default)]
pub struct HashIrPass;

impl Pass for HashIrPass {
    fn name(&self) -> &'static str {
        "hash-ir"
    }

    fn description(&self) -> &'static str {
        "Computes a 64-bit SSA-invariant structural hash and stamps it onto the entry function"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let digest = format!("{:016x}", compute_structural_hash(module));
        module.ops[fn_id].attributes.insert(
            "castforge.structural_hash".to_string(),
            Attribute::String(digest.clone()),
        );
        if let Some(&reg_id) = module.ops[fn_id].regions.first()
            && let Some(&blk_id) = module.regions[reg_id].blocks.first()
            && let Some(&op_id) = module.blocks[blk_id]
                .ops
                .iter()
                .rev()
                .find(|&&id| !module.ops[id].is_op("func", "return"))
        {
            module.ops[op_id].attributes.insert(
                "castforge.structural_hash".to_string(),
                Attribute::String(digest),
            );
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: 0,
        })
    }
}

/// In-Pipeline Structural & Signature Equivalence Verifier Pass (`verify-equiv`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct VerifyEquivPass {
    pub tolerance: f64,
}

impl Default for VerifyEquivPass {
    fn default() -> Self {
        Self { tolerance: 1e-4 }
    }
}

impl Pass for VerifyEquivPass {
    fn name(&self) -> &'static str {
        "verify-equiv"
    }

    fn description(&self) -> &'static str {
        "Verifies structural SSA integrity and function I/O signature consistency in a pipeline"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        module.verify()?;
        for fn_id in module.functions() {
            let fn_op = &module.ops[fn_id];
            let Some(&reg_id) = fn_op.regions.first() else {
                bail!("Function is missing body region");
            };
            let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
                bail!("Function is missing entry block");
            };
            if let Some(Attribute::Type(crate::ir::Type::Function { inputs, results })) =
                fn_op.attributes.get("function_type")
            {
                let blk_args = &module.blocks[blk_id].arguments;
                if inputs.len() != blk_args.len() {
                    bail!(
                        "Function input arity mismatch: signature has {} vs {} block args",
                        inputs.len(),
                        blk_args.len()
                    );
                }
                if let Some(&ret_id) = module.blocks[blk_id]
                    .ops
                    .iter()
                    .find(|&&id| module.ops[id].is_op("func", "return"))
                {
                    let ret_opnds = &module.ops[ret_id].operands;
                    if results.len() != ret_opnds.len() {
                        bail!(
                            "Function result arity mismatch: signature has {} vs {} return vals",
                            results.len(),
                            ret_opnds.len()
                        );
                    }
                }
            }
            module.ops[fn_id].attributes.insert(
                "castforge.verified_tol".to_string(),
                Attribute::Float(self.tolerance),
            );
            if let Some(&op_id) = module.blocks[blk_id]
                .ops
                .iter()
                .rev()
                .find(|&&id| !module.ops[id].is_op("func", "return"))
            {
                module.ops[op_id].attributes.insert(
                    "castforge.verified_tol".to_string(),
                    Attribute::Float(self.tolerance),
                );
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: 0,
        })
    }
}

/// Pass-Pipeline Bisector Checkpoint Pass (`bisect-passes`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct BisectPassesPass {
    /// Checkpoint step limit or label (default: `1`).
    pub limit: usize,
}

impl Default for BisectPassesPass {
    fn default() -> Self {
        Self { limit: 1 }
    }
}

impl Pass for BisectPassesPass {
    fn name(&self) -> &'static str {
        "bisect-passes"
    }

    fn description(&self) -> &'static str {
        "Verifies IR invariants and stamps a numbered bisection checkpoint on the entry function"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        module.verify()?;
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let prev = module.ops[fn_op_id]
            .attributes
            .get("castforge.bisect_step")
            .and_then(|a| match a {
                Attribute::Int(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(0);
        let next = (prev + 1).min(self.limit.max(1));
        module.ops[fn_op_id].attributes.insert(
            "castforge.bisect_step".to_string(),
            Attribute::Int(next as i64),
        );
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: 0,
        })
    }
}

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
