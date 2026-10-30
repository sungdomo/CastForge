//! Unified IR Verification, Numerical Sanitizer, Structural Hashing, Bisection & Roofline
//! Profiling (`verify-ir`).

pub mod profile;

pub use profile::*;

use crate::ir::{Attribute, Module, ValueId};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::verify_stablehlo_shapes;
use anyhow::{Result, anyhow, bail};
use indexmap::IndexMap;
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
        let mut created = 0usize;
        let mut modified = 0usize;

        if self.sanitize {
            let res = SanitizeNumericsPass {
                clamp_max: self.clamp_max,
                eps: self.eps,
            }
            .run(module)?;
            created += res.ops_created;
            modified += res.ops_modified;
        }
        if self.instrument {
            let res = InstrumentProfilingPass {
                filter: self.filter.clone(),
            }
            .run(module)?;
            created += res.ops_created;
            modified += res.ops_modified;
        }

        if self.profile || self.shapes || self.hash || self.equiv || self.bisect_limit.is_some() {
            modified += verify_ir_single_traversal(self, module)?;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn verify_ir_single_traversal(pass: &UnifiedVerifyIrPass, module: &mut Module) -> Result<usize> {
    if pass.equiv || pass.bisect_limit.is_some() {
        module.verify()?;
    }

    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };

    let mut counts: IndexMap<String, usize> = IndexMap::new();
    let mut total_params = 0i64;
    let mut total_flops = 0i64;
    let mut total_bytes = 0i64;

    let fn_ids = module.functions();
    for &fn_id in &fn_ids {
        let fn_op = &module.ops[fn_id];
        if pass.profile {
            *counts.entry(fn_op.name.to_string()).or_insert(0) += 1;
        }
        if pass.shapes
            && let Err(msg) = verify_stablehlo_shapes(module, fn_id)
        {
            return Err(anyhow!(msg));
        }

        let Some(&reg_id) = fn_op.regions.first() else {
            if pass.equiv {
                bail!("Function is missing body region");
            }
            continue;
        };
        let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
            if pass.equiv {
                bail!("Function is missing entry block");
            }
            continue;
        };

        if pass.equiv
            && let Some(Attribute::Type(crate::ir::Type::Function { inputs, results })) =
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

        let block = &module.blocks[blk_id];
        let mut val_slots: HashMap<ValueId, u32> = HashMap::new();
        let mut next_slot = 0u32;
        if pass.hash {
            for &arg_id in &block.arguments {
                mix(module.values[arg_id].ty.to_string().as_bytes());
                val_slots.insert(arg_id, next_slot);
                next_slot += 1;
            }
        }

        for &child_id in &block.ops {
            if pass.shapes
                && let Err(msg) = verify_stablehlo_shapes(module, child_id)
            {
                return Err(anyhow!(msg));
            }
            let child = &module.ops[child_id];
            if pass.profile {
                *counts.entry(child.name.to_string()).or_insert(0) += 1;
                if !child.is_op("func", "func") && !child.is_op("func", "return") {
                    if child.is_op("stablehlo", "constant") {
                        if let Some(&res) = child.results.first()
                            && let Some(t) = module.values[res].ty.as_tensor()
                        {
                            total_params += t.num_elements().unwrap_or(0);
                        }
                    } else {
                        let (fl, by) = op_flops_and_bytes(module, child);
                        total_flops += fl;
                        total_bytes += by;
                    }
                }
            }
            if pass.hash {
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
    }

    if pass.hash {
        for (name, entry) in module.tensor_store.iter() {
            mix(name.as_bytes());
            if let Some(vals) = entry.as_f32_slice() {
                for &v in vals {
                    mix(&v.to_bits().to_le_bytes());
                }
            }
        }
    }

    if pass.profile {
        let ai = if total_bytes > 0 {
            (total_flops as f64) / (total_bytes as f64)
        } else {
            0.0
        }
        .max(0.0);
        let bottleneck = if ai >= pass.ridge_flops_per_byte {
            "Compute-Bound"
        } else {
            "Memory-Bandwidth-Bound"
        };
        eprintln!(
            "[castforge:stats] Total Ops: {} | Params: {} | FLOPs: {} | \
             Bytes: {} B | AI: {:.2} FLOP/B ({})",
            module.ops.len(),
            total_params,
            total_flops,
            total_bytes,
            ai,
            bottleneck
        );
        for (op_name, count) in &counts {
            eprintln!("  - {:<28} : {}", op_name, count);
        }
    }

    let mut modified = 0usize;
    if pass.hash
        && let Some(&fn_id) = fn_ids.first()
    {
        let digest = format!("{hash:016x}");
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
        modified += 1;
    }

    if pass.equiv {
        for &fn_id in &fn_ids {
            module.ops[fn_id].attributes.insert(
                "castforge.verified_tol".to_string(),
                Attribute::Float(pass.tolerance),
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
                    "castforge.verified_tol".to_string(),
                    Attribute::Float(pass.tolerance),
                );
            }
        }
        modified += 1;
    }

    if let Some(limit) = pass.bisect_limit
        && let Some(&fn_op_id) = fn_ids.first()
    {
        let prev = module.ops[fn_op_id]
            .attributes
            .get("castforge.bisect_step")
            .and_then(|a| match a {
                Attribute::Int(v) => Some(*v as usize),
                _ => None,
            })
            .unwrap_or(0);
        let next = (prev + 1).min(limit.max(1));
        module.ops[fn_op_id].attributes.insert(
            "castforge.bisect_step".to_string(),
            Attribute::Int(next as i64),
        );
        modified += 1;
    }

    Ok(modified)
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
