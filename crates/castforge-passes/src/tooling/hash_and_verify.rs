//! Structural Graph Hasher (`hash-ir`) and In-Pipeline Equivalence Verifier (`verify-equiv`)
//! Passes (`Ir -> Ir`).
//!
//! - [`HashIrPass`] (`hash-ir`) computes a deterministic 64-bit FNV-1a structural and weight
//!   fingerprint over the module (invariant to SSA variable naming) and stamps
//!   `"castforge.structural_hash"` onto the entry function.
//! - [`VerifyEquivPass`] (`verify-equiv`) validates structural IR invariants, function signature
//!   consistency, and tensor shape compatibility at any point in a pass chain.

use crate::{Pass, PassResult};
use anyhow::{Result, bail};
use castforge_ir::{Attribute, Module, ValueId};
use std::collections::HashMap;

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
    /// Numerical tolerance metadata for downstream verification (default `1e-4`).
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
            if let Some(Attribute::Type(castforge_ir::Type::Function { inputs, results })) =
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
