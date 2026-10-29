//! Identical Function Deduplicator (`dedup-functions`) and Canonical SSA Renamer
//! (`normalize-ssa`) Passes (`Ir -> Ir`).
//!
//! - [`DedupFunctionsPass`] (`dedup-functions`, inspired by LLVM `-mergefunc`) detects structurally
//!   identical `func.func` definitions (such as repeated layer kernels produced by
//!   `outline-subgraphs`), merges duplicates into a single canonical function, and rewrites all
//!   `func.call` sites.
//! - [`NormalizeSsaPass`] (`normalize-ssa`, inspired by LLVM `-instnamer`) renames block arguments
//!   (`%arg0, %arg1, ...`) and op results (`%v0, %v1, ...`) in deterministic topological order and
//!   sorts operation attribute maps alphabetically.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpId, ValueId};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::fmt::Write as _;

/// Identical Function Deduplicator Pass (`dedup-functions`: `Ir -> Ir`).
#[derive(Debug, Clone, Default)]
pub struct DedupFunctionsPass;

impl Pass for DedupFunctionsPass {
    fn name(&self) -> &'static str {
        "dedup-functions"
    }

    fn description(&self) -> &'static str {
        "Merges structurally identical func.func definitions and rewrites func.call sites"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let funcs = module.functions();
        if funcs.len() < 2 {
            return Ok(PassResult::unchanged(self.name()));
        }

        let mut sig_to_sym: HashMap<String, String> = HashMap::new();
        let mut redirect_map: HashMap<String, String> = HashMap::new();
        let mut duplicate_fn_ids: Vec<OpId> = Vec::new();

        for (idx, &fn_id) in funcs.iter().enumerate() {
            let sym = module.ops[fn_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .unwrap_or("func")
                .to_string();
            let fingerprint = fingerprint_function(module, fn_id);
            if let Some(canon_sym) = sig_to_sym.get(&fingerprint) {
                if idx > 0 {
                    redirect_map.insert(sym, canon_sym.clone());
                    duplicate_fn_ids.push(fn_id);
                }
            } else {
                sig_to_sym.insert(fingerprint, sym);
            }
        }

        if duplicate_fn_ids.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let mut modified_calls = 0usize;
        for (_, op) in module.ops.iter_mut() {
            if !op.is_op("func", "call") {
                continue;
            }
            if let Some(callee) = op
                .attributes
                .get("callee")
                .and_then(|a| a.as_str())
                .map(str::to_string)
                && let Some(canon) = redirect_map.get(&callee)
            {
                op.attributes
                    .insert("callee".to_string(), Attribute::String(canon.clone()));
                modified_calls += 1;
            }
        }

        let mod_body = module.body_block();
        let removed = duplicate_fn_ids.len();
        for fn_id in duplicate_fn_ids {
            module.remove_op(mod_body, fn_id);
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: removed,
            ops_modified: modified_calls,
            ops_created: 0,
        })
    }
}

fn fingerprint_function(module: &Module, fn_id: OpId) -> String {
    let mut fp = String::new();
    let op = &module.ops[fn_id];
    let Some(&reg_id) = op.regions.first() else {
        return fp;
    };
    let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
        return fp;
    };
    let block = &module.blocks[blk_id];

    let mut val_slots: HashMap<ValueId, String> = HashMap::new();
    for (i, &arg_id) in block.arguments.iter().enumerate() {
        let ty = &module.values[arg_id].ty;
        let _ = writeln!(fp, "arg({i}):{ty}");
        val_slots.insert(arg_id, format!("a{i}"));
    }

    for (op_idx, &child_id) in block.ops.iter().enumerate() {
        let child = &module.ops[child_id];
        let opnds: Vec<String> = child
            .operands
            .iter()
            .map(|&v| {
                val_slots
                    .get(&v)
                    .cloned()
                    .unwrap_or_else(|| "?".to_string())
            })
            .collect();
        let res_tys: Vec<String> = child
            .results
            .iter()
            .enumerate()
            .map(|(r_i, &r)| {
                val_slots.insert(r, format!("r{op_idx}_{r_i}"));
                module.values[r].ty.to_string()
            })
            .collect();
        let mut attr_pairs: Vec<(&String, &Attribute)> = child.attributes.iter().collect();
        attr_pairs.sort_by_key(|(k, _)| *k);
        let attrs_str: Vec<String> = attr_pairs
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        let _ = writeln!(
            fp,
            "op({}):{}({})->[{}]{{{}}}",
            op_idx,
            child.name,
            opnds.join(","),
            res_tys.join(","),
            attrs_str.join(",")
        );
    }
    fp
}

/// Canonical SSA Renamer & Attribute Sorter Pass (`normalize-ssa`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct NormalizeSsaPass {
    /// Prefix for normalized operation result SSA names (default: `"v"` -> `%v0, %v1, ...`).
    pub prefix: String,
}

impl Default for NormalizeSsaPass {
    fn default() -> Self {
        Self {
            prefix: "v".to_string(),
        }
    }
}

impl Pass for NormalizeSsaPass {
    fn name(&self) -> &'static str {
        "normalize-ssa"
    }

    fn description(&self) -> &'static str {
        "Renames SSA values in canonical topological order and sorts operation attributes"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let pfx = if self.prefix.trim().is_empty() {
            "v"
        } else {
            self.prefix.trim()
        };
        let mut modified = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let blk_id = module.regions[reg_id].blocks[0];
            let args = module.blocks[blk_id].arguments.clone();
            for (idx, arg_id) in args.into_iter().enumerate() {
                let target_name = format!("arg{idx}");
                if module.values[arg_id].ssa_name.as_deref() != Some(&target_name) {
                    module.values[arg_id].ssa_name = Some(target_name);
                    modified += 1;
                }
            }

            let ops = module.blocks[blk_id].ops.clone();
            let mut val_counter = 0usize;
            for op_id in ops {
                let results: Vec<ValueId> = module.ops[op_id].results.to_vec();
                for res_id in results {
                    let target_name = format!("{pfx}{val_counter}");
                    val_counter += 1;
                    if module.values[res_id].ssa_name.as_deref() != Some(&target_name) {
                        module.values[res_id].ssa_name = Some(target_name);
                        modified += 1;
                    }
                }
                let mut pairs: Vec<(String, Attribute)> =
                    module.ops[op_id].attributes.drain(..).collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                let sorted: IndexMap<String, Attribute> = pairs.into_iter().collect();
                module.ops[op_id].attributes = sorted;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
