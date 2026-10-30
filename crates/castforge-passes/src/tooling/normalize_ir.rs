//! Unified Canonical SSA Normalization & Metadata Stripping Pass (`normalize-ir`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Location, Module, ValueId};
use indexmap::IndexMap;

/// Unified Canonical SSA Renamer & Metadata Stripping Pass (`normalize-ir`).
///
/// Consolidates:
/// - `ssa = true`: Renames block arguments (`%arg0, %arg1`) and op results (`%v0, %v1`) in
///   topological order and sorts operation attributes (`NormalizeSsaPass`).
/// - `debuginfo = true`: Strips `loc(...)` source location metadata (`StripDebugInfoPass`).
/// - `attributes = true`: Strips transient `castforge.*` and schedule attributes
///   (`StripAttributesPass`).
#[derive(Debug, Clone)]
pub struct NormalizeIrPass {
    pub ssa: bool,
    pub prefix: String,
    pub debuginfo: bool,
    pub attributes: bool,
}

impl Default for NormalizeIrPass {
    fn default() -> Self {
        Self {
            ssa: true,
            prefix: "v".to_string(),
            debuginfo: false,
            attributes: false,
        }
    }
}

impl Pass for NormalizeIrPass {
    fn name(&self) -> &'static str {
        "normalize-ir"
    }

    fn description(&self) -> &'static str {
        "Unified canonical SSA renaming, attribute sorting, and debug/schedule metadata stripping"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.ssa {
            steps.push(
                NormalizeSsaPass {
                    prefix: self.prefix.clone(),
                }
                .run(module)?,
            );
        }
        if self.debuginfo || self.attributes {
            steps.push(
                StripMetadataPass {
                    debuginfo: self.debuginfo,
                    attributes: self.attributes,
                }
                .run(module)?,
            );
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

/// Canonical SSA Renamer & Attribute Sorter Pass (`normalize-ssa`).
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

/// Unified Metadata & Debug Info Stripping Pass (`strip-metadata`).
pub struct StripMetadataPass {
    pub debuginfo: bool,
    pub attributes: bool,
}

impl Default for StripMetadataPass {
    fn default() -> Self {
        Self {
            debuginfo: true,
            attributes: true,
        }
    }
}

impl Pass for StripMetadataPass {
    fn name(&self) -> &'static str {
        "strip-metadata"
    }

    fn description(&self) -> &'static str {
        "Strips debug locations and transient castforge.* attributes for portable export"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.debuginfo {
            steps.push(StripDebugInfoPass.run(module)?);
        }
        if self.attributes {
            steps.push(StripAttributesPass.run(module)?);
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

/// Strip DebugInfo Pass (`strip-debuginfo`).
pub struct StripDebugInfoPass;

impl Pass for StripDebugInfoPass {
    fn name(&self) -> &'static str {
        "strip-debuginfo"
    }

    fn description(&self) -> &'static str {
        "Strips source location metadata (loc(...)) from all operations in the module"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in &mut module.ops {
            if op.location != Location::Unknown {
                op.location = Location::Unknown;
                modified += 1;
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

/// Transient Dialect & Schedule Attribute Cleaner Pass (`strip-attributes`).
#[derive(Debug, Clone, Default)]
pub struct StripAttributesPass;

impl Pass for StripAttributesPass {
    fn name(&self) -> &'static str {
        "strip-attributes"
    }

    fn description(&self) -> &'static str {
        "Strips transient castforge.* and schedule attributes for pure portable StableHLO export"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        const TRANSIENT_KEYS: &[&str] = &[
            "tile_m",
            "tile_n",
            "tile_k",
            "tile_size",
            "l2_tile",
            "reg_tile_m",
            "reg_tile_n",
            "loop_order",
            "collapse_loops",
            "skew_loops",
            "unroll_factor",
            "prefetch_distance",
            "split_k",
            "double_buffer",
            "vector_width",
            "simd_isa",
            "parallel_m",
            "inplace_operand",
            "sparsity",
            "data_layout",
            "pad_align",
            "padded_mnk",
            "spmd_sharding",
            "num_devices",
            "async_overlap",
            "coalesced_collectives",
            "stream_id",
            "unroll_step",
        ];

        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let before = op.attributes.len();
            op.attributes.retain(|k, _| {
                !k.starts_with("castforge.") && !TRANSIENT_KEYS.contains(&k.as_str())
            });
            if op.attributes.len() < before {
                modified += 1;
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
