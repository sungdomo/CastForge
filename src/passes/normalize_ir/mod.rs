//! Unified IR Normalization, Subgraph Outlining/Inlining/Extraction, Input Freezing,
//! Parameterization & Activation Tapping (`normalize-ir`).

pub mod inliner;
pub mod outline;

pub use inliner::*;
pub use outline::*;

use crate::ir::{
    Attribute, DType, Location, Module, OpName, TensorEntry, TensorType, Type, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashSet;

/// Surgical IR Normalization, Subgraph Outlining/Inlining/Extraction, Input Freezing,
/// Parameterization & Activation Tapping Pass (`normalize-ir`).
#[derive(Debug, Clone, Default)]
pub struct NormalizeIrPass {
    /// Subgraph mode (`outline`, `all`, `gemm`, `composites`, `inline`, `extract`, `reduce`,
    /// `dedup`, `compose`, `unroll`).
    pub mode: Option<String>,
    pub dedup: bool,
    pub first: String,
    pub second: String,
    pub entry_name: String,
    pub repeat: usize,
    pub target: String,
    pub stop_at: String,
    pub max_ops: Option<usize>,
    pub keep_op: String,
    /// Format: `"old_dialect.old_op:new_dialect.new_op"`
    pub rename_op: Option<String>,
    /// Format: `"dialect.op:attr_key:attr_value"`
    pub annotate_op: Option<String>,
    /// Freeze function arguments to constants (`freeze=true` or `args=%w`).
    pub freeze: bool,
    /// Unfreeze / parameterize constants to function arguments (`unfreeze=true`).
    pub unfreeze: bool,
    /// Comma/semicolon-separated argument names to freeze.
    pub args: String,
    /// Default scalar value when freezing unbacked arguments (default `0.5`).
    pub value: f64,
    /// Minimum element count when parameterizing constants (`unfreeze=true`).
    pub min_elements: usize,
    /// Tap intermediate SSA values into `func.return` (`Some("")` = all compute values).
    pub tap: Option<String>,
    /// Normalize SSA names in topological order and sort operation attributes (`ssa=true`).
    pub ssa: bool,
    /// Prefix for normalized SSA names (`prefix=v`).
    pub prefix: String,
    /// Strip debug location metadata (`debuginfo=true`).
    pub debuginfo: bool,
    /// Strip transient `castforge.*` and schedule attributes (`attributes=true`).
    pub attributes: bool,
}

impl Pass for NormalizeIrPass {
    fn name(&self) -> &'static str {
        "normalize-ir"
    }

    fn description(&self) -> &'static str {
        "Unified IR normalization, subgraph outlining/inlining/extraction, freezing & tapping"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &[
            "outline-subgraphs",
            "parameterize-constants",
            "freeze-inputs",
            "tap-intermediates",
            "strip-debuginfo",
            "strip-attributes",
            "normalize-ssa",
        ]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();

        if self.mode.is_some()
            || self.dedup
            || self.repeat > 1
            || !self.target.is_empty()
            || !self.stop_at.is_empty()
            || self.max_ops.is_some()
            || !self.keep_op.is_empty()
        {
            let m = match self.mode.as_deref() {
                Some("outline") | None => "all".to_string(),
                Some(other) => other.to_string(),
            };
            steps.push(
                OutlineSubgraphsPass {
                    mode: m,
                    dedup: self.dedup,
                    first: self.first.clone(),
                    second: self.second.clone(),
                    entry_name: if self.entry_name.is_empty() {
                        "composed_main".to_string()
                    } else {
                        self.entry_name.clone()
                    },
                    repeat: self.repeat.max(1),
                    target: self.target.clone(),
                    stop_at: self.stop_at.clone(),
                    max_ops: self.max_ops,
                    keep_op: self.keep_op.clone(),
                }
                .run(module)?,
            );
        }

        if self.unfreeze {
            steps.push(
                ParameterizeConstantsPass {
                    min_elements: self.min_elements.max(1),
                }
                .run(module)?,
            );
        } else if self.freeze || !self.args.is_empty() {
            steps.push(
                FreezeInputsPass {
                    args: self.args.clone(),
                    value: if self.value == 0.0 { 0.5 } else { self.value },
                }
                .run(module)?,
            );
        }

        if let Some(ref tap_vals) = self.tap {
            steps.push(
                TapIntermediatesPass {
                    values: tap_vals.clone(),
                }
                .run(module)?,
            );
        }

        if self.rename_op.is_some() || self.annotate_op.is_some() {
            steps.push(self.apply_op_rename_and_annotate(module));
        }
        if self.debuginfo {
            steps.push(StripDebugInfoPass.run(module)?);
        }
        if self.attributes {
            steps.push(StripAttributesPass.run(module)?);
        }
        if self.ssa {
            steps.push(
                NormalizeSsaPass {
                    prefix: self.prefix.clone(),
                }
                .run(module)?,
            );
        }

        Ok(PassResult::aggregate(self.name(), &steps))
    }
}

impl NormalizeIrPass {
    fn apply_op_rename_and_annotate(&self, module: &mut Module) -> PassResult {
        let mut modified = 0usize;
        let rename_pair = self
            .rename_op
            .as_ref()
            .and_then(|r| r.split_once(':'))
            .map(|(f, t)| (OpName::parse(f.trim()), OpName::parse(t.trim())));
        let annotate_triple = self.annotate_op.as_ref().and_then(|r| {
            let parts: Vec<&str> = r.splitn(3, ':').collect();
            (parts.len() == 3).then(|| {
                (
                    OpName::parse(parts[0].trim()),
                    parts[1].trim().to_string(),
                    Attribute::String(parts[2].trim().to_string()),
                )
            })
        });

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let blk_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[blk_id].ops.clone();
            for op_id in ops {
                if let Some((ref from_name, ref to_name)) = rename_pair
                    && module.ops[op_id].name == *from_name
                {
                    module.ops[op_id].name = to_name.clone();
                    modified += 1;
                }
                if let Some((ref target_op, ref attr_key, ref attr_val)) = annotate_triple
                    && module.ops[op_id].name == *target_op
                {
                    module.ops[op_id]
                        .attributes
                        .insert(attr_key.clone(), attr_val.clone());
                    modified += 1;
                }
            }
        }
        PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        }
    }
}

/// Input Freezer Pass (`freeze-inputs`).
#[derive(Debug, Clone)]
pub struct FreezeInputsPass {
    pub args: String,
    pub value: f64,
}

impl Default for FreezeInputsPass {
    fn default() -> Self {
        Self {
            args: String::new(),
            value: 0.5,
        }
    }
}

impl Pass for FreezeInputsPass {
    fn name(&self) -> &'static str {
        "freeze-inputs"
    }

    fn description(&self) -> &'static str {
        "Binds func.func input arguments to in-body stablehlo.constant operations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let orig_args = module.blocks[block_id].arguments.clone();
        if orig_args.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let target_names: HashSet<String> = self
            .args
            .split([',', ';'])
            .map(|s| s.trim().trim_start_matches('%').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut kept_args: Vec<ValueId> = Vec::new();
        let mut frozen_args: Vec<ValueId> = Vec::new();

        for (idx, &arg_id) in orig_args.iter().enumerate() {
            let ssa = module.values[arg_id].ssa_name.as_deref().unwrap_or("");
            let should_freeze = if target_names.is_empty() {
                idx >= 1
            } else {
                target_names.contains(ssa)
            };
            if should_freeze {
                frozen_args.push(arg_id);
            } else {
                kept_args.push(arg_id);
            }
        }

        if frozen_args.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let mut new_const_ops = Vec::with_capacity(frozen_args.len());
        for &arg_id in &frozen_args {
            let ty = module.values[arg_id].ty.clone();
            let ssa = module.values[arg_id]
                .ssa_name
                .clone()
                .unwrap_or_else(|| "frozen_const".to_string());
            let t_ty = ty.as_tensor().cloned().unwrap_or(TensorType {
                shape: Vec::new(),
                dtype: DType::F32,
                encoding: None,
            });
            let val_attr = if module.tensor_store.get(&ssa).is_some() {
                Attribute::TensorRef {
                    key: ssa.clone(),
                    ty: t_ty,
                }
            } else {
                let raw_literal = if t_ty.dtype.is_float() {
                    format!("{:.4}", self.value)
                } else if t_ty.dtype == DType::I1 {
                    if self.value != 0.0 {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                } else {
                    format!("{}", self.value as i64)
                };
                Attribute::DenseElements {
                    ty: t_ty,
                    raw_literal,
                }
            };
            let mut attrs = IndexMap::new();
            attrs.insert("value".to_string(), val_attr);
            let const_op_id = module.add_operation(
                block_id,
                OpName::new("stablehlo", "constant"),
                SmallVec::new(),
                vec![(Some(ssa), ty)],
                attrs,
                Vec::new(),
                true,
                Location::Unknown,
            );
            module.blocks[block_id].ops.pop();
            let const_res = module.ops[const_op_id].results[0];
            module.replace_all_uses(arg_id, const_res);
            module.values.remove(arg_id);
            new_const_ops.push(const_op_id);
        }

        module.blocks[block_id].ops.splice(0..0, new_const_ops);
        for (new_idx, &arg_id) in kept_args.iter().enumerate() {
            module.values[arg_id].def = ValueDef::BlockArgument {
                block: block_id,
                index: new_idx as u32,
            };
        }
        module.blocks[block_id].arguments = kept_args;

        update_func_signature(module, fn_op_id, block_id);

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: frozen_args.len(),
        })
    }
}

/// Constant Parameterizer Pass (`parameterize-constants`).
#[derive(Debug, Clone)]
pub struct ParameterizeConstantsPass {
    pub min_elements: usize,
}

impl Default for ParameterizeConstantsPass {
    fn default() -> Self {
        Self { min_elements: 2 }
    }
}

impl Pass for ParameterizeConstantsPass {
    fn name(&self) -> &'static str {
        "parameterize-constants"
    }

    fn description(&self) -> &'static str {
        "Hoists in-body stablehlo.constant weight tensors into explicit func.func arguments"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let min_elems = self.min_elements.max(1) as i64;
        let mut hoisted = 0usize;

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
                continue;
            }
            let res_val = op.results[0];
            let ty = module.values[res_val].ty.clone();
            let elems = ty.as_tensor().and_then(|t| t.num_elements()).unwrap_or(1);
            if elems < min_elems {
                continue;
            }
            let ssa = module.values[res_val]
                .ssa_name
                .clone()
                .unwrap_or_else(|| format!("param_{hoisted}"));
            if let Some(t_ty) = ty.as_tensor().cloned()
                && let Some(val_attr) = op.attributes.get("value").cloned()
                && let Some(vals) = module
                    .tensor_store
                    .resolve_f32_attr(&val_attr, elems.max(1) as usize)
            {
                if let Attribute::TensorRef { ref key, .. } = val_attr
                    && key != &ssa
                {
                    module.tensor_store.rename_key(key, ssa.clone());
                } else if module.tensor_store.get(&ssa).is_none() {
                    module
                        .tensor_store
                        .insert(TensorEntry::from_f32(ssa.clone(), t_ty, vals));
                }
            }
            let new_arg = module.add_block_argument(block_id, Some(ssa), ty);
            module.replace_all_uses(res_val, new_arg);
            module.remove_op(block_id, op_id);
            hoisted += 1;
        }

        if hoisted == 0 {
            return Ok(PassResult::unchanged(self.name()));
        }

        update_func_signature(module, fn_op_id, block_id);

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: hoisted,
            ops_modified: 1,
            ops_created: hoisted,
        })
    }
}

/// Intermediate Activation Tapper Pass (`tap-intermediates`).
#[derive(Debug, Clone, Default)]
pub struct TapIntermediatesPass {
    /// Semicolon/comma-separated SSA value names to append to `func.return` (empty = all compute).
    pub values: String,
}

impl Pass for TapIntermediatesPass {
    fn name(&self) -> &'static str {
        "tap-intermediates"
    }

    fn description(&self) -> &'static str {
        "Appends intermediate SSA activations to func.return for multi-output layer inspection"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let Some(&ret_op_id) = ops
            .iter()
            .find(|&&id| module.ops[id].is_op("func", "return"))
        else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let filter: HashSet<String> = self
            .values
            .split([',', ';'])
            .map(|s| s.trim().trim_start_matches('%').to_string())
            .filter(|s| !s.is_empty() && s != "all" && s != "true")
            .collect();

        let mut tapped: Vec<ValueId> = Vec::new();
        for &op_id in &ops {
            if op_id == ret_op_id || module.ops[op_id].is_op("stablehlo", "constant") {
                continue;
            }
            for &res in &module.ops[op_id].results {
                let ssa = module.values[res].ssa_name.as_deref().unwrap_or("");
                let matched = if filter.is_empty() {
                    true
                } else {
                    filter.contains(ssa)
                };
                if matched && !module.ops[ret_op_id].operands.contains(&res) {
                    tapped.push(res);
                }
            }
        }

        if tapped.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        for &vid in &tapped {
            module.ops[ret_op_id].operands.push(vid);
        }

        update_func_signature(module, fn_op_id, block_id);

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 1,
            ops_created: tapped.len(),
        })
    }
}

fn update_func_signature(
    module: &mut Module,
    fn_op_id: crate::ir::OpId,
    block_id: crate::ir::BlockId,
) {
    let inputs: Vec<Type> = module.blocks[block_id]
        .arguments
        .iter()
        .map(|&a| module.values[a].ty.clone())
        .collect();
    let results: Vec<Type> = module.blocks[block_id]
        .ops
        .iter()
        .find(|&&id| module.ops[id].is_op("func", "return"))
        .map(|&ret_id| {
            module.ops[ret_id]
                .operands
                .iter()
                .map(|&v| module.values[v].ty.clone())
                .collect()
        })
        .unwrap_or_default();
    module.ops[fn_op_id].attributes.insert(
        "function_type".to_string(),
        Attribute::Type(Type::Function { inputs, results }),
    );
}

/// Canonical SSA Renamer & Attribute Sorter Pass (`normalize-ssa`).
#[derive(Debug, Clone)]
pub struct NormalizeSsaPass {
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
            let regions = module.ops[fn_id].regions.clone();
            let mut val_counter = 0usize;
            for reg_id in regions {
                normalize_region_ssa(module, reg_id, pfx, &mut val_counter, &mut modified);
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

fn normalize_region_ssa(
    module: &mut Module,
    reg_id: crate::ir::RegionId,
    pfx: &str,
    val_counter: &mut usize,
    modified: &mut usize,
) {
    let Some(reg) = module.regions.get(reg_id) else {
        return;
    };
    let blocks = reg.blocks.clone();
    for blk_id in blocks {
        let args = module.blocks[blk_id].arguments.clone();
        for (idx, arg_id) in args.into_iter().enumerate() {
            let target_name = format!("arg{idx}");
            if module.values[arg_id].ssa_name.as_deref() != Some(&target_name) {
                module.values[arg_id].ssa_name = Some(target_name);
                *modified += 1;
            }
        }

        let ops = module.blocks[blk_id].ops.clone();
        for op_id in ops {
            let is_const = module.ops[op_id].is_op("stablehlo", "constant");
            let is_tensor_ref = matches!(
                module.ops[op_id].attributes.get("value"),
                Some(Attribute::TensorRef { .. })
            );
            let results: Vec<ValueId> = module.ops[op_id].results.to_vec();
            for res_id in results {
                let target_name = format!("{pfx}{}", *val_counter);
                *val_counter += 1;
                if module.values[res_id].ssa_name.as_deref() != Some(&target_name) {
                    if is_const
                        && !is_tensor_ref
                        && let Some(old_name) = module.values[res_id].ssa_name.clone()
                    {
                        module
                            .tensor_store
                            .rename_key(&old_name, target_name.clone());
                    }
                    module.values[res_id].ssa_name = Some(target_name);
                    *modified += 1;
                }
            }
            let nested_regions = module.ops[op_id].regions.clone();
            for nested_reg in nested_regions {
                normalize_region_ssa(module, nested_reg, pfx, val_counter, modified);
            }
            let mut pairs: Vec<(String, Attribute)> =
                module.ops[op_id].attributes.drain(..).collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let sorted: IndexMap<String, Attribute> = pairs.into_iter().collect();
            module.ops[op_id].attributes = sorted;
        }
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
        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let before = op.attributes.len();
            op.attributes.retain(|k, _| !is_transient_attribute(k));
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

fn is_transient_attribute(k: &str) -> bool {
    const TRANSIENT_KEYS: &[&str] = &[
        "tile_m",
        "tile_n",
        "tile_k",
        "tile_size",
        "l2_tile",
        "reg_tile_m",
        "reg_tile_n",
        "loop_tile_m",
        "loop_tile_n",
        "loop_tile_k",
        "loop_l2_tile",
        "loop_reg_m",
        "loop_reg_n",
        "loop_order",
        "collapse_loops",
        "loop_collapse",
        "skew_loops",
        "loop_skew",
        "unroll_factor",
        "loop_unroll",
        "prefetch_distance",
        "loop_prefetch",
        "split_k",
        "loop_split_k",
        "double_buffer",
        "loop_double_buffer",
        "vector_width",
        "simd_vectorize",
        "simd_width",
        "simd_isa",
        "parallel_m",
        "loop_parallel",
        "inplace_operand",
        "rematerialized",
        "sparsity",
        "data_layout",
        "pad_align",
        "pad_deltas",
        "padded_mnk",
        "packed_rhs",
        "quant_mode",
        "lhs_scale",
        "rhs_scale",
        "calibration_samples",
        "compute_precision",
        "amp_cast",
        "autodiff_role",
        "spmd_sharding",
        "num_devices",
        "async_overlap",
        "coalesced_collectives",
        "stream_id",
        "stream_synchronized",
        "unroll_step",
    ];
    k.starts_with("castforge.") || TRANSIENT_KEYS.contains(&k)
}
