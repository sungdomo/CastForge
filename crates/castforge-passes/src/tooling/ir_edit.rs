//! Unified Surgical IR Editing, Input Freezing, Parameterization & Activation Tapping (`ir-edit`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, DType, Location, Module, OpName, TensorType, Type, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashSet;

/// Surgical IR Edit, Input Freezing, Parameterization & Activation Tapping Pass (`ir-edit`).
#[derive(Debug, Clone, Default)]
pub struct IrEditPass {
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
}

impl Pass for IrEditPass {
    fn name(&self) -> &'static str {
        "ir-edit"
    }

    fn description(&self) -> &'static str {
        "Unified IR surgery (rename_op, annotate_op, freeze/unfreeze args, tap outputs)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut created = 0usize;
        let mut removed = 0usize;

        if let Some(rule) = &self.rename_op
            && let Some((from, to)) = rule.split_once(':')
        {
            let from_name = OpName::parse(from.trim());
            let to_name = OpName::parse(to.trim());
            for (_, op) in &mut module.ops {
                if op.name == from_name {
                    op.name = to_name.clone();
                    modified += 1;
                }
            }
        }

        if let Some(rule) = &self.annotate_op {
            let parts: Vec<&str> = rule.splitn(3, ':').collect();
            if parts.len() == 3 {
                let target_op = OpName::parse(parts[0].trim());
                let attr_key = parts[1].trim().to_string();
                let attr_val = Attribute::String(parts[2].trim().to_string());
                for (_, op) in &mut module.ops {
                    if op.name == target_op {
                        op.attributes.insert(attr_key.clone(), attr_val.clone());
                        modified += 1;
                    }
                }
            }
        }

        if self.unfreeze {
            let r = ParameterizeConstantsPass {
                min_elements: self.min_elements.max(1),
            }
            .run(module)?;
            modified += r.ops_modified;
            created += r.ops_created;
            removed += r.ops_removed;
        } else if self.freeze || !self.args.is_empty() {
            let r = FreezeInputsPass {
                args: self.args.clone(),
                value: if self.value == 0.0 { 0.5 } else { self.value },
            }
            .run(module)?;
            modified += r.ops_modified;
            created += r.ops_created;
            removed += r.ops_removed;
        }

        if let Some(ref tap_vals) = self.tap {
            let r = TapIntermediatesPass {
                values: tap_vals.clone(),
            }
            .run(module)?;
            modified += r.ops_modified;
            created += r.ops_created;
            removed += r.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || created > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// Unified Input Freezing & Constant Parameterization Pass (`freeze-inputs`).
pub struct UnifiedFreezeInputsPass {
    pub args: String,
    pub value: f64,
    pub unfreeze: bool,
    pub min_elements: usize,
}

impl Default for UnifiedFreezeInputsPass {
    fn default() -> Self {
        Self {
            args: String::new(),
            value: 0.5,
            unfreeze: false,
            min_elements: 2,
        }
    }
}

impl Pass for UnifiedFreezeInputsPass {
    fn name(&self) -> &'static str {
        "freeze-inputs"
    }

    fn description(&self) -> &'static str {
        "Binds func.func args to constants or hoists constants to args (unfreeze=true)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut res = if self.unfreeze {
            ParameterizeConstantsPass {
                min_elements: self.min_elements,
            }
            .run(module)?
        } else {
            FreezeInputsPass {
                args: self.args.clone(),
                value: self.value,
            }
            .run(module)?
        };
        res.pass_name = self.name().to_string();
        Ok(res)
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
                Attribute::DenseElements {
                    ty: t_ty,
                    raw_literal: format!("{:.4}", self.value),
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
    fn_op_id: castforge_ir::OpId,
    block_id: castforge_ir::BlockId,
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
