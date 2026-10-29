//! Module Linker (`link_modules`) & Function Composer Pass (`compose-funcs`: `Ir -> Ir`).
//!
//! - [`link_modules`] merges all functions and [`TensorStore`](castforge_ir::TensorStore) weights
//!   from a secondary [`Module`] into a primary [`Module`].
//! - [`ComposeFuncsPass`] (`compose-funcs`) synthesizes a composed pipeline function (`@composed`)
//!   that chains two functions (`first -> second`) via `func.call` invocations.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Location, Module, OpId, OpName, Type, ValueId};
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

/// Links all functions and [`TensorStore`](castforge_ir::TensorStore) weights from `other` into
/// `base`, disambiguating colliding symbol names automatically.
pub fn link_modules(base: &mut Module, other: &Module) -> Result<usize> {
    for (_, entry) in other.tensor_store.iter() {
        base.tensor_store.insert(entry.clone());
    }

    let existing_syms: HashSet<String> = base
        .functions()
        .into_iter()
        .filter_map(|id| {
            base.ops[id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .map(str::to_string)
        })
        .collect();

    let mut linked = 0usize;
    let mod_body = base.body_block();

    for src_fn_id in other.functions() {
        let src_fn = &other.ops[src_fn_id];
        let orig_sym = src_fn
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("func");
        let new_sym = if existing_syms.contains(orig_sym) {
            format!("{orig_sym}_linked_{linked}")
        } else {
            orig_sym.to_string()
        };

        let Some(&src_reg) = src_fn.regions.first() else {
            continue;
        };
        let Some(&src_blk) = other.regions[src_reg].blocks.first() else {
            continue;
        };

        let dst_blk = base.create_block(None);
        let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
        for &arg_id in &other.blocks[src_blk].arguments {
            let v = &other.values[arg_id];
            let new_arg = base.add_block_argument(dst_blk, v.ssa_name.clone(), v.ty.clone());
            val_map.insert(arg_id, new_arg);
        }

        for &op_id in &other.blocks[src_blk].ops {
            let op = &other.ops[op_id];
            let mapped_operands: SmallVec<[ValueId; 4]> = op
                .operands
                .iter()
                .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                .collect();
            let res_specs: Vec<(Option<String>, Type)> = op
                .results
                .iter()
                .map(|&r| {
                    let rv = &other.values[r];
                    (rv.ssa_name.clone(), rv.ty.clone())
                })
                .collect();
            let new_op_id = base.add_operation(
                dst_blk,
                op.name.clone(),
                mapped_operands,
                res_specs,
                op.attributes.clone(),
                Vec::new(),
                op.custom_syntax,
                op.location.clone(),
            );
            for (&old_r, &new_r) in op.results.iter().zip(base.ops[new_op_id].results.iter()) {
                val_map.insert(old_r, new_r);
            }
        }

        let dst_reg = base.create_region(vec![dst_blk]);
        let mut attrs = src_fn.attributes.clone();
        attrs.insert("sym_name".to_string(), Attribute::String(new_sym));
        base.add_operation(
            mod_body,
            OpName::new("func", "func"),
            SmallVec::new(),
            Vec::new(),
            attrs,
            vec![dst_reg],
            true,
            src_fn.location.clone(),
        );
        linked += 1;
    }

    Ok(linked)
}

/// Function Composer Pass (`compose-funcs`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct ComposeFuncsPass {
    /// Optional first function symbol name (defaults to the first function in the module).
    pub first: String,
    /// Optional second function symbol name (defaults to the second function in the module).
    pub second: String,
    /// Symbol name for the synthesized composed entry function (default: `"composed_main"`).
    pub entry_name: String,
}

impl Default for ComposeFuncsPass {
    fn default() -> Self {
        Self {
            first: String::new(),
            second: String::new(),
            entry_name: "composed_main".to_string(),
        }
    }
}

impl Pass for ComposeFuncsPass {
    fn name(&self) -> &'static str {
        "compose-funcs"
    }

    fn description(&self) -> &'static str {
        "Synthesizes a composed entry function chaining two module functions (f2(f1(x)))"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let funcs = module.functions();
        if funcs.len() < 2 {
            return Ok(PassResult::unchanged(self.name()));
        }

        let find_fn = |name: &str, fallback_idx: usize| -> Option<OpId> {
            let clean = name.trim().trim_start_matches('@');
            if !clean.is_empty() {
                for &fid in &funcs {
                    if module.ops[fid]
                        .attributes
                        .get("sym_name")
                        .and_then(|a| a.as_str())
                        == Some(clean)
                    {
                        return Some(fid);
                    }
                }
            }
            funcs.get(fallback_idx).copied()
        };

        let (Some(f1_id), Some(f2_id)) = (find_fn(&self.first, 0), find_fn(&self.second, 1)) else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let (sym1, in1, out1) = extract_fn_sig(module, f1_id);
        let (sym2, in2, out2) = extract_fn_sig(module, f2_id);
        if out1.is_empty() || in2.is_empty() || out2.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let comp_block = module.create_block(None);
        let mut f1_call_args: SmallVec<[ValueId; 4]> = SmallVec::new();
        let mut comp_inputs = Vec::new();
        for (i, ty) in in1.iter().enumerate() {
            let arg = module.add_block_argument(comp_block, Some(format!("in_{i}")), ty.clone());
            f1_call_args.push(arg);
            comp_inputs.push(ty.clone());
        }

        let mut f2_extra_args: SmallVec<[ValueId; 4]> = SmallVec::new();
        for (j, ty) in in2.iter().enumerate().skip(1) {
            let arg = module.add_block_argument(comp_block, Some(format!("in2_{j}")), ty.clone());
            f2_extra_args.push(arg);
            comp_inputs.push(ty.clone());
        }

        let mut call1_attrs = IndexMap::new();
        call1_attrs.insert("callee".to_string(), Attribute::String(sym1));
        let call1_id = module.add_operation(
            comp_block,
            OpName::new("func", "call"),
            f1_call_args,
            vec![(Some("stage1_out".to_string()), out1[0].clone())],
            call1_attrs,
            Vec::new(),
            true,
            Location::Unknown,
        );
        let stage1_val = module.ops[call1_id].results[0];

        let mut f2_call_args: SmallVec<[ValueId; 4]> = smallvec![stage1_val];
        f2_call_args.extend(f2_extra_args);
        let mut call2_attrs = IndexMap::new();
        call2_attrs.insert("callee".to_string(), Attribute::String(sym2));
        let call2_id = module.add_operation(
            comp_block,
            OpName::new("func", "call"),
            f2_call_args,
            vec![(Some("stage2_out".to_string()), out2[0].clone())],
            call2_attrs,
            Vec::new(),
            true,
            Location::Unknown,
        );
        let stage2_val = module.ops[call2_id].results[0];

        module.add_operation(
            comp_block,
            OpName::new("func", "return"),
            smallvec![stage2_val],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            true,
            Location::Unknown,
        );

        let comp_reg = module.create_region(vec![comp_block]);
        let mut fn_attrs = IndexMap::new();
        let entry_sym = if self.entry_name.trim().is_empty() {
            "composed_main".to_string()
        } else {
            self.entry_name.trim().to_string()
        };
        fn_attrs.insert("sym_name".to_string(), Attribute::String(entry_sym));
        fn_attrs.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: comp_inputs,
                results: vec![out2[0].clone()],
            }),
        );
        let mod_body = module.body_block();
        module.add_operation(
            mod_body,
            OpName::new("func", "func"),
            SmallVec::new(),
            Vec::new(),
            fn_attrs,
            vec![comp_reg],
            true,
            Location::Unknown,
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: true,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 4,
        })
    }
}

fn extract_fn_sig(module: &Module, fn_id: OpId) -> (String, Vec<Type>, Vec<Type>) {
    let op = &module.ops[fn_id];
    let sym = op
        .attributes
        .get("sym_name")
        .and_then(|a| a.as_str())
        .unwrap_or("main")
        .to_string();
    if let Some(Attribute::Type(Type::Function { inputs, results })) =
        op.attributes.get("function_type")
    {
        return (sym, inputs.clone(), results.clone());
    }
    (sym, Vec::new(), Vec::new())
}
