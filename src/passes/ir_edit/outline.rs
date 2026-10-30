//! Submodule `outline` of `ir_edit`.

use super::*;
use crate::ir::{Attribute, Location, Module, OpId, OpName, Type, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

/// Unified Subgraph Outlining, Inlining, Extraction, Deduplication, Composition & Unrolling
/// (`outline-subgraphs`).
pub struct UnifiedOutlineSubgraphsPass {
    pub mode: String,
    pub dedup: bool,
    pub first: String,
    pub second: String,
    pub entry_name: String,
    pub repeat: usize,
    pub target: String,
    pub stop_at: String,
    pub max_ops: Option<usize>,
    pub keep_op: String,
}

impl Default for UnifiedOutlineSubgraphsPass {
    fn default() -> Self {
        Self {
            mode: "all".to_string(),
            dedup: false,
            first: String::new(),
            second: String::new(),
            entry_name: "composed_main".to_string(),
            repeat: 1,
            target: String::new(),
            stop_at: String::new(),
            max_ops: None,
            keep_op: String::new(),
        }
    }
}

impl Pass for UnifiedOutlineSubgraphsPass {
    fn name(&self) -> &'static str {
        "outline-subgraphs"
    }

    fn description(&self) -> &'static str {
        "Unified subgraph outlining, inlining, extraction, function dedup, composition & unrolling"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        match m.as_str() {
            "inline" | "inliner" => {
                steps.push(InlinerPass.run(module)?);
            }
            "extract" | "reduce" => {
                steps.push(
                    ExtractSubgraphPass {
                        target: self.target.clone(),
                        stop_at: self.stop_at.clone(),
                        max_ops: self.max_ops,
                        keep_op: self.keep_op.clone(),
                    }
                    .run(module)?,
                );
            }
            "dedup" => {
                steps.push(DedupFunctionsPass.run(module)?);
            }
            "compose" => {
                steps.push(
                    ComposeFuncsPass {
                        first: self.first.clone(),
                        second: self.second.clone(),
                        entry_name: self.entry_name.clone(),
                    }
                    .run(module)?,
                );
            }
            "unroll" => {
                steps.push(
                    UnrollCallsPass {
                        repeat: self.repeat.max(2),
                    }
                    .run(module)?,
                );
            }
            _ => {
                if !self.target.is_empty()
                    || !self.stop_at.is_empty()
                    || self.max_ops.is_some()
                    || !self.keep_op.is_empty()
                {
                    steps.push(
                        ExtractSubgraphPass {
                            target: self.target.clone(),
                            stop_at: self.stop_at.clone(),
                            max_ops: self.max_ops,
                            keep_op: self.keep_op.clone(),
                        }
                        .run(module)?,
                    );
                } else {
                    steps.push(
                        OutlineSubgraphsPass {
                            mode: self.mode.clone(),
                        }
                        .run(module)?,
                    );
                    if self.dedup {
                        steps.push(DedupFunctionsPass.run(module)?);
                    }
                    if self.repeat > 1 {
                        steps.push(
                            UnrollCallsPass {
                                repeat: self.repeat,
                            }
                            .run(module)?,
                        );
                    }
                }
            }
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

/// Subgraph Outliner Pass (`outline-subgraphs`).
#[derive(Debug, Clone)]
pub struct OutlineSubgraphsPass {
    pub mode: String,
}

impl Default for OutlineSubgraphsPass {
    fn default() -> Self {
        Self {
            mode: "all".to_string(),
        }
    }
}

impl Pass for OutlineSubgraphsPass {
    fn name(&self) -> &'static str {
        "outline-subgraphs"
    }

    fn description(&self) -> &'static str {
        "Outlines GEMM and composite subgraphs into private func.func kernels + func.call sites"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&caller_fn_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let caller_reg = module.ops[caller_fn_id].regions[0];
        let caller_block = module.regions[caller_reg].blocks[0];
        let ops = module.blocks[caller_block].ops.clone();

        let mode_lc = self.mode.to_ascii_lowercase();
        let mut outlined_count = 0usize;
        let mod_body = module.body_block();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let is_match = match mode_lc.as_str() {
                "gemm" | "dot" => op.is_op("stablehlo", "dot_general"),
                "composite" | "composites" => op.is_op("stablehlo", "composite"),
                _ => op.is_op("stablehlo", "dot_general") || op.is_op("stablehlo", "composite"),
            };
            if !is_match || op.results.is_empty() {
                continue;
            }

            let kernel_sym = format!("outlined_kernel_{outlined_count}");
            outlined_count += 1;

            let kernel_block = module.create_block(None);
            let mut kernel_args: SmallVec<[ValueId; 4]> = SmallVec::new();
            let mut in_types = Vec::with_capacity(op.operands.len());
            for (idx, &opnd) in op.operands.iter().enumerate() {
                let ty = module.values[opnd].ty.clone();
                in_types.push(ty.clone());
                let arg_val =
                    module.add_block_argument(kernel_block, Some(format!("arg{idx}")), ty);
                kernel_args.push(arg_val);
            }

            let out_types: Vec<Type> = op
                .results
                .iter()
                .map(|&r| module.values[r].ty.clone())
                .collect();
            let kernel_res_specs: Vec<(Option<String>, Type)> = out_types
                .iter()
                .enumerate()
                .map(|(i, t)| (Some(format!("k_res{i}")), t.clone()))
                .collect();

            let inner_op_id = module.add_operation(
                kernel_block,
                op.name.clone(),
                kernel_args,
                kernel_res_specs,
                op.attributes.clone(),
                Vec::new(),
                op.custom_syntax,
                op.location.clone(),
            );
            let inner_results: SmallVec<[ValueId; 4]> =
                SmallVec::from_slice(&module.ops[inner_op_id].results);
            module.add_operation(
                kernel_block,
                OpName::new("func", "return"),
                inner_results,
                Vec::new(),
                IndexMap::new(),
                Vec::new(),
                true,
                Location::Unknown,
            );

            let kernel_reg = module.create_region(vec![kernel_block]);
            let mut fn_attrs = IndexMap::new();
            fn_attrs.insert(
                "sym_name".to_string(),
                Attribute::String(kernel_sym.clone()),
            );
            fn_attrs.insert(
                "sym_visibility".to_string(),
                Attribute::String("private".to_string()),
            );
            fn_attrs.insert(
                "function_type".to_string(),
                Attribute::Type(Type::Function {
                    inputs: in_types,
                    results: out_types.clone(),
                }),
            );
            module.add_operation(
                mod_body,
                OpName::new("func", "func"),
                SmallVec::new(),
                Vec::new(),
                fn_attrs,
                vec![kernel_reg],
                true,
                Location::Unknown,
            );

            let call_res_specs: Vec<(Option<String>, Type)> = op
                .results
                .iter()
                .map(|&r| {
                    (
                        module.values[r].ssa_name.clone(),
                        module.values[r].ty.clone(),
                    )
                })
                .collect();
            let mut call_attrs = IndexMap::new();
            call_attrs.insert("callee".to_string(), Attribute::String(kernel_sym));

            let call_op_id = module.add_operation(
                caller_block,
                OpName::new("func", "call"),
                op.operands.clone(),
                call_res_specs,
                call_attrs,
                Vec::new(),
                true,
                op.location.clone(),
            );
            module.blocks[caller_block].ops.pop();
            let call_results = module.ops[call_op_id].results.clone();
            for (&old_r, &new_r) in op.results.iter().zip(call_results.iter()) {
                module.replace_all_uses(old_r, new_r);
            }
            if let Some(pos) = module.blocks[caller_block]
                .ops
                .iter()
                .position(|&id| id == op_id)
            {
                module.blocks[caller_block].ops.insert(pos, call_op_id);
            }
            module.remove_op(caller_block, op_id);
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: outlined_count > 0,
            ops_removed: outlined_count,
            ops_modified: 0,
            ops_created: outlined_count * 3,
        })
    }
}

/// Links all functions and [`TensorStore`](crate::ir::TensorStore) weights from `other` into
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

/// Function Composer Pass (`compose-funcs`).
#[derive(Debug, Clone)]
pub struct ComposeFuncsPass {
    pub first: String,
    pub second: String,
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

/// Bounded Call-Chain Unroller Pass (`unroll-calls`).
#[derive(Debug, Clone)]
pub struct UnrollCallsPass {
    pub repeat: usize,
}

impl Default for UnrollCallsPass {
    fn default() -> Self {
        Self { repeat: 2 }
    }
}

impl Pass for UnrollCallsPass {
    fn name(&self) -> &'static str {
        "unroll-calls"
    }

    fn description(&self) -> &'static str {
        "Unrolls repeated func.call invocations (repeat=N) for stacked layer blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let repeat = self.repeat.max(1);
        if repeat <= 1 {
            return Ok(PassResult::unchanged(self.name()));
        }
        let mut created = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("func", "call") || op.operands.is_empty() || op.results.len() != 1 {
                    continue;
                }
                let orig_res = op.results[0];
                let res_ty = module.values[orig_res].ty.clone();
                if module.values[op.operands[0]].ty != res_ty {
                    continue;
                }

                let mut cur_val = orig_res;
                let mut new_call_ids: Vec<OpId> = Vec::with_capacity(repeat - 1);
                for step in 1..repeat {
                    let mut next_operands: SmallVec<[_; 4]> = op.operands.clone();
                    next_operands[0] = cur_val;
                    let mut attrs = op.attributes.clone();
                    attrs.insert("unroll_step".to_string(), Attribute::Int(step as i64));

                    let call_id = module.add_operation(
                        block_id,
                        OpName::new("func", "call"),
                        next_operands,
                        vec![(Some(format!("call_u{step}")), res_ty.clone())],
                        attrs,
                        Vec::new(),
                        true,
                        op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    cur_val = module.ops[call_id].results[0];
                    new_call_ids.push(call_id);
                    created += 1;
                }

                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == op_id || new_call_ids.contains(&other_id) {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = cur_val;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id]
                        .ops
                        .splice((pos + 1)..(pos + 1), new_call_ids);
                }
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
