//! Unified Subgraph Outlining, Function Deduplication, Composition & Call Unrolling
//! (`outline-subgraphs`).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Location, Module, OpId, OpName, Type, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// Unified Subgraph Outlining, Inlining, Extraction, Deduplication, Composition & Unrolling
/// (`outline-subgraphs`).
///
/// Consolidates:
/// - `mode=all|gemm|composites` (`OutlineSubgraphsPass`)
/// - `mode=inline` (`InlinerPass`)
/// - `mode=extract|reduce` (`ExtractSubgraphPass` / `ReduceIrPass`)
/// - `mode=dedup` or `dedup=true` (`DedupFunctionsPass`)
/// - `mode=compose` (`ComposeFuncsPass`)
/// - `mode=unroll` or `repeat > 1` (`UnrollCallsPass`)
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
                steps.push(crate::InlinerPass.run(module)?);
            }
            "extract" | "reduce" => {
                steps.push(
                    crate::ExtractSubgraphPass {
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
                        crate::ExtractSubgraphPass {
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

/// Function Call Inliner Pass (`inliner`).
pub struct InlinerPass;

impl Pass for InlinerPass {
    fn name(&self) -> &'static str {
        "inliner"
    }

    fn description(&self) -> &'static str {
        "Inlines func.call operations by splicing callee bodies into caller blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut created = 0usize;

        let mut func_map: HashMap<String, OpId> = HashMap::new();
        for fn_op_id in module.functions() {
            if let Some(name) = module.ops[fn_op_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
            {
                func_map.insert(name.to_string(), fn_op_id);
            }
        }

        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for call_op_id in ops {
                let Some(call_op) = module.ops.get(call_op_id).cloned() else {
                    continue;
                };
                if !call_op.is_op("func", "call") {
                    continue;
                }
                let Some(callee_name) = call_op.attributes.get("callee").and_then(|a| a.as_str())
                else {
                    continue;
                };
                let Some(&callee_fn_id) = func_map.get(callee_name) else {
                    continue;
                };
                let callee_reg = module.ops[callee_fn_id].regions[0];
                let callee_block = module.regions[callee_reg].blocks[0];
                if callee_block == block_id {
                    continue;
                }

                let callee_args = module.blocks[callee_block].arguments.clone();
                if callee_args.len() != call_op.operands.len() {
                    continue;
                }

                let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
                for (&arg_val, &caller_val) in callee_args.iter().zip(call_op.operands.iter()) {
                    val_map.insert(arg_val, caller_val);
                }

                let callee_ops = module.blocks[callee_block].ops.clone();
                let mut new_op_ids = Vec::new();
                let mut return_values = Vec::new();

                for src_op_id in callee_ops {
                    let src_op = module.ops[src_op_id].clone();
                    if src_op.is_op("func", "return") {
                        for &ret_opnd in &src_op.operands {
                            let mapped = val_map.get(&ret_opnd).copied().unwrap_or(ret_opnd);
                            return_values.push(mapped);
                        }
                        continue;
                    }

                    let mapped_operands: SmallVec<[ValueId; 4]> = src_op
                        .operands
                        .iter()
                        .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                        .collect();

                    let res_types: Vec<(Option<String>, Type)> = src_op
                        .results
                        .iter()
                        .map(|&r| {
                            let v = &module.values[r];
                            let inlined_name = v
                                .ssa_name
                                .as_ref()
                                .map(|n| format!("inl_{callee_name}_{n}"));
                            (inlined_name, v.ty.clone())
                        })
                        .collect();

                    let cloned_id = module.add_operation(
                        block_id,
                        src_op.name.clone(),
                        mapped_operands,
                        res_types,
                        src_op.attributes.clone(),
                        Vec::new(),
                        src_op.custom_syntax,
                        src_op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    for (&old_res, &new_res) in src_op
                        .results
                        .iter()
                        .zip(module.ops[cloned_id].results.iter())
                    {
                        val_map.insert(old_res, new_res);
                    }
                    new_op_ids.push(cloned_id);
                    created += 1;
                }

                for (&call_res, &ret_val) in call_op.results.iter().zip(return_values.iter()) {
                    module.replace_all_uses(call_res, ret_val);
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == call_op_id)
                {
                    module.blocks[block_id].ops.splice(pos..pos, new_op_ids);
                }
                module.remove_op(block_id, call_op_id);
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

/// Unified Subgraph Extractor & Delta IR Reducer Pass (`extract-subgraph`).
#[derive(Debug, Clone, Default)]
pub struct ExtractSubgraphPass {
    pub target: String,
    pub stop_at: String,
    pub max_ops: Option<usize>,
    pub keep_op: String,
}

impl Pass for ExtractSubgraphPass {
    fn name(&self) -> &'static str {
        "extract-subgraph"
    }

    fn description(&self) -> &'static str {
        "Extracts backward dataflow slice or delta-reduces function to a minimal reproducer"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        if self.max_ops.is_some() || !self.keep_op.is_empty() {
            let mut res = ReduceIrPass {
                max_ops: self.max_ops.unwrap_or(1),
                keep_op: self.keep_op.clone(),
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }

        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let compute_ops: Vec<OpId> = ops
            .iter()
            .copied()
            .filter(|&id| !module.ops[id].is_op("func", "return"))
            .collect();
        if compute_ops.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let clean_target = self.target.trim().trim_start_matches('%');
        let mut target_val: Option<ValueId> = None;

        if !clean_target.is_empty() {
            for &op_id in &compute_ops {
                for &res in &module.ops[op_id].results {
                    if module.values[res].ssa_name.as_deref() == Some(clean_target) {
                        target_val = Some(res);
                        break;
                    }
                }
            }
            if target_val.is_none()
                && let Ok(idx) = clean_target.parse::<usize>()
                && let Some(&op_id) = compute_ops.get(idx)
            {
                target_val = module.ops[op_id].results.first().copied();
            }
        } else {
            for &op_id in &compute_ops {
                if !module.ops[op_id].is_op("stablehlo", "constant")
                    && let Some(&res) = module.ops[op_id].results.first()
                {
                    target_val = Some(res);
                    break;
                }
            }
        }

        let Some(target_vid) = target_val else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let stop_names: HashSet<String> = self
            .stop_at
            .split([',', ';'])
            .map(|s| s.trim().trim_start_matches('%').to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut keep_ops: HashSet<OpId> = HashSet::new();
        let mut promoted_vals: Vec<ValueId> = Vec::new();
        let mut visited_vals: HashSet<ValueId> = HashSet::new();
        let mut worklist = vec![target_vid];

        while let Some(vid) = worklist.pop() {
            if !visited_vals.insert(vid) {
                continue;
            }
            let ssa = module.values[vid].ssa_name.as_deref().unwrap_or("");
            if vid != target_vid && stop_names.contains(ssa) {
                if matches!(module.values[vid].def, ValueDef::OpResult { .. }) {
                    promoted_vals.push(vid);
                }
                continue;
            }
            if let ValueDef::OpResult { op: prod_op, .. } = module.values[vid].def {
                keep_ops.insert(prod_op);
                for &opnd in &module.ops[prod_op].operands {
                    worklist.push(opnd);
                }
            }
        }

        for &p_vid in &promoted_vals {
            let ssa = module.values[p_vid].ssa_name.clone();
            let ty = module.values[p_vid].ty.clone();
            let new_arg = module.add_block_argument(block_id, ssa, ty);
            module.replace_all_uses(p_vid, new_arg);
        }

        let mut removed = 0usize;
        let mut modified = 0usize;
        for &op_id in &ops {
            if module.ops[op_id].is_op("func", "return") {
                if module.ops[op_id].operands.as_slice() != [target_vid] {
                    module.ops[op_id].operands = smallvec![target_vid];
                    modified += 1;
                }
                continue;
            }
            if !keep_ops.contains(&op_id) {
                module.remove_op(block_id, op_id);
                removed += 1;
            }
        }

        let in_types: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let out_types = vec![module.values[target_vid].ty.clone()];
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: in_types,
                results: out_types,
            }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || modified > 0 || !promoted_vals.is_empty(),
            ops_removed: removed,
            ops_modified: modified,
            ops_created: promoted_vals.len(),
        })
    }
}

/// Delta IR Reducer Pass (`reduce-ir`).
#[derive(Debug, Clone)]
pub struct ReduceIrPass {
    pub max_ops: usize,
    pub keep_op: String,
}

impl Default for ReduceIrPass {
    fn default() -> Self {
        Self {
            max_ops: 1,
            keep_op: String::new(),
        }
    }
}

impl Pass for ReduceIrPass {
    fn name(&self) -> &'static str {
        "reduce-ir"
    }

    fn description(&self) -> &'static str {
        "Delta-reduces a function to a minimal reproducer of at most max_ops operations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&fn_op_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let reg_id = module.ops[fn_op_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        let compute_ops: Vec<OpId> = ops
            .iter()
            .copied()
            .filter(|&id| !module.ops[id].is_op("func", "return"))
            .collect();
        let limit = self.max_ops.max(1);
        if compute_ops.len() <= limit && self.keep_op.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let filter = self.keep_op.trim().to_ascii_lowercase();
        let mut retained: Vec<OpId> = compute_ops
            .iter()
            .copied()
            .filter(|&id| {
                if filter.is_empty() {
                    !module.ops[id].is_op("stablehlo", "constant")
                } else {
                    module.ops[id]
                        .name
                        .to_string()
                        .to_ascii_lowercase()
                        .contains(&filter)
                }
            })
            .take(limit)
            .collect();

        if retained.is_empty()
            && let Some(&first_op) = compute_ops.first()
        {
            retained.push(first_op);
        }

        let Some(&last_op) = retained.last() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let Some(&ret_val) = module.ops[last_op].results.first() else {
            return Ok(PassResult::unchanged(self.name()));
        };

        let keep_set: HashSet<OpId> = retained.iter().copied().collect();
        let mut to_promote: Vec<ValueId> = Vec::new();
        for &op_id in &retained {
            for &opnd in &module.ops[op_id].operands {
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[opnd].def
                    && !keep_set.contains(&prod_op)
                    && !to_promote.contains(&opnd)
                {
                    to_promote.push(opnd);
                }
            }
        }

        for &vid in &to_promote {
            let ssa = module.values[vid].ssa_name.clone();
            let ty = module.values[vid].ty.clone();
            let new_arg = module.add_block_argument(block_id, ssa, ty);
            module.replace_all_uses(vid, new_arg);
        }

        let mut removed = 0usize;
        for &op_id in &ops {
            if module.ops[op_id].is_op("func", "return") {
                module.ops[op_id].operands = smallvec![ret_val];
                continue;
            }
            if !keep_set.contains(&op_id) {
                module.remove_op(block_id, op_id);
                removed += 1;
            }
        }

        let in_types: Vec<Type> = module.blocks[block_id]
            .arguments
            .iter()
            .map(|&a| module.values[a].ty.clone())
            .collect();
        let out_types = vec![module.values[ret_val].ty.clone()];
        module.ops[fn_op_id].attributes.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: in_types,
                results: out_types,
            }),
        );

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || !to_promote.is_empty(),
            ops_removed: removed,
            ops_modified: 1,
            ops_created: to_promote.len(),
        })
    }
}
