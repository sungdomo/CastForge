//! Subgraph Outliner & Partitioner Pass (`outline-subgraphs`: `Ir -> Ir`).
//!
//! Symmetric inverse of [`InlinerPass`](crate::InlinerPass) (`outline-subgraphs <-> inliner`).
//! Extracts matched compute operations (`dot_general`, `composite`, or `all`) from the entry
//! function into standalone private `@outlined_kernel_{k}` `func.func` definitions and replaces
//! them with `func.call @outlined_kernel_{k}(...)` invocations.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Location, Module, OpName, Type, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Subgraph Outliner Pass (`outline-subgraphs`).
#[derive(Debug, Clone)]
pub struct OutlineSubgraphsPass {
    /// Filter mode: `"all"` (default: `dot_general` + `composite`), `"gemm"`, or `"composite"`.
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

            // Replace original op in caller_block with `func.call @kernel_sym(...)`
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
