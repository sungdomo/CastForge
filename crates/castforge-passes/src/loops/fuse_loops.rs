//! General multi-operation elementwise loop fusion pass (`fuse-loops`).

use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Module, OpId, OpName, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// Loop Optimization Pass: General Elementwise Loop Fusion (`fuse-loops`).
///
/// Fuses arbitrary-length producer-consumer chains/DAGs of elementwise loops (`add`, `subtract`,
/// `multiply`, `divide`, `maximum`, `minimum`, `exponential`, `tanh`, `rsqrt`) sharing the same
/// trip count into a single `stablehlo.composite` (`"castforge.fused_loop_chain"`) that executes
/// in one loop with scalar register temporaries (`s0`, `s1`, ...).
pub struct FuseLoopsPass;

impl Pass for FuseLoopsPass {
    fn name(&self) -> &'static str {
        "fuse-loops"
    }

    fn description(&self) -> &'static str {
        "Fuses multi-op elementwise loop chains into a single register-pipelined loop"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            created += fuse_block_elementwise_chains(module, block_id);
        }

        if created > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

fn fuse_block_elementwise_chains(module: &mut Module, block_id: BlockId) -> usize {
    let op_ids = module.blocks[block_id].ops.clone();
    let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
    for &op_id in &op_ids {
        if let Some(op) = module.ops.get(op_id) {
            for &v in &op.operands {
                *use_counts.entry(v).or_insert(0) += 1;
            }
        }
    }

    let mut def_op: HashMap<ValueId, OpId> = HashMap::new();
    for &op_id in &op_ids {
        if is_fusable_elementwise(module, op_id)
            && let Some(&res) = module.ops[op_id].results.first()
        {
            def_op.insert(res, op_id);
        }
    }

    // Identify single-use intermediate elementwise results consumed by another elementwise op
    // of identical tensor type.
    let mut consumed_internally: HashSet<OpId> = HashSet::new();
    for &op_id in &op_ids {
        if !is_fusable_elementwise(module, op_id) {
            continue;
        }
        let out_ty = &module.values[module.ops[op_id].results[0]].ty;
        for &inp in &module.ops[op_id].operands {
            if let Some(&prod_op_id) = def_op.get(&inp)
                && use_counts.get(&inp).copied().unwrap_or(0) == 1
                && &module.values[inp].ty == out_ty
            {
                consumed_internally.insert(prod_op_id);
            }
        }
    }

    let mut fused_count = 0usize;
    // Process roots (elementwise ops that are NOT consumed internally by another elementwise op)
    for &root_op_id in &op_ids {
        if !is_fusable_elementwise(module, root_op_id) || consumed_internally.contains(&root_op_id)
        {
            continue;
        }

        let mut chain_ops: Vec<OpId> = Vec::new();
        collect_chain_postorder(
            module,
            root_op_id,
            &def_op,
            &consumed_internally,
            &mut chain_ops,
        );
        if chain_ops.len() < 2 {
            continue;
        }

        let mut ext_inputs: Vec<ValueId> = Vec::new();
        let mut step_index_by_val: HashMap<ValueId, usize> = HashMap::new();
        let mut step_specs: Vec<String> = Vec::new();

        for (step_idx, &step_op_id) in chain_ops.iter().enumerate() {
            let op = &module.ops[step_op_id];
            let short = op.name.name.as_str();
            let mut arg_tokens = Vec::with_capacity(op.operands.len());
            for &v in &op.operands {
                if let Some(&prev_step) = step_index_by_val.get(&v) {
                    arg_tokens.push(format!("s{prev_step}"));
                } else {
                    let pos = if let Some(idx) = ext_inputs.iter().position(|&x| x == v) {
                        idx
                    } else {
                        let idx = ext_inputs.len();
                        ext_inputs.push(v);
                        idx
                    };
                    arg_tokens.push(pos.to_string());
                }
            }
            step_specs.push(format!("{short}({})", arg_tokens.join(",")));
            step_index_by_val.insert(op.results[0], step_idx);
        }

        let mut attrs = IndexMap::new();
        attrs.insert(
            "name".to_string(),
            Attribute::String("castforge.fused_loop_chain".to_string()),
        );
        attrs.insert(
            "decomposition".to_string(),
            Attribute::SymbolRef("fused_loop_chain_impl".to_string()),
        );
        attrs.insert("steps".to_string(), Attribute::String(step_specs.join(";")));

        let root_mut = &mut module.ops[root_op_id];
        root_mut.name = OpName::new("stablehlo", "composite");
        root_mut.operands = SmallVec::from_slice(&ext_inputs);
        root_mut.attributes = attrs;
        fused_count += 1;
    }

    fused_count
}

fn collect_chain_postorder(
    module: &Module,
    op_id: OpId,
    def_op: &HashMap<ValueId, OpId>,
    consumed_internally: &HashSet<OpId>,
    out: &mut Vec<OpId>,
) {
    let operands = module.ops[op_id].operands.clone();
    for v in operands {
        if let Some(&prod_op_id) = def_op.get(&v)
            && consumed_internally.contains(&prod_op_id)
            && !out.contains(&prod_op_id)
        {
            collect_chain_postorder(module, prod_op_id, def_op, consumed_internally, out);
        }
    }
    if !out.contains(&op_id) {
        out.push(op_id);
    }
}

fn is_fusable_elementwise(module: &Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    if op.name.dialect != "stablehlo" || op.results.len() != 1 {
        return false;
    }
    matches!(
        op.name.name.as_str(),
        "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "maximum"
            | "minimum"
            | "exponential"
            | "tanh"
            | "rsqrt"
    )
}
