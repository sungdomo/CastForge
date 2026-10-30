//! Unified Elementwise, Map-Reduce & Register-Pipelined Loop-Chain Fusion (`fuse-elementwise`).
//!
//! Consolidates `fuse-elementwise` (`mode=composites`), `fuse-map-reduce` (`mode=map_reduce`),
//! `fuse-loops` (`mode=loops`), and `fission-loops` (`max_steps > 0`).

use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Type, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// Unified Elementwise, Map-Reduce & Loop-Chain Fusion Pass (`fuse-elementwise`).
pub struct UnifiedFuseElementwisePass {
    pub mode: String,
    pub max_steps: usize,
}

impl Default for UnifiedFuseElementwisePass {
    fn default() -> Self {
        Self {
            mode: "composites".to_string(),
            max_steps: 0,
        }
    }
}

impl Pass for UnifiedFuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Unified elementwise, map-reduce, and register loop-chain fusion (with max_steps fission)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        if m == "loops" || m == "chain" || self.max_steps > 0 {
            steps.push(FuseLoopsPass.run(module)?);
            if self.max_steps > 0 {
                steps.push(
                    FissionLoopsPass {
                        max_steps: self.max_steps,
                    }
                    .run(module)?,
                );
            }
        } else if m == "map_reduce" || m == "reduce" {
            steps.push(FuseMapReducePass.run(module)?);
        } else if m == "megakernel" || m == "persistent" {
            steps.push(FuseMegakernelPass.run(module)?);
        } else if m == "all" {
            steps.push(FuseElementwisePass.run(module)?);
            steps.push(FuseMapReducePass.run(module)?);
        } else {
            steps.push(FuseElementwisePass.run(module)?);
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

/// Producer-into-Reduction Fusion Pass (`fuse-map-reduce`).
pub struct FuseMapReducePass;

impl Pass for FuseMapReducePass {
    fn name(&self) -> &'static str {
        "fuse-map-reduce"
    }

    fn description(&self) -> &'static str {
        "Fuses elementwise map producers directly into stablehlo.reduce (fused_map_reduce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Reduce {
                    inputs,
                    init_values,
                    dimensions,
                    ..
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                let reducer = op
                    .attributes
                    .get("reducer")
                    .and_then(|a| a.as_str())
                    .unwrap_or("stablehlo.add")
                    .to_string();
                let Some(&mapped_val) = inputs.first() else {
                    continue;
                };
                let Some(&init_val) = init_values.first() else {
                    continue;
                };
                if module.use_count(mapped_val) != 1 {
                    continue;
                }
                let Some((src_x, map_kind)) = match_map_producer(module, mapped_val) else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_map_reduce".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_map_reduce_impl".to_string()),
                );
                attrs.insert(
                    "map_op".to_string(),
                    Attribute::String(map_kind.to_string()),
                );
                attrs.insert("reducer".to_string(), Attribute::String(reducer));
                attrs.insert(
                    "dimensions".to_string(),
                    Attribute::DenseIntArray(dimensions),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[src_x, init_val]);
                op_mut.attributes = attrs;
                created += 1;
            }
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

fn match_map_producer(module: &Module, val_id: ValueId) -> Option<(ValueId, &'static str)> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match StableHloOp::from_op(op)? {
        StableHloOp::Multiply { lhs, rhs, .. } if lhs == rhs => Some((lhs, "square")),
        StableHloOp::Exp { operand, .. } => Some((operand, "exponential")),
        StableHloOp::Tanh { operand, .. } => Some((operand, "tanh")),
        StableHloOp::Rsqrt { operand, .. } => Some((operand, "rsqrt")),
        _ => None,
    }
}

/// Fuse Elementwise Pass (`stablehlo.add` + `stablehlo.maximum` -> `stablehlo.composite`).
pub struct FuseElementwisePass;

impl Pass for FuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Maximum {
                    lhs,
                    rhs,
                    result: max_res,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let ValueDef::OpResult { op: add_op_id, .. } = module.values[lhs].def else {
                    continue;
                };
                if module.use_count(lhs) != 1 {
                    continue;
                }
                let Some(add_op) = module.ops.get(add_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add {
                    lhs: add_a,
                    rhs: add_b,
                    ..
                }) = StableHloOp::from_op(add_op)
                else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_add_max".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_add_max_impl".to_string()),
                );

                let max_op_mut = &mut module.ops[op_id];
                max_op_mut.name = OpName::new("stablehlo", "composite");
                max_op_mut.operands = SmallVec::from_slice(&[add_a, add_b, rhs]);
                max_op_mut.attributes = attrs;
                let _ = max_res;
                created += 1;

                module.remove_op(block_id, add_op_id);
                removed += 1;
            }
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

/// General Elementwise Loop-Chain Fusion Pass (`fuse-loops`).
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

/// Loop Fission / Loop Distribution Pass (`fission-loops`).
pub struct FissionLoopsPass {
    pub max_steps: usize,
}

impl Default for FissionLoopsPass {
    fn default() -> Self {
        Self { max_steps: 2 }
    }
}

impl Pass for FissionLoopsPass {
    fn name(&self) -> &'static str {
        "fission-loops"
    }

    fn description(&self) -> &'static str {
        "Distributes long fused_loop_chain ops into balanced sub-loops (max_steps=2)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let limit = self.max_steps.max(1);
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Composite {
                    name,
                    inputs,
                    results,
                    ..
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if name != "castforge.fused_loop_chain" || results.is_empty() {
                    continue;
                }
                let Some(steps_str) = op.attributes.get("steps").and_then(|a| a.as_str()) else {
                    continue;
                };
                let raw_steps: Vec<&str> = steps_str.split(';').filter(|s| !s.is_empty()).collect();
                if raw_steps.len() <= limit {
                    continue;
                }

                let out_ty = module.values[results[0]].ty.clone();
                let mut step_val_map: Vec<Option<ValueId>> = vec![None; raw_steps.len()];
                let chunks: Vec<&[&str]> = raw_steps.chunks(limit).collect();
                let mut global_step_offset = 0usize;

                for (chunk_idx, chunk) in chunks.iter().enumerate() {
                    let is_last = chunk_idx + 1 == chunks.len();
                    let (chunk_inputs, chunk_steps_str) =
                        remap_chunk_steps(chunk, global_step_offset, &inputs, &step_val_map);

                    if !is_last {
                        let inter_val = insert_subchain_before(
                            module,
                            block_id,
                            op_id,
                            &chunk_inputs,
                            &chunk_steps_str,
                            out_ty.clone(),
                        );
                        let last_global_in_chunk = global_step_offset + chunk.len() - 1;
                        step_val_map[last_global_in_chunk] = Some(inter_val);
                        created += 1;
                    } else {
                        let op_mut = &mut module.ops[op_id];
                        op_mut.operands = SmallVec::from_slice(&chunk_inputs);
                        op_mut
                            .attributes
                            .insert("steps".to_string(), Attribute::String(chunk_steps_str));
                        modified += 1;
                    }
                    global_step_offset += chunk.len();
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn remap_chunk_steps(
    chunk: &[&str],
    global_offset: usize,
    orig_inputs: &[ValueId],
    step_val_map: &[Option<ValueId>],
) -> (Vec<ValueId>, String) {
    let mut local_inputs: Vec<ValueId> = Vec::new();
    let mut local_specs: Vec<String> = Vec::with_capacity(chunk.len());

    let mut add_ext = |v: ValueId| -> usize {
        if let Some(pos) = local_inputs.iter().position(|&x| x == v) {
            pos
        } else {
            let pos = local_inputs.len();
            local_inputs.push(v);
            pos
        }
    };

    for &spec in chunk {
        let Some((op_name, rest)) = spec.split_once('(') else {
            continue;
        };
        let args_part = rest.trim_end_matches(')');
        let mut mapped_args = Vec::new();
        for tok in args_part.split(',').filter(|s| !s.is_empty()) {
            if let Some(s_idx_str) = tok.strip_prefix('s')
                && let Ok(global_s) = s_idx_str.parse::<usize>()
            {
                if global_s >= global_offset {
                    mapped_args.push(format!("s{}", global_s - global_offset));
                } else if let Some(Some(inter_val)) = step_val_map.get(global_s) {
                    let ext_idx = add_ext(*inter_val);
                    mapped_args.push(ext_idx.to_string());
                }
            } else if let Ok(orig_idx) = tok.parse::<usize>()
                && let Some(&orig_v) = orig_inputs.get(orig_idx)
            {
                let ext_idx = add_ext(orig_v);
                mapped_args.push(ext_idx.to_string());
            }
        }
        local_specs.push(format!("{}({})", op_name.trim(), mapped_args.join(",")));
    }

    (local_inputs, local_specs.join(";"))
}

fn insert_subchain_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    operands: &[ValueId],
    steps_str: &str,
    out_ty: Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.fused_loop_chain".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("fused_loop_chain_impl".to_string()),
    );
    attrs.insert(
        "steps".to_string(),
        Attribute::String(steps_str.to_string()),
    );

    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "composite"),
        operands: SmallVec::from_slice(operands),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(val_id);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, new_op_id);
    } else {
        block_ops.push(new_op_id);
    }
    val_id
}

/// Persistent Megakernel Fusion Pass (`fuse-megakernel`).
#[derive(Default)]
pub struct FuseMegakernelPass;

impl Pass for FuseMegakernelPass {
    fn name(&self) -> &'static str {
        "fuse-megakernel"
    }

    fn description(&self) -> &'static str {
        "Fuses sequential compute ops into a persistent grid-synchronized megakernel"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for fn_id in module.functions() {
            let fn_op = &mut module.ops[fn_id];
            if fn_op.attributes.get("castforge.megakernel") != Some(&Attribute::Bool(true)) {
                fn_op
                    .attributes
                    .insert("castforge.megakernel".to_string(), Attribute::Bool(true));
                fn_op.attributes.insert(
                    "castforge.grid_sync".to_string(),
                    Attribute::String("cooperative_groups".to_string()),
                );
                modified += 1;
            }
        }
        for (_, op) in module.ops.iter_mut() {
            if op.name.dialect == "stablehlo" && op.name.name != "constant" {
                op.attributes
                    .insert("castforge.megakernel".to_string(), Attribute::Bool(true));
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
