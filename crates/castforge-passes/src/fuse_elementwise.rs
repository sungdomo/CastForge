//! Unified Elementwise, Map-Reduce, Loop-Chain Fusion & Composite Decomposition
//! (`fuse-elementwise`).
//!
//! Consolidates `fuse-elementwise` (`mode=composites`), `fuse-map-reduce` (`mode=map_reduce`),
//! `fuse-loops` (`mode=loops`), `fission-loops` (`max_steps > 0`), and `decompose-composites`
//! (`mode=decompose`).

use crate::{DeadCodeEliminationPass, FoldConstantsPass, FuseHorizontalDotPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// Unified Elementwise, Activation, Gated-MLP, MoE, Map-Reduce, Loop-Chain Fusion & Composite
/// Decomposition Pass (`fuse-elementwise`).
pub struct UnifiedFuseElementwisePass {
    pub mode: String,
    pub max_steps: usize,
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for UnifiedFuseElementwisePass {
    fn default() -> Self {
        Self {
            mode: "composites".to_string(),
            max_steps: 0,
            shared_expert: false,
            top_k: 2,
            capacity_factor: 1.25,
        }
    }
}

impl Pass for UnifiedFuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Unified elementwise, activation (SwiGLU/GELU/GeGLU/MoE), map-reduce & loop-chain fusion"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        if matches!(
            m.as_str(),
            "activations" | "swiglu" | "gelu" | "geglu" | "gated_mlp" | "mlp" | "moe"
        ) || self.shared_expert
        {
            let kind = if m == "activations" || m == "composites" {
                "all".to_string()
            } else {
                m
            };
            steps.push(
                FuseActivationsPass {
                    kind,
                    shared_expert: self.shared_expert,
                    top_k: self.top_k,
                    capacity_factor: self.capacity_factor,
                }
                .run(module)?,
            );
        } else if matches!(m.as_str(), "decompose" | "expand" | "lower") {
            steps.push(DecomposeCompositesPass.run(module)?);
        } else if m == "loops" || m == "chain" || self.max_steps > 0 {
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
            steps.push(FuseActivationsPass::default().run(module)?);
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

/// Inverse Macro-Fusion Pass: Expands `stablehlo.composite` operations
/// (`castforge.fused_add_max`, `castforge.fused_matmul_epilogue`,
/// `castforge.fused_softmax`, `castforge.fused_rmsnorm`) back into primitive
/// `stablehlo` operations for targets without specialized composite kernels.
pub struct DecomposeCompositesPass;

impl Pass for DecomposeCompositesPass {
    fn name(&self) -> &'static str {
        "decompose-composites"
    }

    fn description(&self) -> &'static str {
        "Expands stablehlo.composite operations back into primitive StableHLO ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
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
                let Some(&res_val) = results.first() else {
                    continue;
                };
                let out_ty = module.values[res_val].ty.clone();

                match name.as_str() {
                    "castforge.fused_add_max" if inputs.len() == 3 => {
                        let add_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "add"),
                            &[inputs[0], inputs[1]],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "maximum");
                        op_mut.operands = SmallVec::from_slice(&[add_val, inputs[2]]);
                        op_mut.attributes.clear();
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_matmul_epilogue" if inputs.len() >= 3 => {
                        let epilogue = op
                            .attributes
                            .get("epilogue")
                            .and_then(|a| a.as_str())
                            .unwrap_or("bias")
                            .to_string();
                        let mut dot_attrs = IndexMap::new();
                        for k in [
                            "lhs_batching_dimensions",
                            "rhs_batching_dimensions",
                            "lhs_contracting_dimensions",
                            "rhs_contracting_dimensions",
                        ] {
                            if let Some(v) = op.attributes.get(k) {
                                dot_attrs.insert(k.to_string(), v.clone());
                            }
                        }
                        let dot_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[inputs[0], inputs[1]],
                            dot_attrs,
                            out_ty.clone(),
                        );
                        created += 1;

                        match epilogue.as_str() {
                            "bias_relu" if inputs.len() >= 4 => {
                                let add_val = insert_op_before(
                                    module,
                                    block_id,
                                    op_id,
                                    OpName::new("stablehlo", "add"),
                                    &[dot_val, inputs[2]],
                                    IndexMap::new(),
                                    out_ty,
                                );
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "maximum");
                                op_mut.operands = SmallVec::from_slice(&[add_val, inputs[3]]);
                                op_mut.attributes.clear();
                                created += 1;
                                modified += 1;
                            }
                            "bias_tanh" => {
                                let add_val = insert_op_before(
                                    module,
                                    block_id,
                                    op_id,
                                    OpName::new("stablehlo", "add"),
                                    &[dot_val, inputs[2]],
                                    IndexMap::new(),
                                    out_ty,
                                );
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "tanh");
                                op_mut.operands = SmallVec::from_slice(&[add_val]);
                                op_mut.attributes.clear();
                                created += 1;
                                modified += 1;
                            }
                            "bias_silu" => {
                                let add_val = insert_op_before(
                                    module,
                                    block_id,
                                    op_id,
                                    OpName::new("stablehlo", "add"),
                                    &[dot_val, inputs[2]],
                                    IndexMap::new(),
                                    out_ty.clone(),
                                );
                                let denom_val = insert_silu_denom_before(
                                    module, block_id, op_id, add_val, &out_ty,
                                );
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "divide");
                                op_mut.operands = SmallVec::from_slice(&[add_val, denom_val]);
                                op_mut.attributes.clear();
                                created += 6;
                                modified += 1;
                            }
                            "bias_scale" if inputs.len() >= 4 => {
                                let add_val = insert_op_before(
                                    module,
                                    block_id,
                                    op_id,
                                    OpName::new("stablehlo", "add"),
                                    &[dot_val, inputs[2]],
                                    IndexMap::new(),
                                    out_ty,
                                );
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "multiply");
                                op_mut.operands = SmallVec::from_slice(&[add_val, inputs[3]]);
                                op_mut.attributes.clear();
                                created += 1;
                                modified += 1;
                            }
                            _ => {
                                let op_mut = &mut module.ops[op_id];
                                op_mut.name = OpName::new("stablehlo", "add");
                                op_mut.operands = SmallVec::from_slice(&[dot_val, inputs[2]]);
                                op_mut.attributes.clear();
                                modified += 1;
                            }
                        }
                    }
                    "castforge.fused_swiglu" if inputs.len() >= 2 => {
                        let gate = inputs[0];
                        let up = inputs[1];
                        let denom_val =
                            insert_silu_denom_before(module, block_id, op_id, gate, &out_ty);
                        let silu_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "divide"),
                            &[gate, denom_val],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[silu_val, up]);
                        op_mut.attributes.clear();
                        created += 6;
                        modified += 1;
                    }
                    "castforge.fused_softmax" if !inputs.is_empty() => {
                        let x = inputs[0];
                        let exp_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "exponential"),
                            &[x],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let (red_ty, axis, bcast_dims) = reduced_row_type(&out_ty);
                        let zero_init = insert_scalar_const_before(module, block_id, op_id, 0.0);
                        let mut red_attrs = IndexMap::new();
                        red_attrs.insert(
                            "dimensions".to_string(),
                            Attribute::DenseIntArray(vec![axis]),
                        );
                        red_attrs.insert(
                            "reducer".to_string(),
                            Attribute::String("stablehlo.add".to_string()),
                        );
                        let sum_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "reduce"),
                            &[exp_val, zero_init],
                            red_attrs,
                            red_ty,
                        );
                        let mut bcast_attrs = IndexMap::new();
                        bcast_attrs.insert(
                            "broadcast_dimensions".to_string(),
                            Attribute::DenseIntArray(bcast_dims),
                        );
                        let denom_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "broadcast_in_dim"),
                            &[sum_val],
                            bcast_attrs,
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "divide");
                        op_mut.operands = SmallVec::from_slice(&[exp_val, denom_val]);
                        op_mut.attributes.clear();
                        created += 4;
                        modified += 1;
                    }
                    "castforge.fused_rope" if inputs.len() >= 3 => {
                        let (x, cos_v, sin_v) = (inputs[0], inputs[1], inputs[2]);
                        let (half_ty, rows, cols) = half_col_type(&out_ty);
                        let half = cols / 2;
                        let x_lo = insert_slice_2d_before(
                            module,
                            block_id,
                            op_id,
                            x,
                            [0, 0],
                            [rows, half],
                            half_ty.clone(),
                        );
                        let x_hi = insert_slice_2d_before(
                            module,
                            block_id,
                            op_id,
                            x,
                            [0, half],
                            [rows, cols],
                            half_ty.clone(),
                        );
                        let zero_hi = insert_zero_tensor_before(module, block_id, op_id, &half_ty);
                        let neg_hi = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "subtract"),
                            &[zero_hi, x_hi],
                            IndexMap::new(),
                            half_ty,
                        );
                        let mut cat_attrs = IndexMap::new();
                        cat_attrs.insert("dimension".to_string(), Attribute::Int(1));
                        let rot_x = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "concatenate"),
                            &[neg_hi, x_lo],
                            cat_attrs,
                            out_ty.clone(),
                        );
                        let mul_cos = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[x, cos_v],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let mul_sin = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[rot_x, sin_v],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[mul_cos, mul_sin]);
                        op_mut.attributes.clear();
                        created += 7;
                        modified += 1;
                    }
                    "castforge.fused_rmsnorm" if !inputs.is_empty() => {
                        let eps = match op.attributes.get("eps") {
                            Some(Attribute::Float(f)) => *f as f32,
                            _ => 1e-5,
                        };
                        let (norm_val, c_cnt) = insert_rmsnorm_core_before(
                            module, block_id, op_id, inputs[0], eps, &out_ty,
                        );
                        created += c_cnt;
                        if let Some(&w_val) = inputs.get(1) {
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "multiply");
                            op_mut.operands = SmallVec::from_slice(&[norm_val, w_val]);
                            op_mut.attributes.clear();
                            modified += 1;
                        } else {
                            let one_t =
                                insert_splat_tensor_before(module, block_id, op_id, &out_ty, 1.0);
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "multiply");
                            op_mut.operands = SmallVec::from_slice(&[norm_val, one_t]);
                            op_mut.attributes.clear();
                            created += 1;
                            modified += 1;
                        }
                    }
                    "castforge.fused_residual_rmsnorm" if inputs.len() >= 2 => {
                        let eps = match op.attributes.get("eps") {
                            Some(Attribute::Float(f)) => *f as f32,
                            _ => 1e-5,
                        };
                        let h_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "add"),
                            &[inputs[0], inputs[1]],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let (norm_val, c_cnt) = insert_rmsnorm_core_before(
                            module, block_id, op_id, h_val, eps, &out_ty,
                        );
                        created += 1 + c_cnt;
                        let w_val = inputs.get(2).copied().unwrap_or_else(|| {
                            insert_splat_tensor_before(module, block_id, op_id, &out_ty, 1.0)
                        });
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[norm_val, w_val]);
                        op_mut.attributes.clear();
                        modified += 1;
                    }
                    "castforge.fused_gelu" if !inputs.is_empty() => {
                        let gelu_val =
                            insert_gelu_before(module, block_id, op_id, inputs[0], &out_ty);
                        let zero_t = insert_zero_tensor_before(module, block_id, op_id, &out_ty);
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[gelu_val, zero_t]);
                        op_mut.attributes.clear();
                        created += 9;
                        modified += 1;
                    }
                    "castforge.fused_geglu" if inputs.len() >= 2 => {
                        let gelu_val =
                            insert_gelu_before(module, block_id, op_id, inputs[0], &out_ty);
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "multiply");
                        op_mut.operands = SmallVec::from_slice(&[gelu_val, inputs[1]]);
                        op_mut.attributes.clear();
                        created += 8;
                        modified += 1;
                    }
                    "castforge.fused_qmatmul" if inputs.len() >= 2 => {
                        let w_q = inputs[1];
                        let w_f32_ty = if let Some(t) = module.values[w_q].ty.as_tensor() {
                            Type::Tensor(TensorType {
                                shape: t.shape.clone(),
                                dtype: DType::F32,
                                encoding: None,
                            })
                        } else {
                            out_ty.clone()
                        };
                        let deq_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "uniform_dequantize"),
                            &[w_q],
                            IndexMap::new(),
                            w_f32_ty,
                        );
                        let packed = op
                            .attributes
                            .get("packed_rhs")
                            .is_some_and(|a| matches!(a, Attribute::Bool(true)));
                        let mut dot_attrs = IndexMap::new();
                        dot_attrs.insert(
                            "lhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![1]),
                        );
                        dot_attrs.insert(
                            "rhs_contracting_dimensions".to_string(),
                            Attribute::DenseIntArray(vec![if packed { 1 } else { 0 }]),
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "dot_general");
                        op_mut.operands = SmallVec::from_slice(&[inputs[0], deq_val]);
                        op_mut.attributes = dot_attrs;
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_map_reduce" if inputs.len() >= 2 => {
                        let map_op = op
                            .attributes
                            .get("map_op")
                            .and_then(|a| a.as_str())
                            .unwrap_or("exponential")
                            .to_string();
                        let x_ty = module.values[inputs[0]].ty.clone();
                        let mapped_val = if map_op == "square" {
                            insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", "multiply"),
                                &[inputs[0], inputs[0]],
                                IndexMap::new(),
                                x_ty,
                            )
                        } else {
                            insert_op_before(
                                module,
                                block_id,
                                op_id,
                                OpName::new("stablehlo", &map_op),
                                &[inputs[0]],
                                IndexMap::new(),
                                x_ty,
                            )
                        };
                        let mut red_attrs = IndexMap::new();
                        if let Some(d) = op.attributes.get("dimensions") {
                            red_attrs.insert("dimensions".to_string(), d.clone());
                        }
                        if let Some(r) = op.attributes.get("reducer") {
                            red_attrs.insert("reducer".to_string(), r.clone());
                        }
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "reduce");
                        op_mut.operands = SmallVec::from_slice(&[mapped_val, inputs[1]]);
                        op_mut.attributes = red_attrs;
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_log_softmax" if !inputs.is_empty() => {
                        let x = inputs[0];
                        let exp_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "exponential"),
                            &[x],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let (red_ty, axis, bcast_dims) = reduced_row_type(&out_ty);
                        let zero_init = insert_scalar_const_before(module, block_id, op_id, 0.0);
                        let mut red_attrs = IndexMap::new();
                        red_attrs.insert(
                            "dimensions".to_string(),
                            Attribute::DenseIntArray(vec![axis]),
                        );
                        red_attrs.insert(
                            "reducer".to_string(),
                            Attribute::String("stablehlo.add".to_string()),
                        );
                        let sum_val = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "reduce"),
                            &[exp_val, zero_init],
                            red_attrs.clone(),
                            red_ty.clone(),
                        );
                        let log_sum = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "log"),
                            &[sum_val],
                            IndexMap::new(),
                            red_ty,
                        );
                        let mut bcast_attrs = IndexMap::new();
                        bcast_attrs.insert(
                            "broadcast_dimensions".to_string(),
                            Attribute::DenseIntArray(bcast_dims),
                        );
                        let lse_bcast = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "broadcast_in_dim"),
                            &[log_sum],
                            bcast_attrs,
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "subtract");
                        op_mut.operands = SmallVec::from_slice(&[x, lse_bcast]);
                        op_mut.attributes.clear();
                        created += 5;
                        modified += 1;
                    }
                    "castforge.fma" if inputs.len() >= 3 => {
                        let (a, b, c) = (inputs[0], inputs[1], inputs[2]);
                        let prod = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[a, b],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[prod, c]);
                        op_mut.attributes.clear();
                        created += 1;
                        modified += 1;
                    }
                    "castforge.fused_moe" if inputs.len() >= 5 => {
                        let (x, w0, w1, g0, g1) =
                            (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
                        let mut dot_attrs = IndexMap::new();
                        dot_attrs.insert(
                            "lhs_contracting_dimensions".to_string(),
                            Attribute::Array(vec![Attribute::Int(1)]),
                        );
                        dot_attrs.insert(
                            "rhs_contracting_dimensions".to_string(),
                            Attribute::Array(vec![Attribute::Int(0)]),
                        );
                        let e0 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w0],
                            dot_attrs.clone(),
                            out_ty.clone(),
                        );
                        let e1 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "dot_general"),
                            &[x, w1],
                            dot_attrs,
                            out_ty.clone(),
                        );
                        let m0 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[e0, g0],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let m1 = insert_op_before(
                            module,
                            block_id,
                            op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[e1, g1],
                            IndexMap::new(),
                            out_ty,
                        );
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "add");
                        op_mut.operands = SmallVec::from_slice(&[m0, m1]);
                        op_mut.attributes.clear();
                        created += 4;
                        modified += 1;
                    }
                    "castforge.stream_wait"
                    | "castforge.static_quantize"
                    | "castforge.nan_guard"
                    | "castforge.host_offload"
                    | "castforge.profile_region"
                        if !inputs.is_empty() =>
                    {
                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "convert");
                        op_mut.operands = SmallVec::from_slice(&[inputs[0]]);
                        op_mut.attributes.clear();
                        modified += 1;
                    }
                    _ => {}
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

fn reduced_row_type(out_ty: &Type) -> (Type, i64, Vec<i64>) {
    if let Some(t) = out_ty.as_tensor() {
        let rank = t.shape.len();
        if rank >= 2 {
            let reduced_shape = t.shape[..rank - 1].to_vec();
            let axis = (rank - 1) as i64;
            let bcast_dims: Vec<i64> = (0..(rank - 1) as i64).collect();
            return (
                Type::Tensor(TensorType {
                    shape: reduced_shape,
                    dtype: t.dtype.clone(),
                    encoding: None,
                }),
                axis,
                bcast_dims,
            );
        }
    }
    (
        Type::Tensor(TensorType {
            shape: vec![Dim::Static(1)],
            dtype: DType::F32,
            encoding: None,
        }),
        0,
        vec![0],
    )
}

fn insert_silu_denom_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    x: ValueId,
    out_ty: &Type,
) -> ValueId {
    let t_ty = out_ty.as_tensor().cloned().unwrap_or(TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    });
    let mut z_attrs = IndexMap::new();
    z_attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty.clone(),
            raw_literal: "0.0".to_string(),
        },
    );
    let zero_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        z_attrs,
        out_ty.clone(),
    );
    let mut one_attrs = IndexMap::new();
    one_attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty,
            raw_literal: "1.0".to_string(),
        },
    );
    let one_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        one_attrs,
        out_ty.clone(),
    );
    let neg_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "subtract"),
        &[zero_val, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let exp_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "exponential"),
        &[neg_val],
        IndexMap::new(),
        out_ty.clone(),
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[exp_val, one_val],
        IndexMap::new(),
        out_ty.clone(),
    )
}

fn insert_scalar_const_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    val: f32,
) -> ValueId {
    let s_ty = TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    };
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: s_ty.clone(),
            raw_literal: format!("{val:.1}"),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        Type::Tensor(s_ty),
    )
}

fn half_col_type(out_ty: &Type) -> (Type, i64, i64) {
    if let Some(t) = out_ty.as_tensor()
        && let [Dim::Static(r), Dim::Static(c)] = t.shape.as_slice()
    {
        return (
            Type::Tensor(TensorType {
                shape: vec![Dim::Static(*r), Dim::Static(*c / 2)],
                dtype: t.dtype.clone(),
                encoding: None,
            }),
            *r,
            *c,
        );
    }
    (
        Type::Tensor(TensorType {
            shape: vec![Dim::Static(1), Dim::Static(1)],
            dtype: DType::F32,
            encoding: None,
        }),
        1,
        2,
    )
}

fn insert_slice_2d_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    operand: ValueId,
    starts: [i64; 2],
    limits: [i64; 2],
    out_ty: Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "start_indices".to_string(),
        Attribute::DenseIntArray(starts.to_vec()),
    );
    attrs.insert(
        "limit_indices".to_string(),
        Attribute::DenseIntArray(limits.to_vec()),
    );
    attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "slice"),
        &[operand],
        attrs,
        out_ty,
    )
}

fn insert_zero_tensor_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    ty: &Type,
) -> ValueId {
    let t_ty = ty.as_tensor().cloned().unwrap_or(TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    });
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty,
            raw_literal: "0.0".to_string(),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        ty.clone(),
    )
}

fn insert_op_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    attrs: IndexMap<String, Attribute>,
    out_ty: Type,
) -> ValueId {
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
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

fn insert_splat_tensor_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    ty: &Type,
    val: f32,
) -> ValueId {
    let t_ty = ty.as_tensor().cloned().unwrap_or(TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    });
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty,
            raw_literal: format!("{val}"),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        ty.clone(),
    )
}

fn insert_rmsnorm_core_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    x: ValueId,
    eps: f32,
    out_ty: &Type,
) -> (ValueId, usize) {
    let sq_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let (red_ty, axis, bcast_dims) = reduced_row_type(out_ty);
    let cols = out_ty
        .as_tensor()
        .and_then(|t| match t.shape.last() {
            Some(Dim::Static(c)) => Some(*c as f32),
            _ => None,
        })
        .unwrap_or(1.0);
    let zero_init = insert_scalar_const_before(module, block_id, before_op, 0.0);
    let mut red_attrs = IndexMap::new();
    red_attrs.insert(
        "dimensions".to_string(),
        Attribute::DenseIntArray(vec![axis]),
    );
    red_attrs.insert(
        "reducer".to_string(),
        Attribute::String("stablehlo.add".to_string()),
    );
    let sum_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "reduce"),
        &[sq_val, zero_init],
        red_attrs,
        red_ty.clone(),
    );
    let n_const = insert_splat_tensor_before(module, block_id, before_op, &red_ty, cols);
    let mean_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "divide"),
        &[sum_val, n_const],
        IndexMap::new(),
        red_ty.clone(),
    );
    let eps_const = insert_splat_tensor_before(module, block_id, before_op, &red_ty, eps);
    let var_eps = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[mean_val, eps_const],
        IndexMap::new(),
        red_ty.clone(),
    );
    let inv_rms = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "rsqrt"),
        &[var_eps],
        IndexMap::new(),
        red_ty,
    );
    let mut bcast_attrs = IndexMap::new();
    bcast_attrs.insert(
        "broadcast_dimensions".to_string(),
        Attribute::DenseIntArray(bcast_dims),
    );
    let inv_bcast = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "broadcast_in_dim"),
        &[inv_rms],
        bcast_attrs,
        out_ty.clone(),
    );
    let norm_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x, inv_bcast],
        IndexMap::new(),
        out_ty.clone(),
    );
    (norm_val, 9)
}

fn insert_gelu_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    x: ValueId,
    out_ty: &Type,
) -> ValueId {
    let half = insert_splat_tensor_before(module, block_id, before_op, out_ty, 0.5);
    let one = insert_splat_tensor_before(module, block_id, before_op, out_ty, 1.0);
    let c_k = insert_splat_tensor_before(module, block_id, before_op, out_ty, 0.797_884_6);
    let c_a = insert_splat_tensor_before(module, block_id, before_op, out_ty, 0.044715);
    let x2 = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let x3 = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x2, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let ax3 = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[c_a, x3],
        IndexMap::new(),
        out_ty.clone(),
    );
    let poly = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[x, ax3],
        IndexMap::new(),
        out_ty.clone(),
    );
    let inner = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[c_k, poly],
        IndexMap::new(),
        out_ty.clone(),
    );
    let th = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "tanh"),
        &[inner],
        IndexMap::new(),
        out_ty.clone(),
    );
    let one_p = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[one, th],
        IndexMap::new(),
        out_ty.clone(),
    );
    let hx = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[half, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[hx, one_p],
        IndexMap::new(),
        out_ty.clone(),
    )
}

/// Unified Activation, Gated-MLP & Mixture-of-Experts (MoE) Fusion Pass (`fuse-activations`).
pub struct FuseActivationsPass {
    pub kind: String,
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for FuseActivationsPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
            shared_expert: false,
            top_k: 2,
            capacity_factor: 1.25,
        }
    }
}

impl Pass for FuseActivationsPass {
    fn name(&self) -> &'static str {
        "fuse-activations"
    }

    fn description(&self) -> &'static str {
        "Unified activation, gated-MLP & MoE fusion (SwiGLU, GELU, GeGLU, GatedMLP, MoE)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let mut steps = Vec::new();
        if k == "all" || k == "swiglu" {
            steps.push(FuseSwiGluPass.run(module)?);
        }
        if k == "all" || k == "gelu" || k == "geglu" {
            steps.push(FuseGeluPass.run(module)?);
        }
        if k == "all" || k == "gated_mlp" || k == "mlp" {
            steps.push(FuseGatedMlpPass.run(module)?);
        }
        if k == "all" || k == "moe" || self.shared_expert {
            steps.push(
                FuseMoePass {
                    shared_expert: self.shared_expert,
                    top_k: self.top_k,
                    capacity_factor: self.capacity_factor,
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

/// SwiGLU Fusion Pass (`fuse-swiglu`).
pub struct FuseSwiGluPass;

impl Pass for FuseSwiGluPass {
    fn name(&self) -> &'static str {
        "fuse-swiglu"
    }

    fn description(&self) -> &'static str {
        "Fuses SwiGLU (gate / (1 + exp(-gate)) * up) into stablehlo.composite (fused_swiglu)"
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
                let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };

                let matched = match_silu_val(module, lhs)
                    .map(|g| (g, rhs))
                    .or_else(|| match_silu_val(module, rhs).map(|g| (g, lhs)));
                let Some((gate_val, up_val)) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_swiglu".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_swiglu_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[gate_val, up_val]);
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

/// Matches `silu(x) = divide(x, add(1.0, exponential(subtract(0.0, x))))` and returns `Some(x)`.
pub fn match_silu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: div_op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let div_op = module.ops.get(div_op_id)?;
    let StableHloOp::Divide {
        lhs: gate,
        rhs: denom,
        ..
    } = StableHloOp::from_op(div_op)?
    else {
        return None;
    };

    let ValueDef::OpResult { op: add_op_id, .. } = module.values[denom].def else {
        return None;
    };
    let add_op = module.ops.get(add_op_id)?;
    let StableHloOp::Add {
        lhs: a1, rhs: a2, ..
    } = StableHloOp::from_op(add_op)?
    else {
        return None;
    };

    let exp_val = if FoldConstantsPass::const_splat(module, a2)
        .is_some_and(|v| (v - 1.0).abs() < 1e-6)
    {
        a1
    } else if FoldConstantsPass::const_splat(module, a1).is_some_and(|v| (v - 1.0).abs() < 1e-6) {
        a2
    } else {
        return None;
    };

    let ValueDef::OpResult { op: exp_op_id, .. } = module.values[exp_val].def else {
        return None;
    };
    let exp_op = module.ops.get(exp_op_id)?;
    let StableHloOp::Exp {
        operand: neg_val, ..
    } = StableHloOp::from_op(exp_op)?
    else {
        return None;
    };

    let ValueDef::OpResult { op: sub_op_id, .. } = module.values[neg_val].def else {
        return None;
    };
    let sub_op = module.ops.get(sub_op_id)?;
    let StableHloOp::Subtract {
        lhs: zero_val,
        rhs: sub_rhs,
        ..
    } = StableHloOp::from_op(sub_op)?
    else {
        return None;
    };

    if sub_rhs != gate {
        return None;
    }
    if !FoldConstantsPass::const_splat(module, zero_val).is_some_and(|v| v.abs() < 1e-7) {
        return None;
    }

    Some(gate)
}

/// GELU & GeGLU Fusion Pass (`fuse-gelu`).
pub struct FuseGeluPass;

impl Pass for FuseGeluPass {
    fn name(&self) -> &'static str {
        "fuse-gelu"
    }

    fn description(&self) -> &'static str {
        "Fuses GELU and gated GeGLU (gelu(gate) * up) into stablehlo.composite"
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
                let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };
                if match_gelu_op(module, op).is_some() {
                    continue;
                }
                let matched = match_gelu_val(module, lhs)
                    .map(|g| (g, rhs))
                    .or_else(|| match_gelu_val(module, rhs).map(|g| (g, lhs)));
                let Some((gate_val, up_val)) = matched else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_geglu".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_geglu_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[gate_val, up_val]);
                op_mut.attributes = attrs;
                created += 1;
            }

            if created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
            }

            let ops2 = module.block_ops(block_id);
            let mut standalone_created = 0usize;
            for op_id in ops2 {
                let Some(x_val) = module
                    .ops
                    .get(op_id)
                    .and_then(|op| match_gelu_op(module, op))
                else {
                    continue;
                };

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_gelu".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_gelu_impl".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x_val]);
                op_mut.attributes = attrs;
                standalone_created += 1;
            }

            if standalone_created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
                created += standalone_created;
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

/// Matches a value defined by a Tanh-Approximation GELU subgraph and returns its input `x`.
pub fn match_gelu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match_gelu_op(module, op)
}

pub fn match_gelu_op(module: &Module, op: &castforge_ir::Operation) -> Option<ValueId> {
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    let mut factors = Vec::with_capacity(3);
    collect_mul_factors(module, lhs, &mut factors, 2);
    collect_mul_factors(module, rhs, &mut factors, 2);
    if factors.len() != 3 {
        return None;
    }

    let half_idx = factors.iter().position(|&v| {
        FoldConstantsPass::const_splat(module, v).is_some_and(|c| (c - 0.5).abs() < 1e-4)
    })?;

    for i in 0..3 {
        if i == half_idx {
            continue;
        }
        let other_idx = 3 - half_idx - i;
        let x_cand = factors[i];
        let one_plus_tanh_cand = factors[other_idx];
        if match_one_plus_tanh_of_x(module, one_plus_tanh_cand, x_cand) {
            return Some(x_cand);
        }
    }
    None
}

fn collect_mul_factors(module: &Module, val: ValueId, out: &mut Vec<ValueId>, depth: usize) {
    if depth > 0
        && let ValueDef::OpResult { op: op_id, .. } = module.values[val].def
        && let Some(op) = module.ops.get(op_id)
        && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op)
    {
        collect_mul_factors(module, lhs, out, depth - 1);
        collect_mul_factors(module, rhs, out, depth - 1);
        return;
    }
    out.push(val);
}

fn match_one_plus_tanh_of_x(module: &Module, val_id: ValueId, x_val: ValueId) -> bool {
    let ValueDef::OpResult { op: add_id, .. } = module.values[val_id].def else {
        return false;
    };
    let Some(add_op) = module.ops.get(add_id) else {
        return false;
    };
    let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(add_op) else {
        return false;
    };
    let tanh_val = if FoldConstantsPass::const_splat(module, lhs)
        .is_some_and(|c| (c - 1.0).abs() < 1e-4)
    {
        rhs
    } else if FoldConstantsPass::const_splat(module, rhs).is_some_and(|c| (c - 1.0).abs() < 1e-4) {
        lhs
    } else {
        return false;
    };

    let ValueDef::OpResult { op: tanh_id, .. } = module.values[tanh_val].def else {
        return false;
    };
    let Some(tanh_op) = module.ops.get(tanh_id) else {
        return false;
    };
    let Some(StableHloOp::Tanh { operand: inner, .. }) = StableHloOp::from_op(tanh_op) else {
        return false;
    };

    let ValueDef::OpResult { op: mul_id, .. } = module.values[inner].def else {
        return false;
    };
    let Some(mul_op) = module.ops.get(mul_id) else {
        return false;
    };
    let Some(StableHloOp::Multiply {
        lhs: m1, rhs: m2, ..
    }) = StableHloOp::from_op(mul_op)
    else {
        return false;
    };
    let poly_val = if FoldConstantsPass::const_splat(module, m1)
        .is_some_and(|c| (c - 0.79788456).abs() < 1e-2)
    {
        m2
    } else if FoldConstantsPass::const_splat(module, m2)
        .is_some_and(|c| (c - 0.79788456).abs() < 1e-2)
    {
        m1
    } else {
        return false;
    };

    let ValueDef::OpResult { op: p_add_id, .. } = module.values[poly_val].def else {
        return false;
    };
    let Some(p_add_op) = module.ops.get(p_add_id) else {
        return false;
    };
    let Some(StableHloOp::Add {
        lhs: p1, rhs: p2, ..
    }) = StableHloOp::from_op(p_add_op)
    else {
        return false;
    };
    let cubic_val = if p1 == x_val {
        p2
    } else if p2 == x_val {
        p1
    } else {
        return false;
    };

    let mut cubic_factors = Vec::new();
    collect_mul_factors(module, cubic_val, &mut cubic_factors, 3);
    if cubic_factors.len() != 4 {
        return false;
    }
    let x_count = cubic_factors.iter().filter(|&&v| v == x_val).count();
    let coeff_ok = cubic_factors.iter().any(|&v| {
        FoldConstantsPass::const_splat(module, v).is_some_and(|c| (c - 0.044715).abs() < 1e-3)
    });
    x_count == 3 && coeff_ok
}

/// Gated MLP Macro-Fusion Pass (`fuse-gated-mlp`).
pub struct FuseGatedMlpPass;

impl Pass for FuseGatedMlpPass {
    fn name(&self) -> &'static str {
        "fuse-gated-mlp"
    }

    fn description(&self) -> &'static str {
        "Fuses dual gate/up dot_general + slices + SwiGLU/GeGLU into fused_gated_mlp"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let h_res = FuseHorizontalDotPass.run(module)?;
        let sw_res = FuseSwiGluPass.run(module)?;
        let ge_res = FuseGeluPass.run(module)?;

        let mut created = h_res.ops_created + sw_res.ops_created + ge_res.ops_created;
        let mut removed = h_res.ops_removed + sw_res.ops_removed + ge_res.ops_removed;
        let mut fused_count = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                let act_kind = if name == "castforge.fused_swiglu" {
                    "swiglu"
                } else if name == "castforge.fused_geglu" {
                    "geglu"
                } else {
                    continue;
                };
                if inputs.len() < 2 {
                    continue;
                }

                let Some((dot_val_0, start_col_0)) = match_2d_col_slice(module, inputs[0]) else {
                    continue;
                };
                let Some((dot_val_1, start_col_1)) = match_2d_col_slice(module, inputs[1]) else {
                    continue;
                };
                if dot_val_0 != dot_val_1 || start_col_0 != 0 || start_col_1 <= 0 {
                    continue;
                }

                let ValueDef::OpResult { op: dot_op_id, .. } = module.values[dot_val_0].def else {
                    continue;
                };
                let Some(dot_op) = module.ops.get(dot_op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    lhs: x_val,
                    rhs: w_gate_up,
                    dimension_numbers: dims,
                    ..
                }) = StableHloOp::from_op(dot_op)
                else {
                    continue;
                };
                let packed = dims.rhs_contracting_dimensions == [1];

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_gated_mlp".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_gated_mlp_impl".to_string()),
                );
                attrs.insert(
                    "activation".to_string(),
                    Attribute::String(act_kind.to_string()),
                );
                if packed {
                    attrs.insert("packed_rhs".to_string(), Attribute::Bool(true));
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x_val, w_gate_up]);
                op_mut.attributes = attrs;
                fused_count += 1;
            }
        }

        if fused_count > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
            created += fused_count;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || fused_count > 0,
            ops_removed: removed,
            ops_modified: fused_count,
            ops_created: created,
        })
    }
}

fn match_2d_col_slice(module: &Module, val_id: ValueId) -> Option<(ValueId, i64)> {
    let ValueDef::OpResult { op: s_id, .. } = module.values[val_id].def else {
        return None;
    };
    let s_op = module.ops.get(s_id)?;
    let StableHloOp::Slice {
        operand,
        start_indices,
        ..
    } = StableHloOp::from_op(s_op)?
    else {
        return None;
    };
    if start_indices.len() == 2 && start_indices[0] == 0 {
        Some((operand, start_indices[1]))
    } else {
        None
    }
}

/// Mixture-of-Experts (MoE) Routing, Shared-Expert & Grouped-GEMM Fusion Pass (`fuse-moe`).
pub struct FuseMoePass {
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for FuseMoePass {
    fn default() -> Self {
        Self {
            shared_expert: false,
            top_k: 2,
            capacity_factor: 1.25,
        }
    }
}

impl Pass for FuseMoePass {
    fn name(&self) -> &'static str {
        "fuse-moe"
    }

    fn description(&self) -> &'static str {
        "Fuses routed + shared Mixture-of-Experts (MoE) GEMMs and gate combine into fused_moe"
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
                let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op) else {
                    continue;
                };
                let Some((x0, w0, g0)) = match_gated_expert(module, lhs) else {
                    continue;
                };
                let Some((x1, w1, g1)) = match_gated_expert(module, rhs) else {
                    continue;
                };
                if x0 != x1 {
                    continue;
                }

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_moe".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_moe_impl".to_string()),
                );
                attrs.insert("num_experts".to_string(), Attribute::Int(2));
                attrs.insert(
                    "castforge.moe_top_k".to_string(),
                    Attribute::Int(self.top_k.max(1) as i64),
                );
                attrs.insert(
                    "castforge.moe_capacity_factor".to_string(),
                    Attribute::Float(self.capacity_factor.max(0.5) as f64),
                );
                if self.shared_expert {
                    attrs.insert(
                        "castforge.moe_shared_expert".to_string(),
                        Attribute::Bool(true),
                    );
                }

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[x0, w0, w1, g0, g1]);
                op_mut.attributes = attrs;
                created += 1;
            }
        }

        if created > 0 {
            let dce = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: created,
            ops_created: created,
        })
    }
}

fn match_gated_expert(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
    let ValueDef::OpResult { op: mul_id, .. } = module.values[val_id].def else {
        return None;
    };
    let mul_op = module.ops.get(mul_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(mul_op)? else {
        return None;
    };
    match_dot_2d(module, lhs)
        .map(|(x, w)| (x, w, rhs))
        .or_else(|| match_dot_2d(module, rhs).map(|(x, w)| (x, w, lhs)))
}

fn match_dot_2d(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: dot_id, .. } = module.values[val_id].def else {
        return None;
    };
    let dot_op = module.ops.get(dot_id)?;
    let StableHloOp::DotGeneral { lhs, rhs, .. } = StableHloOp::from_op(dot_op)? else {
        return None;
    };
    Some((lhs, rhs))
}
