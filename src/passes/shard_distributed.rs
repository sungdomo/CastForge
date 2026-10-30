//! Unified Distributed Parallelism, Collectives & Stream Overlap (`shard-distributed`).
//!
//! Consolidates:
//! - `mode=tp|spmd|all` (`ShardSpmdPass` — Megatron-style 1D Tensor Parallelism)
//! - `mode=ring_cp|ring|all` (`ShardRingAttentionPass` — RingAttention Context Parallelism)
//! - `mode=pp|pipeline` (`ShardPipelineParallelPass` — 1F1B / Zero-Bubble Pipeline Parallelism)
//! - `mode=coalesce|hoist|all` (`HoistAllReducePass` — collective `all_reduce` coalescing)
//! - `mode=streams|async` (`ScheduleStreamsPass` — multi-stream compute/copy overlap)

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId, format_dense_f32_literal,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;

/// Unified Distributed Parallelism, Collectives, Stream Scheduling & Autodiff Pass
/// (`shard-distributed`).
pub struct ShardDistributedPass {
    pub mode: String,
    pub devices: usize,
    pub ring_size: usize,
    pub coalesce: bool,
    pub stages: usize,
    pub microbatches: usize,
    pub schedule: String,
    pub streams: usize,
    pub learning_rate: f32,
}

impl Default for ShardDistributedPass {
    fn default() -> Self {
        Self {
            mode: "tp".to_string(),
            devices: 2,
            ring_size: 4,
            coalesce: false,
            stages: 2,
            microbatches: 4,
            schedule: "zero_bubble_1f1b".to_string(),
            streams: 2,
            learning_rate: 0.0,
        }
    }
}

impl Pass for ShardDistributedPass {
    fn name(&self) -> &'static str {
        "shard-distributed"
    }

    fn description(&self) -> &'static str {
        "Unified distributed parallelism (TP, Ring-CP, PP, FSDP, EP, streams, & reverse-mode VJP)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
        if matches!(m.as_str(), "autodiff" | "vjp" | "sgd") || self.learning_rate > 0.0 {
            steps.push(
                AutodiffPass {
                    learning_rate: self.learning_rate,
                }
                .run(module)?,
            );
        }
        if m == "tp" || m == "spmd" || m == "all" {
            steps.push(
                ShardSpmdPass {
                    num_devices: self.devices,
                }
                .run(module)?,
            );
        }
        if m == "ring_cp" || m == "ring" || m == "all" {
            steps.push(
                ShardRingAttentionPass {
                    ring_size: self.ring_size,
                }
                .run(module)?,
            );
        }
        if m == "pp" || m == "pipeline" {
            steps.push(
                ShardPipelineParallelPass {
                    stages: self.stages,
                    microbatches: self.microbatches,
                    schedule: self.schedule.clone(),
                }
                .run(module)?,
            );
        }
        if m == "fsdp" || m == "zero3" || m == "zero" {
            steps.push(run_fsdp_sharding(module, self.devices)?);
        }
        if m == "ep" || m == "expert" || m == "moe" {
            steps.push(run_expert_parallelism(module, self.devices)?);
        }
        if m == "streams" || m == "stream" || m == "async" {
            steps.push(
                ScheduleStreamsPass {
                    num_streams: self.streams,
                }
                .run(module)?,
            );
        }
        if m == "coalesce" || m == "hoist" || m == "all" || self.coalesce {
            steps.push(HoistAllReducePass.run(module)?);
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

/// Megatron-Style 1D SPMD Tensor Parallelism Pass (`shard-spmd`).
pub struct ShardSpmdPass {
    pub num_devices: usize,
}

impl Default for ShardSpmdPass {
    fn default() -> Self {
        Self { num_devices: 2 }
    }
}

impl Pass for ShardSpmdPass {
    fn name(&self) -> &'static str {
        "shard-spmd"
    }

    fn description(&self) -> &'static str {
        "Partitions GEMM ops for Megatron-style SPMD tensor parallelism and inserts all_reduce"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let devices = self.num_devices.max(2) as i64;
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let gemm_ops: Vec<OpId> = ops
                .into_iter()
                .filter(|&id| {
                    module.ops.get(id).is_some_and(|op| {
                        !op.attributes.contains_key("spmd_sharding")
                            && (op.is_op("stablehlo", "dot_general")
                                || (op.is_op("stablehlo", "composite")
                                    && op
                                        .attributes
                                        .get("name")
                                        .and_then(|a| a.as_str())
                                        .is_some_and(|n| {
                                            n == "castforge.fused_matmul_epilogue"
                                                || n == "castforge.fused_gated_mlp"
                                                || n == "castforge.fused_qmatmul"
                                        })))
                    })
                })
                .collect();

            let total_gemms = gemm_ops.len();
            for (idx, op_id) in gemm_ops.into_iter().enumerate() {
                let is_row_parallel = (idx % 2 == 1) || (total_gemms == 1);
                let role = if is_row_parallel {
                    "row_parallel"
                } else {
                    "column_parallel"
                };

                let Some(out_val) = module.ops[op_id].results.first().copied() else {
                    continue;
                };
                module.ops[op_id]
                    .attributes
                    .insert("spmd_sharding".to_string(), Attribute::String(role.into()));
                module.ops[op_id]
                    .attributes
                    .insert("num_devices".to_string(), Attribute::Int(devices));
                modified += 1;

                if is_row_parallel {
                    insert_all_reduce_after(module, block_id, op_id, out_val, devices);
                    created += 1;
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

fn insert_all_reduce_after(
    module: &mut Module,
    block_id: BlockId,
    after_op: OpId,
    src_val: ValueId,
    devices: i64,
) {
    let out_ty = module.values[src_val].ty.clone();
    let mut attrs = IndexMap::new();
    attrs.insert(
        "name".to_string(),
        Attribute::String("castforge.all_reduce".to_string()),
    );
    attrs.insert(
        "decomposition".to_string(),
        Attribute::SymbolRef("all_reduce_impl".to_string()),
    );
    attrs.insert(
        "reduction".to_string(),
        Attribute::String("sum".to_string()),
    );
    attrs.insert("num_devices".to_string(), Attribute::Int(devices));

    let ar_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "composite"),
        operands: SmallVec::from_slice(&[src_val]),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let ar_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: ar_op_id,
            index: 0,
        },
    });
    module.ops[ar_op_id].results.push(ar_val);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == after_op) {
        block_ops.insert(pos + 1, ar_op_id);
    } else {
        block_ops.push(ar_op_id);
    }

    let all_ops: Vec<OpId> = module.blocks[block_id].ops.clone();
    for op_id in all_ops {
        if op_id != after_op && op_id != ar_op_id {
            for opnd in &mut module.ops[op_id].operands {
                if *opnd == src_val {
                    *opnd = ar_val;
                }
            }
        }
    }
}

/// RingAttention / Context Parallelism Pass (`shard-ring-attention`).
pub struct ShardRingAttentionPass {
    pub ring_size: usize,
}

impl Default for ShardRingAttentionPass {
    fn default() -> Self {
        Self { ring_size: 4 }
    }
}

impl Pass for ShardRingAttentionPass {
    fn name(&self) -> &'static str {
        "shard-ring-attention"
    }

    fn description(&self) -> &'static str {
        "Shards sequence context across a device ring with overlapped P2P KV send/recv"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let is_attn = op.is_op("stablehlo", "composite")
                && op
                    .attributes
                    .get("name")
                    .and_then(|a| a.as_str())
                    .is_some_and(|n| n == "castforge.fused_attention");
            if is_attn {
                op.attributes.insert(
                    "castforge.ring_attention".to_string(),
                    Attribute::Bool(true),
                );
                op.attributes.insert(
                    "castforge.ring_size".to_string(),
                    Attribute::Int(self.ring_size.max(2) as i64),
                );
                op.attributes.insert(
                    "castforge.p2p_overlap".to_string(),
                    Attribute::String("send_recv_kv".to_string()),
                );
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

/// 1F1B / Zero-Bubble Pipeline Parallelism Pass (`shard-pipeline-parallel`).
pub struct ShardPipelineParallelPass {
    pub stages: usize,
    pub microbatches: usize,
    pub schedule: String,
}

impl Default for ShardPipelineParallelPass {
    fn default() -> Self {
        Self {
            stages: 2,
            microbatches: 4,
            schedule: "zero_bubble_1f1b".to_string(),
        }
    }
}

impl Pass for ShardPipelineParallelPass {
    fn name(&self) -> &'static str {
        "shard-pipeline-parallel"
    }

    fn description(&self) -> &'static str {
        "Partitions compute ops across pipeline-parallel stages with 1F1B / Zero-Bubble scheduling"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let num_stages = self.stages.max(2);
        let mb = self.microbatches.max(1) as i64;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            let compute_ops: Vec<_> = module
                .block_ops(block_id)
                .into_iter()
                .filter(|&op_id| {
                    module.ops.get(op_id).is_some_and(|op| {
                        op.name.dialect == "stablehlo" && op.name.name != "constant"
                    })
                })
                .collect();
            let total = compute_ops.len().max(1);
            for (idx, op_id) in compute_ops.into_iter().enumerate() {
                let stage = (idx * num_stages) / total;
                if let Some(op) = module.ops.get_mut(op_id) {
                    op.attributes.insert(
                        "castforge.pp_stage".to_string(),
                        Attribute::Int(stage as i64),
                    );
                    op.attributes.insert(
                        "castforge.pp_schedule".to_string(),
                        Attribute::String(self.schedule.clone()),
                    );
                    op.attributes
                        .insert("castforge.pp_microbatches".to_string(), Attribute::Int(mb));
                    modified += 1;
                }
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

/// Collective All-Reduce Coalescer & Async Overlap Pass (`hoist-all-reduce`).
#[derive(Debug, Clone, Default)]
pub struct HoistAllReducePass;

impl Pass for HoistAllReducePass {
    fn name(&self) -> &'static str {
        "hoist-all-reduce"
    }

    fn description(&self) -> &'static str {
        "Coalesces linear all_reduce sums, eliminates redundant collectives, and enables overlap"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };

                if is_all_reduce(&op) {
                    if op.attributes.get("async_overlap") != Some(&Attribute::Bool(true)) {
                        module.ops[op_id]
                            .attributes
                            .insert("async_overlap".to_string(), Attribute::Bool(true));
                        modified += 1;
                    }
                    if let Some(&src) = op.operands.first()
                        && let ValueDef::OpResult { op: prod_id, .. } = module.values[src].def
                        && module.ops.get(prod_id).is_some_and(is_all_reduce)
                    {
                        let res = op.results[0];
                        module.replace_all_uses(res, src);
                        module.remove_op(block_id, op_id);
                        removed += 1;
                        continue;
                    }
                }

                if op.is_op("stablehlo", "add") && op.operands.len() == 2 && op.results.len() == 1 {
                    let (lhs, rhs) = (op.operands[0], op.operands[1]);
                    let (
                        ValueDef::OpResult { op: l_op_id, .. },
                        ValueDef::OpResult { op: r_op_id, .. },
                    ) = (module.values[lhs].def, module.values[rhs].def)
                    else {
                        continue;
                    };
                    if l_op_id == r_op_id {
                        continue;
                    }
                    let (Some(l_op), Some(r_op)) = (
                        module.ops.get(l_op_id).cloned(),
                        module.ops.get(r_op_id).cloned(),
                    ) else {
                        continue;
                    };
                    if !is_all_reduce(&l_op)
                        || !is_all_reduce(&r_op)
                        || l_op.operands.is_empty()
                        || r_op.operands.is_empty()
                    {
                        continue;
                    }
                    let raw_a = l_op.operands[0];
                    let raw_b = r_op.operands[0];
                    let add_res = op.results[0];

                    module.ops[op_id].operands[0] = raw_a;
                    module.ops[op_id].operands[1] = raw_b;

                    let mut ar_attrs = l_op.attributes.clone();
                    ar_attrs.insert("async_overlap".to_string(), Attribute::Bool(true));
                    ar_attrs.insert("coalesced_collectives".to_string(), Attribute::Int(2));
                    let out_ty = module.values[add_res].ty.clone();

                    let ar_id = module.add_operation(
                        block_id,
                        l_op.name.clone(),
                        SmallVec::from_slice(&[add_res]),
                        vec![(Some("ar_coalesced".to_string()), out_ty)],
                        ar_attrs,
                        Vec::new(),
                        l_op.custom_syntax,
                        op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    let ar_res = module.ops[ar_id].results[0];

                    for (other_id, other_op) in module.ops.iter_mut() {
                        if other_id == op_id || other_id == ar_id {
                            continue;
                        }
                        for opnd in &mut other_op.operands {
                            if *opnd == add_res {
                                *opnd = ar_res;
                            }
                        }
                    }
                    if let Some(pos) = module.blocks[block_id]
                        .ops
                        .iter()
                        .position(|&id| id == op_id)
                    {
                        module.blocks[block_id].ops.insert(pos + 1, ar_id);
                    }
                    modified += 1;
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn is_all_reduce(op: &Operation) -> bool {
    op.is_op("stablehlo", "all_reduce")
        || (op.is_op("stablehlo", "composite")
            && op.attributes.get("name").and_then(|a| a.as_str()) == Some("castforge.all_reduce"))
}

/// Multi-Stream Async Compute / Copy Overlap Pass (`schedule-streams`).
pub struct ScheduleStreamsPass {
    pub num_streams: usize,
}

impl Default for ScheduleStreamsPass {
    fn default() -> Self {
        Self { num_streams: 2 }
    }
}

impl Pass for ScheduleStreamsPass {
    fn name(&self) -> &'static str {
        "schedule-streams"
    }

    fn description(&self) -> &'static str {
        "Assigns independent branches to concurrent execution streams with stream_wait barriers"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let streams = self.num_streams.max(2);
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.operands.len() < 2
                    || op.is_op("func", "return")
                    || op.attributes.contains_key("stream_synchronized")
                {
                    continue;
                }
                let lhs = op.operands[0];
                let rhs = op.operands[1];
                if lhs == rhs {
                    continue;
                }
                let (
                    ValueDef::OpResult { op: lhs_op_id, .. },
                    ValueDef::OpResult { op: rhs_op_id, .. },
                ) = (module.values[lhs].def, module.values[rhs].def)
                else {
                    continue;
                };
                if lhs_op_id == rhs_op_id {
                    continue;
                }
                let Some(lhs_op) = module.ops.get(lhs_op_id) else {
                    continue;
                };
                let Some(rhs_op) = module.ops.get(rhs_op_id) else {
                    continue;
                };
                if lhs_op.is_op("stablehlo", "constant") || rhs_op.is_op("stablehlo", "constant") {
                    continue;
                }
                if lhs_op.operands.contains(&rhs) || rhs_op.operands.contains(&lhs) {
                    continue;
                }

                let rhs_stream = ((modified + 1) % streams).max(1) as i64;
                module.ops[lhs_op_id]
                    .attributes
                    .insert("stream_id".to_string(), Attribute::Int(0));
                module.ops[rhs_op_id]
                    .attributes
                    .insert("stream_id".to_string(), Attribute::Int(rhs_stream));

                let synced_rhs =
                    insert_stream_wait_before(module, block_id, op_id, rhs, rhs_stream);
                let op_mut = &mut module.ops[op_id];
                op_mut.operands[1] = synced_rhs;
                op_mut
                    .attributes
                    .insert("stream_id".to_string(), Attribute::Int(0));
                op_mut
                    .attributes
                    .insert("stream_synchronized".to_string(), Attribute::Bool(true));
                created += 1;
                modified += 3;
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

fn insert_stream_wait_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    val: ValueId,
    wait_stream: i64,
) -> ValueId {
    let out_ty = module.values[val].ty.clone();
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "composite"),
        operands: SmallVec::from_slice(&[val]),
        results: SmallVec::new(),
        attributes: {
            let mut m = IndexMap::new();
            m.insert(
                "name".to_string(),
                Attribute::String("castforge.stream_wait".to_string()),
            );
            m.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("stream_wait_impl".to_string()),
            );
            m.insert("wait_stream".to_string(), Attribute::Int(wait_stream));
            m.insert("target_stream".to_string(), Attribute::Int(0));
            m
        },
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let res_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(res_val);
    let block_ops = &mut module.blocks[block_id].ops;
    let pos = block_ops
        .iter()
        .position(|&x| x == before_op)
        .unwrap_or(block_ops.len());
    block_ops.insert(pos, new_op_id);
    res_val
}

fn run_fsdp_sharding(module: &mut Module, num_devices: usize) -> Result<PassResult> {
    let devices = num_devices.max(2) as i64;
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();
    for block_id in blocks {
        for op_id in module.block_ops(block_id) {
            let Some(op) = module.ops.get_mut(op_id) else {
                continue;
            };
            let is_gemm = op.is_op("stablehlo", "dot_general")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| {
                            n == "castforge.fused_matmul_epilogue"
                                || n == "castforge.fused_gated_mlp"
                                || n == "castforge.fused_qmatmul"
                        }));
            if is_gemm {
                op.attributes
                    .insert("castforge.fsdp_sharded".to_string(), Attribute::Bool(true));
                op.attributes.insert(
                    "castforge.fsdp_devices".to_string(),
                    Attribute::Int(devices),
                );
                op.attributes.insert(
                    "castforge.fsdp_prefetch".to_string(),
                    Attribute::String("all_gather".to_string()),
                );
                op.attributes.insert(
                    "castforge.grad_sync".to_string(),
                    Attribute::String("reduce_scatter".to_string()),
                );
                modified += 1;
            }
        }
    }
    Ok(PassResult {
        pass_name: "shard-distributed".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

fn run_expert_parallelism(module: &mut Module, num_devices: usize) -> Result<PassResult> {
    let devices = num_devices.max(2) as i64;
    let mut modified = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();
    for block_id in blocks {
        for op_id in module.block_ops(block_id) {
            let Some(op) = module.ops.get_mut(op_id) else {
                continue;
            };
            let is_moe_or_gemm = op.is_op("stablehlo", "dot_general")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| {
                            n == "castforge.fused_moe"
                                || n == "castforge.fused_gated_mlp"
                                || n == "castforge.fused_matmul_epilogue"
                        }));
            if is_moe_or_gemm {
                op.attributes
                    .insert("castforge.ep_sharded".to_string(), Attribute::Bool(true));
                op.attributes
                    .insert("castforge.ep_devices".to_string(), Attribute::Int(devices));
                op.attributes.insert(
                    "castforge.ep_dispatch".to_string(),
                    Attribute::String("all_to_all".to_string()),
                );
                op.attributes.insert(
                    "castforge.ep_combine".to_string(),
                    Attribute::String("all_to_all".to_string()),
                );
                modified += 1;
            }
        }
    }
    Ok(PassResult {
        pass_name: "shard-distributed".to_string(),
        changed: modified > 0,
        ops_removed: 0,
        ops_modified: modified,
        ops_created: 0,
    })
}

/// Reverse-Mode Automatic Differentiation Pass (`autodiff`).
pub struct AutodiffPass {
    pub learning_rate: f32,
}

impl Default for AutodiffPass {
    fn default() -> Self {
        Self { learning_rate: 0.0 }
    }
}

impl Pass for AutodiffPass {
    fn name(&self) -> &'static str {
        "autodiff"
    }

    fn description(&self) -> &'static str {
        "Synthesizes reverse-mode VJP gradient ops in StableHLO (optional SGD step via lr=...)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;

        for fn_id in module.functions() {
            let Some(reg_id) = module.ops[fn_id].regions.first().copied() else {
                continue;
            };
            let Some(block_id) = module.regions[reg_id].blocks.first().copied() else {
                continue;
            };
            let forward_ops = module.block_ops(block_id);
            let Some(&ret_op_id) = forward_ops
                .iter()
                .find(|&&id| module.ops[id].is_op("func", "return"))
            else {
                continue;
            };
            let Some(&ret_val) = module.ops[ret_op_id].operands.first() else {
                continue;
            };
            let ret_ty = module.values[ret_val].ty.clone();

            let seed_grad = insert_splat_const_before(module, block_id, ret_op_id, &ret_ty, 1.0);
            created += 1;

            let mut grads: HashMap<ValueId, ValueId> = HashMap::new();
            grads.insert(ret_val, seed_grad);

            for &op_id in forward_ops.iter().rev() {
                if op_id == ret_op_id {
                    continue;
                }
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(&out_val) = op.results.first() else {
                    continue;
                };
                let Some(&dy) = grads.get(&out_val) else {
                    continue;
                };
                let out_ty = module.values[out_val].ty.clone();

                let Some(hlo) = StableHloOp::from_op(&op) else {
                    continue;
                };

                match hlo {
                    StableHloOp::Add { lhs, rhs, .. } => {
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dy,
                            &mut created,
                        );
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            rhs,
                            dy,
                            &mut created,
                        );
                    }
                    StableHloOp::Subtract { lhs, rhs, .. } => {
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dy,
                            &mut created,
                        );
                        let zero =
                            insert_splat_const_before(module, block_id, ret_op_id, &out_ty, 0.0);
                        let neg_dy = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "subtract"),
                            &[zero, dy],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 2;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            rhs,
                            neg_dy,
                            &mut created,
                        );
                    }
                    StableHloOp::Multiply { lhs, rhs, .. } => {
                        let dlhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, rhs],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let drhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, lhs],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 2;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dlhs,
                            &mut created,
                        );
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            rhs,
                            drhs,
                            &mut created,
                        );
                    }
                    StableHloOp::Divide { lhs, rhs, .. } => {
                        let dlhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "divide"),
                            &[dy, rhs],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dlhs,
                            &mut created,
                        );
                    }
                    StableHloOp::Tanh { operand, .. } => {
                        let one =
                            insert_splat_const_before(module, block_id, ret_op_id, &out_ty, 1.0);
                        let y_sq = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[out_val, out_val],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let dtanh = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "subtract"),
                            &[one, y_sq],
                            IndexMap::new(),
                            out_ty.clone(),
                        );
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, dtanh],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 4;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::Exp { operand, .. } => {
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "multiply"),
                            &[dy, out_val],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::Maximum { lhs, rhs, .. } => {
                        let pred_dims = module.values[out_val]
                            .ty
                            .as_tensor()
                            .map(|t| t.shape.clone())
                            .unwrap_or_default();
                        let pred_ty = Type::Tensor(TensorType {
                            shape: pred_dims,
                            dtype: DType::I1,
                            encoding: None,
                        });
                        let mut cmp_attrs = IndexMap::new();
                        cmp_attrs.insert(
                            "comparison_direction".to_string(),
                            Attribute::String("GT".to_string()),
                        );
                        let mask = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "compare"),
                            &[lhs, rhs],
                            cmp_attrs,
                            pred_ty,
                        );
                        let zero =
                            insert_splat_const_before(module, block_id, ret_op_id, &out_ty, 0.0);
                        let dlhs = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "select"),
                            &[mask, dy, zero],
                            IndexMap::new(),
                            out_ty,
                        );
                        created += 3;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            lhs,
                            dlhs,
                            &mut created,
                        );
                    }
                    StableHloOp::Reshape { operand, .. } => {
                        let in_ty = module.values[operand].ty.clone();
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "reshape"),
                            &[dy],
                            IndexMap::new(),
                            in_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::Transpose {
                        operand,
                        permutation,
                        ..
                    } => {
                        let in_ty = module.values[operand].ty.clone();
                        let mut inv_perm = vec![0i64; permutation.len()];
                        for (dst, &src) in permutation.iter().enumerate() {
                            if (src as usize) < inv_perm.len() {
                                inv_perm[src as usize] = dst as i64;
                            }
                        }
                        let mut t_attrs = IndexMap::new();
                        t_attrs.insert(
                            "permutation".to_string(),
                            Attribute::DenseIntArray(inv_perm),
                        );
                        let dx = insert_vjp_op(
                            module,
                            block_id,
                            ret_op_id,
                            OpName::new("stablehlo", "transpose"),
                            &[dy],
                            t_attrs,
                            in_ty,
                        );
                        created += 1;
                        accumulate_grad(
                            module,
                            block_id,
                            ret_op_id,
                            &mut grads,
                            operand,
                            dx,
                            &mut created,
                        );
                    }
                    StableHloOp::DotGeneral { lhs, rhs, .. } => {
                        let lhs_ty = module.values[lhs].ty.clone();
                        let rhs_ty = module.values[rhs].ty.clone();
                        let lhs_dims = tensor_dims(&lhs_ty);
                        let rhs_dims = tensor_dims(&rhs_ty);
                        if lhs_dims.len() == 2 && rhs_dims.len() == 2 {
                            let bt_ty = Type::Tensor(TensorType {
                                shape: vec![Dim::Static(rhs_dims[1]), Dim::Static(rhs_dims[0])],
                                dtype: DType::F32,
                                encoding: None,
                            });
                            let mut t_attrs = IndexMap::new();
                            t_attrs.insert(
                                "permutation".to_string(),
                                Attribute::DenseIntArray(vec![1, 0]),
                            );
                            let bt = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "transpose"),
                                &[rhs],
                                t_attrs.clone(),
                                bt_ty,
                            );
                            let dlhs = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "dot_general"),
                                &[dy, bt],
                                dot_2d_attrs(),
                                lhs_ty,
                            );

                            let at_ty = Type::Tensor(TensorType {
                                shape: vec![Dim::Static(lhs_dims[1]), Dim::Static(lhs_dims[0])],
                                dtype: DType::F32,
                                encoding: None,
                            });
                            let at = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "transpose"),
                                &[lhs],
                                t_attrs,
                                at_ty,
                            );
                            let drhs = insert_vjp_op(
                                module,
                                block_id,
                                ret_op_id,
                                OpName::new("stablehlo", "dot_general"),
                                &[at, dy],
                                dot_2d_attrs(),
                                rhs_ty,
                            );
                            created += 4;
                            accumulate_grad(
                                module,
                                block_id,
                                ret_op_id,
                                &mut grads,
                                lhs,
                                dlhs,
                                &mut created,
                            );
                            accumulate_grad(
                                module,
                                block_id,
                                ret_op_id,
                                &mut grads,
                                rhs,
                                drhs,
                                &mut created,
                            );
                        }
                    }
                    _ => {}
                }
            }

            let block_args = module.blocks[block_id].arguments.clone();
            let target_arg = block_args
                .iter()
                .rev()
                .copied()
                .find(|&arg| grads.contains_key(&arg) && module.values[arg].ty == ret_ty);

            if let Some(arg_val) = target_arg
                && let Some(&grad_val) = grads.get(&arg_val)
            {
                let new_ret = if self.learning_rate > 0.0 {
                    let lr_const = insert_splat_const_before(
                        module,
                        block_id,
                        ret_op_id,
                        &ret_ty,
                        self.learning_rate,
                    );
                    let step = insert_vjp_op(
                        module,
                        block_id,
                        ret_op_id,
                        OpName::new("stablehlo", "multiply"),
                        &[grad_val, lr_const],
                        IndexMap::new(),
                        ret_ty.clone(),
                    );
                    let updated = insert_vjp_op(
                        module,
                        block_id,
                        ret_op_id,
                        OpName::new("stablehlo", "subtract"),
                        &[arg_val, step],
                        IndexMap::new(),
                        ret_ty,
                    );
                    created += 3;
                    updated
                } else {
                    grad_val
                };
                module.ops[ret_op_id].operands[0] = new_ret;
                modified += 1;
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

fn dot_2d_attrs() -> IndexMap<String, Attribute> {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "lhs_contracting_dimensions".to_string(),
        Attribute::DenseIntArray(vec![1]),
    );
    attrs.insert(
        "rhs_contracting_dimensions".to_string(),
        Attribute::DenseIntArray(vec![0]),
    );
    attrs
}

fn tensor_dims(ty: &Type) -> Vec<i64> {
    ty.as_tensor()
        .map(|t| {
            t.shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n,
                    _ => 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn accumulate_grad(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    grads: &mut HashMap<ValueId, ValueId>,
    target: ValueId,
    contrib: ValueId,
    created: &mut usize,
) {
    if module.values[target].ty != module.values[contrib].ty {
        return;
    }
    if let Some(&existing) = grads.get(&target) {
        let ty = module.values[target].ty.clone();
        let sum = insert_vjp_op(
            module,
            block_id,
            before_op,
            OpName::new("stablehlo", "add"),
            &[existing, contrib],
            IndexMap::new(),
            ty,
        );
        *created += 1;
        grads.insert(target, sum);
    } else {
        grads.insert(target, contrib);
    }
}

fn insert_splat_const_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    ty: &Type,
    val: f32,
) -> ValueId {
    let dims = tensor_dims(ty);
    let dim_objs: Vec<Dim> = dims.iter().map(|&d| Dim::Static(d)).collect();
    let num_elems: usize = dims.iter().map(|&d| d.max(1) as usize).product::<usize>();
    let lit = format_dense_f32_literal(&vec![val; num_elems.max(1)], &dim_objs);
    let tensor_ty = ty
        .as_tensor()
        .cloned()
        .unwrap_or_else(|| TensorType::new(dim_objs, DType::F32));
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: tensor_ty,
            raw_literal: lit,
        },
    );
    insert_vjp_op(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        ty.clone(),
    )
}

fn insert_vjp_op(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    mut attrs: IndexMap<String, Attribute>,
    out_ty: Type,
) -> ValueId {
    if name.name != "constant" {
        attrs.insert(
            "autodiff_role".to_string(),
            Attribute::String("vjp_grad".to_string()),
        );
    }
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
