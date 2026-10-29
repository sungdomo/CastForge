use crate::{HoistAllReducePass, Pass, PassResult, ShardSpmdPass};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Module};

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

/// Unified Distributed Parallelism & Collective Optimization Pass (`shard-distributed`).
///
/// Consolidates `shard-spmd` (`mode=tp`), `shard-ring-attention` (`mode=ring_cp`), and
/// `hoist-all-reduce` (`mode=coalesce|all`).
pub struct ShardDistributedPass {
    pub mode: String,
    pub devices: usize,
    pub ring_size: usize,
    pub coalesce: bool,
}

impl Default for ShardDistributedPass {
    fn default() -> Self {
        Self {
            mode: "tp".to_string(),
            devices: 2,
            ring_size: 4,
            coalesce: false,
        }
    }
}

impl Pass for ShardDistributedPass {
    fn name(&self) -> &'static str {
        "shard-distributed"
    }

    fn description(&self) -> &'static str {
        "Unified distributed SPMD tensor parallelism, RingAttention CP, and all_reduce coalescing"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        let mut steps = Vec::new();
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

/// 1F1B / Zero-Bubble Pipeline Parallelism Pass (`shard-pipeline-parallel`).
///
/// Partitions sequential compute operations across `stages` pipeline ranks with `microbatches`
/// micro-batches and stamps `castforge.pp_stage`, `castforge.pp_schedule` (`"zero_bubble_1f1b"`),
/// and `castforge.pp_microbatches` attributes.
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
