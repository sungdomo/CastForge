use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Multi-Stream Async Compute / Copy Overlap Pass (`schedule-streams`).
///
/// Identifies independent producer branches feeding into a common join operation
/// (such as parallel projections, dual gated/routed branches, or residual branches),
/// assigns concurrent execution stream IDs (`stream_id = 0, 1, ...`), and inserts
/// `stablehlo.composite` (`"castforge.stream_wait"`) synchronization barriers before
/// the join operation.
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
                // Check direct independence: neither op directly consumes the other's result
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
