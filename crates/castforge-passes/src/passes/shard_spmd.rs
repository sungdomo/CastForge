use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Megatron-Style 1D SPMD Tensor Parallelism Pass (`shard-spmd`).
///
/// Annotates 2D `stablehlo.dot_general` (and fused GEMM composite) operations with 1D Tensor
/// Parallel sharding metadata (`spmd_sharding = "column_parallel"` / `"row_parallel"`,
/// `num_devices = N`) and inserts collective `"castforge.all_reduce"` synchronization operations
/// after row-parallel projections (or after standalone projections).
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
