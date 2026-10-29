use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef};
use smallvec::SmallVec;

/// Fold Broadcast Pass (`fold-broadcast`).
///
/// Simplifies `stablehlo.broadcast_in_dim` operations by:
/// - Eliminating identity broadcasts where operand and result types match.
/// - Composing chained `broadcast_in_dim(broadcast_in_dim(x, d1), d2)` into a single op.
/// - Folding `broadcast_in_dim` of splat constants directly into a target-shaped splat constant.
/// - Hoisting unary elementwise ops (`exponential`, `tanh`, `rsqrt`) before `broadcast_in_dim`
///   so unary math runs on the smaller unbroadcasted tensor.
pub struct FoldBroadcastPass;

impl Pass for FoldBroadcastPass {
    fn name(&self) -> &'static str {
        "fold-broadcast"
    }

    fn description(&self) -> &'static str {
        "Composes chained broadcast_in_dim ops, removes identity broadcasts, and hoists unary ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                match hlo {
                    StableHloOp::BroadcastInDim {
                        operand,
                        broadcast_dimensions: dims2,
                        result,
                    } => {
                        let mut cur_operand = operand;
                        let mut cur_dims = dims2;

                        // 1. Compose chained `broadcast_in_dim(broadcast_in_dim(x, dims1), dims2)`
                        if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                            && let Some(prod_op) = module.ops.get(prod_id)
                            && let Some(StableHloOp::BroadcastInDim {
                                operand: base_in,
                                broadcast_dimensions: dims1,
                                ..
                            }) = StableHloOp::from_op(prod_op)
                            && dims1.iter().all(|&d| (d as usize) < cur_dims.len())
                        {
                            let composed: Vec<i64> =
                                dims1.iter().map(|&d| cur_dims[d as usize]).collect();
                            cur_operand = base_in;
                            cur_dims = composed;

                            let op_mut = &mut module.ops[op_id];
                            op_mut.operands[0] = cur_operand;
                            op_mut.attributes.insert(
                                "broadcast_dimensions".to_string(),
                                Attribute::DenseIntArray(cur_dims.clone()),
                            );
                            modified += 1;
                        }

                        // 2. Fold `broadcast_in_dim(splat_const)` into a splat constant
                        if let ValueDef::OpResult { op: prod_id, .. } =
                            module.values[cur_operand].def
                            && let Some(prod_op) = module.ops.get(prod_id)
                            && prod_op.is_op("stablehlo", "constant")
                            && let Some(Attribute::DenseElements { raw_literal, .. }) =
                                prod_op.attributes.get("value")
                            && !raw_literal.trim().starts_with('[')
                            && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
                        {
                            let lit = raw_literal.clone();
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "constant");
                            op_mut.operands.clear();
                            op_mut.attributes.clear();
                            op_mut.attributes.insert(
                                "value".to_string(),
                                Attribute::DenseElements {
                                    ty: res_ty,
                                    raw_literal: lit,
                                },
                            );
                            modified += 1;
                            continue;
                        }

                        // 3. Eliminate identity broadcast `[0, 1, ..., rank - 1]` with same type
                        let is_identity = module.values[cur_operand].ty == module.values[result].ty
                            && cur_dims.iter().enumerate().all(|(i, &d)| d == i as i64);
                        if is_identity {
                            module.replace_all_uses(result, cur_operand);
                            module.remove_op(block_id, op_id);
                            removed += 1;
                        }
                    }
                    // 4. Hoist unary ops above single-use `broadcast_in_dim`:
                    // `unary(broadcast_in_dim(x)) -> broadcast_in_dim(unary(x))`
                    StableHloOp::Exp { operand, result }
                    | StableHloOp::Tanh { operand, result }
                    | StableHloOp::Rsqrt { operand, result } => {
                        if module.use_count(operand) != 1 {
                            continue;
                        }
                        let ValueDef::OpResult { op: bcast_id, .. } = module.values[operand].def
                        else {
                            continue;
                        };
                        let Some(bcast_op) = module.ops.get(bcast_id) else {
                            continue;
                        };
                        let Some(StableHloOp::BroadcastInDim {
                            operand: small_x,
                            broadcast_dimensions,
                            ..
                        }) = StableHloOp::from_op(bcast_op)
                        else {
                            continue;
                        };
                        let unary_name = module.ops[op_id].name.clone();
                        let small_ty = module.values[small_x].ty.clone();
                        let big_ty = module.values[result].ty.clone();
                        if small_ty == big_ty {
                            continue;
                        }

                        // Mutate `bcast_id` in-place into `unary(small_x) -> small_ty`
                        let bcast_mut = &mut module.ops[bcast_id];
                        bcast_mut.name = unary_name;
                        bcast_mut.operands = SmallVec::from_slice(&[small_x]);
                        bcast_mut.attributes.clear();
                        module.values[operand].ty = small_ty;

                        // Mutate `op_id` in-place into `broadcast_in_dim(operand) -> big_ty`
                        let unary_mut = &mut module.ops[op_id];
                        unary_mut.name = OpName::new("stablehlo", "broadcast_in_dim");
                        unary_mut.operands = SmallVec::from_slice(&[operand]);
                        unary_mut.attributes.clear();
                        unary_mut.attributes.insert(
                            "broadcast_dimensions".to_string(),
                            Attribute::DenseIntArray(broadcast_dimensions),
                        );
                        modified += 1;
                    }
                    _ => {}
                }
            }
        }

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
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
