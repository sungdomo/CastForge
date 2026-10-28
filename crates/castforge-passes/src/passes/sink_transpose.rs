use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// AI Optimization Pass: Sinks `stablehlo.transpose` through unary and binary elementwise
/// operations toward producers (such as `stablehlo.constant` weights) so that constant
/// folding or transpose cancellation can eliminate runtime layout conversions.
pub struct SinkTransposePass;

impl Pass for SinkTransposePass {
    fn name(&self) -> &'static str {
        "sink-transpose"
    }

    fn description(&self) -> &'static str {
        "Sinks stablehlo.transpose through elementwise ops toward constants and producers"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            loop {
                let mut iter_changed = false;
                let ops = module.block_ops(block_id);

                for op_id in ops {
                    let Some(op) = module.ops.get(op_id) else {
                        continue;
                    };
                    let Some(StableHloOp::Transpose {
                        operand: elem_val,
                        permutation: perm,
                        result: trans_res,
                    }) = StableHloOp::from_op(op)
                    else {
                        continue;
                    };

                    let ValueDef::OpResult { op: prod_op_id, .. } = module.values[elem_val].def
                    else {
                        continue;
                    };
                    if module.use_count(elem_val) != 1 {
                        continue;
                    }
                    let Some(prod_op) = module.ops.get(prod_op_id).cloned() else {
                        continue;
                    };
                    let Some(prod_hlo) = StableHloOp::from_op(&prod_op) else {
                        continue;
                    };

                    let trans_ty = module.values[trans_res].ty.clone();

                    match prod_hlo {
                        StableHloOp::Exp { operand: x, .. }
                        | StableHloOp::Tanh { operand: x, .. }
                        | StableHloOp::Rsqrt { operand: x, .. }
                        | StableHloOp::Convert { operand: x, .. } => {
                            let tx = insert_transpose_before(
                                module,
                                block_id,
                                op_id,
                                x,
                                &perm,
                                trans_ty.clone(),
                            );
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = prod_op.name.clone();
                            op_mut.operands = SmallVec::from_slice(&[tx]);
                            op_mut.attributes = prod_op.attributes.clone();
                            module.remove_op(block_id, prod_op_id);
                            created += 1;
                            modified += 1;
                            removed += 1;
                            iter_changed = true;
                            break;
                        }
                        StableHloOp::Add { lhs, rhs, .. }
                        | StableHloOp::Subtract { lhs, rhs, .. }
                        | StableHloOp::Multiply { lhs, rhs, .. }
                        | StableHloOp::Divide { lhs, rhs, .. }
                        | StableHloOp::Maximum { lhs, rhs, .. }
                        | StableHloOp::Minimum { lhs, rhs, .. } => {
                            let tlhs = insert_transpose_before(
                                module,
                                block_id,
                                op_id,
                                lhs,
                                &perm,
                                trans_ty.clone(),
                            );
                            let trhs = insert_transpose_before(
                                module,
                                block_id,
                                op_id,
                                rhs,
                                &perm,
                                trans_ty.clone(),
                            );
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = prod_op.name.clone();
                            op_mut.operands = SmallVec::from_slice(&[tlhs, trhs]);
                            op_mut.attributes = prod_op.attributes.clone();
                            module.remove_op(block_id, prod_op_id);
                            created += 2;
                            modified += 1;
                            removed += 1;
                            iter_changed = true;
                            break;
                        }
                        _ => {}
                    }
                }

                if !iter_changed {
                    break;
                }
            }
        }

        if created > 0 || modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn insert_transpose_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input: ValueId,
    perm: &[i64],
    out_ty: castforge_ir::Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "permutation".to_string(),
        Attribute::DenseIntArray(perm.to_vec()),
    );
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "transpose"),
        operands: SmallVec::from_slice(&[input]),
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
