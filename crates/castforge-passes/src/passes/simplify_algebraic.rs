use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Dim, Module, ValueDef, ValueId};

/// 8. Algebraic Simplification Pass (`x + 0 -> x`, `x * 1 -> x`, `x * 0 -> 0`)
pub struct SimplifyAlgebraicPass;

impl SimplifyAlgebraicPass {
    fn const_attr(module: &Module, val_id: ValueId) -> Option<&Attribute> {
        if let ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                return op_node.attributes.get("value");
            }
        }
        None
    }
}

impl Pass for SimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Algebraic identity simplification (x+0->x, x*1->x, x*0->0, identity convert/slice)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
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

                let replacement: Option<ValueId> = match hlo {
                    StableHloOp::Add { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, lhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, rhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Subtract { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, lhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Multiply { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_one()) {
                            Some((result, lhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_one()) {
                            Some((result, rhs))
                        } else if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, rhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, lhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Convert { operand, result } => {
                        if module.values[operand].ty == module.values[result].ty {
                            Some((result, operand))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Slice {
                        operand,
                        start_indices,
                        limit_indices,
                        strides,
                        result,
                    } => {
                        let is_full = module.values[operand].ty == module.values[result].ty
                            && start_indices.iter().all(|&s| s == 0)
                            && strides.iter().all(|&s| s == 1)
                            && module.values[operand].ty.as_tensor().is_some_and(|t| {
                                t.shape.len() == limit_indices.len()
                                    && t.shape
                                        .iter()
                                        .zip(limit_indices.iter())
                                        .all(|(d, &lim)| *d == Dim::Static(lim))
                            });
                        if is_full {
                            Some((result, operand))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
                .map(|(res, rep)| {
                    module.replace_all_uses(res, rep);
                    rep
                });

                if replacement.is_some() {
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        if removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}
