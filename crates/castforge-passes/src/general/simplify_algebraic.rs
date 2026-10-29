use crate::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Dim, Module, OpName, ValueDef, ValueId};

/// Algebraic Simplification, Commutative Canonicalization & Reassociation Pass
/// (`simplify-algebraic`).
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

    fn is_constant(module: &Module, val_id: ValueId) -> bool {
        Self::const_attr(module, val_id).is_some()
    }

    fn format_splat(v: f64) -> String {
        if (v - v.round()).abs() < 1e-9 {
            format!("{:.1}", v.round())
        } else {
            format!("{v}")
        }
    }
}

impl Pass for SimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Algebraic identities (x+0, x-x, x*1, x/1, max(x,x), reassociation, commutative order)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(mut hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                // 1. Commutative canonicalization: place constant operand on RHS
                match &mut hlo {
                    StableHloOp::Add { lhs, rhs, .. }
                    | StableHloOp::Multiply { lhs, rhs, .. }
                    | StableHloOp::Maximum { lhs, rhs, .. }
                    | StableHloOp::Minimum { lhs, rhs, .. }
                        if Self::is_constant(module, *lhs) && !Self::is_constant(module, *rhs) =>
                    {
                        std::mem::swap(lhs, rhs);
                        module.ops[op_id].operands.swap(0, 1);
                        modified += 1;
                    }
                    _ => {}
                }

                // 2. Reassociation of splat constants: `(x + c1) + c2 -> x + (c1 + c2)`
                // and `(x * c1) * c2 -> x * (c1 * c2)`
                match hlo {
                    StableHloOp::Add { lhs, rhs, .. } => {
                        if let Some(c2) = FoldConstantsPass::const_splat(module, rhs)
                            && let ValueDef::OpResult { op: lhs_op_id, .. } = module.values[lhs].def
                            && let Some(lhs_op) = module.ops.get(lhs_op_id)
                            && let Some(StableHloOp::Add {
                                lhs: x,
                                rhs: c1_val,
                                ..
                            }) = StableHloOp::from_op(lhs_op)
                            && let Some(c1) = FoldConstantsPass::const_splat(module, c1_val)
                            && let ValueDef::OpResult { op: c2_op_id, .. } = module.values[rhs].def
                            && let Some(c2_ty) = module.values[rhs].ty.as_tensor().cloned()
                        {
                            let sum_lit = Self::format_splat(c1 + c2);
                            module.ops[c2_op_id].attributes.insert(
                                "value".to_string(),
                                Attribute::DenseElements {
                                    ty: c2_ty,
                                    raw_literal: sum_lit,
                                },
                            );
                            module.ops[op_id].operands[0] = x;
                            modified += 1;
                            continue;
                        }
                    }
                    StableHloOp::Multiply { lhs, rhs, .. } => {
                        if let Some(c2) = FoldConstantsPass::const_splat(module, rhs)
                            && let ValueDef::OpResult { op: lhs_op_id, .. } = module.values[lhs].def
                            && let Some(lhs_op) = module.ops.get(lhs_op_id)
                            && let Some(StableHloOp::Multiply {
                                lhs: x,
                                rhs: c1_val,
                                ..
                            }) = StableHloOp::from_op(lhs_op)
                            && let Some(c1) = FoldConstantsPass::const_splat(module, c1_val)
                            && let ValueDef::OpResult { op: c2_op_id, .. } = module.values[rhs].def
                            && let Some(c2_ty) = module.values[rhs].ty.as_tensor().cloned()
                        {
                            let prod_lit = Self::format_splat(c1 * c2);
                            module.ops[c2_op_id].attributes.insert(
                                "value".to_string(),
                                Attribute::DenseElements {
                                    ty: c2_ty,
                                    raw_literal: prod_lit,
                                },
                            );
                            module.ops[op_id].operands[0] = x;
                            modified += 1;
                            continue;
                        }
                    }
                    _ => {}
                }

                // 3. Self-subtraction `subtract(x, x) -> constant 0`
                if let StableHloOp::Subtract { lhs, rhs, result } = hlo
                    && lhs == rhs
                    && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
                {
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "constant");
                    op_mut.operands.clear();
                    op_mut.attributes.clear();
                    op_mut.attributes.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty: res_ty,
                            raw_literal: "0.0".to_string(),
                        },
                    );
                    modified += 1;
                    continue;
                }

                // 4. Direct value-replacement identities
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
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_one())
                            || Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_zero())
                        {
                            Some((result, lhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_one())
                            || Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero())
                        {
                            Some((result, rhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Divide { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_one()) {
                            Some((result, lhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Maximum { lhs, rhs, result }
                    | StableHloOp::Minimum { lhs, rhs, result } => {
                        if lhs == rhs {
                            Some((result, lhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Select {
                        on_true,
                        on_false,
                        result,
                        ..
                    } => {
                        if on_true == on_false {
                            Some((result, on_true))
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

        if removed > 0 || modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
