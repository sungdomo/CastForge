use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::{Result, anyhow};
use castforge_dialect_stablehlo::{StableHloOp, verify_stablehlo_shapes};
use castforge_ir::{Attribute, BlockId, Dim, Module, ValueDef, ValueId};

/// 3. Fold Transpose Pass (`transpose(transpose(x, p1), p2)` -> `transpose(x, p1[p2])`)
pub struct FoldTransposePass {
    pub eliminate_identity: bool,
}

impl Default for FoldTransposePass {
    fn default() -> Self {
        Self {
            eliminate_identity: true,
        }
    }
}

impl Pass for FoldTransposePass {
    fn name(&self) -> &'static str {
        "fold-transpose"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.transpose ops and eliminates identity transposes"
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
                let Some(StableHloOp::Transpose {
                    operand,
                    permutation: perm2,
                    result,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                // Check if producer of `operand` is also a `stablehlo.transpose`
                let mut current_operand = operand;
                let mut current_perm = perm2;

                if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def {
                    if let Some(prod_op) = module.ops.get(prod_op_id) {
                        if let Some(StableHloOp::Transpose {
                            operand: base_input,
                            permutation: perm1,
                            ..
                        }) = StableHloOp::from_op(prod_op)
                        {
                            if perm1.len() == current_perm.len() {
                                // Compose permutations: composed[i] = perm1[perm2[i]]
                                let composed: Vec<i64> = current_perm
                                    .iter()
                                    .map(|&idx| perm1[idx as usize])
                                    .collect();
                                current_operand = base_input;
                                current_perm = composed;

                                let op_mut = &mut module.ops[op_id];
                                op_mut.operands[0] = current_operand;
                                op_mut.attributes.insert(
                                    "permutation".to_string(),
                                    Attribute::DenseIntArray(current_perm.clone()),
                                );
                                modified += 1;
                            }
                        }
                    }
                }

                // Check if the resulting permutation is an identity `[0, 1, 2, ...]`
                let is_identity = self.eliminate_identity
                    && !current_perm.is_empty()
                    && current_perm.iter().enumerate().all(|(i, &p)| p == i as i64);

                if is_identity {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        // Clean up any producer transpose that became dead
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

/// 4. Fold Reshape Pass (`reshape(reshape(x))` -> `reshape(x)` & identity reshape removal)
pub struct FoldReshapePass;

impl Pass for FoldReshapePass {
    fn name(&self) -> &'static str {
        "fold-reshape"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.reshape ops and eliminates identity reshapes"
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
                let Some(StableHloOp::Reshape { operand, result }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut current_operand = operand;
                if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def {
                    if let Some(prod_op) = module.ops.get(prod_op_id) {
                        if let Some(StableHloOp::Reshape {
                            operand: base_input,
                            ..
                        }) = StableHloOp::from_op(prod_op)
                        {
                            current_operand = base_input;
                            module.ops[op_id].operands[0] = current_operand;
                            modified += 1;
                        }
                    }
                }

                // If input type and output type of reshape are identical, eliminate it!
                if module.values[current_operand].ty == module.values[result].ty {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
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

/// 10. Shape Verification Pass (`verify-shapes`)
pub struct VerifyShapesPass;

impl Pass for VerifyShapesPass {
    fn name(&self) -> &'static str {
        "verify-shapes"
    }

    fn description(&self) -> &'static str {
        "Verifies StableHLO tensor shapes and rank consistency across all ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        for (op_id, _) in &module.ops {
            if let Err(msg) = verify_stablehlo_shapes(module, op_id) {
                return Err(anyhow!(msg));
            }
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: false,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}
