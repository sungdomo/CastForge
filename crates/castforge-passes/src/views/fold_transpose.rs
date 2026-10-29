use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, ValueDef};

/// Fold Transpose Pass (`transpose(transpose(x, p1), p2)` -> `transpose(x, p1[p2])`).
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

                if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def
                    && let Some(prod_op) = module.ops.get(prod_op_id)
                    && let Some(StableHloOp::Transpose {
                        operand: base_input,
                        permutation: perm1,
                        ..
                    }) = StableHloOp::from_op(prod_op)
                    && perm1.len() == current_perm.len()
                {
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
