//! Unified Algebraic Simplification, Constant Folding, Conditional Propagation (SCCP),
//! and E-Graph Factoring/FMA Pass (`simplify-algebraic`), executed in a single forward
//! dataflow traversal.

pub mod rewrites;

use rewrites::{AlgOutcome, try_egraph_and_fma_op, try_simplify_algebraic_op};

use crate::ir::{BlockId, Module, OpId};
use crate::passes::fold::{ConstFoldOutcome, try_fold_constant_op};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use std::collections::HashSet;

/// Unified Algebraic Simplification, Constant Folding & E-Graph Factoring/FMA Pass
/// (`simplify-algebraic`).
pub struct SimplifyAlgebraicPass {
    pub fold_constants: bool,
    pub include_conditional: bool,
    pub egraph: bool,
    pub fuse_fma: bool,
}

impl Default for SimplifyAlgebraicPass {
    fn default() -> Self {
        Self {
            fold_constants: true,
            include_conditional: true,
            egraph: false,
            fuse_fma: true,
        }
    }
}

impl Pass for SimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Unified constant folding, SCCP, algebraic identities, e-graph factoring & FMA extraction"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &["fold-constants", "fuse-fma"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut dead_ops: HashSet<OpId> = HashSet::new();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if StableHloOp::from_op(op).is_none() {
                    continue;
                }

                // 1. Constant folding & SCCP (`compare` / `select` + arithmetic/views)
                if self.fold_constants {
                    match try_fold_constant_op(module, op_id, self.include_conditional) {
                        ConstFoldOutcome::Modified => {
                            modified += 1;
                            continue;
                        }
                        ConstFoldOutcome::Replaced => {
                            dead_ops.insert(op_id);
                            removed += 1;
                            continue;
                        }
                        ConstFoldOutcome::Unchanged => {}
                    }
                }

                // 2. Algebraic identities, commutative canonicalization, and reassociation
                match try_simplify_algebraic_op(module, block_id, op_id) {
                    AlgOutcome::ModifiedAndDone(c) => {
                        created += c;
                        modified += 1;
                        continue;
                    }
                    AlgOutcome::Replaced(swapped_before) => {
                        if swapped_before {
                            modified += 1;
                        }
                        dead_ops.insert(op_id);
                        removed += 1;
                        continue;
                    }
                    AlgOutcome::SwappedOnly => {
                        modified += 1;
                    }
                    AlgOutcome::Unchanged => {}
                }

                // 3. E-Graph distributive factoring (`a*b +/- a*c -> a*(b +/- c)`) & FMA
                if self.egraph {
                    let (c, m) = try_egraph_and_fma_op(module, block_id, op_id, self.fuse_fma);
                    created += c;
                    modified += m;
                }
            }

            if !dead_ops.is_empty() {
                module.blocks[block_id]
                    .ops
                    .retain(|id| !dead_ops.contains(id));
                for dead_id in dead_ops {
                    module.ops.remove(dead_id);
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
