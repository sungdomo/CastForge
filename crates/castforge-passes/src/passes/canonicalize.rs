use crate::{
    CommonSubexpressionEliminationPass, DeadCodeEliminationPass, FoldBroadcastPass,
    FoldConstantsPass, FoldReshapePass, FoldSliceConcatPass, FoldTransposePass, Pass, PassResult,
    SimplifyAlgebraicPass,
};
use anyhow::Result;
use castforge_ir::Module;

/// Composite Canonicalize Pass (`canonicalize`).
pub struct CanonicalizePass;

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn description(&self) -> &'static str {
        "Composite canonicalization pass (constants, algebraic, layout, slice/concat, cse, dce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let steps = [
                FoldConstantsPass::default().run(module)?,
                SimplifyAlgebraicPass.run(module)?,
                FoldTransposePass::default().run(module)?,
                FoldReshapePass.run(module)?,
                FoldBroadcastPass.run(module)?,
                FoldSliceConcatPass.run(module)?,
                CommonSubexpressionEliminationPass.run(module)?,
                DeadCodeEliminationPass::default().run(module)?,
            ];

            let iter_changed = steps.iter().any(|r| r.changed);
            total.changed |= iter_changed;
            total.ops_removed += steps.iter().map(|r| r.ops_removed).sum::<usize>();
            total.ops_modified += steps.iter().map(|r| r.ops_modified).sum::<usize>();
            if !iter_changed {
                break;
            }
        }

        Ok(total)
    }
}
