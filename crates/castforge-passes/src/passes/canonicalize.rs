use crate::{
    CommonSubexpressionEliminationPass, DeadCodeEliminationPass, FoldConstantsPass,
    FoldReshapePass, FoldTransposePass, Pass, PassResult, SimplifyAlgebraicPass,
};
use anyhow::Result;
use castforge_ir::Module;

/// 7. Composite Canonicalize Pass
pub struct CanonicalizePass;

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn description(&self) -> &'static str {
        "Composite canonicalization pass (constants, algebraic, transpose, reshape, cse, dce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let r0 = FoldConstantsPass::default().run(module)?;
            let r_alg = SimplifyAlgebraicPass.run(module)?;
            let r1 = FoldTransposePass::default().run(module)?;
            let r2 = FoldReshapePass.run(module)?;
            let r3 = CommonSubexpressionEliminationPass.run(module)?;
            let r4 = DeadCodeEliminationPass::default().run(module)?;

            let iter_changed =
                r0.changed || r_alg.changed || r1.changed || r2.changed || r3.changed || r4.changed;
            total.changed |= iter_changed;
            total.ops_removed += r0.ops_removed
                + r_alg.ops_removed
                + r1.ops_removed
                + r2.ops_removed
                + r3.ops_removed
                + r4.ops_removed;
            total.ops_modified += r0.ops_modified
                + r_alg.ops_modified
                + r1.ops_modified
                + r2.ops_modified
                + r3.ops_modified
                + r4.ops_modified;
            if !iter_changed {
                break;
            }
        }

        Ok(total)
    }
}
