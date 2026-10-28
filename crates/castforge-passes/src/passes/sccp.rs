use crate::{FoldConstantsPass, Pass, PassResult};
use anyhow::Result;
use castforge_ir::Module;

/// 16. Sparse Conditional Constant Propagation (`sccp`)
/// Evaluates `stablehlo.compare` on constants and folds `stablehlo.select` predicates.
/// Note: `fold-constants` includes conditional constant propagation by default; `sccp` runs
/// `FoldConstantsPass { include_conditional: true }` to a fixed point.
pub struct SccpPass;

impl Pass for SccpPass {
    fn name(&self) -> &'static str {
        "sccp"
    }

    fn description(&self) -> &'static str {
        "Sparse Conditional Constant Propagation (evaluates compare/select and constants)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;

        for _ in 0..8 {
            let fold_res = (FoldConstantsPass {
                include_conditional: true,
            })
            .run(module)?;
            modified += fold_res.ops_modified;
            removed += fold_res.ops_removed;
            if !fold_res.changed {
                break;
            }
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
