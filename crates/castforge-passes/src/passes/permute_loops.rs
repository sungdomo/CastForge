use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module};

/// Loop Permutation, Coalescing & Wavefront Skewing A-to-B IR Translator (`permute-loops`).
///
/// Attaches loop interchange order (`loop_order = "mnk" | "mkn" | "nkm"`), 2D->1D loop
/// coalescing (`loop_collapse = true`), and polyhedral wavefront skewing (`loop_skew = true`)
/// schedule attributes onto operations in the IR.
pub struct PermuteLoopsPass {
    pub order: String,
    pub collapse: bool,
    pub skew: bool,
}

impl Default for PermuteLoopsPass {
    fn default() -> Self {
        Self {
            order: "mkn".to_string(),
            collapse: false,
            skew: false,
        }
    }
}

impl Pass for PermuteLoopsPass {
    fn name(&self) -> &'static str {
        "permute-loops"
    }

    fn description(&self) -> &'static str {
        "Translates loop nests by attaching permutation (mnk/mkn/nkm), collapse, or skew attributes"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.is_op("func", "return") || op.is_op("stablehlo", "constant") {
                    continue;
                }
                let is_gemm = matches!(
                    StableHloOp::from_op(&op),
                    Some(StableHloOp::DotGeneral { .. })
                );
                let attrs = &mut module.ops[op_id].attributes;
                if is_gemm {
                    attrs.insert(
                        "loop_order".to_string(),
                        Attribute::String(self.order.to_ascii_lowercase()),
                    );
                    if self.skew {
                        attrs.insert("loop_skew".to_string(), Attribute::Bool(true));
                    }
                    modified += 1;
                }
                if self.collapse {
                    attrs.insert("loop_collapse".to_string(), Attribute::Bool(true));
                    if !is_gemm {
                        modified += 1;
                    }
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
