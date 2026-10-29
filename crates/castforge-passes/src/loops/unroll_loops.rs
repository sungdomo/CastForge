use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module};

/// Loop Unrolling, Software Pipelining, Split-K & Double-Buffering A-to-B IR Translator
/// (`unroll-loops`).
///
/// Attaches inner-loop unrolling (`loop_unroll`), software prefetch distance (`loop_prefetch`),
/// Split-K reduction partitioning (`loop_split_k`), and 2-stage ping-pong tile double-buffering
/// (`loop_double_buffer`) schedule attributes onto GEMM and reduction operations.
pub struct UnrollLoopsPass {
    pub factor: usize,
    pub prefetch: usize,
    pub split_k: usize,
    pub double_buffer: bool,
}

impl Default for UnrollLoopsPass {
    fn default() -> Self {
        Self {
            factor: 4,
            prefetch: 0,
            split_k: 1,
            double_buffer: false,
        }
    }
}

impl Pass for UnrollLoopsPass {
    fn name(&self) -> &'static str {
        "unroll-loops"
    }

    fn description(&self) -> &'static str {
        "Translates reduction loops by attaching unroll, prefetch, split-k, and double-buffer attrs"
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
                let is_target = matches!(
                    StableHloOp::from_op(&op),
                    Some(StableHloOp::DotGeneral { .. } | StableHloOp::Reduce { .. })
                );
                if !is_target {
                    continue;
                }
                let attrs = &mut module.ops[op_id].attributes;
                if self.factor > 1 {
                    attrs.insert(
                        "loop_unroll".to_string(),
                        Attribute::Int(self.factor as i64),
                    );
                }
                if self.prefetch > 0 {
                    attrs.insert(
                        "loop_prefetch".to_string(),
                        Attribute::Int(self.prefetch as i64),
                    );
                }
                if self.split_k > 1 {
                    attrs.insert(
                        "loop_split_k".to_string(),
                        Attribute::Int(self.split_k as i64),
                    );
                }
                if self.double_buffer {
                    attrs.insert("loop_double_buffer".to_string(), Attribute::Bool(true));
                }
                modified += 1;
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
