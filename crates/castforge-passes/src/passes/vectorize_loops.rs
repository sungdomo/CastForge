use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, BlockId, Module};

/// SIMD Vectorization & Multicore Parallelization A-to-B IR Translator (`vectorize-loops`).
///
/// Attaches hardware SIMD ISA (`simd_isa = "scalar" | "avx2" | "neon"`), vector lane width
/// (`simd_width`), explicit vectorization (`simd_vectorize = true`), and multicore parallel
/// outer-loop work-sharing (`loop_parallel = true`) schedule attributes onto compute operations.
pub struct VectorizeLoopsPass {
    pub width: usize,
    pub isa: String,
    pub parallel: bool,
}

impl Default for VectorizeLoopsPass {
    fn default() -> Self {
        Self {
            width: 8,
            isa: "scalar".to_string(),
            parallel: false,
        }
    }
}

impl Pass for VectorizeLoopsPass {
    fn name(&self) -> &'static str {
        "vectorize-loops"
    }

    fn description(&self) -> &'static str {
        "Translates compute ops by attaching SIMD vectorization, ISA, and parallel attrs"
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
                let attrs = &mut module.ops[op_id].attributes;
                attrs.insert("simd_vectorize".to_string(), Attribute::Bool(true));
                attrs.insert("simd_width".to_string(), Attribute::Int(self.width as i64));
                if !self.isa.is_empty() && self.isa != "scalar" {
                    attrs.insert(
                        "simd_isa".to_string(),
                        Attribute::String(self.isa.to_ascii_lowercase()),
                    );
                }
                if self.parallel {
                    attrs.insert("loop_parallel".to_string(), Attribute::Bool(true));
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
