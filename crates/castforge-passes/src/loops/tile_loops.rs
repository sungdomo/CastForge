use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module};

/// Loop Tiling A-to-B IR Translator (`tile-loops`).
///
/// Stamps cache-blocking (`loop_tile_m`, `loop_tile_n`, `loop_tile_k`), hierarchical L2 blocking
/// (`loop_l2_tile`), and register micro-kernel Unroll-and-Jam (`loop_reg_m`, `loop_reg_n`)
/// schedule attributes onto `dot_general` and composite GEMM operations in the IR.
pub struct TileLoopsPass {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub l2_tile: usize,
    pub reg_m: usize,
    pub reg_n: usize,
}

impl Default for TileLoopsPass {
    fn default() -> Self {
        Self {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            l2_tile: 0,
            reg_m: 0,
            reg_n: 0,
        }
    }
}

impl Pass for TileLoopsPass {
    fn name(&self) -> &'static str {
        "tile-loops"
    }

    fn description(&self) -> &'static str {
        "Translates GEMM ops by attaching L1/L2 cache and register tiling schedule attributes"
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
                let is_gemm = matches!(
                    StableHloOp::from_op(&op),
                    Some(StableHloOp::DotGeneral { .. } | StableHloOp::Convolution { .. })
                );
                if !is_gemm {
                    continue;
                }
                let attrs = &mut module.ops[op_id].attributes;
                if self.tile_m > 0 && self.tile_n > 0 && self.tile_k > 0 {
                    attrs.insert(
                        "loop_tile_m".to_string(),
                        Attribute::Int(self.tile_m as i64),
                    );
                    attrs.insert(
                        "loop_tile_n".to_string(),
                        Attribute::Int(self.tile_n as i64),
                    );
                    attrs.insert(
                        "loop_tile_k".to_string(),
                        Attribute::Int(self.tile_k as i64),
                    );
                }
                if self.l2_tile > 0 {
                    attrs.insert(
                        "loop_l2_tile".to_string(),
                        Attribute::Int(self.l2_tile as i64),
                    );
                }
                if self.reg_m > 1 && self.reg_n > 1 {
                    attrs.insert("loop_reg_m".to_string(), Attribute::Int(self.reg_m as i64));
                    attrs.insert("loop_reg_n".to_string(), Attribute::Int(self.reg_n as i64));
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
