use crate::{
    PadForTilingPass, Pass, PassResult, PermuteLoopsPass, TileLoopsPass, UnrollLoopsPass,
    VectorizeLoopsPass,
};
use anyhow::Result;
use castforge_ir::{Attribute, Module};

/// Unified Loop Schedule Pass (`schedule-loops`).
///
/// Consolidates `pad-for-tiling`, `tile-loops`, `permute-loops`, `unroll-loops`, and
/// `vectorize-loops` into a single pass for configuring polyhedral/SIMD schedules.
pub struct ScheduleLoopsPass {
    pub tile_size: usize,
    pub l2_tile: usize,
    pub reg_tile: usize,
    pub order: String,
    pub collapse: bool,
    pub skew: bool,
    pub unroll_factor: usize,
    pub prefetch: usize,
    pub split_k: usize,
    pub double_buffer: bool,
    pub isa: String,
    pub vector_width: usize,
    pub parallel: bool,
    pub align: i64,
}

impl Default for ScheduleLoopsPass {
    fn default() -> Self {
        Self {
            tile_size: 16,
            l2_tile: 0,
            reg_tile: 0,
            order: "mkn".to_string(),
            collapse: false,
            skew: false,
            unroll_factor: 4,
            prefetch: 0,
            split_k: 1,
            double_buffer: false,
            isa: "avx2".to_string(),
            vector_width: 8,
            parallel: true,
            align: 8,
        }
    }
}

impl Pass for ScheduleLoopsPass {
    fn name(&self) -> &'static str {
        "schedule-loops"
    }

    fn description(&self) -> &'static str {
        "Unified loop schedule pass (tile, permute, unroll, vectorize, and pad-for-tiling)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            PadForTilingPass { align: self.align }.run(module)?,
            TileLoopsPass {
                tile_m: self.tile_size,
                tile_n: self.tile_size,
                tile_k: self.tile_size,
                l2_tile: self.l2_tile,
                reg_m: self.reg_tile,
                reg_n: self.reg_tile,
            }
            .run(module)?,
            PermuteLoopsPass {
                order: self.order.clone(),
                collapse: self.collapse,
                skew: self.skew,
            }
            .run(module)?,
            UnrollLoopsPass {
                factor: self.unroll_factor,
                prefetch: self.prefetch,
                split_k: self.split_k,
                double_buffer: self.double_buffer,
            }
            .run(module)?,
            VectorizeLoopsPass {
                width: self.vector_width,
                isa: self.isa.clone(),
                parallel: self.parallel,
            }
            .run(module)?,
        ];
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// GPU Matrix-Core / Tensor-Core Lowering Pass (`lower-tensor-cores`).
///
/// Annotates `dot_general` and fused GEMM/attention composites with hardware Tensor-Core /
/// Matrix-Core instruction schedules (`arch=hopper_wgmma|ampere_wmma|cdna3_mfma|apple_simdgroup`,
/// `mma_shape = [16, 16, 16]`).
pub struct LowerTensorCoresPass {
    pub arch: String,
    pub mma_m: i64,
    pub mma_n: i64,
    pub mma_k: i64,
}

impl Default for LowerTensorCoresPass {
    fn default() -> Self {
        Self {
            arch: "hopper_wgmma".to_string(),
            mma_m: 16,
            mma_n: 16,
            mma_k: 16,
        }
    }
}

impl Pass for LowerTensorCoresPass {
    fn name(&self) -> &'static str {
        "lower-tensor-cores"
    }

    fn description(&self) -> &'static str {
        "Lowers GEMMs to GPU Tensor Core / Matrix Core instructions (WMMA, WGMMA, MFMA, Simdgroup)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let instr = match self.arch.to_ascii_lowercase().as_str() {
            "cdna3" | "mfma" | "rocm" | "mi300x" => "v_mfma_f32_16x16x16_f16",
            "apple" | "metal" | "simdgroup" => "simdgroup_multiply_accumulate",
            "ampere" | "wmma" => "nvcuda::wmma::mma_sync",
            _ => "wgmma.mma_async.sync.aligned",
        };
        for (_, op) in module.ops.iter_mut() {
            let is_gemm = op.is_op("stablehlo", "dot_general")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n.contains("matmul") || n.contains("attention")));
            if is_gemm {
                op.attributes.insert(
                    "castforge.tensor_core_arch".to_string(),
                    Attribute::String(self.arch.clone()),
                );
                op.attributes.insert(
                    "castforge.tensor_core_intrinsic".to_string(),
                    Attribute::String(instr.to_string()),
                );
                op.attributes.insert(
                    "castforge.mma_shape".to_string(),
                    Attribute::DenseIntArray(vec![self.mma_m, self.mma_n, self.mma_k]),
                );
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
