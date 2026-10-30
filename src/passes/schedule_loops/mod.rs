//! Unified Polyhedral, SIMD, Convolution Lowering & Tensor-Core Loop Scheduling
//! (`schedule-loops`).
//!
//! Consolidates `lower-conv-to-gemm`, `split-large-gemm`, `pad-for-tiling`, `tile-loops`,
//! `permute-loops`, `unroll-loops`, `vectorize-loops`, and `lower-tensor-cores`
//! (`tensor_cores=wgmma|wmma|mfma|simdgroup`) into [`ScheduleLoopsPass`].

pub mod lowering;

pub use lowering::*;

use crate::ir::{Attribute, BlockId, Dim, Module};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;

/// Unified Loop Schedule, Convolution Lowering & Tensor-Core Lowering Pass (`schedule-loops`).
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
    pub tensor_cores: Option<String>,
    pub mma_tile: (i64, i64, i64),
    pub max_n: Option<i64>,
    pub lower_conv: bool,
    pub warp_specialized: bool,
    pub tmem_columns: usize,
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
            tensor_cores: None,
            mma_tile: (16, 16, 16),
            max_n: None,
            lower_conv: false,
            warp_specialized: false,
            tmem_columns: 512,
        }
    }
}

impl Pass for ScheduleLoopsPass {
    fn name(&self) -> &'static str {
        "schedule-loops"
    }

    fn description(&self) -> &'static str {
        "Unified loop schedule (tile, permute, unroll, vectorize, pad, conv-to-gemm, Tensor Cores)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        if self.lower_conv {
            let mut r = LowerConvToGemmPass.run(module)?;
            r.pass_name = self.name().to_string();
            return Ok(r);
        }
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        if let Some(max_n) = self.max_n
            && max_n > 0
        {
            let r = SplitLargeGemmPass { max_n }.run(module)?;
            created += r.ops_created;
            modified += r.ops_modified;
            removed += r.ops_removed;
        }
        modified += self.stamp_all_schedule_attributes_single_pass(module);
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

impl ScheduleLoopsPass {
    fn stamp_all_schedule_attributes_single_pass(&self, module: &mut Module) -> usize {
        let mut modified = 0usize;
        let align = self.align.max(1);
        let round_up = |x: i64| ((x + align - 1) / align) * align;
        let order_low = self.order.to_ascii_lowercase();
        let isa_low = self.isa.to_ascii_lowercase();
        let tc_info = self
            .tensor_cores
            .as_ref()
            .filter(|s| !s.eq_ignore_ascii_case("none") && !s.is_empty())
            .map(|arch| {
                let arch_low = arch.to_ascii_lowercase();
                let is_blackwell = matches!(arch_low.as_str(), "tcgen05" | "blackwell" | "sm100");
                let instr = match arch_low.as_str() {
                    "tcgen05" | "blackwell" | "sm100" => "tcgen05.mma.cta_group::2.kind::f8f6f4",
                    "cdna3" | "mfma" | "rocm" | "mi300x" => "v_mfma_f32_16x16x16_f16",
                    "apple" | "metal" | "simdgroup" => "simdgroup_multiply_accumulate",
                    "ampere" | "wmma" => "nvcuda::wmma::mma_sync",
                    _ => "wgmma.mma_async.sync.aligned",
                };
                (arch.clone(), instr, is_blackwell)
            });

        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if op.is_op("func", "return") || op.is_op("stablehlo", "constant") {
                    continue;
                }
                let hlo = StableHloOp::from_op(op);
                let is_dot = matches!(hlo, Some(StableHloOp::DotGeneral { .. }));
                let is_conv = matches!(hlo, Some(StableHloOp::Convolution { .. }));
                let is_reduce = matches!(hlo, Some(StableHloOp::Reduce { .. }));
                let is_tc_composite = op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n.contains("matmul") || n.contains("attention"));

                let pad_dims = if is_dot && op.operands.len() >= 2 {
                    let lt = module.values[op.operands[0]].ty.as_tensor().cloned();
                    let rt = module.values[op.operands[1]].ty.as_tensor().cloned();
                    match (lt, rt) {
                        (Some(l), Some(r)) if l.shape.len() == 2 && r.shape.len() == 2 => {
                            match (&l.shape[0], &l.shape[1], &r.shape[1]) {
                                (Dim::Static(m), Dim::Static(k), Dim::Static(n)) => {
                                    Some((*m, *k, *n, round_up(*m), round_up(*k), round_up(*n)))
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                let attrs = &mut module.ops[op_id].attributes;
                // 1. Pad-for-tiling alignment metadata
                if let Some((m, k, n, pm, pk, pn)) = pad_dims {
                    attrs.insert("pad_align".to_string(), Attribute::Int(align));
                    attrs.insert(
                        "padded_mnk".to_string(),
                        Attribute::DenseIntArray(vec![pm, pn, pk]),
                    );
                    attrs.insert(
                        "pad_deltas".to_string(),
                        Attribute::DenseIntArray(vec![pm - m, pn - n, pk - k]),
                    );
                }

                // 2. L1/L2/register tiling attributes
                if is_dot || is_conv {
                    if self.tile_size > 0 {
                        let t = self.tile_size as i64;
                        attrs.insert("loop_tile_m".to_string(), Attribute::Int(t));
                        attrs.insert("loop_tile_n".to_string(), Attribute::Int(t));
                        attrs.insert("loop_tile_k".to_string(), Attribute::Int(t));
                    }
                    if self.l2_tile > 0 {
                        attrs.insert(
                            "loop_l2_tile".to_string(),
                            Attribute::Int(self.l2_tile as i64),
                        );
                    }
                    if self.reg_tile > 1 {
                        let rt = self.reg_tile as i64;
                        attrs.insert("loop_reg_m".to_string(), Attribute::Int(rt));
                        attrs.insert("loop_reg_n".to_string(), Attribute::Int(rt));
                    }
                }

                // 3. Loop permutation, collapse & wavefront skew
                if is_dot {
                    attrs.insert(
                        "loop_order".to_string(),
                        Attribute::String(order_low.clone()),
                    );
                    if self.skew {
                        attrs.insert("loop_skew".to_string(), Attribute::Bool(true));
                    }
                }
                if self.collapse {
                    attrs.insert("loop_collapse".to_string(), Attribute::Bool(true));
                }

                // 4. Loop unrolling, prefetching, Split-K & double-buffering
                if is_dot || is_reduce {
                    if self.unroll_factor > 1 {
                        attrs.insert(
                            "loop_unroll".to_string(),
                            Attribute::Int(self.unroll_factor as i64),
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
                }

                // 5. SIMD vectorization & multicore parallelization
                attrs.insert("simd_vectorize".to_string(), Attribute::Bool(true));
                attrs.insert(
                    "simd_width".to_string(),
                    Attribute::Int(self.vector_width as i64),
                );
                if !isa_low.is_empty() && isa_low != "scalar" {
                    attrs.insert("simd_isa".to_string(), Attribute::String(isa_low.clone()));
                }
                if self.parallel {
                    attrs.insert("loop_parallel".to_string(), Attribute::Bool(true));
                }

                // 6. GPU Tensor Cores / Matrix Cores (`wgmma` / `tcgen05` / `wmma` / `mfma`)
                if (is_dot || is_tc_composite)
                    && let Some((ref arch, instr, is_blackwell)) = tc_info
                {
                    attrs.insert(
                        "castforge.tensor_core_arch".to_string(),
                        Attribute::String(arch.clone()),
                    );
                    attrs.insert(
                        "castforge.tensor_core_intrinsic".to_string(),
                        Attribute::String(instr.to_string()),
                    );
                    attrs.insert(
                        "castforge.mma_shape".to_string(),
                        Attribute::DenseIntArray(vec![
                            self.mma_tile.0,
                            self.mma_tile.1,
                            self.mma_tile.2,
                        ]),
                    );
                    if is_blackwell {
                        attrs.insert(
                            "castforge.tmem_allocated".to_string(),
                            Attribute::Bool(true),
                        );
                        attrs.insert(
                            "castforge.tmem_columns".to_string(),
                            Attribute::Int(self.tmem_columns.max(64) as i64),
                        );
                    }
                    if self.warp_specialized || is_blackwell {
                        attrs.insert(
                            "castforge.warp_specialized".to_string(),
                            Attribute::Bool(true),
                        );
                        attrs.insert("castforge.tma_multicast".to_string(), Attribute::Bool(true));
                    }
                }

                modified += 1;
            }
        }
        modified
    }
}

/// GPU Matrix-Core / Tensor-Core Lowering Pass (`lower-tensor-cores`).
pub struct LowerTensorCoresPass {
    pub arch: String,
    pub mma_m: i64,
    pub mma_n: i64,
    pub mma_k: i64,
    pub warp_specialized: bool,
    pub tmem_columns: usize,
}

impl Default for LowerTensorCoresPass {
    fn default() -> Self {
        Self {
            arch: "hopper_wgmma".to_string(),
            mma_m: 16,
            mma_n: 16,
            mma_k: 16,
            warp_specialized: false,
            tmem_columns: 512,
        }
    }
}

impl Pass for LowerTensorCoresPass {
    fn name(&self) -> &'static str {
        "lower-tensor-cores"
    }

    fn description(&self) -> &'static str {
        "Lowers GEMMs to GPU Tensor Core / Matrix Core instructions (WMMA, WGMMA, tcgen05, MFMA)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let arch_low = self.arch.to_ascii_lowercase();
        let is_blackwell = matches!(arch_low.as_str(), "tcgen05" | "blackwell" | "sm100");
        let instr = match arch_low.as_str() {
            "tcgen05" | "blackwell" | "sm100" => "tcgen05.mma.cta_group::2.kind::f8f6f4",
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
                if is_blackwell {
                    op.attributes.insert(
                        "castforge.tmem_allocated".to_string(),
                        Attribute::Bool(true),
                    );
                    op.attributes.insert(
                        "castforge.tmem_columns".to_string(),
                        Attribute::Int(self.tmem_columns.max(64) as i64),
                    );
                }
                if self.warp_specialized || is_blackwell {
                    op.attributes.insert(
                        "castforge.warp_specialized".to_string(),
                        Attribute::Bool(true),
                    );
                    op.attributes
                        .insert("castforge.tma_multicast".to_string(), Attribute::Bool(true));
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

/// Loop Tiling Pass (`tile-loops`).
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

/// Loop Permutation, Coalescing & Wavefront Skewing Pass (`permute-loops`).
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

/// Loop Unrolling, Software Pipelining, Split-K & Double-Buffering Pass (`unroll-loops`).
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

/// SIMD Vectorization & Multicore Parallelization Pass (`vectorize-loops`).
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
