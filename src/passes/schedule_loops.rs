//! Unified Polyhedral, SIMD, Convolution Lowering & Tensor-Core Loop Scheduling
//! (`schedule-loops`).
//!
//! Consolidates `lower-conv-to-gemm`, `split-large-gemm`, `pad-for-tiling`, `tile-loops`,
//! `permute-loops`, `unroll-loops`, `vectorize-loops`, and `lower-tensor-cores`
//! (`tensor_cores=wgmma|wmma|mfma|simdgroup`) into [`ScheduleLoopsPass`].

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

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
        let mut steps = Vec::new();
        if let Some(max_n) = self.max_n
            && max_n > 0
        {
            steps.push(SplitLargeGemmPass { max_n }.run(module)?);
        }
        steps.extend([
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
        ]);
        if let Some(ref tc_arch) = self.tensor_cores
            && !tc_arch.eq_ignore_ascii_case("none")
        {
            steps.push(
                LowerTensorCoresPass {
                    arch: tc_arch.clone(),
                    mma_m: self.mma_tile.0,
                    mma_n: self.mma_tile.1,
                    mma_k: self.mma_tile.2,
                    warp_specialized: self.warp_specialized,
                    tmem_columns: self.tmem_columns,
                }
                .run(module)?,
            );
        }
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

/// Static Tile-Alignment Padding Pass (`pad-for-tiling`).
#[derive(Debug, Clone)]
pub struct PadForTilingPass {
    pub align: i64,
}

impl Default for PadForTilingPass {
    fn default() -> Self {
        Self { align: 8 }
    }
}

impl Pass for PadForTilingPass {
    fn name(&self) -> &'static str {
        "pad-for-tiling"
    }

    fn description(&self) -> &'static str {
        "Computes tile-aligned (M, K, N) multiples for GEMM ops to eliminate remainder tails"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let align = self.align.max(1);
        let round_up = |x: i64| ((x + align - 1) / align) * align;
        let mut modified = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "dot_general") || op.operands.len() < 2 {
                    continue;
                }
                let lhs_ty = module.values[op.operands[0]].ty.as_tensor().cloned();
                let rhs_ty = module.values[op.operands[1]].ty.as_tensor().cloned();
                let (Some(lt), Some(rt)) = (lhs_ty, rhs_ty) else {
                    continue;
                };
                if lt.shape.len() != 2 || rt.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k), Dim::Static(n)) =
                    (&lt.shape[0], &lt.shape[1], &rt.shape[1])
                else {
                    continue;
                };
                let (pm, pk, pn) = (round_up(*m), round_up(*k), round_up(*n));
                let dst = &mut module.ops[op_id].attributes;
                dst.insert("pad_align".to_string(), Attribute::Int(align));
                dst.insert(
                    "padded_mnk".to_string(),
                    Attribute::DenseIntArray(vec![pm, pk, pn]),
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

/// Large GEMM Column-Splitting Pass (`split-large-gemm`).
#[derive(Debug, Clone)]
pub struct SplitLargeGemmPass {
    pub max_n: i64,
}

impl Default for SplitLargeGemmPass {
    fn default() -> Self {
        Self { max_n: 2 }
    }
}

impl Pass for SplitLargeGemmPass {
    fn name(&self) -> &'static str {
        "split-large-gemm"
    }

    fn description(&self) -> &'static str {
        "Splits wide dot_general GEMMs (N > max_n) into sliced sub-GEMMs + concatenate"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let max_n = self.max_n.max(1);
        let mut created = 0usize;
        let mut removed = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "dot_general")
                    || op.operands.len() < 2
                    || op.results.len() != 1
                {
                    continue;
                }
                let lhs_id = op.operands[0];
                let rhs_id = op.operands[1];
                let orig_res = op.results[0];
                let Some(lt) = module.values[lhs_id].ty.as_tensor().cloned() else {
                    continue;
                };
                let Some(rt) = module.values[rhs_id].ty.as_tensor().cloned() else {
                    continue;
                };
                if lt.shape.len() != 2 || rt.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k), Dim::Static(k2), Dim::Static(n)) =
                    (&lt.shape[0], &lt.shape[1], &rt.shape[0], &rt.shape[1])
                else {
                    continue;
                };
                if *k != *k2 || *n <= max_n || *n < 2 {
                    continue;
                }

                let n_left = *n / 2;
                let n_right = *n - n_left;
                let dt = lt.dtype.clone();

                let mut sl_l_attrs = IndexMap::new();
                sl_l_attrs.insert(
                    "start_indices".to_string(),
                    Attribute::DenseIntArray(vec![0, 0]),
                );
                sl_l_attrs.insert(
                    "limit_indices".to_string(),
                    Attribute::DenseIntArray(vec![*k, n_left]),
                );
                sl_l_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
                let w_l_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*k), Dim::Static(n_left)],
                    dt.clone(),
                ));
                let sl_l_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "slice"),
                    smallvec![rhs_id],
                    vec![(Some("w_split_l".to_string()), w_l_ty)],
                    sl_l_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let w_l_val = module.ops[sl_l_op].results[0];

                let mut sl_r_attrs = IndexMap::new();
                sl_r_attrs.insert(
                    "start_indices".to_string(),
                    Attribute::DenseIntArray(vec![0, n_left]),
                );
                sl_r_attrs.insert(
                    "limit_indices".to_string(),
                    Attribute::DenseIntArray(vec![*k, *n]),
                );
                sl_r_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
                let w_r_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*k), Dim::Static(n_right)],
                    dt.clone(),
                ));
                let sl_r_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "slice"),
                    smallvec![rhs_id],
                    vec![(Some("w_split_r".to_string()), w_r_ty)],
                    sl_r_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let w_r_val = module.ops[sl_r_op].results[0];

                let out_l_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*m), Dim::Static(n_left)],
                    dt.clone(),
                ));
                let dot_l_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    smallvec![lhs_id, w_l_val],
                    vec![(Some("gemm_l".to_string()), out_l_ty)],
                    op.attributes.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let dot_l_val = module.ops[dot_l_op].results[0];

                let out_r_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*m), Dim::Static(n_right)],
                    dt.clone(),
                ));
                let dot_r_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    smallvec![lhs_id, w_r_val],
                    vec![(Some("gemm_r".to_string()), out_r_ty)],
                    op.attributes.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let dot_r_val = module.ops[dot_r_op].results[0];

                let mut cat_attrs = IndexMap::new();
                cat_attrs.insert("dimension".to_string(), Attribute::Int(1));
                let full_out_ty = module.values[orig_res].ty.clone();
                let cat_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "concatenate"),
                    smallvec![dot_l_val, dot_r_val],
                    vec![(Some("gemm_cat".to_string()), full_out_ty)],
                    cat_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let cat_val = module.ops[cat_op].results[0];

                module.replace_all_uses(orig_res, cat_val);
                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id]
                        .ops
                        .splice(pos..=pos, [sl_l_op, sl_r_op, dot_l_op, dot_r_op, cat_op]);
                }
                created += 5;
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: removed,
            ops_created: created,
        })
    }
}

/// Vision / CNN Lowering Pass (`lower-conv-to-gemm`).
pub struct LowerConvToGemmPass;

impl Pass for LowerConvToGemmPass {
    fn name(&self) -> &'static str {
        "lower-conv-to-gemm"
    }

    fn description(&self) -> &'static str {
        "Lowers stablehlo.convolution into im2col + stablehlo.dot_general GEMM"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Convolution { lhs, rhs, result }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };

                let lhs_dims = static_dims(module, lhs);
                let rhs_dims = static_dims(module, rhs);
                let out_dims = static_dims(module, result);
                let dtype = module.values[result]
                    .ty
                    .as_tensor()
                    .map(|t| t.dtype.clone())
                    .unwrap_or(DType::F32);

                let is_pointwise_2d =
                    lhs_dims.len() == 2 && rhs_dims.len() == 2 && lhs_dims[1] == rhs_dims[0];
                if is_pointwise_2d {
                    let mut dot_attrs = IndexMap::new();
                    dot_attrs.insert(
                        "lhs_contracting_dimensions".to_string(),
                        Attribute::DenseIntArray(vec![1]),
                    );
                    dot_attrs.insert(
                        "rhs_contracting_dimensions".to_string(),
                        Attribute::DenseIntArray(vec![0]),
                    );
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "dot_general");
                    op_mut.operands = SmallVec::from_slice(&[lhs, rhs]);
                    op_mut.attributes = dot_attrs;
                    modified += 1;
                    continue;
                }

                let (n_batch, h_in, w_in, c_in) = match lhs_dims.as_slice() {
                    [h, w, c] => (1, *h, *w, *c),
                    [n, h, w, c] => (*n, *h, *w, *c),
                    [hw, c] => (1, *hw, 1, *c),
                    _ => (1, 1, 1, 1),
                };
                let (kh, kw, k_total, c_out) = match rhs_dims.as_slice() {
                    [kh, kw, cin, cout] => (*kh, *kw, kh * kw * cin, *cout),
                    [kt, cout] => {
                        let spatial = (kt / c_in.max(1)).max(1);
                        let k_side = (spatial as f64).sqrt().round() as i64;
                        (k_side.max(1), (spatial / k_side.max(1)).max(1), *kt, *cout)
                    }
                    _ => (1, 1, c_in, 1),
                };
                let (oh, ow) = match out_dims.as_slice() {
                    [_, oh, ow, _] => (*oh, *ow),
                    [oh, ow, _] => (*oh, *ow),
                    _ => ((h_in - kh + 1).max(1), (w_in - kw + 1).max(1)),
                };
                let out_elems: i64 = out_dims.iter().product::<i64>().max(1);
                let m_patches = (out_elems / c_out.max(1)).max(n_batch);

                let patches_ty = Type::Tensor(TensorType {
                    shape: vec![Dim::Static(m_patches), Dim::Static(k_total)],
                    dtype: dtype.clone(),
                    encoding: None,
                });
                let mut im2col_attrs = IndexMap::new();
                im2col_attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.im2col".to_string()),
                );
                im2col_attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("im2col_impl".to_string()),
                );
                im2col_attrs.insert("h_in".to_string(), Attribute::Int(h_in));
                im2col_attrs.insert("w_in".to_string(), Attribute::Int(w_in));
                im2col_attrs.insert("c_in".to_string(), Attribute::Int(c_in));
                im2col_attrs.insert("kh".to_string(), Attribute::Int(kh));
                im2col_attrs.insert("kw".to_string(), Attribute::Int(kw));
                im2col_attrs.insert("oh".to_string(), Attribute::Int(oh));
                im2col_attrs.insert("ow".to_string(), Attribute::Int(ow));

                let patches_val = insert_op_before(
                    module,
                    block_id,
                    op_id,
                    OpName::new("stablehlo", "composite"),
                    &[lhs],
                    im2col_attrs,
                    patches_ty,
                );
                created += 1;

                let rhs_2d = if rhs_dims.len() == 2 {
                    rhs
                } else {
                    let w2d_ty = Type::Tensor(TensorType {
                        shape: vec![Dim::Static(k_total), Dim::Static(c_out)],
                        dtype: dtype.clone(),
                        encoding: None,
                    });
                    created += 1;
                    insert_op_before(
                        module,
                        block_id,
                        op_id,
                        OpName::new("stablehlo", "reshape"),
                        &[rhs],
                        IndexMap::new(),
                        w2d_ty,
                    )
                };

                let mut dot_attrs = IndexMap::new();
                dot_attrs.insert(
                    "lhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![1]),
                );
                dot_attrs.insert(
                    "rhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![0]),
                );

                if out_dims.len() == 2 {
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "dot_general");
                    op_mut.operands = SmallVec::from_slice(&[patches_val, rhs_2d]);
                    op_mut.attributes = dot_attrs;
                    modified += 1;
                } else {
                    let dot_2d_ty = Type::Tensor(TensorType {
                        shape: vec![Dim::Static(m_patches), Dim::Static(c_out)],
                        dtype,
                        encoding: None,
                    });
                    let dot_2d_val = insert_op_before(
                        module,
                        block_id,
                        op_id,
                        OpName::new("stablehlo", "dot_general"),
                        &[patches_val, rhs_2d],
                        dot_attrs,
                        dot_2d_ty,
                    );
                    created += 1;

                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "reshape");
                    op_mut.operands = SmallVec::from_slice(&[dot_2d_val]);
                    op_mut.attributes.clear();
                    modified += 1;
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn static_dims(module: &Module, val_id: ValueId) -> Vec<i64> {
    module.values[val_id]
        .ty
        .as_tensor()
        .map(|t| {
            t.shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n,
                    _ => 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn insert_op_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    attrs: IndexMap<String, Attribute>,
    out_ty: Type,
) -> ValueId {
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
        operands: SmallVec::from_slice(operands),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(val_id);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, new_op_id);
    } else {
        block_ops.push(new_op_id);
    }
    val_id
}
