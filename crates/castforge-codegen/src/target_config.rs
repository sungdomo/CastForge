//! Hardware target and loop-transformation configuration (`TargetConfig`, `GemmLoopOrder`).

use serde::{Deserialize, Serialize};

/// Loop permutation order for 2D Matrix Multiplication (`M`, `N`, `K`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GemmLoopOrder {
    /// Classic inner-product reduction order (`for m { for n { for k } }`).
    #[default]
    Mnk,
    /// Broadcast-A / row-streaming outer-product order (`for m { for k { for n } }`),
    /// providing unit-stride `n` access across unpacked `B[k, n]` and `C[m, n]`.
    Mkn,
    /// Column-stationary order (`for n { for k { for m } }`).
    Nkm,
}

impl GemmLoopOrder {
    /// Parses a loop order string (`"mnk"`, `"mkn"`, `"nkm"`).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "mkn" | "ikj" => Self::Mkn,
            "nkm" | "jki" => Self::Nkm,
            _ => Self::Mnk,
        }
    }
}

/// Hardware target descriptor parameterizing cache tiling, loop permutation, unrolling,
/// SIMD vectorization, and GPU workgroup dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetConfig {
    /// Cache / register tile size along the M (row) dimension (`0` disables loop tiling).
    pub tile_m: usize,
    /// Cache / register tile size along the N (column) dimension (`0` disables loop tiling).
    pub tile_n: usize,
    /// Cache tile size along the K (reduction) dimension (`0` disables loop tiling).
    pub tile_k: usize,
    /// Target SIMD vector width in `f32` lanes (e.g., `8` for AVX2, `4` for NEON).
    pub simd_width: usize,
    /// WebGPU `@workgroup_size(X)` thread count per compute workgroup.
    pub wgsl_workgroup_size: u32,
    /// GEMM loop permutation order (`Mnk`, `Mkn`, `Nkm`).
    pub loop_order: GemmLoopOrder,
    /// Inner-loop unroll factor (`1` disables unrolling; `2`, `4`, `8` unroll with tail peeling).
    pub unroll_factor: usize,
    /// Emit explicit SIMD vectorization loops/pragmas with tail-loop peeling.
    pub vectorize: bool,
    /// Register micro-kernel row blocking factor (`mr` for Unroll-and-Jam; `0` or `1` = disabled).
    pub reg_tile_m: usize,
    /// Register micro-kernel col blocking factor (`nr` for Unroll-and-Jam; `0` or `1` = disabled).
    pub reg_tile_n: usize,
    /// Collapse/coalesce perfectly nested 2D loops into a single 1D loop.
    pub collapse_loops: bool,
    /// Emit multicore parallel outer-loop work-sharing (`#pragma omp parallel for`).
    pub parallel: bool,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            tile_m: 0,
            tile_n: 0,
            tile_k: 0,
            simd_width: 8,
            wgsl_workgroup_size: 64,
            loop_order: GemmLoopOrder::Mnk,
            unroll_factor: 1,
            vectorize: false,
            reg_tile_m: 0,
            reg_tile_n: 0,
            collapse_loops: false,
            parallel: false,
        }
    }
}

impl TargetConfig {
    /// Create a symmetric `(tile x tile x tile)` cache-blocked CPU configuration.
    pub fn cpu_tiled(tile: usize) -> Self {
        Self {
            tile_m: tile,
            tile_n: tile,
            tile_k: tile,
            ..Self::default()
        }
    }

    /// Create a WebGPU workgroup-tiled configuration with shared `var<workgroup>` tile memory.
    pub fn gpu_workgroup(tile: usize, workgroup_size: u32) -> Self {
        Self {
            tile_m: tile,
            tile_n: tile,
            tile_k: tile,
            wgsl_workgroup_size: workgroup_size.max(1),
            ..Self::default()
        }
    }

    /// Configure the GEMM loop permutation order (`Mnk`, `Mkn`, `Nkm`).
    pub fn with_loop_order(mut self, order: GemmLoopOrder) -> Self {
        self.loop_order = order;
        self
    }

    /// Configure the inner-loop unroll factor (`1` = disabled, `2`/`4`/`8` = unrolled + peeled).
    pub fn with_unroll(mut self, factor: usize) -> Self {
        self.unroll_factor = factor.max(1);
        self
    }

    /// Enable or disable explicit SIMD vectorization and tail peeling.
    pub fn with_vectorize(mut self, enabled: bool) -> Self {
        self.vectorize = enabled;
        self
    }

    /// Configure `(mr x nr)` outer-loop Unroll-and-Jam register micro-kernel blocking.
    pub fn with_register_tile(mut self, mr: usize, nr: usize) -> Self {
        self.reg_tile_m = mr;
        self.reg_tile_n = nr;
        self
    }

    /// Enable or disable 2D -> 1D loop coalescing / flattening.
    pub fn with_collapse_loops(mut self, enabled: bool) -> Self {
        self.collapse_loops = enabled;
        self
    }

    /// Enable or disable multicore parallel outer-loop work-sharing.
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Returns `true` when 2D/3D loop tiling is enabled.
    pub fn is_tiled(&self) -> bool {
        self.tile_m > 0 && self.tile_n > 0 && self.tile_k > 0
    }

    /// Returns `true` when `(mr x nr)` register micro-kernel Unroll-and-Jam is enabled.
    pub fn is_register_tiled(&self) -> bool {
        self.reg_tile_m > 1 && self.reg_tile_n > 1
    }
}
