use serde::{Deserialize, Serialize};

/// Hardware target descriptor parameterizing cache tiling, SIMD vector width, and GPU workgroup
/// dimensions (inspired by Modular MAX Engine's parameterized target descriptors).
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
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            tile_m: 0,
            tile_n: 0,
            tile_k: 0,
            simd_width: 8,
            wgsl_workgroup_size: 64,
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
            simd_width: 8,
            wgsl_workgroup_size: 64,
        }
    }

    /// Create a WebGPU workgroup-tiled configuration with shared `var<workgroup>` tile memory.
    pub fn gpu_workgroup(tile: usize, workgroup_size: u32) -> Self {
        Self {
            tile_m: tile,
            tile_n: tile,
            tile_k: tile,
            simd_width: 8,
            wgsl_workgroup_size: workgroup_size.max(1),
        }
    }

    /// Returns `true` when 2D/3D loop tiling is enabled.
    pub fn is_tiled(&self) -> bool {
        self.tile_m > 0 && self.tile_n > 0 && self.tile_k > 0
    }
}
