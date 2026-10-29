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

    /// Returns the canonical lowercase string (`"mnk"`, `"mkn"`, `"nkm"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mnk => "mnk",
            Self::Mkn => "mkn",
            Self::Nkm => "nkm",
        }
    }
}

/// Hardware SIMD instruction set architecture for explicit multi-arch intrinsic emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SimdIsa {
    /// Portable scalar / auto-vectorized loops (`#pragma omp simd`).
    #[default]
    Scalar,
    /// x86_64 AVX2 + FMA 256-bit (`8 x f32`) hardware intrinsics (`_mm256_fmadd_ps`).
    Avx2,
    /// x86_64 AVX-512F 512-bit (`16 x f32`) hardware intrinsics (`_mm512_fmadd_ps`).
    Avx512,
    /// ARM64 NEON 128-bit (`4 x f32`) hardware intrinsics (`vfmaq_f32`).
    Neon,
    /// ARM64 SVE/SVE2 scalable vector (`8 x f32`) hardware intrinsics (`svmla_f32_x`).
    Sve,
    /// RISC-V Vector 1.0 (`rv64gcv`, `8 x f32`) hardware intrinsics (`__riscv_vfmacc_vv_f32m1`).
    Rvv,
    /// WebAssembly SIMD128 (`wasm32`, `4 x f32`) hardware intrinsics (`f32x4_mul` + `f32x4_add`).
    WasmSimd128,
}

impl SimdIsa {
    /// Parses a multi-architecture SIMD ISA identifier (`"scalar"`, `"avx2"`, `"avx512"`,
    /// `"neon"`, `"sve"`, `"rvv"`, `"wasm"`).
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "avx512" | "avx-512" | "avx512f" => Self::Avx512,
            "avx2" | "fma" | "x86_64" => Self::Avx2,
            "sve" | "sve2" | "armv9" => Self::Sve,
            "neon" | "aarch64" | "arm64" => Self::Neon,
            "rvv" | "riscv" | "riscv64" => Self::Rvv,
            "wasm" | "wasm32" | "simd128" | "wasm-simd128" => Self::WasmSimd128,
            _ => Self::Scalar,
        }
    }

    /// Canonical lowercase ISA identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::Sve => "sve",
            Self::Rvv => "rvv",
            Self::WasmSimd128 => "wasm_simd128",
        }
    }

    /// Default `f32` vector lane width for this architecture.
    pub fn default_lanes(&self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Avx512 => 16,
            Self::Avx2 | Self::Sve | Self::Rvv => 8,
            Self::Neon | Self::WasmSimd128 => 4,
        }
    }

    /// Primary FMA/MAC hardware intrinsic symbol for this ISA.
    pub fn intrinsic_name(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar_fma",
            Self::Avx2 => "_mm256_fmadd_ps",
            Self::Avx512 => "_mm512_fmadd_ps",
            Self::Neon => "vfmaq_f32",
            Self::Sve => "svmla_f32_x",
            Self::Rvv => "__riscv_vfmacc_vv_f32m1",
            Self::WasmSimd128 => "f32x4_add_mul",
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
    /// Outer L2/L3 cache tile size (`0` disables hierarchical 2-level L2->L1 cache blocking).
    pub l2_tile_size: usize,
    /// Software pipelining / hardware prefetch distance in iterations (`0` disables prefetching).
    pub prefetch_distance: usize,
    /// Split-K reduction partitioning factor (`1` disables Split-K; `> 1` partitions `k`).
    pub split_k: usize,
    /// Enable polyhedral wavefront loop skewing (`w = i + j` diagonal wavefront traversal).
    pub loop_skew: bool,
    /// Enable 2-stage software-pipelined ping-pong tile double-buffering across K tiles.
    pub double_buffer: bool,
    /// Hardware SIMD ISA for explicit intrinsic codegen (`Scalar`, `Avx2`, `Neon`).
    pub simd_isa: SimdIsa,
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
            l2_tile_size: 0,
            prefetch_distance: 0,
            split_k: 1,
            loop_skew: false,
            double_buffer: false,
            simd_isa: SimdIsa::Scalar,
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

    /// Configure hierarchical L2 cache tile size wrapping L1 `tile_m`/`tile_n`/`tile_k` blocks.
    pub fn with_l2_tile_size(mut self, l2_tile: usize) -> Self {
        self.l2_tile_size = l2_tile;
        self
    }

    /// Configure software pipelining / hardware prefetch lookahead distance along `k`.
    pub fn with_prefetch(mut self, distance: usize) -> Self {
        self.prefetch_distance = distance;
        self
    }

    /// Configure Split-K reduction partitioning across `splits` partial accumulators.
    pub fn with_split_k(mut self, splits: usize) -> Self {
        self.split_k = splits.max(1);
        self
    }

    /// Enable or disable polyhedral wavefront loop skewing (`w = i + j`).
    pub fn with_loop_skew(mut self, enabled: bool) -> Self {
        self.loop_skew = enabled;
        self
    }

    /// Enable or disable 2-stage software-pipelined ping-pong tile double-buffering.
    pub fn with_double_buffer(mut self, enabled: bool) -> Self {
        self.double_buffer = enabled;
        self
    }

    /// Select the target hardware SIMD instruction set (`Scalar`, `Avx2`, `Avx512`, `Neon`, `Sve`,
    /// `Rvv`, `WasmSimd128`).
    pub fn with_simd_isa(mut self, isa: SimdIsa) -> Self {
        self.simd_isa = isa;
        if isa != SimdIsa::Scalar {
            self.vectorize = true;
            self.simd_width = isa.default_lanes();
        }
        self
    }

    /// Creates a pre-tuned hardware [`TargetConfig`] for a named CPU or GPU architecture profile
    /// (`"x86_64-avx2"`, `"x86_64-avx512"`, `"aarch64-neon"`, `"aarch64-sve"`, `"riscv64-rvv"`,
    /// `"wasm32-simd128"`, `"cuda-sm80"`, `"rocm-gfx90a"`, `"metal-apple"`, `"webgpu"`).
    pub fn for_arch(arch: &str) -> Self {
        match arch.trim().to_ascii_lowercase().as_str() {
            "x86_64-avx512" | "avx512" => Self::cpu_tiled(32)
                .with_simd_isa(SimdIsa::Avx512)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "x86_64-avx2" | "x86_64" | "avx2" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Avx2)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "aarch64-sve" | "sve" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Sve)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "aarch64-neon" | "aarch64" | "arm64" | "neon" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Neon)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(4)
                .with_parallel(true),
            "riscv64-rvv" | "riscv64" | "rvv" => Self::cpu_tiled(16)
                .with_simd_isa(SimdIsa::Rvv)
                .with_loop_order(GemmLoopOrder::Mkn)
                .with_unroll(2),
            "wasm32-simd128" | "wasm32" | "wasm" => Self::cpu_tiled(8)
                .with_simd_isa(SimdIsa::WasmSimd128)
                .with_loop_order(GemmLoopOrder::Mkn),
            "cuda-sm80" | "cuda" | "rocm-gfx90a" | "hip" => Self::gpu_workgroup(16, 256),
            "metal-apple" | "metal" | "msl" | "opencl" | "webgpu" | "wgsl" => {
                Self::gpu_workgroup(16, 64)
            }
            _ => Self::default(),
        }
    }

    /// Returns `true` when 2D/3D loop tiling is enabled.
    pub fn is_tiled(&self) -> bool {
        self.tile_m > 0 && self.tile_n > 0 && self.tile_k > 0
    }

    /// Returns `true` when `(mr x nr)` register micro-kernel Unroll-and-Jam is enabled.
    pub fn is_register_tiled(&self) -> bool {
        self.reg_tile_m > 1 && self.reg_tile_n > 1
    }

    /// Merges per-operation schedule attributes stamped by IR-to-IR loop passes
    /// (`tile-loops`, `permute-loops`, `unroll-loops`, `vectorize-loops`) into a copy
    /// of this [`TargetConfig`].
    pub fn with_op_attributes(&self, op: &castforge_ir::Operation) -> Self {
        use castforge_ir::Attribute;
        let attrs = &op.attributes;
        let mut cfg = *self;
        if let Some(Attribute::Int(v)) = attrs.get("loop_tile_m") {
            cfg.tile_m = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_tile_n") {
            cfg.tile_n = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_tile_k") {
            cfg.tile_k = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_l2_tile") {
            cfg.l2_tile_size = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_reg_m") {
            cfg.reg_tile_m = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_reg_n") {
            cfg.reg_tile_n = (*v).max(0) as usize;
        }
        if let Some(Attribute::String(s)) = attrs.get("loop_order") {
            cfg.loop_order = GemmLoopOrder::parse(s);
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_collapse") {
            cfg.collapse_loops = *b;
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_skew") {
            cfg.loop_skew = *b;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_unroll") {
            cfg.unroll_factor = (*v).max(1) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_prefetch") {
            cfg.prefetch_distance = (*v).max(0) as usize;
        }
        if let Some(Attribute::Int(v)) = attrs.get("loop_split_k") {
            cfg.split_k = (*v).max(1) as usize;
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_double_buffer") {
            cfg.double_buffer = *b;
        }
        if let Some(Attribute::Bool(b)) = attrs.get("simd_vectorize") {
            cfg.vectorize = *b;
        }
        if let Some(Attribute::Int(v)) = attrs.get("simd_width") {
            cfg.simd_width = (*v).max(1) as usize;
        }
        if let Some(Attribute::String(s)) = attrs.get("simd_isa") {
            cfg = cfg.with_simd_isa(SimdIsa::parse(s));
        }
        if let Some(Attribute::Bool(b)) = attrs.get("loop_parallel") {
            cfg.parallel = *b;
        }
        cfg
    }

    /// Scans `module` for any IR schedule attributes and returns an updated module-level
    /// [`TargetConfig`] (used for emitting top-level SIMD headers or module banners).
    pub fn with_module_attributes(&self, module: &castforge_ir::Module) -> Self {
        let mut cfg = *self;
        for (_, op) in &module.ops {
            cfg = cfg.with_op_attributes(op);
        }
        cfg
    }
}
