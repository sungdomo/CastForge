//! Target-aware conv/matmul weight packing shape functions and symbolic parameter range inference.
//!
//! Implements two core shape/parameter analysis subsystems:
//! 1. **Target-Aware Weight Packing Shape Functions (`PackingShapeFunctions.cpp`)**:
//!    - Computes register-blocked micro-kernel shapes (`select_conv_micro_kernel_shape`) across
//!      32-register AVX-512, 16-register AVX2 (with output-width and channel residual minimization
//!      over `n in 2..=4`), Neoverse-N1 (`{8, 2}`), and 32-register NEON (`{6, 4}`).
//!    - Materializes 5D/6D packed convolution filter shapes (`materialize_pack_conv_filter_shape`
//!      and `materialize_pack_conv_transpose_filter_shape` for `RSCF` and `FCRS` layouts) and
//!      3D/4D packed matmul `B` shapes (`calculate_matmul_tile_nk`,
//!      `materialize_pack_matmul_b_shape`) including VNNI (`k_factor = 4`), I8MM (`k_factor = 8`),
//!      and macOS Accelerate bypasses.
//! 2. **Symbolic Parameter Range Inference & Algebraic Simplification
//!    (`PropagateShapesPass.cpp`)**:
//!    - Bounds parameter expression nesting depth (`MAX_PARAM_EXPR_NESTING_DEPTH = 4`,
//!      `can_param_operator_be_nested`) and propagates closed 64-bit integer intervals
//!      ([`ConstantIntRange`]) through `Add`, `Mul`, `MulNoWrap`, `Div`, `Mod`, `Shl`, `Shr`,
//!      `And`, `Or`, `Xor`, `Max`, `Min`, `Eq`, `Lt`, `Le`, and `Cond`.
//!    - Simplifies symbolic parameter expressions using range proofs (`<max(add(A, B), B)> ->
//!      <add(A, B)>` when `A, B >= 0`, constant-condition `Cond` folding, single-point range
//!      collapse) and synchronizes enclosing graph/function signatures
//!      (`update_graph_signature_from_body`).

use crate::ir::{Attribute, DType, Module, Type};
use indexmap::IndexMap;

/// Maximum nesting depth allowed when folding or inlining symbolic parameter expressions
/// (`kMaxDepth = 4`).
pub const MAX_PARAM_EXPR_NESTING_DEPTH: usize = 4;

/// Target hardware description used by weight-packing shape functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackingTargetDesc {
    /// Target architecture (`"x86_64"`, `"aarch64"`, `"arm64"`).
    pub arch: String,
    /// Enabled ISA features (e.g. `"avx512f"`, `"avx2"`, `"avxvnni"`, `"neon"`, `"i8mm"`).
    pub features: Vec<String>,
    /// Target microarchitecture CPU name (e.g. `"neoverse-n1"`, `"skylake-avx512"`, `"znver4"`).
    pub cpu_name: String,
    /// Target operating system (`"linux"`, `"macos"`, `"darwin"`).
    pub os_name: String,
}

impl PackingTargetDesc {
    /// Creates an x86_64 AVX-512 target descriptor.
    #[must_use]
    pub fn x86_64_avx512(has_vnni: bool) -> Self {
        let mut features = vec!["avx512f".to_string(), "avx2".to_string()];
        if has_vnni {
            features.push("avx512vnni".to_string());
        }
        Self {
            arch: "x86_64".to_string(),
            features,
            cpu_name: "skylake-avx512".to_string(),
            os_name: "linux".to_string(),
        }
    }

    /// Creates an x86_64 AVX2 target descriptor (16 YMM registers).
    #[must_use]
    pub fn x86_64_avx2(has_vnni: bool) -> Self {
        let mut features = vec!["avx2".to_string()];
        if has_vnni {
            features.push("avxvnni".to_string());
        }
        Self {
            arch: "x86_64".to_string(),
            features,
            cpu_name: "haswell".to_string(),
            os_name: "linux".to_string(),
        }
    }

    /// Creates an AArch64 NEON target descriptor.
    #[must_use]
    pub fn aarch64_neon(cpu_name: &str, has_i8mm: bool, os_name: &str) -> Self {
        let mut features = vec!["neon".to_string()];
        if has_i8mm {
            features.push("i8mm".to_string());
        }
        Self {
            arch: "aarch64".to_string(),
            features,
            cpu_name: cpu_name.to_string(),
            os_name: os_name.to_string(),
        }
    }

    /// Returns `true` if the target is x86_64.
    #[must_use]
    pub fn is_x86_64(&self) -> bool {
        self.arch.eq_ignore_ascii_case("x86_64") || self.arch.eq_ignore_ascii_case("amd64")
    }

    /// Returns `true` if the target is AArch64 / ARM64.
    #[must_use]
    pub fn is_aarch64(&self) -> bool {
        self.arch.eq_ignore_ascii_case("aarch64") || self.arch.eq_ignore_ascii_case("arm64")
    }

    /// Returns `true` if the target has AVX-512 support.
    #[must_use]
    pub fn has_avx512(&self) -> bool {
        self.features
            .iter()
            .any(|f| f.eq_ignore_ascii_case("avx512f") || f.eq_ignore_ascii_case("avx512"))
    }

    /// Returns `true` if the target has AVX2 support.
    #[must_use]
    pub fn has_avx2(&self) -> bool {
        self.features.iter().any(|f| f.eq_ignore_ascii_case("avx2")) || self.has_avx512()
    }

    /// Returns `true` if the target has VNNI (`avxvnni` or `avx512vnni`).
    #[must_use]
    pub fn has_vnni(&self) -> bool {
        self.features.iter().any(|f| {
            f.eq_ignore_ascii_case("avxvnni")
                || f.eq_ignore_ascii_case("avx512vnni")
                || f.eq_ignore_ascii_case("vnni")
        })
    }

    /// Returns `true` if the target has ARMv8.6-A `i8mm` matrix multiply instructions.
    #[must_use]
    pub fn has_i8mm(&self) -> bool {
        self.features.iter().any(|f| f.eq_ignore_ascii_case("i8mm"))
    }

    /// Returns `true` if the target OS is macOS / Darwin.
    #[must_use]
    pub fn is_macos(&self) -> bool {
        self.os_name.eq_ignore_ascii_case("macos") || self.os_name.eq_ignore_ascii_case("darwin")
    }

    /// Returns `true` when Apple Accelerate is used for `c_dtype` (`f32` on macOS).
    #[must_use]
    pub fn use_apple_accelerate(&self, c_dtype: &DType) -> bool {
        self.is_macos() && matches!(c_dtype, DType::F32)
    }

    /// Returns the number of architectural SIMD vector registers (`32` for AVX-512 and AArch64
    /// NEON, `16` for AVX2/SSE).
    #[must_use]
    pub fn num_simd_regs(&self) -> u64 {
        if self.is_x86_64() {
            if self.has_avx512() { 32 } else { 16 }
        } else {
            32
        }
    }

    /// Returns the SIMD vector register width in bits (`512` for AVX-512, `256` for AVX2, `128`
    /// for NEON/SSE).
    #[must_use]
    pub fn simd_bit_width(&self) -> u64 {
        if self.is_x86_64() {
            if self.has_avx512() {
                512
            } else if self.has_avx2() {
                256
            } else {
                128
            }
        } else {
            128
        }
    }
}

/// Source layout of a convolution filter tensor before target-specific packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvFilterLayout {
    /// `[R, S, C_per_group, F]` for 2D conv or `[Q, R, S, C_per_group, F]` for 3D conv.
    Rscf,
    /// `[F, C_per_group, R, S]` for 2D conv or `[F, C_per_group, Q, R, S]` for 3D conv.
    Fcrs,
}

/// Returns the byte size of `dtype` (minimum 1 byte).
#[must_use]
pub fn packing_dtype_byte_size(dtype: &DType) -> u64 {
    (dtype.bit_width() as u64).div_ceil(8).max(1)
}

/// Selects the convolution micro-kernel `(m, n)` blocking parameters for `target`.
///
/// Mirrors `getMicroKernelShape` in `PackingShapeFunctions.cpp`:
/// - x86_64 with 32 SIMD registers (AVX-512): `(6, 3)`.
/// - x86_64 with 16 SIMD registers (AVX2): defaults to `(4, 3)`; when `output_width` (`Wo`) is
///   statically known, searches `n in 2..=4` to minimize wasted vector registers in the channel
///   tail `(F / G) % (n * inner_channel_size)`, then searches `m in 1..=((16 - n - 1) / n)` to
///   maximize `Wo % m` (or exact divisibility).
/// - AArch64 `neoverse-n1`: `(8, 2)`.
/// - AArch64 other NEON (32 SIMD registers): `(6, 4)`.
#[must_use]
pub fn select_conv_micro_kernel_shape(
    target: &PackingTargetDesc,
    output_width: Option<u64>,
    inner_channel_size: u64,
    num_groups: u64,
    out_channels: u64,
) -> (u64, u64) {
    if target.is_x86_64() {
        if target.num_simd_regs() == 32 {
            return (6, 3);
        }
        let mut best_m = 4u64;
        let mut best_n = 3u64;
        let Some(wo) = output_width else {
            return (best_m, best_n);
        };
        if wo == 0 || inner_channel_size == 0 || num_groups == 0 {
            return (best_m, best_n);
        }

        let num_f = out_channels / num_groups;
        let mut max_nr_mod = 0u64;
        for cand_n in 2..=4u64 {
            let block_f = cand_n * inner_channel_size;
            let rem_vecs = (num_f % block_f).div_ceil(inner_channel_size);
            let effective_rem = if rem_vecs == 0 { cand_n } else { rem_vecs };
            if effective_rem >= max_nr_mod {
                max_nr_mod = effective_rem;
                best_n = cand_n;
            }
        }

        let max_m = (16u64.saturating_sub(best_n + 1)) / best_n;
        let mut best_m_rem = 0u64;
        for cand_m in 1..=max_m.max(1) {
            let rem = wo % cand_m;
            let score = if rem == 0 { cand_m } else { rem };
            if score >= best_m_rem {
                best_m_rem = score;
                best_m = cand_m;
            }
        }
        (best_m, best_n)
    } else if target.cpu_name.eq_ignore_ascii_case("neoverse-n1") {
        (8, 2)
    } else {
        (6, 4)
    }
}

/// Materializes the packed filter shape (`FRSCf` for 2D or `FQRSCf` for 3D) for `mo.pack.conv`:
/// - 2D input filter (`rank == 4`): returns `[outer_F, R, S, C_per_group, inner_f]`
/// - 3D input filter (`rank == 5`): returns `[outer_F, Q, R, S, C_per_group, inner_f]`
///
/// Here `inner_f = u_n * (simd_bytes / dtype_bytes)` and
/// `outer_F = num_groups * ceil((F / num_groups) / inner_f)`.
pub fn materialize_pack_conv_filter_shape(
    filter_shape: &[u64],
    num_groups: u64,
    layout: ConvFilterLayout,
    output_width: Option<u64>,
    dtype: &DType,
    target: &PackingTargetDesc,
) -> Result<Vec<u64>, String> {
    if filter_shape.len() != 4 && filter_shape.len() != 5 {
        return Err(format!(
            "conv filter rank must be 4 (2D) or 5 (3D), got {}",
            filter_shape.len()
        ));
    }
    if num_groups == 0 {
        return Err("num_groups must be positive".to_string());
    }

    let (spatial, c_per_group, f_total) = match (layout, filter_shape.len()) {
        (ConvFilterLayout::Rscf, 4) => (
            vec![filter_shape[0], filter_shape[1]],
            filter_shape[2],
            filter_shape[3],
        ),
        (ConvFilterLayout::Rscf, 5) => (
            vec![filter_shape[0], filter_shape[1], filter_shape[2]],
            filter_shape[3],
            filter_shape[4],
        ),
        (ConvFilterLayout::Fcrs, 4) => (
            vec![filter_shape[2], filter_shape[3]],
            filter_shape[1],
            filter_shape[0],
        ),
        (ConvFilterLayout::Fcrs, 5) => (
            vec![filter_shape[2], filter_shape[3], filter_shape[4]],
            filter_shape[1],
            filter_shape[0],
        ),
        _ => unreachable!(),
    };

    if f_total % num_groups != 0 {
        return Err(format!(
            "output channels ({f_total}) must be divisible by num_groups ({num_groups})"
        ));
    }

    let simd_bytes = target.simd_bit_width() / 8;
    let elem_bytes = packing_dtype_byte_size(dtype);
    let simd_lanes = (simd_bytes / elem_bytes).max(1);
    let (_, u_n) =
        select_conv_micro_kernel_shape(target, output_width, simd_lanes, num_groups, f_total);
    let inner_f = u_n * simd_lanes;
    let f_per_group = f_total / num_groups;
    let outer_f = num_groups * f_per_group.div_ceil(inner_f);

    let mut packed = Vec::with_capacity(spatial.len() + 3);
    packed.push(outer_f);
    packed.extend(spatial);
    packed.push(c_per_group);
    packed.push(inner_f);
    Ok(packed)
}

/// Materializes the packed filter shape for `mo.pack.conv_transpose`:
/// - For `ConvFilterLayout::Rscf`, input is `[R, S, F_per_group, C]` (or 5D
///   `[Q, R, S, F_per_group, C]`).
/// - For `ConvFilterLayout::Fcrs`, input is `[C, F_per_group, R, S]` (or 5D
///   `[C, F_per_group, Q, R, S]`).
/// - Uses `inner_f = simd_bytes / dtype_bytes` (1 SIMD vector per block) and returns
///   `[outer_F, R, S, C / num_groups, inner_f]` (or 6D with `Q`).
pub fn materialize_pack_conv_transpose_filter_shape(
    filter_shape: &[u64],
    num_groups: u64,
    layout: ConvFilterLayout,
    dtype: &DType,
    target: &PackingTargetDesc,
) -> Result<Vec<u64>, String> {
    if filter_shape.len() != 4 && filter_shape.len() != 5 {
        return Err(format!(
            "conv_transpose filter rank must be 4 or 5, got {}",
            filter_shape.len()
        ));
    }
    if num_groups == 0 {
        return Err("num_groups must be positive".to_string());
    }

    let (spatial, f_per_group, c_total) = match (layout, filter_shape.len()) {
        (ConvFilterLayout::Rscf, 4) => (
            vec![filter_shape[0], filter_shape[1]],
            filter_shape[2],
            filter_shape[3],
        ),
        (ConvFilterLayout::Rscf, 5) => (
            vec![filter_shape[0], filter_shape[1], filter_shape[2]],
            filter_shape[3],
            filter_shape[4],
        ),
        (ConvFilterLayout::Fcrs, 4) => (
            vec![filter_shape[2], filter_shape[3]],
            filter_shape[1],
            filter_shape[0],
        ),
        (ConvFilterLayout::Fcrs, 5) => (
            vec![filter_shape[2], filter_shape[3], filter_shape[4]],
            filter_shape[1],
            filter_shape[0],
        ),
        _ => unreachable!(),
    };

    if c_total % num_groups != 0 {
        return Err(format!(
            "input channels ({c_total}) must be divisible by num_groups ({num_groups})"
        ));
    }

    let simd_bytes = target.simd_bit_width() / 8;
    let elem_bytes = packing_dtype_byte_size(dtype);
    let inner_f = (simd_bytes / elem_bytes).max(1);
    let c_per_group = c_total / num_groups;
    let outer_f = num_groups * f_per_group.div_ceil(inner_f);

    let mut packed = Vec::with_capacity(spatial.len() + 3);
    packed.push(outer_f);
    packed.extend(spatial);
    packed.push(c_per_group);
    packed.push(inner_f);
    Ok(packed)
}

/// Computes `(n_tile, k_factor)` for matmul `B`-matrix packing on `target`.
///
/// Mirrors `calculateTileNK`, `getVNNIFactor`, and `getI8MMFactor` in `PackingShapeFunctions.cpp`.
#[must_use]
pub fn calculate_matmul_tile_nk(
    target: &PackingTargetDesc,
    a_dtype: &DType,
    b_dtype: &DType,
    c_dtype: &DType,
) -> (u64, u64) {
    if target.use_apple_accelerate(c_dtype) {
        return (1, 1);
    }
    let simd_bytes = target.simd_bit_width() / 8;
    let c_bytes = packing_dtype_byte_size(c_dtype);
    let simd_lanes = (simd_bytes / c_bytes).max(1);

    let is_int8_to_i32 = matches!(a_dtype, DType::I8 | DType::U8)
        && matches!(b_dtype, DType::I8 | DType::U8)
        && matches!(c_dtype, DType::I32);

    let n_tile = if target.is_x86_64() {
        if is_int8_to_i32 && !target.has_avx512() && !target.has_vnni() {
            2 * simd_lanes
        } else if target.has_avx512() && !matches!(a_dtype, DType::F32) {
            4 * simd_lanes
        } else {
            2 * simd_lanes
        }
    } else if target.is_aarch64() {
        if is_int8_to_i32 && target.has_i8mm() {
            2 * simd_lanes
        } else {
            4 * simd_lanes
        }
    } else {
        2 * simd_lanes
    };

    let k_factor = if is_int8_to_i32 {
        if target.is_aarch64() && target.has_i8mm() {
            8
        } else {
            4
        }
    } else {
        1
    };

    (n_tile, k_factor)
}

/// Materializes the packed `B` matrix shape for `mo.pack.matmul`:
/// - Rank 2 `[K, N]` (or `[N, K]` when `transpose_b` is `true`) ->
///   `[ceil(N / n_tile), ceil(K / k_factor), n_tile * k_factor]`
///   (or `[K, N]` on macOS Accelerate `f32`).
/// - Rank 3 `[B, K, N]` (or `[B, N, K]` when `transpose_b` is `true`) ->
///   `[B, ceil(N / n_tile), ceil(K / k_factor), n_tile * k_factor]`.
pub fn materialize_pack_matmul_b_shape(
    b_shape: &[u64],
    transpose_b: bool,
    a_dtype: &DType,
    b_dtype: &DType,
    c_dtype: &DType,
    target: &PackingTargetDesc,
) -> Result<Vec<u64>, String> {
    let (batch_prefix, k, n) = match b_shape.len() {
        2 => {
            let (d0, d1) = (b_shape[0], b_shape[1]);
            if transpose_b {
                (None, d1, d0)
            } else {
                (None, d0, d1)
            }
        }
        3 => {
            let (b, d1, d2) = (b_shape[0], b_shape[1], b_shape[2]);
            if transpose_b {
                (Some(b), d2, d1)
            } else {
                (Some(b), d1, d2)
            }
        }
        r => {
            return Err(format!("matmul B shape rank must be 2 or 3, got {r}"));
        }
    };

    if target.use_apple_accelerate(c_dtype) {
        return Ok(match batch_prefix {
            Some(b) => vec![b, k, n],
            None => vec![k, n],
        });
    }

    let (n_tile, k_factor) = calculate_matmul_tile_nk(target, a_dtype, b_dtype, c_dtype);
    let outer_n = n.div_ceil(n_tile);
    let padded_k = k.div_ceil(k_factor);
    let inner_nk = n_tile * k_factor;

    Ok(match batch_prefix {
        Some(b) => vec![b, outer_n, padded_k, inner_nk],
        None => vec![outer_n, padded_k, inner_nk],
    })
}

// ============================================================================
// Symbolic Parameter Range Inference & Range-Proven Algebraic Simplifier
// (`PropagateShapesPass.cpp`)
// ============================================================================

/// Closed signed 64-bit integer interval `[min, max]` for symbolic parameter range inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantIntRange {
    /// Inclusive lower bound.
    pub min: i64,
    /// Inclusive upper bound.
    pub max: i64,
}

impl ConstantIntRange {
    /// Unconstrained 64-bit signed range `[i64::MIN, i64::MAX]`.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            min: i64::MIN,
            max: i64::MAX,
        }
    }

    /// Point interval `[val, val]`.
    #[must_use]
    pub const fn exact(val: i64) -> Self {
        Self { min: val, max: val }
    }

    /// Non-negative interval `[0, i64::MAX]`.
    #[must_use]
    pub const fn non_negative() -> Self {
        Self {
            min: 0,
            max: i64::MAX,
        }
    }

    /// Strictly positive interval `[1, i64::MAX]`.
    #[must_use]
    pub const fn positive() -> Self {
        Self {
            min: 1,
            max: i64::MAX,
        }
    }

    /// Creates a range `[min, max]`, clamping `max >= min`.
    #[must_use]
    pub const fn new(min: i64, max: i64) -> Self {
        if min <= max {
            Self { min, max }
        } else {
            Self { min: max, max: min }
        }
    }

    /// Returns `Some(val)` when `min == max`.
    #[must_use]
    pub const fn single_value(&self) -> Option<i64> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }

    /// Returns `true` when every value in the range is `>= 0`.
    #[must_use]
    pub const fn is_non_negative(&self) -> bool {
        self.min >= 0
    }

    /// Returns the hull `[self.min.min(other.min), self.max.max(other.max)]`.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Symbolic parameter expression AST supporting all operators handled by `PropagateShapesPass.cpp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicParamExpr {
    /// Constant 64-bit integer literal.
    Const(i64),
    /// Named symbolic parameter (e.g. `"batch"`, `"seq_len"`).
    Sym(String),
    /// Wrapping or general integer addition (`lhs + rhs`).
    Add(Box<Self>, Box<Self>),
    /// General integer multiplication (`lhs * rhs`).
    Mul(Box<Self>, Box<Self>),
    /// No-Signed-Wrap multiplication (`mul_no_wrap(lhs, rhs)`).
    MulNoWrap(Box<Self>, Box<Self>),
    /// Floor/truncating integer division (`lhs / rhs`).
    Div(Box<Self>, Box<Self>),
    /// Remainder (`lhs % rhs`).
    Mod(Box<Self>, Box<Self>),
    /// Left shift (`lhs << rhs`).
    Shl(Box<Self>, Box<Self>),
    /// Arithmetic right shift (`lhs >> rhs`).
    Shr(Box<Self>, Box<Self>),
    /// Bitwise AND (`lhs & rhs`).
    And(Box<Self>, Box<Self>),
    /// Bitwise OR (`lhs | rhs`).
    Or(Box<Self>, Box<Self>),
    /// Bitwise XOR (`lhs ^ rhs`).
    Xor(Box<Self>, Box<Self>),
    /// Maximum (`max(lhs, rhs)`).
    Max(Box<Self>, Box<Self>),
    /// Minimum (`min(lhs, rhs)`).
    Min(Box<Self>, Box<Self>),
    /// Equality comparison (`lhs == rhs`, evaluates to `0` or `1`).
    Eq(Box<Self>, Box<Self>),
    /// Strictly-less-than comparison (`lhs < rhs`, evaluates to `0` or `1`).
    Lt(Box<Self>, Box<Self>),
    /// Less-than-or-equal comparison (`lhs <= rhs`, evaluates to `0` or `1`).
    Le(Box<Self>, Box<Self>),
    /// Ternary conditional (`cond ? then_branch : else_branch`).
    Cond {
        /// Condition expression (non-zero selects `then_branch`, zero selects `else_branch`).
        cond: Box<Self>,
        /// Value when `cond != 0`.
        then_branch: Box<Self>,
        /// Value when `cond == 0`.
        else_branch: Box<Self>,
    },
}

/// Computes the AST tree depth of `expr` (leaf `Const` or `Sym` has depth `0`).
#[must_use]
pub fn param_expr_depth(expr: &SymbolicParamExpr) -> usize {
    match expr {
        SymbolicParamExpr::Const(_) | SymbolicParamExpr::Sym(_) => 0,
        SymbolicParamExpr::Add(a, b)
        | SymbolicParamExpr::Mul(a, b)
        | SymbolicParamExpr::MulNoWrap(a, b)
        | SymbolicParamExpr::Div(a, b)
        | SymbolicParamExpr::Mod(a, b)
        | SymbolicParamExpr::Shl(a, b)
        | SymbolicParamExpr::Shr(a, b)
        | SymbolicParamExpr::And(a, b)
        | SymbolicParamExpr::Or(a, b)
        | SymbolicParamExpr::Xor(a, b)
        | SymbolicParamExpr::Max(a, b)
        | SymbolicParamExpr::Min(a, b)
        | SymbolicParamExpr::Eq(a, b)
        | SymbolicParamExpr::Lt(a, b)
        | SymbolicParamExpr::Le(a, b) => 1 + param_expr_depth(a).max(param_expr_depth(b)),
        SymbolicParamExpr::Cond {
            cond,
            then_branch,
            else_branch,
        } => {
            1 + param_expr_depth(cond)
                .max(param_expr_depth(then_branch))
                .max(param_expr_depth(else_branch))
        }
    }
}

/// Returns `true` when `child` can be nested inside `parent` without exceeding
/// [`MAX_PARAM_EXPR_NESTING_DEPTH`] (`kMaxDepth = 4`).
#[must_use]
pub fn can_param_operator_be_nested(parent: &SymbolicParamExpr, child: &SymbolicParamExpr) -> bool {
    param_expr_depth(parent).saturating_add(param_expr_depth(child)) <= MAX_PARAM_EXPR_NESTING_DEPTH
}

/// Infers a tight [`ConstantIntRange`] for `expr` given symbol bounds `sym_bounds` (symbols not
/// present in `sym_bounds` default to `ConstantIntRange::full()`).
#[must_use]
pub fn infer_param_expr_range(
    expr: &SymbolicParamExpr,
    sym_bounds: &IndexMap<String, ConstantIntRange>,
) -> ConstantIntRange {
    match expr {
        SymbolicParamExpr::Const(v) => ConstantIntRange::exact(*v),
        SymbolicParamExpr::Sym(name) => sym_bounds
            .get(name)
            .copied()
            .unwrap_or_else(ConstantIntRange::full),
        SymbolicParamExpr::Add(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            match (l.min.checked_add(r.min), l.max.checked_add(r.max)) {
                (Some(lo), Some(hi)) => ConstantIntRange::new(lo, hi),
                (Some(lo), None) if l.min >= 0 && r.min >= 0 => ConstantIntRange::new(lo, i64::MAX),
                _ => ConstantIntRange::full(),
            }
        }
        SymbolicParamExpr::Mul(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            let p1 = l.min.checked_mul(r.min);
            let p2 = l.min.checked_mul(r.max);
            let p3 = l.max.checked_mul(r.min);
            let p4 = l.max.checked_mul(r.max);
            if let (Some(a), Some(b), Some(c), Some(d)) = (p1, p2, p3, p4) {
                let lo = a.min(b).min(c).min(d);
                let hi = a.max(b).max(c).max(d);
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::MulNoWrap(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::new(l.min.saturating_mul(r.min), l.max.saturating_mul(r.max))
            } else {
                infer_param_expr_range(
                    &SymbolicParamExpr::Mul(lhs.clone(), rhs.clone()),
                    sym_bounds,
                )
            }
        }
        SymbolicParamExpr::Div(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if r.min > 0 {
                let q1 = l.min / r.min;
                let q2 = l.min / r.max;
                let q3 = l.max / r.min;
                let q4 = l.max / r.max;
                let lo = q1.min(q2).min(q3).min(q4);
                let hi = q1.max(q2).max(q3).max(q4);
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Mod(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0 && r.min > 0 {
                let upper = (r.max - 1).min(l.max);
                ConstantIntRange::new(0, upper)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Shl(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0
                && r.min >= 0
                && r.max < 63
                && let (Some(lo), Some(hi)) = (
                    l.min.checked_shl(r.min as u32),
                    l.max.checked_shl(r.max as u32),
                )
                && lo >= 0
                && hi >= lo
            {
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Shr(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0 && r.min >= 0 && r.max < 63 {
                let lo = l.min >> (r.max as u32);
                let hi = l.max >> (r.min as u32);
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::And(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if let (Some(lv), Some(rv)) = (l.single_value(), r.single_value()) {
                ConstantIntRange::exact(lv & rv)
            } else if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::new(0, l.max.min(r.max))
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Or(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if let (Some(lv), Some(rv)) = (l.single_value(), r.single_value()) {
                ConstantIntRange::exact(lv | rv)
            } else if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::new(l.min.max(r.min), i64::MAX)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Xor(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if let (Some(lv), Some(rv)) = (l.single_value(), r.single_value()) {
                ConstantIntRange::exact(lv ^ rv)
            } else if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::non_negative()
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Max(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            ConstantIntRange::new(l.min.max(r.min), l.max.max(r.max))
        }
        SymbolicParamExpr::Min(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            ConstantIntRange::new(l.min.min(r.min), l.max.min(r.max))
        }
        SymbolicParamExpr::Eq(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min == l.max && r.min == r.max && l.min == r.min {
                ConstantIntRange::exact(1)
            } else if l.max < r.min || r.max < l.min {
                ConstantIntRange::exact(0)
            } else {
                ConstantIntRange::new(0, 1)
            }
        }
        SymbolicParamExpr::Lt(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.max < r.min {
                ConstantIntRange::exact(1)
            } else if l.min >= r.max {
                ConstantIntRange::exact(0)
            } else {
                ConstantIntRange::new(0, 1)
            }
        }
        SymbolicParamExpr::Le(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.max <= r.min {
                ConstantIntRange::exact(1)
            } else if l.min > r.max {
                ConstantIntRange::exact(0)
            } else {
                ConstantIntRange::new(0, 1)
            }
        }
        SymbolicParamExpr::Cond {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = infer_param_expr_range(cond, sym_bounds);
            let t = infer_param_expr_range(then_branch, sym_bounds);
            let e = infer_param_expr_range(else_branch, sym_bounds);
            if c.min == 0 && c.max == 0 {
                e
            } else if c.min > 0 || c.max < 0 {
                t
            } else {
                t.union(&e)
            }
        }
    }
}

/// Simplifies `expr` using range analysis over `sym_bounds`:
/// - Collapses any expression whose inferred [`ConstantIntRange`] is a single value `[v, v]` to
///   `SymbolicParamExpr::Const(v)`.
/// - Simplifies `<max(add(A, B), B)> -> <add(A, B)>` (and commutative variants) when both `A` and
///   `B` are provably non-negative (`min >= 0`).
/// - Simplifies `<min(add(A, B), B)> -> <B>` when both `A` and `B` are provably non-negative.
/// - Simplifies `<max(L, R)>` / `<min(L, R)>` when one operand's range dominates the other
///   (`l_range.min >= r_range.max`).
/// - Collapses `Cond` expressions whose condition range cannot be zero (`then_branch`), is
///   identically zero (`else_branch`), or has identical branches.
#[must_use]
pub fn simplify_param_expr_with_ranges(
    expr: &SymbolicParamExpr,
    sym_bounds: &IndexMap<String, ConstantIntRange>,
) -> SymbolicParamExpr {
    let whole_range = infer_param_expr_range(expr, sym_bounds);
    if let Some(c) = whole_range.single_value() {
        return SymbolicParamExpr::Const(c);
    }

    match expr {
        SymbolicParamExpr::Const(_) | SymbolicParamExpr::Sym(_) => expr.clone(),
        SymbolicParamExpr::Add(a, b) => {
            let sa = simplify_param_expr_with_ranges(a, sym_bounds);
            let sb = simplify_param_expr_with_ranges(b, sym_bounds);
            if sa == SymbolicParamExpr::Const(0) {
                return sb;
            }
            if sb == SymbolicParamExpr::Const(0) {
                return sa;
            }
            SymbolicParamExpr::Add(Box::new(sa), Box::new(sb))
        }
        SymbolicParamExpr::Mul(a, b) => {
            let sa = simplify_param_expr_with_ranges(a, sym_bounds);
            let sb = simplify_param_expr_with_ranges(b, sym_bounds);
            if sa == SymbolicParamExpr::Const(1) {
                return sb;
            }
            if sb == SymbolicParamExpr::Const(1) {
                return sa;
            }
            SymbolicParamExpr::Mul(Box::new(sa), Box::new(sb))
        }
        SymbolicParamExpr::MulNoWrap(a, b) => {
            let sa = simplify_param_expr_with_ranges(a, sym_bounds);
            let sb = simplify_param_expr_with_ranges(b, sym_bounds);
            if sa == SymbolicParamExpr::Const(1) {
                return sb;
            }
            if sb == SymbolicParamExpr::Const(1) {
                return sa;
            }
            SymbolicParamExpr::MulNoWrap(Box::new(sa), Box::new(sb))
        }
        SymbolicParamExpr::Max(lhs, rhs) => {
            let sl = simplify_param_expr_with_ranges(lhs, sym_bounds);
            let sr = simplify_param_expr_with_ranges(rhs, sym_bounds);
            if sl == sr {
                return sl;
            }
            // Check `<max(add(A, B), B)> -> <add(A, B)>` when `A >= 0` and `B >= 0`.
            if let SymbolicParamExpr::Add(a, b) = &sl
                && (**b == sr || **a == sr)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sl;
            }
            if let SymbolicParamExpr::Add(a, b) = &sr
                && (**b == sl || **a == sl)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sr;
            }
            let lr = infer_param_expr_range(&sl, sym_bounds);
            let rr = infer_param_expr_range(&sr, sym_bounds);
            if lr.min >= rr.max {
                return sl;
            }
            if rr.min >= lr.max {
                return sr;
            }
            SymbolicParamExpr::Max(Box::new(sl), Box::new(sr))
        }
        SymbolicParamExpr::Min(lhs, rhs) => {
            let sl = simplify_param_expr_with_ranges(lhs, sym_bounds);
            let sr = simplify_param_expr_with_ranges(rhs, sym_bounds);
            if sl == sr {
                return sl;
            }
            // Check `<min(add(A, B), B)> -> <B>` when `A >= 0` and `B >= 0`.
            if let SymbolicParamExpr::Add(a, b) = &sl
                && (**b == sr || **a == sr)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sr;
            }
            if let SymbolicParamExpr::Add(a, b) = &sr
                && (**b == sl || **a == sl)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sl;
            }
            let lr = infer_param_expr_range(&sl, sym_bounds);
            let rr = infer_param_expr_range(&sr, sym_bounds);
            if lr.max <= rr.min {
                return sl;
            }
            if rr.max <= lr.min {
                return sr;
            }
            SymbolicParamExpr::Min(Box::new(sl), Box::new(sr))
        }
        SymbolicParamExpr::Cond {
            cond,
            then_branch,
            else_branch,
        } => {
            let sc = simplify_param_expr_with_ranges(cond, sym_bounds);
            let st = simplify_param_expr_with_ranges(then_branch, sym_bounds);
            let se = simplify_param_expr_with_ranges(else_branch, sym_bounds);
            if st == se {
                return st;
            }
            let cr = infer_param_expr_range(&sc, sym_bounds);
            if cr.min == 0 && cr.max == 0 {
                return se;
            }
            if cr.min > 0 || cr.max < 0 {
                return st;
            }
            SymbolicParamExpr::Cond {
                cond: Box::new(sc),
                then_branch: Box::new(st),
                else_branch: Box::new(se),
            }
        }
        SymbolicParamExpr::Div(a, b) => SymbolicParamExpr::Div(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Mod(a, b) => SymbolicParamExpr::Mod(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Shl(a, b) => SymbolicParamExpr::Shl(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Shr(a, b) => SymbolicParamExpr::Shr(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::And(a, b) => SymbolicParamExpr::And(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Or(a, b) => SymbolicParamExpr::Or(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Xor(a, b) => SymbolicParamExpr::Xor(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Eq(a, b) => SymbolicParamExpr::Eq(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Lt(a, b) => SymbolicParamExpr::Lt(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Le(a, b) => SymbolicParamExpr::Le(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
    }
}

/// Synchronizes the outer signature of every `mo.graph` / `func.func` operation in `module` with
/// the refined types of its entry block arguments and `mo.output` / `func.return` terminator
/// operands (`updateGraphSignature` in `PropagateShapesPass.cpp`).
///
/// Returns the number of graph/function signatures updated.
pub fn update_graph_signature_from_body(module: &mut Module) -> usize {
    let mut updated_count = 0usize;
    let op_ids: Vec<_> = module.ops.keys().collect();

    for op_id in op_ids {
        let Some(op) = module.ops.get(op_id).cloned() else {
            continue;
        };
        let full_name = op.name.full_name();
        if !matches!(full_name.as_str(), "mo.graph" | "func.func" | "mgp.model") {
            continue;
        }
        let Some(&reg_id) = op.regions.first() else {
            continue;
        };
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        let Some(&blk_id) = reg.blocks.first() else {
            continue;
        };
        let Some(blk) = module.blocks.get(blk_id).cloned() else {
            continue;
        };

        let arg_types: Vec<Type> = blk
            .arguments
            .iter()
            .filter_map(|&v| module.values.get(v).map(|vi| vi.ty.clone()))
            .collect();

        let mut ret_types: Option<Vec<Type>> = None;
        if let Some(&last_op_id) = blk.ops.last()
            && let Some(term_op) = module.ops.get(last_op_id)
            && matches!(
                term_op.name.full_name().as_str(),
                "mo.output" | "func.return" | "mgp.yield"
            )
        {
            let tys: Vec<Type> = term_op
                .operands
                .iter()
                .filter_map(|&v| module.values.get(v).map(|vi| vi.ty.clone()))
                .collect();
            ret_types = Some(tys);
        }

        let mut changed = false;
        let new_arg_attr = Attribute::Array(
            arg_types
                .iter()
                .map(|t| Attribute::Type(t.clone()))
                .collect(),
        );
        if op.attributes.get("argument_types") != Some(&new_arg_attr) {
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut
                    .attributes
                    .insert("argument_types".to_string(), new_arg_attr);
            }
            changed = true;
        }

        if let Some(r_types) = ret_types {
            let new_res_attr =
                Attribute::Array(r_types.iter().map(|t| Attribute::Type(t.clone())).collect());
            if op.attributes.get("result_types") != Some(&new_res_attr) {
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut
                        .attributes
                        .insert("result_types".to_string(), new_res_attr);
                }
                changed = true;
            }
            for (&res_val, new_ty) in op.results.iter().zip(r_types.iter()) {
                if let Some(v_mut) = module.values.get_mut(res_val)
                    && &v_mut.ty != new_ty
                {
                    v_mut.ty = new_ty.clone();
                    changed = true;
                }
            }
        }

        if changed {
            updated_count += 1;
        }
    }

    updated_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_micro_kernel_and_filter_packing_shapes() {
        let avx512 = PackingTargetDesc::x86_64_avx512(true);
        // 2D RSCF filter [3, 3, 64, 128], G = 1, f32 (16 lanes per ZMM, u_n = 3 -> inner_f = 48).
        // outer_F = ceil(128 / 48) = 3 -> [3, 3, 3, 64, 48].
        let packed_avx512 = materialize_pack_conv_filter_shape(
            &[3, 3, 64, 128],
            1,
            ConvFilterLayout::Rscf,
            Some(28),
            &DType::F32,
            &avx512,
        )
        .expect("packed shape");
        assert_eq!(packed_avx512, vec![3, 3, 3, 64, 48]);

        // Neoverse-N1 AArch64: micro-kernel is (8, 2), NEON 128b = 4 f32 lanes -> inner_f = 8.
        let neoverse = PackingTargetDesc::aarch64_neon("neoverse-n1", true, "linux");
        assert_eq!(
            select_conv_micro_kernel_shape(&neoverse, Some(14), 4, 1, 64),
            (8, 2)
        );
        let packed_n1 = materialize_pack_conv_filter_shape(
            &[64, 32, 3, 3],
            1,
            ConvFilterLayout::Fcrs,
            Some(14),
            &DType::F32,
            &neoverse,
        )
        .expect("neoverse packed shape");
        assert_eq!(packed_n1, vec![8, 3, 3, 32, 8]);
    }

    #[test]
    fn test_matmul_b_packing_vnni_and_i8mm() {
        let avx2_vnni = PackingTargetDesc::x86_64_avx2(true);
        // AVX2 (256-bit -> 8 i32 lanes): n_tile = 16, VNNI k_factor = 4 -> inner_nk = 64.
        let packed_vnni = materialize_pack_matmul_b_shape(
            &[128, 100],
            false,
            &DType::U8,
            &DType::I8,
            &DType::I32,
            &avx2_vnni,
        )
        .expect("vnni pack");
        // outer_n = ceil(100 / 16) = 7, padded_k = ceil(128 / 4) = 32, inner = 64.
        assert_eq!(packed_vnni, vec![7, 32, 64]);

        // AArch64 with I8MM: 128-bit -> 4 i32 lanes, n_tile = 8, k_factor = 8 -> inner = 64.
        let arm_i8mm = PackingTargetDesc::aarch64_neon("neoverse-v1", true, "linux");
        let packed_i8mm = materialize_pack_matmul_b_shape(
            &[4, 64, 48],
            false,
            &DType::I8,
            &DType::I8,
            &DType::I32,
            &arm_i8mm,
        )
        .expect("i8mm pack");
        assert_eq!(packed_i8mm, vec![4, 6, 8, 64]);
    }

    #[test]
    fn test_symbolic_param_range_inference_and_simplification() {
        let mut bounds = IndexMap::new();
        bounds.insert("A".to_string(), ConstantIntRange::non_negative());
        bounds.insert("B".to_string(), ConstantIntRange::positive());

        let a = SymbolicParamExpr::Sym("A".to_string());
        let b = SymbolicParamExpr::Sym("B".to_string());
        let add_ab = SymbolicParamExpr::Add(Box::new(a.clone()), Box::new(b.clone()));
        let max_expr = SymbolicParamExpr::Max(Box::new(add_ab.clone()), Box::new(b.clone()));

        // Since A >= 0 and B >= 1, <max(add(A, B), B)> simplifies to <add(A, B)>.
        let simplified = simplify_param_expr_with_ranges(&max_expr, &bounds);
        assert_eq!(simplified, add_ab);

        // Conditional with strictly positive condition B > 0 folds to `then_branch`.
        let cond_expr = SymbolicParamExpr::Cond {
            cond: Box::new(b.clone()),
            then_branch: Box::new(a.clone()),
            else_branch: Box::new(SymbolicParamExpr::Const(99)),
        };
        assert_eq!(simplify_param_expr_with_ranges(&cond_expr, &bounds), a);
    }
}
