//! Target-aware convolution and matmul weight-packing shape materialization functions.
//!
//! Computes register-blocked micro-kernel shapes (`select_conv_micro_kernel_shape`) across
//! 32-register AVX-512, 16-register AVX2 (with output-width and channel residual minimization
//! over `n in 2..=4`), Neoverse-N1 (`{8, 2}`), and 32-register NEON (`{6, 4}`), and materializes
//! 5D/6D packed convolution filter shapes (`materialize_pack_conv_filter_shape` and
//! `materialize_pack_conv_transpose_filter_shape` for `RSCF` and `FCRS` layouts) and 3D/4D packed
//! matmul `B` shapes (`calculate_matmul_tile_nk`, `materialize_pack_matmul_b_shape`) including
//! VNNI (`k_factor = 4`), I8MM (`k_factor = 8`), and macOS Accelerate bypasses.

use crate::ir::DType;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_micro_kernel_and_filter_packing_shapes() {
        let avx512 = PackingTargetDesc::x86_64_avx512(true);
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
        let packed_vnni = materialize_pack_matmul_b_shape(
            &[128, 100],
            false,
            &DType::U8,
            &DType::I8,
            &DType::I32,
            &avx2_vnni,
        )
        .expect("vnni pack");
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
}
