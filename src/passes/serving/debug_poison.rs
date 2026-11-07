//! Relocatable compilation cache-key hashing (`RelocatableCacheKeyHasher`) and dtype-aware debug
//! allocation poisoning (`DebugAllocPoisonTracker`).

use crate::ir::{DType, Module};
use crate::passes::normalize::StripDebugInfoPass;
use crate::passes::registry::Pass;
use crate::passes::serving::module_hashing::normalize_all_names;
use sha2::{Digest, Sha256};

// ============================================================================
// Relocatable Compilation Cache-Key Hasher (`MEFCacheKey`)
// ============================================================================

/// Pipeline configuration options fed into [`RelocatableCacheKeyHasher`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelocatablePipelineKeyOptions {
    /// Textual pipeline specification (e.g., `"canonicalize,cse,fuse-elementwise"`).
    pub textual_pipeline: String,
    /// Host/device target triple or architecture descriptor.
    pub target_descriptor: String,
    /// Compile-time parameter defines (`key=value`).
    pub compile_defines: Vec<(String, String)>,
    /// Optional content-addressed storage cache directory path (stripped to a boolean
    /// `cas_enabled` bit when hashing).
    pub cache_dir: Option<String>,
    /// Machine-local kernel library paths (paths are stripped; only their byte contents are
    /// hashed).
    pub kernel_library_paths: Vec<String>,
    /// Machine-local diagnostic temporary output prefix (stripped before hashing).
    pub save_temp_prefix: String,
    /// Machine-local timing output directory (stripped before hashing).
    pub timing_out_dir: String,
    /// Whether source tracebacks are enabled in debug diagnostics.
    pub source_tracebacks: bool,
}

impl RelocatablePipelineKeyOptions {
    /// Returns a path-stripped copy of these options and whether CAS caching is enabled.
    pub fn stripped_for_relocatable_key(&self) -> (Self, bool) {
        let cas_enabled = self
            .cache_dir
            .as_ref()
            .is_some_and(|d| !d.trim().is_empty());
        let stripped = Self {
            textual_pipeline: self.textual_pipeline.clone(),
            target_descriptor: self.target_descriptor.clone(),
            compile_defines: self.compile_defines.clone(),
            cache_dir: None,
            kernel_library_paths: Vec::new(),
            save_temp_prefix: String::new(),
            timing_out_dir: String::new(),
            source_tracebacks: self.source_tracebacks,
        };
        (stripped, cas_enabled)
    }
}

/// Relocatable compilation cache-key hasher that strips machine-specific install/output paths,
/// hashes kernel package byte payloads rather than file paths, and normalizes debug info and symbol
/// names before hashing the IR module.
#[derive(Debug, Clone)]
pub struct RelocatableCacheKeyHasher {
    prefix_digest: [u8; 32],
}

impl RelocatableCacheKeyHasher {
    /// Builds a prefix hasher from `options` and the in-memory byte contents of any referenced
    /// kernel libraries/packages.
    pub fn build(
        options: &RelocatablePipelineKeyOptions,
        kernel_library_contents: &[&[u8]],
    ) -> Self {
        let (stripped, cas_enabled) = options.stripped_for_relocatable_key();
        let mut hasher = Sha256::new();
        hasher.update(b"castforge-relocatable-cache-key-v1\n");
        hasher.update(format!("pipeline={}\n", stripped.textual_pipeline).as_bytes());
        hasher.update(format!("target={}\n", stripped.target_descriptor).as_bytes());
        hasher.update(format!("cas-enabled={cas_enabled}\n").as_bytes());
        for (k, v) in &stripped.compile_defines {
            hasher.update(format!("define:{k}={v}\n").as_bytes());
        }
        for blob in kernel_library_contents {
            hasher.update((blob.len() as u64).to_le_bytes());
            hasher.update(blob);
        }
        hasher.update(format!("source-tracebacks={}\n", stripped.source_tracebacks).as_bytes());
        let out = hasher.finalize();
        let mut prefix_digest = [0u8; 32];
        prefix_digest.copy_from_slice(&out);
        Self { prefix_digest }
    }

    /// Computes the 64-character hex SHA-256 cache key for `module` by cloning the module,
    /// stripping debug locations (`StripDebugInfoPass`), and normalizing all symbol/SSA names
    /// (`normalize_all_names`).
    pub fn key_for_module(&self, module: &Module) -> Result<String, String> {
        let mut cloned = module.clone();
        let _ = StripDebugInfoPass.run(&mut cloned);
        normalize_all_names(&mut cloned)?;
        let ir_text = crate::ir::print_mlir(&cloned);
        let mut hasher = Sha256::new();
        hasher.update(self.prefix_digest);
        hasher.update((ir_text.len() as u64).to_le_bytes());
        hasher.update(ir_text.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest {
            use std::fmt::Write;
            let _ = write!(&mut hex, "{b:02x}");
        }
        Ok(hex)
    }
}

/// Convenience helper computing a relocatable 64-char hex cache key for `module`.
pub fn compute_relocatable_module_cache_key(
    module: &Module,
    options: &RelocatablePipelineKeyOptions,
    kernel_library_contents: &[&[u8]],
) -> Result<String, String> {
    RelocatableCacheKeyHasher::build(options, kernel_library_contents).key_for_module(module)
}

// ============================================================================
// Debug Allocation Poisoning Engine (`DebugAllocPoison`)
// ============================================================================

/// Bit pattern used to poison or zero-fill newly allocated host/device buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoisonPattern {
    /// Little-endian bit pattern value (`0x7BFF` for `f16`, `0x7F7F_FFFF` for `f32`, `0xCD` for
    /// integers, `0` for zero-fill).
    pub bits: u64,
    /// Width of the repeating pattern in bytes (`1`, `2`, `4`, or `8`).
    pub pattern_bytes: usize,
    /// Human-readable pattern classification (`"largest-finite"`, `"byte-fill"`, or `"zero"`).
    pub pattern_kind: &'static str,
}

/// Zero-fill [`PoisonPattern`] constant (`0x00`, 1 byte, `"zero"`).
pub const ZERO_FILL_PATTERN: PoisonPattern = PoisonPattern {
    bits: 0,
    pattern_bytes: 1,
    pattern_kind: "zero",
};

/// Returns the dtype-specific [`PoisonPattern`] for `dtype`:
/// - Standard IEEE floats (`F16`, `BF16`, `F32`, `F64`) return their `largest-finite` bit pattern.
/// - Sub-byte (`I1`, `I4`) and 8-bit float formats (`F8E4M3FN`, `F8E5M2`) return `None`.
/// - Integer and index types return `0xCD` (`1` byte, `"byte-fill"`).
pub fn get_poison_pattern_for_dtype(dtype: &DType) -> Option<PoisonPattern> {
    match dtype {
        DType::I1 | DType::I4 | DType::F8E4M3FN | DType::F8E5M2 | DType::QuantUniform { .. } => {
            None
        }
        DType::F16 => Some(PoisonPattern {
            bits: 0x7BFF,
            pattern_bytes: 2,
            pattern_kind: "largest-finite",
        }),
        DType::BF16 => Some(PoisonPattern {
            bits: 0x7F7F,
            pattern_bytes: 2,
            pattern_kind: "largest-finite",
        }),
        DType::F32 => Some(PoisonPattern {
            bits: 0x7F7F_FFFF,
            pattern_bytes: 4,
            pattern_kind: "largest-finite",
        }),
        DType::F64 => Some(PoisonPattern {
            bits: 0x7FEF_FFFF_FFFF_FFFF,
            pattern_bytes: 8,
            pattern_kind: "largest-finite",
        }),
        DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::Index => Some(PoisonPattern {
            bits: 0xCD,
            pattern_bytes: 1,
            pattern_kind: "byte-fill",
        }),
    }
}

/// Fills `buf` with `pattern` using unaligned-safe byte copies.
pub fn fill_poison_buffer(buf: &mut [u8], pattern: &PoisonPattern) {
    if pattern.pattern_bytes <= 1 {
        buf.fill(pattern.bits as u8);
        return;
    }
    let pat_len = pattern.pattern_bytes.min(8);
    let pat_bytes = pattern.bits.to_le_bytes();
    let mut off = 0usize;
    while off + pat_len <= buf.len() {
        buf[off..off + pat_len].copy_from_slice(&pat_bytes[..pat_len]);
        off += pat_len;
    }
}

/// Selection mode for [`PoisonSelection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoisonMode {
    /// Poison allocations matching `(dtypes, min_ordinal..=max_ordinal)` and zero-fill the rest.
    Select,
    /// Zero-fill every allocation.
    None,
    /// Leave allocations untouched (`fill = None`), performing synchronization only.
    SyncOnly,
}

/// Parsed allocation poison filter specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoisonSelection {
    /// Poisoning mode.
    pub mode: PoisonMode,
    /// Allowed [`DType`] filter (empty means all dtypes match).
    pub dtypes: Vec<DType>,
    /// Inclusive lower allocation ordinal bound.
    pub min_ordinal: u64,
    /// Inclusive upper allocation ordinal bound.
    pub max_ordinal: u64,
}

impl Default for PoisonSelection {
    fn default() -> Self {
        Self {
            mode: PoisonMode::Select,
            dtypes: Vec::new(),
            min_ordinal: 0,
            max_ordinal: u64::MAX,
        }
    }
}

fn parse_dtype_token(tok: &str) -> Option<DType> {
    match tok.trim().to_ascii_lowercase().as_str() {
        "i1" | "bool" => Some(DType::I1),
        "i8" | "int8" => Some(DType::I8),
        "u8" | "ui8" | "uint8" => Some(DType::U8),
        "i16" | "int16" => Some(DType::I16),
        "u16" | "ui16" | "uint16" => Some(DType::U16),
        "i32" | "int32" | "si32" => Some(DType::I32),
        "u32" | "ui32" | "uint32" => Some(DType::U32),
        "i64" | "int64" | "si64" => Some(DType::I64),
        "u64" | "ui64" | "uint64" => Some(DType::U64),
        "f16" | "float16" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f32" | "float32" => Some(DType::F32),
        "f64" | "float64" => Some(DType::F64),
        "f8e4m3fn" => Some(DType::F8E4M3FN),
        "f8e5m2" => Some(DType::F8E5M2),
        "index" => Some(DType::Index),
        _ => None,
    }
}

fn parse_ordinal_range(val: &str) -> Option<(u64, u64)> {
    let val = val.trim();
    let (lo_s, hi_s) = if let Some((l, r)) = val.split_once('-') {
        (l.trim(), r.trim())
    } else {
        (val, val)
    };
    let lo = lo_s.parse::<u64>().ok()?;
    let hi = hi_s.parse::<u64>().ok()?;
    if lo > hi {
        return None;
    }
    Some((lo, hi))
}

/// Parses an allocation poison selection string (`""`, `"all"`, `"none"`, `"sync-only"`, or `&`-
/// separated clauses `dtype=f32,f16&ord=10-20`).
pub fn parse_poison_selection(spec: &str) -> PoisonSelection {
    let mut sel = PoisonSelection::default();
    let trimmed = spec.trim();
    if trimmed.is_empty() || trimmed == "all" {
        return sel;
    }
    if trimmed == "none" {
        sel.mode = PoisonMode::None;
        return sel;
    }
    if trimmed == "sync-only" {
        sel.mode = PoisonMode::SyncOnly;
        return sel;
    }

    let mut have_dtypes = false;
    let mut have_ordinals = false;
    for clause in trimmed.split('&').filter(|c| !c.trim().is_empty()) {
        let Some((k, v)) = clause.trim().split_once('=') else {
            continue;
        };
        let key = k.trim();
        if key == "dtype" && !have_dtypes {
            let mut parsed = Vec::new();
            let mut all_valid = true;
            for item in v.split(',').filter(|s| !s.trim().is_empty()) {
                if let Some(dt) = parse_dtype_token(item) {
                    parsed.push(dt);
                } else {
                    all_valid = false;
                    break;
                }
            }
            if all_valid && !parsed.is_empty() {
                sel.dtypes = parsed;
                have_dtypes = true;
            }
        } else if key == "ord"
            && !have_ordinals
            && let Some((lo, hi)) = parse_ordinal_range(v)
        {
            sel.min_ordinal = lo;
            sel.max_ordinal = hi;
            have_ordinals = true;
        }
    }
    sel
}

/// Decision returned for a single allocation by [`DebugAllocPoisonTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationPoisonDecision {
    /// Epoch-relative ordinal assigned to this allocation.
    pub ordinal: u64,
    /// Fill pattern to apply (`None` when `SyncOnly` or when the dtype has no poison pattern).
    pub fill: Option<PoisonPattern>,
}

/// Metadata version of a live selection control file (`(unique_id, mtime_ns, size_bytes)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionFileVersion {
    /// Filesystem inode / unique file identifier.
    pub unique_id: u64,
    /// Modification timestamp in nanoseconds.
    pub mtime_ns: u64,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// Stateful allocation poison tracker with file-version epoch resets and ordinal progression.
#[derive(Debug, Clone)]
pub struct DebugAllocPoisonTracker {
    selection: PoisonSelection,
    version: Option<SelectionFileVersion>,
    epoch: u64,
    next_ordinal: u64,
}

impl DebugAllocPoisonTracker {
    /// Creates a tracker initialized with `spec`.
    pub fn new(spec: &str) -> Self {
        Self {
            selection: parse_poison_selection(spec),
            version: None,
            epoch: 0,
            next_ordinal: 0,
        }
    }

    /// Current selection epoch counter.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Next ordinal to be assigned within the current epoch.
    pub fn next_ordinal(&self) -> u64 {
        self.next_ordinal
    }

    /// Updates the selection from a watched file when `new_version != self.version`, advancing
    /// `epoch` and resetting `next_ordinal = 0` even if `new_spec` text is identical.
    pub fn refresh_from_file_version(&mut self, new_version: SelectionFileVersion, new_spec: &str) {
        if self.version != Some(new_version) {
            self.selection = parse_poison_selection(new_spec);
            self.version = Some(new_version);
            self.epoch += 1;
            self.next_ordinal = 0;
        }
    }

    /// Evaluates the poison decision for an allocation of `dtype` and advances `next_ordinal`.
    pub fn decide_for_allocation(&mut self, dtype: &DType) -> AllocationPoisonDecision {
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        let fill = match self.selection.mode {
            PoisonMode::SyncOnly => None,
            PoisonMode::None => Some(ZERO_FILL_PATTERN),
            PoisonMode::Select => {
                let in_ord =
                    ordinal >= self.selection.min_ordinal && ordinal <= self.selection.max_ordinal;
                let in_dtype =
                    self.selection.dtypes.is_empty() || self.selection.dtypes.contains(dtype);
                if in_ord && in_dtype {
                    get_poison_pattern_for_dtype(dtype)
                } else {
                    Some(ZERO_FILL_PATTERN)
                }
            }
        };
        AllocationPoisonDecision { ordinal, fill }
    }
}
