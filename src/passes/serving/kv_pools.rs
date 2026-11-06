//! CacheGen arithmetic CDF codec, CacheBlend retrieve waves, ARC eviction, and multi-tenant
//! salt quota registries.

use super::kv_attention::{KvDmaCopyEntry, stamp_fused_provenance};
use crate::ir::{Attribute, DType, Dim, Location, Module, OpId, OpName, Type, ValueDef, ValueId};
use crate::passes::Pass;
use indexmap::IndexMap;
use smallvec::SmallVec;

fn anonymize_type_dims(ty: &mut Type) -> usize {
    match ty {
        Type::Tensor(t) => {
            let mut count = 0usize;
            for dim in &mut t.shape {
                if matches!(dim, Dim::Symbolic(_)) {
                    *dim = Dim::Dynamic;
                    count += 1;
                }
            }
            count
        }
        Type::Tuple(items) => items.iter_mut().map(anonymize_type_dims).sum(),
        Type::Function { inputs, results } => {
            let a: usize = inputs.iter_mut().map(anonymize_type_dims).sum();
            let b: usize = results.iter_mut().map(anonymize_type_dims).sum();
            a + b
        }
        _ => 0,
    }
}

/// Strips symbolic dimension names (`Dim::Symbolic(name)` -> `Dim::Dynamic`) across all SSA
/// values and function signatures in `module`.
///
/// Idempotent: returns the number of symbolic dimension slots anonymized.
pub fn anonymize_symbolic_dims(module: &mut Module) -> usize {
    let mut count = 0usize;
    for (_, val) in module.values.iter_mut() {
        count += anonymize_type_dims(&mut val.ty);
    }
    for (_, op) in module.ops.iter_mut() {
        for (_, attr) in op.attributes.iter_mut() {
            if let Attribute::Type(ty) = attr {
                count += anonymize_type_dims(ty);
            }
        }
    }
    count
}

/// Deduplicates structurally identical `func.func` definitions in `module` modulo `@sym_name`,
/// SSA value numbering, and source locations.
///
/// For each equivalence class of 2 or more functions:
/// 1. Selects a deterministic canonical symbol (`@main` / `@forward` if present, otherwise the
///    lexicographically smallest symbol name),
/// 2. Fuses all duplicate functions' [`Location`]s into the canonical function via
///    [`stamp_fused_provenance`],
/// 3. Rewrites all `func.call` callee attributes across `module` to reference the canonical
///    symbol, and
/// 4. Removes the duplicate `func.func` operations from `module`.
///
/// Returns the number of duplicate functions merged and eliminated.
pub fn merge_identical_functions(module: &mut Module) -> usize {
    let fns = module.functions();
    if fns.len() <= 1 {
        return 0;
    }

    // Group functions by (signature_string, body_structural_hash)
    type FuncStubEntry = (OpId, String, Location);
    let mut groups: IndexMap<(String, u64), Vec<FuncStubEntry>> = IndexMap::new();
    for &fn_id in &fns {
        let Some(op) = module.ops.get(fn_id) else {
            continue;
        };
        let Some(sym_name) = op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        let Some(&reg_id) = op.regions.first() else {
            continue;
        };
        let sig_str = op
            .attributes
            .get("function_type")
            .map(|a| a.to_string())
            .unwrap_or_default();
        let body_hash = crate::codegen::compute_region_structural_hash(module, reg_id);
        groups.entry((sig_str, body_hash)).or_default().push((
            fn_id,
            sym_name,
            op.location.clone(),
        ));
    }

    let mut replacements: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut dead_fn_ops: std::collections::HashSet<OpId> = std::collections::HashSet::new();

    for (_, mut candidates) in groups {
        if candidates.len() <= 1 {
            continue;
        }
        // Sort deterministically: prefer entry points (`main`/`forward`), then lexicographical name
        candidates.sort_by(|a, b| {
            let a_entry = matches!(a.1.as_str(), "main" | "forward");
            let b_entry = matches!(b.1.as_str(), "main" | "forward");
            b_entry.cmp(&a_entry).then_with(|| a.1.cmp(&b.1))
        });

        let (canon_id, canon_name, _) = candidates[0].clone();
        let all_locs: Vec<Location> = candidates.iter().map(|(_, _, loc)| loc.clone()).collect();
        if let Some(canon_op) = module.ops.get_mut(canon_id) {
            stamp_fused_provenance(canon_op, &all_locs);
        }

        for (dup_id, dup_name, _) in candidates.into_iter().skip(1) {
            replacements.insert(dup_name, canon_name.clone());
            dead_fn_ops.insert(dup_id);
        }
    }

    if dead_fn_ops.is_empty() {
        return 0;
    }

    // Rewrite `func.call` callee attributes across all remaining operations
    for (_, op) in module.ops.iter_mut() {
        if op.is_op("func", "call")
            && let Some(callee) = op.attributes.get("callee").and_then(|a| a.as_str())
            && let Some(new_callee) = replacements.get(callee)
        {
            op.attributes.insert(
                "callee".to_string(),
                Attribute::SymbolRef(new_callee.clone()),
            );
        }
    }

    // Remove duplicate functions from the module root block and ops arena
    if let Some(root_reg) = module.regions.get(module.root_region)
        && let Some(&root_blk) = root_reg.blocks.first()
    {
        module.blocks[root_blk]
            .ops
            .retain(|id| !dead_fn_ops.contains(id));
    }
    let merged_count = dead_fn_ops.len();
    for dead_id in dead_fn_ops {
        module.ops.remove(dead_id);
    }
    merged_count
}

/// Fuses elementwise epilogue consumers across an intervening `stablehlo.reshape` into a
/// preceding `dot_general` or `fused_matmul_epilogue`.
///
/// Moves the bijective `reshape` past sole-use unary (`tanh`, `abs`, `negate`, `exponential`,
/// `log`, `logistic`, `sqrt`, `rsqrt`) or binary (`add`, `multiply`, `maximum`) elementwise
/// operations and invokes [`crate::passes::FuseMatmulEpiloguePass`] so `matmul -> reshape ->
/// bias -> activation` collapses into `fused_matmul_epilogue -> reshape`.
///
/// Returns the number of reshape-epilogue commutations performed.
pub fn fuse_reshape_epilogue(module: &mut Module) -> usize {
    let mut total_commuted = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };

        loop {
            let mut commuted_in_iter = false;
            let ops = module.blocks[block_id].ops.clone();

            for &cons_id in &ops {
                let Some(cons_op) = module.ops.get(cons_id).cloned() else {
                    continue;
                };
                if cons_op.name.dialect != "stablehlo" || cons_op.results.len() != 1 {
                    continue;
                }
                let op_short = cons_op.name.name.as_str();
                let is_unary = cons_op.operands.len() == 1
                    && matches!(
                        op_short,
                        "tanh"
                            | "abs"
                            | "negate"
                            | "exponential"
                            | "log"
                            | "logistic"
                            | "sqrt"
                            | "rsqrt"
                    );
                let is_binary = cons_op.operands.len() == 2
                    && matches!(op_short, "add" | "multiply" | "maximum");
                if !is_unary && !is_binary {
                    continue;
                }

                // Check if operand 0 is a sole-use `stablehlo.reshape` fed by GEMM/epilogue/add
                let r_val = cons_op.operands[0];
                if module.use_count(r_val) != 1 {
                    continue;
                }
                let ValueDef::OpResult { op: resh_id, .. } = module.values[r_val].def else {
                    continue;
                };
                let Some(resh_op) = module.ops.get(resh_id).cloned() else {
                    continue;
                };
                if !resh_op.is_op("stablehlo", "reshape") || resh_op.operands.len() != 1 {
                    continue;
                }
                let src_val = resh_op.operands[0];
                let src_ty = module.values[src_val].ty.clone();
                let ValueDef::OpResult { op: prod_id, .. } = module.values[src_val].def else {
                    continue;
                };
                let Some(prod_op) = module.ops.get(prod_id) else {
                    continue;
                };
                let prod_eligible = prod_op.is_op("stablehlo", "dot_general")
                    || prod_op.is_op("stablehlo", "add")
                    || prod_op.is_op("stablehlo", "composite");
                if !prod_eligible {
                    continue;
                }

                // For binary ops, resolve operand 1 in pre-reshape `src_ty` space
                let mut new_operands: SmallVec<[ValueId; 4]> = smallvec::smallvec![src_val];
                let mut dead_other_resh: Option<OpId> = None;
                if is_binary {
                    let other_val = cons_op.operands[1];
                    let ValueDef::OpResult { op: other_def, .. } = module.values[other_val].def
                    else {
                        continue;
                    };
                    let Some(other_op) = module.ops.get(other_def).cloned() else {
                        continue;
                    };
                    if other_op.is_op("stablehlo", "reshape")
                        && other_op.operands.len() == 1
                        && module.values[other_op.operands[0]].ty == src_ty
                    {
                        new_operands.push(other_op.operands[0]);
                        if module.use_count(other_val) == 1 {
                            dead_other_resh = Some(other_def);
                        }
                    } else if other_op.is_op("stablehlo", "constant")
                        && module.use_count(other_val) == 1
                    {
                        // Retype sole-use splat/dense constant to `src_ty`
                        module.values[other_val].ty = src_ty.clone();
                        new_operands.push(other_val);
                    } else {
                        continue;
                    }
                }

                // Commute in-place:
                // - `resh_id` becomes the elementwise op (`cons_op.name`) on `src_ty` -> `r_val`
                // - `cons_id` becomes `stablehlo.reshape` on `[r_val]` -> `cons_op.results[0]`
                module.values[r_val].ty = src_ty;
                let merged_locs = [resh_op.location.clone(), cons_op.location.clone()];
                if let Some(elem_mut) = module.ops.get_mut(resh_id) {
                    elem_mut.name = cons_op.name.clone();
                    elem_mut.operands = new_operands;
                    elem_mut.attributes = cons_op.attributes.clone();
                    stamp_fused_provenance(elem_mut, &merged_locs);
                }
                if let Some(resh_mut) = module.ops.get_mut(cons_id) {
                    resh_mut.name = OpName::new("stablehlo", "reshape");
                    resh_mut.operands = smallvec::smallvec![r_val];
                    resh_mut.attributes = resh_op.attributes.clone();
                    stamp_fused_provenance(resh_mut, &merged_locs);
                }
                if let Some(dead_id) = dead_other_resh {
                    module.remove_op(block_id, dead_id);
                }
                total_commuted += 1;
                commuted_in_iter = true;
                break;
            }

            if !commuted_in_iter {
                break;
            }
        }
    }

    if total_commuted > 0 {
        let _ = crate::passes::FuseMatmulEpiloguePass.run(module);
    }
    total_commuted
}

/// CacheGen 16-bit strictly monotonic Cumulative Distribution Function (CDF) and lossless
/// arithmetic range codec for quantized KV-cache token streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheGenCdfCodec {
    pub max_bins: usize,
}

impl CacheGenCdfCodec {
    pub const MAX_BINS_SUPPORTED: usize = 64;
    pub const PRECISION_BITS: u32 = 16;

    pub fn new(max_bins: usize) -> Self {
        Self {
            max_bins: max_bins.clamp(2, Self::MAX_BINS_SUPPORTED),
        }
    }

    /// Computes the normalized 16-bit strictly monotonic CDF of length `max_bins + 1` across
    /// `symbols` in a single `(layer, channel)` stream using monotonic `normalize_cdf_value`
    /// (`floor((0xFFFF - max_bins) * cum / ntokens) + i`), guaranteeing every bin has width
    /// `>= 1`.
    pub fn compute_channel_cdf(&self, symbols: &[u8]) -> Vec<u16> {
        let bins = self.max_bins;
        let mut hist = vec![0u32; bins + 1];
        for &sym in symbols {
            let idx = (sym as usize).min(bins - 1);
            hist[idx + 1] += 1;
        }
        let mut local_sum = 0u32;
        for count in hist.iter_mut().skip(1) {
            let v = *count;
            *count += local_sum;
            local_sum += v;
        }
        let max_u16_val = (0xFFFFusize).saturating_sub(bins) as u32;
        let mut cdf = vec![0u16; bins + 1];
        for (i, &cum) in hist.iter().enumerate() {
            let norm = (max_u16_val * cum).checked_div(local_sum).unwrap_or(0) as u16;
            cdf[i] = norm + (i as u16);
        }
        cdf
    }

    /// Encodes a channel's quantized `symbols` (`0..max_bins`) into a compact byte stream using
    /// 16-bit precision integer arithmetic coding (`ac_enc.cu`).
    pub fn encode_channel(&self, cdf: &[u16], symbols: &[u8]) -> Vec<u8> {
        struct BitWriter {
            out_bytes: Vec<u8>,
            cur_byte: u8,
            bit_count: u8,
        }
        impl BitWriter {
            fn push_bit(&mut self, bit: u8) {
                self.cur_byte = (self.cur_byte << 1) | (bit & 1);
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.out_bytes.push(self.cur_byte);
                    self.cur_byte = 0;
                    self.bit_count = 0;
                }
            }
            fn emit_with_pending(&mut self, bit: u8, pending: &mut u64) {
                self.push_bit(bit);
                let inv = 1 - (bit & 1);
                while *pending > 0 {
                    self.push_bit(inv);
                    *pending -= 1;
                }
            }
        }

        let max_sym = cdf.len().saturating_sub(2);
        let mut low: u32 = 0;
        let mut high: u32 = 0xFFFF_FFFF;
        let mut pending_bits: u64 = 0;
        let mut writer = BitWriter {
            out_bytes: Vec::new(),
            cur_byte: 0,
            bit_count: 0,
        };

        for &raw_sym in symbols {
            let sym = (raw_sym as usize).min(max_sym);
            let span = (high as u64) - (low as u64) + 1;
            let c_low = cdf[sym] as u64;
            let c_high = if sym == max_sym {
                0x10000u64
            } else {
                cdf[sym + 1] as u64
            };
            high = low
                .wrapping_sub(1)
                .wrapping_add(((span * c_high) >> Self::PRECISION_BITS) as u32);
            low = low.wrapping_add(((span * c_low) >> Self::PRECISION_BITS) as u32);

            loop {
                if high < 0x8000_0000 {
                    writer.emit_with_pending(0, &mut pending_bits);
                    low <<= 1;
                    high = (high << 1) | 1;
                } else if low >= 0x8000_0000 {
                    writer.emit_with_pending(1, &mut pending_bits);
                    low <<= 1;
                    high = (high << 1) | 1;
                } else if low >= 0x4000_0000 && high < 0xC000_0000 {
                    pending_bits += 1;
                    low = (low << 1) & 0x7FFF_FFFF;
                    high = (high << 1) | 0x8000_0001;
                } else {
                    break;
                }
            }
        }

        pending_bits += 1;
        if low < 0x4000_0000 {
            writer.emit_with_pending(0, &mut pending_bits);
        } else {
            writer.emit_with_pending(1, &mut pending_bits);
        }
        if writer.bit_count > 0 {
            writer.cur_byte <<= 8 - writer.bit_count;
            writer.out_bytes.push(writer.cur_byte);
        }
        writer.out_bytes
    }

    /// Decodes `num_tokens` quantized symbols from a CacheGen arithmetic-coded `bitstream` using
    /// binary search over the 16-bit `cdf` (`ac_dec.cu`).
    pub fn decode_channel(&self, cdf: &[u16], bitstream: &[u8], num_tokens: usize) -> Vec<u8> {
        let max_sym = cdf.len().saturating_sub(2);
        let mut low: u32 = 0;
        let mut high: u32 = 0xFFFF_FFFF;
        let mut value: u32 = 0;
        let mut bit_cursor: usize = 0;

        let read_bit = |cursor: &mut usize| -> u32 {
            let byte_idx = *cursor / 8;
            let bit_idx = 7 - (*cursor % 8);
            *cursor += 1;
            if byte_idx < bitstream.len() {
                ((bitstream[byte_idx] >> bit_idx) & 1) as u32
            } else {
                0
            }
        };

        for _ in 0..32 {
            value = (value << 1) | read_bit(&mut bit_cursor);
        }

        let mut decoded = Vec::with_capacity(num_tokens);
        for tok_idx in 0..num_tokens {
            let span = (high as u64) - (low as u64) + 1;
            let count = ((((value as u64) - (low as u64) + 1) * 0x10000u64 - 1) / span) as u16;

            // Binary search over `cdf[0..=max_sym]` (`binsearch` in `ac_dec.cu`)
            let mut left = 0usize;
            let mut right = max_sym + 1;
            while left + 1 < right {
                let mid = (left + right) / 2;
                let v = cdf[mid];
                if v < count {
                    left = mid;
                } else if v > count {
                    right = mid;
                } else {
                    left = mid;
                    break;
                }
            }
            let sym = left;
            decoded.push(sym as u8);
            if tok_idx + 1 == num_tokens {
                break;
            }

            let c_low = cdf[sym] as u64;
            let c_high = if sym == max_sym {
                0x10000u64
            } else {
                cdf[sym + 1] as u64
            };
            high = low
                .wrapping_sub(1)
                .wrapping_add(((span * c_high) >> Self::PRECISION_BITS) as u32);
            low = low.wrapping_add(((span * c_low) >> Self::PRECISION_BITS) as u32);

            loop {
                if high < 0x8000_0000 || low >= 0x8000_0000 {
                    low <<= 1;
                    high = (high << 1) | 1;
                    value = (value << 1) | read_bit(&mut bit_cursor);
                } else if low >= 0x4000_0000 && high < 0xC000_0000 {
                    low = (low << 1) & 0x7FFF_FFFF;
                    high = (high << 1) | 0x8000_0001;
                    value = (value & 0x8000_0000)
                        | ((value << 1) & 0x7FFF_FFFE)
                        | read_bit(&mut bit_cursor);
                } else {
                    break;
                }
            }
        }
        decoded
    }
}

/// Per-layer-group invariants for CacheBlend non-block-aligned KV retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBGroupSpec {
    pub group_idx: usize,
    pub num_layers: usize,
    pub slot_tokens: usize,
    pub num_temp_slots: usize,
    pub bytes_per_token_per_layer: usize,
    pub rope_enabled: bool,
}

/// Descriptor for a single non-block-aligned RAG chunk hit in host KV memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBChunkMatch {
    pub host_offset: usize,
    pub num_tokens: usize,
    pub old_start_pos: i64,
    pub cur_start_pos: i64,
    pub slot_mapping_offset: usize,
}

/// K-only in-place delta re-RoPE launch descriptor for one temporary staging slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBRopeVar {
    pub group_idx: usize,
    pub slot_idx: usize,
    pub old_st: i64,
    pub cur_st: i64,
}

/// Per-token scatter launch descriptor writing `n_tok` tokens from temporary staging slot
/// `slot_idx` into paged KV slots starting at `slot_mapping_offset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBScatterVar {
    pub group_idx: usize,
    pub slot_idx: usize,
    pub slot_mapping_offset: usize,
    pub n_tok: usize,
}

/// One batched wave of CacheBlend temporary staging slots: H2D staging copies, then K-only
/// delta re-RoPE launches, then per-token paged KV scatters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CBRetrieveStep {
    pub step_idx: usize,
    pub staging_copies: Vec<KvDmaCopyEntry>,
    pub ropes: Vec<CBRopeVar>,
    pub scatters: Vec<CBScatterVar>,
}

/// Builds a batched plan-then-execute CacheBlend retrieve schedule (`Vec<CBRetrieveStep>`)
/// over `groups` and `chunks`, partitioning chunks into waves sized by the available temporary
/// GPU staging slots (`num_temp_slots`) and splitting H2D staging copies at `host_buffer_alignment`
/// pin-chunk boundaries.
pub fn plan_cacheblend_retrieve(
    groups: &[CBGroupSpec],
    chunks: &[CBChunkMatch],
    host_buffer_alignment: usize,
) -> Vec<CBRetrieveStep> {
    if groups.is_empty() || chunks.is_empty() {
        return Vec::new();
    }
    let wave_cap = groups
        .iter()
        .map(|g| g.num_temp_slots.max(1))
        .min()
        .unwrap_or(1);
    let align = host_buffer_alignment.max(1);
    let mut steps = Vec::new();

    for (step_idx, wave) in chunks.chunks(wave_cap).enumerate() {
        let mut staging_copies = Vec::new();
        let mut ropes = Vec::new();
        let mut scatters = Vec::new();

        for (slot_idx, chunk) in wave.iter().enumerate() {
            let n_tok = chunk.num_tokens;
            if n_tok == 0 {
                continue;
            }
            let mut group_host_cursor = chunk.host_offset;
            for grp in groups {
                let total_bytes = grp
                    .num_layers
                    .saturating_mul(n_tok)
                    .saturating_mul(grp.bytes_per_token_per_layer);
                let slot_cap_bytes = grp
                    .num_layers
                    .saturating_mul(grp.slot_tokens.max(n_tok))
                    .saturating_mul(grp.bytes_per_token_per_layer);
                let dev_base = slot_idx.saturating_mul(slot_cap_bytes);

                let mut rem = total_bytes;
                let mut h_cur = group_host_cursor;
                let mut d_cur = dev_base;
                while rem > 0 {
                    let next_boundary = ((h_cur / align) + 1) * align;
                    let step_bytes = rem.min(next_boundary - h_cur);
                    staging_copies.push(KvDmaCopyEntry {
                        block_id: slot_idx,
                        host_offset: h_cur as u64,
                        device_offset: d_cur as u64,
                        width_bytes: step_bytes as u64,
                        height_layers: grp.num_layers,
                    });
                    h_cur += step_bytes;
                    d_cur += step_bytes;
                    rem -= step_bytes;
                }
                group_host_cursor += total_bytes;

                if grp.rope_enabled && chunk.old_start_pos != chunk.cur_start_pos {
                    ropes.push(CBRopeVar {
                        group_idx: grp.group_idx,
                        slot_idx,
                        old_st: chunk.old_start_pos,
                        cur_st: chunk.cur_start_pos,
                    });
                }
                scatters.push(CBScatterVar {
                    group_idx: grp.group_idx,
                    slot_idx,
                    slot_mapping_offset: chunk.slot_mapping_offset,
                    n_tok,
                });
            }
        }

        steps.push(CBRetrieveStep {
            step_idx,
            staging_copies,
            ropes,
            scatters,
        });
    }
    steps
}

fn is_narrow_float_dtype(dt: &DType) -> bool {
    matches!(
        dt,
        DType::F16 | DType::BF16 | DType::F8E4M3FN | DType::F8E5M2
    )
}

/// Ensures accumulating operations (`dot_general`, `convolution`, `reduce`, and fused GEMM
/// composites) with narrow floating-point operands or outputs (`f16`, `bf16`, `f8E4M3FN`,
/// `f8E5M2`) accumulate in `f32` before downcasting on epilogue store.
///
/// Stamps `castforge.accumulator_dtype = "f32"` (and `castforge.epilogue_downcast = "<dtype>"`
/// when the output tensor is narrow) and returns the number of accumulating ops promoted.
pub fn promote_narrow_accumulators(module: &mut Module) -> usize {
    let mut promoted = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.attributes.contains_key("castforge.accumulator_dtype") {
                continue;
            }
            let is_accum_op = op.is_op("stablehlo", "dot_general")
                || op.is_op("stablehlo", "convolution")
                || op.is_op("stablehlo", "reduce")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| {
                            matches!(
                                n,
                                "castforge.fused_matmul_epilogue"
                                    | "castforge.fused_dequant_matmul"
                                    | "castforge.fused_qmatmul"
                            )
                        }));
            if !is_accum_op {
                continue;
            }

            let has_narrow_input = op.operands.iter().any(|&v| {
                module
                    .values
                    .get(v)
                    .and_then(|val| val.ty.as_tensor())
                    .is_some_and(|t| is_narrow_float_dtype(&t.dtype))
            });
            let narrow_output_dt = op.results.first().and_then(|&r| {
                module
                    .values
                    .get(r)
                    .and_then(|val| val.ty.as_tensor())
                    .and_then(|t| {
                        if is_narrow_float_dtype(&t.dtype) {
                            Some(t.dtype.to_string())
                        } else {
                            None
                        }
                    })
            });

            if !has_narrow_input && narrow_output_dt.is_none() {
                continue;
            }

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.attributes.insert(
                    "castforge.accumulator_dtype".to_string(),
                    Attribute::String("f32".to_string()),
                );
                if let Some(down_dt) = narrow_output_dt {
                    op_mut.attributes.insert(
                        "castforge.epilogue_downcast".to_string(),
                        Attribute::String(down_dt),
                    );
                }
                promoted += 1;
            }
        }
    }
    promoted
}

/// Adaptive Replacement Cache (ARC) KV eviction policy with `T1` (recent), `T2` (frequent),
/// and `B1`/`B2` ghost-history feedback.
#[derive(Debug, Clone, Default)]
pub struct ArcKvEvictionPolicy {
    t1: IndexMap<u64, ()>,
    t2: IndexMap<u64, ()>,
    b1: IndexMap<u64, ()>,
    b2: IndexMap<u64, ()>,
    target_t1_size: f64,
    capacity: usize,
}

impl ArcKvEvictionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resident_len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    pub fn t1_len(&self) -> usize {
        self.t1.len()
    }

    pub fn t2_len(&self) -> usize {
        self.t2.len()
    }

    pub fn b1_len(&self) -> usize {
        self.b1.len()
    }

    pub fn b2_len(&self) -> usize {
        self.b2.len()
    }

    pub fn target_t1_size(&self) -> f64 {
        self.target_t1_size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn adaptation_delta(hit_len: usize, other_len: usize) -> f64 {
        (other_len as f64 / hit_len.max(1) as f64).max(1.0)
    }

    fn trim_ghost_lists(&mut self) {
        if self.capacity == 0 {
            return;
        }
        while self.b1.len() > self.capacity {
            self.b1.shift_remove_index(0);
        }
        while self.b2.len() > self.capacity {
            self.b2.shift_remove_index(0);
        }
        while self.b1.len() + self.b2.len() > 2 * self.capacity {
            if !self.b2.is_empty() {
                self.b2.shift_remove_index(0);
            } else if !self.b1.is_empty() {
                self.b1.shift_remove_index(0);
            } else {
                break;
            }
        }
    }

    /// Admits newly created KV block keys in reverse request order (so later suffix chunks stay
    /// closer to the LRU head) and applies `B1`/`B2` ghost-history feedback to `target_t1_size`.
    pub fn on_keys_created(&mut self, keys: &[u64]) {
        for &key in keys.iter().rev() {
            if self.t1.shift_remove(&key).is_some() || self.t2.shift_remove(&key).is_some() {
                self.t2.insert(key, ());
            } else if self.b1.contains_key(&key) {
                let delta = Self::adaptation_delta(self.b1.len(), self.b2.len());
                self.target_t1_size = (self.target_t1_size + delta).min(self.capacity as f64);
                self.b1.shift_remove(&key);
                self.t2.insert(key, ());
            } else if self.b2.contains_key(&key) {
                let delta = Self::adaptation_delta(self.b2.len(), self.b1.len());
                self.target_t1_size = (self.target_t1_size - delta).max(0.0);
                self.b2.shift_remove(&key);
                self.t2.insert(key, ());
            } else {
                self.t1.insert(key, ());
            }
            self.capacity = self.capacity.max(self.resident_len());
            self.trim_ghost_lists();
        }
    }

    /// Promotes touched resident keys from `T1` to the MRU tail of `T2`, or refreshes their
    /// recency in `T2`.
    pub fn on_keys_touched(&mut self, keys: &[u64]) {
        for &key in keys.iter().rev() {
            if self.t1.shift_remove(&key).is_some() || self.t2.shift_remove(&key).is_some() {
                self.t2.insert(key, ());
            }
        }
    }

    /// Selects and evicts `expected_ratio` of resident keys into ghost history (`B1` or `B2`)
    /// using the request-independent adaptive boundary rule (`|T1| >= int(p)`).
    pub fn select_and_evict_victims(&mut self, expected_ratio: f64) -> Vec<u64> {
        let resident = self.resident_len();
        if resident == 0 {
            return Vec::new();
        }
        let ratio = expected_ratio.clamp(0.0, 1.0);
        if ratio <= 0.0 {
            return Vec::new();
        }
        let target_count = ((resident as f64 * ratio) as usize).max(1);
        self.capacity = self.capacity.max(resident);
        self.target_t1_size = self.target_t1_size.min(self.capacity as f64);

        let mut victims = Vec::with_capacity(target_count);
        while victims.len() < target_count {
            let prefer_t1 = self.t1.len() >= (self.target_t1_size as usize);
            if prefer_t1 && let Some((k, _)) = self.t1.shift_remove_index(0) {
                self.b1.insert(k, ());
                victims.push(k);
                continue;
            }
            if let Some((k, _)) = self.t2.shift_remove_index(0) {
                self.b2.insert(k, ());
                victims.push(k);
                continue;
            }
            if let Some((k, _)) = self.t1.shift_remove_index(0) {
                self.b1.insert(k, ());
                victims.push(k);
                continue;
            }
            break;
        }
        self.trim_ghost_lists();
        victims
    }
}

/// Multi-tenant `cache_salt` byte-budget quota registry supporting both strict allowlist
/// (`get_limit_bytes` -> `0` for unknown tenants) and coordinator bootstrap-exempt
/// (`effective_limit_bytes` -> `None` until armed) resolution lenses.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvSaltQuotaRegistry {
    limits: IndexMap<String, u64>,
    default_limit_bytes: Option<u64>,
}

impl KvSaltQuotaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_quota(&mut self, cache_salt: impl Into<String>, limit_bytes: u64) {
        self.limits.insert(cache_salt.into(), limit_bytes);
    }

    pub fn delete_quota(&mut self, cache_salt: &str) -> bool {
        self.limits.shift_remove(cache_salt).is_some()
    }

    pub fn has_quota(&self, cache_salt: &str) -> bool {
        self.limits.contains_key(cache_salt)
    }

    /// Strict allowlist lens: unregistered salts resolve to `0` bytes so unknown tenant data is
    /// immediately eligible for eviction on local storage workers.
    pub fn get_limit_bytes(&self, cache_salt: &str) -> u64 {
        self.limits.get(cache_salt).copied().unwrap_or(0)
    }

    pub fn set_default_limit_bytes(&mut self, limit_bytes: Option<u64>) {
        self.default_limit_bytes = limit_bytes;
    }

    pub fn get_default_limit_bytes(&self) -> Option<u64> {
        self.default_limit_bytes
    }

    /// Coordinator bootstrap-safe lens: returns `Some(limit)` for registered salts, or
    /// `default_limit_bytes` (`None` = exempt from eviction until the quota controller syncs
    /// tenant tables and arms enforcement).
    pub fn effective_limit_bytes(&self, cache_salt: &str) -> Option<u64> {
        self.limits
            .get(cache_salt)
            .copied()
            .or(self.default_limit_bytes)
    }

    /// Computes how many bytes `used_bytes` exceeds `cache_salt`'s quota under either the
    /// strict allowlist lens (`allowlist_lens = true`) or the bootstrap-safe coordinator lens
    /// (`allowlist_lens = false`).
    pub fn over_quota_bytes(&self, cache_salt: &str, used_bytes: u64, allowlist_lens: bool) -> u64 {
        let limit = if allowlist_lens {
            Some(self.get_limit_bytes(cache_salt))
        } else {
            self.effective_limit_bytes(cache_salt)
        };
        match limit {
            None => 0,
            Some(lim) => used_bytes.saturating_sub(lim),
        }
    }
}
