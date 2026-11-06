//! Autoregressive KV-cache codecs, CacheBlend, and tiered KV storage passes.

use crate::ir::{Attribute, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId};
use crate::passes::fuse::kv_serving::EngineKvFormat;
use anyhow::Result;
use indexmap::IndexMap;

/// Sentinel scale used when a V tensor, head, or page is identically zero, keeping
/// `dequant(quant(0)) == 0.0` exact and avoiding `0.0 / 0.0` NaNs.
pub const KV_ZERO_SCALE_SENTINEL: f32 = 1.0;

/// Quantization scale granularity for the V plane in [`AsymK16V8Codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvScaleScope {
    /// Single FP32 scale across the entire V buffer (`1` scale).
    PerTensor,
    /// One FP32 scale per KV head across all pages (`num_kv_heads` scales).
    PerLayerHead,
    /// One FP32 scale per `(page, kv_head)` pair (`num_pages * num_kv_heads` scales).
    PerPageHead,
}

/// Asymmetric K16/V8 encoded KV-cache payload.
///
/// Keys (`K`) are preserved in 16-bit IEEE-754 half precision (`u16` bits) to protect RoPE phase
/// and exponential $\text{softmax}(Q K^T / \sqrt{d_k})$ logits, while Values (`V`) are quantized
/// to 8-bit signed codes (`i8`) with scoped FP32 scales.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedAsymKvPage {
    pub k_fp16_bits: Vec<u16>,
    pub v_int8_codes: Vec<i8>,
    pub v_scales: Vec<f32>,
    pub scope: KvScaleScope,
    pub num_pages: usize,
    pub page_tokens: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl EncodedAsymKvPage {
    /// Total compressed byte footprint (`K` in 2B + `V` in 1B + FP32 scale metadata).
    pub fn payload_bytes(&self) -> usize {
        self.k_fp16_bits.len() * 2 + self.v_int8_codes.len() + self.v_scales.len() * 4
    }

    /// Baseline 16-bit (`FP16`/`BF16`) byte footprint for both `K` and `V` (`2B + 2B` per elem).
    pub fn uncompressed_fp16_bytes(&self) -> usize {
        (self.k_fp16_bits.len() + self.v_int8_codes.len()) * 2
    }
}

/// Asymmetric `K16/V8` KV-cache encoder/decoder with `PerTensor`, `PerLayerHead`, and
/// `PerPageHead` scale scopes.
pub struct AsymK16V8Codec;

impl AsymK16V8Codec {
    const QMAX: f32 = 127.0;

    /// Encodes `k` and `v` buffers of shape `[num_pages, page_tokens, num_kv_heads, head_dim]`
    /// into an [`EncodedAsymKvPage`].
    pub fn encode(
        k: &[f32],
        v: &[f32],
        num_pages: usize,
        page_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scope: KvScaleScope,
    ) -> Result<EncodedAsymKvPage> {
        use crate::ir::f32_to_f16_bits;
        let expected = num_pages
            .saturating_mul(page_tokens)
            .saturating_mul(num_kv_heads)
            .saturating_mul(head_dim);
        if expected == 0 || k.len() != expected || v.len() != expected {
            anyhow::bail!(
                "AsymK16V8Codec::encode shape mismatch: expected {expected} elems, \
                 got k={} v={}",
                k.len(),
                v.len()
            );
        }

        let k_fp16_bits: Vec<u16> = k.iter().map(|&x| f32_to_f16_bits(x)).collect();
        let v_scales =
            Self::compute_v_scales(v, num_pages, page_tokens, num_kv_heads, head_dim, scope);
        let mut v_int8_codes = vec![0i8; expected];

        let head_stride = head_dim;
        let tok_stride = num_kv_heads * head_stride;
        let page_stride = page_tokens * tok_stride;

        for p in 0..num_pages {
            for t in 0..page_tokens {
                for h in 0..num_kv_heads {
                    let scale = match scope {
                        KvScaleScope::PerTensor => v_scales[0],
                        KvScaleScope::PerLayerHead => v_scales[h],
                        KvScaleScope::PerPageHead => v_scales[p * num_kv_heads + h],
                    };
                    let base = p * page_stride + t * tok_stride + h * head_stride;
                    for d in 0..head_dim {
                        let val = v[base + d];
                        let q = (val / scale).round().clamp(-Self::QMAX, Self::QMAX) as i8;
                        v_int8_codes[base + d] = q;
                    }
                }
            }
        }

        Ok(EncodedAsymKvPage {
            k_fp16_bits,
            v_int8_codes,
            v_scales,
            scope,
            num_pages,
            page_tokens,
            num_kv_heads,
            head_dim,
        })
    }

    /// Computes scoped FP32 quantization scales for `v`, substituting [`KV_ZERO_SCALE_SENTINEL`]
    /// whenever the maximum absolute value in a scope is `0.0`.
    pub fn compute_v_scales(
        v: &[f32],
        num_pages: usize,
        page_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scope: KvScaleScope,
    ) -> Vec<f32> {
        let head_stride = head_dim;
        let tok_stride = num_kv_heads * head_stride;
        let page_stride = page_tokens * tok_stride;

        match scope {
            KvScaleScope::PerTensor => {
                let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
                let s = if amax > 0.0 {
                    amax / Self::QMAX
                } else {
                    KV_ZERO_SCALE_SENTINEL
                };
                vec![s]
            }
            KvScaleScope::PerLayerHead => {
                let mut scales = vec![KV_ZERO_SCALE_SENTINEL; num_kv_heads];
                for (h, slot) in scales.iter_mut().enumerate() {
                    let mut amax = 0.0f32;
                    for p in 0..num_pages {
                        for t in 0..page_tokens {
                            let base = p * page_stride + t * tok_stride + h * head_stride;
                            for d in 0..head_dim {
                                amax = amax.max(v[base + d].abs());
                            }
                        }
                    }
                    *slot = if amax > 0.0 {
                        amax / Self::QMAX
                    } else {
                        KV_ZERO_SCALE_SENTINEL
                    };
                }
                scales
            }
            KvScaleScope::PerPageHead => {
                let mut scales = vec![KV_ZERO_SCALE_SENTINEL; num_pages * num_kv_heads];
                for p in 0..num_pages {
                    for h in 0..num_kv_heads {
                        let mut amax = 0.0f32;
                        for t in 0..page_tokens {
                            let base = p * page_stride + t * tok_stride + h * head_stride;
                            for d in 0..head_dim {
                                amax = amax.max(v[base + d].abs());
                            }
                        }
                        scales[p * num_kv_heads + h] = if amax > 0.0 {
                            amax / Self::QMAX
                        } else {
                            KV_ZERO_SCALE_SENTINEL
                        };
                    }
                }
                scales
            }
        }
    }

    /// Decodes an [`EncodedAsymKvPage`] back into `(k_f32, v_f32)`.
    pub fn decode(encoded: &EncodedAsymKvPage) -> (Vec<f32>, Vec<f32>) {
        use crate::ir::f16_bits_to_f32;
        let k_f32: Vec<f32> = encoded
            .k_fp16_bits
            .iter()
            .map(|&bits| f16_bits_to_f32(bits))
            .collect();
        let total = encoded.v_int8_codes.len();
        let mut v_f32 = vec![0.0f32; total];

        let head_stride = encoded.head_dim;
        let tok_stride = encoded.num_kv_heads * head_stride;
        let page_stride = encoded.page_tokens * tok_stride;

        for p in 0..encoded.num_pages {
            for t in 0..encoded.page_tokens {
                for h in 0..encoded.num_kv_heads {
                    let scale = match encoded.scope {
                        KvScaleScope::PerTensor => encoded.v_scales[0],
                        KvScaleScope::PerLayerHead => encoded.v_scales[h],
                        KvScaleScope::PerPageHead => encoded.v_scales[p * encoded.num_kv_heads + h],
                    };
                    let base = p * page_stride + t * tok_stride + h * head_stride;
                    for d in 0..encoded.head_dim {
                        v_f32[base + d] = (encoded.v_int8_codes[base + d] as f32) * scale;
                    }
                }
            }
        }
        (k_f32, v_f32)
    }
}

/// Geometry and rotary embedding parameters for CacheBlend in-place delta re-RoPE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheBlendRopeSpec {
    /// Number of KV heads per token.
    pub num_kv_heads: usize,
    /// Element stride between consecutive heads (`== head_size` for contiguous K, or
    /// `2 * head_size` for packed K/V buffers so only the K half is rotated).
    pub head_stride: usize,
    /// Element offset inside each head where RoPE dimensions start (`0` for standard GQA/MHA,
    /// or `kv_lora_rank` for DeepSeek MLA where decoupled RoPE keys trail the latent vector).
    pub rope_base_offset: usize,
    /// Even number of rotary embedding dimensions (`rot_dim <= head_stride - rope_base_offset`).
    pub rot_dim: usize,
    /// Base RoPE frequency (`10000.0` for standard LLaMA, `500000.0` for LLaMA-3).
    pub rope_theta: f32,
    /// `true` for GPT-NeoX / LLaMA half-split pairing `(i, i + rot_dim/2)`; `false` for
    /// GPT-J interleaved pairing `(2*i, 2*i + 1)`.
    pub is_neox: bool,
}

/// Rotates cached Key vectors `key_buf` in-place from `old_positions` to `new_positions`
/// using the abelian group property $\mathcal{R}(p_{\text{new}}) = \mathcal{R}(p_{\text{new}} -
/// p_{\text{old}}) \cdot \mathcal{R}(p_{\text{old}})$.
///
/// This enables non-prefix KV-cache reuse (e.g. RAG document chunks spliced at arbitrary prompt
/// offsets) without recomputing QKV projections.
pub fn re_rope_key_in_place(
    key_buf: &mut [f32],
    old_positions: &[i64],
    new_positions: &[i64],
    spec: &CacheBlendRopeSpec,
) -> Result<()> {
    let num_tokens = old_positions.len();
    if new_positions.len() != num_tokens {
        anyhow::bail!(
            "re_rope_key_in_place position length mismatch: old={} new={}",
            old_positions.len(),
            new_positions.len()
        );
    }
    if spec.rot_dim == 0 || !spec.rot_dim.is_multiple_of(2) {
        anyhow::bail!(
            "re_rope_key_in_place requires positive even rot_dim, got {}",
            spec.rot_dim
        );
    }
    if spec.rope_base_offset + spec.rot_dim > spec.head_stride {
        anyhow::bail!(
            "re_rope_key_in_place rope_base_offset ({}) + rot_dim ({}) exceeds head_stride ({})",
            spec.rope_base_offset,
            spec.rot_dim,
            spec.head_stride
        );
    }
    let tok_stride = spec.num_kv_heads.saturating_mul(spec.head_stride);
    let required = num_tokens.saturating_mul(tok_stride);
    if key_buf.len() < required {
        anyhow::bail!(
            "re_rope_key_in_place buffer too small: len={} < required={required}",
            key_buf.len()
        );
    }

    let half = spec.rot_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / spec.rope_theta.powf((2 * i) as f32 / spec.rot_dim as f32))
        .collect();

    for t in 0..num_tokens {
        let delta_pos = (new_positions[t] - old_positions[t]) as f32;
        if delta_pos == 0.0 {
            continue;
        }
        for h in 0..spec.num_kv_heads {
            let base = t * tok_stride + h * spec.head_stride + spec.rope_base_offset;
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = delta_pos * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let (idx0, idx1) = if spec.is_neox {
                    (base + i, base + i + half)
                } else {
                    (base + 2 * i, base + 2 * i + 1)
                };
                let k0 = key_buf[idx0];
                let k1 = key_buf[idx1];
                key_buf[idx0] = k0 * cos_a - k1 * sin_a;
                key_buf[idx1] = k0 * sin_a + k1 * cos_a;
            }
        }
    }
    Ok(())
}

/// Configuration for chained prefix-cache token-block hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainedBlockHashConfig<'a> {
    /// Number of tokens per KV-cache page block (`> 0`). Trailing ragged tokens
    /// (`tokens.len() % block_size`) are dropped so partial blocks never pollute the cache.
    pub block_size: usize,
    /// 64-bit domain seed for the hash chain.
    pub seed: u64,
    /// Optional model/LoRA-adapter/quantization salt mixed into the chain root.
    pub salt: &'a [u8],
    /// Optional parent block hash when continuing a multi-turn or chunked prefix.
    pub parent_hash: Option<u64>,
}

impl<'a> Default for ChainedBlockHashConfig<'a> {
    fn default() -> Self {
        Self {
            block_size: 16,
            seed: 0,
            salt: b"",
            parent_hash: None,
        }
    }
}

fn mix64(mut x: u64, k: u64) -> u64 {
    x ^= k;
    let wide = (x as u128).wrapping_mul(0x9e3779b97f4a7c15u128);
    let folded = (wide as u64) ^ ((wide >> 64) as u64);
    folded.rotate_left(27).wrapping_mul(0x94d049bb133111ebu64)
}

fn hash_bytes_64(mut state: u64, bytes: &[u8]) -> u64 {
    let (chunks, rem) = bytes.as_chunks::<8>();
    for &chunk in chunks {
        state = mix64(state, u64::from_le_bytes(chunk));
    }
    if !rem.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rem.len()].copy_from_slice(rem);
        state = mix64(state, u64::from_le_bytes(tail) ^ (rem.len() as u64));
    }
    mix64(state, bytes.len() as u64)
}

/// Slices `tokens` into full blocks of `cfg.block_size` (dropping any trailing ragged tokens)
/// and computes a chained 64-bit prefix hash per block:
///
/// $$H_k = \text{Hash}(\text{root},\; H_{k-1} \parallel \text{tokens}_k)$$
///
/// Each returned hash $H_k$ uniquely identifies the entire token prefix `[0 .. (k + 1) * B)`
/// under `(seed, salt)`.
pub fn hash_prefix_token_blocks(
    tokens: &[i32],
    cfg: &ChainedBlockHashConfig<'_>,
) -> Result<Vec<u64>> {
    if cfg.block_size == 0 {
        anyhow::bail!("hash_prefix_token_blocks: block_size must be > 0");
    }
    let root_key = if cfg.salt.is_empty() {
        mix64(cfg.seed, 0x517cc1b727220a95)
    } else {
        hash_bytes_64(mix64(cfg.seed, 0x517cc1b727220a95), cfg.salt)
    };
    let mut prev_hash = cfg.parent_hash.unwrap_or(root_key);
    let num_full_blocks = tokens.len() / cfg.block_size;
    let mut out = Vec::with_capacity(num_full_blocks);

    for blk_idx in 0..num_full_blocks {
        let slice = &tokens[blk_idx * cfg.block_size..(blk_idx + 1) * cfg.block_size];
        let mut block_state = root_key;
        for &tok in slice {
            block_state = mix64(block_state, (tok as u32) as u64);
        }
        block_state = mix64(block_state, cfg.block_size as u64);
        let chained = mix64(mix64(root_key, prev_hash), block_state);
        out.push(chained);
        prev_hash = chained;
    }
    Ok(out)
}

/// Compresses large uniform `stablehlo.constant` tensors with `>= min_elements` (`>= 2`) into a
/// 1-element scalar constant (`tensor<1xdtype>`) followed by `stablehlo.broadcast_in_dim`.
///
/// Also evicts any unreferenced uniform tensor entries from `module.tensor_store`, returning the
/// number of large constants compressed.
pub fn compress_splat_constants(module: &mut Module, min_elements: usize) -> usize {
    use crate::ir::{Dim, TensorType, Type, parse_dense_f32_literal};
    use smallvec::smallvec;

    let threshold = min_elements.max(2) as i64;
    let mut compressed = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "constant") || op.results.is_empty() {
                continue;
            }
            let orig_res = op.results[0];
            let Some(tensor_ty) = module.values[orig_res].ty.as_tensor().cloned() else {
                continue;
            };
            let Some(num_elems) = tensor_ty.num_elements() else {
                continue;
            };
            if num_elems < threshold {
                continue;
            }

            // Extract uniform scalar value if all elements are identical
            let mut evicted_key: Option<String> = None;
            let uniform_scalar: Option<f32> = match op.attributes.get("value") {
                Some(Attribute::DenseElements { raw_literal, .. }) => {
                    parse_dense_f32_literal(raw_literal, num_elems as usize).and_then(|vals| {
                        let &v0 = vals.first()?;
                        vals.iter()
                            .all(|&x| (x - v0).abs() <= f32::EPSILON)
                            .then_some(v0)
                    })
                }
                Some(Attribute::TensorRef { key, .. }) => {
                    if let Some(entry) = module.tensor_store.get(key)
                        && let Some(vals) = entry.as_f32_slice()
                        && let Some(&v0) = vals.first()
                        && vals.iter().all(|&x| (x - v0).abs() <= f32::EPSILON)
                    {
                        evicted_key = Some(key.clone());
                        Some(v0)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let Some(v0) = uniform_scalar else {
                continue;
            };

            let scalar_tensor_ty = TensorType::new(vec![Dim::Static(1)], tensor_ty.dtype);
            let scalar_ty = Type::Tensor(scalar_tensor_ty.clone());
            let lit = format!("{v0:.6}");
            let mut const_attrs = IndexMap::new();
            const_attrs.insert(
                "value".to_string(),
                Attribute::DenseElements {
                    ty: scalar_tensor_ty,
                    raw_literal: lit,
                },
            );

            // Rewrite `op_id` in place to produce the 1-element scalar constant
            let scalar_val_id = module.values.insert_with_key(|vid| Value {
                id: vid,
                ssa_name: Some(format!("splat_scalar_{compressed}")),
                ty: scalar_ty,
                def: ValueDef::OpResult {
                    op: op_id,
                    index: 0,
                },
            });
            module.ops[op_id].results[0] = scalar_val_id;
            module.ops[op_id].attributes = const_attrs;

            // Insert `stablehlo.broadcast_in_dim` producing the original `orig_res` ValueId
            let mut bcast_attrs = IndexMap::new();
            bcast_attrs.insert(
                "broadcast_dimensions".to_string(),
                Attribute::DenseIntArray(vec![0]),
            );
            let bcast_op_id = module.ops.insert_with_key(|oid| Operation {
                id: oid,
                name: OpName::new("stablehlo", "broadcast_in_dim"),
                operands: smallvec![scalar_val_id],
                results: smallvec![orig_res],
                attributes: bcast_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: op.location.clone(),
            });
            module.values[orig_res].def = ValueDef::OpResult {
                op: bcast_op_id,
                index: 0,
            };

            if let Some(pos) = module.blocks[block_id]
                .ops
                .iter()
                .position(|&id| id == op_id)
            {
                module.blocks[block_id].ops.insert(pos + 1, bcast_op_id);
            }
            if let Some(k) = evicted_key {
                module.tensor_store.remove(&k);
            }
            compressed += 1;
        }
    }
    compressed
}

/// Summary report of CUDA Graph / HIP Graph capture stability analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaGraphCaptureReport {
    /// `true` if every operation in the module has static shapes and no host-blocking sync ops.
    pub is_capture_safe: bool,
    /// Number of SSA tensor values backed by address-stable storage (arena slots, in-place
    /// aliases, or constant weights).
    pub stable_value_count: usize,
    /// Number of function input arguments requiring stable device pointers or graph-exec update.
    pub dynamic_input_count: usize,
    /// Operations that prevent full CUDA/HIP Graph capture (e.g. dynamic shapes or host sync).
    pub blocking_ops: Vec<String>,
}

/// Analyzes and stamps `castforge.cuda_graph_stable = true` onto operations whose tensor operands
/// and results reside in address-stable arena slots or constant weights, and stamps
/// `castforge.cuda_graph_capturable` on each `func.func` operation.
pub fn mark_cuda_graph_stable_tensors(module: &mut Module) -> CudaGraphCaptureReport {
    use crate::codegen::build_memory_plan;
    use crate::ir::Dim;

    let plan = build_memory_plan(module);
    let mut stable_value_count = 0usize;
    let mut dynamic_input_count = 0usize;
    let mut blocking_ops = Vec::new();

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        dynamic_input_count += module.blocks[block_id].arguments.len();
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if op.is_op("func", "return") {
                continue;
            }
            let has_dynamic_dim = op
                .results
                .iter()
                .chain(op.operands.iter())
                .filter_map(|&v| module.values[v].ty.as_tensor())
                .any(|t| !t.shape.iter().all(|d| matches!(d, Dim::Static(_))));
            let is_host_sync = op.is_op("castforge", "offload_to_host")
                || (op.is_op("stablehlo", "composite")
                    && op
                        .attributes
                        .get("name")
                        .and_then(|a| a.as_str())
                        .is_some_and(|n| n == "castforge.offload_sync"));
            if has_dynamic_dim || is_host_sync {
                blocking_ops.push(op.name.to_string());
                continue;
            }

            let all_results_stable = op.results.iter().all(|&res| {
                op.is_op("stablehlo", "constant")
                    || plan
                        .allocation_for(res)
                        .is_some_and(|a| a.arena_slot.is_some() || a.inplace_from.is_some())
            });
            if all_results_stable && !op.results.is_empty() {
                stable_value_count += op.results.len();
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.attributes.insert(
                        "castforge.cuda_graph_stable".to_string(),
                        Attribute::Bool(true),
                    );
                }
            }
        }
    }

    let is_capture_safe = blocking_ops.is_empty();
    for fn_op_id in module.functions() {
        if let Some(fn_op) = module.ops.get_mut(fn_op_id) {
            fn_op.attributes.insert(
                "castforge.cuda_graph_capturable".to_string(),
                Attribute::Bool(is_capture_safe),
            );
        }
    }

    CudaGraphCaptureReport {
        is_capture_safe,
        stable_value_count,
        dynamic_input_count,
        blocking_ops,
    }
}

/// Intrusive doubly-linked LRU free-list links stored inline inside each [`KvPoolBlock`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvFreeLinks {
    pub prev: Option<usize>,
    pub next: Option<usize>,
    pub in_free_list: bool,
}

/// Metadata for a single physical block in [`KvBlockPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPoolBlock {
    pub block_id: usize,
    pub ref_count: usize,
    pub prefix_hash: Option<u64>,
    pub links: KvFreeLinks,
}

/// $O(1)$ Intrusive LRU KV-Cache Block Pool with prefix-hit revival, uncommitted front-recycle,
/// and reverse-chain eviction ordering.
#[derive(Debug, Clone)]
pub struct KvBlockPool {
    blocks: Vec<KvPoolBlock>,
    free_head: Option<usize>,
    free_tail: Option<usize>,
    free_len: usize,
    hash_to_block: std::collections::HashMap<u64, usize>,
}

impl KvBlockPool {
    /// Creates a new [`KvBlockPool`] with `num_blocks` physical blocks (`0..num_blocks`), all
    /// initially on the free list in ascending ID order.
    pub fn new(num_blocks: usize) -> Self {
        let mut pool = Self {
            blocks: (0..num_blocks)
                .map(|id| KvPoolBlock {
                    block_id: id,
                    ref_count: 0,
                    prefix_hash: None,
                    links: KvFreeLinks::default(),
                })
                .collect(),
            free_head: None,
            free_tail: None,
            free_len: 0,
            hash_to_block: std::collections::HashMap::new(),
        };
        for id in 0..num_blocks {
            pool.push_free_back(id);
        }
        pool
    }

    /// Number of currently unreferenced blocks on the LRU free list.
    pub fn free_count(&self) -> usize {
        self.free_len
    }

    /// Total capacity in blocks.
    pub fn capacity(&self) -> usize {
        self.blocks.len()
    }

    /// Inspects metadata for `block_id`.
    pub fn block(&self, block_id: usize) -> Option<&KvPoolBlock> {
        self.blocks.get(block_id)
    }

    fn push_free_back(&mut self, id: usize) {
        debug_assert!(!self.blocks[id].links.in_free_list);
        if let Some(tail) = self.free_tail {
            self.blocks[tail].links.next = Some(id);
        } else {
            self.free_head = Some(id);
        }
        self.blocks[id].links = KvFreeLinks {
            prev: self.free_tail,
            next: None,
            in_free_list: true,
        };
        self.free_tail = Some(id);
        self.free_len += 1;
    }

    fn push_free_front(&mut self, id: usize) {
        debug_assert!(!self.blocks[id].links.in_free_list);
        if let Some(head) = self.free_head {
            self.blocks[head].links.prev = Some(id);
        } else {
            self.free_tail = Some(id);
        }
        self.blocks[id].links = KvFreeLinks {
            prev: None,
            next: self.free_head,
            in_free_list: true,
        };
        self.free_head = Some(id);
        self.free_len += 1;
    }

    fn unlink_free(&mut self, id: usize) {
        let links = self.blocks[id].links;
        debug_assert!(links.in_free_list);
        if let Some(prev) = links.prev {
            self.blocks[prev].links.next = links.next;
        } else {
            self.free_head = links.next;
        }
        if let Some(next) = links.next {
            self.blocks[next].links.prev = links.prev;
        } else {
            self.free_tail = links.prev;
        }
        self.blocks[id].links = KvFreeLinks::default();
        self.free_len -= 1;
    }

    /// Allocates a block from the front of the LRU free list in $O(1)$, evicting its old
    /// `prefix_hash` mapping if it was previously committed.
    pub fn allocate(&mut self) -> Option<usize> {
        let id = self.free_head?;
        self.unlink_free(id);
        if let Some(old_hash) = self.blocks[id].prefix_hash.take()
            && self.hash_to_block.get(&old_hash) == Some(&id)
        {
            self.hash_to_block.remove(&old_hash);
        }
        self.blocks[id].ref_count = 1;
        Some(id)
    }

    /// Commits a full block `block_id` under `prefix_hash` so future requests can match it.
    pub fn commit(&mut self, block_id: usize, prefix_hash: u64) {
        if let Some(blk) = self.blocks.get_mut(block_id) {
            blk.prefix_hash = Some(prefix_hash);
            self.hash_to_block.insert(prefix_hash, block_id);
        }
    }

    /// Looks up `prefix_hash` in the pool; if the block is currently sitting on the free list
    /// awaiting eviction, revives it in $O(1)$ by unlinking it from the middle of the free list!
    pub fn touch_or_revive_prefix(&mut self, prefix_hash: u64) -> Option<usize> {
        let &id = self.hash_to_block.get(&prefix_hash)?;
        if self.blocks[id].prefix_hash != Some(prefix_hash) {
            return None;
        }
        if self.blocks[id].links.in_free_list {
            self.unlink_free(id);
        }
        self.blocks[id].ref_count += 1;
        Some(id)
    }

    /// Releases a single block:
    /// - If `ref_count` drops to `0` and the block was **committed** (`prefix_hash.is_some()`),
    ///   appends it to the **back** (MRU tail) of the free list so its cached prefix stays alive
    ///   as long as possible.
    /// - If `ref_count` drops to `0` and the block was **uncommitted** (`prefix_hash.is_none()`),
    ///   prepends it to the **front** (LRU head) of the free list so it is recycled immediately
    ///   before evicting any committed prefix block!
    pub fn release_block(&mut self, block_id: usize) {
        let Some(blk) = self.blocks.get_mut(block_id) else {
            return;
        };
        if blk.ref_count == 0 {
            return;
        }
        blk.ref_count -= 1;
        if blk.ref_count == 0 {
            let committed = blk.prefix_hash.is_some();
            if committed {
                self.push_free_back(block_id);
            } else {
                self.push_free_front(block_id);
            }
        }
    }

    /// Releases a request's chain of blocks `[b_0, b_1, ..., b_{k-1}]` in **reverse order**
    /// (`b_{k-1}` first, `b_0` last) so the tail of the chain sits closer to the LRU eviction
    /// head than the root prefix block `b_0`.
    pub fn release_request_chain(&mut self, chain: &[usize]) {
        for &block_id in chain.iter().rev() {
            self.release_block(block_id);
        }
    }
}

/// Hardware transfer execution path for moving KV-cache pages between host memory and GPU VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDmaTransferMode {
    /// Direct 1D Copy-Engine batch DMA (`cudaMemcpyBatchAsync`), bypassing SM kernels.
    DirectCopy1D,
    /// Direct 2D Pitched Copy-Engine DMA (`cudaMemcpy3DBatchAsync`) collapsing all `num_layers`
    /// of a block into a single pitched transfer (`EngineKvFormat::CrossLayer`).
    DirectCopy2DPitched,
    /// Staged H2D/D2H copy + SM scatter/gather kernel for non-contiguous layouts (`Hnd`).
    StagedSmKernel,
}

/// Returns `true` if `format` stores each page block as a contiguous `[block_size, nh, hs]` run
/// eligible for direct Copy-Engine DMA without SM gather/scatter kernels.
pub fn direct_copy_format_supported(format: EngineKvFormat) -> bool {
    matches!(
        format,
        EngineKvFormat::Nhd | EngineKvFormat::CrossLayer | EngineKvFormat::MlaLatent
    )
}

/// A single DMA copy descriptor, guaranteed not to straddle a `host_buffer_alignment` pin-chunk
/// boundary (`cudaHostRegister` region boundary).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvDmaCopyEntry {
    pub block_id: usize,
    pub host_offset: u64,
    pub device_offset: u64,
    pub width_bytes: u64,
    pub height_layers: usize,
}

/// Planned KV-cache transfer batch across a sequence of page blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvDmaTransferPlan {
    pub mode: KvDmaTransferMode,
    pub entries: Vec<KvDmaCopyEntry>,
    pub pin_boundary_splits: usize,
}

/// Plans a KV-cache H2D/D2H transfer for `block_ids` starting at `initial_host_offset`,
/// selecting Direct 1D/2D Copy-Engine DMA vs. Staged SM kernels and splitting any copy that
/// crosses a `host_buffer_alignment` (`cudaHostRegister`) boundary.
pub fn plan_kv_dma_transfer(
    format: EngineKvFormat,
    num_layers: usize,
    bytes_per_layer_block: u64,
    block_ids: &[usize],
    initial_host_offset: u64,
    host_buffer_alignment: u64,
) -> KvDmaTransferPlan {
    let layers = num_layers.max(1);
    let align = host_buffer_alignment.max(1);
    let mode = if !direct_copy_format_supported(format) {
        KvDmaTransferMode::StagedSmKernel
    } else if format == EngineKvFormat::CrossLayer && layers > 1 {
        KvDmaTransferMode::DirectCopy2DPitched
    } else {
        KvDmaTransferMode::DirectCopy1D
    };

    let mut entries = Vec::new();
    let mut pin_boundary_splits = 0usize;
    let mut cur_host = initial_host_offset;

    for &blk_id in block_ids {
        if mode == KvDmaTransferMode::DirectCopy2DPitched {
            let total_block_bytes = bytes_per_layer_block * (layers as u64);
            let mut rem = total_block_bytes;
            let mut dev_off = (blk_id as u64) * total_block_bytes;
            while rem > 0 {
                let chunk_rem = align - (cur_host % align);
                let step = rem.min(chunk_rem);
                if step < rem {
                    pin_boundary_splits += 1;
                }
                entries.push(KvDmaCopyEntry {
                    block_id: blk_id,
                    host_offset: cur_host,
                    device_offset: dev_off,
                    width_bytes: step,
                    height_layers: 1,
                });
                cur_host += step;
                dev_off += step;
                rem -= step;
            }
        } else {
            for layer_idx in 0..layers {
                let mut rem = bytes_per_layer_block;
                let mut dev_off =
                    ((layer_idx * 1024 + blk_id) as u64).saturating_mul(bytes_per_layer_block);
                while rem > 0 {
                    let chunk_rem = align - (cur_host % align);
                    let step = rem.min(chunk_rem);
                    if step < rem {
                        pin_boundary_splits += 1;
                    }
                    entries.push(KvDmaCopyEntry {
                        block_id: blk_id,
                        host_offset: cur_host,
                        device_offset: dev_off,
                        width_bytes: step,
                        height_layers: 1,
                    });
                    cur_host += step;
                    dev_off += step;
                    rem -= step;
                }
            }
        }
    }

    KvDmaTransferPlan {
        mode,
        entries,
        pin_boundary_splits,
    }
}

/// Fuses sole-use unary elementwise producers (`negate`, `abs`, `tanh`, `exponential`, `log`,
/// `sqrt`, `rsqrt`, `logistic`, `convert`) into the data-operand prologue of a `dot_general`,
/// `gather`, or `reduce` consumer.
///
/// Eliminates the intermediate producer buffer and evaluates the unary transform on-the-fly
/// during the consumer's load. Returns the number of prologues fused.
pub fn fuse_sole_use_prologues(module: &mut Module) -> usize {
    let mut fused_count = 0usize;

    for fn_op_id in module.functions() {
        let Some(&reg_id) = module.ops[fn_op_id].regions.first() else {
            continue;
        };
        let Some(&block_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let ops = module.blocks[block_id].ops.clone();

        let mut use_counts: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        for &op_id in &ops {
            if let Some(op) = module.ops.get(op_id) {
                for &opnd in &op.operands {
                    *use_counts.entry(opnd).or_insert(0) += 1;
                }
            }
        }

        let mut to_remove: std::collections::HashSet<OpId> = std::collections::HashSet::new();

        for &cons_id in &ops {
            if to_remove.contains(&cons_id) {
                continue;
            }
            let Some(cons_op) = module.ops.get(cons_id).cloned() else {
                continue;
            };
            let is_eligible_consumer = cons_op.is_op("stablehlo", "dot_general")
                || cons_op.is_op("stablehlo", "gather")
                || cons_op.is_op("stablehlo", "reduce");
            if !is_eligible_consumer
                || cons_op.operands.is_empty()
                || cons_op.attributes.contains_key("castforge.prologue_op")
            {
                continue;
            }
            let first_opnd = cons_op.operands[0];
            if use_counts.get(&first_opnd).copied().unwrap_or(0) != 1 {
                continue;
            }
            let ValueDef::OpResult { op: prod_id, .. } = module.values[first_opnd].def else {
                continue;
            };
            if to_remove.contains(&prod_id) {
                continue;
            }
            let Some(prod_op) = module.ops.get(prod_id).cloned() else {
                continue;
            };
            let is_unary_elementwise = prod_op.name.dialect == "stablehlo"
                && prod_op.operands.len() == 1
                && matches!(
                    prod_op.name.name.as_str(),
                    "negate"
                        | "abs"
                        | "tanh"
                        | "exponential"
                        | "log"
                        | "sqrt"
                        | "rsqrt"
                        | "logistic"
                        | "convert"
                );
            if !is_unary_elementwise {
                continue;
            }

            let raw_input = prod_op.operands[0];
            let prod_name = prod_op.name.name.clone();
            let merged_locs = [prod_op.location.clone(), cons_op.location.clone()];
            if let Some(cons_mut) = module.ops.get_mut(cons_id) {
                cons_mut.operands[0] = raw_input;
                cons_mut.attributes.insert(
                    "castforge.prologue_op".to_string(),
                    Attribute::String(prod_name),
                );
                stamp_fused_provenance(cons_mut, &merged_locs);
            }
            to_remove.insert(prod_id);
            fused_count += 1;
        }

        if !to_remove.is_empty() {
            module.blocks[block_id]
                .ops
                .retain(|id| !to_remove.contains(id));
            for dead_id in to_remove {
                module.ops.remove(dead_id);
            }
        }
    }
    fused_count
}

/// Default maximum number of kernels allowed in a single fused launch sequence.
pub const DEFAULT_MAX_KERNELS_IN_FUSION_SET: usize = 16;

/// A bounded group of consecutive device operations sharing the same execution context
/// (`device_context` and `stream_id`) that can be dispatched as a single fused sequence
/// without overflowing kernel constant-parameter memory or deadlocking across streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceFusionSet {
    pub set_id: usize,
    pub device_context: String,
    pub stream_id: i64,
    pub op_ids: Vec<OpId>,
    pub op_names: Vec<String>,
}

/// Partitions consecutive device kernels into bounded [`SequenceFusionSet`]s.
///
/// Breaks the active fusion set whenever:
/// 1. A collective, host-offload, or explicit barrier operation is encountered (`all_reduce`,
///    `all_gather`, `reduce_scatter`, `collective_permute`, `kv_offload`, `kv_reload`, or
///    `castforge.sequence_barrier = true`),
/// 2. The operation's `(castforge.device, castforge.stream_id)` context differs from the active
///    set's context (`useSameContexts`), or
/// 3. The active set reaches `max_kernels_per_set.max(1)` operations.
///
/// Each grouped operation is stamped with `castforge.sequence_fusion_set = <set_id>`.
pub fn plan_kernel_sequence_fusion(
    module: &mut Module,
    max_kernels_per_set: usize,
) -> Vec<SequenceFusionSet> {
    let cap = max_kernels_per_set.max(1);
    let mut sets: Vec<SequenceFusionSet> = Vec::new();
    let mut active: Option<SequenceFusionSet> = None;

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
            // Skip non-kernel metadata/constants and function terminators
            if op.is_op("func", "return") || op.is_op("stablehlo", "constant") {
                continue;
            }
            let op_short = op.name.name.as_str();
            let is_barrier = matches!(
                op_short,
                "all_reduce"
                    | "all_gather"
                    | "reduce_scatter"
                    | "collective_permute"
                    | "kv_offload"
                    | "kv_reload"
            ) || matches!(
                op.attributes.get("castforge.sequence_barrier"),
                Some(Attribute::Bool(true))
            );
            if is_barrier {
                if let Some(finished) = active.take() {
                    sets.push(finished);
                }
                continue;
            }

            let dev_ctx = match op.attributes.get("castforge.device") {
                Some(Attribute::String(s)) => s.clone(),
                _ => "gpu:0".to_string(),
            };
            let stream_id = match op.attributes.get("castforge.stream_id") {
                Some(Attribute::Int(id)) => *id,
                _ => 0,
            };
            let full_name = format!("{}.{}", op.name.dialect, op.name.name);

            let needs_new_set = match &active {
                None => true,
                Some(curr) => {
                    curr.device_context != dev_ctx
                        || curr.stream_id != stream_id
                        || curr.op_ids.len() >= cap
                }
            };
            if needs_new_set {
                if let Some(finished) = active.take() {
                    sets.push(finished);
                }
                let next_id = sets.len();
                active = Some(SequenceFusionSet {
                    set_id: next_id,
                    device_context: dev_ctx,
                    stream_id,
                    op_ids: Vec::new(),
                    op_names: Vec::new(),
                });
            }

            if let Some(curr) = active.as_mut() {
                let set_id = curr.set_id;
                curr.op_ids.push(op_id);
                curr.op_names.push(full_name);
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.attributes.insert(
                        "castforge.sequence_fusion_set".to_string(),
                        Attribute::Int(set_id as i64),
                    );
                }
            }
        }
        if let Some(finished) = active.take() {
            sets.push(finished);
        }
    }
    sets
}

/// Combines multiple constituent [`Location`]s into a single deduplicated fused location
/// (`Location::Named("op1+op2+...")`), preserving source provenance across operator fusions.
pub fn fuse_locations(locs: &[Location]) -> Location {
    let mut parts: Vec<String> = Vec::new();
    let mut single_flc: Option<Location> = None;
    let mut non_unknown_count = 0usize;

    for loc in locs {
        match loc {
            Location::Unknown => {}
            Location::Named(name) => {
                non_unknown_count += 1;
                for token in name.split('+') {
                    let trimmed = token.trim();
                    if !trimmed.is_empty() && !parts.iter().any(|p| p == trimmed) {
                        parts.push(trimmed.to_string());
                    }
                }
            }
            Location::FileLineCol { file, line, col } => {
                non_unknown_count += 1;
                if non_unknown_count == 1 {
                    single_flc = Some(loc.clone());
                } else {
                    single_flc = None;
                }
                let formatted = format!("{file}:{line}:{col}");
                if !parts.iter().any(|p| p == &formatted) {
                    parts.push(formatted);
                }
            }
        }
    }

    if parts.is_empty() {
        Location::Unknown
    } else if non_unknown_count == 1
        && parts.len() == 1
        && let Some(flc) = single_flc
    {
        flc
    } else {
        Location::Named(parts.join("+"))
    }
}

/// Stamps `target_op` with the combined [`Location`] from `source_locs` via [`fuse_locations`]
/// and records a human-readable `castforge.provenance` string attribute when known so lowered
/// Loop IR and target codegen retain fused operator provenance comments.
pub fn stamp_fused_provenance(target_op: &mut Operation, source_locs: &[Location]) {
    let fused = fuse_locations(source_locs);
    if fused != Location::Unknown {
        let prov_str = match &fused {
            Location::Named(s) => s.clone(),
            Location::FileLineCol { file, line, col } => format!("{file}:{line}:{col}"),
            Location::Unknown => String::new(),
        };
        target_op.location = fused;
        if !prov_str.is_empty() {
            target_op.attributes.insert(
                "castforge.provenance".to_string(),
                Attribute::String(prov_str),
            );
        }
    }
}
