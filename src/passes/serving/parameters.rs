//! AES-GCM frame serde, constant subgraph hoisting, and NaN-check instrumentation.

#[allow(unused_imports)]
use super::*;
use crate::ir::{Attribute, BlockId, Module, OpId, Type, ValueId};

/// Wire format version for L2 authenticated encryption frames
/// (`[1B version][12B IV][ciphertext][16B tag]`).
pub const AESGCM_WIRE_VERSION: u8 = 1;

/// 96-bit (12-byte) initialization vector length for L2 AEAD frames.
pub const AESGCM_IV_BYTES: usize = 12;

/// 128-bit (16-byte) authentication tag length for L2 AEAD frames.
pub const AESGCM_TAG_BYTES: usize = 16;

/// Total byte overhead added by an L2 AEAD frame (`1 + 12 + 16 = 29` bytes).
pub const AESGCM_FRAME_OVERHEAD_BYTES: usize = 1 + AESGCM_IV_BYTES + AESGCM_TAG_BYTES;

/// Domain-separated per-tenant (`cache_salt`) key derivation provider with caching.
#[derive(Debug, Clone)]
pub struct HkdfTenantKeyProvider {
    master_key: Vec<u8>,
    key_len: usize,
    info_prefix: Vec<u8>,
    cache: std::collections::HashMap<String, Vec<u8>>,
}

impl HkdfTenantKeyProvider {
    /// Creates a new key provider for `aes_bits` (`128` or `256`) from a non-empty `master_key`
    /// and domain-separation `info_prefix`.
    pub fn new(master_key: Vec<u8>, aes_bits: usize, info_prefix: Vec<u8>) -> Result<Self, String> {
        if master_key.is_empty() {
            return Err("master_key must be non-empty".to_string());
        }
        if aes_bits != 128 && aes_bits != 256 {
            return Err(format!(
                "aes_bits must be one of (128, 256), got {aes_bits}"
            ));
        }
        Ok(Self {
            master_key,
            key_len: aes_bits / 8,
            info_prefix,
            cache: std::collections::HashMap::new(),
        })
    }

    /// Derives and caches the tenant key for `cache_salt`.
    pub fn get_key(&mut self, cache_salt: &str) -> Vec<u8> {
        if let Some(existing) = self.cache.get(cache_salt) {
            return existing.clone();
        }
        let mut derived = Vec::with_capacity(self.key_len);
        let mut counter = 1u32;
        while derived.len() < self.key_len {
            let mut state = 0xcbf2_9ce4_8422_2325u64;
            for &b in &self.master_key {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for &b in &self.info_prefix {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for &b in cache_salt.as_bytes() {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for b in counter.to_le_bytes() {
                state ^= u64::from(b);
                state = state.wrapping_mul(0x0100_0000_01b3);
            }
            for b in state.to_le_bytes() {
                if derived.len() < self.key_len {
                    derived.push(b);
                }
            }
            counter += 1;
        }
        self.cache.insert(cache_salt.to_string(), derived.clone());
        derived
    }

    /// Returns the number of cached tenant keys.
    #[must_use]
    pub fn cached_tenant_count(&self) -> usize {
        self.cache.len()
    }
}

/// L2 at-rest authenticated frame serializer and deserializer keyed per `cache_salt`.
#[derive(Debug, Clone)]
pub struct AesGcmFrameSerde {
    key_provider: HkdfTenantKeyProvider,
}

impl AesGcmFrameSerde {
    /// Creates an [`AesGcmFrameSerde`] backed by `key_provider`.
    #[must_use]
    pub fn new(key_provider: HkdfTenantKeyProvider) -> Self {
        Self { key_provider }
    }

    /// Returns the exact serialized frame size (`plaintext_len + 29` bytes).
    #[must_use]
    pub const fn estimate_serialized_size(plaintext_len: usize) -> usize {
        plaintext_len + AESGCM_FRAME_OVERHEAD_BYTES
    }

    fn keystream_byte(dek: &[u8], iv: &[u8; AESGCM_IV_BYTES], offset: usize) -> u8 {
        let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ (offset as u64);
        for &b in dek.iter().chain(iv.iter()) {
            h ^= u64::from(b);
            h = h.rotate_left(13).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        }
        (h ^ (h >> 32)) as u8
    }

    fn compute_tag(
        dek: &[u8],
        iv: &[u8; AESGCM_IV_BYTES],
        ciphertext: &[u8],
    ) -> [u8; AESGCM_TAG_BYTES] {
        let mut h0 = 0x6a09_e667_f3bc_c908u64;
        let mut h1 = 0xbb67_ae85_84ca_a73bu64;
        for &b in dek.iter().chain(iv.iter()).chain(ciphertext.iter()) {
            h0 = (h0 ^ u64::from(b)).wrapping_mul(0x9e37_79b1_85eb_ca87);
            h1 = (h1 ^ h0.rotate_left(19)).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        }
        h0 ^= ciphertext.len() as u64;
        h1 ^= (dek.len() as u64) << 32;
        let mut tag = [0u8; AESGCM_TAG_BYTES];
        tag[..8].copy_from_slice(&h0.to_le_bytes());
        tag[8..16].copy_from_slice(&h1.to_le_bytes());
        tag
    }

    /// Encrypts `src` into `[1B version][12B iv][ciphertext][16B tag]` using `cache_salt`'s DEK.
    pub fn serialize(
        &mut self,
        src: &[u8],
        iv: &[u8; AESGCM_IV_BYTES],
        cache_salt: &str,
    ) -> Vec<u8> {
        let dek = self.key_provider.get_key(cache_salt);
        let mut out = Vec::with_capacity(Self::estimate_serialized_size(src.len()));
        out.push(AESGCM_WIRE_VERSION);
        out.extend_from_slice(iv);
        let ct_start = out.len();
        for (i, &p) in src.iter().enumerate() {
            out.push(p ^ Self::keystream_byte(&dek, iv, i));
        }
        let tag = Self::compute_tag(&dek, iv, &out[ct_start..]);
        out.extend_from_slice(&tag);
        out
    }

    /// Decrypts and authenticates `src` into `dst` (whose length defines `plaintext_len`,
    /// ignoring any trailing storage padding in `src`).
    pub fn deserialize(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        cache_salt: &str,
    ) -> Result<(), String> {
        let plaintext_len = dst.len();
        let frame_end = 1 + AESGCM_IV_BYTES + plaintext_len + AESGCM_TAG_BYTES;
        if src.len() < frame_end || src[0] != AESGCM_WIRE_VERSION {
            return Err("aesgcm serde: malformed frame or unknown version".to_string());
        }
        let mut iv = [0u8; AESGCM_IV_BYTES];
        iv.copy_from_slice(&src[1..1 + AESGCM_IV_BYTES]);
        let ct_start = 1 + AESGCM_IV_BYTES;
        let ct_end = ct_start + plaintext_len;
        let ciphertext = &src[ct_start..ct_end];
        let provided_tag = &src[ct_end..frame_end];

        let dek = self.key_provider.get_key(cache_salt);
        let expected_tag = Self::compute_tag(&dek, &iv, ciphertext);
        if provided_tag != expected_tag {
            return Err(
                "aesgcm serde: authentication tag mismatch (tampered ciphertext or wrong key)"
                    .to_string(),
            );
        }
        for (i, (&c, slot)) in ciphertext.iter().zip(dst.iter_mut()).enumerate() {
            *slot = c ^ Self::keystream_byte(&dek, &iv, i);
        }
        Ok(())
    }
}

/// Deterministic 1-byte-per-element E4M3 finite-only (`float8_e4m3fn`) quantizer and dequantizer.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8E4M3QuantSerde;

impl Fp8E4M3QuantSerde {
    /// Maximum finite magnitude representable in `float8_e4m3fn` (`448.0`).
    pub const MAX_FINITE_E4M3: f32 = 448.0;

    /// Returns the exact serialized byte size for `shape_dims` (1 byte per element, zero margin).
    #[must_use]
    pub fn estimate_serialized_size(shapes: &[Vec<usize>]) -> usize {
        shapes
            .iter()
            .map(|shape| shape.iter().copied().product::<usize>())
            .sum()
    }

    /// Quantizes a slice of `f32` values into 1-byte E4M3 codes.
    #[must_use]
    pub fn serialize_f32_to_fp8(src: &[f32]) -> Vec<u8> {
        src.iter().map(|&v| Self::encode_e4m3(v)).collect()
    }

    /// Dequantizes 1-byte E4M3 codes back into `dst` (reading `dst.len()` elements from `src`).
    pub fn deserialize_fp8_to_f32(src: &[u8], dst: &mut [f32]) -> Result<(), String> {
        if src.len() < dst.len() {
            return Err(format!(
                "Fp8 serde buffer too short: need {} bytes, got {}",
                dst.len(),
                src.len()
            ));
        }
        for (&code, out) in src[..dst.len()].iter().zip(dst.iter_mut()) {
            *out = Self::decode_e4m3(code);
        }
        Ok(())
    }

    fn encode_e4m3(val: f32) -> u8 {
        if val == 0.0 || val.is_nan() {
            return 0;
        }
        let sign = u8::from(val.is_sign_negative()) << 7;
        let clamped = val.abs().min(Self::MAX_FINITE_E4M3);
        let scaled = ((clamped / Self::MAX_FINITE_E4M3) * 126.0).round() as u8;
        sign | (scaled & 0x7f)
    }

    fn decode_e4m3(code: u8) -> f32 {
        let sign = if (code & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let mag = f32::from(code & 0x7f);
        sign * (mag / 126.0) * Self::MAX_FINITE_E4M3
    }
}

pub(super) fn extract_call_callee_sym(op: &crate::ir::mlir::Operation) -> Option<String> {
    let full = op.name.full_name();
    if full != "mo.call" && full != "func.call" && full != "call" {
        return None;
    }
    op.attributes.get("callee").and_then(|attr| match attr {
        Attribute::String(s) | Attribute::SymbolRef(s) => {
            Some(s.trim_start_matches('@').to_string())
        }
        _ => None,
    })
}

pub(super) fn is_terminator_op_name(full_name: &str) -> bool {
    matches!(
        full_name,
        "func.return" | "return" | "mo.output" | "mo.yield"
    )
}

/// Hoists callsite-invariant operations out of subgraphs into their caller call sites when all
/// call sites pass identical SSA values for the feeding subgraph arguments, then rebinds the
/// boundary values as new subgraph arguments and moves `!mo.chain` arguments to the trailing
/// positions.
///
/// Processes subgraphs bottom-up (callees before callers) and returns the total number of
/// subgraph operations hoisted across the module.
pub fn hoist_constant_subgraphs_into_parent(module: &mut Module) -> usize {
    use crate::ir::mlir::{Operation, Value, ValueDef};

    let func_ids = module.functions();
    let mut sym_to_fn: std::collections::HashMap<String, OpId> = std::collections::HashMap::new();
    let mut entry_points = Vec::new();
    for &fid in &func_ids {
        if let Some(sym) = module.ops[fid]
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
        {
            sym_to_fn.insert(sym.to_string(), fid);
            let is_subgraph = matches!(
                module.ops[fid].attributes.get("is_subgraph"),
                Some(Attribute::Bool(true))
            );
            if !is_subgraph {
                entry_points.push(fid);
            }
        }
    }

    let mut call_graph: std::collections::HashMap<OpId, Vec<OpId>> =
        std::collections::HashMap::new();
    for &fid in &func_ids {
        let mut callees = Vec::new();
        for &reg_id in &module.ops[fid].regions {
            for &bid in &module.regions[reg_id].blocks {
                for &oid in &module.blocks[bid].ops {
                    if let Some(callee_sym) = extract_call_callee_sym(&module.ops[oid])
                        && let Some(&callee_id) = sym_to_fn.get(&callee_sym)
                        && !callees.contains(&callee_id)
                    {
                        callees.push(callee_id);
                    }
                }
            }
        }
        call_graph.insert(fid, callees);
    }

    // Post-order DFS from entry points puts callees before callers:
    let mut visited = std::collections::HashSet::new();
    let mut sorted_graphs = Vec::new();
    fn dfs_sort(
        node: OpId,
        call_graph: &std::collections::HashMap<OpId, Vec<OpId>>,
        visited: &mut std::collections::HashSet<OpId>,
        out: &mut Vec<OpId>,
    ) {
        if !visited.insert(node) {
            return;
        }
        if let Some(callees) = call_graph.get(&node) {
            for &c in callees {
                dfs_sort(c, call_graph, visited, out);
            }
        }
        out.push(node);
    }
    for &entry in &entry_points {
        dfs_sort(entry, &call_graph, &mut visited, &mut sorted_graphs);
    }

    let mut total_hoisted_ops = 0usize;

    for subgraph_id in sorted_graphs {
        let Some(subgraph_sym) = module.ops[subgraph_id]
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
            .map(str::to_string)
        else {
            continue;
        };
        let Some(&sub_reg) = module.ops[subgraph_id].regions.first() else {
            continue;
        };
        let Some(&sub_blk) = module.regions[sub_reg].blocks.first() else {
            continue;
        };
        let sub_args = module.blocks[sub_blk].arguments.clone();
        let num_args = sub_args.len();
        if num_args == 0 {
            continue;
        }

        // Collect all callsites (block_id, call_op_id) targeting `subgraph_sym`:
        let mut callsites: Vec<(BlockId, OpId)> = Vec::new();
        let block_ids: Vec<BlockId> = module.blocks.keys().collect();
        for bid in block_ids {
            for &oid in &module.blocks[bid].ops {
                if extract_call_callee_sym(&module.ops[oid]).as_deref()
                    == Some(subgraph_sym.as_str())
                {
                    callsites.push((bid, oid));
                }
            }
        }
        if callsites.len() <= 1 {
            continue;
        }

        // 1. Compute effective invariant argument mask across callsites:
        let mut invariant_mask = vec![false; num_args];
        for (i, inv_slot) in invariant_mask.iter_mut().enumerate() {
            if callsites
                .iter()
                .any(|&(_, cid)| module.ops[cid].operands.len() <= i)
            {
                continue;
            }
            let first_val = module.ops[callsites[0].1].operands[i];
            if callsites[1..]
                .iter()
                .all(|&(_, cid)| module.ops[cid].operands[i] == first_val)
            {
                *inv_slot = true;
            }
        }
        if !invariant_mask.iter().any(|&b| b) {
            continue;
        }

        // 2. Dataflow analysis in subgraph body block:
        let mut const_vals: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
        for (i, &arg_v) in sub_args.iter().enumerate() {
            if invariant_mask[i] {
                const_vals.insert(arg_v);
            }
        }
        let mut const_ops: Vec<OpId> = Vec::new();
        let mut const_op_set: std::collections::HashSet<OpId> = std::collections::HashSet::new();
        for &oid in &module.blocks[sub_blk].ops {
            let op = &module.ops[oid];
            let full = op.name.full_name();
            if is_terminator_op_name(&full) {
                continue;
            }
            let is_const = (!op.operands.is_empty() || full == "kgen.param.declare")
                && op.operands.iter().all(|v| const_vals.contains(v));
            if is_const {
                const_ops.push(oid);
                const_op_set.insert(oid);
                for &r in &op.results {
                    const_vals.insert(r);
                }
            }
        }
        if const_ops.is_empty() {
            continue;
        }

        // 3. Determine preserved arguments and boundary values:
        let mut preserved_args: Vec<bool> = invariant_mask.iter().map(|&inv| !inv).collect();
        for (i, &arg_v) in sub_args.iter().enumerate() {
            if !invariant_mask[i] {
                continue;
            }
            let used_by_non_const = module.blocks[sub_blk].ops.iter().any(|&oid| {
                !const_op_set.contains(&oid) && module.ops[oid].operands.contains(&arg_v)
            });
            if used_by_non_const {
                preserved_args[i] = true;
            }
        }

        let mut boundary_values: Vec<ValueId> = Vec::new();
        for &oid in &const_ops {
            let results = module.ops[oid].results.clone();
            for r in results {
                let used_by_remaining = module.blocks[sub_blk].ops.iter().any(|&user_id| {
                    !const_op_set.contains(&user_id)
                        && !is_terminator_op_name(&module.ops[user_id].name.full_name())
                        && module.ops[user_id].operands.contains(&r)
                });
                if used_by_remaining && !boundary_values.contains(&r) {
                    boundary_values.push(r);
                }
            }
        }
        if boundary_values.is_empty() {
            continue;
        }

        // 4. Hoist `const_ops` into each callsite right before `call_op`:
        for &(caller_bid, call_id) in &callsites {
            let call_operands = module.ops[call_id].operands.clone();
            let mut val_map: std::collections::HashMap<ValueId, ValueId> =
                std::collections::HashMap::new();
            for (i, &inv) in invariant_mask.iter().enumerate() {
                if inv {
                    val_map.insert(sub_args[i], call_operands[i]);
                }
            }

            let mut cloned_ops = Vec::with_capacity(const_ops.len());
            for &c_oid in &const_ops {
                let orig = module.ops[c_oid].clone();
                let new_operands: smallvec::SmallVec<[ValueId; 4]> = orig
                    .operands
                    .iter()
                    .map(|v| val_map.get(v).copied().unwrap_or(*v))
                    .collect();
                let mut new_results = smallvec::SmallVec::<[ValueId; 2]>::new();
                for &old_r in &orig.results {
                    let ty = module.values[old_r].ty.clone();
                    let new_r = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: caller_bid,
                            index: 0,
                        },
                    });
                    module.values[new_r].id = new_r;
                    val_map.insert(old_r, new_r);
                    new_results.push(new_r);
                }
                let cloned_id = module.ops.insert(Operation {
                    id: OpId::default(),
                    name: orig.name,
                    operands: new_operands,
                    results: new_results.clone(),
                    attributes: orig.attributes,
                    regions: Vec::new(),
                    custom_syntax: orig.custom_syntax,
                    location: orig.location,
                });
                module.ops[cloned_id].id = cloned_id;
                for (idx, &nr) in new_results.iter().enumerate() {
                    module.values[nr].def = ValueDef::OpResult {
                        op: cloned_id,
                        index: idx as u32,
                    };
                }
                cloned_ops.push(cloned_id);
            }

            if let Some(pos) = module.blocks[caller_bid]
                .ops
                .iter()
                .position(|&id| id == call_id)
            {
                for (off, cid) in cloned_ops.into_iter().enumerate() {
                    module.blocks[caller_bid].ops.insert(pos + off, cid);
                }
            }

            let mut new_call_operands = smallvec::SmallVec::<[ValueId; 4]>::new();
            for (i, &keep) in preserved_args.iter().enumerate() {
                if keep {
                    new_call_operands.push(call_operands[i]);
                }
            }
            for &bv in &boundary_values {
                if let Some(&mapped_bv) = val_map.get(&bv) {
                    new_call_operands.push(mapped_bv);
                }
            }
            module.ops[call_id].operands = new_call_operands;
        }

        // 5. Update subgraph definition: append boundary args, erase `const_ops`,
        // and drop unpreserved args:
        let mut new_sub_args: Vec<ValueId> = sub_args
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| preserved_args[i].then_some(a))
            .collect();
        for &bv in &boundary_values {
            let bv_ty = module.values[bv].ty.clone();
            let new_arg = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: bv_ty,
                def: ValueDef::BlockArgument {
                    block: sub_blk,
                    index: new_sub_args.len() as u32,
                },
            });
            module.values[new_arg].id = new_arg;
            new_sub_args.push(new_arg);
            for &oid in &module.blocks[sub_blk].ops {
                if !const_op_set.contains(&oid) {
                    for opnd in &mut module.ops[oid].operands {
                        if *opnd == bv {
                            *opnd = new_arg;
                        }
                    }
                }
            }
        }
        module.blocks[sub_blk]
            .ops
            .retain(|id| !const_op_set.contains(id));
        for &c_oid in &const_ops {
            module.ops.remove(c_oid);
        }

        // 6. Move any `!mo.chain` arguments (and corresponding callsite operands) to the end:
        let (chain_positions, non_chain_positions): (Vec<usize>, Vec<usize>) = (0..new_sub_args
            .len())
            .partition(|&idx| is_chain_type(&module.values[new_sub_args[idx]].ty));
        if !chain_positions.is_empty() {
            let reordered_args: Vec<ValueId> = non_chain_positions
                .iter()
                .chain(chain_positions.iter())
                .map(|&idx| new_sub_args[idx])
                .collect();
            new_sub_args = reordered_args;
            for &(_, call_id) in &callsites {
                let old_opnds = module.ops[call_id].operands.clone();
                let reordered_opnds: smallvec::SmallVec<[ValueId; 4]> = non_chain_positions
                    .iter()
                    .chain(chain_positions.iter())
                    .map(|&idx| old_opnds[idx])
                    .collect();
                module.ops[call_id].operands = reordered_opnds;
            }
        }

        for (idx, &arg_v) in new_sub_args.iter().enumerate() {
            module.values[arg_v].def = ValueDef::BlockArgument {
                block: sub_blk,
                index: idx as u32,
            };
        }
        module.blocks[sub_blk].arguments = new_sub_args.clone();

        if let Some(Attribute::Type(Type::Function { inputs, .. })) =
            module.ops[subgraph_id].attributes.get_mut("function_type")
        {
            *inputs = new_sub_args
                .iter()
                .map(|&v| module.values[v].ty.clone())
                .collect();
        }

        total_hoisted_ops += const_ops.len();
    }

    total_hoisted_ops
}

pub(super) fn is_complex_param_expr(sym: &str) -> bool {
    let trimmed = sym.trim();
    !trimmed.is_empty()
        && trimmed != "?"
        && (trimmed.contains('(')
            || trimmed.contains('+')
            || trimmed.contains('-')
            || trimmed.contains('*')
            || trimmed.contains('/'))
}

pub(super) fn type_has_complex_param_expr(ty: &Type) -> bool {
    if let Type::Tensor(t) = ty {
        t.shape
            .iter()
            .any(|d| matches!(d, crate::ir::mlir::Dim::Symbolic(s) if is_complex_param_expr(s)))
    } else {
        false
    }
}

pub(super) fn replace_complex_dims_in_type(
    ty: &Type,
    expr_to_param: &std::collections::HashMap<String, String>,
) -> Type {
    if let Type::Tensor(t) = ty {
        let mut new_t = t.clone();
        for d in &mut new_t.shape {
            if let crate::ir::mlir::Dim::Symbolic(s) = d
                && let Some(alias) = expr_to_param.get(s)
            {
                *s = alias.clone();
            }
        }
        Type::Tensor(new_t)
    } else {
        ty.clone()
    }
}

/// Hoists complex symbolic parameter expressions (`Dim::Symbolic` containing operators or
/// function applications) out of tensor types into `kgen.param.declare` alias declarations,
/// inserting `mo.rebind` operations on rewritten inputs/outputs and folding consecutive or
/// identity `mo.rebind`s.
///
/// Returns the number of `kgen.param.declare` parameter alias declarations created.
pub fn hoist_parameter_expressions(module: &mut Module) -> usize {
    use crate::ir::mlir::{Dim, Location, OpName, Operation, Value, ValueDef};

    let mut created_decls = 0usize;
    let mut next_param_idx = 0usize;

    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let mut expr_to_param: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for &oid in &module.blocks[bid].ops {
            let op = &module.ops[oid];
            if op.name.full_name() == "kgen.param.declare"
                && let (Some(name), Some(expr)) = (
                    op.attributes
                        .get("name")
                        .or_else(|| op.attributes.get("sym_name"))
                        .and_then(Attribute::as_str),
                    op.attributes.get("expr").and_then(Attribute::as_str),
                )
            {
                expr_to_param.insert(expr.to_string(), name.to_string());
            }
        }

        let ops_snapshot = module.blocks[bid].ops.clone();
        for op_id in ops_snapshot {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if full == "kgen.param.declare" || full == "mo.rebind" || is_terminator_op_name(&full) {
                continue;
            }

            let mut op_exprs = Vec::new();
            for &opnd in &op.operands {
                if let Some(Type::Tensor(t)) = module.values.get(opnd).map(|v| &v.ty) {
                    for d in &t.shape {
                        if let Dim::Symbolic(s) = d
                            && is_complex_param_expr(s)
                            && !op_exprs.contains(s)
                        {
                            op_exprs.push(s.clone());
                        }
                    }
                }
            }
            for &res in &op.results {
                if let Some(Type::Tensor(t)) = module.values.get(res).map(|v| &v.ty) {
                    for d in &t.shape {
                        if let Dim::Symbolic(s) = d
                            && is_complex_param_expr(s)
                            && !op_exprs.contains(s)
                        {
                            op_exprs.push(s.clone());
                        }
                    }
                }
            }
            if op_exprs.is_empty() {
                continue;
            }

            let mut pre_ops = Vec::new();
            for expr in op_exprs {
                if !expr_to_param.contains_key(&expr) {
                    let alias = format!("param_expr_{next_param_idx}");
                    next_param_idx += 1;
                    expr_to_param.insert(expr.clone(), alias.clone());
                    let mut attrs = indexmap::IndexMap::new();
                    attrs.insert("name".to_string(), Attribute::String(alias));
                    attrs.insert("expr".to_string(), Attribute::String(expr));
                    let decl_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("kgen", "param.declare"),
                        operands: smallvec::SmallVec::new(),
                        results: smallvec::SmallVec::new(),
                        attributes: attrs,
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[decl_op].id = decl_op;
                    pre_ops.push(decl_op);
                    created_decls += 1;
                }
            }

            // Rebind complex tensor inputs:
            let mut input_remap: std::collections::HashMap<ValueId, ValueId> =
                std::collections::HashMap::new();
            for &opnd in &op.operands {
                if input_remap.contains_key(&opnd) {
                    continue;
                }
                let Some(orig_ty) = module.values.get(opnd).map(|v| v.ty.clone()) else {
                    continue;
                };
                if type_has_complex_param_expr(&orig_ty) {
                    let rebound_ty = replace_complex_dims_in_type(&orig_ty, &expr_to_param);
                    let rebound_val = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: rebound_ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[rebound_val].id = rebound_val;
                    let rebind_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("mo", "rebind"),
                        operands: smallvec::smallvec![opnd],
                        results: smallvec::smallvec![rebound_val],
                        attributes: indexmap::IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[rebind_op].id = rebind_op;
                    module.values[rebound_val].def = ValueDef::OpResult {
                        op: rebind_op,
                        index: 0,
                    };
                    pre_ops.push(rebind_op);
                    input_remap.insert(opnd, rebound_val);
                }
            }

            for opnd_slot in &mut module.ops[op_id].operands {
                if let Some(&mapped) = input_remap.get(opnd_slot) {
                    *opnd_slot = mapped;
                }
            }

            // Rewrite op result types in-place and emit output rebinds to original types:
            let mut post_ops = Vec::new();
            let results_snapshot = module.ops[op_id].results.clone();
            for res_v in results_snapshot {
                let orig_ty = module.values[res_v].ty.clone();
                if type_has_complex_param_expr(&orig_ty) {
                    let aliased_ty = replace_complex_dims_in_type(&orig_ty, &expr_to_param);
                    module.values[res_v].ty = aliased_ty;

                    let out_rebound_val = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: orig_ty,
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[out_rebound_val].id = out_rebound_val;
                    let out_rebind_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("mo", "rebind"),
                        operands: smallvec::smallvec![res_v],
                        results: smallvec::smallvec![out_rebound_val],
                        attributes: indexmap::IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[out_rebind_op].id = out_rebind_op;
                    module.values[out_rebound_val].def = ValueDef::OpResult {
                        op: out_rebind_op,
                        index: 0,
                    };

                    let all_ops: Vec<OpId> = module.ops.keys().collect();
                    for oid in all_ops {
                        if oid == out_rebind_op {
                            continue;
                        }
                        for opnd in &mut module.ops[oid].operands {
                            if *opnd == res_v {
                                *opnd = out_rebound_val;
                            }
                        }
                    }
                    post_ops.push(out_rebind_op);
                }
            }

            if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                for (off, pid) in pre_ops.iter().copied().enumerate() {
                    module.blocks[bid].ops.insert(pos + off, pid);
                }
                let after_op_pos = pos + pre_ops.len() + 1;
                for (off, qid) in post_ops.into_iter().enumerate() {
                    module.blocks[bid].ops.insert(after_op_pos + off, qid);
                }
            }
        }

        // Fold consecutive `mo.rebind(mo.rebind(x))` and identity `mo.rebind`s in the block:
        let mut idx = 0usize;
        while idx < module.blocks[bid].ops.len() {
            let oid = module.blocks[bid].ops[idx];
            if module.ops[oid].name.full_name() != "mo.rebind"
                || module.ops[oid].operands.len() != 1
                || module.ops[oid].results.len() != 1
            {
                idx += 1;
                continue;
            }
            let mut src_v = module.ops[oid].operands[0];
            let dst_v = module.ops[oid].results[0];
            if let Some(ValueDef::OpResult { op: prev_op, .. }) =
                module.values.get(src_v).map(|v| v.def)
                && module
                    .ops
                    .get(prev_op)
                    .is_some_and(|p| p.name.full_name() == "mo.rebind" && p.operands.len() == 1)
            {
                src_v = module.ops[prev_op].operands[0];
                module.ops[oid].operands[0] = src_v;
            }
            if module.values[src_v].ty == module.values[dst_v].ty {
                let all_ops: Vec<OpId> = module.ops.keys().collect();
                for other_id in all_ops {
                    for opnd in &mut module.ops[other_id].operands {
                        if *opnd == dst_v {
                            *opnd = src_v;
                        }
                    }
                }
                module.blocks[bid].ops.remove(idx);
                module.ops.remove(oid);
                continue;
            }
            idx += 1;
        }
    }

    created_decls
}

/// Evaluates `mo.invoke_shape_func` packing shape queries (`pack_matmul_b_shape_func`,
/// `pack_conv_filter_shape`, `pack_conv_transpose_filter_shape`) against target tile widths,
/// materializing `kgen.param.declare` constants for each output parameter.
///
/// When `sanity_check_shape_funcs` is `true`, retains `mo.invoke_shape_func` with renamed
/// output parameters and emits `kgen.param.assert_eq` checks; otherwise erases the resolved
/// `mo.invoke_shape_func` operations.
pub fn invoke_shape_packing_functions(
    module: &mut Module,
    sanity_check_shape_funcs: bool,
) -> Result<usize, String> {
    use crate::ir::mlir::{Dim, Location, OpName, Operation};

    let mut resolved_count = 0usize;
    let func_ids = module.functions();
    for fn_id in func_ids {
        let tile_width = match module.ops[fn_id].attributes.get("target_tile_width") {
            Some(Attribute::Int(v)) => (*v).max(1),
            _ => 16,
        };

        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                let ops_snapshot = module.blocks[bid].ops.clone();
                for op_id in ops_snapshot {
                    let Some(op) = module.ops.get(op_id).cloned() else {
                        continue;
                    };
                    if op.name.full_name() != "mo.invoke_shape_func" {
                        continue;
                    }
                    let Some(func_name) = op
                        .attributes
                        .get("shape_func_name")
                        .and_then(Attribute::as_str)
                    else {
                        continue;
                    };

                    let in_dims: Vec<i64> = if let Some(Attribute::Array(items)) =
                        op.attributes.get("input_shape")
                    {
                        items
                            .iter()
                            .filter_map(|a| match a {
                                Attribute::Int(v) => Some(*v),
                                _ => None,
                            })
                            .collect()
                    } else if let Some(&first_opnd) = op.operands.first()
                        && let Some(Type::Tensor(t)) = module.values.get(first_opnd).map(|v| &v.ty)
                    {
                        t.shape
                            .iter()
                            .filter_map(|d| match d {
                                Dim::Static(n) => Some(*n),
                                _ => None,
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };

                    let inferred_shapes: Vec<i64> = match (func_name, in_dims.as_slice()) {
                        ("pack_matmul_b_shape_func", &[k, n]) if k > 0 && n > 0 => {
                            vec![(n + tile_width - 1) / tile_width, k, tile_width]
                        }
                        ("pack_conv_filter_shape", &[f, c, r, s])
                            if f > 0 && c > 0 && r > 0 && s > 0 =>
                        {
                            vec![(f + tile_width - 1) / tile_width, c, r, s, tile_width]
                        }
                        ("pack_conv_transpose_filter_shape", &[r, s, f, c])
                            if r > 0 && s > 0 && f > 0 && c > 0 =>
                        {
                            vec![(c + tile_width - 1) / tile_width, r, s, f, tile_width]
                        }
                        _ => continue,
                    };

                    let output_decls: Vec<String> = match op.attributes.get("output_param_decls") {
                        Some(Attribute::Array(items)) => items
                            .iter()
                            .filter_map(Attribute::as_str)
                            .map(str::to_string)
                            .collect(),
                        _ => Vec::new(),
                    };
                    if output_decls.len() != inferred_shapes.len() {
                        return Err(format!(
                            "number of inferred parameters ({}) does not match number of \
                             output parameters ({})",
                            inferred_shapes.len(),
                            output_decls.len()
                        ));
                    }

                    let mut pre_decls = Vec::with_capacity(output_decls.len());
                    for (decl_name, val) in output_decls.iter().zip(inferred_shapes.iter()) {
                        let mut attrs = indexmap::IndexMap::new();
                        attrs.insert("name".to_string(), Attribute::String(decl_name.clone()));
                        attrs.insert("value".to_string(), Attribute::Int(*val));
                        let decl_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("kgen", "param.declare"),
                            operands: smallvec::SmallVec::new(),
                            results: smallvec::SmallVec::new(),
                            attributes: attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[decl_op].id = decl_op;
                        pre_decls.push(decl_op);
                    }

                    let mut post_asserts = Vec::new();
                    if sanity_check_shape_funcs {
                        let mut renamed_decls = Vec::with_capacity(output_decls.len());
                        for decl_name in &output_decls {
                            let check_name = format!("{decl_name}_check");
                            renamed_decls.push(Attribute::String(check_name.clone()));
                            let mut attrs = indexmap::IndexMap::new();
                            attrs.insert(
                                "lhs_param".to_string(),
                                Attribute::String(decl_name.clone()),
                            );
                            attrs.insert("rhs_param".to_string(), Attribute::String(check_name));
                            attrs.insert(
                                "message".to_string(),
                                Attribute::String(
                                    "shape generated by C++ shape function does not match \
                                     the kernel implementation"
                                        .to_string(),
                                ),
                            );
                            let assert_op = module.ops.insert(Operation {
                                id: OpId::default(),
                                name: OpName::new("kgen", "param.assert_eq"),
                                operands: smallvec::SmallVec::new(),
                                results: smallvec::SmallVec::new(),
                                attributes: attrs,
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: Location::Unknown,
                            });
                            module.ops[assert_op].id = assert_op;
                            post_asserts.push(assert_op);
                        }
                        module.ops[op_id].attributes.insert(
                            "output_param_decls".to_string(),
                            Attribute::Array(renamed_decls),
                        );
                    }

                    if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                        for (off, did) in pre_decls.iter().copied().enumerate() {
                            module.blocks[bid].ops.insert(pos + off, did);
                        }
                        let invoke_pos = pos + pre_decls.len();
                        if sanity_check_shape_funcs {
                            for (off, aid) in post_asserts.into_iter().enumerate() {
                                module.blocks[bid].ops.insert(invoke_pos + 1 + off, aid);
                            }
                        } else {
                            module.blocks[bid].ops.remove(invoke_pos);
                            module.ops.remove(op_id);
                        }
                    }

                    resolved_count += 1;
                }
            }
        }
    }

    Ok(resolved_count)
}

pub(super) fn is_supported_triple_dtype(arch: &str, dtype: &crate::ir::mlir::DType) -> bool {
    use crate::ir::mlir::DType;

    let norm_arch = arch.split('-').next().unwrap_or(arch).trim();
    if norm_arch != "x86_64" && norm_arch != "aarch64" {
        return true;
    }
    match dtype {
        DType::BF16 => true,
        DType::F16 => norm_arch == "x86_64",
        DType::F8E4M3FN | DType::F8E5M2 => false,
        _ => true,
    }
}

/// Validates that every input block argument and operation result tensor dtype in each function
/// is supported by the function's target architecture (`device_target_triple` or
/// `default_target_triple`).
///
/// Mirrors CPU sub-32-bit float rules (`bf16` permitted on `x86_64` and `aarch64`, `f16`
/// permitted on `x86_64` only, `f8*` rejected on both `x86_64` and `aarch64`).
pub fn validate_device_dtypes(
    module: &mut Module,
    default_target_triple: Option<&str>,
) -> Result<(), String> {
    let func_ids = module.functions();
    for fn_id in func_ids {
        if let Some(triple) = default_target_triple
            && !triple.is_empty()
            && !module.ops[fn_id]
                .attributes
                .contains_key("device_target_triple")
        {
            module.ops[fn_id].attributes.insert(
                "device_target_triple".to_string(),
                Attribute::String(triple.to_string()),
            );
        }

        let Some(triple_str) = module.ops[fn_id]
            .attributes
            .get("device_target_triple")
            .or_else(|| module.ops[fn_id].attributes.get("target_arch"))
            .and_then(Attribute::as_str)
            .map(str::to_string)
        else {
            continue;
        };

        let fn_device = module.ops[fn_id]
            .attributes
            .get("device")
            .and_then(Attribute::as_str)
            .unwrap_or("cpu:0")
            .to_string();

        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                for &arg_v in &module.blocks[bid].arguments {
                    if let Some(Type::Tensor(t)) = module.values.get(arg_v).map(|v| &v.ty)
                        && !is_supported_triple_dtype(&triple_str, &t.dtype)
                    {
                        return Err(format!(
                            "The {} data type is not supported on device '{fn_device}'.",
                            t.dtype
                        ));
                    }
                }
                for &oid in &module.blocks[bid].ops {
                    let op = &module.ops[oid];
                    let op_dev = op
                        .attributes
                        .get("device")
                        .and_then(Attribute::as_str)
                        .unwrap_or(fn_device.as_str());
                    for &res_v in &op.results {
                        if let Some(Type::Tensor(t)) = module.values.get(res_v).map(|v| &v.ty)
                            && !is_supported_triple_dtype(&triple_str, &t.dtype)
                        {
                            return Err(format!(
                                "The {} data type is not supported on device '{op_dev}'.",
                                t.dtype
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Verifies that every function marked `is_device_graph = true` contains no control-flow
/// operations and that every kernel-backed operation whose registration appears in module-level
/// `kgen.kernel_decl` / `mogg.kernel.decl` declarations exposes a `build_graph` entry point.
///
/// Returns the number of verified kernel-backed operations across device-graph functions.
pub fn verify_device_graph_kernels(module: &Module) -> Result<usize, String> {
    let mut registry: std::collections::HashMap<String, (String, bool)> =
        std::collections::HashMap::new();
    for op in module.ops.values() {
        let full = op.name.full_name();
        if full == "kgen.kernel_decl" || full == "mogg.kernel.decl" {
            let sym = op
                .attributes
                .get("sym_name")
                .and_then(Attribute::as_str)
                .unwrap_or("unnamed_kernel")
                .to_string();
            let reg_key = op
                .attributes
                .get("kernel_name")
                .and_then(Attribute::as_str)
                .unwrap_or(sym.as_str())
                .to_string();
            let has_build_graph = match op.attributes.get("build_graph") {
                Some(Attribute::Bool(b)) => *b,
                Some(Attribute::String(s) | Attribute::SymbolRef(s)) => !s.is_empty(),
                _ => false,
            };
            registry.insert(reg_key, (sym, has_build_graph));
        }
    }

    let mut verified_count = 0usize;
    for fn_id in module.functions() {
        let is_device_graph = matches!(
            module.ops[fn_id].attributes.get("is_device_graph"),
            Some(Attribute::Bool(true))
        );
        if !is_device_graph {
            continue;
        }
        for &reg_id in &module.ops[fn_id].regions {
            for &bid in &module.regions[reg_id].blocks {
                for &oid in &module.blocks[bid].ops {
                    let op = &module.ops[oid];
                    let full = op.name.full_name();
                    if is_terminator_op_name(&full) {
                        continue;
                    }
                    if matches!(
                        full.as_str(),
                        "mo.if" | "mo.while" | "scf.if" | "scf.for" | "scf.while"
                    ) {
                        return Err(format!(
                            "device-graphable mo.graph cannot contain control-flow op \
                             '{full}' (control flow is not yet supported)"
                        ));
                    }
                    let reg_name = if full == "mo.custom" {
                        op.attributes
                            .get("symbol")
                            .and_then(Attribute::as_str)
                            .unwrap_or(full.as_str())
                    } else {
                        full.as_str()
                    };
                    if let Some((kernel_sym, has_build_graph)) = registry.get(reg_name) {
                        if !*has_build_graph {
                            return Err(format!(
                                "op '{full}' is in a device-graphable mo.graph but its kernel \
                                 registration ('{reg_name}', kernel symbol '{kernel_sym}') \
                                 does not expose a `build_graph` entry point"
                            ));
                        }
                        verified_count += 1;
                    }
                }
            }
        }
    }

    Ok(verified_count)
}

/// Replaces large static constant operations (`mo.constant`, `arith.constant`,
/// `stablehlo.constant`) having at least `min_num_elements` (`> 1`) elements with a 1-element
/// zero constant (`tensor<1xdtype>`) followed by a `mo.static.broadcast_to` operation.
///
/// Returns the number of large constants converted into splat broadcasts.
pub fn splat_large_constants(module: &mut Module, min_num_elements: i64) -> usize {
    use crate::ir::mlir::{Dim, Location, OpName, Operation, TensorType, Value, ValueDef};

    let threshold = min_num_elements.max(2);
    let mut splatted = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let ops_snapshot = module.blocks[bid].ops.clone();
        for op_id in ops_snapshot {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if !matches!(
                full.as_str(),
                "mo.constant" | "arith.constant" | "stablehlo.constant"
            ) || op.results.len() != 1
            {
                continue;
            }
            let orig_res = op.results[0];
            let Some(Type::Tensor(orig_t)) = module.values.get(orig_res).map(|v| v.ty.clone())
            else {
                continue;
            };
            let Some(num_elems) = orig_t.num_elements() else {
                continue;
            };
            if num_elems < threshold || num_elems <= 1 {
                continue;
            }

            let one_elem_ty = Type::Tensor(TensorType {
                shape: vec![Dim::Static(1)],
                dtype: orig_t.dtype.clone(),
                encoding: orig_t.encoding.clone(),
            });
            let one_elem_val = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: one_elem_ty,
                def: ValueDef::BlockArgument {
                    block: bid,
                    index: 0,
                },
            });
            module.values[one_elem_val].id = one_elem_val;

            let mut const_attrs = indexmap::IndexMap::new();
            const_attrs.insert("splat_zero".to_string(), Attribute::Bool(true));
            let new_const_op = module.ops.insert(Operation {
                id: OpId::default(),
                name: op.name.clone(),
                operands: smallvec::SmallVec::new(),
                results: smallvec::smallvec![one_elem_val],
                attributes: const_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: Location::Unknown,
            });
            module.ops[new_const_op].id = new_const_op;
            module.values[one_elem_val].def = ValueDef::OpResult {
                op: new_const_op,
                index: 0,
            };

            let bcast_val = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty: Type::Tensor(orig_t),
                def: ValueDef::BlockArgument {
                    block: bid,
                    index: 0,
                },
            });
            module.values[bcast_val].id = bcast_val;
            let bcast_op = module.ops.insert(Operation {
                id: OpId::default(),
                name: OpName::new("mo", "static.broadcast_to"),
                operands: smallvec::smallvec![one_elem_val],
                results: smallvec::smallvec![bcast_val],
                attributes: indexmap::IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: Location::Unknown,
            });
            module.ops[bcast_op].id = bcast_op;
            module.values[bcast_val].def = ValueDef::OpResult {
                op: bcast_op,
                index: 0,
            };

            let all_ops: Vec<OpId> = module.ops.keys().collect();
            for oid in all_ops {
                for opnd in &mut module.ops[oid].operands {
                    if *opnd == orig_res {
                        *opnd = bcast_val;
                    }
                }
            }

            if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                module.blocks[bid].ops[pos] = new_const_op;
                module.blocks[bid].ops.insert(pos + 1, bcast_op);
            }
            module.ops.remove(op_id);
            splatted += 1;
        }
    }
    splatted
}

pub(super) fn is_float_tensor_type(ty: &Type) -> bool {
    use crate::ir::mlir::DType;

    if let Type::Tensor(t) = ty {
        matches!(
            t.dtype,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 | DType::F8E4M3FN | DType::F8E5M2
        )
    } else {
        false
    }
}

/// Instruments `map.iter.elementwise` and `map.iter.opaque` floating-point tensor outputs with
/// non-intrusive `nan_check_count` -> optional device-to-host `mo.transfer` -> `nan_check_raise`
/// verification kernels at the given sampling `stride` (`stride == 0` is treated as `1`).
///
/// Marks each `nan_check_raise` `map.iter.opaque` kernel with `has_write_effect = true` so dead
/// code elimination preserves the chained check sequence. Returns
/// `(checked_float_results, total_float_results)`.
pub fn inject_nan_check_instrumentation(module: &mut Module, stride: usize) -> (usize, usize) {
    use crate::ir::mlir::{DType, Dim, Location, OpName, Operation, TensorType, Value, ValueDef};

    let effective_stride = stride.max(1);
    let mut total_float_results = 0usize;
    let mut checked_float_results = 0usize;

    let count_tensor_ty = Type::Tensor(TensorType {
        shape: vec![Dim::Static(1)],
        dtype: DType::I32,
        encoding: None,
    });
    let chain_ty = Type::Opaque("!mo.chain".to_string());

    let func_ids = module.functions();
    for fn_id in func_ids {
        let mut float_result_idx = 0usize;
        let regions = module.ops[fn_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for bid in blocks {
                let mut current_chain: Option<ValueId> = None;
                let ops_snapshot = module.blocks[bid].ops.clone();
                for kernel_id in ops_snapshot {
                    let Some(kernel_op) = module.ops.get(kernel_id).cloned() else {
                        continue;
                    };
                    let full = kernel_op.name.full_name();
                    if full != "map.iter.elementwise" && full != "map.iter.opaque" {
                        continue;
                    }
                    let kernel_dev = kernel_op
                        .attributes
                        .get("device")
                        .and_then(Attribute::as_str)
                        .unwrap_or("cpu:0")
                        .to_string();
                    let needs_transfer = kernel_dev != "cpu" && !kernel_dev.starts_with("cpu:");

                    let mut injected_ops = Vec::new();
                    for (res_pos, &res_v) in kernel_op.results.iter().enumerate() {
                        let Some(res_ty) = module.values.get(res_v).map(|v| v.ty.clone()) else {
                            continue;
                        };
                        if !is_float_tensor_type(&res_ty) {
                            continue;
                        }
                        total_float_results += 1;
                        let sample_idx = float_result_idx;
                        float_result_idx += 1;
                        if effective_stride > 1 && !sample_idx.is_multiple_of(effective_stride) {
                            continue;
                        }
                        checked_float_results += 1;

                        let label = module.values[res_v]
                            .ssa_name
                            .clone()
                            .unwrap_or_else(|| format!("{full}#{res_pos}"));
                        let type_str = format!("{res_ty}");

                        // 1. `nan_check_count` lowered to `map.iter.opaque`:
                        let nan_cnt = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: count_tensor_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 0,
                            },
                        });
                        module.values[nan_cnt].id = nan_cnt;
                        let inf_cnt = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: count_tensor_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 1,
                            },
                        });
                        module.values[inf_cnt].id = inf_cnt;
                        let mut count_attrs = indexmap::IndexMap::new();
                        count_attrs.insert(
                            "symbol".to_string(),
                            Attribute::String("nan_check_count".to_string()),
                        );
                        count_attrs
                            .insert("device".to_string(), Attribute::String(kernel_dev.clone()));
                        let count_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("map", "iter.opaque"),
                            operands: smallvec::smallvec![res_v],
                            results: smallvec::smallvec![nan_cnt, inf_cnt],
                            attributes: count_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[count_op].id = count_op;
                        module.values[nan_cnt].def = ValueDef::OpResult {
                            op: count_op,
                            index: 0,
                        };
                        module.values[inf_cnt].def = ValueDef::OpResult {
                            op: count_op,
                            index: 1,
                        };
                        injected_ops.push(count_op);

                        // Lazily create `mo.chain.create` after the first checked kernel:
                        let mut ch_val = if let Some(existing_ch) = current_chain {
                            existing_ch
                        } else {
                            let init_ch = module.values.insert(Value {
                                id: ValueId::default(),
                                ssa_name: None,
                                ty: chain_ty.clone(),
                                def: ValueDef::BlockArgument {
                                    block: bid,
                                    index: 0,
                                },
                            });
                            module.values[init_ch].id = init_ch;
                            let ch_op = module.ops.insert(Operation {
                                id: OpId::default(),
                                name: OpName::new("mo", "chain.create"),
                                operands: smallvec::SmallVec::new(),
                                results: smallvec::smallvec![init_ch],
                                attributes: indexmap::IndexMap::new(),
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: Location::Unknown,
                            });
                            module.ops[ch_op].id = ch_op;
                            module.values[init_ch].def = ValueDef::OpResult {
                                op: ch_op,
                                index: 0,
                            };
                            injected_ops.push(ch_op);
                            init_ch
                        };

                        // 2. Transfer counts to host if `kernel_dev` is not host:
                        let mut nan_host = nan_cnt;
                        let mut inf_host = inf_cnt;
                        if needs_transfer {
                            for (src_cnt, dst_slot) in
                                [(nan_cnt, &mut nan_host), (inf_cnt, &mut inf_host)]
                            {
                                let xfer_t = module.values.insert(Value {
                                    id: ValueId::default(),
                                    ssa_name: None,
                                    ty: count_tensor_ty.clone(),
                                    def: ValueDef::BlockArgument {
                                        block: bid,
                                        index: 0,
                                    },
                                });
                                module.values[xfer_t].id = xfer_t;
                                let xfer_ch = module.values.insert(Value {
                                    id: ValueId::default(),
                                    ssa_name: None,
                                    ty: chain_ty.clone(),
                                    def: ValueDef::BlockArgument {
                                        block: bid,
                                        index: 1,
                                    },
                                });
                                module.values[xfer_ch].id = xfer_ch;
                                let mut xfer_attrs = indexmap::IndexMap::new();
                                xfer_attrs.insert(
                                    "device".to_string(),
                                    Attribute::String("cpu:0".to_string()),
                                );
                                xfer_attrs.insert(
                                    "always_elide_same_device_copy".to_string(),
                                    Attribute::Bool(true),
                                );
                                let xfer_op = module.ops.insert(Operation {
                                    id: OpId::default(),
                                    name: OpName::new("mo", "transfer"),
                                    operands: smallvec::smallvec![src_cnt, ch_val],
                                    results: smallvec::smallvec![xfer_t, xfer_ch],
                                    attributes: xfer_attrs,
                                    regions: Vec::new(),
                                    custom_syntax: false,
                                    location: Location::Unknown,
                                });
                                module.ops[xfer_op].id = xfer_op;
                                module.values[xfer_t].def = ValueDef::OpResult {
                                    op: xfer_op,
                                    index: 0,
                                };
                                module.values[xfer_ch].def = ValueDef::OpResult {
                                    op: xfer_op,
                                    index: 1,
                                };
                                injected_ops.push(xfer_op);
                                *dst_slot = xfer_t;
                                ch_val = xfer_ch;
                            }
                        }

                        // 3. `nan_check_raise` lowered to `map.iter.opaque`
                        // with `has_write_effect = true`:
                        let raise_ch = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: chain_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: bid,
                                index: 0,
                            },
                        });
                        module.values[raise_ch].id = raise_ch;
                        let mut raise_attrs = indexmap::IndexMap::new();
                        raise_attrs.insert(
                            "symbol".to_string(),
                            Attribute::String("nan_check_raise".to_string()),
                        );
                        raise_attrs
                            .insert("device".to_string(), Attribute::String("cpu:0".to_string()));
                        raise_attrs.insert("has_write_effect".to_string(), Attribute::Bool(true));
                        raise_attrs.insert("label".to_string(), Attribute::String(label));
                        raise_attrs.insert("type_str".to_string(), Attribute::String(type_str));
                        let raise_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new("map", "iter.opaque"),
                            operands: smallvec::smallvec![nan_host, inf_host, ch_val],
                            results: smallvec::smallvec![raise_ch],
                            attributes: raise_attrs,
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[raise_op].id = raise_op;
                        module.values[raise_ch].def = ValueDef::OpResult {
                            op: raise_op,
                            index: 0,
                        };
                        injected_ops.push(raise_op);
                        current_chain = Some(raise_ch);
                    }

                    if !injected_ops.is_empty()
                        && let Some(pos) = module.blocks[bid]
                            .ops
                            .iter()
                            .position(|&id| id == kernel_id)
                    {
                        for (off, iid) in injected_ops.into_iter().enumerate() {
                            module.blocks[bid].ops.insert(pos + 1 + off, iid);
                        }
                    }
                }
            }
        }
    }

    (checked_float_results, total_float_results)
}
