//! Cost-bounded MAP input fusion and destination-passing output/epilogue fusion.
//!
//! Implements the two core region-splicing fusion engines for structured index-space kernels
//! (`map.foreach`, `map.iter.elementwise`, and `map.iter.opaque`):
//! 1. **Cost-Bounded Input Fusion (`fuse_map_inputs`)**:
//!    - Evaluates tensor operand footprints (`LARGE_TENSOR_ELEMENT_THRESHOLD = 4096`) and target
//!      execution domains to determine when a multi-consumer elementwise producer is cheap to
//!      duplicate (`is_cheap_to_duplicate_map_foreach`).
//!    - Enforces mutable-chain single-use direct-edge legality (`check_mutable_kernel_fusible`)
//!      and verifies all users of a duplicated producer can be fused
//!      (`duplication_allowed_for_producer`) so the intermediate tensor buffer is completely
//!      eliminated.
//!    - Deduplicates redundant `map.load` operations at identical `(tensor, index)` coordinates
//!      within each consumer region (`merge_redundant_map_loads`) before splicing the producer's
//!      body at each load site, prioritizing single-use producers before multi-use candidates.
//! 2. **Destination-Passing Output & Epilogue Fusion (`fuse_map_outputs`)**:
//!    - Inspects how an elementwise consumer accesses a producer's output tensor: untransformed
//!      iteration coordinates (`consumer_reads_untransformed`), single bijective index reshape
//!      (`consumer_reads_through_reshape`), pure index reshape (`consumer_is_pure_reshape`), and
//!      mutable store legality (`is_fusible_mutable_store`).
//!    - Expands `map.iter.opaque` destination-passing outputs (`grow_opaque_outputs`) when the
//!      original output is still live, and inverts a consumer's `map.index.reshape` (swapping
//!      `in_shape` and `out_shape`) onto the fused epilogue `map.store` coordinate
//!      (`invert_reshape_on_epilogue_store`).

use crate::ir::{
    Attribute, BlockId, Dim, Location, Module, OpId, OpName, Operation, RegionId, Type, Value,
    ValueDef, ValueId,
};
use crate::passes::serving::is_chain_type;
use crate::passes::serving::kv_attention::stamp_fused_provenance;
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

/// Element count threshold above which a static tensor operand is classified as "large" for
/// producer duplication cost analysis (`4096` elements).
pub const LARGE_TENSOR_ELEMENT_THRESHOLD: i64 = 4096;

/// Returns `true` when `ty` is a tensor type that either has at least one dynamic/symbolic
/// dimension (`Dim::Dynamic` or `Dim::Symbolic`) or has a static element count strictly greater
/// than [`LARGE_TENSOR_ELEMENT_THRESHOLD`] (`4096`).
#[must_use]
pub fn is_large_tensor_operand(ty: &Type) -> bool {
    let Type::Tensor(t) = ty else {
        return false;
    };
    let mut prod = 1i64;
    for dim in &t.shape {
        match dim {
            Dim::Dynamic | Dim::Symbolic(_) => return true,
            Dim::Static(d) => {
                if *d < 0 {
                    return true;
                }
                prod = prod.saturating_mul(*d);
            }
        }
    }
    prod > LARGE_TENSOR_ELEMENT_THRESHOLD
}

/// Returns `true` when `op_id` is an elementwise map kernel (`map.foreach` or
/// `map.iter.elementwise`).
#[must_use]
pub fn is_elementwise_map_kernel(module: &Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    matches!(
        op.name.full_name().as_str(),
        "map.foreach" | "map.iter.elementwise"
    )
}

/// Returns `true` when `op_id` executes on a host/CPU device where duplicating producer kernels
/// across multiple consumers is disallowed.
#[must_use]
pub fn is_host_map_kernel(module: &Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    for key in ["device", "target", "execution_domain"] {
        if let Some(Attribute::String(s)) = op.attributes.get(key) {
            let lower = s.to_ascii_lowercase();
            if lower == "cpu" || lower == "host" || lower.starts_with("cpu:") {
                return true;
            }
        }
    }
    false
}

/// Returns `true` when `op_id` is an elementwise map kernel that is cheap to duplicate across
/// multiple consumers:
/// - Must not execute on a host/CPU target (`!is_host_map_kernel`).
/// - Must not carry an explicit `is_cheap_to_duplicate = false` attribute.
/// - Must read from at most 1 unique large tensor operand (`num_large_tensor_operands <= 1`) via
///   `map.load` operations in its body region (or carry `is_cheap_to_duplicate = true`).
#[must_use]
pub fn is_cheap_to_duplicate_map_foreach(module: &Module, op_id: OpId) -> bool {
    if !is_elementwise_map_kernel(module, op_id) || is_host_map_kernel(module, op_id) {
        return false;
    }
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    if let Some(Attribute::Bool(explicit)) = op.attributes.get("is_cheap_to_duplicate") {
        return *explicit;
    }

    let mut large_tensors: IndexSet<ValueId> = IndexSet::new();
    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &inner_op_id in &blk.ops {
                let Some(inner_op) = module.ops.get(inner_op_id) else {
                    continue;
                };
                if inner_op.name.full_name() == "map.load"
                    && let Some(&tensor_val) = inner_op.operands.first()
                    && let Some(val_info) = module.values.get(tensor_val)
                    && is_large_tensor_operand(&val_info.ty)
                {
                    large_tensors.insert(tensor_val);
                }
            }
        }
    }
    large_tensors.len() <= 1
}

/// Checks whether `producer_op` and `consumer_op` satisfy mutable-effect chain fusion legality:
/// - If neither kernel produces a `!mo.chain` token (or `has_mutable_load = true`), returns `true`.
/// - If `producer_op` produces a chain token, that token must have a single use, and that user
///   must be `consumer_op` directly (no intervening side-effecting operations on the chain).
#[must_use]
pub fn check_mutable_kernel_fusible(module: &Module, producer_op: OpId, consumer_op: OpId) -> bool {
    let Some(prod) = module.ops.get(producer_op) else {
        return false;
    };
    let Some(cons) = module.ops.get(consumer_op) else {
        return false;
    };

    let prod_chain_res = prod
        .results
        .iter()
        .copied()
        .find(|&r| module.values.get(r).is_some_and(|v| is_chain_type(&v.ty)));
    let prod_has_mutable = prod_chain_res.is_some()
        || matches!(
            prod.attributes.get("has_mutable_load"),
            Some(Attribute::Bool(true))
        );
    let cons_has_mutable = cons
        .results
        .iter()
        .any(|&r| module.values.get(r).is_some_and(|v| is_chain_type(&v.ty)))
        || matches!(
            cons.attributes.get("has_mutable_load"),
            Some(Attribute::Bool(true))
        );

    if !prod_has_mutable && !cons_has_mutable {
        return true;
    }

    if let Some(chain_val) = prod_chain_res {
        let mut chain_users: Vec<OpId> = Vec::new();
        for (cand_id, cand_op) in &module.ops {
            if cand_op.operands.contains(&chain_val) {
                chain_users.push(cand_id);
            }
        }
        if chain_users.len() != 1 || chain_users[0] != consumer_op {
            return false;
        }
    }
    true
}

/// Returns the number of fused kernels already accumulated into `op_id` (defaults to `1`).
#[must_use]
pub fn fused_kernel_count(module: &Module, op_id: OpId) -> usize {
    let Some(op) = module.ops.get(op_id) else {
        return 1;
    };
    match op.attributes.get("fused_kernel_count") {
        Some(Attribute::Int(n)) if *n > 0 => *n as usize,
        _ => 1,
    }
}

/// Inserts an unlinked operation with `res_types` into `module.ops` and `module.values` without
/// appending it to a block yet.
fn create_detached_op(
    module: &mut Module,
    name: OpName,
    operands: SmallVec<[ValueId; 4]>,
    res_types: &[Type],
    attributes: IndexMap<String, Attribute>,
    location: Location,
) -> (OpId, SmallVec<[ValueId; 2]>) {
    let op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
        operands,
        results: SmallVec::new(),
        attributes,
        regions: Vec::new(),
        custom_syntax: false,
        location,
    });
    let mut results = SmallVec::with_capacity(res_types.len());
    for (idx, ty) in res_types.iter().cloned().enumerate() {
        let val_id = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty,
            def: ValueDef::OpResult {
                op: op_id,
                index: idx as u32,
            },
        });
        results.push(val_id);
    }
    module.ops[op_id].results = results.clone();
    (op_id, results)
}

/// Deduplicates redundant `map.load` operations within each block of `region_id` that read from
/// the exact same `(tensor_operand, index_operands)` tuple.
///
/// Returns the number of redundant `map.load` operations eliminated.
pub fn merge_redundant_map_loads(module: &mut Module, region_id: RegionId) -> usize {
    let Some(reg) = module.regions.get(region_id) else {
        return 0;
    };
    let block_ids = reg.blocks.clone();
    let mut merged_count = 0usize;

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let ops = blk.ops.clone();
        let mut seen_loads: IndexMap<SmallVec<[ValueId; 4]>, ValueId> = IndexMap::new();
        let mut replacements: IndexMap<ValueId, ValueId> = IndexMap::new();
        let mut dead_ops: IndexSet<OpId> = IndexSet::new();

        for &op_id in &ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.name.full_name() == "map.load" && op.results.len() == 1 && !op.operands.is_empty()
            {
                let key: SmallVec<[ValueId; 4]> = op
                    .operands
                    .iter()
                    .map(|&v| replacements.get(&v).copied().unwrap_or(v))
                    .collect();
                let res_val = op.results[0];
                if let Some(&canon_val) = seen_loads.get(&key) {
                    replacements.insert(res_val, canon_val);
                    dead_ops.insert(op_id);
                    merged_count += 1;
                } else {
                    seen_loads.insert(key, res_val);
                }
            }
        }

        if !replacements.is_empty() {
            for &op_id in &ops {
                if dead_ops.contains(&op_id) {
                    continue;
                }
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    for opnd in &mut op_mut.operands {
                        if let Some(&new_v) = replacements.get(opnd) {
                            *opnd = new_v;
                        }
                    }
                }
            }
            for dead_id in dead_ops {
                module.remove_op(blk_id, dead_id);
            }
        }
    }

    merged_count
}

/// Finds the enclosing kernel operation (`map.foreach`, `map.iter.elementwise`, or
/// `map.iter.opaque`) in `enclosing_block` whose body/prologue region contains `inner_op_id`.
fn find_enclosing_map_kernel_in_block(
    module: &Module,
    enclosing_block: BlockId,
    inner_op_id: OpId,
) -> Option<(OpId, RegionId)> {
    let blk = module.blocks.get(enclosing_block)?;
    for &cand_op_id in &blk.ops {
        let Some(cand_op) = module.ops.get(cand_op_id) else {
            continue;
        };
        let full_name = cand_op.name.full_name();
        if !matches!(
            full_name.as_str(),
            "map.foreach" | "map.iter.elementwise" | "map.iter.opaque"
        ) {
            continue;
        }
        for (reg_idx, &reg_id) in cand_op.regions.iter().enumerate() {
            if full_name == "map.iter.opaque" && reg_idx > 0 {
                continue;
            }
            let Some(reg) = module.regions.get(reg_id) else {
                continue;
            };
            for &b_id in &reg.blocks {
                if let Some(b) = module.blocks.get(b_id)
                    && b.ops.contains(&inner_op_id)
                {
                    return Some((cand_op_id, reg_id));
                }
            }
        }
    }
    None
}

/// Collects all uses of `tensor_val` across the module, returning `(load_op_id, consumer_kernel_id,
/// target_region_id)` if every single use of `tensor_val` is either:
/// - A `map.load` tensor operand (operand 0) inside a valid consumer kernel in `enclosing_block`,
///   or
/// - A top-level operand on that consumer kernel corresponding to the `map.load`.
fn collect_fusible_load_sites(
    module: &Module,
    enclosing_block: BlockId,
    producer_op: OpId,
    tensor_val: ValueId,
    max_fusion_set_size: Option<usize>,
) -> Option<Vec<(OpId, OpId, RegionId)>> {
    let prod_count = fused_kernel_count(module, producer_op);
    let mut sites: Vec<(OpId, OpId, RegionId)> = Vec::new();
    let mut consumer_kernels: IndexSet<OpId> = IndexSet::new();

    for (op_id, op) in &module.ops {
        if !op.operands.contains(&tensor_val) {
            continue;
        }
        if op.name.full_name() == "map.load" {
            if op.operands.first().copied() != Some(tensor_val)
                || op.operands.iter().skip(1).any(|&v| v == tensor_val)
            {
                return None;
            }
            let (cons_id, reg_id) =
                find_enclosing_map_kernel_in_block(module, enclosing_block, op_id)?;
            if cons_id == producer_op || !check_mutable_kernel_fusible(module, producer_op, cons_id)
            {
                return None;
            }
            if let Some(limit) = max_fusion_set_size {
                let cons_count = fused_kernel_count(module, cons_id);
                if prod_count + cons_count > limit {
                    return None;
                }
            }
            consumer_kernels.insert(cons_id);
            sites.push((op_id, cons_id, reg_id));
        } else if matches!(
            op.name.full_name().as_str(),
            "map.foreach" | "map.iter.elementwise" | "map.iter.opaque"
        ) {
            if op_id == producer_op {
                return None;
            }
            consumer_kernels.insert(op_id);
        } else {
            return None;
        }
    }

    if sites.is_empty() {
        return None;
    }
    for cons_id in consumer_kernels {
        if !sites.iter().any(|&(_, c, _)| c == cons_id) {
            return None;
        }
    }
    Some(sites)
}

/// Returns `true` if `producer_op` can be fused (and duplicated if multi-use) into all of its
/// consumers within `enclosing_block` under `max_fusion_set_size`.
#[must_use]
pub fn duplication_allowed_for_producer(
    module: &Module,
    enclosing_block: BlockId,
    producer_op: OpId,
    max_fusion_set_size: Option<usize>,
) -> bool {
    if !is_elementwise_map_kernel(module, producer_op) {
        return false;
    }
    let Some(prod) = module.ops.get(producer_op) else {
        return false;
    };
    let Some(&tensor_val) = prod.results.first() else {
        return false;
    };
    let Some(sites) = collect_fusible_load_sites(
        module,
        enclosing_block,
        producer_op,
        tensor_val,
        max_fusion_set_size,
    ) else {
        return false;
    };
    let distinct_consumers: IndexSet<OpId> = sites.iter().map(|&(_, c, _)| c).collect();
    if distinct_consumers.len() > 1 && !is_cheap_to_duplicate_map_foreach(module, producer_op) {
        return false;
    }
    true
}

/// Splices the body of `producer_op` right before `load_op_id` inside `consumer_op`'s region,
/// mapping the producer's iteration-index block arguments to `load_op_id`'s index operands
/// (`operands[1..]`), replacing `load_op_id`'s result with the value yielded by `producer_op`,
/// and removing `load_op_id`.
fn splice_producer_at_load_site(
    module: &mut Module,
    producer_op: OpId,
    load_op_id: OpId,
    consumer_op: OpId,
) -> bool {
    let Some(prod_op) = module.ops.get(producer_op).cloned() else {
        return false;
    };
    let Some(&prod_reg_id) = prod_op.regions.first() else {
        return false;
    };
    let Some(prod_reg) = module.regions.get(prod_reg_id) else {
        return false;
    };
    let Some(&prod_blk_id) = prod_reg.blocks.first() else {
        return false;
    };
    let Some(prod_blk) = module.blocks.get(prod_blk_id).cloned() else {
        return false;
    };

    let Some(load_op) = module.ops.get(load_op_id).cloned() else {
        return false;
    };
    if load_op.results.len() != 1 || load_op.operands.is_empty() {
        return false;
    }
    let load_res = load_op.results[0];
    let load_indices: SmallVec<[ValueId; 4]> = load_op.operands.iter().skip(1).copied().collect();

    if !prod_blk.arguments.is_empty() && prod_blk.arguments.len() != load_indices.len() {
        return false;
    }

    let Some(cons_op) = module.ops.get(consumer_op).cloned() else {
        return false;
    };
    let mut target_blk_id: Option<BlockId> = None;
    for &r_id in &cons_op.regions {
        if let Some(r) = module.regions.get(r_id) {
            for &b_id in &r.blocks {
                if let Some(b) = module.blocks.get(b_id)
                    && b.ops.contains(&load_op_id)
                {
                    target_blk_id = Some(b_id);
                    break;
                }
            }
        }
    }
    let Some(cons_blk_id) = target_blk_id else {
        return false;
    };

    let mut val_map: IndexMap<ValueId, ValueId> = IndexMap::new();
    for (&arg_val, &idx_val) in prod_blk.arguments.iter().zip(load_indices.iter()) {
        val_map.insert(arg_val, idx_val);
    }

    let mut cloned_ops: Vec<OpId> = Vec::new();
    let mut yielded_val: Option<ValueId> = None;

    for &inner_id in &prod_blk.ops {
        let Some(inner_op) = module.ops.get(inner_id).cloned() else {
            continue;
        };
        let full_name = inner_op.name.full_name();
        if full_name == "map.yield" {
            if let Some(&y_opnd) = inner_op.operands.first() {
                yielded_val = Some(val_map.get(&y_opnd).copied().unwrap_or(y_opnd));
            }
            break;
        }

        let new_operands: SmallVec<[ValueId; 4]> = inner_op
            .operands
            .iter()
            .map(|&v| val_map.get(&v).copied().unwrap_or(v))
            .collect();
        let res_types: Vec<Type> = inner_op
            .results
            .iter()
            .filter_map(|&r| module.values.get(r).map(|v| v.ty.clone()))
            .collect();

        let (new_op_id, new_results) = create_detached_op(
            module,
            inner_op.name.clone(),
            new_operands,
            &res_types,
            inner_op.attributes.clone(),
            inner_op.location.clone(),
        );
        for (&old_r, &new_r) in inner_op.results.iter().zip(new_results.iter()) {
            val_map.insert(old_r, new_r);
        }
        cloned_ops.push(new_op_id);
    }

    let Some(replacement_val) = yielded_val else {
        return false;
    };

    if let Some(cons_blk) = module.blocks.get_mut(cons_blk_id)
        && let Some(pos) = cons_blk.ops.iter().position(|&id| id == load_op_id)
    {
        for (offset, &new_id) in cloned_ops.iter().enumerate() {
            cons_blk.ops.insert(pos + offset, new_id);
        }
    }

    if let Some(cons_blk) = module.blocks.get(cons_blk_id) {
        let blk_ops = cons_blk.ops.clone();
        for &op_id in &blk_ops {
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                for opnd in &mut op_mut.operands {
                    if *opnd == load_res {
                        *opnd = replacement_val;
                    }
                }
            }
        }
    }

    module.remove_op(cons_blk_id, load_op_id);

    let prod_tensor = prod_op.results.first().copied();
    let prod_inputs: SmallVec<[ValueId; 4]> = prod_op
        .operands
        .iter()
        .copied()
        .filter(|&v| module.values.get(v).is_none_or(|vi| !is_chain_type(&vi.ty)))
        .collect();
    let prod_fused = fused_kernel_count(module, producer_op);
    let cons_fused = fused_kernel_count(module, consumer_op);
    let merged_locs = [prod_op.location.clone(), cons_op.location.clone()];

    if let Some(cons_mut) = module.ops.get_mut(consumer_op) {
        if let Some(pt) = prod_tensor {
            cons_mut.operands.retain(|v| *v != pt);
        }
        for &inp in &prod_inputs {
            if !cons_mut.operands.contains(&inp) {
                cons_mut.operands.push(inp);
            }
        }
        cons_mut.attributes.insert(
            "fused_kernel_count".to_string(),
            Attribute::Int((prod_fused + cons_fused) as i64),
        );
        stamp_fused_provenance(cons_mut, &merged_locs);
    }

    true
}

/// Fuses elementwise map producers (`map.foreach` / `map.iter.elementwise`) into consumer map
/// kernels (`map.foreach`, `map.iter.elementwise`, and `map.iter.opaque` prologues).
///
/// Runs in two priority phases per block:
/// 1. Single-use producers (`single_use_only = true`).
/// 2. Multi-use cheap-to-duplicate producers (`single_use_only = false`) where every consumer load
///    site is valid (`duplication_allowed_for_producer`), ensuring the intermediate buffer is
///    completely eliminated.
///
/// Before splicing a producer into a consumer region, deduplicates redundant `map.load` operations
/// at identical `(tensor, index)` coordinates (`merge_redundant_map_loads`).
///
/// Returns the total number of producer-to-consumer input fusions performed.
pub fn fuse_map_inputs(module: &mut Module, max_fusion_set_size: Option<usize>) -> usize {
    let mut total_fused = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        for single_use_only in [true, false] {
            while let Some(blk) = module.blocks.get(blk_id) {
                let ops = blk.ops.clone();
                let mut fused_in_pass = false;

                for &prod_id in &ops {
                    if !is_elementwise_map_kernel(module, prod_id) {
                        continue;
                    }
                    let Some(prod_op) = module.ops.get(prod_id).cloned() else {
                        continue;
                    };
                    let Some(&tensor_val) = prod_op.results.first() else {
                        continue;
                    };
                    if !duplication_allowed_for_producer(
                        module,
                        blk_id,
                        prod_id,
                        max_fusion_set_size,
                    ) {
                        continue;
                    }
                    let Some(initial_sites) = collect_fusible_load_sites(
                        module,
                        blk_id,
                        prod_id,
                        tensor_val,
                        max_fusion_set_size,
                    ) else {
                        continue;
                    };
                    let distinct_consumers: IndexSet<OpId> =
                        initial_sites.iter().map(|&(_, c, _)| c).collect();
                    if single_use_only && distinct_consumers.len() > 1 {
                        continue;
                    }

                    let target_regions: IndexSet<RegionId> =
                        initial_sites.iter().map(|&(_, _, r)| r).collect();
                    for reg_id in target_regions {
                        merge_redundant_map_loads(module, reg_id);
                    }

                    let Some(sites) = collect_fusible_load_sites(
                        module,
                        blk_id,
                        prod_id,
                        tensor_val,
                        max_fusion_set_size,
                    ) else {
                        continue;
                    };

                    let mut all_spliced = true;
                    for (load_id, cons_id, _) in sites {
                        if splice_producer_at_load_site(module, prod_id, load_id, cons_id) {
                            total_fused += 1;
                        } else {
                            all_spliced = false;
                        }
                    }

                    if all_spliced {
                        if let Some(&chain_res) = prod_op
                            .results
                            .iter()
                            .find(|&&r| module.values.get(r).is_some_and(|v| is_chain_type(&v.ty)))
                            && let Some(&chain_in) = prod_op.operands.iter().find(|&&v| {
                                module.values.get(v).is_some_and(|vi| is_chain_type(&vi.ty))
                            })
                        {
                            for &cons_id in &distinct_consumers {
                                if let Some(cons_mut) = module.ops.get_mut(cons_id) {
                                    for opnd in &mut cons_mut.operands {
                                        if *opnd == chain_res {
                                            *opnd = chain_in;
                                        }
                                    }
                                }
                            }
                        }
                        module.remove_op(blk_id, prod_id);
                        fused_in_pass = true;
                        break;
                    }
                }

                if !fused_in_pass {
                    break;
                }
            }
        }
    }

    total_fused
}

// ============================================================================
// Output & Epilogue Fusion (`OutputFusion.cpp`)
// ============================================================================

/// Returns `true` when `consumer_op` (`map.foreach` / `map.iter.elementwise`) reads `target_tensor`
/// exclusively through `map.load` operations whose index coordinates are the exact untransformed
/// block arguments of `consumer_op`'s body block.
#[must_use]
pub fn consumer_reads_untransformed(
    module: &Module,
    consumer_op: OpId,
    target_tensor: ValueId,
) -> bool {
    if !is_elementwise_map_kernel(module, consumer_op) {
        return false;
    }
    let Some(cons) = module.ops.get(consumer_op) else {
        return false;
    };
    let Some(&reg_id) = cons.regions.first() else {
        return false;
    };
    let Some(reg) = module.regions.get(reg_id) else {
        return false;
    };
    let Some(&blk_id) = reg.blocks.first() else {
        return false;
    };
    let Some(blk) = module.blocks.get(blk_id) else {
        return false;
    };

    let mut found_load = false;
    for &inner_id in &blk.ops {
        let Some(inner_op) = module.ops.get(inner_id) else {
            continue;
        };
        if !inner_op.operands.contains(&target_tensor) {
            continue;
        }
        if inner_op.name.full_name() != "map.load" {
            return false;
        }
        if inner_op.operands.first().copied() != Some(target_tensor) {
            return false;
        }
        let load_indices = &inner_op.operands[1..];
        if load_indices != blk.arguments.as_slice() {
            return false;
        }
        found_load = true;
    }
    found_load
}

/// Returns `true` when `consumer_op` is a pure index reshape over `target_tensor`: its body block
/// contains only a single `map.index.reshape` on the block arguments, a single `map.load` of
/// `target_tensor` at those reshaped coordinates, and a `map.yield` of that loaded value.
#[must_use]
pub fn consumer_is_pure_reshape(
    module: &Module,
    consumer_op: OpId,
    target_tensor: ValueId,
) -> bool {
    if !is_elementwise_map_kernel(module, consumer_op) {
        return false;
    }
    let Some(cons) = module.ops.get(consumer_op) else {
        return false;
    };
    let Some(&reg_id) = cons.regions.first() else {
        return false;
    };
    let Some(reg) = module.regions.get(reg_id) else {
        return false;
    };
    let Some(&blk_id) = reg.blocks.first() else {
        return false;
    };
    let Some(blk) = module.blocks.get(blk_id) else {
        return false;
    };
    if blk.ops.len() != 3 {
        return false;
    }
    let Some(op0) = module.ops.get(blk.ops[0]) else {
        return false;
    };
    let Some(op1) = module.ops.get(blk.ops[1]) else {
        return false;
    };
    let Some(op2) = module.ops.get(blk.ops[2]) else {
        return false;
    };
    if op0.name.full_name() != "map.index.reshape"
        || op0.operands.as_slice() != blk.arguments.as_slice()
    {
        return false;
    }
    if op1.name.full_name() != "map.load"
        || op1.operands.first().copied() != Some(target_tensor)
        || op1.operands[1..] != *op0.results
        || op1.results.len() != 1
    {
        return false;
    }
    op2.name.full_name() == "map.yield" && op2.operands.as_slice() == op1.results.as_slice()
}

/// Checks whether `consumer_op` reads `target_tensor` exclusively through a single
/// `map.index.reshape` applied to `consumer_op`'s untransformed block arguments.
///
/// When `consumer_op` performs additional elementwise compute beyond a pure reshape, requires that
/// `consumer_op` does not load from any other tensor so that inverting the reshape onto the
/// epilogue store coordinate remains valid.
#[must_use]
pub fn consumer_reads_through_reshape(
    module: &Module,
    consumer_op: OpId,
    target_tensor: ValueId,
) -> Option<OpId> {
    if !is_elementwise_map_kernel(module, consumer_op) {
        return None;
    }
    let cons = module.ops.get(consumer_op)?;
    let &reg_id = cons.regions.first()?;
    let reg = module.regions.get(reg_id)?;
    let &blk_id = reg.blocks.first()?;
    let blk = module.blocks.get(blk_id)?;

    let mut reshape_op_id: Option<OpId> = None;
    let mut load_count = 0usize;
    let mut other_load_count = 0usize;

    for &inner_id in &blk.ops {
        let Some(inner_op) = module.ops.get(inner_id) else {
            continue;
        };
        let full_name = inner_op.name.full_name();
        if full_name == "map.index.reshape" {
            if reshape_op_id.is_some() || inner_op.operands.as_slice() != blk.arguments.as_slice() {
                return None;
            }
            reshape_op_id = Some(inner_id);
        } else if full_name == "map.load" {
            if inner_op.operands.first().copied() == Some(target_tensor) {
                let r_id = reshape_op_id?;
                let r_op = module.ops.get(r_id)?;
                if inner_op.operands[1..] != *r_op.results {
                    return None;
                }
                load_count += 1;
            } else {
                other_load_count += 1;
            }
        } else if inner_op.operands.contains(&target_tensor) {
            return None;
        }
    }

    if load_count != 1 {
        return None;
    }
    if other_load_count > 0 && !consumer_is_pure_reshape(module, consumer_op, target_tensor) {
        return None;
    }
    reshape_op_id
}

/// Checks whether a consumer kernel containing a `map.store` writes at untransformed iteration
/// coordinates and its stored value does not conflict with `target_tensor`.
#[must_use]
pub fn is_fusible_mutable_store(
    module: &Module,
    consumer_op: OpId,
    target_tensor: ValueId,
) -> bool {
    let Some(cons) = module.ops.get(consumer_op) else {
        return false;
    };
    let Some(&reg_id) = cons.regions.first() else {
        return false;
    };
    let Some(reg) = module.regions.get(reg_id) else {
        return false;
    };
    let Some(&blk_id) = reg.blocks.first() else {
        return false;
    };
    let Some(blk) = module.blocks.get(blk_id) else {
        return false;
    };

    for &inner_id in &blk.ops {
        let Some(inner_op) = module.ops.get(inner_id) else {
            continue;
        };
        if inner_op.name.full_name() == "map.store" {
            if inner_op.operands.is_empty() {
                return false;
            }
            if inner_op.operands.first().copied() == Some(target_tensor) {
                return false;
            }
            let store_indices = &inner_op.operands[1..];
            if !blk.arguments.is_empty() && store_indices != blk.arguments.as_slice() {
                return false;
            }
        }
    }
    true
}

/// Appends `extra_output_types` as additional destination-passing tensor outputs on an opaque
/// kernel (`map.iter.opaque`), inserting them before any trailing `!mo.chain` token result and
/// updating `num_outputs`.
///
/// Returns the newly created result [`ValueId`]s.
pub fn grow_opaque_outputs(
    module: &mut Module,
    opaque_op: OpId,
    extra_output_types: &[Type],
) -> Vec<ValueId> {
    let mut new_vals = Vec::with_capacity(extra_output_types.len());
    let Some(op_snapshot) = module.ops.get(opaque_op).cloned() else {
        return new_vals;
    };
    let has_trailing_chain = op_snapshot
        .results
        .last()
        .is_some_and(|&r| module.values.get(r).is_some_and(|v| is_chain_type(&v.ty)));
    let insert_pos = if has_trailing_chain {
        op_snapshot.results.len() - 1
    } else {
        op_snapshot.results.len()
    };

    for (idx, ty) in extra_output_types.iter().cloned().enumerate() {
        let res_index = (insert_pos + idx) as u32;
        let val_id = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty,
            def: ValueDef::OpResult {
                op: opaque_op,
                index: res_index,
            },
        });
        new_vals.push(val_id);
    }

    if let Some(op_mut) = module.ops.get_mut(opaque_op) {
        for (offset, &val_id) in new_vals.iter().enumerate() {
            op_mut.results.insert(insert_pos + offset, val_id);
        }
        let updated_results = op_mut.results.clone();
        let num_tensor_outputs = updated_results
            .iter()
            .filter(|&&r| module.values.get(r).is_some_and(|v| !is_chain_type(&v.ty)))
            .count();
        op_mut.attributes.insert(
            "num_outputs".to_string(),
            Attribute::Int(num_tensor_outputs as i64),
        );
        for (i, &r_id) in updated_results.iter().enumerate() {
            if let Some(v_mut) = module.values.get_mut(r_id) {
                v_mut.def = ValueDef::OpResult {
                    op: opaque_op,
                    index: i as u32,
                };
            }
        }
    }

    new_vals
}

/// Inverts a `map.index.reshape` operation in-place by swapping its `in_shape` and `out_shape`
/// attributes so it maps producer iteration coordinates to reshaped consumer store coordinates.
pub fn invert_reshape_on_epilogue_store(module: &mut Module, reshape_op: OpId) {
    let Some(op_mut) = module.ops.get_mut(reshape_op) else {
        return;
    };
    let in_shape = op_mut.attributes.shift_remove("in_shape");
    let out_shape = op_mut.attributes.shift_remove("out_shape");
    if let Some(in_s) = in_shape {
        op_mut.attributes.insert("out_shape".to_string(), in_s);
    }
    if let Some(out_s) = out_shape {
        op_mut.attributes.insert("in_shape".to_string(), out_s);
    }
}

/// Fuses a single eligible elementwise consumer (`consumer_op`) into the output/epilogue of
/// `producer_op` (`map.iter.opaque` or `map.foreach`), either replacing `producer_op`'s original
/// output in-place (when `keep_original_result == false`) or growing a new output slot via
/// [`grow_opaque_outputs`] (when `keep_original_result == true`).
fn fuse_output_once(
    module: &mut Module,
    enclosing_block: BlockId,
    producer_op: OpId,
    consumer_op: OpId,
    target_tensor: ValueId,
    keep_original_result: bool,
) -> bool {
    let untransformed = consumer_reads_untransformed(module, consumer_op, target_tensor);
    let reshape_id = if !untransformed {
        let Some(r_id) = consumer_reads_through_reshape(module, consumer_op, target_tensor) else {
            return false;
        };
        if consumer_is_pure_reshape(module, consumer_op, target_tensor) {
            return false;
        }
        Some(r_id)
    } else {
        None
    };

    if !is_fusible_mutable_store(module, consumer_op, target_tensor)
        || !check_mutable_kernel_fusible(module, producer_op, consumer_op)
    {
        return false;
    }

    let Some(prod_snapshot) = module.ops.get(producer_op).cloned() else {
        return false;
    };
    let Some(cons_snapshot) = module.ops.get(consumer_op).cloned() else {
        return false;
    };
    let Some(&cons_res) = cons_snapshot.results.first() else {
        return false;
    };
    let Some(cons_res_ty) = module.values.get(cons_res).map(|v| v.ty.clone()) else {
        return false;
    };
    let Some(&cons_reg_id) = cons_snapshot.regions.first() else {
        return false;
    };
    let Some(cons_reg) = module.regions.get(cons_reg_id) else {
        return false;
    };
    let Some(&cons_blk_id) = cons_reg.blocks.first() else {
        return false;
    };

    merge_redundant_map_loads(module, cons_reg_id);
    let Some(cons_blk) = module.blocks.get(cons_blk_id).cloned() else {
        return false;
    };

    let prod_is_opaque = prod_snapshot.name.full_name() == "map.iter.opaque";
    if keep_original_result && !prod_is_opaque {
        return false;
    }

    let Some(&prod_reg_id) = prod_snapshot.regions.last() else {
        return false;
    };
    let Some(prod_reg) = module.regions.get(prod_reg_id) else {
        return false;
    };
    let Some(&prod_blk_id) = prod_reg.blocks.first() else {
        return false;
    };
    let Some(prod_blk) = module.blocks.get(prod_blk_id).cloned() else {
        return false;
    };

    let Some(&term_op_id) = prod_blk.ops.last() else {
        return false;
    };
    let Some(term_op) = module.ops.get(term_op_id).cloned() else {
        return false;
    };
    let term_name = term_op.name.full_name();
    if !matches!(term_name.as_str(), "map.store" | "map.yield") || term_op.operands.is_empty() {
        return false;
    }
    let prod_computed_val = term_op.operands[0];
    let prod_store_indices: SmallVec<[ValueId; 4]> = if term_op.operands.len() > 1 {
        term_op.operands[1..].iter().copied().collect()
    } else {
        prod_blk.arguments.iter().copied().collect()
    };

    if reshape_id.is_none()
        && !cons_blk.arguments.is_empty()
        && cons_blk.arguments.len() != prod_store_indices.len()
    {
        return false;
    }

    let mut val_map: IndexMap<ValueId, ValueId> = IndexMap::new();
    if reshape_id.is_none() {
        for (&c_arg, &p_idx) in cons_blk.arguments.iter().zip(prod_store_indices.iter()) {
            val_map.insert(c_arg, p_idx);
        }
    }

    let mut cloned_ops: Vec<OpId> = Vec::new();
    let mut inverted_store_indices: Option<SmallVec<[ValueId; 4]>> = None;
    let mut cons_yielded_val: Option<ValueId> = None;

    for &inner_id in &cons_blk.ops {
        let Some(inner_op) = module.ops.get(inner_id).cloned() else {
            continue;
        };
        let full_name = inner_op.name.full_name();
        if full_name == "map.load" && inner_op.operands.first().copied() == Some(target_tensor) {
            if let Some(&load_res) = inner_op.results.first() {
                val_map.insert(load_res, prod_computed_val);
            }
            continue;
        }
        if full_name == "map.yield" || full_name == "map.store" {
            if let Some(&y_val) = inner_op.operands.first() {
                cons_yielded_val = Some(val_map.get(&y_val).copied().unwrap_or(y_val));
            }
            break;
        }

        if Some(inner_id) == reshape_id {
            // Invert `map.index.reshape`: it now takes `prod_store_indices` as inputs and produces
            // the reshaped coordinates matching `cons_blk.arguments`.
            let inv_res_types: Vec<Type> = cons_blk
                .arguments
                .iter()
                .filter_map(|&a| module.values.get(a).map(|v| v.ty.clone()))
                .collect();
            let (new_op_id, new_results) = create_detached_op(
                module,
                inner_op.name.clone(),
                prod_store_indices.clone(),
                &inv_res_types,
                inner_op.attributes.clone(),
                inner_op.location.clone(),
            );
            invert_reshape_on_epilogue_store(module, new_op_id);
            for (&c_arg, &inv_idx) in cons_blk.arguments.iter().zip(new_results.iter()) {
                val_map.insert(c_arg, inv_idx);
            }
            inverted_store_indices = Some(new_results.iter().copied().collect());
            cloned_ops.push(new_op_id);
            continue;
        }

        let new_operands: SmallVec<[ValueId; 4]> = inner_op
            .operands
            .iter()
            .map(|&v| val_map.get(&v).copied().unwrap_or(v))
            .collect();
        let res_types: Vec<Type> = inner_op
            .results
            .iter()
            .filter_map(|&r| module.values.get(r).map(|v| v.ty.clone()))
            .collect();

        let (new_op_id, new_results) = create_detached_op(
            module,
            inner_op.name.clone(),
            new_operands,
            &res_types,
            inner_op.attributes.clone(),
            inner_op.location.clone(),
        );
        for (&old_r, &new_r) in inner_op.results.iter().zip(new_results.iter()) {
            val_map.insert(old_r, new_r);
        }
        cloned_ops.push(new_op_id);
    }

    let Some(epilogue_val) = cons_yielded_val else {
        return false;
    };

    if let Some(p_blk_mut) = module.blocks.get_mut(prod_blk_id)
        && let Some(pos) = p_blk_mut.ops.iter().position(|&id| id == term_op_id)
    {
        for (offset, &cid) in cloned_ops.iter().enumerate() {
            p_blk_mut.ops.insert(pos + offset, cid);
        }
    }

    let final_store_indices = inverted_store_indices.unwrap_or(prod_store_indices);

    let out_val_to_replace_consumer = if keep_original_result {
        let grown = grow_opaque_outputs(module, producer_op, std::slice::from_ref(&cons_res_ty));
        let Some(&grown_val) = grown.first() else {
            return false;
        };
        let mut store_operands: SmallVec<[ValueId; 4]> = smallvec::smallvec![epilogue_val];
        store_operands.extend(final_store_indices);
        let mut attrs = IndexMap::new();
        attrs.insert(
            "output_index".to_string(),
            Attribute::Int(grown.len() as i64),
        );
        module.add_operation(
            prod_blk_id,
            OpName::new("map", "store"),
            store_operands,
            Vec::new(),
            attrs,
            Vec::new(),
            false,
            cons_snapshot.location.clone(),
        );
        grown_val
    } else {
        if let Some(term_mut) = module.ops.get_mut(term_op_id) {
            if term_name == "map.store" {
                let mut new_opnds: SmallVec<[ValueId; 4]> = smallvec::smallvec![epilogue_val];
                new_opnds.extend(final_store_indices);
                term_mut.operands = new_opnds;
            } else {
                term_mut.operands[0] = epilogue_val;
            }
        }
        if let Some(v_mut) = module.values.get_mut(target_tensor) {
            v_mut.ty = cons_res_ty;
        }
        target_tensor
    };

    let extra_cons_operands: SmallVec<[ValueId; 4]> = cons_snapshot
        .operands
        .iter()
        .copied()
        .filter(|&v| v != target_tensor)
        .collect();
    let prod_fused = fused_kernel_count(module, producer_op);
    let cons_fused = fused_kernel_count(module, consumer_op);
    let merged_locs = [prod_snapshot.location, cons_snapshot.location];

    if let Some(prod_mut) = module.ops.get_mut(producer_op) {
        for &ext in &extra_cons_operands {
            if !prod_mut.operands.contains(&ext) {
                prod_mut.operands.push(ext);
            }
        }
        prod_mut.attributes.insert(
            "fused_kernel_count".to_string(),
            Attribute::Int((prod_fused + cons_fused) as i64),
        );
        stamp_fused_provenance(prod_mut, &merged_locs);
    }

    module.replace_all_uses(cons_res, out_val_to_replace_consumer);
    module.remove_op(enclosing_block, consumer_op);
    true
}

/// Fuses elementwise consumers (`map.foreach` / `map.iter.elementwise`) into the output/epilogue
/// region of `map.iter.opaque` and `map.foreach` producers.
///
/// Supports both untransformed consumers (`consumer_reads_untransformed`) and consumers that read
/// the producer's output through a single `map.index.reshape` (`consumer_reads_through_reshape`),
/// inverting the reshape onto the epilogue store coordinates (`invert_reshape_on_epilogue_store`)
/// and expanding opaque kernel outputs (`grow_opaque_outputs`) when the intermediate result is
/// still live.
///
/// Returns the total number of output/epilogue fusions performed.
pub fn fuse_map_outputs(module: &mut Module) -> usize {
    let mut total_fused = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        while let Some(blk) = module.blocks.get(blk_id) {
            let ops = blk.ops.clone();
            let mut fused_in_iter = false;

            for &prod_id in &ops {
                let Some(prod_op) = module.ops.get(prod_id).cloned() else {
                    continue;
                };
                let prod_name = prod_op.name.full_name();
                if !matches!(
                    prod_name.as_str(),
                    "map.iter.opaque" | "map.foreach" | "map.iter.elementwise"
                ) {
                    continue;
                }
                let Some(&target_tensor) = prod_op
                    .results
                    .iter()
                    .find(|&&r| module.values.get(r).is_some_and(|v| !is_chain_type(&v.ty)))
                else {
                    continue;
                };

                let mut candidate_consumers: Vec<OpId> = Vec::new();
                let mut has_non_elementwise_user = false;
                for &cand_id in &ops {
                    if cand_id == prod_id {
                        continue;
                    }
                    let Some(cand_op) = module.ops.get(cand_id) else {
                        continue;
                    };
                    let uses_in_op_or_region = cand_op.operands.contains(&target_tensor)
                        || cand_op.regions.iter().any(|&r_id| {
                            module.regions.get(r_id).is_some_and(|r| {
                                r.blocks.iter().any(|&b_id| {
                                    module.blocks.get(b_id).is_some_and(|b| {
                                        b.ops.iter().any(|&inner_id| {
                                            module.ops.get(inner_id).is_some_and(|iop| {
                                                iop.operands.contains(&target_tensor)
                                            })
                                        })
                                    })
                                })
                            })
                        });
                    if uses_in_op_or_region {
                        if is_elementwise_map_kernel(module, cand_id) {
                            candidate_consumers.push(cand_id);
                        } else {
                            has_non_elementwise_user = true;
                        }
                    }
                }

                if candidate_consumers.is_empty() {
                    continue;
                }

                let total_users = candidate_consumers.len() + usize::from(has_non_elementwise_user);
                for &cons_id in &candidate_consumers {
                    let keep_original = total_users > 1;
                    if fuse_output_once(
                        module,
                        blk_id,
                        prod_id,
                        cons_id,
                        target_tensor,
                        keep_original,
                    ) {
                        total_fused += 1;
                        fused_in_iter = true;
                        break;
                    }
                }

                if fused_in_iter {
                    break;
                }
            }

            if !fused_in_iter {
                break;
            }
        }
    }

    total_fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, TensorType};

    fn make_tensor_ty(dims: &[i64]) -> Type {
        Type::Tensor(TensorType::static_shape(dims, DType::F32))
    }

    #[test]
    fn test_is_large_tensor_operand_and_cheap_to_duplicate() {
        let small_ty = make_tensor_ty(&[64, 64]); // 4096 -> not strictly > 4096
        let large_ty = make_tensor_ty(&[64, 65]); // 4160 -> > 4096
        let dyn_ty = Type::Tensor(TensorType::new(
            vec![Dim::Symbolic("N".to_string()), Dim::Static(16)],
            DType::F32,
        ));
        assert!(!is_large_tensor_operand(&small_ty));
        assert!(is_large_tensor_operand(&large_ty));
        assert!(is_large_tensor_operand(&dyn_ty));
    }

    #[test]
    fn test_fuse_map_inputs_merges_redundant_loads_and_duplicates_cheap_producer() {
        let mut module = Module::new(Some("map_input_fusion_test".to_string()));
        let body_blk = module.body_block();
        let x_in =
            module.add_block_argument(body_blk, Some("%x".to_string()), make_tensor_ty(&[128]));

        let prod_blk = module.create_block(None);
        let idx_arg =
            module.add_block_argument(prod_blk, Some("%i".to_string()), Type::Scalar(DType::I64));
        let load_id = module.add_operation(
            prod_blk,
            OpName::new("map", "load"),
            smallvec::smallvec![x_in, idx_arg],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let load_res = module.ops[load_id].results[0];
        let neg_id = module.add_operation(
            prod_blk,
            OpName::new("rmo", "neg"),
            smallvec::smallvec![load_res],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let neg_res = module.ops[neg_id].results[0];
        module.add_operation(
            prod_blk,
            OpName::new("map", "yield"),
            smallvec::smallvec![neg_res],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let prod_reg = module.create_region(vec![prod_blk]);
        let prod_op = module.add_operation(
            body_blk,
            OpName::new("map", "foreach"),
            smallvec::smallvec![x_in],
            vec![(None, make_tensor_ty(&[128]))],
            IndexMap::new(),
            vec![prod_reg],
            false,
            Location::Unknown,
        );
        let prod_res = module.ops[prod_op].results[0];

        let cons_blk = module.create_block(None);
        let j_arg =
            module.add_block_argument(cons_blk, Some("%j".to_string()), Type::Scalar(DType::I64));
        let c_load1 = module.add_operation(
            cons_blk,
            OpName::new("map", "load"),
            smallvec::smallvec![prod_res, j_arg],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let c_res1 = module.ops[c_load1].results[0];
        let c_load2 = module.add_operation(
            cons_blk,
            OpName::new("map", "load"),
            smallvec::smallvec![prod_res, j_arg],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let c_res2 = module.ops[c_load2].results[0];
        let add_id = module.add_operation(
            cons_blk,
            OpName::new("rmo", "add"),
            smallvec::smallvec![c_res1, c_res2],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let add_res = module.ops[add_id].results[0];
        module.add_operation(
            cons_blk,
            OpName::new("map", "yield"),
            smallvec::smallvec![add_res],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let cons_reg = module.create_region(vec![cons_blk]);
        let cons_op = module.add_operation(
            body_blk,
            OpName::new("map", "foreach"),
            smallvec::smallvec![prod_res],
            vec![(None, make_tensor_ty(&[128]))],
            IndexMap::new(),
            vec![cons_reg],
            false,
            Location::Unknown,
        );

        let fused = fuse_map_inputs(&mut module, Some(8));
        assert_eq!(fused, 1);
        assert_eq!(module.blocks[body_blk].ops, vec![cons_op]);
        assert_eq!(fused_kernel_count(&module, cons_op), 2);
        let neg_count = module.blocks[cons_blk]
            .ops
            .iter()
            .filter(|&&id| module.ops[id].name.full_name() == "rmo.neg")
            .count();
        assert_eq!(neg_count, 1);
    }

    #[test]
    fn test_fuse_map_outputs_with_inverse_reshape_and_grown_opaque_output() {
        let mut module = Module::new(Some("map_output_fusion_test".to_string()));
        let body_blk = module.body_block();
        let x_in = module.add_block_argument(body_blk, None, make_tensor_ty(&[4, 8]));

        let p_blk = module.create_block(None);
        let p_i0 = module.add_block_argument(p_blk, None, Type::Scalar(DType::I64));
        let p_i1 = module.add_block_argument(p_blk, None, Type::Scalar(DType::I64));
        let val_op = module.add_operation(
            p_blk,
            OpName::new("rmo", "constant"),
            SmallVec::new(),
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let val_res = module.ops[val_op].results[0];
        module.add_operation(
            p_blk,
            OpName::new("map", "store"),
            smallvec::smallvec![val_res, p_i0, p_i1],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let p_reg = module.create_region(vec![p_blk]);
        let prod_op = module.add_operation(
            body_blk,
            OpName::new("map", "iter.opaque"),
            smallvec::smallvec![x_in],
            vec![(None, make_tensor_ty(&[4, 8]))],
            IndexMap::new(),
            vec![p_reg],
            false,
            Location::Unknown,
        );
        let prod_res = module.ops[prod_op].results[0];

        let c_blk = module.create_block(None);
        let c_flat = module.add_block_argument(c_blk, None, Type::Scalar(DType::I64));
        let mut r_attrs = IndexMap::new();
        r_attrs.insert(
            "in_shape".to_string(),
            Attribute::Array(vec![Attribute::Int(32)]),
        );
        r_attrs.insert(
            "out_shape".to_string(),
            Attribute::Array(vec![Attribute::Int(4), Attribute::Int(8)]),
        );
        let r_id = module.add_operation(
            c_blk,
            OpName::new("map", "index.reshape"),
            smallvec::smallvec![c_flat],
            vec![
                (None, Type::Scalar(DType::I64)),
                (None, Type::Scalar(DType::I64)),
            ],
            r_attrs,
            Vec::new(),
            false,
            Location::Unknown,
        );
        let r0 = module.ops[r_id].results[0];
        let r1 = module.ops[r_id].results[1];
        let ld_id = module.add_operation(
            c_blk,
            OpName::new("map", "load"),
            smallvec::smallvec![prod_res, r0, r1],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let ld_res = module.ops[ld_id].results[0];
        let exp_id = module.add_operation(
            c_blk,
            OpName::new("rmo", "exp"),
            smallvec::smallvec![ld_res],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let exp_res = module.ops[exp_id].results[0];
        module.add_operation(
            c_blk,
            OpName::new("map", "yield"),
            smallvec::smallvec![exp_res],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let c_reg = module.create_region(vec![c_blk]);
        module.add_operation(
            body_blk,
            OpName::new("map", "foreach"),
            smallvec::smallvec![prod_res],
            vec![(None, make_tensor_ty(&[32]))],
            IndexMap::new(),
            vec![c_reg],
            false,
            Location::Unknown,
        );

        let fused = fuse_map_outputs(&mut module);
        assert_eq!(fused, 1);
        assert_eq!(module.blocks[body_blk].ops, vec![prod_op]);
        assert_eq!(module.values[prod_res].ty, make_tensor_ty(&[32]));
        let inverted_reshape = module.blocks[p_blk]
            .ops
            .iter()
            .copied()
            .find(|&id| module.ops[id].name.full_name() == "map.index.reshape")
            .expect("spliced map.index.reshape");
        assert_eq!(
            module.ops[inverted_reshape].attributes.get("in_shape"),
            Some(&Attribute::Array(vec![
                Attribute::Int(4),
                Attribute::Int(8)
            ]))
        );
        assert_eq!(
            module.ops[inverted_reshape].attributes.get("out_shape"),
            Some(&Attribute::Array(vec![Attribute::Int(32)]))
        );
    }
}
