//! Tenant quota management, accumulating buffer allocation, and tensor wire envelopes.

use super::format_expected_tensor_shape;
use crate::ir::{Attribute, BlockId, Module, OpId, Type, ValueId};

/// Snapshot entry for a single tenant's (`cache_salt`) byte quota.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantQuotaEntry {
    /// Tenant selector salt.
    pub cache_salt: String,
    /// Configured byte budget (`>= 0`).
    pub limit_bytes: i64,
}

/// Durable snapshot of a [`TenantQuotaManager`]'s per-salt limits and default limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantQuotaSnapshot {
    /// Explicit per-salt limits (`cache_salt -> limit_bytes`).
    pub limits: std::collections::BTreeMap<String, i64>,
    /// Default limit applied to unregistered salts under coordinator default-quota semantics.
    pub default_limit_bytes: Option<i64>,
}

/// Per-`cache_salt` byte budget registry supporting both allowlist eviction semantics
/// (`get_limit_bytes`) and coordinator boot-safe default-quota semantics
/// (`effective_limit_bytes`).
#[derive(Debug, Clone)]
pub struct TenantQuotaManager {
    section_name: String,
    limits: std::collections::BTreeMap<String, i64>,
    default_limit_bytes: Option<i64>,
}

impl TenantQuotaManager {
    /// Creates a new [`TenantQuotaManager`] with the given durable `section_name`.
    #[must_use]
    pub fn new(section_name: impl Into<String>) -> Self {
        Self {
            section_name: section_name.into(),
            limits: std::collections::BTreeMap::new(),
            default_limit_bytes: None,
        }
    }

    /// Returns the durable document section name for this quota registry.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.section_name
    }

    /// Creates or updates the byte quota for `cache_salt`, rejecting negative values.
    pub fn set_quota(
        &mut self,
        cache_salt: impl Into<String>,
        limit_bytes: i64,
    ) -> Result<(), String> {
        if limit_bytes < 0 {
            return Err(format!(
                "limit_bytes must be non-negative (got {limit_bytes})"
            ));
        }
        self.limits.insert(cache_salt.into(), limit_bytes);
        Ok(())
    }

    /// Removes the explicit quota registration for `cache_salt`, returning `true` if present.
    pub fn delete_quota(&mut self, cache_salt: &str) -> bool {
        self.limits.remove(cache_salt).is_some()
    }

    /// Returns the effective byte limit under allowlist semantics (unregistered salts resolve
    /// to `0` bytes so their objects are evicted on the next cycle).
    #[must_use]
    pub fn get_limit_bytes(&self, cache_salt: &str) -> i64 {
        self.limits.get(cache_salt).copied().unwrap_or(0)
    }

    /// Configures the default byte limit applied to unregistered salts by
    /// [`Self::effective_limit_bytes`] (`None` exempts unregistered salts from eviction;
    /// `Some(0)` arms strict allowlist enforcement).
    pub fn set_default_limit_bytes(&mut self, limit_bytes: Option<i64>) -> Result<(), String> {
        if let Some(v) = limit_bytes
            && v < 0
        {
            return Err(format!("limit_bytes must be non-negative (got {v})"));
        }
        self.default_limit_bytes = limit_bytes;
        Ok(())
    }

    /// Returns the configured default limit for unregistered salts (`None` = exempt).
    #[must_use]
    pub const fn get_default_limit_bytes(&self) -> Option<i64> {
        self.default_limit_bytes
    }

    /// Returns the byte limit for `cache_salt` under default-quota semantics: explicit limit if
    /// registered, otherwise [`Self::get_default_limit_bytes`].
    #[must_use]
    pub fn effective_limit_bytes(&self, cache_salt: &str) -> Option<i64> {
        self.limits
            .get(cache_salt)
            .copied()
            .or(self.default_limit_bytes)
    }

    /// Returns `true` if `cache_salt` has an explicit quota entry (distinguishing an explicit
    /// `0` limit from an unregistered salt).
    #[must_use]
    pub fn has_quota(&self, cache_salt: &str) -> bool {
        self.limits.contains_key(cache_salt)
    }

    /// Captures a detached durable snapshot of all per-salt limits and the default limit.
    #[must_use]
    pub fn capture(&self) -> TenantQuotaSnapshot {
        TenantQuotaSnapshot {
            limits: self.limits.clone(),
            default_limit_bytes: self.default_limit_bytes,
        }
    }

    /// Atomically replaces the quota table and default limit from `snapshot`.
    pub fn restore(&mut self, snapshot: TenantQuotaSnapshot) {
        self.limits = snapshot.limits;
        self.default_limit_bytes = snapshot.default_limit_bytes;
    }

    /// Returns a detached list of all explicitly registered tenant quotas in sorted key order.
    #[must_use]
    pub fn list_quotas(&self) -> Vec<TenantQuotaEntry> {
        self.limits
            .iter()
            .map(|(salt, &limit)| TenantQuotaEntry {
                cache_salt: salt.clone(),
                limit_bytes: limit,
            })
            .collect()
    }
}

/// Validates that a multi-output serde tuple (`role` in `"src"`, `"dst"`, `"layout"`) has exact
/// arity `expected`.
pub fn validate_multi_serde_group_size<T>(
    group: &[T],
    expected: usize,
    role: &str,
) -> Result<(), String> {
    if group.len() != expected {
        return Err(format!(
            "MultiSerde {role} group length {} does not match expected {expected}",
            group.len()
        ));
    }
    Ok(())
}

/// Checks whether a `map.iter.opaque` kernel contains an accumulating `map.call_opaque`
/// operation (`concat`, `matmul`, `batch_matmul`, `conv`, or `conv_transpose`).
pub(super) fn kernel_contains_accumulating_call(module: &Module, kernel_id: OpId) -> bool {
    let regions = module.ops[kernel_id].regions.clone();
    for rid in regions {
        for &bid in &module.regions[rid].blocks {
            for &op_id in &module.blocks[bid].ops {
                let op = &module.ops[op_id];
                if op.name.full_name() == "map.call_opaque"
                    && let Some(Attribute::String(kname)) = op
                        .attributes
                        .get("kernel_name")
                        .or_else(|| op.attributes.get("callee"))
                    && (kname.contains("mo.concat")
                        || kname.contains("mo.matmul")
                        || kname.contains("mo.batch_matmul")
                        || kname.contains("mo.conv")
                        || kname.contains("mo.conv_transpose"))
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Resolves the outer store-target tensor for a `map.iter.epilogue` op by scanning its body's
/// `map.store` ops and mapping captured block arguments back to the epilogue's capture operands.
/// Prefers a captured tensor whose element type matches the epilogue's input value element type.
pub(super) fn find_epilogue_store_target(
    module: &Module,
    epi_id: OpId,
) -> Option<(ValueId, crate::ir::mlir::DType)> {
    use crate::ir::mlir::{DType, ValueDef};

    let epi_op = &module.ops[epi_id];
    let rid = *epi_op.regions.first()?;
    let bid = *module.regions[rid].blocks.first()?;
    let block = &module.blocks[bid];
    if block.arguments.len() < 2 {
        return None;
    }
    let val_arg = block.arguments[0];
    let val_dtype = match &module.values[val_arg].ty {
        Type::Scalar(dt) => dt.clone(),
        Type::Tensor(tt) => tt.dtype.clone(),
        _ => DType::F32,
    };

    let mut fallback: Option<(ValueId, DType)> = None;
    for &op_id in &block.ops {
        let op = &module.ops[op_id];
        if op.name.full_name() == "map.store" && op.operands.len() >= 2 {
            // In `map.store %val, %tensor, %idx`, operand 1 (or operand 0 if 2-operand) is the
            // target tensor block argument captured by the epilogue closure.
            let tensor_val = if op.operands.len() >= 3 {
                op.operands[1]
            } else {
                op.operands[0]
            };
            if let ValueDef::BlockArgument {
                block: arg_blk,
                index: arg_idx,
            } = module.values[tensor_val].def
                && arg_blk == bid
                && (arg_idx as usize) >= 2
            {
                let capture_idx = (arg_idx as usize) - 2;
                if let Some(&outer_capture) = epi_op.operands.get(capture_idx) {
                    let outer_ty = module.values[outer_capture].ty.clone();
                    if let Type::Tensor(ref tt) = outer_ty
                        && tt.dtype == val_dtype
                    {
                        return Some((outer_capture, val_dtype));
                    }
                    if fallback.is_none() {
                        fallback = Some((outer_capture, val_dtype.clone()));
                    }
                }
            }
        }
    }
    fallback
}

/// Assigns a concrete `$intermediateTensor` store-target operand (and `"has_intermediate_tensor"`
/// attribute) to every fused `map.iter.epilogue` inside `map.iter.opaque` kernels.
///
/// When an accumulating kernel (`matmul`, `batch_matmul`, `conv`, `conv_transpose`, `concat`)
/// has a dtype-changing epilogue (the epilogue body's input element type differs from the store
/// target tensor's element type), allocates a fresh pre-cast accumulation scratch result on the
/// enclosing `map.iter.opaque` (inserted before any trailing `!mo.chain` result) plus a matching
/// trailing block argument on the kernel body, and binds the epilogue's intermediate tensor to
/// that scratch argument.
///
/// Returns `(assigned_epilogues, allocated_scratch_buffers)`.
pub fn allocate_accumulating_buffers(module: &mut Module) -> (usize, usize) {
    use crate::ir::mlir::{RegionId, Value, ValueDef};

    let mut assigned_epilogues = 0usize;
    let mut allocated_scratch = 0usize;

    let region_ids: Vec<RegionId> = module.regions.iter().map(|(r, _)| r).collect();
    for rid in region_ids {
        let block_ids = module.regions[rid].blocks.clone();
        for bid in block_ids {
            let op_ids = module.blocks[bid].ops.clone();
            for kernel_id in op_ids {
                if module.ops[kernel_id].name.full_name() != "map.iter.opaque" {
                    continue;
                }
                let is_accumulating = kernel_contains_accumulating_call(module, kernel_id);
                let Some(&k_rid) = module.ops[kernel_id].regions.first() else {
                    continue;
                };
                let Some(&k_bid) = module.regions[k_rid].blocks.first() else {
                    continue;
                };
                let k_ops = module.blocks[k_bid].ops.clone();
                for epi_id in k_ops {
                    if module.ops[epi_id].name.full_name() != "map.iter.epilogue" {
                        continue;
                    }
                    if module.ops[epi_id]
                        .attributes
                        .contains_key("has_intermediate_tensor")
                    {
                        continue;
                    }
                    let Some((store_target, in_dtype)) = find_epilogue_store_target(module, epi_id)
                    else {
                        continue;
                    };
                    let target_ty = module.values[store_target].ty.clone();
                    let dtype_changes = match &target_ty {
                        Type::Tensor(tt) => tt.dtype != in_dtype,
                        _ => false,
                    };

                    let intermediate_val = if dtype_changes && is_accumulating {
                        let Type::Tensor(ref tt) = target_ty else {
                            continue;
                        };
                        let mut scratch_tt = tt.clone();
                        scratch_tt.dtype = in_dtype;
                        let scratch_ty = Type::Tensor(scratch_tt);

                        // Add a new non-chain result on `kernel_id` before any trailing chain.
                        let existing_results = module.ops[kernel_id].results.clone();
                        let has_trailing_chain = existing_results.last().is_some_and(|&v| {
                            matches!(&module.values[v].ty, Type::Opaque(s) if s == "!mo.chain")
                        });
                        let insert_pos = if has_trailing_chain {
                            existing_results.len() - 1
                        } else {
                            existing_results.len()
                        };
                        let scratch_res = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: scratch_ty.clone(),
                            def: ValueDef::OpResult {
                                op: kernel_id,
                                index: insert_pos as u32,
                            },
                        });
                        module.values[scratch_res].id = scratch_res;
                        module.ops[kernel_id]
                            .results
                            .insert(insert_pos, scratch_res);
                        if has_trailing_chain
                            && let Some(&chain_v) = module.ops[kernel_id].results.last()
                        {
                            module.values[chain_v].def = ValueDef::OpResult {
                                op: kernel_id,
                                index: (module.ops[kernel_id].results.len() - 1) as u32,
                            };
                        }

                        // Append matching output block argument to the kernel body block.
                        let arg_idx = module.blocks[k_bid].arguments.len() as u32;
                        let scratch_arg = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: scratch_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: k_bid,
                                index: arg_idx,
                            },
                        });
                        module.values[scratch_arg].id = scratch_arg;
                        module.blocks[k_bid].arguments.push(scratch_arg);
                        allocated_scratch += 1;
                        scratch_arg
                    } else {
                        store_target
                    };

                    module.ops[epi_id].operands.push(intermediate_val);
                    module.ops[epi_id]
                        .attributes
                        .insert("has_intermediate_tensor".to_string(), Attribute::Bool(true));
                    assigned_epilogues += 1;
                }
            }
        }
    }

    (assigned_epilogues, allocated_scratch)
}

pub(super) fn is_closure_index_transform_op(full_name: &str) -> bool {
    full_name.starts_with("map.index.") && !full_name.ends_with(".stride")
}

pub(super) fn is_capturing_fusion_closure(full_name: &str) -> bool {
    matches!(
        full_name,
        "map.iter.elementwise.body" | "map.iter.prologue" | "map.iter.epilogue"
    )
}

/// Walks backward from `idx_val` through a chain of `map.index.*` transform ops inside `block_id`,
/// returning the chain closest-to-load first.
pub(super) fn collect_closure_index_transform_chain(
    module: &Module,
    idx_val: ValueId,
) -> Vec<OpId> {
    use crate::ir::mlir::ValueDef;

    let mut chain = Vec::new();
    let mut current = idx_val;
    while let Some(ValueDef::OpResult { op, .. }) = module.values.get(current).map(|v| v.def) {
        let Some(def_op) = module.ops.get(op) else {
            break;
        };
        if !is_closure_index_transform_op(&def_op.name.full_name()) || def_op.operands.is_empty() {
            break;
        }
        chain.push(op);
        current = def_op.operands[0];
    }
    chain
}

/// Checks whether `transform_op`'s index result transitively reaches the index operand of a
/// `map.store` within `closure_block` through index-space transforms.
pub(super) fn closure_transform_feeds_store(
    module: &Module,
    closure_block: BlockId,
    transform_op: OpId,
) -> bool {
    let Some(&start_res) = module.ops[transform_op].results.first() else {
        return false;
    };
    let mut worklist = vec![start_res];
    let mut visited = std::collections::HashSet::new();
    visited.insert(start_res);

    while let Some(val) = worklist.pop() {
        for &op_id in &module.blocks[closure_block].ops {
            let op = &module.ops[op_id];
            let full = op.name.full_name();
            if full == "map.store" {
                // In `map.store(%val, %tensor, %idx)`, the last operand is the index:
                if op.operands.last().copied() == Some(val) {
                    return true;
                }
            } else if is_closure_index_transform_op(&full)
                && op.operands.first().copied() == Some(val)
                && let Some(&res) = op.results.first()
                && visited.insert(res)
            {
                worklist.push(res);
            }
        }
    }
    false
}

/// Hoists load-side `map.index.*` transform chains (`map.index.slice`, `map.index.broadcast`,
/// `map.index.reshape`, `map.index.transpose`) that do not transitively feed a `map.store` index
/// out of capturing fusion closures (`map.iter.elementwise.body`, `map.iter.prologue`,
/// `map.iter.epilogue`) into `<op>.stride` view operations placed immediately before the closure.
///
/// Also rewires `map.call_elementwise` index operands through hoistable index-transform prefixes
/// and erases dead index-transform ops from the closure body. Returns
/// `(hoisted_stride_ops, erased_index_transforms)`.
pub fn loop_invariant_view_motion(module: &mut Module) -> (usize, usize) {
    use crate::ir::mlir::{Location, OpName, Operation, RegionId, Value, ValueDef};

    let mut hoisted_stride_ops = 0usize;
    let mut erased_transforms = 0usize;

    let region_ids: Vec<RegionId> = module.regions.iter().map(|(r, _)| r).collect();
    for rid in region_ids {
        let block_ids = module.regions[rid].blocks.clone();
        for outer_bid in block_ids {
            let outer_ops = module.blocks[outer_bid].ops.clone();
            for closure_id in outer_ops {
                let Some(closure_op) = module.ops.get(closure_id).cloned() else {
                    continue;
                };
                let closure_name = closure_op.name.full_name();
                if !is_capturing_fusion_closure(&closure_name) {
                    continue;
                }
                let Some(&c_rid) = closure_op.regions.first() else {
                    continue;
                };
                let Some(&c_bid) = module.regions[c_rid].blocks.first() else {
                    continue;
                };

                // Map capture block arguments to outer capture operands.
                // `map.iter.epilogue` has 2 leading non-capture args `(%val, %idx)`;
                // `map.iter.elementwise.body` and `map.iter.prologue` have 1 leading `%idx` arg.
                let num_captures = closure_op.operands.len();
                let c_args = module.blocks[c_bid].arguments.clone();
                if c_args.len() < num_captures {
                    continue;
                }
                let capture_arg_offset = c_args.len() - num_captures;
                let mut outer_map = std::collections::HashMap::new();
                for (i, &outer_val) in closure_op.operands.iter().enumerate() {
                    outer_map.insert(c_args[capture_arg_offset + i], outer_val);
                }

                let inner_ops = module.blocks[c_bid].ops.clone();
                let mut inserted_before_closure = Vec::new();

                // 1. Hoist load-side index-transform prefixes into `.stride` ops before `closure`:
                for &inner_id in &inner_ops {
                    let Some(load_op) = module.ops.get(inner_id).cloned() else {
                        continue;
                    };
                    if load_op.name.full_name() != "map.load" || load_op.operands.len() < 2 {
                        continue;
                    }
                    let load_tensor_arg = load_op.operands[0];
                    let load_idx_val = load_op.operands[1];
                    let chain = collect_closure_index_transform_chain(module, load_idx_val);
                    let mut prefix_len = 0usize;
                    while prefix_len < chain.len()
                        && !closure_transform_feeds_store(module, c_bid, chain[prefix_len])
                    {
                        prefix_len += 1;
                    }
                    if prefix_len == 0 {
                        continue;
                    }
                    let hoist_slice = &chain[..prefix_len];
                    let outermost_transform = hoist_slice[hoist_slice.len() - 1];
                    let outermost_idx = module.ops[outermost_transform].operands[0];

                    let mut current_outer_tensor = outer_map
                        .get(&load_tensor_arg)
                        .copied()
                        .unwrap_or(load_tensor_arg);
                    let tensor_ty = module.values[current_outer_tensor].ty.clone();

                    for &t_id in hoist_slice {
                        let t_op = module.ops[t_id].clone();
                        let stride_op_name = format!("{}.stride", t_op.name.name);
                        let mut stride_operands = smallvec::smallvec![current_outer_tensor];
                        for &extra_opnd in t_op.operands.iter().skip(1) {
                            let mapped = outer_map.get(&extra_opnd).copied().unwrap_or(extra_opnd);
                            stride_operands.push(mapped);
                        }
                        let stride_res = module.values.insert(Value {
                            id: ValueId::default(),
                            ssa_name: None,
                            ty: tensor_ty.clone(),
                            def: ValueDef::BlockArgument {
                                block: outer_bid,
                                index: 0,
                            },
                        });
                        module.values[stride_res].id = stride_res;
                        let stride_op = module.ops.insert(Operation {
                            id: OpId::default(),
                            name: OpName::new(&t_op.name.dialect, &stride_op_name),
                            operands: stride_operands,
                            results: smallvec::smallvec![stride_res],
                            attributes: t_op.attributes.clone(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: Location::Unknown,
                        });
                        module.ops[stride_op].id = stride_op;
                        module.values[stride_res].def = ValueDef::OpResult {
                            op: stride_op,
                            index: 0,
                        };
                        inserted_before_closure.push(stride_op);
                        current_outer_tensor = stride_res;
                        hoisted_stride_ops += 1;
                    }

                    // Add new capture operand on `closure_id` and matching block arg on `c_bid`:
                    module.ops[closure_id].operands.push(current_outer_tensor);
                    let new_cap_arg = module.add_block_argument(c_bid, None, tensor_ty.clone());
                    outer_map.insert(new_cap_arg, current_outer_tensor);

                    module.ops[inner_id].operands[0] = new_cap_arg;
                    module.ops[inner_id].operands[1] = outermost_idx;
                }

                if !inserted_before_closure.is_empty()
                    && let Some(pos) = module.blocks[outer_bid]
                        .ops
                        .iter()
                        .position(|&id| id == closure_id)
                {
                    for (off, sid) in inserted_before_closure.into_iter().enumerate() {
                        module.blocks[outer_bid].ops.insert(pos + off, sid);
                    }
                }

                // 2. Rewire `map.call_elementwise` index operands through hoistable prefixes:
                for &inner_id in &inner_ops {
                    let Some(call_op) = module.ops.get(inner_id).cloned() else {
                        continue;
                    };
                    if call_op.name.full_name() != "map.call_elementwise"
                        || call_op.operands.is_empty()
                    {
                        continue;
                    }
                    let idx_pos = call_op.operands.len() - 1;
                    let chain =
                        collect_closure_index_transform_chain(module, call_op.operands[idx_pos]);
                    let mut prefix_len = 0usize;
                    while prefix_len < chain.len()
                        && !closure_transform_feeds_store(module, c_bid, chain[prefix_len])
                    {
                        prefix_len += 1;
                    }
                    if prefix_len > 0 {
                        let outermost = chain[prefix_len - 1];
                        let new_idx = module.ops[outermost].operands[0];
                        module.ops[inner_id].operands[idx_pos] = new_idx;
                    }
                }

                // 3. Erase dead `map.index.*` transforms inside `c_bid` in reverse order:
                let rev_ops: Vec<OpId> = module.blocks[c_bid].ops.iter().copied().rev().collect();
                for tid in rev_ops {
                    let Some(top) = module.ops.get(tid).cloned() else {
                        continue;
                    };
                    if !is_closure_index_transform_op(&top.name.full_name()) {
                        continue;
                    }
                    let used = top.results.iter().any(|&r| {
                        module.blocks[c_bid]
                            .ops
                            .iter()
                            .any(|&oid| oid != tid && module.ops[oid].operands.contains(&r))
                    });
                    if !used {
                        module.remove_op(c_bid, tid);
                        erased_transforms += 1;
                    }
                }
            }
        }
    }

    (hoisted_stride_ops, erased_transforms)
}

/// Stages device-graph execution inside functions/models annotated with
/// `is_device_graph = true`:
/// 1. Rewrites each `mgp.guard` into a forwarding `mogg.routine.async` region whose body drops
///    the leading ordering chain argument and yields the remaining forwarded arguments via
///    `mogg.output`.
/// 2. Collects all `mogg.routine.async` and `mgp.device_graph.stable.create` operations in block
///    order and moves them into a single isolated `mgp.device_graph.staging` region just before
///    the terminator, capturing free outside operands as staging inputs and yielding values with
///    escaping uses via `mgp.device_graph.staging.yield`.
///
/// Returns `(lowered_guards, staged_ops_count)`.
pub fn stage_device_graphs(module: &mut Module) -> (usize, usize) {
    use crate::ir::mlir::{Location, OpName, Operation, Value, ValueDef};

    let mut lowered_guards = 0usize;
    let mut staged_ops_count = 0usize;

    let fn_ids = module.functions();
    for fn_id in fn_ids {
        let is_device_graph = matches!(
            module.ops[fn_id].attributes.get("is_device_graph"),
            Some(Attribute::Bool(true))
        );
        if !is_device_graph {
            continue;
        }
        let Some(&fn_reg) = module.ops[fn_id].regions.first() else {
            continue;
        };
        let Some(&exec_blk) = module.regions[fn_reg].blocks.first() else {
            continue;
        };

        // Step 1: Lower `mgp.guard` ops in `exec_blk` into forwarding `mogg.routine.async` ops:
        let initial_ops = module.blocks[exec_blk].ops.clone();
        for op_id in initial_ops {
            let Some(guard_op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if guard_op.name.full_name() != "mgp.guard" {
                continue;
            }
            let body_blk = module.create_block(None);
            let mut forwarded_args = smallvec::SmallVec::<[ValueId; 4]>::new();
            for (i, &opnd) in guard_op.operands.iter().enumerate() {
                let opnd_ty = module.values[opnd].ty.clone();
                let arg_v = module.add_block_argument(body_blk, None, opnd_ty);
                if i > 0 {
                    forwarded_args.push(arg_v);
                }
            }
            module.add_operation(
                body_blk,
                OpName::new("mogg", "output"),
                forwarded_args,
                vec![],
                indexmap::IndexMap::new(),
                vec![],
                false,
                Location::Unknown,
            );
            let body_reg = module.create_region(vec![body_blk]);

            let mut attrs = guard_op.attributes.clone();
            attrs.insert(
                "dispatch".to_string(),
                Attribute::String("Immediate".to_string()),
            );
            attrs.insert("can_fuse_into_other".to_string(), Attribute::Bool(false));
            attrs.insert("can_be_fused_into".to_string(), Attribute::Bool(false));
            attrs.insert("can_duplicate".to_string(), Attribute::Bool(false));

            module.ops[op_id].name = OpName::new("mogg", "routine.async");
            module.ops[op_id].attributes = attrs;
            module.ops[op_id].regions = vec![body_reg];
            lowered_guards += 1;
        }

        // Step 2: Collect all `mogg.routine.async` and `mgp.device_graph.stable.create` ops:
        let current_ops = module.blocks[exec_blk].ops.clone();
        let mut staged_ops = Vec::new();
        let mut staged_set = std::collections::HashSet::new();
        for &op_id in &current_ops {
            let full = module.ops[op_id].name.full_name();
            if full == "mogg.routine.async" || full == "mgp.device_graph.stable.create" {
                staged_ops.push(op_id);
                staged_set.insert(op_id);
            }
        }
        if staged_ops.is_empty() {
            continue;
        }
        staged_ops_count += staged_ops.len();

        // Collect deduplicated free inputs defined outside `staged_set`:
        let mut free_inputs = Vec::new();
        let mut seen_inputs = std::collections::HashSet::new();
        for &sop_id in &staged_ops {
            for &opnd in &module.ops[sop_id].operands {
                let def_inside = match module.values[opnd].def {
                    ValueDef::OpResult { op, .. } => staged_set.contains(&op),
                    ValueDef::BlockArgument { .. } => false,
                };
                if !def_inside && seen_inputs.insert(opnd) {
                    free_inputs.push(opnd);
                }
            }
        }

        // Collect escaping uses (`(user_op, operand_idx, produced_val)`) outside `staged_set`:
        let mut escaping_uses = Vec::new();
        for &sop_id in &staged_ops {
            let results = module.ops[sop_id].results.clone();
            for res_v in results {
                for &other_id in &current_ops {
                    if staged_set.contains(&other_id) {
                        continue;
                    }
                    for (opnd_idx, &u_val) in module.ops[other_id].operands.iter().enumerate() {
                        if u_val == res_v {
                            escaping_uses.push((other_id, opnd_idx, res_v));
                        }
                    }
                }
            }
        }

        // Create staging region & block:
        let staging_blk = module.create_block(None);
        let mut free_map = std::collections::HashMap::new();
        for &inp in &free_inputs {
            let ty = module.values[inp].ty.clone();
            let barg = module.add_block_argument(staging_blk, None, ty);
            free_map.insert(inp, barg);
        }

        // Move `staged_ops` from `exec_blk` into `staging_blk` and remap free operands:
        module.blocks[exec_blk]
            .ops
            .retain(|id| !staged_set.contains(id));
        for &sop_id in &staged_ops {
            for opnd in &mut module.ops[sop_id].operands {
                if let Some(&mapped) = free_map.get(opnd) {
                    *opnd = mapped;
                }
            }
            module.blocks[staging_blk].ops.push(sop_id);
        }

        let yield_operands: smallvec::SmallVec<[ValueId; 4]> =
            escaping_uses.iter().map(|&(_, _, v)| v).collect();
        module.add_operation(
            staging_blk,
            OpName::new("mgp", "device_graph.staging.yield"),
            yield_operands,
            vec![],
            indexmap::IndexMap::new(),
            vec![],
            false,
            Location::Unknown,
        );
        let staging_reg = module.create_region(vec![staging_blk]);

        let mut staging_results = smallvec::SmallVec::<[ValueId; 2]>::new();
        for (idx, &(_, _, prod_v)) in escaping_uses.iter().enumerate() {
            let ty = module.values[prod_v].ty.clone();
            let s_res = module.values.insert(Value {
                id: ValueId::default(),
                ssa_name: None,
                ty,
                def: ValueDef::BlockArgument {
                    block: exec_blk,
                    index: idx as u32,
                },
            });
            module.values[s_res].id = s_res;
            staging_results.push(s_res);
        }

        let staging_op = module.ops.insert(Operation {
            id: OpId::default(),
            name: OpName::new("mgp", "device_graph.staging"),
            operands: free_inputs.into_iter().collect(),
            results: staging_results.clone(),
            attributes: indexmap::IndexMap::new(),
            regions: vec![staging_reg],
            custom_syntax: false,
            location: Location::Unknown,
        });
        module.ops[staging_op].id = staging_op;
        for (idx, &s_res) in staging_results.iter().enumerate() {
            module.values[s_res].def = ValueDef::OpResult {
                op: staging_op,
                index: idx as u32,
            };
        }

        // Insert `staging_op` right before the terminator in `exec_blk`:
        let insert_idx = module.blocks[exec_blk].ops.len().saturating_sub(1);
        module.blocks[exec_blk].ops.insert(insert_idx, staging_op);

        // Rewire escaping uses to the corresponding `staging_op` results:
        for ((user_op, opnd_idx, _), new_res) in escaping_uses.into_iter().zip(staging_results) {
            module.ops[user_op].operands[opnd_idx] = new_res;
        }
    }

    (lowered_guards, staged_ops_count)
}

/// Raises lowered `mgp.buffer_to_index(mgp.resource.constant {resource_slot = i})` sequences back
/// into `index.constant {value = c}` inside single-caller MGP-ABI kernel stubs (`abi = "mgp"`),
/// looking up the 64-bit integer slot `i` from the sole `mogg.compiled_call` caller's
/// `constant_resources` attribute.
///
/// Returns the number of `mgp.buffer_to_index` operations raised back to `index.constant`.
pub fn raise_back_index_constants(module: &mut Module) -> usize {
    use crate::ir::mlir::{DType, Location, OpName, Operation, Value, ValueDef};

    // 1. Collect all `mogg.compiled_call` callers per callee symbol:
    let mut symbol_callers: std::collections::HashMap<String, Vec<OpId>> =
        std::collections::HashMap::new();
    for (op_id, op) in &module.ops {
        if op.name.full_name() == "mogg.compiled_call" {
            let callee = op
                .attributes
                .get("callee")
                .or_else(|| op.attributes.get("callee_symbol"))
                .and_then(|a| match a {
                    Attribute::SymbolRef(s) | Attribute::String(s) => {
                        Some(s.trim_start_matches('@').to_string())
                    }
                    _ => None,
                });
            if let Some(sym) = callee {
                symbol_callers.entry(sym).or_default().push(op_id);
            }
        }
    }

    let mut raised = 0usize;
    let fn_ids = module.functions();
    for fn_id in fn_ids {
        let is_mgp_abi = matches!(
            module.ops[fn_id].attributes.get("abi"),
            Some(Attribute::String(s)) if s.eq_ignore_ascii_case("mgp")
        );
        if !is_mgp_abi {
            continue;
        }
        let Some(sym_name) = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(Attribute::as_str)
            .map(ToOwned::to_owned)
        else {
            continue;
        };
        let Some(callers) = symbol_callers.get(&sym_name) else {
            continue;
        };
        // Only single-caller kernel stubs can safely inline their caller's constant resource:
        if callers.len() != 1 {
            continue;
        }
        let caller_id = callers[0];
        let resource_values: Vec<Option<i64>> =
            match module.ops[caller_id].attributes.get("constant_resources") {
                Some(Attribute::DenseIntArray(vals)) => vals.iter().copied().map(Some).collect(),
                Some(Attribute::Array(items)) => items
                    .iter()
                    .map(|item| match item {
                        Attribute::Int(v) => Some(*v),
                        _ => None,
                    })
                    .collect(),
                _ => continue,
            };

        let regions = module.ops[fn_id].regions.clone();
        for rid in regions {
            let blocks = module.regions[rid].blocks.clone();
            for bid in blocks {
                let ops_snapshot = module.blocks[bid].ops.clone();
                for op_id in ops_snapshot {
                    let Some(b2i_op) = module.ops.get(op_id).cloned() else {
                        continue;
                    };
                    if b2i_op.name.full_name() != "mgp.buffer_to_index"
                        || b2i_op.operands.is_empty()
                        || b2i_op.results.is_empty()
                    {
                        continue;
                    }
                    let buf_val = b2i_op.operands[0];
                    let Some(ValueDef::OpResult { op: res_op_id, .. }) =
                        module.values.get(buf_val).map(|v| v.def)
                    else {
                        continue;
                    };
                    let Some(res_op) = module.ops.get(res_op_id).cloned() else {
                        continue;
                    };
                    if res_op.name.full_name() != "mgp.resource.constant" {
                        continue;
                    }
                    let Some(slot_idx) =
                        res_op
                            .attributes
                            .get("resource_slot")
                            .and_then(|a| match a {
                                Attribute::Int(v) => Some(*v),
                                Attribute::Opaque(raw) => raw
                                    .split(':')
                                    .next()
                                    .and_then(|s| s.trim().parse::<i64>().ok()),
                                _ => None,
                            })
                    else {
                        continue;
                    };
                    if slot_idx < 0 {
                        continue;
                    }
                    let Some(Some(const_val)) = resource_values.get(slot_idx as usize).copied()
                    else {
                        continue;
                    };

                    // Create `index.constant {value = const_val}` and replace uses of `b2i_op`:
                    let old_res = b2i_op.results[0];
                    let idx_res = module.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty: Type::Scalar(DType::Index),
                        def: ValueDef::BlockArgument {
                            block: bid,
                            index: 0,
                        },
                    });
                    module.values[idx_res].id = idx_res;
                    let mut attrs = indexmap::IndexMap::new();
                    attrs.insert("value".to_string(), Attribute::Int(const_val));
                    let idx_const_op = module.ops.insert(Operation {
                        id: OpId::default(),
                        name: OpName::new("index", "constant"),
                        operands: smallvec::SmallVec::new(),
                        results: smallvec::smallvec![idx_res],
                        attributes: attrs,
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: Location::Unknown,
                    });
                    module.ops[idx_const_op].id = idx_const_op;
                    module.values[idx_res].def = ValueDef::OpResult {
                        op: idx_const_op,
                        index: 0,
                    };

                    if let Some(pos) = module.blocks[bid].ops.iter().position(|&id| id == op_id) {
                        module.blocks[bid].ops[pos] = idx_const_op;
                    }
                    module.replace_all_uses(old_res, idx_res);
                    module.ops.remove(op_id);
                    raised += 1;
                }
            }
        }
    }

    raised
}

/// Inserts runtime shape-verification assertions (`mgp.tensor.extract_tensor_spec`,
/// `mgp.tensor_spec.get_dim`, `index.constant`, `index.cmp`, `pop.and`, `mgp.assert`) at the
/// start of each function's entry block for every tensor block argument (after skipping the first
/// `implicit_arg_count` arguments) that has at least one static dimension.
///
/// Tensor arguments whose dimensions are all dynamic (`?` or symbolic) are skipped. Returns the
/// number of `mgp.assert` operations inserted.
pub fn verify_model_arguments(module: &mut Module, implicit_arg_count: usize) -> usize {
    use crate::ir::mlir::{DType, Dim, Location, OpName, Operation, Value, ValueDef};

    let mut inserted_asserts = 0usize;
    let idx_ty = Type::Scalar(DType::Index);
    let i1_ty = Type::Scalar(DType::I1);
    let spec_ty = Type::Opaque("!mgp.tensor_spec".to_string());

    let fn_ids = module.functions();
    for fn_id in fn_ids {
        let Some(&fn_reg) = module.ops[fn_id].regions.first() else {
            continue;
        };
        let Some(&entry_blk) = module.regions[fn_reg].blocks.first() else {
            continue;
        };
        let args = module.blocks[entry_blk].arguments.clone();
        if args.len() <= implicit_arg_count {
            continue;
        }

        let mut prologue_ops = Vec::new();
        for (extra_idx, &arg_val) in args[implicit_arg_count..].iter().enumerate() {
            let Type::Tensor(tt) = module.values[arg_val].ty.clone() else {
                continue;
            };
            if tt.shape.is_empty() {
                continue;
            }
            let static_dims: Vec<(usize, i64)> = tt
                .shape
                .iter()
                .enumerate()
                .filter_map(|(axis, d)| match d {
                    Dim::Static(n) => Some((axis, *n)),
                    _ => None,
                })
                .collect();
            if static_dims.is_empty() {
                continue;
            }

            let alloc_op_with_res = |m: &mut Module,
                                     name: OpName,
                                     operands: smallvec::SmallVec<[ValueId; 4]>,
                                     attrs: indexmap::IndexMap<String, Attribute>,
                                     res_ty: Option<Type>|
             -> (OpId, Option<ValueId>) {
                let mut results = smallvec::SmallVec::<[ValueId; 2]>::new();
                let res_v = res_ty.map(|ty| {
                    let v = m.values.insert(Value {
                        id: ValueId::default(),
                        ssa_name: None,
                        ty,
                        def: ValueDef::BlockArgument {
                            block: entry_blk,
                            index: 0,
                        },
                    });
                    m.values[v].id = v;
                    results.push(v);
                    v
                });
                let oid = m.ops.insert(Operation {
                    id: OpId::default(),
                    name,
                    operands,
                    results,
                    attributes: attrs,
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: Location::Unknown,
                });
                m.ops[oid].id = oid;
                if let Some(v) = res_v {
                    m.values[v].def = ValueDef::OpResult { op: oid, index: 0 };
                }
                (oid, res_v)
            };

            let (extract_op, Some(runtime_spec)) = alloc_op_with_res(
                module,
                OpName::new("mgp", "tensor.extract_tensor_spec"),
                smallvec::smallvec![arg_val],
                indexmap::IndexMap::new(),
                Some(spec_ty.clone()),
            ) else {
                unreachable!()
            };
            prologue_ops.push(extract_op);

            let mut are_equal: Option<ValueId> = None;
            for (axis, expected_len) in static_dims {
                let mut c_attrs = indexmap::IndexMap::new();
                c_attrs.insert("value".to_string(), Attribute::Int(expected_len));
                let (const_op, Some(expected_dim)) = alloc_op_with_res(
                    module,
                    OpName::new("index", "constant"),
                    smallvec::SmallVec::new(),
                    c_attrs,
                    Some(idx_ty.clone()),
                ) else {
                    unreachable!()
                };
                prologue_ops.push(const_op);

                let mut dim_attrs = indexmap::IndexMap::new();
                dim_attrs.insert("axis".to_string(), Attribute::Int(axis as i64));
                let (get_dim_op, Some(runtime_dim)) = alloc_op_with_res(
                    module,
                    OpName::new("mgp", "tensor_spec.get_dim"),
                    smallvec::smallvec![runtime_spec],
                    dim_attrs,
                    Some(idx_ty.clone()),
                ) else {
                    unreachable!()
                };
                prologue_ops.push(get_dim_op);

                let mut cmp_attrs = indexmap::IndexMap::new();
                cmp_attrs.insert("predicate".to_string(), Attribute::String("eq".to_string()));
                let (cmp_op, Some(dim_eq)) = alloc_op_with_res(
                    module,
                    OpName::new("index", "cmp"),
                    smallvec::smallvec![runtime_dim, expected_dim],
                    cmp_attrs,
                    Some(i1_ty.clone()),
                ) else {
                    unreachable!()
                };
                prologue_ops.push(cmp_op);

                are_equal = Some(match are_equal {
                    None => dim_eq,
                    Some(prev_eq) => {
                        let (and_op, Some(conj)) = alloc_op_with_res(
                            module,
                            OpName::new("pop", "and"),
                            smallvec::smallvec![prev_eq, dim_eq],
                            indexmap::IndexMap::new(),
                            Some(i1_ty.clone()),
                        ) else {
                            unreachable!()
                        };
                        prologue_ops.push(and_op);
                        conj
                    }
                });
            }

            if let Some(cond_val) = are_equal {
                let msg = format!(
                    "Tensor type mismatch for argument {extra_idx}: Expected shape: [{}]",
                    format_expected_tensor_shape(&tt.shape)
                );
                let mut assert_attrs = indexmap::IndexMap::new();
                assert_attrs.insert("msg".to_string(), Attribute::String(msg));
                let (assert_op, _) = alloc_op_with_res(
                    module,
                    OpName::new("mgp", "assert"),
                    smallvec::smallvec![cond_val],
                    assert_attrs,
                    None,
                );
                prologue_ops.push(assert_op);
                inserted_asserts += 1;
            }
        }

        if !prologue_ops.is_empty() {
            let mut new_block_ops = prologue_ops;
            new_block_ops.append(&mut module.blocks[entry_blk].ops);
            module.blocks[entry_blk].ops = new_block_ops;
        }
    }

    inserted_asserts
}

/// Scans every block in `module` for non-terminator operations with unused results (selecting at
/// most one unused result per operation, since a failing op sets all its results to error) and
/// synchronizes them before the block's terminator by inserting `mo.chain.create` (or `mgp.sync`)
/// and wiring the merged chain into the terminator's trailing chain operand.
///
/// Returns the total number of overhanging operations synchronized across all blocks.
pub fn fixup_overhanging_chains(module: &mut Module) -> usize {
    use crate::ir::mlir::{Location, OpName, Operation, Value, ValueDef};

    let mut synchronized_ops = 0usize;
    let chain_ty = Type::Opaque("!mo.chain".to_string());

    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for bid in block_ids {
        let ops_snapshot = module.blocks[bid].ops.clone();
        let Some(&term_id) = ops_snapshot.last() else {
            continue;
        };
        let term_name = module.ops[term_id].name.full_name();
        if !matches!(
            term_name.as_str(),
            "func.return" | "return" | "mo.output" | "mo.yield" | "mgp.output"
        ) {
            continue;
        }

        let mut hanging_results = smallvec::SmallVec::<[ValueId; 4]>::new();
        for &op_id in &ops_snapshot[..ops_snapshot.len() - 1] {
            let results = module.ops[op_id].results.clone();
            for res_v in results {
                let is_used = ops_snapshot.iter().any(|&other_id| {
                    other_id != op_id && module.ops[other_id].operands.contains(&res_v)
                });
                if !is_used {
                    hanging_results.push(res_v);
                    break;
                }
            }
        }
        if hanging_results.is_empty() {
            continue;
        }
        synchronized_ops += hanging_results.len();

        // If the terminator's last operand is already a `!mo.chain`, include it in the join:
        let term_operands = module.ops[term_id].operands.clone();
        let trailing_chain_slot = term_operands.last().and_then(|&last_v| {
            if matches!(&module.values[last_v].ty, Type::Opaque(s) if s == "!mo.chain") {
                Some((term_operands.len() - 1, last_v))
            } else {
                None
            }
        });
        if let Some((_, existing_ch)) = trailing_chain_slot {
            hanging_results.push(existing_ch);
        }

        let merged_ch = module.values.insert(Value {
            id: ValueId::default(),
            ssa_name: None,
            ty: chain_ty.clone(),
            def: ValueDef::BlockArgument {
                block: bid,
                index: 0,
            },
        });
        module.values[merged_ch].id = merged_ch;
        let sync_op = module.ops.insert(Operation {
            id: OpId::default(),
            name: OpName::new("mo", "chain.create"),
            operands: hanging_results,
            results: smallvec::smallvec![merged_ch],
            attributes: indexmap::IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: Location::Unknown,
        });
        module.ops[sync_op].id = sync_op;
        module.values[merged_ch].def = ValueDef::OpResult {
            op: sync_op,
            index: 0,
        };

        let term_pos = module.blocks[bid].ops.len() - 1;
        module.blocks[bid].ops.insert(term_pos, sync_op);

        if let Some((slot, _)) = trailing_chain_slot {
            module.ops[term_id].operands[slot] = merged_ch;
        } else {
            module.ops[term_id].operands.push(merged_ch);
        }
    }

    synchronized_ops
}

/// Maximum number of tensors permitted in a single inline tensor transfer envelope.
pub const MAX_INLINE_TENSOR_COUNT: usize = 65_536;

/// Maximum rank permitted for an inline transfer tensor.
pub const MAX_INLINE_TENSOR_RANK: usize = 16;

/// Maximum raw byte length (`2 GiB`) for a single inline transfer tensor.
pub const MAX_SINGLE_TENSOR_WIRE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Maximum cumulative raw byte length (`16 GiB`) across all tensors in an inline transfer
/// envelope.
pub const MAX_TOTAL_TENSOR_WIRE_BYTES: usize = 16 * 1024 * 1024 * 1024;

/// Maximum number of KV groups permitted in a grouped inline tensor transfer envelope.
pub const MAX_INLINE_TENSOR_GROUP_COUNT: usize = 64;

/// Maximum metadata wire overhead (`1024` bytes) budgeted per tensor frame.
pub const MAX_METADATA_BYTES_PER_TENSOR: usize = 1024;

pub(super) const SUPPORTED_INLINE_TENSOR_DTYPES: &[(&str, usize)] = &[
    ("uint8", 1),
    ("int8", 1),
    ("int16", 2),
    ("uint16", 2),
    ("float16", 2),
    ("bfloat16", 2),
    ("float8_e4m3fn", 1),
    ("float8_e5m2", 1),
    ("int32", 4),
    ("uint32", 4),
    ("float32", 4),
    ("int64", 8),
    ("uint64", 8),
    ("float64", 8),
    ("bool", 1),
];

/// Returns the element byte size for a canonical or `torch.`-prefixed inline tensor dtype name.
#[must_use]
pub fn inline_tensor_dtype_itemsize(dtype: &str) -> Option<usize> {
    let stripped = dtype.strip_prefix("torch.").unwrap_or(dtype);
    SUPPORTED_INLINE_TENSOR_DTYPES
        .iter()
        .find_map(|&(name, sz)| (name == stripped).then_some(sz))
}

/// Typed metadata and raw payload bytes for a single contiguous CPU tensor in an inline transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineTensorFrame {
    /// Canonical dtype string (`"float16"`, `"bfloat16"`, `"float32"`, `"uint8"`, etc.).
    pub dtype: String,
    /// Tensor dimensions (`rank <= 16`).
    pub shape: Vec<usize>,
    /// Raw contiguous little-endian tensor bytes (`<= 2 GiB`).
    pub data: Vec<u8>,
}

/// One KV group's sparse chunk indices and corresponding contiguous CPU tensor frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineTensorGroup {
    /// Strictly increasing chunk indices (`0 <= idx < num_chunks`), one per tensor frame.
    pub chunk_indices: Vec<usize>,
    /// Contiguous CPU tensor frames for each chunk index in `chunk_indices`.
    pub tensors: Vec<InlineTensorFrame>,
}

/// Validates that `indices` has length `num_tensors`, is strictly increasing, and (when
/// `num_chunks` is `Some(limit)`) every index is `< limit`.
pub fn validate_grouped_chunk_indices(
    gid: usize,
    indices: &[usize],
    num_tensors: usize,
    num_chunks: Option<usize>,
) -> Result<(), String> {
    if indices.len() != num_tensors {
        return Err(format!(
            "Inline tensor group {gid} has {} chunk indices for {num_tensors} tensors",
            indices.len()
        ));
    }
    if indices.windows(2).any(|w| w[1] <= w[0]) {
        return Err(format!(
            "Inline tensor group {gid} chunk indices are not strictly increasing"
        ));
    }
    if let Some(&last) = indices.last()
        && let Some(limit) = num_chunks
        && last >= limit
    {
        return Err(format!(
            "Inline tensor group {gid} chunk index out of range (chunks={limit})"
        ));
    }
    Ok(())
}

pub(super) fn validate_and_write_frames(
    frames: &[InlineTensorFrame],
    total_bytes: &mut usize,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    out.extend_from_slice(&(frames.len() as u32).to_be_bytes());
    for frame in frames {
        let dtype_norm = frame.dtype.strip_prefix("torch.").unwrap_or(&frame.dtype);
        let Some(itemsize) = inline_tensor_dtype_itemsize(dtype_norm) else {
            return Err(format!("Unsupported inline tensor dtype: {}", frame.dtype));
        };
        if frame.shape.len() > MAX_INLINE_TENSOR_RANK {
            return Err("Inline tensor rank exceeds 16".to_string());
        }
        let elem_count: usize = frame.shape.iter().copied().product();
        let expected_bytes = elem_count.saturating_mul(itemsize);
        if frame.data.len() != expected_bytes {
            return Err(format!(
                "Inline tensor byte length mismatch: expected {expected_bytes}, got {}",
                frame.data.len()
            ));
        }
        if frame.data.len() > MAX_SINGLE_TENSOR_WIRE_BYTES {
            return Err("Inline tensor exceeds 2 GiB".to_string());
        }
        *total_bytes = total_bytes.saturating_add(frame.data.len());
        if *total_bytes > MAX_TOTAL_TENSOR_WIRE_BYTES {
            return Err("Inline tensor payload exceeds 16 GiB".to_string());
        }
        let dt_bytes = dtype_norm.as_bytes();
        out.extend_from_slice(&(dt_bytes.len() as u16).to_be_bytes());
        out.extend_from_slice(dt_bytes);
        out.push(frame.shape.len() as u8);
        for &dim in &frame.shape {
            out.extend_from_slice(&(dim as u64).to_be_bytes());
        }
        out.extend_from_slice(&(frame.data.len() as u64).to_be_bytes());
        out.extend_from_slice(&frame.data);
    }
    Ok(())
}

pub(super) fn read_and_validate_frames(
    payload: &[u8],
    cursor: &mut usize,
    expected_shape: &[usize],
    expected_dtype: &str,
    expected_count: Option<usize>,
    total_bytes: &mut usize,
) -> Result<Vec<InlineTensorFrame>, String> {
    if *cursor + 4 > payload.len() {
        return Err("Unexpected EOF reading inline tensor frame count".to_string());
    }
    let count = u32::from_be_bytes(payload[*cursor..*cursor + 4].try_into().unwrap()) as usize;
    *cursor += 4;
    if count > MAX_INLINE_TENSOR_COUNT {
        return Err("Inline tensor count exceeds 65536".to_string());
    }
    if let Some(exp_cnt) = expected_count
        && count != exp_cnt
    {
        return Err(format!(
            "Inline tensor count mismatch: expected {exp_cnt}, got {count}"
        ));
    }

    let exp_dtype_norm = expected_dtype
        .strip_prefix("torch.")
        .unwrap_or(expected_dtype);
    let Some(exp_itemsize) = inline_tensor_dtype_itemsize(exp_dtype_norm) else {
        return Err(format!("Unsupported inline tensor dtype: {expected_dtype}"));
    };
    let expected_tensor_bytes = expected_shape
        .iter()
        .copied()
        .product::<usize>()
        .saturating_mul(exp_itemsize);

    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        if *cursor + 2 > payload.len() {
            return Err("Unexpected EOF reading inline tensor dtype length".to_string());
        }
        let dt_len = u16::from_be_bytes(payload[*cursor..*cursor + 2].try_into().unwrap()) as usize;
        *cursor += 2;
        if *cursor + dt_len > payload.len() {
            return Err("Unexpected EOF reading inline tensor dtype".to_string());
        }
        let dtype_str = std::str::from_utf8(&payload[*cursor..*cursor + dt_len])
            .map_err(|e| format!("Invalid UTF-8 in inline tensor dtype: {e}"))?
            .to_string();
        *cursor += dt_len;

        if inline_tensor_dtype_itemsize(&dtype_str).is_none() {
            return Err(format!("Unsupported inline tensor dtype: {dtype_str}"));
        }
        if dtype_str != exp_dtype_norm {
            return Err(format!(
                "Inline tensor dtype mismatch: expected {exp_dtype_norm}, got {dtype_str}"
            ));
        }

        if *cursor >= payload.len() {
            return Err("Unexpected EOF reading inline tensor rank".to_string());
        }
        let rank = payload[*cursor] as usize;
        *cursor += 1;
        if rank > MAX_INLINE_TENSOR_RANK {
            return Err("Inline tensor rank exceeds 16".to_string());
        }
        if *cursor + rank * 8 > payload.len() {
            return Err("Unexpected EOF reading inline tensor shape".to_string());
        }
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = u64::from_be_bytes(payload[*cursor..*cursor + 8].try_into().unwrap()) as usize;
            *cursor += 8;
            shape.push(d);
        }
        if shape != expected_shape {
            return Err(format!(
                "Inline tensor shape mismatch: expected {expected_shape:?}, got {shape:?}"
            ));
        }

        if *cursor + 8 > payload.len() {
            return Err("Unexpected EOF reading inline tensor data length".to_string());
        }
        let data_len =
            u64::from_be_bytes(payload[*cursor..*cursor + 8].try_into().unwrap()) as usize;
        *cursor += 8;
        if data_len != expected_tensor_bytes {
            return Err(format!(
                "Inline tensor byte length mismatch: expected \
                 {expected_tensor_bytes}, got {data_len}"
            ));
        }
        if *cursor + data_len > payload.len() {
            return Err("Unexpected EOF reading inline tensor data".to_string());
        }
        let data = payload[*cursor..*cursor + data_len].to_vec();
        *cursor += data_len;

        *total_bytes = total_bytes.saturating_add(data_len);
        if *total_bytes > MAX_TOTAL_TENSOR_WIRE_BYTES {
            return Err("Inline tensor payload exceeds 16 GiB".to_string());
        }
        frames.push(InlineTensorFrame {
            dtype: dtype_str,
            shape,
            data,
        });
    }
    Ok(frames)
}

/// Encodes a flat collection of CPU tensor frames into a versioned (`version = 1`, tag `0x54`)
/// wire envelope, enforcing per-tensor (`2 GiB`), cumulative (`16 GiB`), rank (`<= 16`), and
/// count (`<= 65536`) limits.
pub fn encode_inline_tensor_envelope(tensors: &[InlineTensorFrame]) -> Result<Vec<u8>, String> {
    if tensors.len() > MAX_INLINE_TENSOR_COUNT {
        return Err("Inline tensor count exceeds 65536".to_string());
    }
    let mut out = Vec::new();
    out.push(0x54); // 'T' = flat TensorEnvelope
    out.push(1); // version = 1
    let mut total_bytes = 0usize;
    validate_and_write_frames(tensors, &mut total_bytes, &mut out)?;
    Ok(out)
}

/// Decodes and validates a flat inline tensor wire envelope against `expected_shape`,
/// `expected_dtype`, and `expected_count`. Rejects grouped envelopes (`0x47`) so neither envelope
/// kind decodes as the other.
pub fn decode_inline_tensor_envelope(
    payload: &[u8],
    expected_shape: &[usize],
    expected_dtype: &str,
    expected_count: usize,
) -> Result<Vec<InlineTensorFrame>, String> {
    if expected_count > MAX_INLINE_TENSOR_COUNT {
        return Err("Expected tensor count must be between 0 and 65536".to_string());
    }
    let max_envelope_bytes = MAX_TOTAL_TENSOR_WIRE_BYTES
        .saturating_add(MAX_METADATA_BYTES_PER_TENSOR.saturating_mul(expected_count));
    if payload.len() > max_envelope_bytes {
        return Err("Inline tensor envelope exceeds the wire-format limit".to_string());
    }
    if payload.len() < 2 || payload[0] != 0x54 || payload[1] != 1 {
        return Err("Invalid or mismatched TensorEnvelope header".to_string());
    }
    let mut cursor = 2usize;
    let mut total_bytes = 0usize;
    read_and_validate_frames(
        payload,
        &mut cursor,
        expected_shape,
        expected_dtype,
        Some(expected_count),
        &mut total_bytes,
    )
}

/// Encodes a group-major collection of hybrid-KV CPU tensor groups into a versioned
/// (`version = 1`, tag `0x47`) wire envelope, enforcing group count (`<= 64`), strictly increasing
/// chunk indices per group, and cumulative byte/count limits.
pub fn encode_grouped_tensor_envelope(groups: &[InlineTensorGroup]) -> Result<Vec<u8>, String> {
    if groups.len() > MAX_INLINE_TENSOR_GROUP_COUNT {
        return Err("Inline tensor group count exceeds 64".to_string());
    }
    let total_tensors: usize = groups.iter().map(|g| g.tensors.len()).sum();
    if total_tensors > MAX_INLINE_TENSOR_COUNT {
        return Err("Inline tensor count exceeds 65536".to_string());
    }
    let mut out = Vec::new();
    out.push(0x47); // 'G' = GroupedTensorEnvelope
    out.push(1); // version = 1
    out.extend_from_slice(&(groups.len() as u16).to_be_bytes());
    let mut total_bytes = 0usize;
    for (gid, group) in groups.iter().enumerate() {
        validate_grouped_chunk_indices(gid, &group.chunk_indices, group.tensors.len(), None)?;
        out.extend_from_slice(&(group.chunk_indices.len() as u32).to_be_bytes());
        for &cidx in &group.chunk_indices {
            out.extend_from_slice(&(cidx as u64).to_be_bytes());
        }
        validate_and_write_frames(&group.tensors, &mut total_bytes, &mut out)?;
    }
    Ok(out)
}

/// Decodes and validates a group-major inline tensor wire envelope against per-group
/// `(expected_shape, expected_dtype, num_chunks)` specifications.
pub fn decode_grouped_tensor_envelope(
    payload: &[u8],
    expected: &[(Vec<usize>, String, usize)],
) -> Result<Vec<InlineTensorGroup>, String> {
    if expected.len() > MAX_INLINE_TENSOR_GROUP_COUNT {
        return Err("Expected group count must be between 0 and 64".to_string());
    }
    let expected_total: usize = expected.iter().map(|(_, _, c)| *c).sum();
    if expected_total > MAX_INLINE_TENSOR_COUNT {
        return Err("Expected tensor count must be between 0 and 65536".to_string());
    }
    let max_envelope_bytes = MAX_TOTAL_TENSOR_WIRE_BYTES
        .saturating_add(MAX_METADATA_BYTES_PER_TENSOR.saturating_mul(expected_total));
    if payload.len() > max_envelope_bytes {
        return Err("Inline tensor envelope exceeds the wire-format limit".to_string());
    }
    if payload.len() < 4 || payload[0] != 0x47 || payload[1] != 1 {
        return Err("Invalid or mismatched GroupedTensorEnvelope header".to_string());
    }
    let group_count = u16::from_be_bytes(payload[2..4].try_into().unwrap()) as usize;
    if group_count != expected.len() {
        return Err(format!(
            "Inline tensor group count mismatch: expected {}, got {group_count}",
            expected.len()
        ));
    }
    let mut cursor = 4usize;
    let mut total_bytes = 0usize;
    let mut groups = Vec::with_capacity(group_count);
    for (gid, (exp_shape, exp_dtype, num_chunks)) in expected.iter().enumerate() {
        if cursor + 4 > payload.len() {
            return Err("Unexpected EOF reading group chunk index count".to_string());
        }
        let idx_len = u32::from_be_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + idx_len * 8 > payload.len() {
            return Err("Unexpected EOF reading group chunk indices".to_string());
        }
        let mut chunk_indices = Vec::with_capacity(idx_len);
        for _ in 0..idx_len {
            let cidx = u64::from_be_bytes(payload[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;
            chunk_indices.push(cidx);
        }
        let tensors = read_and_validate_frames(
            payload,
            &mut cursor,
            exp_shape,
            exp_dtype,
            None,
            &mut total_bytes,
        )?;
        validate_grouped_chunk_indices(gid, &chunk_indices, tensors.len(), Some(*num_chunks))?;
        groups.push(InlineTensorGroup {
            chunk_indices,
            tensors,
        });
    }
    Ok(groups)
}
