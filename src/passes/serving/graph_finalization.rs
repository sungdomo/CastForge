//! Device-graph finalization (`FinalizeDeviceGraphsPass`), relational async dataflow analysis
//! (`AsyncDataflowDependencies`), RMO legalization (`LegalizeRMOOps`), and symbolic algebraic
//! optimization (`SymbolicOptimization`).

use crate::ir::{
    Attribute, Block, BlockId, DType, Dim, Module, OpId, OpName, Operation, Region, TensorType,
    Type, Value, ValueDef, ValueId,
};
use crate::passes::serving::device_graphs::{
    is_shape_only_in_region, reconcile_and_resolve_device_graph_builder_casts,
};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

// ============================================================================
// Device Graph Finalization (`FinalizeDeviceGraphsPass`)
// ============================================================================

/// Summary report produced by [`finalize_device_graphs`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceGraphFinalizationReport {
    /// Number of builder-to-context conversion casts reconciled or resolved.
    pub resolved_builder_casts: usize,
    /// Number of shape-only region inputs annotated to skip device buffer materialization.
    pub shape_only_inputs_marked: usize,
    /// Number of redundant pass-through chains stripped from collective yields.
    pub stripped_passthrough_chains: usize,
    /// Number of `mgp.device_graph.collective` ops lowered into async create/cache + execute pairs.
    pub collectives_lowered: usize,
}

/// Finalizes staged/partitioned `mgp.device_graph.collective` operations in `module`:
/// 1. Runs [`reconcile_and_resolve_device_graph_builder_casts`] to fold inverse casts and lower
///    audited `builder_to_device_context` bridges into `mgp.device_graph.device_context`.
/// 2. Identifies shape-only arguments (`is_shape_only_in_region`) in each collective region and
///    records their indices in `shape_only_input_indices`.
/// 3. Strips pass-through `!grt.chain` operands from `mgp.device_graph.collective.yield` when the
///    yielded chain is an unmodified block argument.
/// 4. Lowers each `mgp.device_graph.collective` into an `mogg.routine.async` operation (wrapping
///    `mgp.device_graph.collective.create` + `mgp.device_graph.cache` + `mogg.routine.return`)
///    followed by `mgp.device_graph.collective.execute`, rewiring all downstream uses.
pub fn finalize_device_graphs(
    module: &mut Module,
) -> Result<DeviceGraphFinalizationReport, String> {
    let mut report = DeviceGraphFinalizationReport {
        resolved_builder_casts: reconcile_and_resolve_device_graph_builder_casts(module)?,
        ..Default::default()
    };

    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        let mut new_block_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len() + 2);

        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if op.name.full_name() != "mgp.device_graph.collective" {
                new_block_ops.push(op_id);
                continue;
            }

            // Inspect collective regions for shape-only block arguments and pass-through chains.
            let mut shape_only_indices: Vec<Attribute> = Vec::new();
            for &reg_id in &op.regions {
                let Some(reg) = module.regions.get(reg_id).cloned() else {
                    continue;
                };
                if let Some(&first_blk) = reg.blocks.first()
                    && let Some(r_blk) = module.blocks.get(first_blk).cloned()
                {
                    let blk_args: IndexSet<ValueId> = r_blk.arguments.iter().copied().collect();
                    for (arg_idx, &arg_val) in r_blk.arguments.iter().enumerate() {
                        if is_shape_only_in_region(module, reg_id, arg_val) {
                            shape_only_indices.push(Attribute::Int(arg_idx as i64));
                            report.shape_only_inputs_marked += 1;
                        }
                    }
                    if let Some(&last_op_id) = r_blk.ops.last()
                        && let Some(yield_op) = module.ops.get_mut(last_op_id)
                        && yield_op.name.full_name() == "mgp.device_graph.collective.yield"
                    {
                        let before_len = yield_op.operands.len();
                        yield_op.operands.retain(|v| {
                            let is_chain = matches!(
                                &module.values[*v].ty,
                                Type::Opaque(c) if c.contains("chain")
                            );
                            !(blk_args.contains(v) && is_chain)
                        });
                        report.stripped_passthrough_chains += before_len - yield_op.operands.len();
                    }
                }
            }

            let loc = op.location.clone();
            let handle_ty = Type::Opaque("!mgp.device_graph.handle".to_string());

            // Build the inner `mogg.routine.async` body:
            //   %raw = mgp.device_graph.collective.create(...) { regions = ... }
            //   %cached = mgp.device_graph.cache(%raw)
            //   mogg.routine.return(%cached)
            let async_blk = module.blocks.insert_with_key(|id| Block {
                id,
                label: None,
                arguments: Vec::new(),
                ops: Vec::new(),
            });
            let mut create_attrs = op.attributes.clone();
            create_attrs.insert(
                "shape_only_input_indices".to_string(),
                Attribute::Array(shape_only_indices),
            );
            let create_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "device_graph.collective.create"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: create_attrs,
                regions: op.regions.clone(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let raw_handle = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: handle_ty.clone(),
                def: ValueDef::OpResult {
                    op: create_op,
                    index: 0,
                },
            });
            module.ops[create_op].results.push(raw_handle);
            module.blocks[async_blk].ops.push(create_op);

            let cache_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "device_graph.cache"),
                operands: SmallVec::from_slice(&[raw_handle]),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let cached_handle = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: handle_ty.clone(),
                def: ValueDef::OpResult {
                    op: cache_op,
                    index: 0,
                },
            });
            module.ops[cache_op].results.push(cached_handle);
            module.blocks[async_blk].ops.push(cache_op);

            let ret_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg", "routine.return"),
                operands: SmallVec::from_slice(&[cached_handle]),
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            module.blocks[async_blk].ops.push(ret_op);

            let async_reg = module.regions.insert_with_key(|id| Region {
                id,
                blocks: vec![async_blk],
            });

            let mut async_attrs = IndexMap::new();
            async_attrs.insert(
                "dispatch".to_string(),
                Attribute::String("deferred".to_string()),
            );
            async_attrs.insert("can_fuse_into_other".to_string(), Attribute::Bool(false));
            async_attrs.insert("can_be_fused_into".to_string(), Attribute::Bool(false));
            async_attrs.insert("can_duplicate".to_string(), Attribute::Bool(false));

            let async_routine_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg", "routine.async"),
                operands: op.operands.clone(),
                results: SmallVec::new(),
                attributes: async_attrs,
                regions: vec![async_reg],
                custom_syntax: false,
                location: loc.clone(),
            });
            let outer_handle = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: handle_ty,
                def: ValueDef::OpResult {
                    op: async_routine_op,
                    index: 0,
                },
            });
            module.ops[async_routine_op].results.push(outer_handle);
            new_block_ops.push(async_routine_op);

            // Rewrite `op_id` in place into `mgp.device_graph.collective.execute(%outer_handle)`.
            let mut exec_opds: SmallVec<[ValueId; 4]> =
                SmallVec::with_capacity(1 + op.operands.len());
            exec_opds.push(outer_handle);
            exec_opds.extend(op.operands.iter().copied());
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName::new("mgp", "device_graph.collective.execute");
                op_mut.operands = exec_opds;
                op_mut.regions.clear();
            }
            new_block_ops.push(op_id);
            report.collectives_lowered += 1;
        }

        if let Some(blk_mut) = module.blocks.get_mut(blk_id) {
            blk_mut.ops = new_block_ops;
        }
    }

    Ok(report)
}

// ============================================================================
// Relational Async Dataflow Analysis (`AsyncDataflowDependencies`)
// ============================================================================

/// Relational dataflow analysis over SSA values, chains, and operations in a [`Module`].
///
/// Tracks:
/// - `same_as(a, b)`: identity equivalence classes (`mgp.alias`, `mogg.view.identity`,
///   single-operand `grt.chain.join`).
/// - `maybe_same_as(a, b)`: potential storage aliasing across lifetime-extending views
///   (`mgp.extend_lifetime`, `mgp.buffer.slice`, `mgp.tensor.slice`, `mgp.view`) and in-place
///   mutators (`mgp.mutable_load`, `mgp.in_place_op`).
/// - `is_always_ready(val)`: values that are ready at function entry without asynchronous device
///   waits (constants, entry parameters, and `grt.chain.create` whose inputs are all ready).
/// - `implies_ready(src, dst)`: whether readiness of `src` transitively guarantees readiness of
///   `dst` (`src == dst`, `is_always_ready(dst)`, or `src` post-dominates `dst` through chain
///   joins and async completions).
/// - `are_disjoint(op_a, op_b)`: whether two operations cannot execute concurrently because one
///   strictly happens-after the other (`runs_later` / `runs_earlier`) or because they reside in
///   mutually exclusive conditional branches (`structurally_cannot_overlap` / `exclusive_set`).
/// - `is_non_self_overlapping(loop_op, arg_index)`: whether a loop-carried token/buffer in a
///   `scf.while` / `mo.while` loop depends on the previous iteration's completion before reuse.
#[derive(Debug, Clone)]
pub struct AsyncDataflowAnalysis {
    same_root: IndexMap<ValueId, ValueId>,
    alias_sources: IndexMap<ValueId, IndexSet<ValueId>>,
    always_ready: IndexSet<ValueId>,
    ready_ancestors: IndexMap<ValueId, IndexSet<ValueId>>,
    op_ancestors: IndexMap<OpId, IndexSet<OpId>>,
    op_branch_tag: IndexMap<OpId, (OpId, usize)>,
}

impl AsyncDataflowAnalysis {
    /// Computes the relational async dataflow analysis for `module`.
    pub fn analyze(module: &Module) -> Self {
        let mut same_root: IndexMap<ValueId, ValueId> = IndexMap::new();
        let mut alias_sources: IndexMap<ValueId, IndexSet<ValueId>> = IndexMap::new();
        let mut always_ready: IndexSet<ValueId> = IndexSet::new();
        let mut ready_ancestors: IndexMap<ValueId, IndexSet<ValueId>> = IndexMap::new();
        let mut op_ancestors: IndexMap<OpId, IndexSet<OpId>> = IndexMap::new();
        let mut op_branch_tag: IndexMap<OpId, (OpId, usize)> = IndexMap::new();

        for (val_id, val) in &module.values {
            same_root.insert(val_id, val_id);
            let mut self_set = IndexSet::new();
            self_set.insert(val_id);
            alias_sources.insert(val_id, self_set);
            ready_ancestors.insert(val_id, IndexSet::from([val_id]));
            if matches!(val.def, ValueDef::BlockArgument { .. }) {
                always_ready.insert(val_id);
            }
        }

        // Tag operations inside conditional branches (`scf.if`, `mo.if`, `cf.cond_br_region`).
        for (op_id, op) in &module.ops {
            let full = op.name.full_name();
            if matches!(full.as_str(), "scf.if" | "mo.if") && op.regions.len() >= 2 {
                for (branch_idx, &reg_id) in op.regions.iter().enumerate() {
                    if let Some(reg) = module.regions.get(reg_id) {
                        for &b_id in &reg.blocks {
                            if let Some(b) = module.blocks.get(b_id) {
                                for &inner_op in &b.ops {
                                    op_branch_tag.insert(inner_op, (op_id, branch_idx));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Propagate equivalence, alias closures, readiness, and op ordering across blocks.
        for (_, blk) in &module.blocks {
            for &op_id in &blk.ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let full = op.name.full_name();

                // Record transitive op ancestors from operand producers.
                let mut anc: IndexSet<OpId> = IndexSet::new();
                for &opd in &op.operands {
                    if let ValueDef::OpResult { op: prod_op, .. } = module.values[opd].def {
                        anc.insert(prod_op);
                        if let Some(prev_anc) = op_ancestors.get(&prod_op) {
                            anc.extend(prev_anc.iter().copied());
                        }
                    }
                }
                op_ancestors.insert(op_id, anc);

                // Identity equivalence (`same_as`).
                if (matches!(full.as_str(), "mgp.alias" | "mogg.view.identity")
                    || (full == "grt.chain.join" && op.operands.len() == 1))
                    && !op.operands.is_empty()
                    && !op.results.is_empty()
                {
                    let src_root = *same_root.get(&op.operands[0]).unwrap_or(&op.operands[0]);
                    for &res in &op.results {
                        same_root.insert(res, src_root);
                    }
                }

                // Lifetime-extension / may-alias closure (`maybe_same_as`).
                let is_lifetime_or_alias = matches!(
                    full.as_str(),
                    "mgp.alias"
                        | "mogg.view.identity"
                        | "mgp.extend_lifetime"
                        | "mgp.buffer.slice"
                        | "mgp.buffer.bulk_slice"
                        | "mgp.tensor.slice"
                        | "mgp.view"
                        | "mgp.mutable_load"
                        | "mgp.in_place_op"
                );
                if is_lifetime_or_alias && !op.operands.is_empty() {
                    let base = op.operands[0];
                    let base_roots = alias_sources
                        .get(&base)
                        .cloned()
                        .unwrap_or_else(|| IndexSet::from([base]));
                    for &res in &op.results {
                        alias_sources
                            .entry(res)
                            .or_default()
                            .extend(base_roots.iter().copied());
                    }
                }

                // Readiness (`is_always_ready` and `implies_ready`).
                let is_const = full.ends_with(".constant")
                    || full == "stablehlo.constant"
                    || full == "kgen.param.constant";
                if is_const
                    || (full == "grt.chain.create"
                        && op.operands.iter().all(|v| always_ready.contains(v)))
                {
                    for &res in &op.results {
                        always_ready.insert(res);
                    }
                }

                // Any chain/token or op result implies its operand tokens are ready once it is
                // ready.
                let mut implied: IndexSet<ValueId> = IndexSet::new();
                for &opd in &op.operands {
                    let root_opd = *same_root.get(&opd).unwrap_or(&opd);
                    implied.insert(opd);
                    implied.insert(root_opd);
                    if let Some(prev) = ready_ancestors.get(&opd) {
                        implied.extend(prev.iter().copied());
                    }
                }
                for &res in &op.results {
                    let mut set = implied.clone();
                    set.insert(res);
                    if let Some(&r) = same_root.get(&res) {
                        set.insert(r);
                    }
                    ready_ancestors.insert(res, set);
                }
            }
        }

        Self {
            same_root,
            alias_sources,
            always_ready,
            ready_ancestors,
            op_ancestors,
            op_branch_tag,
        }
    }

    /// Returns `true` if `a` and `b` are provably identical values (`sameAs`).
    pub fn same_as(&self, a: ValueId, b: ValueId) -> bool {
        let ra = self.same_root.get(&a).copied().unwrap_or(a);
        let rb = self.same_root.get(&b).copied().unwrap_or(b);
        ra == rb
    }

    /// Returns `true` if `a` and `b` may alias the same underlying storage (`maybeSameAs`).
    pub fn maybe_same_as(&self, a: ValueId, b: ValueId) -> bool {
        if self.same_as(a, b) {
            return true;
        }
        let Some(sa) = self.alias_sources.get(&a) else {
            return false;
        };
        let Some(sb) = self.alias_sources.get(&b) else {
            return false;
        };
        sa.iter().any(|x| sb.contains(x))
    }

    /// Returns `true` if `val` is always ready without waiting on asynchronous device work.
    pub fn is_always_ready(&self, val: ValueId) -> bool {
        let root = self.same_root.get(&val).copied().unwrap_or(val);
        self.always_ready.contains(&val) || self.always_ready.contains(&root)
    }

    /// Returns `true` if readiness of `src` guarantees readiness of `dst` (`impliesReady`).
    pub fn implies_ready(&self, src: ValueId, dst: ValueId) -> bool {
        if self.same_as(src, dst) || self.is_always_ready(dst) {
            return true;
        }
        let dst_root = self.same_root.get(&dst).copied().unwrap_or(dst);
        self.ready_ancestors
            .get(&src)
            .is_some_and(|set| set.contains(&dst) || set.contains(&dst_root))
    }

    /// Returns `true` if `later_op` transitively depends on `earlier_op` (`runsLater`).
    pub fn runs_later(&self, later_op: OpId, earlier_op: OpId) -> bool {
        self.op_ancestors
            .get(&later_op)
            .is_some_and(|anc| anc.contains(&earlier_op))
    }

    /// Returns `true` if `op_a` and `op_b` reside in mutually exclusive branches of the same
    /// conditional operation (`structurallyCannotOverlap` / `exclusiveSet`).
    pub fn structurally_cannot_overlap(&self, op_a: OpId, op_b: OpId) -> bool {
        if let (Some(&(parent_a, branch_a)), Some(&(parent_b, branch_b))) =
            (self.op_branch_tag.get(&op_a), self.op_branch_tag.get(&op_b))
        {
            parent_a == parent_b && branch_a != branch_b
        } else {
            false
        }
    }

    /// Returns `true` if `op_a` and `op_b` have disjoint execution lifetimes (`areDisjoint`).
    pub fn are_disjoint(&self, op_a: OpId, op_b: OpId) -> bool {
        if op_a == op_b {
            return false;
        }
        self.runs_later(op_a, op_b)
            || self.runs_later(op_b, op_a)
            || self.structurally_cannot_overlap(op_a, op_b)
    }

    /// Returns `true` if the loop-carried token/argument at `arg_index` in `loop_op` (`scf.while`
    /// or `mo.while`) is non-self-overlapping across iterations (the yielded value for iteration
    /// `i + 1` transitively depends on the block argument from iteration `i`).
    pub fn is_non_self_overlapping(
        &self,
        module: &Module,
        loop_op: OpId,
        arg_index: usize,
    ) -> bool {
        let Some(op) = module.ops.get(loop_op) else {
            return false;
        };
        let Some(&reg_id) = op.regions.last() else {
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
        let Some(&iter_arg) = blk.arguments.get(arg_index) else {
            return false;
        };
        let Some(&term_id) = blk.ops.last() else {
            return false;
        };
        let Some(term_op) = module.ops.get(term_id) else {
            return false;
        };
        let Some(&yielded_val) = term_op.operands.get(arg_index) else {
            return false;
        };
        self.implies_ready(yielded_val, iter_arg)
    }
}

/// Uses [`AsyncDataflowAnalysis`] to prune `is_always_ready` and implied (`implies_ready`) chain
/// operands from `grt.chain.join`, `mgp.device.wait`, and `mogg.routine.async` operations across
/// `module`. Returns the total number of redundant dependency edges removed.
pub fn optimize_async_dataflow_dependencies(module: &mut Module) -> usize {
    let analysis = AsyncDataflowAnalysis::analyze(module);
    let mut pruned_edges = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let ops = blk.ops.clone();
        for &op_id in &ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if !matches!(
                full.as_str(),
                "grt.chain.join" | "mgp.device.wait" | "mogg.routine.async"
            ) || op.operands.len() <= 1
            {
                continue;
            }

            let mut kept: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (i, &cand) in op.operands.iter().enumerate() {
                let is_chain = matches!(
                    &module.values[cand].ty,
                    Type::Opaque(c) if c.contains("chain") || c.contains("token")
                ) || full == "grt.chain.join";
                if !is_chain {
                    kept.push(cand);
                    continue;
                }
                if analysis.is_always_ready(cand) && (kept.len() + (op.operands.len() - i - 1)) >= 1
                {
                    pruned_edges += 1;
                    continue;
                }
                let implied_by_other = op.operands.iter().enumerate().any(|(j, &other)| {
                    i != j
                        && (!analysis.same_as(cand, other) || j < i)
                        && analysis.implies_ready(other, cand)
                });
                if implied_by_other && (kept.len() + (op.operands.len() - i - 1)) >= 1 {
                    pruned_edges += 1;
                } else {
                    kept.push(cand);
                }
            }

            if kept.len() != op.operands.len()
                && let Some(op_mut) = module.ops.get_mut(op_id)
            {
                op_mut.operands = kept;
            }
        }
    }

    pruned_edges
}

// ============================================================================
// RMO Legalization (`LegalizeRMOOps`) & Symbolic Algebraic Optimization
// ============================================================================

/// Returns `true` if `lhs` and `rhs` have the same rank and no dimension position has conflicting
/// static sizes (`Dim::Static(a)` vs. `Dim::Static(b)` with `a != b`).
pub fn is_compatible_shape(lhs: &[Dim], rhs: &[Dim]) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    lhs.iter().zip(rhs.iter()).all(|(d0, d1)| match (d0, d1) {
        (Dim::Static(a), Dim::Static(b)) => a == b,
        _ => true,
    })
}

fn dtype_promotion_rank(dt: &DType) -> u8 {
    match dt {
        DType::I1 => 1,
        DType::I4 => 2,
        DType::U8 => 3,
        DType::I8 => 4,
        DType::U16 => 5,
        DType::I16 => 6,
        DType::U32 => 7,
        DType::I32 => 8,
        DType::U64 => 9,
        DType::I64 | DType::Index => 10,
        DType::F8E4M3FN | DType::F8E5M2 => 11,
        DType::F16 => 12,
        DType::BF16 => 13,
        DType::F32 => 14,
        DType::F64 => 15,
        DType::QuantUniform { .. } => 0,
    }
}

/// Promotes two scalar/element [`DType`]s along the numerical promotion lattice (`Bool < Int <
/// Float < Complex`, with `F16 + BF16 -> F32`).
pub fn promote_rmo_dtypes(lhs: &DType, rhs: &DType) -> DType {
    if lhs == rhs {
        return lhs.clone();
    }
    if matches!(
        (lhs, rhs),
        (DType::F16, DType::BF16) | (DType::BF16, DType::F16)
    ) {
        return DType::F32;
    }
    if dtype_promotion_rank(lhs) >= dtype_promotion_rank(rhs) {
        lhs.clone()
    } else {
        rhs.clone()
    }
}

fn tensor_rank_and_dtype(ty: &Type) -> (usize, DType) {
    match ty {
        Type::Tensor(tt) => (tt.shape.len(), tt.dtype.clone()),
        Type::Scalar(dt) => (0, dt.clone()),
        _ => (1, DType::F32),
    }
}

/// Summary report produced by [`legalize_rmo_ops`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RmoLegalizationReport {
    /// Number of duplicate `rmo.assert` / `mo.assert` operations deduplicated per block.
    pub deduplicated_asserts: usize,
    /// Number of `rmo.rebind_shape` operations lowered to `rmo.rebind_tensor_shape` with filtered
    /// `output_param_decls`.
    pub legalized_rebinds: usize,
    /// Number of composite broadcast elementwise operations decomposed into explicit broadcast +
    /// dtype promotion + primitive elementwise ops.
    pub decomposed_elementwise: usize,
}

/// Legalizes high-level RMO operations in `module`:
/// 1. Deduplicates `rmo.assert` and `mo.assert` operations within each block by
///    `(condition_value, message)` (`AssertCache`), rewiring duplicate assert results to the
///    first dominating assert in the block.
/// 2. Validates `rmo.rebind_shape` / `mo.rebind_shape` input-vs-output shapes with
///    [`is_compatible_shape`], returning `Err` on incompatible shapes, and lowers valid rebinds
///    into `rmo.rebind_tensor_shape` with `output_param_decls` filtered to only the symbolic
///    dimension names referenced by the output tensor type.
/// 3. Decomposes `rmo.broadcast_elementwise.<op>` operations into `mo.shape_of`,
///    `mo.broadcast_shape`, `mo.broadcast_to`, `mo.cast` (to [`promote_rmo_dtypes`]), and
///    `mo.elementwise.<op>`.
pub fn legalize_rmo_ops(module: &mut Module) -> Result<RmoLegalizationReport, String> {
    let mut report = RmoLegalizationReport::default();
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        let mut assert_cache: IndexMap<(ValueId, String), ValueId> = IndexMap::new();
        let mut new_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len() + 4);

        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();

            // 1. Deduplicate `rmo.assert` / `mo.assert` by `(condition, message)`.
            if matches!(full.as_str(), "rmo.assert" | "mo.assert") && !op.operands.is_empty() {
                let cond_val = op.operands[0];
                let msg = match op.attributes.get("message") {
                    Some(Attribute::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let key = (cond_val, msg);
                if let Some(&existing_res) = assert_cache.get(&key) {
                    if let Some(&dup_res) = op.results.first() {
                        module.replace_all_uses(dup_res, existing_res);
                    }
                    module.ops.remove(op_id);
                    report.deduplicated_asserts += 1;
                    continue;
                }
                if let Some(&first_res) = op.results.first() {
                    assert_cache.insert(key, first_res);
                }
                new_ops.push(op_id);
                continue;
            }

            // 2. Legalize `rmo.rebind_shape` / `mo.rebind_shape` into `rmo.rebind_tensor_shape`.
            if matches!(full.as_str(), "rmo.rebind_shape" | "mo.rebind_shape")
                && !op.operands.is_empty()
                && !op.results.is_empty()
            {
                let in_ty = &module.values[op.operands[0]].ty;
                let out_ty = &module.values[op.results[0]].ty;
                if let (Type::Tensor(in_tt), Type::Tensor(out_tt)) = (in_ty, out_ty) {
                    if !is_compatible_shape(&in_tt.shape, &out_tt.shape) {
                        return Err(format!(
                            "incompatible rebind_shape between {:?} and {:?}",
                            in_tt.shape, out_tt.shape
                        ));
                    }
                    let mut referenced_syms: IndexSet<String> = IndexSet::new();
                    for d in &out_tt.shape {
                        if let Dim::Symbolic(s) = d {
                            referenced_syms.insert(s.clone());
                        }
                    }
                    let in_decls = op.attributes.get("input_param_decls");
                    let filtered_decls: Vec<Attribute> = match in_decls {
                        Some(Attribute::Array(arr)) => arr
                            .iter()
                            .filter(|attr| {
                                matches!(attr, Attribute::String(s) if referenced_syms.contains(s))
                            })
                            .cloned()
                            .collect(),
                        _ => referenced_syms
                            .into_iter()
                            .map(Attribute::String)
                            .collect(),
                    };
                    if let Some(op_mut) = module.ops.get_mut(op_id) {
                        op_mut.name = OpName::new("rmo", "rebind_tensor_shape");
                        op_mut.attributes.insert(
                            "output_param_decls".to_string(),
                            Attribute::Array(filtered_decls),
                        );
                    }
                    report.legalized_rebinds += 1;
                }
                new_ops.push(op_id);
                continue;
            }

            // 3. Decompose `rmo.broadcast_elementwise.<op>`.
            if let Some(elem_fn) = full.strip_prefix("rmo.broadcast_elementwise.")
                && op.operands.len() == 2
                && !op.results.is_empty()
            {
                let lhs = op.operands[0];
                let rhs = op.operands[1];
                let out_val = op.results[0];
                let loc = op.location.clone();
                let (_, lhs_dt) = tensor_rank_and_dtype(&module.values[lhs].ty);
                let (_, rhs_dt) = tensor_rank_and_dtype(&module.values[rhs].ty);
                let promoted_dt = promote_rmo_dtypes(&lhs_dt, &rhs_dt);
                let out_shape = match &module.values[out_val].ty {
                    Type::Tensor(tt) => tt.shape.clone(),
                    _ => vec![Dim::Static(1)],
                };
                let promoted_tensor_ty = Type::Tensor(TensorType {
                    dtype: promoted_dt.clone(),
                    shape: out_shape,
                    encoding: None,
                });
                let shape_ty = Type::Tensor(TensorType {
                    dtype: DType::I64,
                    shape: vec![Dim::Static(2)],
                    encoding: None,
                });

                let shape_lhs_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mo", "shape_of"),
                    operands: SmallVec::from_slice(&[lhs]),
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let shape_lhs_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: shape_ty.clone(),
                    def: ValueDef::OpResult {
                        op: shape_lhs_op,
                        index: 0,
                    },
                });
                module.ops[shape_lhs_op].results.push(shape_lhs_val);
                new_ops.push(shape_lhs_op);

                let shape_rhs_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mo", "shape_of"),
                    operands: SmallVec::from_slice(&[rhs]),
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let shape_rhs_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: shape_ty.clone(),
                    def: ValueDef::OpResult {
                        op: shape_rhs_op,
                        index: 0,
                    },
                });
                module.ops[shape_rhs_op].results.push(shape_rhs_val);
                new_ops.push(shape_rhs_op);

                let bshape_op = module.ops.insert_with_key(|id| Operation {
                    id,
                    name: OpName::new("mo", "broadcast_shape"),
                    operands: SmallVec::from_slice(&[shape_lhs_val, shape_rhs_val]),
                    results: SmallVec::new(),
                    attributes: IndexMap::new(),
                    regions: Vec::new(),
                    custom_syntax: false,
                    location: loc.clone(),
                });
                let bshape_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: shape_ty,
                    def: ValueDef::OpResult {
                        op: bshape_op,
                        index: 0,
                    },
                });
                module.ops[bshape_op].results.push(bshape_val);
                new_ops.push(bshape_op);

                let mut cast_and_bcast = |input_val: ValueId, input_dt: DType| -> ValueId {
                    let casted = if input_dt != promoted_dt {
                        let c_op = module.ops.insert_with_key(|id| Operation {
                            id,
                            name: OpName::new("mo", "cast"),
                            operands: SmallVec::from_slice(&[input_val]),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: loc.clone(),
                        });
                        let c_val = module.values.insert_with_key(|id| Value {
                            id,
                            ssa_name: None,
                            ty: promoted_tensor_ty.clone(),
                            def: ValueDef::OpResult { op: c_op, index: 0 },
                        });
                        module.ops[c_op].results.push(c_val);
                        new_ops.push(c_op);
                        c_val
                    } else {
                        input_val
                    };
                    let b_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mo", "broadcast_to"),
                        operands: SmallVec::from_slice(&[casted, bshape_val]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: loc.clone(),
                    });
                    let b_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: promoted_tensor_ty.clone(),
                        def: ValueDef::OpResult { op: b_op, index: 0 },
                    });
                    module.ops[b_op].results.push(b_val);
                    new_ops.push(b_op);
                    b_val
                };

                let b_lhs = cast_and_bcast(lhs, lhs_dt);
                let b_rhs = cast_and_bcast(rhs, rhs_dt);

                module.values[out_val].ty = promoted_tensor_ty;
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.name = OpName::new("mo", format!("elementwise.{elem_fn}"));
                    op_mut.operands = SmallVec::from_slice(&[b_lhs, b_rhs]);
                }
                new_ops.push(op_id);
                report.decomposed_elementwise += 1;
                continue;
            }

            new_ops.push(op_id);
        }

        if let Some(blk_mut) = module.blocks.get_mut(blk_id) {
            blk_mut.ops = new_ops;
        }
    }

    Ok(report)
}

/// Simplifies symbolic algebraic dimension expressions (`sym.add`, `sym.sub`, `sym.mul`,
/// `sym.div`, `sym.max`, `sym.min`, `mo.symbolic.select`, and `sym.assert_eq` dominating equality
/// replacements) across `module`. Returns the number of simplified symbolic operations.
pub fn optimize_symbolic_dims(module: &mut Module) -> usize {
    let mut simplified = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        let mut const_vals: IndexMap<ValueId, i64> = IndexMap::new();
        let mut zero_val_id: Option<ValueId> = None;
        let mut one_val_id: Option<ValueId> = None;
        let mut eq_replacements: IndexMap<ValueId, ValueId> = IndexMap::new();
        let mut dead_ops: IndexSet<OpId> = IndexSet::new();

        for &op_id in &orig_ops {
            if !module.ops.contains_key(op_id) {
                continue;
            }
            // Apply any dominating equality replacements (`sym.assert_eq(a, b)` -> `a`).
            if !eq_replacements.is_empty()
                && let Some(op_mut) = module.ops.get_mut(op_id)
            {
                for opd in &mut op_mut.operands {
                    while let Some(&rep) = eq_replacements.get(opd) {
                        *opd = rep;
                    }
                }
            }
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();

            if matches!(full.as_str(), "sym.const" | "index.constant")
                && let Some(&res) = op.results.first()
                && let Some(Attribute::Int(v)) = op.attributes.get("value")
            {
                const_vals.insert(res, *v);
                if *v == 0 && zero_val_id.is_none() {
                    zero_val_id = Some(res);
                }
                if *v == 1 && one_val_id.is_none() {
                    one_val_id = Some(res);
                }
                continue;
            }

            if full == "sym.assert_eq" && op.operands.len() == 2 {
                let lhs = op.operands[0];
                let rhs = op.operands[1];
                if lhs != rhs {
                    eq_replacements.insert(rhs, lhs);
                    simplified += 1;
                }
                continue;
            }

            if op.results.len() != 1 {
                continue;
            }
            let res = op.results[0];

            if op.operands.len() == 2 {
                let a = op.operands[0];
                let b = op.operands[1];
                let ca = const_vals.get(&a).copied();
                let cb = const_vals.get(&b).copied();

                let replacement: Option<ValueId> = match full.as_str() {
                    "sym.add" => {
                        if cb == Some(0) {
                            Some(a)
                        } else if ca == Some(0) {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    "sym.sub" => {
                        if cb == Some(0) {
                            Some(a)
                        } else if a == b {
                            zero_val_id
                        } else {
                            None
                        }
                    }
                    "sym.mul" => {
                        if cb == Some(1) {
                            Some(a)
                        } else if ca == Some(1) {
                            Some(b)
                        } else if ca == Some(0) {
                            Some(a)
                        } else if cb == Some(0) {
                            Some(b)
                        } else {
                            None
                        }
                    }
                    "sym.div" => {
                        if cb == Some(1) {
                            Some(a)
                        } else if a == b {
                            one_val_id
                        } else {
                            None
                        }
                    }
                    "sym.max" | "sym.min" if a == b => Some(a),
                    _ => None,
                };

                if let Some(rep_val) = replacement {
                    module.replace_all_uses(res, rep_val);
                    dead_ops.insert(op_id);
                    simplified += 1;
                }
            } else if full == "mo.symbolic.select" && op.operands.len() == 3 {
                let cond = op.operands[0];
                let t_val = op.operands[1];
                let f_val = op.operands[2];
                let rep = if t_val == f_val || const_vals.get(&cond).copied() == Some(1) {
                    Some(t_val)
                } else if const_vals.get(&cond).copied() == Some(0) {
                    Some(f_val)
                } else {
                    None
                };
                if let Some(rep_val) = rep {
                    module.replace_all_uses(res, rep_val);
                    dead_ops.insert(op_id);
                    simplified += 1;
                }
            }
        }

        for &dead_id in &dead_ops {
            module.remove_op(blk_id, dead_id);
        }
    }

    simplified
}
