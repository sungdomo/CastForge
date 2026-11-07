//! Parametric dependency graph analysis and Kahn sweep-and-swap topological
//! sorting within graph blocks.
//!
//! Covers:
//! 1. **Parametric Dependency Graph (`ParametricTopologyAnalysis`)**:
//!    Combines SSA use-def dependencies with symbolic parameter declarations
//!    (`param_decls` / `decl_params`) and symbolic parameter references
//!    (`param_uses` / `use_params` / symbolic dimension strings in tensor
//!    shapes). When a parameter provider or SSA producer resides in an
//!    enclosing or sibling region/block, propagates the dependency up the
//!    consumer's parent operation chain until both operations share a common
//!    block.
//! 2. **Kahn Sweep-and-Swap Topological Sort
//!    (`topologically_sort_block_with_parameters`)**:
//!    Computes in-degree blocker counts across all first-level non-terminator
//!    operations in a block (including nested region dependencies), schedules
//!    ready operations by sweep-and-swap while keeping block terminators
//!    anchored at the end of the block, and deterministically reports
//!    unschedulable operations in block order when a dependency cycle exists.
//! 3. **Rooted Transitive User Components (`rooted_transitive_user_components`)**:
//!    Collects an operation and all its transitive SSA/parametric users in
//!    topological order.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::mlir::{Attribute, BlockId, Module, OpId, Operation, RegionId, ValueDef};

fn is_terminator_name(full: &str) -> bool {
    matches!(
        full,
        "mef.output" | "mgp.yield" | "mo.output" | "func.return" | "return"
    )
}

/// Extracts declared symbolic parameter names from `op`.
fn collect_declared_params(op: &Operation) -> Vec<String> {
    let mut out = Vec::new();
    for key in ["param_decls", "decl_params", "output_param_decls"] {
        match op.attributes.get(key) {
            Some(Attribute::Array(items)) => {
                for item in items {
                    if let Attribute::String(s) = item
                        && !s.is_empty()
                    {
                        out.push(s.clone());
                    }
                }
            }
            Some(Attribute::String(s)) if !s.is_empty() => {
                out.push(s.clone());
            }
            _ => {}
        }
    }
    out
}

/// Extracts referenced symbolic parameter names from `op`.
fn collect_referenced_params(op: &Operation) -> Vec<String> {
    let mut out = Vec::new();
    for key in ["param_uses", "use_params", "input_params", "shape_attr"] {
        match op.attributes.get(key) {
            Some(Attribute::Array(items)) => {
                for item in items {
                    if let Attribute::String(s) = item
                        && !s.is_empty()
                        && s != "dyn"
                        && s != "?"
                    {
                        out.push(s.clone());
                    }
                }
            }
            Some(Attribute::String(s)) if !s.is_empty() && s != "dyn" && s != "?" => {
                out.push(s.clone());
            }
            _ => {}
        }
    }
    out
}

/// Precomputed parent-block and parent-op relationships across a [`Module`].
#[derive(Debug, Default)]
struct HierarchyIndex {
    op_to_block: HashMap<OpId, BlockId>,
    block_to_region: HashMap<BlockId, RegionId>,
    region_to_parent_op: HashMap<RegionId, OpId>,
}

impl HierarchyIndex {
    fn build(module: &Module) -> Self {
        let mut idx = Self::default();
        for (blk_id, blk) in &module.blocks {
            for &op_id in &blk.ops {
                idx.op_to_block.insert(op_id, blk_id);
            }
        }
        for (reg_id, reg) in &module.regions {
            for &blk_id in &reg.blocks {
                idx.block_to_region.insert(blk_id, reg_id);
            }
        }
        for (op_id, op) in &module.ops {
            for &reg_id in &op.regions {
                idx.region_to_parent_op.insert(reg_id, op_id);
            }
        }
        idx
    }

    fn parent_op(&self, op_id: OpId) -> Option<OpId> {
        let blk = self.op_to_block.get(&op_id)?;
        let reg = self.block_to_region.get(blk)?;
        self.region_to_parent_op.get(reg).copied()
    }

    fn parent_region(&self, op_id: OpId) -> Option<RegionId> {
        let blk = self.op_to_block.get(&op_id)?;
        self.block_to_region.get(blk).copied()
    }

    /// Returns `true` if `anc_reg` is a strict ancestor region of `desc_reg`.
    fn is_proper_ancestor_region(&self, anc_reg: RegionId, mut desc_reg: RegionId) -> bool {
        while let Some(&parent_op) = self.region_to_parent_op.get(&desc_reg) {
            let Some(next_reg) = self.parent_region(parent_op) else {
                break;
            };
            if next_reg == anc_reg {
                return true;
            }
            desc_reg = next_reg;
        }
        false
    }
}

/// Combined SSA + symbolic parameter dependency analysis for a [`Module`].
#[derive(Debug, Clone, Default)]
pub struct ParametricTopologyAnalysis {
    users_of: BTreeMap<OpId, Vec<OpId>>,
    providers_of: BTreeMap<OpId, Vec<OpId>>,
}

impl ParametricTopologyAnalysis {
    /// Builds the parametric and SSA dependency graph for `module`.
    #[must_use]
    pub fn analyze(module: &Module) -> Self {
        let hier = HierarchyIndex::build(module);
        let mut users_set: BTreeMap<OpId, HashSet<OpId>> = BTreeMap::new();
        let mut providers_set: BTreeMap<OpId, HashSet<OpId>> = BTreeMap::new();

        let mut apply_dep = |from: OpId, mut to: OpId, mut do_at_least_once: bool| {
            if from == to {
                return;
            }
            if let (Some(from_reg), Some(to_reg)) =
                (hier.parent_region(from), hier.parent_region(to))
                && hier.is_proper_ancestor_region(to_reg, from_reg)
            {
                return;
            }

            let from_blk = hier.op_to_block.get(&from).copied();
            while hier.op_to_block.get(&to).copied() != from_blk {
                do_at_least_once = true;
                if from != to {
                    users_set.entry(from).or_default().insert(to);
                    providers_set.entry(to).or_default().insert(from);
                }
                let Some(p) = hier.parent_op(to) else {
                    return;
                };
                to = p;
            }
            if do_at_least_once && from != to {
                users_set.entry(from).or_default().insert(to);
                providers_set.entry(to).or_default().insert(from);
            }
        };

        // 1. Collect symbolic parameter declarations across all live operations.
        let live_ops = module.live_ops();
        let mut param_decl_owner: HashMap<String, OpId> = HashMap::new();
        for &oid in &live_ops {
            if let Some(op) = module.ops.get(oid) {
                for name in collect_declared_params(op) {
                    param_decl_owner.entry(name).or_insert(oid);
                }
            }
        }

        // 2. Wire parametric edges (`do_at_least_once = true`) and SSA edges.
        for &oid in &live_ops {
            let Some(op) = module.ops.get(oid) else {
                continue;
            };
            for param_name in collect_referenced_params(op) {
                if let Some(&decl_op) = param_decl_owner.get(&param_name) {
                    apply_dep(decl_op, oid, true);
                }
            }
            for &opd in &op.operands {
                if let Some(val) = module.values.get(opd)
                    && let ValueDef::OpResult { op: def_op, .. } = val.def
                {
                    // Same-block SSA edges are also recorded so Kahn blocker
                    // counting sees both SSA and parametric providers uniformly.
                    apply_dep(def_op, oid, true);
                }
            }
        }

        let mut users_of = BTreeMap::new();
        for (k, set) in users_set {
            let mut v: Vec<OpId> = set.into_iter().collect();
            v.sort_by_key(|id| format!("{id:?}"));
            users_of.insert(k, v);
        }
        let mut providers_of = BTreeMap::new();
        for (k, set) in providers_set {
            let mut v: Vec<OpId> = set.into_iter().collect();
            v.sort_by_key(|id| format!("{id:?}"));
            providers_of.insert(k, v);
        }

        Self {
            users_of,
            providers_of,
        }
    }

    /// Returns the direct SSA and parametric users of `op_id`.
    #[must_use]
    pub fn users(&self, op_id: OpId) -> &[OpId] {
        self.users_of.get(&op_id).map_or(&[], Vec::as_slice)
    }

    /// Returns the direct SSA and parametric providers of `op_id`.
    #[must_use]
    pub fn providers(&self, op_id: OpId) -> &[OpId] {
        self.providers_of.get(&op_id).map_or(&[], Vec::as_slice)
    }
}

/// Collects `op_id` and all nested operations inside `op_id`'s regions.
fn collect_op_and_nested(module: &Module, op_id: OpId, out: &mut Vec<OpId>) {
    out.push(op_id);
    let Some(op) = module.ops.get(op_id) else {
        return;
    };
    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &child_id in &blk.ops {
                collect_op_and_nested(module, child_id, out);
            }
        }
    }
}

/// Topologically sorts the first-level operations of `block_id` using in-degree
/// (Kahn) sweep-and-swap scheduling over both SSA and symbolic parameter
/// dependencies, keeping any block terminator anchored at the end of `block_id`.
///
/// # Errors
///
/// Returns `Err` if a dependency cycle prevents scheduling, listing the stuck
/// operations in deterministic block order.
pub fn topologically_sort_block_with_parameters(
    module: &mut Module,
    block_id: BlockId,
) -> Result<usize, String> {
    let orig_ops = module
        .blocks
        .get(block_id)
        .map(|b| b.ops.clone())
        .unwrap_or_default();
    if orig_ops.len() <= 1 {
        return Ok(0);
    }

    let analysis = ParametricTopologyAnalysis::analyze(module);
    let hier = HierarchyIndex::build(module);

    // Separate trailing terminator if present so it always remains at the end.
    let has_trailing_term = orig_ops.last().is_some_and(|&last_id| {
        module
            .ops
            .get(last_id)
            .is_some_and(|op| is_terminator_name(&op.name.full_name()))
    });
    let sched_len = if has_trailing_term {
        orig_ops.len() - 1
    } else {
        orig_ops.len()
    };
    if sched_len <= 1 {
        return Ok(0);
    }

    let mut ops: Vec<OpId> = orig_ops[..sched_len].to_vec();
    let mut id_of: HashMap<OpId, usize> = HashMap::with_capacity(sched_len);
    for (idx, &oid) in ops.iter().enumerate() {
        id_of.insert(oid, idx);
    }

    let mut num_blockers = vec![0usize; sched_len];
    let mut waiters_of: Vec<Vec<usize>> = vec![Vec::new(); sched_len];

    for (id, &op_id) in ops.iter().enumerate() {
        let mut nested = Vec::new();
        collect_op_and_nested(module, op_id, &mut nested);
        let mut blockers: HashSet<usize> = HashSet::new();
        for nested_op in nested {
            for &provider in analysis.providers(nested_op) {
                let mut cur = Some(provider);
                while let Some(parent) = cur {
                    if parent == op_id {
                        break;
                    }
                    if let Some(&blocker_idx) = id_of.get(&parent) {
                        blockers.insert(blocker_idx);
                    }
                    cur = hier.parent_op(parent);
                }
            }
        }
        num_blockers[id] = blockers.len();
        for b_idx in blockers {
            waiters_of[b_idx].push(id);
        }
    }

    let mut ids: Vec<usize> = (0..sched_len).collect();
    let mut next_scheduled = 0usize;

    while next_scheduled < sched_len {
        let mut scheduled_at_least_once = false;
        let sweep_start = next_scheduled;
        for i in sweep_start..sched_len {
            if num_blockers[ids[i]] != 0 {
                continue;
            }
            for &waiter in &waiters_of[ids[i]] {
                num_blockers[waiter] = num_blockers[waiter].saturating_sub(1);
            }
            ops.swap(i, next_scheduled);
            ids.swap(i, next_scheduled);
            scheduled_at_least_once = true;
            next_scheduled += 1;
        }

        if !scheduled_at_least_once {
            let stuck_names: Vec<String> = ops[next_scheduled..]
                .iter()
                .filter_map(|&oid| module.ops.get(oid).map(|o| o.name.full_name()))
                .collect();
            return Err(format!(
                "cannot schedule operation: dependency cycle in the graph ({})",
                stuck_names.join(", ")
            ));
        }
    }

    let moved_count = ops
        .iter()
        .zip(orig_ops[..sched_len].iter())
        .filter(|(a, b)| a != b)
        .count();

    if has_trailing_term {
        ops.push(orig_ops[sched_len]);
    }
    if let Some(blk_mut) = module.blocks.get_mut(block_id) {
        blk_mut.ops = ops;
    }

    Ok(moved_count)
}

/// Collects `root_op` and all its transitive SSA/parametric users within the
/// same block, ordered topologically.
#[must_use]
pub fn rooted_transitive_user_components(module: &Module, root_op: OpId) -> Vec<OpId> {
    let analysis = ParametricTopologyAnalysis::analyze(module);
    let mut fringe = vec![root_op];
    let mut enqueued: HashSet<OpId> = HashSet::from([root_op]);
    let mut reachable: HashSet<OpId> = HashSet::new();

    while let Some(cur) = fringe.pop() {
        reachable.insert(cur);
        for &user in analysis.users(cur) {
            if enqueued.insert(user) {
                fringe.push(user);
            }
        }
    }

    // Order reachable ops according to block topological position.
    let mut ordered = Vec::with_capacity(reachable.len());
    for &oid in &module.live_ops() {
        if reachable.contains(&oid) {
            ordered.push(oid);
        }
    }
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::mlir::{DType, Dim, Location, OpName, TensorType, Type};
    use indexmap::IndexMap;
    use smallvec::{SmallVec, smallvec};

    #[test]
    fn test_parametric_topological_sort_reorders_decl_before_use() {
        let mut module = Module::new(Some("param_topo".to_string()));
        let blk = module.body_block();
        let t_ty = Type::Tensor(TensorType {
            shape: vec![Dim::Static(4)],
            dtype: DType::F32,
            encoding: None,
        });

        // Insert `op_use` (uses parameter "D0") BEFORE `op_decl` (declares "D0").
        let mut use_attrs = IndexMap::new();
        use_attrs.insert(
            "param_uses".to_string(),
            Attribute::Array(vec![Attribute::String("D0".to_string())]),
        );
        let op_use = module.add_operation(
            blk,
            OpName::new("mo", " broadcast_to"),
            SmallVec::new(),
            vec![(None, t_ty.clone())],
            use_attrs,
            Vec::new(),
            false,
            Location::Unknown,
        );

        let mut decl_attrs = IndexMap::new();
        decl_attrs.insert(
            "param_decls".to_string(),
            Attribute::Array(vec![Attribute::String("D0".to_string())]),
        );
        let op_decl = module.add_operation(
            blk,
            OpName::new("mo", "rebind"),
            SmallVec::new(),
            vec![(None, t_ty)],
            decl_attrs,
            Vec::new(),
            false,
            Location::Unknown,
        );

        let res_use = module.ops[op_use].results[0];
        let term = module.add_operation(
            blk,
            OpName::new("mo", "output"),
            smallvec![res_use],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );

        let moved = topologically_sort_block_with_parameters(&mut module, blk).expect("topo sort");
        assert!(moved >= 1);
        assert_eq!(module.blocks[blk].ops, vec![op_decl, op_use, term]);
    }
}
