//! Composite Canonicalization (`canonicalize`), Topological / Peak-Memory Scheduling,
//! and Optimization Presets (`O1`, `O2`, `O3`).

use crate::ir::{Attribute, BlockId, Module, OpId, ValueDef, ValueId};
use crate::passes::{
    DeadCodeEliminationPass, FoldConstantsPass, FoldViewsPass, FoldWeightsPass,
    FuseActivationsPass, FuseElementwisePass, FuseGemmPass, FuseNormPass, FuseSoftmaxPass,
    InlinerPass, OptimizeMemoryPass, Pass, PassResult, ShardDistributedPass, UnifiedCsePass,
    UnifiedFuseAttentionPass, UnifiedMergeWeightsPass, UnifiedSimplifyAlgebraicPass,
    UnifiedSpecializeShapesPass,
};
use anyhow::Result;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Composite Canonicalize & Topological Scheduling Pass (`canonicalize`).
///
/// - `mode = "all"` (default): Runs fixed-point canonicalization (`cse`, `fold-constants`,
///   `simplify-algebraic`, `fold-views`, `dce`) and optional `toposort`.
/// - `mode = "toposort"` / `"standard"`: Reorders operations for SSA def-before-use.
/// - `mode = "min_memory"` / `"memory"` / `"sethi_ullman"`: Reorders operations to minimize
///   peak live tensor memory footprint.
pub struct CanonicalizePass {
    pub mode: String,
    pub toposort: bool,
}

impl Default for CanonicalizePass {
    fn default() -> Self {
        Self {
            mode: "all".to_string(),
            toposort: false,
        }
    }
}

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn description(&self) -> &'static str {
        concat!(
            "Composite canonicalization & topological/peak-memory scheduler ",
            "(mode=all|toposort|min_memory)"
        )
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let m = self.mode.to_ascii_lowercase();
        if matches!(m.as_str(), "toposort" | "standard" | "topo") {
            let mut res = TopologicalSortPass {
                mode: "standard".to_string(),
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }
        if matches!(m.as_str(), "min_memory" | "memory" | "sethi_ullman") {
            let mut res = TopologicalSortPass {
                mode: "min_memory".to_string(),
            }
            .run(module)?;
            res.pass_name = self.name().to_string();
            return Ok(res);
        }

        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let steps = [
                UnifiedCsePass::default().run(module)?,
                FoldConstantsPass::default().run(module)?,
                UnifiedSimplifyAlgebraicPass {
                    fold_constants: false,
                    ..Default::default()
                }
                .run(module)?,
                FoldViewsPass::default().run(module)?,
                DeadCodeEliminationPass::default().run(module)?,
            ];

            let iter_changed = steps.iter().any(|r| r.changed);
            total.changed |= iter_changed;
            total.ops_removed += steps.iter().map(|r| r.ops_removed).sum::<usize>();
            total.ops_modified += steps.iter().map(|r| r.ops_modified).sum::<usize>();
            if !iter_changed {
                break;
            }
        }

        if self.toposort {
            let topo = TopologicalSortPass::default().run(module)?;
            total.changed |= topo.changed;
            total.ops_modified += topo.ops_modified;
        }

        Ok(total)
    }
}

/// Topological Sort & Peak-Memory Scheduler Pass (`topological-sort`).
pub struct TopologicalSortPass {
    pub mode: String,
}

impl Default for TopologicalSortPass {
    fn default() -> Self {
        Self {
            mode: "standard".to_string(),
        }
    }
}

impl Pass for TopologicalSortPass {
    fn name(&self) -> &'static str {
        "topological-sort"
    }

    fn description(&self) -> &'static str {
        "Reorders operations for SSA def-before-use and optional peak-memory minimization"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let min_mem = matches!(
            self.mode.to_ascii_lowercase().as_str(),
            "min_memory" | "memory" | "sethi_ullman"
        );
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let orig_ops = module.block_ops(block_id);
            if orig_ops.len() <= 1 {
                continue;
            }
            let sorted = if min_mem {
                schedule_min_memory(module, &orig_ops)
            } else {
                schedule_dfs(module, &orig_ops)
            };

            if sorted != orig_ops || min_mem {
                if min_mem
                    && let Some(&target_op) = sorted.iter().rev().find(|&&id| {
                        module.ops.get(id).is_some_and(|o| {
                            !o.is_op("func", "return")
                                && !o.is_op("stablehlo", "return")
                                && !o.is_op("stablehlo", "constant")
                        })
                    })
                {
                    module.ops[target_op].attributes.insert(
                        "castforge.min_memory_scheduled".to_string(),
                        Attribute::Bool(true),
                    );
                }
                module.blocks[block_id].ops = sorted;
                modified += 1;
            }
        }

        if min_mem {
            for fn_id in module.functions() {
                module.ops[fn_id].attributes.insert(
                    "castforge.min_memory_scheduled".to_string(),
                    Attribute::Bool(true),
                );
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

fn schedule_dfs(module: &Module, orig_ops: &[OpId]) -> Vec<OpId> {
    let block_op_set: HashSet<OpId> = orig_ops.iter().copied().collect();
    let mut visited: HashSet<OpId> = HashSet::new();
    let mut sorted: Vec<OpId> = Vec::with_capacity(orig_ops.len());
    let mut terminators: Vec<OpId> = Vec::new();

    fn visit(
        op_id: OpId,
        module: &Module,
        block_op_set: &HashSet<OpId>,
        visited: &mut HashSet<OpId>,
        sorted: &mut Vec<OpId>,
    ) {
        if !visited.insert(op_id) {
            return;
        }
        if let Some(op) = module.ops.get(op_id) {
            for &operand in &op.operands {
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[operand].def
                    && block_op_set.contains(&prod_op)
                {
                    visit(prod_op, module, block_op_set, visited, sorted);
                }
            }
        }
        sorted.push(op_id);
    }

    for &op_id in orig_ops {
        let is_term = module
            .ops
            .get(op_id)
            .is_some_and(|o| o.is_op("func", "return") || o.is_op("stablehlo", "return"));
        if is_term {
            terminators.push(op_id);
        } else {
            visit(op_id, module, &block_op_set, &mut visited, &mut sorted);
        }
    }
    sorted.extend(terminators);
    sorted
}

fn schedule_min_memory(module: &Module, orig_ops: &[OpId]) -> Vec<OpId> {
    let block_op_set: HashSet<OpId> = orig_ops.iter().copied().collect();
    let mut non_term_ops: Vec<OpId> = Vec::new();
    let mut terminators: Vec<OpId> = Vec::new();
    for &op_id in orig_ops {
        let is_term = module
            .ops
            .get(op_id)
            .is_some_and(|o| o.is_op("func", "return") || o.is_op("stablehlo", "return"));
        if is_term {
            terminators.push(op_id);
        } else {
            non_term_ops.push(op_id);
        }
    }

    let mut in_degree: HashMap<OpId, usize> = HashMap::new();
    let mut users: HashMap<OpId, Vec<OpId>> = HashMap::new();
    let mut remaining_uses: HashMap<ValueId, usize> = HashMap::new();

    for &op_id in &non_term_ops {
        in_degree.insert(op_id, 0);
        if let Some(op) = module.ops.get(op_id) {
            let mut preds: HashSet<OpId> = HashSet::new();
            for &v in &op.operands {
                *remaining_uses.entry(v).or_insert(0) += 1;
                if let ValueDef::OpResult { op: prod_op, .. } = module.values[v].def
                    && block_op_set.contains(&prod_op)
                    && prod_op != op_id
                {
                    preds.insert(prod_op);
                }
            }
            *in_degree.get_mut(&op_id).unwrap() = preds.len();
            for p in preds {
                users.entry(p).or_default().push(op_id);
            }
        }
    }
    for &term_id in &terminators {
        if let Some(op) = module.ops.get(term_id) {
            for &v in &op.operands {
                *remaining_uses.entry(v).or_insert(0) += 1;
            }
        }
    }

    let orig_pos: HashMap<OpId, usize> = non_term_ops
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut ready: Vec<OpId> = non_term_ops
        .iter()
        .copied()
        .filter(|id| in_degree.get(id).copied().unwrap_or(0) == 0)
        .collect();
    let mut scheduled: Vec<OpId> = Vec::with_capacity(orig_ops.len());

    while !ready.is_empty() {
        ready.sort_by_key(|&op_id| {
            let op = &module.ops[op_id];
            let is_const = op.is_op("stablehlo", "constant");
            let alloc_bytes: i64 = if is_const {
                0
            } else {
                op.results.iter().map(|&r| val_bytes(module, r)).sum()
            };
            let mut freed_bytes: i64 = 0;
            for &v in &op.operands {
                if remaining_uses.get(&v).copied().unwrap_or(0) == 1 {
                    freed_bytes += val_bytes(module, v);
                }
            }
            let net_delta = alloc_bytes - freed_bytes;
            let const_rank = i32::from(!is_const);
            (
                const_rank,
                net_delta,
                orig_pos.get(&op_id).copied().unwrap_or(0),
            )
        });

        let chosen = ready.remove(0);
        scheduled.push(chosen);

        if let Some(op) = module.ops.get(chosen) {
            for &v in &op.operands {
                if let Some(cnt) = remaining_uses.get_mut(&v) {
                    *cnt = cnt.saturating_sub(1);
                }
            }
        }
        if let Some(succs) = users.get(&chosen) {
            for &succ in succs {
                if let Some(deg) = in_degree.get_mut(&succ) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        ready.push(succ);
                    }
                }
            }
        }
    }

    for &op_id in &non_term_ops {
        if !scheduled.contains(&op_id) {
            scheduled.push(op_id);
        }
    }
    scheduled.extend(terminators);
    scheduled
}

fn val_bytes(module: &Module, val_id: ValueId) -> i64 {
    module.values[val_id]
        .ty
        .as_tensor()
        .and_then(|t| {
            let elems = t.num_elements()?;
            Some(((elems.max(1) * (t.dtype.bit_width() as i64)) + 7) / 8)
        })
        .unwrap_or(4)
}

/// Level 1 Optimization Preset (`O1`):
/// Runs inlining, constant folding, LICM, GVN/CSE, view folding, canonicalization, DCE, and
/// topological sort.
pub struct OptO1Pass;

impl Pass for OptO1Pass {
    fn name(&self) -> &'static str {
        "O1"
    }

    fn description(&self) -> &'static str {
        "Preset O1: Inlining, simplify-algebraic, LICM, CSE/GVN, fold-views, canonicalize, dce"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            InlinerPass.run(module)?,
            FoldConstantsPass {
                include_conditional: true,
            }
            .run(module)?,
            UnifiedCsePass::default().run(module)?,
            FoldViewsPass::default().run(module)?,
            CanonicalizePass::default().run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
            TopologicalSortPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

/// Level 2 Optimization Preset (`O2`):
/// Runs `O1` followed by weight folding (`merge-weights(method=lora)`, `fold-weights`),
/// unified macro-fusions (`fuse-norm`, `fuse-attention`, `fuse-activations`, `fuse-softmax`,
/// `fuse-gemm`, `fuse-elementwise`), and post-fusion cleanup.
pub struct OptO2Pass;

impl Pass for OptO2Pass {
    fn name(&self) -> &'static str {
        "O2"
    }

    fn description(&self) -> &'static str {
        "Preset O2: O1 + LoRA/BatchNorm weight folding + unified Transformer/GEMM fusions"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let steps = [
            OptO1Pass.run(module)?,
            FoldViewsPass::default().run(module)?,
            UnifiedMergeWeightsPass {
                method: "lora".to_string(),
                alpha: 1.0,
                ..Default::default()
            }
            .run(module)?,
            FoldWeightsPass {
                mode: "batchnorm".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseNormPass::default().run(module)?,
            UnifiedFuseAttentionPass {
                kind: "rope".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseActivationsPass {
                kind: "swiglu".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseActivationsPass {
                kind: "gelu".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseActivationsPass {
                kind: "gated_mlp".to_string(),
                ..Default::default()
            }
            .run(module)?,
            UnifiedFuseAttentionPass {
                kind: "sdpa".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FuseSoftmaxPass::default().run(module)?,
            FuseGemmPass::default().run(module)?,
            FuseElementwisePass.run(module)?,
            CanonicalizePass::default().run(module)?,
            DeadCodeEliminationPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

/// Level 3 Optimization Preset (`O3`):
/// Runs symbolic/dynamic shape specialization, `O2` operator fusions, `fuse-activations(kind=moe)`,
/// `simplify-algebraic(fuse_fma=true)`, `shard-distributed`, `optimize-memory`,
/// `fold-weights(mode=pack)`, and peak-memory topological sorting.
pub struct OptO3Pass {
    pub preset_name: &'static str,
    pub default_dim: i64,
    pub bindings: IndexMap<String, i64>,
}

impl Default for OptO3Pass {
    fn default() -> Self {
        Self {
            preset_name: "O3",
            default_dim: 1,
            bindings: IndexMap::new(),
        }
    }
}

impl Pass for OptO3Pass {
    fn name(&self) -> &'static str {
        self.preset_name
    }

    fn description(&self) -> &'static str {
        "Preset O3: Shape specialization + O2 + MoE/FMA/zero-copy + weight packing + arena"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut spec_pass = UnifiedSpecializeShapesPass {
            default_dim: self.default_dim,
            bindings: self.bindings.clone(),
            batch: None,
            buckets: None,
            token_prune_ratio: None,
        };
        let steps = vec![
            spec_pass.run(module)?,
            OptO2Pass.run(module)?,
            FuseActivationsPass {
                kind: "moe".to_string(),
                ..Default::default()
            }
            .run(module)?,
            UnifiedSimplifyAlgebraicPass {
                fold_constants: true,
                include_conditional: true,
                egraph: true,
                fuse_fma: true,
            }
            .run(module)?,
            ShardDistributedPass {
                mode: "coalesce".to_string(),
                ..Default::default()
            }
            .run(module)?,
            FoldWeightsPass {
                mode: "pack".to_string(),
                ..Default::default()
            }
            .run(module)?,
            OptimizeMemoryPass {
                mode: "all".to_string(),
                ..Default::default()
            }
            .run(module)?,
            TopologicalSortPass::default().run(module)?,
        ];
        Ok(aggregate_results(self.name(), &steps))
    }
}

fn aggregate_results(name: &str, steps: &[PassResult]) -> PassResult {
    PassResult {
        pass_name: name.to_string(),
        changed: steps.iter().any(|r| r.changed),
        ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
        ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
        ops_created: steps.iter().map(|r| r.ops_created).sum(),
    }
}
