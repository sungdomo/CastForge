use crate::{
    FoldConstantsPass, FoldReshapePass, FoldTransposePass, Pass, PassResult, SimplifyAlgebraicPass,
};
use anyhow::Result;
use castforge_ir::{BlockId, Dim, Location, Module, OpId, ValueDef};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// 1. Dead Code Elimination (DCE — SSA values + optional Symbol DCE)
pub struct DeadCodeEliminationPass {
    pub include_symbols: bool,
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self {
            include_symbols: true,
        }
    }
}

impl DeadCodeEliminationPass {
    pub fn ssa_only() -> Self {
        Self {
            include_symbols: false,
        }
    }
}

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn description(&self) -> &'static str {
        "Dead Code Elimination (removes unused pure SSA ops and dead private symbols)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        loop {
            let mut pass_removed = 0usize;
            for &block_id in &blocks {
                let ops = module.block_ops(block_id);
                // Iterate in reverse so chains of dead ops are cleaned up quickly
                for &op_id in ops.iter().rev() {
                    let Some(op) = module.ops.get(op_id) else {
                        continue;
                    };
                    if !op.is_pure() || op.results.is_empty() {
                        continue;
                    }
                    let all_unused = op.results.iter().all(|&res| module.use_count(res) == 0);
                    if all_unused {
                        module.remove_op(block_id, op_id);
                        pass_removed += 1;
                    }
                }
            }
            removed += pass_removed;
            if pass_removed == 0 {
                break;
            }
        }

        if self.include_symbols {
            let sym_res = SymbolDcePass.run(module)?;
            removed += sym_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 2. Common Subexpression Elimination (CSE)
pub struct CommonSubexpressionEliminationPass;

impl Pass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Common Subexpression Elimination (deduplicates identical pure operations)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let mut seen: HashMap<String, OpId> = HashMap::new();
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                    continue;
                }
                let res_val = op.results[0];
                let res_ty = &module.values[res_val].ty;
                let sig = format!(
                    "{}|operands:{:?}|attrs:{:?}|ty:{}",
                    op.name, op.operands, op.attributes, res_ty
                );

                if let Some(&existing_op_id) = seen.get(&sig) {
                    let existing_res = module.ops[existing_op_id].results[0];
                    module.replace_all_uses(res_val, existing_res);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                } else {
                    seen.insert(sig, op_id);
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 7. Composite Canonicalize Pass
pub struct CanonicalizePass;

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn description(&self) -> &'static str {
        "Composite canonicalization pass (constants, algebraic, transpose, reshape, cse, dce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let r0 = FoldConstantsPass::default().run(module)?;
            let r_alg = SimplifyAlgebraicPass.run(module)?;
            let r1 = FoldTransposePass::default().run(module)?;
            let r2 = FoldReshapePass.run(module)?;
            let r3 = CommonSubexpressionEliminationPass.run(module)?;
            let r4 = DeadCodeEliminationPass::default().run(module)?;

            let iter_changed =
                r0.changed || r_alg.changed || r1.changed || r2.changed || r3.changed || r4.changed;
            total.changed |= iter_changed;
            total.ops_removed += r0.ops_removed
                + r_alg.ops_removed
                + r1.ops_removed
                + r2.ops_removed
                + r3.ops_removed
                + r4.ops_removed;
            total.ops_modified += r0.ops_modified
                + r_alg.ops_modified
                + r1.ops_modified
                + r2.ops_modified
                + r3.ops_modified
                + r4.ops_modified;
            if !iter_changed {
                break;
            }
        }

        Ok(total)
    }
}

/// 15. Symbol Dead Code Elimination Pass (`symbol-dce`)
/// Eliminates unreachable `private` functions from the module.
pub struct SymbolDcePass;

impl Pass for SymbolDcePass {
    fn name(&self) -> &'static str {
        "symbol-dce"
    }

    fn description(&self) -> &'static str {
        "Symbol Dead Code Elimination (removes unreachable private func.func symbols)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut referenced_symbols = HashSet::new();
        for (_, op) in &module.ops {
            if op.is_op("func", "call") {
                if let Some(callee) = op.attributes.get("callee").and_then(|a| a.as_str()) {
                    referenced_symbols.insert(callee.to_string());
                }
            }
            if op.is_op("stablehlo", "composite") {
                if let Some(decomp) = op.attributes.get("decomposition").and_then(|a| a.as_str()) {
                    referenced_symbols.insert(decomp.to_string());
                }
            }
        }

        let root_block = module.body_block();
        let mut removed = 0usize;
        for fn_op_id in module.functions() {
            let op = &module.ops[fn_op_id];
            let is_private =
                op.attributes.get("sym_visibility").and_then(|a| a.as_str()) == Some("private");
            let sym_name = op
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            if is_private && !referenced_symbols.contains(sym_name) {
                let regions = op.regions.clone();
                for reg_id in regions {
                    let blocks = module.regions[reg_id].blocks.clone();
                    for blk_id in blocks {
                        let inner_ops = module.block_ops(blk_id);
                        for inner_op in inner_ops {
                            module.remove_op(blk_id, inner_op);
                        }
                    }
                }
                module.remove_op(root_block, fn_op_id);
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 17. Topological Sort Pass (`topological-sort`)
/// Reorders operations inside each block so all SSA definitions precede their uses.
pub struct TopologicalSortPass;

impl Pass for TopologicalSortPass {
    fn name(&self) -> &'static str {
        "topological-sort"
    }

    fn description(&self) -> &'static str {
        "Reorders operations inside blocks to satisfy SSA def-before-use ordering"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let orig_ops = module.block_ops(block_id);
            if orig_ops.len() <= 1 {
                continue;
            }
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
                        if let ValueDef::OpResult { op: prod_op, .. } = module.values[operand].def {
                            if block_op_set.contains(&prod_op) {
                                visit(prod_op, module, block_op_set, visited, sorted);
                            }
                        }
                    }
                }
                sorted.push(op_id);
            }

            for &op_id in &orig_ops {
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

            if sorted != orig_ops {
                module.blocks[block_id].ops = sorted;
                modified += 1;
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

/// 18. Strip DebugInfo Pass (`strip-debuginfo`)
/// Resets all operation `Location` annotations to `Location::Unknown`.
pub struct StripDebugInfoPass;

impl Pass for StripDebugInfoPass {
    fn name(&self) -> &'static str {
        "strip-debuginfo"
    }

    fn description(&self) -> &'static str {
        "Strips source location metadata (loc(...)) from all operations in the module"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in &mut module.ops {
            if op.location != Location::Unknown {
                op.location = Location::Unknown;
                modified += 1;
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

/// 19. Print Op Stats Pass (`print-op-stats`)
/// Computes operation frequency, parameter counts, and FLOP estimates across the module.
pub struct PrintOpStatsPass;

impl Pass for PrintOpStatsPass {
    fn name(&self) -> &'static str {
        "print-op-stats"
    }

    fn description(&self) -> &'static str {
        "Computes and prints operation counts, parameter sizes, and FLOP estimates"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut counts: IndexMap<String, usize> = IndexMap::new();
        let mut total_params = 0i64;
        let mut est_flops = 0i64;

        for (_, op) in &module.ops {
            *counts.entry(op.name.to_string()).or_insert(0) += 1;
            if op.is_op("stablehlo", "constant") {
                if let Some(&res) = op.results.first() {
                    if let Some(t) = module.values[res].ty.as_tensor() {
                        total_params += t.num_elements().unwrap_or(0);
                    }
                }
            } else if op.is_op("stablehlo", "dot_general") && op.operands.len() == 2 {
                let lhs_ty = module.values[op.operands[0]].ty.as_tensor();
                let rhs_ty = module.values[op.operands[1]].ty.as_tensor();
                if let (Some(l), Some(r)) = (lhs_ty, rhs_ty) {
                    if let (Some(Dim::Static(m)), Some(Dim::Static(k)), Some(Dim::Static(n))) =
                        (l.shape.first(), l.shape.get(1), r.shape.get(1))
                    {
                        est_flops += 2 * m * k * n;
                    }
                }
            }
        }

        eprintln!(
            "[castforge:stats] Total Ops: {} | Constant Params: {} | Est. MatMul FLOPs: {}",
            module.ops.len(),
            total_params,
            est_flops
        );
        for (op_name, count) in &counts {
            eprintln!("  - {:<28} : {}", op_name, count);
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: false,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}
