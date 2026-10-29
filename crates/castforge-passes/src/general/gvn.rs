use crate::{
    CommonSubexpressionEliminationPass, EgraphSimplifyPass, HoistConstantsPass, Pass, PassResult,
    SimplifyAlgebraicPass,
};
use anyhow::Result;
use castforge_ir::{BlockId, Module, Operation, RegionId, Type, ValueId};
use std::collections::HashMap;

/// Unified Common Subexpression Elimination, Dominator GVN & Constant Hoisting Pass (`cse`).
pub struct UnifiedCsePass {
    pub gvn: bool,
    pub hoist_constants: bool,
}

impl Default for UnifiedCsePass {
    fn default() -> Self {
        Self {
            gvn: true,
            hoist_constants: false,
        }
    }
}

impl Pass for UnifiedCsePass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Unified CSE, cross-region Dominator GVN, and constant hoisting/deduplication"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.hoist_constants {
            steps.push(HoistConstantsPass.run(module)?);
        }
        if self.gvn {
            steps.push(GlobalValueNumberingPass.run(module)?);
        } else {
            steps.push(CommonSubexpressionEliminationPass.run(module)?);
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Unified Algebraic Simplification & E-Graph Factoring/FMA Pass (`simplify-algebraic`).
pub struct UnifiedSimplifyAlgebraicPass {
    pub egraph: bool,
    pub fuse_fma: bool,
}

impl Default for UnifiedSimplifyAlgebraicPass {
    fn default() -> Self {
        Self {
            egraph: false,
            fuse_fma: true,
        }
    }
}

impl Pass for UnifiedSimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Unified algebraic identities, reassociation, e-graph factoring, and FMA extraction"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = vec![SimplifyAlgebraicPass.run(module)?];
        if self.egraph {
            steps.push(
                EgraphSimplifyPass {
                    fuse_fma: self.fuse_fma,
                }
                .run(module)?,
            );
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Global Value Numbering Pass (`gvn`).
///
/// Extends single-block CSE to a dominator-tree / cross-region Global Value Numbering pass:
/// - Propagates canonical value numbers from outer regions into nested regions and basic blocks.
/// - Canonicalizes commutative binary operations (`add`, `multiply`, `maximum`, `minimum`, `and`,
///   `or`, `xor`) and attribute ordering.
#[derive(Default)]
pub struct GlobalValueNumberingPass;

impl Pass for GlobalValueNumberingPass {
    fn name(&self) -> &'static str {
        "gvn"
    }

    fn description(&self) -> &'static str {
        "Dominator-aware Global Value Numbering across nested regions and basic blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut value_leader: HashMap<ValueId, ValueId> = HashMap::new();
        for fn_id in module.functions() {
            let regions: Vec<RegionId> = module.ops[fn_id].regions.clone();
            let mut env: HashMap<String, ValueId> = HashMap::new();
            for reg_id in regions {
                removed += gvn_region(module, reg_id, &mut env, &mut value_leader);
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

fn gvn_region(
    module: &mut Module,
    region_id: RegionId,
    outer_env: &mut HashMap<String, ValueId>,
    value_leader: &mut HashMap<ValueId, ValueId>,
) -> usize {
    let mut removed = 0usize;
    let Some(reg) = module.regions.get(region_id) else {
        return 0;
    };
    let blocks: Vec<BlockId> = reg.blocks.clone();
    for block_id in blocks {
        let mut block_env = outer_env.clone();
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            for &nested_reg in &op.regions {
                removed += gvn_region(module, nested_reg, &mut block_env, value_leader);
            }
            if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                continue;
            }
            let res_val = op.results[0];
            let res_ty = module.values[res_val].ty.clone();
            let sig = gvn_signature(&op, &res_ty, value_leader);
            if let Some(&leader_val) = block_env.get(&sig)
                && leader_val != res_val
            {
                value_leader.insert(res_val, leader_val);
                module.replace_all_uses(res_val, leader_val);
                module.remove_op(block_id, op_id);
                removed += 1;
            } else {
                value_leader.insert(res_val, res_val);
                block_env.insert(sig, res_val);
            }
        }
        *outer_env = block_env;
    }
    removed
}

fn gvn_signature(
    op: &Operation,
    res_ty: &Type,
    value_leader: &HashMap<ValueId, ValueId>,
) -> String {
    let mut ops_vec: Vec<String> = op
        .operands
        .iter()
        .map(|v| {
            let canon = value_leader.get(v).copied().unwrap_or(*v);
            format!("{canon:?}")
        })
        .collect();
    let is_comm = op.name.dialect == "stablehlo"
        && op.operands.len() == 2
        && matches!(
            op.name.name.as_str(),
            "add" | "multiply" | "maximum" | "minimum" | "and" | "or" | "xor"
        );
    if is_comm {
        ops_vec.sort();
    }
    let mut attrs: Vec<(&String, String)> = op
        .attributes
        .iter()
        .filter(|(k, _)| !k.starts_with("castforge.") && *k != "inplace_operand")
        .map(|(k, v)| (k, format!("{v:?}")))
        .collect();
    attrs.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{}|operands:{:?}|attrs:{:?}|ty:{}",
        op.name, ops_vec, attrs, res_ty
    )
}
