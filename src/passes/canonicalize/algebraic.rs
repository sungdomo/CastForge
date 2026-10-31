//! Unified Algebraic Simplification, Constant Folding, Conditional Propagation (SCCP),
//! and E-Graph Factoring/FMA Pass (`simplify-algebraic`), executed in a single forward
//! dataflow traversal.

use super::constants::{ConstFoldOutcome, FoldConstantsPass, try_fold_constant_op};
use crate::ir::{
    Attribute, BlockId, Dim, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashSet;

/// Unified Algebraic Simplification, Constant Folding & E-Graph Factoring/FMA Pass
/// (`simplify-algebraic`).
pub struct SimplifyAlgebraicPass {
    pub fold_constants: bool,
    pub include_conditional: bool,
    pub egraph: bool,
    pub fuse_fma: bool,
}

impl Default for SimplifyAlgebraicPass {
    fn default() -> Self {
        Self {
            fold_constants: true,
            include_conditional: true,
            egraph: false,
            fuse_fma: true,
        }
    }
}

impl Pass for SimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Unified constant folding, SCCP, algebraic identities, e-graph factoring & FMA extraction"
    }

    fn child_passes(&self) -> &'static [&'static str] {
        &["fold-constants", "fuse-fma"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            let mut dead_ops: HashSet<OpId> = HashSet::new();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if StableHloOp::from_op(op).is_none() {
                    continue;
                }

                // 1. Constant folding & SCCP (`compare` / `select` + arithmetic/views)
                if self.fold_constants {
                    match try_fold_constant_op(module, op_id, self.include_conditional) {
                        ConstFoldOutcome::Modified => {
                            modified += 1;
                            continue;
                        }
                        ConstFoldOutcome::Replaced => {
                            dead_ops.insert(op_id);
                            removed += 1;
                            continue;
                        }
                        ConstFoldOutcome::Unchanged => {}
                    }
                }

                // 2. Algebraic identities, commutative canonicalization, and reassociation
                match try_simplify_algebraic_op(module, block_id, op_id) {
                    AlgOutcome::ModifiedAndDone(c) => {
                        created += c;
                        modified += 1;
                        continue;
                    }
                    AlgOutcome::Replaced(swapped_before) => {
                        if swapped_before {
                            modified += 1;
                        }
                        dead_ops.insert(op_id);
                        removed += 1;
                        continue;
                    }
                    AlgOutcome::SwappedOnly => {
                        modified += 1;
                    }
                    AlgOutcome::Unchanged => {}
                }

                // 3. E-Graph distributive factoring (`a*b +/- a*c -> a*(b +/- c)`) & FMA
                if self.egraph {
                    let (c, m) = try_egraph_and_fma_op(module, block_id, op_id, self.fuse_fma);
                    created += c;
                    modified += m;
                }
            }

            if !dead_ops.is_empty() {
                module.blocks[block_id]
                    .ops
                    .retain(|id| !dead_ops.contains(id));
                for dead_id in dead_ops {
                    module.ops.remove(dead_id);
                }
            }
        }

        if created > 0 || modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

enum AlgOutcome {
    Unchanged,
    SwappedOnly,
    ModifiedAndDone(usize),
    Replaced(bool),
}

fn try_simplify_algebraic_op(module: &mut Module, block_id: BlockId, op_id: OpId) -> AlgOutcome {
    let Some(op) = module.ops.get(op_id) else {
        return AlgOutcome::Unchanged;
    };
    let Some(mut hlo) = StableHloOp::from_op(op) else {
        return AlgOutcome::Unchanged;
    };

    let mut swapped = false;
    match &mut hlo {
        StableHloOp::Add { lhs, rhs, .. }
        | StableHloOp::Multiply { lhs, rhs, .. }
        | StableHloOp::Maximum { lhs, rhs, .. }
        | StableHloOp::Minimum { lhs, rhs, .. }
            if is_constant(module, *lhs) && !is_constant(module, *rhs) =>
        {
            std::mem::swap(lhs, rhs);
            module.ops[op_id].operands.swap(0, 1);
            swapped = true;
        }
        _ => {}
    }

    // Reassociation of splat constants: `(x + c1) + c2 -> x + (c1 + c2)`
    // and `(x * c1) * c2 -> x * (c1 * c2)`
    match hlo {
        StableHloOp::Add { lhs, rhs, .. } => {
            if let Some(c2) = FoldConstantsPass::const_splat(module, rhs)
                && let ValueDef::OpResult { op: lhs_op_id, .. } = module.values[lhs].def
                && let Some(lhs_op) = module.ops.get(lhs_op_id)
                && let Some(StableHloOp::Add {
                    lhs: x,
                    rhs: c1_val,
                    ..
                }) = StableHloOp::from_op(lhs_op)
                && let Some(c1) = FoldConstantsPass::const_splat(module, c1_val)
                && let ValueDef::OpResult { op: c2_op_id, .. } = module.values[rhs].def
                && let Some(c2_ty) = module.values[rhs].ty.as_tensor().cloned()
            {
                let sum_lit = format_splat(c1 + c2);
                let created = update_or_clone_reassoc_const(
                    module, block_id, op_id, lhs, c1_val, rhs, c2_op_id, x, c2_ty, sum_lit,
                );
                return AlgOutcome::ModifiedAndDone(created);
            }
        }
        StableHloOp::Multiply { lhs, rhs, .. } => {
            if let Some(c2) = FoldConstantsPass::const_splat(module, rhs)
                && let ValueDef::OpResult { op: lhs_op_id, .. } = module.values[lhs].def
                && let Some(lhs_op) = module.ops.get(lhs_op_id)
                && let Some(StableHloOp::Multiply {
                    lhs: x,
                    rhs: c1_val,
                    ..
                }) = StableHloOp::from_op(lhs_op)
                && let Some(c1) = FoldConstantsPass::const_splat(module, c1_val)
                && let ValueDef::OpResult { op: c2_op_id, .. } = module.values[rhs].def
                && let Some(c2_ty) = module.values[rhs].ty.as_tensor().cloned()
            {
                let prod_lit = format_splat(c1 * c2);
                let created = update_or_clone_reassoc_const(
                    module, block_id, op_id, lhs, c1_val, rhs, c2_op_id, x, c2_ty, prod_lit,
                );
                return AlgOutcome::ModifiedAndDone(created);
            }
        }
        _ => {}
    }

    // Self-subtraction `subtract(x, x) -> constant 0`
    if let StableHloOp::Subtract { lhs, rhs, result } = hlo
        && lhs == rhs
        && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
    {
        let zero_lit = if res_ty.dtype.is_float() { "0.0" } else { "0" };
        let op_mut = &mut module.ops[op_id];
        op_mut.name = OpName::new("stablehlo", "constant");
        op_mut.operands.clear();
        op_mut.attributes.clear();
        op_mut.attributes.insert(
            "value".to_string(),
            Attribute::DenseElements {
                ty: res_ty,
                raw_literal: zero_lit.to_string(),
            },
        );
        return AlgOutcome::ModifiedAndDone(0);
    }

    let replacement: Option<(ValueId, ValueId)> = match hlo {
        StableHloOp::Add { lhs, rhs, result } => {
            if is_splat_zero(module, rhs) {
                Some((result, lhs))
            } else if is_splat_zero(module, lhs) {
                Some((result, rhs))
            } else {
                None
            }
        }
        StableHloOp::Subtract { lhs, rhs, result } => {
            if is_splat_zero(module, rhs) {
                Some((result, lhs))
            } else {
                None
            }
        }
        StableHloOp::Multiply { lhs, rhs, result } => {
            if is_splat_one(module, rhs) || is_splat_zero(module, lhs) {
                Some((result, lhs))
            } else if is_splat_one(module, lhs) || is_splat_zero(module, rhs) {
                Some((result, rhs))
            } else {
                None
            }
        }
        StableHloOp::Divide { lhs, rhs, result } => {
            if is_splat_one(module, rhs) {
                Some((result, lhs))
            } else {
                None
            }
        }
        StableHloOp::Maximum { lhs, rhs, result } | StableHloOp::Minimum { lhs, rhs, result } => {
            if lhs == rhs {
                Some((result, lhs))
            } else {
                None
            }
        }
        StableHloOp::Select {
            on_true,
            on_false,
            result,
            ..
        } => {
            if on_true == on_false {
                Some((result, on_true))
            } else {
                None
            }
        }
        StableHloOp::Convert { operand, result } => {
            if module.values[operand].ty == module.values[result].ty {
                Some((result, operand))
            } else {
                None
            }
        }
        StableHloOp::Slice {
            operand,
            start_indices,
            limit_indices,
            strides,
            result,
        } => {
            let is_full = module.values[operand].ty == module.values[result].ty
                && start_indices.iter().all(|&s| s == 0)
                && strides.iter().all(|&s| s == 1)
                && module.values[operand].ty.as_tensor().is_some_and(|t| {
                    t.shape.len() == limit_indices.len()
                        && t.shape
                            .iter()
                            .zip(limit_indices.iter())
                            .all(|(d, &lim)| *d == Dim::Static(lim))
                });
            if is_full {
                Some((result, operand))
            } else {
                None
            }
        }
        _ => None,
    };

    if let Some((res, rep)) = replacement {
        module.replace_all_uses(res, rep);
        return AlgOutcome::Replaced(swapped);
    }

    if swapped {
        AlgOutcome::SwappedOnly
    } else {
        AlgOutcome::Unchanged
    }
}

fn try_egraph_and_fma_op(
    module: &mut Module,
    block_id: BlockId,
    op_id: OpId,
    fuse_fma: bool,
) -> (usize, usize) {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return (0, 0);
    };
    let (lhs, rhs, is_sub) = match StableHloOp::from_op(&op) {
        Some(StableHloOp::Add { lhs, rhs, .. }) => (lhs, rhs, false),
        Some(StableHloOp::Subtract { lhs, rhs, .. }) => (lhs, rhs, true),
        _ => return (0, 0),
    };

    if let (Some((l0, l1)), Some((r0, r1))) = (
        match_mul_operands(module, lhs),
        match_mul_operands(module, rhs),
    ) {
        let factored = if l0 == r0 {
            Some((l0, l1, r1))
        } else if l0 == r1 {
            Some((l0, l1, r0))
        } else if l1 == r0 {
            Some((l1, l0, r1))
        } else if l1 == r1 {
            Some((l1, l0, r0))
        } else {
            None
        };
        if let Some((common, b_val, c_val)) = factored {
            let out_ty = module.values[op.results[0]].ty.clone();
            let inner_op_name = if is_sub {
                OpName::new("stablehlo", "subtract")
            } else {
                OpName::new("stablehlo", "add")
            };
            let bc_val = insert_op_before(
                module,
                block_id,
                op_id,
                inner_op_name,
                &[b_val, c_val],
                out_ty,
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "multiply");
            op_mut.operands = SmallVec::from_slice(&[common, bc_val]);
            op_mut.attributes.clear();
            return (1, 1);
        }
    }

    if fuse_fma && !is_sub {
        let fma_match = match_mul_operands(module, lhs)
            .map(|(a, b)| (a, b, rhs))
            .or_else(|| match_mul_operands(module, rhs).map(|(a, b)| (a, b, lhs)));
        if let Some((a_val, b_val, c_val)) = fma_match {
            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fma".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fma_impl".to_string()),
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[a_val, b_val, c_val]);
            op_mut.attributes = attrs;
            return (0, 1);
        }
    }

    (0, 0)
}

fn const_attr(module: &Module, val_id: ValueId) -> Option<&Attribute> {
    if let ValueDef::OpResult { op, .. } = module.values[val_id].def {
        let op_node = module.ops.get(op)?;
        if op_node.is_op("stablehlo", "constant") {
            return op_node.attributes.get("value");
        }
    }
    None
}

fn is_constant(module: &Module, val_id: ValueId) -> bool {
    const_attr(module, val_id).is_some()
}

fn is_splat_zero(module: &Module, val_id: ValueId) -> bool {
    const_attr(module, val_id).is_some_and(|a| a.is_splat_zero())
        || FoldConstantsPass::const_splat(module, val_id).is_some_and(|v| v == 0.0)
}

fn is_splat_one(module: &Module, val_id: ValueId) -> bool {
    const_attr(module, val_id).is_some_and(|a| a.is_splat_one())
        || FoldConstantsPass::const_splat(module, val_id).is_some_and(|v| (v - 1.0).abs() < 1e-9)
}

#[allow(clippy::too_many_arguments)]
fn update_or_clone_reassoc_const(
    module: &mut Module,
    block_id: BlockId,
    op_id: OpId,
    lhs: ValueId,
    c1_val: ValueId,
    rhs: ValueId,
    c2_op_id: OpId,
    x: ValueId,
    c2_ty: crate::ir::TensorType,
    raw_literal: String,
) -> usize {
    let rhs_uses = module.use_count(rhs);
    let can_mutate_c2 = (c1_val != rhs && rhs_uses == 1)
        || (c1_val == rhs && rhs_uses == 2 && module.use_count(lhs) == 1);
    if can_mutate_c2 {
        if let Some(ref ssa) = module.values[rhs].ssa_name {
            module.tensor_store.remove(ssa);
        }
        module.ops[c2_op_id].attributes.insert(
            "value".to_string(),
            Attribute::DenseElements {
                ty: c2_ty,
                raw_literal,
            },
        );
        module.ops[op_id].operands[0] = x;
        0
    } else {
        let new_c_val = insert_op_before(
            module,
            block_id,
            op_id,
            OpName::new("stablehlo", "constant"),
            &[],
            crate::ir::Type::Tensor(c2_ty.clone()),
        );
        if let ValueDef::OpResult { op: new_c_op, .. } = module.values[new_c_val].def {
            module.ops[new_c_op].attributes.insert(
                "value".to_string(),
                Attribute::DenseElements {
                    ty: c2_ty,
                    raw_literal,
                },
            );
        }
        module.ops[op_id].operands = SmallVec::from_slice(&[x, new_c_val]);
        1
    }
}

fn format_splat(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{:.1}", v.round())
    } else {
        format!("{v}")
    }
}

fn match_mul_operands(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: mul_id, .. } = module.values[val_id].def else {
        return None;
    };
    let mul_op = module.ops.get(mul_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(mul_op)? else {
        return None;
    };
    Some((lhs, rhs))
}

fn insert_op_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    out_ty: crate::ir::Type,
) -> ValueId {
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
        operands: SmallVec::from_slice(operands),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: true,
        location: Location::Unknown,
    });
    let val_id = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: out_ty,
        def: ValueDef::OpResult {
            op: new_op_id,
            index: 0,
        },
    });
    module.ops[new_op_id].results.push(val_id);

    let block_ops = &mut module.blocks[block_id].ops;
    if let Some(pos) = block_ops.iter().position(|&id| id == before_op) {
        block_ops.insert(pos, new_op_id);
    } else {
        block_ops.push(new_op_id);
    }
    val_id
}
