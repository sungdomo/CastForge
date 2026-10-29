use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, Location, Module, OpId, OpName, Operation, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Cost-Guided E-Graph Algebraic Factoring & FMA Extraction Pass (`egraph-simplify`).
///
/// Performs two cost-guided rewrites:
/// 1. **Distributive Factoring**: Rewrites `a * b + a * c -> a * (b + c)` and
///    `a * b - a * c -> a * (b - c)` across all commutative operand orders, eliminating redundant
///    tensor multiplications.
/// 2. **Fused Multiply-Add Extraction (`castforge.fma`)**: When `fuse_fma = true`, fuses
///    `add(multiply(a, b), c)` into a single hardware FMA composite (`"castforge.fma"`).
pub struct EgraphSimplifyPass {
    pub fuse_fma: bool,
}

impl Default for EgraphSimplifyPass {
    fn default() -> Self {
        Self { fuse_fma: true }
    }
}

impl Pass for EgraphSimplifyPass {
    fn name(&self) -> &'static str {
        "egraph-simplify"
    }

    fn description(&self) -> &'static str {
        "Applies distributive algebraic factoring (a*b + a*c -> a*(b+c)) and FMA extraction"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            // Phase 1: Distributive factoring `a * b +/- a * c -> a * (b +/- c)`
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let (lhs, rhs, is_sub) = match StableHloOp::from_op(&op) {
                    Some(StableHloOp::Add { lhs, rhs, .. }) => (lhs, rhs, false),
                    Some(StableHloOp::Subtract { lhs, rhs, .. }) => (lhs, rhs, true),
                    _ => continue,
                };
                let Some((l0, l1)) = match_mul_operands(module, lhs) else {
                    continue;
                };
                let Some((r0, r1)) = match_mul_operands(module, rhs) else {
                    continue;
                };

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
                let Some((common, b_val, c_val)) = factored else {
                    continue;
                };

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
                created += 1;
                modified += 1;
            }

            // Phase 2: Fused Multiply-Add (`add(multiply(a, b), c) -> castforge.fma(a, b, c)`)
            if self.fuse_fma {
                let ops2 = module.block_ops(block_id);
                for op_id in ops2 {
                    let Some(op) = module.ops.get(op_id).cloned() else {
                        continue;
                    };
                    let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op) else {
                        continue;
                    };
                    let fma_match = match_mul_operands(module, lhs)
                        .map(|(a, b)| (a, b, rhs))
                        .or_else(|| match_mul_operands(module, rhs).map(|(a, b)| (a, b, lhs)));
                    let Some((a_val, b_val, c_val)) = fma_match else {
                        continue;
                    };

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
                    modified += 1;
                }
            }
        }

        if created > 0 || modified > 0 {
            let dce = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
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
    out_ty: castforge_ir::Type,
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
