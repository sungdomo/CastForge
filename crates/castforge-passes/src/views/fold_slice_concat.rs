use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Dim, Module, ValueDef};

/// Fold Slice & Concatenate Pass (`fold-slice-concat`).
///
/// Simplifies `stablehlo.slice` and `stablehlo.concatenate` patterns by:
/// - Eliminating single-input `concatenate([x]) -> x`.
/// - Composing nested unit-stride `slice(slice(x, s1, l1), s2, l2) -> slice(x, s1 + s2, s1 + l2)`.
/// - Folding `slice(concatenate(inputs, dim), start, limit)` when the slice extracts a region
///   that lies entirely within a single concatenated operand `inputs[k]`.
pub struct FoldSliceConcatPass;

impl Pass for FoldSliceConcatPass {
    fn name(&self) -> &'static str {
        "fold-slice-concat"
    }

    fn description(&self) -> &'static str {
        "Composes nested slices and folds slices of concatenate ops directly to source operands"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                match hlo {
                    StableHloOp::Concatenate { inputs, result, .. } => {
                        if inputs.len() == 1 {
                            module.replace_all_uses(result, inputs[0]);
                            module.remove_op(block_id, op_id);
                            removed += 1;
                        }
                    }
                    StableHloOp::Slice {
                        operand,
                        start_indices: mut starts,
                        limit_indices: mut limits,
                        strides,
                        result,
                    } => {
                        if !strides.iter().all(|&s| s == 1) {
                            continue;
                        }

                        let mut cur_operand = operand;

                        // 1. Compose nested `slice(slice(x, s1, l1, 1), s2, l2, 1)`
                        if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                            && let Some(prod_op) = module.ops.get(prod_id)
                            && let Some(StableHloOp::Slice {
                                operand: base_in,
                                start_indices: s1,
                                strides: st1,
                                ..
                            }) = StableHloOp::from_op(prod_op)
                            && st1.iter().all(|&s| s == 1)
                            && s1.len() == starts.len()
                        {
                            for i in 0..starts.len() {
                                starts[i] += s1[i];
                                limits[i] += s1[i];
                            }
                            cur_operand = base_in;
                            let op_mut = &mut module.ops[op_id];
                            op_mut.operands[0] = cur_operand;
                            op_mut.attributes.insert(
                                "start_indices".to_string(),
                                Attribute::DenseIntArray(starts.clone()),
                            );
                            op_mut.attributes.insert(
                                "limit_indices".to_string(),
                                Attribute::DenseIntArray(limits.clone()),
                            );
                            modified += 1;
                        }

                        // 2. Fold `slice(concatenate(inputs, dim))` if within a single input
                        if let ValueDef::OpResult { op: prod_id, .. } =
                            module.values[cur_operand].def
                            && let Some(prod_op) = module.ops.get(prod_id)
                            && let Some(StableHloOp::Concatenate {
                                inputs, dimension, ..
                            }) = StableHloOp::from_op(prod_op)
                        {
                            let d = dimension as usize;
                            if d < starts.len() {
                                let slice_start = starts[d];
                                let slice_limit = limits[d];
                                let mut offset = 0i64;
                                let mut matched_in = None;

                                for inp in inputs {
                                    let Some(t) = module.values[inp].ty.as_tensor() else {
                                        matched_in = None;
                                        break;
                                    };
                                    let Some(Dim::Static(dim_len)) = t.shape.get(d) else {
                                        matched_in = None;
                                        break;
                                    };
                                    let next_offset = offset + *dim_len;
                                    if slice_start >= offset && slice_limit <= next_offset {
                                        matched_in = Some((inp, offset, *dim_len));
                                        break;
                                    }
                                    offset = next_offset;
                                }

                                if let Some((target_in, in_offset, in_dim_len)) = matched_in {
                                    starts[d] -= in_offset;
                                    limits[d] -= in_offset;
                                    cur_operand = target_in;

                                    let is_full_target = starts.iter().all(|&s| s == 0)
                                        && limits[d] == in_dim_len
                                        && module.values[target_in].ty == module.values[result].ty;

                                    if is_full_target {
                                        module.replace_all_uses(result, target_in);
                                        module.remove_op(block_id, op_id);
                                        removed += 1;
                                        continue;
                                    }

                                    let op_mut = &mut module.ops[op_id];
                                    op_mut.operands[0] = cur_operand;
                                    op_mut.attributes.insert(
                                        "start_indices".to_string(),
                                        Attribute::DenseIntArray(starts),
                                    );
                                    op_mut.attributes.insert(
                                        "limit_indices".to_string(),
                                        Attribute::DenseIntArray(limits),
                                    );
                                    modified += 1;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
