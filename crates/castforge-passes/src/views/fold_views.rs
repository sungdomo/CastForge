//! Unified View, Shape, Layout & Type-Cast Folding Pass (`fold-views`).

use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

/// Unified View, Shape, Layout & Type-Cast Folding Pass (`fold-views`).
///
/// Consolidates `sink-transpose`, `fold-transpose`, `fold-reshape`, `fold-broadcast`,
/// `fold-slice-concat`, `fold-convert`, and `convert-layout` (`layout=nhwc|nchw|col_major`,
/// `propagate=true`) in a single pass.
#[derive(Debug, Clone, Default)]
pub struct FoldViewsPass {
    /// Optional target spatial/matrix layout (`"nhwc"`, `"nchw"`, or `"col_major"`).
    pub layout: Option<String>,
    /// Whether to propagate `data_layout` across downstream ops and fold inverse transposes.
    pub propagate: bool,
}

impl Pass for FoldViewsPass {
    fn name(&self) -> &'static str {
        "fold-views"
    }

    fn description(&self) -> &'static str {
        "Unified folding of transpose, reshape, broadcast, slice/concat, convert, and layout views"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        if self.layout.is_some() || self.propagate {
            let mut conv = ConvertLayoutPass {
                layout: self.layout.clone().unwrap_or_else(|| "nhwc".to_string()),
                propagate: self.propagate,
            };
            let mut r = conv.run(module)?;
            r.pass_name = self.name().to_string();
            return Ok(r);
        }

        let steps = [
            SinkTransposePass.run(module)?,
            FoldTransposePass {
                eliminate_identity: true,
            }
            .run(module)?,
            FoldReshapePass.run(module)?,
            FoldBroadcastPass.run(module)?,
            FoldSliceConcatPass.run(module)?,
            FoldConvertPass.run(module)?,
        ];
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}

/// Fold Transpose Pass (`transpose(transpose(x, p1), p2)` -> `transpose(x, p1[p2])`).
pub struct FoldTransposePass {
    pub eliminate_identity: bool,
}

impl Default for FoldTransposePass {
    fn default() -> Self {
        Self {
            eliminate_identity: true,
        }
    }
}

impl Pass for FoldTransposePass {
    fn name(&self) -> &'static str {
        "fold-transpose"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.transpose ops and eliminates identity transposes"
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
                let Some(StableHloOp::Transpose {
                    operand,
                    permutation: perm2,
                    result,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut current_operand = operand;
                let mut current_perm = perm2;

                if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def
                    && let Some(prod_op) = module.ops.get(prod_op_id)
                    && let Some(StableHloOp::Transpose {
                        operand: base_input,
                        permutation: perm1,
                        ..
                    }) = StableHloOp::from_op(prod_op)
                    && perm1.len() == current_perm.len()
                {
                    let composed: Vec<i64> = current_perm
                        .iter()
                        .map(|&idx| perm1[idx as usize])
                        .collect();
                    current_operand = base_input;
                    current_perm = composed;

                    let op_mut = &mut module.ops[op_id];
                    op_mut.operands[0] = current_operand;
                    op_mut.attributes.insert(
                        "permutation".to_string(),
                        Attribute::DenseIntArray(current_perm.clone()),
                    );
                    modified += 1;
                }

                let is_identity = self.eliminate_identity
                    && !current_perm.is_empty()
                    && current_perm.iter().enumerate().all(|(i, &p)| p == i as i64);

                if is_identity {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
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

/// Fold Reshape Pass (`reshape(reshape(x))` -> `reshape(x)` & identity reshape removal).
pub struct FoldReshapePass;

impl Pass for FoldReshapePass {
    fn name(&self) -> &'static str {
        "fold-reshape"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.reshape ops and eliminates identity reshapes"
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
                let Some(StableHloOp::Reshape { operand, result }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut current_operand = operand;
                if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def
                    && let Some(prod_op) = module.ops.get(prod_op_id)
                    && let Some(StableHloOp::Reshape {
                        operand: base_input,
                        ..
                    }) = StableHloOp::from_op(prod_op)
                {
                    current_operand = base_input;
                    module.ops[op_id].operands[0] = current_operand;
                    modified += 1;
                }

                if module.values[current_operand].ty == module.values[result].ty {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
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

/// Fold Broadcast Pass (`fold-broadcast`).
pub struct FoldBroadcastPass;

impl Pass for FoldBroadcastPass {
    fn name(&self) -> &'static str {
        "fold-broadcast"
    }

    fn description(&self) -> &'static str {
        "Composes chained broadcast_in_dim ops, removes identity broadcasts, and hoists unary ops"
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
                    StableHloOp::BroadcastInDim {
                        operand,
                        broadcast_dimensions: dims2,
                        result,
                    } => {
                        let mut cur_operand = operand;
                        let mut cur_dims = dims2;

                        if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                            && let Some(prod_op) = module.ops.get(prod_id)
                            && let Some(StableHloOp::BroadcastInDim {
                                operand: base_in,
                                broadcast_dimensions: dims1,
                                ..
                            }) = StableHloOp::from_op(prod_op)
                            && dims1.iter().all(|&d| (d as usize) < cur_dims.len())
                        {
                            let composed: Vec<i64> =
                                dims1.iter().map(|&d| cur_dims[d as usize]).collect();
                            cur_operand = base_in;
                            cur_dims = composed;

                            let op_mut = &mut module.ops[op_id];
                            op_mut.operands[0] = cur_operand;
                            op_mut.attributes.insert(
                                "broadcast_dimensions".to_string(),
                                Attribute::DenseIntArray(cur_dims.clone()),
                            );
                            modified += 1;
                        }

                        if let ValueDef::OpResult { op: prod_id, .. } =
                            module.values[cur_operand].def
                            && let Some(prod_op) = module.ops.get(prod_id)
                            && prod_op.is_op("stablehlo", "constant")
                            && let Some(Attribute::DenseElements { raw_literal, .. }) =
                                prod_op.attributes.get("value")
                            && !raw_literal.trim().starts_with('[')
                            && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
                        {
                            let lit = raw_literal.clone();
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = OpName::new("stablehlo", "constant");
                            op_mut.operands.clear();
                            op_mut.attributes.clear();
                            op_mut.attributes.insert(
                                "value".to_string(),
                                Attribute::DenseElements {
                                    ty: res_ty,
                                    raw_literal: lit,
                                },
                            );
                            modified += 1;
                            continue;
                        }

                        let is_identity = module.values[cur_operand].ty == module.values[result].ty
                            && cur_dims.iter().enumerate().all(|(i, &d)| d == i as i64);
                        if is_identity {
                            module.replace_all_uses(result, cur_operand);
                            module.remove_op(block_id, op_id);
                            removed += 1;
                        }
                    }
                    StableHloOp::Exp { operand, result }
                    | StableHloOp::Tanh { operand, result }
                    | StableHloOp::Rsqrt { operand, result } => {
                        if module.use_count(operand) != 1 {
                            continue;
                        }
                        let ValueDef::OpResult { op: bcast_id, .. } = module.values[operand].def
                        else {
                            continue;
                        };
                        let Some(bcast_op) = module.ops.get(bcast_id) else {
                            continue;
                        };
                        let Some(StableHloOp::BroadcastInDim {
                            operand: small_x,
                            broadcast_dimensions,
                            ..
                        }) = StableHloOp::from_op(bcast_op)
                        else {
                            continue;
                        };
                        let unary_name = module.ops[op_id].name.clone();
                        let small_ty = module.values[small_x].ty.clone();
                        let big_ty = module.values[result].ty.clone();
                        if small_ty == big_ty {
                            continue;
                        }

                        let bcast_mut = &mut module.ops[bcast_id];
                        bcast_mut.name = unary_name;
                        bcast_mut.operands = SmallVec::from_slice(&[small_x]);
                        bcast_mut.attributes.clear();
                        module.values[operand].ty = small_ty;

                        let unary_mut = &mut module.ops[op_id];
                        unary_mut.name = OpName::new("stablehlo", "broadcast_in_dim");
                        unary_mut.operands = SmallVec::from_slice(&[operand]);
                        unary_mut.attributes.clear();
                        unary_mut.attributes.insert(
                            "broadcast_dimensions".to_string(),
                            Attribute::DenseIntArray(broadcast_dimensions),
                        );
                        modified += 1;
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

/// Fold Slice & Concatenate Pass (`fold-slice-concat`).
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

/// Fold Convert Pass (`fold-convert`).
pub struct FoldConvertPass;

fn is_float_dtype(dt: &DType) -> bool {
    matches!(
        dt,
        DType::F8E4M3FN | DType::F8E5M2 | DType::F16 | DType::BF16 | DType::F32 | DType::F64
    )
}

fn is_lossless_intermediate(t0: &DType, t1: &DType, t2: &DType) -> bool {
    if t0 == t1 || t1 == t2 {
        return true;
    }
    if is_float_dtype(t0)
        && is_float_dtype(t1)
        && is_float_dtype(t2)
        && t1.bit_width() >= t0.bit_width()
        && t1.bit_width() >= t2.bit_width()
    {
        return true;
    }
    let is_sint = |d: &DType| {
        matches!(
            d,
            DType::I4 | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    };
    is_sint(t0)
        && is_sint(t1)
        && is_sint(t2)
        && t1.bit_width() >= t0.bit_width()
        && t1.bit_width() >= t2.bit_width()
}

impl Pass for FoldConvertPass {
    fn name(&self) -> &'static str {
        "fold-convert"
    }

    fn description(&self) -> &'static str {
        "Eliminates redundant or round-trip stablehlo.convert casts and folds constant casts"
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
                let Some(StableHloOp::Convert { operand, result }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut cur_operand = operand;

                if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                    && let Some(prod_op) = module.ops.get(prod_id)
                    && let Some(StableHloOp::Convert {
                        operand: base_in, ..
                    }) = StableHloOp::from_op(prod_op)
                    && let (Some(t0), Some(t1), Some(t2)) = (
                        module.values[base_in].ty.as_tensor().map(|t| &t.dtype),
                        module.values[operand].ty.as_tensor().map(|t| &t.dtype),
                        module.values[result].ty.as_tensor().map(|t| &t.dtype),
                    )
                    && is_lossless_intermediate(t0, t1, t2)
                {
                    cur_operand = base_in;
                    module.ops[op_id].operands[0] = cur_operand;
                    modified += 1;
                }

                if module.values[cur_operand].ty == module.values[result].ty {
                    module.replace_all_uses(result, cur_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                    continue;
                }

                if let ValueDef::OpResult { op: prod_id, .. } = module.values[cur_operand].def
                    && let Some(prod_op) = module.ops.get(prod_id)
                    && prod_op.is_op("stablehlo", "constant")
                    && let Some(Attribute::DenseElements { raw_literal, .. }) =
                        prod_op.attributes.get("value")
                    && !raw_literal.trim().starts_with('[')
                    && let Some(res_ty) = module.values[result].ty.as_tensor().cloned()
                {
                    let lit = raw_literal.clone();
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "constant");
                    op_mut.operands.clear();
                    op_mut.attributes.clear();
                    op_mut.attributes.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty: res_ty,
                            raw_literal: lit,
                        },
                    );
                    modified += 1;
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

/// Sinks `stablehlo.transpose` through unary and binary elementwise operations toward producers.
pub struct SinkTransposePass;

impl Pass for SinkTransposePass {
    fn name(&self) -> &'static str {
        "sink-transpose"
    }

    fn description(&self) -> &'static str {
        "Sinks stablehlo.transpose through elementwise ops toward constants and producers"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            loop {
                let mut iter_changed = false;
                let ops = module.block_ops(block_id);

                for op_id in ops {
                    let Some(op) = module.ops.get(op_id) else {
                        continue;
                    };
                    let Some(StableHloOp::Transpose {
                        operand: elem_val,
                        permutation: perm,
                        result: trans_res,
                    }) = StableHloOp::from_op(op)
                    else {
                        continue;
                    };

                    let ValueDef::OpResult { op: prod_op_id, .. } = module.values[elem_val].def
                    else {
                        continue;
                    };
                    if module.use_count(elem_val) != 1 {
                        continue;
                    }
                    let Some(prod_op) = module.ops.get(prod_op_id).cloned() else {
                        continue;
                    };
                    let Some(prod_hlo) = StableHloOp::from_op(&prod_op) else {
                        continue;
                    };

                    let trans_ty = module.values[trans_res].ty.clone();

                    match prod_hlo {
                        StableHloOp::Exp { operand: x, .. }
                        | StableHloOp::Tanh { operand: x, .. }
                        | StableHloOp::Rsqrt { operand: x, .. }
                        | StableHloOp::Convert { operand: x, .. } => {
                            let tx = insert_transpose_before(
                                module,
                                block_id,
                                op_id,
                                x,
                                &perm,
                                trans_ty.clone(),
                            );
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = prod_op.name.clone();
                            op_mut.operands = SmallVec::from_slice(&[tx]);
                            op_mut.attributes = prod_op.attributes.clone();
                            module.remove_op(block_id, prod_op_id);
                            created += 1;
                            modified += 1;
                            removed += 1;
                            iter_changed = true;
                            break;
                        }
                        StableHloOp::Add { lhs, rhs, .. }
                        | StableHloOp::Subtract { lhs, rhs, .. }
                        | StableHloOp::Multiply { lhs, rhs, .. }
                        | StableHloOp::Divide { lhs, rhs, .. }
                        | StableHloOp::Maximum { lhs, rhs, .. }
                        | StableHloOp::Minimum { lhs, rhs, .. } => {
                            let tlhs = insert_transpose_before(
                                module,
                                block_id,
                                op_id,
                                lhs,
                                &perm,
                                trans_ty.clone(),
                            );
                            let trhs = insert_transpose_before(
                                module,
                                block_id,
                                op_id,
                                rhs,
                                &perm,
                                trans_ty.clone(),
                            );
                            let op_mut = &mut module.ops[op_id];
                            op_mut.name = prod_op.name.clone();
                            op_mut.operands = SmallVec::from_slice(&[tlhs, trhs]);
                            op_mut.attributes = prod_op.attributes.clone();
                            module.remove_op(block_id, prod_op_id);
                            created += 2;
                            modified += 1;
                            removed += 1;
                            iter_changed = true;
                            break;
                        }
                        _ => {}
                    }
                }

                if !iter_changed {
                    break;
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

fn insert_transpose_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input: ValueId,
    perm: &[i64],
    out_ty: castforge_ir::Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "permutation".to_string(),
        Attribute::DenseIntArray(perm.to_vec()),
    );
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "transpose"),
        operands: SmallVec::from_slice(&[input]),
        results: SmallVec::new(),
        attributes: attrs,
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

/// Spatial & Matrix Layout Converter & Global Layout Propagation Pass (`convert-layout`).
#[derive(Debug, Clone)]
pub struct ConvertLayoutPass {
    /// Target layout (`"nhwc"`, `"nchw"`, or `"col_major"`).
    pub layout: String,
    /// Whether to propagate `data_layout` across downstream ops and cancel inverse transposes.
    pub propagate: bool,
}

impl Default for ConvertLayoutPass {
    fn default() -> Self {
        Self {
            layout: "nhwc".to_string(),
            propagate: false,
        }
    }
}

impl Pass for ConvertLayoutPass {
    fn name(&self) -> &'static str {
        "convert-layout"
    }

    fn description(&self) -> &'static str {
        "Inserts layout-conversion transposes (NCHW<->NHWC / row<->col) and propagates layout"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let tgt_layout = if self.layout.trim().is_empty() {
            "nhwc".to_string()
        } else {
            self.layout.trim().to_ascii_lowercase()
        };
        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "convolution") && !op.is_op("stablehlo", "dot_general") {
                    continue;
                }
                if op.attributes.get("data_layout").and_then(|a| a.as_str()) == Some(&tgt_layout) {
                    continue;
                }
                let Some(&in_val) = op.operands.first() else {
                    continue;
                };
                let Some(in_tensor) = module.values[in_val].ty.as_tensor().cloned() else {
                    continue;
                };

                module.ops[op_id].attributes.insert(
                    "data_layout".to_string(),
                    Attribute::String(tgt_layout.clone()),
                );
                if self.propagate {
                    module.ops[op_id].attributes.insert(
                        "castforge.layout_propagated".to_string(),
                        Attribute::Bool(true),
                    );
                }
                modified += 1;

                if in_tensor.shape.len() == 2 {
                    let mid_ty = Type::Tensor(TensorType {
                        shape: vec![in_tensor.shape[1].clone(), in_tensor.shape[0].clone()],
                        dtype: in_tensor.dtype.clone(),
                        encoding: None,
                    });
                    let orig_ty = Type::Tensor(in_tensor);
                    let mut p_attrs = IndexMap::new();
                    p_attrs.insert(
                        "permutation".to_string(),
                        Attribute::DenseIntArray(vec![1, 0]),
                    );

                    let t1_id = module.add_operation(
                        block_id,
                        OpName::new("stablehlo", "transpose"),
                        smallvec![in_val],
                        vec![(Some("layout_in".to_string()), mid_ty)],
                        p_attrs.clone(),
                        Vec::new(),
                        true,
                        op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    let t1_res = module.ops[t1_id].results[0];

                    let t2_id = module.add_operation(
                        block_id,
                        OpName::new("stablehlo", "transpose"),
                        smallvec![t1_res],
                        vec![(Some("layout_restored".to_string()), orig_ty)],
                        p_attrs,
                        Vec::new(),
                        true,
                        op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    let t2_res = module.ops[t2_id].results[0];

                    module.ops[op_id].operands[0] = t2_res;
                    if let Some(pos) = module.blocks[block_id]
                        .ops
                        .iter()
                        .position(|&id| id == op_id)
                    {
                        module.blocks[block_id].ops.splice(pos..pos, [t1_id, t2_id]);
                    }
                    created += 2;
                }
            }
        }

        if self.propagate {
            for fn_id in module.functions() {
                let reg_id = module.ops[fn_id].regions[0];
                let block_id = module.regions[reg_id].blocks[0];
                let ops = module.blocks[block_id].ops.clone();
                for op_id in ops {
                    let Some(op) = module.ops.get(op_id).cloned() else {
                        continue;
                    };
                    if op.name.dialect != "stablehlo" || op.is_op("stablehlo", "constant") {
                        continue;
                    }
                    let has_layout_input = op.operands.iter().any(|&opnd| {
                        if let ValueDef::OpResult { op: def_op, .. } = module.values[opnd].def {
                            module
                                .ops
                                .get(def_op)
                                .is_some_and(|d| d.attributes.contains_key("data_layout"))
                        } else {
                            false
                        }
                    });
                    if has_layout_input && !op.attributes.contains_key("data_layout") {
                        module.ops[op_id].attributes.insert(
                            "data_layout".to_string(),
                            Attribute::String(tgt_layout.clone()),
                        );
                        module.ops[op_id].attributes.insert(
                            "castforge.layout_propagated".to_string(),
                            Attribute::Bool(true),
                        );
                        modified += 1;
                    }
                }
            }
            let sink_res = SinkTransposePass.run(module)?;
            let fold_res = FoldTransposePass::default().run(module)?;
            modified += sink_res.ops_modified + fold_res.ops_modified;
            removed += sink_res.ops_removed + fold_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || created > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}
