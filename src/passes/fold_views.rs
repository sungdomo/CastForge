//! Unified View, Shape, Layout & Type-Cast Folding Pass (`fold-views`), executed in a single
//! forward block traversal after transpose sinking.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use crate::passes::{DeadCodeEliminationPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};
use std::collections::HashSet;

/// Unified View, Shape, Layout & Type-Cast Folding Pass (`fold-views`).
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

    fn child_passes(&self) -> &'static [&'static str] {
        &["convert-layout"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        if self.layout.is_some() || self.propagate {
            let tgt_layout = self
                .layout
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("nhwc")
                .to_string();
            return Ok(ConvertLayoutPass {
                layout: tgt_layout,
                propagate: self.propagate,
            }
            .run(module)?
            .with_name(self.name()));
        }

        let (created, modified, removed) = fold_all_views_single_pass(module)?;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// Spatial & Matrix Layout Conversion Pass (`convert-layout`).
#[derive(Debug, Clone)]
pub struct ConvertLayoutPass {
    pub layout: String,
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
        "Converts spatial/matrix layouts (nhwc/nchw/col_major) and propagates layout across ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let tgt = self.layout.trim().to_ascii_lowercase();
        let tgt_layout = if tgt.is_empty() { "nhwc" } else { tgt.as_str() };
        run_convert_layout(module, tgt_layout, self.propagate, self.name())
    }
}

fn fold_all_views_single_pass(module: &mut Module) -> Result<(usize, usize, usize)> {
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
            let Some(hlo) = StableHloOp::from_op(op) else {
                continue;
            };

            match hlo {
                StableHloOp::Transpose {
                    operand,
                    permutation: perm2,
                    result,
                } => {
                    if try_sink_transpose_through_elementwise(
                        module,
                        block_id,
                        op_id,
                        operand,
                        &perm2,
                        result,
                        &mut dead_ops,
                        &mut created,
                        &mut modified,
                        &mut removed,
                    ) {
                        continue;
                    }

                    let mut current_operand = operand;
                    let mut current_perm = perm2;

                    if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def
                        && !dead_ops.contains(&prod_op_id)
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

                    let is_identity = !current_perm.is_empty()
                        && current_perm.iter().enumerate().all(|(i, &p)| p == i as i64);
                    if is_identity {
                        module.replace_all_uses(result, current_operand);
                        dead_ops.insert(op_id);
                        removed += 1;
                    }
                }

                StableHloOp::Reshape { operand, result } => {
                    let mut current_operand = operand;
                    if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[operand].def
                        && !dead_ops.contains(&prod_op_id)
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
                        dead_ops.insert(op_id);
                        removed += 1;
                    }
                }

                StableHloOp::BroadcastInDim {
                    operand,
                    broadcast_dimensions: dims2,
                    result,
                } => {
                    let mut cur_operand = operand;
                    let mut cur_dims = dims2;

                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                        && !dead_ops.contains(&prod_id)
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

                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[cur_operand].def
                        && !dead_ops.contains(&prod_id)
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
                        dead_ops.insert(op_id);
                        removed += 1;
                    }
                }

                StableHloOp::Exp { operand, result }
                | StableHloOp::Tanh { operand, result }
                | StableHloOp::Rsqrt { operand, result } => {
                    if module.use_count(operand) != 1 {
                        continue;
                    }
                    let ValueDef::OpResult { op: bcast_id, .. } = module.values[operand].def else {
                        continue;
                    };
                    if dead_ops.contains(&bcast_id) {
                        continue;
                    }
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

                StableHloOp::Concatenate { inputs, result, .. } => {
                    if inputs.len() == 1 {
                        module.replace_all_uses(result, inputs[0]);
                        dead_ops.insert(op_id);
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
                        && !dead_ops.contains(&prod_id)
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

                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[cur_operand].def
                        && !dead_ops.contains(&prod_id)
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
                                    dead_ops.insert(op_id);
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

                StableHloOp::Convert { operand, result } => {
                    let mut cur_operand = operand;
                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[operand].def
                        && !dead_ops.contains(&prod_id)
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
                        dead_ops.insert(op_id);
                        removed += 1;
                        continue;
                    }

                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[cur_operand].def
                        && !dead_ops.contains(&prod_id)
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

                _ => {}
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

    Ok((created, modified, removed))
}

fn try_sink_transpose_through_elementwise(
    module: &mut Module,
    block_id: BlockId,
    op_id: OpId,
    elem_val: ValueId,
    perm: &[i64],
    trans_res: ValueId,
    dead_ops: &mut HashSet<OpId>,
    created: &mut usize,
    modified: &mut usize,
    removed: &mut usize,
) -> bool {
    let ValueDef::OpResult { op: prod_op_id, .. } = module.values[elem_val].def else {
        return false;
    };
    if dead_ops.contains(&prod_op_id) || module.use_count(elem_val) != 1 {
        return false;
    }
    let Some(prod_op) = module.ops.get(prod_op_id).cloned() else {
        return false;
    };
    let Some(prod_hlo) = StableHloOp::from_op(&prod_op) else {
        return false;
    };

    let trans_ty = module.values[trans_res].ty.clone();

    match prod_hlo {
        StableHloOp::Exp { operand: x, .. }
        | StableHloOp::Tanh { operand: x, .. }
        | StableHloOp::Rsqrt { operand: x, .. }
        | StableHloOp::Convert { operand: x, .. } => {
            let tx = insert_folded_transpose_before(
                module, block_id, op_id, x, perm, trans_ty, dead_ops, created, modified,
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.name = prod_op.name;
            op_mut.operands = SmallVec::from_slice(&[tx]);
            op_mut.attributes = prod_op.attributes;
            dead_ops.insert(prod_op_id);
            *modified += 1;
            *removed += 1;
            true
        }
        StableHloOp::Add { lhs, rhs, .. }
        | StableHloOp::Subtract { lhs, rhs, .. }
        | StableHloOp::Multiply { lhs, rhs, .. }
        | StableHloOp::Divide { lhs, rhs, .. }
        | StableHloOp::Maximum { lhs, rhs, .. }
        | StableHloOp::Minimum { lhs, rhs, .. } => {
            let tlhs = insert_folded_transpose_before(
                module,
                block_id,
                op_id,
                lhs,
                perm,
                trans_ty.clone(),
                dead_ops,
                created,
                modified,
            );
            let trhs = insert_folded_transpose_before(
                module, block_id, op_id, rhs, perm, trans_ty, dead_ops, created, modified,
            );
            let op_mut = &mut module.ops[op_id];
            op_mut.name = prod_op.name;
            op_mut.operands = SmallVec::from_slice(&[tlhs, trhs]);
            op_mut.attributes = prod_op.attributes;
            dead_ops.insert(prod_op_id);
            *modified += 1;
            *removed += 1;
            true
        }
        _ => false,
    }
}

fn insert_folded_transpose_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input: ValueId,
    perm: &[i64],
    out_ty: crate::ir::Type,
    dead_ops: &HashSet<OpId>,
    created: &mut usize,
    modified: &mut usize,
) -> ValueId {
    let mut cur_input = input;
    let mut cur_perm: Vec<i64> = perm.to_vec();

    if let ValueDef::OpResult { op: prod_op_id, .. } = module.values[input].def
        && !dead_ops.contains(&prod_op_id)
        && let Some(prod_op) = module.ops.get(prod_op_id)
        && let Some(StableHloOp::Transpose {
            operand: base_input,
            permutation: perm1,
            ..
        }) = StableHloOp::from_op(prod_op)
        && perm1.len() == cur_perm.len()
    {
        cur_perm = cur_perm.iter().map(|&idx| perm1[idx as usize]).collect();
        cur_input = base_input;
        *modified += 1;
    }

    let is_identity = !cur_perm.is_empty()
        && cur_perm.iter().enumerate().all(|(i, &p)| p == i as i64)
        && module.values[cur_input].ty == out_ty;
    if is_identity {
        return cur_input;
    }

    *created += 1;
    insert_transpose_before(module, block_id, before_op, cur_input, &cur_perm, out_ty)
}

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

fn insert_transpose_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    input: ValueId,
    perm: &[i64],
    out_ty: crate::ir::Type,
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

fn run_convert_layout(
    module: &mut Module,
    tgt_layout: &str,
    propagate: bool,
    pass_name: &str,
) -> Result<PassResult> {
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
            if op.attributes.get("data_layout").and_then(|a| a.as_str()) == Some(tgt_layout) {
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
                Attribute::String(tgt_layout.to_string()),
            );
            if propagate {
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

    if propagate {
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
                        Attribute::String(tgt_layout.to_string()),
                    );
                    module.ops[op_id].attributes.insert(
                        "castforge.layout_propagated".to_string(),
                        Attribute::Bool(true),
                    );
                    modified += 1;
                }
            }
        }
        let (c, m, r) = fold_all_views_single_pass(module)?;
        created += c;
        modified += m;
        removed += r;
    }

    Ok(PassResult {
        pass_name: pass_name.to_string(),
        changed: modified > 0 || created > 0 || removed > 0,
        ops_removed: removed,
        ops_modified: modified,
        ops_created: created,
    })
}
