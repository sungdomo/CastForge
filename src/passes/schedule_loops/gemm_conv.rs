//! Submodule `lowering` of `schedule_loops`.

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use crate::passes::{Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

/// Static Tile-Alignment Padding Pass (`pad-for-tiling`).
#[derive(Debug, Clone)]
pub struct PadForTilingPass {
    pub align: i64,
}

impl Default for PadForTilingPass {
    fn default() -> Self {
        Self { align: 8 }
    }
}

impl Pass for PadForTilingPass {
    fn name(&self) -> &'static str {
        "pad-for-tiling"
    }

    fn description(&self) -> &'static str {
        "Computes tile-aligned (M, K, N) multiples for GEMM ops to eliminate remainder tails"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let align = self.align.max(1);
        let round_up = |x: i64| ((x + align - 1) / align) * align;
        let mut modified = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "dot_general") || op.operands.len() < 2 {
                    continue;
                }
                let lhs_ty = module.values[op.operands[0]].ty.as_tensor().cloned();
                let rhs_ty = module.values[op.operands[1]].ty.as_tensor().cloned();
                let (Some(lt), Some(rt)) = (lhs_ty, rhs_ty) else {
                    continue;
                };
                if lt.shape.len() != 2 || rt.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k), Dim::Static(n)) =
                    (&lt.shape[0], &lt.shape[1], &rt.shape[1])
                else {
                    continue;
                };
                let (pm, pk, pn) = (round_up(*m), round_up(*k), round_up(*n));
                let dst = &mut module.ops[op_id].attributes;
                dst.insert("pad_align".to_string(), Attribute::Int(align));
                dst.insert(
                    "padded_mnk".to_string(),
                    Attribute::DenseIntArray(vec![pm, pn, pk]),
                );
                dst.insert(
                    "pad_deltas".to_string(),
                    Attribute::DenseIntArray(vec![pm - m, pn - n, pk - k]),
                );
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

/// Large GEMM Column-Splitting Pass (`split-large-gemm`).
#[derive(Debug, Clone)]
pub struct SplitLargeGemmPass {
    pub max_n: i64,
}

impl Default for SplitLargeGemmPass {
    fn default() -> Self {
        Self { max_n: 2 }
    }
}

impl Pass for SplitLargeGemmPass {
    fn name(&self) -> &'static str {
        "split-large-gemm"
    }

    fn description(&self) -> &'static str {
        "Splits wide dot_general GEMMs (N > max_n) into sliced sub-GEMMs + concatenate"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let max_n = self.max_n.max(1);
        let mut created = 0usize;
        let mut removed = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if !op.is_op("stablehlo", "dot_general")
                    || op.operands.len() < 2
                    || op.results.len() != 1
                {
                    continue;
                }
                let lhs_id = op.operands[0];
                let rhs_id = op.operands[1];
                let orig_res = op.results[0];
                let Some(lt) = module.values[lhs_id].ty.as_tensor().cloned() else {
                    continue;
                };
                let Some(rt) = module.values[rhs_id].ty.as_tensor().cloned() else {
                    continue;
                };
                if lt.shape.len() != 2 || rt.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(m), Dim::Static(k), Dim::Static(k2), Dim::Static(n)) =
                    (&lt.shape[0], &lt.shape[1], &rt.shape[0], &rt.shape[1])
                else {
                    continue;
                };
                if *k != *k2 || *n <= max_n || *n < 2 {
                    continue;
                }

                let n_left = *n / 2;
                let n_right = *n - n_left;
                let dt = lt.dtype.clone();

                let mut sl_l_attrs = IndexMap::new();
                sl_l_attrs.insert(
                    "start_indices".to_string(),
                    Attribute::DenseIntArray(vec![0, 0]),
                );
                sl_l_attrs.insert(
                    "limit_indices".to_string(),
                    Attribute::DenseIntArray(vec![*k, n_left]),
                );
                sl_l_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
                let w_l_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*k), Dim::Static(n_left)],
                    dt.clone(),
                ));
                let sl_l_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "slice"),
                    smallvec![rhs_id],
                    vec![(Some("w_split_l".to_string()), w_l_ty)],
                    sl_l_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let w_l_val = module.ops[sl_l_op].results[0];

                let mut sl_r_attrs = IndexMap::new();
                sl_r_attrs.insert(
                    "start_indices".to_string(),
                    Attribute::DenseIntArray(vec![0, n_left]),
                );
                sl_r_attrs.insert(
                    "limit_indices".to_string(),
                    Attribute::DenseIntArray(vec![*k, *n]),
                );
                sl_r_attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
                let w_r_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*k), Dim::Static(n_right)],
                    dt.clone(),
                ));
                let sl_r_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "slice"),
                    smallvec![rhs_id],
                    vec![(Some("w_split_r".to_string()), w_r_ty)],
                    sl_r_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let w_r_val = module.ops[sl_r_op].results[0];

                let out_l_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*m), Dim::Static(n_left)],
                    dt.clone(),
                ));
                let dot_l_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    smallvec![lhs_id, w_l_val],
                    vec![(Some("gemm_l".to_string()), out_l_ty)],
                    op.attributes.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let dot_l_val = module.ops[dot_l_op].results[0];

                let out_r_ty = Type::Tensor(TensorType::new(
                    vec![Dim::Static(*m), Dim::Static(n_right)],
                    dt.clone(),
                ));
                let dot_r_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "dot_general"),
                    smallvec![lhs_id, w_r_val],
                    vec![(Some("gemm_r".to_string()), out_r_ty)],
                    op.attributes.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let dot_r_val = module.ops[dot_r_op].results[0];

                let mut cat_attrs = IndexMap::new();
                cat_attrs.insert("dimension".to_string(), Attribute::Int(1));
                let full_out_ty = module.values[orig_res].ty.clone();
                let cat_op = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "concatenate"),
                    smallvec![dot_l_val, dot_r_val],
                    vec![(Some("gemm_cat".to_string()), full_out_ty)],
                    cat_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let cat_val = module.ops[cat_op].results[0];

                module.replace_all_uses(orig_res, cat_val);
                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id]
                        .ops
                        .splice(pos..=pos, [sl_l_op, sl_r_op, dot_l_op, dot_r_op, cat_op]);
                }
                created += 5;
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: removed,
            ops_created: created,
        })
    }
}

/// Vision / CNN Lowering Pass (`lower-conv-to-gemm`).
pub struct LowerConvToGemmPass;

impl Pass for LowerConvToGemmPass {
    fn name(&self) -> &'static str {
        "lower-conv-to-gemm"
    }

    fn description(&self) -> &'static str {
        "Lowers stablehlo.convolution into im2col + stablehlo.dot_general GEMM"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Convolution { lhs, rhs, result }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };

                let lhs_dims = static_dims(module, lhs);
                let rhs_dims = static_dims(module, rhs);
                let out_dims = static_dims(module, result);
                let dtype = module.values[result]
                    .ty
                    .as_tensor()
                    .map(|t| t.dtype.clone())
                    .unwrap_or(DType::F32);

                let is_pointwise_2d =
                    lhs_dims.len() == 2 && rhs_dims.len() == 2 && lhs_dims[1] == rhs_dims[0];
                if is_pointwise_2d {
                    let mut dot_attrs = IndexMap::new();
                    dot_attrs.insert(
                        "lhs_contracting_dimensions".to_string(),
                        Attribute::DenseIntArray(vec![1]),
                    );
                    dot_attrs.insert(
                        "rhs_contracting_dimensions".to_string(),
                        Attribute::DenseIntArray(vec![0]),
                    );
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "dot_general");
                    op_mut.operands = SmallVec::from_slice(&[lhs, rhs]);
                    op_mut.attributes = dot_attrs;
                    modified += 1;
                    continue;
                }

                let (n_batch, h_in, w_in, c_in) = match lhs_dims.as_slice() {
                    [h, w, c] => (1, *h, *w, *c),
                    [n, h, w, c] => (*n, *h, *w, *c),
                    [hw, c] => (1, *hw, 1, *c),
                    _ => (1, 1, 1, 1),
                };
                let (kh, kw, k_total, c_out) = match rhs_dims.as_slice() {
                    [kh, kw, cin, cout] => (*kh, *kw, kh * kw * cin, *cout),
                    [kt, cout] => {
                        let spatial = (kt / c_in.max(1)).max(1);
                        let k_side = (spatial as f64).sqrt().round() as i64;
                        (k_side.max(1), (spatial / k_side.max(1)).max(1), *kt, *cout)
                    }
                    _ => (1, 1, c_in, 1),
                };
                let (oh, ow) = match out_dims.as_slice() {
                    [_, oh, ow, _] => (*oh, *ow),
                    [oh, ow, _] => (*oh, *ow),
                    _ => ((h_in - kh + 1).max(1), (w_in - kw + 1).max(1)),
                };
                let out_elems: i64 = out_dims.iter().product::<i64>().max(1);
                let m_patches = (out_elems / c_out.max(1)).max(n_batch);

                let patches_ty = Type::Tensor(TensorType {
                    shape: vec![Dim::Static(m_patches), Dim::Static(k_total)],
                    dtype: dtype.clone(),
                    encoding: None,
                });
                let mut im2col_attrs = IndexMap::new();
                im2col_attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.im2col".to_string()),
                );
                im2col_attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("im2col_impl".to_string()),
                );
                im2col_attrs.insert("h_in".to_string(), Attribute::Int(h_in));
                im2col_attrs.insert("w_in".to_string(), Attribute::Int(w_in));
                im2col_attrs.insert("c_in".to_string(), Attribute::Int(c_in));
                im2col_attrs.insert("kh".to_string(), Attribute::Int(kh));
                im2col_attrs.insert("kw".to_string(), Attribute::Int(kw));
                im2col_attrs.insert("oh".to_string(), Attribute::Int(oh));
                im2col_attrs.insert("ow".to_string(), Attribute::Int(ow));

                let patches_val = insert_op_before(
                    module,
                    block_id,
                    op_id,
                    OpName::new("stablehlo", "composite"),
                    &[lhs],
                    im2col_attrs,
                    patches_ty,
                );
                created += 1;

                let rhs_2d = if rhs_dims.len() == 2 {
                    rhs
                } else {
                    let w2d_ty = Type::Tensor(TensorType {
                        shape: vec![Dim::Static(k_total), Dim::Static(c_out)],
                        dtype: dtype.clone(),
                        encoding: None,
                    });
                    created += 1;
                    insert_op_before(
                        module,
                        block_id,
                        op_id,
                        OpName::new("stablehlo", "reshape"),
                        &[rhs],
                        IndexMap::new(),
                        w2d_ty,
                    )
                };

                let mut dot_attrs = IndexMap::new();
                dot_attrs.insert(
                    "lhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![1]),
                );
                dot_attrs.insert(
                    "rhs_contracting_dimensions".to_string(),
                    Attribute::DenseIntArray(vec![0]),
                );

                if out_dims.len() == 2 {
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "dot_general");
                    op_mut.operands = SmallVec::from_slice(&[patches_val, rhs_2d]);
                    op_mut.attributes = dot_attrs;
                    modified += 1;
                } else {
                    let dot_2d_ty = Type::Tensor(TensorType {
                        shape: vec![Dim::Static(m_patches), Dim::Static(c_out)],
                        dtype,
                        encoding: None,
                    });
                    let dot_2d_val = insert_op_before(
                        module,
                        block_id,
                        op_id,
                        OpName::new("stablehlo", "dot_general"),
                        &[patches_val, rhs_2d],
                        dot_attrs,
                        dot_2d_ty,
                    );
                    created += 1;

                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "reshape");
                    op_mut.operands = SmallVec::from_slice(&[dot_2d_val]);
                    op_mut.attributes.clear();
                    modified += 1;
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn static_dims(module: &Module, val_id: ValueId) -> Vec<i64> {
    module.values[val_id]
        .ty
        .as_tensor()
        .map(|t| {
            t.shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n,
                    _ => 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn insert_op_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    name: OpName,
    operands: &[ValueId],
    attrs: IndexMap<String, Attribute>,
    out_ty: Type,
) -> ValueId {
    let new_op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name,
        operands: SmallVec::from_slice(operands),
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
