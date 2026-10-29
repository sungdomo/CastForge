use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Vision / CNN Lowering Pass (`lower-conv-to-gemm`).
///
/// Lowers `stablehlo.convolution` operations into explicit `im2col` patch extraction
/// (`"castforge.im2col"` composite when spatial kernel $> 1\times 1$) followed by
/// `stablehlo.dot_general` matrix multiplication so that all downstream GEMM fusions
/// (`fuse-matmul-epilogue`, `pack-weights`, `fuse-dequant-matmul`) and polyhedral loop
/// transformations apply seamlessly to convolutional layers.
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

                // Pointwise 1x1 or already-2D [M, K] x [K, N] convolution -> direct dot_general
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

                // General spatial convolution: insert `castforge.im2col` -> `dot_general`
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
