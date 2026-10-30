//! Primitive StableHLO op builders used by [`super::decompose::DecomposeCompositesPass`].

use crate::ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, TensorType, Type,
    Value, ValueDef, ValueId,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

pub(super) fn reduced_row_type(out_ty: &Type) -> (Type, i64, Vec<i64>) {
    if let Some(t) = out_ty.as_tensor() {
        let rank = t.shape.len();
        if rank >= 2 {
            let reduced_shape = t.shape[..rank - 1].to_vec();
            let axis = (rank - 1) as i64;
            let bcast_dims: Vec<i64> = (0..(rank - 1) as i64).collect();
            return (
                Type::Tensor(TensorType {
                    shape: reduced_shape,
                    dtype: t.dtype.clone(),
                    encoding: None,
                }),
                axis,
                bcast_dims,
            );
        }
    }
    (
        Type::Tensor(TensorType {
            shape: vec![Dim::Static(1)],
            dtype: DType::F32,
            encoding: None,
        }),
        0,
        vec![0],
    )
}

pub(super) fn insert_silu_denom_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    x: ValueId,
    out_ty: &Type,
) -> ValueId {
    let t_ty = out_ty.as_tensor().cloned().unwrap_or(TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    });
    let mut z_attrs = IndexMap::new();
    z_attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty.clone(),
            raw_literal: "0.0".to_string(),
        },
    );
    let zero_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        z_attrs,
        out_ty.clone(),
    );
    let mut one_attrs = IndexMap::new();
    one_attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty,
            raw_literal: "1.0".to_string(),
        },
    );
    let one_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        one_attrs,
        out_ty.clone(),
    );
    let neg_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "subtract"),
        &[zero_val, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let exp_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "exponential"),
        &[neg_val],
        IndexMap::new(),
        out_ty.clone(),
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[exp_val, one_val],
        IndexMap::new(),
        out_ty.clone(),
    )
}

pub(super) fn insert_scalar_const_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    val: f32,
) -> ValueId {
    let s_ty = TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    };
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: s_ty.clone(),
            raw_literal: format!("{val:.1}"),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        Type::Tensor(s_ty),
    )
}

pub(super) fn half_col_type(out_ty: &Type) -> (Type, i64, i64) {
    if let Some(t) = out_ty.as_tensor()
        && let [Dim::Static(r), Dim::Static(c)] = t.shape.as_slice()
    {
        return (
            Type::Tensor(TensorType {
                shape: vec![Dim::Static(*r), Dim::Static(*c / 2)],
                dtype: t.dtype.clone(),
                encoding: None,
            }),
            *r,
            *c,
        );
    }
    (
        Type::Tensor(TensorType {
            shape: vec![Dim::Static(1), Dim::Static(1)],
            dtype: DType::F32,
            encoding: None,
        }),
        1,
        2,
    )
}

pub(super) fn insert_slice_2d_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    operand: ValueId,
    starts: [i64; 2],
    limits: [i64; 2],
    out_ty: Type,
) -> ValueId {
    let mut attrs = IndexMap::new();
    attrs.insert(
        "start_indices".to_string(),
        Attribute::DenseIntArray(starts.to_vec()),
    );
    attrs.insert(
        "limit_indices".to_string(),
        Attribute::DenseIntArray(limits.to_vec()),
    );
    attrs.insert("strides".to_string(), Attribute::DenseIntArray(vec![1, 1]));
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "slice"),
        &[operand],
        attrs,
        out_ty,
    )
}

pub(super) fn insert_zero_tensor_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    ty: &Type,
) -> ValueId {
    let t_ty = ty.as_tensor().cloned().unwrap_or(TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    });
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty,
            raw_literal: "0.0".to_string(),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        ty.clone(),
    )
}

pub(super) fn insert_op_before(
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

pub(super) fn insert_splat_tensor_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    ty: &Type,
    val: f32,
) -> ValueId {
    let t_ty = ty.as_tensor().cloned().unwrap_or(TensorType {
        shape: Vec::new(),
        dtype: DType::F32,
        encoding: None,
    });
    let mut attrs = IndexMap::new();
    attrs.insert(
        "value".to_string(),
        Attribute::DenseElements {
            ty: t_ty,
            raw_literal: format!("{val}"),
        },
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "constant"),
        &[],
        attrs,
        ty.clone(),
    )
}

pub(super) fn insert_rmsnorm_core_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    x: ValueId,
    eps: f32,
    out_ty: &Type,
) -> (ValueId, usize) {
    let sq_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let (red_ty, axis, bcast_dims) = reduced_row_type(out_ty);
    let cols = out_ty
        .as_tensor()
        .and_then(|t| match t.shape.last() {
            Some(Dim::Static(c)) => Some(*c as f32),
            _ => None,
        })
        .unwrap_or(1.0);
    let zero_init = insert_scalar_const_before(module, block_id, before_op, 0.0);
    let mut red_attrs = IndexMap::new();
    red_attrs.insert(
        "dimensions".to_string(),
        Attribute::DenseIntArray(vec![axis]),
    );
    red_attrs.insert(
        "reducer".to_string(),
        Attribute::String("stablehlo.add".to_string()),
    );
    let sum_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "reduce"),
        &[sq_val, zero_init],
        red_attrs,
        red_ty.clone(),
    );
    let n_const = insert_splat_tensor_before(module, block_id, before_op, &red_ty, cols);
    let mean_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "divide"),
        &[sum_val, n_const],
        IndexMap::new(),
        red_ty.clone(),
    );
    let eps_const = insert_splat_tensor_before(module, block_id, before_op, &red_ty, eps);
    let var_eps = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[mean_val, eps_const],
        IndexMap::new(),
        red_ty.clone(),
    );
    let inv_rms = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "rsqrt"),
        &[var_eps],
        IndexMap::new(),
        red_ty,
    );
    let mut bcast_attrs = IndexMap::new();
    bcast_attrs.insert(
        "broadcast_dimensions".to_string(),
        Attribute::DenseIntArray(bcast_dims),
    );
    let inv_bcast = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "broadcast_in_dim"),
        &[inv_rms],
        bcast_attrs,
        out_ty.clone(),
    );
    let norm_val = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x, inv_bcast],
        IndexMap::new(),
        out_ty.clone(),
    );
    (norm_val, 9)
}

pub(super) fn insert_gelu_before(
    module: &mut Module,
    block_id: BlockId,
    before_op: OpId,
    x: ValueId,
    out_ty: &Type,
) -> ValueId {
    let half = insert_splat_tensor_before(module, block_id, before_op, out_ty, 0.5);
    let one = insert_splat_tensor_before(module, block_id, before_op, out_ty, 1.0);
    let c_k = insert_splat_tensor_before(module, block_id, before_op, out_ty, 0.797_884_6);
    let c_a = insert_splat_tensor_before(module, block_id, before_op, out_ty, 0.044715);
    let x2 = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let x3 = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[x2, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    let ax3 = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[c_a, x3],
        IndexMap::new(),
        out_ty.clone(),
    );
    let poly = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[x, ax3],
        IndexMap::new(),
        out_ty.clone(),
    );
    let inner = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[c_k, poly],
        IndexMap::new(),
        out_ty.clone(),
    );
    let th = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "tanh"),
        &[inner],
        IndexMap::new(),
        out_ty.clone(),
    );
    let one_p = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "add"),
        &[one, th],
        IndexMap::new(),
        out_ty.clone(),
    );
    let hx = insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[half, x],
        IndexMap::new(),
        out_ty.clone(),
    );
    insert_op_before(
        module,
        block_id,
        before_op,
        OpName::new("stablehlo", "multiply"),
        &[hx, one_p],
        IndexMap::new(),
        out_ty.clone(),
    )
}
