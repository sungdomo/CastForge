use crate::StableHloOp;
use castforge_ir::{Attribute, Dim, Module, OpId, Type};

/// Validates StableHLO tensor shape rules for `op_id` inside `module`.
pub fn verify_stablehlo_shapes(module: &Module, op_id: OpId) -> Result<(), String> {
    let Some(op) = module.ops.get(op_id) else {
        return Ok(());
    };
    let Some(hlo) = StableHloOp::from_op(op) else {
        return Ok(());
    };

    match hlo {
        StableHloOp::Transpose {
            operand,
            permutation,
            result,
        } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            if in_ty.rank() != permutation.len() || out_ty.rank() != permutation.len() {
                return Err(format!(
                    "stablehlo.transpose rank mismatch: input rank {}, perm {:?}, output rank {}",
                    in_ty.rank(),
                    permutation,
                    out_ty.rank()
                ));
            }
            for (out_idx, &in_idx) in permutation.iter().enumerate() {
                let Some(in_dim) = in_ty.shape.get(in_idx as usize) else {
                    return Err(format!(
                        "stablehlo.transpose permutation index {in_idx} out of bounds"
                    ));
                };
                let out_dim = &out_ty.shape[out_idx];
                if let (Dim::Static(a), Dim::Static(b)) = (in_dim, out_dim)
                    && a != b
                {
                    return Err(format!(
                        "stablehlo.transpose dim mismatch at axis {out_idx}: {a} vs {b}"
                    ));
                }
            }
        }
        StableHloOp::Reshape { operand, result } => {
            let Some(in_ty) = module.values[operand].ty.as_tensor() else {
                return Ok(());
            };
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            if let (Some(in_elems), Some(out_elems)) = (in_ty.num_elements(), out_ty.num_elements())
                && in_elems != out_elems
            {
                return Err(format!(
                    "stablehlo.reshape count mismatch: {in_ty} ({in_elems}) vs {out_ty}"
                ));
            }
        }
        StableHloOp::Slice {
            operand,
            start_indices,
            limit_indices,
            strides,
            result,
        } => {
            let Some(out_ty) = module.values[result].ty.as_tensor() else {
                return Ok(());
            };
            let _ = operand;
            for (i, &start) in start_indices.iter().enumerate() {
                let limit = *limit_indices.get(i).unwrap_or(&start);
                let stride = (*strides.get(i).unwrap_or(&1)).max(1);
                let expected = (limit - start + stride - 1) / stride;
                if let Some(Dim::Static(actual)) = out_ty.shape.get(i)
                    && *actual != expected
                {
                    return Err(format!(
                        "stablehlo.slice axis {i} expected dim {expected}, got {actual}"
                    ));
                }
            }
        }
        _ => {}
    }

    Ok(())
}

/// Helper to query function metadata from a `func.func` operation.
pub struct FuncView<'a> {
    pub op_id: OpId,
    pub sym_name: &'a str,
    pub function_type: Option<&'a Type>,
}

impl<'a> FuncView<'a> {
    pub fn from_op(module: &'a Module, op_id: OpId) -> Option<Self> {
        let op = module.ops.get(op_id)?;
        if !op.is_op("func", "func") {
            return None;
        }
        let sym_name = op.attributes.get("sym_name")?.as_str()?;
        let function_type = match op.attributes.get("function_type") {
            Some(Attribute::Type(ty)) => Some(ty),
            _ => None,
        };
        Some(Self {
            op_id,
            sym_name,
            function_type,
        })
    }
}
