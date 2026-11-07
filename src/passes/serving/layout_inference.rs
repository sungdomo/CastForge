//! Bidirectional tensor layout inference and device architecture dtype
//! validation.
//!
//! Seeds layout constraints from 2D/3D `mo.conv`, `mo.conv_transpose`, and
//! `mo.matmul`, propagates constraints bidirectionally across `mo.guard`,
//! `mo.transfer`, and elementwise operations, and validates tensor element
//! types against target CPU/accelerator architectures (`x86_64`, `aarch64`,
//! `gpu`).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ir::mlir::{Attribute, DType, Module, OpId, Type, ValueDef, ValueId};

fn is_elementwise_or_same_layout_op(full: &str) -> bool {
    matches!(
        full,
        "mo.transfer"
            | "mo.add"
            | "mo.sub"
            | "mo.mul"
            | "mo.div"
            | "mo.relu"
            | "mo.tanh"
            | "mo.exp"
            | "mo.neg"
            | "mo.abs"
            | "stablehlo.add"
            | "stablehlo.multiply"
    )
}

/// Seeds tensor layout constraints from `mo.conv`, `mo.conv_transpose`, and
/// `mo.matmul` operations in `module`, propagates them bidirectionally across
/// `mo.guard`, `mo.transfer`, and elementwise operations, and assigns the
/// inferred layouts to `TensorType::encoding`.
///
/// Values that already carry a non-empty `encoding` layout are left untouched
/// so the pass is idempotent.
///
/// # Errors
///
/// Returns `Err` if a `mo.conv` or `mo.conv_transpose` has unsupported rank
/// (must be 4D or 5D) or if conflicting layout constraints meet on the same
/// SSA value.
pub fn propagate_bidirectional_tensor_layouts(module: &mut Module) -> Result<usize, String> {
    let mut inferred: HashMap<ValueId, String> = HashMap::new();

    let constrain = |inferred: &mut HashMap<ValueId, String>,
                     module: &Module,
                     vid: ValueId,
                     layout: &str|
     -> Result<bool, String> {
        let Some(val) = module.values.get(vid) else {
            return Ok(false);
        };
        let Type::Tensor(tt) = &val.ty else {
            return Ok(false);
        };
        if tt.encoding.is_some() {
            return Ok(false);
        }
        if let Some(existing) = inferred.get(&vid) {
            if existing != layout {
                return Err(format!(
                    "attempting to constrain value to layout: {layout} when it's already \
                     constrained to layout: {existing}"
                ));
            }
            return Ok(false);
        }
        inferred.insert(vid, layout.to_string());
        Ok(true)
    };

    // Step 1: Seed constraints from `mo.conv`, `mo.conv_transpose`, and `mo.matmul`.
    for &op_id in &module.live_ops() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let full = op.name.full_name();
        match full.as_str() {
            "mo.conv" | "mo.conv_transpose" => {
                if op.operands.len() < 2 {
                    return Err(format!("{full} requires input and filter operands"));
                }
                let input_v = op.operands[0];
                let filter_v = op.operands[1];
                let in_rank = match &module.values[input_v].ty {
                    Type::Tensor(tt) => tt.shape.len(),
                    _ => 0,
                };
                let flt_rank = match &module.values[filter_v].ty {
                    Type::Tensor(tt) => tt.shape.len(),
                    _ => 0,
                };
                let (in_layout, flt_layout) = match (full.as_str(), in_rank, flt_rank) {
                    ("mo.conv", 4, 4) => ("NHWC", "FCRS"),
                    ("mo.conv", 5, 5) => ("NDHWC", "FCQRS"),
                    ("mo.conv_transpose", 4, 4) => ("NHWC", "RSFC"),
                    ("mo.conv_transpose", 5, 5) => ("NDHWC", "QRSFC"),
                    _ => {
                        return Err(format!("Only support 2D and 3D layout for {full}"));
                    }
                };
                constrain(&mut inferred, module, input_v, in_layout)?;
                constrain(&mut inferred, module, filter_v, flt_layout)?;
            }
            "mo.matmul" if op.operands.len() >= 2 => {
                let weights_v = op.operands[1];
                constrain(&mut inferred, module, weights_v, "KN")?;
            }
            _ => {}
        }
    }

    // Build value -> user ops map for bidirectional propagation.
    let mut users_of_val: HashMap<ValueId, Vec<OpId>> = HashMap::new();
    for &oid in &module.live_ops() {
        if let Some(op) = module.ops.get(oid) {
            for &opd in &op.operands {
                users_of_val.entry(opd).or_default().push(oid);
            }
        }
    }

    // Step 2: Propagate layouts through `mo.guard`, `mo.transfer`, and elementwise ops.
    let mut worklist: VecDeque<ValueId> = inferred.keys().copied().collect();
    let mut in_history: HashSet<ValueId> = inferred.keys().copied().collect();

    while let Some(vid) = worklist.pop_front() {
        let Some(layout) = inferred.get(&vid).cloned() else {
            continue;
        };
        let mut affected_ops: Vec<OpId> = users_of_val.get(&vid).cloned().unwrap_or_default();
        if let Some(val) = module.values.get(vid)
            && let ValueDef::OpResult { op: def_op, .. } = val.def
        {
            affected_ops.push(def_op);
        }

        for oid in affected_ops {
            let Some(op) = module.ops.get(oid) else {
                continue;
            };
            let full = op.name.full_name();
            let mut propagate_vals: Vec<ValueId> = Vec::new();
            if full == "mo.guard" {
                propagate_vals.extend(op.operands.iter().skip(1).copied());
                propagate_vals.extend(op.results.iter().copied());
            } else if is_elementwise_or_same_layout_op(&full) {
                propagate_vals.extend(op.operands.iter().copied());
                propagate_vals.extend(op.results.iter().copied());
            }

            for target_v in propagate_vals {
                constrain(&mut inferred, module, target_v, &layout)?;
                if in_history.insert(target_v) {
                    worklist.push_back(target_v);
                }
            }
        }
    }

    // Step 3: Assign inferred layouts to `TensorType::encoding`.
    let count = inferred.len();
    for (vid, layout) in inferred {
        if let Some(val_mut) = module.values.get_mut(vid)
            && let Type::Tensor(tt) = &mut val_mut.ty
        {
            tt.encoding = Some(layout);
        }
    }
    Ok(count)
}

/// Returns `true` if `dtype` is supported for computation/casting on `target_arch`:
/// - `x86_64`: supports `BF16`, `F16`, floats $\ge 32$ bits, and all non-float types;
///   rejects sub-32-bit floats other than `BF16`/`F16` (e.g. `F8E4M3FN`, `F8E5M2`).
/// - `aarch64`: supports `BF16`, floats $\ge 32$ bits, and all non-float types;
///   rejects `F16` and 8-bit floats.
/// - GPU / accelerator architectures (`"nvptx64"`, `"amdgcn"`, `"gpu"`): support all dtypes.
#[must_use]
pub fn is_supported_target_arch_dtype(target_arch: &str, dtype: &DType) -> bool {
    let arch = target_arch
        .split('-')
        .next()
        .unwrap_or(target_arch)
        .trim()
        .to_ascii_lowercase();
    if arch == "x86_64" || arch == "aarch64" {
        if *dtype == DType::BF16 {
            return true;
        }
        if arch == "x86_64" && *dtype == DType::F16 {
            return true;
        }
        if dtype.is_float() && dtype.bit_width() < 32 {
            return false;
        }
    }
    true
}

/// Validates that all tensor operands and results in `module` (excluding
/// operations marked with `dtype_agnostic = true`) use element types supported
/// on `target_arch`.
///
/// # Errors
///
/// Returns `Err` on the first unsupported `(dtype, device)` combination.
pub fn validate_module_target_arch_dtypes(
    module: &Module,
    target_arch: &str,
) -> Result<(), String> {
    if target_arch.trim().is_empty() {
        return Ok(());
    }
    for op_id in module.live_ops() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        if matches!(
            op.attributes.get("dtype_agnostic"),
            Some(Attribute::Bool(true))
        ) {
            continue;
        }
        let op_dev = op
            .attributes
            .get("device_ref")
            .or_else(|| op.attributes.get("device"))
            .and_then(Attribute::as_str)
            .unwrap_or("cpu:0");

        for &vid in op.operands.iter().chain(op.results.iter()) {
            let Some(val) = module.values.get(vid) else {
                continue;
            };
            let Type::Tensor(tt) = &val.ty else {
                continue;
            };
            if !is_supported_target_arch_dtype(target_arch, &tt.dtype) {
                let dev = tt.encoding.as_deref().unwrap_or(op_dev);
                return Err(format!(
                    "The {} data type is not supported on device '{dev}'.",
                    tt.dtype
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arch_dtype_support_rules() {
        assert!(is_supported_target_arch_dtype(
            "x86_64-unknown-linux-gnu",
            &DType::F16
        ));
        assert!(is_supported_target_arch_dtype("x86_64", &DType::BF16));
        assert!(!is_supported_target_arch_dtype("x86_64", &DType::F8E4M3FN));

        assert!(is_supported_target_arch_dtype(
            "aarch64-apple-darwin",
            &DType::BF16
        ));
        assert!(!is_supported_target_arch_dtype(
            "aarch64-apple-darwin",
            &DType::F16
        ));
        assert!(is_supported_target_arch_dtype(
            "nvptx64-nvidia-cuda",
            &DType::F8E4M3FN
        ));
    }
}
