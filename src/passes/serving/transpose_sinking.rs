//! Composite activation recognition, transpose sinking, and DAX slot arenas.

use super::{extract_int_list_attr, stamp_fused_provenance};
use crate::ir::{
    Attribute, Module, OpId, OpName, Operation, TensorType, Type, Value, ValueDef, ValueId,
};
use indexmap::IndexMap;

pub(super) fn def_op_of(module: &Module, val: ValueId) -> Option<&Operation> {
    let v_info = module.values.get(val)?;
    let ValueDef::OpResult { op, .. } = v_info.def else {
        return None;
    };
    module.ops.get(op)
}

pub(super) fn match_splat_f64(module: &Module, mut val: ValueId) -> Option<f64> {
    for _ in 0..3 {
        let op = def_op_of(module, val)?;
        if matches!(
            op.name.to_string().as_str(),
            "stablehlo.broadcast_in_dim" | "mo.static.broadcast_to" | "mo.broadcast_to"
        ) && !op.operands.is_empty()
        {
            val = op.operands[0];
            continue;
        }
        if matches!(
            op.name.to_string().as_str(),
            "stablehlo.constant" | "arith.constant" | "mo.constant"
        ) {
            return op.attributes.get("value")?.as_splat_f64();
        }
        break;
    }
    None
}

pub(super) fn is_splat_approx(module: &Module, val: ValueId, target: f64, tol: f64) -> bool {
    match_splat_f64(module, val).is_some_and(|v| (v - target).abs() <= tol)
}

pub(super) fn is_binary_op(
    module: &Module,
    val: ValueId,
    names: &[&str],
) -> Option<(ValueId, ValueId, OpId)> {
    let op = def_op_of(module, val)?;
    if op.operands.len() == 2 && names.contains(&op.name.to_string().as_str()) {
        Some((op.operands[0], op.operands[1], op.id))
    } else {
        None
    }
}

pub(super) fn is_unary_op(
    module: &Module,
    val: ValueId,
    names: &[&str],
) -> Option<(ValueId, OpId)> {
    let op = def_op_of(module, val)?;
    if op.operands.len() == 1 && names.contains(&op.name.to_string().as_str()) {
        Some((op.operands[0], op.id))
    } else {
        None
    }
}

pub(super) fn match_comm_const_other(
    module: &Module,
    lhs: ValueId,
    rhs: ValueId,
    target: f64,
    tol: f64,
) -> Option<(ValueId, ValueId)> {
    if is_splat_approx(module, lhs, target, tol) {
        Some((rhs, lhs))
    } else if is_splat_approx(module, rhs, target, tol) {
        Some((lhs, rhs))
    } else {
        None
    }
}

pub(super) const ADD_OPS: &[&str] = &["stablehlo.add", "mo.add", "arith.addf"];
pub(super) const MUL_OPS: &[&str] = &["stablehlo.multiply", "mo.mul", "arith.mulf"];
pub(super) const DIV_OPS: &[&str] = &["stablehlo.divide", "mo.div", "arith.divf"];
pub(super) const NEG_OPS: &[&str] = &["stablehlo.negate", "mo.negative", "arith.negf"];
pub(super) const EXP_OPS: &[&str] = &["stablehlo.exponential", "mo.exp", "math.exp"];
pub(super) const SIGMOID_OPS: &[&str] = &["mo.sigmoid", "stablehlo.logistic"];
pub(super) const ERF_OPS: &[&str] = &["stablehlo.erf", "mo.erf", "onnx.Erf", "math.erf"];
pub(super) const TANH_OPS: &[&str] = &["stablehlo.tanh", "mo.tanh", "math.tanh"];
pub(super) const POW_OPS: &[&str] = &["stablehlo.power", "mo.pow", "math.powf"];

pub(super) fn match_exp_to_sigmoid(
    module: &Module,
    op: &Operation,
) -> Option<(ValueId, Vec<OpId>)> {
    if op.operands.len() != 2 || !DIV_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    let num = op.operands[0];
    let den = op.operands[1];
    if !is_splat_approx(module, num, 1.0, 1e-5) {
        return None;
    }
    let (add_l, add_r, add_id) = is_binary_op(module, den, ADD_OPS)?;
    let (exp_val, _) = match_comm_const_other(module, add_l, add_r, 1.0, 1e-5)?;
    let (neg_val, exp_id) = is_unary_op(module, exp_val, EXP_OPS)?;
    let (x_val, neg_id) = is_unary_op(module, neg_val, NEG_OPS)?;
    Some((x_val, vec![add_id, exp_id, neg_id]))
}

pub(super) fn match_sigmoid_mul_to_silu(
    module: &Module,
    op: &Operation,
) -> Option<(ValueId, Vec<OpId>)> {
    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (cand_sig, cand_x) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        if let Some((sig_in, sig_id)) = is_unary_op(module, cand_sig, SIGMOID_OPS)
            && sig_in == cand_x
        {
            return Some((cand_x, vec![sig_id]));
        }
    }
    None
}

pub(super) fn match_sigmoid_mul_to_gelu_quick(
    module: &Module,
    op: &Operation,
) -> Option<(ValueId, Vec<OpId>)> {
    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (cand_sig, cand_x) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        if let Some((sig_in, sig_id)) = is_unary_op(module, cand_sig, SIGMOID_OPS)
            && let Some((mul_l, mul_r, inner_mul_id)) = is_binary_op(module, sig_in, MUL_OPS)
            && let Some((inner_x, _)) = match_comm_const_other(module, mul_l, mul_r, 1.702, 1e-3)
            && inner_x == cand_x
        {
            return Some((cand_x, vec![sig_id, inner_mul_id]));
        }
    }
    None
}

pub(super) fn match_erf_to_gelu(module: &Module, op: &Operation) -> Option<(ValueId, Vec<OpId>)> {
    const SQRT2: f64 = std::f64::consts::SQRT_2;
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (half_side, erf_side) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        let Some((hl, hr, half_mul_id)) = is_binary_op(module, half_side, MUL_OPS) else {
            continue;
        };
        let Some((x_half, _)) = match_comm_const_other(module, hl, hr, 0.5, 1e-5) else {
            continue;
        };
        let Some((al, ar, add_id)) = is_binary_op(module, erf_side, ADD_OPS) else {
            continue;
        };
        let Some((erf_val, _)) = match_comm_const_other(module, al, ar, 1.0, 1e-5) else {
            continue;
        };
        let Some((erf_arg, erf_id)) = is_unary_op(module, erf_val, ERF_OPS) else {
            continue;
        };

        // Match either `x / sqrt(2)` or `x * (1 / sqrt(2))`.
        if let Some((dl, dr, scale_id)) = is_binary_op(module, erf_arg, DIV_OPS)
            && dl == x_half
            && is_splat_approx(module, dr, SQRT2, 1e-4)
        {
            return Some((x_half, vec![half_mul_id, add_id, erf_id, scale_id]));
        }
        if let Some((ml, mr, scale_id)) = is_binary_op(module, erf_arg, MUL_OPS)
            && let Some((x_erf, _)) = match_comm_const_other(module, ml, mr, INV_SQRT2, 1e-4)
            && x_erf == x_half
        {
            return Some((x_half, vec![half_mul_id, add_id, erf_id, scale_id]));
        }
    }
    None
}

pub(super) fn match_cube_of(module: &Module, val: ValueId, x: ValueId) -> Option<Vec<OpId>> {
    if let Some((pl, pr, pow_id)) = is_binary_op(module, val, POW_OPS)
        && pl == x
        && is_splat_approx(module, pr, 3.0, 1e-5)
    {
        return Some(vec![pow_id]);
    }
    let (cl, cr, cube_id) = is_binary_op(module, val, MUL_OPS)?;
    for (sq_cand, other) in [(cl, cr), (cr, cl)] {
        if other == x
            && let Some((sl, sr, sq_id)) = is_binary_op(module, sq_cand, MUL_OPS)
            && sl == x
            && sr == x
        {
            return Some(vec![cube_id, sq_id]);
        }
    }
    None
}

pub(super) fn match_tanh_to_gelu_tanh(
    module: &Module,
    op: &Operation,
) -> Option<(ValueId, Vec<OpId>)> {
    // sqrt(2 / pi) = 0.7978845608028654
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    const COEFF: f64 = 0.044715;

    if op.operands.len() != 2 || !MUL_OPS.contains(&op.name.to_string().as_str()) {
        return None;
    }
    for (half_side, tanh_side) in [
        (op.operands[0], op.operands[1]),
        (op.operands[1], op.operands[0]),
    ] {
        let Some((hl, hr, half_mul_id)) = is_binary_op(module, half_side, MUL_OPS) else {
            continue;
        };
        let Some((x_half, _)) = match_comm_const_other(module, hl, hr, 0.5, 1e-5) else {
            continue;
        };
        let Some((al, ar, outer_add_id)) = is_binary_op(module, tanh_side, ADD_OPS) else {
            continue;
        };
        let Some((tanh_val, _)) = match_comm_const_other(module, al, ar, 1.0, 1e-5) else {
            continue;
        };
        let Some((tanh_arg, tanh_id)) = is_unary_op(module, tanh_val, TANH_OPS) else {
            continue;
        };
        let Some((sl, sr, scale_mul_id)) = is_binary_op(module, tanh_arg, MUL_OPS) else {
            continue;
        };
        let Some((poly_val, _)) = match_comm_const_other(module, sl, sr, SQRT_2_OVER_PI, 1e-4)
        else {
            continue;
        };
        let Some((pl, pr, poly_add_id)) = is_binary_op(module, poly_val, ADD_OPS) else {
            continue;
        };
        for (cand_x, cand_term) in [(pl, pr), (pr, pl)] {
            if cand_x != x_half {
                continue;
            }
            let Some((tl, tr, term_mul_id)) = is_binary_op(module, cand_term, MUL_OPS) else {
                continue;
            };
            let Some((cube_val, _)) = match_comm_const_other(module, tl, tr, COEFF, 1e-5) else {
                continue;
            };
            if let Some(mut cube_ids) = match_cube_of(module, cube_val, x_half) {
                let mut dead = vec![
                    half_mul_id,
                    outer_add_id,
                    tanh_id,
                    scale_mul_id,
                    poly_add_id,
                    term_mul_id,
                ];
                dead.append(&mut cube_ids);
                return Some((x_half, dead));
            }
        }
    }
    None
}

/// Recognizes decomposed arithmetic subgraphs for `sigmoid`, `silu`, `gelu_quick`, `gelu`, and
/// `gelu_tanh` and folds each root operation into its canonical `mo.<activation>` operator while
/// pruning dead intermediate arithmetic nodes.
///
/// Patterns recognized:
/// - `1 / (1 + exp(-x))` -> `mo.sigmoid(x)`
/// - `x * sigmoid(x)` -> `mo.silu(x)`
/// - `x * sigmoid(1.702 * x)` -> `mo.gelu_quick(x)`
/// - `0.5 * x * (1 + erf(x / sqrt(2)))` -> `mo.gelu(x)`
/// - `0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))` -> `mo.gelu_tanh(x)`
///
/// Returns the number of composite activations recognized and folded.
pub fn recognize_composite_activations(module: &mut Module) -> usize {
    let mut folded = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        let op_ids = module.blocks[block_id].ops.clone();
        let mut candidate_dead: std::collections::HashSet<OpId> = std::collections::HashSet::new();

        for &op_id in &op_ids {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let matched: Option<(&str, ValueId, Vec<OpId>)> =
                if let Some((x, dead)) = match_exp_to_sigmoid(module, &op) {
                    Some(("sigmoid", x, dead))
                } else if let Some((x, dead)) = match_sigmoid_mul_to_silu(module, &op) {
                    Some(("silu", x, dead))
                } else if let Some((x, dead)) = match_sigmoid_mul_to_gelu_quick(module, &op) {
                    Some(("gelu_quick", x, dead))
                } else if let Some((x, dead)) = match_erf_to_gelu(module, &op) {
                    Some(("gelu", x, dead))
                } else if let Some((x, dead)) = match_tanh_to_gelu_tanh(module, &op) {
                    Some(("gelu_tanh", x, dead))
                } else {
                    None
                };

            let Some((act_name, x_val, dead_ids)) = matched else {
                continue;
            };

            let mut locs = vec![op.location.clone()];
            for &d_id in &dead_ids {
                if let Some(d_op) = module.ops.get(d_id) {
                    locs.push(d_op.location.clone());
                }
                candidate_dead.insert(d_id);
            }

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName::new("mo", act_name);
                op_mut.operands = smallvec::smallvec![x_val];
                op_mut.attributes.insert(
                    "castforge.composite_activation".to_string(),
                    Attribute::String(act_name.to_string()),
                );
                stamp_fused_provenance(op_mut, &locs);
            }
            folded += 1;
        }

        if !candidate_dead.is_empty() {
            // Sweep dead intermediate ops in reverse topological order.
            for &op_id in op_ids.iter().rev() {
                if !candidate_dead.contains(&op_id) {
                    continue;
                }
                let is_unused = module
                    .ops
                    .get(op_id)
                    .is_some_and(|op| op.results.iter().all(|&r| module.use_count(r) == 0));
                if is_unused {
                    module.remove_op(block_id, op_id);
                }
            }
        }
    }

    folded
}

/// Applies the inverse of `perm` to `items` (`result[perm[i]] = items[i]`).
///
/// Returns `None` if `items.len() != perm.len()` or if `perm` is not a valid permutation of
/// `0..items.len()`.
pub fn unpermute_slice<T: Clone>(items: &[T], perm: &[usize]) -> Option<Vec<T>> {
    let n = items.len();
    if perm.len() != n {
        return None;
    }
    let mut out: Vec<Option<T>> = vec![None; n];
    for (idx, &perm_idx) in perm.iter().enumerate() {
        if perm_idx >= n || out[perm_idx].is_some() {
            return None;
        }
        out[perm_idx] = Some(items[idx].clone());
    }
    out.into_iter().collect()
}

pub(super) fn extract_transpose_perm(op: &Operation) -> Option<Vec<usize>> {
    if !matches!(
        op.name.to_string().as_str(),
        "stablehlo.transpose" | "mo.transpose" | "onnx.Transpose"
    ) || op.operands.is_empty()
        || op.results.len() != 1
    {
        return None;
    }
    let raw =
        extract_int_list_attr(op, "permutation").or_else(|| extract_int_list_attr(op, "perm"))?;
    let mut perm = Vec::with_capacity(raw.len());
    for v in raw {
        if v < 0 {
            return None;
        }
        perm.push(v as usize);
    }
    let n = perm.len();
    let mut seen = vec![false; n];
    for &p in &perm {
        if p >= n || seen[p] {
            return None;
        }
        seen[p] = true;
    }
    Some(perm)
}

pub(super) fn is_sinkable_unary_elementwise(op: &Operation) -> bool {
    op.operands.len() == 1
        && op.results.len() == 1
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.tanh"
                | "stablehlo.exponential"
                | "stablehlo.negate"
                | "stablehlo.abs"
                | "stablehlo.log"
                | "stablehlo.sqrt"
                | "stablehlo.rsqrt"
                | "stablehlo.logistic"
                | "stablehlo.erf"
                | "stablehlo.convert"
                | "mo.relu"
                | "mo.sigmoid"
                | "mo.silu"
                | "mo.gelu"
                | "mo.gelu_quick"
                | "mo.gelu_tanh"
                | "mo.tanh"
                | "mo.exp"
                | "mo.negative"
        )
}

pub(super) fn is_sinkable_binary_elementwise(op: &Operation) -> bool {
    op.operands.len() == 2
        && op.results.len() == 1
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.add"
                | "stablehlo.subtract"
                | "stablehlo.multiply"
                | "stablehlo.divide"
                | "stablehlo.maximum"
                | "stablehlo.minimum"
                | "stablehlo.power"
                | "mo.add"
                | "mo.sub"
                | "mo.mul"
                | "mo.div"
                | "mo.max"
                | "mo.min"
                | "mo.pow"
        )
}

pub(super) fn is_sinkable_broadcast_op(op: &Operation) -> bool {
    op.operands.len() == 1
        && op.results.len() == 1
        && matches!(
            op.name.to_string().as_str(),
            "stablehlo.broadcast_in_dim" | "mo.static.broadcast_to" | "mo.broadcast_to"
        )
}

/// Sinks sole-use `transpose` operations past unary elementwise ops, matching-permutation binary
/// elementwise ops (coalescing two input transposes into one output transpose), and static
/// broadcasts (padding the permutation for leading broadcast dimensions and unpermuting the
/// intermediate broadcast shape).
///
/// Returns the number of transpose-sinking rewrites applied.
pub fn sink_transpose_through_elementwise_and_broadcast(module: &mut Module) -> usize {
    let mut sunk_count = 0usize;
    let block_ids: Vec<_> = module.blocks.keys().collect();

    for block_id in block_ids {
        loop {
            let mut changed_in_pass = false;
            let op_ids = module.blocks[block_id].ops.clone();

            for &op_id in &op_ids {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.results.len() != 1 {
                    continue;
                }
                let orig_res_val = op.results[0];
                let Some(orig_res_tt) = module
                    .values
                    .get(orig_res_val)
                    .and_then(|v| v.ty.as_tensor())
                    .cloned()
                else {
                    continue;
                };

                // Case 1: Unary `unary(transpose(x, perm))` -> `transpose(unary(x), perm)`
                if is_sinkable_unary_elementwise(&op) {
                    let trans_val = op.operands[0];
                    if module.use_count(trans_val) != 1 {
                        continue;
                    }
                    let Some(trans_op) = def_op_of(module, trans_val).cloned() else {
                        continue;
                    };
                    if extract_transpose_perm(&trans_op).is_none() {
                        continue;
                    }
                    if trans_op.attributes.get("castforge.device")
                        != op.attributes.get("castforge.device")
                    {
                        continue;
                    }
                    let raw_in = trans_op.operands[0];
                    let Some(raw_in_tt) = module
                        .values
                        .get(raw_in)
                        .and_then(|v| v.ty.as_tensor())
                        .cloned()
                    else {
                        continue;
                    };

                    let inter_tt = TensorType {
                        shape: raw_in_tt.shape,
                        dtype: orig_res_tt.dtype.clone(),
                        encoding: orig_res_tt.encoding.clone(),
                    };
                    let inter_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: Type::Tensor(inter_tt),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: 0,
                        },
                    });

                    if let Some(u_mut) = module.ops.get_mut(op_id) {
                        u_mut.operands[0] = raw_in;
                        u_mut.results[0] = inter_val;
                    }
                    if let Some(t_mut) = module.ops.get_mut(trans_op.id) {
                        t_mut.operands[0] = inter_val;
                        t_mut.results[0] = orig_res_val;
                    }
                    if let Some(v_mut) = module.values.get_mut(orig_res_val) {
                        v_mut.def = ValueDef::OpResult {
                            op: trans_op.id,
                            index: 0,
                        };
                    }
                    module.values.remove(trans_val);

                    let blk = &mut module.blocks[block_id].ops;
                    blk.retain(|&id| id != trans_op.id);
                    if let Some(pos) = blk.iter().position(|&id| id == op_id) {
                        blk.insert(pos + 1, trans_op.id);
                    }

                    sunk_count += 1;
                    changed_in_pass = true;
                    break;
                }

                // Case 2: Binary elementwise `bin(transpose(x, p), transpose(y, p))`
                // -> `transpose(bin(x, y), p)`
                if is_sinkable_binary_elementwise(&op) {
                    let lhs_val = op.operands[0];
                    let rhs_val = op.operands[1];
                    if lhs_val == rhs_val
                        || module.use_count(lhs_val) != 1
                        || module.use_count(rhs_val) != 1
                    {
                        continue;
                    }
                    let (Some(t1_op), Some(t2_op)) = (
                        def_op_of(module, lhs_val).cloned(),
                        def_op_of(module, rhs_val).cloned(),
                    ) else {
                        continue;
                    };
                    let (Some(perm1), Some(perm2)) = (
                        extract_transpose_perm(&t1_op),
                        extract_transpose_perm(&t2_op),
                    ) else {
                        continue;
                    };
                    if perm1 != perm2 {
                        continue;
                    }
                    let dev = op.attributes.get("castforge.device");
                    if t1_op.attributes.get("castforge.device") != dev
                        || t2_op.attributes.get("castforge.device") != dev
                    {
                        continue;
                    }
                    let in1 = t1_op.operands[0];
                    let in2 = t2_op.operands[0];
                    let (Some(in1_tt), Some(in2_tt)) = (
                        module
                            .values
                            .get(in1)
                            .and_then(|v| v.ty.as_tensor())
                            .cloned(),
                        module
                            .values
                            .get(in2)
                            .and_then(|v| v.ty.as_tensor())
                            .cloned(),
                    ) else {
                        continue;
                    };
                    if in1_tt.shape != in2_tt.shape {
                        continue;
                    }

                    let inter_tt = TensorType {
                        shape: in1_tt.shape,
                        dtype: orig_res_tt.dtype.clone(),
                        encoding: orig_res_tt.encoding.clone(),
                    };
                    let inter_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: Type::Tensor(inter_tt),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: 0,
                        },
                    });

                    if let Some(b_mut) = module.ops.get_mut(op_id) {
                        b_mut.operands[0] = in1;
                        b_mut.operands[1] = in2;
                        b_mut.results[0] = inter_val;
                    }
                    let locs = vec![t1_op.location.clone(), t2_op.location.clone()];
                    if let Some(t1_mut) = module.ops.get_mut(t1_op.id) {
                        t1_mut.operands[0] = inter_val;
                        t1_mut.results[0] = orig_res_val;
                        stamp_fused_provenance(t1_mut, &locs);
                    }
                    if let Some(v_mut) = module.values.get_mut(orig_res_val) {
                        v_mut.def = ValueDef::OpResult {
                            op: t1_op.id,
                            index: 0,
                        };
                    }
                    module.values.remove(lhs_val);
                    module.remove_op(block_id, t2_op.id);

                    let blk = &mut module.blocks[block_id].ops;
                    blk.retain(|&id| id != t1_op.id);
                    if let Some(pos) = blk.iter().position(|&id| id == op_id) {
                        blk.insert(pos + 1, t1_op.id);
                    }

                    sunk_count += 1;
                    changed_in_pass = true;
                    break;
                }

                // Case 3: Static broadcast `broadcast(transpose(x, perm))`
                // -> `transpose(broadcast(x), padded_perm)`
                if is_sinkable_broadcast_op(&op) {
                    let trans_val = op.operands[0];
                    if module.use_count(trans_val) != 1 {
                        continue;
                    }
                    let Some(trans_op) = def_op_of(module, trans_val).cloned() else {
                        continue;
                    };
                    let Some(perm) = extract_transpose_perm(&trans_op) else {
                        continue;
                    };
                    if trans_op.attributes.get("castforge.device")
                        != op.attributes.get("castforge.device")
                    {
                        continue;
                    }
                    let raw_in = trans_op.operands[0];
                    let Some(trans_tt) = module
                        .values
                        .get(trans_val)
                        .and_then(|v| v.ty.as_tensor())
                        .cloned()
                    else {
                        continue;
                    };
                    if orig_res_tt.shape.len() < trans_tt.shape.len() {
                        continue;
                    }
                    // If `broadcast_dimensions` is present on `stablehlo.broadcast_in_dim`,
                    // verify it represents standard right-aligned broadcasting `[extra..out_rank]`.
                    let extra_dims = orig_res_tt.shape.len() - trans_tt.shape.len();
                    if let Some(bcast_dims) = extract_int_list_attr(&op, "broadcast_dimensions") {
                        let expected: Vec<i64> =
                            (extra_dims as i64..orig_res_tt.shape.len() as i64).collect();
                        if bcast_dims != expected {
                            continue;
                        }
                    }

                    let mut padded_perm: Vec<usize> = (0..extra_dims).collect();
                    for p in perm {
                        padded_perm.push(p + extra_dims);
                    }
                    let Some(new_bcast_shape) = unpermute_slice(&orig_res_tt.shape, &padded_perm)
                    else {
                        continue;
                    };

                    let inter_tt = TensorType {
                        shape: new_bcast_shape,
                        dtype: orig_res_tt.dtype.clone(),
                        encoding: orig_res_tt.encoding.clone(),
                    };
                    let inter_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: Type::Tensor(inter_tt),
                        def: ValueDef::OpResult {
                            op: op_id,
                            index: 0,
                        },
                    });

                    if let Some(b_mut) = module.ops.get_mut(op_id) {
                        b_mut.operands[0] = raw_in;
                        b_mut.results[0] = inter_val;
                    }
                    let new_perm_i64: Vec<i64> = padded_perm.iter().map(|&p| p as i64).collect();
                    if let Some(t_mut) = module.ops.get_mut(trans_op.id) {
                        t_mut.operands[0] = inter_val;
                        t_mut.results[0] = orig_res_val;
                        let key = if t_mut.attributes.contains_key("perm") {
                            "perm"
                        } else {
                            "permutation"
                        };
                        t_mut
                            .attributes
                            .insert(key.to_string(), Attribute::DenseIntArray(new_perm_i64));
                    }
                    if let Some(v_mut) = module.values.get_mut(orig_res_val) {
                        v_mut.def = ValueDef::OpResult {
                            op: trans_op.id,
                            index: 0,
                        };
                    }
                    module.values.remove(trans_val);

                    let blk = &mut module.blocks[block_id].ops;
                    blk.retain(|&id| id != trans_op.id);
                    if let Some(pos) = blk.iter().position(|&id| id == op_id) {
                        blk.insert(pos + 1, trans_op.id);
                    }

                    sunk_count += 1;
                    changed_in_pass = true;
                    break;
                }
            }

            if !changed_in_pass {
                break;
            }
        }
    }

    sunk_count
}

/// Object-group role in a hybrid CacheBlend registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendObjectGroupKind {
    /// Full-attention KV object group read by both the prefix and non-prefix blend legs.
    Attention,
    /// Connector-private auxiliary object group read alongside attention on the blend leg.
    Aux,
    /// Recurrent/SSM state object group read only on the contiguous prefix leg.
    Recurrent,
}

/// Per-leg object-group read routing for CacheBlend hybrid lookups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlendReadGroupSet {
    /// Ascending group IDs read by the relocatable blend leg (`attention + aux`).
    pub blend_gids: Vec<usize>,
    /// Ascending group IDs read by the contiguous prefix leg (`attention + recurrent`).
    pub prefix_gids: Vec<usize>,
    /// Ascending group IDs holding recurrent state alone.
    pub recurrent_gids: Vec<usize>,
    /// The single full-attention object group ID.
    pub attn_gid: usize,
}

/// Classifies a registration's `num_object_groups` and `group_kinds` into per-leg read sets.
///
/// Single-group (`<= 1`) fused layouts map both legs to group `0`. Multi-group layouts must
/// contain exactly one [`BlendObjectGroupKind::Attention`] group and at most one
/// [`BlendObjectGroupKind::Aux`] group.
pub fn classify_blend_read_groups(
    num_object_groups: usize,
    group_kinds: &[BlendObjectGroupKind],
) -> Option<BlendReadGroupSet> {
    if num_object_groups <= 1 {
        return Some(BlendReadGroupSet {
            blend_gids: vec![0],
            prefix_gids: vec![0],
            recurrent_gids: Vec::new(),
            attn_gid: 0,
        });
    }
    if group_kinds.len() != num_object_groups {
        return None;
    }
    let mut attn = Vec::new();
    let mut aux = Vec::new();
    let mut recurrent = Vec::new();
    for (idx, &kind) in group_kinds.iter().enumerate() {
        match kind {
            BlendObjectGroupKind::Attention => attn.push(idx),
            BlendObjectGroupKind::Aux => aux.push(idx),
            BlendObjectGroupKind::Recurrent => recurrent.push(idx),
        }
    }
    if attn.len() != 1 || aux.len() > 1 {
        return None;
    }
    let attn_gid = attn[0];
    let mut blend_gids = attn.clone();
    blend_gids.extend_from_slice(&aux);
    blend_gids.sort_unstable();

    let mut prefix_gids = attn;
    prefix_gids.extend_from_slice(&recurrent);
    prefix_gids.sort_unstable();

    Some(BlendReadGroupSet {
        blend_gids,
        prefix_gids,
        recurrent_gids: recurrent,
        attn_gid,
    })
}

/// Interleaves per-group expanded object keys into a chunk-major key sequence with uniform stride
/// `per_group_keys.len() * expansion` per chunk so prefix bitmaps stay leading-ones-aligned.
pub fn interleave_chunk_major_object_keys<T: Clone>(
    per_group_keys: &[Vec<T>],
    num_chunks: usize,
) -> Option<Vec<T>> {
    if per_group_keys.is_empty() || num_chunks == 0 {
        return Some(Vec::new());
    }
    let first_len = per_group_keys[0].len();
    if first_len == 0 || !first_len.is_multiple_of(num_chunks) {
        return None;
    }
    if per_group_keys.iter().any(|g| g.len() != first_len) {
        return None;
    }
    let expansion = first_len / num_chunks;
    let mut out = Vec::with_capacity(per_group_keys.len() * first_len);
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * expansion;
        let end = start + expansion;
        for group_keys in per_group_keys {
            out.extend_from_slice(&group_keys[start..end]);
        }
    }
    Some(out)
}

/// Resolved per-group RoPE geometry for CacheBlend in-place key rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CBGroupRopeGeometry {
    /// Whether K and V are fused-packed into a doubled `2 * head_size` per-head stride.
    pub fused_packed: bool,
    /// Element stride per head (`2 * head_size` when `fused_packed`, `hidden_dim` for MLA, else
    /// `head_size`).
    pub per_head_elems: usize,
    /// Number of heads rotated across the row (`1` for MLA latents).
    pub num_heads: usize,
    /// Starting element offset of the rotated slice within each head (`> 0` for MLA).
    pub rot_offset: usize,
}

/// Computes the per-group CacheBlend RoPE geometry across split K/V, fused-packed K/V, and
/// Multi-head Latent Attention (MLA `[content | rope]`) layouts.
#[allow(clippy::too_many_arguments)]
pub fn compute_cb_group_rope_geometry(
    tokens_per_block: usize,
    slots_per_block: usize,
    kv_size: usize,
    hidden_dim: usize,
    head_size: usize,
    is_fused_format: bool,
    declared_hs: Option<usize>,
    rot_window: Option<(usize, usize)>,
) -> Option<CBGroupRopeGeometry> {
    if tokens_per_block == 0
        || tokens_per_block != slots_per_block
        || hidden_dim == 0
        || head_size == 0
    {
        return None;
    }
    if let Some((rot_offset, rot_width)) = rot_window
        && rot_offset > 0
    {
        if kv_size != 1 || rot_width == 0 || rot_offset + rot_width != hidden_dim {
            return None;
        }
        return Some(CBGroupRopeGeometry {
            fused_packed: false,
            per_head_elems: hidden_dim,
            num_heads: 1,
            rot_offset,
        });
    }
    if !matches!(kv_size, 1 | 2) {
        return None;
    }
    let fused_packed = is_fused_format && declared_hs.is_none_or(|hs| hs == 2 * head_size);
    let per_head_elems = head_size * if fused_packed { 2 } else { 1 };
    let num_heads = hidden_dim / per_head_elems;
    if num_heads == 0 || num_heads * per_head_elems != hidden_dim {
        return None;
    }
    Some(CBGroupRopeGeometry {
        fused_packed,
        per_head_elems,
        num_heads,
        rot_offset: 0,
    })
}

/// Outcome of reserving or committing a payload in [`DaxGenerationSlotArena`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaxPutOutcome {
    /// Payload was newly reserved/committed into a slot.
    Inserted,
    /// Key is already committed in the index.
    AlreadyExists,
    /// Key already has an active uncommitted write reservation.
    Inflight,
    /// Arena has no free slots available.
    NoSpace,
    /// Payload size is zero or exceeds `slot_bytes`.
    InvalidSize,
}

/// Read reservation pinning a committed DAX slot generation against concurrent recycling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DaxReadReservation {
    /// Position of the key in the caller's request slice.
    pub result_index: usize,
    /// Reserved key identifier.
    pub key: u64,
    /// Byte offset of the slot within the arena (`slot_id * slot_bytes`).
    pub offset_bytes: u64,
    /// Payload size in bytes stored in the slot.
    pub size_bytes: u64,
    /// Physical slot index.
    pub slot_id: usize,
    /// Monotonic slot generation counter at reservation time.
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DaxIndexEntry {
    offset_bytes: u64,
    size_bytes: u64,
    slot_id: usize,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DaxInflightEntry {
    offset_bytes: u64,
    size_bytes: u64,
    slot_id: usize,
    generation: u64,
    canceled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DaxSlotState {
    generation: u64,
    committed: bool,
    borrow_count: u32,
    pending_free: bool,
}

/// Generational fixed-slot persistent memory (DAX) arena manager with read-borrow reference
/// counting and deferred `pending_free` slot reclamation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaxGenerationSlotArena {
    slot_bytes: u64,
    max_slots: usize,
    next_slot: usize,
    free_slots: Vec<usize>,
    index: IndexMap<u64, DaxIndexEntry>,
    inflight: IndexMap<u64, DaxInflightEntry>,
    slot_states: IndexMap<usize, DaxSlotState>,
    external_lock_counts: IndexMap<u64, u32>,
    active_reads: usize,
    active_writes: usize,
}

impl DaxGenerationSlotArena {
    /// Creates a new arena with `max_dax_size_bytes` capacity partitioned into fixed `slot_bytes`
    /// slots. Returns `None` if `slot_bytes == 0` or the arena cannot fit at least one slot.
    pub fn new(max_dax_size_bytes: u64, slot_bytes: u64) -> Option<Self> {
        if slot_bytes == 0 {
            return None;
        }
        let max_slots = (max_dax_size_bytes / slot_bytes) as usize;
        if max_slots == 0 {
            return None;
        }
        Some(Self {
            slot_bytes,
            max_slots,
            next_slot: 0,
            free_slots: Vec::new(),
            index: IndexMap::new(),
            inflight: IndexMap::new(),
            slot_states: IndexMap::new(),
            external_lock_counts: IndexMap::new(),
            active_reads: 0,
            active_writes: 0,
        })
    }

    /// Maximum number of fixed-size slots in the arena.
    pub fn max_slots(&self) -> usize {
        self.max_slots
    }

    /// Fixed byte capacity of each slot.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    fn allocate_slot(&mut self) -> Option<usize> {
        if let Some(slot_id) = self.free_slots.pop() {
            return Some(slot_id);
        }
        if self.next_slot < self.max_slots {
            let slot_id = self.next_slot;
            self.next_slot += 1;
            return Some(slot_id);
        }
        None
    }

    fn reserve_slot_state(&mut self, slot_id: usize) -> u64 {
        let next_gen = self
            .slot_states
            .get(&slot_id)
            .map_or(1, |s| s.generation + 1);
        self.slot_states.insert(
            slot_id,
            DaxSlotState {
                generation: next_gen,
                committed: false,
                borrow_count: 0,
                pending_free: false,
            },
        );
        next_gen
    }

    fn schedule_slot_reclaim(&mut self, slot_id: usize, generation: u64) {
        let Some(state) = self.slot_states.get_mut(&slot_id) else {
            return;
        };
        if state.generation != generation {
            return;
        }
        state.committed = false;
        if state.borrow_count == 0 {
            state.pending_free = false;
            if !self.free_slots.contains(&slot_id) {
                self.free_slots.push(slot_id);
            }
        } else {
            state.pending_free = true;
        }
    }

    fn is_entry_committed(&self, entry: &DaxIndexEntry) -> bool {
        self.slot_states
            .get(&entry.slot_id)
            .is_some_and(|s| s.generation == entry.generation && s.committed)
    }

    /// Reserves an in-flight write slot for `key` of `size_bytes`, returning the outcome and
    /// `Some((slot_id, generation, offset_bytes))` when a new slot was allocated.
    pub fn begin_put(
        &mut self,
        key: u64,
        size_bytes: u64,
    ) -> (DaxPutOutcome, Option<(usize, u64, u64)>) {
        if size_bytes == 0 || size_bytes > self.slot_bytes {
            return (DaxPutOutcome::InvalidSize, None);
        }
        if self.index.contains_key(&key) {
            return (DaxPutOutcome::AlreadyExists, None);
        }
        if self.inflight.contains_key(&key) {
            return (DaxPutOutcome::Inflight, None);
        }
        let Some(slot_id) = self.allocate_slot() else {
            return (DaxPutOutcome::NoSpace, None);
        };
        let offset_bytes = (slot_id as u64) * self.slot_bytes;
        let generation = self.reserve_slot_state(slot_id);
        self.inflight.insert(
            key,
            DaxInflightEntry {
                offset_bytes,
                size_bytes,
                slot_id,
                generation,
                canceled: false,
            },
        );
        self.active_writes += 1;
        (
            DaxPutOutcome::Inserted,
            Some((slot_id, generation, offset_bytes)),
        )
    }

    /// Finalizes an in-flight write for `key`. Commits the slot if `!write_failed` and the write
    /// was not force-canceled; otherwise reclaims the slot.
    pub fn finalize_put(&mut self, key: u64, write_failed: bool) -> bool {
        self.active_writes = self.active_writes.saturating_sub(1);
        let Some(inflight) = self.inflight.shift_remove(&key) else {
            return false;
        };
        if inflight.canceled || write_failed {
            self.schedule_slot_reclaim(inflight.slot_id, inflight.generation);
            return false;
        }
        if let Some(state) = self.slot_states.get_mut(&inflight.slot_id)
            && state.generation == inflight.generation
        {
            state.committed = true;
            state.pending_free = false;
        }
        self.index.insert(
            key,
            DaxIndexEntry {
                offset_bytes: inflight.offset_bytes,
                size_bytes: inflight.size_bytes,
                slot_id: inflight.slot_id,
                generation: inflight.generation,
            },
        );
        true
    }

    /// Synchronously allocates and commits a slot for `key` (`begin_put` + `finalize_put`).
    pub fn put(&mut self, key: u64, size_bytes: u64) -> DaxPutOutcome {
        let (outcome, reserved) = self.begin_put(key, size_bytes);
        if reserved.is_some() {
            self.finalize_put(key, false);
        }
        outcome
    }

    /// Checks whether `keys` are committed in the arena, optionally incrementing external lock
    /// reference counts for hits.
    pub fn exists_many(&mut self, keys: &[u64], lock: bool) -> Vec<bool> {
        let mut results = vec![false; keys.len()];
        for (i, &key) in keys.iter().enumerate() {
            let Some(&entry) = self.index.get(&key) else {
                continue;
            };
            if !self.is_entry_committed(&entry) {
                continue;
            }
            results[i] = true;
            if lock {
                *self.external_lock_counts.entry(key).or_insert(0) += 1;
            }
        }
        results
    }

    /// Decrements external lock reference counts for `keys`.
    pub fn unlock_many(&mut self, keys: &[u64]) {
        for &key in keys {
            if let Some(count) = self.external_lock_counts.get_mut(&key) {
                if *count <= 1 {
                    self.external_lock_counts.shift_remove(&key);
                } else {
                    *count -= 1;
                }
            }
        }
    }

    /// Reserves committed entries for a read pipeline, incrementing each hit slot's
    /// `borrow_count` so concurrent eviction cannot recycle the physical slot until
    /// [`Self::finalize_reads`] is called.
    pub fn reserve_reads(
        &mut self,
        keys: &[u64],
        prefix_only: bool,
    ) -> (Vec<DaxReadReservation>, Vec<bool>) {
        let mut reservations = Vec::new();
        let mut hits = vec![false; keys.len()];

        for (i, &key) in keys.iter().enumerate() {
            let Some(&entry) = self.index.get(&key) else {
                if prefix_only {
                    break;
                }
                continue;
            };
            let Some(state) = self.slot_states.get_mut(&entry.slot_id) else {
                if prefix_only {
                    break;
                }
                continue;
            };
            if state.generation != entry.generation || !state.committed {
                if prefix_only {
                    break;
                }
                continue;
            }
            state.borrow_count += 1;
            reservations.push(DaxReadReservation {
                result_index: i,
                key,
                offset_bytes: entry.offset_bytes,
                size_bytes: entry.size_bytes,
                slot_id: entry.slot_id,
                generation: entry.generation,
            });
            hits[i] = true;
        }
        if !reservations.is_empty() {
            self.active_reads += 1;
        }
        (reservations, hits)
    }

    /// Releases read reservations and recycles any `pending_free` slots whose `borrow_count`
    /// reaches zero.
    pub fn finalize_reads(&mut self, reservations: &[DaxReadReservation]) {
        if reservations.is_empty() {
            return;
        }
        self.active_reads = self.active_reads.saturating_sub(1);
        for res in reservations {
            let Some(state) = self.slot_states.get_mut(&res.slot_id) else {
                continue;
            };
            if state.generation != res.generation {
                continue;
            };
            state.borrow_count = state.borrow_count.saturating_sub(1);
            if state.pending_free && state.borrow_count == 0 {
                state.pending_free = false;
                if !self.free_slots.contains(&res.slot_id) {
                    self.free_slots.push(res.slot_id);
                }
            }
        }
    }

    /// Deletes `keys` from the index. When `!force`, externally locked and in-flight keys are
    /// skipped. Slots with active read borrows enter `pending_free` until borrows drain.
    pub fn delete_many(&mut self, keys: &[u64], force: bool) -> Vec<bool> {
        let mut results = vec![false; keys.len()];
        for (i, &key) in keys.iter().enumerate() {
            if let Some(inflight) = self.inflight.get_mut(&key) {
                if !force {
                    continue;
                }
                inflight.canceled = true;
                self.external_lock_counts.shift_remove(&key);
                results[i] = true;
                continue;
            }
            let Some(&entry) = self.index.get(&key) else {
                continue;
            };
            if !force && self.external_lock_counts.get(&key).copied().unwrap_or(0) > 0 {
                continue;
            }
            self.index.shift_remove(&key);
            self.external_lock_counts.shift_remove(&key);
            self.schedule_slot_reclaim(entry.slot_id, entry.generation);
            results[i] = true;
        }
        results
    }

    /// Returns `(current_usage, usage_after_ongoing_eviction)` as fractions of `max_slots`.
    /// The second value subtracts `pending_free` slots that have been evicted from the index but
    /// are waiting for active read borrows to drain.
    pub fn usage(&self) -> (f64, f64) {
        if self.max_slots == 0 {
            return (0.0, 0.0);
        }
        let live_slot_count = self.next_slot.saturating_sub(self.free_slots.len());
        let pending_reclaim_count = self.slot_states.values().filter(|s| s.pending_free).count();
        let current_usage = (live_slot_count as f64) / (self.max_slots as f64);
        let post_eviction_slots = live_slot_count.saturating_sub(pending_reclaim_count);
        let usage_after_eviction = (post_eviction_slots as f64) / (self.max_slots as f64);
        (current_usage, usage_after_eviction)
    }

    /// Returns `true` iff every committed, non-canceled in-flight, or borrowed slot has
    /// `slot_id < new_max_slots`.
    pub fn can_shrink_to(&self, new_max_slots: usize) -> bool {
        if new_max_slots == 0 {
            return false;
        }
        for entry in self.index.values() {
            if self.is_entry_committed(entry) && entry.slot_id >= new_max_slots {
                return false;
            }
        }
        for inflight in self.inflight.values() {
            if !inflight.canceled && inflight.slot_id >= new_max_slots {
                return false;
            }
        }
        for (&slot_id, state) in &self.slot_states {
            if slot_id >= new_max_slots && state.borrow_count > 0 {
                return false;
            }
        }
        true
    }
}

/// Default chunk-count histogram upper bounds (`0, 1, 2, 4, ..., 2048`).
pub const CHUNK_COUNT_HISTOGRAM_BOUNDS: &[f64] = &[
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
];

/// Fixed-bound bucket histogram with Prometheus-compatible cumulative bucket export.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketHistogram {
    bounds: Vec<f64>,
    bucket_counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl BucketHistogram {
    /// Creates an empty histogram with strictly ascending `bounds`. Returns `None` if `bounds` is
    /// empty or not strictly ascending.
    pub fn with_bounds(bounds: &[f64]) -> Option<Self> {
        if bounds.is_empty() || bounds.windows(2).any(|w| w[0] >= w[1] || w[0].is_nan()) {
            return None;
        }
        if bounds.last().is_some_and(|b| b.is_nan()) {
            return None;
        }
        Some(Self {
            bounds: bounds.to_vec(),
            bucket_counts: vec![0; bounds.len() + 1],
            sum: 0.0,
            count: 0,
        })
    }

    /// Records one observation of `value` into the first bucket with `bound >= value` (or the
    /// overflow `+Inf` bucket).
    pub fn observe(&mut self, value: f64) {
        let idx = self.bounds.partition_point(|&b| b < value);
        self.bucket_counts[idx] += 1;
        self.sum += value;
        self.count += 1;
    }

    /// Total number of recorded observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sum of all recorded observation values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Returns cumulative `(le_label, cumulative_count)` pairs followed by `("+Inf", count)`.
    pub fn cumulative_buckets(&self) -> Vec<(String, u64)> {
        let mut out = Vec::with_capacity(self.bounds.len() + 1);
        let mut acc = 0u64;
        for (&bound, &cnt) in self.bounds.iter().zip(self.bucket_counts.iter()) {
            acc += cnt;
            out.push((format!("{bound}"), acc));
        }
        out.push(("+Inf".to_string(), self.count));
        out
    }
}
