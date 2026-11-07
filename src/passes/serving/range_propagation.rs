//! Symbolic parameter range inference, range-proven algebraic simplification, and signature sync.
//!
//! Bounds parameter expression nesting depth (`MAX_PARAM_EXPR_NESTING_DEPTH = 4`,
//! `can_param_operator_be_nested`), propagates closed 64-bit integer intervals
//! ([`ConstantIntRange`]) through `Add`, `Mul`, `MulNoWrap`, `Div`, `Mod`, `Shl`, `Shr`, `And`,
//! `Or`, `Xor`, `Max`, `Min`, `Eq`, `Lt`, `Le`, and `Cond`, simplifies symbolic parameter
//! expressions using range proofs (`<max(add(A, B), B)> -> <add(A, B)>` when `A, B >= 0`,
//! constant-condition `Cond` folding, single-point range collapse), and synchronizes enclosing
//! graph/function signatures (`update_graph_signature_from_body`).

use crate::ir::{Attribute, Module, Type};
use indexmap::IndexMap;

/// Maximum nesting depth allowed when folding or inlining symbolic parameter expressions
/// (`kMaxDepth = 4`).
pub const MAX_PARAM_EXPR_NESTING_DEPTH: usize = 4;

/// Closed signed 64-bit integer interval `[min, max]` for symbolic parameter range inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantIntRange {
    /// Inclusive lower bound.
    pub min: i64,
    /// Inclusive upper bound.
    pub max: i64,
}

impl ConstantIntRange {
    /// Unconstrained 64-bit signed range `[i64::MIN, i64::MAX]`.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            min: i64::MIN,
            max: i64::MAX,
        }
    }

    /// Point interval `[val, val]`.
    #[must_use]
    pub const fn exact(val: i64) -> Self {
        Self { min: val, max: val }
    }

    /// Non-negative interval `[0, i64::MAX]`.
    #[must_use]
    pub const fn non_negative() -> Self {
        Self {
            min: 0,
            max: i64::MAX,
        }
    }

    /// Strictly positive interval `[1, i64::MAX]`.
    #[must_use]
    pub const fn positive() -> Self {
        Self {
            min: 1,
            max: i64::MAX,
        }
    }

    /// Creates a range `[min, max]`, clamping `max >= min`.
    #[must_use]
    pub const fn new(min: i64, max: i64) -> Self {
        if min <= max {
            Self { min, max }
        } else {
            Self { min: max, max: min }
        }
    }

    /// Returns `Some(val)` when `min == max`.
    #[must_use]
    pub const fn single_value(&self) -> Option<i64> {
        if self.min == self.max {
            Some(self.min)
        } else {
            None
        }
    }

    /// Returns `true` when every value in the range is `>= 0`.
    #[must_use]
    pub const fn is_non_negative(&self) -> bool {
        self.min >= 0
    }

    /// Returns the hull `[self.min.min(other.min), self.max.max(other.max)]`.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Symbolic parameter expression AST supporting all operators handled by `PropagateShapesPass.cpp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicParamExpr {
    /// Constant 64-bit integer literal.
    Const(i64),
    /// Named symbolic parameter (e.g. `"batch"`, `"seq_len"`).
    Sym(String),
    /// Wrapping or general integer addition (`lhs + rhs`).
    Add(Box<Self>, Box<Self>),
    /// General integer multiplication (`lhs * rhs`).
    Mul(Box<Self>, Box<Self>),
    /// No-Signed-Wrap multiplication (`mul_no_wrap(lhs, rhs)`).
    MulNoWrap(Box<Self>, Box<Self>),
    /// Floor/truncating integer division (`lhs / rhs`).
    Div(Box<Self>, Box<Self>),
    /// Remainder (`lhs % rhs`).
    Mod(Box<Self>, Box<Self>),
    /// Left shift (`lhs << rhs`).
    Shl(Box<Self>, Box<Self>),
    /// Arithmetic right shift (`lhs >> rhs`).
    Shr(Box<Self>, Box<Self>),
    /// Bitwise AND (`lhs & rhs`).
    And(Box<Self>, Box<Self>),
    /// Bitwise OR (`lhs | rhs`).
    Or(Box<Self>, Box<Self>),
    /// Bitwise XOR (`lhs ^ rhs`).
    Xor(Box<Self>, Box<Self>),
    /// Maximum (`max(lhs, rhs)`).
    Max(Box<Self>, Box<Self>),
    /// Minimum (`min(lhs, rhs)`).
    Min(Box<Self>, Box<Self>),
    /// Equality comparison (`lhs == rhs`, evaluates to `0` or `1`).
    Eq(Box<Self>, Box<Self>),
    /// Strictly-less-than comparison (`lhs < rhs`, evaluates to `0` or `1`).
    Lt(Box<Self>, Box<Self>),
    /// Less-than-or-equal comparison (`lhs <= rhs`, evaluates to `0` or `1`).
    Le(Box<Self>, Box<Self>),
    /// Ternary conditional (`cond ? then_branch : else_branch`).
    Cond {
        /// Condition expression (non-zero selects `then_branch`, zero selects `else_branch`).
        cond: Box<Self>,
        /// Value when `cond != 0`.
        then_branch: Box<Self>,
        /// Value when `cond == 0`.
        else_branch: Box<Self>,
    },
}

/// Computes the AST tree depth of `expr` (leaf `Const` or `Sym` has depth `0`).
#[must_use]
pub fn param_expr_depth(expr: &SymbolicParamExpr) -> usize {
    match expr {
        SymbolicParamExpr::Const(_) | SymbolicParamExpr::Sym(_) => 0,
        SymbolicParamExpr::Add(a, b)
        | SymbolicParamExpr::Mul(a, b)
        | SymbolicParamExpr::MulNoWrap(a, b)
        | SymbolicParamExpr::Div(a, b)
        | SymbolicParamExpr::Mod(a, b)
        | SymbolicParamExpr::Shl(a, b)
        | SymbolicParamExpr::Shr(a, b)
        | SymbolicParamExpr::And(a, b)
        | SymbolicParamExpr::Or(a, b)
        | SymbolicParamExpr::Xor(a, b)
        | SymbolicParamExpr::Max(a, b)
        | SymbolicParamExpr::Min(a, b)
        | SymbolicParamExpr::Eq(a, b)
        | SymbolicParamExpr::Lt(a, b)
        | SymbolicParamExpr::Le(a, b) => 1 + param_expr_depth(a).max(param_expr_depth(b)),
        SymbolicParamExpr::Cond {
            cond,
            then_branch,
            else_branch,
        } => {
            1 + param_expr_depth(cond)
                .max(param_expr_depth(then_branch))
                .max(param_expr_depth(else_branch))
        }
    }
}

/// Returns `true` when `child` can be nested inside `parent` without exceeding
/// [`MAX_PARAM_EXPR_NESTING_DEPTH`] (`kMaxDepth = 4`).
#[must_use]
pub fn can_param_operator_be_nested(parent: &SymbolicParamExpr, child: &SymbolicParamExpr) -> bool {
    param_expr_depth(parent).saturating_add(param_expr_depth(child)) <= MAX_PARAM_EXPR_NESTING_DEPTH
}

/// Infers a tight [`ConstantIntRange`] for `expr` given symbol bounds `sym_bounds` (symbols not
/// present in `sym_bounds` default to `ConstantIntRange::full()`).
#[must_use]
pub fn infer_param_expr_range(
    expr: &SymbolicParamExpr,
    sym_bounds: &IndexMap<String, ConstantIntRange>,
) -> ConstantIntRange {
    match expr {
        SymbolicParamExpr::Const(v) => ConstantIntRange::exact(*v),
        SymbolicParamExpr::Sym(name) => sym_bounds
            .get(name)
            .copied()
            .unwrap_or_else(ConstantIntRange::full),
        SymbolicParamExpr::Add(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            match (l.min.checked_add(r.min), l.max.checked_add(r.max)) {
                (Some(lo), Some(hi)) => ConstantIntRange::new(lo, hi),
                (Some(lo), None) if l.min >= 0 && r.min >= 0 => ConstantIntRange::new(lo, i64::MAX),
                _ => ConstantIntRange::full(),
            }
        }
        SymbolicParamExpr::Mul(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            let p1 = l.min.checked_mul(r.min);
            let p2 = l.min.checked_mul(r.max);
            let p3 = l.max.checked_mul(r.min);
            let p4 = l.max.checked_mul(r.max);
            if let (Some(a), Some(b), Some(c), Some(d)) = (p1, p2, p3, p4) {
                let lo = a.min(b).min(c).min(d);
                let hi = a.max(b).max(c).max(d);
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::MulNoWrap(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::new(l.min.saturating_mul(r.min), l.max.saturating_mul(r.max))
            } else {
                infer_param_expr_range(
                    &SymbolicParamExpr::Mul(lhs.clone(), rhs.clone()),
                    sym_bounds,
                )
            }
        }
        SymbolicParamExpr::Div(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if r.min > 0 {
                let q1 = l.min / r.min;
                let q2 = l.min / r.max;
                let q3 = l.max / r.min;
                let q4 = l.max / r.max;
                let lo = q1.min(q2).min(q3).min(q4);
                let hi = q1.max(q2).max(q3).max(q4);
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Mod(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0 && r.min > 0 {
                let upper = (r.max - 1).min(l.max);
                ConstantIntRange::new(0, upper)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Shl(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0
                && r.min >= 0
                && r.max < 63
                && let (Some(lo), Some(hi)) = (
                    l.min.checked_shl(r.min as u32),
                    l.max.checked_shl(r.max as u32),
                )
                && lo >= 0
                && hi >= lo
            {
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Shr(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min >= 0 && r.min >= 0 && r.max < 63 {
                let lo = l.min >> (r.max as u32);
                let hi = l.max >> (r.min as u32);
                ConstantIntRange::new(lo, hi)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::And(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if let (Some(lv), Some(rv)) = (l.single_value(), r.single_value()) {
                ConstantIntRange::exact(lv & rv)
            } else if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::new(0, l.max.min(r.max))
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Or(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if let (Some(lv), Some(rv)) = (l.single_value(), r.single_value()) {
                ConstantIntRange::exact(lv | rv)
            } else if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::new(l.min.max(r.min), i64::MAX)
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Xor(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if let (Some(lv), Some(rv)) = (l.single_value(), r.single_value()) {
                ConstantIntRange::exact(lv ^ rv)
            } else if l.min >= 0 && r.min >= 0 {
                ConstantIntRange::non_negative()
            } else {
                ConstantIntRange::full()
            }
        }
        SymbolicParamExpr::Max(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            ConstantIntRange::new(l.min.max(r.min), l.max.max(r.max))
        }
        SymbolicParamExpr::Min(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            ConstantIntRange::new(l.min.min(r.min), l.max.min(r.max))
        }
        SymbolicParamExpr::Eq(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.min == l.max && r.min == r.max && l.min == r.min {
                ConstantIntRange::exact(1)
            } else if l.max < r.min || r.max < l.min {
                ConstantIntRange::exact(0)
            } else {
                ConstantIntRange::new(0, 1)
            }
        }
        SymbolicParamExpr::Lt(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.max < r.min {
                ConstantIntRange::exact(1)
            } else if l.min >= r.max {
                ConstantIntRange::exact(0)
            } else {
                ConstantIntRange::new(0, 1)
            }
        }
        SymbolicParamExpr::Le(lhs, rhs) => {
            let l = infer_param_expr_range(lhs, sym_bounds);
            let r = infer_param_expr_range(rhs, sym_bounds);
            if l.max <= r.min {
                ConstantIntRange::exact(1)
            } else if l.min > r.max {
                ConstantIntRange::exact(0)
            } else {
                ConstantIntRange::new(0, 1)
            }
        }
        SymbolicParamExpr::Cond {
            cond,
            then_branch,
            else_branch,
        } => {
            let c = infer_param_expr_range(cond, sym_bounds);
            let t = infer_param_expr_range(then_branch, sym_bounds);
            let e = infer_param_expr_range(else_branch, sym_bounds);
            if c.min == 0 && c.max == 0 {
                e
            } else if c.min > 0 || c.max < 0 {
                t
            } else {
                t.union(&e)
            }
        }
    }
}

/// Simplifies `expr` using range analysis over `sym_bounds`:
/// - Collapses any expression whose inferred [`ConstantIntRange`] is a single value `[v, v]` to
///   `SymbolicParamExpr::Const(v)`.
/// - Simplifies `<max(add(A, B), B)> -> <add(A, B)>` (and commutative variants) when both `A` and
///   `B` are provably non-negative (`min >= 0`).
/// - Simplifies `<min(add(A, B), B)> -> <B>` when both `A` and `B` are provably non-negative.
/// - Simplifies `<max(L, R)>` / `<min(L, R)>` when one operand's range dominates the other
///   (`l_range.min >= r_range.max`).
/// - Collapses `Cond` expressions whose condition range cannot be zero (`then_branch`), is
///   identically zero (`else_branch`), or has identical branches.
#[must_use]
pub fn simplify_param_expr_with_ranges(
    expr: &SymbolicParamExpr,
    sym_bounds: &IndexMap<String, ConstantIntRange>,
) -> SymbolicParamExpr {
    let whole_range = infer_param_expr_range(expr, sym_bounds);
    if let Some(c) = whole_range.single_value() {
        return SymbolicParamExpr::Const(c);
    }

    match expr {
        SymbolicParamExpr::Const(_) | SymbolicParamExpr::Sym(_) => expr.clone(),
        SymbolicParamExpr::Add(a, b) => {
            let sa = simplify_param_expr_with_ranges(a, sym_bounds);
            let sb = simplify_param_expr_with_ranges(b, sym_bounds);
            if sa == SymbolicParamExpr::Const(0) {
                return sb;
            }
            if sb == SymbolicParamExpr::Const(0) {
                return sa;
            }
            SymbolicParamExpr::Add(Box::new(sa), Box::new(sb))
        }
        SymbolicParamExpr::Mul(a, b) => {
            let sa = simplify_param_expr_with_ranges(a, sym_bounds);
            let sb = simplify_param_expr_with_ranges(b, sym_bounds);
            if sa == SymbolicParamExpr::Const(1) {
                return sb;
            }
            if sb == SymbolicParamExpr::Const(1) {
                return sa;
            }
            SymbolicParamExpr::Mul(Box::new(sa), Box::new(sb))
        }
        SymbolicParamExpr::MulNoWrap(a, b) => {
            let sa = simplify_param_expr_with_ranges(a, sym_bounds);
            let sb = simplify_param_expr_with_ranges(b, sym_bounds);
            if sa == SymbolicParamExpr::Const(1) {
                return sb;
            }
            if sb == SymbolicParamExpr::Const(1) {
                return sa;
            }
            SymbolicParamExpr::MulNoWrap(Box::new(sa), Box::new(sb))
        }
        SymbolicParamExpr::Max(lhs, rhs) => {
            let sl = simplify_param_expr_with_ranges(lhs, sym_bounds);
            let sr = simplify_param_expr_with_ranges(rhs, sym_bounds);
            if sl == sr {
                return sl;
            }
            if let SymbolicParamExpr::Add(a, b) = &sl
                && (**b == sr || **a == sr)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sl;
            }
            if let SymbolicParamExpr::Add(a, b) = &sr
                && (**b == sl || **a == sl)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sr;
            }
            let lr = infer_param_expr_range(&sl, sym_bounds);
            let rr = infer_param_expr_range(&sr, sym_bounds);
            if lr.min >= rr.max {
                return sl;
            }
            if rr.min >= lr.max {
                return sr;
            }
            SymbolicParamExpr::Max(Box::new(sl), Box::new(sr))
        }
        SymbolicParamExpr::Min(lhs, rhs) => {
            let sl = simplify_param_expr_with_ranges(lhs, sym_bounds);
            let sr = simplify_param_expr_with_ranges(rhs, sym_bounds);
            if sl == sr {
                return sl;
            }
            if let SymbolicParamExpr::Add(a, b) = &sl
                && (**b == sr || **a == sr)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sr;
            }
            if let SymbolicParamExpr::Add(a, b) = &sr
                && (**b == sl || **a == sl)
                && infer_param_expr_range(a, sym_bounds).is_non_negative()
                && infer_param_expr_range(b, sym_bounds).is_non_negative()
            {
                return sl;
            }
            let lr = infer_param_expr_range(&sl, sym_bounds);
            let rr = infer_param_expr_range(&sr, sym_bounds);
            if lr.max <= rr.min {
                return sl;
            }
            if rr.max <= lr.min {
                return sr;
            }
            SymbolicParamExpr::Min(Box::new(sl), Box::new(sr))
        }
        SymbolicParamExpr::Cond {
            cond,
            then_branch,
            else_branch,
        } => {
            let sc = simplify_param_expr_with_ranges(cond, sym_bounds);
            let st = simplify_param_expr_with_ranges(then_branch, sym_bounds);
            let se = simplify_param_expr_with_ranges(else_branch, sym_bounds);
            if st == se {
                return st;
            }
            let cr = infer_param_expr_range(&sc, sym_bounds);
            if cr.min == 0 && cr.max == 0 {
                return se;
            }
            if cr.min > 0 || cr.max < 0 {
                return st;
            }
            SymbolicParamExpr::Cond {
                cond: Box::new(sc),
                then_branch: Box::new(st),
                else_branch: Box::new(se),
            }
        }
        SymbolicParamExpr::Div(a, b) => SymbolicParamExpr::Div(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Mod(a, b) => SymbolicParamExpr::Mod(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Shl(a, b) => SymbolicParamExpr::Shl(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Shr(a, b) => SymbolicParamExpr::Shr(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::And(a, b) => SymbolicParamExpr::And(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Or(a, b) => SymbolicParamExpr::Or(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Xor(a, b) => SymbolicParamExpr::Xor(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Eq(a, b) => SymbolicParamExpr::Eq(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Lt(a, b) => SymbolicParamExpr::Lt(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
        SymbolicParamExpr::Le(a, b) => SymbolicParamExpr::Le(
            Box::new(simplify_param_expr_with_ranges(a, sym_bounds)),
            Box::new(simplify_param_expr_with_ranges(b, sym_bounds)),
        ),
    }
}

/// Synchronizes the outer signature of every `mo.graph` / `func.func` operation in `module` with
/// the refined types of its entry block arguments and `mo.output` / `func.return` terminator
/// operands (`updateGraphSignature` in `PropagateShapesPass.cpp`).
///
/// Returns the number of graph/function signatures updated.
pub fn update_graph_signature_from_body(module: &mut Module) -> usize {
    let mut updated_count = 0usize;
    let op_ids: Vec<_> = module.ops.keys().collect();

    for op_id in op_ids {
        let Some(op) = module.ops.get(op_id).cloned() else {
            continue;
        };
        let full_name = op.name.full_name();
        if !matches!(full_name.as_str(), "mo.graph" | "func.func" | "mgp.model") {
            continue;
        }
        let Some(&reg_id) = op.regions.first() else {
            continue;
        };
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        let Some(&blk_id) = reg.blocks.first() else {
            continue;
        };
        let Some(blk) = module.blocks.get(blk_id).cloned() else {
            continue;
        };

        let arg_types: Vec<Type> = blk
            .arguments
            .iter()
            .filter_map(|&v| module.values.get(v).map(|vi| vi.ty.clone()))
            .collect();

        let mut ret_types: Option<Vec<Type>> = None;
        if let Some(&last_op_id) = blk.ops.last()
            && let Some(term_op) = module.ops.get(last_op_id)
            && matches!(
                term_op.name.full_name().as_str(),
                "mo.output" | "func.return" | "mgp.yield"
            )
        {
            let tys: Vec<Type> = term_op
                .operands
                .iter()
                .filter_map(|&v| module.values.get(v).map(|vi| vi.ty.clone()))
                .collect();
            ret_types = Some(tys);
        }

        let mut changed = false;
        let new_arg_attr = Attribute::Array(
            arg_types
                .iter()
                .map(|t| Attribute::Type(t.clone()))
                .collect(),
        );
        if op.attributes.get("argument_types") != Some(&new_arg_attr) {
            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut
                    .attributes
                    .insert("argument_types".to_string(), new_arg_attr);
            }
            changed = true;
        }

        if let Some(r_types) = ret_types {
            let new_res_attr =
                Attribute::Array(r_types.iter().map(|t| Attribute::Type(t.clone())).collect());
            if op.attributes.get("result_types") != Some(&new_res_attr) {
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut
                        .attributes
                        .insert("result_types".to_string(), new_res_attr);
                }
                changed = true;
            }
            for (&res_val, new_ty) in op.results.iter().zip(r_types.iter()) {
                if let Some(v_mut) = module.values.get_mut(res_val)
                    && &v_mut.ty != new_ty
                {
                    v_mut.ty = new_ty.clone();
                    changed = true;
                }
            }
        }

        if changed {
            updated_count += 1;
        }
    }

    updated_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_param_range_inference_and_simplification() {
        let mut bounds = IndexMap::new();
        bounds.insert("A".to_string(), ConstantIntRange::non_negative());
        bounds.insert("B".to_string(), ConstantIntRange::positive());

        let a = SymbolicParamExpr::Sym("A".to_string());
        let b = SymbolicParamExpr::Sym("B".to_string());
        let add_ab = SymbolicParamExpr::Add(Box::new(a.clone()), Box::new(b.clone()));
        let max_expr = SymbolicParamExpr::Max(Box::new(add_ab.clone()), Box::new(b.clone()));

        let simplified = simplify_param_expr_with_ranges(&max_expr, &bounds);
        assert_eq!(simplified, add_ab);

        let cond_expr = SymbolicParamExpr::Cond {
            cond: Box::new(b.clone()),
            then_branch: Box::new(a.clone()),
            else_branch: Box::new(SymbolicParamExpr::Const(99)),
        };
        assert_eq!(simplify_param_expr_with_ranges(&cond_expr, &bounds), a);
    }
}
