//! Call-site shape normalization, dead parameter declaration elimination,
//! parameter expression lowering, compile-time tensor shape staticization, and
//! multi-device graph-to-model lowering.
//!
//! Covers:
//! 1. **Call-Site Shape Normalization (`normalize_call_operands_to_callee`)**:
//!    Compares caller operand tensor types against callee parameter tensor types;
//!    when `dtype` and `device_ref` match (`encoding` matches) but shape
//!    dimensions differ, inserts `mo.rebind` before the call with
//!    `output_param_decls` containing only symbolic dimensions from the expected
//!    shape not already declared in the caller region, recording newly declared
//!    symbols so subsequent call sites do not redeclare them.
//! 2. **Dead Parameter Declaration Elimination (`remove_dead_param_declares`)**:
//!    Worklist liveness dead-code elimination over `kgen.param.declare`
//!    operations, retaining impure `opcode == "apply"` operations and any
//!    declaration transitively referenced via `param_uses` or symbolic dimensions
//!    of live operations.
//! 3. **Parameter Expression Lowering (`ParameterLoweringState`)**:
//!    Lowers parameter expressions per region with memoization, creating
//!    `kgen.param.placeholder` on forward references to `ParamDeclRef` and
//!    resolving placeholders when `lower_param_decl` is called.
//! 4. **Compile-Time Shape & Tensor Staticization (`staticize_shape_and_tensor_ops`)**:
//!    Evaluates compile-time `shape_attr` values across `[1] <-> []` reshapes,
//!    1D constant broadcasts, rank-aligned symbolic/static `mo.broadcast_shape`
//!    (returning `Err` on conflicting static non-1 dimensions), axis-0 1D
//!    `mo.concat`, and integer `mo.cast` / elementwise `mo.add`.
//! 5. **Multi-Device Graph-to-Model Lowering (`lower_graph_to_mgp_model`)**:
//!    Collects sorted device references (`"cpu:0"` first, followed by remaining
//!    devices in lexicographical order) and lowers `mo.graph` into `mgp.model`
//!    with `init` and `execute` regions and `minimum_cpu_alignment` enforcement.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use smallvec::SmallVec;

use crate::ir::mlir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, RegionId,
    TensorType, Type, Value, ValueDef, ValueId,
};

fn make_op_name(full: &str) -> OpName {
    if let Some((dialect, name)) = full.split_once('.') {
        OpName::new(dialect, name)
    } else {
        OpName::new("builtin", full)
    }
}

fn new_unlinked_op(module: &mut Module, full_name: &str) -> OpId {
    let op_name = make_op_name(full_name);
    module.ops.insert_with_key(|id| Operation {
        id,
        name: op_name,
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    })
}

// ---------------------------------------------------------------------------
// 1. Call-Site Shape Normalization
// ---------------------------------------------------------------------------

/// Collects all symbolic parameter names already declared inside `region_id`
/// via `output_param_decls`, block argument shapes, or `kgen.param.declare`
/// operations.
#[must_use]
pub fn collect_declared_params_in_region(module: &Module, region_id: RegionId) -> BTreeSet<String> {
    let mut declared = BTreeSet::new();
    let Some(region) = module.regions.get(region_id) else {
        return declared;
    };
    for &block_id in &region.blocks {
        let Some(block) = module.blocks.get(block_id) else {
            continue;
        };
        for &val_id in &block.arguments {
            if let Some(val) = module.values.get(val_id)
                && let Type::Tensor(tt) = &val.ty
            {
                for dim in &tt.shape {
                    if let Dim::Symbolic(sym) = dim {
                        declared.insert(sym.clone());
                    }
                }
            }
        }
        for &op_id in &block.ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            if op.name.full_name() == "kgen.param.declare"
                && let Some(Attribute::String(name)) = op.attributes.get("sym_name")
            {
                declared.insert(name.clone());
            }
            if let Some(Attribute::Array(decls)) = op.attributes.get("output_param_decls") {
                for item in decls {
                    if let Attribute::String(sym) = item {
                        declared.insert(sym.clone());
                    }
                }
            }
            for &nested_region in &op.regions {
                declared.extend(collect_declared_params_in_region(module, nested_region));
            }
        }
    }
    declared
}

/// Normalizes call-site tensor operands to match callee parameter tensor types
/// by inserting `mo.rebind` operations when `dtype` and `device_ref` match
/// (`encoding` matches) but shape dimensions differ.
///
/// Newly introduced symbolic dimensions on `mo.rebind` are emitted in
/// `output_param_decls` and recorded in the caller region's declared parameter
/// set so subsequent call sites do not redeclare them.
///
/// Returns the number of `mo.rebind` operations inserted.
pub fn normalize_call_operands_to_callee(module: &mut Module) -> usize {
    // 1. Collect callee signatures: sym_name -> Vec<Type> of entry block args.
    let mut callee_param_types: HashMap<String, Vec<Type>> = HashMap::new();
    let all_op_ids: Vec<OpId> = module.ops.keys().collect();
    for op_id in &all_op_ids {
        let Some(op) = module.ops.get(*op_id) else {
            continue;
        };
        let full = op.name.full_name();
        if (full == "func.func" || full == "mo.func" || full == "kgen.func")
            && let Some(Attribute::String(sym_name)) = op.attributes.get("sym_name")
            && let Some(&region_id) = op.regions.first()
            && let Some(region) = module.regions.get(region_id)
            && let Some(&entry_block) = region.blocks.first()
            && let Some(block) = module.blocks.get(entry_block)
        {
            let param_tys = block
                .arguments
                .iter()
                .filter_map(|&vid| module.values.get(vid).map(|v| v.ty.clone()))
                .collect();
            callee_param_types.insert(sym_name.clone(), param_tys);
        }
    }

    let mut rebinds_inserted = 0usize;
    let region_ids: Vec<RegionId> = module.regions.keys().collect();
    for region_id in region_ids {
        let mut declared_params = collect_declared_params_in_region(module, region_id);
        let Some(region) = module.regions.get(region_id) else {
            continue;
        };
        let block_ids: Vec<BlockId> = region.blocks.clone();
        for block_id in block_ids {
            let Some(block) = module.blocks.get(block_id) else {
                continue;
            };
            let orig_ops = block.ops.clone();
            let mut new_block_ops = Vec::with_capacity(orig_ops.len());

            for op_id in orig_ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let full = op.name.full_name();
                if full != "mo.call" && full != "func.call" {
                    new_block_ops.push(op_id);
                    continue;
                }
                let Some(Attribute::String(callee)) = op.attributes.get("callee").cloned() else {
                    new_block_ops.push(op_id);
                    continue;
                };
                let Some(expected_tys) = callee_param_types.get(&callee).cloned() else {
                    new_block_ops.push(op_id);
                    continue;
                };

                let operands = op.operands.clone();
                let mut updated_operands = operands.clone();

                for (idx, (&operand_val, expected_ty)) in
                    operands.iter().zip(expected_tys.iter()).enumerate()
                {
                    let Some(actual_ty) = module.values.get(operand_val).map(|v| v.ty.clone())
                    else {
                        continue;
                    };
                    let (Type::Tensor(actual_tt), Type::Tensor(expected_tt)) =
                        (&actual_ty, expected_ty)
                    else {
                        continue;
                    };

                    // Must match dtype and device_ref (encoding), and differ in shape.
                    if actual_tt.dtype != expected_tt.dtype
                        || actual_tt.encoding != expected_tt.encoding
                        || actual_tt.shape == expected_tt.shape
                    {
                        continue;
                    }

                    // Collect symbolic dimensions from expected_tt.shape not yet declared.
                    let mut new_param_decls = Vec::new();
                    for dim in &expected_tt.shape {
                        if let Dim::Symbolic(sym) = dim
                            && declared_params.insert(sym.clone())
                        {
                            new_param_decls.push(Attribute::String(sym.clone()));
                        }
                    }

                    let rebind_op_id = new_unlinked_op(module, "mo.rebind");
                    if let Some(rop) = module.ops.get_mut(rebind_op_id) {
                        rop.operands.push(operand_val);
                        if !new_param_decls.is_empty() {
                            rop.attributes.insert(
                                "output_param_decls".to_string(),
                                Attribute::Array(new_param_decls),
                            );
                        }
                    }
                    let rebound_ty = expected_ty.clone();
                    let rebound_val = module.values.insert_with_key(|vid| Value {
                        id: vid,
                        ssa_name: None,
                        ty: rebound_ty,
                        def: ValueDef::OpResult {
                            op: rebind_op_id,
                            index: 0,
                        },
                    });
                    if let Some(rop) = module.ops.get_mut(rebind_op_id) {
                        rop.results.push(rebound_val);
                    }
                    new_block_ops.push(rebind_op_id);
                    updated_operands[idx] = rebound_val;
                    rebinds_inserted += 1;
                }

                if let Some(call_op) = module.ops.get_mut(op_id) {
                    call_op.operands = updated_operands;
                }
                new_block_ops.push(op_id);
            }

            if let Some(block_mut) = module.blocks.get_mut(block_id) {
                block_mut.ops = new_block_ops;
            }
        }
    }

    rebinds_inserted
}

// ---------------------------------------------------------------------------
// 2. Dead Parameter Declaration Elimination
// ---------------------------------------------------------------------------

/// Extracts all symbolic parameter names referenced by `op` (via `param_uses`
/// attribute or operand/result symbolic dimensions).
fn extract_referenced_param_names(module: &Module, op: &Operation) -> Vec<String> {
    let mut refs = Vec::new();
    if let Some(Attribute::Array(uses)) = op.attributes.get("param_uses") {
        for item in uses {
            if let Attribute::String(name) = item {
                refs.push(name.clone());
            }
        }
    }
    for &val_id in op.operands.iter().chain(op.results.iter()) {
        if let Some(val) = module.values.get(val_id)
            && let Type::Tensor(tt) = &val.ty
        {
            for dim in &tt.shape {
                if let Dim::Symbolic(sym) = dim {
                    refs.push(sym.clone());
                }
            }
        }
    }
    refs
}

/// Removes dead `kgen.param.declare` operations via worklist liveness analysis.
///
/// Any `kgen.param.declare` with `opcode == "apply"` is treated as having
/// potential side effects and is retained as a liveness root alongside all
/// non-`kgen.param.declare` operations.
///
/// Returns the number of dead `kgen.param.declare` operations removed.
pub fn remove_dead_param_declares(module: &mut Module) -> usize {
    // Map sym_name -> OpId for all kgen.param.declare ops.
    let mut param_decl_by_name: HashMap<String, OpId> = HashMap::new();
    let mut worklist: Vec<OpId> = Vec::new();
    let mut live_decls: HashSet<OpId> = HashSet::new();

    for (op_id, op) in &module.ops {
        if op.name.full_name() == "kgen.param.declare" {
            if let Some(Attribute::String(sym_name)) = op.attributes.get("sym_name") {
                param_decl_by_name.insert(sym_name.clone(), op_id);
            }
            let is_impure_apply = matches!(
                op.attributes.get("opcode"),
                Some(Attribute::String(code)) if code == "apply"
            );
            if is_impure_apply {
                live_decls.insert(op_id);
                worklist.push(op_id);
            }
        } else {
            worklist.push(op_id);
        }
    }

    while let Some(op_id) = worklist.pop() {
        let Some(op) = module.ops.get(op_id) else {
            continue;
        };
        let refs = extract_referenced_param_names(module, op);
        for name in refs {
            if let Some(&decl_op_id) = param_decl_by_name.get(&name)
                && live_decls.insert(decl_op_id)
            {
                worklist.push(decl_op_id);
            }
        }
    }

    let mut removed = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();
    for block_id in block_ids {
        let Some(block) = module.blocks.get(block_id) else {
            continue;
        };
        let mut retained_ops = Vec::with_capacity(block.ops.len());
        let mut dead_ops = Vec::new();
        for &op_id in &block.ops {
            let is_dead_decl = module.ops.get(op_id).is_some_and(|op| {
                op.name.full_name() == "kgen.param.declare" && !live_decls.contains(&op_id)
            });
            if is_dead_decl {
                dead_ops.push(op_id);
            } else {
                retained_ops.push(op_id);
            }
        }
        if !dead_ops.is_empty() {
            if let Some(block_mut) = module.blocks.get_mut(block_id) {
                block_mut.ops = retained_ops;
            }
            for dead_op in dead_ops {
                module.ops.remove(dead_op);
                removed += 1;
            }
        }
    }

    removed
}

// ---------------------------------------------------------------------------
// 3. Parameter Expression Lowering with Forward-Reference Placeholders
// ---------------------------------------------------------------------------

/// High-level parameter expression node lowered by [`ParameterLoweringState`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParamExpr {
    /// Constant integer parameter literal.
    Constant(i64),
    /// Reference to a named parameter declaration (`sym_name`).
    ParamDeclRef(String),
    /// Binary arithmetic expression (`add`, `sub`, `mul`, `div`).
    BinaryOp {
        /// Arithmetic opcode (`"add"`, `"sub"`, `"mul"`, `"div"`).
        opcode: String,
        /// Left-hand side parameter expression.
        lhs: Box<ParamExpr>,
        /// Right-hand side parameter expression.
        rhs: Box<ParamExpr>,
    },
}

/// Per-region parameter lowering state with memoization and forward-reference
/// `kgen.param.placeholder` resolution.
#[derive(Debug, Default)]
pub struct ParameterLoweringState {
    /// Memoized lowered `ValueId` per `(RegionId, ParamExpr)`.
    lowered_exprs: HashMap<(RegionId, ParamExpr), ValueId>,
    /// Forward-referenced `kgen.param.placeholder` ops keyed by `(RegionId, sym_name)`.
    placeholders: HashMap<(RegionId, String), (OpId, ValueId)>,
    /// Already-defined parameter declarations keyed by `(RegionId, sym_name)`.
    defined_decls: HashMap<(RegionId, String), ValueId>,
}

impl ParameterLoweringState {
    /// Creates a new empty parameter lowering state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Lowers `expr` inside `region_id` (appending materialized ops to the entry
    /// block of `region_id`), reusing memoized values and creating a
    /// `kgen.param.placeholder` if a `ParamDeclRef` has not been declared yet.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `region_id` has no entry block.
    pub fn lower_expr(
        &mut self,
        module: &mut Module,
        region_id: RegionId,
        expr: &ParamExpr,
    ) -> Result<ValueId, String> {
        let key = (region_id, expr.clone());
        if let Some(&cached) = self.lowered_exprs.get(&key) {
            return Ok(cached);
        }

        let entry_block = module
            .regions
            .get(region_id)
            .and_then(|r| r.blocks.first().copied())
            .ok_or_else(|| "target region has no entry block".to_string())?;

        let val_id = match expr {
            ParamExpr::Constant(c) => {
                let op_id = new_unlinked_op(module, "kgen.param.constant");
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut
                        .attributes
                        .insert("value".to_string(), Attribute::Int(*c));
                }
                let vid = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: Type::Scalar(DType::I64),
                    def: ValueDef::OpResult {
                        op: op_id,
                        index: 0,
                    },
                });
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.results.push(vid);
                }
                if let Some(block_mut) = module.blocks.get_mut(entry_block) {
                    block_mut.ops.push(op_id);
                }
                vid
            }
            ParamExpr::ParamDeclRef(sym_name) => {
                if let Some(&defined_val) = self.defined_decls.get(&(region_id, sym_name.clone())) {
                    defined_val
                } else if let Some(&(_ph_op, ph_val)) =
                    self.placeholders.get(&(region_id, sym_name.clone()))
                {
                    ph_val
                } else {
                    let ph_op_id = new_unlinked_op(module, "kgen.param.placeholder");
                    if let Some(op_mut) = module.ops.get_mut(ph_op_id) {
                        op_mut
                            .attributes
                            .insert("sym_name".to_string(), Attribute::String(sym_name.clone()));
                    }
                    let ph_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: Type::Scalar(DType::I64),
                        def: ValueDef::OpResult {
                            op: ph_op_id,
                            index: 0,
                        },
                    });
                    if let Some(op_mut) = module.ops.get_mut(ph_op_id) {
                        op_mut.results.push(ph_val);
                    }
                    if let Some(block_mut) = module.blocks.get_mut(entry_block) {
                        block_mut.ops.push(ph_op_id);
                    }
                    self.placeholders
                        .insert((region_id, sym_name.clone()), (ph_op_id, ph_val));
                    ph_val
                }
            }
            ParamExpr::BinaryOp { opcode, lhs, rhs } => {
                let lhs_val = self.lower_expr(module, region_id, lhs)?;
                let rhs_val = self.lower_expr(module, region_id, rhs)?;
                let op_id = new_unlinked_op(module, &format!("kgen.param.{opcode}"));
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.operands.push(lhs_val);
                    op_mut.operands.push(rhs_val);
                }
                let vid = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: Type::Scalar(DType::I64),
                    def: ValueDef::OpResult {
                        op: op_id,
                        index: 0,
                    },
                });
                if let Some(op_mut) = module.ops.get_mut(op_id) {
                    op_mut.results.push(vid);
                }
                if let Some(block_mut) = module.blocks.get_mut(entry_block) {
                    block_mut.ops.push(op_id);
                }
                vid
            }
        };

        self.lowered_exprs.insert(key, val_id);
        Ok(val_id)
    }

    /// Lowers a parameter declaration `sym_name = rhs_expr` in `region_id`,
    /// replacing all uses of any forward-reference `kgen.param.placeholder`
    /// with the newly lowered `ValueId` and erasing the placeholder op.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `sym_name` was already declared in `region_id`.
    pub fn lower_param_decl(
        &mut self,
        module: &mut Module,
        region_id: RegionId,
        sym_name: &str,
        rhs_expr: &ParamExpr,
    ) -> Result<ValueId, String> {
        let decl_key = (region_id, sym_name.to_string());
        if self.defined_decls.contains_key(&decl_key) {
            return Err(format!(
                "duplicate parameter declaration '{sym_name}' in region"
            ));
        }

        let lowered_val = self.lower_expr(module, region_id, rhs_expr)?;
        self.defined_decls.insert(decl_key.clone(), lowered_val);
        self.lowered_exprs.insert(
            (region_id, ParamExpr::ParamDeclRef(sym_name.to_string())),
            lowered_val,
        );

        // If a forward-reference placeholder existed, replace its uses and remove it.
        if let Some((ph_op_id, ph_val_id)) = self.placeholders.remove(&decl_key) {
            module.replace_all_uses(ph_val_id, lowered_val);
            for (_bid, block) in &mut module.blocks {
                block.ops.retain(|&oid| oid != ph_op_id);
            }
            module.remove_op_unlinked(ph_op_id);
        }

        Ok(lowered_val)
    }
}

// ---------------------------------------------------------------------------
// 4. Compile-Time Shape & Tensor Staticization
// ---------------------------------------------------------------------------

/// Extracts a 1D integer vector from a `shape_attr` or `value` array attribute.
fn extract_int_vector_attr(op: &Operation, attr_key: &str) -> Option<Vec<i64>> {
    let Attribute::Array(items) = op.attributes.get(attr_key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if let Attribute::Int(v) = item {
            out.push(*v);
        } else {
            return None;
        }
    }
    Some(out)
}

/// Extracts a 1D symbolic/static dimension list from `shape_attr` where entries
/// may be `Attribute::Int(d)` or `Attribute::String(sym)`.
fn extract_dim_vector_attr(op: &Operation, attr_key: &str) -> Option<Vec<Dim>> {
    let Attribute::Array(items) = op.attributes.get(attr_key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Attribute::Int(v) if *v >= 0 => out.push(Dim::Static(*v)),
            Attribute::Int(_) => out.push(Dim::Dynamic),
            Attribute::String(s) => out.push(Dim::Symbolic(s.clone())),
            _ => return None,
        }
    }
    Some(out)
}

/// Encodes a slice of [`Dim`] back into an `Attribute::Array`.
fn encode_dim_vector_attr(dims: &[Dim]) -> Attribute {
    Attribute::Array(
        dims.iter()
            .map(|d| match d {
                Dim::Static(n) => Attribute::Int(*n),
                Dim::Dynamic => Attribute::Int(-1),
                Dim::Symbolic(s) => Attribute::String(s.clone()),
            })
            .collect(),
    )
}

/// Attempts to staticize a `[1] <-> []` reshape/squeeze/unsqueeze operation on
/// a compile-time 1-element constant vector.
#[must_use]
pub fn try_staticize_reshape_like(input_vals: &[i64], target_shape: &[Dim]) -> Option<Vec<i64>> {
    // Only staticize scalar `[]` <-> 1-element `[1]` reshapes.
    let is_scalar_or_single =
        target_shape.is_empty() || (target_shape.len() == 1 && target_shape[0] == Dim::Static(1));
    if input_vals.len() == 1 && is_scalar_or_single {
        Some(vec![input_vals[0]])
    } else {
        None
    }
}

/// Attempts to staticize a 1D `mo.broadcast_to` operation when the input is a
/// single-element compile-time constant and the target shape is 1D static `[N]`.
#[must_use]
pub fn try_staticize_broadcast_to(input_vals: &[i64], target_shape: &[Dim]) -> Option<Vec<i64>> {
    if input_vals.len() != 1 || target_shape.len() != 1 {
        return None;
    }
    if let Dim::Static(n) = target_shape[0]
        && n >= 0
    {
        Some(vec![input_vals[0]; n as usize])
    } else {
        None
    }
}

/// Computes the rank-aligned broadcasted shape of `lhs` and `rhs` following
/// symbolic and static dimension broadcasting rules:
/// - Ranks are right-aligned by prepending `Dim::Static(1)` to the shorter shape.
/// - If either dimension is `Dim::Static(1)`, the result is the other dimension.
/// - If both dimensions are equal (`Static(d) == Static(d)` or
///   `Symbolic(s) == Symbolic(s)`), the result is that dimension.
/// - If both dimensions are distinct static non-1 values, returns `Err`.
///
/// # Errors
///
/// Returns `Err` when two aligned static dimensions are both `!= 1` and unequal.
pub fn try_staticize_broadcast_shape(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, String> {
    let out_rank = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(out_rank);

    for i in 0..out_rank {
        let l_dim = if i < out_rank - lhs.len() {
            &Dim::Static(1)
        } else {
            &lhs[i - (out_rank - lhs.len())]
        };
        let r_dim = if i < out_rank - rhs.len() {
            &Dim::Static(1)
        } else {
            &rhs[i - (out_rank - rhs.len())]
        };

        let merged = match (l_dim, r_dim) {
            (Dim::Static(1), other) | (other, Dim::Static(1)) => other.clone(),
            (Dim::Static(a), Dim::Static(b)) if a == b => Dim::Static(*a),
            (Dim::Static(a), Dim::Static(b)) => {
                return Err(format!(
                    "incompatible static broadcast dimensions at axis {i}: {a} vs {b}"
                ));
            }
            (Dim::Symbolic(a), Dim::Symbolic(b)) if a == b => Dim::Symbolic(a.clone()),
            _ => Dim::Dynamic,
        };
        out.push(merged);
    }

    Ok(out)
}

/// Propagates compile-time `shape_attr` values across reshape, broadcast,
/// `mo.broadcast_shape`, 1D axis-0 `mo.concat`, `mo.cast`, and `mo.add`
/// operations in `module`.
///
/// Returns the number of operations annotated with a newly folded `shape_attr`.
///
/// # Errors
///
/// Returns `Err` if `mo.broadcast_shape` encounters conflicting static non-1
/// dimensions.
pub fn staticize_shape_and_tensor_ops(module: &mut Module) -> Result<usize, String> {
    let mut folded_count = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in block_ids {
        let Some(block) = module.blocks.get(block_id) else {
            continue;
        };
        let op_ids = block.ops.clone();

        for op_id in op_ids {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if op.attributes.contains_key("shape_attr") {
                continue;
            }

            // Look up producer operations of operands.
            let operand_ops: Vec<Option<Operation>> = op
                .operands
                .iter()
                .map(|&vid| {
                    let val = module.values.get(vid)?;
                    if let ValueDef::OpResult { op: prod_id, .. } = val.def {
                        module.ops.get(prod_id).cloned()
                    } else {
                        None
                    }
                })
                .collect();

            let result_shape: Option<Vec<Dim>> = op
                .results
                .first()
                .and_then(|&vid| module.values.get(vid))
                .and_then(|v| match &v.ty {
                    Type::Tensor(tt) => Some(tt.shape.clone()),
                    _ => None,
                });

            let full = op.name.full_name();
            let folded_attr: Option<Attribute> = match full.as_str() {
                "mo.reshape" | "mo.squeeze" | "mo.unsqueeze" => {
                    if let (Some(Some(in_op)), Some(target_shape)) =
                        (operand_ops.first(), result_shape.as_ref())
                        && let Some(in_vals) = extract_int_vector_attr(in_op, "shape_attr")
                        && let Some(out_vals) = try_staticize_reshape_like(&in_vals, target_shape)
                    {
                        Some(Attribute::Array(
                            out_vals.into_iter().map(Attribute::Int).collect(),
                        ))
                    } else {
                        None
                    }
                }
                "mo.broadcast_to" => {
                    if let (Some(Some(in_op)), Some(target_shape)) =
                        (operand_ops.first(), result_shape.as_ref())
                        && let Some(in_vals) = extract_int_vector_attr(in_op, "shape_attr")
                        && let Some(out_vals) = try_staticize_broadcast_to(&in_vals, target_shape)
                    {
                        Some(Attribute::Array(
                            out_vals.into_iter().map(Attribute::Int).collect(),
                        ))
                    } else {
                        None
                    }
                }
                "mo.broadcast_shape" => {
                    if operand_ops.len() == 2
                        && let (Some(lhs_op), Some(rhs_op)) = (&operand_ops[0], &operand_ops[1])
                        && let (Some(lhs_dims), Some(rhs_dims)) = (
                            extract_dim_vector_attr(lhs_op, "shape_attr"),
                            extract_dim_vector_attr(rhs_op, "shape_attr"),
                        )
                    {
                        let merged = try_staticize_broadcast_shape(&lhs_dims, &rhs_dims)?;
                        Some(encode_dim_vector_attr(&merged))
                    } else {
                        None
                    }
                }
                "mo.concat" => {
                    let axis = match op.attributes.get("axis") {
                        Some(Attribute::Int(a)) => *a,
                        None => 0,
                        _ => -1,
                    };
                    let is_1d = result_shape.as_ref().is_none_or(|s| s.len() == 1);
                    if axis == 0 && is_1d && !operand_ops.is_empty() {
                        let mut combined = Vec::new();
                        let mut all_valid = true;
                        for maybe_op in &operand_ops {
                            if let Some(in_op) = maybe_op
                                && let Some(dims) = extract_dim_vector_attr(in_op, "shape_attr")
                            {
                                combined.extend(dims);
                            } else {
                                all_valid = false;
                                break;
                            }
                        }
                        if all_valid {
                            Some(encode_dim_vector_attr(&combined))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "mo.cast" => {
                    if let Some(Some(in_op)) = operand_ops.first()
                        && let Some(in_vals) = extract_int_vector_attr(in_op, "shape_attr")
                    {
                        Some(Attribute::Array(
                            in_vals.into_iter().map(Attribute::Int).collect(),
                        ))
                    } else {
                        None
                    }
                }
                "mo.add" => {
                    if operand_ops.len() == 2
                        && let (Some(lhs_op), Some(rhs_op)) = (&operand_ops[0], &operand_ops[1])
                        && let (Some(lhs_vals), Some(rhs_vals)) = (
                            extract_int_vector_attr(lhs_op, "shape_attr"),
                            extract_int_vector_attr(rhs_op, "shape_attr"),
                        )
                        && lhs_vals.len() == rhs_vals.len()
                    {
                        let summed: Vec<Attribute> = lhs_vals
                            .into_iter()
                            .zip(rhs_vals)
                            .map(|(a, b)| Attribute::Int(a.wrapping_add(b)))
                            .collect();
                        Some(Attribute::Array(summed))
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(attr) = folded_attr
                && let Some(op_mut) = module.ops.get_mut(op_id)
            {
                op_mut.attributes.insert("shape_attr".to_string(), attr);
                folded_count += 1;
            }
        }
    }

    Ok(folded_count)
}

// ---------------------------------------------------------------------------
// 5. Multi-Device Graph-to-Model Lowering
// ---------------------------------------------------------------------------

/// Walks `graph_op` and all nested regions to collect unique device labels
/// referenced by `device` / `device_ref` attributes or tensor `encoding` fields.
///
/// The host device `"cpu:0"` is always placed at index `0`, followed by all
/// other discovered devices sorted in lexicographical order.
#[must_use]
pub fn collect_sorted_graph_devices(module: &Module, graph_op: OpId) -> Vec<String> {
    let mut non_host_devices = BTreeSet::new();

    fn record_device(label: &str, non_host: &mut BTreeSet<String>) {
        if !label.is_empty() && label != "cpu:0" {
            non_host.insert(label.to_string());
        }
    }

    fn scan_op(module: &Module, op_id: OpId, non_host: &mut BTreeSet<String>) {
        let Some(op) = module.ops.get(op_id) else {
            return;
        };
        for key in ["device", "device_ref"] {
            if let Some(Attribute::String(dev)) = op.attributes.get(key) {
                record_device(dev, non_host);
            }
        }
        for &vid in op.operands.iter().chain(op.results.iter()) {
            if let Some(val) = module.values.get(vid)
                && let Type::Tensor(TensorType {
                    encoding: Some(dev),
                    ..
                }) = &val.ty
            {
                record_device(dev, non_host);
            }
        }
        for &region_id in &op.regions {
            let Some(region) = module.regions.get(region_id) else {
                continue;
            };
            for &block_id in &region.blocks {
                let Some(block) = module.blocks.get(block_id) else {
                    continue;
                };
                for &arg_id in &block.arguments {
                    if let Some(val) = module.values.get(arg_id)
                        && let Type::Tensor(TensorType {
                            encoding: Some(dev),
                            ..
                        }) = &val.ty
                    {
                        record_device(dev, non_host);
                    }
                }
                for &child_op in &block.ops {
                    scan_op(module, child_op, non_host);
                }
            }
        }
    }

    scan_op(module, graph_op, &mut non_host_devices);

    let mut sorted = Vec::with_capacity(1 + non_host_devices.len());
    sorted.push("cpu:0".to_string());
    sorted.extend(non_host_devices);
    sorted
}

/// Lowers `mo.graph` operations in `module` into two-region `mgp.model`
/// operations:
/// - Region 0 (`init`): Creates `mgp.context.create_device_runtime` for each
///   device in [`collect_sorted_graph_devices`] order (`"cpu:0"` first) and
///   yields the device runtime contexts.
/// - Region 1 (`execute`): Receives the execute chain and original graph
///   arguments, enforces `minimum_cpu_alignment` on `mo.constant` /
///   `mogg.constant` operations, and replaces `mo.output` with `mgp.yield`.
///
/// Returns the number of `mo.graph` operations lowered.
pub fn lower_graph_to_mgp_model(module: &mut Module, minimum_cpu_alignment: u64) -> usize {
    let mut lowered_count = 0usize;
    let op_ids: Vec<OpId> = module.ops.keys().collect();

    for op_id in op_ids {
        let is_mo_graph = module
            .ops
            .get(op_id)
            .is_some_and(|op| op.name.full_name() == "mo.graph");
        if !is_mo_graph {
            continue;
        }

        let sorted_devices = collect_sorted_graph_devices(module, op_id);
        let Some(orig_regions) = module.ops.get(op_id).map(|op| op.regions.clone()) else {
            continue;
        };
        let Some(&execute_region_id) = orig_regions.first() else {
            continue;
        };

        // Build Region 0 (init region) with mgp.context.create_device_runtime per device.
        let init_block_id = module.create_block(Some("init".to_string()));
        let mut dev_ctx_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
        let mut device_to_ctx: BTreeMap<String, ValueId> = BTreeMap::new();

        for (dev_idx, dev_label) in sorted_devices.iter().enumerate() {
            let mut attrs = IndexMap::new();
            attrs.insert(
                "device_label".to_string(),
                Attribute::String(dev_label.clone()),
            );
            attrs.insert("device_index".to_string(), Attribute::Int(dev_idx as i64));
            let create_dev_id = module.add_operation(
                init_block_id,
                make_op_name("mgp.context.create_device_runtime"),
                SmallVec::new(),
                vec![(None, Type::Opaque("!mgp.device.context".to_string()))],
                attrs,
                Vec::new(),
                false,
                Location::Unknown,
            );
            let ctx_val = module.ops[create_dev_id].results[0];
            dev_ctx_vals.push(ctx_val);
            device_to_ctx.insert(dev_label.clone(), ctx_val);
        }

        module.add_operation(
            init_block_id,
            make_op_name("mgp.yield"),
            dev_ctx_vals,
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let init_region_id = module.create_region(vec![init_block_id]);

        // Update Region 1 (execute region): enforce minimum_cpu_alignment on constants
        // and rewrite mo.output -> mgp.yield.
        if let Some(exec_region) = module.regions.get(execute_region_id) {
            let exec_blocks = exec_region.blocks.clone();
            for bid in exec_blocks {
                let Some(block) = module.blocks.get(bid) else {
                    continue;
                };
                let block_ops = block.ops.clone();
                for child_op_id in block_ops {
                    let Some(child_op) = module.ops.get_mut(child_op_id) else {
                        continue;
                    };
                    let child_full = child_op.name.full_name();
                    if child_full == "mo.constant" || child_full == "mogg.constant" {
                        let current_align = match child_op.attributes.get("alignment") {
                            Some(&Attribute::Int(a)) if a > 0 => a as u64,
                            _ => 1,
                        };
                        let effective_align = current_align.max(minimum_cpu_alignment);
                        child_op.attributes.insert(
                            "alignment".to_string(),
                            Attribute::Int(effective_align as i64),
                        );
                    } else if child_full == "mo.output" {
                        child_op.name = make_op_name("mgp.yield");
                    }
                }
            }
        }

        if let Some(graph_mut) = module.ops.get_mut(op_id) {
            graph_mut.name = make_op_name("mgp.model");
            graph_mut.attributes.insert(
                "sorted_devices".to_string(),
                Attribute::Array(sorted_devices.into_iter().map(Attribute::String).collect()),
            );
            graph_mut.attributes.insert(
                "minimum_cpu_alignment".to_string(),
                Attribute::Int(minimum_cpu_alignment as i64),
            );
            graph_mut.regions = vec![init_region_id, execute_region_id];
            lowered_count += 1;
        }
    }

    lowered_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_static_and_symbolic() {
        let lhs = vec![
            Dim::Symbolic("batch".to_string()),
            Dim::Static(1),
            Dim::Static(8),
        ];
        let rhs = vec![Dim::Static(16), Dim::Static(1)];
        let merged = try_staticize_broadcast_shape(&lhs, &rhs);
        assert_eq!(
            merged,
            Ok(vec![
                Dim::Symbolic("batch".to_string()),
                Dim::Static(16),
                Dim::Static(8),
            ])
        );

        let bad_lhs = vec![Dim::Static(4)];
        let bad_rhs = vec![Dim::Static(8)];
        assert!(try_staticize_broadcast_shape(&bad_lhs, &bad_rhs).is_err());
    }

    #[test]
    fn test_param_lowering_forward_placeholder_resolution() {
        let mut module = Module::new(Some("param_test".to_string()));
        let root_region = module.root_region;
        let mut state = ParameterLoweringState::new();

        // Reference 'K' before it is declared.
        let expr = ParamExpr::BinaryOp {
            opcode: "add".to_string(),
            lhs: Box::new(ParamExpr::ParamDeclRef("K".to_string())),
            rhs: Box::new(ParamExpr::Constant(1)),
        };
        let add_val = state.lower_expr(&mut module, root_region, &expr);
        assert!(add_val.is_ok());

        // Now declare 'K = 32'; placeholder should be resolved and removed.
        let k_val = state.lower_param_decl(&mut module, root_region, "K", &ParamExpr::Constant(32));
        assert!(k_val.is_ok());
        assert!(
            state
                .lower_param_decl(&mut module, root_region, "K", &ParamExpr::Constant(64))
                .is_err()
        );
    }
}
