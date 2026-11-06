//! Tensor-to-kernel dialect lowering (`MOToMAP`), kernel-to-routine lowering (`MAPToMOGG`),
//! primitive async routine wrapping (`LowerPrimitives`), and view-producer fusion (`ViewFuser`).

use crate::ir::{
    Attribute, Block, BlockId, DType, Dim, Module, OpId, OpName, Operation, Region, Type, Value,
    ValueDef, ValueId,
};
use indexmap::{IndexMap, IndexSet};
use smallvec::SmallVec;

/// Maximum tensor rank supported for direct `map.iter.elementwise` foreach loop lowering.
pub const MAX_FOREACH_RANK: usize = 6;

/// Asynchronous launch dispatch mode attached to lowered kernel iterations and async routines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncLaunchDispatch {
    /// Launch kernel or host callback immediately on the active stream.
    Immediate,
    /// Defer launch into an asynchronous stream queue or device-graph capture region.
    Deferred,
}

impl AsyncLaunchDispatch {
    /// Canonical attribute string representation (`"immediate"` or `"deferred"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Immediate => "immediate",
            Self::Deferred => "deferred",
        }
    }
}

/// Associative reduction monoid for `map.iter.rowwise` kernel lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowwiseReductionMonoid {
    /// Additive reduction (`0` identity).
    Add,
    /// Multiplicative reduction (`1` identity).
    Mul,
    /// Minimum reduction (`+inf` / `MAX` identity).
    Min,
    /// Maximum reduction (`-inf` / `MIN` identity).
    Max,
    /// Bitwise/logical AND reduction (`all-ones` / `true` identity).
    And,
    /// Bitwise/logical OR reduction (`0` / `false` identity).
    Or,
}

impl RowwiseReductionMonoid {
    /// Resolves a reduction op suffix (`"add"`, `"sum"`, `"mul"`, `"prod"`, `"min"`, `"max"`,
    /// `"and"`, `"or"`) into a [`RowwiseReductionMonoid`].
    pub fn from_op_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "add" | "sum" => Some(Self::Add),
            "mul" | "prod" => Some(Self::Mul),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "and" => Some(Self::And),
            "or" => Some(Self::Or),
            _ => None,
        }
    }

    /// Canonical monoid name stored on `map.iter.rowwise`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
        }
    }

    /// Combiner operation name inside the `map.iter.rowwise` body region.
    pub fn combiner_op_name(self) -> OpName {
        OpName::new("map", format!("combiner.{}", self.as_str()))
    }

    /// Identity neutral element attribute for `dtype`.
    pub fn identity_attr(self, dtype: &DType) -> Attribute {
        match (self, dtype) {
            (Self::Add | Self::Or, DType::F16 | DType::BF16 | DType::F32 | DType::F64) => {
                Attribute::Float(0.0)
            }
            (Self::Add | Self::Or, _) => Attribute::Int(0),
            (Self::Mul, DType::F16 | DType::BF16 | DType::F32 | DType::F64) => {
                Attribute::Float(1.0)
            }
            (Self::Mul, _) => Attribute::Int(1),
            (Self::Min, DType::F16 | DType::BF16 | DType::F32 | DType::F64) => {
                Attribute::Float(f64::INFINITY)
            }
            (Self::Min, _) => Attribute::Int(i64::MAX),
            (Self::Max, DType::F16 | DType::BF16 | DType::F32 | DType::F64) => {
                Attribute::Float(f64::NEG_INFINITY)
            }
            (Self::Max, _) => Attribute::Int(i64::MIN),
            (Self::And, DType::I1) => Attribute::Bool(true),
            (Self::And, _) => Attribute::Int(-1),
        }
    }
}

/// Summary report produced by [`lower_tensor_ops_to_map_kernels`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MapKernelLoweringReport {
    /// Number of elementwise ops lowered to `map.iter.elementwise`.
    pub elementwise_lowered: usize,
    /// Number of reduction ops lowered to `map.iter.rowwise`.
    pub rowwise_lowered: usize,
    /// Number of opaque/custom ops lowered to `map.iter.opaque`.
    pub opaque_lowered: usize,
    /// Number of elementwise ops skipped because their rank exceeded [`MAX_FOREACH_RANK`].
    pub rank_exceeded_skipped: usize,
}

impl MapKernelLoweringReport {
    /// Total number of high-level tensor operations lowered into `map.iter.*` kernels.
    pub fn total_lowered(&self) -> usize {
        self.elementwise_lowered + self.rowwise_lowered + self.opaque_lowered
    }
}

fn tensor_rank_and_dtype(ty: &Type) -> (usize, DType) {
    match ty {
        Type::Tensor(tt) => (tt.shape.len(), tt.dtype.clone()),
        Type::Scalar(dt) => (0, dt.clone()),
        _ => (1, DType::F32),
    }
}

fn scalar_type(dtype: DType) -> Type {
    Type::Scalar(dtype)
}

fn index_scalar_type() -> Type {
    Type::Scalar(DType::I64)
}

/// Lowers high-level tensor operations in `module` into structured `map.iter.*` kernel operations:
/// - `mo.elementwise.<op>` / `rmo.elementwise.<op>` -> `map.iter.elementwise` with an inner region
///   containing `(idx_0, ..., idx_{r-1})` block arguments, explicit `map.load` per tensor input,
///   `map.call.elementwise`, and `map.store` + `map.yield`. Operations whose output rank exceeds
///   [`MAX_FOREACH_RANK`] (`6`) are left unlowered and counted in `rank_exceeded_skipped`.
/// - `mo.reduce.<monoid>` / `rmo.reduce.<monoid>` -> `map.iter.rowwise` with `reduction_monoid`,
///   `identity`, and `axis` attributes and an inner `(acc, elem)` combiner region.
/// - `mo.opaque_kernel` / `mo.custom_op` -> `map.iter.opaque` wrapping an inner region with
///   `map.iter.prologue` input loads, `map.call.opaque`, and `map.iter.epilogue` output stores,
///   preserving `reads_memory`, `writes_memory`, and trailing chain operands.
pub fn lower_tensor_ops_to_map_kernels(
    module: &mut Module,
    dispatch: AsyncLaunchDispatch,
) -> MapKernelLoweringReport {
    let mut report = MapKernelLoweringReport::default();
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let orig_ops = module.blocks[blk_id].ops.clone();
        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            let elem_suffix = full
                .strip_prefix("mo.elementwise.")
                .or_else(|| full.strip_prefix("rmo.elementwise."));
            if let Some(kernel_fn) = elem_suffix {
                if op.results.is_empty() {
                    continue;
                }
                let out_val = op.results[0];
                let (rank, out_dtype) = tensor_rank_and_dtype(&module.values[out_val].ty);
                if rank > MAX_FOREACH_RANK {
                    report.rank_exceeded_skipped += 1;
                    continue;
                }
                lower_single_elementwise_op(
                    module, op_id, &op, kernel_fn, rank, out_dtype, dispatch,
                );
                report.elementwise_lowered += 1;
                continue;
            }

            let reduce_suffix = full
                .strip_prefix("mo.reduce.")
                .or_else(|| full.strip_prefix("rmo.reduce."));
            if let Some(monoid_str) = reduce_suffix
                && let Some(monoid) = RowwiseReductionMonoid::from_op_suffix(monoid_str)
            {
                if op.operands.is_empty() || op.results.is_empty() {
                    continue;
                }
                lower_single_rowwise_op(module, op_id, &op, monoid, dispatch);
                report.rowwise_lowered += 1;
                continue;
            }

            if matches!(full.as_str(), "mo.opaque_kernel" | "mo.custom_op") {
                lower_single_opaque_op(module, op_id, &op, dispatch);
                report.opaque_lowered += 1;
            }
        }
    }

    report
}

fn lower_single_elementwise_op(
    module: &mut Module,
    op_id: OpId,
    op: &Operation,
    kernel_fn: &str,
    rank: usize,
    out_dtype: DType,
    dispatch: AsyncLaunchDispatch,
) {
    let loc = op.location.clone();
    let idx_ty = index_scalar_type();
    let body_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });

    let mut idx_args: SmallVec<[ValueId; 4]> = SmallVec::new();
    for dim_i in 0..rank {
        let arg_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: idx_ty.clone(),
            def: ValueDef::BlockArgument {
                block: body_blk,
                index: dim_i as u32,
            },
        });
        module.blocks[body_blk].arguments.push(arg_val);
        idx_args.push(arg_val);
    }

    let mut loaded_scalars: SmallVec<[ValueId; 4]> = SmallVec::new();
    for &in_tensor in &op.operands {
        let (_, in_dt) = tensor_rank_and_dtype(&module.values[in_tensor].ty);
        let mut load_opds: SmallVec<[ValueId; 4]> = SmallVec::with_capacity(1 + idx_args.len());
        load_opds.push(in_tensor);
        load_opds.extend(idx_args.iter().copied());
        let load_op = module.ops.insert_with_key(|id| Operation {
            id,
            name: OpName::new("map", "load"),
            operands: load_opds,
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: loc.clone(),
        });
        let load_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: scalar_type(in_dt),
            def: ValueDef::OpResult {
                op: load_op,
                index: 0,
            },
        });
        module.ops[load_op].results.push(load_val);
        module.blocks[body_blk].ops.push(load_op);
        loaded_scalars.push(load_val);
    }

    let mut call_attrs = IndexMap::new();
    call_attrs.insert(
        "callee".to_string(),
        Attribute::String(kernel_fn.to_string()),
    );
    let call_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("map", "call.elementwise"),
        operands: loaded_scalars,
        results: SmallVec::new(),
        attributes: call_attrs,
        regions: Vec::new(),
        custom_syntax: false,
        location: loc.clone(),
    });
    let call_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: scalar_type(out_dtype),
        def: ValueDef::OpResult {
            op: call_op,
            index: 0,
        },
    });
    module.ops[call_op].results.push(call_val);
    module.blocks[body_blk].ops.push(call_op);

    let mut store_opds: SmallVec<[ValueId; 4]> = SmallVec::with_capacity(1 + idx_args.len());
    store_opds.push(call_val);
    store_opds.extend(idx_args.iter().copied());
    let store_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("map", "store"),
        operands: store_opds,
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc.clone(),
    });
    module.blocks[body_blk].ops.push(store_op);

    let yield_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("map", "yield"),
        operands: SmallVec::from_slice(&[call_val]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc,
    });
    module.blocks[body_blk].ops.push(yield_op);

    let body_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![body_blk],
    });

    if let Some(op_mut) = module.ops.get_mut(op_id) {
        op_mut.name = OpName::new("map", "iter.elementwise");
        op_mut.regions = vec![body_reg];
        op_mut
            .attributes
            .insert("rank".to_string(), Attribute::Int(rank as i64));
        op_mut.attributes.insert(
            "dispatch".to_string(),
            Attribute::String(dispatch.as_str().to_string()),
        );
        op_mut.attributes.insert(
            "kernel_fn".to_string(),
            Attribute::String(kernel_fn.to_string()),
        );
    }
}

fn lower_single_rowwise_op(
    module: &mut Module,
    op_id: OpId,
    op: &Operation,
    monoid: RowwiseReductionMonoid,
    dispatch: AsyncLaunchDispatch,
) {
    let loc = op.location.clone();
    let (_, elem_dtype) = tensor_rank_and_dtype(&module.values[op.operands[0]].ty);
    let elem_ty = scalar_type(elem_dtype.clone());

    let body_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let acc_arg = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: elem_ty.clone(),
        def: ValueDef::BlockArgument {
            block: body_blk,
            index: 0,
        },
    });
    let val_arg = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: elem_ty.clone(),
        def: ValueDef::BlockArgument {
            block: body_blk,
            index: 1,
        },
    });
    module.blocks[body_blk].arguments.push(acc_arg);
    module.blocks[body_blk].arguments.push(val_arg);

    let comb_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: monoid.combiner_op_name(),
        operands: SmallVec::from_slice(&[acc_arg, val_arg]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc.clone(),
    });
    let comb_val = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: elem_ty,
        def: ValueDef::OpResult {
            op: comb_op,
            index: 0,
        },
    });
    module.ops[comb_op].results.push(comb_val);
    module.blocks[body_blk].ops.push(comb_op);

    let yield_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("map", "yield"),
        operands: SmallVec::from_slice(&[comb_val]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc,
    });
    module.blocks[body_blk].ops.push(yield_op);

    let body_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![body_blk],
    });

    let axis = match op.attributes.get("axis") {
        Some(Attribute::Int(a)) => *a,
        _ => -1,
    };

    if let Some(op_mut) = module.ops.get_mut(op_id) {
        op_mut.name = OpName::new("map", "iter.rowwise");
        op_mut.regions = vec![body_reg];
        op_mut.attributes.insert(
            "reduction_monoid".to_string(),
            Attribute::String(monoid.as_str().to_string()),
        );
        op_mut
            .attributes
            .insert("identity".to_string(), monoid.identity_attr(&elem_dtype));
        op_mut
            .attributes
            .insert("axis".to_string(), Attribute::Int(axis));
        op_mut.attributes.insert(
            "dispatch".to_string(),
            Attribute::String(dispatch.as_str().to_string()),
        );
    }
}

fn lower_single_opaque_op(
    module: &mut Module,
    op_id: OpId,
    op: &Operation,
    dispatch: AsyncLaunchDispatch,
) {
    let loc = op.location.clone();
    let symbol = match op
        .attributes
        .get("symbol")
        .or_else(|| op.attributes.get("callee"))
    {
        Some(Attribute::String(s)) => s.clone(),
        _ => "opaque_kernel".to_string(),
    };
    let reads_memory = match op.attributes.get("reads_memory") {
        Some(Attribute::Bool(b)) => *b,
        _ => true,
    };
    let writes_memory = match op.attributes.get("writes_memory") {
        Some(Attribute::Bool(b)) => *b,
        _ => false,
    };

    let body_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });

    // Separate trailing `!grt.chain` operand if present.
    let mut tensor_operands: SmallVec<[ValueId; 4]> = SmallVec::new();
    let mut chain_operands: SmallVec<[ValueId; 4]> = SmallVec::new();
    for &opd in &op.operands {
        if matches!(&module.values[opd].ty, Type::Opaque(c) if c.contains("chain")) {
            chain_operands.push(opd);
        } else {
            tensor_operands.push(opd);
        }
    }

    let mut prologue_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
    for (idx, &t_opd) in tensor_operands.iter().enumerate() {
        let mut p_attrs = IndexMap::new();
        p_attrs.insert("input_index".to_string(), Attribute::Int(idx as i64));
        let p_op = module.ops.insert_with_key(|id| Operation {
            id,
            name: OpName::new("map", "iter.prologue"),
            operands: SmallVec::from_slice(&[t_opd]),
            results: SmallVec::new(),
            attributes: p_attrs,
            regions: Vec::new(),
            custom_syntax: false,
            location: loc.clone(),
        });
        let p_ty = module.values[t_opd].ty.clone();
        let p_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: p_ty,
            def: ValueDef::OpResult { op: p_op, index: 0 },
        });
        module.ops[p_op].results.push(p_val);
        module.blocks[body_blk].ops.push(p_op);
        prologue_vals.push(p_val);
    }

    let mut call_opds = prologue_vals;
    call_opds.extend(chain_operands.iter().copied());
    let mut call_attrs = IndexMap::new();
    call_attrs.insert("symbol".to_string(), Attribute::String(symbol.clone()));
    call_attrs.insert("reads_memory".to_string(), Attribute::Bool(reads_memory));
    call_attrs.insert("writes_memory".to_string(), Attribute::Bool(writes_memory));

    let call_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("map", "call.opaque"),
        operands: call_opds,
        results: SmallVec::new(),
        attributes: call_attrs,
        regions: Vec::new(),
        custom_syntax: false,
        location: loc.clone(),
    });

    let mut epilogue_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
    let mut out_idx = 0usize;
    for (r_idx, &orig_res) in op.results.iter().enumerate() {
        let res_ty = module.values[orig_res].ty.clone();
        let is_chain = matches!(&res_ty, Type::Opaque(c) if c.contains("chain"));
        let c_val = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: res_ty.clone(),
            def: ValueDef::OpResult {
                op: call_op,
                index: r_idx as u32,
            },
        });
        module.ops[call_op].results.push(c_val);
        if is_chain {
            epilogue_vals.push(c_val);
        } else {
            let mut e_attrs = IndexMap::new();
            e_attrs.insert("output_index".to_string(), Attribute::Int(out_idx as i64));
            out_idx += 1;
            let e_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("map", "iter.epilogue"),
                operands: SmallVec::from_slice(&[c_val]),
                results: SmallVec::new(),
                attributes: e_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let e_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: res_ty,
                def: ValueDef::OpResult { op: e_op, index: 0 },
            });
            module.ops[e_op].results.push(e_val);
            module.blocks[body_blk].ops.push(e_op);
            epilogue_vals.push(e_val);
        }
    }

    // Ensure `map.call.opaque` precedes all `map.iter.epilogue` ops.
    let epilogue_count = module.blocks[body_blk].ops.len() - tensor_operands.len();
    let insert_idx = module.blocks[body_blk].ops.len() - epilogue_count;
    module.blocks[body_blk].ops.insert(insert_idx, call_op);

    let yield_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("map", "yield"),
        operands: epilogue_vals,
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc,
    });
    module.blocks[body_blk].ops.push(yield_op);

    let body_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![body_blk],
    });

    if let Some(op_mut) = module.ops.get_mut(op_id) {
        op_mut.name = OpName::new("map", "iter.opaque");
        op_mut.regions = vec![body_reg];
        op_mut
            .attributes
            .insert("symbol".to_string(), Attribute::String(symbol));
        op_mut
            .attributes
            .insert("reads_memory".to_string(), Attribute::Bool(reads_memory));
        op_mut
            .attributes
            .insert("writes_memory".to_string(), Attribute::Bool(writes_memory));
        op_mut.attributes.insert(
            "dispatch".to_string(),
            Attribute::String(dispatch.as_str().to_string()),
        );
    }
}

// ============================================================================
// MAPToMOGG & LowerPrimitives: Routine Lowering & Async Wrapping
// ============================================================================

fn collect_unique_symbolic_dims_from_type(ty: &Type, out: &mut IndexSet<String>) {
    if let Type::Tensor(tt) = ty {
        for d in &tt.shape {
            if let Dim::Symbolic(s) = d {
                out.insert(s.clone());
            }
        }
    }
}

/// Lowers `map.iter.opaque` and `mo.invoke_shape_func` operations in `module` into `mogg.routine`
/// operations whose inner bodies reconstruct tensor specifications from `!mogg.pointer` block
/// arguments (`mogg.shape.create` + `mogg.tensor.create`) and deduplicate dynamic symbolic
/// dimension arguments (`!mogg.dim`) across all input and output tensor shapes.
///
/// Returns the number of operations lowered into `mogg.routine` invocations.
pub fn lower_map_kernels_to_routines(module: &mut Module) -> usize {
    let mut lowered_count = 0usize;
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let orig_ops = module.blocks[blk_id].ops.clone();
        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if !matches!(full.as_str(), "map.iter.opaque" | "mo.invoke_shape_func") {
                continue;
            }
            let loc = op.location.clone();
            let routine_sym = match op
                .attributes
                .get("symbol")
                .or_else(|| op.attributes.get("shape_func"))
            {
                Some(Attribute::String(s)) => s.clone(),
                _ => format!("routine_{lowered_count}"),
            };

            // Collect deduplicated dynamic symbolic dimensions across operands and results.
            let mut unique_dims: IndexSet<String> = IndexSet::new();
            for &opd in &op.operands {
                collect_unique_symbolic_dims_from_type(&module.values[opd].ty, &mut unique_dims);
            }
            for &res in &op.results {
                collect_unique_symbolic_dims_from_type(&module.values[res].ty, &mut unique_dims);
            }

            let routine_blk = module.blocks.insert_with_key(|id| Block {
                id,
                label: None,
                arguments: Vec::new(),
                ops: Vec::new(),
            });

            let ptr_ty = Type::Opaque("!mogg.pointer".to_string());
            let dim_ty = Type::Opaque("!mogg.dim".to_string());

            // 1. Pointer block arguments for each tensor operand.
            let mut ptr_args: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (idx, &opd) in op.operands.iter().enumerate() {
                let arg_ty = if matches!(module.values[opd].ty, Type::Tensor(_)) {
                    ptr_ty.clone()
                } else {
                    module.values[opd].ty.clone()
                };
                let arg_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: arg_ty,
                    def: ValueDef::BlockArgument {
                        block: routine_blk,
                        index: idx as u32,
                    },
                });
                module.blocks[routine_blk].arguments.push(arg_val);
                ptr_args.push(arg_val);
            }

            // 2. Deduplicated `!mogg.dim` block arguments for dynamic symbolic dimensions.
            let mut dim_arg_map: IndexMap<String, ValueId> = IndexMap::new();
            for sym_name in &unique_dims {
                let arg_idx = module.blocks[routine_blk].arguments.len() as u32;
                let dim_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: Some(sym_name.clone()),
                    ty: dim_ty.clone(),
                    def: ValueDef::BlockArgument {
                        block: routine_blk,
                        index: arg_idx,
                    },
                });
                module.blocks[routine_blk].arguments.push(dim_val);
                dim_arg_map.insert(sym_name.clone(), dim_val);
            }

            // 3. Rebuild tensor values inside the routine body from pointer + shape.
            let mut rebuilt_tensors: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (&opd, &ptr_arg) in op.operands.iter().zip(ptr_args.iter()) {
                let orig_ty = module.values[opd].ty.clone();
                if let Type::Tensor(ref tt) = orig_ty {
                    let mut shape_dim_opds: SmallVec<[ValueId; 4]> = SmallVec::new();
                    for d in &tt.shape {
                        if let Dim::Symbolic(s) = d
                            && let Some(&dv) = dim_arg_map.get(s)
                        {
                            shape_dim_opds.push(dv);
                        }
                    }
                    let shape_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mogg", "shape.create"),
                        operands: shape_dim_opds,
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: loc.clone(),
                    });
                    let shape_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: Type::Opaque("!mogg.shape".to_string()),
                        def: ValueDef::OpResult {
                            op: shape_op,
                            index: 0,
                        },
                    });
                    module.ops[shape_op].results.push(shape_val);
                    module.blocks[routine_blk].ops.push(shape_op);

                    let tensor_op = module.ops.insert_with_key(|id| Operation {
                        id,
                        name: OpName::new("mogg", "tensor.create"),
                        operands: SmallVec::from_slice(&[ptr_arg, shape_val]),
                        results: SmallVec::new(),
                        attributes: IndexMap::new(),
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: loc.clone(),
                    });
                    let tensor_val = module.values.insert_with_key(|id| Value {
                        id,
                        ssa_name: None,
                        ty: orig_ty,
                        def: ValueDef::OpResult {
                            op: tensor_op,
                            index: 0,
                        },
                    });
                    module.ops[tensor_op].results.push(tensor_val);
                    module.blocks[routine_blk].ops.push(tensor_op);
                    rebuilt_tensors.push(tensor_val);
                } else {
                    rebuilt_tensors.push(ptr_arg);
                }
            }

            let mut exec_attrs = IndexMap::new();
            exec_attrs.insert("symbol".to_string(), Attribute::String(routine_sym.clone()));
            let exec_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg", "kernel.invoke"),
                operands: rebuilt_tensors,
                results: SmallVec::new(),
                attributes: exec_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let mut yield_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (r_idx, &orig_res) in op.results.iter().enumerate() {
                let r_ty = module.values[orig_res].ty.clone();
                let r_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: r_ty,
                    def: ValueDef::OpResult {
                        op: exec_op,
                        index: r_idx as u32,
                    },
                });
                module.ops[exec_op].results.push(r_val);
                yield_vals.push(r_val);
            }
            module.blocks[routine_blk].ops.push(exec_op);

            let ret_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg", "routine.return"),
                operands: yield_vals,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            module.blocks[routine_blk].ops.push(ret_op);

            let routine_reg = module.regions.insert_with_key(|id| Region {
                id,
                blocks: vec![routine_blk],
            });

            let dedup_dims_attr: Vec<Attribute> = unique_dims
                .iter()
                .map(|d| Attribute::String(d.clone()))
                .collect();

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName::new("mogg", "routine");
                op_mut.regions = vec![routine_reg];
                op_mut
                    .attributes
                    .insert("symbol".to_string(), Attribute::String(routine_sym));
                op_mut.attributes.insert(
                    "deduplicated_dynamic_dims".to_string(),
                    Attribute::Array(dedup_dims_attr),
                );
                op_mut.attributes.insert(
                    "num_deduplicated_dims".to_string(),
                    Attribute::Int(unique_dims.len() as i64),
                );
            }
            lowered_count += 1;
        }
    }

    lowered_count
}

fn is_primitive_op_for_async_wrap(full_name: &str) -> bool {
    full_name == "kgen.param.constant"
        || full_name.starts_with("index.")
        || full_name.starts_with("pop.")
        || (full_name.starts_with("mgp.")
            && !full_name.starts_with("mgp.routine.")
            && !full_name.starts_with("mgp.device_graph."))
}

fn is_primitive_excluded_from_fusion(op: &Operation) -> bool {
    let full = op.name.full_name();
    if matches!(
        full.as_str(),
        "mgp.sync" | "mgp.buffer.alloc" | "mgp.buffer.constant.external"
    ) {
        return true;
    }
    if full == "mgp.device.wait" {
        let src_stream = op.attributes.get("src_stream");
        let dst_stream = op.attributes.get("dst_stream");
        if src_stream != dst_stream {
            return true;
        }
    }
    false
}

/// Lowers scalar `index.constant` operations inside kernel/routine regions into 8-byte aligned
/// `mgp.buffer.constant` + `mgp.buffer.to_index` pairs (so compiled kernel structural hashes stay
/// invariant across scalar literal changes) and wraps outer primitive device/index/parameter
/// operations into `mogg.routine.async` operations with accurate fusion and duplication flags.
///
/// Returns `(converted_index_constants, wrapped_async_primitives)`.
pub fn lower_primitives_to_async_routines(
    module: &mut Module,
    dispatch: AsyncLaunchDispatch,
) -> (usize, usize) {
    let mut converted_constants = 0usize;
    let mut wrapped_primitives = 0usize;

    // Collect blocks belonging to nested kernel/routine regions.
    let mut kernel_blocks: IndexSet<BlockId> = IndexSet::new();
    for (_, op) in &module.ops {
        let full = op.name.full_name();
        if full.starts_with("map.iter.")
            || full.starts_with("mogg.routine")
            || full.starts_with("mgp.routine")
        {
            for &reg_id in &op.regions {
                if let Some(reg) = module.regions.get(reg_id) {
                    for &b_id in &reg.blocks {
                        kernel_blocks.insert(b_id);
                    }
                }
            }
        }
    }

    // Phase 1: Lower `index.constant` inside kernel blocks into `mgp.buffer.constant` +
    // `mgp.buffer.to_index`.
    for &blk_id in &kernel_blocks {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        let mut new_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len());
        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if op.name.full_name() != "index.constant" || op.results.is_empty() {
                new_ops.push(op_id);
                continue;
            }
            let val_attr = op
                .attributes
                .get("value")
                .cloned()
                .unwrap_or(Attribute::Int(0));
            let loc = op.location.clone();
            let mut buf_attrs = IndexMap::new();
            buf_attrs.insert("value".to_string(), val_attr);
            buf_attrs.insert("alignment".to_string(), Attribute::Int(8));

            let buf_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mgp", "buffer.constant"),
                operands: SmallVec::new(),
                results: SmallVec::new(),
                attributes: buf_attrs,
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let buf_val = module.values.insert_with_key(|id| Value {
                id,
                ssa_name: None,
                ty: Type::Opaque("!mgp.buffer".to_string()),
                def: ValueDef::OpResult {
                    op: buf_op,
                    index: 0,
                },
            });
            module.ops[buf_op].results.push(buf_val);
            new_ops.push(buf_op);

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName::new("mgp", "buffer.to_index");
                op_mut.operands = SmallVec::from_slice(&[buf_val]);
                op_mut.attributes.shift_remove("value");
            }
            new_ops.push(op_id);
            converted_constants += 1;
        }
        if let Some(blk_mut) = module.blocks.get_mut(blk_id) {
            blk_mut.ops = new_ops;
        }
    }

    // Phase 2: Wrap primitive ops in non-kernel blocks into `mogg.routine.async`.
    let all_blocks: Vec<BlockId> = module.blocks.keys().collect();
    for blk_id in all_blocks {
        if kernel_blocks.contains(&blk_id) {
            continue;
        }
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();
        for &op_id in &orig_ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let full = op.name.full_name();
            if !is_primitive_op_for_async_wrap(&full) {
                continue;
            }
            let excluded = is_primitive_excluded_from_fusion(&op);
            let can_fuse = !excluded;
            let can_duplicate = full.starts_with("index.") || full == "kgen.param.constant";
            let eff_dispatch = if full == "mgp.sync" {
                AsyncLaunchDispatch::Immediate
            } else {
                dispatch
            };

            let inner_blk = module.blocks.insert_with_key(|id| Block {
                id,
                label: None,
                arguments: Vec::new(),
                ops: Vec::new(),
            });
            let mut inner_opds: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (idx, &opd) in op.operands.iter().enumerate() {
                let arg_ty = module.values[opd].ty.clone();
                let arg_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: arg_ty,
                    def: ValueDef::BlockArgument {
                        block: inner_blk,
                        index: idx as u32,
                    },
                });
                module.blocks[inner_blk].arguments.push(arg_val);
                inner_opds.push(arg_val);
            }

            let loc = op.location.clone();
            let cloned_prim = module.ops.insert_with_key(|id| Operation {
                id,
                name: op.name.clone(),
                operands: inner_opds,
                results: SmallVec::new(),
                attributes: op.attributes.clone(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc.clone(),
            });
            let mut yield_vals: SmallVec<[ValueId; 4]> = SmallVec::new();
            for (r_idx, &orig_res) in op.results.iter().enumerate() {
                let r_ty = module.values[orig_res].ty.clone();
                let r_val = module.values.insert_with_key(|id| Value {
                    id,
                    ssa_name: None,
                    ty: r_ty,
                    def: ValueDef::OpResult {
                        op: cloned_prim,
                        index: r_idx as u32,
                    },
                });
                module.ops[cloned_prim].results.push(r_val);
                yield_vals.push(r_val);
            }
            module.blocks[inner_blk].ops.push(cloned_prim);

            let ret_op = module.ops.insert_with_key(|id| Operation {
                id,
                name: OpName::new("mogg", "routine.return"),
                operands: yield_vals,
                results: SmallVec::new(),
                attributes: IndexMap::new(),
                regions: Vec::new(),
                custom_syntax: false,
                location: loc,
            });
            module.blocks[inner_blk].ops.push(ret_op);

            let inner_reg = module.regions.insert_with_key(|id| Region {
                id,
                blocks: vec![inner_blk],
            });

            if let Some(op_mut) = module.ops.get_mut(op_id) {
                op_mut.name = OpName::new("mogg", "routine.async");
                op_mut.regions = vec![inner_reg];
                op_mut
                    .attributes
                    .insert("wrapped_primitive".to_string(), Attribute::String(full));
                op_mut.attributes.insert(
                    "dispatch".to_string(),
                    Attribute::String(eff_dispatch.as_str().to_string()),
                );
                op_mut
                    .attributes
                    .insert("can_fuse_into_other".to_string(), Attribute::Bool(can_fuse));
                op_mut
                    .attributes
                    .insert("can_be_fused_into".to_string(), Attribute::Bool(can_fuse));
                op_mut
                    .attributes
                    .insert("can_duplicate".to_string(), Attribute::Bool(can_duplicate));
            }
            wrapped_primitives += 1;
        }
    }

    (converted_constants, wrapped_primitives)
}

// ============================================================================
// ViewFuser / InputFusion / OutputFusion
// ============================================================================

/// Summary report produced by [`fuse_view_producers`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ViewFusionReport {
    /// Number of view-producing `map.iter.elementwise` ops fused into consumer
    /// `map.iter.elementwise` bodies.
    pub fused_into_elementwise: usize,
    /// Number of view-producing `map.iter.elementwise` ops fused into consumer
    /// `map.iter.opaque` input prologues (`map.iter.prologue`).
    pub fused_into_prologue: usize,
    /// Number of single-use elementwise consumers fused into producer `map.iter.opaque`
    /// output epilogues (`map.iter.epilogue`).
    pub fused_into_epilogue: usize,
    /// Number of multi-consumer view-producing ops rejected by the residual-chain duplication
    /// guard (`!is_pure_view_foreach && !is_cheap_to_duplicate`).
    pub residual_duplication_blocked: usize,
}

impl ViewFusionReport {
    /// Total number of view/prologue/epilogue fusions performed.
    pub fn total_fused(&self) -> usize {
        self.fused_into_elementwise + self.fused_into_prologue + self.fused_into_epilogue
    }
}

/// Inspects a `map.iter.elementwise` operation and returns `(produces_through_view,
/// is_pure_view_foreach, is_cheap_to_duplicate)`.
pub fn classify_view_foreach(module: &Module, op_id: OpId) -> (bool, bool, bool) {
    let Some(op) = module.ops.get(op_id) else {
        return (false, false, false);
    };
    if op.name.full_name() != "map.iter.elementwise" {
        return (false, false, false);
    }
    let produces_view_attr = matches!(
        op.attributes.get("produces_through_view"),
        Some(Attribute::Bool(true))
    );
    let mut has_view_op = produces_view_attr;
    let mut compute_op_count = 0usize;

    for &reg_id in &op.regions {
        let Some(reg) = module.regions.get(reg_id) else {
            continue;
        };
        for &blk_id in &reg.blocks {
            let Some(blk) = module.blocks.get(blk_id) else {
                continue;
            };
            for &inner_op_id in &blk.ops {
                let Some(inner_op) = module.ops.get(inner_op_id) else {
                    continue;
                };
                let name = inner_op.name.full_name();
                if matches!(
                    name.as_str(),
                    "map.index_rebind" | "map.view_index" | "map.broadcast_index"
                ) {
                    has_view_op = true;
                } else if !matches!(name.as_str(), "map.load" | "map.store" | "map.yield") {
                    if name == "map.call.elementwise" {
                        let callee = match inner_op.attributes.get("callee") {
                            Some(Attribute::String(s)) => s.as_str(),
                            _ => "",
                        };
                        if matches!(
                            callee,
                            "identity" | "view" | "reshape" | "broadcast" | "slice"
                        ) {
                            has_view_op = true;
                        } else {
                            compute_op_count += 1;
                        }
                    } else {
                        compute_op_count += 1;
                    }
                }
            }
        }
    }

    let is_pure_view = has_view_op && compute_op_count == 0;
    let is_cheap = matches!(
        op.attributes.get("is_cheap_to_duplicate"),
        Some(Attribute::Bool(true))
    );
    (has_view_op, is_pure_view, is_cheap)
}

/// Fuses view-producing `map.iter.elementwise` kernels into consumer `map.iter.elementwise` bodies
/// and `map.iter.opaque` input prologues (`map.iter.prologue`), and fuses single-use elementwise
/// consumers of `map.iter.opaque` into `map.iter.epilogue` store epilogues.
///
/// Enforces the residual-chain duplication guard: when a view producer has multiple consumers
/// (`consumer_count > 1`) and has already absorbed non-trivial compute (`!is_pure_view_foreach &&
/// !is_cheap_to_duplicate`), it is not duplicated across consumers and
/// `residual_duplication_blocked` is incremented.
pub fn fuse_view_producers(module: &mut Module) -> ViewFusionReport {
    let mut report = ViewFusionReport::default();
    let block_ids: Vec<BlockId> = module.blocks.keys().collect();

    for blk_id in block_ids {
        let Some(blk) = module.blocks.get(blk_id) else {
            continue;
        };
        let orig_ops = blk.ops.clone();

        // Count consumers per SSA value in `blk_id`.
        let mut consumer_ops: IndexMap<ValueId, Vec<OpId>> = IndexMap::new();
        for &op_id in &orig_ops {
            if let Some(op) = module.ops.get(op_id) {
                for &opd in &op.operands {
                    consumer_ops.entry(opd).or_default().push(op_id);
                }
            }
        }

        let mut dead_ops: IndexSet<OpId> = IndexSet::new();

        // Phase 1: View-producer input fusion into consumer `map.iter.elementwise` or
        // `map.iter.opaque`.
        for &prod_id in &orig_ops {
            if dead_ops.contains(&prod_id) {
                continue;
            }
            let (produces_view, is_pure_view, is_cheap) = classify_view_foreach(module, prod_id);
            if !produces_view {
                continue;
            }
            let Some(prod_op) = module.ops.get(prod_id).cloned() else {
                continue;
            };
            if prod_op.results.len() != 1 || prod_op.operands.is_empty() {
                continue;
            }
            let prod_res = prod_op.results[0];
            let prod_src = prod_op.operands[0];
            let consumers = consumer_ops.get(&prod_res).cloned().unwrap_or_default();
            if consumers.is_empty() {
                continue;
            }

            // Residual-chain duplication guard.
            if consumers.len() > 1 && !is_pure_view && !is_cheap {
                report.residual_duplication_blocked += 1;
                continue;
            }

            let mut all_consumers_fused = true;
            for &cons_id in &consumers {
                let Some(cons_op) = module.ops.get(cons_id).cloned() else {
                    all_consumers_fused = false;
                    continue;
                };
                let cons_name = cons_op.name.full_name();
                if cons_name == "map.iter.elementwise" {
                    // Rewire consumer operand from `prod_res` to `prod_src` and annotate view
                    // fusion inside the consumer.
                    if let Some(c_mut) = module.ops.get_mut(cons_id) {
                        for opd in &mut c_mut.operands {
                            if *opd == prod_res {
                                *opd = prod_src;
                            }
                        }
                        let prev = match c_mut.attributes.get("fused_view_producers") {
                            Some(Attribute::Int(n)) => *n,
                            _ => 0,
                        };
                        c_mut
                            .attributes
                            .insert("fused_view_producers".to_string(), Attribute::Int(prev + 1));
                        c_mut
                            .attributes
                            .insert("produces_through_view".to_string(), Attribute::Bool(true));
                    }
                    // Also rewire `map.load` inside consumer body if it directly referenced
                    // `prod_res`.
                    for &reg_id in &cons_op.regions {
                        if let Some(reg) = module.regions.get(reg_id).cloned() {
                            for &b_id in &reg.blocks {
                                if let Some(b) = module.blocks.get(b_id).cloned() {
                                    for &inner_id in &b.ops {
                                        if let Some(inner_mut) = module.ops.get_mut(inner_id) {
                                            for opd in &mut inner_mut.operands {
                                                if *opd == prod_res {
                                                    *opd = prod_src;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    report.fused_into_elementwise += 1;
                } else if cons_name == "map.iter.opaque" {
                    if let Some(c_mut) = module.ops.get_mut(cons_id) {
                        for opd in &mut c_mut.operands {
                            if *opd == prod_res {
                                *opd = prod_src;
                            }
                        }
                        let prev = match c_mut.attributes.get("fused_input_prologues") {
                            Some(Attribute::Int(n)) => *n,
                            _ => 0,
                        };
                        c_mut.attributes.insert(
                            "fused_input_prologues".to_string(),
                            Attribute::Int(prev + 1),
                        );
                    }
                    // Update `map.iter.prologue` inside the opaque kernel body.
                    for &reg_id in &cons_op.regions {
                        if let Some(reg) = module.regions.get(reg_id).cloned() {
                            for &b_id in &reg.blocks {
                                if let Some(b) = module.blocks.get(b_id).cloned() {
                                    for &inner_id in &b.ops {
                                        if let Some(inner_mut) = module.ops.get_mut(inner_id)
                                            && inner_mut.name.full_name() == "map.iter.prologue"
                                            && inner_mut.operands.contains(&prod_res)
                                        {
                                            for opd in &mut inner_mut.operands {
                                                if *opd == prod_res {
                                                    *opd = prod_src;
                                                }
                                            }
                                            inner_mut.attributes.insert(
                                                "fused_view_rebind".to_string(),
                                                Attribute::Bool(true),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    report.fused_into_prologue += 1;
                } else {
                    all_consumers_fused = false;
                }
            }

            if all_consumers_fused {
                dead_ops.insert(prod_id);
            }
        }

        // Phase 2: Output epilogue fusion (`map.iter.opaque` -> single-use `map.iter.elementwise`).
        for &op_id in &orig_ops {
            if dead_ops.contains(&op_id) {
                continue;
            }
            let Some(opaque_op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if opaque_op.name.full_name() != "map.iter.opaque" || opaque_op.results.is_empty() {
                continue;
            }
            let opaque_res = opaque_op.results[0];
            let consumers = consumer_ops.get(&opaque_res).cloned().unwrap_or_default();
            if consumers.len() != 1 {
                continue;
            }
            let cons_id = consumers[0];
            if dead_ops.contains(&cons_id) {
                continue;
            }
            let Some(cons_op) = module.ops.get(cons_id).cloned() else {
                continue;
            };
            if cons_op.name.full_name() != "map.iter.elementwise"
                || cons_op.operands.len() != 1
                || cons_op.results.len() != 1
            {
                continue;
            }
            let cons_res = cons_op.results[0];
            let epilogue_fn = match cons_op.attributes.get("kernel_fn") {
                Some(Attribute::String(s)) => s.clone(),
                _ => "elementwise_epilogue".to_string(),
            };

            if let Some(opaque_mut) = module.ops.get_mut(op_id) {
                let prev = match opaque_mut.attributes.get("fused_output_epilogues") {
                    Some(Attribute::Int(n)) => *n,
                    _ => 0,
                };
                opaque_mut.attributes.insert(
                    "fused_output_epilogues".to_string(),
                    Attribute::Int(prev + 1),
                );
                opaque_mut.attributes.insert(
                    "epilogue_kernel_fn".to_string(),
                    Attribute::String(epilogue_fn.clone()),
                );
            }
            for &reg_id in &opaque_op.regions {
                if let Some(reg) = module.regions.get(reg_id).cloned() {
                    for &b_id in &reg.blocks {
                        if let Some(b) = module.blocks.get(b_id).cloned() {
                            for &inner_id in &b.ops {
                                if let Some(inner_mut) = module.ops.get_mut(inner_id)
                                    && inner_mut.name.full_name() == "map.iter.epilogue"
                                {
                                    inner_mut.attributes.insert(
                                        "fused_epilogue_fn".to_string(),
                                        Attribute::String(epilogue_fn.clone()),
                                    );
                                }
                            }
                        }
                    }
                }
            }

            module.values[opaque_res].ty = module.values[cons_res].ty.clone();
            module.replace_all_uses(cons_res, opaque_res);
            dead_ops.insert(cons_id);
            report.fused_into_epilogue += 1;
        }

        for &dead_id in &dead_ops {
            module.remove_op(blk_id, dead_id);
        }
    }

    report
}
