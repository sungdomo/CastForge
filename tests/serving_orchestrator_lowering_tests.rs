#![cfg(feature = "serving")]
//! Integration tests for dialect lowering (`MOToMAP`, `MAPToMOGG`, `LowerPrimitives`),
//! view-producer fusion (`ViewFuser`), device-graph finalization (`FinalizeDeviceGraphsPass`),
//! relational async dataflow analysis (`AsyncDataflowDependencies`), RMO legalization
//! (`LegalizeRMOOps`), symbolic dimension optimization (`SymbolicOptimization`), shared-memory
//! region lease authority (`SharedRegionState`), idempotent reply window
//! (`OrchestratorReplyWindow`), pre-prefetch KV load policy (`KvLoadPolicy`), and multi-field
//! object group framing (`ObjectGroupFrame`).

use castforge::ir::{
    Attribute, Block, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, Region,
    TensorType, Type, Value, ValueDef, ValueId,
};
use castforge::passes::serving::{
    AsyncDataflowAnalysis, AsyncLaunchDispatch, ExtentSpec, KvLoadContext, ObjectFieldSpec,
    ObjectGroupFrame, OrchestratorReplyWindow, REGION_CONTRACT_VERSION, RegionContract,
    ReplyWindowOutcome, RowwiseReductionMonoid, SharedRegionState, WriteReserveStatus,
    classify_view_foreach, evaluate_kv_load_policy, finalize_device_graphs, fuse_view_producers,
    is_compatible_shape, legalize_rmo_ops, lower_map_kernels_to_routines,
    lower_primitives_to_async_routines, lower_tensor_ops_to_map_kernels,
    optimize_async_dataflow_dependencies, optimize_symbolic_dims, promote_rmo_dtypes,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

fn make_test_module_with_block() -> (Module, BlockId) {
    let mut module = Module::new(Some("test_module".to_string()));
    let blk_id = module.blocks.insert_with_key(|id| Block {
        id,
        label: Some("entry".to_string()),
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let reg_id = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![blk_id],
    });
    let mut fn_attrs = IndexMap::new();
    fn_attrs.insert(
        "sym_name".to_string(),
        Attribute::String("main".to_string()),
    );
    let fn_op = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("func", "func"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: fn_attrs,
        regions: vec![reg_id],
        custom_syntax: false,
        location: Location::Unknown,
    });
    let root_blk = module.body_block();
    module.blocks[root_blk].ops.push(fn_op);
    (module, blk_id)
}

fn add_block_arg(module: &mut Module, blk_id: BlockId, ty: Type) -> ValueId {
    let idx = module.blocks[blk_id].arguments.len() as u32;
    let v = module.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty,
        def: ValueDef::BlockArgument {
            block: blk_id,
            index: idx,
        },
    });
    module.blocks[blk_id].arguments.push(v);
    v
}

fn push_op(
    module: &mut Module,
    blk_id: BlockId,
    dialect: &str,
    name: &str,
    operands: &[ValueId],
    result_types: &[Type],
    attrs: IndexMap<String, Attribute>,
) -> (OpId, Vec<ValueId>) {
    let op_id = module.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new(dialect, name),
        operands: SmallVec::from_slice(operands),
        results: SmallVec::new(),
        attributes: attrs,
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    let mut results = Vec::with_capacity(result_types.len());
    for (i, ty) in result_types.iter().enumerate() {
        let v = module.values.insert_with_key(|id| Value {
            id,
            ssa_name: None,
            ty: ty.clone(),
            def: ValueDef::OpResult {
                op: op_id,
                index: i as u32,
            },
        });
        module.ops[op_id].results.push(v);
        results.push(v);
    }
    module.blocks[blk_id].ops.push(op_id);
    (op_id, results)
}

fn tensor_ty(dtype: DType, dims: &[i64]) -> Type {
    Type::Tensor(TensorType {
        dtype,
        shape: dims.iter().map(|&d| Dim::Static(d)).collect(),
        encoding: None,
    })
}

fn sym_tensor_ty(dtype: DType, dims: &[Dim]) -> Type {
    Type::Tensor(TensorType {
        dtype,
        shape: dims.to_vec(),
        encoding: None,
    })
}

// ============================================================================
// Batch #29 Tests: Dialect Lowering & View-Producer Fusion
// ============================================================================

#[test]
fn test_lower_tensor_ops_to_map_kernels_elementwise_rowwise_opaque_and_rank_guard() {
    let (mut module, blk_id) = make_test_module_with_block();
    let t2d = tensor_ty(DType::F32, &[4, 8]);
    let t1d = tensor_ty(DType::F32, &[4]);
    let t7d = tensor_ty(DType::F32, &[2, 2, 2, 2, 2, 2, 2]);
    let chain_ty = Type::Opaque("!grt.chain".to_string());

    let a = add_block_arg(&mut module, blk_id, t2d.clone());
    let b = add_block_arg(&mut module, blk_id, t2d.clone());
    let high_rank_in = add_block_arg(&mut module, blk_id, t7d.clone());
    let ch_in = add_block_arg(&mut module, blk_id, chain_ty.clone());

    // 1. 2D elementwise add -> `map.iter.elementwise`
    let (elem_op, elem_res) = push_op(
        &mut module,
        blk_id,
        "mo",
        "elementwise.add",
        &[a, b],
        std::slice::from_ref(&t2d),
        IndexMap::new(),
    );

    // 2. 7D elementwise exp -> skipped because rank 7 > MAX_FOREACH_RANK (6)
    let (high_op, _) = push_op(
        &mut module,
        blk_id,
        "mo",
        "elementwise.exp",
        &[high_rank_in],
        std::slice::from_ref(&t7d),
        IndexMap::new(),
    );

    // 3. Rowwise reduction -> `map.iter.rowwise`
    let mut red_attrs = IndexMap::new();
    red_attrs.insert("axis".to_string(), Attribute::Int(1));
    let (red_op, _) = push_op(
        &mut module,
        blk_id,
        "mo",
        "reduce.sum",
        &[elem_res[0]],
        std::slice::from_ref(&t1d),
        red_attrs,
    );

    // 4. Opaque kernel with trailing chain -> `map.iter.opaque`
    let mut opq_attrs = IndexMap::new();
    opq_attrs.insert(
        "symbol".to_string(),
        Attribute::String("custom_gemm".to_string()),
    );
    opq_attrs.insert("reads_memory".to_string(), Attribute::Bool(true));
    opq_attrs.insert("writes_memory".to_string(), Attribute::Bool(true));
    let (opq_op, _) = push_op(
        &mut module,
        blk_id,
        "mo",
        "opaque_kernel",
        &[a, b, ch_in],
        &[t2d, chain_ty],
        opq_attrs,
    );

    let report = lower_tensor_ops_to_map_kernels(&mut module, AsyncLaunchDispatch::Deferred);
    assert_eq!(report.elementwise_lowered, 1);
    assert_eq!(report.rowwise_lowered, 1);
    assert_eq!(report.opaque_lowered, 1);
    assert_eq!(report.rank_exceeded_skipped, 1);
    assert_eq!(report.total_lowered(), 3);

    // Verify lowered elementwise op structure.
    assert_eq!(module.ops[elem_op].name.full_name(), "map.iter.elementwise");
    assert_eq!(
        module.ops[elem_op].attributes.get("dispatch"),
        Some(&Attribute::String("deferred".to_string()))
    );
    let elem_reg = module.ops[elem_op].regions[0];
    let elem_blk = module.regions[elem_reg].blocks[0];
    assert_eq!(module.blocks[elem_blk].arguments.len(), 2);
    let inner_names: Vec<String> = module.blocks[elem_blk]
        .ops
        .iter()
        .map(|&id| module.ops[id].name.full_name())
        .collect();
    assert_eq!(
        inner_names,
        vec![
            "map.load",
            "map.load",
            "map.call.elementwise",
            "map.store",
            "map.yield"
        ]
    );

    // Verify 7D op remained unlowered.
    assert_eq!(module.ops[high_op].name.full_name(), "mo.elementwise.exp");

    // Verify rowwise reduction op and combiner monoid.
    assert_eq!(module.ops[red_op].name.full_name(), "map.iter.rowwise");
    assert_eq!(
        module.ops[red_op].attributes.get("reduction_monoid"),
        Some(&Attribute::String("add".to_string()))
    );
    assert_eq!(
        RowwiseReductionMonoid::Max.identity_attr(&DType::F32),
        Attribute::Float(f64::NEG_INFINITY)
    );
    assert_eq!(
        RowwiseReductionMonoid::Min.identity_attr(&DType::I32),
        Attribute::Int(i64::MAX)
    );

    // Verify opaque kernel prologues, call.opaque, and epilogues.
    assert_eq!(module.ops[opq_op].name.full_name(), "map.iter.opaque");
    let opq_reg = module.ops[opq_op].regions[0];
    let opq_blk = module.regions[opq_reg].blocks[0];
    let opq_inner: Vec<String> = module.blocks[opq_blk]
        .ops
        .iter()
        .map(|&id| module.ops[id].name.full_name())
        .collect();
    assert_eq!(
        opq_inner,
        vec![
            "map.iter.prologue",
            "map.iter.prologue",
            "map.call.opaque",
            "map.iter.epilogue",
            "map.yield"
        ]
    );
}

#[test]
fn test_lower_map_kernels_to_routines_and_primitives_to_async_routines() {
    let (mut module, blk_id) = make_test_module_with_block();
    let sym_t1 = sym_tensor_ty(
        DType::F32,
        &[
            Dim::Symbolic("B".to_string()),
            Dim::Symbolic("N".to_string()),
        ],
    );
    let sym_t2 = sym_tensor_ty(
        DType::F32,
        &[
            Dim::Symbolic("B".to_string()),
            Dim::Symbolic("K".to_string()),
        ],
    );
    let x = add_block_arg(&mut module, blk_id, sym_t1.clone());
    let y = add_block_arg(&mut module, blk_id, sym_t2.clone());

    let mut opq_attrs = IndexMap::new();
    opq_attrs.insert(
        "symbol".to_string(),
        Attribute::String("flash_kernel".to_string()),
    );
    let (opq_op, _) = push_op(
        &mut module,
        blk_id,
        "map",
        "iter.opaque",
        &[x, y],
        std::slice::from_ref(&sym_t1),
        opq_attrs,
    );

    // Attach an inner `index.constant` inside `opq_op` before routine lowering to test phase 1.
    let lowered_routines = lower_map_kernels_to_routines(&mut module);
    assert_eq!(lowered_routines, 1);
    assert_eq!(module.ops[opq_op].name.full_name(), "mogg.routine");
    assert_eq!(
        module.ops[opq_op].attributes.get("num_deduplicated_dims"),
        Some(&Attribute::Int(3))
    );

    // Inject an `index.constant` inside the routine's inner block and primitive ops in the outer
    // block.
    let routine_reg = module.ops[opq_op].regions[0];
    let routine_blk = module.regions[routine_reg].blocks[0];
    let mut idx_attrs = IndexMap::new();
    idx_attrs.insert("value".to_string(), Attribute::Int(42));
    let _ = push_op(
        &mut module,
        routine_blk,
        "index",
        "constant",
        &[],
        &[Type::Scalar(DType::I64)],
        idx_attrs,
    );

    // Outer primitives: `index.add` (fusible + duplicable), `mgp.sync` (excluded from fusion,
    // immediate dispatch), and cross-stream `mgp.device.wait` (excluded from fusion).
    let (idx_add_op, _) = push_op(
        &mut module,
        blk_id,
        "index",
        "add",
        &[x, y],
        &[Type::Scalar(DType::I64)],
        IndexMap::new(),
    );
    let (sync_op, _) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "sync",
        &[x],
        &[Type::Scalar(DType::I64)],
        IndexMap::new(),
    );
    let mut wait_attrs = IndexMap::new();
    wait_attrs.insert(
        "src_stream".to_string(),
        Attribute::String("s0".to_string()),
    );
    wait_attrs.insert(
        "dst_stream".to_string(),
        Attribute::String("s1".to_string()),
    );
    let (wait_op, _) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "device.wait",
        &[x],
        &[Type::Scalar(DType::I64)],
        wait_attrs,
    );

    let (converted_consts, wrapped_prims) =
        lower_primitives_to_async_routines(&mut module, AsyncLaunchDispatch::Deferred);
    assert_eq!(converted_consts, 1);
    assert_eq!(wrapped_prims, 3);

    // Check that `index.constant` inside `routine_blk` became `mgp.buffer.constant` +
    // `mgp.buffer.to_index`.
    let routine_op_names: Vec<String> = module.blocks[routine_blk]
        .ops
        .iter()
        .map(|&id| module.ops[id].name.full_name())
        .collect();
    assert!(routine_op_names.contains(&"mgp.buffer.constant".to_string()));
    assert!(routine_op_names.contains(&"mgp.buffer.to_index".to_string()));

    // Verify fusion/duplication flags on wrapped outer primitives.
    assert_eq!(
        module.ops[idx_add_op].attributes.get("can_fuse_into_other"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        module.ops[idx_add_op].attributes.get("can_duplicate"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        module.ops[sync_op].attributes.get("can_fuse_into_other"),
        Some(&Attribute::Bool(false))
    );
    assert_eq!(
        module.ops[sync_op].attributes.get("dispatch"),
        Some(&Attribute::String("immediate".to_string()))
    );
    assert_eq!(
        module.ops[wait_op].attributes.get("can_fuse_into_other"),
        Some(&Attribute::Bool(false))
    );
}

#[test]
fn test_fuse_view_producers_and_residual_chain_duplication_guard() {
    let (mut module, blk_id) = make_test_module_with_block();
    let t2d = tensor_ty(DType::F32, &[4, 8]);
    let src = add_block_arg(&mut module, blk_id, t2d.clone());

    // 1. Pure view producer (`produces_through_view = true`, no compute) feeding two consumers:
    //    one `map.iter.elementwise` and one `map.iter.opaque`.
    let mut pure_view_attrs = IndexMap::new();
    pure_view_attrs.insert("produces_through_view".to_string(), Attribute::Bool(true));
    let (pure_view_op, pure_view_res) = push_op(
        &mut module,
        blk_id,
        "map",
        "iter.elementwise",
        &[src],
        std::slice::from_ref(&t2d),
        pure_view_attrs,
    );
    assert_eq!(
        classify_view_foreach(&module, pure_view_op),
        (true, true, false)
    );

    let (cons_elem_op, _) = push_op(
        &mut module,
        blk_id,
        "map",
        "iter.elementwise",
        &[pure_view_res[0]],
        std::slice::from_ref(&t2d),
        IndexMap::new(),
    );
    let (cons_opq_op, opq_res) = push_op(
        &mut module,
        blk_id,
        "mo",
        "opaque_kernel",
        &[pure_view_res[0]],
        std::slice::from_ref(&t2d),
        IndexMap::new(),
    );
    lower_tensor_ops_to_map_kernels(&mut module, AsyncLaunchDispatch::Immediate);

    // Single-use elementwise consumer of `cons_opq_op` for epilogue fusion.
    let mut epilogue_cons_attrs = IndexMap::new();
    epilogue_cons_attrs.insert(
        "kernel_fn".to_string(),
        Attribute::String("relu".to_string()),
    );
    let _ = push_op(
        &mut module,
        blk_id,
        "map",
        "iter.elementwise",
        &[opq_res[0]],
        std::slice::from_ref(&t2d),
        epilogue_cons_attrs,
    );

    // 2. Non-pure view producer (already fused with expensive `exp` compute) feeding TWO consumers
    //    -> must be blocked by the residual-chain duplication guard!
    let mut residual_attrs = IndexMap::new();
    residual_attrs.insert("produces_through_view".to_string(), Attribute::Bool(true));
    let (res_view_op, res_view_out) = push_op(
        &mut module,
        blk_id,
        "mo",
        "elementwise.exp",
        &[src],
        std::slice::from_ref(&t2d),
        residual_attrs,
    );
    lower_tensor_ops_to_map_kernels(&mut module, AsyncLaunchDispatch::Immediate);
    assert_eq!(
        classify_view_foreach(&module, res_view_op),
        (true, false, false)
    );
    let _ = push_op(
        &mut module,
        blk_id,
        "map",
        "iter.elementwise",
        &[res_view_out[0]],
        std::slice::from_ref(&t2d),
        IndexMap::new(),
    );
    let _ = push_op(
        &mut module,
        blk_id,
        "map",
        "iter.elementwise",
        &[res_view_out[0]],
        std::slice::from_ref(&t2d),
        IndexMap::new(),
    );

    let report = fuse_view_producers(&mut module);
    assert_eq!(report.fused_into_elementwise, 1);
    assert_eq!(report.fused_into_prologue, 1);
    assert_eq!(report.fused_into_epilogue, 1);
    assert_eq!(report.residual_duplication_blocked, 1);
    assert_eq!(report.total_fused(), 3);

    assert_eq!(module.ops[cons_elem_op].operands[0], src);
    assert_eq!(module.ops[cons_opq_op].operands[0], src);
    assert_eq!(
        module.ops[cons_opq_op]
            .attributes
            .get("fused_output_epilogues"),
        Some(&Attribute::Int(1))
    );
    assert!(module.blocks[blk_id].ops.contains(&res_view_op));
}

// ============================================================================
// Batch #30 Tests: Device Graph Finalization, Async Dataflow & RMO Legalization
// ============================================================================

#[test]
fn test_finalize_device_graphs_casts_shape_only_and_collective_lowering() {
    let (mut module, blk_id) = make_test_module_with_block();
    let builder_ty = Type::Opaque("!mgp.device_graph.builder".to_string());
    let ctx_ty = Type::Opaque("!mgp.device_context".to_string());
    let t_ty = tensor_ty(DType::F32, &[8, 8]);
    let chain_ty = Type::Opaque("!grt.chain".to_string());

    let b_arg = add_block_arg(&mut module, blk_id, builder_ty);
    let t_arg = add_block_arg(&mut module, blk_id, t_ty.clone());

    // Audited `builder_to_device_context` bridge cast used by `mgp.buffer.slice`.
    let mut cast_attrs = IndexMap::new();
    cast_attrs.insert(
        "bridge_kind".to_string(),
        Attribute::String("builder_to_device_context".to_string()),
    );
    let (cast_op, cast_res) = push_op(
        &mut module,
        blk_id,
        "builtin",
        "unrealized_conversion_cast",
        &[b_arg],
        &[ctx_ty],
        cast_attrs,
    );
    let _ = push_op(
        &mut module,
        blk_id,
        "mgp",
        "buffer.slice",
        &[cast_res[0]],
        std::slice::from_ref(&t_ty),
        IndexMap::new(),
    );

    // Build an `mgp.device_graph.collective` op whose region has:
    // - arg 0: shape-only tensor (used only by `mgp.tensor.extract.tensor_spec`)
    // - arg 1: pass-through `!grt.chain` yielded directly in `mgp.device_graph.collective.yield`
    let coll_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let r_shape_arg = add_block_arg(&mut module, coll_blk, t_ty.clone());
    let r_chain_arg = add_block_arg(&mut module, coll_blk, chain_ty.clone());
    let (_, spec_res) = push_op(
        &mut module,
        coll_blk,
        "mgp",
        "tensor.extract.tensor_spec",
        &[r_shape_arg],
        &[Type::Scalar(DType::I64)],
        IndexMap::new(),
    );
    let _ = push_op(
        &mut module,
        coll_blk,
        "mgp",
        "device_graph.collective.yield",
        &[spec_res[0], r_chain_arg],
        &[],
        IndexMap::new(),
    );
    let coll_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![coll_blk],
    });
    let (coll_op, _) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "device_graph.collective",
        &[t_arg],
        &[Type::Scalar(DType::I64)],
        IndexMap::new(),
    );
    module.ops[coll_op].regions.push(coll_reg);

    let report = finalize_device_graphs(&mut module).expect("finalization should succeed");
    assert_eq!(report.resolved_builder_casts, 1);
    assert_eq!(report.shape_only_inputs_marked, 1);
    assert_eq!(report.stripped_passthrough_chains, 1);
    assert_eq!(report.collectives_lowered, 1);
    assert_eq!(
        module.ops[cast_op].name.full_name(),
        "mgp.device_graph.device_context"
    );
    assert_eq!(
        module.ops[coll_op].name.full_name(),
        "mgp.device_graph.collective.execute"
    );
}

#[test]
fn test_async_dataflow_analysis_relations_and_dependency_pruning() {
    let (mut module, blk_id) = make_test_module_with_block();
    let chain_ty = Type::Opaque("!grt.chain".to_string());
    let buf_ty = Type::Opaque("!mgp.buffer".to_string());

    let entry_ch = add_block_arg(&mut module, blk_id, chain_ty.clone());
    let base_buf = add_block_arg(&mut module, blk_id, buf_ty.clone());

    let (_, ready_ch) = push_op(
        &mut module,
        blk_id,
        "grt",
        "chain.create",
        &[entry_ch],
        std::slice::from_ref(&chain_ty),
        IndexMap::new(),
    );
    let (_, alias_buf) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "alias",
        &[base_buf],
        std::slice::from_ref(&buf_ty),
        IndexMap::new(),
    );
    let (_, slice_buf) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "buffer.slice",
        &[alias_buf[0]],
        std::slice::from_ref(&buf_ty),
        IndexMap::new(),
    );

    // Async op `k1` producing `ch1`, followed by `k2` consuming `ch1` and producing `ch2`.
    let (k1_op, ch1) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "kernel.launch",
        &[slice_buf[0]],
        std::slice::from_ref(&chain_ty),
        IndexMap::new(),
    );
    let (k2_op, ch2) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "kernel.launch",
        &[ch1[0]],
        std::slice::from_ref(&chain_ty),
        IndexMap::new(),
    );

    // `grt.chain.join(ready_ch, ch1, ch2)`: `ready_ch` is always ready and `ch2` implies `ch1`!
    let (join_op, _) = push_op(
        &mut module,
        blk_id,
        "grt",
        "chain.join",
        &[ready_ch[0], ch1[0], ch2[0]],
        std::slice::from_ref(&chain_ty),
        IndexMap::new(),
    );

    // Mutually exclusive `scf.if` branches.
    let then_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let else_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let (then_op, _) = push_op(
        &mut module,
        then_blk,
        "mgp",
        "branch_work",
        &[base_buf],
        &[],
        IndexMap::new(),
    );
    let (else_op, _) = push_op(
        &mut module,
        else_blk,
        "mgp",
        "branch_work",
        &[base_buf],
        &[],
        IndexMap::new(),
    );
    let then_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![then_blk],
    });
    let else_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![else_blk],
    });
    let (if_op, _) = push_op(
        &mut module,
        blk_id,
        "scf",
        "if",
        &[entry_ch],
        &[],
        IndexMap::new(),
    );
    module.ops[if_op].regions = vec![then_reg, else_reg];

    // `scf.while` loop with a non-self-overlapping loop-carried token.
    let while_blk = module.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let iter_tok = add_block_arg(&mut module, while_blk, chain_ty.clone());
    let (_, next_tok) = push_op(
        &mut module,
        while_blk,
        "mgp",
        "step_async",
        &[iter_tok],
        std::slice::from_ref(&chain_ty),
        IndexMap::new(),
    );
    let _ = push_op(
        &mut module,
        while_blk,
        "scf",
        "yield",
        &[next_tok[0]],
        &[],
        IndexMap::new(),
    );
    let while_reg = module.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![while_blk],
    });
    let (while_op, _) = push_op(
        &mut module,
        blk_id,
        "scf",
        "while",
        &[ch2[0]],
        &[],
        IndexMap::new(),
    );
    module.ops[while_op].regions = vec![while_reg];

    let analysis = AsyncDataflowAnalysis::analyze(&module);
    assert!(analysis.same_as(base_buf, alias_buf[0]));
    assert!(!analysis.same_as(base_buf, slice_buf[0]));
    assert!(analysis.maybe_same_as(base_buf, slice_buf[0]));
    assert!(analysis.is_always_ready(entry_ch));
    assert!(analysis.is_always_ready(ready_ch[0]));
    assert!(!analysis.is_always_ready(ch1[0]));
    assert!(analysis.implies_ready(ch2[0], ch1[0]));
    assert!(analysis.runs_later(k2_op, k1_op));
    assert!(analysis.are_disjoint(k1_op, k2_op));
    assert!(analysis.structurally_cannot_overlap(then_op, else_op));
    assert!(analysis.are_disjoint(then_op, else_op));
    assert!(analysis.is_non_self_overlapping(&module, while_op, 0));

    let pruned = optimize_async_dataflow_dependencies(&mut module);
    assert_eq!(pruned, 2);
    assert_eq!(
        module.ops[join_op].operands.as_slice(),
        std::slice::from_ref(&ch2[0])
    );
}

#[test]
fn test_legalize_rmo_ops_and_optimize_symbolic_dims() {
    let (mut module, blk_id) = make_test_module_with_block();
    let cond = add_block_arg(&mut module, blk_id, Type::Scalar(DType::I1));
    let chain_ty = Type::Opaque("!grt.chain".to_string());

    // 1. Duplicate `rmo.assert` with identical `(cond, message)`.
    let mut assert_attrs = IndexMap::new();
    assert_attrs.insert(
        "message".to_string(),
        Attribute::String("bounds_check".to_string()),
    );
    let (_, a1_res) = push_op(
        &mut module,
        blk_id,
        "rmo",
        "assert",
        &[cond],
        std::slice::from_ref(&chain_ty),
        assert_attrs.clone(),
    );
    let (_, a2_res) = push_op(
        &mut module,
        blk_id,
        "rmo",
        "assert",
        &[cond],
        std::slice::from_ref(&chain_ty),
        assert_attrs,
    );
    let (use_assert_op, _) = push_op(
        &mut module,
        blk_id,
        "mgp",
        "consume_chain",
        &[a2_res[0]],
        &[],
        IndexMap::new(),
    );

    // 2. Compatible `rmo.rebind_shape` filtering `input_param_decls` -> `output_param_decls`.
    let in_sym_ty = sym_tensor_ty(
        DType::F32,
        &[Dim::Symbolic("B".to_string()), Dim::Static(16)],
    );
    let out_sym_ty = sym_tensor_ty(
        DType::F32,
        &[Dim::Symbolic("Batch".to_string()), Dim::Static(16)],
    );
    let sym_in = add_block_arg(&mut module, blk_id, in_sym_ty);
    let mut rebind_attrs = IndexMap::new();
    rebind_attrs.insert(
        "input_param_decls".to_string(),
        Attribute::Array(vec![
            Attribute::String("Batch".to_string()),
            Attribute::String("UnusedSym".to_string()),
        ]),
    );
    let (rebind_op, _) = push_op(
        &mut module,
        blk_id,
        "rmo",
        "rebind_shape",
        &[sym_in],
        &[out_sym_ty],
        rebind_attrs,
    );

    // 3. `rmo.broadcast_elementwise.add` with `F16` + `BF16` -> promoted to `F32` with broadcast.
    let lhs_f16 = add_block_arg(&mut module, blk_id, tensor_ty(DType::F16, &[1, 8]));
    let rhs_bf16 = add_block_arg(&mut module, blk_id, tensor_ty(DType::BF16, &[4, 8]));
    let (bcast_elem_op, bcast_elem_res) = push_op(
        &mut module,
        blk_id,
        "rmo",
        "broadcast_elementwise.add",
        &[lhs_f16, rhs_bf16],
        &[tensor_ty(DType::F16, &[4, 8])],
        IndexMap::new(),
    );

    assert!(is_compatible_shape(
        &[Dim::Symbolic("B".to_string()), Dim::Static(8)],
        &[Dim::Static(4), Dim::Static(8)]
    ));
    assert!(!is_compatible_shape(
        &[Dim::Static(4), Dim::Static(8)],
        &[Dim::Static(4), Dim::Static(16)]
    ));
    assert_eq!(promote_rmo_dtypes(&DType::F16, &DType::BF16), DType::F32);

    let report = legalize_rmo_ops(&mut module).expect("legalize_rmo_ops should succeed");
    assert_eq!(report.deduplicated_asserts, 1);
    assert_eq!(report.legalized_rebinds, 1);
    assert_eq!(report.decomposed_elementwise, 1);
    assert_eq!(module.ops[use_assert_op].operands[0], a1_res[0]);
    assert_eq!(
        module.ops[rebind_op].name.full_name(),
        "rmo.rebind_tensor_shape"
    );
    assert_eq!(
        module.ops[rebind_op].attributes.get("output_param_decls"),
        Some(&Attribute::Array(vec![Attribute::String(
            "Batch".to_string()
        )]))
    );
    assert_eq!(
        module.ops[bcast_elem_op].name.full_name(),
        "mo.elementwise.add"
    );
    assert_eq!(
        module.values[bcast_elem_res[0]].ty,
        tensor_ty(DType::F32, &[4, 8])
    );

    // 4. Symbolic algebraic simplification (`x + 0 -> x`, `x - x -> 0`, `x * 1 -> x`,
    //    `sym.assert_eq(x, y)` replaces `y` with `x`).
    let idx_ty = Type::Scalar(DType::I64);
    let sym_x = add_block_arg(&mut module, blk_id, idx_ty.clone());
    let sym_y = add_block_arg(&mut module, blk_id, idx_ty.clone());
    let mut zero_attrs = IndexMap::new();
    zero_attrs.insert("value".to_string(), Attribute::Int(0));
    let (_, c0) = push_op(
        &mut module,
        blk_id,
        "sym",
        "const",
        &[],
        std::slice::from_ref(&idx_ty),
        zero_attrs,
    );
    let mut one_attrs = IndexMap::new();
    one_attrs.insert("value".to_string(), Attribute::Int(1));
    let (_, c1) = push_op(
        &mut module,
        blk_id,
        "sym",
        "const",
        &[],
        std::slice::from_ref(&idx_ty),
        one_attrs,
    );
    let _ = push_op(
        &mut module,
        blk_id,
        "sym",
        "assert_eq",
        &[sym_x, sym_y],
        &[],
        IndexMap::new(),
    );
    let (_, add_res) = push_op(
        &mut module,
        blk_id,
        "sym",
        "add",
        &[sym_y, c0[0]],
        std::slice::from_ref(&idx_ty),
        IndexMap::new(),
    );
    let (_, mul_res) = push_op(
        &mut module,
        blk_id,
        "sym",
        "mul",
        &[add_res[0], c1[0]],
        std::slice::from_ref(&idx_ty),
        IndexMap::new(),
    );
    let (_, sub_res) = push_op(
        &mut module,
        blk_id,
        "sym",
        "sub",
        &[sym_x, sym_y],
        std::slice::from_ref(&idx_ty),
        IndexMap::new(),
    );
    let (consumer_op, _) = push_op(
        &mut module,
        blk_id,
        "sym",
        "use",
        &[mul_res[0], sub_res[0]],
        &[],
        IndexMap::new(),
    );

    let simplified = optimize_symbolic_dims(&mut module);
    assert_eq!(simplified, 4);
    assert_eq!(module.ops[consumer_op].operands[0], sym_x);
    assert_eq!(module.ops[consumer_op].operands[1], c0[0]);
}

// ============================================================================
// Batch #31 Tests: SharedRegionState, ReplyWindow, KvLoadPolicy & ObjectGroupFrame
// ============================================================================

#[test]
fn test_shared_region_state_incarnations_all_or_nothing_ttl_and_leases() {
    let contract = RegionContract {
        version: REGION_CONTRACT_VERSION,
        epoch: 7,
        capacity_bytes: 1024,
        alignment_bytes: 64,
        write_ttl_ms: 500,
    };
    let mut state = SharedRegionState::new(contract).expect("valid contract");
    assert!(state.validate_contract(REGION_CONTRACT_VERSION, 7).is_ok());
    assert!(state.validate_contract(REGION_CONTRACT_VERSION, 8).is_err());

    // Register client `worker_a` with incarnation 1 and reserve a write + a committed key with a
    // read lease.
    assert_eq!(state.register_client("worker_a", 1, 100), (0, 0));
    let batch1 = state
        .reserve_write_batch(
            "worker_a",
            1,
            &[
                ("k_committed".to_string(), 100),
                ("k_stale".to_string(), 200),
            ],
            100,
        )
        .expect("reserve batch 1");
    let tok_committed = match batch1[0] {
        WriteReserveStatus::Reserved {
            write_token,
            extent,
        } => {
            assert_eq!(
                extent,
                ExtentSpec {
                    offset: 0,
                    length: 100,
                    aligned_length: 128,
                }
            );
            write_token
        }
        other => panic!("unexpected status: {other:?}"),
    };
    let ext_committed = state
        .finish_write("worker_a", 1, tok_committed, 110)
        .expect("finish_write");
    // Idempotent duplicate `finish_write` replays the exact same `ExtentSpec`.
    assert_eq!(
        state
            .finish_write("worker_a", 1, tok_committed, 115)
            .expect("idempotent finish_write"),
        ext_committed
    );
    let _lease = state
        .reserve_read("worker_a", 1, "k_committed", 120)
        .expect("reserve_read");

    // Re-register `worker_a` with incarnation 2: retires the uncommitted `k_stale` write and the
    // active read lease on `k_committed`.
    assert_eq!(state.register_client("worker_a", 2, 130), (1, 1));
    let usage = state.usage(130);
    assert_eq!(usage.live_keys, 1);
    assert_eq!(usage.inflight_writes, 0);
    assert_eq!(usage.active_read_leases, 0);
    assert_eq!(usage.used_bytes, 128);
    assert_eq!(usage.free_bytes, 896);

    // All-or-nothing batch reservation: requesting `[500, 600]` exceeds remaining 896 bytes, so
    // the first 512-byte allocation is rolled back and `free_bytes` remains 896!
    let oos_batch = state
        .reserve_write_batch(
            "worker_a",
            2,
            &[("k_part1".to_string(), 500), ("k_part2".to_string(), 600)],
            140,
        )
        .expect("all-or-nothing batch");
    assert_eq!(
        oos_batch,
        vec![
            WriteReserveStatus::OutOfSpace,
            WriteReserveStatus::OutOfSpace
        ]
    );
    assert_eq!(state.usage(140).free_bytes, 896);

    // Lazy write-TTL expiration: reserve `k_ttl` at t=200ms; at t=700ms (`200 + 500`), it expires
    // and frees its extent.
    let ttl_batch = state
        .reserve_write_batch("worker_a", 2, &[("k_ttl".to_string(), 256)], 200)
        .expect("ttl reserve");
    assert!(matches!(ttl_batch[0], WriteReserveStatus::Reserved { .. }));
    assert_eq!(state.usage(400).inflight_writes, 1);
    assert_eq!(state.usage(700).inflight_writes, 0);
    assert_eq!(state.usage(700).free_bytes, 896);

    // Evict `k_committed` now that its reader count is 0 -> full 1024 bytes free.
    let evicted = state.evict_key("k_committed", 710).expect("evict");
    assert_eq!(evicted.aligned_length, 128);
    assert_eq!(state.usage(710).free_bytes, 1024);
}

#[test]
fn test_orchestrator_reply_window_idempotency_conflict_and_incarnation_reset() {
    let mut window = OrchestratorReplyWindow::new(2);
    assert_eq!(
        window.check_or_replay("client_1", 10, 1, b"req_payload_1"),
        ReplyWindowOutcome::Miss
    );
    window
        .record_reply("client_1", 10, 1, b"req_payload_1", b"reply_1".to_vec())
        .expect("record reply 1");

    // Byte-identical retry replays the cached reply.
    assert_eq!(
        window.check_or_replay("client_1", 10, 1, b"req_payload_1"),
        ReplyWindowOutcome::ReplayCached(b"reply_1".to_vec())
    );
    // Reusing `request_id = 1` with a mutated payload fails with `ConflictAlreadyExists`.
    assert_eq!(
        window.check_or_replay("client_1", 10, 1, b"mutated_payload"),
        ReplyWindowOutcome::ConflictAlreadyExists
    );
    assert!(
        window
            .record_reply("client_1", 10, 1, b"mutated_payload", b"bad".to_vec())
            .is_err()
    );

    // FIFO eviction when exceeding `window_size = 2`.
    window
        .record_reply("client_1", 10, 2, b"req_2", b"reply_2".to_vec())
        .expect("record 2");
    window
        .record_reply("client_1", 10, 3, b"req_3", b"reply_3".to_vec())
        .expect("record 3");
    assert_eq!(
        window.check_or_replay("client_1", 10, 1, b"req_payload_1"),
        ReplyWindowOutcome::Miss
    );
    assert_eq!(
        window.check_or_replay("client_1", 10, 3, b"req_3"),
        ReplyWindowOutcome::ReplayCached(b"reply_3".to_vec())
    );

    // Advancing incarnation clears prior entries.
    assert_eq!(
        window.check_or_replay("client_1", 11, 3, b"req_3"),
        ReplyWindowOutcome::Miss
    );
}

#[test]
fn test_kv_load_policy_and_object_group_frame_roundtrip() {
    // 1. `skip_load` override
    let d1 = evaluate_kv_load_policy(&KvLoadContext {
        request_id: "r1".to_string(),
        num_lookup_tokens: 1024,
        chunk_size: 256,
        skip_load: true,
        engine_computed_tokens: Some(0),
        min_novel_tokens_threshold: 256,
    });
    assert!(!d1.should_load);
    assert_eq!(d1.reason, "skip_load_override");

    // 2. Engine already computed 896 of 1024 tokens -> only 128 novel tokens (< 256 chunk_size)
    let d2 = evaluate_kv_load_policy(&KvLoadContext {
        request_id: "r2".to_string(),
        num_lookup_tokens: 1024,
        chunk_size: 256,
        skip_load: false,
        engine_computed_tokens: Some(896),
        min_novel_tokens_threshold: 256,
    });
    assert!(!d2.should_load);
    assert_eq!(d2.effective_novel_tokens, 128);
    assert_eq!(d2.reason, "recompute_cheaper_than_load");

    // 3. Novel tokens (768) >= threshold (512) -> load approved
    let d3 = evaluate_kv_load_policy(&KvLoadContext {
        request_id: "r3".to_string(),
        num_lookup_tokens: 1024,
        chunk_size: 256,
        skip_load: false,
        engine_computed_tokens: Some(256),
        min_novel_tokens_threshold: 512,
    });
    assert!(d3.should_load);
    assert_eq!(d3.effective_novel_tokens, 768);
    assert_eq!(d3.reason, "load_approved");

    // 4. `ObjectGroupFrame` multi-field encode/decode roundtrip & validation
    let make_obj = |seed: u8| {
        vec![
            ObjectFieldSpec {
                name: "k".to_string(),
                dtype: DType::F16,
                shape: vec![2, 4],
                data: vec![seed; 16],
            },
            ObjectFieldSpec {
                name: "scales".to_string(),
                dtype: DType::F32,
                shape: vec![2],
                data: vec![seed.wrapping_add(1); 8],
            },
        ]
    };
    let frame = ObjectGroupFrame {
        group_id: "layer_group_0".to_string(),
        objects: vec![make_obj(0xAA), make_obj(0xBB)],
    };
    let encoded = frame.encode().expect("encode should succeed");
    let decoded = ObjectGroupFrame::decode(&encoded).expect("decode should succeed");
    assert_eq!(decoded, frame);

    // Schema mismatch between object 0 and object 1 is rejected.
    let bad_frame = ObjectGroupFrame {
        group_id: "bad".to_string(),
        objects: vec![
            make_obj(1),
            vec![ObjectFieldSpec {
                name: "k".to_string(),
                dtype: DType::F16,
                shape: vec![2, 4],
                data: vec![0; 16],
            }],
        ],
    };
    assert!(bad_frame.encode().is_err());
}
