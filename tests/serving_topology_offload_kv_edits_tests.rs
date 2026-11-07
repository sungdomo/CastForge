#![cfg(feature = "serving")]

use std::collections::{BTreeMap, HashMap, HashSet};

use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

use castforge::ir::mlir::{Attribute, DType, Dim, Location, Module, OpName, TensorType, Type};
use castforge::passes::serving::{
    DrainSignals, EvictionAwarePolicyConfig, EvictionAwareStoreQueue, KvCacheGroupSpec,
    KvCacheSpecKind, LazyOffloadRequestRegistry, LazyStoreSliceOp, OVERDUE_RANK,
    PagedKvSequenceView, ParametricTopologyAnalysis, StridedTensorMeta, apply_mamba_page_view_edit,
    apply_mamba_slice_plane_edit, apply_mamba_unified_view_edit,
    apply_subpaged_attention_view_edit, coalesce_contiguous_store_ops,
    compute_compressed_mamba_slots, declares_slot_compression, globalize_mgp_call_slot_indices,
    raise_back_single_caller_index_constants, rooted_transitive_user_components,
    topologically_sort_block_with_parameters, validate_kv_cache_group_specs,
    verify_device_graph_kernel_registrations,
};

#[test]
fn test_parametric_topology_slot_globalization_and_device_graph_verification() {
    // 1. Parametric topology + Kahn topological sort + rooted transitive components
    let mut topo_mod = Module::new(Some("topo".to_string()));
    let blk = topo_mod.body_block();
    let t_ty = Type::Tensor(TensorType {
        shape: vec![Dim::Static(8)],
        dtype: DType::F32,
        encoding: None,
    });

    // Insert consumer before producer via symbolic parameter "SYM_K"
    let mut use_attrs = IndexMap::new();
    use_attrs.insert(
        "param_uses".to_string(),
        Attribute::Array(vec![Attribute::String("SYM_K".to_string())]),
    );
    let op_consumer = topo_mod.add_operation(
        blk,
        OpName::new("mo", "reshape"),
        SmallVec::new(),
        vec![(None, t_ty.clone())],
        use_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );

    let mut decl_attrs = IndexMap::new();
    decl_attrs.insert(
        "param_decls".to_string(),
        Attribute::Array(vec![Attribute::String("SYM_K".to_string())]),
    );
    let op_producer = topo_mod.add_operation(
        blk,
        OpName::new("mo", "rebind"),
        SmallVec::new(),
        vec![(None, t_ty)],
        decl_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );

    let analysis = ParametricTopologyAnalysis::analyze(&topo_mod);
    assert_eq!(analysis.users(op_producer), &[op_consumer]);
    assert_eq!(analysis.providers(op_consumer), &[op_producer]);

    let moved = topologically_sort_block_with_parameters(&mut topo_mod, blk).expect("sort");
    assert_eq!(moved, 2);
    assert_eq!(topo_mod.blocks[blk].ops, vec![op_producer, op_consumer]);
    assert_eq!(
        rooted_transitive_user_components(&topo_mod, op_producer),
        vec![op_producer, op_consumer]
    );

    // 2. Bottom-up call-graph slot globalization (`globalize_mgp_call_slot_indices`)
    let mut call_mod = Module::new(Some("globalize_slots".to_string()));
    let root = call_mod.body_block();

    // Subgraph model `@sub` with `num_slots = 3`
    let sub_init = call_mod.create_block(Some("init".to_string()));
    let mut sub_ctx_attrs = IndexMap::new();
    sub_ctx_attrs.insert("num_slots".to_string(), Attribute::Int(3));
    call_mod.add_operation(
        sub_init,
        OpName::new("mgp.context", "create"),
        SmallVec::new(),
        vec![(None, Type::Opaque("!mgp.context".to_string()))],
        sub_ctx_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );
    let sub_exec = call_mod.create_block(Some("execute".to_string()));
    let sub_init_reg = call_mod.create_region(vec![sub_init]);
    let sub_exec_reg = call_mod.create_region(vec![sub_exec]);
    let mut sub_attrs = IndexMap::new();
    sub_attrs.insert("sym_name".to_string(), Attribute::String("sub".to_string()));
    call_mod.add_operation(
        root,
        OpName::new("mgp", "model"),
        SmallVec::new(),
        Vec::new(),
        sub_attrs,
        vec![sub_init_reg, sub_exec_reg],
        false,
        Location::Unknown,
    );

    // Main model `@main` with `num_slots = 5` and two `mgp.call @sub` invocations
    let main_init = call_mod.create_block(Some("init".to_string()));
    let mut main_ctx_attrs = IndexMap::new();
    main_ctx_attrs.insert("num_slots".to_string(), Attribute::Int(5));
    call_mod.add_operation(
        main_init,
        OpName::new("mgp.context", "create"),
        SmallVec::new(),
        vec![(None, Type::Opaque("!mgp.context".to_string()))],
        main_ctx_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );
    let main_exec = call_mod.create_block(Some("execute".to_string()));
    let mut call1_attrs = IndexMap::new();
    call1_attrs.insert("callee".to_string(), Attribute::String("sub".to_string()));
    let call1 = call_mod.add_operation(
        main_exec,
        OpName::new("mgp", "call"),
        SmallVec::new(),
        Vec::new(),
        call1_attrs.clone(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let call2 = call_mod.add_operation(
        main_exec,
        OpName::new("mgp", "call"),
        SmallVec::new(),
        Vec::new(),
        call1_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );
    let main_init_reg = call_mod.create_region(vec![main_init]);
    let main_exec_reg = call_mod.create_region(vec![main_exec]);
    let mut main_attrs = IndexMap::new();
    main_attrs.insert(
        "sym_name".to_string(),
        Attribute::String("main".to_string()),
    );
    call_mod.add_operation(
        root,
        OpName::new("mgp", "model"),
        SmallVec::new(),
        Vec::new(),
        main_attrs,
        vec![main_init_reg, main_exec_reg],
        false,
        Location::Unknown,
    );

    let globalized = globalize_mgp_call_slot_indices(&mut call_mod).expect("globalize slots");
    assert_eq!(globalized, 2);
    assert_eq!(
        call_mod.ops[call1].attributes.get("is_globalized_index"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        call_mod.ops[call2].attributes.get("is_globalized_index"),
        Some(&Attribute::Bool(true))
    );
    // First call gets offset 5, second call gets offset 5 + 3 = 8.
    let exec_ops = &call_mod.blocks[main_exec].ops;
    assert_eq!(exec_ops.len(), 4);
    assert_eq!(
        call_mod.ops[exec_ops[0]].attributes.get("value"),
        Some(&Attribute::Int(5))
    );
    assert_eq!(
        call_mod.ops[exec_ops[2]].attributes.get("value"),
        Some(&Attribute::Int(8))
    );

    // 3. Device-graph verification & single-caller index-constant raising
    let mut dg_mod = Module::new(Some("dg_verify".to_string()));
    let dg_root = dg_mod.body_block();
    let mut kdecl_attrs = IndexMap::new();
    kdecl_attrs.insert(
        "sym_name".to_string(),
        Attribute::String("add_kernel".to_string()),
    );
    kdecl_attrs.insert(
        "registration".to_string(),
        Attribute::String("mo.add".to_string()),
    );
    kdecl_attrs.insert("build_graph".to_string(), Attribute::Bool(true));
    dg_mod.add_operation(
        dg_root,
        OpName::new("mogg.kernel", "decl"),
        SmallVec::new(),
        Vec::new(),
        kdecl_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );
    let g_blk = dg_mod.create_block(Some("body".to_string()));
    dg_mod.add_operation(
        g_blk,
        OpName::new("mo", "add"),
        SmallVec::new(),
        Vec::new(),
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let g_reg = dg_mod.create_region(vec![g_blk]);
    let mut g_attrs = IndexMap::new();
    g_attrs.insert("is_device_graph".to_string(), Attribute::Bool(true));
    dg_mod.add_operation(
        dg_root,
        OpName::new("mo", "graph"),
        SmallVec::new(),
        Vec::new(),
        g_attrs,
        vec![g_reg],
        false,
        Location::Unknown,
    );
    assert_eq!(verify_device_graph_kernel_registrations(&dg_mod), Ok(1));

    // Raise back single-caller index constant
    let stub_blk = dg_mod.create_block(Some("entry".to_string()));
    let mut rc_attrs = IndexMap::new();
    rc_attrs.insert("resource_slot".to_string(), Attribute::Int(0));
    let rc_op = dg_mod.add_operation(
        stub_blk,
        OpName::new("mgp.resource", "constant"),
        SmallVec::new(),
        vec![(None, Type::Opaque("!mgp.buffer".to_string()))],
        rc_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );
    let rc_val = dg_mod.ops[rc_op].results[0];
    let b2i_op = dg_mod.add_operation(
        stub_blk,
        OpName::new("mgp.buffer", "to_index"),
        smallvec![rc_val],
        vec![(None, Type::Scalar(DType::Index))],
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let stub_reg = dg_mod.create_region(vec![stub_blk]);
    let mut stub_attrs = IndexMap::new();
    stub_attrs.insert(
        "sym_name".to_string(),
        Attribute::String("k_single".to_string()),
    );
    stub_attrs.insert("abi".to_string(), Attribute::String("mgp".to_string()));
    dg_mod.add_operation(
        dg_root,
        OpName::new("mogg", "func"),
        SmallVec::new(),
        Vec::new(),
        stub_attrs,
        vec![stub_reg],
        false,
        Location::Unknown,
    );

    let mut cc_attrs = IndexMap::new();
    cc_attrs.insert(
        "callee_symbol".to_string(),
        Attribute::String("k_single".to_string()),
    );
    cc_attrs.insert(
        "constant_resources".to_string(),
        Attribute::Array(vec![Attribute::DenseIntArray(vec![77])]),
    );
    dg_mod.add_operation(
        dg_root,
        OpName::new("mogg", "compiled_call"),
        SmallVec::new(),
        Vec::new(),
        cc_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );

    assert_eq!(raise_back_single_caller_index_constants(&mut dg_mod), 1);
    assert_eq!(dg_mod.ops[b2i_op].name.full_name(), "index.constant");
    assert_eq!(
        dg_mod.ops[b2i_op].attributes.get("value"),
        Some(&Attribute::Int(77))
    );
}

#[test]
fn test_lazy_offload_request_registry_store_coalescing_and_eviction_queue() {
    // 1. Request generation registry & orphaned batch handling
    let mut reg = LazyOffloadRequestRegistry::new();
    reg.arrive("req_a");
    reg.register_batch("req_a", vec![10, 11]).expect("batch 1");
    assert!(reg.has_in_flight("req_a"));
    assert!(!reg.in_flight_is_orphaned("req_a"));

    // Preemption reset orphans the in-flight batch while keeping phase Active.
    reg.reset("req_a");
    assert!(reg.in_flight_is_orphaned("req_a"));
    assert!(!reg.is_finished("req_a"));

    let b = reg.complete_batch("req_a").expect("complete batch");
    assert!(b.orphaned);
    assert_eq!(b.block_ids, vec![10, 11]);
    reg.finish("req_a");
    assert!(reg.can_end_session("req_a"));
    reg.session_ended("req_a");

    // 2. Contiguous store metadata coalescing
    let s1 = LazyStoreSliceOp {
        request_id: "req_c".to_string(),
        start: 0,
        end: 16,
        block_ids: vec![vec![1], vec![101]],
        token_bytes: vec![1, 2],
        cache_salt: "salt".to_string(),
    };
    let s2 = LazyStoreSliceOp {
        request_id: "req_c".to_string(),
        start: 16,
        end: 32,
        block_ids: vec![vec![2], vec![102]],
        token_bytes: vec![1, 2, 3, 4],
        cache_salt: "salt".to_string(),
    };
    let merged = coalesce_contiguous_store_ops(&[s1.clone(), s2]).expect("coalesce");
    assert_eq!(merged.start, 0);
    assert_eq!(merged.end, 32);
    assert_eq!(merged.block_ids, vec![vec![1, 2], vec![101, 102]]);
    assert_eq!(merged.token_bytes, vec![1, 2, 3, 4]);

    // Non-contiguous gap rejected
    let s_gap = LazyStoreSliceOp {
        start: 20,
        end: 36,
        ..s1.clone()
    };
    assert!(coalesce_contiguous_store_ops(&[s1, s_gap]).is_err());

    // 3. Eviction-aware store queue: snapshot mismatch drops suffix & overdue rank
    assert_eq!(OVERDUE_RANK, 1usize << 62);
    let cfg = EvictionAwarePolicyConfig::new(2.0, 2, 5.0).expect("cfg");
    let mut queue = EvictionAwareStoreQueue::new(cfg);

    // Unhashed block rejected and breaks prefix chain
    let op_unhashed = LazyStoreSliceOp {
        request_id: "req_bad".to_string(),
        start: 0,
        end: 16,
        block_ids: vec![vec![99]],
        token_bytes: vec![],
        cache_salt: String::new(),
    };
    queue.add(op_unhashed.clone(), &BTreeMap::from([(99, None)]), 1.0);
    queue.add(op_unhashed, &BTreeMap::from([(99, Some(1))]), 2.0);
    assert_eq!(queue.counters().rejected_unhashed, 1);
    assert_eq!(queue.counters().rejected_prefix_broken, 1);

    // Add an op whose block gets recycled before drain, and an overdue op
    let op_evicted = LazyStoreSliceOp {
        request_id: "req_ev".to_string(),
        start: 0,
        end: 16,
        block_ids: vec![vec![50]],
        token_bytes: vec![],
        cache_salt: String::new(),
    };
    let op_overdue = LazyStoreSliceOp {
        request_id: "req_due".to_string(),
        start: 0,
        end: 16,
        block_ids: vec![vec![60]],
        token_bytes: vec![],
        cache_salt: String::new(),
    };
    queue.add(op_evicted, &BTreeMap::from([(50, Some(500))]), 1.0);
    queue.add(op_overdue, &BTreeMap::from([(60, Some(600))]), 1.0);

    let drain_res = queue.drain(&DrainSignals {
        now_secs: 10.0,
        new_blocks_allocated: 0,
        est_next_step_blocks: 0,
        blocked_request_ids: HashSet::new(),
        free_queue_order: vec![],
        // Block 50 hash changed from 500 -> 999 (recycled); block 60 intact (600)
        live_block_hashes: HashMap::from([(50, Some(999)), (60, Some(600))]),
    });
    assert_eq!(queue.counters().dropped_evicted, 1);
    assert_eq!(queue.counters().emitted, 1);
    assert_eq!(queue.counters().emitted_overdue, 1);
    assert_eq!(drain_res.items.len(), 1);
    assert_eq!(drain_res.items[0].request_id, "req_due");
    assert!(queue.ledger_line().ends_with("pending=0"));
}

#[test]
fn test_kv_group_edits_mamba_subpaged_attention_and_token_drop_compaction() {
    let valid_mamba = KvCacheGroupSpec {
        kind: KvCacheSpecKind::Mamba,
        block_size: 16,
        page_size_bytes: 1024,
        mamba_cache_mode: "align".to_string(),
        tokens_per_state: 1,
        compress_ratio: 1,
        tq_slot_size: 0,
    };
    let valid_attn = KvCacheGroupSpec {
        kind: KvCacheSpecKind::FullAttention,
        block_size: 544,
        page_size_bytes: 17 * 2 * 32 * 4 * 8 * 2, // 17 kernel pages of [2, 32, 4, 8] fp16
        mamba_cache_mode: String::new(),
        tokens_per_state: 1,
        compress_ratio: 1,
        tq_slot_size: 0,
    };
    assert!(!declares_slot_compression(&valid_attn));
    assert!(valid_attn.kind.is_subpageable_attention());
    assert!(validate_kv_cache_group_specs(&[valid_mamba.clone(), valid_attn.clone()]).is_ok());

    let bad_cross = KvCacheGroupSpec {
        kind: KvCacheSpecKind::CrossAttention,
        ..valid_mamba.clone()
    };
    assert!(validate_kv_cache_group_specs(&[bad_cross]).is_err());

    // 1. Mamba page view edit (`stride[0] * elem_bytes == page_size_bytes`)
    let conv_padded = StridedTensorMeta {
        shape: vec![8, 128],
        strides: vec![512, 1],
        element_size: 2,
        storage_offset: 0,
    };
    let page_view =
        apply_mamba_page_view_edit(&valid_mamba, &[conv_padded]).expect("mamba page view");
    assert_eq!(page_view.shape, vec![8, 2, 16, 1, 16]);

    // 2. Mamba slice plane edit (`uint8` planes `[num_blocks, block_size, 1, width]`)
    let conv_slice = StridedTensorMeta {
        shape: vec![8, 64],
        strides: vec![64, 1],
        element_size: 2,
        storage_offset: 0,
    };
    let planes =
        apply_mamba_slice_plane_edit(&valid_mamba, &[conv_slice]).expect("mamba slice plane");
    assert_eq!(planes[0].shape, vec![8, 16, 1, 8]);
    assert_eq!(planes[0].element_size, 1);

    // 3. Sub-paged attention view edit (kernel_block_size=32, logical_block_size=544, ratio=17)
    let kernel_paged = StridedTensorMeta {
        shape: vec![34, 2, 32, 4, 8],
        strides: vec![2 * 32 * 4 * 8, 32 * 4 * 8, 4 * 8, 8, 1],
        element_size: 2,
        storage_offset: 0,
    };
    let logical_view =
        apply_subpaged_attention_view_edit(&valid_attn, &kernel_paged).expect("subpaged edit");
    assert_eq!(logical_view.shape, vec![2, 2, 544, 1, 32]);

    // 4. Mamba unified view edit (including compressed slots fallback)
    let (slots, h_size) = compute_compressed_mamba_slots(795 * 8, 16, 2, 795 * 8).expect("slots");
    assert_eq!(slots * h_size, 795 * 8);
    let unified_tensor = StridedTensorMeta {
        shape: vec![4, 1, 1, 256],
        strides: vec![512, 256, 256, 1],
        element_size: 2,
        storage_offset: 0,
    };
    let nhd_view =
        apply_mamba_unified_view_edit(&valid_mamba, &unified_tensor, "NHD").expect("unified NHD");
    assert_eq!(nhd_view.shape, vec![4, 16, 1, 16]);

    // 5. In-place token-drop compaction
    let cache = vec![1.0f32; 2 * 2 * 4 * 2 * 2];
    let mut seq_view =
        PagedKvSequenceView::new(2, 4, 2, 2, cache, vec![0, 1, 2, 3, 4, 5]).expect("seq view");
    assert_eq!(
        seq_view.compact_kv_in_place(&[vec![1, 2], vec![3, 4], vec![5, 0]]),
        Ok(3)
    );
}

#[test]
fn test_optimize_serving_pass_registry_pipeline() {
    use castforge::passes::{PassManager, PassRegistry};

    let reg = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &reg,
        "optimize-serving(mode=topology),optimize-serving(mode=all)",
    )
    .expect("build optimize-serving pipeline");

    let mut module = Module::new(Some("serving_pipeline_test".to_string()));
    let blk = module.body_block();
    let t_ty = Type::Tensor(TensorType {
        shape: vec![Dim::Static(4)],
        dtype: DType::F32,
        encoding: None,
    });

    let mut use_attrs = IndexMap::new();
    use_attrs.insert(
        "param_uses".to_string(),
        Attribute::Array(vec![Attribute::String("SEQ_LEN".to_string())]),
    );
    let op_use = module.add_operation(
        blk,
        OpName::new("mo", "broadcast_to"),
        SmallVec::new(),
        vec![(None, t_ty.clone())],
        use_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );

    let mut decl_attrs = IndexMap::new();
    decl_attrs.insert(
        "param_decls".to_string(),
        Attribute::Array(vec![Attribute::String("SEQ_LEN".to_string())]),
    );
    let op_decl = module.add_operation(
        blk,
        OpName::new("mo", "rebind"),
        SmallVec::new(),
        vec![(None, t_ty)],
        decl_attrs,
        Vec::new(),
        false,
        Location::Unknown,
    );

    let results = pm.run(&mut module).expect("run optimize-serving pipeline");
    assert!(results.iter().any(|r| r.changed));
    assert_eq!(module.blocks[blk].ops, vec![op_decl, op_use]);
}
