//! Integration tests for device-graph staging, layout propagation, collective partitioning,
//! async lifetime folding, and buffer dependency analyses.

use castforge::ir::parse_mlir;

#[test]
fn test_infer_and_propagate_tensor_layouts_and_conflict_detection() {
    use castforge::passes::infer_and_propagate_tensor_layouts;

    let mlir_ok = concat!(
        "module @infer_layouts {\n",
        "  func.func @main(%x: tensor<1x8x8x3xf32>, %b: tensor<1x8x8x3xf32>,",
        " %w: tensor<16x3x3x3xf32>, %mw: tensor<3x16xf32>) -> tensor<1x8x8x16xf32> {\n",
        "    %x_add = mo.add %x, %b",
        " : (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>\n",
        "    %w_tr = mo.transfer %w : (tensor<16x3x3x3xf32>) -> tensor<16x3x3x3xf32>\n",
        "    %y = mo.conv %x_add, %w_tr",
        " : (tensor<1x8x8x3xf32>, tensor<16x3x3x3xf32>) -> tensor<1x8x8x16xf32>\n",
        "    %mm = mo.matmul %x, %mw",
        " : (tensor<1x8x8x3xf32>, tensor<3x16xf32>) -> tensor<1x8x8x16xf32>\n",
        "    return %y : tensor<1x8x8x16xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m_ok = parse_mlir(mlir_ok).expect("valid MLIR");
    let layouts = infer_and_propagate_tensor_layouts(&mut m_ok).expect("consistent layouts");
    let func_id = m_ok.functions()[0];
    let blk_id = m_ok.regions[m_ok.ops[func_id].regions[0]].blocks[0];
    let args = &m_ok.blocks[blk_id].arguments;
    assert_eq!(layouts.get(&args[0]).map(String::as_str), Some("NHWC"));
    assert_eq!(layouts.get(&args[1]).map(String::as_str), Some("NHWC"));
    assert_eq!(layouts.get(&args[2]).map(String::as_str), Some("FCRS"));
    assert_eq!(layouts.get(&args[3]).map(String::as_str), Some("KN"));

    // Conflict: same 4D tensor used as both conv filter ("FCRS") and conv_transpose ("RSFC"):
    let mlir_conflict = concat!(
        "module @layout_conflict {\n",
        "  func.func @main(%x: tensor<1x8x8x3xf32>, %w: tensor<3x3x3x3xf32>)",
        " -> tensor<1x8x8x3xf32> {\n",
        "    %y0 = mo.conv %x, %w",
        " : (tensor<1x8x8x3xf32>, tensor<3x3x3x3xf32>) -> tensor<1x8x8x3xf32>\n",
        "    %y1 = mo.conv_transpose %y0, %w",
        " : (tensor<1x8x8x3xf32>, tensor<3x3x3x3xf32>) -> tensor<1x8x8x3xf32>\n",
        "    return %y1 : tensor<1x8x8x3xf32>\n",
        "  }\n",
        "}\n",
    );
    let mut m_conflict = parse_mlir(mlir_conflict).expect("valid MLIR");
    let err = infer_and_propagate_tensor_layouts(&mut m_conflict).unwrap_err();
    assert!(err.contains("already constrained 'FCRS'"));
}

#[test]
fn test_globalize_call_graph_slot_indices_bottom_up() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::globalize_call_graph_slot_indices;

    // Call hierarchy:
    //   @entry (local slots = 4)
    //     -> calls @mid (local slots = 3, calls @leaf with local slots = 5 => total @mid = 8)
    //        at offset 4
    //     -> calls @leaf (total @leaf = 5) at offset 4 + 8 = 12
    //   => total @entry = 17
    let mlir = concat!(
        "module @slot_globalize {\n",
        "  func.func @entry(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %ctx0 = mgp.context.create {num_slots = 4 : i64} : () -> tensor<1xi64>\n",
        "    %a = mgp.call %x {callee = \"@mid\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %b = mgp.call %a {callee = \"@leaf\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %b : tensor<4xf32>\n",
        "  }\n",
        "  func.func @mid(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %ctx1 = mgp.context.create {num_slots = 3 : i64} : () -> tensor<1xi64>\n",
        "    %y = mgp.call %x {callee = \"@leaf\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %y : tensor<4xf32>\n",
        "  }\n",
        "  func.func @leaf(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %ctx2 = mgp.context.create {num_slots = 5 : i64} : () -> tensor<1xi64>\n",
        "    return %x : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let totals = globalize_call_graph_slot_indices(&mut m).expect("valid call graph");
    assert_eq!(totals.get("leaf"), Some(&5));
    assert_eq!(totals.get("mid"), Some(&8));
    assert_eq!(totals.get("entry"), Some(&17));

    // Check @entry's two `mgp.call` ops received `slot_offset = 4` and `slot_offset = 12`:
    let entry_func = m.functions()[0];
    let entry_blk = m.regions[m.ops[entry_func].regions[0]].blocks[0];
    let call_offsets: Vec<i64> = m.blocks[entry_blk]
        .ops
        .iter()
        .filter(|&&op_id| m.ops[op_id].name.full_name() == "mgp.call")
        .filter_map(|&op_id| match m.ops[op_id].attributes.get("slot_offset") {
            Some(Attribute::Int(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert_eq!(call_offsets, vec![4, 12]);
}

#[test]
fn test_store_policy_planner_and_binary_trace_stream_codec() {
    use castforge::passes::{
        L2AdapterSpec, StorePolicyKind, TRACE_FORMAT_VERSION, TRACE_MAGIC, TRACE_SCHEMA_VERSION,
        TraceHeader, TraceRecord, decode_trace_stream, encode_trace_stream,
        plan_l1_l2_store_policy,
    };

    let adapters = vec![
        L2AdapterSpec {
            index: 0,
            name: "nvme".to_string(),
            writable: true,
        },
        L2AdapterSpec {
            index: 1,
            name: "readonly-archive".to_string(),
            writable: false,
        },
    ];
    let keys = vec!["k0".to_string(), "k1".to_string()];
    let default_plan =
        plan_l1_l2_store_policy(StorePolicyKind::DefaultKeepL1, &keys, &adapters, &keys);
    assert_eq!(default_plan.targets_by_adapter.len(), 1);
    assert_eq!(default_plan.targets_by_adapter[&0], keys);
    assert!(default_plan.l1_deletions.is_empty());

    let skip_l1_plan =
        plan_l1_l2_store_policy(StorePolicyKind::BufferOnlySkipL1, &keys, &adapters, &keys);
    assert_eq!(skip_l1_plan.l1_deletions, keys);

    // Binary trace stream roundtrip & partial tail handling:
    let hdr = TraceHeader {
        magic: TRACE_MAGIC,
        format_version: TRACE_FORMAT_VERSION,
        level: "storage".to_string(),
        trace_schema_version: TRACE_SCHEMA_VERSION,
        t_mono_start_ns: 1_000,
        t_wall_start_ns: 2_000,
        sm_config_digest: "deadbeef".to_string(),
    };
    let recs = vec![
        TraceRecord {
            t_mono_ns: 100,
            t_wall_ns: 2100,
            qualname: "StorageManager.reserve_write".to_string(),
            args_json: "{\"keys\":2}".to_string(),
        },
        TraceRecord {
            t_mono_ns: 250,
            t_wall_ns: 2250,
            qualname: "StorageManager.commit_write".to_string(),
            args_json: "{\"ok\":true}".to_string(),
        },
    ];
    let mut encoded = encode_trace_stream(&hdr, &recs);
    let (dec_hdr, dec_recs, truncated) = decode_trace_stream(&encoded).expect("valid trace");
    assert_eq!(dec_hdr, hdr);
    assert_eq!(dec_recs, recs);
    assert!(!truncated);

    // Simulate SIGKILL truncating a 3rd trailing frame mid-body:
    encoded.extend_from_slice(&100u32.to_be_bytes());
    encoded.extend_from_slice(&[1, 2, 3]);
    let (_, dec_recs2, truncated2) = decode_trace_stream(&encoded).expect("graceful tail");
    assert_eq!(dec_recs2, recs);
    assert!(truncated2);
}

#[test]
fn test_hoist_loop_invariant_index_views_preserves_store_transforms() {
    use castforge::passes::hoist_loop_invariant_index_views;

    // `%idx_r` (`map.index.reshape`) feeds BOTH `map.store` (at operand 1) and `%idx_b`
    // (`map.index.broadcast`), which feeds `%idx_s` (`map.index.slice`) -> `map.load`.
    // Only `[%idx_s, %idx_b]` is hoistable; `%idx_r` feeds the store index and must stay!
    let mlir = concat!(
        "module @hoist_index_views {\n",
        "  func.func @main(%in_t: tensor<4xf32>, %out_t: tensor<4xf32>, %idx0: tensor<1xi64>)",
        " -> tensor<4xf32> {\n",
        "    %idx_r = map.index.reshape %idx0 : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    %idx_b = map.index.broadcast %idx_r : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    %idx_s = map.index.slice %idx_b : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    %val = map.load %in_t, %idx_s : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    %stored = map.store %out_t, %idx_r, %val",
        " : (tensor<4xf32>, tensor<1xi64>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %stored : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let hoisted = hoist_loop_invariant_index_views(&mut m);
    assert_eq!(hoisted, 2);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk_id]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();

    assert!(op_names.contains(&"map.index.slice.stride".to_string()));
    assert!(op_names.contains(&"map.index.broadcast.stride".to_string()));
    assert!(!op_names.contains(&"map.index.slice".to_string()));
    assert!(!op_names.contains(&"map.index.broadcast".to_string()));
    // Store-feeding `map.index.reshape` remains in place:
    assert!(op_names.contains(&"map.index.reshape".to_string()));
}

#[test]
fn test_raise_back_contiguous_views_and_slice_dim_matcher() {
    use castforge::passes::{
        find_contiguous_slice_dim, match_contiguous_slice_dim, raise_back_contiguous_views,
    };

    // 1. Contiguous slice dimension matcher:
    assert_eq!(find_contiguous_slice_dim(&[1, 8, 16]), 1);
    assert_eq!(
        match_contiguous_slice_dim(
            &[1, 8, 16],
            &[1, 4, 16],
            &[0, 2, 0],
            &[1, 6, 16],
            &[1, 1, 1]
        ),
        Some(1)
    );
    // Non-contiguous slice (slicing dim 2 when dim 1 is not 1, or step != 1) returns None:
    assert_eq!(
        match_contiguous_slice_dim(&[1, 8, 16], &[1, 8, 8], &[0, 0, 0], &[1, 8, 8], &[1, 1, 1]),
        None
    );
    assert_eq!(
        match_contiguous_slice_dim(
            &[1, 8, 16],
            &[1, 4, 16],
            &[0, 0, 0],
            &[1, 8, 16],
            &[1, 2, 1]
        ),
        None
    );

    // 2. Raising `map.iter.elementwise` with body `slice -> reshape -> load` raises in reverse
    // (`mo.static.reshape` then `mogg.slice`):
    let mlir = concat!(
        "module @raise_views {\n",
        "  func.func @main(%x: tensor<1x8x16xf32>) -> tensor<1x4x16xf32> {\n",
        "    %y = map.iter.elementwise %x {view_chain = [\"slice\", \"reshape\"],",
        " in_dims = array<i64: 1, 8, 16>, out_dims = array<i64: 1, 4, 16>,",
        " starts = array<i64: 0, 2, 0>, stops = array<i64: 1, 6, 16>,",
        " steps = array<i64: 1, 1, 1>} : (tensor<1x8x16xf32>) -> tensor<1x4x16xf32>\n",
        "    return %y : tensor<1x4x16xf32>\n",
        "  }\n",
        "}\n",
    );
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let (slices, reshapes) = raise_back_contiguous_views(&mut m);
    assert_eq!((slices, reshapes), (1, 1));

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk_id]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec!["mo.static.reshape", "mogg.slice", "func.return"]
    );
}

#[test]
fn test_allocate_accumulating_epilogue_buffers_mixed_and_same_precision() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::{
        allocate_accumulating_epilogue_buffers, is_accumulating_opaque_kernel_name,
    };

    assert!(is_accumulating_opaque_kernel_name("mo.matmul.f32"));
    assert!(is_accumulating_opaque_kernel_name("mo.conv"));
    assert!(!is_accumulating_opaque_kernel_name("mo.softmax"));

    // Kernel 0: `mo.matmul` accumulating in `f32` with an epilogue casting to `f16` (mixed
    // precision -> allocates a scratch `tensor<4x8xf32>` result before the trailing chain!).
    // Kernel 1: `mo.matmul` in `f32` with an `f32` epilogue (same precision -> binds directly).
    let mlir = concat!(
        "module @accum_buffers {\n",
        "  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x8xf32>, %c0: tensor<1xi64>)",
        " -> (tensor<4x8xf16>, tensor<4x8xf32>, tensor<1xi64>) {\n",
        "    %mm0, %ch1 = map.iter.opaque %a, %b, %c0 {kernel_name = \"mo.matmul\"}",
        " : (tensor<4x8xf32>, tensor<8x8xf32>, tensor<1xi64>)",
        " -> (tensor<4x8xf32>, tensor<1xi64>)\n",
        "    %epi0 = map.iter.epilogue %mm0 : (tensor<4x8xf32>) -> tensor<4x8xf16>\n",
        "    %mm1 = map.iter.opaque %a, %b {kernel_name = \"mo.matmul\"}",
        " : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>\n",
        "    %epi1 = map.iter.epilogue %mm1 : (tensor<4x8xf32>) -> tensor<4x8xf32>\n",
        "    return %epi0, %epi1, %ch1 : tensor<4x8xf16>, tensor<4x8xf32>, tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let (scratch, direct) = allocate_accumulating_epilogue_buffers(&mut m);
    assert_eq!((scratch, direct), (1, 1));

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    // `mm0` now has 3 results: `[%mm0, %scratch_f32, %ch1]`:
    assert_eq!(m.ops[ops[0]].results.len(), 3);
    assert_eq!(
        m.ops[ops[1]].attributes.get("uses_scratch_accumulator"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        m.ops[ops[3]].attributes.get("uses_scratch_accumulator"),
        Some(&Attribute::Bool(false))
    );
}

#[test]
fn test_compute_device_graph_dependencies_and_stable_place_ordering() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::compute_device_graph_dependencies;

    // Routine 0 (`%r0`) -> `mgp.device_graph.stable.place` (`%addr`) ->
    // `mgp.device_graph.stable.tensor` (`%st`) -> Routine 1 (`%r1`).
    // Routine 2 consumes both `%r0` and `%r1`.
    let mlir = concat!(
        "module @device_graph_deps {\n",
        "  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %r0 = mogg.routine.async %x {constant_resources = [\"w0\"]}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %addr = mgp.device_graph.stable.place %r0 : (tensor<4xf32>) -> tensor<1xi64>\n",
        "    %st = mgp.device_graph.stable.tensor %addr : (tensor<1xi64>) -> tensor<4xf32>\n",
        "    %r1 = mogg.routine.async %st {constant_resources = [\"w1\", \"w0\"]}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r2 = mogg.routine.async %r0, %r1",
        " : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r2 : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let converted = compute_device_graph_dependencies(&mut m);
    assert_eq!(converted, 3);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let deps_ops: Vec<_> = m.blocks[blk_id]
        .ops
        .iter()
        .copied()
        .filter(|&id| m.ops[id].name.full_name() == "mgp.device_graph.dependencies")
        .collect();
    assert_eq!(deps_ops.len(), 3);

    assert_eq!(
        m.ops[deps_ops[0]].attributes.get("dep_node_indices"),
        Some(&Attribute::DenseIntArray(vec![]))
    );
    // Routine 1 depends on Routine 0 through `stable.tensor` -> `stable.place`:
    assert_eq!(
        m.ops[deps_ops[1]].attributes.get("dep_node_indices"),
        Some(&Attribute::DenseIntArray(vec![0]))
    );
    // Routine 2 depends on both Routine 0 and Routine 1:
    assert_eq!(
        m.ops[deps_ops[2]].attributes.get("dep_node_indices"),
        Some(&Attribute::DenseIntArray(vec![0, 1]))
    );
    // Enclosing function aggregates deduplicated `"constant_resources"`:
    assert_eq!(
        m.ops[func_id].attributes.get("constant_resources"),
        Some(&Attribute::Array(vec![
            Attribute::String("w0".to_string()),
            Attribute::String("w1".to_string()),
        ]))
    );
}

#[test]
fn test_optimal_bloom_filter_and_sliding_window_block_downsampler() {
    use castforge::passes::{
        OptimalBloomFilter, batched_ranges_with_skip, downsample_sliding_window_block_ids,
        recalculate_sliding_window_blocks_to_skip,
    };

    // 1. Sub-chunk sliding-window block downsampler (e.g. 2 chunks of 4 blocks each, SWA keeps 2):
    let swa_blocks = vec![11u64, 12, 13, 14, 15, 16, 17, 18];
    let downsampled = downsample_sliding_window_block_ids(&swa_blocks, 4, 2).unwrap();
    assert_eq!(downsampled, vec![13, 14, 17, 18]);

    // 2. Sliding-window block skip recalculation:
    assert_eq!(recalculate_sliding_window_blocks_to_skip(4, 4, 5), 5);
    // 1 full chunk (2 window blocks) + 3 tail blocks where first 2 are outside window -> 2 + 1 = 3:
    assert_eq!(recalculate_sliding_window_blocks_to_skip(4, 2, 7), 3);
    // Batched iteration with skip:
    assert_eq!(
        batched_ranges_with_skip(10, 3, 2).unwrap(),
        vec![(2, 3), (5, 3), (8, 2)]
    );

    // 3. Optimal Bloom filter:
    let mut bf = OptimalBloomFilter::new(1000, 0.01);
    assert!(bf.size_bits() > 9000);
    assert!(bf.hash_count() >= 6);
    bf.add(b"chunk-alpha");
    bf.add(b"chunk-beta");
    assert!(bf.contains(b"chunk-alpha"));
    assert!(bf.contains(b"chunk-beta"));
    assert!(!bf.contains(b"chunk-never-inserted"));
    let p_alpha = bf.hash_positions(b"chunk-alpha");
    let p_gamma = bf.hash_positions(b"chunk-gamma");
    let added = bf.add_batch_with_positions_and_check(&[p_alpha, p_gamma]);
    assert_eq!(added, 1);
    assert_eq!(bf.item_count(), 3);
    assert!(bf.fill_rate() > 0.0);
}

#[test]
fn test_simplify_device_graph_dependencies_transitive_reduction() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::{
        compute_device_graph_dependencies, simplify_device_graph_dependencies,
    };

    // Routine 0 -> Routine 1 -> Routine 2, while Routine 2 also directly consumes Routine 0's
    // output (creating a redundant edge 0 -> 2 implied by 0 -> 1 -> 2).
    let mlir = concat!(
        "module @simplify_dev_graph {\n",
        "  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %r0 = mogg.routine.async %x : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = mogg.routine.async %r0 : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r2 = mogg.routine.async %r0, %r1",
        " : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r2 : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(compute_device_graph_dependencies(&mut m), 3);
    let eliminated = simplify_device_graph_dependencies(&mut m);
    assert_eq!(eliminated, 1);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let deps_ops: Vec<_> = m.blocks[blk_id]
        .ops
        .iter()
        .copied()
        .filter(|&id| m.ops[id].name.full_name() == "mgp.device_graph.dependencies")
        .collect();
    // Node 2's redundant edge to Node 0 was pruned, leaving only `[1]`:
    assert_eq!(
        m.ops[deps_ops[2]].attributes.get("dep_node_indices"),
        Some(&Attribute::DenseIntArray(vec![1]))
    );
}

#[test]
fn test_simplify_extend_lifetimes_absorbs_into_async_routine() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::simplify_extend_lifetimes;

    let mlir = concat!(
        "module @simplify_extend {\n",
        "  func.func @main(%x: tensor<4xf32>, %buf0: tensor<4xf32>, %buf1: tensor<4xf32>)",
        " -> tensor<1xi64> {\n",
        "    %gate = mogg.routine.async %x {in_out_signature = [\"in\"]}",
        " : (tensor<4xf32>) -> tensor<1xi64>\n",
        "    %ch = mgp.extend_lifetime %gate, %buf0, %buf1",
        " : (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    return %ch : tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(simplify_extend_lifetimes(&mut m), 1);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    assert_eq!(ops.len(), 2); // `mogg.routine.async` + `func.return`
    let routine_id = ops[0];
    assert_eq!(m.ops[routine_id].operands.len(), 3);
    assert_eq!(
        m.ops[routine_id].attributes.get("in_out_signature"),
        Some(&Attribute::Array(vec![
            Attribute::String("in".to_string()),
            Attribute::String("none".to_string()),
            Attribute::String("none".to_string()),
        ]))
    );
}

#[test]
fn test_defer_buffer_allocations_to_mutators_and_join_chains() {
    use castforge::passes::defer_buffer_allocations_to_mutators;

    let mlir = concat!(
        "module @defer_alloc {\n",
        "  func.func @main(%ch0: tensor<1xi64>, %x: tensor<4xf32>) -> tensor<1xi64> {\n",
        "    %buf = mgp.buffer.alloc %ch0 : (tensor<1xi64>) -> tensor<4xf32>\n",
        "    %ch1 = mgp.async_compute %ch0, %x : (tensor<1xi64>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %ch2 = mo.mutable.store %ch1, %buf, %x",
        " : (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    return %ch2 : tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(defer_buffer_allocations_to_mutators(&mut m), 1);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk_id]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mgp.async_compute",
            "mo.chain.create",
            "mgp.buffer.alloc",
            "mo.mutable.store",
            "func.return"
        ]
    );
}

#[test]
fn test_analyze_exec_invariant_code_motion_fixed_point() {
    use castforge::passes::analyze_exec_invariant_code_motion;

    let mlir = concat!(
        "module @eicm {\n",
        "  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %w = mgp.buffer.constant : () -> tensor<4xf32>\n",
        "    %view = mgp.buffer.view %w {recomputable = true, can_be_conveyed = false}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %triv = index.constant : () -> tensor<1xi64>\n",
        "    %alloc = mgp.buffer.alloc %view : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %mut = mo.mutable.store %alloc, %x, %triv",
        " : (tensor<4xf32>, tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    return %mut : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let m = parse_mlir(mlir).expect("valid MLIR");
    let part = analyze_exec_invariant_code_motion(&m);

    let liftable_names: Vec<String> = part
        .liftable_ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    let unliftable_names: Vec<String> = part
        .unliftable_ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    let clone_names: Vec<String> = part
        .clone_ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();

    // `%w` and `%view` are liftable; `%alloc` is mutated by unliftable `%mut` so `SuppressMutated`
    // pulls `%alloc` into `unliftable_ops`, which in turn causes `NeedToClone` to mark `%view`
    // (non-conveyable recomputable) as needing a clone in the execute region. `%triv` is only
    // used by `%mut` so `SuppressTrivial` marks `%triv` unliftable.
    assert_eq!(
        liftable_names,
        vec!["mgp.buffer.constant", "mgp.buffer.view"]
    );
    assert_eq!(
        unliftable_names,
        vec![
            "index.constant",
            "mgp.buffer.alloc",
            "mo.mutable.store",
            "func.return"
        ]
    );
    assert_eq!(clone_names, vec!["mgp.buffer.view"]);
}

#[test]
fn test_multi_group_fold_unfold_and_grouped_presence_bitmaps() {
    use castforge::passes::{
        FULL_ATTENTION_GROUP_WINDOW, all_grouped_retain_masks, fold_multi_group_presence,
        fold_unfold_grouped_presence, highest_servable_prefix_len, merge_presence_bitmaps,
        unfold_group_chunk_range, unfold_multi_group_retain_mask,
    };

    assert_eq!(
        unfold_group_chunk_range(4, FULL_ATTENTION_GROUP_WINDOW),
        (0, 4)
    );
    assert_eq!(unfold_group_chunk_range(4, 2), (2, 4));

    let found = vec![
        true, false, // chunk 0: g0=true, g1=false
        true, false, // chunk 1: g0=true, g1=false
        true, true, // chunk 2: g0=true, g1=true
        false, true, // chunk 3: g0=false, g1=true
    ];
    let windows = [FULL_ATTENTION_GROUP_WINDOW, 1];
    let servable = fold_multi_group_presence(&found, 4, 1, &windows).expect("valid fold");
    assert_eq!(servable, vec![true, false, false, true, false]);
    let hit_len = highest_servable_prefix_len(&servable);
    assert_eq!(hit_len, 3);

    let retain = unfold_multi_group_retain_mask(hit_len, 4, 1, &windows).expect("valid unfold");
    assert_eq!(
        retain,
        vec![
            true, false, // chunk 0
            true, false, // chunk 1
            true, true, // chunk 2
            false, false, // chunk 3
        ]
    );

    let rows = vec![
        vec![true, true, true, false],
        vec![false, false, true, true],
    ];
    let (grouped_hit, grouped_retain) =
        fold_unfold_grouped_presence(&rows, &windows).expect("valid grouped fold_unfold");
    assert_eq!(grouped_hit, 3);
    assert_eq!(
        grouped_retain,
        vec![
            vec![true, true, true, false],
            vec![false, false, true, false],
        ]
    );

    let all_ret = all_grouped_retain_masks(&rows).expect("valid all_grouped");
    assert_eq!(all_ret[0], vec![false, false, true, false]);

    let merged = merge_presence_bitmaps(&rows, 4);
    assert_eq!(merged, vec![true, true, true, true]);
}

#[test]
fn test_lower_buffer_concat_in_place_with_alignment_check() {
    use castforge::passes::{is_provable_multiple_of, lower_buffer_concat_in_place};

    let mlir = concat!(
        "module @concat_in_place {\n",
        "  func.func @main(%ch0: tensor<1xi64>, %dyn: tensor<1xi64>) -> tensor<1xi64> {\n",
        "    %stride = index.constant {value = 256 : i64} : () -> tensor<1xi64>\n",
        "    %sz0 = index.mul %dyn, %stride : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>\n",
        "    %sz1 = index.constant {value = 100 : i64} : () -> tensor<1xi64>\n",
        "    %tot = index.add %sz0, %sz1 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>\n",
        "    %in0 = mgp.buffer.alloc %sz0 : (tensor<1xi64>) -> tensor<4xf32>\n",
        "    %in1 = mgp.buffer.alloc %sz1 : (tensor<1xi64>) -> tensor<4xf32>\n",
        "    %out = mgp.buffer.alloc %tot : (tensor<1xi64>) -> tensor<8xf32>\n",
        "    %ch1 = mgp.buffer.concat %ch0, %out, %in0, %in1",
        " : (tensor<1xi64>, tensor<8xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    return %ch1 : tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let sz0_val = m.ops[m.blocks[blk_id].ops[1]].results[0];
    let sz1_val = m.ops[m.blocks[blk_id].ops[2]].results[0];
    assert!(is_provable_multiple_of(&m, sz0_val, 256));
    assert!(!is_provable_multiple_of(&m, sz1_val, 256));

    // Because `sz0` (the only non-final slice size) is a provable multiple of 256, the concat is
    // lowered in-place into two `mgp.buffer.slice` ops and the `out` alloc is hoisted before them.
    assert_eq!(lower_buffer_concat_in_place(&mut m, 256), 1);

    let slice_count = m.blocks[blk_id]
        .ops
        .iter()
        .filter(|&&id| m.ops[id].name.full_name() == "mgp.buffer.slice")
        .count();
    let alloc_count = m.blocks[blk_id]
        .ops
        .iter()
        .filter(|&&id| m.ops[id].name.full_name() == "mgp.buffer.alloc")
        .count();
    assert_eq!(slice_count, 2);
    assert_eq!(alloc_count, 1);
}

#[test]
fn test_fixup_hanging_chain_overhangs_awaits_unused_chains() {
    use castforge::passes::fixup_hanging_chain_overhangs;

    let mlir = concat!(
        "module @overhangs {\n",
        "  func.func @main(%ch0: tensor<1xi64>, %x: tensor<4xf32>) -> tensor<1xi64> {\n",
        "    %ch_main = mgp.async_compute %ch0, %x",
        " : (tensor<1xi64>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %ch_hang1 = mgp.side_effect %ch0, %x",
        " : (tensor<1xi64>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %ch_hang2 = mgp.telemetry %ch0 : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %ch_main : tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(fixup_hanging_chain_overhangs(&mut m), 2);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    assert_eq!(ops.len(), 5);
    let join_op = ops[3];
    assert_eq!(m.ops[join_op].name.full_name(), "mo.chain.create");
    assert_eq!(m.ops[join_op].operands.len(), 3);
}

#[test]
fn test_outline_structured_kernel_bodies_into_opaque_closure() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::outline_structured_kernel_bodies;

    let mlir = concat!(
        "module @outline_structured {\n",
        "  func.func @main(%ch0: tensor<1xi64>, %x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %out = map.iter.elementwise %ch0, %x {has_read_effect = true}",
        " : (tensor<1xi64>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %out : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(outline_structured_kernel_bodies(&mut m), 1);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let opaque_op = m.blocks[blk_id].ops[0];
    assert_eq!(m.ops[opaque_op].name.full_name(), "map.iter.opaque");
    assert_eq!(
        m.ops[opaque_op].attributes.get("device_ref"),
        Some(&Attribute::String("cpu:0".to_string()))
    );
    assert_eq!(
        m.ops[opaque_op].attributes.get("has_read_effect"),
        Some(&Attribute::Bool(true))
    );

    // Outer opaque block captures the 1 non-chain input (`%x`) + 1 non-chain output (`%out`).
    let opaque_blk = m.regions[m.ops[opaque_op].regions[0]].blocks[0];
    assert_eq!(m.blocks[opaque_blk].arguments.len(), 2);
    let closure_op = m.blocks[opaque_blk].ops[0];
    assert_eq!(m.ops[closure_op].name.full_name(), "map.closure");
    assert_eq!(m.ops[closure_op].operands.len(), 2);

    // Inner closure block has its trailing `%idx` (`tensor<1xi64>`) rotated to argument 0.
    let closure_blk = m.regions[m.ops[closure_op].regions[0]].blocks[0];
    assert_eq!(m.blocks[closure_blk].arguments.len(), 3);
}

#[test]
fn test_parametric_topology_sort_and_rooted_components() {
    use castforge::passes::{
        extract_rooted_parametric_components, topologically_sort_parametric_block,
    };

    // Graph region where `%user` appears before `%decl` in textual order, but `%user` references
    // parameter `"tile_k"` declared by `%decl`, and `%comb` consumes both `%user` and `%aux`.
    let mlir = concat!(
        "module @param_topo {\n",
        "  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %user = mo.param.consumer %x {use_params = [\"tile_k\"]}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %decl = mo.param.producer %x {decl_param = \"tile_k\"}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %aux = mo.aux %x : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %comb = mo.combine %user, %aux",
        " : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %comb : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];

    let moved = topologically_sort_parametric_block(&mut m, blk_id).expect("acyclic");
    assert!(moved >= 2);

    let op_names: Vec<String> = m.blocks[blk_id]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    // `mo.param.producer` (`decl_param = "tile_k"`) is scheduled before `mo.param.consumer`:
    let prod_pos = op_names
        .iter()
        .position(|n| n == "mo.param.producer")
        .unwrap();
    let cons_pos = op_names
        .iter()
        .position(|n| n == "mo.param.consumer")
        .unwrap();
    assert!(prod_pos < cons_pos);
    assert_eq!(op_names.last().unwrap(), "func.return");

    // Rooted component starting from `mo.param.producer` pulls in `mo.param.consumer`,
    // `mo.combine`, `func.return`, and `mo.combine`'s provider `mo.aux`, all topologically sorted:
    let prod_op = m.blocks[blk_id].ops[prod_pos];
    let comp = extract_rooted_parametric_components(&m, blk_id, prod_op).expect("acyclic");
    assert_eq!(comp.len(), 5);
}

#[test]
fn test_strided_kv_contiguous_view_recovery_and_sub_byte_serde_layout() {
    use castforge::passes::{
        SubByteKvSerdePreset, SubByteKvSerdeSpec, recover_strided_kv_contiguous_view,
    };

    // 1. Logical shape `[2, 4, 16, 8, 64]` (`[2, NB, BS, NH, HS]`) physically laid out as HND
    // `[2, 4, 8, 16, 64]` (`[2, NB, NH, BS, HS]`), so strides are `[32768, 8192, 64, 1024, 1]`:
    let shape = [2, 4, 16, 8, 64];
    let strides = [32768, 8192, 64, 1024, 1];
    let view = recover_strided_kv_contiguous_view(&shape, &strides).expect("valid HND recovery");
    assert_eq!(view.perm, vec![0, 1, 3, 2, 4]);
    assert_eq!(view.canonical_shape, vec![2, 4, 8, 16, 64]);
    assert_eq!(view.padding_per_block_elems, 0);

    // 2. Dim-0-padded pool view (`stride[0]` inflated by 512 elements per block):
    let padded_shape = [4, 8, 64];
    let padded_strides = [1024, 64, 1]; // tight inner = 8 * 64 = 512; padding = 512
    let padded_view =
        recover_strided_kv_contiguous_view(&padded_shape, &padded_strides).expect("dim0 padded");
    assert_eq!(padded_view.padding_per_block_elems, 512);

    // 3. Sub-byte KV serde layout with 4-bit norm-corrected preset:
    let spec = SubByteKvSerdeSpec {
        preset: SubByteKvSerdePreset::K4V4NormCorr,
        head_dim: 128,
        block_size: 16,
        skip_first_layers: 2,
        skip_last_layers: 2,
    };
    assert_eq!(spec.key_packed_size(), 66); // 128 * 4 / 8 + 2 = 66
    assert_eq!(spec.value_packed_size(), 68); // 128 * 4 / 8 + 4 = 68
    assert_eq!(spec.slot_size_aligned(), 134);
    assert_eq!(spec.quantized_layer_range(10), (2, 8));
    // Shape [2, 10, 32, 256] (10 layers, 32 tokens = 2 blocks of 16, 2 heads of 128):
    // 4 raw fp16 layers = 2 * 4 * 32 * 256 * 2 = 131072 bytes
    // 6 quant layers = 6 * 2 * 16 * 2 * 134 = 51456 bytes
    let total = spec
        .serialized_nbytes(&[2, 10, 32, 256], 2)
        .expect("valid layout");
    assert_eq!(total, 131_072 + 51_456);
}

#[test]
fn test_verify_model_argument_shapes_emits_spec_checks_and_assert() {
    use castforge::ir::mlir::{Attribute, Dim};
    use castforge::passes::{format_expected_tensor_shape, verify_model_argument_shapes};

    assert_eq!(
        format_expected_tensor_shape(&[Dim::Dynamic, Dim::Static(16), Dim::Static(64)]),
        "any,16,64"
    );

    let mlir = concat!(
        "module @verify_args {\n",
        "  func.func @main(%ch0: tensor<1xi64>, %x: tensor<?x16x64xf32>, %dyn: tensor<?x?xf32>)",
        " -> tensor<1xi64> {\n",
        "    return %ch0 : tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    // `%ch0` is a chain, `%dyn` has all dynamic dimensions, so only `%x` (argument 0) is verified:
    assert_eq!(verify_model_argument_shapes(&mut m), 1);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk_id]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mgp.tensor.extract.tensor_spec",
            "index.constant",
            "mgp.tensor_spec.get_dim",
            "index.cmp",
            "index.constant",
            "mgp.tensor_spec.get_dim",
            "index.cmp",
            "pop.and",
            "mgp.assert",
            "func.return",
        ]
    );
    let assert_op = m.blocks[blk_id].ops[8];
    assert_eq!(
        m.ops[assert_op].attributes.get("message"),
        Some(&Attribute::String(
            "Tensor type mismatch for argument 0: Expected shape: [any,16,64]".to_string()
        ))
    );
}

#[test]
fn test_stage_device_graph_regions_lowers_guards_and_isolates_body() {
    use castforge::passes::stage_device_graph_regions;

    let mlir = concat!(
        "module @stage_dg {\n",
        "  func.func @main(%ch0: tensor<1xi64>, %x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %r0 = mgp.routine.async %x : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %g = mgp.guard %ch0, %r0 : (tensor<1xi64>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %st = mgp.device_graph.stable.create %g : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = mgp.routine.async %st : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r1 : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    // 4 ops staged: `%r0`, lowered `%g` (`mgp.guard` -> `mgp.routine.async`), `%st`, and `%r1`:
    assert_eq!(stage_device_graph_regions(&mut m), 4);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let outer_ops = &m.blocks[blk_id].ops;
    assert_eq!(outer_ops.len(), 2); // `mgp.device_graph.staging` + `func.return`
    let staging_op = outer_ops[0];
    assert_eq!(
        m.ops[staging_op].name.full_name(),
        "mgp.device_graph.staging"
    );
    assert_eq!(m.ops[staging_op].operands.len(), 2); // free inputs: `%x` and `%ch0`
    assert_eq!(m.ops[staging_op].results.len(), 1); // escaping output: `%r1`

    let inner_blk = m.regions[m.ops[staging_op].regions[0]].blocks[0];
    let inner_names: Vec<String> = m.blocks[inner_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        inner_names,
        vec![
            "mgp.routine.async",
            "mgp.routine.async",
            "mgp.device_graph.stable.create",
            "mgp.routine.async",
            "mgp.device_graph.staging.yield",
        ]
    );
}

#[test]
fn test_partition_device_graph_collective_across_multiple_accelerators() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::{partition_device_graph_collective, stage_device_graph_regions};

    let mlir = concat!(
        "module @partition_collective {\n",
        "  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %r0 = mgp.routine.async %x {device_ref = \"gpu:0\"}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %st = mgp.device_graph.stable.create %r0 {device_ref = \"gpu:0\"}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = mgp.routine.async %st {device_ref = \"gpu:1\"}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r1 : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );

    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(stage_device_graph_regions(&mut m), 3);
    assert_eq!(
        partition_device_graph_collective(&mut m).expect("valid partition"),
        1
    );

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let coll_op = m.blocks[blk_id].ops[0];
    assert_eq!(
        m.ops[coll_op].name.full_name(),
        "mgp.device_graph.collective"
    );
    assert_eq!(m.ops[coll_op].regions.len(), 2);
    assert_eq!(
        m.ops[coll_op].attributes.get("num_stable_slots"),
        Some(&Attribute::Int(1))
    );

    // Region 0 (`gpu:0`) publishes the stable slot via `stable.place`:
    let reg0_blk = m.regions[m.ops[coll_op].regions[0]].blocks[0];
    let reg0_names: Vec<String> = m.blocks[reg0_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        reg0_names,
        vec![
            "mgp.routine.async",
            "mgp.device_graph.stable.place",
            "mgp.tensor.extract.tensor_spec",
            "mgp.device_graph.stable.tensor",
            "mgp.device_graph.collective.yield",
        ]
    );

    // Region 1 (`gpu:1`) reads the stable slot via `stable.get` before running its routine:
    let reg1_blk = m.regions[m.ops[coll_op].regions[1]].blocks[0];
    let reg1_names: Vec<String> = m.blocks[reg1_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        reg1_names,
        vec![
            "mgp.device_graph.stable.get",
            "mgp.tensor_spec.create",
            "mgp.device_graph.stable.tensor",
            "mgp.routine.async",
            "mgp.device_graph.collective.yield",
        ]
    );
}

#[test]
fn test_reconcile_and_resolve_device_graph_builder_casts_and_shape_only_check() {
    use castforge::passes::{
        is_shape_only_in_region, reconcile_and_resolve_device_graph_builder_casts,
    };

    let ir = concat!(
        "module @finalize_graph_mod {\n",
        "  func.func @finalize_graph(%builder: tensor<1xi64>, %buf: tensor<8xf32>)",
        " -> tensor<4xf32> {\n",
        "    %dev_ctx_tmp = builtin.unrealized_conversion_cast %builder",
        " : (tensor<1xi64>) -> tensor<2xi64>\n",
        "    %builder_roundtrip = builtin.unrealized_conversion_cast %dev_ctx_tmp",
        " : (tensor<2xi64>) -> tensor<1xi64>\n",
        "    %bridge_ctx = builtin.unrealized_conversion_cast %builder_roundtrip",
        " {bridge_kind = \"builder_to_device_context\"}",
        " : (tensor<1xi64>) -> tensor<2xi64>\n",
        "    %sliced = mgp.buffer.slice %buf, %bridge_ctx",
        " : (tensor<8xf32>, tensor<2xi64>) -> tensor<4xf32>\n",
        "    %spec = mgp.tensor.extract.tensor_spec %sliced",
        " : (tensor<4xf32>) -> tensor<1xi64>\n",
        "    return %sliced : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(ir).expect("valid MLIR");
    let func_op = m.functions()[0];
    let func_reg = m.ops[func_op].regions[0];
    let func_blk = m.regions[func_reg].blocks[0];

    // Check `is_shape_only_in_region`: `%buf` is used by `mgp.buffer.slice` (not shape-only),
    // while `%sliced` is used by `extract.tensor_spec` and `func.return` (not shape-only).
    let buf_val = m.blocks[func_blk].arguments[1];
    assert!(!is_shape_only_in_region(&m, func_reg, buf_val));

    let resolved =
        reconcile_and_resolve_device_graph_builder_casts(&mut m).expect("cast resolution succeeds");
    assert_eq!(resolved, 3);

    let op_names: Vec<String> = m.blocks[func_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mgp.device_graph.device_context",
            "mgp.buffer.slice",
            "mgp.tensor.extract.tensor_spec",
            "func.return",
        ]
    );

    // Verify that an unaudited user of a builder_to_device_context cast returns an error:
    let bad_ir = concat!(
        "module @unaudited_mod {\n",
        "  func.func @unaudited(%builder: tensor<1xi64>, %x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %bridge = builtin.unrealized_conversion_cast %builder",
        " {bridge_kind = \"builder_to_device_context\"}",
        " : (tensor<1xi64>) -> tensor<2xi64>\n",
        "    %bad = mgp.routine.async %x, %bridge",
        " : (tensor<4xf32>, tensor<2xi64>) -> tensor<4xf32>\n",
        "    return %bad : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut bad_m = parse_mlir(bad_ir).expect("valid MLIR");
    let err = reconcile_and_resolve_device_graph_builder_casts(&mut bad_m)
        .expect_err("unaudited op should fail");
    assert!(err.contains("unaudited op 'mgp.routine.async'"));
}

#[test]
fn test_raw_block_layout_fdp_slot_allocator_and_crc32_checkpoint_recovery() {
    use castforge::passes::{
        MAX_FIXED_BUFFER_REGION_BYTES, RawBlockLayoutConfig, RawBlockMetaHeader,
        RawBlockSlotAllocator, build_recovery_item_ranges, compute_crc32_ieee,
        partition_fixed_buffer_registration_ranges, select_latest_valid_checkpoint,
    };

    let layout = RawBlockLayoutConfig::new(
        "/dev/nvme0n1",
        0,
        1024 * 1024,
        4096,
        4096,
        65536,
        65536,
        *b"CFGIDX01",
        1,
    )
    .expect("valid layout");
    assert_eq!(layout.meta_container_bytes, 32768);
    assert_eq!(layout.metadata_container_offsets(), [0, 32768]);
    assert_eq!(layout.data_base_offset, 65536);
    assert_eq!(layout.max_slots, 15);
    assert!(layout.payload_fits_slot(60000, true));
    assert!(!layout.payload_fits_slot(61441, true));

    // Fixed-buffer registration chunking & parallel recovery work ranges:
    let reg_ranges = partition_fixed_buffer_registration_ranges(
        0x1000_0000,
        2 * MAX_FIXED_BUFFER_REGION_BYTES + 4096,
    )
    .expect("valid fixed buffer range");
    assert_eq!(
        reg_ranges,
        vec![
            (0x1000_0000, MAX_FIXED_BUFFER_REGION_BYTES),
            (
                0x1000_0000 + MAX_FIXED_BUFFER_REGION_BYTES,
                MAX_FIXED_BUFFER_REGION_BYTES
            ),
            (0x1000_0000 + 2 * MAX_FIXED_BUFFER_REGION_BYTES, 4096),
        ]
    );
    assert_eq!(
        build_recovery_item_ranges(10, 3),
        vec![(0, 4), (4, 8), (8, 10)]
    );

    // Dual-container CRC32 checkpoint selection:
    let payload_v1 = br#"{"version":1,"next_slot":2}"#;
    let hdr_v1 = RawBlockMetaHeader {
        magic: *b"CFGIDX01",
        version: 1,
        seq: 4,
        payload_len: payload_v1.len() as u64,
        crc32: compute_crc32_ieee(payload_v1),
    }
    .encode();

    let payload_v2 = br#"{"version":1,"next_slot":3}"#;
    let hdr_v2 = RawBlockMetaHeader {
        magic: *b"CFGIDX01",
        version: 1,
        seq: 5,
        payload_len: payload_v2.len() as u64,
        crc32: compute_crc32_ieee(payload_v2),
    }
    .encode();

    let (chosen_hdr, chosen_payload) = select_latest_valid_checkpoint(
        &[(&hdr_v1, payload_v1), (&hdr_v2, payload_v2)],
        b"CFGIDX01",
        1,
        layout.meta_payload_capacity(),
    )
    .expect("valid checkpoint");
    assert_eq!(chosen_hdr.seq, 5);
    assert_eq!(chosen_payload, payload_v2);

    // If seq 5 is corrupted, fallback to seq 4:
    let mut corrupted_v2 = payload_v2.to_vec();
    corrupted_v2[0] ^= 0xFF;
    let (fallback_hdr, _) = select_latest_valid_checkpoint(
        &[(&hdr_v1, payload_v1), (&hdr_v2, &corrupted_v2)],
        b"CFGIDX01",
        1,
        layout.meta_payload_capacity(),
    )
    .expect("fallback checkpoint");
    assert_eq!(fallback_hdr.seq, 4);

    // FDP placement-affinity slot allocation & crash-recovery validation:
    let mut alloc = RawBlockSlotAllocator::new(&layout, true);
    assert!(alloc.allocate_slot(Some(0)).is_err());
    let off0 = alloc.allocate_slot(Some(1)).expect("slot 0");
    let off1 = alloc.allocate_slot(Some(2)).expect("slot 1");
    assert_eq!(off0, 65536);
    assert_eq!(off1, 65536 + 65536);

    // Free both and re-allocate for placement_id = 1 (hit) then placement_id = 3 (fallback):
    alloc.free_slot_by_offset(off0);
    alloc.free_slot_by_offset(off1);
    let reused_p1 = alloc.allocate_slot(Some(1)).expect("affinity hit");
    assert_eq!(reused_p1, off0);
    assert_eq!(alloc.fdp_slot_affinity_hit_count, 1);

    let fallback_p3 = alloc.allocate_slot(Some(3)).expect("affinity fallback");
    assert_eq!(fallback_p3, off1);
    assert_eq!(alloc.fdp_slot_affinity_fallback_count, 1);

    // Rebuild from committed offsets (next_slot = 3, slot 0 committed -> slots 1, 2 free)
    // and validate on-disk headers:
    alloc
        .rebuild_from_committed_offsets(3, &[off0])
        .expect("rebuild succeeds");
    assert_eq!(alloc.free_slot_count(), 2);

    let dropped = alloc
        .validate_recovered_entries(&[("k0".to_string(), off0, 100, 4096)], &[Some((999, 4096))]);
    assert_eq!(dropped, vec!["k0".to_string()]);
    assert_eq!(alloc.free_slot_count(), 3);
}

#[test]
fn test_prepare_device_primitives_and_hierarchical_profile_scope_tracing() {
    use castforge::ir::Attribute;
    use castforge::passes::{add_profile_scope_tracing, prepare_device_primitives};

    let ir = concat!(
        "module @profile_and_dev_primitives {\n",
        "  func.func @kernel_entry(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %q = mogg.call_opaque %x",
        " {device_ref = \"gpu:2\", profile_scope = \"model/layer0/q_proj\",",
        " profile_color = \"blue\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %spec = mgp.tensor.extract.tensor_spec %q : (tensor<4xf32>) -> tensor<1xi64>\n",
        "    %k = mogg.call_opaque %q",
        " {device_ref = \"gpu:2\", profile_scope = \"model/layer0/k_proj\",",
        " profile_color = \"blue\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %k : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(ir).expect("valid MLIR");

    // 1. Add hierarchical profile scope tracing (2 leaf scopes + 2 shared ancestor scopes
    // "model" and "model/layer0"):
    let inserted = add_profile_scope_tracing(&mut m, false);
    assert_eq!(inserted, 4);

    // 2. Specialize device primitives (`mogg.call_opaque` -> `mogg.call_opaque.gpu`,
    // `device_ref = "gpu:2"` -> `device_ref_id = 2`):
    let specialized = prepare_device_primitives(&mut m);
    assert_eq!(specialized, 2);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk_id]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mogg.profile_scope.begin",
            "mogg.profile_scope.begin",
            "mogg.profile_scope.begin",
            "mogg.call_opaque.gpu",
            "mgp.tensor.extract.tensor_spec",
            "mogg.profile_scope.end",
            "mogg.profile_scope.begin",
            "mogg.call_opaque.gpu",
            "mogg.profile_scope.end",
            "mogg.profile_scope.end",
            "mogg.profile_scope.end",
            "func.return",
        ]
    );

    let first_begin = m.blocks[blk_id].ops[0];
    assert_eq!(
        m.ops[first_begin].attributes.get("name"),
        Some(&Attribute::String("model/kernel_entry".to_string()))
    );
    assert_eq!(
        m.ops[first_begin].attributes.get("color"),
        Some(&Attribute::String("blue".to_string()))
    );

    let gpu_op = m.blocks[blk_id].ops[3];
    assert_eq!(
        m.ops[gpu_op].attributes.get("device_ref_id"),
        Some(&Attribute::Int(2))
    );
    assert!(!m.ops[gpu_op].attributes.contains_key("device_ref"));
}

#[test]
fn test_generate_device_graph_build_stubs_and_builder_alloc_lowering() {
    use castforge::ir::Attribute;
    use castforge::passes::generate_device_graph_build_stubs;

    let ir = concat!(
        "module @device_graph_stubs {\n",
        "  func.func @matmul_stub(%in: tensor<4xf32>, %ctx: tensor<1xi64>)",
        " -> tensor<4xf32> {\n",
        "    %pack = mogg.tuple.create %in {is_stable_variadic = true}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %trace = mogg.with_trace %pack : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %res = mogg.call_opaque %pack, %ctx {has_build_graph = true}",
        " : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    return %res : tensor<4xf32>\n",
        "  }\n",
        "  func.func @copy_stub(%in: tensor<4xf32>, %ctx: tensor<1xi64>)",
        " -> tensor<4xf32> {\n",
        "    %res = mogg.call.view_copy %in, %ctx",
        " : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    return %res : tensor<4xf32>\n",
        "  }\n",
        "  func.func @graph_body(%builder: tensor<1xi64>, %x: tensor<4xf32>,",
        " %eager_ctx: tensor<1xi64>) -> tensor<4xf32> {\n",
        "    %buf = mgp.buffer.alloc %x : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %c1 = mogg.call %buf, %eager_ctx {callee = \"matmul_stub\"}",
        " : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    %c2 = mogg.call %c1, %eager_ctx {callee = \"copy_stub\"}",
        " : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    return %c2 : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(ir).expect("valid MLIR");
    let graph_func = m.functions()[2];
    m.ops[graph_func]
        .attributes
        .insert("is_device_graph".to_string(), Attribute::Bool(true));

    let rewritten =
        generate_device_graph_build_stubs(&mut m).expect("stub generation should succeed");
    assert_eq!(rewritten, 3);

    // Module now has 5 functions: matmul_stub, matmul_stub_build_graph, copy_stub,
    // copy_stub_build_graph, and graph_body.
    let funcs = m.functions();
    assert_eq!(funcs.len(), 5);

    // Verify `matmul_stub_build_graph` has `mogg.tuple.create` with `variadic_abi = "stable"`
    // and `mogg.call.build_graph` (with `mogg.with_trace` dropped):
    let matmul_bg = funcs[1];
    assert_eq!(
        m.ops[matmul_bg].attributes.get("sym_name"),
        Some(&Attribute::String("matmul_stub_build_graph".to_string()))
    );
    let matmul_bg_blk = m.regions[m.ops[matmul_bg].regions[0]].blocks[0];
    let matmul_bg_ops: Vec<String> = m.blocks[matmul_bg_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        matmul_bg_ops,
        vec!["mogg.tuple.create", "mogg.call.build_graph", "func.return"]
    );
    let tuple_op = m.blocks[matmul_bg_blk].ops[0];
    assert_eq!(
        m.ops[tuple_op].attributes.get("variadic_abi"),
        Some(&Attribute::String("stable".to_string()))
    );

    // Verify `copy_stub_build_graph` inserts `mgp.device_graph.recording_context` at block start:
    let copy_bg = funcs[3];
    let copy_bg_blk = m.regions[m.ops[copy_bg].regions[0]].blocks[0];
    let copy_bg_ops: Vec<String> = m.blocks[copy_bg_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        copy_bg_ops,
        vec![
            "mgp.device_graph.recording_context",
            "mogg.call.view_copy",
            "func.return"
        ]
    );

    // Verify `graph_body` rewrote `mgp.buffer.alloc` -> `mgp.device_graph.alloc` and calls:
    let gb_blk = m.regions[m.ops[graph_func].regions[0]].blocks[0];
    let builder_arg = m.blocks[gb_blk].arguments[0];
    let alloc_op = m.blocks[gb_blk].ops[0];
    assert_eq!(m.ops[alloc_op].name.full_name(), "mgp.device_graph.alloc");
    assert_eq!(m.ops[alloc_op].operands.last(), Some(&builder_arg));

    let call1_op = m.blocks[gb_blk].ops[1];
    assert_eq!(
        m.ops[call1_op].attributes.get("callee"),
        Some(&Attribute::String("matmul_stub_build_graph".to_string()))
    );
    assert_eq!(m.ops[call1_op].operands[1], builder_arg);
}

#[test]
fn test_globalize_and_inline_model_subgraphs_offsets_slots_and_prefixes_weights() {
    use castforge::ir::Attribute;
    use castforge::passes::globalize_and_inline_model_subgraphs;

    let ir = concat!(
        "module @subgraph_inlining {\n",
        "  func.func @sub_mlp(%x: tensor<4xf32>, %w_name: tensor<1xi64>) -> tensor<4xf32> {\n",
        "    %ctx = mgp.context.create %x {num_slots = 3} : (tensor<4xf32>) -> tensor<1xi64>\n",
        "    %w = mgp.buffer.constant_external %w_name {e_is_placeholder = true}",
        " : (tensor<1xi64>) -> tensor<4xf32>\n",
        "    %cached = mgp.buffer.cache %w, %ctx {b_buffer_slot = 1}",
        " : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    %out = mgp.execute.generic %x, %cached {a_runtime_slot = 2}",
        " : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %out : tensor<4xf32>\n",
        "  }\n",
        "  func.func @main_model(%in: tensor<4xf32>, %w_name: tensor<1xi64>) -> tensor<4xf32> {\n",
        "    %ctx = mgp.context.create %in {num_slots = 5} : (tensor<4xf32>) -> tensor<1xi64>\n",
        "    %res = mgp.call %in, %w_name {callee = \"sub_mlp\", prefix = \"layers.0.\"}",
        " : (tensor<4xf32>, tensor<1xi64>) -> tensor<4xf32>\n",
        "    return %res : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );

    // 1. Test globalize-only mode (`inline_subgraphs = false`):
    let mut m_glob = parse_mlir(ir).expect("valid MLIR");
    let count_glob =
        globalize_and_inline_model_subgraphs(&mut m_glob, false).expect("globalize succeeds");
    assert_eq!(count_glob, 1);
    let main_glob = m_glob.functions()[1];
    let main_glob_blk = m_glob.regions[m_glob.ops[main_glob].regions[0]].blocks[0];
    let glob_ops: Vec<String> = m_glob.blocks[main_glob_blk]
        .ops
        .iter()
        .map(|&id| m_glob.ops[id].name.full_name())
        .collect();
    assert_eq!(
        glob_ops,
        vec![
            "mgp.context.create",
            "index.constant",
            "mgp.call",
            "func.return"
        ]
    );
    let const_op = m_glob.blocks[main_glob_blk].ops[1];
    assert_eq!(
        m_glob.ops[const_op].attributes.get("value"),
        Some(&Attribute::Int(5))
    );
    let call_op = m_glob.blocks[main_glob_blk].ops[2];
    assert_eq!(
        m_glob.ops[call_op].attributes.get("is_globalized_index"),
        Some(&Attribute::Bool(true))
    );

    // 2. Test full subgraph inlining mode (`inline_subgraphs = true`):
    let mut m_inl = parse_mlir(ir).expect("valid MLIR");
    let sub_func = m_inl.functions()[0];
    m_inl.ops[sub_func]
        .attributes
        .insert("is_subgraph".to_string(), Attribute::Bool(true));

    let count_inl =
        globalize_and_inline_model_subgraphs(&mut m_inl, true).expect("inlining succeeds");
    assert_eq!(count_inl, 1);
    // `sub_mlp` was erased because it was marked `is_subgraph = true` and had no remaining uses:
    assert_eq!(m_inl.functions().len(), 1);

    let main_inl = m_inl.functions()[0];
    let main_inl_blk = m_inl.regions[m_inl.ops[main_inl].regions[0]].blocks[0];
    let inl_ops: Vec<String> = m_inl.blocks[main_inl_blk]
        .ops
        .iter()
        .map(|&id| m_inl.ops[id].name.full_name())
        .collect();
    assert_eq!(
        inl_ops,
        vec![
            "mgp.context.create",
            "mgp.string.create",
            "mgp.string.concat",
            "mgp.buffer.constant_external",
            "mgp.buffer.cache",
            "mgp.execute.generic",
            "func.return",
        ]
    );
    // Parent context `num_slots` updated to 5 + 3 = 8:
    let ctx_op = m_inl.blocks[main_inl_blk].ops[0];
    assert_eq!(
        m_inl.ops[ctx_op].attributes.get("num_slots"),
        Some(&Attribute::Int(8))
    );
    // `b_buffer_slot` offset from 1 -> 1 + 5 = 6, `a_runtime_slot` offset from 2 -> 2 + 5 = 7:
    let cache_op = m_inl.blocks[main_inl_blk].ops[4];
    assert_eq!(
        m_inl.ops[cache_op].attributes.get("b_buffer_slot"),
        Some(&Attribute::Int(6))
    );
    let exec_op = m_inl.blocks[main_inl_blk].ops[5];
    assert_eq!(
        m_inl.ops[exec_op].attributes.get("a_runtime_slot"),
        Some(&Attribute::Int(7))
    );
}

#[test]
fn test_legalize_rowwise_kernels_primary_selection_monoid_binding_and_launch_params() {
    use castforge::ir::{
        Attribute, Block, DType, Dim, OpName, Operation, Region, TensorType, Type, Value, ValueDef,
    };
    use castforge::passes::legalize_rowwise_kernels;
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    let ir = concat!(
        "module @rowwise_mod {\n",
        "  func.func @row_reduce(%in0: tensor<4xf32>, %out0: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %rw = mogg.call.rowwise %in0, %out0 {static_cols = 128}",
        " : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %rw : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(ir).expect("valid MLIR");
    let func_id = m.functions()[0];
    let func_blk = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let rw_op = m.blocks[func_blk].ops[0];
    let loc = m.ops[rw_op].location.clone();
    let f32_ty = Type::Tensor(TensorType::new(vec![Dim::Static(4)], DType::F32));
    let idx_ty = Type::Tensor(TensorType::new(vec![Dim::Static(1)], DType::I64));

    // Attach the rowwise body region with `[%cap_in, %cap_out, %row_idx]`, a `mogg.rowwise.reduce`
    // fold region with `[%elem_idx, %acc]`, an element-index `mogg.tensor.load`, an expensive
    // `mogg.call.elementwise` (`mo.exp`), and a row-indexed `mogg.tensor.store`:
    let rw_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let cap_in = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::BlockArgument {
            block: rw_blk,
            index: 0,
        },
    });
    let cap_out = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::BlockArgument {
            block: rw_blk,
            index: 1,
        },
    });
    let row_idx = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: idx_ty.clone(),
        def: ValueDef::BlockArgument {
            block: rw_blk,
            index: 2,
        },
    });
    m.blocks[rw_blk].arguments = vec![cap_in, cap_out, row_idx];

    let fold_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let elem_idx = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: idx_ty,
        def: ValueDef::BlockArgument {
            block: fold_blk,
            index: 0,
        },
    });
    m.blocks[fold_blk].arguments = vec![elem_idx];

    let load_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg.tensor", "load"),
        operands: smallvec::smallvec![cap_in, elem_idx],
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc.clone(),
    });
    let loaded_val = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::OpResult {
            op: load_op,
            index: 0,
        },
    });
    m.ops[load_op].results.push(loaded_val);

    let elem_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg.call", "elementwise"),
        operands: smallvec::smallvec![loaded_val],
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "mo_op_name".to_string(),
                Attribute::String("mo.exp".to_string()),
            );
            a
        },
        regions: Vec::new(),
        custom_syntax: false,
        location: loc.clone(),
    });
    let elem_val = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::OpResult {
            op: elem_op,
            index: 0,
        },
    });
    m.ops[elem_op].results.push(elem_val);
    m.blocks[fold_blk].ops = vec![load_op, elem_op];

    let fold_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![fold_blk],
    });
    let fold_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg.rowwise", "reduce"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "monoid".to_string(),
                Attribute::String("online_softmax".to_string()),
            );
            a
        },
        regions: vec![fold_reg],
        custom_syntax: false,
        location: loc.clone(),
    });
    let fold_res = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty,
        def: ValueDef::OpResult {
            op: fold_op,
            index: 0,
        },
    });
    m.ops[fold_op].results.push(fold_res);

    let store_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mogg.tensor", "store"),
        operands: smallvec::smallvec![cap_out, row_idx, fold_res],
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: loc,
    });
    m.blocks[rw_blk].ops = vec![fold_op, store_op];
    let rw_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![rw_blk],
    });
    m.ops[rw_op].regions.push(rw_reg);

    let count = legalize_rowwise_kernels(&mut m, 16).expect("rowwise legalization succeeds");
    assert_eq!(count, 1);

    assert_eq!(
        m.ops[rw_op].attributes.get("primary"),
        Some(&Attribute::Int(0))
    );
    assert_eq!(
        m.ops[rw_op].attributes.get("is_cached"),
        Some(&Attribute::Bool(false))
    );
    assert_eq!(
        m.ops[rw_op].attributes.get("accum_bitwidth"),
        Some(&Attribute::Int(32))
    );
    assert_eq!(
        m.ops[rw_op].attributes.get("computationally_expensive"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        m.ops[rw_op].attributes.get("simd_monoid"),
        Some(&Attribute::String("online_softmax".to_string()))
    );
    assert_eq!(
        m.ops[rw_op].attributes.get("cols"),
        Some(&Attribute::Int(128))
    );

    // Load inside fold became `mogg.rowwise.primary`, and store moved into `mogg.rowwise.emit`:
    assert_eq!(m.ops[load_op].name.full_name(), "mogg.rowwise.primary");
    let rw_body_ops: Vec<String> = m.blocks[rw_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        rw_body_ops,
        vec!["mogg.rowwise.reduce", "mogg.rowwise.emit"]
    );
}

#[test]
fn test_dynamic_file_storage_agent_l1_pages_and_sharded_atomic_store_plan() {
    use castforge::passes::{
        DynamicFileStorageAgent, build_single_file_xfer_descs, compute_l1_page_indices,
        is_orphan_temp_storage_file,
    };

    let pages = compute_l1_page_indices(4096 * 10, 4096 * 3, 4096).expect("aligned range");
    assert_eq!(pages, vec![10, 11, 12]);
    assert!(compute_l1_page_indices(4097, 4096, 4096).is_err());

    let file_descs = build_single_file_xfer_descs(7, 8192, 4096).expect("valid file xfer descs");
    assert_eq!(file_descs, vec![(0, 4096, 7), (4096, 4096, 7)]);

    assert!(is_orphan_temp_storage_file("model.bin.tmp.a1b2c3"));
    assert!(!is_orphan_temp_storage_file("model.bin"));

    let mut agent = DynamicFileStorageAgent::new(
        "/mnt/kv_store",
        "cuda",
        0x2000_0000,
        4096 * 16,
        4096,
        0,
        true,
        true,
    )
    .expect("valid agent");
    assert_eq!(agent.memory_type(), "VRAM");
    assert!(agent.use_direct_io());
    assert_eq!(agent.registered_page_count(), 16);

    // First store to shard `ab/cd` triggers `needs_mkdir`:
    let plan1 = agent
        .plan_atomic_store(
            &pages,
            "org/model-7b",
            1,
            2,
            "abcdef012345",
            "tenantA",
            "99ff",
        )
        .expect("plan 1");
    assert_eq!(plan1.needs_mkdir, Some("/mnt/kv_store/ab/cd".to_string()));
    assert_eq!(
        plan1.final_path,
        "/mnt/kv_store/ab/cd/org--model-7b_00000001_2_abcdef012345@tenantA.bin"
    );
    assert_eq!(
        plan1.temp_path,
        "/mnt/kv_store/ab/cd/org--model-7b_00000001_2_abcdef012345@tenantA.bin.tmp.99ff"
    );
    assert_eq!(plan1.file_size, 3 * 4096);

    // Second store with the same chunk hash prefix (`ab/cd`) but different salt skips mkdir:
    let plan2 = agent
        .plan_atomic_store(
            &pages,
            "org/model-7b",
            1,
            2,
            "abcd99999999",
            "tenantB",
            "88ee",
        )
        .expect("plan 2");
    assert_eq!(plan2.needs_mkdir, None);
}

#[test]
fn test_add_try_except_blocks_and_lookup_constant_resources() {
    use castforge::ir::Attribute;
    use castforge::passes::{add_try_except_blocks, lookup_constant_resources};

    let mlir_src = r#"
module {
  func.func @runtime_kernel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c0 = "castforge.buffer_constant"() {value = "res_alpha"} : () -> tensor<4xf32>
    %0 = "stablehlo.add"(%arg0, %c0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func @graph_kernel(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = "castforge.buffer_constant"() {value = "res_beta"} : () -> tensor<4xf32>
    %1 = "stablehlo.multiply"(%arg0, %c1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
"#;
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let funcs = m.functions();
    let f_runtime = funcs[0];
    let f_graph = funcs[1];

    m.ops[f_runtime]
        .attributes
        .insert("abi".to_string(), Attribute::String("runtime".to_string()));
    m.ops[f_runtime].attributes.insert(
        "constant_resources".to_string(),
        Attribute::Array(vec![
            Attribute::String("res_zero".to_string()),
            Attribute::String("res_alpha".to_string()),
        ]),
    );

    m.ops[f_graph]
        .attributes
        .insert("abi".to_string(), Attribute::String("graph".to_string()));
    m.ops[f_graph]
        .attributes
        .insert("throws".to_string(), Attribute::Bool(true));
    m.ops[f_graph].attributes.insert(
        "constant_resources".to_string(),
        Attribute::Array(vec![Attribute::String("res_beta".to_string())]),
    );

    let rewritten = lookup_constant_resources(&mut m).expect("lookup succeeds");
    assert_eq!(rewritten, 2);

    // Verify constant_resources on f_runtime was replaced by 2 unit placeholders:
    assert_eq!(
        m.ops[f_runtime].attributes.get("constant_resources"),
        Some(&Attribute::Array(vec![
            Attribute::Opaque("unit".to_string()),
            Attribute::Opaque("unit".to_string()),
        ]))
    );

    let wrapped = add_try_except_blocks(&mut m);
    assert_eq!(wrapped, 2);

    // Verify runtime_kernel has [castforge.try, func.return] in its entry block, and its except
    // block contains `castforge.raise`:
    let rt_entry = m.regions[m.ops[f_runtime].regions[0]].blocks[0];
    assert_eq!(m.blocks[rt_entry].ops.len(), 2);
    let rt_try_id = m.blocks[rt_entry].ops[0];
    assert!(m.ops[rt_try_id].is_op("castforge", "try"));
    let rt_except_blk = m.regions[m.ops[rt_try_id].regions[1]].blocks[0];
    let rt_except_op = m.blocks[rt_except_blk].ops[0];
    assert!(m.ops[rt_except_op].is_op("castforge", "raise"));

    // Verify graph_kernel's except block synthesizes `castforge.async_error` + `castforge.yield`:
    let gr_entry = m.regions[m.ops[f_graph].regions[0]].blocks[0];
    let gr_try_id = m.blocks[gr_entry].ops[0];
    let gr_except_blk = m.regions[m.ops[gr_try_id].regions[1]].blocks[0];
    assert_eq!(m.blocks[gr_except_blk].ops.len(), 2);
    assert!(m.ops[m.blocks[gr_except_blk].ops[0]].is_op("castforge", "async_error"));
    assert!(m.ops[m.blocks[gr_except_blk].ops[1]].is_op("castforge", "yield"));
}

#[test]
fn test_canonicalize_funcs_for_merge_and_merge_identical_kernel_stubs() {
    use castforge::ir::Attribute;
    use castforge::passes::{
        canonicalize_device_ref_str, canonicalize_funcs_for_merge, merge_identical_kernel_stubs,
    };

    assert_eq!(
        canonicalize_device_ref_str("gpu:3"),
        Some("gpu:0".to_string())
    );
    assert_eq!(canonicalize_device_ref_str("gpu:0"), None);

    let mlir_src = concat!(
        "module {\n",
        "  func.func @leaf_b(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %0 = \"stablehlo.exponential\"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %0 : tensor<4xf32>\n",
        "  }\n",
        "  func.func @leaf_a(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %0 = \"stablehlo.exponential\"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %0 : tensor<4xf32>\n",
        "  }\n",
        "  func.func @caller_b(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %0 = \"castforge.call\"(%arg0) {callee = @leaf_b}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %0 : tensor<4xf32>\n",
        "  }\n",
        "  func.func @caller_a(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %0 = \"castforge.call\"(%arg0) {callee = @leaf_a}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %0 : tensor<4xf32>\n",
        "  }\n",
        "  func.func @entry(%arg0: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %0 = \"castforge.compiled_call\"(%arg0) {callee_symbol = @caller_b}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %0 : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let funcs = m.functions();
    let f_leaf_b = funcs[0];
    let f_leaf_a = funcs[1];
    let f_caller_b = funcs[2];
    let f_caller_a = funcs[3];
    let f_entry = funcs[4];

    // Give leaf_b and leaf_a different device ordinals (`gpu:2` vs `gpu:1`) and different
    // non-SIMD symbolic shape dim names (`batch_x` vs `batch_y`):
    m.ops[f_leaf_b].attributes.insert(
        "device_ref".to_string(),
        Attribute::String("gpu:2".to_string()),
    );
    m.ops[f_leaf_a].attributes.insert(
        "device_ref".to_string(),
        Attribute::String("gpu:1".to_string()),
    );
    let op_b = m.blocks[m.regions[m.ops[f_leaf_b].regions[0]].blocks[0]].ops[0];
    let op_a = m.blocks[m.regions[m.ops[f_leaf_a].regions[0]].blocks[0]].ops[0];
    m.ops[op_b].attributes.insert(
        "shape_dims".to_string(),
        Attribute::Array(vec![
            Attribute::String("batch_x".to_string()),
            Attribute::String("simd:4".to_string()),
        ]),
    );
    m.ops[op_a].attributes.insert(
        "shape_dims".to_string(),
        Attribute::Array(vec![
            Attribute::String("batch_y".to_string()),
            Attribute::String("simd:4".to_string()),
        ]),
    );
    m.ops[f_caller_b].attributes.insert(
        "fusion_info".to_string(),
        Attribute::String("fused_from_layer_2".to_string()),
    );
    m.ops[f_caller_a].attributes.insert(
        "fusion_info".to_string(),
        Attribute::String("fused_from_layer_1".to_string()),
    );

    let canon_changes = canonicalize_funcs_for_merge(&mut m);
    assert_eq!(canon_changes, 4);

    // Bottom-up merge first collapses (@leaf_a, @leaf_b) -> @leaf_a, which makes @caller_a and
    // @caller_b identical modulo `fusion_info`, collapsing (@caller_a, @caller_b) -> @caller_a
    // and rewriting @entry's `castforge.compiled_call` to `@caller_a`:
    let merged = merge_identical_kernel_stubs(&mut m);
    assert_eq!(merged, 2);
    assert_eq!(m.functions().len(), 3);

    let entry_blk = m.regions[m.ops[f_entry].regions[0]].blocks[0];
    let compiled_call_id = m.blocks[entry_blk].ops[0];
    assert_eq!(
        m.ops[compiled_call_id]
            .attributes
            .get("callee_symbol")
            .and_then(|a| a.as_str()),
        Some("caller_a")
    );
}

#[test]
fn test_lower_compiled_routines_module_load_and_init_guard() {
    use castforge::ir::Attribute;
    use castforge::passes::lower_compiled_routines;

    let mlir_src = concat!(
        "module {\n",
        "  func.func @model_init(%ctx: tensor<1xi64>) -> tensor<4xf32> {\n",
        "    %rt = \"castforge.runtime.create\"(%ctx) {slot = 5}",
        " : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    %ld = \"castforge.compiled_load\"(%rt) : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    %ph = \"castforge.module.load.placeholder\"() : () -> tensor<1xi64>\n",
        "    %w0 = \"castforge.compiled_call\"()",
        " {callee_symbol = @init_weights, dispatch = \"deferred\"} : () -> tensor<4xf32>\n",
        "    return %w0 : tensor<4xf32>\n",
        "  }\n",
        "  func.func @model_execute(%ctx: tensor<1xi64>, %x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %y = \"castforge.compiled_call\"(%x)",
        " {callee_symbol = @fwd_kernel} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %z = \"castforge.to_graph_from_opaque\"(%y)",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %z : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let funcs = m.functions();
    let f_init = funcs[0];
    let f_exec = funcs[1];

    let init_blk = m.regions[m.ops[f_init].regions[0]].blocks[0];
    let load_op = m.blocks[init_blk].ops[1];
    m.ops[load_op].attributes.insert(
        "symbols".to_string(),
        Attribute::Array(vec![
            Attribute::String("init_weights".to_string()),
            Attribute::String("fwd_kernel".to_string()),
        ]),
    );

    let lowered = lower_compiled_routines(&mut m).expect("lowering succeeds");
    assert_eq!(lowered, 5);

    // Verify `castforge.compiled_load` became `castforge.module.load` with slot=5, module_index=0:
    assert!(m.ops[load_op].is_op("castforge", "module.load"));
    assert_eq!(
        m.ops[load_op].attributes.get("slot"),
        Some(&Attribute::Int(5))
    );
    assert_eq!(
        m.ops[load_op].attributes.get("module_index"),
        Some(&Attribute::Int(0))
    );

    // Verify `castforge.module.load.placeholder` became `castforge.chain.create` over [rt, ld],
    // and a `castforge.guard` was inserted before the init `castforge.generic.execute`:
    let init_ops = &m.blocks[init_blk].ops;
    assert!(m.ops[init_ops[2]].is_op("castforge", "chain.create"));
    assert!(m.ops[init_ops[3]].is_op("castforge", "guard"));
    assert!(m.ops[init_ops[4]].is_op("castforge", "generic.execute"));
    assert_eq!(
        m.ops[init_ops[4]].attributes.get("kernel_index"),
        Some(&Attribute::Int(0))
    );
    assert_eq!(
        m.ops[init_ops[4]].attributes.get("is_async"),
        Some(&Attribute::Bool(true))
    );

    // Verify execute function lowered `compiled_call` -> `generic.execute` (kernel_index = 1)
    // and elided `to_graph_from_opaque`:
    let exec_blk = m.regions[m.ops[f_exec].regions[0]].blocks[0];
    let exec_ops = &m.blocks[exec_blk].ops;
    assert_eq!(exec_ops.len(), 2);
    assert!(m.ops[exec_ops[0]].is_op("castforge", "generic.execute"));
    assert_eq!(
        m.ops[exec_ops[0]].attributes.get("kernel_index"),
        Some(&Attribute::Int(1))
    );
}
