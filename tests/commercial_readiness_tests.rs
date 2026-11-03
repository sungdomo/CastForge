//! Commercial-readiness integration tests covering the C FFI ABI (`src/ffi.rs`), untrusted input
//! hardening & resource quotas, lazy `.safetensors` indexing, structured pass telemetry, and
//! extended ONNX operators (`Einsum`, `Tile`, `TopK`, negative-axis ops).

use castforge::codegen::{
    BackendRegistry, CompilationCache, TargetConfig, compute_function_structural_hashes,
    compute_module_structural_hash, evaluate_module, lower_to_loop_ir,
};
use castforge::ffi::{
    castforge_compile_mlir, castforge_free_string, castforge_optimize_mlir, castforge_version,
};
use castforge::ir::formats::{import_gguf_bytes, import_onnx_json};
use castforge::ir::{
    DType, Dim, Location, TensorBacking, TensorEntry, TensorStore, TensorType, parse_mlir,
};
use castforge::passes::{
    ArcKvEvictionPolicy, AsymK16V8Codec, BLEND_REORDER_LMIN, BlendMatchSpan, BlendObjectGroupKind,
    BlendReadGroupSet, BlendTokenRangeMatcher, BucketHistogram, CBChunkMatch, CBGroupRopeGeometry,
    CBGroupSpec, CHUNK_COUNT_HISTOGRAM_BOUNDS, CacheBlendRopeSpec, CacheEventBatchSpec,
    CacheEventEntrySpec, CacheEventOpKind, CacheEventTier, CacheGenCdfCodec,
    ChainedBlockHashConfig, CollectiveGraphNodeSpec, ContiguousViewPlan, CoordinatorEventGate,
    CoordinatorKeyDirectory, DEFAULT_MAX_KERNELS_IN_FUSION_SET, DaxGenerationSlotArena,
    DaxPutOutcome, DeviceGraphStage, DeviceTargetArch, DynamicAllocRequest, EngineKvFormat,
    EventIngestOutcome, FULL_ATTENTION_WINDOW, FdpRawBlockSlotManager, FleetCacheUsageView,
    FleetWatermarkEvictionController, GDS_DEFAULT_ALIGN_BYTES, GdsSlabAddressManager,
    IsolatedLruEvictionPolicy, KV_ZERO_SCALE_SENTINEL, KvAttentionMaskKind, KvBlockPool,
    KvDmaTransferMode, KvPageSpec, KvPrefixBloomFilter, KvSaltQuotaRegistry, KvScaleScope,
    KvStorageKeySpec, L2FaultInjectConfig, L2FaultInjectPolicy, LayerGroupPayloadSharder,
    P2P_DEFAULT_MAX_MISSES, P2PPeerDiscoveryTable, P2PPeerSpec, PassManager, PassRegistry,
    PlacementRecord, RAW_BLOCK_META_MAGIC, RAW_BLOCK_SLOT_MAGIC, ROWWISE_CHEAP_OP_COST,
    ROWWISE_EXPENSIVE_OP_COST, RawBlockMetaCheckpointHeader, ReshapeDimMapping, TOKEN_WIRE_STRIDE,
    TurboQuantPreset, TurboQuantSerdeConfig, WarmPrefetchJobTable, WarmPrefetchStatus,
    analyze_kv_layer_groups, analyze_static_reshape_dim_mapping,
    annotate_squeeze_unsqueeze_reshapes, anonymize_symbolic_dims, await_hanging_effect_chains,
    build_orthonormal_hadamard, classify_blend_read_groups, coalesce_exhaustive_gathers_to_split,
    coalesce_partitioning_slices_to_split, compress_splat_constants,
    compute_cb_group_rope_geometry, compute_ieee_crc32, decode_kv_fs_filename,
    decode_kv_storage_key, decode_raw_block_slot_header, defer_allocs_to_first_use,
    direct_copy_format_supported, downsample_subchunk_swa_block_ids,
    elide_redundant_state_transfers, eliminate_implied_dependencies, encode_kv_fs_filename,
    encode_kv_storage_key, encode_raw_block_meta_checkpoint, encode_raw_block_slot_header,
    encode_wide_column_row_key, expected_contiguous_shape_from_strides, find_squeeze_axis,
    find_unsqueeze_axis, flatten_elementwise_chains, fold_extend_lifetimes_into_async_kernels,
    fold_unfold_hybrid_prefix_bitmaps, fuse_horizontal_shared_input_ops, fuse_locations,
    fuse_reshape_epilogue, fuse_sole_use_prologues, group_p2p_lookup_keys,
    hash_packed_token_blocks_be, hash_prefix_token_blocks, hoist_callsite_invariant_subgraph_ops,
    hoist_exec_invariant_ops, hoist_symbolic_dim_expressions, interleave_chunk_major_object_keys,
    is_supported_arch_dtype, legalize_rowwise_reduction_launch, longest_full_attention_hit,
    longest_joint_prefix_hit, longest_sliding_window_hit, map_axis_through_static_reshape,
    mark_cuda_graph_stable_tensors, merge_adjacent_parallel_loops, merge_duplicate_shape_ops,
    merge_identical_functions, merge_parallel_matmuls, num_packed_tokens, pack_token_ids_be,
    plan_cacheblend_retrieve, plan_cacheblend_token_reorder, plan_concat_in_place,
    plan_device_graph_collective_partition, plan_dynamic_memory_pool, plan_jenga_kv_geometry,
    plan_kernel_sequence_fusion, plan_kv_dma_transfer, promote_narrow_accumulators,
    re_rope_key_in_place, recalculate_subchunk_blocks_to_skip, recognize_composite_activations,
    recover_contiguous_or_dim0_padded_view, reuse_dead_loop_buffers,
    select_latest_valid_meta_checkpoint, sink_broadcast_through_reduction,
    sink_split_through_reshape, sink_transpose_through_elementwise_and_broadcast,
    solve_lloyd_max_gaussian, stamp_fused_provenance, swim_reshape_through_slice,
    unfold_chunk_range, unique_blend_token_coverage, unpack_token_ids_be, unpermute_slice,
    validate_target_device_dtypes,
};
use std::ffi::{CStr, CString, c_char};
use std::ptr;

#[test]
fn test_c_ffi_abi_optimize_compile_and_error_handling() {
    unsafe {
        let ver_ptr = castforge_version();
        assert!(!ver_ptr.is_null());
        let ver = CStr::from_ptr(ver_ptr).to_str().unwrap();
        assert_eq!(ver, "1.0.0");

        let mlir = CString::new(
            r#"
module @ffi_test {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %a = stablehlo.add %x, %z : tensor<2x4xf32>
    %b = stablehlo.maximum %a, %z : tensor<2x4xf32>
    func.return %b : tensor<2x4xf32>
  }
}
"#,
        )
        .unwrap();
        let preset = CString::new("O2").unwrap();
        let mut out_mlir: *mut c_char = ptr::null_mut();
        let mut out_tel: *mut c_char = ptr::null_mut();
        let mut out_err: *mut c_char = ptr::null_mut();

        let rc = castforge_optimize_mlir(
            mlir.as_ptr(),
            preset.as_ptr(),
            &mut out_mlir,
            &mut out_tel,
            &mut out_err,
        );
        assert_eq!(rc, 0);
        assert!(!out_mlir.is_null());
        assert!(!out_tel.is_null());
        assert!(out_err.is_null());

        let opt_str = CStr::from_ptr(out_mlir).to_str().unwrap().to_string();
        let tel_str = CStr::from_ptr(out_tel).to_str().unwrap().to_string();
        castforge_free_string(out_mlir);
        castforge_free_string(out_tel);

        assert!(opt_str.contains("func.func @main"));
        assert!(tel_str.contains("\"pass_name\""));
        assert!(tel_str.contains("\"elapsed_us\""));

        for target in ["rust", "c", "wgsl"] {
            let target_c = CString::new(target).unwrap();
            let mut out_code: *mut c_char = ptr::null_mut();
            let mut err_ptr: *mut c_char = ptr::null_mut();
            let rc = castforge_compile_mlir(
                mlir.as_ptr(),
                preset.as_ptr(),
                target_c.as_ptr(),
                &mut out_code,
                &mut err_ptr,
            );
            assert_eq!(rc, 0);
            assert!(!out_code.is_null());
            let code_str = CStr::from_ptr(out_code).to_str().unwrap().to_string();
            assert!(!code_str.is_empty());
            castforge_free_string(out_code);
        }

        // Verify FFI error handling on invalid target
        let bad_target = CString::new("unknown_gpu").unwrap();
        let mut out_code: *mut c_char = ptr::null_mut();
        let mut err_ptr: *mut c_char = ptr::null_mut();
        let rc = castforge_compile_mlir(
            mlir.as_ptr(),
            preset.as_ptr(),
            bad_target.as_ptr(),
            &mut out_code,
            &mut err_ptr,
        );
        assert_ne!(rc, 0);
        assert!(!err_ptr.is_null());
        let err_msg = CStr::from_ptr(err_ptr).to_str().unwrap().to_string();
        assert!(err_msg.contains("Unknown codegen target"));
        castforge_free_string(err_ptr);
    }
}

#[test]
fn test_untrusted_input_hardening_and_resource_quotas() {
    // 1. Tensor dimension overflow guard in MLIR parser
    let overflow_mlir = r#"
module @overflow {
  func.func @main(%x: tensor<1000000000x1000000000xf32>) -> tensor<1000000000x1000000000xf32> {
    func.return %x : tensor<1000000000x1000000000xf32>
  }
}
"#;
    assert!(
        parse_mlir(overflow_mlir).is_err(),
        "Parser must reject overflowing static tensor dimensions"
    );

    // 2. Negative static dimension guard in MLIR parser
    let neg_dim_mlir = r#"
module @neg {
  func.func @main(%x: tensor<-4x4xf32>) -> tensor<-4x4xf32> {
    func.return %x : tensor<-4x4xf32>
  }
}
"#;
    assert!(
        parse_mlir(neg_dim_mlir).is_err(),
        "Parser must reject negative static tensor dimensions"
    );

    // 3. Deep region nesting limit (> 64 nested regions)
    let mut deep_mlir = String::from("module @deep {\n  func.func @main(%x: tensor<2xf32>) {\n");
    for _ in 0..70 {
        deep_mlir.push_str("    \"scf.if\"(%x) {\n");
    }
    for _ in 0..70 {
        deep_mlir.push_str("    }\n");
    }
    deep_mlir.push_str("    func.return\n  }\n}\n");
    assert!(
        parse_mlir(&deep_mlir).is_err(),
        "Parser must reject region nesting exceeding MAX_REGION_NESTING_DEPTH"
    );

    // 4. SafeTensors oversized header guard (> 100 MiB claimed in 8-byte prefix)
    let mut bad_st = Vec::new();
    let huge_hdr: u64 = 500 * 1024 * 1024;
    bad_st.extend_from_slice(&huge_hdr.to_le_bytes());
    bad_st.extend_from_slice(b"{}");
    assert!(
        TensorStore::from_safetensors_bytes(&bad_st).is_err(),
        "SafeTensors loader must reject headers > 100 MiB"
    );

    // 5. GGUF excessive tensor count guard
    let mut bad_gguf = Vec::new();
    bad_gguf.extend_from_slice(b"GGUF");
    bad_gguf.extend_from_slice(&3u32.to_le_bytes());
    bad_gguf.extend_from_slice(&1_000_000u64.to_le_bytes()); // tensor_count > 65536
    bad_gguf.extend_from_slice(&0u64.to_le_bytes());
    assert!(
        import_gguf_bytes(&bad_gguf).is_err(),
        "GGUF importer must reject excessive header tensor counts"
    );
}

#[test]
fn test_lazy_safetensors_external_file_indexing_and_telemetry() {
    let mut store = TensorStore::new();
    store.insert(TensorEntry::from_f32(
        "layer0.weight",
        TensorType::new(vec![Dim::Static(2), Dim::Static(2)], DType::F32),
        vec![1.5, -2.5, 3.5, 4.5],
    ));

    let tmp_path = std::env::temp_dir().join("castforge_lazy_test.safetensors");
    store.save_safetensors_file(&tmp_path).unwrap();

    let mut lazy_store = TensorStore::new();
    let count = lazy_store.load_safetensors_lazy_file(&tmp_path).unwrap();
    assert_eq!(count, 1);

    let entry = lazy_store.get("layer0.weight").unwrap();
    assert!(matches!(entry.backing, TensorBacking::ExternalFile { .. }));

    // Round-trip lazy store back to bytes (reads ExternalFile slice on demand)
    let roundtrip_bytes = lazy_store.to_safetensors_bytes().unwrap();
    let reloaded = TensorStore::from_safetensors_bytes(&roundtrip_bytes).unwrap();
    let vals = reloaded
        .get("layer0.weight")
        .unwrap()
        .as_f32_slice()
        .unwrap();
    assert_eq!(vals, &[1.5, -2.5, 3.5, 4.5]);
    let _ = std::fs::remove_file(&tmp_path);

    // Verify structured telemetry JSON on an optimization pipeline
    let mut module = parse_mlir(
        r#"
module @tel {
  func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %a = stablehlo.add %x, %z : tensor<2x2xf32>
    func.return %a : tensor<2x2xf32>
  }
}
"#,
    )
    .unwrap();
    let reg = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(&reg, "O1").unwrap();
    let telemetry = pm.run_with_telemetry(&mut module).unwrap();
    assert!(!telemetry.is_empty());
    assert!(telemetry.iter().any(|t| t.changed));
}

#[test]
fn test_extended_onnx_operators_einsum_tile_topk_and_negative_axes() {
    let onnx_json = r#"{
  "name": "extended_onnx",
  "inputs": [
    { "name": "x", "shape": [2, 2], "dtype": "f32" }
  ],
  "initializers": [
    { "name": "w", "shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0] }
  ],
  "nodes": [
    {
      "op_type": "Einsum",
      "inputs": ["x", "w"],
      "outputs": ["mm"],
      "output_shape": [2, 2],
      "attrs": { "equation": "ij,jk->ik" }
    },
    {
      "op_type": "Tile",
      "inputs": ["mm"],
      "outputs": ["tiled"],
      "output_shape": [2, 4],
      "attrs": {}
    },
    {
      "op_type": "TopK",
      "inputs": ["tiled"],
      "outputs": ["top2"],
      "output_shape": [2, 2],
      "attrs": { "k": 2 }
    },
    {
      "op_type": "ReduceSum",
      "inputs": ["top2"],
      "outputs": ["out"],
      "output_shape": [2],
      "attrs": { "axis": -1 }
    }
  ],
  "outputs": [
    { "name": "out", "shape": [2], "dtype": "f32" }
  ]
}"#;

    let module = import_onnx_json(onnx_json).unwrap();
    let out = evaluate_module(&module, &[vec![1.0_f32, 0.0, 0.0, 1.0]])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    // x = I_2, so mm = w = [[1, 2], [3, 4]]
    // tiled = [[1, 2, 1, 2], [3, 4, 3, 4]]
    // top2 = [[2, 2], [4, 4]]
    // ReduceSum(axis=-1) = [4.0, 8.0]
    assert_eq!(out, vec![4.0, 8.0]);
}

#[test]
fn test_engine_kv_formats_and_heterogeneous_kv_layer_groups() {
    // Verify all 4 physical KV-cache layouts (NHD, HND, CrossLayer, MlaLatent)
    assert_eq!(
        EngineKvFormat::Nhd.physical_shape(32, 64, 16, 8, 128),
        vec![2, 64, 16, 8, 128]
    );
    assert_eq!(
        EngineKvFormat::Hnd.physical_shape(32, 64, 16, 8, 128),
        vec![2, 64, 8, 16, 128]
    );
    assert_eq!(
        EngineKvFormat::CrossLayer.physical_shape(32, 64, 16, 8, 128),
        vec![64, 32, 2, 16, 8, 128]
    );
    assert_eq!(
        EngineKvFormat::MlaLatent.physical_shape(32, 64, 16, 1, 512),
        vec![64, 16, 512]
    );
    assert!(
        EngineKvFormat::CrossLayer
            .index_expr("b", "l", "kv", "t", "h", "d")
            .contains("* NL + l")
    );

    // Build a 2-layer attention module and lower with `kv_format=cross_layer`
    let mlir = r#"
module @paged_kv_test {
  func.func @main(%q: tensor<2x4xf32>, %k: tensor<2x4xf32>, %v: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %a0 = stablehlo.composite %q, %k, %v
      {name = "castforge.fused_attention", decomposition = @fused_attention_impl}
      : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %a1 = stablehlo.composite %a0, %k, %v
      {name = "castforge.fused_attention", decomposition = @fused_attention_impl}
      : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %a1 : tensor<2x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    let reg = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &reg,
        "lower-paged-attention(page_block_size=16,max_blocks=32,kv_format=cross_layer)",
    )
    .unwrap();
    pm.run(&mut module).unwrap();

    let groups = analyze_kv_layer_groups(&module);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].format, EngineKvFormat::CrossLayer);
    assert_eq!(groups[0].layer_indices, vec![0, 1]);
    assert_eq!(groups[0].page_block_size, 16);
    assert_eq!(groups[0].head_size, 4);

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("castforge.kv_format = \"cross_layer\""));
    assert!(printed.contains("32, 2, 2, 16, 1, 4"));
    assert!(printed.contains("castforge.kv_layer_group = 0"));
    assert!(printed.contains("castforge.jenga_huge_block_bytes = 256"));
    assert!(printed.contains("castforge.jenga_pages_per_huge_block = 1"));
}

#[test]
fn test_jenga_kv_pool_geometry_and_hybrid_prefix_hit_planner() {
    // 1. Jenga Block-Pool Geometry:
    // Group 0 (e.g. MLA latent page): 3584 bytes (7 * 512), aligned to 256 -> can pad to 4096
    // Group 1 (e.g. GQA KV page): 4096 bytes, aligned to 256
    // Without padding, lcm(3584, 4096) = 28672. With <= 15% padding, 3584 pads to 4096.
    let specs = vec![
        KvPageSpec {
            data_bytes: 3584,
            stride_alignment: Some(256),
        },
        KvPageSpec {
            data_bytes: 4096,
            stride_alignment: Some(256),
        },
    ];
    let exact = plan_jenga_kv_geometry(&specs, 0.0, 1024 * 1024).unwrap();
    assert_eq!(exact.huge_block_bytes, 28672);
    assert_eq!(exact.pages_per_huge_block, vec![8, 7]);

    let padded = plan_jenga_kv_geometry(&specs, 0.15, 1024 * 1024).unwrap();
    assert_eq!(padded.huge_block_bytes, 4096);
    assert_eq!(padded.padded_page_bytes, vec![4096, 4096]);
    assert_eq!(padded.pages_per_huge_block, vec![1, 1]);

    // 2. Hybrid Full-Attention + Sliding-Window Prefix-Hit Planner:
    // Suppose we have 6 prompt blocks [0..6):
    // - Full-attention group has cached blocks: [true, true, true, true, true, false] -> max hit 5
    // - Sliding-window group (window = 2 blocks) evicted early blocks [0, 1] and kept [2, 3, 4]:
    //   cached = [false, false, true, true, true, false]
    //   Stopping at h = 5 requires blocks [3, 4] cached -> true!
    //   Stopping at h = 4 requires blocks [2, 3] cached -> true!
    //   Stopping at h = 3 requires blocks [1, 2] cached -> false (block 1 evicted)!
    let full_cached = [true, true, true, true, true, false];
    let sw_cached = [false, false, true, true, true, false];

    assert_eq!(longest_full_attention_hit(6, |i| full_cached[i]), 5);
    assert_eq!(longest_sliding_window_hit(6, 2, |i| sw_cached[i]), 5);

    let f_full = |i: usize| full_cached[i];
    let f_sw = |i: usize| sw_cached[i];
    let joint = longest_joint_prefix_hit(
        6,
        &[
            (KvAttentionMaskKind::FullAttention, &f_full),
            (
                KvAttentionMaskKind::SlidingWindow {
                    blocks_in_window: 2,
                },
                &f_sw,
            ),
        ],
    );
    assert_eq!(joint, 5);

    // Now suppose full-attention only cached [0, 1, 2] (hit = 3), while sliding-window cached
    // [false, false, true, true, true, false]. At upper = 3, sliding-window needs [1, 2] cached,
    // which fails, narrowing to h = 0!
    let full_short = [true, true, true, false, false, false];
    let f_full_short = |i: usize| full_short[i];
    let joint_narrowed = longest_joint_prefix_hit(
        6,
        &[
            (KvAttentionMaskKind::FullAttention, &f_full_short),
            (
                KvAttentionMaskKind::SlidingWindow {
                    blocks_in_window: 2,
                },
                &f_sw,
            ),
        ],
    );
    assert_eq!(joint_narrowed, 0);
}

#[test]
fn test_subgraph_structural_hash_and_compilation_cache() {
    let mlir_layer_0 = r#"
module @decoder_block_0 {
  func.func @layer_0(%x0: tensor<2x4xf32>, %w0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %sum0 = stablehlo.add %x0, %w0 : tensor<2x4xf32>
    %act0 = stablehlo.tanh %sum0 : tensor<2x4xf32>
    func.return %act0 : tensor<2x4xf32>
  }
}
"#;
    let mlir_layer_31 = r#"
module @decoder_block_31 {
  func.func @layer_31(%hidden: tensor<2x4xf32>, %weight: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %added = stablehlo.add %hidden, %weight : tensor<2x4xf32>
    %out = stablehlo.tanh %added : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    let mod_0 = parse_mlir(mlir_layer_0).unwrap();
    let mod_31 = parse_mlir(mlir_layer_31).unwrap();

    // Despite different module, function, and SSA variable names, structural hashes match!
    assert_eq!(
        compute_module_structural_hash(&mod_0),
        compute_module_structural_hash(&mod_31)
    );

    let registry = BackendRegistry::with_builtin_backends();
    let config = TargetConfig::default();
    let mut cache = CompilationCache::new();

    // First compilation: cold miss
    let (code_0, hit_0) = registry
        .generate_cached("c", &mod_0, &config, &mut cache)
        .unwrap();
    assert!(!hit_0);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 0);
    assert_eq!(cache.len(), 1);

    // Second compilation of structurally identical decoder block 31: warm hit in O(1)
    let (code_31, hit_31) = registry
        .generate_cached("c", &mod_31, &config, &mut cache)
        .unwrap();
    assert!(hit_31);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.len(), 1);
    assert_eq!(code_0, code_31);

    // Multi-function module with 3 identical decoder layers and 1 main function:
    let multi_layer_mlir = r#"
module @transformer_3_layers {
  func.func @layer_0(%x: tensor<2x4xf32>, %w: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.add %x, %w : tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
  func.func @layer_1(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.add %a, %b : tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
  func.func @layer_2(%p: tensor<2x4xf32>, %q: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.add %p, %q : tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
  func.func @main(%in: tensor<2x4xf32>, %w: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.multiply %in, %w : tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
"#;
    let multi_mod = parse_mlir(multi_layer_mlir).unwrap();
    let fn_hashes = compute_function_structural_hashes(&multi_mod);
    assert_eq!(fn_hashes.len(), 4);
    assert_eq!(fn_hashes[0].1, fn_hashes[1].1);
    assert_eq!(fn_hashes[1].1, fn_hashes[2].1);
    assert_ne!(fn_hashes[0].1, fn_hashes[3].1);
    assert_eq!(CompilationCache::analyze_layer_reuse(&multi_mod), (2, 4));
}

#[test]
fn test_nvtx_roctx_hardware_profiler_range_emission() {
    let mlir = r#"
module @profiler_test {
  func.func @main(%x: tensor<2x4xf32>, %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.add %x, %y : tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    let pass_reg = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&pass_reg, "instrument-profiling(filter=all)").unwrap();
    pm.run(&mut module).unwrap();

    let backend_reg = BackendRegistry::with_builtin_backends();
    let cfg = TargetConfig::default();

    let c_src = backend_reg.generate("c", &module, &cfg).unwrap();
    assert!(c_src.contains("#ifdef CASTFORGE_ENABLE_NVTX"));
    assert!(c_src.contains("#include <nvToolsExt.h>"));
    assert!(c_src.contains("nvtxRangePushA(\"stablehlo.add\");"));
    assert!(c_src.contains("nvtxRangePop();"));

    let cuda_src = backend_reg.generate("cuda", &module, &cfg).unwrap();
    assert!(cuda_src.contains("#ifdef CASTFORGE_ENABLE_NVTX"));
    assert!(cuda_src.contains("#include <nvToolsExt.h>"));
    assert!(cuda_src.contains("void main_cuda_launch("));
    assert!(cuda_src.contains("nvtxRangePushA(\"main:stablehlo.add\");"));
    assert!(cuda_src.contains("nvtxRangePop();"));

    let hip_src = backend_reg.generate("hip", &module, &cfg).unwrap();
    assert!(hip_src.contains("#ifdef CASTFORGE_ENABLE_ROCTX"));
    assert!(hip_src.contains("#include <roctx.h>"));
    assert!(hip_src.contains("void main_hip_launch("));
    assert!(hip_src.contains("roctxRangePushA(\"main:stablehlo.add\");"));
    assert!(hip_src.contains("roctxRangePop();"));

    let rust_src = backend_reg.generate("rust", &module, &cfg).unwrap();
    assert!(rust_src.contains("// [profile:region(id=0, label=\"stablehlo.add\")]"));
    assert!(rust_src.contains("std::hint::black_box(\"stablehlo.add\")"));
}

#[test]
fn test_asym_k16_v8_kv_codec_and_scoped_scales() {
    // Shape: [num_pages = 2, page_tokens = 4, num_kv_heads = 2, head_dim = 8] -> 128 elems
    let (num_pages, page_tokens, num_kv_heads, head_dim) = (2, 4, 2, 8);
    let total = num_pages * page_tokens * num_kv_heads * head_dim;

    let k_orig: Vec<f32> = (0..total).map(|i| ((i as f32) - 64.0) * 0.05).collect();
    // Make page 0 normal-range and page 1 head 1 all-zero to verify KV_ZERO_SCALE_SENTINEL
    let mut v_orig: Vec<f32> = (0..total).map(|i| ((i as f32) - 32.0) * 0.1).collect();
    let head_stride = head_dim;
    let tok_stride = num_kv_heads * head_stride;
    let page_stride = page_tokens * tok_stride;
    for t in 0..page_tokens {
        for d in 0..head_dim {
            v_orig[page_stride + t * tok_stride + head_stride + d] = 0.0;
        }
    }

    for scope in [
        KvScaleScope::PerTensor,
        KvScaleScope::PerLayerHead,
        KvScaleScope::PerPageHead,
    ] {
        let enc = AsymK16V8Codec::encode(
            &k_orig,
            &v_orig,
            num_pages,
            page_tokens,
            num_kv_heads,
            head_dim,
            scope,
        )
        .unwrap();

        match scope {
            KvScaleScope::PerTensor => assert_eq!(enc.v_scales.len(), 1),
            KvScaleScope::PerLayerHead => assert_eq!(enc.v_scales.len(), num_kv_heads),
            KvScaleScope::PerPageHead => {
                assert_eq!(enc.v_scales.len(), num_pages * num_kv_heads);
                // Page 1, head 1 was zeroed -> scale must equal KV_ZERO_SCALE_SENTINEL (1.0)
                assert_eq!(enc.v_scales[3], KV_ZERO_SCALE_SENTINEL);
            }
        }

        // Verify ~25% byte reduction compared to 16-bit K+V (512B -> 384B + scale bytes)
        assert_eq!(enc.uncompressed_fp16_bytes(), 512);
        assert!(enc.payload_bytes() <= 400);

        let (k_dec, v_dec) = AsymK16V8Codec::decode(&enc);
        for (&ko, &kd) in k_orig.iter().zip(k_dec.iter()) {
            assert!((ko - kd).abs() < 5e-3, "K drift: {ko} vs {kd}");
        }
        for (&vo, &vd) in v_orig.iter().zip(v_dec.iter()) {
            assert!(!vd.is_nan());
            assert!((vo - vd).abs() < 0.08, "V drift: {vo} vs {vd}");
        }
    }
}

#[test]
fn test_cacheblend_delta_re_rope_in_place() {
    // Verify that rotating unrotated keys K_raw from 0 -> old_pos and then re-RoPEing in-place
    // from old_pos -> new_pos matches rotating K_raw directly from 0 -> new_pos, for both
    // NeoX half-split and GPT-J interleaved RoPE, including MLA trailing rope_base_offset!
    let k_raw: Vec<f32> = (0..48).map(|i| ((i as f32) + 1.0) * 0.125).collect();
    let zeros = [0i64, 0, 0];
    let old_pos = [5i64, 6, 7];
    let new_pos = [42i64, 43, 44];

    for (is_neox, rope_base_offset) in [(true, 0usize), (false, 0usize), (true, 4usize)] {
        let spec = CacheBlendRopeSpec {
            num_kv_heads: 2,
            head_stride: 8,
            rope_base_offset,
            rot_dim: 4,
            rope_theta: 10000.0,
            is_neox,
        };

        let mut k_via_delta = k_raw.clone();
        re_rope_key_in_place(&mut k_via_delta, &zeros, &old_pos, &spec).unwrap();
        re_rope_key_in_place(&mut k_via_delta, &old_pos, &new_pos, &spec).unwrap();

        let mut k_direct = k_raw.clone();
        re_rope_key_in_place(&mut k_direct, &zeros, &new_pos, &spec).unwrap();

        for (idx, (&a, &b)) in k_via_delta.iter().zip(k_direct.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Re-RoPE mismatch at idx={idx} (neox={is_neox}, offset={rope_base_offset}): \
                 {a} vs {b}"
            );
        }

        // For MLA (rope_base_offset = 4), verify the leading 4 latent dims per head were untouched
        if rope_base_offset == 4 {
            for t in 0..3 {
                for h in 0..2 {
                    let base = t * 16 + h * 8;
                    assert_eq!(&k_via_delta[base..base + 4], &k_raw[base..base + 4]);
                }
            }
        }
    }
}

#[test]
fn test_chained_prefix_cache_block_hashing_and_adapter_salt() {
    // 3 full blocks of 4 + 2 trailing ragged tokens
    let prompt_a = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 99, 100];
    let prompt_b = [10, 11, 12, 13, 20, 21, 22, 23, 40, 41, 42, 43]; // shares first 2 blocks of 4

    let cfg = ChainedBlockHashConfig {
        block_size: 4,
        seed: 7,
        salt: b"llama3-8b:lora_finance_v1",
        parent_hash: None,
    };

    let hashes_a = hash_prefix_token_blocks(&prompt_a, &cfg).unwrap();
    let hashes_b = hash_prefix_token_blocks(&prompt_b, &cfg).unwrap();

    // Trailing 2 ragged tokens in prompt_a are dropped -> 3 full blocks
    assert_eq!(hashes_a.len(), 3);
    assert_eq!(hashes_b.len(), 3);
    // First 2 blocks match identically; 3rd block diverges
    assert_eq!(hashes_a[0], hashes_b[0]);
    assert_eq!(hashes_a[1], hashes_b[1]);
    assert_ne!(hashes_a[2], hashes_b[2]);

    // Resuming from parent_hash = Some(hashes_a[1]) for block 2 must match hashes_a[2]
    let cfg_resume = ChainedBlockHashConfig {
        block_size: 4,
        seed: 7,
        salt: b"llama3-8b:lora_finance_v1",
        parent_hash: Some(hashes_a[1]),
    };
    let resumed = hash_prefix_token_blocks(&prompt_a[8..12], &cfg_resume).unwrap();
    assert_eq!(resumed, vec![hashes_a[2]]);

    // Changing the LoRA adapter salt changes every block hash from block 0 onward
    let cfg_diff_salt = ChainedBlockHashConfig {
        block_size: 4,
        seed: 7,
        salt: b"llama3-8b:lora_medical_v2",
        parent_hash: None,
    };
    let hashes_diff = hash_prefix_token_blocks(&prompt_a, &cfg_diff_salt).unwrap();
    assert_ne!(hashes_a[0], hashes_diff[0]);
}

#[test]
fn test_strided_nan_check_sanitizer_pass_and_gpu_lowering() {
    // 5 floating-point compute ops (indices 0, 1, 2, 3, 4).
    // With `sanitize-numerics(stride=2,all=true,clamp_max=1024)`, only indices 0, 2, 4
    // (3 out of 5 ops) should receive `castforge.nan_guard`!
    let mlir = r#"
module @strided_nan_check {
  func.func @main(%x: tensor<2x4xf32>, %y: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.add %x, %y : tensor<2x4xf32>
    %1 = stablehlo.multiply %0, %x : tensor<2x4xf32>
    %2 = stablehlo.subtract %1, %y : tensor<2x4xf32>
    %3 = stablehlo.exponential %2 : tensor<2x4xf32>
    %4 = stablehlo.tanh %3 : tensor<2x4xf32>
    func.return %4 : tensor<2x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    let reg = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &reg,
        "sanitize-numerics(stride=2,all=true,clamp_max=1024.0)",
    )
    .unwrap();
    let res = pm.run(&mut module).unwrap();
    assert_eq!(res[0].ops_created, 3);

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("sample_index = 0"));
    assert!(printed.contains("sample_index = 2"));
    assert!(printed.contains("sample_index = 4"));
    assert!(!printed.contains("sample_index = 1"));
    assert!(!printed.contains("sample_index = 3"));

    // Verify CUDA and C codegen emit finite clamps for `castforge.nan_guard`
    let backend_reg = BackendRegistry::with_builtin_backends();
    let cuda_src = backend_reg
        .generate("cuda", &module, &TargetConfig::default())
        .unwrap();
    assert!(cuda_src.contains("isnan("));
    assert!(cuda_src.contains("-1024.0f"));
    assert!(cuda_src.contains("1024.0f"));
}

#[test]
fn test_splat_constant_compression_and_cuda_graph_capture_stability() {
    let mlir = r#"
module @splat_and_cuda_graph {
  func.func @main(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense_resource<uniform_w> : tensor<4x4xf32>
    %0 = stablehlo.add %x, %c : tensor<4x4xf32>
    %1 = stablehlo.tanh %0 : tensor<4x4xf32>
    func.return %1 : tensor<4x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    module.tensor_store.insert(TensorEntry::from_f32(
        "uniform_w",
        TensorType::new(vec![Dim::Static(4), Dim::Static(4)], DType::F32),
        vec![2.5f32; 16],
    ));

    let input = vec![1.0f32; 16];
    let out_before = evaluate_module(&module, std::slice::from_ref(&input)).unwrap();

    // Compress uniform constants with >= 8 elements into 1-element scalar + broadcast_in_dim
    let compressed = compress_splat_constants(&mut module, 8);
    assert_eq!(compressed, 1);
    assert!(module.tensor_store.get("uniform_w").is_none());

    let out_after = evaluate_module(&module, &[input]).unwrap();
    assert_eq!(out_before, out_after);

    let report = mark_cuda_graph_stable_tensors(&mut module);
    assert!(report.is_capture_safe);
    assert!(report.blocking_ops.is_empty());
    assert_eq!(report.dynamic_input_count, 1);
    assert!(report.stable_value_count >= 2);

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("stablehlo.broadcast_in_dim"));
    assert!(printed.contains("castforge.cuda_graph_capturable = true"));
    assert!(printed.contains("castforge.cuda_graph_stable = true"));
}

#[test]
fn test_intrusive_lru_kv_block_pool_revival_and_reverse_chain_eviction() {
    let mut pool = KvBlockPool::new(3);
    assert_eq!(pool.free_count(), 3);

    // Allocate 3 blocks: commit b0 (root prefix) and b1 (child prefix), leave b2 uncommitted
    let b0 = pool.allocate().unwrap();
    let b1 = pool.allocate().unwrap();
    let b2 = pool.allocate().unwrap();
    assert_eq!((b0, b1, b2), (0, 1, 2));
    assert_eq!(pool.free_count(), 0);

    pool.commit(b0, 0xAAAA);
    pool.commit(b1, 0xBBBB);

    // Release chain [b0, b1] in reverse order -> b1 goes to free list first, then b0 at MRU tail.
    // Release uncommitted b2 -> goes to the FRONT (LRU head) so it is reused before evicting b1/b0!
    pool.release_request_chain(&[b0, b1]);
    pool.release_block(b2);
    assert_eq!(pool.free_count(), 3);

    // Next allocation MUST recycle uncommitted b2 (front of free list) without evicting b0 or b1!
    let recycled = pool.allocate().unwrap();
    assert_eq!(recycled, b2);

    // Revive b0 from the free list via prefix hit in O(1):
    let revived_b0 = pool.touch_or_revive_prefix(0xAAAA).unwrap();
    assert_eq!(revived_b0, b0);
    assert_eq!(pool.free_count(), 1); // Only b1 remains on the free list

    // Next allocation must evict b1 (the tail of the chain, which was freed before b0)
    let evicted = pool.allocate().unwrap();
    assert_eq!(evicted, b1);
    assert!(pool.touch_or_revive_prefix(0xBBBB).is_none());
}

#[test]
fn test_kv_dma_transfer_planner_and_pin_chunk_boundary_splitting() {
    assert!(direct_copy_format_supported(EngineKvFormat::Nhd));
    assert!(direct_copy_format_supported(EngineKvFormat::CrossLayer));
    assert!(direct_copy_format_supported(EngineKvFormat::MlaLatent));
    assert!(!direct_copy_format_supported(EngineKvFormat::Hnd));

    // 1. Hnd format must select StagedSmKernel
    let hnd_plan = plan_kv_dma_transfer(EngineKvFormat::Hnd, 4, 1024, &[0], 0, 4096);
    assert_eq!(hnd_plan.mode, KvDmaTransferMode::StagedSmKernel);
    assert_eq!(hnd_plan.entries.len(), 4);

    // 2. CrossLayer collapses all 4 layers (4 * 1024 = 4096B) into a single DirectCopy2DPitched
    // entry when aligned, or splits across pin-chunk boundaries when straddling `align = 4096`!
    let cross_aligned = plan_kv_dma_transfer(EngineKvFormat::CrossLayer, 4, 1024, &[2], 0, 4096);
    assert_eq!(cross_aligned.mode, KvDmaTransferMode::DirectCopy2DPitched);
    assert_eq!(cross_aligned.entries.len(), 1);
    assert_eq!(cross_aligned.pin_boundary_splits, 0);
    assert_eq!(cross_aligned.entries[0].width_bytes, 4096);

    // Start at host_offset = 3072 with pin-chunk alignment = 4096 -> 4096B transfer straddles
    // boundary 4096 and must split into [3072..4096) (1024B) + [4096..7168) (3072B)!
    let cross_split = plan_kv_dma_transfer(EngineKvFormat::CrossLayer, 4, 1024, &[2], 3072, 4096);
    assert_eq!(cross_split.mode, KvDmaTransferMode::DirectCopy2DPitched);
    assert_eq!(cross_split.pin_boundary_splits, 1);
    assert_eq!(cross_split.entries.len(), 2);
    assert_eq!(cross_split.entries[0].host_offset, 3072);
    assert_eq!(cross_split.entries[0].width_bytes, 1024);
    assert_eq!(cross_split.entries[1].host_offset, 4096);
    assert_eq!(cross_split.entries[1].width_bytes, 3072);
}

#[test]
fn test_prologue_input_fusion_into_gather_and_gemm() {
    let mlir = r#"
module @prologue_fusion {
  func.func @main(%x: tensor<4x4xf32>, %w: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.tanh %x : tensor<4x4xf32>
    %1 = stablehlo.dot_general %0, %w : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %1 : tensor<4x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    let x = vec![0.5f32; 16];
    let w = vec![1.0f32; 16];
    let out_before = evaluate_module(&module, &[x.clone(), w.clone()]).unwrap();

    let fused = fuse_sole_use_prologues(&mut module);
    assert_eq!(fused, 1);

    let printed = castforge::ir::print_mlir(&module);
    assert!(!printed.contains("stablehlo.tanh"));
    assert!(printed.contains("castforge.prologue_op = \"tanh\""));

    let out_after = evaluate_module(&module, &[x, w]).unwrap();
    assert_eq!(out_before, out_after);

    let backend_reg = BackendRegistry::with_builtin_backends();
    let cuda_src = backend_reg
        .generate("cuda", &module, &TargetConfig::default())
        .unwrap();
    assert!(cuda_src.contains("tanhf("));
}

#[test]
fn test_bounded_kernel_sequence_fusion_horizon_and_device_contexts() {
    assert_eq!(DEFAULT_MAX_KERNELS_IN_FUSION_SET, 16);

    let mlir = r#"
module @sequence_fusion {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.abs %x : tensor<4xf32>
    %1 = stablehlo.tanh %0 : tensor<4xf32>
    %2 = stablehlo.negate %1 : tensor<4xf32>
    %3 = stablehlo.exponential %2 {castforge.stream_id = 1} : tensor<4xf32>
    %4 = stablehlo.all_reduce %3 : tensor<4xf32>
    %5 = stablehlo.log %4 {castforge.device = "gpu:1"} : tensor<4xf32>
    func.return %5 : tensor<4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    // Cap at max_kernels_per_set = 2:
    // - Set 0: [%0 (abs), %1 (tanh)] on ("gpu:0", stream 0) -> capped at 2
    // - Set 1: [%2 (negate)] on ("gpu:0", stream 0)
    // - Set 2: [%3 (exponential)] on ("gpu:0", stream 1) -> context switch (GEX-2590)
    // - %4 (all_reduce) is a collective barrier -> breaks active set and is excluded
    // - Set 3: [%5 (log)] on ("gpu:1", stream 0) -> device context switch
    let sets = plan_kernel_sequence_fusion(&mut module, 2);
    assert_eq!(sets.len(), 4);

    assert_eq!(sets[0].set_id, 0);
    assert_eq!(sets[0].device_context, "gpu:0");
    assert_eq!(sets[0].stream_id, 0);
    assert_eq!(sets[0].op_names, vec!["stablehlo.abs", "stablehlo.tanh"]);

    assert_eq!(sets[1].set_id, 1);
    assert_eq!(sets[1].op_names, vec!["stablehlo.negate"]);

    assert_eq!(sets[2].set_id, 2);
    assert_eq!(sets[2].stream_id, 1);
    assert_eq!(sets[2].op_names, vec!["stablehlo.exponential"]);

    assert_eq!(sets[3].set_id, 3);
    assert_eq!(sets[3].device_context, "gpu:1");
    assert_eq!(sets[3].op_names, vec!["stablehlo.log"]);
}

#[test]
fn test_fused_location_provenance_preservation_across_lowering() {
    let fused_loc = fuse_locations(&[
        Location::Named("attn_q_tanh".to_string()),
        Location::Unknown,
        Location::FileLineCol {
            file: "llama.py".to_string(),
            line: 142,
            col: 8,
        },
        Location::Named("attn_q_tanh".to_string()), // duplicate should be deduplicated
    ]);
    assert_eq!(
        fused_loc,
        Location::Named("attn_q_tanh+llama.py:142:8".to_string())
    );

    let mlir = r#"
module @provenance_fusion {
  func.func @main(%x: tensor<4x4xf32>, %w: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = stablehlo.tanh %x : tensor<4x4xf32>
    %1 = stablehlo.dot_general %0, %w : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %1 : tensor<4x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    // Stamp constituent source locations on `tanh` and `dot_general` before prologue fusion
    for (_, op) in module.ops.iter_mut() {
        if op.is_op("stablehlo", "tanh") {
            op.location = Location::Named("layer0.input_tanh".to_string());
        } else if op.is_op("stablehlo", "dot_general") {
            stamp_fused_provenance(
                op,
                &[Location::FileLineCol {
                    file: "model.py".to_string(),
                    line: 88,
                    col: 12,
                }],
            );
        }
    }

    let fused = fuse_sole_use_prologues(&mut module);
    assert_eq!(fused, 1);

    let backend_reg = BackendRegistry::with_builtin_backends();
    let cuda_src = backend_reg
        .generate("cuda", &module, &TargetConfig::default())
        .unwrap();
    assert!(cuda_src.contains("Provenance: layer0.input_tanh+model.py:88:12"));
}

#[test]
fn test_anonymize_symbolic_dims_and_merge_identical_functions() {
    let mlir = r#"
module @dedup_funcs {
  func.func @layer_b(%x: tensor<sym_bxf32>) -> tensor<sym_bxf32> {
    %0 = stablehlo.tanh %x : tensor<sym_bxf32>
    %1 = stablehlo.abs %0 : tensor<sym_bxf32>
    func.return %1 : tensor<sym_bxf32>
  }
  func.func @layer_a(%y: tensor<sym_axf32>) -> tensor<sym_axf32> {
    %p = stablehlo.tanh %y : tensor<sym_axf32>
    %q = stablehlo.abs %p : tensor<sym_axf32>
    func.return %q : tensor<sym_axf32>
  }
  func.func @main(%in: tensor<?xf32>) -> tensor<?xf32> {
    %t0 = func.call @layer_b(%in) : (tensor<?xf32>) -> tensor<?xf32>
    %t1 = func.call @layer_a(%t0) : (tensor<?xf32>) -> tensor<?xf32>
    func.return %t1 : tensor<?xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).unwrap();
    for &fn_id in &module.functions() {
        let sym = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap()
            .to_string();
        module.ops[fn_id].location = Location::Named(format!("src_{sym}"));
    }

    // 1. Anonymize symbolic dimensions (`sym_a`, `sym_b` -> `?`) so `@layer_a` and `@layer_b`
    // become structurally identical
    let anon_count = anonymize_symbolic_dims(&mut module);
    assert!(anon_count >= 6);
    assert_eq!(anonymize_symbolic_dims(&mut module), 0); // idempotent

    // 2. Merge identical functions -> `@layer_a` is chosen lexicographically over `@layer_b`,
    // `@layer_b` is removed, its location is fused into `@layer_a`, and both `func.call` ops
    // now invoke `@layer_a`!
    let merged = merge_identical_functions(&mut module);
    assert_eq!(merged, 1);
    assert_eq!(module.functions().len(), 2);

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("func.func @layer_a("));
    assert!(!printed.contains("func.func @layer_b("));
    assert_eq!(printed.matches("func.call @layer_a(").count(), 2);
    assert!(printed.contains("src_layer_a+src_layer_b"));
}

#[test]
fn test_reshape_epilogue_fusion_into_matmul_bias_tanh() {
    let mlir = concat!(
        "module @reshape_epilogue {\n",
        "  func.func @main(%x: tensor<4x4xf32>, %w: tensor<4x4xf32>, ",
        "%b: tensor<4x4xf32>) -> tensor<2x2x4xf32> {\n",
        "    %dot = stablehlo.dot_general %x, %w ",
        ": (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>\n",
        "    %r0 = stablehlo.reshape %dot : (tensor<4x4xf32>) -> tensor<2x2x4xf32>\n",
        "    %rb = stablehlo.reshape %b : (tensor<4x4xf32>) -> tensor<2x2x4xf32>\n",
        "    %add = stablehlo.add %r0, %rb : tensor<2x2x4xf32>\n",
        "    %act = stablehlo.tanh %add : tensor<2x2x4xf32>\n",
        "    func.return %act : tensor<2x2x4xf32>\n",
        "  }\n",
        "}\n",
    );
    let mut module = parse_mlir(mlir).unwrap();
    let x = vec![0.25f32; 16];
    let w = vec![0.5f32; 16];
    let b = vec![0.1f32; 16];
    let out_before = evaluate_module(&module, &[x.clone(), w.clone(), b.clone()]).unwrap();

    // Commute `reshape` past `add` and `tanh` and fuse into `castforge.fused_matmul_epilogue`
    let commuted = fuse_reshape_epilogue(&mut module);
    assert_eq!(commuted, 2);

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("castforge.fused_matmul_epilogue"));
    assert!(printed.contains("epilogue = \"bias_tanh\""));
    assert_eq!(printed.matches("stablehlo.reshape").count(), 1);

    let out_after = evaluate_module(&module, &[x, w, b]).unwrap();
    assert_eq!(out_before, out_after);
}

#[test]
fn test_cachegen_16bit_monotonic_cdf_and_arithmetic_coding_roundtrip() {
    let codec = CacheGenCdfCodec::new(16);
    // Highly skewed KV quantization bin stream (concentrated around center bin 8)
    let symbols: Vec<u8> = (0..64)
        .map(|i| match i % 8 {
            0..=5 => 8,
            6 => 7,
            _ => 9,
        })
        .collect();

    let cdf = codec.compute_channel_cdf(&symbols);
    assert_eq!(cdf.len(), 17);
    assert_eq!(cdf[0], 0);
    // Strictly monotonic (`cdf[i + 1] > cdf[i]` even for zero-count bins!)
    for pair in cdf.windows(2) {
        assert!(pair[1] > pair[0]);
    }

    let bitstream = codec.encode_channel(&cdf, &symbols);
    // Skewed 64-symbol stream compresses well below 64 raw bytes
    assert!(bitstream.len() < 32);

    let decoded = codec.decode_channel(&cdf, &bitstream, symbols.len());
    assert_eq!(decoded, symbols);
}

#[test]
fn test_cacheblend_batched_retrieve_plan_waves_and_staging() {
    let groups = vec![CBGroupSpec {
        group_idx: 0,
        num_layers: 2,
        slot_tokens: 16,
        num_temp_slots: 2, // double-buffered temporary staging slots
        bytes_per_token_per_layer: 128,
        rope_enabled: true,
    }];

    // 3 non-block-aligned RAG chunks -> must partition into 2 waves (`[chunk0, chunk1]` and
    // `[chunk2]`) because `num_temp_slots = 2`!
    let chunks = vec![
        CBChunkMatch {
            host_offset: 0,
            num_tokens: 8, // 2 * 8 * 128 = 2048B (aligned within [0..4096))
            old_start_pos: 0,
            cur_start_pos: 0, // no position shift -> no re-RoPE needed!
            slot_mapping_offset: 0,
        },
        CBChunkMatch {
            host_offset: 3072,
            num_tokens: 8, // 2048B starting at 3072 -> straddles 4096 pin boundary!
            old_start_pos: 10,
            cur_start_pos: 42, // position shifted -> emits CBRopeVar(old=10, cur=42)
            slot_mapping_offset: 8,
        },
        CBChunkMatch {
            host_offset: 8192,
            num_tokens: 6,
            old_start_pos: 5,
            cur_start_pos: 20,
            slot_mapping_offset: 16,
        },
    ];

    let steps = plan_cacheblend_retrieve(&groups, &chunks, 4096);
    assert_eq!(steps.len(), 2);

    // Wave 0: 2 chunks -> chunk 0 has 1 aligned staging copy, chunk 1 splits into 2 copies
    assert_eq!(steps[0].staging_copies.len(), 3);
    assert_eq!(steps[0].ropes.len(), 1); // only chunk 1 (slot_idx = 1) needed re-RoPE
    assert_eq!(steps[0].ropes[0].slot_idx, 1);
    assert_eq!(steps[0].ropes[0].old_st, 10);
    assert_eq!(steps[0].ropes[0].cur_st, 42);
    assert_eq!(steps[0].scatters.len(), 2);

    // Wave 1: chunk 2 reuses `slot_idx = 0`!
    assert_eq!(steps[1].staging_copies.len(), 1);
    assert_eq!(steps[1].ropes.len(), 1);
    assert_eq!(steps[1].ropes[0].slot_idx, 0);
    assert_eq!(steps[1].scatters.len(), 1);
    assert_eq!(steps[1].scatters[0].slot_idx, 0);
    assert_eq!(steps[1].scatters[0].n_tok, 6);
}

#[test]
fn test_promote_narrow_accumulators_and_epilogue_downcast_codegen() {
    let mlir = concat!(
        "module @narrow_accum {\n",
        "  func.func @main(%x: tensor<4x4xbf16>, %w: tensor<4x4xbf16>) -> tensor<4x4xbf16> {\n",
        "    %0 = stablehlo.dot_general %x, %w ",
        ": (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>\n",
        "    func.return %0 : tensor<4x4xbf16>\n",
        "  }\n",
        "}\n",
    );
    let mut module = parse_mlir(mlir).unwrap();
    let promoted = promote_narrow_accumulators(&mut module);
    assert_eq!(promoted, 1);
    assert_eq!(promote_narrow_accumulators(&mut module), 0); // idempotent

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("castforge.accumulator_dtype = \"f32\""));
    assert!(printed.contains("castforge.epilogue_downcast = \"bf16\""));

    let backend_reg = BackendRegistry::with_builtin_backends();
    let cuda_src = backend_reg
        .generate("cuda", &module, &TargetConfig::default())
        .unwrap();
    assert!(cuda_src.contains("Mixed-precision accumulator (f32, epilogue downcast -> bf16)"));
}

#[test]
fn test_arc_kv_eviction_policy_ghost_adaptation_and_scan_resistance() {
    let mut arc = ArcKvEvictionPolicy::new();
    // Admit request chain [100, 101, 102, 103] -> processed in reverse so 103 is at T1 LRU head
    arc.on_keys_created(&[100, 101, 102, 103]);
    assert_eq!(arc.t1_len(), 4);
    assert_eq!(arc.t2_len(), 0);

    // Touch shared root prefix [100, 101] -> promoted to frequent list `T2`!
    arc.on_keys_touched(&[100, 101]);
    assert_eq!(arc.t1_len(), 2); // [103, 102] remain in T1
    assert_eq!(arc.t2_len(), 2); // [101, 100] in T2

    // Evict 50% (2 keys) -> since |T1| >= int(p=0), victims come from T1 in suffix-first order!
    let victims = arc.select_and_evict_victims(0.5);
    assert_eq!(victims, vec![103, 102]);
    assert_eq!(arc.t1_len(), 0);
    assert_eq!(arc.t2_len(), 2); // Frequent prefix [100, 101] survived the eviction!
    assert_eq!(arc.b1_len(), 2); // [103, 102] recorded in ghost list B1

    // Recreate ghost key 103 from B1 -> grows `target_t1_size` (`p > 0`) and promotes 103 to T2
    arc.on_keys_created(&[103]);
    assert!(arc.target_t1_size() >= 1.0);
    assert_eq!(arc.b1_len(), 1);
    assert_eq!(arc.t2_len(), 3);
}

#[test]
fn test_kv_salt_quota_registry_allowlist_and_bootstrap_exempt_lenses() {
    let mut reg = KvSaltQuotaRegistry::new();
    reg.set_quota("tenant_alpha", 4096);

    // 1. Registered tenant resolves to 4096 under both lenses
    assert!(reg.has_quota("tenant_alpha"));
    assert_eq!(reg.get_limit_bytes("tenant_alpha"), 4096);
    assert_eq!(reg.effective_limit_bytes("tenant_alpha"), Some(4096));
    assert_eq!(reg.over_quota_bytes("tenant_alpha", 5120, false), 1024);

    // 2. Unregistered tenant before `set_default_limit_bytes`:
    // - Strict allowlist lens resolves to `0` (evicts all 2048 bytes on local workers)
    // - Coordinator bootstrap lens resolves to `None` (exempt: 0 bytes over quota!)
    assert!(!reg.has_quota("tenant_unknown"));
    assert_eq!(reg.get_limit_bytes("tenant_unknown"), 0);
    assert_eq!(reg.effective_limit_bytes("tenant_unknown"), None);
    assert_eq!(reg.over_quota_bytes("tenant_unknown", 2048, true), 2048);
    assert_eq!(reg.over_quota_bytes("tenant_unknown", 2048, false), 0);

    // 3. Arm strict allowlist on coordinator (`set_default_limit_bytes(Some(0))`) after sync
    reg.set_default_limit_bytes(Some(0));
    assert_eq!(reg.effective_limit_bytes("tenant_unknown"), Some(0));
    assert_eq!(reg.over_quota_bytes("tenant_unknown", 2048, false), 2048);

    assert!(reg.delete_quota("tenant_alpha"));
    assert!(!reg.has_quota("tenant_alpha"));
}

#[test]
fn test_flatten_elementwise_chains_eliminates_interior_reshapes() {
    let mlir = concat!(
        "module @flatten_elemwise {\n",
        "  func.func @main(%x: tensor<2x4xf32>) -> tensor<4x2xf32> {\n",
        "    %0 = stablehlo.tanh %x : tensor<2x4xf32>\n",
        "    %1 = stablehlo.reshape %0 : (tensor<2x4xf32>) -> tensor<4x2xf32>\n",
        "    %2 = stablehlo.abs %1 : tensor<4x2xf32>\n",
        "    func.return %2 : tensor<4x2xf32>\n",
        "  }\n",
        "}\n",
    );
    let mut module = parse_mlir(mlir).unwrap();
    let input = vec![-0.5, 0.25, -1.0, 0.75, -0.2, 0.4, -0.8, 0.1];
    let out_before = evaluate_module(&module, std::slice::from_ref(&input)).unwrap();

    // Flatten both `tanh` (`2x4`) and `abs` (`4x2`) to 1D `tensor<8xf32>` and fold the
    // interior `2x4 -> 4x2` reshape so `tanh` and `abs` become adjacent 1D operations!
    let flattened = flatten_elementwise_chains(&mut module);
    assert_eq!(flattened, 2);
    assert_eq!(flatten_elementwise_chains(&mut module), 0); // idempotent

    let printed = castforge::ir::print_mlir(&module);
    assert!(printed.contains("stablehlo.tanh"));
    assert!(printed.contains("tensor<8xf32>"));

    let out_after = evaluate_module(&module, &[input]).unwrap();
    assert_eq!(out_before, out_after);
}

#[test]
fn test_elide_redundant_state_transfers_store_load_and_dead_store() {
    let mlir = concat!(
        "module @elide_state {\n",
        "  func.func @main(%buf: tensor<8xf32>, %p1: tensor<4xf32>, ",
        "%p2: tensor<4xf32>, %idx: tensor<i32>) -> tensor<4xf32> {\n",
        "    %u1 = stablehlo.dynamic_update_slice %buf, %p1, %idx ",
        ": (tensor<8xf32>, tensor<4xf32>, tensor<i32>) -> tensor<8xf32>\n",
        "    %u2 = stablehlo.dynamic_update_slice %u1, %p2, %idx ",
        ": (tensor<8xf32>, tensor<4xf32>, tensor<i32>) -> tensor<8xf32>\n",
        "    %sl = stablehlo.dynamic_slice %u2, %idx ",
        ": (tensor<8xf32>, tensor<i32>) -> tensor<4xf32>\n",
        "    %off = stablehlo.composite %sl {name = \"castforge.kv_offload\"} ",
        ": (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %rel = stablehlo.composite %off {name = \"castforge.kv_reload\"} ",
        ": (tensor<4xf32>) -> tensor<4xf32>\n",
        "    func.return %rel : tensor<4xf32>\n",
        "  }\n",
        "}\n",
    );
    let mut module = parse_mlir(mlir).unwrap();
    // 1. `%u1` is overwritten by `%u2` at the exact same `%idx` -> `%u1` elided (1)
    // 2. `%sl` reads exact slice written by `%u2` -> forwards `%p2`, elides `%sl` + `%u2` (2)
    // 3. `%rel` reloads `%off` -> forwards `%p2` and elides `%rel` + `%off` (2)
    let elided = elide_redundant_state_transfers(&mut module);
    assert_eq!(elided, 5);

    let printed = castforge::ir::print_mlir(&module);
    assert!(!printed.contains("dynamic_update_slice"));
    assert!(!printed.contains("dynamic_slice"));
    assert!(!printed.contains("kv_offload"));
    assert!(!printed.contains("kv_reload"));
    assert!(printed.contains("return %p2 : tensor<4xf32>"));
}

#[test]
fn test_validate_target_device_dtypes_across_cpu_wgsl_and_gpu_profiles() {
    assert!(is_supported_arch_dtype(
        DeviceTargetArch::X86_64,
        &DType::BF16
    ));
    assert!(is_supported_arch_dtype(
        DeviceTargetArch::X86_64,
        &DType::F16
    ));
    assert!(!is_supported_arch_dtype(
        DeviceTargetArch::X86_64,
        &DType::F8E4M3FN
    ));
    assert!(is_supported_arch_dtype(
        DeviceTargetArch::AArch64,
        &DType::BF16
    ));
    assert!(!is_supported_arch_dtype(
        DeviceTargetArch::AArch64,
        &DType::F16
    ));
    assert!(!is_supported_arch_dtype(
        DeviceTargetArch::Wgsl,
        &DType::BF16
    ));
    assert!(is_supported_arch_dtype(
        DeviceTargetArch::GpuSm90,
        &DType::F8E4M3FN
    ));

    let mlir = concat!(
        "module @validate_devices {\n",
        "  func.func @main(%x: tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN> {\n",
        "    %0 = stablehlo.abs %x : tensor<4xf8E4M3FN>\n",
        "    func.return %0 : tensor<4xf8E4M3FN>\n",
        "  }\n",
        "}\n",
    );
    let module = parse_mlir(mlir).unwrap();

    // `GpuSm90` supports `f8E4M3FN` natively -> 0 diagnostics
    assert!(validate_target_device_dtypes(&module, DeviceTargetArch::GpuSm90).is_empty());

    // `X86_64` rejects sub-16-bit floats (`f8E4M3FN`) on both block arg and `stablehlo.abs`
    let x86_diags = validate_target_device_dtypes(&module, DeviceTargetArch::X86_64);
    assert_eq!(x86_diags.len(), 2);
    assert_eq!(x86_diags[0].dtype, DType::F8E4M3FN);
    assert!(
        x86_diags[0]
            .message
            .contains("The f8E4M3FN data type is not supported on device 'cpu:x86_64'.")
    );
}

#[test]
fn test_hybrid_prefix_bitmap_fold_unfold_non_monotonic_recovery() {
    assert_eq!(unfold_chunk_range(4, FULL_ATTENTION_WINDOW), (0, 4));
    assert_eq!(unfold_chunk_range(4, 2), (2, 4));
    assert_eq!(unfold_chunk_range(4, 1), (3, 4)); // Mamba w=1

    // 3 object groups over 4 chunks:
    // - Group 0 (Full attention, w = -1): chunks [0, 1, 2] present, chunk 3 missing
    // - Group 1 (Sliding window, w = 2): chunk 0 evicted! chunks [1, 2, 3] present
    // - Group 2 (Mamba state, w = 1): only chunk 2 present!
    // Notice that at length L = 1 (chunk 0) and L = 2 (chunks 0..2), Group 1 & 2 cannot serve
    // because chunk 0 is missing, so `servable` is `[false, false, true, false]`!
    // Picking `highest_set_bit` recovers `hit_length = 3` (`j = 2`)!
    let presence = vec![
        vec![true, true, true, false],
        vec![false, true, true, true],
        vec![false, false, true, false],
    ];
    let windows = vec![FULL_ATTENTION_WINDOW, 2, 1];
    let (hit_len, servable, retain) = fold_unfold_hybrid_prefix_bitmaps(&presence, &windows);

    assert_eq!(hit_len, 3);
    assert_eq!(servable, vec![false, false, true, false]);
    // Unfold retain masks for `hit_len = 3`:
    assert_eq!(retain[0], vec![true, true, true, false]); // Full attention: [0..3)
    assert_eq!(retain[1], vec![false, true, true, false]); // Sliding w=2: [1..3)
    assert_eq!(retain[2], vec![false, false, true, false]); // Mamba w=1: [2..3)
}

#[test]
fn test_isolated_lru_per_salt_eviction_preserves_other_tenants() {
    let mut policy = IsolatedLruEvictionPolicy::new();

    // Tenant A creates sequence chunks [10, 11, 12, 13] -> inserted in reverse order
    // so LRU order is [13, 12, 11, 10] (later chunks evicted before earlier prefix chunks!)
    policy.on_keys_created("tenant-a", &[10, 11, 12, 13]);
    policy.on_keys_created("tenant-b", &[20, 21]);

    assert_eq!(policy.resident_len_for_salt("tenant-a"), 4);
    assert_eq!(policy.resident_len_for_salt("tenant-b"), 2);
    assert_eq!(policy.total_resident_len(), 6);

    // Touching chunk 13 in tenant-a promotes it to MRU -> LRU order becomes [12, 11, 10, 13]
    policy.on_keys_touched("tenant-a", &[13]);

    // Evict 50% of tenant-a's bucket -> evicts [12, 11]; tenant-b is completely untouched!
    let evicted = policy.evict_for_salt("tenant-a", 0.5);
    assert_eq!(evicted, vec![12, 11]);
    assert_eq!(policy.resident_len_for_salt("tenant-a"), 2);
    assert_eq!(policy.resident_len_for_salt("tenant-b"), 2);
}

#[test]
fn test_gds_slab_address_manager_alignment_all_or_nothing_and_coalescing() {
    // 16 KiB total backing file (4 x 4096-byte pages)
    let mut mgr = GdsSlabAddressManager::new(16_384, GDS_DEFAULT_ALIGN_BYTES);
    assert_eq!(mgr.free_size_bytes(), 16_384);
    assert_eq!(mgr.align_up(5_000), 8_192);

    // Allocate 2 slabs of 3,000 logical bytes -> each rounds up to 4,096 physical bytes
    let batch1 = mgr
        .allocate_batch(3_000, 2)
        .expect("2 x 4KiB should succeed");
    assert_eq!(batch1.len(), 2);
    assert_eq!(batch1[0].offset, 0);
    assert_eq!(batch1[0].physical_size, 4_096);
    assert_eq!(batch1[1].offset, 4_096);
    assert_eq!(mgr.free_size_bytes(), 8_192);

    // Request 3 more 4KiB slabs when only 2 (8,192B) remain -> all-or-nothing rollback!
    assert!(mgr.allocate_batch(3_000, 3).is_none());
    // Free size and fragment count must be completely restored after rollback!
    assert_eq!(mgr.free_size_bytes(), 8_192);
    assert_eq!(mgr.free_fragment_count(), 1);

    // Freeing the first slab creates 2 fragments ([0..4096) and [8192..16384));
    // freeing the second slab ([4096..8192)) coalesces everything back into 1 contiguous region!
    mgr.free_region(batch1[0]);
    assert_eq!(mgr.free_fragment_count(), 2);
    mgr.free_region(batch1[1]);
    assert_eq!(mgr.free_fragment_count(), 1);
    assert_eq!(mgr.free_size_bytes(), 16_384);
}

#[test]
fn test_defer_allocs_to_first_use_shrinks_loop_ir_live_ranges() {
    use castforge::ir::loops::{LoopKernel, LoopModule, LoopStmt};

    // Construct a kernel where `buf_late` and `buf_dead` are hoisted to the top of the function
    // before `buf_early`'s loop, inflating peak live memory across the first loop.
    let mut loop_mod = LoopModule {
        name: "defer_test".to_string(),
        kernels: vec![LoopKernel {
            name: "main".to_string(),
            params: vec![("arg0".to_string(), 16)],
            output: ("out0".to_string(), 16),
            body: vec![
                LoopStmt::Alloc {
                    name: "buf_early".to_string(),
                    len: 16,
                },
                LoopStmt::Alloc {
                    name: "buf_late".to_string(),
                    len: 16,
                },
                LoopStmt::Alloc {
                    name: "buf_dead".to_string(),
                    len: 16,
                },
                LoopStmt::For {
                    var: "i".to_string(),
                    lower: 0,
                    upper: 16,
                    step: 1,
                    parallel: true,
                    vectorized: true,
                    unroll: 4,
                    body: vec![LoopStmt::Assign {
                        dst: "buf_early".to_string(),
                        index: "i".to_string(),
                        expr: "arg0[i] + 1.0".to_string(),
                    }],
                },
                LoopStmt::For {
                    var: "i".to_string(),
                    lower: 0,
                    upper: 16,
                    step: 1,
                    parallel: true,
                    vectorized: true,
                    unroll: 4,
                    body: vec![LoopStmt::Assign {
                        dst: "buf_late".to_string(),
                        index: "i".to_string(),
                        expr: "buf_early[i] * 2.0".to_string(),
                    }],
                },
            ],
        }],
    };

    // Also verify `lower_to_loop_ir` + `defer_allocs_to_first_use` works on a parsed MLIR module
    let mlir = r#"
module @chain {
  func.func @main(%x: tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.abs %x : tensor<8xf32>
    %1 = stablehlo.negate %0 : tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
"#;
    let parsed = parse_mlir(mlir).expect("valid MLIR");
    let mut lowered = lower_to_loop_ir(&parsed, &TargetConfig::default());
    let _ = defer_allocs_to_first_use(&mut lowered);

    let changed = defer_allocs_to_first_use(&mut loop_mod);
    // `buf_late` deferred past first `For` loop + `buf_dead` eliminated = 2
    assert_eq!(changed, 2);
    let body = &loop_mod.kernels[0].body;
    assert_eq!(body.len(), 4);
    assert!(matches!(&body[0], LoopStmt::Alloc { name, .. } if name == "buf_early"));
    assert!(matches!(&body[1], LoopStmt::For { .. }));
    assert!(matches!(&body[2], LoopStmt::Alloc { name, .. } if name == "buf_late"));
    assert!(matches!(&body[3], LoopStmt::For { .. }));
}

#[test]
fn test_merge_duplicate_shape_materializations_and_provenance() {
    let mlir = r#"
module @shape_dedup {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<8xf32> {
    %r0 = stablehlo.reshape %x : (tensor<2x4xf32>) -> tensor<8xf32>
    %r1 = stablehlo.reshape %x : (tensor<2x4xf32>) -> tensor<8xf32>
    %i0 = stablehlo.iota {iota_dimension = 0 : i64} : tensor<8xf32>
    %i1 = stablehlo.iota {iota_dimension = 0 : i64} : tensor<8xf32>
    %sum0 = stablehlo.add %r0, %r1 : tensor<8xf32>
    %sum1 = stablehlo.add %i0, %i1 : tensor<8xf32>
    %out = stablehlo.add %sum0, %sum1 : tensor<8xf32>
    return %out : tensor<8xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    // Assign named locations to the two reshapes so we can verify provenance fusion
    let func_id = module.functions()[0];
    let func_reg = module.ops[func_id].regions[0];
    let func_block = module.regions[func_reg].blocks[0];
    let op_ids = module.blocks[func_block].ops.clone();
    module.ops[op_ids[0]].location = Location::Named("reshape_lhs".to_string());
    module.ops[op_ids[1]].location = Location::Named("reshape_rhs".to_string());

    let merged = merge_duplicate_shape_ops(&mut module);
    // Both duplicate `%r1` (`stablehlo.reshape`) and duplicate `%i1` (`stablehlo.iota`) are merged!
    assert_eq!(merged, 2);

    let remaining_ops = &module.blocks[func_block].ops;
    let primary_reshape = &module.ops[remaining_ops[0]];
    assert_eq!(primary_reshape.name.to_string(), "stablehlo.reshape");
    assert_eq!(
        primary_reshape.location,
        Location::Named("reshape_lhs+reshape_rhs".to_string())
    );

    let printed = castforge::ir::print_mlir(&module);
    assert_eq!(printed.matches("stablehlo.reshape").count(), 1);
    assert_eq!(printed.matches("stablehlo.iota").count(), 1);
}

#[test]
fn test_kv_prefix_bloom_filter_optimal_sizing_and_batch_dedup() {
    let mut bf = KvPrefixBloomFilter::new(1_000, 0.01);
    // m = floor(-(1000 * ln(0.01)) / (ln(2)^2)) = 9585, k = floor((9585/1000) * ln(2)) = 6
    assert_eq!(bf.bit_size(), 9_585);
    assert_eq!(bf.hash_count(), 6);
    assert_eq!(bf.bits_set(), 0);

    // Batch with duplicates [101, 202, 101, 303, 202] -> exactly 3 newly observed unique keys!
    let new_count = bf.add_batch_and_count_new(&[101, 202, 101, 303, 202]);
    assert_eq!(new_count, 3);
    assert_eq!(bf.item_count(), 3);
    assert!(bf.contains(101));
    assert!(bf.contains(202));
    assert!(bf.contains(303));
    assert!(!bf.contains(999_999));
    assert!(bf.fill_rate() > 0.0);

    bf.clear();
    assert_eq!(bf.item_count(), 0);
    assert_eq!(bf.bits_set(), 0);
}

#[test]
fn test_warm_prefetch_job_table_consume_once_lifecycle() {
    let mut table = WarmPrefetchJobTable::new();
    assert_eq!(table.poll("req-missing"), WarmPrefetchStatus::Unknown);

    // Submit 2 object groups of 3 keys each (6 total requested keys)
    table.submit("req-1", &[vec![10, 11, 12], vec![20, 21, 22]]);
    assert_eq!(table.active_job_count(), 1);
    assert_eq!(table.poll("req-1"), WarmPrefetchStatus::Pending);

    // Complete with 4 out of 6 keys found in L2
    assert!(table.record_completion("req-1", &[vec![true, true, false], vec![true, false, true]]));

    // First poll after completion returns `Completed { found_keys: 4, total_keys: 6 }` and drops it
    assert_eq!(
        table.poll("req-1"),
        WarmPrefetchStatus::Completed {
            found_keys: 4,
            total_keys: 6,
        }
    );
    assert_eq!(table.active_job_count(), 0);
    // Subsequent poll for the same request_id returns `Unknown` (consume-once semantics)
    assert_eq!(table.poll("req-1"), WarmPrefetchStatus::Unknown);
}

#[test]
fn test_layer_group_payload_sharder_qualifiers_and_roundtrip() {
    // 5 layers, group size 2 -> 3 groups: layers_0_1 (2), layers_2_3 (2), layers_4_4 (1)
    let sharder = LayerGroupPayloadSharder::new(5, 2, 2);
    assert_eq!(sharder.num_groups(), 3);
    assert_eq!(sharder.group_qualifier(0), "layers_0_1");
    assert_eq!(sharder.group_qualifier(1), "layers_2_3");
    assert_eq!(sharder.group_qualifier(2), "layers_4_4");

    // 4 bytes per layer per KV component -> total = 2 * 5 * 4 = 40 bytes
    let payload: Vec<u8> = (0u8..40u8).collect();
    // Indivisible payload length (39) must be rejected
    assert!(sharder.shard(&payload[..39]).is_none());

    let mut shards = sharder.shard(&payload).expect("valid payload");
    assert_eq!(shards.len(), 3);
    assert_eq!(shards["layers_0_1"].len(), 16);
    assert_eq!(shards["layers_2_3"].len(), 16);
    assert_eq!(shards["layers_4_4"].len(), 8);

    let reassembled = sharder.reassemble(&shards).expect("valid shards");
    assert_eq!(reassembled, payload);

    // Removing a required group shard causes reassembly to fail cleanly
    shards.shift_remove("layers_2_3");
    assert!(sharder.reassemble(&shards).is_none());
}

#[test]
fn test_plan_concat_in_place_alignment_and_offset_computation() {
    use castforge::ir::Attribute;

    // Case 1: Two distinct op-produced `tensor<4x16xf32>` slices (256 bytes each) concatenated
    // along dimension 0 -> aligned to 64 bytes, uniform stride = true, offsets = [0, 256]
    // Case 2: Unaligned `tensor<3xf32>` (12 bytes) preceding `tensor<4xf32>` -> rejected for 64B!
    let mlir = r#"
module @concat_inplace {
  func.func @main(%x: tensor<4x16xf32>, %u: tensor<3xf32>, %v: tensor<4xf32>) -> tensor<8x16xf32> {
    %a = stablehlo.abs %x : tensor<4x16xf32>
    %b = stablehlo.negate %x : tensor<4x16xf32>
    %c_ok = stablehlo.concatenate %a, %b : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
    %u1 = stablehlo.abs %u : tensor<3xf32>
    %v1 = stablehlo.abs %v : tensor<4xf32>
    %c_bad = stablehlo.concatenate %u1, %v1 : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
    return %c_ok : tensor<8x16xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let planned = plan_concat_in_place(&mut module, 64);
    assert_eq!(planned, 1);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;
    let c_ok_op = &module.ops[ops[2]];
    assert_eq!(
        c_ok_op.attributes.get("castforge.concat_in_place"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        c_ok_op.attributes.get("castforge.concat_uniform_stride"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        c_ok_op
            .attributes
            .get("castforge.concat_slice_byte_offsets"),
        Some(&Attribute::DenseIntArray(vec![0, 256]))
    );

    // Unaligned concat (`%u1` is 12 bytes, not a multiple of 64) must NOT be marked in-place!
    let c_unaligned_op = &module.ops[ops[5]];
    assert!(
        !c_unaligned_op
            .attributes
            .contains_key("castforge.concat_in_place")
    );
}

#[test]
fn test_legalize_rowwise_reduction_launch_cost_model_and_attributes() {
    use castforge::ir::Attribute;

    assert_eq!(ROWWISE_CHEAP_OP_COST, 1);
    assert_eq!(ROWWISE_EXPENSIVE_OP_COST, 16);

    let mlir = r#"
module @rowwise_cost {
  func.func @main(%x: tensor<4x128xf16>, %i: tensor<f16>) -> tensor<4x128xf16> {
    %a = stablehlo.add %x, %x : tensor<4x128xf16>
    %r = stablehlo.reduce %a, %i : (tensor<4x128xf16>, tensor<f16>) -> tensor<4xf16>
    %sm = "castforge.softmax"(%x) {axis = 1 : i64} : (tensor<4x128xf16>) -> tensor<4x128xf16>
    return %sm : tensor<4x128xf16>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    // Threshold = 16:
    // - `%r` (`stablehlo.reduce` fed by cheap `stablehlo.add`): cost = 1 + 1 = 2 (< 16),
    //   `num_phases = 1`, `accum_bits = 32` (promoted from `f16`), `cols = Some(128)`
    // - `%sm` (`castforge.softmax`): cost = 2*16 + 3 = 35 (>= 16 -> expensive!),
    //   `num_phases = 2`, `cols = Some(128)`
    let plans = legalize_rowwise_reduction_launch(&mut module, 16);
    assert_eq!(plans.len(), 2);

    assert_eq!(plans[0].op_name, "stablehlo.reduce");
    assert_eq!(plans[0].element_cost, 2);
    assert!(!plans[0].computationally_expensive);
    assert_eq!(plans[0].num_phases, 1);
    assert_eq!(plans[0].accum_bits, 32);
    assert_eq!(plans[0].cols, Some(128));

    assert_eq!(plans[1].op_name, "castforge.softmax");
    assert_eq!(plans[1].element_cost, 35);
    assert!(plans[1].computationally_expensive);
    assert_eq!(plans[1].num_phases, 2);
    assert_eq!(plans[1].accum_bits, 32);
    assert_eq!(plans[1].cols, Some(128));

    let sm_op = &module.ops[plans[1].op_id];
    assert_eq!(
        sm_op.attributes.get("castforge.rowwise_expensive"),
        Some(&Attribute::Bool(true))
    );
    assert_eq!(
        sm_op.attributes.get("castforge.rowwise_cols"),
        Some(&Attribute::Int(128))
    );
}

#[test]
fn test_reversible_kv_storage_key_and_fs_filename_codec() {
    let unsalted = KvStorageKeySpec {
        model_name: "meta-llama/Llama-3-8B".to_string(),
        kv_rank: 3,
        object_group_id: 1,
        chunk_hash: 0x0123_4567_89ab_cdef,
        cache_salt: String::new(),
    };
    let wire = encode_kv_storage_key(&unsalted);
    // Unsalted key omits trailing `@salt`
    assert_eq!(wire, "meta-llama/Llama-3-8B@00000003@1@0123456789abcdef");
    assert_eq!(decode_kv_storage_key(&wire), Some(unsalted.clone()));

    let fs_name = encode_kv_fs_filename(&unsalted);
    assert_eq!(
        fs_name,
        "meta-llama-SEP-Llama-3-8B@0x00000003@1@0123456789abcdef.data"
    );
    assert_eq!(decode_kv_fs_filename(&fs_name), Some(unsalted));

    let salted = KvStorageKeySpec {
        model_name: "deepseek-ai/DeepSeek-V3".to_string(),
        kv_rank: 12,
        object_group_id: 2,
        chunk_hash: 0xfedc_ba98_7654_3210,
        cache_salt: "tenant-prod".to_string(),
    };
    let salted_fs = encode_kv_fs_filename(&salted);
    assert!(salted_fs.ends_with("@tenant-prod.data"));
    assert_eq!(decode_kv_fs_filename(&salted_fs), Some(salted));
    assert!(decode_kv_fs_filename("missing_extension").is_none());
}

#[test]
fn test_l2_fault_inject_policy_tail_ratios_and_trim_mask() {
    let keys: Vec<KvStorageKeySpec> = (0..5)
        .map(|i| KvStorageKeySpec {
            model_name: "llama".to_string(),
            kv_rank: 0,
            object_group_id: 0,
            chunk_hash: 1000 + i,
            cache_salt: String::new(),
        })
        .collect();

    // `gap_indices = [1]` (2nd chunk) + `gap_tail_ratios = [0.5, 0.0]`
    // For n = 5:
    // - ratio 0.5 -> round((1 - 0.5) * 4) = 2 (middle chunk)
    // - ratio 0.0 -> round((1 - 0.0) * 4) = 4 (last chunk)
    let policy = L2FaultInjectPolicy::new(L2FaultInjectConfig {
        rate: 0.0,
        seed: 42,
        gap_indices: vec![1],
        gap_tail_ratios: vec![0.5, 0.0],
    });
    assert_eq!(policy.drop_positions(&keys), vec![1, 2, 4]);

    let lookup_bitmap = vec![true, true, true, true, true];
    let (load_bitmap, trim_mask) = policy.fault_load_bitmap(&keys, &lookup_bitmap);
    assert_eq!(load_bitmap, vec![true, false, false, true, false]);
    assert_eq!(trim_mask, vec![false, true, true, false, true]);
}

#[test]
fn test_reuse_dead_loop_buffers_by_use_count_and_disjoint_liveness() {
    use castforge::ir::loops::{LoopKernel, LoopModule, LoopStmt};

    // `buf_a` (len 16) is used in statements 0 and 1 (2 uses).
    // `buf_b` (len 16) is used in statements 2 and 3 (2 uses, strictly after `buf_a`).
    // `buf_c` (len 32) has different length -> not merged with `buf_a`.
    let mut loop_mod = LoopModule {
        name: "reuse_test".to_string(),
        kernels: vec![LoopKernel {
            name: "main".to_string(),
            params: vec![("x".to_string(), 16)],
            output: ("out".to_string(), 16),
            body: vec![
                LoopStmt::Alloc {
                    name: "buf_a".to_string(),
                    len: 16,
                },
                LoopStmt::Alloc {
                    name: "buf_b".to_string(),
                    len: 16,
                },
                LoopStmt::Alloc {
                    name: "buf_c".to_string(),
                    len: 32,
                },
                LoopStmt::Assign {
                    dst: "buf_a".to_string(),
                    index: "0".to_string(),
                    expr: "x[0] + 1.0".to_string(),
                },
                LoopStmt::Assign {
                    dst: "buf_c".to_string(),
                    index: "0".to_string(),
                    expr: "buf_a[0] * 2.0".to_string(),
                },
                LoopStmt::Assign {
                    dst: "buf_b".to_string(),
                    index: "0".to_string(),
                    expr: "buf_c[0] + 3.0".to_string(),
                },
                LoopStmt::Assign {
                    dst: "out".to_string(),
                    index: "0".to_string(),
                    expr: "buf_b[0]".to_string(),
                },
            ],
        }],
    };

    let reused = reuse_dead_loop_buffers(&mut loop_mod);
    assert_eq!(reused, 1);

    let printed = format!("{loop_mod}");
    assert!(printed.contains("alloc buf_a: f32[16]"));
    assert!(!printed.contains("alloc buf_b"));
    assert!(printed.contains("out[0] = buf_a[0]"));
}

#[test]
fn test_hoist_exec_invariant_ops_trivial_filter_and_slot_assignment() {
    use castforge::ir::Attribute;

    // `%x_abs` depends on runtime argument `%x` (not invariant).
    // `%c0` (`stablehlo.constant`) + `%w_trans` (`stablehlo.transpose` of `%c0`) form a non-trivial
    // execution-invariant subgraph placed AFTER `%x_abs` -> should be lifted & hoisted to top!
    // `%i_trivial` (`stablehlo.iota`) only feeds runtime `%out` -> trivial op, NOT lifted!
    let mlir = r#"
module @exec_invariant {
  func.func @main(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %x_abs = stablehlo.abs %x : tensor<4x4xf32>
    %c0 = stablehlo.constant dense<2.0> : tensor<4x4xf32>
    %w_t = stablehlo.transpose %c0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %i_trivial = stablehlo.iota {iota_dimension = 0 : i64} : tensor<4x4xf32>
    %sum = stablehlo.add %x_abs, %w_t : tensor<4x4xf32>
    %out = stablehlo.add %sum, %i_trivial : tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let cached = hoist_exec_invariant_ops(&mut module);
    assert_eq!(cached, 1);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;

    // `%c0` and `%w_trans` are hoisted to positions 0 and 1 before `%x_abs`!
    assert_eq!(module.ops[ops[0]].name.to_string(), "stablehlo.constant");
    assert_eq!(module.ops[ops[1]].name.to_string(), "stablehlo.transpose");
    assert_eq!(
        module.ops[ops[1]]
            .attributes
            .get("castforge.exec_invariant_slot"),
        Some(&Attribute::Int(0))
    );
    assert_eq!(module.ops[ops[2]].name.to_string(), "stablehlo.abs");

    // Standalone trivial `%i_trivial` (`stablehlo.iota`) is NOT marked `exec_invariant_init`
    let iota_op = &module.ops[ops[3]];
    assert_eq!(iota_op.name.to_string(), "stablehlo.iota");
    assert!(
        !iota_op
            .attributes
            .contains_key("castforge.exec_invariant_init")
    );
}

#[test]
fn test_hoist_symbolic_dim_expressions_deduplication_and_aliasing() {
    use castforge::ir::{Attribute, Type};

    let mlir = r#"
module @param_expr {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.abs %x : tensor<4x8xf32>
    %1 = stablehlo.negate %0 : tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let arg_val = module.blocks[func_block].arguments[0];
    let op0_res = module.ops[module.blocks[func_block].ops[0]].results[0];
    let op1_res = module.ops[module.blocks[func_block].ops[1]].results[0];

    // Inject compound symbolic dimension expressions: `"batch*seq"` (used twice) and `"hidden+64"`
    // plus a simple identifier `"batch"` (which should NOT be hoisted as a complex expression).
    let complex_ty = Type::Tensor(TensorType {
        shape: vec![
            Dim::Symbolic("batch".to_string()),
            Dim::Symbolic("batch*seq".to_string()),
            Dim::Symbolic("hidden+64".to_string()),
        ],
        dtype: DType::F32,
        encoding: None,
    });
    module.values[arg_val].ty = complex_ty.clone();
    module.values[op0_res].ty = complex_ty.clone();
    module.values[op1_res].ty = complex_ty;

    let hoisted = hoist_symbolic_dim_expressions(&mut module);
    // 2 unique compound expressions (`"batch*seq"` -> `param_expr_0`, `"hidden+64"` -> `1`)
    assert_eq!(hoisted, 2);

    let Type::Tensor(ref rewritten_tt) = module.values[op0_res].ty else {
        panic!("expected tensor type");
    };
    assert_eq!(
        rewritten_tt.shape,
        vec![
            Dim::Symbolic("batch".to_string()),
            Dim::Symbolic("param_expr_0".to_string()),
            Dim::Symbolic("param_expr_1".to_string()),
        ]
    );

    let Some(Attribute::Dictionary(decls)) = module.ops[func_id]
        .attributes
        .get("castforge.param_expr_decls")
    else {
        panic!("expected castforge.param_expr_decls dictionary attribute");
    };
    assert_eq!(
        decls.get("param_expr_0"),
        Some(&Attribute::String("batch*seq".to_string()))
    );
    assert_eq!(
        decls.get("param_expr_1"),
        Some(&Attribute::String("hidden+64".to_string()))
    );
}

#[test]
fn test_subchunk_swa_block_id_downsampling_and_skip_recalculation() {
    // Example from LMCache `downsample_and_stage_block_ids`:
    // block_size = 32, chunk_tokens = 128 (4 blocks/chunk), 2 chunks (8 blocks total):
    // - Full attention (subchunk_sw = 128): keeps all [1..=8]
    // - SWA (subchunk_sw = 64 -> 2 blocks/chunk): keeps trailing 2 blocks -> [13, 14, 17, 18]
    let full_blocks = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let swa_blocks = vec![11, 12, 13, 14, 15, 16, 17, 18];

    assert_eq!(
        downsample_subchunk_swa_block_ids(&full_blocks, 128, 128, 32),
        Some(vec![1, 2, 3, 4, 5, 6, 7, 8])
    );
    assert_eq!(
        downsample_subchunk_swa_block_ids(&swa_blocks, 128, 64, 32),
        Some(vec![13, 14, 17, 18])
    );
    // Non-multiple of blocks_per_chunk (4) is rejected
    assert!(downsample_subchunk_swa_block_ids(&[11, 12, 13], 128, 64, 32).is_none());

    // Skip recalculation when blocks_per_chunk = 4, blocks_per_window = 2:
    // - skipping 5 original blocks = 1 full chunk (2 window blocks) + 1 tail block (which falls
    //   inside the 2 already-dropped blocks of chunk 1 -> 0 extra tail skip) = 2
    // - skipping 7 original blocks = 1 full chunk (2 window blocks) + (3 - 2 = 1 tail skip) = 3
    assert_eq!(recalculate_subchunk_blocks_to_skip(4, 2, 5), 2);
    assert_eq!(recalculate_subchunk_blocks_to_skip(4, 2, 7), 3);
}

#[test]
fn test_token_wire_codec_be_and_zero_unpack_chunk_hashing() {
    assert_eq!(TOKEN_WIRE_STRIDE, 4);
    let tokens = vec![101u32, 2054, 3099, 4096, 5120, 6144];
    let packed = pack_token_ids_be(&tokens);
    assert_eq!(packed.len(), 6 * 4);
    assert_eq!(num_packed_tokens(&packed), 6);
    assert_eq!(unpack_token_ids_be(&packed), Some(tokens));

    // Ragged byte slice not divisible by `TOKEN_WIRE_STRIDE` (4) is rejected
    assert!(unpack_token_ids_be(&packed[..23]).is_none());

    // Zero-unpack chained hashing over 2 chunks of 3 tokens each
    let h_all = hash_packed_token_blocks_be(&packed, 3, 0);
    assert_eq!(h_all.len(), 2);
    assert_ne!(h_all[0], h_all[1]);

    // Modifying a token in chunk 0 cascades to chunk 1's chained hash!
    let mut modified = packed.clone();
    modified[3] ^= 1;
    let h_mod = hash_packed_token_blocks_be(&modified, 3, 0);
    assert_ne!(h_all[0], h_mod[0]);
    assert_ne!(h_all[1], h_mod[1]);
}

#[test]
fn test_dynamic_memory_pool_lcm_alignment_can_share_and_bulk_slice() {
    // Provide 3 allocations out of topological order:
    // - `buf_late` (first_use = 5..8, size = 256, align = 128)
    // - `buf_a` (first_use = 1..3, size = 256, align = 64)
    // - `buf_b` (first_use = 2..4, size = 128, align = 96) -> overlaps `buf_a`!
    // Notice `buf_late` (5..8) executes strictly after both `buf_a` (1..3) and `buf_b` (2..4),
    // so `buf_late` can share offset 0 with `buf_a`!
    let allocs = vec![
        DynamicAllocRequest {
            name: "buf_late".to_string(),
            size_bytes: 256,
            alignment_bytes: 128,
            first_use_idx: 5,
            last_use_idx: 8,
            location: Location::Named("loc_late".to_string()),
        },
        DynamicAllocRequest {
            name: "buf_a".to_string(),
            size_bytes: 256,
            alignment_bytes: 64,
            first_use_idx: 1,
            last_use_idx: 3,
            location: Location::Named("loc_a".to_string()),
        },
        DynamicAllocRequest {
            name: "buf_b".to_string(),
            size_bytes: 128,
            alignment_bytes: 96,
            first_use_idx: 2,
            last_use_idx: 4,
            location: Location::Named("loc_b".to_string()),
        },
    ];

    let plan = plan_dynamic_memory_pool(&allocs);
    // Ordered by `first_use_idx`: ["buf_a", "buf_b", "buf_late"]
    assert_eq!(plan.ordered_names, vec!["buf_a", "buf_b", "buf_late"]);
    // LCM(64, 96, 128) = 384
    assert_eq!(plan.pool_alignment_bytes, 384);
    // `buf_a` at 0 (size 256); `buf_b` conflicts with `buf_a` and needs 96-byte alignment -> 288;
    // `buf_late` shares with both `buf_a` and `buf_b` -> placed at offset 0!
    assert_eq!(plan.bulk_slice_offsets, vec![0, 288, 0]);
    assert_eq!(plan.high_watermark_bytes, 288 + 128);
    assert_eq!(
        plan.fused_location,
        Location::Named("loc_a+loc_b+loc_late".to_string())
    );
}

#[test]
fn test_merge_adjacent_parallel_loops_forwards_yield_and_rejects_cross_iter_hazard() {
    use castforge::ir::loops::{LoopKernel, LoopModule, LoopStmt};

    // Kernel 0: Producer writes `tmp[i] = x[i] + 1.0`, consumer reads `tmp[i] * 2.0` -> merged!
    // `tmp` is used only by the consumer -> `tmp`'s `Alloc` and store are dropped!
    // Kernel 1: Consumer reads `tmp[i + 1]` (cross-iteration stencil hazard) -> NOT merged!
    let mut loop_mod = LoopModule {
        name: "merge_parallels".to_string(),
        kernels: vec![
            LoopKernel {
                name: "pointwise_ok".to_string(),
                params: vec![("x".to_string(), 16)],
                output: ("out".to_string(), 16),
                body: vec![
                    LoopStmt::Alloc {
                        name: "tmp".to_string(),
                        len: 16,
                    },
                    LoopStmt::For {
                        var: "i".to_string(),
                        lower: 0,
                        upper: 16,
                        step: 1,
                        parallel: true,
                        vectorized: true,
                        unroll: 4,
                        body: vec![LoopStmt::Assign {
                            dst: "tmp".to_string(),
                            index: "i".to_string(),
                            expr: "x[i] + 1.0".to_string(),
                        }],
                    },
                    LoopStmt::For {
                        var: "i".to_string(),
                        lower: 0,
                        upper: 16,
                        step: 1,
                        parallel: true,
                        vectorized: true,
                        unroll: 4,
                        body: vec![LoopStmt::Assign {
                            dst: "out".to_string(),
                            index: "i".to_string(),
                            expr: "tmp[i] * 2.0".to_string(),
                        }],
                    },
                ],
            },
            LoopKernel {
                name: "cross_iter_hazard".to_string(),
                params: vec![("x".to_string(), 16)],
                output: ("out".to_string(), 16),
                body: vec![
                    LoopStmt::Alloc {
                        name: "tmp".to_string(),
                        len: 16,
                    },
                    LoopStmt::For {
                        var: "i".to_string(),
                        lower: 0,
                        upper: 16,
                        step: 1,
                        parallel: true,
                        vectorized: true,
                        unroll: 1,
                        body: vec![LoopStmt::Assign {
                            dst: "tmp".to_string(),
                            index: "i".to_string(),
                            expr: "x[i] + 1.0".to_string(),
                        }],
                    },
                    LoopStmt::For {
                        var: "i".to_string(),
                        lower: 0,
                        upper: 16,
                        step: 1,
                        parallel: true,
                        vectorized: false,
                        unroll: 1,
                        body: vec![LoopStmt::Assign {
                            dst: "out".to_string(),
                            index: "i".to_string(),
                            expr: "tmp[i + 1] * 2.0".to_string(),
                        }],
                    },
                ],
            },
        ],
    };

    let merged = merge_adjacent_parallel_loops(&mut loop_mod);
    assert_eq!(merged, 1);

    // Kernel 0 now has a single `For` loop (and `alloc tmp` was eliminated!)
    assert_eq!(loop_mod.kernels[0].body.len(), 1);
    let printed_k0 = format!("{}", loop_mod);
    assert!(printed_k0.contains("out[i] = (x[i] + 1.0) * 2.0"));

    // Kernel 1 (cross-iteration hazard `tmp[i + 1]`) remains unmerged (Alloc + 2 For loops)
    assert_eq!(loop_mod.kernels[1].body.len(), 3);
}

#[test]
fn test_await_hanging_effect_chains_inserts_barrier_before_terminator() {
    use castforge::ir::Attribute;

    // `%offload` (`castforge.kv_offload`) is an asynchronous effectful op whose result is unused
    // (`use_empty()`), while `%y` (`stablehlo.abs`) is returned by `func.return`.
    let mlir = r#"
module @fixup_overhangs {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %offload = "castforge.kv_offload"(%x) {slot = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
    %y = stablehlo.abs %x : tensor<4xf32>
    return %y : tensor<4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let awaited = await_hanging_effect_chains(&mut module);
    assert_eq!(awaited, 1);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;
    assert_eq!(ops.len(), 4);
    assert_eq!(module.ops[ops[2]].name.to_string(), "castforge.chain_await");
    assert_eq!(
        module.ops[ops[3]]
            .attributes
            .get("castforge.awaited_hanging_chains"),
        Some(&Attribute::Int(1))
    );

    // Running `await_hanging_effect_chains` a second time is idempotent (`%offload` is now used
    // by `castforge.chain_await`!)
    assert_eq!(await_hanging_effect_chains(&mut module), 0);
}

#[test]
fn test_eliminate_implied_dependencies_transitive_reduction_and_cycle_safety() {
    // Node 0: []
    // Node 1: [0]
    // Node 2: [1, 0] -> edge `2 -> 0` is implied transitively via `2 -> 1 -> 0`!
    // Node 3: [2, 1, 0] -> edges `3 -> 1` and `3 -> 0` are implied via `3 -> 2 -> 1 -> 0`!
    let dag = vec![vec![], vec![0], vec![1, 0], vec![2, 1, 0]];
    let (reduced, eliminated) = eliminate_implied_dependencies(&dag);
    assert_eq!(eliminated, 3);
    assert_eq!(reduced, vec![vec![], vec![0], vec![1], vec![2]]);

    // Malformed cycle (`0 -> 1 -> 0`) must terminate cleanly without hanging
    let cyclic = vec![vec![1], vec![0]];
    let (_, cyc_elim) = eliminate_implied_dependencies(&cyclic);
    assert_eq!(cyc_elim, 2);
}

#[test]
fn test_coalesce_partitioning_slices_to_split_contiguity_and_ordering() {
    use castforge::ir::Attribute;

    // `%x` has shape `tensor<4x16xf32>`. Three out-of-order slices along dim 1:
    // - `%s_mid`: [0..4, 4..12] (size 8, start 4)
    // - `%s_hi`:  [0..4, 12..16] (size 4, start 12)
    // - `%s_lo`:  [0..4, 0..4] (size 4, start 0)
    // Together they partition dim 1 (`4 + 8 + 4 = 16`)!
    let mlir = r#"
module @slice_to_split {
  func.func @main(%x: tensor<4x16xf32>) -> tensor<4x4xf32> {
    %s_mid = stablehlo.slice %x
      {start_indices = [0, 4], limit_indices = [4, 12], strides = [1, 1]}
      : (tensor<4x16xf32>) -> tensor<4x8xf32>
    %s_hi = stablehlo.slice %x
      {start_indices = [0, 12], limit_indices = [4, 16], strides = [1, 1]}
      : (tensor<4x16xf32>) -> tensor<4x4xf32>
    %s_lo = stablehlo.slice %x
      {start_indices = [0, 0], limit_indices = [4, 4], strides = [1, 1]}
      : (tensor<4x16xf32>) -> tensor<4x4xf32>
    %out = stablehlo.add %s_lo, %s_hi : tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let groups = coalesce_partitioning_slices_to_split(&mut module);
    assert_eq!(groups, 1);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;

    // `%s_lo` (`ops[2]`) has `start = 0` -> sorted split index 0, sizes `[4, 8, 4]`
    // `%s_mid` (`ops[0]`) has `start = 4` -> sorted split index 1
    // `%s_hi` (`ops[1]`) has `start = 12` -> sorted split index 2
    assert_eq!(
        module.ops[ops[2]]
            .attributes
            .get("castforge.coalesced_split_index"),
        Some(&Attribute::Int(0))
    );
    assert_eq!(
        module.ops[ops[0]]
            .attributes
            .get("castforge.coalesced_split_index"),
        Some(&Attribute::Int(1))
    );
    assert_eq!(
        module.ops[ops[1]]
            .attributes
            .get("castforge.coalesced_split_index"),
        Some(&Attribute::Int(2))
    );
    assert_eq!(
        module.ops[ops[2]]
            .attributes
            .get("castforge.coalesced_split_sizes"),
        Some(&Attribute::DenseIntArray(vec![4, 8, 4]))
    );
}

#[test]
fn test_find_squeeze_unsqueeze_axis_and_reshape_annotation() {
    use castforge::ir::Attribute;

    // Symbolic + static dims example from Modular `ShapeOpSinking.cpp`:
    // `[D1, 2, 1, D3]` -> `[D1, 2, D3]` => `Some(2)`
    let in_dims = vec![
        Dim::Symbolic("D1".to_string()),
        Dim::Static(2),
        Dim::Static(1),
        Dim::Symbolic("D3".to_string()),
    ];
    let out_dims = vec![
        Dim::Symbolic("D1".to_string()),
        Dim::Static(2),
        Dim::Symbolic("D3".to_string()),
    ];
    assert_eq!(find_squeeze_axis(&in_dims, &out_dims), Some(2));
    assert_eq!(find_unsqueeze_axis(&out_dims, &in_dims), Some(2));

    // Mismatched trailing symbolic dim (`[D1, 2, 1, D4]` -> `[D1, 2, D3]`) => `None`
    let bad_in = vec![
        Dim::Symbolic("D1".to_string()),
        Dim::Static(2),
        Dim::Static(1),
        Dim::Symbolic("D4".to_string()),
    ];
    assert_eq!(find_squeeze_axis(&bad_in, &out_dims), None);

    let mlir = r#"
module @squeeze_unsqueeze {
  func.func @main(%x: tensor<4x1x8xf32>) -> tensor<1x4x8xf32> {
    %sq = stablehlo.reshape %x : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
    %unsq = stablehlo.reshape %sq : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
    return %unsq : tensor<1x4x8xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let count = annotate_squeeze_unsqueeze_reshapes(&mut module);
    assert_eq!(count, 2);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;
    assert_eq!(
        module.ops[ops[0]].attributes.get("castforge.squeeze_axis"),
        Some(&Attribute::Int(1))
    );
    assert_eq!(
        module.ops[ops[1]]
            .attributes
            .get("castforge.unsqueeze_axis"),
        Some(&Attribute::Int(0))
    );
}

#[test]
fn test_p2p_peer_discovery_consecutive_misses_and_group_lookup_keys() {
    let mut table = P2PPeerDiscoveryTable::new("node-self", P2P_DEFAULT_MAX_MISSES);
    let peer_a = P2PPeerSpec {
        instance_id: "node-a".to_string(),
        ip: "10.0.0.2".to_string(),
        p2p_advertised_url: "tcp://10.0.0.2:5555".to_string(),
        mq_port: 5556,
    };
    let peer_self = P2PPeerSpec {
        instance_id: "node-self".to_string(),
        ip: "10.0.0.1".to_string(),
        p2p_advertised_url: "tcp://10.0.0.1:5555".to_string(),
        mq_port: 5556,
    };

    // Poll 1: discovers `node-a` (ignores `node-self`) -> creates adapter 1
    assert_eq!(
        table.reconcile_peers(&[peer_self, peer_a.clone()]),
        (vec![1], vec![])
    );
    assert_eq!(table.active_peer_count(), 1);

    // Polls 2 & 3: `node-a` missing -> 2 consecutive misses (< 3), adapter still retained!
    assert_eq!(table.reconcile_peers(&[]), (vec![], vec![]));
    assert_eq!(table.reconcile_peers(&[]), (vec![], vec![]));
    assert_eq!(table.active_peer_count(), 1);

    // Poll 4: 3rd consecutive miss -> adapter 1 is evicted!
    assert_eq!(table.reconcile_peers(&[]), (vec![], vec![1]));
    assert_eq!(table.active_peer_count(), 0);

    // Test `group_p2p_lookup_keys`: equal counts across registered groups [0, 1] succeed,
    // unequal counts or unregistered group IDs return empty!
    let k0 = KvStorageKeySpec {
        model_name: "m".to_string(),
        kv_rank: 0,
        object_group_id: 0,
        chunk_hash: 1,
        cache_salt: String::new(),
    };
    let k1 = KvStorageKeySpec {
        object_group_id: 1,
        chunk_hash: 2,
        ..k0.clone()
    };
    let grouped = group_p2p_lookup_keys(&[k0.clone(), k1.clone()], &[0, 1]);
    assert_eq!(grouped.len(), 2);
    // Unequal group counts (2 keys in group 0, 1 key in group 1) -> treated as all misses
    assert!(group_p2p_lookup_keys(&[k0.clone(), k0, k1], &[0, 1]).is_empty());
}

#[test]
fn test_encode_wide_column_row_key_hash_prefix_and_layer_group_suffix() {
    let key = KvStorageKeySpec {
        model_name: "meta-llama/Llama-3-70B".to_string(),
        kv_rank: 2,
        object_group_id: 1,
        chunk_hash: 0xabcd_ef01_2345_6789,
        cache_salt: "tenant-x".to_string(),
    };
    let row_key = encode_wide_column_row_key(&key, 8, Some(16));
    assert_eq!(
        row_key,
        "abcd@meta-llama/Llama-3-70B@lg8@00000002@1@abcdef0123456789@tenant-x@layer_16"
    );

    let unsalted = KvStorageKeySpec {
        cache_salt: String::new(),
        ..key
    };
    assert_eq!(
        encode_wide_column_row_key(&unsalted, 0, None),
        "abcd@meta-llama/Llama-3-70B@00000002@1@abcdef0123456789"
    );
}

#[test]
fn test_blend_token_range_matcher_sliding_window_and_coverage() {
    let mut matcher = BlendTokenRangeMatcher::new(4, 10, true);
    let stored_tokens = vec![10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40];
    let chunk_hashes = vec![0x1111, 0x2222, 0x3333];
    // Chunk 2 (`[10, 20, 30, 40]`) duplicates chunk 0's content, so `dedup_content = true`
    // indexes only 2 chunks (`0x1111` at offset 0 and `0x2222` at offset 4).
    assert_eq!(
        matcher.on_new_token_hashes(&stored_tokens, &chunk_hashes, 0),
        2
    );

    // Query contains chunk 1 at offset 2 and chunk 0 at offset 7:
    let query = vec![99, 98, 50, 60, 70, 80, 97, 10, 20, 30, 40, 96];
    let matches = matcher.match_sub_sequence(&query);
    assert_eq!(
        matches,
        vec![
            BlendMatchSpan {
                old_st: 4,
                old_ed: 8,
                cur_st: 2,
                cur_ed: 6,
                chunk_hash: 0x2222,
            },
            BlendMatchSpan {
                old_st: 0,
                old_ed: 4,
                cur_st: 7,
                cur_ed: 11,
                chunk_hash: 0x1111,
            },
        ]
    );
    assert_eq!(unique_blend_token_coverage(&matches), 8);

    // Overlapping spans `[2, 6)` and `[4, 8)` merge to coverage 6 (not 8):
    let overlapping = vec![
        BlendMatchSpan {
            old_st: 0,
            old_ed: 4,
            cur_st: 2,
            cur_ed: 6,
            chunk_hash: 1,
        },
        BlendMatchSpan {
            old_st: 4,
            old_ed: 8,
            cur_st: 4,
            cur_ed: 8,
            chunk_hash: 2,
        },
    ];
    assert_eq!(unique_blend_token_coverage(&overlapping), 6);

    // Evicting chunk `0x2222` removes it from subsequent probes:
    matcher.remove_chunks(&[0x2222]);
    let after_evict = matcher.match_sub_sequence(&query);
    assert_eq!(after_evict.len(), 1);
    assert_eq!(after_evict[0].chunk_hash, 0x1111);
}

#[test]
fn test_plan_cacheblend_token_reorder_glued_pieces_and_pinned_tail() {
    // System prefix (4 tokens): `[1, 2, 3, 4]`
    // Document A (16 tokens): `[100..116]`
    // Glue separator (2 tokens): `[999, 998]`
    // Document B (16 tokens): `[200..216]`
    // Tail question (16 tokens = `BLEND_REORDER_LMIN`): `[500..516]`
    let prefix: Vec<u32> = vec![1, 2, 3, 4];
    let doc_a: Vec<u32> = (100..116).collect();
    let sep: Vec<u32> = vec![999, 998];
    let doc_b: Vec<u32> = (200..216).collect();
    let tail_q: Vec<u32> = (500..500 + BLEND_REORDER_LMIN as u32).collect();

    // Cached candidate H has `[prefix, doc_a, sep, doc_b]`:
    let mut cand_h = Vec::new();
    cand_h.extend_from_slice(&prefix);
    cand_h.extend_from_slice(&doc_a);
    cand_h.extend_from_slice(&sep);
    cand_h.extend_from_slice(&doc_b);

    // Incoming prompt P has `[prefix, doc_b, sep, doc_a, tail_q]` (documents swapped!):
    let mut prompt_p = Vec::new();
    prompt_p.extend_from_slice(&prefix);
    prompt_p.extend_from_slice(&doc_b);
    prompt_p.extend_from_slice(&sep);
    prompt_p.extend_from_slice(&doc_a);
    prompt_p.extend_from_slice(&tail_q);

    // Reorder P with `chunk_size = 8`, `baseline_chunks = 1`, `keep_prefix = 4`:
    let perm = plan_cacheblend_token_reorder(&prompt_p, &[cand_h.clone()], 8, 1, 4)
        .expect("should find a valid token reorder plan");
    assert_eq!(perm.len(), prompt_p.len());

    let reordered: Vec<u32> = perm.iter().map(|&idx| prompt_p[idx]).collect();
    // Reordered prompt starts with exact copy of `cand_h` (`[prefix, doc_a, sep, doc_b]` = 38
    // tokens) and keeps `tail_q` pinned at the very end!
    assert_eq!(&reordered[..cand_h.len()], cand_h.as_slice());
    assert_eq!(
        &reordered[reordered.len() - BLEND_REORDER_LMIN..],
        tail_q.as_slice()
    );
}

#[test]
fn test_coordinator_event_gate_incarnation_fencing_dedup_and_gaps() {
    let mut gate = CoordinatorEventGate::new();

    // 1. First batch (`seq = 1`, `incarnation = 10`) is admitted cleanly:
    let b1 = CacheEventBatchSpec {
        instance_id: "worker-1".to_string(),
        incarnation: 10,
        seq: 1,
        dropped_events: 0,
        tier: CacheEventTier::L1,
    };
    assert_eq!(gate.ingest(&b1), EventIngestOutcome::Admitted);
    // Re-offering `seq = 1` is rejected as `Duplicate`:
    assert_eq!(gate.ingest(&b1), EventIngestOutcome::Duplicate);
    // Older incarnation (`9 < 10`) is rejected as `StaleIncarnation`:
    let stale = CacheEventBatchSpec {
        incarnation: 9,
        seq: 2,
        ..b1.clone()
    };
    assert_eq!(gate.ingest(&stale), EventIngestOutcome::StaleIncarnation);

    // 2. Jumping to `seq = 4` (skipping 2 and 3) with `dropped_events = 5` detects gap & loss:
    let b4 = CacheEventBatchSpec {
        seq: 4,
        dropped_events: 5,
        ..b1.clone()
    };
    assert_eq!(gate.ingest(&b4), EventIngestOutcome::Admitted);
    let c = gate.cursor("worker-1").expect("cursor exists");
    assert!(c.gap_detected);
    assert_eq!(c.missing_batches, 2);
    assert_eq!(c.events_dropped, 5);

    // 3. Newer incarnation (`11 > 10`) fences `worker-1` and resets its cursor:
    let b_restart = CacheEventBatchSpec {
        incarnation: 11,
        seq: 1,
        dropped_events: 0,
        ..b1.clone()
    };
    assert_eq!(gate.ingest(&b_restart), EventIngestOutcome::Admitted);
    assert_eq!(gate.fenced_instances(), &["worker-1".to_string()]);
    assert!(!gate.cursor("worker-1").unwrap().gap_detected);

    // 4. `drop_instance("worker-1")` fences L1 batches at incarnation 11, while L2 passes through:
    gate.drop_instance("worker-1");
    let b_l1_after_drop = CacheEventBatchSpec {
        incarnation: 11,
        seq: 2,
        tier: CacheEventTier::L1,
        ..b1.clone()
    };
    assert_eq!(
        gate.ingest(&b_l1_after_drop),
        EventIngestOutcome::StaleIncarnation
    );
    let b_l2_after_drop = CacheEventBatchSpec {
        incarnation: 11,
        seq: 2,
        tier: CacheEventTier::L2,
        ..b1
    };
    assert_eq!(gate.ingest(&b_l2_after_drop), EventIngestOutcome::Admitted);
}

#[test]
fn test_merge_parallel_matmuls_shared_activation_and_transpose_guard() {
    use castforge::ir::Attribute;

    let mlir = r#"
module @parallel_matmul {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<8x32xf32>) {
    %c1 = stablehlo.dot %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %c2 = stablehlo.dot %a, %c : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    %act1 = stablehlo.tanh %c1 : tensor<4x16xf32>
    %act2 = stablehlo.tanh %c2 : tensor<4x32xf32>
    return %act1 : tensor<4x16xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let merged = merge_parallel_matmuls(&mut module);
    assert_eq!(merged, 1);

    let func_id = module.functions()[0];
    let block_id = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[block_id].ops;
    // Expected: `stablehlo.concatenate`, merged `stablehlo.dot`, `mo.split`,
    // `stablehlo.tanh`, `stablehlo.tanh`, `func.return`
    assert_eq!(ops.len(), 6);
    assert!(module.ops[ops[0]].is_op("stablehlo", "concatenate"));
    assert!(module.ops[ops[1]].is_op("stablehlo", "dot"));
    assert!(module.ops[ops[2]].is_op("mo", "split"));
    assert_eq!(
        module.ops[ops[2]].attributes.get("split_sizes"),
        Some(&Attribute::DenseIntArray(vec![16, 32]))
    );

    // Verify that when all matmul users only feed directly into `func.return`, merging is skipped:
    let return_only_mlir = r#"
module @parallel_matmul_return_only {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<8x32xf32>) {
    %c1 = stablehlo.dot %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %c2 = stablehlo.dot %a, %c : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    return %c1, %c2 : tensor<4x16xf32>, tensor<4x32xf32>
  }
}
"#;
    let mut ret_module = parse_mlir(return_only_mlir).expect("valid MLIR");
    assert_eq!(merge_parallel_matmuls(&mut ret_module), 0);
}

#[test]
fn test_static_reshape_dim_mapping_and_axis_through_reshape() {
    // 1. Static prefix with trailing dynamic dim: [3, 5, 6, -1] -> [15, 3, 2, -1]
    let before = [3, 5, 6, -1];
    let after = [15, 3, 2, -1];
    let mapping = analyze_static_reshape_dim_mapping(&before, &after).expect("supported mapping");
    assert_eq!(
        mapping,
        ReshapeDimMapping {
            forward: vec![vec![0], vec![0], vec![1, 2], vec![]],
            backward: vec![vec![0, 1], vec![2], vec![2], vec![]],
        }
    );

    // Axis 0 (`3`) merges with right neighbor (`5`) into output axis 0 (`15`), which is valid:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 0), Some(0));
    // Axis 1 (`5`) merges with left neighbor (`3`) into output axis 0 (`15`), which is rejected:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 1), None);
    // Axis 2 (`6`) splits into output axes 1 (`3`) and 2 (`2`), mapping to outer split axis 1:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 2), Some(1));
    // Axis 3 (`-1`, dynamic) cannot be statically mapped:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 3), None);

    // 2. Leading dynamic dim (right-to-left scan + reversed order): [-1, 3, 5, 6] -> [-1, 15, 3, 2]
    let dyn_lead_before = [-1, 3, 5, 6];
    let dyn_lead_after = [-1, 15, 3, 2];
    let rev_mapping = analyze_static_reshape_dim_mapping(&dyn_lead_before, &dyn_lead_after)
        .expect("supported right-to-left mapping");
    assert_eq!(
        rev_mapping,
        ReshapeDimMapping {
            forward: vec![vec![], vec![1], vec![1], vec![2, 3]],
            backward: vec![vec![], vec![1, 2], vec![3], vec![3]],
        }
    );
    assert_eq!(
        map_axis_through_static_reshape(&dyn_lead_before, &dyn_lead_after, 3),
        Some(2)
    );

    // 3. Skipping leading unit (`1`) dimensions when mapping an axis:
    let unit_before = [1, 1, 4, 8];
    let unit_after = [32];
    assert_eq!(
        map_axis_through_static_reshape(&unit_before, &unit_after, 0),
        Some(0)
    );

    // 4. Rejections: zero dimensions, mismatched volumes, or static/dynamic mixing:
    assert!(analyze_static_reshape_dim_mapping(&[0, 2], &[0, 4]).is_none());
    assert!(analyze_static_reshape_dim_mapping(&[3, 5], &[16]).is_none());
    assert!(analyze_static_reshape_dim_mapping(&[3, -1, 5], &[15, -1]).is_none());
}

#[test]
fn test_coalesce_exhaustive_gathers_to_split_dominance_and_partial_rejection() {
    use castforge::ir::Attribute;

    let mlir = r#"
module @gathers_to_split {
  func.func @main(%x: tensor<3x8xf32>) {
    %g2 = mo.gather %x {axis = 0, index = 2} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %g0 = mo.gather %x {axis = 0, index = 0} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %g1 = mo.gather %x {axis = 0, index = 1} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %sum01 = stablehlo.add %g0, %g1 : tensor<1x8xf32>
    %sum012 = stablehlo.add %sum01, %g2 : tensor<1x8xf32>
    return %sum012 : tensor<1x8xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(coalesce_exhaustive_gathers_to_split(&mut module), 1);

    let func_id = module.functions()[0];
    let block_id = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[block_id].ops;
    // The 3 gathers are replaced by a single `mo.split` at the dominant position (index 0):
    assert_eq!(ops.len(), 4);
    assert!(module.ops[ops[0]].is_op("mo", "split"));
    assert_eq!(
        module.ops[ops[0]].attributes.get("split_sizes"),
        Some(&Attribute::DenseIntArray(vec![1, 1, 1]))
    );
    // Verify %g0, %g1, %g2 map to split results #0, #1, #2 in index order (not block order):
    let split_results = &module.ops[ops[0]].results;
    assert_eq!(split_results.len(), 3);
    assert_eq!(module.ops[ops[1]].operands[0], split_results[0]);
    assert_eq!(module.ops[ops[1]].operands[1], split_results[1]);
    assert_eq!(module.ops[ops[2]].operands[1], split_results[2]);

    // Partial gather coverage (missing index 2 of 3) must NOT be rewritten:
    let partial_mlir = r#"
module @partial_gathers {
  func.func @main(%x: tensor<3x8xf32>) {
    %g0 = mo.gather %x {axis = 0, index = 0} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %g1 = mo.gather %x {axis = 0, index = 1} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %sum01 = stablehlo.add %g0, %g1 : tensor<1x8xf32>
    return %sum01 : tensor<1x8xf32>
  }
}
"#;
    let mut partial_mod = parse_mlir(partial_mlir).expect("valid MLIR");
    assert_eq!(coalesce_exhaustive_gathers_to_split(&mut partial_mod), 0);
}

#[test]
fn test_recognize_composite_activations_sigmoid_silu_gelu_variants() {
    // 1. Decomposed `x * (1 / (1 + exp(-x)))` folds `1 / (1 + exp(-x))` -> `mo.sigmoid`
    //    and then `x * mo.sigmoid(x)` -> `mo.silu(x)` in a single pass!
    let silu_mlir = r#"
module @decomposed_silu {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %neg = stablehlo.negate %x : tensor<4xf32>
    %exp = stablehlo.exponential %neg : tensor<4xf32>
    %den = stablehlo.add %one, %exp : tensor<4xf32>
    %sig = stablehlo.divide %one, %den : tensor<4xf32>
    %silu = stablehlo.multiply %x, %sig : tensor<4xf32>
    return %silu : tensor<4xf32>
  }
}
"#;
    let mut silu_mod = parse_mlir(silu_mlir).expect("valid MLIR");
    assert_eq!(recognize_composite_activations(&mut silu_mod), 2);
    let func_id = silu_mod.functions()[0];
    let blk_id = silu_mod.regions[silu_mod.ops[func_id].regions[0]].blocks[0];
    let ops = &silu_mod.blocks[blk_id].ops;
    assert!(ops.iter().any(|&id| silu_mod.ops[id].is_op("mo", "silu")));
    assert!(
        !ops.iter()
            .any(|&id| silu_mod.ops[id].is_op("mo", "sigmoid"))
    );

    // 2. Quick-GELU (`x * sigmoid(1.702 * x)`) and Erf-GELU (`0.5 * x * (1 + erf(x / sqrt(2)))`):
    let gelu_mlir = r#"
module @decomposed_gelus {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c1702 = stablehlo.constant dense<1.702> : tensor<4xf32>
    %half = stablehlo.constant dense<0.5> : tensor<4xf32>
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %sqrt2 = stablehlo.constant dense<1.41421356> : tensor<4xf32>
    %scaled_q = stablehlo.multiply %c1702, %x : tensor<4xf32>
    %sig_q = stablehlo.logistic %scaled_q : tensor<4xf32>
    %qgelu = stablehlo.multiply %x, %sig_q : tensor<4xf32>
    %hx = stablehlo.multiply %half, %qgelu : tensor<4xf32>
    %div_s2 = stablehlo.divide %qgelu, %sqrt2 : tensor<4xf32>
    %erf_v = stablehlo.erf %div_s2 : tensor<4xf32>
    %one_p = stablehlo.add %one, %erf_v : tensor<4xf32>
    %gelu = stablehlo.multiply %hx, %one_p : tensor<4xf32>
    return %gelu : tensor<4xf32>
  }
}
"#;
    let mut gelu_mod = parse_mlir(gelu_mlir).expect("valid MLIR");
    assert_eq!(recognize_composite_activations(&mut gelu_mod), 2);
    let g_func = gelu_mod.functions()[0];
    let g_blk = gelu_mod.regions[gelu_mod.ops[g_func].regions[0]].blocks[0];
    let g_ops = &gelu_mod.blocks[g_blk].ops;
    assert!(
        g_ops
            .iter()
            .any(|&id| gelu_mod.ops[id].is_op("mo", "gelu_quick"))
    );
    assert!(g_ops.iter().any(|&id| gelu_mod.ops[id].is_op("mo", "gelu")));

    // 3. Tanh-GELU (`0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * (x * x * x))))`):
    let tanh_gelu_mlir = r#"
module @decomposed_gelu_tanh {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<4xf32>
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %c044 = stablehlo.constant dense<0.044715> : tensor<4xf32>
    %c797 = stablehlo.constant dense<0.79788456> : tensor<4xf32>
    %hx = stablehlo.multiply %half, %x : tensor<4xf32>
    %x2 = stablehlo.multiply %x, %x : tensor<4xf32>
    %x3 = stablehlo.multiply %x2, %x : tensor<4xf32>
    %term = stablehlo.multiply %c044, %x3 : tensor<4xf32>
    %poly = stablehlo.add %x, %term : tensor<4xf32>
    %targ = stablehlo.multiply %c797, %poly : tensor<4xf32>
    %th = stablehlo.tanh %targ : tensor<4xf32>
    %one_p = stablehlo.add %one, %th : tensor<4xf32>
    %gtanh = stablehlo.multiply %hx, %one_p : tensor<4xf32>
    return %gtanh : tensor<4xf32>
  }
}
"#;
    let mut gt_mod = parse_mlir(tanh_gelu_mlir).expect("valid MLIR");
    assert_eq!(recognize_composite_activations(&mut gt_mod), 1);
    let gt_func = gt_mod.functions()[0];
    let gt_blk = gt_mod.regions[gt_mod.ops[gt_func].regions[0]].blocks[0];
    let gt_ops = &gt_mod.blocks[gt_blk].ops;
    assert!(
        gt_ops
            .iter()
            .any(|&id| gt_mod.ops[id].is_op("mo", "gelu_tanh"))
    );
}

#[test]
fn test_sink_transpose_through_unary_binary_and_rank_increasing_broadcast() {
    use castforge::ir::Attribute;

    // Verify `unpermute_slice` inverse property (`perm = [2, 0, 1]` -> `inverse = [1, 2, 0]`):
    assert_eq!(
        unpermute_slice(&[10, 20, 30], &[2, 0, 1]),
        Some(vec![20, 30, 10])
    );

    // 1. Binary + Unary cascade:
    //    `%t1 = transpose(%x, [1, 0])`, `%t2 = transpose(%y, [1, 0])`
    //    `%sum = add(%t1, %t2)` -> sinks to `add(%x, %y)` followed by 1 transpose
    //    `%act = tanh(%sum)`    -> sinks that transpose past `tanh` as well!
    let bin_unary_mlir = r#"
module @sink_transpose_bin_unary {
  func.func @main(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %t1 = stablehlo.transpose %x {permutation = [1, 0]} : (tensor<4x8xf32>) -> tensor<8x4xf32>
    %t2 = stablehlo.transpose %y {permutation = [1, 0]} : (tensor<4x8xf32>) -> tensor<8x4xf32>
    %sum = stablehlo.add %t1, %t2 : tensor<8x4xf32>
    %act = stablehlo.tanh %sum : tensor<8x4xf32>
    return %act : tensor<8x4xf32>
  }
}
"#;
    let mut bu_mod = parse_mlir(bin_unary_mlir).expect("valid MLIR");
    assert_eq!(
        sink_transpose_through_elementwise_and_broadcast(&mut bu_mod),
        2
    );
    let func_id = bu_mod.functions()[0];
    let blk_id = bu_mod.regions[bu_mod.ops[func_id].regions[0]].blocks[0];
    let ops = &bu_mod.blocks[blk_id].ops;
    assert_eq!(ops.len(), 4);
    assert!(bu_mod.ops[ops[0]].is_op("stablehlo", "add"));
    assert!(bu_mod.ops[ops[1]].is_op("stablehlo", "tanh"));
    assert!(bu_mod.ops[ops[2]].is_op("stablehlo", "transpose"));

    // 2. Rank-increasing broadcast (`[1, 3, 2]` -> transpose `[0, 2, 1]` -> `[1, 2, 3]`
    //    -> broadcast `[9, 1, 2, 3]`):
    //    After sinking transpose: broadcast produces `[9, 1, 3, 2]` and transpose uses
    //    padded permutation `[0, 1, 3, 2]` to produce `[9, 1, 2, 3]`!
    let bcast_mlir = r#"
module @sink_transpose_bcast {
  func.func @main(%x: tensor<1x3x2xf32>) -> tensor<9x1x2x3xf32> {
    %t = stablehlo.transpose %x {permutation = [0, 2, 1]}
      : (tensor<1x3x2xf32>) -> tensor<1x2x3xf32>
    %b = mo.static.broadcast_to %t : (tensor<1x2x3xf32>) -> tensor<9x1x2x3xf32>
    return %b : tensor<9x1x2x3xf32>
  }
}
"#;
    let mut bc_mod = parse_mlir(bcast_mlir).expect("valid MLIR");
    assert_eq!(
        sink_transpose_through_elementwise_and_broadcast(&mut bc_mod),
        1
    );
    let b_func = bc_mod.functions()[0];
    let b_blk = bc_mod.regions[bc_mod.ops[b_func].regions[0]].blocks[0];
    let b_ops = &bc_mod.blocks[b_blk].ops;
    assert_eq!(b_ops.len(), 3);
    assert!(bc_mod.ops[b_ops[0]].is_op("mo", "static.broadcast_to"));
    assert!(bc_mod.ops[b_ops[1]].is_op("stablehlo", "transpose"));
    let inter_val = bc_mod.ops[b_ops[0]].results[0];
    assert_eq!(
        bc_mod.values[inter_val].ty.as_tensor().unwrap().shape,
        vec![
            Dim::Static(9),
            Dim::Static(1),
            Dim::Static(3),
            Dim::Static(2)
        ]
    );
    assert_eq!(
        bc_mod.ops[b_ops[1]].attributes.get("permutation"),
        Some(&Attribute::DenseIntArray(vec![0, 1, 3, 2]))
    );
}

#[test]
fn test_classify_blend_read_groups_and_cb_group_rope_geometry() {
    // 1. Single-group fused layout maps both prefix and blend legs to group 0:
    assert_eq!(
        classify_blend_read_groups(1, &[]),
        Some(BlendReadGroupSet {
            blend_gids: vec![0],
            prefix_gids: vec![0],
            recurrent_gids: vec![],
            attn_gid: 0,
        })
    );

    // 2. Multi-group hybrid layout (Recurrent=0, Attention=1, Aux=2):
    let kinds = [
        BlendObjectGroupKind::Recurrent,
        BlendObjectGroupKind::Attention,
        BlendObjectGroupKind::Aux,
    ];
    assert_eq!(
        classify_blend_read_groups(3, &kinds),
        Some(BlendReadGroupSet {
            blend_gids: vec![1, 2],
            prefix_gids: vec![0, 1],
            recurrent_gids: vec![0],
            attn_gid: 1,
        })
    );
    // Rejects multiple attention groups or >1 aux group:
    assert!(
        classify_blend_read_groups(
            2,
            &[
                BlendObjectGroupKind::Attention,
                BlendObjectGroupKind::Attention
            ]
        )
        .is_none()
    );

    // 3. Chunk-major key interleaving (`expansion = 2` keys per chunk across 2 groups):
    let g0 = vec!["c0_g0_a", "c0_g0_b", "c1_g0_a", "c1_g0_b"];
    let g1 = vec!["c0_g1_a", "c0_g1_b", "c1_g1_a", "c1_g1_b"];
    assert_eq!(
        interleave_chunk_major_object_keys(&[g0, g1], 2),
        Some(vec![
            "c0_g0_a", "c0_g0_b", "c0_g1_a", "c0_g1_b", "c1_g0_a", "c1_g0_b", "c1_g1_a", "c1_g1_b"
        ])
    );

    // 4. RoPE geometry: split K/V, fused-packed K/V, and MLA `[content | rope]`:
    assert_eq!(
        compute_cb_group_rope_geometry(16, 16, 2, 1024, 128, false, None, None),
        Some(CBGroupRopeGeometry {
            fused_packed: false,
            per_head_elems: 128,
            num_heads: 8,
            rot_offset: 0,
        })
    );
    assert_eq!(
        compute_cb_group_rope_geometry(16, 16, 1, 2048, 128, true, Some(256), None),
        Some(CBGroupRopeGeometry {
            fused_packed: true,
            per_head_elems: 256,
            num_heads: 8,
            rot_offset: 0,
        })
    );
    // MLA latent (`kv_size = 1`, `hidden_dim = 576 = 512 content + 64 rope`):
    assert_eq!(
        compute_cb_group_rope_geometry(16, 16, 1, 576, 64, false, None, Some((512, 64))),
        Some(CBGroupRopeGeometry {
            fused_packed: false,
            per_head_elems: 576,
            num_heads: 1,
            rot_offset: 512,
        })
    );
    // Rejects MLA window when `kv_size != 1` or window does not end the row:
    assert!(
        compute_cb_group_rope_geometry(16, 16, 2, 576, 64, false, None, Some((512, 64))).is_none()
    );
    assert!(
        compute_cb_group_rope_geometry(16, 16, 1, 576, 64, false, None, Some((500, 64))).is_none()
    );
}

#[test]
fn test_dax_generation_slot_arena_read_borrows_deferred_reclaim_and_shrink() {
    let mut arena = DaxGenerationSlotArena::new(4 * 4096, 4096).expect("4-slot DAX arena");
    assert_eq!(arena.max_slots(), 4);

    // Insert 3 keys into slots 0, 1, 2:
    assert_eq!(arena.put(100, 4096), DaxPutOutcome::Inserted);
    assert_eq!(arena.put(200, 4096), DaxPutOutcome::Inserted);
    assert_eq!(arena.put(300, 4096), DaxPutOutcome::Inserted);
    assert_eq!(arena.put(100, 4096), DaxPutOutcome::AlreadyExists);
    assert_eq!(arena.put(999, 8192), DaxPutOutcome::InvalidSize);

    // Reserve read on key 200 (slot 1, generation 1):
    let (reservations, hits) = arena.reserve_reads(&[200, 404], false);
    assert_eq!(hits, vec![true, false]);
    assert_eq!(reservations.len(), 1);
    assert_eq!(reservations[0].slot_id, 1);
    assert_eq!(reservations[0].generation, 1);

    // Evict key 200 while borrowed by an active read:
    // Key 200 is removed from the index immediately, but slot 1 enters `pending_free`!
    assert_eq!(arena.delete_many(&[200], false), vec![true]);
    assert_eq!(arena.exists_many(&[200], false), vec![false]);

    // `usage()` reports 3/4 = 0.75 current usage (slot 1 still borrowed),
    // and 2/4 = 0.50 post-eviction usage!
    let (cur_u, post_u) = arena.usage();
    assert!((cur_u - 0.75).abs() < 1e-9);
    assert!((post_u - 0.50).abs() < 1e-9);

    // Finalizing the read drains `borrow_count` to 0 and recycles slot 1 into `free_slots`:
    arena.finalize_reads(&reservations);
    let (cur_after, post_after) = arena.usage();
    assert!((cur_after - 0.50).abs() < 1e-9);
    assert!((post_after - 0.50).abs() < 1e-9);

    // Inserting key 400 reuses recycled slot 1 with incremented generation 2:
    let (out, slot_info) = arena.begin_put(400, 2048);
    assert_eq!(out, DaxPutOutcome::Inserted);
    assert_eq!(slot_info, Some((1, 2, 4096)));
    assert!(arena.finalize_put(400, false));

    // Highest occupied slot is slot 2 (`0, 1, 2`), so shrinking to 3 slots is valid,
    // while shrinking to 2 slots is rejected because slot 2 is live:
    assert!(arena.can_shrink_to(3));
    assert!(!arena.can_shrink_to(2));
}

#[test]
fn test_coordinator_key_directory_l1_fencing_shared_l2_and_histogram() {
    let mut dir = CoordinatorKeyDirectory::new();

    // Worker A stores keys 10 and 20 in L1 ("cpu") and key 20 in shared L2 ("redis"):
    dir.consume_batch(
        "worker-a",
        1,
        CacheEventTier::L1,
        "cpu",
        false,
        CacheEventOpKind::Store,
        100,
        &[
            CacheEventEntrySpec {
                key: 10,
                size_bytes: 1024,
            },
            CacheEventEntrySpec {
                key: 20,
                size_bytes: 2048,
            },
        ],
    );
    dir.consume_batch(
        "worker-a",
        1,
        CacheEventTier::L2,
        "redis",
        true,
        CacheEventOpKind::Store,
        101,
        &[CacheEventEntrySpec {
            key: 20,
            size_bytes: 2048,
        }],
    );
    // Worker B re-stores key 20 into the same shared L2 ("redis") pool -> upserts in place!
    dir.consume_batch(
        "worker-b",
        1,
        CacheEventTier::L2,
        "redis",
        true,
        CacheEventOpKind::Store,
        102,
        &[CacheEventEntrySpec {
            key: 20,
            size_bytes: 2048,
        }],
    );
    dir.consume_batch(
        "worker-b",
        1,
        CacheEventTier::L1,
        "cpu",
        false,
        CacheEventOpKind::Access,
        105,
        &[CacheEventEntrySpec {
            key: 20,
            size_bytes: 0,
        }],
    );

    assert_eq!(dir.num_keys(), 2);
    assert_eq!(dir.tier_stats(), (2, 3072, 1, 2048));
    assert_eq!(dir.get_access_counts(&[10, 20, 99]), vec![0, 1, 0]);

    // Fencing `worker-a` drops its 2 L1 placements (removing key 10 completely),
    // while key 20's L2 placement survives!
    assert_eq!(dir.fence_instance("worker-a"), 2);
    assert_eq!(dir.num_keys(), 1);
    assert_eq!(dir.tier_stats(), (0, 0, 1, 2048));
    let looked_up = dir.lookup(&[10, 20]);
    assert!(looked_up[0].is_empty());
    assert_eq!(
        looked_up[1],
        vec![PlacementRecord {
            instance_id: "worker-b".to_string(),
            incarnation: 1,
            tier: CacheEventTier::L2,
            backend: "redis".to_string(),
            size_bytes: 2048,
            shared: true,
        }]
    );

    // Verify `BucketHistogram` observation and Prometheus cumulative bucket series:
    let mut hist = BucketHistogram::with_bounds(&[1.0, 4.0, 16.0]).expect("valid bounds");
    assert!(BucketHistogram::with_bounds(&[4.0, 2.0]).is_none());
    assert!(BucketHistogram::with_bounds(CHUNK_COUNT_HISTOGRAM_BOUNDS).is_some());
    hist.observe(0.5);
    hist.observe(4.0);
    hist.observe(32.0);
    assert_eq!(hist.count(), 3);
    assert!((hist.sum() - 36.5).abs() < 1e-9);
    assert_eq!(
        hist.cumulative_buckets(),
        vec![
            ("1".to_string(), 1),
            ("4".to_string(), 2),
            ("16".to_string(), 2),
            ("+Inf".to_string(), 3),
        ]
    );
}

#[test]
fn test_fdp_raw_block_slot_manager_and_double_buffered_meta_checkpoint() {
    // 1. Verify raw-block per-slot header codec (`LMCBLK01` + slot_identity + payload_len):
    assert_eq!(RAW_BLOCK_SLOT_MAGIC, b"LMCBLK01");
    assert_eq!(RAW_BLOCK_META_MAGIC, b"LMCIDX01");
    assert!(encode_raw_block_slot_header(0x1122_3344_5566_7788, 65536, 16).is_none());
    let slot_hdr =
        encode_raw_block_slot_header(0x1122_3344_5566_7788, 65536, 4096).expect("valid header");
    assert_eq!(slot_hdr.len(), 4096);
    assert_eq!(
        decode_raw_block_slot_header(&slot_hdr),
        Some((0x1122_3344_5566_7788, 65536))
    );

    // 2. Verify double-buffered metadata checkpoint container selection and CRC32 validation:
    let payload_v1 = br#"{"next_slot":3,"entries":[0,2]}"#;
    let payload_v2 = br#"{"next_slot":4,"entries":[0,2,3]}"#;
    let container_a =
        encode_raw_block_meta_checkpoint(1, 10, payload_v1, 4096).expect("valid checkpoint a");
    let container_b =
        encode_raw_block_meta_checkpoint(1, 11, payload_v2, 4096).expect("valid checkpoint b");

    // Both containers valid -> selects higher `seq = 11`:
    let (hdr, chosen_payload) = select_latest_valid_meta_checkpoint(
        &[(0, container_a.clone()), (65536, container_b.clone())],
        4096,
        1,
    )
    .expect("should pick valid checkpoint");
    assert_eq!(
        hdr,
        RawBlockMetaCheckpointHeader {
            seq: 11,
            payload_len: payload_v2.len() as u64,
            crc32: compute_ieee_crc32(payload_v2),
            container_offset: 65536,
        }
    );
    assert_eq!(chosen_payload, payload_v2);

    // Corrupt a byte in container B's payload -> falls back to container A (`seq = 10`):
    let mut corrupted_b = container_b;
    corrupted_b[4100] ^= 0x5A;
    let (fallback_hdr, fallback_payload) =
        select_latest_valid_meta_checkpoint(&[(0, container_a), (65536, corrupted_b)], 4096, 1)
            .expect("should fall back to container A");
    assert_eq!(fallback_hdr.seq, 10);
    assert_eq!(fallback_hdr.container_offset, 0);
    assert_eq!(fallback_payload, payload_v1);

    // 3. Verify NVMe FDP placement-affinity slot recycling and crash-safe free-list rebuild:
    let mut mgr = FdpRawBlockSlotManager::new(4, true);
    let s0 = mgr.allocate_slot(Some(100)).expect("slot 0");
    let s1 = mgr.allocate_slot(Some(200)).expect("slot 1");
    let s2 = mgr.allocate_slot(Some(100)).expect("slot 2");
    assert_eq!((s0, s1, s2), (0, 1, 2));
    assert_eq!(mgr.slot_placement_id(s1), Some(200));

    // Free slots 0 (PID 100) and 1 (PID 200), then request PID 100 -> affinity hit on slot 0!
    mgr.free_slot(s0);
    mgr.free_slot(s1);
    let s_reused_100 = mgr.allocate_slot(Some(100)).expect("affinity hit");
    assert_eq!(s_reused_100, 0);
    assert_eq!(mgr.stats(), (3, 1, 1, 0));

    // Request PID 300 while slot 1 (PID 200) is in free list -> fallback recycle of slot 1!
    let s_fallback_300 = mgr.allocate_slot(Some(300)).expect("fallback recycle");
    assert_eq!(s_fallback_300, 1);
    assert_eq!(mgr.slot_placement_id(1), Some(300));
    assert_eq!(mgr.stats(), (3, 0, 1, 1));

    // Simulate crash recovery where `next_slot = 4` had uncommitted in-flight slot 1 and freed
    // slot 3, with only slots `[0, 2]` committed in the checkpoint:
    assert!(mgr.rebuild_free_slots_from_committed(4, &[0, 2]));
    assert_eq!(mgr.stats(), (4, 2, 1, 1));
    assert_eq!(mgr.allocate_slot(None), Some(3));
    assert_eq!(mgr.allocate_slot(None), Some(1));
    assert_eq!(mgr.allocate_slot(None), None);
}

#[test]
fn test_swim_reshape_through_slice_and_sink_split_through_reshape() {
    // 1. Verify `swim_reshape_through_slice`:
    //    `slice(tensor<8x6xf32> -> tensor<4x6xf32>, start=[2, 0], limit=[6, 6], strides=[1, 1])`
    //    followed by `reshape(tensor<4x6xf32> -> tensor<2x2x6xf32>)`
    //    commutes into `reshape(tensor<8x6xf32> -> tensor<4x2x6xf32>)` followed by
    //    `slice(tensor<4x2x6xf32> -> tensor<2x2x6xf32>, start=[1, 0, 0], limit=[3, 2, 6])`!
    let mlir_slice_reshape = concat!(
        "module @swim_reshape_slice {\n",
        "  func.func @main(%x: tensor<8x6xf32>) -> tensor<2x2x6xf32> {\n",
        "    %s = stablehlo.slice %x ",
        "{start_indices = [2, 0], limit_indices = [6, 6], strides = [1, 1]} ",
        ": (tensor<8x6xf32>) -> tensor<4x6xf32>\n",
        "    %r = stablehlo.reshape %s : (tensor<4x6xf32>) -> tensor<2x2x6xf32>\n",
        "    return %r : tensor<2x2x6xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m1 = parse_mlir(mlir_slice_reshape).expect("valid MLIR");
    assert_eq!(swim_reshape_through_slice(&mut m1), 1);
    let f1 = m1.functions()[0];
    let b1 = m1.regions[m1.ops[f1].regions[0]].blocks[0];
    let ops1 = &m1.blocks[b1].ops;
    assert!(m1.ops[ops1[0]].is_op("stablehlo", "reshape"));
    assert!(m1.ops[ops1[1]].is_op("stablehlo", "slice"));
    let mid_v = m1.ops[ops1[0]].results[0];
    assert_eq!(
        m1.values[mid_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(4), Dim::Static(2), Dim::Static(6)]
    );

    // 2. Verify `sink_split_through_reshape`:
    //    `mo.split %x {axis = 0} : (tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)`
    //    followed by two identical `stablehlo.reshape` ops `tensor<2x6xf32> -> tensor<2x2x3xf32>`
    //    commutes into a single `stablehlo.reshape` (`tensor<4x6xf32> -> tensor<4x2x3xf32>`)
    //    followed by `mo.split` along `axis = 0`!
    let mlir_split_reshape = r#"
module @sink_split_reshape {
  func.func @main(%x: tensor<4x6xf32>) -> tensor<2x2x3xf32> {
    %s0, %s1 = mo.split %x {axis = 0} : (tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)
    %r0 = stablehlo.reshape %s0 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
    %r1 = stablehlo.reshape %s1 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
    %sum = stablehlo.add %r0, %r1 : tensor<2x2x3xf32>
    return %sum : tensor<2x2x3xf32>
  }
}
"#;
    let mut m2 = parse_mlir(mlir_split_reshape).expect("valid MLIR");
    assert_eq!(sink_split_through_reshape(&mut m2), 1);
    let f2 = m2.functions()[0];
    let b2 = m2.regions[m2.ops[f2].regions[0]].blocks[0];
    let ops2 = &m2.blocks[b2].ops;
    assert_eq!(ops2.len(), 4);
    assert!(m2.ops[ops2[0]].is_op("stablehlo", "reshape"));
    assert!(m2.ops[ops2[1]].is_op("mo", "split"));
    let pre_split_v = m2.ops[ops2[0]].results[0];
    assert_eq!(
        m2.values[pre_split_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(4), Dim::Static(2), Dim::Static(3)]
    );
}

#[test]
fn test_sink_broadcast_through_reduction_non_broadcast_and_idempotent() {
    // 1. Non-broadcasted axis (`axis = 2` on `tensor<3x4x8xf32>` broadcast from `tensor<4x8xf32>`):
    //    Commutes into `mo.reduce_add %x {axis = 1} : (tensor<4x8xf32>) -> tensor<4x1xf32>`
    //    followed by `mo.static.broadcast_to -> tensor<3x4x1xf32>`!
    let mlir_non_bcast = r#"
module @sink_bcast_reduce_non_bcast {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<3x4x1xf32> {
    %b = mo.static.broadcast_to %x : (tensor<4x8xf32>) -> tensor<3x4x8xf32>
    %r = mo.reduce_add %b {axis = 2} : (tensor<3x4x8xf32>) -> tensor<3x4x1xf32>
    return %r : tensor<3x4x1xf32>
  }
}
"#;
    let mut m1 = parse_mlir(mlir_non_bcast).expect("valid MLIR");
    assert_eq!(sink_broadcast_through_reduction(&mut m1), 1);
    let f1 = m1.functions()[0];
    let b1 = m1.regions[m1.ops[f1].regions[0]].blocks[0];
    let ops1 = &m1.blocks[b1].ops;
    assert_eq!(ops1.len(), 3);
    assert!(m1.ops[ops1[0]].is_op("mo", "reduce_add"));
    assert_eq!(
        m1.ops[ops1[0]].attributes.get("axis"),
        Some(&castforge::ir::Attribute::Int(1))
    );
    assert!(m1.ops[ops1[1]].is_op("mo", "static.broadcast_to"));
    let mid_v = m1.ops[ops1[0]].results[0];
    assert_eq!(
        m1.values[mid_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(4), Dim::Static(1)]
    );

    // 2. Broadcasted axis (`axis = 1` on `tensor<3x4x8xf32>` broadcast from `tensor<1x8xf32>`)
    //    with idempotent `mo.reduce_max`: folds directly into a single `mo.static.broadcast_to`!
    let mlir_idempotent = r#"
module @sink_bcast_reduce_idempotent {
  func.func @main(%x: tensor<1x8xf32>) -> tensor<3x1x8xf32> {
    %b = mo.static.broadcast_to %x : (tensor<1x8xf32>) -> tensor<3x4x8xf32>
    %r = mo.reduce_max %b {axis = 1} : (tensor<3x4x8xf32>) -> tensor<3x1x8xf32>
    return %r : tensor<3x1x8xf32>
  }
}
"#;
    let mut m2 = parse_mlir(mlir_idempotent).expect("valid MLIR");
    assert_eq!(sink_broadcast_through_reduction(&mut m2), 1);
    let f2 = m2.functions()[0];
    let b2 = m2.regions[m2.ops[f2].regions[0]].blocks[0];
    let ops2 = &m2.blocks[b2].ops;
    assert_eq!(ops2.len(), 2);
    assert!(m2.ops[ops2[0]].is_op("mo", "static.broadcast_to"));
    let out_v = m2.ops[ops2[0]].results[0];
    assert_eq!(
        m2.values[out_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(3), Dim::Static(1), Dim::Static(8)]
    );
}

#[test]
fn test_fold_extend_lifetimes_and_device_graph_collective_partition() {
    // 1. Verify `fold_extend_lifetimes_into_async_kernels`:
    //    `%c1 = mogg.async_exec %c0, %a` followed by `%c2 = mgp.extend_lifetimes %c1, %b, %a`
    //    and a no-op `%c3 = mgp.extend_lifetimes %c2` folds `%b` into `mogg.async_exec`
    //    (deduplicating `%a`) and forwards `%c3` -> `%c1`!
    let mlir_lifetime = concat!(
        "module @simplify_extend_lifetimes {\n",
        "  func.func @main(%c0: tensor<1xi64>, %a: tensor<4xf32>, %b: tensor<4xf32>) ",
        "-> tensor<1xi64> {\n",
        "    %c1 = mogg.async_exec %c0, %a : (tensor<1xi64>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %c2 = mgp.extend_lifetimes %c1, %b, %a ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %c3 = mgp.extend_lifetimes %c2 : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %c3 : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_lifetime).expect("valid MLIR");
    assert_eq!(fold_extend_lifetimes_into_async_kernels(&mut m), 2);
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    assert_eq!(ops.len(), 2);
    assert!(m.ops[ops[0]].is_op("mogg", "async_exec"));
    assert_eq!(m.ops[ops[0]].operands.len(), 3);
    assert_eq!(
        m.ops[ops[0]]
            .attributes
            .get("castforge.extended_lifetime_count"),
        Some(&castforge::ir::Attribute::Int(1))
    );
    assert_eq!(m.ops[ops[1]].operands[0], m.ops[ops[0]].results[0]);

    // 2. Verify `plan_device_graph_collective_partition` across 2 devices:
    let nodes = vec![
        CollectiveGraphNodeSpec {
            node_id: 10,
            devices: vec![0],
            is_collective: false,
            is_graph_capturable: true,
        },
        CollectiveGraphNodeSpec {
            node_id: 11,
            devices: vec![1],
            is_collective: false,
            is_graph_capturable: true,
        },
        CollectiveGraphNodeSpec {
            node_id: 20,
            devices: vec![0, 1],
            is_collective: true,
            is_graph_capturable: true,
        },
        CollectiveGraphNodeSpec {
            node_id: 30,
            devices: vec![0, 1],
            is_collective: false,
            is_graph_capturable: false,
        },
        CollectiveGraphNodeSpec {
            node_id: 40,
            devices: vec![0, 1],
            is_collective: false,
            is_graph_capturable: true,
        },
    ];

    // When `capture_collectives_in_graph == false`, node 20 splits the capture into a
    // pre-collective graph, an `EagerCollectiveStage`, an `EagerHostStage` (node 30), and a
    // post-host graph (node 40):
    let stages_split =
        plan_device_graph_collective_partition(2, &nodes, false).expect("valid partition");
    assert_eq!(
        stages_split,
        vec![
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![10], vec![11]],
            },
            DeviceGraphStage::EagerCollectiveStage { node_ids: vec![20] },
            DeviceGraphStage::EagerHostStage { node_ids: vec![30] },
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![40], vec![40]],
            },
        ]
    );

    // When `capture_collectives_in_graph == true`, the all-device collective (node 20) is captured
    // directly inside the first `DeviceGraphCapture` stage!
    let stages_fused =
        plan_device_graph_collective_partition(2, &nodes, true).expect("valid partition");
    assert_eq!(
        stages_fused,
        vec![
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![10, 20], vec![11, 20]],
            },
            DeviceGraphStage::EagerHostStage { node_ids: vec![30] },
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![40], vec![40]],
            },
        ]
    );
}

#[test]
fn test_turboquant_serde_config_lloyd_max_and_orthonormal_hadamard() {
    assert_eq!(
        TurboQuantPreset::from_name("turboquant_k3v4_nc"),
        Some(TurboQuantPreset::K3V4Nc)
    );
    assert!(TurboQuantPreset::from_name("unknown").is_none());

    // 1. Verify `TurboQuantSerdeConfig` slot geometry and serialized byte calculation:
    //    For `turboquant_k3v4_nc` with `head_dim = 128`:
    //    - `key_packed_size = ceil(128 * 3 / 8) + 2 = 48 + 2 = 50`
    //    - `value_packed_size = ceil(128 * 4 / 8) + 4 = 64 + 4 = 68`
    //    - `slot_size = 118`, `slot_size_aligned = 118`
    let cfg = TurboQuantSerdeConfig::new(TurboQuantPreset::K3V4Nc, 128, 16, 2, 2).expect("valid");
    assert!(!cfg.key_fp8());
    assert_eq!(cfg.key_mse_bits(), 3);
    assert_eq!(cfg.centroid_bits(), 3);
    assert_eq!(cfg.key_packed_size(), 50);
    assert_eq!(cfg.value_packed_size(), 68);
    assert_eq!(cfg.slot_size(), 118);
    assert_eq!(cfg.slot_size_aligned(), 118);
    assert_eq!(cfg.layer_ranges(10), (2, 8));
    // 10 layers (4 raw, 6 quantized), 32 tokens (2 blocks of 16), hidden_dim=512 (4 heads), fp16:
    // raw_bytes = 2 * 4 * 32 * 512 * 2 = 262_144
    // quant_bytes = 6 * 2 * 16 * 4 * 118 = 90_624
    assert_eq!(
        cfg.serialized_nbytes_for_shape(10, 32, 512, 2),
        Some(262_144 + 90_624)
    );

    // 2. Verify `solve_lloyd_max_gaussian` symmetry, strict monotonicity, and midpoint boundaries:
    let (centroids, boundaries) =
        solve_lloyd_max_gaussian(128, 3, 200, 1e-10).expect("valid lloyd-max");
    assert_eq!(centroids.len(), 8);
    assert_eq!(boundaries.len(), 7);
    for i in 0..7 {
        assert!(centroids[i] < boundaries[i]);
        assert!(boundaries[i] < centroids[i + 1]);
    }
    for i in 0..4 {
        assert!((centroids[i] + centroids[7 - i]).abs() < 1e-5);
    }

    // 3. Verify `build_orthonormal_hadamard` produces $H^T H = I$:
    assert!(build_orthonormal_hadamard(6).is_none());
    let h = build_orthonormal_hadamard(8).expect("valid hadamard");
    assert_eq!(h.len(), 64);
    for i in 0..8 {
        for j in 0..8 {
            let dot: f32 = (0..8).map(|k| h[i * 8 + k] * h[j * 8 + k]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-5);
        }
    }
}

#[test]
fn test_recover_contiguous_or_dim0_padded_view_and_stride_ties() {
    assert_eq!(
        expected_contiguous_shape_from_strides(&[16, 4, 1], 64),
        Some(vec![4, 4, 4])
    );

    // 1. Transposed HND physical layout `[2, 4, 8, 16, 64]` exposed logically as
    //    `[2, 4, 16, 8, 64]` with strides `[32768, 8192, 64, 1024, 1]`:
    let hnd =
        recover_contiguous_or_dim0_padded_view(&[2, 4, 16, 8, 64], &[32768, 8192, 64, 1024, 1])
            .expect("valid transposed contiguous view");
    assert_eq!(
        hnd,
        ContiguousViewPlan {
            permutation: vec![0, 1, 3, 2, 4],
            permuted_shape: vec![2, 4, 8, 16, 64],
            permuted_strides: vec![32768, 8192, 1024, 64, 1],
            is_strictly_contiguous: true,
            padding_per_block_elems: 0,
            block_stride_elems: 32768,
        }
    );

    // 2. Size-1 dimension tying on stride with neighbor (`[NB=4, NH=1, BS=16, HS=64]` with
    //    strides `[1024, 64, 64, 1]`) places the size-1 dim innermost among tied dims
    //    (`permutation = [0, 2, 1, 3]`, `permuted_shape = [4, 16, 1, 64]`):
    let tied = recover_contiguous_or_dim0_padded_view(&[4, 1, 16, 64], &[1024, 64, 64, 1])
        .expect("valid tied-stride view");
    assert_eq!(tied.permutation, vec![0, 2, 1, 3]);
    assert_eq!(tied.permuted_shape, vec![4, 16, 1, 64]);
    assert!(tied.is_strictly_contiguous);

    // 3. DeepSeek-V4 style `dim-0`-padded view (`shape = [4, 16, 64]`, `inner_tight = 1024`,
    //    `strides = [1536, 64, 1]` -> `padding_per_block_elems = 512`):
    let padded = recover_contiguous_or_dim0_padded_view(&[4, 16, 64], &[1536, 64, 1])
        .expect("valid dim-0-padded view");
    assert!(!padded.is_strictly_contiguous);
    assert_eq!(padded.padding_per_block_elems, 512);
    assert_eq!(padded.block_stride_elems, 1536);

    // 4. Reject interior-dimension padding (`strides = [2048, 128, 1]` for `[4, 16, 64]`) and
    //    overlapping outer blocks (`strides = [512, 64, 1]`):
    assert!(recover_contiguous_or_dim0_padded_view(&[4, 16, 64], &[2048, 128, 1]).is_none());
    assert!(recover_contiguous_or_dim0_padded_view(&[4, 16, 64], &[512, 64, 1]).is_none());
}

#[test]
fn test_fleet_cache_usage_view_and_watermark_eviction_controller() {
    let mut usage = FleetCacheUsageView::new();
    let mut ctrl = FleetWatermarkEvictionController::new(0.5, 0.8, 2);

    // 1. Store 4 keys under `tenant-a` in L2 (each 1000 bytes -> 4000 bytes total), plus key 10
    //    replicated on a second non-shared L2 backend (`nvme`) and in `worker-1` L1:
    let entries_a = vec![
        CacheEventEntrySpec {
            key: 10,
            size_bytes: 1000,
        },
        CacheEventEntrySpec {
            key: 20,
            size_bytes: 1000,
        },
        CacheEventEntrySpec {
            key: 30,
            size_bytes: 1000,
        },
        CacheEventEntrySpec {
            key: 40,
            size_bytes: 1000,
        },
    ];
    usage.consume_batch(
        "worker-1",
        CacheEventTier::L2,
        "redis",
        true,
        CacheEventOpKind::Store,
        "tenant-a",
        &entries_a,
    );
    ctrl.consume_l2_batch(
        &usage,
        CacheEventOpKind::Store,
        "tenant-a",
        &[10, 20, 30, 40],
    );

    usage.consume_batch(
        "worker-2",
        CacheEventTier::L2,
        "nvme",
        false,
        CacheEventOpKind::Store,
        "tenant-a",
        &[CacheEventEntrySpec {
            key: 10,
            size_bytes: 1000,
        }],
    );
    usage.consume_batch(
        "worker-1",
        CacheEventTier::L1,
        "dram",
        false,
        CacheEventOpKind::Store,
        "tenant-a",
        &[CacheEventEntrySpec {
            key: 10,
            size_bytes: 1000,
        }],
    );

    assert_eq!(usage.get_key_bytes(CacheEventTier::L2, 10), 2000);
    assert_eq!(usage.get_salt_bytes(CacheEventTier::L2, "tenant-a"), 5000);
    assert_eq!(usage.get_total_bytes(CacheEventTier::L1), 1000);

    // Fencing `worker-1` drops its L1 bytes (`1000`) while leaving L2 (`5000`) untouched:
    assert_eq!(usage.fence_instance("worker-1"), 1000);
    assert_eq!(usage.get_total_bytes(CacheEventTier::L1), 0);
    assert_eq!(usage.get_salt_bytes(CacheEventTier::L2, "tenant-a"), 5000);

    // Deleting key 10 from `nvme` still leaves its `redis` L2 copy (`1000` bytes), so key 10
    // stays in the LRU!
    usage.consume_batch(
        "worker-2",
        CacheEventTier::L2,
        "nvme",
        false,
        CacheEventOpKind::Delete,
        "tenant-a",
        &[CacheEventEntrySpec {
            key: 10,
            size_bytes: 0,
        }],
    );
    ctrl.consume_l2_batch(&usage, CacheEventOpKind::Delete, "tenant-a", &[10]);
    assert_eq!(ctrl.policy.resident_len_for_salt("tenant-a"), 4);

    // 2. Pin key 40 (the LRU head, since `on_keys_created` inserts in reverse request order
    //    `[40, 30, 20, 10]`) and set quota for `tenant-a` to `5000` (`watermark = 4000`):
    ctrl.pin(&[40]);
    assert_eq!(ctrl.pin_count(40), 1);
    assert_eq!(ctrl.filter_unpinned(&[40, 30, 20]), vec![30, 20]);

    // Without quota configured, `tenant-a` is exempt:
    assert!(ctrl.compute_eviction_plan(&usage).is_empty());

    // Arm quota `5000` -> `current_bytes = 4000 >= 0.8 * 5000` triggers eviction of 2
    // unpinned keys (`[30, 20]`, skipping pinned key `40`):
    ctrl.quota.set_quota("tenant-a", 5000);
    let plan = ctrl.compute_eviction_plan(&usage);
    assert_eq!(plan.get("tenant-a"), Some(&vec![30, 20]));

    // With quota `0`, `effective_ratio = 1.0` selects all 3 unpinned keys (`[30, 20, 10]`) and
    // chunks them into batches of `max_delete_batch = 2`:
    ctrl.quota.set_quota("tenant-a", 0);
    let full_plan = ctrl.compute_eviction_plan(&usage);
    assert_eq!(full_plan.get("tenant-a"), Some(&vec![30, 20, 10]));
    assert_eq!(
        ctrl.chunk_delete_batches(&full_plan),
        vec![
            ("tenant-a".to_string(), vec![30, 20]),
            ("tenant-a".to_string(), vec![10]),
        ]
    );
}

#[test]
fn test_fuse_horizontal_shared_input_ops_and_def_before_use_guard() {
    // 1. Sibling elementwise ops `%a = stablehlo.add %x, %y` and `%b = stablehlo.multiply %x, %z`
    //    share `%x` and have no dependency between them -> fuse into `mo.horizontal_fused`!
    let mlir_fusible = r#"
module @horizontal_fuse_ok {
  func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> tensor<4xf32> {
    %a = stablehlo.add %x, %y : tensor<4xf32>
    %b = stablehlo.multiply %x, %z : tensor<4xf32>
    %out = stablehlo.subtract %a, %b : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
"#;
    let mut m1 = parse_mlir(mlir_fusible).expect("valid MLIR");
    assert_eq!(fuse_horizontal_shared_input_ops(&mut m1), 1);
    let f1 = m1.functions()[0];
    let b1 = m1.regions[m1.ops[f1].regions[0]].blocks[0];
    let ops1 = &m1.blocks[b1].ops;
    assert_eq!(ops1.len(), 3);
    assert!(m1.ops[ops1[0]].is_op("mo", "horizontal_fused"));
    assert_eq!(m1.ops[ops1[0]].operands.len(), 3);
    assert_eq!(m1.ops[ops1[0]].results.len(), 2);

    // 2. When an intermediate op `%mid` between `%a` and `%b` reads `%a`, moving `%a` down to `%b`
    //    would violate def-before-use (`resultsUnusedBefore(a, b) == false`), so horizontal fusion
    //    is rejected!
    let mlir_blocked = r#"
module @horizontal_fuse_blocked {
  func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> tensor<4xf32> {
    %a = stablehlo.add %x, %y : tensor<4xf32>
    %mid = stablehlo.tanh %a : tensor<4xf32>
    %b = stablehlo.multiply %x, %z : tensor<4xf32>
    %out = stablehlo.subtract %mid, %b : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
"#;
    let mut m2 = parse_mlir(mlir_blocked).expect("valid MLIR");
    assert_eq!(fuse_horizontal_shared_input_ops(&mut m2), 0);
}

#[test]
fn test_hoist_callsite_invariant_subgraph_ops() {
    // Subgraph `@sub` is called twice from `@main` with the same weight `%w` as argument 0 and
    // varying activations `%x1`, `%x2` as argument 1.
    // Inside `@sub`, `%w_exp = stablehlo.exponential %arg0` and `%w_tan = stablehlo.tanh %w_exp`
    // depend only on callsite-invariant `%arg0`, so both ops are hoisted into `@main` before the
    // first `func.call @sub`, `%arg0` is pruned, and `%w_tan` is passed into `@sub`!
    let mlir = r#"
module @hoist_subgraph_invariant {
  func.func @sub(%w: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
    %w_exp = stablehlo.exponential %w : tensor<4xf32>
    %w_tan = stablehlo.tanh %w_exp : tensor<4xf32>
    %y = stablehlo.add %x, %w_tan : tensor<4xf32>
    return %y : tensor<4xf32>
  }
  func.func @main(%w: tensor<4xf32>, %x1: tensor<4xf32>, %x2: tensor<4xf32>) -> tensor<4xf32> {
    %r1 = func.call @sub(%w, %x1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %r2 = func.call @sub(%w, %x2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %out = stablehlo.add %r1, %r2 : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
"#;
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(hoist_callsite_invariant_subgraph_ops(&mut m), 2);

    let fns = m.functions();
    let sub_fn = fns[0];
    let main_fn = fns[1];
    let sub_blk = m.regions[m.ops[sub_fn].regions[0]].blocks[0];
    let main_blk = m.regions[m.ops[main_fn].regions[0]].blocks[0];

    // `@sub` now has only `stablehlo.add` and `func.return` (2 ops instead of 4), and 2 args
    // (`%x` and the hoisted boundary value `%w_tan`):
    assert_eq!(m.blocks[sub_blk].ops.len(), 2);
    assert!(m.ops[m.blocks[sub_blk].ops[0]].is_op("stablehlo", "add"));
    assert_eq!(m.blocks[sub_blk].arguments.len(), 2);

    // `@main` now begins with the 2 hoisted ops (`stablehlo.exponential` and `stablehlo.tanh`)
    // before the 2 `func.call` ops:
    let main_ops = &m.blocks[main_blk].ops;
    assert_eq!(main_ops.len(), 6);
    assert!(m.ops[main_ops[0]].is_op("stablehlo", "exponential"));
    assert!(m.ops[main_ops[1]].is_op("stablehlo", "tanh"));
    assert!(m.ops[main_ops[2]].is_op("func", "call"));
    assert!(m.ops[main_ops[3]].is_op("func", "call"));
}
