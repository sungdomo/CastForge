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
    ArcKvEvictionPolicy, AsymK16V8Codec, CBChunkMatch, CBGroupSpec, CacheBlendRopeSpec,
    CacheGenCdfCodec, ChainedBlockHashConfig, DEFAULT_MAX_KERNELS_IN_FUSION_SET, DeviceTargetArch,
    EngineKvFormat, FULL_ATTENTION_WINDOW, GDS_DEFAULT_ALIGN_BYTES, GdsSlabAddressManager,
    IsolatedLruEvictionPolicy, KV_ZERO_SCALE_SENTINEL, KvAttentionMaskKind, KvBlockPool,
    KvDmaTransferMode, KvPageSpec, KvPrefixBloomFilter, KvSaltQuotaRegistry, KvScaleScope,
    KvStorageKeySpec, L2FaultInjectConfig, L2FaultInjectPolicy, LayerGroupPayloadSharder,
    PassManager, PassRegistry, ROWWISE_CHEAP_OP_COST, ROWWISE_EXPENSIVE_OP_COST,
    WarmPrefetchJobTable, WarmPrefetchStatus, analyze_kv_layer_groups, anonymize_symbolic_dims,
    compress_splat_constants, decode_kv_fs_filename, decode_kv_storage_key,
    defer_allocs_to_first_use, direct_copy_format_supported, elide_redundant_state_transfers,
    encode_kv_fs_filename, encode_kv_storage_key, flatten_elementwise_chains,
    fold_unfold_hybrid_prefix_bitmaps, fuse_locations, fuse_reshape_epilogue,
    fuse_sole_use_prologues, hash_prefix_token_blocks, hoist_exec_invariant_ops,
    hoist_symbolic_dim_expressions, is_supported_arch_dtype, legalize_rowwise_reduction_launch,
    longest_full_attention_hit, longest_joint_prefix_hit, longest_sliding_window_hit,
    mark_cuda_graph_stable_tensors, merge_duplicate_shape_ops, merge_identical_functions,
    plan_cacheblend_retrieve, plan_concat_in_place, plan_jenga_kv_geometry,
    plan_kernel_sequence_fusion, plan_kv_dma_transfer, promote_narrow_accumulators,
    re_rope_key_in_place, reuse_dead_loop_buffers, stamp_fused_provenance, unfold_chunk_range,
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
