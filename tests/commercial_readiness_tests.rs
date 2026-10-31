//! Commercial-readiness integration tests covering the C FFI ABI (`src/ffi.rs`), untrusted input
//! hardening & resource quotas, lazy `.safetensors` indexing, structured pass telemetry, and
//! extended ONNX operators (`Einsum`, `Tile`, `TopK`, negative-axis ops).

use castforge::codegen::evaluate_module;
use castforge::ffi::{
    castforge_compile_mlir, castforge_free_string, castforge_optimize_mlir, castforge_version,
};
use castforge::ir::formats::{import_gguf_bytes, import_onnx_json};
use castforge::ir::{DType, Dim, TensorBacking, TensorEntry, TensorStore, TensorType, parse_mlir};
use castforge::passes::{PassManager, PassRegistry};
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
