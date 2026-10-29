use castforge_mlir::{parse_mlir, print_mlir};
use castforge_passes::{PassManager, PassRegistry};

const SURGERY_INLINE_MLIR: &str = r#"
module @safetensors_surgery {
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %w_a = stablehlo.constant dense<[
      [0.1, 4.0, 0.2, 8.0],
      [0.1, 2.0, 0.1, 6.0],
      [0.2, 3.0, 0.1, 5.0]
    ]> : tensor<3x4xf32>
    %w_b = stablehlo.constant dense<[
      [0.3, 6.0, 0.4, 10.0],
      [0.1, 4.0, 0.3, 8.0],
      [0.2, 5.0, 0.1, 7.0]
    ]> : tensor<3x4xf32>
    %w_merged = stablehlo.custom_call %w_a, %w_b
      {call_target_name = "castforge.merge"}
      : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    %out = stablehlo.dot_general %x, %w_merged
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
"#;

#[test]
fn test_externalize_safetensors_roundtrip_and_weight_surgery() {
    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(SURGERY_INLINE_MLIR).expect("parse failed");

    // 1. Externalize inline `dense<...>` constants to `dense_resource<...>`
    let mut ext_pm = PassManager::from_pipeline_str(&registry, "externalize-weights")
        .expect("pipeline build failed");
    ext_pm.run(&mut module).expect("externalize failed");

    let ext_mlir = print_mlir(&module);
    assert!(ext_mlir.contains("dense_resource<w_a> : tensor<3x4xf32>"));
    assert!(ext_mlir.contains("dense_resource<w_b> : tensor<3x4xf32>"));
    assert!(!ext_mlir.contains("dense<[["));

    // 2. Serialize TensorStore to binary `.safetensors` and reload into a fresh module
    let st_bytes = module
        .tensor_store
        .to_safetensors_bytes()
        .expect("safetensors serialize failed");

    let mut reloaded_mod = parse_mlir(&ext_mlir).expect("reparse dense_resource mlir failed");
    assert!(reloaded_mod.tensor_store.is_empty());
    let loaded_count = reloaded_mod
        .tensor_store
        .load_safetensors_bytes(&st_bytes)
        .expect("safetensors load failed");
    assert_eq!(loaded_count, 2);

    // 3. Run full weight surgery (SLERP merge + L1 prune + i8 quantize + dce) on dense_resource IR
    let mut surgery_pm = PassManager::from_pipeline_str(
        &registry,
        "merge-weights(method=slerp,t=0.5),\
         prune-weights(mode=structured,sparsity=0.5),\
         quantize(bits=8),\
         dce",
    )
    .expect("surgery pipeline build failed");
    surgery_pm
        .run(&mut reloaded_mod)
        .expect("surgery pipeline run failed");

    let post_surgery_mlir = print_mlir(&reloaded_mod);
    assert!(post_surgery_mlir.contains("dense_resource<w_merged>"));
    assert!(post_surgery_mlir.contains("!quant.uniform<i8:f32,"));
    assert!(post_surgery_mlir.contains("stablehlo.uniform_dequantize"));
    assert!(post_surgery_mlir.contains("-> tensor<2x2xf32>"));

    // 4. Save quantized weights to `.safetensors`, reload, and inline back to `dense<...>`
    let q_st_bytes = reloaded_mod
        .tensor_store
        .to_safetensors_bytes()
        .expect("quantized safetensors serialize failed");
    let mut final_mod = parse_mlir(&post_surgery_mlir).expect("parse post surgery failed");
    final_mod
        .tensor_store
        .load_safetensors_bytes(&q_st_bytes)
        .expect("load quantized safetensors failed");

    let mut inline_pm =
        PassManager::from_pipeline_str(&registry, "externalize-weights(inline=true)")
            .expect("inline pipeline build failed");
    inline_pm.run(&mut final_mod).expect("inline run failed");

    let inlined_mlir = print_mlir(&final_mod);
    assert!(!inlined_mlir.contains("dense_resource<"));
    assert!(inlined_mlir.contains("dense<[["));
}
