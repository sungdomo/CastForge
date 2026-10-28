use castforge_codegen::{CodegenTarget, build_memory_plan, generate_code};
use castforge_mlir::parse_mlir;
use castforge_passes::{PassManager, PassRegistry};

#[test]
fn test_end_to_end_pipeline_to_rust_c_and_wgsl_codegen() {
    let mlir = r#"
module @weight_surgery_demo {
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %w_a = stablehlo.constant dense<[
      [1.0, 0.05, 2.0, -0.02],
      [0.8, -0.01, 1.5, 0.03],
      [1.2, 0.02, 2.5, -0.01]
    ]> : tensor<3x4xf32>
    %w_b = stablehlo.constant dense<[
      [1.4, -0.03, 1.8, 0.01],
      [1.0, 0.02, 1.7, -0.02],
      [0.9, 0.01, 2.1, 0.04]
    ]> : tensor<3x4xf32>
    %w_merged = stablehlo.custom_call %w_a, %w_b {call_target_name = "castforge.merge"}
      : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    %out = stablehlo.dot_general %x, %w_merged
      {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]}
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
"#;

    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(mlir).expect("parse failed");
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "merge-weights(method=slerp,t=0.5),prune-structured(sparsity=0.5),quantize-weights(bits=8)",
    )
    .expect("pipeline build failed");
    pm.run(&mut module).expect("passes failed");

    let plan = build_memory_plan(&module);
    // Quantized 3x2 i8 weight buffer = 6 bytes!
    assert_eq!(plan.total_constant_bytes, 6);

    let rust_code = generate_code(&module, CodegenTarget::Rust).expect("rust codegen failed");
    assert!(rust_code.contains("pub fn forward(x: &[f32; 6]) -> [f32; 4]"));
    assert!(rust_code.contains("let w_merged: [i8; 6]"));

    let c_code = generate_code(&module, CodegenTarget::C).expect("c codegen failed");
    assert!(c_code.contains("void forward(const float* restrict x, float* restrict out_buf)"));

    let wgsl_code = generate_code(&module, CodegenTarget::Wgsl).expect("wgsl codegen failed");
    assert!(wgsl_code.contains("@compute @workgroup_size(64)"));
}

#[test]
fn test_liveness_and_arena_buffer_reuse_savings() {
    let mlir = r#"
module @arena_reuse_demo {
  func.func @chain(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %t1 = stablehlo.exponential %x : tensor<4x4xf32>
    %t2 = stablehlo.tanh %t1 : tensor<4x4xf32>
    %t3 = stablehlo.rsqrt %t2 : tensor<4x4xf32>
    %t4 = stablehlo.exponential %t3 : tensor<4x4xf32>
    func.return %t4 : tensor<4x4xf32>
  }
}
"#;
    let module = parse_mlir(mlir).expect("parse chain mlir");
    let plan = build_memory_plan(&module);

    // 1 input arg (64 B) + 4 intermediate 4x4xf32 activations (4 * 64 B = 256 B) = 320 B naive
    assert_eq!(plan.naive_activation_bytes, 320);
    // With liveness-guided ping-pong reuse, only 2 arena slots (2 * 64 B = 128 B) are needed
    // for the 4 sequential temporaries (%t3 reuses %t1's slot; %t4 reuses %t2's slot)!
    assert_eq!(plan.reusable_slot_count, 2);
    assert_eq!(plan.total_activation_bytes, 192);
    assert_eq!(plan.saved_activation_bytes(), 128);

    let rust_code = generate_code(&module, CodegenTarget::Rust).expect("rust codegen");
    assert!(rust_code.contains("Saved: 128 B across 2 arena slots"));
    assert!(rust_code.contains("// [arena] t1: slot #0 @ offset 0 B"));
    assert!(rust_code.contains("// [arena] t3: slot #0 @ offset 0 B"));
}
