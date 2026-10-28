use castforge_mlir::{PrintOptions, parse_mlir, print_mlir, print_mlir_with_options};
use castforge_passes::{PassManager, PassRegistry};

const SAMPLE_MLIR: &str = r#"
module @test_mod {
  func.func @main(%arg0: tensor<4x8xf32>, %bias: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    %t1 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    %t2 = stablehlo.transpose %t1, dims = [1, 0] : (tensor<8x4xf32>) -> tensor<4x8xf32>
    %r1 = stablehlo.reshape %t2 : (tensor<4x8xf32>) -> tensor<32xf32>
    %r2 = stablehlo.reshape %r1 : (tensor<32xf32>) -> tensor<4x8xf32>
    %sum = stablehlo.add %r2, %bias : tensor<4x8xf32>
    %dup = stablehlo.add %r2, %bias : tensor<4x8xf32>
    %dead = stablehlo.multiply %dup, %bias : tensor<4x8xf32>
    %out = stablehlo.maximum %sum, %zero : tensor<4x8xf32>
    return %out : tensor<4x8xf32>
  }
}
"#;

#[test]
fn test_dynamic_cli_pipeline_and_fusion() {
    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(SAMPLE_MLIR).expect("parse failed");
    module.verify().expect("initial verify failed");

    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "fold-transpose(eliminate_identity=true),fold-reshape,cse,dce,fuse-elementwise",
    )
    .expect("pipeline build failed");

    let results = pm.run(&mut module).expect("pipeline execution failed");
    assert!(results.iter().any(|r| r.changed));

    let out_mlir = print_mlir(&module);
    // Double transpose and double reshape should be eliminated
    assert!(!out_mlir.contains("stablehlo.transpose"));
    assert!(!out_mlir.contains("stablehlo.reshape"));
    // Dead multiply should be eliminated
    assert!(!out_mlir.contains("stablehlo.multiply"));
    // Add + Maximum should be fused into stablehlo.composite
    assert!(out_mlir.contains("stablehlo.composite"));
}

#[test]
fn test_generic_mlir_roundtrip() {
    let module = parse_mlir(SAMPLE_MLIR).expect("parse failed");
    let generic = print_mlir_with_options(
        &module,
        PrintOptions {
            generic_form: true,
            print_locations: false,
        },
    );
    assert!(generic.contains("\"stablehlo.add\""));
    let reparsed = parse_mlir(&generic).expect("reparse generic failed");
    reparsed.verify().expect("reparse verify failed");
}

#[test]
fn test_constant_folding_and_algebraic_simplification() {
    let mlir = r#"
module @rmsnorm_test {
  func.func @forward(%x: tensor<4x128xf32>, %w: tensor<128x64xf32>) -> tensor<4x64xf32> {
    %c2 = stablehlo.constant dense<2.0> : tensor<4x128xf32>
    %c3 = stablehlo.constant dense<3.0> : tensor<4x128xf32>
    %c5 = stablehlo.add %c2, %c3 : tensor<4x128xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<4x128xf32>
    %one = stablehlo.constant dense<1.0> : tensor<4x128xf32>
    %x_plus_0 = stablehlo.add %x, %zero : tensor<4x128xf32>
    %x_mul_1 = stablehlo.multiply %x_plus_0, %one : tensor<4x128xf32>
    %x_conv = stablehlo.convert %x_mul_1 : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %scaled = stablehlo.multiply %x_conv, %c5 : tensor<4x128xf32>
    %init = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%scaled init: %init) applies stablehlo.add across dimensions = [1] : (tensor<4x128xf32>, tensor<f32>) -> tensor<4xf32>
    %out = stablehlo.dot_general %scaled, %w {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]} : (tensor<4x128xf32>, tensor<128x64xf32>) -> tensor<4x64xf32>
    return %out : tensor<4x64xf32>
  }
}
"#;

    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(mlir).expect("parse failed");
    let mut pm = PassManager::from_pipeline_str(&registry, "verify-shapes,canonicalize")
        .expect("pipeline build failed");
    pm.run(&mut module).expect("pipeline run failed");

    let printed = print_mlir(&module);
    // 2.0 + 3.0 should fold to dense<5.0>
    assert!(printed.contains("dense<5.0>"));
    // No-op convert, x+0, x*1, and unused reduce should all be eliminated!
    assert!(!printed.contains("stablehlo.convert"));
    assert!(!printed.contains("stablehlo.reduce"));
    assert!(printed.contains("stablehlo.dot_general"));
}

#[test]
fn test_merge_prune_and_quantize_pipeline() {
    let mlir = r#"
module @weight_surgery {
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %w_a = stablehlo.constant dense<[[1.0, 0.05, 2.0, -0.02], [0.8, -0.01, 1.5, 0.03], [1.2, 0.02, 2.5, -0.01]]> : tensor<3x4xf32>
    %w_b = stablehlo.constant dense<[[1.4, -0.03, 1.8, 0.01], [1.0, 0.02, 1.7, -0.02], [0.9, 0.01, 2.1, 0.04]]> : tensor<3x4xf32>
    %w_merged = stablehlo.custom_call %w_a, %w_b {call_target_name = "castforge.merge"} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    %out = stablehlo.dot_general %x, %w_merged {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
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

    pm.run(&mut module).expect("pipeline execution failed");

    let printed = print_mlir(&module);
    // 1. Donor constants w_a and w_b were merged into w_merged and DCE'd
    assert!(!printed.contains("castforge.merge"));
    // 2. Structured pruning reduced columns from 4 -> 2 (tensor<3x2x...> and output tensor<2x2xf32>)
    assert!(printed.contains("-> tensor<2x2xf32>"));
    // 3. Quantization converted the 3x2 weight matrix into !quant.uniform<i8:f32, ...> + uniform_dequantize
    assert!(printed.contains("!quant.uniform<i8:f32,"));
    assert!(printed.contains("stablehlo.uniform_dequantize"));

    // Verify the quantized & pruned MLIR round-trips through the parser!
    let reparsed = parse_mlir(&printed).expect("reparse quantized mlir failed");
    reparsed.verify().expect("verify reparsed failed");
}

#[test]
fn test_run_with_trace_captures_every_pass_step() {
    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(SAMPLE_MLIR).expect("parse failed");
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "fold-transpose,fold-reshape,cse,dce,fuse-elementwise",
    )
    .expect("pipeline build failed");

    let steps = pm.run_with_trace(&mut module).expect("trace failed");
    // Initial IR + 5 passes = 6 steps
    assert_eq!(steps.len(), 6);
    assert_eq!(steps[0].label, "Initial IR");
    assert_eq!(steps[1].label, "fold-transpose");
    assert_eq!(steps[5].label, "fuse-elementwise");
}
