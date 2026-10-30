use castforge::mlir::{PrintOptions, parse_mlir, print_mlir, print_mlir_with_options};
use castforge::passes::{PassManager, PassRegistry};

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

    let mut pm = PassManager::from_pipeline_str(&registry, "fold-views,cse,dce,fuse-elementwise")
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
    %red = stablehlo.reduce(%scaled init: %init)
      applies stablehlo.add across dimensions = [1]
      : (tensor<4x128xf32>, tensor<f32>) -> tensor<4xf32>
    %out = stablehlo.dot_general %scaled, %w
      {lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]}
      : (tensor<4x128xf32>, tensor<128x64xf32>) -> tensor<4x64xf32>
    return %out : tensor<4x64xf32>
  }
}
"#;

    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(mlir).expect("parse failed");
    let mut pm = PassManager::from_pipeline_str(&registry, "verify-ir,canonicalize")
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
        "merge-weights(method=slerp,t=0.5),fold-weights(mode=prune,sparsity=0.5),quantize(bits=8)",
    )
    .expect("pipeline build failed");

    pm.run(&mut module).expect("pipeline execution failed");

    let printed = print_mlir(&module);
    // 1. Donor constants w_a and w_b were merged into w_merged and DCE'd
    assert!(!printed.contains("castforge.merge"));
    // 2. Structured pruning reduced columns from 4 -> 2 (output tensor<2x2xf32>)
    assert!(printed.contains("-> tensor<2x2xf32>"));
    // 3. Quantization converted the 3x2 weight matrix into !quant.uniform + uniform_dequantize
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
    let mut pm = PassManager::from_pipeline_str(&registry, "fold-views,cse,dce,fuse-elementwise")
        .expect("pipeline build failed");

    let steps = pm.run_with_trace(&mut module).expect("trace failed");
    // Initial IR + 4 passes = 5 steps
    assert_eq!(steps.len(), 5);
    assert_eq!(steps[0].label, "Initial IR");
    assert_eq!(steps[1].label, "fold-views");
    assert_eq!(steps[4].label, "fuse-elementwise");
}

#[test]
fn test_mlir_general_passes_inliner_symbol_dce_sccp_toposort_and_debuginfo() {
    let mlir = r#"
module @general_passes {
  func.func private @helper(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %sum = stablehlo.add %a, %b : tensor<4x8xf32> loc("helper_add")
    return %sum : tensor<4x8xf32>
  }

  func.func private @unused_fn(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
    return %x : tensor<4x8xf32>
  }

  func.func @main(%arg0: tensor<4x8xf32>, %bias: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %call_res = func.call @helper(%arg0, %bias)
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32> loc("call_op")
    %c10 = stablehlo.constant dense<10.0> : tensor<4x8xf32>
    %c5 = stablehlo.constant dense<5.0> : tensor<4x8xf32>
    %cmp = stablehlo.compare %c10, %c5, GT
      : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xi1>
    %sel = stablehlo.select %cmp, %call_res, %arg0
      : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    return %sel : tensor<4x8xf32>
  }
}
"#;

    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(mlir).expect("parse failed");
    assert_eq!(module.functions().len(), 3);

    let mut pm = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "ir-edit(mode=inline),dce,simplify-algebraic(mode=constants),",
            "canonicalize(mode=toposort),ir-edit(debuginfo=true),",
            "verify-ir(profile=true,equiv=false,hash=false)"
        ),
    )
    .expect("pipeline build failed");

    pm.run(&mut module).expect("pipeline execution failed");

    // Both private functions (@helper and @unused_fn) should be eliminated by symbol-dce!
    assert_eq!(module.functions().len(), 1);

    let printed = print_mlir_with_options(
        &module,
        PrintOptions {
            generic_form: false,
            print_locations: true,
        },
    );
    // func.call was inlined into stablehlo.add
    assert!(!printed.contains("func.call"));
    assert!(printed.contains("stablehlo.add"));
    // SCCP folded compare(10 > 5) -> true and select(true, %call_res, %arg0) -> %call_res
    assert!(!printed.contains("stablehlo.compare"));
    assert!(!printed.contains("stablehlo.select"));
    // strip-debuginfo reset all named locations to loc(unknown)
    assert!(!printed.contains("helper_add"));
}

#[test]
fn test_unified_dce_and_fold_constants() {
    let mlir = r#"
module @unified_passes {
  func.func private @add_vec(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %sum = stablehlo.add %a, %b : tensor<4xf32>
    return %sum : tensor<4xf32>
  }

  func.func private @dead_private(%x: tensor<4xf32>) -> tensor<4xf32> {
    return %x : tensor<4xf32>
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %v1 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %v2 = stablehlo.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
    %vsum = func.call @add_vec(%v1, %v2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %c1 = stablehlo.constant dense<2.0> : tensor<4xf32>
    %c2 = stablehlo.constant dense<1.0> : tensor<4xf32>
    %cmp = stablehlo.compare %c1, %c2, GT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %sel = stablehlo.select %cmp, %vsum, %arg0
      : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %sel : tensor<4xf32>
  }
}
"#;

    let registry = PassRegistry::with_builtin_passes();
    let mut module = parse_mlir(mlir).expect("parse failed");
    assert_eq!(module.functions().len(), 3);

    // Running just `inliner,fold-constants,dce` (or `inliner,canonicalize`) should:
    // 1. Inline @add_vec
    // 2. Fold dense<[1,2,3,4]> + dense<[10,20,30,40]> -> dense<[11,22,33,44]>
    // 3. Fold compare(2.0 > 1.0) and select -> %vsum
    // 4. Eliminate both dead private functions (@add_vec and @dead_private) via unified `dce`
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "ir-edit(mode=inline),simplify-algebraic(mode=constants),dce",
    )
    .expect("pipeline build failed");
    pm.run(&mut module).expect("pipeline execution failed");

    assert_eq!(module.functions().len(), 1);
    let printed = print_mlir(&module);
    assert!(!printed.contains("func.call"));
    assert!(!printed.contains("stablehlo.add"));
    assert!(!printed.contains("stablehlo.compare"));
    assert!(!printed.contains("stablehlo.select"));
    assert!(printed.contains("dense<[11.0, 22.0, 33.0, 44.0]>"));
}

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
    let mut ext_pm = PassManager::from_pipeline_str(&registry, "fold-weights(mode=externalize)")
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
         fold-weights(mode=structured,sparsity=0.5),\
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

    let mut inline_pm = PassManager::from_pipeline_str(&registry, "fold-weights(mode=inline)")
        .expect("inline pipeline build failed");
    inline_pm.run(&mut final_mod).expect("inline run failed");

    let inlined_mlir = print_mlir(&final_mod);
    assert!(!inlined_mlir.contains("dense_resource<"));
    assert!(inlined_mlir.contains("dense<[["));
}
