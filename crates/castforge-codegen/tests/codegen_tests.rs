use castforge_codegen::{
    BottleneckKind, CodegenTarget, GemmLoopOrder, HardwareRoofline, TargetConfig,
    analyze_module_cost_with_roofline, build_memory_plan, evaluate_module, generate_code,
    generate_code_with_config, verify_module_numerics,
};
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
    assert!(rust_code.contains("let mut arena = [0.0_f32; 32];"));
    assert!(rust_code.contains("// [arena] t1: slot #0 @ offset 0 B"));
    assert!(rust_code.contains("// [arena] t3: slot #0 @ offset 0 B"));
    assert!(rust_code.contains("arena[16 + i] = (arena[i]).tanh();"));
    assert!(rust_code.contains("arena[i] = 1.0_f32 / (arena[16 + i]).sqrt();"));

    let c_code = generate_code(&module, CodegenTarget::C).expect("c codegen");
    assert!(c_code.contains("float arena[32];"));
    assert!(c_code.contains("float* const t1 = &arena[0];"));
    assert!(c_code.contains("float* const t2 = &arena[16];"));
    assert!(c_code.contains("float* const t3 = &arena[0];"));
    assert!(c_code.contains("float* const t4 = &arena[16];"));
}

#[test]
fn test_ir_interpreter_and_numeric_verification_across_ai_optimizations() {
    let mlir = r#"
module @numeric_verify_demo {
  func.func @norm_linear_relu(
    %x: tensor<2x2xf32>,
    %bias: tensor<2x2xf32>
  ) -> tensor<2x2xf32> {
    %scale = stablehlo.constant dense<[[2.0, 0.5], [2.0, 0.5]]> : tensor<2x2xf32>
    %w = stablehlo.constant dense<[[1.0, -1.0], [2.0, 4.0]]> : tensor<2x2xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %normed = stablehlo.multiply %x, %scale : tensor<2x2xf32>
    %mm = stablehlo.dot_general %normed, %w, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %added = stablehlo.add %mm, %bias : tensor<2x2xf32>
    %act = stablehlo.maximum %added, %zero : tensor<2x2xf32>
    func.return %act : tensor<2x2xf32>
  }
}
"#;
    let before = parse_mlir(mlir).expect("parse mlir");
    let mut after = before.clone();

    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "fold-norm-weights,fuse-matmul-epilogue,fold-constants,dce",
    )
    .expect("build pipeline");
    pm.run(&mut after).expect("run pipeline");

    let rep = verify_module_numerics(&before, &after, 1e-5).expect("verify numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);
    assert!(rep.max_abs_diff < 1e-6);

    // Check explicit evaluation on known inputs
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let bias = vec![0.5_f32, -10.0, 0.0, 1.0];
    let out = evaluate_module(&after, &[x, bias]).expect("eval module");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], vec![4.5_f32, 0.0, 10.0, 3.0]);
}

#[test]
fn test_tiled_and_workgroup_parameterized_codegen() {
    let mlir = r#"
module @tiled_gemm_demo {
  func.func @matmul(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %w = stablehlo.constant dense<1.0> : tensor<4x4xf32>
    %out = stablehlo.dot_general %x, %w : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %out : tensor<4x4xf32>
  }
}
"#;
    let module = parse_mlir(mlir).expect("parse tiled gemm mlir");
    let cpu_cfg = TargetConfig::cpu_tiled(2);
    let rust_tiled =
        generate_code_with_config(&module, CodegenTarget::Rust, &cpu_cfg).expect("tiled rust");
    assert!(rust_tiled.contains("// Hardware Config: Tiled GEMM (2x2x2)"));
    assert!(rust_tiled.contains("for m0 in (0..4).step_by(2)"));
    assert!(rust_tiled.contains("for k0 in (0..4).step_by(2)"));

    let c_tiled = generate_code_with_config(&module, CodegenTarget::C, &cpu_cfg).expect("tiled c");
    assert!(c_tiled.contains("for (int m0 = 0; m0 < 4; m0 += 2)"));
    assert!(c_tiled.contains("for (int k0 = 0; k0 < 4; k0 += 2)"));

    let gpu_cfg = TargetConfig::gpu_workgroup(4, 128);
    let wgsl_tiled =
        generate_code_with_config(&module, CodegenTarget::Wgsl, &gpu_cfg).expect("tiled wgsl");
    assert!(wgsl_tiled.contains("var<workgroup> wg_tile_a: array<f32, 16>;"));
    assert!(wgsl_tiled.contains("@compute @workgroup_size(128)"));
}

#[test]
fn test_roofline_cost_model_before_and_after_fusion() {
    let mlir = r#"
module @roofline_demo {
  func.func @mlp_block(
    %x: tensor<16x64xf32>,
    %w: tensor<64x64xf32>,
    %b: tensor<16x64xf32>
  ) -> tensor<16x64xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<16x64xf32>
    %mm = stablehlo.dot_general %x, %w : (tensor<16x64xf32>, tensor<64x64xf32>) -> tensor<16x64xf32>
    %add = stablehlo.add %mm, %b : tensor<16x64xf32>
    %relu = stablehlo.maximum %add, %zero : tensor<16x64xf32>
    func.return %relu : tensor<16x64xf32>
  }
}
"#;
    let before_mod = parse_mlir(mlir).expect("parse roofline mlir");
    let mut after_mod = before_mod.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&registry, "fuse-matmul-epilogue,dce").expect("pipeline");
    pm.run(&mut after_mod).expect("run fusion");

    let roofline = HardwareRoofline::new(200.0, 50.0); // ridge point = 4.0 FLOP/Byte
    let before_cost = analyze_module_cost_with_roofline(&before_mod, &roofline);
    let after_cost = analyze_module_cost_with_roofline(&after_mod, &roofline);

    assert_eq!(before_cost.total_flops, after_cost.total_flops);
    assert!(after_cost.total_memory_traffic_bytes < before_cost.total_memory_traffic_bytes);
    assert!(after_cost.arithmetic_intensity > before_cost.arithmetic_intensity);
    assert_eq!(after_cost.overall_bottleneck, BottleneckKind::ComputeBound);
    assert!(after_cost.estimated_latency_ns < before_cost.estimated_latency_ns);
    assert!(
        after_cost
            .summary_table()
            .contains("castforge.fused_matmul_epilogue")
    );
}

#[test]
fn test_bufferize_inplace_reduces_arena_to_single_slot() {
    let mlir = r#"
module @inplace_demo {
  func.func @chain(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense<1.0> : tensor<4x4xf32>
    %t1 = stablehlo.add %x, %c : tensor<4x4xf32>
    %t2 = stablehlo.multiply %t1, %c : tensor<4x4xf32>
    %t3 = stablehlo.tanh %t2 : tensor<4x4xf32>
    %t4 = stablehlo.maximum %t3, %c : tensor<4x4xf32>
    func.return %t4 : tensor<4x4xf32>
  }
}
"#;
    let baseline = parse_mlir(mlir).expect("parse inplace mlir");
    let baseline_plan = build_memory_plan(&baseline);
    assert_eq!(baseline_plan.reusable_slot_count, 2);
    assert_eq!(baseline_plan.arena_workspace_bytes, 128);

    let mut inplace_mod = baseline.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&registry, "bufferize-inplace").expect("inplace pipeline");
    let res = pm.run(&mut inplace_mod).expect("run bufferize-inplace");
    assert!(res[0].changed);

    let inplace_plan = build_memory_plan(&inplace_mod);
    assert_eq!(inplace_plan.inplace_reuse_count, 3);
    assert_eq!(inplace_plan.reusable_slot_count, 1);
    assert_eq!(inplace_plan.arena_workspace_bytes, 64);

    let rust_code = generate_code(&inplace_mod, CodegenTarget::Rust).expect("rust inplace");
    assert!(rust_code.contains("[arena:inplace]"));
    assert!(rust_code.contains("let mut arena = [0.0_f32; 16]; // 64 B"));

    let c_code = generate_code(&inplace_mod, CodegenTarget::C).expect("c inplace");
    assert!(c_code.contains("[arena:inplace]"));
    assert!(c_code.contains("float arena[16]; /* 64 B"));
}

#[test]
fn test_loop_interchange_unroll_vectorize_and_fuse_loops() {
    let mlir = r#"
module @loop_transforms_demo {
  func.func @forward(
    %x: tensor<4x5xf32>,
    %w: tensor<5x4xf32>,
    %bias: tensor<4x4xf32>
  ) -> tensor<4x4xf32> {
    %c = stablehlo.constant dense<0.5> : tensor<4x4xf32>
    %mm = stablehlo.dot_general %x, %w : (tensor<4x5xf32>, tensor<5x4xf32>) -> tensor<4x4xf32>
    %s0 = stablehlo.add %mm, %bias : tensor<4x4xf32>
    %s1 = stablehlo.multiply %s0, %c : tensor<4x4xf32>
    %s2 = stablehlo.tanh %s1 : tensor<4x4xf32>
    %s3 = stablehlo.subtract %s2, %c : tensor<4x4xf32>
    func.return %s3 : tensor<4x4xf32>
  }
}
"#;
    let before = parse_mlir(mlir).expect("parse loop mlir");
    let mut fused = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(&registry, "fuse-loops").expect("fuse-loops");
    let res = pm.run(&mut fused).expect("run fuse-loops");
    assert!(res[0].changed);

    let rep = verify_module_numerics(&before, &fused, 1e-5).expect("verify fuse-loops numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // 1. Test MKN (Broadcast-A) loop interchange + SIMD vectorization
    let mkn_cfg = TargetConfig::default()
        .with_loop_order(GemmLoopOrder::Mkn)
        .with_vectorize(true);
    let rust_mkn =
        generate_code_with_config(&fused, CodegenTarget::Rust, &mkn_cfg).expect("rust mkn");
    assert!(rust_mkn.contains("[loop:interchange(MKN)]"));
    assert!(rust_mkn.contains("[simd:vectorize(width=8)]"));
    assert!(rust_mkn.contains("let s0 ="));
    assert!(rust_mkn.contains("let s3 ="));

    let c_mkn = generate_code_with_config(&fused, CodegenTarget::C, &mkn_cfg).expect("c mkn");
    assert!(c_mkn.contains("[loop:interchange(MKN)]"));
    assert!(c_mkn.contains("#pragma omp simd"));
    assert!(c_mkn.contains("float s0 ="));
    assert!(c_mkn.contains("float s3 ="));

    // 2. Test inner-loop unrolling (factor=2) with tail-loop peeling (K=5 -> main 0..4, tail 4..5)
    let unroll_cfg = TargetConfig::default().with_unroll(2);
    let rust_unroll =
        generate_code_with_config(&fused, CodegenTarget::Rust, &unroll_cfg).expect("rust unroll");
    assert!(rust_unroll.contains("[loop:unroll(factor=2)]"));
    assert!(rust_unroll.contains("for k_base in (0..4).step_by(2)"));
    assert!(rust_unroll.contains("for k in 4..5"));

    let c_unroll =
        generate_code_with_config(&fused, CodegenTarget::C, &unroll_cfg).expect("c unroll");
    assert!(c_unroll.contains("[loop:unroll(factor=2)]"));
    assert!(c_unroll.contains("for (int k_base = 0; k_base < 4; k_base += 2)"));
    assert!(c_unroll.contains("for (int k = 4; k < 5; ++k)"));
}
