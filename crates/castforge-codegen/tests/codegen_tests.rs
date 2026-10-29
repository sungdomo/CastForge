use castforge_codegen::{
    BottleneckKind, CodegenTarget, GemmLoopOrder, HardwareRoofline, SimdIsa, TargetConfig,
    analyze_module_cost_with_roofline, build_memory_plan, evaluate_module, generate_code,
    generate_code_with_config, verify_module_numerics,
};
use castforge_mlir::{parse_mlir, print_mlir};
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

#[test]
fn test_unroll_and_jam_register_microkernel_collapse_parallel_and_fission() {
    let mlir = r#"
module @advanced_loops_demo {
  func.func @forward(
    %x: tensor<4x4xf32>,
    %cos: tensor<4x4xf32>,
    %sin: tensor<4x4xf32>,
    %w: tensor<4x4xf32>,
    %bias: tensor<4x4xf32>
  ) -> tensor<4x4xf32> {
    %zero_half = stablehlo.constant dense<0.0> : tensor<4x2xf32>
    %c = stablehlo.constant dense<0.5> : tensor<4x4xf32>
    %x_lo = stablehlo.slice %x
      {start_indices = [0, 0], limit_indices = [4, 2], strides = [1, 1]}
      : (tensor<4x4xf32>) -> tensor<4x2xf32>
    %x_hi = stablehlo.slice %x
      {start_indices = [0, 2], limit_indices = [4, 4], strides = [1, 1]}
      : (tensor<4x4xf32>) -> tensor<4x2xf32>
    %neg_hi = stablehlo.subtract %zero_half, %x_hi : tensor<4x2xf32>
    %x_rot = stablehlo.concatenate %neg_hi, %x_lo {dimension = 1}
      : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x4xf32>
    %x_cos = stablehlo.multiply %x, %cos : tensor<4x4xf32>
    %x_sin = stablehlo.multiply %x_rot, %sin : tensor<4x4xf32>
    %rope = stablehlo.add %x_cos, %x_sin : tensor<4x4xf32>

    %mm = stablehlo.dot_general %rope, %w : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %s0 = stablehlo.add %mm, %bias : tensor<4x4xf32>
    %s1 = stablehlo.multiply %s0, %c : tensor<4x4xf32>
    %s2 = stablehlo.tanh %s1 : tensor<4x4xf32>
    %s3 = stablehlo.subtract %s2, %c : tensor<4x4xf32>
    func.return %s3 : tensor<4x4xf32>
  }
}
"#;
    let before = parse_mlir(mlir).expect("parse advanced loops mlir");
    let mut transformed = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "fuse-rope,fuse-loops,fission-loops(max_steps=2)",
    )
    .expect("fission pipeline");
    let res = pm.run(&mut transformed).expect("run pipeline");
    assert!(res[0].changed);
    assert!(res[1].changed);
    assert!(res[2].changed);

    let rep = verify_module_numerics(&before, &transformed, 1e-5).expect("verify fission");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    let cfg = TargetConfig::default()
        .with_register_tile(2, 2)
        .with_collapse_loops(true)
        .with_parallel(true);

    let rust_code =
        generate_code_with_config(&transformed, CodegenTarget::Rust, &cfg).expect("rust code");
    assert!(rust_code.contains("[loop:parallel(collapse=2, schedule=static)]"));
    assert!(rust_code.contains("[loop:unroll_and_jam(mr=2, nr=2)]"));
    assert!(rust_code.contains("let mut acc = [[0.0_f32; 2]; 2];"));
    assert!(rust_code.contains("[loop:coalesce(2D->1D, len=16)]"));

    let c_code = generate_code_with_config(&transformed, CodegenTarget::C, &cfg).expect("c code");
    assert!(c_code.contains("#pragma omp parallel for collapse(2) schedule(static)"));
    assert!(c_code.contains("[loop:unroll_and_jam(mr=2, nr=2)]"));
    assert!(c_code.contains("float acc[2][2] = {0};"));
    assert!(c_code.contains("[loop:coalesce(2D->1D, len=16)]"));
}

#[test]
fn test_masked_attention_map_reduce_log_softmax_and_polyhedral_loops() {
    let mlir = r#"
module @complete_fusion_and_polyhedral {
  func.func @forward(
    %q: tensor<2x4xf32>,
    %k: tensor<2x4xf32>,
    %v: tensor<2x4xf32>,
    %mask: tensor<2x2xf32>,
    %w: tensor<4x4xf32>
  ) -> tensor<2xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %scale = stablehlo.constant dense<0.5> : tensor<2x2xf32>

    // 1. Masked Attention: softmax((Q @ K^T) * scale + mask) @ V
    %kt = stablehlo.transpose %k, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %qk = stablehlo.dot_general %q, %kt, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    %scaled = stablehlo.multiply %qk, %scale : tensor<2x2xf32>
    %masked = stablehlo.add %scaled, %mask : tensor<2x2xf32>
    %exp = stablehlo.exponential %masked : tensor<2x2xf32>
    %sum = stablehlo.reduce %exp, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %denom = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x2xf32>
    %probs = stablehlo.divide %exp, %denom : tensor<2x2xf32>
    %attn = stablehlo.dot_general %probs, %v, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // 2. GEMM + LogSoftmax + Map-Reduce (reduce_sum(square(log_softmax(mm))))
    %mm = stablehlo.dot_general %attn, %w, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %e2 = stablehlo.exponential %mm : tensor<2x4xf32>
    %s2 = stablehlo.reduce %e2, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %d2 = stablehlo.broadcast_in_dim %s2, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %sm2 = stablehlo.divide %e2, %d2 : tensor<2x4xf32>
    %lsm = stablehlo.log %sm2 : tensor<2x4xf32>
    %sq = stablehlo.multiply %lsm, %lsm : tensor<2x4xf32>
    %out = stablehlo.reduce %sq, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    func.return %out : tensor<2xf32>
  }
}
"#;
    let before = parse_mlir(mlir).expect("parse complete mlir");
    let mut fused = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "fuse-attention,fuse-log-softmax,fuse-map-reduce",
    )
    .expect("complete fusion pipeline");
    let res = pm.run(&mut fused).expect("run complete fusion");
    assert!(res[0].changed);
    assert!(res[1].changed);
    assert!(res[2].changed);

    let printed = print_mlir(&fused);
    assert!(printed.contains("castforge.fused_attention"));
    assert!(printed.contains("masked = true"));
    assert!(printed.contains("castforge.fused_log_softmax"));
    assert!(printed.contains("castforge.fused_map_reduce"));

    let rep = verify_module_numerics(&before, &fused, 1e-5).expect("verify complete fusion");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Verify decompose-composites round-trip on map_reduce + log_softmax
    let mut decomposed = fused.clone();
    let mut dec_pm =
        PassManager::from_pipeline_str(&registry, "decompose-composites").expect("decompose");
    dec_pm.run(&mut decomposed).expect("run decompose");
    let dec_rep = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(dec_rep.passed, "max_abs_diff = {}", dec_rep.max_abs_diff);

    // Verify Split-K, L2 tiling, and Software Pipelining codegen
    let cfg_sk = TargetConfig::cpu_tiled(2)
        .with_l2_tile_size(4)
        .with_prefetch(4)
        .with_split_k(2);
    let rust_sk =
        generate_code_with_config(&fused, CodegenTarget::Rust, &cfg_sk).expect("rust split-k");
    assert!(rust_sk.contains("[loop:l2_tile(L2=4, L1=2)]"));
    assert!(rust_sk.contains("[loop:software_pipeline(prefetch_dist=4)]"));
    assert!(rust_sk.contains("[loop:split_k(splits=2, chunk=2)]"));
    assert!(rust_sk.contains("let lse = sum.ln();"));

    let c_sk = generate_code_with_config(&fused, CodegenTarget::C, &cfg_sk).expect("c split-k");
    assert!(c_sk.contains("[loop:l2_tile(L2=4, L1=2)]"));
    assert!(c_sk.contains("__builtin_prefetch"));
    assert!(c_sk.contains("float acc_parts[2] = {0};"));
    assert!(c_sk.contains("float lse = logf(sum);"));

    // Verify Polyhedral Wavefront Loop Skewing codegen
    let cfg_skew = TargetConfig::default().with_loop_skew(true);
    let rust_skew =
        generate_code_with_config(&fused, CodegenTarget::Rust, &cfg_skew).expect("rust skew");
    assert!(rust_skew.contains("[loop:skew(wavefront=\"w = m + n\")]"));
    let c_skew = generate_code_with_config(&fused, CodegenTarget::C, &cfg_skew).expect("c skew");
    assert!(c_skew.contains("[loop:skew(wavefront=\"w = m + n\")]"));
}

#[test]
fn test_autotune_lower_conv_to_gemm_and_llama_block() {
    let conv_mlir = r#"
module @conv_model {
  func.func @main(
    %x: tensor<1x4x4x2xf32>,
    %w: tensor<3x3x2x4xf32>,
    %b: tensor<4x4xf32>
  ) -> tensor<1x2x2x4xf32> {
    %conv = stablehlo.convolution %x, %w {window_strides = [1, 1], padding = [0, 0]}
      : (tensor<1x4x4x2xf32>, tensor<3x3x2x4xf32>) -> tensor<1x2x2x4xf32>
    %flat = stablehlo.reshape %conv : (tensor<1x2x2x4xf32>) -> tensor<4x4xf32>
    %biased = stablehlo.add %flat, %b : tensor<4x4xf32>
    %out = stablehlo.reshape %biased : (tensor<4x4xf32>) -> tensor<1x2x2x4xf32>
    func.return %out : tensor<1x2x2x4xf32>
  }
}
"#;
    let before = parse_mlir(conv_mlir).expect("parse conv_mlir");
    let mut lowered = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "lower-conv-to-gemm,fold-reshape,fuse-matmul-epilogue,dce",
    )
    .expect("conv pipeline");
    pm.run(&mut lowered).expect("run conv pipeline");

    let printed = print_mlir(&lowered);
    assert!(printed.contains("castforge.im2col"));
    assert!(printed.contains("castforge.fused_matmul_epilogue"));

    let rep = verify_module_numerics(&before, &lowered, 1e-5).expect("verify conv numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Verify Roofline schedule auto-tuner
    let roofline = castforge_codegen::HardwareRoofline::new(1000.0, 100.0);
    let tune = castforge_codegen::autotune_module(&lowered, &roofline);
    assert!(!tune.candidates.is_empty());
    assert!(tune.best.speedup_over_baseline >= 1.0);
    let table = tune.summary_table(3);
    assert!(table.contains("Recommended Schedule:"));
    assert!(table.contains("--loop-order"));

    // Verify codegen on lowered conv + autotuned schedule
    let rust_conv =
        generate_code_with_config(&lowered, CodegenTarget::Rust, &tune.best.config).expect("rust");
    assert!(rust_conv.contains("let ih = (y * 1 + r) as isize - 0;"));
    let c_conv =
        generate_code_with_config(&lowered, CodegenTarget::C, &tune.best.config).expect("c");
    assert!(c_conv.contains("int ih = y * 1 + r - 0;"));

    // Verify end-to-end LLaMA Block showcase (examples/llama_block.mlir)
    let llama_src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/llama_block.mlir"
    ))
    .expect("read examples/llama_block.mlir");
    let llama_before = parse_mlir(&llama_src).expect("parse llama_block");
    let mut llama_opt = llama_before.clone();
    let mut llama_pm = PassManager::from_pipeline_str(&registry, "O2").expect("O2");
    llama_pm.run(&mut llama_opt).expect("run optimize on llama");

    let llama_ir = print_mlir(&llama_opt);
    assert!(llama_ir.contains("castforge.fused_residual_rmsnorm"));
    assert!(llama_ir.contains("castforge.fused_rope"));
    assert!(llama_ir.contains("castforge.fused_attention"));
    assert!(llama_ir.contains("castforge.fused_gated_mlp"));

    let llama_rep =
        verify_module_numerics(&llama_before, &llama_opt, 1e-4).expect("verify llama numerics");
    assert!(
        llama_rep.passed,
        "max_abs_diff = {}",
        llama_rep.max_abs_diff
    );
}

#[test]
fn test_frontend_importers_autodiff_and_native_jit() {
    use castforge_codegen::{autotune_module_jit, jit_execute_and_benchmark};
    use castforge_mlir::{export_gguf_bytes, import_gguf_bytes, import_onnx_json};

    // 1. Test ONNX-JSON Frontend Importer + GGUF v3 Binary Round-Trip
    let onnx_json = r#"{
      "name": "onnx_mlp",
      "inputs": [{ "name": "x", "shape": [2, 4], "dtype": "f32" }],
      "initializers": [
        {
          "name": "w1",
          "shape": [4, 4],
          "values": [
            0.5, 0.1, -0.2, 0.3,
            0.2, 0.4, 0.1, -0.1,
            -0.3, 0.2, 0.6, 0.1,
            0.1, -0.2, 0.2, 0.5
          ]
        },
        {
          "name": "b1",
          "shape": [2, 4],
          "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        }
      ],
      "nodes": [
        {
          "op_type": "Gemm",
          "inputs": ["x", "w1", "b1"],
          "outputs": ["h1"],
          "output_shape": [2, 4]
        },
        {
          "op_type": "Relu",
          "inputs": ["h1"],
          "outputs": ["y"],
          "output_shape": [2, 4]
        }
      ],
      "outputs": [{ "name": "y", "shape": [2, 4], "dtype": "f32" }]
    }"#;

    let onnx_mod = import_onnx_json(onnx_json).expect("import onnx json");
    assert_eq!(onnx_mod.tensor_store.len(), 2);
    let mut onnx_opt = onnx_mod.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&registry, "fuse-matmul-epilogue,dce").expect("pipeline");
    pm.run(&mut onnx_opt).expect("run fuse-matmul-epilogue");
    assert!(print_mlir(&onnx_opt).contains("castforge.fused_matmul_epilogue"));

    let gguf_bytes = export_gguf_bytes(&onnx_mod);
    assert_eq!(&gguf_bytes[0..4], b"GGUF");
    let gguf_mod = import_gguf_bytes(&gguf_bytes).expect("import gguf bytes");
    assert_eq!(gguf_mod.tensor_store.len(), 2);
    assert!(print_mlir(&gguf_mod).contains("stablehlo.dot_general"));

    // 2. Test Reverse-Mode Automatic Differentiation (autodiff & autodiff(lr=0.1))
    let grad_mlir = r#"
module @autodiff_demo {
  func.func @loss(%x: tensor<2x2xf32>, %w: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %mm = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %act = stablehlo.tanh %mm : tensor<2x2xf32>
    func.return %act : tensor<2x2xf32>
  }
}
"#;
    let fwd_mod = parse_mlir(grad_mlir).expect("parse grad_mlir");
    let mut vjp_mod = fwd_mod.clone();
    let mut ad_pm = PassManager::from_pipeline_str(&registry, "autodiff").expect("autodiff");
    let ad_res = ad_pm.run(&mut vjp_mod).expect("run autodiff");
    assert!(ad_res[0].changed);
    let vjp_ir = print_mlir(&vjp_mod);
    assert!(vjp_ir.contains("autodiff_role = \"vjp_grad\""));

    // Verify analytical VJP gradient dL/dW against finite-difference gradient
    let x_in = vec![0.5_f32, -0.3, 0.2, 0.8];
    let w_in = vec![0.4_f32, 0.1, -0.2, 0.6];
    let vjp_out =
        evaluate_module(&vjp_mod, &[x_in.clone(), w_in.clone()]).expect("eval vjp")[0].clone();
    let eps = 1e-3_f32;
    for i in 0..4 {
        let mut w_plus = w_in.clone();
        let mut w_minus = w_in.clone();
        w_plus[i] += eps;
        w_minus[i] -= eps;
        let loss_p: f32 = evaluate_module(&fwd_mod, &[x_in.clone(), w_plus]).expect("p")[0]
            .iter()
            .sum();
        let loss_m: f32 = evaluate_module(&fwd_mod, &[x_in.clone(), w_minus]).expect("m")[0]
            .iter()
            .sum();
        let fd_grad = (loss_p - loss_m) / (2.0 * eps);
        assert!(
            (vjp_out[i] - fd_grad).abs() < 1e-3,
            "grad mismatch at {i}: vjp={} vs fd={fd_grad}",
            vjp_out[i]
        );
    }

    // Verify SGD step via autodiff(lr=0.1)
    let mut sgd_mod = fwd_mod.clone();
    let mut sgd_pm =
        PassManager::from_pipeline_str(&registry, "autodiff(lr=0.1)").expect("autodiff lr");
    sgd_pm.run(&mut sgd_mod).expect("run sgd");
    let w_updated =
        evaluate_module(&sgd_mod, &[x_in.clone(), w_in.clone()]).expect("eval sgd")[0].clone();
    for i in 0..4 {
        let expected = w_in[i] - 0.1 * vjp_out[i];
        assert!((w_updated[i] - expected).abs() < 1e-5);
    }

    // 3. Test Native CPU JIT Execution (Rust & C99) and Hardware-in-the-Loop Auto-Tuning
    let rust_jit = jit_execute_and_benchmark(
        &onnx_opt,
        CodegenTarget::Rust,
        &TargetConfig::default(),
        2,
        5,
    )
    .expect("rust jit benchmark");
    assert!(
        rust_jit.numerics_verified,
        "rust jit max_abs_diff = {}",
        rust_jit.max_abs_diff
    );
    assert!(rust_jit.min_latency_ns > 0.0);

    let c_jit =
        jit_execute_and_benchmark(&onnx_opt, CodegenTarget::C, &TargetConfig::default(), 2, 5)
            .expect("c jit benchmark");
    assert!(
        c_jit.numerics_verified,
        "c jit max_abs_diff = {}",
        c_jit.max_abs_diff
    );

    let roofline = HardwareRoofline::new(1000.0, 100.0);
    let jit_tune = autotune_module_jit(&onnx_opt, &roofline, 2).expect("autotune jit");
    assert!(jit_tune.best.estimated_latency_ns > 0.0);
}

#[test]
fn test_rematerialize_shard_spmd_inject_kv_cache_and_egraph_simplify() {
    let mlir = r#"
module @advanced_compiler_frontiers {
  func.func @forward(
    %q: tensor<2x4xf32>,
    %k: tensor<2x4xf32>,
    %v: tensor<2x4xf32>,
    %w1: tensor<4x4xf32>,
    %w2: tensor<4x4xf32>
  ) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %scale = stablehlo.constant dense<0.5> : tensor<2x2xf32>

    // Cheap activation %cheap used both early and late across heavy ops
    %cheap = stablehlo.tanh %q : tensor<2x4xf32>

    // SDPA Attention -> fused + KV-cache injected
    %kt = stablehlo.transpose %k, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %qk = stablehlo.dot_general %cheap, %kt, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    %scores = stablehlo.multiply %qk, %scale : tensor<2x2xf32>
    %ex = stablehlo.exponential %scores : tensor<2x2xf32>
    %sum = stablehlo.reduce %ex, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %denom = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x2xf32>
    %probs = stablehlo.divide %ex, %denom : tensor<2x2xf32>
    %attn = stablehlo.dot_general %probs, %v, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Column-Parallel + Row-Parallel GEMMs -> sharded via shard-spmd(devices=4)
    %h1 = stablehlo.dot_general %attn, %w1, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %h2 = stablehlo.dot_general %h1, %w2, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>

    // Distributive factoring + FMA: (attn * h1 + attn * h2) + cheap -> fma(attn, h1 + h2, cheap)
    %p1 = stablehlo.multiply %attn, %h1 : tensor<2x4xf32>
    %p2 = stablehlo.multiply %attn, %h2 : tensor<2x4xf32>
    %dist = stablehlo.add %p1, %p2 : tensor<2x4xf32>
    %out = stablehlo.add %dist, %cheap : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    let before = parse_mlir(mlir).expect("parse frontiers mlir");
    let mut after = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "inject-kv-cache(max_seq_len=4096),shard-spmd(devices=4),egraph-simplify,rematerialize",
    )
    .expect("frontiers pipeline");
    let res = pm.run(&mut after).expect("run frontiers pipeline");
    assert!(res[0].changed);
    assert!(res[1].changed);
    assert!(res[2].changed);
    assert!(res[3].changed);

    let ir = print_mlir(&after);
    assert!(ir.contains("castforge.kv_cache_update"));
    assert!(ir.contains("max_seq_len = 4096"));
    assert!(ir.contains("spmd_sharding = \"column_parallel\""));
    assert!(ir.contains("spmd_sharding = \"row_parallel\""));
    assert!(ir.contains("castforge.all_reduce"));
    assert!(ir.contains("castforge.fma"));
    assert!(ir.contains("rematerialized = true"));

    let rep = verify_module_numerics(&before, &after, 1e-5).expect("verify frontiers numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Verify Rust and C99 codegen emit hardware FMA (mul_add / fmaf)
    let rust_code = generate_code(&after, CodegenTarget::Rust).expect("rust fma");
    assert!(rust_code.contains(".mul_add("));
    let c_code = generate_code(&after, CodegenTarget::C).expect("c fma");
    assert!(c_code.contains("fmaf("));

    // Verify decompose-composites lowers castforge.fma back to multiply + add
    let mut decomposed = after.clone();
    let mut dec_pm =
        PassManager::from_pipeline_str(&registry, "decompose-composites").expect("decompose");
    dec_pm.run(&mut decomposed).expect("run decompose");
    let dec_rep = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(dec_rep.passed);
}

#[test]
fn test_mixed_precision_fuse_moe_flash_attention_and_double_buffer() {
    let moe_mlir = r#"
module @moe_block {
  func.func @main(
    %x: tensor<2x4xf32>,
    %w0: tensor<4x4xf32>,
    %w1: tensor<4x4xf32>,
    %g0: tensor<2x4xf32>,
    %g1: tensor<2x4xf32>,
    %k: tensor<2x4xf32>,
    %v: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %e0 = stablehlo.dot_general %x, %w0, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %e1 = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %m0 = stablehlo.multiply %e0, %g0 : tensor<2x4xf32>
    %m1 = stablehlo.multiply %e1, %g1 : tensor<2x4xf32>
    %moe = stablehlo.add %m0, %m1 : tensor<2x4xf32>

    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %scale = stablehlo.constant dense<0.5> : tensor<2x2xf32>
    %kt = stablehlo.transpose %k, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %qk = stablehlo.dot_general %moe, %kt, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    %scores = stablehlo.multiply %qk, %scale : tensor<2x2xf32>
    %ex = stablehlo.exponential %scores : tensor<2x2xf32>
    %sum = stablehlo.reduce %ex, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>
    %denom = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x2xf32>
    %probs = stablehlo.divide %ex, %denom : tensor<2x2xf32>
    %out = stablehlo.dot_general %probs, %v, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    let before = parse_mlir(moe_mlir).expect("parse moe_block");
    let mut fused = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(&registry, "fuse-moe,fuse-attention")
        .expect("moe+attn pipeline");
    let res = pm.run(&mut fused).expect("run moe+attn");
    assert!(res[0].changed);
    assert!(res[1].changed);

    let ir = print_mlir(&fused);
    assert!(ir.contains("castforge.fused_moe"));
    assert!(ir.contains("castforge.fused_attention"));

    let rep = verify_module_numerics(&before, &fused, 1e-5).expect("verify moe+attn");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Verify FlashAttention-2 online softmax and fused MoE in Rust & C99 codegen
    let rust_code = generate_code(&fused, CodegenTarget::Rust).expect("rust moe+flash");
    assert!(rust_code.contains("[moe:fused_routed_experts(num_experts=2)]"));
    assert!(rust_code.contains("[attention:flash_online_softmax(O(1)_workspace"));
    assert!(rust_code.contains("m_prev.max(s)"));
    let c_code = generate_code(&fused, CodegenTarget::C).expect("c moe+flash");
    assert!(c_code.contains("[moe:fused_routed_experts(num_experts=2)]"));
    assert!(c_code.contains("[attention:flash_online_softmax(O(1)_workspace"));

    // Verify decompose-composites lowers castforge.fused_moe back to dot_general + multiply + add
    let mut decomposed = fused.clone();
    let mut dec_pm =
        PassManager::from_pipeline_str(&registry, "decompose-composites").expect("decompose");
    dec_pm.run(&mut decomposed).expect("run decompose");
    let dec_rep = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(dec_rep.passed);

    // Verify mixed-precision pass (bf16 & f16) and ping-pong double buffering
    let mut mp_mod = before.clone();
    let mut mp_pm =
        PassManager::from_pipeline_str(&registry, "mixed-precision(dtype=bf16)").expect("mp");
    let mp_res = mp_pm.run(&mut mp_mod).expect("run mp");
    assert!(mp_res[0].changed);
    let mp_ir = print_mlir(&mp_mod);
    assert!(mp_ir.contains("stablehlo.convert"));
    assert!(mp_ir.contains("tensor<2x4xbf16>"));
    assert!(mp_ir.contains("tensor<4x4xbf16>"));
    let mp_rep = verify_module_numerics(&before, &mp_mod, 1e-5).expect("verify mp");
    assert!(mp_rep.passed);

    let cfg_db = TargetConfig::cpu_tiled(2).with_double_buffer(true);
    let rust_db = generate_code_with_config(&before, CodegenTarget::Rust, &cfg_db).expect("r db");
    assert!(rust_db.contains("[loop:double_buffer(stages=2, ping_pong"));
    let c_db = generate_code_with_config(&before, CodegenTarget::C, &cfg_db).expect("c db");
    assert!(c_db.contains("[loop:double_buffer(stages=2, ping_pong"));
}

#[test]
fn test_simd_intrinsics_calibrate_quantize_and_schedule_streams() {
    let mlir = r#"
module @parallel_branches {
  func.func @main(
    %x: tensor<2x4xf32>,
    %w0: tensor<4x4xf32>,
    %w1: tensor<4x4xf32>
  ) -> tensor<2x4xf32> {
    %b0 = stablehlo.dot_general %x, %w0, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %b1 = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %out = stablehlo.add %b0, %b1 : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    let before = parse_mlir(mlir).expect("parse parallel_branches");
    let mut after = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "calibrate-quantize(bits=8,samples=4),schedule-streams(streams=2)",
    )
    .expect("calibrate + streams pipeline");
    let res = pm.run(&mut after).expect("run calibrate + streams");
    assert!(res[0].changed);
    assert!(res[1].changed);

    let ir = print_mlir(&after);
    assert!(ir.contains("castforge.static_quantize"));
    assert!(ir.contains("castforge.stream_wait"));

    // Verify numerical accuracy within INT8 quantization error bound
    let rep = verify_module_numerics(&before, &after, 2e-2).expect("verify w8a8 + streams");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Verify AVX2 & NEON hardware intrinsic emission + stream/quant codegen
    let cfg_avx2 = TargetConfig::default().with_simd_isa(SimdIsa::Avx2);
    let rust_avx2 =
        generate_code_with_config(&after, CodegenTarget::Rust, &cfg_avx2).expect("rust avx2");
    assert!(rust_avx2.contains("_mm256_fmadd_ps"));
    assert!(rust_avx2.contains("[stream:launch(stream_id=0)]"));
    assert!(rust_avx2.contains("[stream:launch(stream_id=1)]"));
    assert!(rust_avx2.contains("[stream:wait(wait_stream=1, target_stream=0)]"));
    assert!(rust_avx2.contains("[quant:w8a8_static("));

    let c_avx2 = generate_code_with_config(&after, CodegenTarget::C, &cfg_avx2).expect("c avx2");
    assert!(c_avx2.contains("#include <immintrin.h>"));
    assert!(c_avx2.contains("_mm256_fmadd_ps"));
    assert!(c_avx2.contains("[stream:wait(wait_stream=1, target_stream=0)]"));
    assert!(c_avx2.contains("[quant:w8a8_static("));

    let cfg_neon = TargetConfig::default().with_simd_isa(SimdIsa::Neon);
    let c_neon = generate_code_with_config(&after, CodegenTarget::C, &cfg_neon).expect("c neon");
    assert!(c_neon.contains("#include <arm_neon.h>"));
    assert!(c_neon.contains("vfmaq_f32"));

    // Verify decompose-composites lowers static_quantize and stream_wait cleanly
    let mut decomposed = after.clone();
    let mut dec_pm =
        PassManager::from_pipeline_str(&registry, "decompose-composites").expect("decompose");
    dec_pm.run(&mut decomposed).expect("run decompose");
    let dec_rep = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(dec_rep.passed);
}

#[test]
fn test_modular_a_to_b_translator_chain_and_loop_ir_passes() {
    use castforge_codegen::CodegenTranslator;
    use castforge_ir::{Payload, TranslatorChain};
    use castforge_mlir::MlirParseTranslator;
    use castforge_passes::{
        PassTranslator, PermuteLoopsPass, TileLoopsPass, UnrollLoopsPass, VectorizeLoopsPass,
    };

    let mlir = r#"
module @chain_demo {
  func.func @main(%x: tensor<4x4xf32>, %w: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %y : tensor<4x4xf32>
  }
}
"#;
    // 1. Test pipe-separated IR pass pipeline (A -> B -> C -> D) stamping loop attributes on IR
    let before = parse_mlir(mlir).expect("parse chain_demo");
    let mut scheduled = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(
        &registry,
        "tile-loops(size=2) | unroll-loops(double_buffer=true) | vectorize-loops(isa=avx2)",
    )
    .expect("pipe-separated loop passes");
    let res = pm.run(&mut scheduled).expect("run loop schedule passes");
    assert!(res.iter().all(|r| r.changed));

    // Codegen with default TargetConfig reads the schedule attributes directly from the IR
    let rust_scheduled = generate_code(&scheduled, CodegenTarget::Rust).expect("rust from ir");
    assert!(rust_scheduled.contains("for m0 in (0..4).step_by(2)"));
    assert!(rust_scheduled.contains("[loop:double_buffer(stages=2, ping_pong"));
    assert!(rust_scheduled.contains("_mm256_fmadd_ps"));

    let c_scheduled = generate_code(&scheduled, CodegenTarget::C).expect("c from ir");
    assert!(c_scheduled.contains("for (int m0 = 0; m0 < 4; m0 += 2)"));
    assert!(c_scheduled.contains("[loop:double_buffer(stages=2, ping_pong"));
    assert!(c_scheduled.contains("#include <immintrin.h>"));

    // 2. Test unified TranslatorChain concatenating Text -> IR -> Loop Passes -> Rust Codegen
    let mut chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(PassTranslator(PermuteLoopsPass {
            order: "mkn".to_string(),
            collapse: false,
            skew: false,
        }))
        .then(PassTranslator(TileLoopsPass {
            tile_m: 0,
            tile_n: 0,
            tile_k: 0,
            l2_tile: 0,
            reg_m: 0,
            reg_n: 0,
        }))
        .then(PassTranslator(UnrollLoopsPass {
            factor: 1,
            prefetch: 2,
            split_k: 1,
            double_buffer: false,
        }))
        .then(PassTranslator(VectorizeLoopsPass {
            width: 8,
            isa: "avx2".to_string(),
            parallel: true,
        }))
        .then(CodegenTranslator::new(CodegenTarget::Rust));

    assert_eq!(chain.len(), 6);
    let out_payload = chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("execute concatenated translator chain");
    let rust_out = out_payload.into_text().expect("text output");
    assert!(rust_out.contains("[loop:interchange(MKN)]"));
    assert!(rust_out.contains("[loop:software_pipeline(prefetch_dist=2)]"));
    assert!(rust_out.contains("[loop:parallel(collapse=2, schedule=static)]"));
    assert!(rust_out.contains("_mm256_fmadd_ps"));
}

#[test]
fn test_symmetric_model_exports_loop_ir_gpu_backends_autotune_and_diff_translators() {
    use castforge_codegen::{
        AutotuneScheduleTranslator, CHeaderTranslator, CodegenTranslator, MemoryPlanTranslator,
        RooflineReportTranslator, SharedLibTranslator, TensorToLoopTranslator,
    };
    use castforge_ir::{Payload, TranslatorChain};
    use castforge_mlir::{
        GgufExportTranslator, GgufImportTranslator, MlirParseTranslator, OnnxExportTranslator,
        OnnxImportTranslator, SafetensorsExportTranslator,
    };
    use castforge_passes::PassRegistry;
    use castforge_viz::{IrDiffTranslator, SnapshotIrTranslator};

    let mlir = r#"
module @export_and_gpu_demo {
  func.func @forward(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %w = stablehlo.constant dense<0.5> : tensor<4x4xf32>
    %b = stablehlo.constant dense<0.1> : tensor<2x4xf32>
    %mm = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %y = stablehlo.add %mm, %b : tensor<2x4xf32>
    func.return %y : tensor<2x4xf32>
  }
}
"#;

    // Step 1: Symmetric Model Export Translators (IR -> ONNX -> IR, IR -> GGUF -> IR, SafeTensors)
    let mut onnx_roundtrip = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(OnnxExportTranslator)
        .then(OnnxImportTranslator);
    let onnx_mod = onnx_roundtrip
        .execute(Payload::Text(mlir.to_string()))
        .expect("IR -> ONNX -> IR roundtrip")
        .into_ir()
        .expect("onnx module");
    assert_eq!(onnx_mod.tensor_store.len(), 2);

    let mut gguf_roundtrip = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(GgufExportTranslator)
        .then(GgufImportTranslator);
    let gguf_mod = gguf_roundtrip
        .execute(Payload::Text(mlir.to_string()))
        .expect("IR -> GGUF -> IR roundtrip")
        .into_ir()
        .expect("gguf module");
    assert_eq!(gguf_mod.tensor_store.len(), 2);

    let mut st_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(SafetensorsExportTranslator);
    let st_bytes = st_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("IR -> SafeTensors")
        .into_bytes()
        .expect("safetensors bytes");
    assert!(st_bytes.len() > 64);

    // Step 2: Progressive LoopIr, CUDA, Metal (MSL), C Header, and Native Shared Library
    let mut loop_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(TensorToLoopTranslator::default());
    let loop_txt = loop_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("lower-to-loops")
        .into_text()
        .expect("loop text");
    assert!(loop_txt.contains("kernel @forward("));
    assert!(loop_txt.contains("GEMM 2x4x4"));

    let mut cuda_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(CodegenTranslator::new(CodegenTarget::Cuda));
    let cuda_src = cuda_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("codegen-cuda")
        .into_text()
        .expect("cuda text");
    assert!(cuda_src.contains("__global__ void forward_cuda_kernel("));
    assert!(cuda_src.contains("__shared__ float shared_tile["));

    let mut msl_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(CodegenTranslator::new(CodegenTarget::Msl));
    let msl_src = msl_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("codegen-msl")
        .into_text()
        .expect("msl text");
    assert!(msl_src.contains("#include <metal_stdlib>"));
    assert!(msl_src.contains("kernel void forward_msl_kernel("));

    let mut hdr_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(CHeaderTranslator);
    let hdr_src = hdr_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("codegen-header")
        .into_text()
        .expect("header text");
    assert!(hdr_src.contains("#ifndef CASTFORGE_EXPORT_AND_GPU_DEMO_H"));
    assert!(hdr_src.contains("void forward("));

    let mut lib_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(SharedLibTranslator::default());
    let lib_bytes = lib_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("compile-lib")
        .into_bytes()
        .expect("shared lib bytes");
    assert!(lib_bytes.len() > 1024);

    // Step 3: Pipeable AutotuneScheduleTranslator (IR -> Autotune -> Codegen)
    let mut autotune_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(AutotuneScheduleTranslator::default())
        .then(CodegenTranslator::new(CodegenTarget::Rust));
    let tuned_rust = autotune_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("autotune-schedule -> codegen-rust")
        .into_text()
        .expect("tuned rust");
    assert!(tuned_rust.contains("pub fn forward("));

    // Step 4: SnapshotIrTranslator + IrDiffTranslator, RooflineReportTranslator, MemoryPlan
    let registry = PassRegistry::with_builtin_passes();
    let fuse_pass = registry
        .create_pass("fuse-matmul-epilogue", &Default::default())
        .expect("fuse-matmul-epilogue");
    let dce_pass = registry
        .create_pass("dce", &Default::default())
        .expect("dce");
    let mut diff_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(SnapshotIrTranslator)
        .then(fuse_pass)
        .then(dce_pass)
        .then(IrDiffTranslator::default());
    let diff_txt = diff_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("snapshot-ir -> fuse-elementwise -> dce -> ir-diff")
        .into_text()
        .expect("diff text");
    assert!(diff_txt.contains("--- a/module.mlir"));
    assert!(diff_txt.contains("castforge.fused_matmul_epilogue"));

    let mut roofline_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(RooflineReportTranslator::default());
    let roofline_json = roofline_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("roofline-report")
        .into_text()
        .expect("roofline json");
    assert!(roofline_json.contains("\"total_flops\""));

    let mut mem_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(MemoryPlanTranslator);
    let mem_json = mem_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("memory-plan")
        .into_text()
        .expect("memory plan json");
    assert!(mem_json.contains("\"total_activation_bytes\""));
}

#[test]
fn test_ir_manipulation_tools_extract_outline_reduce_bisect_sanitize_and_compose() {
    use castforge_codegen::bisect_module_snapshots;
    use castforge_passes::link_modules;

    let mlir = r#"
module @ir_tools_demo {
  func.func @main(
    %x: tensor<2x4xf32>,
    %w: tensor<4x4xf32>,
    %b: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %mm = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %biased = stablehlo.add %mm, %b : tensor<2x4xf32>
    %act = stablehlo.exponential %biased : tensor<2x4xf32>
    func.return %act : tensor<2x4xf32>
  }
}
"#;
    let base = parse_mlir(mlir).expect("parse ir_tools_demo");
    let registry = PassRegistry::with_builtin_passes();

    // 1. Subgraph Extractor (`extract-subgraph`): slice backward cone up to %mm
    let mut sliced = base.clone();
    let mut slice_pm = PassManager::from_pipeline_str(&registry, "extract-subgraph(target=mm)")
        .expect("extract-subgraph");
    let slice_res = slice_pm.run(&mut sliced).expect("run extract-subgraph");
    assert!(slice_res[0].changed);
    let sliced_ir = print_mlir(&sliced);
    assert!(sliced_ir.contains("stablehlo.dot_general"));
    assert!(!sliced_ir.contains("stablehlo.exponential"));

    // Also test cutting at `stop_at=mm` to promote `%mm` to a function input argument
    let mut cut_sliced = base.clone();
    let mut cut_pm =
        PassManager::from_pipeline_str(&registry, "extract-subgraph(target=act,stop_at=mm)")
            .expect("cut extract-subgraph");
    cut_pm
        .run(&mut cut_sliced)
        .expect("run cut extract-subgraph");
    let cut_ir = print_mlir(&cut_sliced);
    assert!(!cut_ir.contains("stablehlo.dot_general"));
    assert!(cut_ir.contains("stablehlo.add"));
    assert!(cut_ir.contains("stablehlo.exponential"));

    // 2. Subgraph Outliner (`outline-subgraphs`) <-> Inliner (`inliner`) symmetric round-trip
    let mut outlined = base.clone();
    let mut out_pm =
        PassManager::from_pipeline_str(&registry, "outline-subgraphs(mode=gemm)").expect("outline");
    let out_res = out_pm.run(&mut outlined).expect("run outline");
    assert!(out_res[0].changed);
    let out_ir = print_mlir(&outlined);
    assert!(out_ir.contains("func.func private @outlined_kernel_0("));
    assert!(out_ir.contains("func.call @outlined_kernel_0("));
    let out_rep = verify_module_numerics(&base, &outlined, 1e-5).expect("verify outlined");
    assert!(out_rep.passed);

    let mut re_inlined = outlined.clone();
    let mut inl_pm =
        PassManager::from_pipeline_str(&registry, "inliner | symbol-dce").expect("inliner");
    inl_pm.run(&mut re_inlined).expect("run inliner");
    let inl_ir = print_mlir(&re_inlined);
    assert!(!inl_ir.contains("func.call"));
    let inl_rep = verify_module_numerics(&base, &re_inlined, 1e-5).expect("verify re-inlined");
    assert!(inl_rep.passed);

    // 3. Delta IR Reducer (`reduce-ir`) & Pass Bisector (`bisect-passes`)
    let mut reduced = base.clone();
    let mut red_pm =
        PassManager::from_pipeline_str(&registry, "reduce-ir(max_ops=1,keep_op=exponential)")
            .expect("reduce-ir");
    let red_res = red_pm.run(&mut reduced).expect("run reduce-ir");
    assert!(red_res[0].changed);
    let red_ir = print_mlir(&reduced);
    assert!(red_ir.contains("stablehlo.exponential"));
    assert!(!red_ir.contains("stablehlo.dot_general"));

    let mut step1 = base.clone();
    PassManager::from_pipeline_str(&registry, "fuse-matmul-epilogue | bisect-passes(limit=2)")
        .unwrap()
        .run(&mut step1)
        .unwrap();
    let bisect_rep = bisect_module_snapshots(
        &base,
        &[("fuse-matmul-epilogue".to_string(), step1.clone())],
        1e-5,
    )
    .expect("bisect snapshots");
    assert!(bisect_rep.first_failing_pass.is_none());
    assert!(bisect_rep.steps[0].passed);

    // 4. Numerical Sanitizer (`sanitize-numerics`) + Codegen + Decompose
    let mut sanitized = base.clone();
    let mut san_pm =
        PassManager::from_pipeline_str(&registry, "sanitize-numerics(clamp_max=65504.0)")
            .expect("sanitize-numerics");
    let san_res = san_pm.run(&mut sanitized).expect("run sanitize");
    assert!(san_res[0].changed);
    let san_ir = print_mlir(&sanitized);
    assert!(san_ir.contains("castforge.nan_guard"));
    let san_rep = verify_module_numerics(&base, &sanitized, 1e-5).expect("verify sanitized");
    assert!(san_rep.passed);

    let san_rust = generate_code(&sanitized, CodegenTarget::Rust).expect("rust sanitized");
    assert!(san_rust.contains("[sanitize:nan_guard(clamp_max=65504.0)]"));
    let san_c = generate_code(&sanitized, CodegenTarget::C).expect("c sanitized");
    assert!(san_c.contains("[sanitize:nan_guard(clamp_max=65504.0)]"));

    let mut dec_san = sanitized.clone();
    PassManager::from_pipeline_str(&registry, "decompose-composites")
        .unwrap()
        .run(&mut dec_san)
        .unwrap();
    let dec_san_rep = verify_module_numerics(&base, &dec_san, 1e-5).expect("verify dec_san");
    assert!(dec_san_rep.passed);

    // 5. Module Linker (`link_modules`) & Function Composer (`compose-funcs`)
    let stage2_mlir = r#"
module @stage2_mod {
  func.func @postprocess(%h: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %y = stablehlo.tanh %h : tensor<2x4xf32>
    func.return %y : tensor<2x4xf32>
  }
}
"#;
    let stage2 = parse_mlir(stage2_mlir).expect("parse stage2");
    let mut linked = base.clone();
    let count = link_modules(&mut linked, &stage2).expect("link_modules");
    assert_eq!(count, 1);
    assert_eq!(linked.functions().len(), 2);

    let mut comp_pm = PassManager::from_pipeline_str(
        &registry,
        "compose-funcs(first=main,second=postprocess,entry=end_to_end)",
    )
    .expect("compose-funcs");
    let comp_res = comp_pm.run(&mut linked).expect("run compose-funcs");
    assert!(comp_res[0].changed);
    let comp_ir = print_mlir(&linked);
    assert!(comp_ir.contains("func.func @end_to_end("));
    assert!(comp_ir.contains("func.call @main("));
    assert!(comp_ir.contains("func.call @postprocess("));
}
