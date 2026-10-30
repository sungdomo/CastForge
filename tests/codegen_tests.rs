use castforge::codegen::{
    BottleneckKind, CodegenTarget, GemmLoopOrder, HardwareRoofline, SimdIsa, TargetConfig,
    analyze_module_cost_with_roofline, build_memory_plan, evaluate_module, generate_code,
    generate_code_with_config, verify_module_numerics,
};
use castforge::mlir::{parse_mlir, print_mlir};
use castforge::passes::{PassManager, PassRegistry};

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
        "merge-weights(method=slerp,t=0.5),fold-weights(mode=prune,sparsity=0.5),quantize(bits=8)",
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
        "fold-weights,fuse-gemm(kind=epilogue),simplify-algebraic(mode=constants),dce",
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
    let mut pm = PassManager::from_pipeline_str(&registry, "fuse-gemm(kind=epilogue),dce")
        .expect("pipeline");
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
    let mut pm = PassManager::from_pipeline_str(&registry, "optimize-memory(mode=inplace)")
        .expect("inplace pipeline");
    let res = pm.run(&mut inplace_mod).expect("run optimize-memory");
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
    let mut pm = PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=loops)")
        .expect("fuse-elementwise(mode=loops)");
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
        concat!(
            "fuse-attention(kind=rope),",
            "fuse-elementwise(mode=loops),",
            "fuse-elementwise(mode=loops,max_steps=2)"
        ),
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
        "fuse-norm(kind=softmax),fuse-attention,fuse-elementwise(mode=map_reduce)",
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
    let mut dec_pm = PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=decompose)")
        .expect("decompose");
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
        "schedule-loops(lower_conv=true),fold-views,fuse-gemm(kind=epilogue),dce",
    )
    .expect("conv pipeline");
    pm.run(&mut lowered).expect("run conv pipeline");

    let printed = print_mlir(&lowered);
    assert!(printed.contains("castforge.im2col"));
    assert!(printed.contains("castforge.fused_matmul_epilogue"));

    let rep = verify_module_numerics(&before, &lowered, 1e-5).expect("verify conv numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Verify Roofline schedule auto-tuner
    let roofline = castforge::codegen::HardwareRoofline::new(1000.0, 100.0);
    let tune = castforge::codegen::autotune_module(&lowered, &roofline);
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
        "/examples/llama_block.mlir"
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
    use castforge::codegen::{autotune_module_jit, jit_execute_and_benchmark};
    use castforge::mlir::{export_gguf_bytes, import_gguf_bytes, import_onnx_json};

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
    let mut pm = PassManager::from_pipeline_str(&registry, "fuse-gemm(kind=epilogue),dce")
        .expect("pipeline");
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
    let mut ad_pm = PassManager::from_pipeline_str(&registry, "shard-distributed(mode=autodiff)")
        .expect("autodiff");
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

    // Verify SGD step via shard-distributed(mode=autodiff,lr=0.1)
    let mut sgd_mod = fwd_mod.clone();
    let mut sgd_pm =
        PassManager::from_pipeline_str(&registry, "shard-distributed(mode=autodiff,lr=0.1)")
            .expect("autodiff lr");
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
        concat!(
            "fuse-attention(mode=kv_cache,max_seq_len=4096),",
            "shard-distributed(mode=tp,devices=4,coalesce=false),",
            "simplify-algebraic(egraph=true),",
            "optimize-memory(mode=remat)"
        ),
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
    let mut dec_pm = PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=decompose)")
        .expect("decompose");
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
    let mut pm =
        PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=moe),fuse-attention")
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
    let mut dec_pm = PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=decompose)")
        .expect("decompose");
    dec_pm.run(&mut decomposed).expect("run decompose");
    let dec_rep = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(dec_rep.passed);

    // Verify mixed-precision via quantize(dtype=bf16) and ping-pong double buffering
    let mut mp_mod = before.clone();
    let mut mp_pm = PassManager::from_pipeline_str(&registry, "quantize(dtype=bf16)").expect("mp");
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
        "quantize(scheme=w8a8,bits=8,samples=4),shard-distributed(mode=streams,streams=2)",
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
    let mut dec_pm = PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=decompose)")
        .expect("decompose");
    dec_pm.run(&mut decomposed).expect("run decompose");
    let dec_rep = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(dec_rep.passed);
}

#[test]
fn test_modular_a_to_b_translator_chain_and_loop_ir_passes() {
    use castforge::codegen::CodegenTranslator;
    use castforge::ir::{Payload, TranslatorChain};
    use castforge::mlir::MlirParseTranslator;
    use castforge::passes::{
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
        "schedule-loops(size=2,double_buffer=true,isa=avx2)",
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
    use castforge::codegen::{
        AutotuneScheduleTranslator, CHeaderTranslator, CodegenTranslator, MemoryPlanTranslator,
        RooflineReportTranslator, SharedLibTranslator, TensorToLoopTranslator,
    };
    use castforge::ir::{Payload, TranslatorChain};
    use castforge::mlir::{
        GgufExportTranslator, GgufImportTranslator, MlirParseTranslator, OnnxExportTranslator,
        OnnxImportTranslator, SafetensorsExportTranslator,
    };
    use castforge::passes::PassRegistry;
    use castforge::viz::{IrDiffTranslator, SnapshotIrTranslator};

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
        .create_pass("fuse-gemm", &Default::default())
        .expect("fuse-gemm");
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
    use castforge::codegen::bisect_module_snapshots;
    use castforge::passes::link_modules;

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

    // 1. Subgraph Extractor (`normalize-ir(target=mm)`): slice backward cone up to %mm
    let mut sliced = base.clone();
    let mut slice_pm = PassManager::from_pipeline_str(&registry, "normalize-ir(target=mm)")
        .expect("normalize-ir(target=mm)");
    let slice_res = slice_pm.run(&mut sliced).expect("run extract-subgraph");
    assert!(slice_res[0].changed);
    let sliced_ir = print_mlir(&sliced);
    assert!(sliced_ir.contains("stablehlo.dot_general"));
    assert!(!sliced_ir.contains("stablehlo.exponential"));

    // Also test cutting at `stop_at=mm` to promote `%mm` to a function input argument
    let mut cut_sliced = base.clone();
    let mut cut_pm =
        PassManager::from_pipeline_str(&registry, "normalize-ir(target=act,stop_at=mm)")
            .expect("cut normalize-ir");
    cut_pm
        .run(&mut cut_sliced)
        .expect("run cut extract-subgraph");
    let cut_ir = print_mlir(&cut_sliced);
    assert!(!cut_ir.contains("stablehlo.dot_general"));
    assert!(cut_ir.contains("stablehlo.add"));
    assert!(cut_ir.contains("stablehlo.exponential"));

    // 2. Subgraph Outliner (`normalize-ir(mode=gemm)`) <-> Inliner (`mode=inline`) round-trip
    let mut outlined = base.clone();
    let mut out_pm =
        PassManager::from_pipeline_str(&registry, "normalize-ir(mode=gemm)").expect("outline");
    let out_res = out_pm.run(&mut outlined).expect("run outline");
    assert!(out_res[0].changed);
    let out_ir = print_mlir(&outlined);
    assert!(out_ir.contains("func.func private @outlined_kernel_0("));
    assert!(out_ir.contains("func.call @outlined_kernel_0("));
    let out_rep = verify_module_numerics(&base, &outlined, 1e-5).expect("verify outlined");
    assert!(out_rep.passed);

    let mut re_inlined = outlined.clone();
    let mut inl_pm = PassManager::from_pipeline_str(&registry, "normalize-ir(mode=inline) | dce")
        .expect("inline-calls");
    inl_pm.run(&mut re_inlined).expect("run inline-calls");
    let inl_ir = print_mlir(&re_inlined);
    assert!(!inl_ir.contains("func.call"));
    let inl_rep = verify_module_numerics(&base, &re_inlined, 1e-5).expect("verify re-inlined");
    assert!(inl_rep.passed);

    // 3. Delta IR Reducer (`normalize-ir(max_ops=1)`) & Pass Bisector (`verify-ir(bisect=2)`)
    let mut reduced = base.clone();
    let mut red_pm =
        PassManager::from_pipeline_str(&registry, "normalize-ir(max_ops=1,keep_op=exponential)")
            .expect("normalize-ir(max_ops=1)");
    let red_res = red_pm.run(&mut reduced).expect("run reduce-ir");
    assert!(red_res[0].changed);
    let red_ir = print_mlir(&reduced);
    assert!(red_ir.contains("stablehlo.exponential"));
    assert!(!red_ir.contains("stablehlo.dot_general"));

    let mut step1 = base.clone();
    PassManager::from_pipeline_str(&registry, "fuse-gemm(kind=epilogue) | verify-ir(bisect=2)")
        .unwrap()
        .run(&mut step1)
        .unwrap();
    let bisect_rep =
        bisect_module_snapshots(&base, &[("fuse-gemm".to_string(), step1.clone())], 1e-5)
            .expect("bisect snapshots");
    assert!(bisect_rep.first_failing_pass.is_none());
    assert!(bisect_rep.steps[0].passed);

    // 4. Numerical Sanitizer (`verify-ir(sanitize=true)`) + Codegen + Decompose
    let mut sanitized = base.clone();
    let mut san_pm =
        PassManager::from_pipeline_str(&registry, "verify-ir(sanitize=true,clamp_max=65504.0)")
            .expect("verify-ir(sanitize=true)");
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
    PassManager::from_pipeline_str(&registry, "fuse-elementwise(mode=decompose)")
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
        "normalize-ir(mode=compose,first=main,second=postprocess,entry=end_to_end)",
    )
    .expect("normalize-ir(mode=compose)");
    let comp_res = comp_pm.run(&mut linked).expect("run compose-funcs");
    assert!(comp_res[0].changed);
    let comp_ir = print_mlir(&linked);
    assert!(comp_ir.contains("func.func @end_to_end("));
    assert!(comp_ir.contains("func.call @main("));
    assert!(comp_ir.contains("func.call @postprocess("));
}

#[test]
fn test_freeze_parameterize_tap_resize_dedup_and_normalize_passes() {
    let mlir = r#"
module @two_layer_gemm {
  func.func @main(
    %x: tensor<2x4xf32>,
    %w1: tensor<4x4xf32>,
    %w2: tensor<4x4xf32>
  ) -> tensor<2x4xf32> {
    %h1 = stablehlo.dot_general %x, %w1, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %h2 = stablehlo.dot_general %h1, %w2, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    func.return %h2 : tensor<2x4xf32>
  }
}
"#;
    let base = parse_mlir(mlir).expect("parse two_layer_gemm");
    let registry = PassRegistry::with_builtin_passes();

    // 1. Symmetric `normalize-ir(args=...)` <-> `normalize-ir(unfreeze=true)` round-trip
    let mut frozen = base.clone();
    let mut frz_pm = PassManager::from_pipeline_str(&registry, "normalize-ir(args=w1;w2)")
        .expect("normalize-ir(args=w1;w2)");
    let frz_res = frz_pm.run(&mut frozen).expect("run normalize-ir(args)");
    assert!(frz_res[0].changed);
    let frz_ir = print_mlir(&frozen);
    assert!(frz_ir.contains("func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32>"));
    assert!(frz_ir.contains("stablehlo.constant dense<0.5000> : tensor<4x4xf32>"));

    let mut re_param = frozen.clone();
    let mut prm_pm =
        PassManager::from_pipeline_str(&registry, "normalize-ir(unfreeze=true,min_elements=2)")
            .expect("normalize-ir(unfreeze=true)");
    let prm_res = prm_pm
        .run(&mut re_param)
        .expect("run normalize-ir(unfreeze=true)");
    assert!(prm_res[0].changed);
    let prm_ir = print_mlir(&re_param);
    assert!(!prm_ir.contains("stablehlo.constant"));
    assert!(prm_ir.contains("%w1: tensor<4x4xf32>, %w2: tensor<4x4xf32>"));

    // 2. Intermediate Activation Tapper (`normalize-ir(tap=h1)`)
    let mut tapped = base.clone();
    let mut tap_pm = PassManager::from_pipeline_str(&registry, "normalize-ir(tap=h1)")
        .expect("normalize-ir(tap=h1)");
    let tap_res = tap_pm.run(&mut tapped).expect("run normalize-ir(tap=h1)");
    assert!(tap_res[0].changed);
    let tap_ir = print_mlir(&tapped);
    assert!(tap_ir.contains("-> (tensor<2x4xf32>, tensor<2x4xf32>)"));
    assert!(tap_ir.contains("return %h2, %h1 : tensor<2x4xf32>, tensor<2x4xf32>"));

    // 3. Static Batch Dimension Resizer (`specialize-shapes(batch=8)`)
    let mut resized = base.clone();
    let mut rsz_pm = PassManager::from_pipeline_str(&registry, "specialize-shapes(batch=8)")
        .expect("specialize-shapes(batch=8)");
    let rsz_res = rsz_pm.run(&mut resized).expect("run resize-batch");
    assert!(rsz_res[0].changed);
    let rsz_ir = print_mlir(&resized);
    assert!(rsz_ir.contains("%x: tensor<8x4xf32>"));
    assert!(rsz_ir.contains("%w1: tensor<4x4xf32>"));
    assert!(rsz_ir.contains("-> tensor<8x4xf32>"));

    // 4. Outlining + Identical Function Deduplicator (`normalize-ir(mode=gemm,dedup=true)`)
    let mut deduped = base.clone();
    let mut dedup_pm = PassManager::from_pipeline_str(
        &registry,
        "normalize-ir(mode=gemm,dedup=false) | normalize-ir(mode=dedup)",
    )
    .expect("outline + dedup");
    let dedup_res = dedup_pm.run(&mut deduped).expect("run outline + dedup");
    assert!(dedup_res[0].changed);
    assert!(dedup_res[1].changed);
    assert_eq!(dedup_res[1].ops_removed, 1);
    let dedup_ir = print_mlir(&deduped);
    assert!(dedup_ir.contains("func.func private @outlined_kernel_0("));
    assert!(!dedup_ir.contains("func.func private @outlined_kernel_1("));
    let dedup_rep = verify_module_numerics(&base, &deduped, 1e-5).expect("verify deduped");
    assert!(dedup_rep.passed);

    // 5. Canonical SSA Renamer & Attribute Sorter (`normalize-ir(prefix=t)`)
    let mut norm = base.clone();
    let mut norm_pm = PassManager::from_pipeline_str(&registry, "normalize-ir(prefix=t)")
        .expect("normalize-ir(prefix=t)");
    let norm_res = norm_pm.run(&mut norm).expect("run normalize-ir(prefix=t)");
    assert!(norm_res[0].changed);
    let norm_ir = print_mlir(&norm);
    assert!(norm_ir.contains("%arg0: tensor<2x4xf32>"));
    assert!(norm_ir.contains("%t0 = stablehlo.dot_general"));
    assert!(norm_ir.contains("%t1 = stablehlo.dot_general"));
}

#[test]
fn test_hash_verify_sparsify_2to4_bucket_shapes_convert_layout_and_offload() {
    let mlir = r#"
module @advanced_ir_tools {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %w1 = stablehlo.constant dense<[[0.1, 0.9, 0.2, 0.8],
                                    [0.7, 0.1, 0.3, 0.9],
                                    [0.4, 0.6, 0.1, 0.2],
                                    [0.8, 0.5, 0.9, 0.3]]> : tensor<4x4xf32>
    %h0 = stablehlo.tanh %x : tensor<2x4xf32>
    %h1 = stablehlo.dot_general %h0, %w1, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %h2 = stablehlo.tanh %h1 : tensor<2x4xf32>
    %out = stablehlo.add %h2, %h0 : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    let base = parse_mlir(mlir).expect("parse advanced_ir_tools");
    let registry = PassRegistry::with_builtin_passes();

    // 1. Structural Hashing + Invariant Verification (`verify-ir`)
    let mut hashed = base.clone();
    let mut hash_pm = PassManager::from_pipeline_str(
        &registry,
        "verify-ir(hash=true,equiv=false) | verify-ir(hash=false,equiv=true)",
    )
    .expect("hash+verify");
    let hash_res = hash_pm.run(&mut hashed).expect("run verify-ir");
    assert!(hash_res[0].changed);
    assert!(hash_res[1].changed);
    let hash_ir = print_mlir(&hashed);
    assert!(hash_ir.contains("castforge.structural_hash"));
    assert!(hash_ir.contains("castforge.verified_tol"));

    // 2. NVIDIA Ampere/Hopper 2:4 Structured Sparsity (`fold-weights(mode=2to4)`)
    let mut sparse = base.clone();
    let mut sp_pm = PassManager::from_pipeline_str(&registry, "fold-weights(mode=2to4)")
        .expect("fold-weights(mode=2to4)");
    let sp_res = sp_pm.run(&mut sparse).expect("run fold-weights(mode=2to4)");
    assert!(sp_res[0].changed);
    let sp_ir = print_mlir(&sparse);
    assert!(sp_ir.contains("sparsity = \"2:4\""));

    // 3. Multi-Bucket Dynamic Shape Dispatcher (`specialize-shapes(buckets=1;4)`)
    let mut bucketed = base.clone();
    let mut bkt_pm = PassManager::from_pipeline_str(&registry, "specialize-shapes(buckets=1;4)")
        .expect("specialize-shapes(buckets=1;4)");
    let bkt_res = bkt_pm.run(&mut bucketed).expect("run bucket-shapes");
    assert!(bkt_res[0].changed);
    assert_eq!(bkt_res[0].ops_created, 2);
    let bkt_ir = print_mlir(&bucketed);
    assert!(bkt_ir.contains("func.func @main_batch_1(%x: tensor<1x4xf32>)"));
    assert!(bkt_ir.contains("func.func @main_batch_4(%x: tensor<4x4xf32>)"));

    // 4. Host Memory Activation Offloader (`optimize-memory(mode=offload)`) + Decompose round-trip
    let mut offloaded = base.clone();
    let mut off_pm =
        PassManager::from_pipeline_str(&registry, "optimize-memory(mode=offload,min_span=2)")
            .expect("optimize-memory(mode=offload)");
    let off_res = off_pm.run(&mut offloaded).expect("run offload-activations");
    assert!(off_res[0].changed);
    let off_ir = print_mlir(&offloaded);
    assert!(off_ir.contains("castforge.host_offload"));
    let off_rust = generate_code(&offloaded, CodegenTarget::Rust).expect("emit rust offloaded");
    assert!(off_rust.contains("[offload:d2h_h2d_pinned("));
    let off_c = generate_code(&offloaded, CodegenTarget::C).expect("emit c offloaded");
    assert!(off_c.contains("[offload:d2h_h2d_pinned("));
    let off_rep = verify_module_numerics(&base, &offloaded, 1e-5).expect("verify offloaded");
    assert!(off_rep.passed);

    // 5. Spatial & Matrix Layout Canonicalizer (`fold-views(layout=nhwc)`)
    let mut layout_mod = base.clone();
    let mut layout_pm = PassManager::from_pipeline_str(&registry, "fold-views(layout=nhwc)")
        .expect("fold-views(layout=nhwc)");
    let layout_res = layout_pm
        .run(&mut layout_mod)
        .expect("run fold-views(layout=nhwc)");
    assert!(layout_res[0].changed);
    let layout_ir = print_mlir(&layout_mod);
    assert!(layout_ir.contains("data_layout = \"nhwc\""));
    assert!(layout_ir.contains("dims = [1, 0]"));
    let layout_rep = verify_module_numerics(&base, &layout_mod, 1e-5).expect("verify layout_mod");
    assert!(layout_rep.passed);
}

#[test]
fn test_pad_for_tiling_split_gemm_unroll_calls_hoist_all_reduce_instrument_and_strip() {
    let registry = PassRegistry::with_builtin_passes();
    assert_eq!(registry.list().count(), 38);

    let mlir = r#"
module @final_ninety_suite {
  func.func @main(%x: tensor<3x4xf32>, %w: tensor<4x4xf32>) -> tensor<3x4xf32> {
    %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<3x4xf32>, tensor<4x4xf32>) -> tensor<3x4xf32>
    func.return %y : tensor<3x4xf32>
  }
}
"#;
    let base = parse_mlir(mlir).expect("parse final_ninety_suite");

    // 1. Tile-alignment padding (`schedule-loops(align=8)`) + metadata stripping (`normalize-ir`)
    let mut padded = base.clone();
    let mut pad_pm = PassManager::from_pipeline_str(&registry, "schedule-loops(align=8)")
        .expect("schedule-loops(align=8)");
    let pad_res = pad_pm
        .run(&mut padded)
        .expect("run schedule-loops(align=8)");
    assert!(pad_res[0].changed);
    let pad_ir = print_mlir(&padded);
    assert!(pad_ir.contains("pad_align = 8"));
    assert!(pad_ir.contains("padded_mnk = array<i64: 8, 8, 8>"));

    let mut stripped = padded.clone();
    let mut strip_pm =
        PassManager::from_pipeline_str(&registry, "normalize-ir(attributes=true,ssa=false)")
            .expect("normalize-ir(attributes=true)");
    let strip_res = strip_pm.run(&mut stripped).expect("run normalize-ir");
    assert!(strip_res[0].changed);
    let strip_ir = print_mlir(&stripped);
    assert!(!strip_ir.contains("pad_align"));
    assert!(!strip_ir.contains("padded_mnk"));

    // 2. Large GEMM Column-Splitting (`schedule-loops(max_n=2)`) + exact numerical verification
    let mut split_gemm = base.clone();
    let mut split_pm = PassManager::from_pipeline_str(&registry, "schedule-loops(max_n=2)")
        .expect("schedule-loops(max_n=2)");
    let split_res = split_pm
        .run(&mut split_gemm)
        .expect("run schedule-loops(max_n=2)");
    assert!(split_res[0].changed);
    assert_eq!(split_res[0].ops_created, 5);
    let split_ir = print_mlir(&split_gemm);
    assert!(split_ir.contains("stablehlo.slice"));
    assert!(split_ir.contains("stablehlo.concatenate"));
    let split_rep = verify_module_numerics(&base, &split_gemm, 1e-5).expect("verify split_gemm");
    assert!(split_rep.passed);

    // 3. Outlining + Bounded Call Unroller (`outline-subgraphs | unroll-calls(repeat=3)`)
    let sq_mlir = r#"
module @square_block {
  func.func @main(%x: tensor<4x4xf32>, %w: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %y : tensor<4x4xf32>
  }
}
"#;
    let mut unrolled = parse_mlir(sq_mlir).expect("parse square_block");
    let mut unroll_pm = PassManager::from_pipeline_str(
        &registry,
        "normalize-ir(mode=gemm,dedup=false) | normalize-ir(mode=unroll,repeat=3)",
    )
    .expect("outline + unroll-calls");
    let unroll_res = unroll_pm.run(&mut unrolled).expect("run unroll-calls");
    assert!(unroll_res[1].changed);
    assert_eq!(unroll_res[1].ops_created, 2);
    let unroll_ir = print_mlir(&unrolled);
    assert!(unroll_ir.contains("%call_u1 = func.call @outlined_kernel_0("));
    assert!(unroll_ir.contains("%call_u2 = func.call @outlined_kernel_0("));

    // 4. Collective Communication Coalescer (`shard-distributed(mode=tp,devices=4,coalesce=true)`)
    let mut spmd_mod = base.clone();
    let mut spmd_pm = PassManager::from_pipeline_str(
        &registry,
        "shard-distributed(mode=tp,devices=4,coalesce=true)",
    )
    .expect("shard-distributed");
    let spmd_res = spmd_pm.run(&mut spmd_mod).expect("run shard-distributed");
    assert!(spmd_res[0].changed);
    let spmd_ir = print_mlir(&spmd_mod);
    assert!(spmd_ir.contains("async_overlap = true"));

    // 5. Per-Op Profiling Region Injector (`verify-ir(instrument=true)`) + Rust/C99 codegen
    let mut prof_mod = base.clone();
    let mut prof_pm = PassManager::from_pipeline_str(
        &registry,
        "verify-ir(instrument=true,filter=gemm,equiv=false,hash=false)",
    )
    .expect("verify-ir(instrument=true)");
    let prof_res = prof_pm.run(&mut prof_mod).expect("run profile-ir");
    assert!(prof_res[0].changed);
    let prof_ir = print_mlir(&prof_mod);
    assert!(prof_ir.contains("castforge.profile_region"));
    let prof_rust = generate_code(&prof_mod, CodegenTarget::Rust).expect("emit rust prof");
    assert!(prof_rust.contains("[profile:region(id=0, label=\"stablehlo.dot_general\")]"));
    let prof_c = generate_code(&prof_mod, CodegenTarget::C).expect("emit c prof");
    assert!(prof_c.contains("[profile:region(id=0, label=\"stablehlo.dot_general\")]"));
    let prof_rep = verify_module_numerics(&base, &prof_mod, 1e-5).expect("verify prof_mod");
    assert!(prof_rep.passed);
}

#[test]
fn test_multi_arch_simd_profiles_and_pluggable_backend_registry() {
    use castforge::codegen::{BackendRegistry, CodegenBackend, LoopModule};

    let mlir = r#"
module @multi_arch_demo {
  func.func @main(%x: tensor<2x4xf32>, %w: tensor<4x4xf32>) -> tensor<2x4xf32> {
    %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    func.return %y : tensor<2x4xf32>
  }
}
"#;
    let module = parse_mlir(mlir).expect("parse multi_arch_demo");

    // 1. Multi-arch hardware profiles (`x86_64-avx512`, `aarch64-sve`, `riscv64-rvv`, `wasm32`)
    for (arch, expected_isa, expected_intr, expected_lanes) in [
        ("x86_64-avx512", "avx512", "_mm512_fmadd_ps", 16),
        ("aarch64-sve", "sve", "svmla_f32_x", 8),
        ("riscv64-rvv", "rvv", "__riscv_vfmacc_vv_f32m1", 8),
        ("wasm32-simd128", "wasm_simd128", "f32x4_add_mul", 4),
    ] {
        let cfg = TargetConfig::for_arch(arch)
            .with_loop_order(GemmLoopOrder::Mnk)
            .with_unroll(1);
        assert_eq!(cfg.simd_isa.as_str(), expected_isa);
        assert_eq!(cfg.simd_isa.intrinsic_name(), expected_intr);
        assert_eq!(cfg.simd_width, expected_lanes);
        let c_code = generate_code_with_config(
            &module,
            CodegenTarget::C,
            &TargetConfig::default().with_simd_isa(cfg.simd_isa),
        )
        .expect("emit C for arch");
        assert!(c_code.contains(expected_intr));
        let rs_code = generate_code_with_config(
            &module,
            CodegenTarget::Rust,
            &TargetConfig::default().with_simd_isa(cfg.simd_isa),
        )
        .expect("emit Rust for arch");
        assert!(rs_code.contains(expected_intr));
    }

    // 2. Built-in AMD ROCm HIP (`codegen-hip`) and Khronos OpenCL (`codegen-opencl`) targets
    let hip_code = generate_code(&module, CodegenTarget::Hip).expect("emit hip");
    assert!(hip_code.contains("#include <hip/hip_runtime.h>"));
    assert!(hip_code.contains("hipLaunchKernelGGL(main_hip_kernel"));

    let ocl_code = generate_code(&module, CodegenTarget::OpenCl).expect("emit opencl");
    assert!(ocl_code.contains("__kernel void main_opencl_kernel("));
    assert!(ocl_code.contains("get_global_id(0)"));

    // 3. Pluggable `CodegenBackend` + `BackendRegistry` for adding a new target in ~15 lines
    struct CustomHalideBackend;
    impl CodegenBackend for CustomHalideBackend {
        fn name(&self) -> &'static str {
            "halide"
        }
        fn file_extension(&self) -> &'static str {
            "hl.cpp"
        }
        fn description(&self) -> &'static str {
            "Custom Halide/DSL backend registered dynamically via BackendRegistry"
        }
        fn emit_from_loop_ir(
            &self,
            loop_mod: &LoopModule,
            _config: &TargetConfig,
        ) -> anyhow::Result<String> {
            Ok(format!(
                "// Halide DSL for {} ({} kernels)\n{}",
                loop_mod.name,
                loop_mod.kernels.len(),
                loop_mod
            ))
        }
    }

    let mut backends = BackendRegistry::with_builtin_backends();
    assert_eq!(backends.list().count(), 7);
    backends.register(Box::new(CustomHalideBackend));
    assert_eq!(backends.list().count(), 8);

    let halide_src = backends
        .generate("halide", &module, &TargetConfig::default())
        .expect("generate custom halide target");
    assert!(halide_src.contains("// Halide DSL for multi_arch_demo (1 kernels)"));
    assert!(halide_src.contains("kernel @main("));
}

#[test]
fn test_sota_commutative_cse_refcount_dce_and_extended_eval() {
    use castforge::passes::prune_2d_columns_by_l1;

    // 1. Zero-dimension and short-slice safety in prune_2d_columns_by_l1
    let (pruned_empty, kept_empty) = prune_2d_columns_by_l1(&[], 0, 0, 4);
    assert!(pruned_empty.is_empty());
    assert!(kept_empty.is_empty());

    // 2. Commutative CSE & attribute-order-independent CSE + O(N) reference-counted DCE
    let mlir = r#"
module @sota_cse_dce {
  func.func @main(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %sum1 = stablehlo.add %a, %b : tensor<4xf32>
    %sum2 = stablehlo.add %b, %a : tensor<4xf32>
    %prod = stablehlo.multiply %sum1, %sum2 : tensor<4xf32>
    %dead1 = stablehlo.negate %prod : tensor<4xf32>
    %dead2 = stablehlo.abs %dead1 : tensor<4xf32>
    %dead3 = stablehlo.cosine %dead2 : tensor<4xf32>
    func.return %prod : tensor<4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("parse mlir");
    let use_counts = module.compute_use_counts();
    assert!(!use_counts.is_empty());

    let mut pm = PassManager::new()
        .add_pass(castforge::passes::CsePass::default())
        .add_pass(castforge::passes::DeadCodeEliminationPass::default());
    let results = pm.run(&mut module).expect("run cse + dce");
    assert_eq!(
        results[0].ops_removed, 1,
        "commutative add(%b, %a) deduplicated"
    );
    assert_eq!(
        results[1].ops_removed, 3,
        "3-op dead chain removed in single sweep"
    );

    // 3. Reference interpreter coverage for iota, negate, abs, cosine, sine, reverse
    let eval_mlir = r#"
module @eval_extended {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %iot = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xf32>
    %sum = stablehlo.add %x, %iot : tensor<4xf32>
    %neg = stablehlo.negate %sum : tensor<4xf32>
    %ab = stablehlo.abs %neg : tensor<4xf32>
    %rev = "stablehlo.reverse"(%ab) {dimensions = array<i64: 0>} : (tensor<4xf32>) -> tensor<4xf32>
    func.return %rev : tensor<4xf32>
  }
}
"#;
    let eval_mod = parse_mlir(eval_mlir).expect("parse eval_mlir");
    let out = castforge::codegen::evaluate_module(&eval_mod, &[vec![1.0, 2.0, 3.0, 4.0]])
        .expect("evaluate extended ops");
    assert_eq!(out[0], vec![7.0, 5.0, 3.0, 1.0]);
}

#[test]
fn test_consolidated_and_sota_passes_groupwise_batchnorm_concat_paged_gvn() {
    let registry = PassRegistry::with_builtin_passes();

    // 1. Test fold-batchnorm, quantize-groupwise, eliminate-concat-copies, gvn, strip-metadata
    let mlir = concat!(
        "module @sota_pipeline {\n",
        "  func.func @main(%x: tensor<2x2xf32>, %r: tensor<2x2xf32>) -> tensor<2x4xf32> {\n",
        "    %w = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>\n",
        "    %scale = stablehlo.constant dense<[[0.5, 2.0], [0.5, 2.0]]> : tensor<2x2xf32>\n",
        "    %mm = \"stablehlo.dot_general\"(%x, %w) ",
        "{lhs_contracting_dimensions = array<i64: 1>, ",
        "rhs_contracting_dimensions = array<i64: 0>} ",
        ": (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n",
        "    %bn = stablehlo.multiply %mm, %scale : tensor<2x2xf32>\n",
        "    %dup1 = stablehlo.add %bn, %r : tensor<2x2xf32>\n",
        "    %dup2 = stablehlo.add %r, %bn : tensor<2x2xf32>\n",
        "    %cat = \"stablehlo.concatenate\"(%dup1, %dup2) {dimension = 1 : i64} ",
        ": (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>\n",
        "    func.return %cat : tensor<2x4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut module = parse_mlir(mlir).expect("parse sota_pipeline");
    let baseline = module.clone();

    let mut pm_bn = PassManager::from_pipeline_str(
        &registry,
        "fold-weights,cse,optimize-memory(mode=zero_copy)",
    )
    .expect("build sota pipeline");
    let res = pm_bn.run(&mut module).expect("run sota pipeline");
    assert!(res[0].changed, "fold-weights absorbed scale into weight");
    assert_eq!(res[1].ops_removed, 1, "cse deduplicated commutative add");
    assert!(res[2].changed, "eliminate-concat-copies routed producer");

    let rep = verify_module_numerics(&baseline, &module, 1e-4).expect("verify numerics");
    assert!(rep.passed, "fold-weights + cse numerically equivalent");

    let printed = print_mlir(&module);
    assert!(printed.contains("castforge.batchnorm_folded = true"));
    assert!(printed.contains("castforge.zero_copy_concat = true"));

    // 2. Test quantize(scheme=groupwise) + normalize-ir(attributes=true)
    let mut pm_gw = PassManager::from_pipeline_str(
        &registry,
        "quantize(scheme=awq,bits=4,group_size=2,salient_ratio=0.25)",
    )
    .expect("build quantize(scheme=awq)");
    let gw_res = pm_gw.run(&mut module).expect("run quantize(scheme=awq)");
    assert!(gw_res[0].changed);
    assert!(print_mlir(&module).contains("awq_groupwise"));
    assert!(print_mlir(&module).contains("castforge.awq_salient_channels = 1"));

    let mut pm_strip = PassManager::from_pipeline_str(&registry, "normalize-ir(attributes=true)")
        .expect("normalize-ir(attributes=true)");
    pm_strip.run(&mut module).expect("run normalize-ir");
    assert!(!print_mlir(&module).contains("castforge.batchnorm_folded"));

    // 3. Test fuse-norm, fuse-activations, and fuse-attention(mode=paged)
    let attn_mlir = concat!(
        "module @paged_attn_demo {\n",
        "  func.func @main(%q: tensor<2x2xf32>, %k: tensor<2x2xf32>, %v: tensor<2x2xf32>) ",
        "-> tensor<2x2xf32> {\n",
        "    %att = \"stablehlo.composite\"(%q, %k, %v) ",
        "{decomposition = @attn, name = \"castforge.fused_attention\", scale = 0.70710678 : f32} ",
        ": (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>\n",
        "    func.return %att : tensor<2x2xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut attn_mod = parse_mlir(attn_mlir).expect("parse attn_mlir");
    let mut pm_paged = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "fuse-norm,fuse-elementwise(mode=activations),",
            "fuse-attention(mode=paged,page_block_size=16,max_blocks=64)"
        ),
    )
    .expect("build paged-attention pipeline");
    let paged_res = pm_paged.run(&mut attn_mod).expect("run paged-attention");
    assert!(paged_res[2].changed);
    let cuda_src =
        castforge::codegen::emit_cuda_module_with_config(&attn_mod, &TargetConfig::default())
            .expect("emit cuda for paged attention");
    assert!(cuda_src.contains("PagedAttention KV block-table addressing (page_block_size=16)"));

    // 4. Test fuse-attention(mode=flash_decoding|tree), shard-distributed(mode=ring_cp),
    //    quantize(scheme=fp8), and fuse-elementwise(mode=megakernel)
    let mut sota5_mod = parse_mlir(attn_mlir).expect("parse attn_mlir for sota5");
    let mut pm_sota5 = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "fuse-attention(mode=flash_decoding,split_kv=8),",
            "shard-distributed(mode=ring_cp,ring_size=4),",
            "fuse-attention(mode=tree,tree_depth=5,draft_tokens=4),",
            "fuse-elementwise(mode=megakernel)"
        ),
    )
    .expect("build sota5 pipeline");
    let sota5_res = pm_sota5.run(&mut sota5_mod).expect("run sota5 pipeline");
    assert!(sota5_res.iter().all(|r| r.changed));
    let sota5_printed = print_mlir(&sota5_mod);
    assert!(sota5_printed.contains("castforge.flash_decoding = true"));
    assert!(sota5_printed.contains("castforge.ring_size = 4"));
    assert!(sota5_printed.contains("castforge.speculative_tree = true"));
    assert!(sota5_printed.contains("castforge.megakernel = true"));

    let mut fp8_mod = parse_mlir(mlir).expect("parse mlir for fp8");
    let mut pm_fp8 = PassManager::from_pipeline_str(&registry, "quantize(scheme=fp8,format=e4m3)")
        .expect("build quantize(scheme=fp8)");
    let fp8_res = pm_fp8.run(&mut fp8_mod).expect("run quantize(scheme=fp8)");
    assert!(fp8_res[0].changed);
    assert!(print_mlir(&fp8_mod).contains("fp8_e4m3"));

    // 5. Test the 5 SOTA passes via unified entrypoints: fuse-attention(kind=mla),
    //    quantize(scheme=hadamard), fuse-norm(kind=cross_entropy),
    //    shard-distributed(mode=pp), and schedule-loops(tensor_cores=wgmma)
    let mut new5_mod = parse_mlir(mlir).expect("parse mlir for new5");
    let mut pm_new5 = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "fuse-attention(kind=mla,kv_lora_rank=512),",
            "quantize(scheme=hadamard,block_size=64),",
            "fuse-norm(kind=cross_entropy,chunk_size=4096),",
            "shard-distributed(mode=pp,stages=2,microbatches=4,schedule=1f1b),",
            "schedule-loops(tensor_cores=wgmma,mma_tile=64x128x16),",
            "canonicalize(mode=min_memory)"
        ),
    )
    .expect("build new5 pipeline");
    let new5_res = pm_new5.run(&mut new5_mod).expect("run new5 pipeline");
    assert!(new5_res.iter().all(|r| r.changed));
    let new5_printed = print_mlir(&new5_mod);
    assert!(new5_printed.contains("castforge.mla_compressed_kv = true"));
    assert!(new5_printed.contains("castforge.hadamard_rotated = true"));
    assert!(new5_printed.contains("castforge.fused_linear_cross_entropy = true"));
    assert!(new5_printed.contains("castforge.pp_schedule = \"1f1b\""));
    assert!(new5_printed.contains("castforge.tensor_core_arch = \"wgmma\""));
    assert!(new5_printed.contains("castforge.min_memory_scheduled = true"));

    // 6. Test SmoothQuant, Global Layout Propagation, Sliding-Window + Attention Sinks,
    //    NVFP4 E2M1 Quantization, Cross-Layer KV Sharing, and Dynamic Activation Sparsity
    let mut sq_mod = parse_mlir(mlir).expect("parse mlir for smoothquant");
    let mut pm_sq = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "quantize(scheme=smoothquant,alpha=0.5),",
            "fold-views(layout=nhwc,propagate=true),",
            "fold-weights(mode=activation_sparse,sparsity=0.5)"
        ),
    )
    .expect("build smoothquant + propagate layout + activation_sparse");
    let sq_res = pm_sq.run(&mut sq_mod).expect("run smoothquant + layout");
    assert!(sq_res.iter().all(|r| r.changed));
    let sq_ir = print_mlir(&sq_mod);
    assert!(sq_ir.contains("castforge.quant_scheme = \"smoothquant\""));
    assert!(sq_ir.contains("castforge.layout_propagated = true"));
    assert!(sq_ir.contains("castforge.activation_sparsity = 0.5"));

    let mut fp4_mod = parse_mlir(mlir).expect("parse mlir for nvfp4");
    let mut pm_fp4 = PassManager::from_pipeline_str(&registry, "quantize(scheme=nvfp4)")
        .expect("build quantize(scheme=nvfp4)");
    let fp4_res = pm_fp4
        .run(&mut fp4_mod)
        .expect("run quantize(scheme=nvfp4)");
    assert!(fp4_res[0].changed);
    assert!(print_mlir(&fp4_mod).contains("castforge.quant_scheme = \"nvfp4\""));

    let mut sw_mod = parse_mlir(attn_mlir).expect("parse attn_mlir for sliding_window");
    let mut pm_sw = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "fuse-attention(mode=sliding_window,window_size=4096,sink_tokens=4),",
            "fuse-attention(kind=kv_sharing,share_every=2)"
        ),
    )
    .expect("build sliding_window + kv_sharing attention");
    let sw_res = pm_sw
        .run(&mut sw_mod)
        .expect("run sliding_window + kv_sharing");
    assert!(sw_res.iter().all(|r| r.changed));
    let sw_ir = print_mlir(&sw_mod);
    assert!(sw_ir.contains("castforge.sliding_window = 4096"));
    assert!(sw_ir.contains("castforge.attention_sinks = 4"));
    assert!(sw_ir.contains("castforge.kv_shared = true"));

    let mut loop_mod = castforge::codegen::lower_to_loop_ir(&baseline, &TargetConfig::default());
    let mut opt_loop = castforge::codegen::OptimizeLoopIrPass::default();
    let changed = castforge::codegen::LoopPass::run(&mut opt_loop, &mut loop_mod);
    let _ = changed;

    // 7. Test chained GEMM epilogue + residual add fusion (`bias_relu_residual`)
    let res_gemm_mlir = concat!(
        "module @res_gemm {\n",
        "  func.func @main(%x: tensor<2x4xf32>, %w: tensor<4x4xf32>, ",
        "%b: tensor<2x4xf32>, %r: tensor<2x4xf32>) -> tensor<2x4xf32> {\n",
        "    %z = \"stablehlo.constant\"() {value = dense<0.0> : tensor<2x4xf32>} ",
        ": () -> tensor<2x4xf32>\n",
        "    %mm = \"stablehlo.dot_general\"(%x, %w) ",
        ": (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>\n",
        "    %biased = \"stablehlo.add\"(%mm, %b) ",
        ": (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>\n",
        "    %act = \"stablehlo.maximum\"(%biased, %z) ",
        ": (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>\n",
        "    %out = \"stablehlo.add\"(%act, %r) ",
        ": (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>\n",
        "    \"func.return\"(%out) : (tensor<2x4xf32>) -> ()\n",
        "  }\n",
        "}\n"
    );
    let mut rg_mod = parse_mlir(res_gemm_mlir).expect("parse res_gemm_mlir");
    let mut pm_rg = PassManager::from_pipeline_str(&registry, "fuse-gemm(kind=epilogue)")
        .expect("build fuse-gemm(kind=epilogue)");
    let rg_res = pm_rg
        .run(&mut rg_mod)
        .expect("run fuse-gemm(kind=epilogue)");
    assert!(rg_res[0].changed);
    let rg_ir = print_mlir(&rg_mod);
    assert!(rg_ir.contains("epilogue = \"bias_relu_residual\""));
    assert!(rg_ir.contains("castforge.has_residual = true"));

    // 8. Test ZeRO-3 / FSDP parameter sharding (`shard-distributed(mode=fsdp,devices=4)`)
    //    and Dynamic Token Pruning (`specialize-shapes(token_prune_ratio=0.5)`)
    let mut fsdp_mod = parse_mlir(mlir).expect("parse mlir for fsdp + token_prune");
    let mut pm_fsdp = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "shard-distributed(mode=fsdp,devices=4),",
            "specialize-shapes(token_prune_ratio=0.5)"
        ),
    )
    .expect("build fsdp + token_prune_ratio");
    let fsdp_res = pm_fsdp.run(&mut fsdp_mod).expect("run fsdp + token_prune");
    assert!(fsdp_res.iter().all(|r| r.changed));
    let fsdp_ir = print_mlir(&fsdp_mod);
    assert!(fsdp_ir.contains("castforge.fsdp_sharded = true"));
    assert!(fsdp_ir.contains("castforge.fsdp_devices = 4"));
    assert!(fsdp_ir.contains("castforge.token_prune_ratio = 0.5"));
    assert!(fsdp_ir.contains("castforge.token_merging = \"tome_bipartite\""));

    // 9. Test Expert Parallelism (EP), RoPE YaRN Scaling, Attention Logit Softcap + QK-Norm,
    //    and FP8 KV-Cache Quantization
    let mut ep_attn_mod = parse_mlir(attn_mlir).expect("parse attn_mlir for ep + rope + softcap");
    let mut pm_ep = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "fuse-attention(rope_scaling=yarn,rope_factor=8.0,logit_softcap=50.0,qk_norm=true,",
            "kv_dtype=fp8,max_seq_len=8192),",
            "shard-distributed(mode=ep,devices=8)"
        ),
    )
    .expect("build ep + rope_scaling + softcap + kv_dtype pipeline");
    let ep_res = pm_ep.run(&mut ep_attn_mod).expect("run ep + attn sota");
    assert!(ep_res[0].changed);
    let ep_ir = print_mlir(&ep_attn_mod);
    assert!(ep_ir.contains("castforge.rope_scaling = \"yarn\""));
    assert!(ep_ir.contains("castforge.rope_factor = 8.0"));
    assert!(ep_ir.contains("castforge.logit_softcap = 50.0"));
    assert!(ep_ir.contains("castforge.qk_norm = true"));
    assert!(ep_ir.contains("castforge.kv_cache_dtype = \"fp8\""));
    assert!(ep_ir.contains("castforge.kv_per_head_scale = true"));

    let mut moe_ep_mod = parse_mlir(mlir).expect("parse mlir for ep");
    let mut pm_moe_ep =
        PassManager::from_pipeline_str(&registry, "shard-distributed(mode=ep,devices=8)")
            .expect("build shard-distributed(mode=ep)");
    let moe_ep_res = pm_moe_ep.run(&mut moe_ep_mod).expect("run ep on gemm");
    assert!(moe_ep_res[0].changed);
    let moe_ep_ir = print_mlir(&moe_ep_mod);
    assert!(moe_ep_ir.contains("castforge.ep_sharded = true"));
    assert!(moe_ep_ir.contains("castforge.ep_devices = 8"));
    assert!(moe_ep_ir.contains("castforge.ep_dispatch = \"all_to_all\""));

    // 10. Test Mamba-2 SSD Chunk-Scan, Blackwell tcgen05 TMEM & Warp Specialization,
    //     DoRA Weight Decomposition, QAT Straight-Through Estimator, and Selective Remat
    let mut sota4_mod = parse_mlir(mlir).expect("parse mlir for mamba2/tcgen05/dora/qat/remat");
    let mut pm_sota4 = PassManager::from_pipeline_str(
        &registry,
        concat!(
            "fuse-attention(kind=mamba2,state_dim=128,chunk_size=64),",
            "schedule-loops(tensor_cores=tcgen05,warp_specialized=true,tmem_columns=512),",
            "merge-weights(method=dora),",
            "quantize(scheme=qat_ste,bits=8),",
            "optimize-memory(mode=selective_remat)"
        ),
    )
    .expect("build mamba2 + tcgen05 + dora + qat_ste + selective_remat pipeline");
    let sota4_res = pm_sota4.run(&mut sota4_mod).expect("run sota4 pipeline");
    assert!(sota4_res.iter().all(|r| r.changed));
    let sota4_ir = print_mlir(&sota4_mod);
    assert!(sota4_ir.contains("castforge.ssm_chunk_scan = true"));
    assert!(sota4_ir.contains("castforge.ssm_kind = \"mamba2\""));
    assert!(sota4_ir.contains("castforge.ssm_state_dim = 128"));
    assert!(sota4_ir.contains("castforge.ssm_chunk_size = 64"));
    assert!(sota4_ir.contains("castforge.tensor_core_intrinsic = \"tcgen05.mma"));
    assert!(sota4_ir.contains("castforge.tmem_allocated = true"));
    assert!(sota4_ir.contains("castforge.tmem_columns = 512"));
    assert!(sota4_ir.contains("castforge.warp_specialized = true"));
    assert!(sota4_ir.contains("castforge.dora_baked = true"));
    assert!(sota4_ir.contains("castforge.qat_ste = true"));
    assert!(sota4_ir.contains("castforge.selective_remat = \"checkpoint\""));
}
