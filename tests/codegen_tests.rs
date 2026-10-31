use castforge::codegen::{
    BottleneckKind, CodegenTarget, GemmLoopOrder, HardwareRoofline, SimdIsa, TargetConfig,
    analyze_module_cost_with_roofline, build_memory_plan, evaluate_module, generate_code,
    generate_code_with_config, verify_module_numerics,
};
use castforge::ir::{parse_mlir, print_mlir};
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
fn test_iterative_cli_ir_codegen_audit_fixes() {
    // 1. Batched 3D GEMM cost model FLOPs & compare/select/nan_guard/static_quantize FLOPs
    let cost_mlir = concat!(
        "module @cost_audit {\n",
        "  func.func @forward(%a: tensor<2x3x4xf32>, %b: tensor<2x4x5xf32>) ",
        "-> tensor<2x3x5xf32> {\n",
        "    %mm = \"stablehlo.dot_general\"(%a, %b) ",
        "{lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], ",
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]} ",
        ": (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>\n",
        "    %cmp = \"stablehlo.compare\"(%mm, %mm) {comparison_direction = \"GT\"} ",
        ": (tensor<2x3x5xf32>, tensor<2x3x5xf32>) -> tensor<2x3x5xi1>\n",
        "    %sel = \"stablehlo.select\"(%cmp, %mm, %mm) ",
        ": (tensor<2x3x5xi1>, tensor<2x3x5xf32>, tensor<2x3x5xf32>) -> tensor<2x3x5xf32>\n",
        "    %ng = \"castforge.nan_guard\"(%sel) {clamp_max = 65504.0} ",
        ": (tensor<2x3x5xf32>) -> tensor<2x3x5xf32>\n",
        "    %sq = \"castforge.static_quantize\"(%ng) {bits = 8, scale = 0.01} ",
        ": (tensor<2x3x5xf32>) -> tensor<2x3x5xf32>\n",
        "    func.return %sq : tensor<2x3x5xf32>\n",
        "  }\n",
        "}\n"
    );
    let cost_mod = parse_mlir(cost_mlir).expect("parse cost_audit");
    let cost_rep = castforge::codegen::analyze_module_cost(&cost_mod);
    // 3D dot_general: 2 * (2*3*5) * 4 = 240 FLOPs
    // compare (30) + select (30) + nan_guard (30) + static_quantize (90) = 180 FLOPs
    assert_eq!(cost_rep.total_flops, 240 + 180);

    // 2. Reduce along axis 0 on 2D tensor + 2D strided Slice + 2D Concatenate + Iota
    let ops_mlir = concat!(
        "module @ops_2d {\n",
        "  func.func @forward(%x: tensor<2x3xf32>) -> tensor<3xf32> {\n",
        "    %z = stablehlo.constant dense<0.0> : tensor<f32>\n",
        "    %red0 = stablehlo.reduce %x, %z ",
        "{dimensions = [0], reducer = \"stablehlo.maximum\"} ",
        ": (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>\n",
        "    func.return %red0 : tensor<3xf32>\n",
        "  }\n",
        "}\n"
    );
    let ops_mod = parse_mlir(ops_mlir).expect("parse ops_2d");
    // Input [[1, 5, 2], [4, 3, 6]] -> max along axis 0 -> [4, 5, 6]
    let eval_out =
        evaluate_module(&ops_mod, &[vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0]]).expect("eval red0");
    assert_eq!(eval_out[0], vec![4.0, 5.0, 6.0]);
    let c_code = generate_code(&ops_mod, CodegenTarget::C).expect("c red0");
    assert!(c_code.contains("[k * 3 + o]"));
    let rs_code = generate_code(&ops_mod, CodegenTarget::Rust).expect("rust red0");
    assert!(rs_code.contains("[k * 3 + o]"));
    let wgsl_code = generate_code(&ops_mod, CodegenTarget::Wgsl).expect("wgsl red0");
    assert!(wgsl_code.contains("[c * 3u + r]"));

    // 3. Nested region liveness in compute_block_liveness & viz graph recursion
    let while_mlir = concat!(
        "module @while_live {\n",
        "  func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {\n",
        "    %step = stablehlo.constant dense<[1.0, 1.0]> : tensor<2xf32>\n",
        "    %w = \"stablehlo.while\"(%x) : (tensor<2xf32>) -> tensor<2xf32> {\n",
        "      %c = \"stablehlo.compare\"(%x, %step) {comparison_direction = \"LT\"} ",
        ": (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>\n",
        "      %next = stablehlo.add %x, %step : tensor<2xf32>\n",
        "      func.return %next : tensor<2xf32>\n",
        "    }\n",
        "    func.return %w : tensor<2xf32>\n",
        "  }\n",
        "}\n"
    );
    let while_mod = parse_mlir(while_mlir).expect("parse while_live");
    let fn_id = while_mod.functions()[0];
    let blk_id = while_mod.regions[while_mod.ops[fn_id].regions[0]].blocks[0];
    let intervals = castforge::codegen::compute_block_liveness(&while_mod, blk_id);
    // %step is defined at step 1 and used inside the nested regions of %w at step 2
    let step_iv = intervals
        .iter()
        .find(|iv| iv.first_def == 1)
        .expect("step iv");
    assert_eq!(step_iv.last_use, 2);
    let viz_json =
        castforge::ir::viz::render_module(&while_mod, castforge::ir::viz::VizFormat::Json);
    assert!(viz_json.contains("stablehlo.compare"));

    // 4. ONNX ReduceMax/ReduceMin/ReduceProd & Slice/Pad round-trip
    let onnx_rt_mlir = concat!(
        "module @onnx_rt {\n",
        "  func.func @forward(%x: tensor<2x2xf32>) -> tensor<2xf32> {\n",
        "    %z = stablehlo.constant dense<1.0> : tensor<f32>\n",
        "    %sl = stablehlo.slice %x ",
        "{start_indices = [0, 0], limit_indices = [2, 2], strides = [1, 1]} ",
        ": (tensor<2x2xf32>) -> tensor<2x2xf32>\n",
        "    %r = stablehlo.reduce %sl, %z ",
        "{dimensions = [1], reducer = \"stablehlo.multiply\"} ",
        ": (tensor<2x2xf32>, tensor<f32>) -> tensor<2xf32>\n",
        "    func.return %r : tensor<2xf32>\n",
        "  }\n",
        "}\n"
    );
    let onnx_orig = parse_mlir(onnx_rt_mlir).expect("parse onnx_rt");
    let onnx_json = castforge::ir::export_onnx_json(&onnx_orig).expect("export onnx");
    let onnx_reimp = castforge::ir::import_onnx_json(&onnx_json).expect("reimport onnx");
    let rep_onnx =
        verify_module_numerics(&onnx_orig, &onnx_reimp, 1e-5).expect("verify onnx roundtrip");
    assert!(rep_onnx.passed);
}
