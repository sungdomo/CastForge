use castforge::codegen::{
    CodegenTarget, GemmLoopOrder, TargetConfig, generate_code, generate_code_with_config,
    verify_module_numerics,
};
use castforge::ir::{parse_mlir, print_mlir};
use castforge::passes::{PassManager, PassRegistry};

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
fn test_modular_a_to_b_translator_chain_and_loop_ir_passes() {
    use castforge::codegen::CodegenTranslator;
    use castforge::ir::{MlirParseTranslator, Payload, TranslatorChain};
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
fn test_pad_for_tiling_split_gemm_unroll_calls_hoist_all_reduce_instrument_and_strip() {
    let registry = PassRegistry::with_builtin_passes();
    assert_eq!(registry.list().count(), 115);
    for (name, _) in registry.list() {
        let pass = registry
            .create_pass(name, &Default::default())
            .expect("instantiate registered pass");
        assert_eq!(
            pass.name(),
            name,
            "Registered pass '{name}' returned mismatched Pass::name() '{}'",
            pass.name()
        );
        for &child in pass.child_passes() {
            assert!(
                registry.list().any(|(registered, _)| registered == child),
                "Composite pass '{name}' references unregistered child pass '{child}'"
            );
        }
    }

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
