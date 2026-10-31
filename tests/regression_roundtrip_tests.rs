//! Systematic numerical round-trip & regression tests for composite fusion,
//! inverse decomposition (`decompose-composites`), and native JIT execution
//! across both `CodegenTarget::Rust` and `CodegenTarget::C`.

use castforge::codegen::{
    CodegenTarget, TargetConfig, generate_code_with_config, jit_execute_and_benchmark,
    verify_module_numerics,
};
use castforge::ir::{parse_mlir, print_mlir};
use castforge::passes::{PassManager, PassRegistry};

fn assert_fuse_decompose_roundtrip(mlir: &str, fuse_pipeline: &str, expected_composite: &str) {
    let registry = PassRegistry::with_builtin_passes();
    let orig = parse_mlir(mlir).expect("parse original MLIR");
    orig.verify().expect("verify original MLIR");

    let mut fused = orig.clone();
    let mut fuse_pm =
        PassManager::from_pipeline_str(&registry, fuse_pipeline).expect("build fuse pipeline");
    fuse_pm.run(&mut fused).expect("run fuse pipeline");
    fused.verify().expect("verify fused MLIR");

    let fused_text = print_mlir(&fused);
    assert!(
        fused_text.contains(expected_composite),
        "expected composite `{expected_composite}` in:\n{fused_text}"
    );

    let rep_fused = verify_module_numerics(&orig, &fused, 1e-4).expect("eval fused");
    assert!(
        rep_fused.passed,
        "orig vs fused mismatch for `{expected_composite}`: max_abs={}",
        rep_fused.max_abs_diff
    );

    let mut decomposed = fused.clone();
    let mut dec_pm =
        PassManager::from_pipeline_str(&registry, "decompose-composites,dce,verify-shapes")
            .expect("build decompose pipeline");
    dec_pm.run(&mut decomposed).expect("run decompose pipeline");
    decomposed.verify().expect("verify decomposed MLIR");

    let dec_text = print_mlir(&decomposed);
    assert!(
        !dec_text.contains(expected_composite),
        "composite `{expected_composite}` was not decomposed:\n{dec_text}"
    );

    let rep_dec = verify_module_numerics(&orig, &decomposed, 1e-4).expect("eval decomposed");
    assert!(
        rep_dec.passed,
        "orig vs decomposed mismatch for `{expected_composite}`: max_abs={}",
        rep_dec.max_abs_diff
    );
}

#[test]
fn test_norm_and_softmax_fuse_decompose_roundtrips() {
    // 1. RMSNorm & Residual+RMSNorm
    let rms_mlir = r#"
module @rms_roundtrip {
  func.func @main(
    %x: tensor<2x4xf32>,
    %r: tensor<2x4xf32>,
    %w: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %h = stablehlo.add %x, %r : tensor<2x4xf32>
    %sq = stablehlo.multiply %h, %h : tensor<2x4xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%sq init: %z)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %n = stablehlo.constant dense<4.0> : tensor<2xf32>
    %mean = stablehlo.divide %sum, %n : tensor<2xf32>
    %eps = stablehlo.constant dense<0.00001> : tensor<2xf32>
    %ve = stablehlo.add %mean, %eps : tensor<2xf32>
    %inv = stablehlo.rsqrt %ve : tensor<2xf32>
    %b = stablehlo.broadcast_in_dim %inv, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %norm = stablehlo.multiply %h, %b : tensor<2x4xf32>
    %out = stablehlo.multiply %norm, %w : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(rms_mlir, "fuse-rmsnorm", "castforge.fused_rmsnorm");
    assert_fuse_decompose_roundtrip(
        rms_mlir,
        "fuse-residual-norm",
        "castforge.fused_residual_rmsnorm",
    );

    // 2. LayerNorm (with gamma & beta)
    let ln_mlir = r#"
module @ln_roundtrip {
  func.func @main(
    %x: tensor<2x4xf32>,
    %g: tensor<2x4xf32>,
    %b: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %n = stablehlo.constant dense<4.0> : tensor<2xf32>
    %sum = stablehlo.reduce(%x init: %z)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean = stablehlo.divide %sum, %n : tensor<2xf32>
    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %xc = stablehlo.subtract %x, %mean_b : tensor<2x4xf32>
    %sq = stablehlo.multiply %xc, %xc : tensor<2x4xf32>
    %vsum = stablehlo.reduce(%sq init: %z)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %var = stablehlo.divide %vsum, %n : tensor<2xf32>
    %eps = stablehlo.constant dense<0.00001> : tensor<2xf32>
    %ve = stablehlo.add %var, %eps : tensor<2xf32>
    %inv = stablehlo.rsqrt %ve : tensor<2xf32>
    %inv_b = stablehlo.broadcast_in_dim %inv, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %norm = stablehlo.multiply %xc, %inv_b : tensor<2x4xf32>
    %scaled = stablehlo.multiply %norm, %g : tensor<2x4xf32>
    %out = stablehlo.add %scaled, %b : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(ln_mlir, "fuse-layernorm", "castforge.fused_layernorm");

    // 3. Softmax & Log-Softmax
    let sm_mlir = r#"
module @sm_roundtrip {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %e = stablehlo.exponential %x : tensor<2x4xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %s = stablehlo.reduce(%e init: %z)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %sb = stablehlo.broadcast_in_dim %s, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %out = stablehlo.divide %e, %sb : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(sm_mlir, "fuse-softmax", "castforge.fused_softmax");

    let lsm_mlir = r#"
module @lsm_roundtrip {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %e = stablehlo.exponential %x : tensor<2x4xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %s = stablehlo.reduce(%e init: %z)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %sb = stablehlo.broadcast_in_dim %s, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %sm = stablehlo.divide %e, %sb : tensor<2x4xf32>
    %out = stablehlo.log %sm : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(lsm_mlir, "fuse-softmax", "castforge.fused_log_softmax");
}

#[test]
fn test_mlp_epilogue_and_elementwise_fuse_decompose_roundtrips() {
    // 1. SwiGLU & Gated MLP
    let swiglu_mlir = r#"
module @swiglu_rt {
  func.func @main(%x: tensor<2x2xf32>, %wd: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %wg = stablehlo.constant dense<[[0.5, -0.3], [0.2, 0.4]]> : tensor<2x2xf32>
    %wu = stablehlo.constant dense<[[0.3, 0.6], [-0.4, 0.2]]> : tensor<2x2xf32>
    %g = stablehlo.dot_general %x, %wg, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %u = stablehlo.dot_general %x, %wu, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %z = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %o = stablehlo.constant dense<1.0> : tensor<2x2xf32>
    %ng = stablehlo.subtract %z, %g : tensor<2x2xf32>
    %eg = stablehlo.exponential %ng : tensor<2x2xf32>
    %den = stablehlo.add %o, %eg : tensor<2x2xf32>
    %silu = stablehlo.divide %g, %den : tensor<2x2xf32>
    %hid = stablehlo.multiply %silu, %u : tensor<2x2xf32>
    %out = stablehlo.dot_general %hid, %wd, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %out : tensor<2x2xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(swiglu_mlir, "fuse-swiglu", "castforge.fused_swiglu");
    assert_fuse_decompose_roundtrip(swiglu_mlir, "fuse-gated-mlp", "castforge.fused_gated_mlp");

    // 2. GELU & GeGLU
    let geglu_mlir = r#"
module @geglu_rt {
  func.func @main(%g: tensor<2x4xf32>, %u: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<2x4xf32>
    %one = stablehlo.constant dense<1.0> : tensor<2x4xf32>
    %c0 = stablehlo.constant dense<0.79788456> : tensor<2x4xf32>
    %c1 = stablehlo.constant dense<0.044715> : tensor<2x4xf32>
    %g2 = stablehlo.multiply %g, %g : tensor<2x4xf32>
    %g3 = stablehlo.multiply %g2, %g : tensor<2x4xf32>
    %cg3 = stablehlo.multiply %g3, %c1 : tensor<2x4xf32>
    %in_s = stablehlo.add %g, %cg3 : tensor<2x4xf32>
    %arg = stablehlo.multiply %in_s, %c0 : tensor<2x4xf32>
    %th = stablehlo.tanh %arg : tensor<2x4xf32>
    %th1 = stablehlo.add %one, %th : tensor<2x4xf32>
    %hg = stablehlo.multiply %g, %half : tensor<2x4xf32>
    %gelu = stablehlo.multiply %hg, %th1 : tensor<2x4xf32>
    %out = stablehlo.multiply %gelu, %u : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(geglu_mlir, "fuse-gelu", "castforge.fused_geglu");

    // 3. FMA & Fused RoPE
    let rope_fma_mlir = r#"
module @rope_fma_rt {
  func.func @main(
    %x: tensor<2x4xf32>,
    %cos: tensor<2x4xf32>,
    %sin: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %x_lo = stablehlo.slice %x
      {start_indices = [0, 0], limit_indices = [2, 2], strides = [1, 1]}
      : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %x_hi = stablehlo.slice %x
      {start_indices = [0, 2], limit_indices = [2, 4], strides = [1, 1]}
      : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %z = stablehlo.constant dense<0.0> : tensor<2x2xf32>
    %neg_hi = stablehlo.subtract %z, %x_hi : tensor<2x2xf32>
    %rot = stablehlo.concatenate %neg_hi, %x_lo {dimension = 1}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    %mc = stablehlo.multiply %x, %cos : tensor<2x4xf32>
    %ms = stablehlo.multiply %rot, %sin : tensor<2x4xf32>
    %out = stablehlo.add %mc, %ms : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(rope_fma_mlir, "fuse-rope", "castforge.fused_rope");
    assert_fuse_decompose_roundtrip(rope_fma_mlir, "fuse-fma", "castforge.fma");

    // 4. Matmul Epilogue (bias_tanh_residual) & Map-Reduce
    let epi_mlir = r#"
module @epi_rt {
  func.func @main(
    %x: tensor<2x4xf32>,
    %w: tensor<4x4xf32>,
    %b: tensor<2x4xf32>,
    %r: tensor<2x4xf32>
  ) -> tensor<2xf32> {
    %dot = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %add = stablehlo.add %dot, %b : tensor<2x4xf32>
    %th = stablehlo.tanh %add : tensor<2x4xf32>
    %res = stablehlo.add %th, %r : tensor<2x4xf32>
    %sq = stablehlo.multiply %res, %res : tensor<2x4xf32>
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %red = stablehlo.reduce(%sq init: %z)
      applies stablehlo.add across dimensions = [1]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    func.return %red : tensor<2xf32>
  }
}
"#;
    assert_fuse_decompose_roundtrip(
        epi_mlir,
        "fuse-matmul-epilogue",
        "castforge.fused_matmul_epilogue",
    );
    assert_fuse_decompose_roundtrip(epi_mlir, "fuse-map-reduce", "castforge.fused_map_reduce");
}

#[test]
fn test_rust_and_c_native_jit_execution_parity() {
    let mlir = r#"
module @native_jit_parity {
  func.func @forward(
    %x: tensor<2x4xf32>,
    %w: tensor<4x4xf32>,
    %b: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %dot = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %add = stablehlo.add %dot, %b : tensor<2x4xf32>
    %relu = stablehlo.maximum %add, %z : tensor<2x4xf32>
    %out = stablehlo.tanh %relu : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;
    let registry = PassRegistry::with_builtin_passes();
    let orig = parse_mlir(mlir).expect("parse mlir");
    let mut opt = orig.clone();
    PassManager::from_pipeline_str(&registry, "O2")
        .unwrap()
        .run(&mut opt)
        .unwrap();

    let cfg = TargetConfig::default();

    // 1. Native Rust JIT execution (`rustc -O`)
    let rust_code =
        generate_code_with_config(&opt, CodegenTarget::Rust, &cfg).expect("generate Rust code");
    assert!(rust_code.contains("pub fn forward("));
    let rust_jit = jit_execute_and_benchmark(&opt, CodegenTarget::Rust, &cfg, 1, 2)
        .expect("JIT compile and run Rust target");
    assert!(
        rust_jit.numerics_verified,
        "Rust JIT numerics failed: max_abs_diff={}",
        rust_jit.max_abs_diff
    );

    // 2. Native C JIT execution (`cc -O3`)
    let c_jit = jit_execute_and_benchmark(&opt, CodegenTarget::C, &cfg, 1, 2)
        .expect("JIT compile and run C target");
    assert!(
        c_jit.numerics_verified,
        "C JIT numerics failed: max_abs_diff={}",
        c_jit.max_abs_diff
    );
}
