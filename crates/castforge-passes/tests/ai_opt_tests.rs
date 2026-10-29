use castforge_codegen::{CodegenTarget, generate_code, verify_module_numerics};
use castforge_mlir::{parse_mlir, print_mlir};
use castforge_passes::{PassManager, PassRegistry, analyze_roofline};

const SINK_TRANSPOSE_MLIR: &str = r#"
module @sink_transpose_demo {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %w = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    %scaled = stablehlo.exponential %w : tensor<2x3xf32>
    %w_t = stablehlo.transpose %scaled, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %out = stablehlo.add %arg0, %w : tensor<2x3xf32>
    %out_t = stablehlo.transpose %out, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    %res = stablehlo.add %w_t, %out_t : tensor<3x2xf32>
    func.return %res : tensor<3x2xf32>
  }
}
"#;

#[test]
fn test_sink_transpose_and_constant_fold() {
    let mut module = parse_mlir(SINK_TRANSPOSE_MLIR).expect("parse sink transpose MLIR");
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(&registry, "sink-transpose,fold-constants,dce")
        .expect("valid pipeline");
    let history = pm.run(&mut module).expect("run pipeline");
    assert!(history.iter().any(|r| r.changed));

    let printed = print_mlir(&module);
    // The exponential on constant %w should have had transpose sunk into %w and folded into a
    // 3x2 constant directly without runtime `stablehlo.exponential`.
    assert!(!printed.contains("stablehlo.exponential"));
    assert!(printed.contains("tensor<3x2xf32>"));
}

const FUSE_MATMUL_EPILOGUE_MLIR: &str = r#"
module @fuse_matmul_demo {
  func.func @linear_relu(
    %x: tensor<2x3xf32>,
    %w: tensor<3x4xf32>,
    %bias: tensor<2x4xf32>
  ) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %mm = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %added = stablehlo.add %mm, %bias : tensor<2x4xf32>
    %act = stablehlo.maximum %added, %zero : tensor<2x4xf32>
    func.return %act : tensor<2x4xf32>
  }
}
"#;

#[test]
fn test_fuse_matmul_epilogue_and_codegen() {
    let mut module = parse_mlir(FUSE_MATMUL_EPILOGUE_MLIR).expect("parse matmul epilogue MLIR");
    let registry = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&registry, "fuse-matmul-epilogue").expect("valid pipeline");
    let res = pm.run(&mut module).expect("run fuse-matmul-epilogue");
    assert!(res[0].changed);
    assert_eq!(res[0].ops_created, 1);
    assert_eq!(res[0].ops_removed, 2);

    let printed = print_mlir(&module);
    assert!(printed.contains("stablehlo.composite"));
    assert!(printed.contains("castforge.fused_matmul_epilogue"));
    assert!(!printed.contains("stablehlo.dot_general"));

    let rust_code = generate_code(&module, CodegenTarget::Rust).expect("rust codegen");
    assert!(rust_code.contains("let val = acc +"));
    assert!(rust_code.contains("val.max("));

    let c_code = generate_code(&module, CodegenTarget::C).expect("c codegen");
    assert!(c_code.contains("fmaxf(val,"));
}

const FOLD_NORM_WEIGHTS_MLIR: &str = r#"
module @fold_norm_demo {
  func.func @norm_linear(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %scale = stablehlo.constant dense<[[2.0, 3.0], [2.0, 3.0]]> : tensor<2x2xf32>
    %w = stablehlo.constant dense<[[1.0, 2.0], [4.0, 5.0]]> : tensor<2x2xf32>
    %normed = stablehlo.multiply %x, %scale : tensor<2x2xf32>
    %out = stablehlo.dot_general %normed, %w, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %out : tensor<2x2xf32>
  }
}
"#;

#[test]
fn test_fold_norm_weights() {
    let mut module = parse_mlir(FOLD_NORM_WEIGHTS_MLIR).expect("parse fold norm MLIR");
    let registry = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&registry, "fold-norm-weights").expect("valid pipeline");
    let res = pm.run(&mut module).expect("run fold-norm-weights");
    assert!(res[0].changed);

    let printed = print_mlir(&module);
    // Runtime multiply should be eliminated and absorbed into %w:
    // row 0 scaled by 2.0 -> [2.0, 4.0], row 1 scaled by 3.0 -> [12.0, 15.0]
    assert!(!printed.contains("stablehlo.multiply"));
    assert!(printed.contains("[[2.0, 4.0], [12.0, 15.0]]"));
}

const SOFTMAX_MLIR: &str = r#"
module @softmax_demo {
  func.func @attn_softmax(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %exp = stablehlo.exponential %x : tensor<2x4xf32>
    %sum = stablehlo.reduce %exp, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %denom = stablehlo.broadcast_in_dim %sum, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %out = stablehlo.divide %exp, %denom : tensor<2x4xf32>
    func.return %out : tensor<2x4xf32>
  }
}
"#;

#[test]
fn test_fuse_softmax_and_decompose_composites() {
    let before = parse_mlir(SOFTMAX_MLIR).expect("parse softmax MLIR");
    let mut fused = before.clone();
    let registry = PassRegistry::with_builtin_passes();

    let mut pm =
        PassManager::from_pipeline_str(&registry, "fuse-softmax").expect("fuse-softmax pipeline");
    let res = pm.run(&mut fused).expect("run fuse-softmax");
    assert!(res[0].changed);

    let printed = print_mlir(&fused);
    assert!(printed.contains("castforge.fused_softmax"));
    assert!(!printed.contains("stablehlo.divide"));

    let rep = verify_module_numerics(&before, &fused, 1e-5).expect("verify softmax numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);

    // Decompose back to primitive StableHLO ops and verify numerical equivalence again
    let mut decomposed = fused.clone();
    let mut dec_pm = PassManager::from_pipeline_str(&registry, "decompose-composites")
        .expect("decompose pipeline");
    dec_pm.run(&mut decomposed).expect("run decompose");

    let dec_printed = print_mlir(&decomposed);
    assert!(!dec_printed.contains("castforge.fused_softmax"));
    assert!(dec_printed.contains("stablehlo.divide"));

    let rep2 = verify_module_numerics(&before, &decomposed, 1e-5).expect("verify decomposed");
    assert!(rep2.passed);
}

const RMSNORM_MLIR: &str = r#"
module @rmsnorm_demo {
  func.func @rmsnorm(%x: tensor<2x4xf32>, %weight: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %four = stablehlo.constant dense<4.0> : tensor<2xf32>
    %eps = stablehlo.constant dense<0.00001> : tensor<2xf32>
    %sq = stablehlo.multiply %x, %x : tensor<2x4xf32>
    %sum = stablehlo.reduce %sq, %zero
      {dimensions = [1], reducer = "stablehlo.add"}
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean = stablehlo.divide %sum, %four : tensor<2xf32>
    %var_eps = stablehlo.add %mean, %eps : tensor<2xf32>
    %inv_rms = stablehlo.rsqrt %var_eps : tensor<2xf32>
    %inv_bcast = stablehlo.broadcast_in_dim %inv_rms, dims = [0]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    %normed = stablehlo.multiply %x, %inv_bcast : tensor<2x4xf32>
    %scaled = stablehlo.multiply %normed, %weight : tensor<2x4xf32>
    func.return %scaled : tensor<2x4xf32>
  }
}
"#;

#[test]
fn test_fuse_rmsnorm_and_numeric_verification() {
    let before = parse_mlir(RMSNORM_MLIR).expect("parse rmsnorm MLIR");
    let mut fused = before.clone();
    let registry = PassRegistry::with_builtin_passes();

    let mut pm =
        PassManager::from_pipeline_str(&registry, "fuse-rmsnorm").expect("fuse-rmsnorm pipeline");
    let res = pm.run(&mut fused).expect("run fuse-rmsnorm");
    assert!(res[0].changed);

    let printed = print_mlir(&fused);
    assert!(printed.contains("castforge.fused_rmsnorm"));
    assert!(!printed.contains("stablehlo.rsqrt"));

    let rep = verify_module_numerics(&before, &fused, 1e-5).expect("verify rmsnorm numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);
}

const BAKE_LORA_MLIR: &str = r#"
module @bake_lora_demo {
  func.func @lora_linear(%x: tensor<2x3xf32>) -> tensor<2x2xf32> {
    %w = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>
    %a = stablehlo.constant dense<[[1.0], [0.5], [2.0]]> : tensor<3x1xf32>
    %b = stablehlo.constant dense<[[2.0, -1.0]]> : tensor<1x2xf32>
    %alpha = stablehlo.constant dense<0.5> : tensor<2x2xf32>
    %base = stablehlo.dot_general %x, %w, contracting_dims = [1] x [0]
      : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %xa = stablehlo.dot_general %x, %a, contracting_dims = [1] x [0]
      : (tensor<2x3xf32>, tensor<3x1xf32>) -> tensor<2x1xf32>
    %xab = stablehlo.dot_general %xa, %b, contracting_dims = [1] x [0]
      : (tensor<2x1xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %scaled = stablehlo.multiply %xab, %alpha : tensor<2x2xf32>
    %out = stablehlo.add %base, %scaled : tensor<2x2xf32>
    func.return %out : tensor<2x2xf32>
  }
}
"#;

#[test]
fn test_bake_lora_and_numeric_verification() {
    let before = parse_mlir(BAKE_LORA_MLIR).expect("parse bake lora MLIR");
    let mut baked = before.clone();
    let registry = PassRegistry::with_builtin_passes();

    let mut pm =
        PassManager::from_pipeline_str(&registry, "bake-lora").expect("bake-lora pipeline");
    let res = pm.run(&mut baked).expect("run bake-lora");
    assert!(res[0].changed);

    let printed = print_mlir(&baked);
    // Only 1 dot_general (`%base`) should remain; `%xa`, `%xab`, `%scaled`, `%out` are eliminated!
    assert_eq!(printed.matches("stablehlo.dot_general").count(), 1);
    assert!(!printed.contains("stablehlo.add"));

    let rep = verify_module_numerics(&before, &baked, 1e-5).expect("verify bake-lora numerics");
    assert!(rep.passed, "max_abs_diff = {}", rep.max_abs_diff);
}

const TASK_ARITHMETIC_MLIR: &str = r#"
module @task_arithmetic_demo {
  func.func @forward(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %w_base = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %w_ft1 = stablehlo.constant dense<[[1.1, 4.0], [5.0, 4.05]]> : tensor<2x2xf32>
    %w_ft2 = stablehlo.constant dense<[[1.05, 3.0], [4.0, 4.02]]> : tensor<2x2xf32>
    %w_task = stablehlo.custom_call %w_base, %w_ft1, %w_ft2
      {call_target_name = "castforge.task_arithmetic"}
      : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %out = stablehlo.dot_general %x, %w_task, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    func.return %out : tensor<2x2xf32>
  }
}
"#;

#[test]
fn test_task_arithmetic_and_dare_pruning() {
    let mut module = parse_mlir(TASK_ARITHMETIC_MLIR).expect("parse task arithmetic MLIR");
    let registry = PassRegistry::with_builtin_passes();
    let mut pm =
        PassManager::from_pipeline_str(&registry, "task-arithmetic(lambda=0.5,drop_rate=0.5)")
            .expect("task-arithmetic pipeline");
    let res = pm.run(&mut module).expect("run task-arithmetic");
    assert!(res[0].changed);

    let printed = print_mlir(&module);
    assert!(!printed.contains("castforge.task_arithmetic"));
    // Smallest 2 deltas ([0,0] and [1,1]) are dropped by DARE (drop_rate=0.5),
    // and surviving deltas ([0,1] and [1,0]) are rescaled by 1/(1-0.5)=2.0 * lambda(0.5) = 1.0:
    // w_task[0,0] = 1.0, w_task[0,1] = 2.0 + (2.0 + 1.0) = 5.0,
    // w_task[1,0] = 3.0 + (2.0 + 1.0) = 6.0, w_task[1,1] = 4.0
    assert!(printed.contains("[[1.0, 5.0], [6.0, 4.0]]"));

    let roof = analyze_roofline(&module, 10.0);
    assert!(roof.total_flops > 0);
    assert!(roof.total_bytes_transferred > 0);
    assert!(roof.arithmetic_intensity > 0.0);
}

const HORIZONTAL_DOT_AND_CANON_MLIR: &str = r#"
module @qkv_horizontal_and_canon {
  func.func @forward(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %wq = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %wk = stablehlo.constant dense<[[0.5, 1.5], [2.5, 3.5]]> : tensor<2x2xf32>
    %c1 = stablehlo.constant dense<2.0> : tensor<2x2xf32>
    %c2 = stablehlo.constant dense<3.0> : tensor<2x2xf32>
    %q = stablehlo.dot_general %x, %wq, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %k = stablehlo.dot_general %x, %wk, contracting_dims = [1] x [0]
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %cat = stablehlo.concatenate %q, %k {dimension = 1}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    %slice_q = stablehlo.slice %cat
      {start_indices = [0, 0], limit_indices = [2, 2], strides = [1, 1]}
      : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %bcast = stablehlo.broadcast_in_dim %slice_q, dims = [0, 1]
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %add1 = stablehlo.add %c1, %bcast : tensor<2x2xf32>
    %add2 = stablehlo.add %add1, %c2 : tensor<2x2xf32>
    %max_id = stablehlo.maximum %add2, %add2 : tensor<2x2xf32>
    func.return %max_id : tensor<2x2xf32>
  }
}
"#;

#[test]
fn test_horizontal_dot_broadcast_slice_concat_and_reassociation() {
    let before = parse_mlir(HORIZONTAL_DOT_AND_CANON_MLIR).expect("parse horizontal dot MLIR");

    // 1. Test `fuse-horizontal-dot` in isolation: merges %q and %k into 1 `dot_general` + 2 slices
    let mut horiz = before.clone();
    let registry = PassRegistry::with_builtin_passes();
    let mut pm_h = PassManager::from_pipeline_str(&registry, "fuse-horizontal-dot")
        .expect("fuse-horizontal-dot pipeline");
    let res_h = pm_h.run(&mut horiz).expect("run fuse-horizontal-dot");
    assert!(res_h[0].changed);
    let printed_h = print_mlir(&horiz);
    assert_eq!(printed_h.matches("stablehlo.dot_general").count(), 1);
    let rep_h =
        verify_module_numerics(&before, &horiz, 1e-5).expect("verify horizontal dot numerics");
    assert!(rep_h.passed, "max_abs_diff = {}", rep_h.max_abs_diff);

    // 2. Test `canonicalize` (`fold-slice-concat`, `fold-broadcast`, `simplify-algebraic`, `dce`)
    let mut canon = before.clone();
    let mut pm_c =
        PassManager::from_pipeline_str(&registry, "canonicalize").expect("canonicalize pipeline");
    let res_c = pm_c.run(&mut canon).expect("run canonicalize");
    assert!(res_c[0].changed);
    let printed_c = print_mlir(&canon);
    // `slice_q` folds through `concatenate(%q, %k)` directly to `%q` (killing `%k` and `%cat`),
    // `bcast` is eliminated, `(c1 + q) + c2` reassociates to `q + 5.0`, and `max(x, x)` folds!
    assert!(!printed_c.contains("stablehlo.concatenate"));
    assert!(!printed_c.contains("stablehlo.broadcast_in_dim"));
    assert!(!printed_c.contains("stablehlo.maximum"));
    assert_eq!(printed_c.matches("stablehlo.add").count(), 1);
    assert!(printed_c.contains("dense<5.0>"));

    let rep_c = verify_module_numerics(&before, &canon, 1e-5).expect("verify canonicalize");
    assert!(rep_c.passed, "max_abs_diff = {}", rep_c.max_abs_diff);
}
