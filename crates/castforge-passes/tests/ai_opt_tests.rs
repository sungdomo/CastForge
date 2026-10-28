use castforge_codegen::{CodegenTarget, generate_code};
use castforge_mlir::{parse_mlir, print_mlir};
use castforge_passes::{PassManager, PassRegistry};

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
