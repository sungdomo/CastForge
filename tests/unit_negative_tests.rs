//! Isolated unit tests and negative tests for MLIR parser/lexer error paths,
//! SSA & StableHLO shape verification (`verify-shapes`, `verify-equiv`),
//! pass registry/pipeline error handling, and single-rule canonicalization.

use castforge::codegen::verify_module_numerics;
use castforge::ir::{parse_mlir, parse_type, print_mlir};
use castforge::passes::{PassManager, PassRegistry};

#[test]
fn test_parser_and_type_negative_cases() {
    let invalid_inputs = [
        // Missing closing brace on module
        "module @unclosed {\n  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {\n    \
         func.return %x : tensor<2xf32>\n  }",
        // Undefined SSA value reference `%missing`
        "module @undef_val {\n  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {\n    \
         %y = stablehlo.add %x, %missing : tensor<2xf32>\n    \
         func.return %y : tensor<2xf32>\n  }\n}",
        // Invalid tensor element type
        "module @bad_dtype {\n  func.func @main(%x: tensor<2xbadtype>) -> tensor<2xf32> {\n    \
         func.return %x : tensor<2xf32>\n  }\n}",
        // Unterminated string literal
        "module @unterminated_str {\n  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {\n    \
         %y = \"stablehlo.add(%x, %x) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>\n    \
         func.return %y : tensor<2xf32>\n  }\n}",
        // Malformed function header missing `@name`
        "module @no_fn_name {\n  func.func (%x: tensor<2xf32>) -> tensor<2xf32> {\n    \
         func.return %x : tensor<2xf32>\n  }\n}",
    ];

    for (idx, bad_mlir) in invalid_inputs.iter().enumerate() {
        let res = parse_mlir(bad_mlir);
        assert!(
            res.is_err(),
            "expected parse_mlir to reject invalid input #{idx}:\n{bad_mlir}"
        );
    }

    assert!(parse_type("tensor<4x8xunknown_dtype>", 1).is_err());
    assert!(parse_type("tensor<4x8xf32", 1).is_err());
}

#[test]
fn test_verify_shapes_negative_cases() {
    let registry = PassRegistry::with_builtin_passes();
    let bad_shape_modules = [
        // 1. Elementwise `stablehlo.add` incompatible static dimensions (4 vs 8)
        r#"
module @bad_add {
  func.func @main(%a: tensor<2x4xf32>, %b: tensor<2x8xf32>) -> tensor<2x4xf32> {
    %0 = "stablehlo.add"(%a, %b) : (tensor<2x4xf32>, tensor<2x8xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
"#,
        // 2. `stablehlo.dot_general` contracting dimension mismatch (lhs[1]=4 vs rhs[0]=8)
        r#"
module @bad_dot {
  func.func @main(%a: tensor<2x4xf32>, %b: tensor<8x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<8x3xf32>) -> tensor<2x3xf32>
    func.return %0 : tensor<2x3xf32>
  }
}
"#,
        // 3. `stablehlo.transpose` permutation out of bounds / dimension mismatch
        r#"
module @bad_transpose {
  func.func @main(%a: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.transpose %a, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
"#,
        // 4. `stablehlo.reshape` element count mismatch (8 vs 12)
        r#"
module @bad_reshape {
  func.func @main(%a: tensor<2x4xf32>) -> tensor<3x4xf32> {
    %0 = stablehlo.reshape %a : (tensor<2x4xf32>) -> tensor<3x4xf32>
    func.return %0 : tensor<3x4xf32>
  }
}
"#,
        // 5. `stablehlo.slice` output dimension mismatch (limit - start = 2 vs output 3)
        r#"
module @bad_slice {
  func.func @main(%a: tensor<4x4xf32>) -> tensor<3x4xf32> {
    %0 = stablehlo.slice %a
      {start_indices = [0, 0], limit_indices = [2, 4], strides = [1, 1]}
      : (tensor<4x4xf32>) -> tensor<3x4xf32>
    func.return %0 : tensor<3x4xf32>
  }
}
"#,
        // 6. `stablehlo.concatenate` axis sum mismatch (2 + 2 = 4 vs output 5)
        r#"
module @bad_concat {
  func.func @main(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<5x4xf32> {
    %0 = stablehlo.concatenate %a, %b {dimension = 0}
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<5x4xf32>
    func.return %0 : tensor<5x4xf32>
  }
}
"#,
        // 7. `stablehlo.reduce` axis out of bounds (axis 3 on rank-2 tensor)
        r#"
module @bad_reduce {
  func.func @main(%a: tensor<2x4xf32>) -> tensor<2xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<f32>
    %0 = stablehlo.reduce(%a init: %z)
      applies stablehlo.add across dimensions = [3]
      : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }
}
"#,
        // 8. `stablehlo.broadcast_in_dim` out-of-bounds broadcast axis
        r#"
module @bad_bcast {
  func.func @main(%a: tensor<2xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.broadcast_in_dim %a, dims = [5]
      : (tensor<2xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
"#,
        // 9. `stablehlo.select` mismatched predicate rank (rank 1 vs output rank 2)
        r#"
module @bad_select {
  func.func @main(%p: tensor<2xi1>, %a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.select %p, %a, %b
      : (tensor<2xi1>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %0 : tensor<2x4xf32>
  }
}
"#,
        // 10. Unary op (`stablehlo.exponential`) input/output shape mismatch
        r#"
module @bad_unary {
  func.func @main(%a: tensor<2x4xf32>) -> tensor<2x8xf32> {
    %0 = "stablehlo.exponential"(%a) : (tensor<2x4xf32>) -> tensor<2x8xf32>
    func.return %0 : tensor<2x8xf32>
  }
}
"#,
    ];

    for (idx, mlir) in bad_shape_modules.iter().enumerate() {
        let mut module = parse_mlir(mlir).expect("parse structurally valid MLIR");
        let mut pm = PassManager::from_pipeline_str(&registry, "verify-shapes").unwrap();
        let res = pm.run(&mut module);
        assert!(
            res.is_err(),
            "expected verify-shapes to reject bad shape module #{idx}:\n{mlir}"
        );
    }
}

#[test]
fn test_pass_registry_and_numerical_equivalence_negative_cases() {
    let registry = PassRegistry::with_builtin_passes();

    // Unknown pass name must return Err
    assert!(PassManager::from_pipeline_str(&registry, "nonexistent-pass-xyz").is_err());
    assert!(PassManager::from_pipeline_str(&registry, "dce,invalid-pass-name").is_err());

    // Numerical equivalence verifier must detect divergent outputs
    let m1 = parse_mlir(
        "module @m1 {\n  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {\n    \
         %y = stablehlo.add %x, %x : tensor<2xf32>\n    func.return %y : tensor<2xf32>\n  }\n}\n",
    )
    .unwrap();
    let m2 = parse_mlir(
        "module @m2 {\n  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {\n    \
         %y = stablehlo.multiply %x, %x : tensor<2xf32>\n    \
         func.return %y : tensor<2xf32>\n  }\n}\n",
    )
    .unwrap();
    let rep = verify_module_numerics(&m1, &m2, 1e-4).unwrap();
    assert!(
        !rep.passed,
        "x + x and x * x should not be numerically equivalent"
    );
    assert!(rep.max_abs_diff > 1e-2);
}

#[test]
fn test_isolated_canonicalization_unit_rules() {
    let registry = PassRegistry::with_builtin_passes();

    // 1. Isolated Algebraic Simplification (`simplify-algebraic(mode=algebraic)`):
    //    x + 0 -> x, x - x -> 0, x * 1 -> x, x / 1 -> x
    let alg_mlir = r#"
module @alg_unit {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %z = stablehlo.constant dense<0.0> : tensor<4xf32>
    %o = stablehlo.constant dense<1.0> : tensor<4xf32>
    %a = stablehlo.add %x, %z : tensor<4xf32>
    %b = stablehlo.multiply %a, %o : tensor<4xf32>
    %c = stablehlo.divide %b, %o : tensor<4xf32>
    func.return %c : tensor<4xf32>
  }
}
"#;
    let orig_alg = parse_mlir(alg_mlir).unwrap();
    let mut opt_alg = orig_alg.clone();
    PassManager::from_pipeline_str(&registry, "simplify-algebraic(mode=algebraic),dce")
        .unwrap()
        .run(&mut opt_alg)
        .unwrap();
    let alg_text = print_mlir(&opt_alg);
    assert!(!alg_text.contains("stablehlo.add"));
    assert!(!alg_text.contains("stablehlo.multiply"));
    assert!(!alg_text.contains("stablehlo.divide"));
    assert!(
        verify_module_numerics(&orig_alg, &opt_alg, 1e-6)
            .unwrap()
            .passed
    );

    // 2. Isolated View Folding (`fold-views`):
    //    transpose(transpose(x, [1,0]), [1,0]) -> x
    //    reshape(reshape(x, [8]), [2,4]) -> x
    let view_mlir = r#"
module @view_unit {
  func.func @main(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %t1 = stablehlo.transpose %x, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
    %t2 = stablehlo.transpose %t1, dims = [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %r1 = stablehlo.reshape %t2 : (tensor<2x4xf32>) -> tensor<8xf32>
    %r2 = stablehlo.reshape %r1 : (tensor<8xf32>) -> tensor<2x4xf32>
    func.return %r2 : tensor<2x4xf32>
  }
}
"#;
    let orig_view = parse_mlir(view_mlir).unwrap();
    let mut opt_view = orig_view.clone();
    PassManager::from_pipeline_str(&registry, "fold-views,dce")
        .unwrap()
        .run(&mut opt_view)
        .unwrap();
    let view_text = print_mlir(&opt_view);
    assert!(!view_text.contains("stablehlo.transpose"));
    assert!(!view_text.contains("stablehlo.reshape"));
    assert!(
        verify_module_numerics(&orig_view, &opt_view, 1e-6)
            .unwrap()
            .passed
    );
}
