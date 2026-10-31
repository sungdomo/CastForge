//! Property-based fuzzing, random pass-order invariance, and pass idempotency
//! tests across synthesized StableHLO DAGs.

use castforge::codegen::verify_module_numerics;
use castforge::ir::{parse_mlir, print_mlir};
use castforge::passes::{PassManager, PassRegistry};

/// Simple deterministic SplitMix64 PRNG for reproducible fuzzing without external crates.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
    }

    fn next_u64(&mut self) -> u64 {
        let mut z = self.0;
        self.0 = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

/// Synthesizes a random valid 2D `tensor<2x4xf32>` StableHLO DAG with `num_steps` operations.
fn synthesize_random_dag(rng: &mut Rng, num_steps: usize) -> String {
    let mut lines = vec![
        "module @fuzz_dag {".to_string(),
        "  func.func @main(%x: tensor<2x4xf32>, %y: tensor<2x4xf32>, %w: tensor<4x4xf32>) \
         -> tensor<2x4xf32> {"
            .to_string(),
        "    %c0 = stablehlo.constant dense<0.0> : tensor<2x4xf32>".to_string(),
        "    %c1 = stablehlo.constant dense<1.0> : tensor<2x4xf32>".to_string(),
        "    %ch = stablehlo.constant dense<0.5> : tensor<2x4xf32>".to_string(),
    ];

    let mut pool = vec![
        "%x".to_string(),
        "%y".to_string(),
        "%c0".to_string(),
        "%c1".to_string(),
        "%ch".to_string(),
    ];

    for step in 0..num_steps {
        let v = format!("%v{step}");
        let lhs = pool[rng.gen_range(pool.len())].clone();
        let rhs = pool[rng.gen_range(pool.len())].clone();
        match rng.gen_range(9) {
            0 => {
                lines.push(format!(
                    "    {v} = stablehlo.add {lhs}, {rhs} : tensor<2x4xf32>"
                ));
            }
            1 => {
                lines.push(format!(
                    "    {v} = stablehlo.subtract {lhs}, {rhs} : tensor<2x4xf32>"
                ));
            }
            2 => {
                lines.push(format!(
                    "    {v} = stablehlo.multiply {lhs}, {rhs} : tensor<2x4xf32>"
                ));
            }
            3 => {
                lines.push(format!(
                    "    {v} = stablehlo.maximum {lhs}, %c0 : tensor<2x4xf32>"
                ));
            }
            4 => {
                lines.push(format!("    {v} = stablehlo.tanh {lhs} : tensor<2x4xf32>"));
            }
            5 => {
                lines.push(format!("    {v} = stablehlo.abs {lhs} : tensor<2x4xf32>"));
            }
            6 => {
                // Double transpose back-to-back (canonicalizable to identity)
                let t = format!("%t{step}");
                lines.push(format!(
                    "    {t} = stablehlo.transpose {lhs}, dims = [1, 0] \
                     : (tensor<2x4xf32>) -> tensor<4x2xf32>"
                ));
                lines.push(format!(
                    "    {v} = stablehlo.transpose {t}, dims = [1, 0] \
                     : (tensor<4x2xf32>) -> tensor<2x4xf32>"
                ));
            }
            7 => {
                // Double reshape back-to-back (canonicalizable to identity)
                let r = format!("%r{step}");
                lines.push(format!(
                    "    {r} = stablehlo.reshape {lhs} : (tensor<2x4xf32>) -> tensor<8xf32>"
                ));
                lines.push(format!(
                    "    {v} = stablehlo.reshape {r} : (tensor<8xf32>) -> tensor<2x4xf32>"
                ));
            }
            _ => {
                // Matmul with %w: (2x4) x (4x4) -> (2x4)
                lines.push(format!(
                    "    {v} = stablehlo.dot_general {lhs}, %w, contracting_dims = [1] x [0] \
                     : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>"
                ));
            }
        }
        pool.push(v);
    }

    let last = pool.last().unwrap();
    lines.push(format!("    func.return {last} : tensor<2x4xf32>"));
    lines.push("  }".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

#[test]
fn test_random_dag_pass_order_numerical_invariance() {
    let registry = PassRegistry::with_builtin_passes();
    let candidate_passes = [
        "simplify-algebraic",
        "fold-views",
        "cse",
        "dce",
        "fuse-elementwise",
        "fuse-matmul-epilogue",
        "fuse-fma",
        "decompose-composites",
        "topological-sort",
        "normalize-ssa",
        "schedule-loops",
        "optimize-memory",
        "strip-attributes",
    ];

    for seed in 1..=30u64 {
        let mut rng = Rng::new(seed);
        let mlir = synthesize_random_dag(&mut rng, 10 + (seed as usize % 8));
        let orig = parse_mlir(&mlir).expect("parse synthesized DAG");
        orig.verify().expect("verify synthesized DAG");

        // Pick a random sequence of 6 passes + verify-shapes
        let mut chosen = Vec::with_capacity(7);
        for _ in 0..6 {
            chosen.push(candidate_passes[rng.gen_range(candidate_passes.len())]);
        }
        chosen.push("verify-shapes");
        let pipeline = chosen.join(",");

        let mut transformed = orig.clone();
        let mut pm = PassManager::from_pipeline_str(&registry, &pipeline)
            .unwrap_or_else(|e| panic!("seed {seed} pipeline `{pipeline}` build failed: {e}"));
        pm.run(&mut transformed)
            .unwrap_or_else(|e| panic!("seed {seed} pipeline `{pipeline}` run failed: {e}"));
        transformed.verify().expect("verify transformed DAG");

        // Also verify parser/printer round-trip on the transformed module
        let printed = print_mlir(&transformed);
        let reparsed = parse_mlir(&printed).expect("reparse transformed DAG");
        reparsed.verify().expect("verify reparsed DAG");

        let rep = verify_module_numerics(&orig, &reparsed, 1e-3)
            .unwrap_or_else(|e| panic!("seed {seed} numerics eval failed: {e}"));
        assert!(
            rep.passed,
            "seed {seed} `{pipeline}` drift: max_abs={}\nOrig:\n{mlir}\nOpt:\n{printed}",
            rep.max_abs_diff
        );
    }
}

#[test]
fn test_pass_fixed_point_idempotency() {
    let registry = PassRegistry::with_builtin_passes();
    let idempotent_passes = [
        "dce",
        "cse",
        "fold-views",
        "simplify-algebraic(mode=constants)",
        "topological-sort",
        "topological-sort(schedule=min_memory)",
        "fuse-rmsnorm",
        "fuse-layernorm",
        "fuse-softmax",
        "fuse-swiglu",
        "fuse-matmul-epilogue",
        "decompose-composites",
        "tile-loops",
        "permute-loops",
        "unroll-loops",
        "vectorize-loops",
        "pad-for-tiling",
        "strip-debuginfo",
        "strip-attributes",
    ];

    let mut rng = Rng::new(2026);
    let mlir = synthesize_random_dag(&mut rng, 14);
    let base_mod = parse_mlir(&mlir).expect("parse base DAG");

    for pass_spec in idempotent_passes {
        let mut m = base_mod.clone();
        let mut pm = PassManager::from_pipeline_str(&registry, pass_spec)
            .unwrap_or_else(|e| panic!("build `{pass_spec}` failed: {e}"));

        // Run #1 brings `m` to the pass's target state
        let _ = pm.run(&mut m).expect("first pass run");
        m.verify().expect("verify after first run");

        // Run #2 on the already-transformed module must report `changed == false`
        let res2 = pm.run(&mut m).expect("second pass run");
        assert!(
            !res2[0].changed,
            "expected pass `{pass_spec}` to be idempotent on second run, but changed=true"
        );
    }
}
