//! Zero-dependency compiler & codegen throughput benchmark suite (`cargo bench`).

use castforge::codegen::{
    CodegenTarget, LoopPass, OptimizeLoopIrPass, TargetConfig, build_memory_plan,
    generate_code_with_config, lower_to_loop_ir,
};
use castforge::ir::{import_hf_config_json, parse_mlir, print_mlir};
use castforge::passes::{PassManager, PassRegistry};
use std::time::Instant;

fn bench_op<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    // Warmup
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1e6 / (iters as f64);
    println!("{name:<48} | {iters:>4} iters | {per_iter_us:>9.2} us/iter");
}

fn main() {
    println!("============================================================================");
    println!("CastForge Compiler & Codegen Benchmark Suite");
    println!("============================================================================");

    let llama_8l_cfg = r#"{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_hidden_layers": 8,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "vocab_size": 128,
  "rms_norm_eps": 1e-5
}"#;

    let base_mod = import_hf_config_json(llama_8l_cfg, None).expect("import 8L LLaMA");
    let mlir_text = print_mlir(&base_mod);
    let registry = PassRegistry::with_builtin_passes();

    bench_op("1. MLIR Parser + Printer Round-Trip (8L LLaMA)", 20, || {
        let m = parse_mlir(&mlir_text).unwrap();
        let s = print_mlir(&m);
        std::hint::black_box(s.len());
    });

    bench_op("2. Full O3 Optimization Pipeline (8L LLaMA)", 15, || {
        let mut m = base_mod.clone();
        let mut pm = PassManager::from_pipeline_str(&registry, "O3").unwrap();
        pm.run(&mut m).unwrap();
        std::hint::black_box(m.live_ops().len());
    });

    let mut opt_mod = base_mod.clone();
    PassManager::from_pipeline_str(&registry, "O3")
        .unwrap()
        .run(&mut opt_mod)
        .unwrap();
    let cfg = TargetConfig::default();

    bench_op("3. Linear-Scan Arena Memory Planner (8L LLaMA)", 50, || {
        let plan = build_memory_plan(&opt_mod);
        std::hint::black_box(plan.total_activation_bytes);
    });

    bench_op(
        "4. Loop-IR Lowering + Polyhedral Opt (8L LLaMA)",
        30,
        || {
            let mut lm = lower_to_loop_ir(&opt_mod, &cfg);
            let mut opt = OptimizeLoopIrPass::default();
            let ch = opt.run(&mut lm);
            std::hint::black_box(ch);
        },
    );

    bench_op("5. Multi-Target Codegen (Rust/C/CUDA/WGSL/MSL)", 20, || {
        for target in [
            CodegenTarget::Rust,
            CodegenTarget::C,
            CodegenTarget::Cuda,
            CodegenTarget::Wgsl,
            CodegenTarget::Msl,
        ] {
            let code = generate_code_with_config(&opt_mod, target, &cfg).unwrap();
            std::hint::black_box(code.len());
        }
    });

    println!("============================================================================");
}
