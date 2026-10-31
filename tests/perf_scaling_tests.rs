//! Performance, compile-time scaling, and arena memory efficiency tests on
//! multi-layer Transformer graphs (4-layer and 12-layer LLaMA architectures).

use castforge::codegen::{
    CodegenTarget, HardwareRoofline, LoopPass, OptimizeLoopIrPass, TargetConfig,
    analyze_module_cost_with_roofline, build_memory_plan, generate_code_with_config,
    lower_to_loop_ir,
};
use castforge::ir::import_hf_config_json;
use castforge::passes::{PassManager, PassRegistry};
use std::time::Instant;

fn build_llama_config_json(
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
) -> String {
    format!(
        r#"{{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": {hidden_size},
  "intermediate_size": {intermediate_size},
  "num_hidden_layers": {num_layers},
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "vocab_size": 128,
  "rms_norm_eps": 1e-5
}}"#
    )
}

#[test]
fn test_multi_layer_transformer_compile_scaling_and_arena_compression() {
    let registry = PassRegistry::with_builtin_passes();

    // 1. Synthesize a 4-layer LLaMA graph and a 12-layer LLaMA graph
    let cfg_4l = build_llama_config_json(4, 64, 128);
    let cfg_12l = build_llama_config_json(12, 64, 128);

    let mut mod_4l = import_hf_config_json(&cfg_4l, None).expect("import 4-layer LLaMA");
    let mut mod_12l = import_hf_config_json(&cfg_12l, None).expect("import 12-layer LLaMA");

    let ops_before_4l = mod_4l.live_ops().len();
    let ops_before_12l = mod_12l.live_ops().len();
    assert!(ops_before_12l > ops_before_4l * 2);

    // 2. Run O3 optimization pipeline and verify sub-second compile time scaling
    let t0 = Instant::now();
    let mut pm4 = PassManager::from_pipeline_str(&registry, "O3").expect("O3 pipeline");
    pm4.run(&mut mod_4l).expect("run O3 on 4L");
    let dur_4l = t0.elapsed();

    let t1 = Instant::now();
    let mut pm12 = PassManager::from_pipeline_str(&registry, "O3").expect("O3 pipeline");
    pm12.run(&mut mod_12l).expect("run O3 on 12L");
    let dur_12l = t1.elapsed();

    mod_4l.verify().expect("verify 4L post-O3");
    mod_12l.verify().expect("verify 12L post-O3");

    // Macro-fusion in O3 must reduce operation count significantly
    let ops_after_12l = mod_12l.live_ops().len();
    assert!(
        ops_after_12l < ops_before_12l,
        "expected O3 fusion to reduce op count ({ops_before_12l} -> {ops_after_12l})"
    );
    assert!(
        dur_4l.as_secs_f64() < 5.0 && dur_12l.as_secs_f64() < 10.0,
        "O3 compile time exceeded budget: 4L={dur_4l:?}, 12L={dur_12l:?}"
    );

    // 3. Linear-scan arena memory planner must achieve substantial buffer reuse on 12-layer graph
    let plan_12l = build_memory_plan(&mod_12l);
    assert!(plan_12l.naive_activation_bytes > 0);
    assert!(
        plan_12l.total_activation_bytes < plan_12l.naive_activation_bytes,
        "expected arena reuse on 12-layer graph: arena={} vs naive={}",
        plan_12l.total_activation_bytes,
        plan_12l.naive_activation_bytes
    );
    let compression_ratio =
        plan_12l.naive_activation_bytes as f64 / plan_12l.total_activation_bytes.max(1) as f64;
    assert!(
        compression_ratio > 1.5,
        "expected >1.5x arena memory compression across 12 layers, got {compression_ratio:.2}x"
    );

    // 4. Roofline analysis & Loop-IR lowering + multi-target codegen scaling
    let roof = analyze_module_cost_with_roofline(&mod_12l, &HardwareRoofline::default());
    assert!(roof.total_flops > 0);

    let cfg = TargetConfig::default();
    let mut loop_mod = lower_to_loop_ir(&mod_12l, &cfg);
    let mut loop_opt = OptimizeLoopIrPass::default();
    let _ = loop_opt.run(&mut loop_mod);
    assert!(!loop_mod.kernels.is_empty());

    for target in [
        CodegenTarget::Rust,
        CodegenTarget::C,
        CodegenTarget::Cuda,
        CodegenTarget::Wgsl,
        CodegenTarget::Msl,
    ] {
        let code = generate_code_with_config(&mod_12l, target, &cfg).expect("codegen 12L");
        assert!(!code.is_empty());
    }
}
