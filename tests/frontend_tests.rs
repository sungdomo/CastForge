use castforge::codegen::{
    CodegenTarget, HardwareRoofline, TargetConfig, analyze_module_cost_with_roofline,
    evaluate_module, generate_code, verify_module_numerics,
};
use castforge::ir::{parse_mlir, print_mlir};
use castforge::passes::{PassManager, PassRegistry};

#[test]
fn test_frontend_importers_autodiff_and_native_jit() {
    use castforge::codegen::{autotune_module_jit, jit_execute_and_benchmark};
    use castforge::ir::{export_gguf_bytes, import_gguf_bytes, import_onnx_json};

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
fn test_symmetric_model_exports_loop_ir_gpu_backends_autotune_and_diff_translators() {
    use castforge::codegen::{
        AutotuneScheduleTranslator, CHeaderTranslator, CodegenTranslator, MemoryPlanTranslator,
        OptimizeLoopIrTranslator, RooflineReportTranslator, SharedLibTranslator,
        TensorToLoopTranslator,
    };
    use castforge::ir::{
        GgufExportTranslator, GgufImportTranslator, MlirParseTranslator, OnnxExportTranslator,
        OnnxImportTranslator, Payload, SafetensorsExportTranslator, TranslatorChain,
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

    // Also verify quantized GGUF (Q8_0) round-trip and ONNX composite/modern-op round-trip
    let reg = PassRegistry::with_builtin_passes();
    let q_pass = reg
        .create_pass("quantize-weights", &Default::default())
        .expect("quantize-weights");
    let mut q_gguf_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(q_pass)
        .then(GgufExportTranslator)
        .then(GgufImportTranslator);
    let q_gguf_mod = q_gguf_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("IR -> quantize -> GGUF Q8_0 -> IR")
        .into_ir()
        .expect("q_gguf module");
    assert_eq!(q_gguf_mod.tensor_store.len(), 2);

    let fuse_ep = reg
        .create_pass("fuse-matmul-epilogue", &Default::default())
        .expect("fuse-matmul-epilogue");
    let mut comp_onnx_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(fuse_ep)
        .then(OnnxExportTranslator);
    let comp_onnx_json = comp_onnx_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("IR -> fuse -> ONNX JSON")
        .into_text()
        .expect("onnx json");
    assert!(comp_onnx_json.contains("castforge.fused_matmul"));

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

    let mut opt_loop_chain = TranslatorChain::new()
        .then(MlirParseTranslator)
        .then(OptimizeLoopIrTranslator::default());
    let opt_loop_txt = opt_loop_chain
        .execute(Payload::Text(mlir.to_string()))
        .expect("optimize-loop-ir")
        .into_text()
        .expect("opt loop text");
    assert!(opt_loop_txt.contains("kernel @forward("));
    assert!(opt_loop_txt.contains("parallel"));
    assert!(opt_loop_txt.contains("vectorize"));

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

    // Verify Rust, C99, WGSL codegen and Roofline cost model on multi-function `func.call` module
    let out_rust = generate_code(&outlined, CodegenTarget::Rust).expect("rust outlined");
    assert!(out_rust.contains("outlined_kernel_0("));
    let out_c = generate_code(&outlined, CodegenTarget::C).expect("c outlined");
    assert!(out_c.contains("void outlined_kernel_0("));
    assert!(out_c.contains("outlined_kernel_0("));
    let out_wgsl = generate_code(&outlined, CodegenTarget::Wgsl).expect("wgsl outlined");
    assert!(out_wgsl.contains("// func.call @outlined_kernel_0"));
    let out_cost = analyze_module_cost_with_roofline(&outlined, &HardwareRoofline::default());
    assert!(out_cost.total_flops > 0);
    assert!(
        out_cost
            .ops
            .iter()
            .any(|o| o.op_name.contains("func.call(@outlined_kernel_0)"))
    );

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

    // 5. Module Linker (`link_modules`) & Function Composer (`compose-functions`)
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
    let comp_res = comp_pm.run(&mut linked).expect("run compose-functions");
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

#[test]
fn test_stablehlo_gather_scatter_dynamic_slice_clamp_and_while_eval() {
    let mlir = r#"
module @gather_scatter_mod {
  func.func @main(%idx: tensor<2xf32>, %upd: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %tbl = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0],
                                     [5.0, 6.0, 7.0, 8.0],
                                     [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
    %lo = stablehlo.constant dense<0.0> : tensor<2x4xf32>
    %hi = stablehlo.constant dense<20.0> : tensor<2x4xf32>
    %updated = "stablehlo.scatter"(%tbl, %idx, %upd)
      : (tensor<3x4xf32>, tensor<2xf32>, tensor<2x4xf32>) -> tensor<3x4xf32>
    %gathered = "stablehlo.gather"(%updated, %idx) {slice_sizes = array<i64: 1, 4>}
      : (tensor<3x4xf32>, tensor<2xf32>) -> tensor<2x4xf32>
    %clamped = "stablehlo.clamp"(%lo, %gathered, %hi)
      : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %clamped : tensor<2x4xf32>
  }
}
"#;
    let module = parse_mlir(mlir).expect("parse gather_scatter_mod");
    let idx_in = vec![2.0_f32, 0.0_f32];
    let upd_in = vec![15.0, 16.0, 17.0, 25.0, 1.5, 2.5, 3.5, 4.5];
    let outs = evaluate_module(&module, &[idx_in, upd_in]).expect("eval gather_scatter");
    assert_eq!(outs[0], vec![15.0, 16.0, 17.0, 20.0, 1.5, 2.5, 3.5, 4.5]);

    let rust_code = generate_code(&module, CodegenTarget::Rust).expect("rust gather/scatter");
    assert!(rust_code.contains(".clamp("));
    let c_code = generate_code(&module, CodegenTarget::C).expect("c gather/scatter");
    assert!(c_code.contains("roundf("));

    // Verify HuggingFace config.json Transformer graph synthesis + O2 optimization
    let hf_cfg = r#"{
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "seq_len": 2
    }"#;
    let mut hf_mod =
        castforge::ir::import_hf_config_json(hf_cfg, None).expect("import_hf_config_json");
    assert_eq!(hf_mod.tensor_store.len(), 7);
    let registry = PassRegistry::with_builtin_passes();
    let mut pm = PassManager::from_pipeline_str(&registry, "O2").expect("O2 on hf_mod");
    let res = pm.run(&mut hf_mod).expect("run O2 on hf_mod");
    assert!(res[0].changed);

    // Verify LoopModule lowering of fused composites + MemoryPlan arena bindings
    let fused_mlir = concat!(
        "module @fused_loop_mod {\n",
        "  func.func @forward(%x: tensor<2x4xf32>, %w: tensor<4xf32>, ",
        "%k: tensor<2x4xf32>, %v: tensor<2x4xf32>) -> tensor<2x4xf32> {\n",
        "    %norm = \"castforge.fused_rmsnorm\"(%x, %w) ",
        ": (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>\n",
        "    %attn = \"castforge.fused_attention\"(%norm, %k, %v) ",
        ": (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>\n",
        "    %mlp = \"castforge.fused_swiglu\"(%attn, %norm) ",
        ": (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>\n",
        "    func.return %mlp : tensor<2x4xf32>\n",
        "  }\n",
        "}\n"
    );
    let fused_mod = parse_mlir(fused_mlir).expect("parse fused_loop_mod");
    let mut loop_mod = castforge::codegen::lower_to_loop_ir(
        &fused_mod,
        &castforge::codegen::TargetConfig::default(),
    );
    let mut loop_opt = castforge::codegen::OptimizeLoopIrPass::default();
    let loop_changes = castforge::codegen::LoopPass::run(&mut loop_opt, &mut loop_mod);
    assert!(loop_changes > 0);
    let loop_txt = loop_mod.to_string();
    assert!(loop_txt.contains("MemoryPlan: arena_workspace_bytes="));
    assert!(loop_txt.contains("Arena bind"));
    assert!(loop_txt.contains("Fused scaled dot-product attention"));
    assert!(loop_txt.contains("Row-wise reduction + normalization"));

    // Verify extended reverse-mode autodiff (divide, log, sqrt, logistic, gather) + AdamW
    let ad_mlir = concat!(
        "module @ad_mod {\n",
        "  func.func @main(%x: tensor<2x4xf32>, %w: tensor<2x4xf32>) -> tensor<2x4xf32> {\n",
        "    %q = stablehlo.divide %x, %w : tensor<2x4xf32>\n",
        "    %s = stablehlo.logistic %q : tensor<2x4xf32>\n",
        "    %r = stablehlo.sqrt %s : tensor<2x4xf32>\n",
        "    %l = stablehlo.log %r : tensor<2x4xf32>\n",
        "    func.return %l : tensor<2x4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut ad_mod = parse_mlir(ad_mlir).expect("parse ad_mod");
    let mut ad_pm = PassManager::from_pipeline_str(
        &registry,
        "autodiff(lr=0.01,optimizer=adamw,weight_decay=0.02)",
    )
    .expect("adamw autodiff");
    let ad_res = ad_pm.run(&mut ad_mod).expect("run adamw autodiff");
    assert!(ad_res[0].changed);
    let ad_printed = print_mlir(&ad_mod);
    assert!(ad_printed.contains("adamw"));
    let ad_outs = evaluate_module(&ad_mod, &[vec![2.0; 8], vec![1.0; 8]]).expect("eval adamw");
    assert_eq!(ad_outs[0].len(), 8);

    // Verify WebGPU WGSL operator coverage & emit_from_loop_ir across all 7 backends
    let wgsl_fused = generate_code(&fused_mod, CodegenTarget::Wgsl).expect("wgsl fused_mod");
    assert!(wgsl_fused.contains("inverseSqrt(sq / f32("));
    let wgsl_gather = generate_code(&module, CodegenTarget::Wgsl).expect("wgsl gather_mod");
    assert!(wgsl_gather.contains("clamp("));
    let backend_reg = castforge::codegen::BackendRegistry::with_builtin_backends();
    for (tgt_name, _, _) in backend_reg.list() {
        let b = backend_reg.get(tgt_name).expect("get backend");
        let loop_src = b
            .emit_from_loop_ir(&loop_mod, &castforge::codegen::TargetConfig::default())
            .expect("emit_from_loop_ir");
        assert!(loop_src.contains("forward"));
    }

    // Verify 3D batched dot_general, 3D transpose, and N-D broadcast_in_dim evaluation + shapes
    let bmm_mlir = concat!(
        "module @bmm_3d {\n",
        "  func.func @forward(%q: tensor<2x2x3xf32>, %k: tensor<2x2x3xf32>, ",
        "%bias: tensor<2xf32>) -> tensor<2x2x2xf32> {\n",
        "    %kt = stablehlo.transpose %k, dims = [0, 2, 1] ",
        ": (tensor<2x2x3xf32>) -> tensor<2x3x2xf32>\n",
        "    %scores = \"stablehlo.dot_general\"(%q, %kt) ",
        "{lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], ",
        "lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]} ",
        ": (tensor<2x2x3xf32>, tensor<2x3x2xf32>) -> tensor<2x2x2xf32>\n",
        "    %b_exp = \"stablehlo.broadcast_in_dim\"(%bias) ",
        "{broadcast_dimensions = [2]} : (tensor<2xf32>) -> tensor<2x2x2xf32>\n",
        "    %out = stablehlo.add %scores, %b_exp : tensor<2x2x2xf32>\n",
        "    func.return %out : tensor<2x2x2xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut bmm_mod = parse_mlir(bmm_mlir).expect("parse bmm_3d");
    PassManager::from_pipeline_str(&registry, "verify-shapes")
        .unwrap()
        .run(&mut bmm_mod)
        .expect("verify-shapes on 3D bmm");
    let q_in = vec![
        1.0, 0.0, 2.0, 0.0, 1.0, 1.0, // batch 0: 2x3
        2.0, 1.0, 0.0, 1.0, 2.0, 1.0, // batch 1: 2x3
    ];
    let k_in = q_in.clone();
    let bias_in = vec![0.5, -0.5];
    let bmm_outs = evaluate_module(&bmm_mod, &[q_in, k_in, bias_in]).expect("eval 3D bmm");
    assert_eq!(bmm_outs[0].len(), 8);
    // Batch 0, row 0 dot row 0 = 1^2 + 0 + 2^2 = 5.0; + bias[0] (0.5) = 5.5
    assert!((bmm_outs[0][0] - 5.5).abs() < 1e-5);
    // Batch 0, row 0 dot row 1 = 0 + 0 + 2*1 = 2.0; + bias[1] (-0.5) = 1.5
    assert!((bmm_outs[0][1] - 1.5).abs() < 1e-5);
    // Batch 1, row 1 dot row 1 = 1^2 + 2^2 + 1^2 = 6.0; + bias[1] (-0.5) = 5.5
    assert!((bmm_outs[0][7] - 5.5).abs() < 1e-5);

    // Verify HardwareRoofline::for_arch presets and composite Pass::child_passes hierarchy
    let hopper_rf = HardwareRoofline::for_arch("nvidia-hopper");
    assert!(hopper_rf.peak_gflops >= 60_000.0);
    assert!(hopper_rf.ridge_point() > 10.0);
    let apple_rf = HardwareRoofline::for_arch("apple-m-series");
    assert!(apple_rf.peak_bandwidth_gbps >= 400.0);
    let o3_pass = registry.create_pass("O3", &Default::default()).expect("O3");
    assert!(o3_pass.is_composite());
    assert!(!o3_pass.child_passes().is_empty());
}
