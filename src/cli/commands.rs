//! Command execution handlers (`run`) for all `castforge` CLI subcommands.

use crate::cli::args::{Cli, Commands};
use crate::cli::chain::{
    build_translator_chain_with_weights, parse_codegen_target_strict, parse_loop_order_strict,
    parse_simd_isa_strict, parse_viz_format_strict, validate_arch_profile,
};
use crate::cli::inspect::{run_inspect, run_list_targets};
use crate::codegen::{
    CodegenTarget, GemmLoopOrder, HardwareRoofline, OptimizeLoopIrPass, TargetConfig,
    analyze_module_cost_with_roofline, apply_schedule_to_module, autotune_module,
    autotune_module_jit, build_memory_plan, compare_output_tensors, compile_shared_lib,
    emit_c_header, evaluate_function, evaluate_module, generate_code_with_config,
    generate_synthetic_inputs, jit_execute_and_benchmark, lower_to_loop_ir,
};
use crate::ir::viz::{
    TimelineFrame, VizFormat, build_graph_json, diff_modules, render_module, render_timeline_html,
};
use crate::ir::{
    Module, OpId, PassResult, Payload, PrintOptions, collect_module_tensors, export_gguf_file,
    export_onnx_file, import_gguf_file, import_hf_model_dir, import_onnx_file,
    import_safetensors_file, parse_mlir, print_mlir, print_mlir_with_options,
    verify_stablehlo_shapes,
};
use crate::passes::{ComposeFunctionsPass, Pass, PassManager, PassRegistry, link_modules};
use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;

/// Parses CLI arguments and executes the requested `castforge` subcommand.
pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Opt {
            input,
            passes,
            config,
            output,
            weights,
            save_weights,
            generic,
            locations,
            verify_shapes,
            stats,
            quiet,
            verify_numerics,
            bisect,
            tolerance,
        } => {
            let show_stats = stats && !quiet;
            let registry = PassRegistry::with_builtin_passes();
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), show_stats)?;

            let need_verify = verify_numerics || bisect;
            let baseline_eval = if need_verify {
                let inputs = generate_synthetic_inputs(&module, None)?;
                let out_before = evaluate_module(&module, &inputs)?;
                Some((inputs, out_before))
            } else {
                None
            };

            let mut pm = if let Some(cfg_path) = config {
                let toml_str = fs::read_to_string(&cfg_path)
                    .with_context(|| format!("Failed to read config {}", cfg_path.display()))?;
                PassManager::from_toml_str(&registry, &toml_str)?
            } else if let Some(spec) = passes {
                PassManager::from_pipeline_str(&registry, &spec)?
            } else {
                PassManager::from_pipeline_str(&registry, "canonicalize")?
            };

            if bisect {
                let (inputs, out_before) = baseline_eval.as_ref().expect("baseline initialized");
                run_pipeline_with_bisect(
                    &mut pm,
                    &mut module,
                    inputs,
                    out_before,
                    tolerance,
                    show_stats,
                )?;
            } else {
                let results = pm.run(&mut module)?;
                if show_stats {
                    log_pass_results(&results);
                }
                if let Some((inputs, out_before)) = &baseline_eval {
                    let out_after = evaluate_module(&module, inputs)?;
                    let rep = compare_output_tensors(out_before, &out_after, tolerance)?;
                    eprintln!(
                        "[castforge:verify] passed: {} | max_abs_diff: {:.6e} (tol: {:.2e}, \
                         checked {} elements)",
                        rep.passed, rep.max_abs_diff, rep.tolerance, rep.num_outputs_checked
                    );
                    if !rep.passed {
                        anyhow::bail!(
                            "Numerical verification failed: max_abs_diff {:.6e} > tol {:.2e}",
                            rep.max_abs_diff,
                            rep.tolerance
                        );
                    }
                }
            }

            if verify_shapes {
                module.verify()?;
                verify_all_stablehlo_shapes(&module)?;
            }

            if let Some(sw_path) = save_weights {
                module
                    .tensor_store
                    .save_safetensors_file(&sw_path)
                    .with_context(|| format!("Failed to save weights {}", sw_path.display()))?;
                if show_stats {
                    eprintln!(
                        "[castforge:weights] Saved {} tensors to {}",
                        module.tensor_store.len(),
                        sw_path.display()
                    );
                }
            }

            let formatted = print_mlir_with_options(
                &module,
                PrintOptions {
                    generic_form: generic,
                    print_locations: locations,
                },
            );
            write_text_output(output.as_ref(), &formatted)?;
        }

        Commands::Run {
            input,
            passes,
            weights,
            func,
            inputs: raw_inputs,
            bisect,
            tolerance,
            json,
            output,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;
            let fn_id = resolve_target_func(&module, func.as_deref())?;
            let inputs = match raw_inputs {
                Some(spec) => parse_cli_inputs(&spec)?,
                None => generate_synthetic_inputs(&module, Some(fn_id))?,
            };
            let baseline = evaluate_function(&module, fn_id, &inputs)?;
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                if bisect {
                    run_pipeline_with_bisect_func(
                        &mut pm,
                        &mut module,
                        func.as_deref(),
                        &inputs,
                        &baseline,
                        tolerance,
                        true,
                    )?;
                } else {
                    pm.run(&mut module)?;
                }
                let opt_fn_id = resolve_target_func(&module, func.as_deref())?;
                let optimized_out = evaluate_function(&module, opt_fn_id, &inputs)?;
                let rep = compare_output_tensors(&baseline, &optimized_out, tolerance)?;
                let text = if json {
                    let payload = serde_json::json!({
                        "passed": rep.passed,
                        "max_abs_diff": rep.max_abs_diff,
                        "tolerance": rep.tolerance,
                        "outputs": optimized_out,
                    });
                    format!("{}\n", serde_json::to_string_pretty(&payload)?)
                } else {
                    format!(
                        "Numerical verification: {} (max_abs_diff = {:.6e}, tol = {:.2e})\n\
                         Outputs: {optimized_out:?}\n",
                        if rep.passed { "PASSED" } else { "FAILED" },
                        rep.max_abs_diff,
                        rep.tolerance
                    )
                };
                write_text_output(output.as_ref(), &text)?;
                if !rep.passed {
                    anyhow::bail!(
                        "Numerical verification failed: max_abs_diff {:.6e} > tol {:.2e}",
                        rep.max_abs_diff,
                        rep.tolerance
                    );
                }
            } else {
                let text = if json {
                    let payload = serde_json::json!({ "outputs": baseline });
                    format!("{}\n", serde_json::to_string_pretty(&payload)?)
                } else {
                    format!("Outputs: {baseline:?}\n")
                };
                write_text_output(output.as_ref(), &text)?;
            }
        }

        Commands::Parse {
            input,
            output,
            generic,
            locations,
            verify_shapes,
        } => {
            let src = read_input_text(input.as_ref())?;
            let module = parse_mlir(&src)?;
            module.verify()?;
            if verify_shapes {
                verify_all_stablehlo_shapes(&module)?;
            }
            let formatted = print_mlir_with_options(
                &module,
                PrintOptions {
                    generic_form: generic,
                    print_locations: locations,
                },
            );
            write_text_output(output.as_ref(), &formatted)?;
        }

        Commands::Viz {
            input,
            format,
            passes,
            weights,
            output,
        } => {
            let fmt = parse_viz_format_strict(&format)?;
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;

            let rendered = if fmt == VizFormat::Html {
                let trace_steps = if let Some(spec) = passes {
                    let registry = PassRegistry::with_builtin_passes();
                    let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                    pm.run_with_trace(&mut module)?
                } else {
                    let mut pm = PassManager::new();
                    pm.run_with_trace(&mut module)?
                };

                let frames: Vec<TimelineFrame> = trace_steps
                    .into_iter()
                    .map(|step| {
                        let (changed, rem, modif, creat) = match &step.result {
                            Some(r) => (r.changed, r.ops_removed, r.ops_modified, r.ops_created),
                            None => (false, 0, 0, 0),
                        };
                        TimelineFrame {
                            step_index: step.step_index,
                            pass_name: step.label,
                            changed,
                            ops_removed: rem,
                            ops_modified: modif,
                            ops_created: creat,
                            mlir_text: print_mlir(&step.module),
                            graph: build_graph_json(&step.module),
                            memory_plan: build_memory_plan(&step.module),
                        }
                    })
                    .collect();
                render_timeline_html(&frames)
            } else {
                if let Some(spec) = passes {
                    let registry = PassRegistry::with_builtin_passes();
                    let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                    pm.run(&mut module)?;
                }
                render_module(&module, fmt)
            };

            write_text_output(output.as_ref(), &rendered)?;
            if let Some(out_path) = &output {
                eprintln!(
                    "[castforge:viz] Wrote {} visualization to {}",
                    format,
                    out_path.display()
                );
            }
        }

        Commands::Codegen {
            input,
            target,
            passes,
            weights,
            output,
            emit_header,
            autotune,
            tile_size,
            workgroup_size,
            loop_order,
            unroll,
            vectorize,
            reg_tile,
            collapse_loops,
            parallel,
            l2_tile_size,
            prefetch,
            split_k,
            loop_skew,
            double_buffer,
            simd_isa,
            arch,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;

            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }

            let mut cfg = match &arch {
                Some(arch_name) => {
                    validate_arch_profile(arch_name)?;
                    TargetConfig::for_arch(arch_name)
                }
                None => TargetConfig::default(),
            };
            if autotune {
                let roofline = build_roofline(arch.as_deref(), None, None)?;
                let report = autotune_module(&module, &roofline);
                apply_schedule_to_module(&mut module, &report.best.config);
                let simd_isa_base = cfg.simd_isa;
                let simd_w_base = cfg.simd_width;
                cfg = report.best.config;
                if arch.is_some() {
                    cfg.simd_isa = simd_isa_base;
                    cfg.simd_width = simd_w_base;
                }
            }
            if let Some(ts) = tile_size {
                cfg.tile_m = ts;
                cfg.tile_n = ts;
                cfg.tile_k = ts;
            }
            if let Some(wg) = workgroup_size {
                cfg.wgsl_workgroup_size = wg.max(1);
            }
            if let Some(order) = loop_order {
                cfg.loop_order = parse_loop_order_strict(&order)?;
            }
            if let Some(uf) = unroll {
                cfg.unroll_factor = uf.max(1);
            }
            if let Some(vec_flag) = vectorize {
                cfg.vectorize = vec_flag;
            }
            if let Some(rt) = reg_tile {
                cfg.reg_tile_m = rt;
                cfg.reg_tile_n = rt;
            }
            if let Some(col_flag) = collapse_loops {
                cfg.collapse_loops = col_flag;
            }
            if let Some(par_flag) = parallel {
                cfg.parallel = par_flag;
            }
            if let Some(l2) = l2_tile_size {
                cfg.l2_tile_size = l2;
            }
            if let Some(pf) = prefetch {
                cfg.prefetch_distance = pf;
            }
            if let Some(sk) = split_k {
                cfg.split_k = sk.max(1);
            }
            if let Some(skew_flag) = loop_skew {
                cfg.loop_skew = skew_flag;
            }
            if let Some(db_flag) = double_buffer {
                cfg.double_buffer = db_flag;
            }
            if let Some(isa_str) = simd_isa {
                let parsed = parse_simd_isa_strict(&isa_str)?;
                cfg.simd_isa = parsed;
                cfg.simd_width = parsed.default_lanes().max(1);
            }

            if let Some(hdr_path) = emit_header {
                let hdr = emit_c_header(&module)?;
                fs::write(&hdr_path, hdr)
                    .with_context(|| format!("Failed to write header {}", hdr_path.display()))?;
            }

            let plan = build_memory_plan(&module);
            let target_lower = target.trim().to_ascii_lowercase();
            match target_lower.as_str() {
                "header" | "c-header" => {
                    let hdr = emit_c_header(&module)?;
                    write_text_output(output.as_ref(), &hdr)?;
                }
                "loop-ir" | "loops" => {
                    let mut loop_mod = lower_to_loop_ir(&module, &cfg);
                    let mut loop_pass = OptimizeLoopIrPass {
                        fuse_loops: true,
                        vectorize: cfg.vectorize,
                        unroll_factor: cfg.unroll_factor.max(1),
                        parallel: cfg.parallel,
                        interchange: cfg.loop_order == GemmLoopOrder::Mkn,
                    };
                    use crate::codegen::LoopPass;
                    let _ = loop_pass.run(&mut loop_mod);
                    write_text_output(output.as_ref(), &loop_mod.to_string())?;
                }
                "lib" | "shared-lib" | "so" | "dylib" => {
                    let bytes = compile_shared_lib(&module, &cfg)?;
                    write_binary_output(output.as_ref(), &bytes)?;
                }
                _ => {
                    let tgt = parse_codegen_target_strict(&target)?;
                    eprintln!(
                        "[castforge:codegen] Target: {:?} | Constant Weights: {} B | \
                         Activation Workspace: {} B (Naive: {} B, Saved: {} B across {} slots)",
                        tgt,
                        plan.total_constant_bytes,
                        plan.total_activation_bytes,
                        plan.naive_activation_bytes,
                        plan.saved_activation_bytes(),
                        plan.reusable_slot_count
                    );
                    let code = generate_code_with_config(&module, tgt, &cfg)?;
                    write_text_output(output.as_ref(), &code)?;
                }
            }
        }

        Commands::Bench {
            input,
            target,
            passes,
            weights,
            arch,
            autotune,
            warmup,
            iters,
            json,
            output,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            let tgt = match parse_codegen_target_strict(&target)? {
                CodegenTarget::Rust => CodegenTarget::Rust,
                CodegenTarget::C => CodegenTarget::C,
                other => anyhow::bail!(
                    "Native CPU JIT benchmarking supports `rust` or `c` targets (got `{other:?}`)"
                ),
            };
            let mut cfg = match &arch {
                Some(a) => {
                    validate_arch_profile(a)?;
                    TargetConfig::for_arch(a)
                }
                None => TargetConfig::default(),
            };
            if autotune {
                let rf = build_roofline(arch.as_deref(), None, None)?;
                let rep = autotune_module(&module, &rf);
                apply_schedule_to_module(&mut module, &rep.best.config);
                let simd_isa_base = cfg.simd_isa;
                let simd_w_base = cfg.simd_width;
                cfg = rep.best.config;
                if arch.is_some() {
                    cfg.simd_isa = simd_isa_base;
                    cfg.simd_width = simd_w_base;
                }
            }
            let bench = jit_execute_and_benchmark(&module, tgt, &cfg, warmup, iters)?;
            let text = if json {
                format!("{}\n", serde_json::to_string_pretty(&bench)?)
            } else {
                format!(
                    "=== CastForge Native JIT Benchmark ({:?}) ===\n\
                     Iterations: {} (Warmup: {})\n\
                     Min Latency: {:.2} ns | Avg Latency: {:.2} ns\n\
                     Numerics Verified: {} (max_abs_diff = {:.6e})\n",
                    bench.target,
                    bench.iterations,
                    warmup,
                    bench.min_latency_ns,
                    bench.avg_latency_ns,
                    if bench.numerics_verified {
                        "PASSED"
                    } else {
                        "FAILED"
                    },
                    bench.max_abs_diff
                )
            };
            write_text_output(output.as_ref(), &text)?;
            if !bench.numerics_verified {
                anyhow::bail!(
                    "JIT benchmark numerical check failed: max_abs_diff {:.6e}",
                    bench.max_abs_diff
                );
            }
        }

        Commands::Profile {
            input,
            passes,
            weights,
            peak_gflops,
            peak_bw_gbps,
            arch,
            json,
            output,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;
            let roofline = build_roofline(arch.as_deref(), peak_gflops, peak_bw_gbps)?;
            let before = analyze_module_cost_with_roofline(&module, &roofline);
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
                let after = analyze_module_cost_with_roofline(&module, &roofline);
                if json {
                    let payload = serde_json::json!({
                        "baseline": before,
                        "optimized": after,
                    });
                    let formatted = serde_json::to_string_pretty(&payload)?;
                    write_text_output(output.as_ref(), &format!("{formatted}\n"))?;
                } else {
                    use std::fmt::Write as _;
                    let mut report = String::new();
                    let _ = writeln!(report, "=== Baseline Cost Profile ===");
                    report.push_str(&before.summary_table());
                    let _ = writeln!(report, "\n=== Optimized Cost Profile ({spec}) ===");
                    report.push_str(&after.summary_table());
                    let traffic_saved = before
                        .total_memory_traffic_bytes
                        .saturating_sub(after.total_memory_traffic_bytes);
                    let _ = writeln!(
                        report,
                        "\nMemory Traffic Saved: {} B ({:.1}% reduction) | \
                         Est. Speedup: {:.2}x",
                        traffic_saved,
                        if before.total_memory_traffic_bytes > 0 {
                            100.0 * traffic_saved as f64 / before.total_memory_traffic_bytes as f64
                        } else {
                            0.0
                        },
                        if after.estimated_latency_ns > 0.0 {
                            before.estimated_latency_ns / after.estimated_latency_ns
                        } else {
                            1.0
                        }
                    );
                    write_text_output(output.as_ref(), &report)?;
                }
            } else if json {
                let formatted = serde_json::to_string_pretty(&before)?;
                write_text_output(output.as_ref(), &format!("{formatted}\n"))?;
            } else {
                write_text_output(output.as_ref(), &before.summary_table())?;
            }
        }

        Commands::Autotune {
            input,
            passes,
            weights,
            peak_gflops,
            peak_bw_gbps,
            top_k,
            bench_jit,
            arch,
            json,
            output,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            let roofline = build_roofline(arch.as_deref(), peak_gflops, peak_bw_gbps)?;
            let report = if bench_jit {
                autotune_module_jit(&module, &roofline, top_k)?
            } else {
                autotune_module(&module, &roofline)
            };
            if json {
                let formatted = serde_json::to_string_pretty(&report)?;
                write_text_output(output.as_ref(), &format!("{formatted}\n"))?;
            } else if let Some(out_path) = output {
                eprint!("{}", report.summary_table(top_k));
                apply_schedule_to_module(&mut module, &report.best.config);
                let mlir_out = print_mlir(&module);
                fs::write(&out_path, mlir_out)
                    .with_context(|| format!("Failed to write {}", out_path.display()))?;
            } else {
                print!("{}", report.summary_table(top_k));
            }
        }

        Commands::Diff {
            input,
            second,
            passes,
            weights,
            output,
        } => {
            let src_a = read_input_text(input.as_ref())?;
            let mut mod_a = parse_mlir(&src_a)?;
            load_optional_weights(&mut mod_a, weights.as_ref(), false)?;
            let mod_b = if let Some(sec_path) = second {
                let src_b = read_input_text(Some(&sec_path))?;
                let mut parsed_b = parse_mlir(&src_b)?;
                load_optional_weights(&mut parsed_b, weights.as_ref(), false)?;
                if let Some(spec) = passes {
                    let registry = PassRegistry::with_builtin_passes();
                    let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                    pm.run(&mut parsed_b)?;
                }
                parsed_b
            } else {
                let mut cloned = mod_a.clone();
                let spec = passes.as_deref().unwrap_or("canonicalize");
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, spec)?;
                pm.run(&mut cloned)?;
                cloned
            };
            let diff_txt = diff_modules(&mod_a, &mod_b);
            write_text_output(output.as_ref(), &diff_txt)?;
        }

        Commands::Inspect {
            input,
            weights,
            passes,
            json,
            output,
        } => {
            run_inspect(
                input.as_ref(),
                weights.as_ref(),
                passes.as_deref(),
                json,
                output.as_ref(),
            )?;
        }

        Commands::Import {
            input,
            format,
            passes,
            output,
            save_weights,
        } => {
            let fmt = match format {
                Some(f) => f,
                None => {
                    let s = input.to_string_lossy().to_ascii_lowercase();
                    if s.ends_with(".gguf") {
                        "gguf".to_string()
                    } else if s.ends_with(".safetensors") {
                        "safetensors".to_string()
                    } else if input.is_dir() || s.ends_with("config.json") {
                        "hf".to_string()
                    } else {
                        "onnx".to_string()
                    }
                }
            };
            let mut module = match fmt.trim().to_ascii_lowercase().as_str() {
                "gguf" => import_gguf_file(&input)?,
                "safetensors" | "st" => import_safetensors_file(&input)?,
                "hf" | "huggingface" | "config.json" => import_hf_model_dir(&input)?,
                "onnx" | "onnx.json" | "json" => import_onnx_file(&input)?,
                other => {
                    anyhow::bail!(
                        "Unknown import format `{other}` \
                         (expected: gguf, safetensors, onnx, or hf)"
                    )
                }
            };
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            if let Some(sw_path) = save_weights {
                module.tensor_store.save_safetensors_file(&sw_path)?;
            }
            let formatted = print_mlir(&module);
            write_text_output(output.as_ref(), &formatted)?;
        }

        Commands::Export {
            input,
            output,
            format,
            passes,
            weights,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            load_optional_weights(&mut module, weights.as_ref(), false)?;
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            let fmt = match format {
                Some(f) => f,
                None => {
                    let s = output.to_string_lossy().to_ascii_lowercase();
                    if s.ends_with(".gguf") {
                        "gguf".to_string()
                    } else if s.ends_with(".safetensors") {
                        "safetensors".to_string()
                    } else {
                        "onnx".to_string()
                    }
                }
            };
            match fmt.trim().to_ascii_lowercase().as_str() {
                "gguf" => export_gguf_file(&module, &output)?,
                "safetensors" | "st" => {
                    let store = collect_module_tensors(&module);
                    store.save_safetensors_file(&output)?;
                }
                "onnx" | "onnx.json" | "json" => export_onnx_file(&module, &output)?,
                other => anyhow::bail!(
                    "Unknown export format `{other}` (expected: onnx, gguf, or safetensors)"
                ),
            }
        }

        Commands::Chain {
            input,
            stages,
            weights,
            output,
        } => {
            let registry = PassRegistry::with_builtin_passes();
            let (mut chain, first_in, last_out) =
                build_translator_chain_with_weights(&registry, &stages, weights)?;
            for (idx, desc) in chain.describe_stages().iter().enumerate() {
                eprintln!("[castforge:chain] stage #{}: {desc}", idx + 1);
            }
            let initial = if first_in == "bytes" {
                let bytes = match input {
                    Some(p) if p.as_os_str() != "-" => {
                        fs::read(&p).with_context(|| format!("Failed to read {}", p.display()))?
                    }
                    _ => {
                        let mut b = Vec::new();
                        std::io::stdin().read_to_end(&mut b)?;
                        b
                    }
                };
                Payload::Bytes(bytes)
            } else {
                Payload::Text(read_input_text(input.as_ref())?)
            };
            let result = chain.execute(initial)?;
            match result {
                Payload::Bytes(b) => {
                    write_binary_output(output.as_ref(), &b)?;
                }
                Payload::Text(s) => {
                    write_text_output(output.as_ref(), &s)?;
                }
                Payload::Ir(m) => {
                    let _ = last_out;
                    let text = print_mlir(&m);
                    write_text_output(output.as_ref(), &text)?;
                }
            }
        }

        Commands::Link {
            inputs,
            compose,
            passes,
            weights,
            output,
        } => {
            let first_src = read_input_text(inputs.first())?;
            let mut base = parse_mlir(&first_src)?;
            for path in inputs.iter().skip(1) {
                let src = read_input_text(Some(path))?;
                let other = parse_mlir(&src)?;
                link_modules(&mut base, &other)?;
            }
            load_optional_weights(&mut base, weights.as_ref(), false)?;
            if let Some(pair) = compose {
                let Some((f1, rest)) = pair.split_once(',') else {
                    anyhow::bail!(
                        "Invalid --compose `{pair}`: expected `first_fn,second_fn` \
                         or `first_fn,second_fn:entry_name`"
                    );
                };
                let (f2, entry_name) = match rest.split_once(':') {
                    Some((second, entry)) => (second.trim(), entry.trim()),
                    None => (rest.trim(), "composed_main"),
                };
                if f1.trim().is_empty() || f2.is_empty() || entry_name.is_empty() {
                    anyhow::bail!("Invalid --compose `{pair}`: function names must be non-empty");
                }
                let mut comp = ComposeFunctionsPass {
                    first: f1.trim().to_string(),
                    second: f2.to_string(),
                    entry_name: entry_name.to_string(),
                };
                comp.run(&mut base)?;
            }
            if let Some(spec) = passes {
                let registry = PassRegistry::with_builtin_passes();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut base)?;
            }
            let formatted = print_mlir(&base);
            write_text_output(output.as_ref(), &formatted)?;
        }

        Commands::ListPasses {
            filter,
            tree,
            options,
            json,
            output,
        } => {
            use std::fmt::Write as _;
            let registry = PassRegistry::with_builtin_passes();
            let q = filter.as_deref().unwrap_or("").to_ascii_lowercase();
            let entries: Vec<(&str, &str)> = registry
                .list()
                .filter(|(n, d)| {
                    if q.is_empty() {
                        return true;
                    }
                    let in_meta = registry.catalog_entry(n).is_some_and(|cat| {
                        cat.family.to_ascii_lowercase().contains(&q)
                            || cat
                                .options
                                .iter()
                                .any(|o| o.name.to_ascii_lowercase().contains(&q))
                    });
                    n.to_ascii_lowercase().contains(&q)
                        || d.to_ascii_lowercase().contains(&q)
                        || in_meta
                })
                .collect();

            let text = if json {
                let items: Vec<serde_json::Value> = entries
                    .iter()
                    .map(|&(name, desc)| {
                        let cat = registry.catalog_entry(name);
                        let children: Vec<String> =
                            cat.map(|c| c.child_passes.clone()).unwrap_or_else(|| {
                                registry
                                    .create_pass(name, &Default::default())
                                    .ok()
                                    .map(|p| {
                                        p.child_passes().iter().map(|s| (*s).to_string()).collect()
                                    })
                                    .unwrap_or_default()
                            });
                        let family = cat.map(|c| c.family.as_str()).unwrap_or("custom");
                        let pass_opts = cat.map(|c| c.options.as_slice()).unwrap_or(&[]);
                        serde_json::json!({
                            "name": name,
                            "family": family,
                            "description": desc,
                            "is_composite": !children.is_empty(),
                            "child_passes": children,
                            "options": pass_opts,
                        })
                    })
                    .collect();
                format!("{}\n", serde_json::to_string_pretty(&items)?)
            } else {
                let mut out = String::new();
                let _ = writeln!(
                    out,
                    "Available CastForge Passes ({} matched):",
                    entries.len()
                );
                let _ = writeln!(out, "{:<28} {:<18} DESCRIPTION", "PASS NAME", "FAMILY");
                let _ = writeln!(out, "{}", "-".repeat(96));
                for (name, desc) in entries {
                    let cat = registry.catalog_entry(name);
                    let family = cat.map(|c| c.family.as_str()).unwrap_or("custom");
                    let _ = writeln!(out, "{:<28} {:<18} {}", name, family, desc);
                    if tree {
                        let children: Vec<String> =
                            cat.map(|c| c.child_passes.clone()).unwrap_or_else(|| {
                                registry
                                    .create_pass(name, &Default::default())
                                    .ok()
                                    .map(|p| {
                                        p.child_passes().iter().map(|s| (*s).to_string()).collect()
                                    })
                                    .unwrap_or_default()
                            });
                        for (idx, child) in children.iter().enumerate() {
                            let branch = if idx + 1 == children.len() {
                                "└─"
                            } else {
                                "├─"
                            };
                            let _ = writeln!(out, "  {branch} {child}");
                        }
                    }
                    if options
                        && let Some(cat_entry) = cat
                        && !cat_entry.options.is_empty()
                    {
                        for opt in &cat_entry.options {
                            let def = if opt.default.is_empty() {
                                "\"\"".to_string()
                            } else {
                                opt.default.clone()
                            };
                            let _ = writeln!(
                                out,
                                "      arg: {}=<val> (default: {}) — {}",
                                opt.name, def, opt.desc
                            );
                        }
                    }
                }
                out
            };
            write_text_output(output.as_ref(), &text)?;
        }

        Commands::ListTargets {
            filter,
            json,
            output,
        } => {
            run_list_targets(filter.as_deref(), json, output.as_ref())?;
        }

        Commands::DirectPass(args) => {
            let registry = PassRegistry::with_builtin_passes();
            run_direct_pass(&registry, &args)?;
        }
    }

    Ok(())
}

fn run_pipeline_with_bisect_func(
    pm: &mut PassManager,
    module: &mut Module,
    func_name: Option<&str>,
    inputs: &[Vec<f32>],
    out_before: &[Vec<f32>],
    tolerance: f32,
    show_stats: bool,
) -> Result<()> {
    let trace_steps = pm.run_with_trace(module)?;
    let mut first_failing: Option<(usize, String, f32)> = None;
    for step in &trace_steps {
        if let Some(r) = &step.result
            && show_stats
        {
            eprintln!(
                "[castforge:pass] {:<18} | changed: {:<5} | \
                 removed: {:<3} | modified: {:<3} | created: {:<3}",
                r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
            );
        }
        if step.step_index == 0 {
            continue;
        }
        let step_fn_id = resolve_target_func(&step.module, func_name)?;
        let out_step = evaluate_function(&step.module, step_fn_id, inputs)?;
        let rep = compare_output_tensors(out_before, &out_step, tolerance)?;
        eprintln!(
            "[castforge:bisect] step #{:<2} {:<22} | passed: {:<5} | max_abs_diff: {:.6e}",
            step.step_index, step.label, rep.passed, rep.max_abs_diff
        );
        if !rep.passed && first_failing.is_none() {
            first_failing = Some((step.step_index, step.label.clone(), rep.max_abs_diff));
        }
    }
    if let Some((idx, name, diff)) = first_failing {
        anyhow::bail!(
            "Pass bisection identified numerical failure at step #{idx} (`{name}`): \
             max_abs_diff {diff:.6e} > tol {tolerance:.2e}"
        );
    }
    Ok(())
}

/// Executes a pass invoked directly as a top-level subcommand
/// (`castforge <pass-spec> [key=val ...] [input] [-o <output>] [-w <weights>]`).
fn run_direct_pass(registry: &PassRegistry, args: &[String]) -> Result<()> {
    let Some(pass_name) = args.first() else {
        anyhow::bail!("Missing pass name for direct pass invocation");
    };
    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut weights_path: Option<PathBuf> = None;
    let mut save_weights_path: Option<PathBuf> = None;
    let mut extra_opts: Vec<String> = Vec::new();
    let mut generic = false;
    let mut locations = false;
    let mut quiet = false;
    let mut verify_shapes = false;
    let mut verify_numerics = false;
    let mut bisect = false;
    let mut tolerance = 1e-4_f32;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-o" | "--output" => {
                idx += 1;
                let val = args
                    .get(idx)
                    .ok_or_else(|| anyhow::anyhow!("Missing file path after `-o`/`--output`"))?;
                output_path = Some(PathBuf::from(val));
            }
            "-w" | "--weights" => {
                idx += 1;
                let val = args
                    .get(idx)
                    .ok_or_else(|| anyhow::anyhow!("Missing file path after `-w`/`--weights`"))?;
                weights_path = Some(PathBuf::from(val));
            }
            "--save-weights" => {
                idx += 1;
                let val = args
                    .get(idx)
                    .ok_or_else(|| anyhow::anyhow!("Missing file path after `--save-weights`"))?;
                save_weights_path = Some(PathBuf::from(val));
            }
            "--tolerance" => {
                idx += 1;
                let val = args
                    .get(idx)
                    .ok_or_else(|| anyhow::anyhow!("Missing float value after `--tolerance`"))?;
                tolerance = val
                    .parse()
                    .with_context(|| format!("Invalid tolerance `{val}`"))?;
            }
            "--generic" => {
                generic = true;
            }
            "--locations" => {
                locations = true;
            }
            "--verify-shapes" => {
                verify_shapes = true;
            }
            "-q" | "--quiet" => {
                quiet = true;
            }
            "--verify-numerics" => {
                verify_numerics = true;
            }
            "--bisect" => {
                bisect = true;
            }
            "--stats=false" => {
                quiet = true;
            }
            "--stats" | "--stats=true" => {
                quiet = false;
            }
            kv if !kv.starts_with('-')
                && kv.contains('=')
                && !std::path::Path::new(kv).exists() =>
            {
                extra_opts.push(kv.to_string());
            }
            positional if !positional.starts_with('-') && input_path.is_none() => {
                input_path = Some(PathBuf::from(positional));
            }
            "-" if input_path.is_none() => {
                input_path = Some(PathBuf::from("-"));
            }
            other => {
                anyhow::bail!("Unrecognized argument `{other}` for pass `{pass_name}`");
            }
        }
        idx += 1;
    }

    let full_spec = if extra_opts.is_empty() {
        pass_name.clone()
    } else if let Some(prefix) = pass_name.strip_suffix(')') {
        format!("{prefix},{})", extra_opts.join(","))
    } else {
        format!("{pass_name}({})", extra_opts.join(","))
    };

    let src = read_input_text(input_path.as_ref())?;
    let mut module = parse_mlir(&src)?;
    load_optional_weights(&mut module, weights_path.as_ref(), !quiet)?;
    let baseline_eval = if verify_numerics || bisect {
        let inputs = generate_synthetic_inputs(&module, None)?;
        let out_before = evaluate_module(&module, &inputs)?;
        Some((inputs, out_before))
    } else {
        None
    };

    let mut pm = PassManager::from_pipeline_str(registry, &full_spec)?;
    if bisect {
        let (inputs, out_before) = baseline_eval.as_ref().expect("baseline initialized");
        run_pipeline_with_bisect(&mut pm, &mut module, inputs, out_before, tolerance, !quiet)?;
    } else {
        let results = pm.run(&mut module)?;
        if !quiet {
            log_pass_results(&results);
        }
        if let Some((inputs, out_before)) = &baseline_eval {
            let out_after = evaluate_module(&module, inputs)?;
            let rep = compare_output_tensors(out_before, &out_after, tolerance)?;
            if !rep.passed {
                anyhow::bail!(
                    "Numerical verification failed: max_abs_diff {:.6e} > tol {:.2e}",
                    rep.max_abs_diff,
                    rep.tolerance
                );
            }
        }
    }
    if verify_shapes {
        module.verify()?;
        verify_all_stablehlo_shapes(&module)?;
    }
    if let Some(sw_path) = save_weights_path {
        module
            .tensor_store
            .save_safetensors_file(&sw_path)
            .with_context(|| format!("Failed to save weights {}", sw_path.display()))?;
    }
    let formatted = print_mlir_with_options(
        &module,
        PrintOptions {
            generic_form: generic,
            print_locations: locations,
        },
    );
    write_text_output(output_path.as_ref(), &formatted)
}

fn resolve_target_func(module: &Module, func_name: Option<&str>) -> Result<OpId> {
    let fns = module.functions();
    let Some(target) = func_name else {
        return fns
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Module contains no `func.func` operations"));
    };
    let clean = target.trim().trim_start_matches('@');
    for fn_id in fns {
        if module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .is_some_and(|s| s == clean)
        {
            return Ok(fn_id);
        }
    }
    anyhow::bail!("Function `@{clean}` not found in module")
}

fn parse_cli_inputs(spec: &str) -> Result<Vec<Vec<f32>>> {
    let mut tensors = Vec::new();
    for group in spec.split(';') {
        let g = group.trim();
        if g.is_empty() {
            continue;
        }
        let mut vals = Vec::new();
        for tok in g.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let v: f32 = t
                .parse()
                .with_context(|| format!("Invalid float value `{t}` in --inputs"))?;
            vals.push(v);
        }
        tensors.push(vals);
    }
    Ok(tensors)
}

fn load_optional_weights(
    module: &mut Module,
    weights: Option<&PathBuf>,
    log_stats: bool,
) -> Result<()> {
    if let Some(w_path) = weights {
        let count = module
            .tensor_store
            .load_safetensors_file(w_path)
            .with_context(|| format!("Failed to load weights {}", w_path.display()))?;
        if log_stats {
            eprintln!(
                "[castforge:weights] Loaded {} tensors from {}",
                count,
                w_path.display()
            );
        }
    }
    Ok(())
}

fn log_pass_results(results: &[PassResult]) {
    for r in results {
        eprintln!(
            "[castforge:pass] {:<18} | changed: {:<5} | \
             removed: {:<3} | modified: {:<3} | created: {:<3}",
            r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
        );
    }
}

fn run_pipeline_with_bisect(
    pm: &mut PassManager,
    module: &mut Module,
    inputs: &[Vec<f32>],
    out_before: &[Vec<f32>],
    tolerance: f32,
    show_stats: bool,
) -> Result<()> {
    run_pipeline_with_bisect_func(pm, module, None, inputs, out_before, tolerance, show_stats)
}

fn build_roofline(
    arch: Option<&str>,
    peak_gflops: Option<f64>,
    peak_bw_gbps: Option<f64>,
) -> Result<HardwareRoofline> {
    let mut roofline = match arch {
        Some(a) => {
            validate_arch_profile(a)?;
            HardwareRoofline::for_arch(a)
        }
        None => HardwareRoofline::default(),
    };
    if let Some(g) = peak_gflops {
        roofline.peak_gflops = g.max(1e-3);
    }
    if let Some(bw) = peak_bw_gbps {
        roofline.peak_bandwidth_gbps = bw.max(1e-3);
    }
    Ok(roofline)
}

fn read_input_text(input: Option<&PathBuf>) -> Result<String> {
    match input {
        Some(path) if path.as_os_str() != "-" => {
            fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))
        }
        _ => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .context("Failed to read from stdin")?;
            Ok(buf)
        }
    }
}

fn write_text_output(output: Option<&PathBuf>, content: &str) -> Result<()> {
    if let Some(out_path) = output {
        fs::write(out_path, content)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
    } else {
        print!("{content}");
    }
    Ok(())
}

fn verify_all_stablehlo_shapes(module: &Module) -> Result<()> {
    for op_id in module.live_ops() {
        verify_stablehlo_shapes(module, op_id).map_err(|e| anyhow::anyhow!(e))?;
    }
    Ok(())
}

fn write_binary_output(output: Option<&PathBuf>, bytes: &[u8]) -> Result<()> {
    if let Some(out_path) = output {
        fs::write(out_path, bytes)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
    } else {
        let mut stdout = std::io::stdout();
        if stdout.is_terminal() {
            println!(
                "({} bytes — use -o <FILE> to save binary output)",
                bytes.len()
            );
        } else {
            stdout.write_all(bytes)?;
            stdout.flush()?;
        }
    }
    Ok(())
}
