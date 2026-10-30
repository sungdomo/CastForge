//! Command execution handlers (`run`) for all `castforge` CLI subcommands.

use crate::cli::args::{Cli, Commands};
use crate::cli::chain::build_translator_chain;
use crate::codegen::{
    CodegenTarget, GemmLoopOrder, HardwareRoofline, SimdIsa, TargetConfig,
    analyze_module_cost_with_roofline, autotune_module, autotune_module_jit, build_memory_plan,
    evaluate_module, generate_code_with_config, generate_synthetic_inputs, verify_module_numerics,
};
use crate::ir::{
    Payload, collect_module_tensors, export_gguf_file, export_onnx_file, import_hf_model_dir,
};
use crate::mlir::{
    PrintOptions, import_gguf_file, import_onnx_file, parse_mlir, print_mlir,
    print_mlir_with_options,
};
use crate::passes::{ComposeFunctionsPass, Pass, PassManager, PassRegistry, link_modules};
use crate::viz::{
    TimelineFrame, VizFormat, build_graph_json, render_module, render_timeline_html,
};
use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;

/// Parses CLI arguments and executes the requested `castforge` subcommand.
pub fn run() -> Result<()> {
    let registry = PassRegistry::with_builtin_passes();
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
            stats,
            verify_numerics,
            tolerance,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            if let Some(w_path) = weights {
                let count = module
                    .tensor_store
                    .load_safetensors_file(&w_path)
                    .with_context(|| format!("Failed to load weights {}", w_path.display()))?;
                if stats {
                    eprintln!(
                        "[castforge:weights] Loaded {} tensors from {}",
                        count,
                        w_path.display()
                    );
                }
            }
            let before_snapshot = verify_numerics.then(|| module.clone());

            let mut pm = if let Some(cfg_path) = config {
                let toml_str = fs::read_to_string(&cfg_path)
                    .with_context(|| format!("Failed to read config {}", cfg_path.display()))?;
                PassManager::from_toml_str(&registry, &toml_str)?
            } else if let Some(spec) = passes {
                PassManager::from_pipeline_str(&registry, &spec)?
            } else {
                PassManager::from_pipeline_str(&registry, "canonicalize")?
            };

            let results = pm.run(&mut module)?;
            if stats {
                for r in &results {
                    eprintln!(
                        "[castforge:pass] {:<18} | changed: {:<5} | \
                         removed: {:<3} | modified: {:<3} | created: {:<3}",
                        r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
                    );
                }
            }
            if let Some(before_mod) = &before_snapshot {
                let rep = verify_module_numerics(before_mod, &module, tolerance)?;
                eprintln!(
                    "[castforge:verify] passed: {} | max_abs_diff: {:.6e} (tol: {:.2e}, \
                     checked {} elements)",
                    rep.passed, rep.max_abs_diff, rep.tolerance, rep.num_outputs_checked
                );
                if !rep.passed {
                    anyhow::bail!(
                        "Numerical verification failed: max_abs_diff {:.6e} > tolerance {:.2e}",
                        rep.max_abs_diff,
                        rep.tolerance
                    );
                }
            }

            if let Some(sw_path) = save_weights {
                module
                    .tensor_store
                    .save_safetensors_file(&sw_path)
                    .with_context(|| format!("Failed to save weights {}", sw_path.display()))?;
                if stats {
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
                    print_locations: false,
                },
            );

            write_text_output(output.as_ref(), &formatted)?;
        }

        Commands::Run {
            input,
            passes,
            weights,
            tolerance,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            if let Some(w_path) = weights {
                module.tensor_store.load_safetensors_file(&w_path)?;
            }
            let inputs = generate_synthetic_inputs(&module, None)?;
            let baseline = evaluate_module(&module, &inputs)?;
            if let Some(spec) = passes {
                let before_mod = module.clone();
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
                let rep = verify_module_numerics(&before_mod, &module, tolerance)?;
                println!(
                    "Numerical verification: {} (max_abs_diff = {:.6e}, tol = {:.2e})",
                    if rep.passed { "PASSED" } else { "FAILED" },
                    rep.max_abs_diff,
                    rep.tolerance
                );
                let optimized_out = evaluate_module(&module, &inputs)?;
                println!("Outputs: {optimized_out:?}");
            } else {
                println!("Outputs: {baseline:?}");
            }
        }

        Commands::Parse { input, generic } => {
            let src = read_input_text(input.as_ref())?;
            let module = parse_mlir(&src)?;
            module.verify()?;
            let formatted = print_mlir_with_options(
                &module,
                PrintOptions {
                    generic_form: generic,
                    print_locations: false,
                },
            );
            print!("{formatted}");
        }

        Commands::Viz {
            input,
            format,
            passes,
            output,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;

            let fmt = match format.to_lowercase().as_str() {
                "dot" | "graphviz" => VizFormat::Dot,
                "json" => VizFormat::Json,
                "html" => VizFormat::Html,
                _ => VizFormat::Mermaid,
            };

            let rendered = if fmt == VizFormat::Html {
                let trace_steps = if let Some(spec) = passes {
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
                    let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                    pm.run(&mut module)?;
                }
                render_module(&module, fmt)
            };

            if let Some(out_path) = output {
                fs::write(&out_path, rendered)?;
                eprintln!(
                    "[castforge:viz] Wrote {} visualization to {}",
                    format,
                    out_path.display()
                );
            } else {
                println!("{rendered}");
            }
        }

        Commands::Codegen {
            input,
            target,
            passes,
            weights,
            output,
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
            if let Some(w_path) = weights {
                module
                    .tensor_store
                    .load_safetensors_file(&w_path)
                    .with_context(|| format!("Failed to load weights {}", w_path.display()))?;
            }

            if let Some(spec) = passes {
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }

            let tgt = CodegenTarget::parse(&target);

            let plan = build_memory_plan(&module);
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

            let mut cfg = match arch {
                Some(arch_name) => TargetConfig::for_arch(&arch_name),
                None => TargetConfig::default(),
            };
            if let Some(ts) = tile_size {
                cfg.tile_m = ts;
                cfg.tile_n = ts;
                cfg.tile_k = ts;
            }
            if let Some(wg) = workgroup_size {
                cfg.wgsl_workgroup_size = wg.max(1);
            }
            if let Some(order) = loop_order {
                cfg.loop_order = GemmLoopOrder::parse(&order);
            }
            if let Some(uf) = unroll {
                cfg.unroll_factor = uf.max(1);
            }
            if vectorize {
                cfg.vectorize = true;
            }
            if let Some(rt) = reg_tile {
                cfg.reg_tile_m = rt;
                cfg.reg_tile_n = rt;
            }
            if collapse_loops {
                cfg.collapse_loops = true;
            }
            if parallel {
                cfg.parallel = true;
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
            if loop_skew {
                cfg.loop_skew = true;
            }
            if double_buffer {
                cfg.double_buffer = true;
            }
            if let Some(isa_str) = simd_isa {
                let parsed = SimdIsa::parse(&isa_str);
                cfg.simd_isa = parsed;
                cfg.simd_width = parsed.default_lanes().max(1);
            }

            let code = generate_code_with_config(&module, tgt, &cfg)?;
            write_text_output(output.as_ref(), &code)?;
        }

        Commands::Profile {
            input,
            passes,
            peak_gflops,
            peak_bw_gbps,
            arch,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            let roofline = match arch {
                Some(a) => HardwareRoofline::for_arch(&a),
                None => HardwareRoofline::new(peak_gflops, peak_bw_gbps),
            };
            let before = analyze_module_cost_with_roofline(&module, &roofline);
            if let Some(spec) = passes {
                println!("=== Baseline Cost Profile ===");
                print!("{}", before.summary_table());
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
                let after = analyze_module_cost_with_roofline(&module, &roofline);
                println!("\n=== Optimized Cost Profile ({spec}) ===");
                print!("{}", after.summary_table());
                let traffic_saved = before
                    .total_memory_traffic_bytes
                    .saturating_sub(after.total_memory_traffic_bytes);
                println!(
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
            } else {
                print!("{}", before.summary_table());
            }
        }

        Commands::Autotune {
            input,
            passes,
            peak_gflops,
            peak_bw_gbps,
            top_k,
            bench_jit,
            arch,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            if let Some(spec) = passes {
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            let roofline = match arch {
                Some(a) => HardwareRoofline::for_arch(&a),
                None => HardwareRoofline::new(peak_gflops, peak_bw_gbps),
            };
            let report = if bench_jit {
                autotune_module_jit(&module, &roofline, top_k)?
            } else {
                autotune_module(&module, &roofline)
            };
            print!("{}", report.summary_table(top_k));
        }

        Commands::Import {
            input,
            format,
            passes,
            output,
            save_weights,
        } => {
            let fmt = format.unwrap_or_else(|| {
                let s = input.to_string_lossy().to_ascii_lowercase();
                if s.ends_with(".gguf") {
                    "gguf".to_string()
                } else if input.is_dir() || s.ends_with("config.json") {
                    "hf".to_string()
                } else {
                    "onnx".to_string()
                }
            });
            let mut module = match fmt.to_ascii_lowercase().as_str() {
                "gguf" => import_gguf_file(&input)?,
                "hf" | "huggingface" | "config.json" => import_hf_model_dir(&input)?,
                _ => import_onnx_file(&input)?,
            };
            if let Some(spec) = passes {
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
            if let Some(w_path) = weights {
                module.tensor_store.load_safetensors_file(&w_path)?;
            }
            if let Some(spec) = passes {
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            let fmt = format.unwrap_or_else(|| {
                let s = output.to_string_lossy().to_ascii_lowercase();
                if s.ends_with(".gguf") {
                    "gguf".to_string()
                } else if s.ends_with(".safetensors") {
                    "safetensors".to_string()
                } else {
                    "onnx".to_string()
                }
            });
            match fmt.to_ascii_lowercase().as_str() {
                "gguf" => export_gguf_file(&module, &output)?,
                "safetensors" | "st" => {
                    let store = collect_module_tensors(&module);
                    store.save_safetensors_file(&output)?;
                }
                _ => export_onnx_file(&module, &output)?,
            }
        }

        Commands::Chain {
            input,
            stages,
            output,
        } => {
            let (mut chain, first_in, last_out) = build_translator_chain(&registry, &stages)?;
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
            match (result, output) {
                (Payload::Bytes(b), Some(out_path)) => {
                    fs::write(&out_path, &b)?;
                }
                (Payload::Bytes(b), None) => {
                    let mut stdout = std::io::stdout();
                    if stdout.is_terminal() {
                        println!("({} bytes — use -o <FILE> to save binary output)", b.len());
                    } else {
                        stdout.write_all(&b)?;
                        stdout.flush()?;
                    }
                }
                (Payload::Text(s), out_path) => {
                    write_text_output(out_path.as_ref(), &s)?;
                }
                (Payload::Ir(m), out_path) => {
                    let _ = last_out;
                    let text = print_mlir(&m);
                    write_text_output(out_path.as_ref(), &text)?;
                }
            }
        }

        Commands::Link {
            inputs,
            compose,
            passes,
            output,
        } => {
            let first_src = read_input_text(inputs.first())?;
            let mut base = parse_mlir(&first_src)?;
            for path in inputs.iter().skip(1) {
                let src = read_input_text(Some(path))?;
                let other = parse_mlir(&src)?;
                link_modules(&mut base, &other)?;
            }
            if let Some(pair) = compose
                && let Some((f1, f2)) = pair.split_once(',')
            {
                let mut comp = ComposeFunctionsPass {
                    first: f1.trim().to_string(),
                    second: f2.trim().to_string(),
                    entry_name: "composed_main".to_string(),
                };
                comp.run(&mut base)?;
            }
            if let Some(spec) = passes {
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut base)?;
            }
            let formatted = print_mlir(&base);
            write_text_output(output.as_ref(), &formatted)?;
        }

        Commands::ListPasses { tree } => {
            println!("Available CastForge Passes:");
            println!("{:<28} DESCRIPTION", "PASS NAME");
            println!("{}", "-".repeat(86));
            for (name, desc) in registry.list() {
                println!("{:<28} {}", name, desc);
                if tree
                    && let Ok(instance) = registry.create_pass(name, &Default::default())
                    && instance.is_composite()
                {
                    let children = instance.child_passes();
                    for (idx, child) in children.iter().enumerate() {
                        let branch = if idx + 1 == children.len() {
                            "└─"
                        } else {
                            "├─"
                        };
                        println!("  {branch} {child}");
                    }
                }
            }
        }

        Commands::DirectPass(args) => {
            run_direct_pass(&registry, &args)?;
        }
    }

    Ok(())
}

/// Executes a pass invoked directly as a top-level subcommand
/// (`castforge <pass-spec> [input] [-o <output>] [-w <weights>] [--save-weights <path>]`).
fn run_direct_pass(registry: &PassRegistry, args: &[String]) -> Result<()> {
    let Some(pass_name) = args.first() else {
        anyhow::bail!("Missing pass name for direct pass invocation");
    };
    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut weights_path: Option<PathBuf> = None;
    let mut save_weights_path: Option<PathBuf> = None;
    let mut generic = false;

    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-o" | "--output" => {
                idx += 1;
                if let Some(val) = args.get(idx) {
                    output_path = Some(PathBuf::from(val));
                }
            }
            "-w" | "--weights" => {
                idx += 1;
                if let Some(val) = args.get(idx) {
                    weights_path = Some(PathBuf::from(val));
                }
            }
            "--save-weights" => {
                idx += 1;
                if let Some(val) = args.get(idx) {
                    save_weights_path = Some(PathBuf::from(val));
                }
            }
            "--generic" => {
                generic = true;
            }
            positional if input_path.is_none() => {
                input_path = Some(PathBuf::from(positional));
            }
            other => {
                anyhow::bail!("Unrecognized argument `{other}` for pass `{pass_name}`");
            }
        }
        idx += 1;
    }

    let src = read_input_text(input_path.as_ref())?;
    let mut module = parse_mlir(&src)?;
    if let Some(w_path) = weights_path {
        module
            .tensor_store
            .load_safetensors_file(&w_path)
            .with_context(|| format!("Failed to load weights {}", w_path.display()))?;
    }
    let mut pm = PassManager::from_pipeline_str(registry, pass_name)?;
    let results = pm.run(&mut module)?;
    for r in &results {
        eprintln!(
            "[castforge:pass] {:<18} | changed: {:<5} | \
             removed: {:<3} | modified: {:<3} | created: {:<3}",
            r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
        );
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
            print_locations: false,
        },
    );
    write_text_output(output_path.as_ref(), &formatted)
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
