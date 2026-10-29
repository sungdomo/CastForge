use anyhow::{Context, Result};
use castforge_codegen::{
    CodegenTarget, CodegenTranslator, GemmLoopOrder, HardwareRoofline, TargetConfig,
    analyze_module_cost_with_roofline, autotune_module, autotune_module_jit, build_memory_plan,
    evaluate_module, generate_code_with_config, generate_synthetic_inputs, verify_module_numerics,
};
use castforge_ir::{Payload, TranslatorChain};
use castforge_mlir::{
    GgufImportTranslator, MlirParseTranslator, MlirPrintTranslator, OnnxImportTranslator,
    PrintOptions, import_gguf_file, import_onnx_file, parse_mlir, print_mlir,
    print_mlir_with_options,
};
use castforge_passes::{PassManager, PassRegistry, parse_pass_spec_item, split_pipeline_spec};
use castforge_viz::{
    TimelineFrame, VizFormat, VizTranslator, build_graph_json, render_module, render_timeline_html,
};
use clap::{Parser, Subcommand};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "castforge",
    version,
    about = "CastForge: Pure-Rust AI Model IR Transformation, Optimization & Visualization Toolset"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a configurable pipeline of IR transformation passes (via -p or --config)
    Opt {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Comma- or pipe-separated pass pipeline (e.g. "fold-transpose | cse | dce")
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Path to TOML pipeline configuration file
        #[arg(short = 'c', long = "config")]
        config: Option<PathBuf>,

        /// Output MLIR file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,

        /// Optional input `.safetensors` file to load into TensorStore before passes
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional output `.safetensors` file to save TensorStore weights after passes
        #[arg(long = "save-weights")]
        save_weights: Option<PathBuf>,

        /// Emit canonical MLIR Generic Form ("dialect.op"(...) : ...)
        #[arg(long)]
        generic: bool,

        /// Print pass execution statistics to stderr
        #[arg(long, default_value_t = true)]
        stats: bool,

        /// Verify numerical equivalence before vs. after optimization passes
        #[arg(long = "verify-numerics")]
        verify_numerics: bool,

        /// Maximum allowed absolute difference when --verify-numerics is enabled
        #[arg(long = "tolerance", default_value_t = 1e-4)]
        tolerance: f32,
    },

    /// Execute an MLIR module directly in the reference interpreter and verify/print outputs
    Run {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Optional pass pipeline to run and numerically verify against the unoptimized module
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before execution
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Numerical verification tolerance when -p/--passes is used
        #[arg(long = "tolerance", default_value_t = 1e-4)]
        tolerance: f32,
    },

    /// Parse and round-trip an MLIR file (pretty or --generic form)
    Parse {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Emit canonical MLIR Generic Form
        #[arg(long)]
        generic: bool,
    },

    /// Export the IR computation graph or interactive Pass Timeline Debugger
    Viz {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Visualization format: mermaid, dot, json, or html
        #[arg(short = 'f', long = "format", default_value = "mermaid")]
        format: String,

        /// Optional pass pipeline to run and trace before visualizing
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional output file path (e.g., trace.html)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Generate executable source code (rust, c, wgsl) from an MLIR module
    Codegen {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Codegen target: rust, c, or wgsl
        #[arg(short = 't', long = "target", default_value = "rust")]
        target: String,

        /// Optional pass pipeline to run before code generation
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before passes/codegen
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional output file path
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,

        /// Optional cache/register tile size for GEMM loops (0 = untiled)
        #[arg(long = "tile-size", default_value_t = 0)]
        tile_size: usize,

        /// WebGPU `@workgroup_size` thread count per workgroup
        #[arg(long = "workgroup-size", default_value_t = 64)]
        workgroup_size: u32,

        /// GEMM loop permutation order: mnk, mkn (Broadcast-A), or nkm
        #[arg(long = "loop-order", default_value = "mnk")]
        loop_order: String,

        /// Inner-loop unroll factor (1 = disabled, 2/4/8 = unrolled with tail peeling)
        #[arg(long = "unroll", default_value_t = 1)]
        unroll: usize,

        /// Emit explicit SIMD vectorization pragmas/annotations
        #[arg(long = "vectorize")]
        vectorize: bool,

        /// Register micro-kernel Unroll-and-Jam tile size (0 = disabled, e.g. 2 for 2x2)
        #[arg(long = "reg-tile", default_value_t = 0)]
        reg_tile: usize,

        /// Collapse/coalesce nested 2D loops into 1D loops
        #[arg(long = "collapse-loops")]
        collapse_loops: bool,

        /// Emit multicore parallel outer-loop work-sharing
        #[arg(long = "parallel")]
        parallel: bool,

        /// Outer L2 cache tile size (0 = disabled)
        #[arg(long = "l2-tile-size", default_value_t = 0)]
        l2_tile_size: usize,

        /// Software pipelining / prefetch distance in iterations (0 = disabled)
        #[arg(long = "prefetch", default_value_t = 0)]
        prefetch: usize,

        /// Split-K reduction partitioning factor (1 = disabled)
        #[arg(long = "split-k", default_value_t = 1)]
        split_k: usize,

        /// Enable polyhedral wavefront loop skewing (w = m + n)
        #[arg(long = "loop-skew")]
        loop_skew: bool,

        /// Enable 2-stage software-pipelined ping-pong tile double-buffering
        #[arg(long = "double-buffer")]
        double_buffer: bool,

        /// Hardware SIMD ISA for explicit intrinsics: scalar, avx2, or neon
        #[arg(long = "simd-isa", default_value = "scalar")]
        simd_isa: String,
    },

    /// Analyze FLOPs, memory bandwidth traffic, arithmetic intensity, and Roofline bottlenecks
    Profile {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Optional pass pipeline to compare before vs. after Roofline metrics
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Peak hardware compute throughput in GFLOP/s
        #[arg(long = "peak-gflops", default_value_t = 1000.0)]
        peak_gflops: f64,

        /// Peak hardware memory bandwidth in GB/s
        #[arg(long = "peak-bw-gbps", default_value_t = 100.0)]
        peak_bw_gbps: f64,
    },

    /// Auto-tune polyhedral/HPC loop schedules using the Roofline & cache model
    Autotune {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Optional pass pipeline to run before schedule auto-tuning
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Peak hardware compute throughput in GFLOP/s
        #[arg(long = "peak-gflops", default_value_t = 1000.0)]
        peak_gflops: f64,

        /// Peak hardware memory bandwidth in GB/s
        #[arg(long = "peak-bw-gbps", default_value_t = 100.0)]
        peak_bw_gbps: f64,

        /// Number of top-ranked schedule candidates to display
        #[arg(long = "top-k", default_value_t = 5)]
        top_k: usize,

        /// Compile and empirically benchmark top schedules via native JIT
        #[arg(long = "bench-jit")]
        bench_jit: bool,
    },

    /// Import an external model (GGUF v2/v3 binary or ONNX JSON) into StableHLO IR + TensorStore
    Import {
        /// Input model path (.gguf or .onnx.json)
        input: PathBuf,

        /// Frontend format: gguf or onnx (auto-detected from extension by default)
        #[arg(short = 'f', long = "format")]
        format: Option<String>,

        /// Optional pass pipeline to run after importing
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional output MLIR file path
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,

        /// Optional output `.safetensors` file to save imported TensorStore weights
        #[arg(long = "save-weights")]
        save_weights: Option<PathBuf>,
    },

    /// Execute a concatenated sequence of modular A-to-B translators (Frontend -> IR -> Backend)
    #[command(visible_alias = "pipe")]
    Chain {
        /// Input file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Concatenated A-to-B translator stages separated by `|` or `,`
        /// (e.g. "fuse-attention | tile-loops(size=16) | vectorize-loops(isa=avx2) | codegen-rust")
        #[arg(short = 's', long = "stages")]
        stages: String,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// List all registered transformation and optimization passes
    ListPasses,

    /// Dynamically invoke a single pass or preset directly by subcommand name
    /// (e.g., `castforge canonicalize model.mlir` or `castforge fold-transpose model.mlir`)
    #[command(external_subcommand)]
    DirectPass(Vec<String>),
}

fn main() -> Result<()> {
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

            if let Some(out_path) = output {
                fs::write(&out_path, formatted)?;
            } else {
                print!("{formatted}");
            }
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

            let tgt = match target.to_lowercase().as_str() {
                "c" | "c99" => CodegenTarget::C,
                "wgsl" | "webgpu" => CodegenTarget::Wgsl,
                _ => CodegenTarget::Rust,
            };

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

            let cfg = TargetConfig {
                tile_m: tile_size,
                tile_n: tile_size,
                tile_k: tile_size,
                simd_width: 8,
                wgsl_workgroup_size: workgroup_size,
                loop_order: GemmLoopOrder::parse(&loop_order),
                unroll_factor: unroll.max(1),
                vectorize,
                reg_tile_m: reg_tile,
                reg_tile_n: reg_tile,
                collapse_loops,
                parallel,
                l2_tile_size,
                prefetch_distance: prefetch,
                split_k: split_k.max(1),
                loop_skew,
                double_buffer,
                simd_isa: castforge_codegen::SimdIsa::parse(&simd_isa),
            };
            let code = generate_code_with_config(&module, tgt, &cfg)?;
            if let Some(out_path) = output {
                fs::write(&out_path, code)?;
            } else {
                print!("{code}");
            }
        }

        Commands::Profile {
            input,
            passes,
            peak_gflops,
            peak_bw_gbps,
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            let roofline = HardwareRoofline::new(peak_gflops, peak_bw_gbps);
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
        } => {
            let src = read_input_text(input.as_ref())?;
            let mut module = parse_mlir(&src)?;
            if let Some(spec) = passes {
                let mut pm = PassManager::from_pipeline_str(&registry, &spec)?;
                pm.run(&mut module)?;
            }
            let roofline = HardwareRoofline::new(peak_gflops, peak_bw_gbps);
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
                } else {
                    "onnx".to_string()
                }
            });
            let mut module = match fmt.to_ascii_lowercase().as_str() {
                "gguf" => import_gguf_file(&input)?,
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
            if let Some(out_path) = output {
                fs::write(&out_path, formatted)?;
            } else {
                print!("{formatted}");
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
            let text_out = match (result, last_out) {
                (Payload::Text(s), _) => s,
                (Payload::Ir(m), _) => print_mlir(&m),
                (Payload::Bytes(b), _) => format!("({} bytes)", b.len()),
            };
            if let Some(out_path) = output {
                fs::write(&out_path, text_out)?;
            } else {
                print!("{text_out}");
            }
        }

        Commands::ListPasses => {
            println!("Available CastForge Passes:");
            println!("{:<20} DESCRIPTION", "PASS NAME");
            println!("{}", "-".repeat(78));
            for (name, desc) in registry.list() {
                println!("{:<20} {}", name, desc);
            }
        }

        Commands::DirectPass(args) => {
            let pass_name = &args[0];
            let input_path = args.get(1).map(PathBuf::from);
            let src = read_input_text(input_path.as_ref())?;
            let mut module = parse_mlir(&src)?;
            let mut pm = PassManager::from_pipeline_str(&registry, pass_name)?;
            let results = pm.run(&mut module)?;
            for r in &results {
                eprintln!(
                    "[castforge:pass] {:<18} | changed: {:<5} | \
                     removed: {:<3} | modified: {:<3} | created: {:<3}",
                    r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
                );
            }
            let formatted = print_mlir_with_options(&module, PrintOptions::default());
            print!("{formatted}");
        }
    }

    Ok(())
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

fn build_translator_chain(
    registry: &PassRegistry,
    spec: &str,
) -> Result<(TranslatorChain, &'static str, &'static str)> {
    let mut chain = TranslatorChain::new();
    let items: Vec<&str> = split_pipeline_spec(spec)
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if items.is_empty() {
        anyhow::bail!("Empty translator chain specification");
    }

    let (first_name, _) = parse_pass_spec_item(items[0])?;
    let starts_with_frontend = matches!(
        first_name.as_str(),
        "parse-mlir" | "parse" | "import-onnx" | "import-gguf"
    );
    let mut first_in = "text";
    if !starts_with_frontend {
        chain.push(Box::new(MlirParseTranslator));
    }

    let mut last_out = "ir";
    for (idx, item) in items.iter().enumerate() {
        let (name, opts) = parse_pass_spec_item(item)?;
        match name.as_str() {
            "parse-mlir" | "parse" => {
                if idx == 0 {
                    first_in = "text";
                }
                chain.push(Box::new(MlirParseTranslator));
                last_out = "ir";
            }
            "import-onnx" => {
                if idx == 0 {
                    first_in = "text";
                }
                chain.push(Box::new(OnnxImportTranslator));
                last_out = "ir";
            }
            "import-gguf" => {
                if idx == 0 {
                    first_in = "bytes";
                }
                chain.push(Box::new(GgufImportTranslator));
                last_out = "ir";
            }
            "print-mlir" | "print" => {
                let generic = opts
                    .get("generic")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                chain.push(Box::new(MlirPrintTranslator {
                    options: PrintOptions {
                        generic_form: generic,
                        print_locations: false,
                    },
                }));
                last_out = "text";
            }
            "codegen" | "codegen-rust" | "codegen-c" | "codegen-wgsl" => {
                let tgt_str =
                    opts.get("target")
                        .map(String::as_str)
                        .unwrap_or(match name.as_str() {
                            "codegen-c" => "c",
                            "codegen-wgsl" => "wgsl",
                            _ => "rust",
                        });
                let tgt = match tgt_str.to_ascii_lowercase().as_str() {
                    "c" | "c99" => CodegenTarget::C,
                    "wgsl" | "webgpu" => CodegenTarget::Wgsl,
                    _ => CodegenTarget::Rust,
                };
                chain.push(Box::new(CodegenTranslator::new(tgt)));
                last_out = "text";
            }
            "viz" | "viz-mermaid" | "viz-dot" | "viz-json" | "viz-html" => {
                let fmt_str =
                    opts.get("format")
                        .map(String::as_str)
                        .unwrap_or(match name.as_str() {
                            "viz-dot" => "dot",
                            "viz-json" => "json",
                            "viz-html" => "html",
                            _ => "mermaid",
                        });
                let fmt = match fmt_str.to_ascii_lowercase().as_str() {
                    "dot" | "graphviz" => VizFormat::Dot,
                    "json" => VizFormat::Json,
                    "html" => VizFormat::Html,
                    _ => VizFormat::Mermaid,
                };
                chain.push(Box::new(VizTranslator { format: fmt }));
                last_out = "text";
            }
            pass_name => {
                let pass = registry.create_pass(pass_name, &opts)?;
                chain.push(Box::new(pass));
                last_out = "ir";
            }
        }
    }

    if last_out == "ir" {
        chain.push(Box::new(MlirPrintTranslator::default()));
        last_out = "text";
    }

    Ok((chain, first_in, last_out))
}
