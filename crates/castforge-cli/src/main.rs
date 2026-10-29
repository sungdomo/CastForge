use anyhow::{Context, Result};
use castforge_codegen::{
    CodegenTarget, GemmLoopOrder, HardwareRoofline, TargetConfig,
    analyze_module_cost_with_roofline, build_memory_plan, evaluate_module,
    generate_code_with_config, generate_synthetic_inputs, verify_module_numerics,
};
use castforge_mlir::{PrintOptions, parse_mlir, print_mlir, print_mlir_with_options};
use castforge_passes::{PassManager, PassRegistry};
use castforge_viz::{
    TimelineFrame, VizFormat, build_graph_json, render_module, render_timeline_html,
};
use clap::{Parser, Subcommand};
use std::fs;
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
        /// Input MLIR file path
        input: PathBuf,

        /// Comma-separated pass pipeline (e.g. "fold-transpose,fold-reshape,cse,dce")
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
        /// Input MLIR file path
        input: PathBuf,

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
        /// Input MLIR file path
        input: PathBuf,

        /// Emit canonical MLIR Generic Form
        #[arg(long)]
        generic: bool,
    },

    /// Export the IR computation graph or interactive Pass Timeline Debugger
    Viz {
        /// Input MLIR file path
        input: PathBuf,

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
        /// Input MLIR file path
        input: PathBuf,

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
    },

    /// Analyze FLOPs, memory bandwidth traffic, arithmetic intensity, and Roofline bottlenecks
    Profile {
        /// Input MLIR file path
        input: PathBuf,

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
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
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
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
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
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
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
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
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
        } => {
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
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
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
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
            let input_path = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("Usage: castforge {pass_name} <input.mlir>"))?;
            let src = fs::read_to_string(input_path)
                .with_context(|| format!("Failed to read {input_path}"))?;
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
