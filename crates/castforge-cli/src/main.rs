use anyhow::{Context, Result};
use castforge_codegen::{CodegenTarget, build_memory_plan, generate_code};
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

        /// Emit canonical MLIR Generic Form ("dialect.op"(...) : ...)
        #[arg(long)]
        generic: bool,

        /// Print pass execution statistics to stderr
        #[arg(long, default_value_t = true)]
        stats: bool,
    },

    /// Parse and round-trip an MLIR file (pretty or --generic form)
    Parse {
        /// Input MLIR file path
        input: PathBuf,

        /// Emit canonical MLIR Generic Form
        #[arg(long)]
        generic: bool,
    },

    /// Export the IR computation graph or interactive Pass Timeline Debugger (mermaid, dot, json, html)
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

        /// Optional output file path
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
            generic,
            stats,
        } => {
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
            let mut module = parse_mlir(&src)?;

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
                        "[castforge:pass] {:<18} | changed: {:<5} | removed: {:<3} | modified: {:<3} | created: {:<3}",
                        r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
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
            output,
        } => {
            let src = fs::read_to_string(&input)
                .with_context(|| format!("Failed to read {}", input.display()))?;
            let mut module = parse_mlir(&src)?;

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
                "[castforge:codegen] Target: {:?} | Constant Weights: {} B | Activation Workspace: {} B ({} buffers)",
                tgt,
                plan.total_constant_bytes,
                plan.total_activation_bytes,
                plan.allocations.len()
            );

            let code = generate_code(&module, tgt)?;
            if let Some(out_path) = output {
                fs::write(&out_path, code)?;
            } else {
                print!("{code}");
            }
        }

        Commands::ListPasses => {
            println!("Available CastForge Passes:");
            println!("{:<20} {}", "PASS NAME", "DESCRIPTION");
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
                    "[castforge:pass] {:<18} | changed: {:<5} | removed: {:<3} | modified: {:<3} | created: {:<3}",
                    r.pass_name, r.changed, r.ops_removed, r.ops_modified, r.ops_created
                );
            }
            let formatted = print_mlir_with_options(&module, PrintOptions::default());
            print!("{formatted}");
        }
    }

    Ok(())
}
