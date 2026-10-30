//! Clap command-line argument definitions (`Cli` and `Commands`) for `castforge`.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Top-level `castforge` command-line parser.
#[derive(Parser)]
#[command(
    name = "castforge",
    version,
    about = "CastForge: Pure-Rust AI Model IR Transformation, Optimization & Visualization Toolset"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Supported `castforge` subcommands.
#[derive(Subcommand)]
pub enum Commands {
    /// Run a configurable pipeline of IR transformation passes (via -p or --config)
    Opt {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Comma- or pipe-separated pass pipeline (e.g. "fold-views | cse | dce")
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

    /// Generate executable source code (rust, c, wgsl, cuda, msl, hip, opencl) from an MLIR module
    Codegen {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Codegen target: rust, c, wgsl, cuda, msl (metal), hip (rocm), or opencl
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
        #[arg(long = "tile-size")]
        tile_size: Option<usize>,

        /// WebGPU/GPU workgroup thread count per block (default: 64)
        #[arg(long = "workgroup-size")]
        workgroup_size: Option<u32>,

        /// GEMM loop permutation order: mnk, mkn (Broadcast-A), or nkm
        #[arg(long = "loop-order")]
        loop_order: Option<String>,

        /// Inner-loop unroll factor (1 = disabled, 2/4/8 = unrolled with tail peeling)
        #[arg(long = "unroll")]
        unroll: Option<usize>,

        /// Emit explicit SIMD vectorization pragmas/annotations
        #[arg(long = "vectorize")]
        vectorize: bool,

        /// Register micro-kernel Unroll-and-Jam tile size (0 = disabled, e.g. 2 for 2x2)
        #[arg(long = "reg-tile")]
        reg_tile: Option<usize>,

        /// Collapse/coalesce nested 2D loops into 1D loops
        #[arg(long = "collapse-loops")]
        collapse_loops: bool,

        /// Emit multicore parallel outer-loop work-sharing
        #[arg(long = "parallel")]
        parallel: bool,

        /// Outer L2 cache tile size (0 = disabled)
        #[arg(long = "l2-tile-size")]
        l2_tile_size: Option<usize>,

        /// Software pipelining / prefetch distance in iterations (0 = disabled)
        #[arg(long = "prefetch")]
        prefetch: Option<usize>,

        /// Split-K reduction partitioning factor (1 = disabled)
        #[arg(long = "split-k")]
        split_k: Option<usize>,

        /// Enable polyhedral wavefront loop skewing (w = m + n)
        #[arg(long = "loop-skew")]
        loop_skew: bool,

        /// Enable 2-stage software-pipelined ping-pong tile double-buffering
        #[arg(long = "double-buffer")]
        double_buffer: bool,

        /// Hardware SIMD ISA: scalar, avx2, avx512, neon, sve, rvv, or wasm
        #[arg(long = "simd-isa")]
        simd_isa: Option<String>,

        /// Optional hardware architecture profile (e.g. x86_64-avx2, apple-m-series, cuda-sm80)
        #[arg(long = "arch")]
        arch: Option<String>,
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

        /// Optional hardware architecture Roofline preset (e.g. nvidia-hopper, apple-m-series)
        #[arg(long = "arch")]
        arch: Option<String>,
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

        /// Optional hardware architecture Roofline preset (e.g. x86_64-avx512, nvidia-hopper)
        #[arg(long = "arch")]
        arch: Option<String>,
    },

    /// Import an external model (GGUF v2/v3, ONNX JSON, or HuggingFace config.json) into IR
    Import {
        /// Input model path (.gguf, .onnx.json, config.json, or HF model directory)
        input: PathBuf,

        /// Frontend format: gguf, onnx, or hf (auto-detected from path by default)
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

    /// Export a CastForge MLIR module to an external model format (ONNX JSON, GGUF v3, SafeTensors)
    Export {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Output model file path (.onnx.json, .gguf, or .safetensors)
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Export format: onnx, gguf, or safetensors (auto-detected from -o extension by default)
        #[arg(short = 'f', long = "format")]
        format: Option<String>,

        /// Optional pass pipeline to run before exporting
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before passes/export
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,
    },

    /// Execute a concatenated sequence of modular A-to-B translators (Frontend -> IR -> Backend)
    #[command(visible_alias = "pipe")]
    Chain {
        /// Input file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Concatenated A-to-B translator stages separated by `|` or `,`
        /// (e.g. "fuse-attention | schedule-loops(size=16,isa=avx2) | codegen-rust")
        #[arg(short = 's', long = "stages")]
        stages: String,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Link two or more MLIR modules into a single module and optionally compose entry functions
    Link {
        /// Input MLIR file paths to link in order
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Optional pair of functions `first,second` to chain via `compose-functions`
        #[arg(long = "compose")]
        compose: Option<String>,

        /// Optional pass pipeline to run after linking
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional output MLIR file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// List all registered transformation and optimization passes
    ListPasses {
        /// Display composite pass hierarchy (`child_passes`) as a tree
        #[arg(short = 't', long = "tree")]
        tree: bool,
    },

    /// Dynamically invoke a single pass or preset directly by subcommand name
    /// (e.g., `castforge canonicalize model.mlir -o out.mlir`)
    #[command(external_subcommand)]
    DirectPass(Vec<String>),
}
