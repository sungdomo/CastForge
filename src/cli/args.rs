//! Clap command-line argument definitions (`Cli` and `Commands`) for `castforge`.

use clap::{ArgAction, Parser, Subcommand};
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

        /// Emit source location annotations (`loc(...)`) in printed MLIR
        #[arg(long = "locations")]
        locations: bool,

        /// Verify StableHLO shape & type inference rules after passes
        #[arg(long = "verify-shapes")]
        verify_shapes: bool,

        /// Print pass execution statistics to stderr (--stats=false or -q to suppress)
        #[arg(
            long,
            default_value_t = true,
            num_args = 0..=1,
            default_missing_value = "true",
            action = ArgAction::Set
        )]
        stats: bool,

        /// Suppress pass statistics logging to stderr
        #[arg(short = 'q', long = "quiet")]
        quiet: bool,

        /// Verify numerical equivalence before vs. after optimization passes
        #[arg(long = "verify-numerics")]
        verify_numerics: bool,

        /// Bisect the pass pipeline step-by-step to pinpoint numerical drift
        #[arg(long = "bisect")]
        bisect: bool,

        /// Maximum allowed absolute difference when --verify-numerics or --bisect is enabled
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

        /// Optional target function name to evaluate (defaults to first `func.func`)
        #[arg(long = "func")]
        func: Option<String>,

        /// Optional explicit input tensors (`"v1,v2;v3,v4"`), overriding synthetic inputs
        #[arg(long = "inputs")]
        inputs: Option<String>,

        /// Bisect the pass pipeline step-by-step to pinpoint numerical drift
        #[arg(long = "bisect")]
        bisect: bool,

        /// Numerical verification tolerance when -p/--passes or --bisect is used
        #[arg(long = "tolerance", default_value_t = 1e-4)]
        tolerance: f32,

        /// Emit evaluation outputs and verification metrics as JSON
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Parse and round-trip an MLIR file (pretty or --generic form)
    Parse {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Output MLIR file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,

        /// Emit canonical MLIR Generic Form
        #[arg(long)]
        generic: bool,

        /// Emit source location annotations (`loc(...)`) in printed MLIR
        #[arg(long = "locations")]
        locations: bool,

        /// Verify StableHLO shape & type inference rules in addition to SSA dominance
        #[arg(long = "verify-shapes")]
        verify_shapes: bool,
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

        /// Optional input `.safetensors` file to load into TensorStore before passes
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional output file path (e.g., trace.html)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Generate executable source code (rust, c, wgsl, cuda, msl, hip, opencl, header, loop-ir)
    Codegen {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Codegen target: rust, c, wgsl, cuda, msl, hip, opencl, header, loop-ir, or lib
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

        /// Optional path to emit a companion C/C++ `.h` FFI header alongside codegen
        #[arg(long = "emit-header")]
        emit_header: Option<PathBuf>,

        /// Automatically search and apply the Roofline-optimal loop schedule before codegen
        #[arg(long = "autotune")]
        autotune: bool,

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

        /// Emit explicit SIMD vectorization pragmas/annotations (--vectorize or --vectorize=false)
        #[arg(long = "vectorize", num_args = 0..=1, default_missing_value = "true")]
        vectorize: Option<bool>,

        /// Register micro-kernel Unroll-and-Jam tile size (0 = disabled, e.g. 2 for 2x2)
        #[arg(long = "reg-tile")]
        reg_tile: Option<usize>,

        /// Collapse/coalesce nested 2D loops into 1D loops
        #[arg(long = "collapse-loops", num_args = 0..=1, default_missing_value = "true")]
        collapse_loops: Option<bool>,

        /// Emit multicore parallel outer-loop work-sharing (--parallel or --parallel=false)
        #[arg(long = "parallel", num_args = 0..=1, default_missing_value = "true")]
        parallel: Option<bool>,

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
        #[arg(long = "loop-skew", num_args = 0..=1, default_missing_value = "true")]
        loop_skew: Option<bool>,

        /// Enable 2-stage software-pipelined ping-pong tile double-buffering
        #[arg(long = "double-buffer", num_args = 0..=1, default_missing_value = "true")]
        double_buffer: Option<bool>,

        /// Hardware SIMD ISA: scalar, avx2, avx512, neon, sve, rvv, or wasm
        #[arg(long = "simd-isa")]
        simd_isa: Option<String>,

        /// Optional hardware architecture profile (e.g. x86_64-avx2, apple-m-series, cuda-sm80)
        #[arg(long = "arch")]
        arch: Option<String>,
    },

    /// Compile and benchmark generated Rust/C99 code natively on the host CPU with numeric check
    Bench {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Native JIT target to benchmark: rust or c
        #[arg(short = 't', long = "target", default_value = "rust")]
        target: String,

        /// Optional pass pipeline to run before codegen & benchmarking
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional hardware architecture profile (e.g. x86_64-avx2, aarch64-neon)
        #[arg(long = "arch")]
        arch: Option<String>,

        /// Automatically search and apply the Roofline-optimal loop schedule before benchmarking
        #[arg(long = "autotune")]
        autotune: bool,

        /// Number of warmup iterations before timing
        #[arg(long = "warmup", default_value_t = 3)]
        warmup: usize,

        /// Number of timed benchmark iterations
        #[arg(long = "iters", default_value_t = 20)]
        iters: usize,

        /// Emit benchmark report as JSON
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Analyze FLOPs, memory bandwidth traffic, arithmetic intensity, and Roofline bottlenecks
    Profile {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Optional pass pipeline to compare before vs. after Roofline metrics
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before passes
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Peak hardware compute throughput in GFLOP/s (default: 1000.0 or --arch preset)
        #[arg(long = "peak-gflops")]
        peak_gflops: Option<f64>,

        /// Peak hardware memory bandwidth in GB/s (default: 100.0 or --arch preset)
        #[arg(long = "peak-bw-gbps")]
        peak_bw_gbps: Option<f64>,

        /// Optional hardware architecture Roofline preset (e.g. nvidia-hopper, apple-m-series)
        #[arg(long = "arch")]
        arch: Option<String>,

        /// Emit Roofline cost report as formatted JSON instead of an ASCII table
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Auto-tune polyhedral/HPC loop schedules using the Roofline & cache model
    Autotune {
        /// Input MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Optional pass pipeline to run before schedule auto-tuning
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before passes
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Peak hardware compute throughput in GFLOP/s (default: 1000.0 or --arch preset)
        #[arg(long = "peak-gflops")]
        peak_gflops: Option<f64>,

        /// Peak hardware memory bandwidth in GB/s (default: 100.0 or --arch preset)
        #[arg(long = "peak-bw-gbps")]
        peak_bw_gbps: Option<f64>,

        /// Number of top-ranked schedule candidates to display
        #[arg(long = "top-k", default_value_t = 5)]
        top_k: usize,

        /// Compile and empirically benchmark top schedules via native JIT
        #[arg(long = "bench-jit")]
        bench_jit: bool,

        /// Optional hardware architecture Roofline preset (e.g. x86_64-avx512, nvidia-hopper)
        #[arg(long = "arch")]
        arch: Option<String>,

        /// Emit auto-tuning report as JSON instead of an ASCII table
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path to write the schedule-stamped MLIR module (or JSON if --json)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Compute a unified MLIR diff between two files or before vs. after a pass pipeline
    Diff {
        /// First (baseline) MLIR file path (reads from stdin if omitted or "-")
        input: Option<PathBuf>,

        /// Optional second MLIR file path to compare against `input`
        second: Option<PathBuf>,

        /// Optional pass pipeline to run on `input` when `second` is omitted
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before passes
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Inspect a model, MLIR module, or `.safetensors` weight file (functions, ops, weights, arena)
    #[command(visible_alias = "info")]
    Inspect {
        /// Input path (.mlir, .gguf, .onnx.json, config.json, or .safetensors)
        input: Option<PathBuf>,

        /// Optional `.safetensors` weights file to include in inspection
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional pass pipeline to run before inspecting the module
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Emit inspection report as formatted JSON
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Import an external model (GGUF, SafeTensors, ONNX JSON, or HF config.json) into IR
    Import {
        /// Input model path (.gguf, .safetensors, .onnx.json, config.json, or HF model dir)
        input: PathBuf,

        /// Frontend format: gguf, safetensors, onnx, or hf (auto-detected by default)
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

        /// Optional input `.safetensors` file to preload into TensorStore when starting from IR
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Link two or more MLIR modules into a single module and optionally compose entry functions
    Link {
        /// Input MLIR file paths to link in order
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Optional pair of functions `first,second` (or `first,second:entry`) to compose
        #[arg(long = "compose")]
        compose: Option<String>,

        /// Optional pass pipeline to run after linking
        #[arg(short = 'p', long = "passes")]
        passes: Option<String>,

        /// Optional input `.safetensors` file to load into TensorStore before passes
        #[arg(short = 'w', long = "weights")]
        weights: Option<PathBuf>,

        /// Optional output MLIR file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// List all registered transformation and optimization passes (from `config/passes.toml`)
    ListPasses {
        /// Optional substring filter to match pass names, families, options, or descriptions
        filter: Option<String>,

        /// Display composite pass hierarchy (`child_passes`) as a tree
        #[arg(short = 't', long = "tree")]
        tree: bool,

        /// Display accepted `key=val` command-line options and defaults for each pass
        #[arg(short = 'a', long = "options", visible_alias = "args")]
        options: bool,

        /// Emit registered passes (including family, hierarchy, and options) as formatted JSON
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// List all codegen targets, hardware `--arch` profiles, and composable translator stages
    #[command(visible_alias = "targets")]
    ListTargets {
        /// Optional substring filter to match target, arch profile, or translator names
        filter: Option<String>,

        /// Emit targets, arch profiles, and translators as formatted JSON
        #[arg(long = "json")]
        json: bool,

        /// Optional output file path (defaults to stdout)
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
    },

    /// Dynamically invoke a single pass or preset directly by subcommand name
    /// (e.g., `castforge canonicalize model.mlir -o out.mlir`)
    #[command(external_subcommand)]
    DirectPass(Vec<String>),
}
