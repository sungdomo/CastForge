//! Builder for composable A-to-B [`TranslatorChain`] pipelines (`castforge chain` / `pipe`).

use crate::codegen::{
    AutotuneScheduleTranslator, CHeaderTranslator, CodegenTarget, CodegenTranslator, GemmLoopOrder,
    HardwareRoofline, MemoryPlanTranslator, OptimizeLoopIrPass, OptimizeLoopIrTranslator,
    RooflineReportTranslator, SharedLibTranslator, SimdIsa, TargetConfig, TensorToLoopTranslator,
};
use crate::ir::{HfConfigImportTranslator, Payload, Translator, TranslatorChain};
use crate::mlir::{
    GgufExportTranslator, GgufImportTranslator, MlirParseTranslator, MlirPrintTranslator,
    OnnxExportTranslator, OnnxImportTranslator, PrintOptions, SafetensorsExportTranslator,
};
use crate::passes::{PassRegistry, parse_pass_spec_item, split_pipeline_spec};
use crate::viz::{IrDiffTranslator, SnapshotIrTranslator, VizFormat, VizTranslator};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use std::path::PathBuf;

/// Translator stage that loads a `.safetensors` file into `Module::tensor_store` (`Ir -> Ir`).
pub struct LoadWeightsTranslator {
    pub path: PathBuf,
}

impl Translator for LoadWeightsTranslator {
    fn name(&self) -> &str {
        "load-weights"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let mut module = input.into_ir()?;
        module
            .tensor_store
            .load_safetensors_file(&self.path)
            .with_context(|| format!("Failed to load weights {}", self.path.display()))?;
        Ok(Payload::from_ir(module))
    }
}

/// Strictly validates and parses a [`CodegenTarget`] name, returning an error on unknown targets.
pub(crate) fn parse_codegen_target_strict(s: &str) -> Result<CodegenTarget> {
    match s.trim().to_ascii_lowercase().as_str() {
        "rust" | "rs" => Ok(CodegenTarget::Rust),
        "c" | "c99" => Ok(CodegenTarget::C),
        "wgsl" | "webgpu" => Ok(CodegenTarget::Wgsl),
        "cuda" | "cu" | "nvptx" => Ok(CodegenTarget::Cuda),
        "msl" | "metal" => Ok(CodegenTarget::Msl),
        "hip" | "rocm" => Ok(CodegenTarget::Hip),
        "opencl" | "cl" => Ok(CodegenTarget::OpenCl),
        other => anyhow::bail!(
            "Unknown codegen target `{other}` \
             (expected: rust, c, wgsl, cuda, msl, hip, opencl, header, loop-ir, or lib)"
        ),
    }
}

/// Strictly validates and parses a [`VizFormat`] name, returning an error on unknown formats.
pub(crate) fn parse_viz_format_strict(s: &str) -> Result<VizFormat> {
    match s.trim().to_ascii_lowercase().as_str() {
        "mermaid" | "mmd" => Ok(VizFormat::Mermaid),
        "dot" | "graphviz" => Ok(VizFormat::Dot),
        "json" => Ok(VizFormat::Json),
        "html" | "timeline" => Ok(VizFormat::Html),
        other => anyhow::bail!(
            "Unknown visualization format `{other}` (expected: mermaid, dot, json, or html)"
        ),
    }
}

/// Parses a `|`- or `,`-delimited translator chain specification into a validated
/// [`TranslatorChain`], returning `(chain, initial_input_kind, final_output_kind)`.
pub fn build_translator_chain(
    registry: &PassRegistry,
    spec: &str,
) -> Result<(TranslatorChain, &'static str, &'static str)> {
    build_translator_chain_with_weights(registry, spec, None)
}

/// Parses a translator chain specification and optionally injects a [`LoadWeightsTranslator`]
/// right after the initial frontend import/parse stage.
pub fn build_translator_chain_with_weights(
    registry: &PassRegistry,
    spec: &str,
    weights: Option<PathBuf>,
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
        "parse-mlir" | "parse" | "import-onnx" | "import-gguf" | "import-hf-config"
    );
    let mut first_in = "text";
    if !starts_with_frontend {
        chain.push(Box::new(MlirParseTranslator));
        if let Some(w_path) = weights.clone() {
            chain.push(Box::new(LoadWeightsTranslator { path: w_path }));
        }
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
                if idx == 0
                    && let Some(w_path) = weights.clone()
                {
                    chain.push(Box::new(LoadWeightsTranslator { path: w_path }));
                }
                last_out = "ir";
            }
            "import-onnx" => {
                if idx == 0 {
                    first_in = "text";
                }
                chain.push(Box::new(OnnxImportTranslator));
                if idx == 0
                    && let Some(w_path) = weights.clone()
                {
                    chain.push(Box::new(LoadWeightsTranslator { path: w_path }));
                }
                last_out = "ir";
            }
            "import-hf-config" => {
                if idx == 0 {
                    first_in = "text";
                }
                chain.push(Box::new(HfConfigImportTranslator));
                if idx == 0
                    && let Some(w_path) = weights.clone()
                {
                    chain.push(Box::new(LoadWeightsTranslator { path: w_path }));
                }
                last_out = "ir";
            }
            "import-gguf" => {
                if idx == 0 {
                    first_in = "bytes";
                }
                chain.push(Box::new(GgufImportTranslator));
                if idx == 0
                    && let Some(w_path) = weights.clone()
                {
                    chain.push(Box::new(LoadWeightsTranslator { path: w_path }));
                }
                last_out = "ir";
            }
            "load-weights" => {
                let path_str = opts
                    .get("path")
                    .or_else(|| opts.get("file"))
                    .ok_or_else(|| anyhow::anyhow!("`load-weights` requires `path=<file>`"))?;
                chain.push(Box::new(LoadWeightsTranslator {
                    path: PathBuf::from(path_str),
                }));
                last_out = "ir";
            }
            "export-onnx" => {
                chain.push(Box::new(OnnxExportTranslator));
                last_out = "text";
            }
            "export-gguf" => {
                chain.push(Box::new(GgufExportTranslator));
                last_out = "bytes";
            }
            "export-safetensors" => {
                chain.push(Box::new(SafetensorsExportTranslator));
                last_out = "bytes";
            }
            "print-mlir" | "print" => {
                let generic = parse_bool_opt(&opts, "generic", false);
                let locations = parse_bool_opt(&opts, "locations", false)
                    || parse_bool_opt(&opts, "print_locations", false);
                chain.push(Box::new(MlirPrintTranslator {
                    options: PrintOptions {
                        generic_form: generic,
                        print_locations: locations,
                    },
                }));
                last_out = "text";
            }
            "lower-to-loops" => {
                let config = parse_target_config_opts(&opts);
                chain.push(Box::new(TensorToLoopTranslator { config }));
                last_out = "text";
            }
            "optimize-loop-ir" => {
                let config = parse_target_config_opts(&opts);
                let mut pass = OptimizeLoopIrPass::default();
                if opts.contains_key("fuse") || opts.contains_key("fuse_loops") {
                    let key = if opts.contains_key("fuse") {
                        "fuse"
                    } else {
                        "fuse_loops"
                    };
                    pass.fuse_loops = parse_bool_opt(&opts, key, pass.fuse_loops);
                }
                if opts.contains_key("vectorize") {
                    pass.vectorize = parse_bool_opt(&opts, "vectorize", pass.vectorize);
                }
                if let Some(v) = opts.get("unroll").or_else(|| opts.get("unroll_factor"))
                    && let Ok(n) = v.parse::<usize>()
                {
                    pass.unroll_factor = n.max(1);
                }
                if opts.contains_key("parallel") {
                    pass.parallel = parse_bool_opt(&opts, "parallel", pass.parallel);
                }
                if opts.contains_key("interchange") {
                    pass.interchange = parse_bool_opt(&opts, "interchange", pass.interchange);
                }
                chain.push(Box::new(OptimizeLoopIrTranslator { config, pass }));
                last_out = "text";
            }
            "codegen-header" | "c-header" | "emit-header" => {
                chain.push(Box::new(CHeaderTranslator));
                last_out = "text";
            }
            "compile-lib" | "shared-lib" => {
                let config = parse_target_config_opts(&opts);
                chain.push(Box::new(SharedLibTranslator { config }));
                last_out = "bytes";
            }
            "autotune-schedule" => {
                let roofline = parse_roofline_opts(&opts);
                chain.push(Box::new(AutotuneScheduleTranslator { roofline }));
                last_out = "ir";
            }
            "roofline-report" => {
                let roofline = parse_roofline_opts(&opts);
                chain.push(Box::new(RooflineReportTranslator { roofline }));
                last_out = "text";
            }
            "memory-plan" => {
                chain.push(Box::new(MemoryPlanTranslator));
                last_out = "text";
            }
            "snapshot-ir" => {
                chain.push(Box::new(SnapshotIrTranslator));
                last_out = "ir";
            }
            "ir-diff" => {
                chain.push(Box::new(IrDiffTranslator::default()));
                last_out = "text";
            }
            "codegen" | "codegen-rust" | "codegen-c" | "codegen-wgsl" | "codegen-cuda"
            | "codegen-msl" | "codegen-metal" | "codegen-hip" | "codegen-rocm"
            | "codegen-opencl" | "codegen-cl" => {
                let tgt_str =
                    opts.get("target")
                        .map(String::as_str)
                        .unwrap_or(match name.as_str() {
                            "codegen-c" => "c",
                            "codegen-wgsl" => "wgsl",
                            "codegen-cuda" => "cuda",
                            "codegen-msl" | "codegen-metal" => "msl",
                            "codegen-hip" | "codegen-rocm" => "hip",
                            "codegen-opencl" | "codegen-cl" => "opencl",
                            _ => "rust",
                        });
                let tgt = parse_codegen_target_strict(tgt_str)?;
                let cfg = parse_target_config_opts(&opts);
                chain.push(Box::new(CodegenTranslator::with_config(tgt, cfg)));
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
                let fmt = parse_viz_format_strict(fmt_str)?;
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

fn parse_bool_opt(opts: &IndexMap<String, String>, key: &str, default: bool) -> bool {
    opts.get(key)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "yes"))
        .unwrap_or(default)
}

/// Builds a [`TargetConfig`] from translator stage options (`arch`, `tile_size`, `unroll`, etc.).
pub(crate) fn parse_target_config_opts(opts: &IndexMap<String, String>) -> TargetConfig {
    let mut cfg = match opts.get("arch") {
        Some(arch) => TargetConfig::for_arch(arch),
        None => TargetConfig::default(),
    };
    if let Some(v) = opts.get("tile_size").or_else(|| opts.get("size"))
        && let Ok(n) = v.parse::<usize>()
    {
        cfg.tile_m = n;
        cfg.tile_n = n;
        cfg.tile_k = n;
    }
    if let Some(v) = opts.get("workgroup_size")
        && let Ok(n) = v.parse::<u32>()
    {
        cfg.wgsl_workgroup_size = n.max(1);
    }
    if let Some(order) = opts.get("loop_order").or_else(|| opts.get("order")) {
        cfg.loop_order = GemmLoopOrder::parse(order);
    }
    if let Some(v) = opts.get("unroll").or_else(|| opts.get("unroll_factor"))
        && let Ok(n) = v.parse::<usize>()
    {
        cfg.unroll_factor = n.max(1);
    }
    if opts.contains_key("vectorize") {
        cfg.vectorize = parse_bool_opt(opts, "vectorize", cfg.vectorize);
    }
    if let Some(v) = opts.get("reg_tile")
        && let Ok(n) = v.parse::<usize>()
    {
        cfg.reg_tile_m = n;
        cfg.reg_tile_n = n;
    }
    if opts.contains_key("collapse_loops") {
        cfg.collapse_loops = parse_bool_opt(opts, "collapse_loops", cfg.collapse_loops);
    }
    if opts.contains_key("parallel") {
        cfg.parallel = parse_bool_opt(opts, "parallel", cfg.parallel);
    }
    if let Some(v) = opts.get("l2_tile_size")
        && let Ok(n) = v.parse::<usize>()
    {
        cfg.l2_tile_size = n;
    }
    if let Some(v) = opts.get("prefetch")
        && let Ok(n) = v.parse::<usize>()
    {
        cfg.prefetch_distance = n;
    }
    if let Some(v) = opts.get("split_k")
        && let Ok(n) = v.parse::<usize>()
    {
        cfg.split_k = n.max(1);
    }
    if opts.contains_key("loop_skew") {
        cfg.loop_skew = parse_bool_opt(opts, "loop_skew", cfg.loop_skew);
    }
    if opts.contains_key("double_buffer") {
        cfg.double_buffer = parse_bool_opt(opts, "double_buffer", cfg.double_buffer);
    }
    if let Some(isa) = opts.get("simd_isa").or_else(|| opts.get("isa")) {
        let parsed = SimdIsa::parse(isa);
        cfg.simd_isa = parsed;
        cfg.simd_width = parsed.default_lanes().max(1);
    }
    cfg
}

/// Builds a [`HardwareRoofline`] from translator stage options (`arch`, `peak_gflops`, etc.).
pub(crate) fn parse_roofline_opts(opts: &IndexMap<String, String>) -> HardwareRoofline {
    let mut roofline = match opts.get("arch") {
        Some(arch) => HardwareRoofline::for_arch(arch),
        None => HardwareRoofline::default(),
    };
    if let Some(v) = opts.get("peak_gflops")
        && let Ok(g) = v.parse::<f64>()
    {
        roofline.peak_gflops = g.max(1e-3);
    }
    if let Some(v) = opts
        .get("peak_bw_gbps")
        .or_else(|| opts.get("peak_bandwidth_gbps"))
        && let Ok(bw) = v.parse::<f64>()
    {
        roofline.peak_bandwidth_gbps = bw.max(1e-3);
    }
    roofline
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn test_cli_arg_parsing_and_chain_options() {
        let cli = Cli::try_parse_from([
            "castforge",
            "opt",
            "model.mlir",
            "-p",
            "canonicalize",
            "--stats=false",
            "--bisect",
        ])
        .expect("opt flags parse cleanly");
        if let Commands::Opt { stats, bisect, .. } = cli.command {
            assert!(!stats);
            assert!(bisect);
        } else {
            panic!("expected Commands::Opt");
        }

        let cli_cg = Cli::try_parse_from([
            "castforge",
            "codegen",
            "model.mlir",
            "--arch",
            "x86_64-avx512",
            "--tile-size",
            "128",
            "--vectorize=false",
            "--autotune",
        ])
        .expect("codegen flags parse cleanly");
        if let Commands::Codegen {
            arch,
            tile_size,
            vectorize,
            autotune,
            ..
        } = cli_cg.command
        {
            assert_eq!(arch.as_deref(), Some("x86_64-avx512"));
            assert_eq!(tile_size, Some(128));
            assert_eq!(vectorize, Some(false));
            assert!(autotune);
        } else {
            panic!("expected Commands::Codegen");
        }

        let reg = PassRegistry::with_builtin_passes();
        let (chain, first_in, last_out) = build_translator_chain(
            &reg,
            "autotune-schedule(arch=nvidia-hopper,peak_gflops=80000) | \
             codegen-c(arch=x86_64-avx512,tile_size=64,unroll=4)",
        )
        .expect("chain builds");
        assert_eq!(first_in, "text");
        assert_eq!(last_out, "text");
        assert_eq!(chain.describe_stages().len(), 3);

        assert!(build_translator_chain(&reg, "codegen(target=invalid_target)").is_err());
        assert!(build_translator_chain(&reg, "viz(format=invalid_format)").is_err());

        let cli_bench = Cli::try_parse_from([
            "castforge",
            "bench",
            "model.mlir",
            "-t",
            "c",
            "--autotune",
            "--iters",
            "10",
            "--json",
        ])
        .expect("bench parses");
        assert!(matches!(cli_bench.command, Commands::Bench { .. }));

        let cli_diff =
            Cli::try_parse_from(["castforge", "diff", "a.mlir", "b.mlir"]).expect("diff parses");
        assert!(matches!(cli_diff.command, Commands::Diff { .. }));

        let cli_inspect =
            Cli::try_parse_from(["castforge", "inspect", "model.mlir", "--json"]).unwrap();
        assert!(matches!(cli_inspect.command, Commands::Inspect { .. }));

        let cli_targets = Cli::try_parse_from(["castforge", "targets", "--json"]).unwrap();
        assert!(matches!(cli_targets.command, Commands::ListTargets { .. }));

        let mod_ir = crate::mlir::parse_mlir(
            r#"
            module @demo {
              func.func @main(%x: tensor<2x2xf32>) -> tensor<2x2xf32> {
                %0 = stablehlo.add %x, %x : tensor<2x2xf32>
                func.return %0 : tensor<2x2xf32>
              }
            }
            "#,
        )
        .unwrap();
        let rep = crate::cli::inspect_module(&mod_ir);
        assert_eq!(rep.name, "demo");
        assert_eq!(rep.functions.len(), 1);
        assert_eq!(rep.total_ops, 1);
        assert_eq!(rep.op_histogram.get("stablehlo.add"), Some(&1));

        let pcat = crate::ir::pass_catalog();
        assert_eq!(pcat.pass.len(), reg.list().count());
        for (name, desc) in reg.list() {
            let entry = reg
                .catalog_entry(name)
                .unwrap_or_else(|| panic!("missing pass `{name}` in config/passes.toml"));
            assert_eq!(entry.description, desc);
            let instance = reg
                .create_pass(name, &Default::default())
                .unwrap_or_else(|e| panic!("failed to instantiate `{name}`: {e}"));
            let expected_children: Vec<String> =
                instance.child_passes().iter().map(|s| (*s).to_string()).collect();
            assert_eq!(
                entry.child_passes, expected_children,
                "child_passes mismatch for `{name}` in config/passes.toml"
            );
        }

        let tcat = crate::cli::target_catalog();
        assert_eq!(tcat.target.len(), 10);
        assert_eq!(tcat.arch_profile.len(), 11);
        assert_eq!(tcat.translator.len(), 20);
    }
}
