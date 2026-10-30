//! Builder for composable A-to-B [`TranslatorChain`] pipelines (`castforge chain` / `pipe`).

use crate::codegen::{
    AutotuneScheduleTranslator, CHeaderTranslator, CodegenTarget, CodegenTranslator, GemmLoopOrder,
    HardwareRoofline, MemoryPlanTranslator, OptimizeLoopIrPass, OptimizeLoopIrTranslator,
    RooflineReportTranslator, SharedLibTranslator, SimdIsa, TargetConfig, TensorToLoopTranslator,
};
use crate::ir::{HfConfigImportTranslator, TranslatorChain};
use crate::mlir::{
    GgufExportTranslator, GgufImportTranslator, MlirParseTranslator, MlirPrintTranslator,
    OnnxExportTranslator, OnnxImportTranslator, PrintOptions, SafetensorsExportTranslator,
};
use crate::passes::{PassRegistry, parse_pass_spec_item, split_pipeline_spec};
use crate::viz::{IrDiffTranslator, SnapshotIrTranslator, VizFormat, VizTranslator};
use anyhow::Result;
use indexmap::IndexMap;

/// Parses a `|`- or `,`-delimited translator chain specification into a validated
/// [`TranslatorChain`], returning `(chain, initial_input_kind, final_output_kind)`.
pub fn build_translator_chain(
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
        "parse-mlir" | "parse" | "import-onnx" | "import-gguf" | "import-hf-config"
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
            "import-hf-config" => {
                if idx == 0 {
                    first_in = "text";
                }
                chain.push(Box::new(HfConfigImportTranslator));
                last_out = "ir";
            }
            "import-gguf" => {
                if idx == 0 {
                    first_in = "bytes";
                }
                chain.push(Box::new(GgufImportTranslator));
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
                chain.push(Box::new(MlirPrintTranslator {
                    options: PrintOptions {
                        generic_form: generic,
                        print_locations: false,
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
                if let Some(v) = opts.get("fuse").or_else(|| opts.get("fuse_loops")) {
                    pass.fuse_loops = v == "true" || v == "1";
                }
                if let Some(v) = opts.get("vectorize") {
                    pass.vectorize = v == "true" || v == "1";
                }
                if let Some(v) = opts.get("unroll").or_else(|| opts.get("unroll_factor"))
                    && let Ok(n) = v.parse::<usize>()
                {
                    pass.unroll_factor = n.max(1);
                }
                if let Some(v) = opts.get("parallel") {
                    pass.parallel = v == "true" || v == "1";
                }
                if let Some(v) = opts.get("interchange") {
                    pass.interchange = v == "true" || v == "1";
                }
                chain.push(Box::new(OptimizeLoopIrTranslator { config, pass }));
                last_out = "text";
            }
            "codegen-header" => {
                chain.push(Box::new(CHeaderTranslator));
                last_out = "text";
            }
            "compile-lib" => {
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
                let tgt = CodegenTarget::parse(tgt_str);
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
