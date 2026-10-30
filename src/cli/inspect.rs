//! Model, IR, weight (`TensorStore`), and hardware target inspection (`castforge inspect` &
//! `castforge list-targets`).

use crate::codegen::{HardwareRoofline, TargetConfig, analyze_module_cost, build_memory_plan};
use crate::ir::{
    Attribute, BlockId, Module, TensorStore, import_gguf_file, import_hf_model_dir,
    import_onnx_file, parse_mlir,
};
use crate::passes::{PassManager, PassRegistry};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Summary of a single `func.func` inside an inspected [`Module`].
#[derive(Debug, Clone, Serialize)]
pub struct FunctionSummary {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub num_ops: usize,
}

/// Summary of a single weight tensor inside a [`TensorStore`].
#[derive(Debug, Clone, Serialize)]
pub struct WeightTensorSummary {
    pub name: String,
    pub dtype: String,
    pub shape: String,
    pub num_elements: u64,
    pub bytes: usize,
}

/// Structured inspection report for a model, MLIR module, or `.safetensors` weight file.
#[derive(Debug, Clone, Serialize)]
pub struct ModuleInspectionReport {
    pub name: String,
    pub functions: Vec<FunctionSummary>,
    pub total_ops: usize,
    pub op_histogram: IndexMap<String, usize>,
    pub total_parameters: u64,
    pub total_weight_bytes: usize,
    pub tensor_store_entries: Vec<WeightTensorSummary>,
    pub peak_activation_bytes: usize,
    pub naive_activation_bytes: usize,
    pub reusable_arena_slots: usize,
    pub total_flops: u64,
}

/// Declarative hardware architecture profile entry from `config/targets.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchProfileSpec {
    pub arch: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub peak_gflops: f64,
    pub peak_bandwidth_gbps: f64,
    pub simd_isa: String,
    pub simd_lanes: usize,
    pub default_tile: usize,
    pub reg_tile: usize,
    pub unroll_factor: usize,
}

/// Hardware architecture profile entry for `castforge list-targets`.
#[derive(Debug, Clone, Serialize)]
pub struct ArchProfileInfo {
    pub arch: String,
    pub aliases: Vec<String>,
    pub peak_gflops: f64,
    pub peak_bandwidth_gbps: f64,
    pub ridge_point_flops_per_byte: f64,
    pub simd_isa: String,
    pub simd_lanes: usize,
    pub default_tile: usize,
    pub reg_tile: usize,
    pub unroll_factor: usize,
}

/// Codegen backend entry parsed from `config/targets.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetBackendInfo {
    pub target: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub output_kind: String,
    pub description: String,
}

/// Translator stage entry parsed from `config/targets.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslatorStageInfo {
    pub stage: String,
    pub signature: String,
    pub description: String,
}

/// Top-level catalog of targets, hardware profiles, and translator stages parsed from
/// `config/targets.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetCatalog {
    #[serde(default)]
    pub target: Vec<TargetBackendInfo>,
    #[serde(default)]
    pub arch_profile: Vec<ArchProfileSpec>,
    #[serde(default)]
    pub translator: Vec<TranslatorStageInfo>,
}

/// Returns the compile-time embedded [`TargetCatalog`] parsed from `config/targets.toml`.
pub fn target_catalog() -> &'static TargetCatalog {
    static CATALOG: OnceLock<TargetCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        toml::from_str(include_str!("../../config/targets.toml"))
            .expect("config/targets.toml must be valid TOML matching TargetCatalog")
    })
}

/// Inspects an input model (`.mlir`, `.gguf`, `.onnx.json`, `config.json`, or `.safetensors`).
pub fn run_inspect(
    input: Option<&PathBuf>,
    weights: Option<&PathBuf>,
    passes: Option<&str>,
    json: bool,
    output: Option<&PathBuf>,
) -> Result<()> {
    if let Some(p) = input
        && p.to_string_lossy()
            .to_ascii_lowercase()
            .ends_with(".safetensors")
    {
        let mut store = TensorStore::new();
        store
            .load_safetensors_file(p)
            .with_context(|| format!("Failed to load {}", p.display()))?;
        let report = inspect_tensor_store_only(&p.display().to_string(), &store);
        return emit_inspection_report(&report, json, output);
    }

    let mut module = load_any_module(input)?;
    if let Some(w_path) = weights {
        module
            .tensor_store
            .load_safetensors_file(w_path)
            .with_context(|| format!("Failed to load weights {}", w_path.display()))?;
    }
    if let Some(spec) = passes {
        let registry = PassRegistry::with_builtin_passes();
        let mut pm = PassManager::from_pipeline_str(&registry, spec)?;
        pm.run(&mut module)?;
    }

    let report = inspect_module(&module);
    emit_inspection_report(&report, json, output)
}

/// Builds a [`ModuleInspectionReport`] from an in-memory [`Module`].
pub fn inspect_module(module: &Module) -> ModuleInspectionReport {
    let name = module
        .name
        .clone()
        .unwrap_or_else(|| "castforge_module".to_string());
    let mut functions = Vec::new();
    let mut op_counts: IndexMap<String, usize> = IndexMap::new();
    let mut total_ops = 0usize;
    let mut inline_params = 0u64;

    for func_id in module.functions() {
        let fn_op = &module.ops[func_id];
        let fn_name = fn_op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("main")
            .to_string();
        let Some(reg_id) = fn_op.regions.first().copied() else {
            continue;
        };
        let Some(blk_id) = module.regions[reg_id].blocks.first().copied() else {
            continue;
        };
        let block = &module.blocks[blk_id];
        let inputs: Vec<String> = block
            .arguments
            .iter()
            .map(|&arg_id| module.values[arg_id].ty.to_string())
            .collect();
        let mut outputs = Vec::new();
        let mut fn_op_count = 0usize;
        for (b_idx, &b_id) in module.regions[reg_id].blocks.iter().enumerate() {
            inspect_block_ops(
                module,
                b_id,
                b_idx == 0,
                &mut fn_op_count,
                &mut total_ops,
                &mut op_counts,
                &mut inline_params,
                &mut outputs,
            );
        }

        functions.push(FunctionSummary {
            name: fn_name,
            inputs,
            outputs,
            num_ops: fn_op_count,
        });
    }

    let mut hist_pairs: Vec<(String, usize)> = op_counts.into_iter().collect();
    hist_pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let op_histogram: IndexMap<String, usize> = hist_pairs.into_iter().collect();

    let (store_entries, store_params, store_bytes) = summarize_tensor_store(&module.tensor_store);
    let mem_plan = build_memory_plan(module);
    let cost = analyze_module_cost(module);

    ModuleInspectionReport {
        name,
        functions,
        total_ops,
        op_histogram,
        total_parameters: inline_params + store_params,
        total_weight_bytes: mem_plan.total_constant_bytes.max(store_bytes),
        tensor_store_entries: store_entries,
        peak_activation_bytes: mem_plan.total_activation_bytes,
        naive_activation_bytes: mem_plan.naive_activation_bytes,
        reusable_arena_slots: mem_plan.reusable_slot_count,
        total_flops: cost.total_flops,
    }
}

#[allow(clippy::too_many_arguments)]
fn inspect_block_ops(
    module: &Module,
    blk_id: BlockId,
    is_entry_block: bool,
    fn_op_count: &mut usize,
    total_ops: &mut usize,
    op_counts: &mut IndexMap<String, usize>,
    inline_params: &mut u64,
    outputs: &mut Vec<String>,
) {
    let block = &module.blocks[blk_id];
    for &op_id in &block.ops {
        let op = &module.ops[op_id];
        if op.is_op("func", "return") {
            if is_entry_block {
                for &v in &op.operands {
                    outputs.push(module.values[v].ty.to_string());
                }
            }
            continue;
        }
        if op.is_op("stablehlo", "return") {
            continue;
        }
        *fn_op_count += 1;
        *total_ops += 1;
        let op_key = op.name.full_name();
        *op_counts.entry(op_key).or_insert(0) += 1;

        if op.is_op("stablehlo", "constant")
            && !matches!(
                op.attributes.get("value"),
                Some(Attribute::TensorRef { .. })
            )
            && let Some(&res_id) = op.results.first()
            && let Some(tt) = module.values[res_id].ty.as_tensor()
        {
            *inline_params += tt.num_elements().unwrap_or(1).max(1) as u64;
        }

        for &sub_reg in &op.regions {
            for &sub_blk in &module.regions[sub_reg].blocks {
                inspect_block_ops(
                    module,
                    sub_blk,
                    false,
                    fn_op_count,
                    total_ops,
                    op_counts,
                    inline_params,
                    outputs,
                );
            }
        }
    }
}

fn inspect_tensor_store_only(name: &str, store: &TensorStore) -> ModuleInspectionReport {
    let (entries, total_parameters, total_weight_bytes) = summarize_tensor_store(store);
    ModuleInspectionReport {
        name: name.to_string(),
        functions: Vec::new(),
        total_ops: 0,
        op_histogram: IndexMap::new(),
        total_parameters,
        total_weight_bytes,
        tensor_store_entries: entries,
        peak_activation_bytes: 0,
        naive_activation_bytes: 0,
        reusable_arena_slots: 0,
        total_flops: 0,
    }
}

fn summarize_tensor_store(store: &TensorStore) -> (Vec<WeightTensorSummary>, u64, usize) {
    let mut entries = Vec::with_capacity(store.len());
    let mut total_params = 0u64;
    let mut total_bytes = 0usize;
    for (_, entry) in store.iter() {
        let elems = entry.ty.num_elements().unwrap_or(1).max(1) as u64;
        let bytes = ((elems * entry.ty.dtype.bit_width() as u64).div_ceil(8)) as usize;
        total_params += elems;
        total_bytes += bytes;
        entries.push(WeightTensorSummary {
            name: entry.name.clone(),
            dtype: entry.ty.dtype.to_string(),
            shape: entry.ty.to_string(),
            num_elements: elems,
            bytes,
        });
    }
    (entries, total_params, total_bytes)
}

fn emit_inspection_report(
    report: &ModuleInspectionReport,
    json: bool,
    output: Option<&PathBuf>,
) -> Result<()> {
    let text = if json {
        format!("{}\n", serde_json::to_string_pretty(report)?)
    } else {
        let mut out = String::new();
        let _ = writeln!(out, "=== CastForge Inspection Report: {} ===", report.name);
        let _ = writeln!(
            out,
            "Functions: {} | Compute Ops: {} | Total Parameters: {} ({} B)",
            report.functions.len(),
            report.total_ops,
            report.total_parameters,
            report.total_weight_bytes
        );
        let _ = writeln!(
            out,
            "Total FLOPs: {} | Peak Arena: {} B (Naive: {} B, Slots: {})",
            report.total_flops,
            report.peak_activation_bytes,
            report.naive_activation_bytes,
            report.reusable_arena_slots
        );
        if !report.functions.is_empty() {
            let _ = writeln!(out, "\nFunctions:");
            for f in &report.functions {
                let _ = writeln!(
                    out,
                    "  @{}({}) -> ({}) [{} ops]",
                    f.name,
                    f.inputs.join(", "),
                    f.outputs.join(", "),
                    f.num_ops
                );
            }
        }
        if !report.op_histogram.is_empty() {
            let _ = writeln!(out, "\nOperation Histogram:");
            for (op, count) in &report.op_histogram {
                let _ = writeln!(out, "  {:<36} {:>6}", op, count);
            }
        }
        if !report.tensor_store_entries.is_empty() {
            let _ = writeln!(
                out,
                "\nTensorStore Weights ({} entries):",
                report.tensor_store_entries.len()
            );
            for t in &report.tensor_store_entries {
                let _ = writeln!(
                    out,
                    "  {:<28} {:<24} {:>10} elems ({:>10} B)",
                    t.name, t.shape, t.num_elements, t.bytes
                );
            }
        }
        out
    };

    if let Some(out_path) = output {
        fs::write(out_path, text)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
    } else {
        print!("{text}");
    }
    Ok(())
}

fn load_any_module(input: Option<&PathBuf>) -> Result<Module> {
    if let Some(path) = input
        && path.as_os_str() != "-"
    {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        if lower.ends_with(".gguf") {
            return import_gguf_file(path);
        } else if path.is_dir() || lower.ends_with("config.json") {
            return import_hf_model_dir(path);
        } else if lower.ends_with(".onnx") || lower.ends_with(".onnx.json") {
            return import_onnx_file(path);
        }
        let src = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        return Ok(parse_mlir(&src)?);
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("Failed to read from stdin")?;
    Ok(parse_mlir(&buf)?)
}

/// Lists all codegen targets, hardware `--arch` profiles, and composable translator stages.
pub fn run_list_targets(filter: Option<&str>, json: bool, output: Option<&PathBuf>) -> Result<()> {
    let catalog = target_catalog();
    let q = filter.unwrap_or("").trim().to_ascii_lowercase();
    let targets: Vec<&TargetBackendInfo> = catalog
        .target
        .iter()
        .filter(|t| {
            q.is_empty()
                || t.target.to_ascii_lowercase().contains(&q)
                || t.aliases
                    .iter()
                    .any(|a| a.to_ascii_lowercase().contains(&q))
                || t.description.to_ascii_lowercase().contains(&q)
        })
        .collect();
    let arch_profiles: Vec<ArchProfileInfo> = catalog
        .arch_profile
        .iter()
        .filter(|spec| {
            q.is_empty()
                || spec.arch.to_ascii_lowercase().contains(&q)
                || spec
                    .aliases
                    .iter()
                    .any(|a| a.to_ascii_lowercase().contains(&q))
                || spec.simd_isa.to_ascii_lowercase().contains(&q)
        })
        .map(|spec| {
            let rf = HardwareRoofline::for_arch(&spec.arch);
            let _ = TargetConfig::for_arch(&spec.arch);
            ArchProfileInfo {
                arch: spec.arch.clone(),
                aliases: spec.aliases.clone(),
                peak_gflops: spec.peak_gflops,
                peak_bandwidth_gbps: spec.peak_bandwidth_gbps,
                ridge_point_flops_per_byte: rf.ridge_point(),
                simd_isa: spec.simd_isa.clone(),
                simd_lanes: spec.simd_lanes,
                default_tile: spec.default_tile,
                reg_tile: spec.reg_tile,
                unroll_factor: spec.unroll_factor,
            }
        })
        .collect();
    let translators: Vec<&TranslatorStageInfo> = catalog
        .translator
        .iter()
        .filter(|tr| {
            q.is_empty()
                || tr.stage.to_ascii_lowercase().contains(&q)
                || tr.description.to_ascii_lowercase().contains(&q)
        })
        .collect();

    let text = if json {
        let payload = serde_json::json!({
            "targets": targets,
            "arch_profiles": arch_profiles,
            "translators": translators,
        });
        format!("{}\n", serde_json::to_string_pretty(&payload)?)
    } else {
        let mut out = String::new();
        let _ = writeln!(out, "=== Codegen Targets (-t / --target) ===");
        let _ = writeln!(
            out,
            "{:<12} {:<18} {:<18} DESCRIPTION",
            "TARGET", "ALIASES", "OUTPUT"
        );
        let _ = writeln!(out, "{}", "-".repeat(92));
        for t in targets {
            let _ = writeln!(
                out,
                "{:<12} {:<18} {:<18} {}",
                t.target,
                t.aliases.join(", "),
                t.output_kind,
                t.description
            );
        }

        let _ = writeln!(out, "\n=== Hardware Architecture Profiles (--arch) ===");
        let _ = writeln!(
            out,
            "{:<18} {:>12} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6}",
            "ARCH", "PEAK GFLOP/S", "BW (GB/S)", "RIDGE F/B", "SIMD ISA", "LANES", "TILE", "UNROLL"
        );
        let _ = writeln!(out, "{}", "-".repeat(92));
        for a in &arch_profiles {
            let _ = writeln!(
                out,
                "{:<18} {:>12.1} {:>10.1} {:>10.2} {:>10} {:>6} {:>6} {:>6}",
                a.arch,
                a.peak_gflops,
                a.peak_bandwidth_gbps,
                a.ridge_point_flops_per_byte,
                a.simd_isa,
                a.simd_lanes,
                a.default_tile,
                a.unroll_factor
            );
        }

        let _ = writeln!(
            out,
            "\n=== Translator Chain Stages (castforge chain -s) ==="
        );
        let _ = writeln!(out, "{:<24} {:<14} DESCRIPTION", "STAGE", "SIGNATURE");
        let _ = writeln!(out, "{}", "-".repeat(92));
        for tr in translators {
            let _ = writeln!(
                out,
                "{:<24} {:<14} {}",
                tr.stage, tr.signature, tr.description
            );
        }
        out
    };

    if let Some(out_path) = output {
        fs::write(out_path, text)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
    } else {
        print!("{text}");
    }
    Ok(())
}
