//! Unified [`Pass`], [`PassRegistry`], [`PassManager`], and **21 built-in A-to-B IR translator
//! passes & presets** (`18` passes + `3` presets `O1`/`O2`/`O3` in `castforge::passes`), organized
//! with 100% 1-to-1 pass-to-file mapping directly under `src/passes/`.

pub mod canonicalize;
pub mod cse;
pub mod dce;
pub mod fold_views;
pub mod fold_weights;
pub mod fuse_attention;
pub mod fuse_elementwise;
pub mod fuse_gemm;
pub mod fuse_norm;
pub mod ir_edit;
pub mod merge_weights;
pub mod optimize_memory;
pub mod quantize;
pub mod schedule_loops;
pub mod shard_distributed;
pub mod simplify_algebraic;
pub mod specialize_shapes;
pub mod verify_ir;

pub use canonicalize::*;
pub use cse::*;
pub use dce::*;
pub use fold_views::*;
pub use fold_weights::*;
pub use fuse_attention::*;
pub use fuse_elementwise::*;
pub use fuse_gemm::*;
pub use fuse_norm::*;
pub use ir_edit::*;
pub use merge_weights::*;
pub use optimize_memory::*;
pub use quantize::*;
pub use schedule_loops::*;
pub use shard_distributed::*;
pub use simplify_algebraic::*;
pub use specialize_shapes::*;
pub use verify_ir::*;

use crate::ir::Module;
use anyhow::{Result, anyhow};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Statistics and change status returned by a single pass execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PassResult {
    pub pass_name: String,
    pub changed: bool,
    pub ops_removed: usize,
    pub ops_modified: usize,
    pub ops_created: usize,
}

impl PassResult {
    /// Returns an unchanged [`PassResult`] for `pass_name`.
    pub fn unchanged(pass_name: &str) -> Self {
        Self {
            pass_name: pass_name.to_string(),
            changed: false,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 0,
        }
    }
}

/// Snapshot of the IR before/after a pass for step-by-step visual debugging.
#[derive(Debug, Clone)]
pub struct PassTraceStep {
    pub step_index: usize,
    pub label: String,
    pub result: Option<PassResult>,
    pub module: Module,
}

/// Key-value options passed to a Pass via CLI (`pass(k=v)`) or config file (`TOML`/`JSON`).
pub type PassOptions = IndexMap<String, String>;

/// Core trait implemented by all IR transformation, optimization, and surgery passes.
pub trait Pass: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn run(&mut self, module: &mut Module) -> Result<PassResult>;
}

impl crate::ir::Translator for Box<dyn Pass> {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let mut module = input.into_ir()?;
        (**self).run(&mut module)?;
        Ok(crate::ir::Payload::from_ir(module))
    }
}

/// Adapter wrapping any concrete [`Pass`] as a [`crate::ir::Translator`].
pub struct PassTranslator<P: Pass>(pub P);

impl<P: Pass> crate::ir::Translator for PassTranslator<P> {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let mut module = input.into_ir()?;
        self.0.run(&mut module)?;
        Ok(crate::ir::Payload::from_ir(module))
    }
}

/// Declarative configuration file format (`pipeline.toml` or `.json`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub settings: PipelineSettings,
    #[serde(default)]
    pub pipeline: Vec<PassStepConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSettings {
    #[serde(default = "default_true")]
    pub verify_after_each_pass: bool,
    #[serde(default = "default_one")]
    pub max_iterations: usize,
}

fn default_true() -> bool {
    true
}
fn default_one() -> usize {
    1
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            verify_after_each_pass: true,
            max_iterations: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassStepConfig {
    pub pass: String,
    #[serde(default)]
    pub options: IndexMap<String, String>,
}

/// Orchestrates static, CLI-driven, or config-file-driven pass execution.
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    pub verify_after_each_pass: bool,
    pub max_iterations: usize,
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            verify_after_each_pass: true,
            max_iterations: 1,
        }
    }

    /// Add a pass statically in Rust code.
    pub fn add_pass<P: Pass + 'static>(mut self, pass: P) -> Self {
        self.passes.push(Box::new(pass));
        self
    }

    /// Add a boxed pass dynamically.
    pub fn push_boxed(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Converts this [`PassManager`] into a [`crate::ir::TranslatorChain`] of `ir -> ir` stages.
    pub fn into_translator_chain(self) -> crate::ir::TranslatorChain {
        let mut chain = crate::ir::TranslatorChain::new();
        for pass in self.passes {
            chain.push(Box::new(pass));
        }
        chain
    }

    /// Build a `PassManager` from a CLI pipeline specification string, e.g.:
    /// `"fold-views,cse,dce,fuse-elementwise"`
    pub fn from_pipeline_str(registry: &PassRegistry, spec: &str) -> Result<Self> {
        let mut pm = Self::new();
        for item in split_pipeline_spec(spec) {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let (name, opts) = parse_pass_spec_item(item)?;
            let pass = registry.create_pass(&name, &opts)?;
            pm.push_boxed(pass);
        }
        Ok(pm)
    }

    /// Build a `PassManager` from a `PipelineConfig` (loaded from TOML or JSON).
    pub fn from_config(registry: &PassRegistry, config: &PipelineConfig) -> Result<Self> {
        let mut pm = Self::new();
        pm.verify_after_each_pass = config.settings.verify_after_each_pass;
        pm.max_iterations = config.settings.max_iterations.max(1);
        for step in &config.pipeline {
            let pass = registry.create_pass(&step.pass, &step.options)?;
            pm.push_boxed(pass);
        }
        Ok(pm)
    }

    /// Parse a TOML configuration string into a `PassManager`.
    pub fn from_toml_str(registry: &PassRegistry, toml_str: &str) -> Result<Self> {
        let cfg: PipelineConfig = toml::from_str(toml_str)?;
        Self::from_config(registry, &cfg)
    }

    /// Execute the configured pipeline on `module` and return execution logs per pass.
    pub fn run(&mut self, module: &mut Module) -> Result<Vec<PassResult>> {
        let mut history = Vec::new();
        for _iter in 0..self.max_iterations {
            let mut any_changed = false;
            for pass in &mut self.passes {
                let res = pass.run(module)?;
                if self.verify_after_each_pass {
                    module.verify()?;
                }
                any_changed |= res.changed;
                history.push(res);
            }
            if !any_changed {
                break;
            }
        }
        Ok(history)
    }

    /// Execute the pipeline and capture a full step-by-step IR trace
    /// for interactive visualization and pass diffing.
    pub fn run_with_trace(&mut self, module: &mut Module) -> Result<Vec<PassTraceStep>> {
        let mut steps = Vec::new();
        steps.push(PassTraceStep {
            step_index: 0,
            label: "Initial IR".to_string(),
            result: None,
            module: module.clone(),
        });

        let mut step_idx = 1usize;
        for _iter in 0..self.max_iterations {
            let mut any_changed = false;
            for pass in &mut self.passes {
                let res = pass.run(module)?;
                if self.verify_after_each_pass {
                    module.verify()?;
                }
                any_changed |= res.changed;
                steps.push(PassTraceStep {
                    step_index: step_idx,
                    label: res.pass_name.clone(),
                    result: Some(res),
                    module: module.clone(),
                });
                step_idx += 1;
            }
            if !any_changed {
                break;
            }
        }
        Ok(steps)
    }
}

/// Splits a pipeline or translator-chain specification string on top-level `,` or `|` separators.
pub fn split_pipeline_spec(spec: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (i, c) in spec.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' | '|' if depth == 0 => {
                out.push(&spec[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&spec[start..]);
    out
}

/// Parses a single `name(k1=v1,k2=v2)` stage specification into `(name, options)`.
pub fn parse_pass_spec_item(item: &str) -> Result<(String, PassOptions)> {
    let mut opts = PassOptions::new();
    if let Some(open) = item.find('(') {
        let close = item
            .rfind(')')
            .ok_or_else(|| anyhow!("Unclosed `(` in pass specification `{item}`"))?;
        let name = item[..open].trim().to_string();
        let inner = &item[open + 1..close];
        for kv in inner.split(',') {
            let kv = kv.trim();
            if kv.is_empty() {
                continue;
            }
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| anyhow!("Expected `key=value` option in `{item}`"))?;
            opts.insert(k.trim().to_string(), v.trim().to_string());
        }
        Ok((name, opts))
    } else {
        Ok((item.trim().to_string(), opts))
    }
}

pub type PassFactory = Box<dyn Fn(&PassOptions) -> Result<Box<dyn Pass>> + Send + Sync>;

pub struct PassDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub factory: PassFactory,
}

/// Registry holding all available passes for dynamic lookup by CLI, command name, or config file.
pub struct PassRegistry {
    passes: IndexMap<&'static str, PassDescriptor>,
}

impl Default for PassRegistry {
    fn default() -> Self {
        Self::with_builtin_passes()
    }
}

impl PassRegistry {
    pub fn new() -> Self {
        Self {
            passes: IndexMap::new(),
        }
    }

    /// Create a registry pre-populated with all 21 built-in CastForge passes (`18` passes + `3`
    /// presets `O1`/`O2`/`O3`).
    pub fn with_builtin_passes() -> Self {
        let mut reg = Self::new();
        reg.register(
            "dce",
            "Dead Code Elimination (removes unused pure SSA ops and dead private symbols)",
            |opts| {
                let include_symbols = opts.get("symbols").map(|v| v != "false").unwrap_or(true);
                Ok(Box::new(DeadCodeEliminationPass { include_symbols }))
            },
        );
        reg.register(
            "cse",
            "Unified CSE, dominator-aware GVN, LICM, and entry constant hoisting",
            |opts| {
                let gvn = opts.get("gvn").map(|v| v != "false").unwrap_or(true);
                let hoist_constants = opts
                    .get("hoist_constants")
                    .or_else(|| opts.get("hoist"))
                    .is_some_and(|v| v == "true" || v == "1");
                let licm = opts
                    .get("licm")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                Ok(Box::new(UnifiedCsePass {
                    gvn,
                    hoist_constants,
                    licm,
                }))
            },
        );
        reg.register(
            "fold-views",
            "Unified folding of transpose, reshape, broadcast, slice/concat, convert, and layout",
            |opts| {
                let layout = opts.get("layout").or_else(|| opts.get("target")).cloned();
                let propagate = opts
                    .get("propagate")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(FoldViewsPass { layout, propagate }))
            },
        );
        reg.register(
            "simplify-algebraic",
            "Unified constant folding, SCCP, algebraic identities, e-graph factoring & FMA",
            |opts| {
                let fold_constants = opts
                    .get("fold_constants")
                    .or_else(|| opts.get("constants"))
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                let include_conditional = opts
                    .get("conditional")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(true);
                let const_only = opts
                    .get("mode")
                    .is_some_and(|m| m == "constants" || m == "fold");
                let egraph = opts
                    .get("egraph")
                    .map(|v| v != "false" && v != "0")
                    .unwrap_or(!const_only);
                let fuse_fma = opts
                    .get("fuse_fma")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!const_only);
                Ok(Box::new(UnifiedSimplifyAlgebraicPass {
                    fold_constants,
                    include_conditional,
                    egraph,
                    fuse_fma,
                }))
            },
        );
        reg.register(
            "quantize",
            "Unified quantization & precision (weight|awq|fp8|nvfp4|f16|bf16|w8a8|smooth|hadamard)",
            |opts| {
                let scheme = opts
                    .get("scheme")
                    .or_else(|| opts.get("mode"))
                    .or_else(|| opts.get("dtype"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts.contains_key("alpha") {
                            "smoothquant".to_string()
                        } else if opts.contains_key("salient_ratio") {
                            "awq".to_string()
                        } else if opts.contains_key("group_size")
                            || opts.contains_key("awq_salient")
                        {
                            "groupwise".to_string()
                        } else if opts.contains_key("format") || opts.contains_key("mxfp8") {
                            "fp8".to_string()
                        } else if opts.contains_key("samples") || opts.contains_key("num_samples") {
                            "w8a8".to_string()
                        } else if opts.contains_key("block_size") && !opts.contains_key("bits") {
                            "hadamard".to_string()
                        } else {
                            "weight".to_string()
                        }
                    });
                let default_bits = if matches!(
                    scheme.as_str(),
                    "groupwise" | "awq" | "gptq" | "fp4" | "mxfp4" | "nvfp4"
                ) {
                    4
                } else {
                    8
                };
                let bits = opts
                    .get("bits")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(default_bits);
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let group_size = opts
                    .get("group_size")
                    .or_else(|| opts.get("group"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32);
                let awq_salient = opts
                    .get("awq_salient")
                    .map(|s| s != "false")
                    .unwrap_or(true);
                let salient_ratio = opts
                    .get("salient_ratio")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.01);
                let format = opts
                    .get("format")
                    .or_else(|| opts.get("dtype"))
                    .cloned()
                    .unwrap_or_else(|| "e4m3".to_string());
                let mxfp8 = opts.get("mxfp8").map(|s| s != "false").unwrap_or(true);
                let block_size = opts
                    .get("block_size")
                    .or_else(|| opts.get("size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32);
                let samples = opts
                    .get("samples")
                    .or_else(|| opts.get("num_samples"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let alpha = opts
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                let hadamard = opts
                    .get("hadamard")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(QuantizePass {
                    scheme,
                    bits,
                    min_elements,
                    group_size,
                    awq_salient,
                    salient_ratio,
                    format,
                    mxfp8,
                    block_size,
                    samples,
                    alpha,
                    hadamard,
                }))
            },
        );
        reg.register(
            "merge-weights",
            "Unified model merging & adapter baking (method=linear|slerp|ties|dare|lora)",
            |opts| {
                let method = opts.get("method").cloned().unwrap_or_else(|| {
                    if opts.contains_key("alpha") {
                        "lora".to_string()
                    } else if opts.contains_key("lambda") || opts.contains_key("drop_rate") {
                        "dare".to_string()
                    } else {
                        "linear".to_string()
                    }
                });
                let t = opts
                    .get("t")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                let alpha = opts
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let lambda = opts
                    .get("lambda")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let drop_rate = opts
                    .get("drop_rate")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                Ok(Box::new(UnifiedMergeWeightsPass {
                    method,
                    t,
                    alpha,
                    lambda,
                    drop_rate,
                }))
            },
        );
        reg.register(
            "fold-weights",
            "Unified weight folding, SIMD pre-packing, SafeTensors externalization & pruning",
            |opts| {
                let inline = opts.get("inline").is_some_and(|v| v == "true" || v == "1");
                let mode = opts
                    .get("mode")
                    .or_else(|| opts.get("kind"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if inline {
                            "inline".to_string()
                        } else if opts.contains_key("sparsity") {
                            "structured".to_string()
                        } else if opts.contains_key("min_elements") {
                            "externalize".to_string()
                        } else {
                            "batchnorm".to_string()
                        }
                    });
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let sparsity = opts
                    .get("sparsity")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(FoldWeightsPass {
                    mode,
                    min_elements,
                    inline,
                    sparsity,
                }))
            },
        );
        reg.register(
            "fuse-elementwise",
            "Unified elementwise, activation (SwiGLU/GELU/GeGLU/MoE), map-reduce & loop fusion",
            |opts| {
                let shared_expert = opts
                    .get("shared_expert")
                    .is_some_and(|s| s == "true" || s == "1");
                let mode = opts
                    .get("mode")
                    .or_else(|| opts.get("kind"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts
                            .get("decompose")
                            .is_some_and(|v| v == "true" || v == "1")
                        {
                            "decompose".to_string()
                        } else if shared_expert
                            || opts.contains_key("top_k")
                            || opts.contains_key("capacity_factor")
                        {
                            "moe".to_string()
                        } else if opts.contains_key("max_steps") {
                            "loops".to_string()
                        } else {
                            "composites".to_string()
                        }
                    });
                let max_steps = opts
                    .get("max_steps")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let top_k = opts
                    .get("top_k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let capacity_factor = opts
                    .get("capacity_factor")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.25);
                Ok(Box::new(UnifiedFuseElementwisePass {
                    mode,
                    max_steps,
                    shared_expert,
                    top_k,
                    capacity_factor,
                }))
            },
        );
        reg.register(
            "fuse-gemm",
            "Unified GEMM fusion (horizontal QKV/MLP dot, dequant+GEMM, and GEMM+bias/act/res)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .cloned()
                    .unwrap_or_else(|| "all".to_string());
                Ok(Box::new(FuseGemmPass { kind }))
            },
        );
        reg.register(
            "fuse-norm",
            "Unified normalization & softmax fusion (RMSNorm, LayerNorm, Softmax, Cross-Entropy)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .or_else(|| opts.get("mode"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts.contains_key("chunk_size") || opts.contains_key("chunk") {
                            "cross_entropy".to_string()
                        } else {
                            "all".to_string()
                        }
                    });
                let chunk_size = opts
                    .get("chunk_size")
                    .or_else(|| opts.get("chunk"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(1024);
                Ok(Box::new(FuseNormPass { kind, chunk_size }))
            },
        );
        reg.register(
            "fuse-attention",
            "Unified attention fusion & serving (SDPA/RoPE/MLA/KV-cache/Paged/Flash/Sliding/Share)",
            |opts| {
                let kind = opts
                    .get("kind")
                    .or_else(|| opts.get("mode"))
                    .cloned()
                    .unwrap_or_else(|| {
                        if opts.contains_key("kv_lora_rank")
                            || opts.contains_key("latent_rank")
                            || opts.contains_key("rank")
                        {
                            "mla".to_string()
                        } else if opts.contains_key("share_every") {
                            "kv_sharing".to_string()
                        } else if opts.contains_key("window_size")
                            || opts.contains_key("sink_tokens")
                        {
                            "sliding_window".to_string()
                        } else if opts.contains_key("page_block_size")
                            || opts.contains_key("block_size")
                            || opts.contains_key("max_blocks")
                        {
                            "paged".to_string()
                        } else if opts.contains_key("split_kv") || opts.contains_key("splits") {
                            "flash_decoding".to_string()
                        } else if opts.contains_key("draft_tokens")
                            || opts.contains_key("tokens")
                            || opts.contains_key("tree_depth")
                            || opts.contains_key("depth")
                        {
                            "tree".to_string()
                        } else if opts.contains_key("max_seq_len")
                            || opts.contains_key("seq_len")
                            || opts.contains_key("kv_dtype")
                        {
                            "kv_cache".to_string()
                        } else if opts.contains_key("rope_scaling")
                            || opts.contains_key("rope_factor")
                        {
                            "rope".to_string()
                        } else if opts.contains_key("state_dim") {
                            "ssm".to_string()
                        } else {
                            "all".to_string()
                        }
                    });
                let kv_lora_rank = opts
                    .get("kv_lora_rank")
                    .or_else(|| opts.get("latent_rank"))
                    .or_else(|| opts.get("rank"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(512);
                let max_seq_len = opts
                    .get("max_seq_len")
                    .or_else(|| opts.get("seq_len"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2048);
                let page_block_size = opts
                    .get("page_block_size")
                    .or_else(|| opts.get("block_size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let max_blocks = opts
                    .get("max_blocks")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(64);
                let split_kv = opts
                    .get("split_kv")
                    .or_else(|| opts.get("splits"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let draft_tokens = opts
                    .get("draft_tokens")
                    .or_else(|| opts.get("tokens"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let tree_depth = opts
                    .get("tree_depth")
                    .or_else(|| opts.get("depth"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let window_size = opts
                    .get("window_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4096);
                let sink_tokens = opts
                    .get("sink_tokens")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let share_every = opts
                    .get("share_every")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let rope_scaling = opts.get("rope_scaling").cloned();
                let rope_factor = opts
                    .get("rope_factor")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);
                let logit_softcap = opts
                    .get("logit_softcap")
                    .or_else(|| opts.get("softcap"))
                    .and_then(|s| s.parse::<f64>().ok());
                let qk_norm = opts.get("qk_norm").is_some_and(|v| v == "true" || v == "1");
                let kv_dtype = opts.get("kv_dtype").cloned();
                let state_dim = opts
                    .get("state_dim")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(128);
                let ssm_chunk_size = opts
                    .get("chunk_size")
                    .or_else(|| opts.get("ssm_chunk_size"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(64);
                Ok(Box::new(UnifiedFuseAttentionPass {
                    kind,
                    kv_lora_rank,
                    max_seq_len,
                    page_block_size,
                    max_blocks,
                    split_kv,
                    draft_tokens,
                    tree_depth,
                    window_size,
                    sink_tokens,
                    share_every,
                    rope_scaling,
                    rope_factor,
                    logit_softcap,
                    qk_norm,
                    kv_dtype,
                    state_dim,
                    ssm_chunk_size,
                }))
            },
        );
        reg.register(
            "canonicalize",
            "Composite canonicalization & topological/peak-memory scheduler (mode=all|min_memory)",
            |opts| {
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts
                        .get("min_memory")
                        .is_some_and(|v| v == "true" || v == "1")
                    {
                        "min_memory".to_string()
                    } else {
                        "all".to_string()
                    }
                });
                let toposort = opts
                    .get("toposort")
                    .is_some_and(|v| v == "true" || v == "1");
                Ok(Box::new(CanonicalizePass { mode, toposort }))
            },
        );
        reg.register(
            "specialize-shapes",
            "Unified shape specialization, static batch resizing, bucketing, and token pruning",
            |opts| {
                let mut default_dim = 1i64;
                let mut bindings = IndexMap::new();
                let batch = opts
                    .get("batch")
                    .or_else(|| opts.get("size"))
                    .and_then(|s| s.parse::<i64>().ok());
                let buckets = opts.get("buckets").cloned();
                let token_prune_ratio = opts
                    .get("token_prune_ratio")
                    .or_else(|| opts.get("prune_ratio"))
                    .and_then(|s| s.parse::<f32>().ok());
                for (k, v) in opts {
                    if matches!(
                        k.as_str(),
                        "batch" | "size" | "buckets" | "token_prune_ratio" | "prune_ratio"
                    ) {
                        continue;
                    }
                    if let Ok(n) = v.parse::<i64>() {
                        if k == "default" || k == "?" {
                            default_dim = n;
                        } else {
                            bindings.insert(k.clone(), n);
                        }
                    }
                }
                Ok(Box::new(UnifiedSpecializeShapesPass {
                    default_dim,
                    bindings,
                    batch,
                    buckets,
                    token_prune_ratio,
                }))
            },
        );
        reg.register(
            "optimize-memory",
            "Unified memory pass (mode=all|inplace|zero_copy|remat|offload, min_distance)",
            |opts| {
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts.contains_key("min_distance") {
                        "remat".to_string()
                    } else if opts.contains_key("min_span") {
                        "offload".to_string()
                    } else {
                        "all".to_string()
                    }
                });
                let min_distance = opts
                    .get("min_distance")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(3);
                let min_span = opts
                    .get("min_span")
                    .or_else(|| opts.get("min_distance"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(OptimizeMemoryPass {
                    mode,
                    min_distance,
                    min_span,
                }))
            },
        );
        reg.register(
            "shard-distributed",
            "Unified distributed parallelism (TP, Ring-CP, PP, FSDP, EP, streams, & reverse VJP)",
            |opts| {
                let learning_rate = opts
                    .get("lr")
                    .or_else(|| opts.get("learning_rate"))
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let mode = opts.get("mode").cloned().unwrap_or_else(|| {
                    if opts.contains_key("lr") || opts.contains_key("learning_rate") {
                        "autodiff".to_string()
                    } else if opts.contains_key("ring_size") {
                        "ring_cp".to_string()
                    } else if opts.contains_key("stages")
                        || opts.contains_key("microbatches")
                        || opts.contains_key("schedule")
                    {
                        "pp".to_string()
                    } else if opts.contains_key("streams") || opts.contains_key("num_streams") {
                        "streams".to_string()
                    } else {
                        "tp".to_string()
                    }
                });
                let devices = opts
                    .get("devices")
                    .or_else(|| opts.get("num_devices"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let ring_size = opts
                    .get("ring_size")
                    .or_else(|| opts.get("devices"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let coalesce = opts
                    .get("coalesce")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!matches!(mode.as_str(), "autodiff" | "vjp" | "sgd"));
                let stages = opts
                    .get("stages")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let microbatches = opts
                    .get("microbatches")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let schedule = opts
                    .get("schedule")
                    .cloned()
                    .unwrap_or_else(|| "1f1b".to_string());
                let streams = opts
                    .get("streams")
                    .or_else(|| opts.get("num_streams"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(ShardDistributedPass {
                    mode,
                    devices,
                    ring_size,
                    coalesce,
                    stages,
                    microbatches,
                    schedule,
                    streams,
                    learning_rate,
                }))
            },
        );
        reg.register(
            "schedule-loops",
            "Unified loop schedule, conv-to-gemm & tensor-core pass (tile, permute, unroll, max_n)",
            |opts| {
                let tile_size = opts
                    .get("size")
                    .or_else(|| opts.get("tile_size"))
                    .or_else(|| opts.get("tile").filter(|t| !t.contains('x')))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(16);
                let l2_tile = opts
                    .get("l2_size")
                    .or_else(|| opts.get("l2_tile"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let reg_tile = opts
                    .get("reg_tile")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let order = opts
                    .get("order")
                    .cloned()
                    .unwrap_or_else(|| "mkn".to_string());
                let collapse = opts
                    .get("collapse")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let skew = opts
                    .get("skew")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let unroll_factor = opts
                    .get("factor")
                    .or_else(|| opts.get("unroll"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(4);
                let prefetch = opts
                    .get("prefetch")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let split_k = opts
                    .get("split_k")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1);
                let double_buffer = opts
                    .get("double_buffer")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let isa = opts
                    .get("isa")
                    .cloned()
                    .unwrap_or_else(|| "avx2".to_string());
                let default_w = match isa.to_ascii_lowercase().as_str() {
                    "avx512" | "avx-512" | "avx512f" => 16,
                    "neon" | "aarch64" | "arm64" | "wasm" | "wasm32" | "simd128" => 4,
                    _ => 8,
                };
                let vector_width = opts
                    .get("width")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(default_w);
                let parallel = opts
                    .get("parallel")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(true);
                let align = opts
                    .get("align")
                    .or_else(|| opts.get("multiple"))
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(8);
                let tensor_cores = opts
                    .get("tensor_cores")
                    .or_else(|| opts.get("arch"))
                    .map(|s| s.to_string())
                    .filter(|s| s != "none" && !s.is_empty());
                let mma_tile = opts
                    .get("mma_tile")
                    .or_else(|| opts.get("tile").filter(|t| t.contains('x')))
                    .map(|t| {
                        let parts: Vec<i64> =
                            t.split('x').filter_map(|p| p.parse::<i64>().ok()).collect();
                        if parts.len() == 3 {
                            (parts[0], parts[1], parts[2])
                        } else {
                            (16, 16, 16)
                        }
                    })
                    .unwrap_or((16, 16, 16));
                let max_n = opts
                    .get("max_n")
                    .or_else(|| opts.get("limit"))
                    .and_then(|s| s.parse::<i64>().ok());
                let lower_conv = opts
                    .get("lower_conv")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let warp_specialized = opts
                    .get("warp_specialized")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let tmem_columns = opts
                    .get("tmem_columns")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(512);
                Ok(Box::new(ScheduleLoopsPass {
                    tile_size,
                    l2_tile,
                    reg_tile,
                    order,
                    collapse,
                    skew,
                    unroll_factor,
                    prefetch,
                    split_k,
                    double_buffer,
                    isa,
                    vector_width,
                    parallel,
                    align,
                    tensor_cores,
                    mma_tile,
                    max_n,
                    lower_conv,
                    warp_specialized,
                    tmem_columns,
                }))
            },
        );
        reg.register(
            "ir-edit",
            "Unified IR surgery, subgraph outlining/inlining/extraction, freezing, tapping & ssa",
            |opts| {
                let target = opts
                    .get("target")
                    .or_else(|| opts.get("output"))
                    .cloned()
                    .unwrap_or_default();
                let stop_at = opts
                    .get("stop_at")
                    .or_else(|| opts.get("inputs"))
                    .cloned()
                    .unwrap_or_default();
                let max_ops = opts.get("max_ops").and_then(|s| s.parse::<usize>().ok());
                let keep_op = opts.get("keep_op").cloned().unwrap_or_default();
                let mode = opts.get("mode").cloned().or_else(|| {
                    if opts.get("inline").is_some_and(|v| v == "true" || v == "1") {
                        Some("inline".to_string())
                    } else if opts.get("outline").is_some_and(|v| v == "true" || v == "1") {
                        Some("outline".to_string())
                    } else if max_ops.is_some() || !keep_op.is_empty() {
                        Some("reduce".to_string())
                    } else if !target.is_empty() || !stop_at.is_empty() {
                        Some("extract".to_string())
                    } else {
                        None
                    }
                });
                let dedup = opts
                    .get("dedup")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or_else(|| {
                        matches!(
                            mode.as_deref(),
                            Some("outline" | "all" | "gemm" | "composites" | "dedup")
                        )
                    });
                let first = opts.get("first").cloned().unwrap_or_default();
                let second = opts.get("second").cloned().unwrap_or_default();
                let entry_name = opts
                    .get("entry")
                    .or_else(|| opts.get("name"))
                    .cloned()
                    .unwrap_or_else(|| "composed_main".to_string());
                let repeat = opts
                    .get("repeat")
                    .or_else(|| opts.get("count"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let rename_op = opts.get("rename_op").cloned();
                let annotate_op = opts
                    .get("annotate_op")
                    .or_else(|| opts.get("set_attr"))
                    .cloned();
                let unfreeze = opts
                    .get("unfreeze")
                    .or_else(|| opts.get("parameterize"))
                    .is_some_and(|v| v == "true" || v == "1")
                    || opts.contains_key("min_elements");
                let freeze = opts.get("freeze").is_some_and(|v| v == "true" || v == "1")
                    || opts.contains_key("args")
                    || opts.contains_key("value");
                let args = opts.get("args").cloned().unwrap_or_default();
                let value = opts
                    .get("value")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.5);
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let tap = opts
                    .get("tap")
                    .or_else(|| opts.get("values"))
                    .or_else(|| opts.get("targets"))
                    .cloned();
                let ssa = opts
                    .get("ssa")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or_else(|| opts.contains_key("prefix"));
                let prefix = opts
                    .get("prefix")
                    .cloned()
                    .unwrap_or_else(|| "v".to_string());
                let debuginfo = opts
                    .get("debuginfo")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                let attributes = opts
                    .get("attributes")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);
                Ok(Box::new(IrEditPass {
                    mode,
                    dedup,
                    first,
                    second,
                    entry_name,
                    repeat,
                    target,
                    stop_at,
                    max_ops,
                    keep_op,
                    rename_op,
                    annotate_op,
                    freeze,
                    unfreeze,
                    args,
                    value,
                    min_elements,
                    tap,
                    ssa,
                    prefix,
                    debuginfo,
                    attributes,
                }))
            },
        );
        reg.register(
            "verify-ir",
            "Unified IR verifier, structural hasher, equivalence, sanitizer, bisection & profiling",
            |opts| {
                let sanitize = opts
                    .get("sanitize")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or_else(|| opts.contains_key("clamp_max") || opts.contains_key("eps"));
                let clamp_max = opts
                    .get("clamp_max")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(65504.0);
                let eps = opts
                    .get("eps")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1e-6);
                let instrument = opts
                    .get("instrument")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or_else(|| opts.contains_key("filter") || opts.contains_key("ops"));
                let profile = opts
                    .get("profile")
                    .or_else(|| opts.get("stats"))
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or_else(|| opts.contains_key("ridge"));
                let filter = opts
                    .get("filter")
                    .or_else(|| opts.get("ops"))
                    .cloned()
                    .unwrap_or_else(|| "gemm".to_string());
                let ridge_flops_per_byte = opts
                    .get("ridge")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(10.0);
                let shapes = opts
                    .get("shapes")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(true);
                let equiv = opts
                    .get("equiv")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!sanitize && !instrument);
                let tolerance = opts
                    .get("tol")
                    .or_else(|| opts.get("tolerance"))
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1e-4);
                let hash = opts
                    .get("hash")
                    .map(|s| s != "false" && s != "0")
                    .unwrap_or(!sanitize && !instrument);
                let bisect_limit = opts
                    .get("bisect")
                    .or_else(|| opts.get("limit"))
                    .and_then(|s| s.parse::<usize>().ok());
                Ok(Box::new(UnifiedVerifyIrPass {
                    shapes,
                    equiv,
                    tolerance,
                    hash,
                    bisect_limit,
                    sanitize,
                    clamp_max,
                    eps,
                    profile,
                    instrument,
                    filter,
                    ridge_flops_per_byte,
                }))
            },
        );
        reg.register(
            "O1",
            "Preset O1: Inlining, simplify-algebraic, LICM, GVN, fold-views, canonicalize, dce",
            |_opts| Ok(Box::new(OptO1Pass)),
        );
        reg.register(
            "O2",
            "Preset O2: O1 + LoRA/BatchNorm weight folding + unified Transformer/GEMM fusions",
            |_opts| Ok(Box::new(OptO2Pass)),
        );
        reg.register(
            "O3",
            "Preset O3: Shape specialization + O2 + MoE/FMA/zero-copy + weight packing + arena",
            |opts| Ok(Box::new(build_o3_preset("O3", opts))),
        );
        reg
    }

    pub fn register<F>(&mut self, name: &'static str, description: &'static str, factory: F)
    where
        F: Fn(&PassOptions) -> Result<Box<dyn Pass>> + Send + Sync + 'static,
    {
        self.passes.insert(
            name,
            PassDescriptor {
                name,
                description,
                factory: Box::new(factory),
            },
        );
    }

    pub fn create_pass(&self, name: &str, opts: &PassOptions) -> Result<Box<dyn Pass>> {
        let desc = self.passes.get(name).ok_or_else(|| {
            anyhow!("Unknown pass `{name}`. Run `castforge list-passes` to see available passes.")
        })?;
        (desc.factory)(opts)
    }

    pub fn list(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.passes.values().map(|d| (d.name, d.description))
    }
}

fn build_o3_preset(preset_name: &'static str, opts: &PassOptions) -> OptO3Pass {
    let mut default_dim = 1i64;
    let mut bindings = IndexMap::new();
    for (k, v) in opts {
        if let Ok(n) = v.parse::<i64>() {
            if k == "default" || k == "?" {
                default_dim = n;
            } else {
                bindings.insert(k.clone(), n);
            }
        }
    }
    OptO3Pass {
        preset_name,
        default_dim,
        bindings,
    }
}
