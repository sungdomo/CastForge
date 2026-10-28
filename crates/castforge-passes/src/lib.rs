pub mod passes;
pub use passes::*;

use anyhow::{Result, anyhow};
use castforge_ir::{MergeMethod, Module};
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

    /// Create a registry pre-populated with all built-in CastForge passes.
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
            "Common Subexpression Elimination (deduplicates identical pure operations)",
            |_opts| Ok(Box::new(CommonSubexpressionEliminationPass)),
        );
        reg.register(
            "fold-transpose",
            "Folds chained stablehlo.transpose ops and eliminates identity transposes",
            |opts| {
                let eliminate_identity = opts
                    .get("eliminate_identity")
                    .map(|v| v != "false")
                    .unwrap_or(true);
                Ok(Box::new(FoldTransposePass { eliminate_identity }))
            },
        );
        reg.register(
            "fold-reshape",
            "Folds chained stablehlo.reshape ops and eliminates identity reshapes",
            |_opts| Ok(Box::new(FoldReshapePass)),
        );
        reg.register(
            "simplify-algebraic",
            "Algebraic identity simplification (x+0->x, x*1->x, x*0->0, identity convert/slice)",
            |_opts| Ok(Box::new(SimplifyAlgebraicPass)),
        );
        reg.register(
            "fold-constants",
            "Constant folding & conditional propagation (arithmetic, shape, compare/select)",
            |opts| {
                let include_conditional = opts
                    .get("conditional")
                    .map(|v| v != "false")
                    .unwrap_or(true);
                Ok(Box::new(FoldConstantsPass {
                    include_conditional,
                }))
            },
        );
        reg.register(
            "verify-shapes",
            "Verifies StableHLO tensor shapes and rank consistency across all ops",
            |_opts| Ok(Box::new(VerifyShapesPass)),
        );
        reg.register(
            "quantize-weights",
            "Post-training symmetric weight quantization (bits=8|4) with uniform_dequantize",
            |opts| {
                let bits = opts
                    .get("bits")
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(8);
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                Ok(Box::new(QuantizeWeightsPass { bits, min_elements }))
            },
        );
        reg.register(
            "prune-structured",
            "Structured L1 channel pruning on 2D weight matrices in dot_general (sparsity=0.5)",
            |opts| {
                let sparsity = opts
                    .get("sparsity")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(StructuredPruningPass { sparsity }))
            },
        );
        reg.register(
            "merge-weights",
            "Merges paired constant weight tensors (method=linear|slerp|ties, t=0.5)",
            |opts| {
                let method = match opts.get("method").map(|s| s.to_lowercase()).as_deref() {
                    Some("slerp") => MergeMethod::Slerp,
                    Some("ties") => MergeMethod::Ties,
                    _ => MergeMethod::Linear,
                };
                let t = opts
                    .get("t")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.5);
                Ok(Box::new(MergeWeightsPass { method, t }))
            },
        );
        reg.register(
            "fuse-elementwise",
            "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite",
            |_opts| Ok(Box::new(FuseElementwisePass)),
        );
        reg.register(
            "ir-edit",
            "Surgical IR editing pass (rename_op=from:to, set_attr=op:key:val)",
            |opts| {
                let rename_op = opts.get("rename_op").cloned();
                let annotate_op = opts.get("annotate_op").cloned();
                Ok(Box::new(IrEditPass {
                    rename_op,
                    annotate_op,
                }))
            },
        );
        reg.register(
            "canonicalize",
            "Composite canonicalization pass (fold-transpose + fold-reshape + cse + dce)",
            |_opts| Ok(Box::new(CanonicalizePass)),
        );
        reg.register(
            "inliner",
            "Inlines func.call operations by splicing callee bodies into caller blocks",
            |_opts| Ok(Box::new(InlinerPass)),
        );
        reg.register(
            "symbol-dce",
            "Symbol Dead Code Elimination (removes unreachable private func.func symbols)",
            |_opts| Ok(Box::new(SymbolDcePass)),
        );
        reg.register(
            "sccp",
            "Sparse Conditional Constant Propagation (evaluates compare/select and constants)",
            |_opts| Ok(Box::new(SccpPass)),
        );
        reg.register(
            "topological-sort",
            "Reorders operations inside blocks to satisfy SSA def-before-use ordering",
            |_opts| Ok(Box::new(TopologicalSortPass)),
        );
        reg.register(
            "strip-debuginfo",
            "Strips source location metadata (loc(...)) from all operations in the module",
            |_opts| Ok(Box::new(StripDebugInfoPass)),
        );
        reg.register(
            "print-op-stats",
            "Computes and prints operation counts, parameter sizes, and FLOP estimates",
            |_opts| Ok(Box::new(PrintOpStatsPass)),
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

    /// Build a `PassManager` from a CLI pipeline specification string, e.g.:
    /// `"fold-transpose(eliminate_identity=true),fold-reshape,cse,dce"`
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

fn split_pipeline_spec(spec: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    for (i, c) in spec.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(&spec[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&spec[start..]);
    out
}

fn parse_pass_spec_item(item: &str) -> Result<(String, PassOptions)> {
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
