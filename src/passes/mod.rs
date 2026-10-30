//! Unified [`Pass`], [`PassRegistry`], [`PassManager`], and **21 built-in A-to-B IR translator
//! passes & presets** (`18` passes + `3` presets `O1`/`O2`/`O3` in `castforge::passes`), organized
//! into modular single-traversal pass packages under `src/passes/`.

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
pub mod registry;
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
pub use registry::*;
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
