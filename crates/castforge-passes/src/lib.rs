use anyhow::{Result, anyhow};
use castforge_dialect_stablehlo::{StableHloOp, verify_stablehlo_shapes};
use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, MergeMethod, Module, OpId, OpName, TensorBacking,
    TensorEntry, TensorType, Type, ValueDef, ValueId, format_dense_f32_literal,
    format_dense_i8_literal, merge_f32_tensors, parse_dense_f32_literal, prune_2d_columns_by_l1,
    quantize_symmetric_f32,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;

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

/// Core trait implemented by all IR transformation, optimization, pruning, quantization, and editing passes.
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
            "Dead Code Elimination (removes unused pure operations)",
            |_opts| Ok(Box::new(DeadCodeEliminationPass)),
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
            "Constant folding for arithmetic and shape ops on constant splat tensors",
            |_opts| Ok(Box::new(FoldConstantsPass)),
        );
        reg.register(
            "verify-shapes",
            "Verifies StableHLO tensor shapes and rank consistency across all ops",
            |_opts| Ok(Box::new(VerifyShapesPass)),
        );
        reg.register(
            "quantize-weights",
            "Post-training symmetric weight quantization (bits=8|4, min_elements=2) with uniform_dequantize",
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
            "Merges paired constant weight tensors using linear, slerp, or ties (method=slerp, t=0.5)",
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
            "Fuses eligible elementwise producer-consumer pairs (e.g., add + maximum) into stablehlo.composite",
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
        let desc = self
            .passes
            .get(name)
            .ok_or_else(|| anyhow!("Unknown pass `{name}`. Run `castforge list-passes` to see available passes."))?;
        (desc.factory)(opts)
    }

    pub fn list(&self) -> impl Iterator<Item = (&'static str, &'static str)> + '_ {
        self.passes
            .values()
            .map(|d| (d.name, d.description))
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

    /// Execute the pipeline and capture a full step-by-step IR trace (initial state + after each pass)
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

// ============================================================================
// Built-in Transformation & Optimization Passes
// ============================================================================

/// 1. Dead Code Elimination (DCE)
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn description(&self) -> &'static str {
        "Dead Code Elimination (removes unused pure operations)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        loop {
            let mut pass_removed = 0usize;
            for &block_id in &blocks {
                let ops = module.block_ops(block_id);
                // Iterate in reverse so chains of dead ops are cleaned up quickly
                for &op_id in ops.iter().rev() {
                    let Some(op) = module.ops.get(op_id) else {
                        continue;
                    };
                    if !op.is_pure() || op.results.is_empty() {
                        continue;
                    }
                    let all_unused = op.results.iter().all(|&res| module.use_count(res) == 0);
                    if all_unused {
                        module.remove_op(block_id, op_id);
                        pass_removed += 1;
                    }
                }
            }
            removed += pass_removed;
            if pass_removed == 0 {
                break;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 2. Common Subexpression Elimination (CSE)
pub struct CommonSubexpressionEliminationPass;

impl Pass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn description(&self) -> &'static str {
        "Common Subexpression Elimination (deduplicates identical pure operations)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let mut seen: HashMap<String, OpId> = HashMap::new();
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_pure() || op.results.len() != 1 || !op.regions.is_empty() {
                    continue;
                }
                let res_val = op.results[0];
                let res_ty = &module.values[res_val].ty;
                let sig = format!(
                    "{}|operands:{:?}|attrs:{:?}|ty:{}",
                    op.name, op.operands, op.attributes, res_ty
                );

                if let Some(&existing_op_id) = seen.get(&sig) {
                    let existing_res = module.ops[existing_op_id].results[0];
                    module.replace_all_uses(res_val, existing_res);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                } else {
                    seen.insert(sig, op_id);
                }
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 3. Fold Transpose Pass (`transpose(transpose(x, p1), p2)` -> `transpose(x, p1[p2])` & identity removal)
pub struct FoldTransposePass {
    pub eliminate_identity: bool,
}

impl Default for FoldTransposePass {
    fn default() -> Self {
        Self {
            eliminate_identity: true,
        }
    }
}

impl Pass for FoldTransposePass {
    fn name(&self) -> &'static str {
        "fold-transpose"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.transpose ops and eliminates identity transposes"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Transpose {
                    operand,
                    permutation: perm2,
                    result,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                // Check if producer of `operand` is also a `stablehlo.transpose`
                let mut current_operand = operand;
                let mut current_perm = perm2;

                if let castforge_ir::ValueDef::OpResult { op: prod_op_id, .. } =
                    module.values[operand].def
                {
                    if let Some(prod_op) = module.ops.get(prod_op_id) {
                        if let Some(StableHloOp::Transpose {
                            operand: base_input,
                            permutation: perm1,
                            ..
                        }) = StableHloOp::from_op(prod_op)
                        {
                            if perm1.len() == current_perm.len() {
                                // Compose permutations: composed[i] = perm1[perm2[i]]
                                let composed: Vec<i64> = current_perm
                                    .iter()
                                    .map(|&idx| perm1[idx as usize])
                                    .collect();
                                current_operand = base_input;
                                current_perm = composed;

                                let op_mut = &mut module.ops[op_id];
                                op_mut.operands[0] = current_operand;
                                op_mut.attributes.insert(
                                    "permutation".to_string(),
                                    Attribute::DenseIntArray(current_perm.clone()),
                                );
                                modified += 1;
                            }
                        }
                    }
                }

                // Check if the resulting permutation is an identity `[0, 1, 2, ...]`
                let is_identity = self.eliminate_identity
                    && !current_perm.is_empty()
                    && current_perm
                        .iter()
                        .enumerate()
                        .all(|(i, &p)| p == i as i64);

                if is_identity {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        // Clean up any producer transpose that became dead
        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass.run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// 4. Fold Reshape Pass (`reshape(reshape(x))` -> `reshape(x)` & identity reshape removal)
pub struct FoldReshapePass;

impl Pass for FoldReshapePass {
    fn name(&self) -> &'static str {
        "fold-reshape"
    }

    fn description(&self) -> &'static str {
        "Folds chained stablehlo.reshape ops and eliminates identity reshapes"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Reshape { operand, result }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let mut current_operand = operand;
                if let castforge_ir::ValueDef::OpResult { op: prod_op_id, .. } =
                    module.values[operand].def
                {
                    if let Some(prod_op) = module.ops.get(prod_op_id) {
                        if let Some(StableHloOp::Reshape {
                            operand: base_input,
                            ..
                        }) = StableHloOp::from_op(prod_op)
                        {
                            current_operand = base_input;
                            module.ops[op_id].operands[0] = current_operand;
                            modified += 1;
                        }
                    }
                }

                // If input type and output type of reshape are identical, eliminate it!
                if module.values[current_operand].ty == module.values[result].ty {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass.run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// 5. Fuse Elementwise Pass (`stablehlo.add` + `stablehlo.maximum` -> `stablehlo.composite`)
pub struct FuseElementwisePass;

impl Pass for FuseElementwisePass {
    fn name(&self) -> &'static str {
        "fuse-elementwise"
    }

    fn description(&self) -> &'static str {
        "Fuses eligible elementwise producer-consumer pairs (e.g., add + maximum) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Maximum {
                    lhs,
                    rhs,
                    result: max_res,
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                // Check if `lhs` is produced by `stablehlo.add` with a single use
                let castforge_ir::ValueDef::OpResult { op: add_op_id, .. } =
                    module.values[lhs].def
                else {
                    continue;
                };
                if module.use_count(lhs) != 1 {
                    continue;
                }
                let Some(add_op) = module.ops.get(add_op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add {
                    lhs: add_a,
                    rhs: add_b,
                    ..
                }) = StableHloOp::from_op(add_op)
                else {
                    continue;
                };

                // Rewrite `max_op` in-place into `stablehlo.composite` representing fused bias+relu
                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_add_max".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_add_max_impl".to_string()),
                );

                let max_op_mut = &mut module.ops[op_id];
                max_op_mut.name = OpName::new("stablehlo", "composite");
                max_op_mut.operands = SmallVec::from_slice(&[add_a, add_b, rhs]);
                max_op_mut.attributes = attrs;
                let _ = max_res;
                created += 1;

                // Remove the now-dead `add` op
                module.remove_op(block_id, add_op_id);
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

/// 6. Surgical IR Edit Pass (`ir-edit`)
pub struct IrEditPass {
    /// Format: `"old_dialect.old_op:new_dialect.new_op"`
    pub rename_op: Option<String>,
    /// Format: `"dialect.op:attr_key:attr_value"`
    pub annotate_op: Option<String>,
}

impl Pass for IrEditPass {
    fn name(&self) -> &'static str {
        "ir-edit"
    }

    fn description(&self) -> &'static str {
        "Surgical IR editing pass (rename_op=from:to, annotate_op=op:key:val)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;

        if let Some(rule) = &self.rename_op {
            if let Some((from, to)) = rule.split_once(':') {
                let from_name = OpName::parse(from.trim());
                let to_name = OpName::parse(to.trim());
                for (_, op) in &mut module.ops {
                    if op.name == from_name {
                        op.name = to_name.clone();
                        modified += 1;
                    }
                }
            }
        }

        if let Some(rule) = &self.annotate_op {
            let parts: Vec<&str> = rule.splitn(3, ':').collect();
            if parts.len() == 3 {
                let target_op = OpName::parse(parts[0].trim());
                let attr_key = parts[1].trim().to_string();
                let attr_val = Attribute::String(parts[2].trim().to_string());
                for (_, op) in &mut module.ops {
                    if op.name == target_op {
                        op.attributes.insert(attr_key.clone(), attr_val.clone());
                        modified += 1;
                    }
                }
            }
        }

        let _ = Location::Unknown;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// 7. Composite Canonicalize Pass
pub struct CanonicalizePass;

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str {
        "canonicalize"
    }

    fn description(&self) -> &'static str {
        "Composite canonicalization pass (fold-constants + simplify-algebraic + fold-transpose + fold-reshape + cse + dce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let r0 = FoldConstantsPass.run(module)?;
            let r_alg = SimplifyAlgebraicPass.run(module)?;
            let r1 = FoldTransposePass::default().run(module)?;
            let r2 = FoldReshapePass.run(module)?;
            let r3 = CommonSubexpressionEliminationPass.run(module)?;
            let r4 = DeadCodeEliminationPass.run(module)?;

            let iter_changed =
                r0.changed || r_alg.changed || r1.changed || r2.changed || r3.changed || r4.changed;
            total.changed |= iter_changed;
            total.ops_removed += r0.ops_removed
                + r_alg.ops_removed
                + r1.ops_removed
                + r2.ops_removed
                + r3.ops_removed
                + r4.ops_removed;
            total.ops_modified += r0.ops_modified
                + r_alg.ops_modified
                + r1.ops_modified
                + r2.ops_modified
                + r3.ops_modified
                + r4.ops_modified;
            if !iter_changed {
                break;
            }
        }

        Ok(total)
    }
}

/// 8. Algebraic Simplification Pass (`x + 0 -> x`, `x * 1 -> x`, `x * 0 -> 0`, identity `convert`/`slice`)
pub struct SimplifyAlgebraicPass;

impl SimplifyAlgebraicPass {
    fn const_attr(module: &Module, val_id: ValueId) -> Option<&Attribute> {
        if let castforge_ir::ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                return op_node.attributes.get("value");
            }
        }
        None
    }
}

impl Pass for SimplifyAlgebraicPass {
    fn name(&self) -> &'static str {
        "simplify-algebraic"
    }

    fn description(&self) -> &'static str {
        "Algebraic identity simplification (x+0->x, x*1->x, x*0->0, identity convert/slice)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                let replacement: Option<ValueId> = match hlo {
                    StableHloOp::Add { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, lhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, rhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Subtract { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, lhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Multiply { lhs, rhs, result } => {
                        if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_one()) {
                            Some((result, lhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_one()) {
                            Some((result, rhs))
                        } else if Self::const_attr(module, rhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, rhs))
                        } else if Self::const_attr(module, lhs).is_some_and(|a| a.is_splat_zero()) {
                            Some((result, lhs))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Convert { operand, result } => {
                        if module.values[operand].ty == module.values[result].ty {
                            Some((result, operand))
                        } else {
                            None
                        }
                    }
                    StableHloOp::Slice {
                        operand,
                        start_indices,
                        limit_indices,
                        strides,
                        result,
                    } => {
                        let is_full = module.values[operand].ty == module.values[result].ty
                            && start_indices.iter().all(|&s| s == 0)
                            && strides.iter().all(|&s| s == 1)
                            && module.values[operand]
                                .ty
                                .as_tensor()
                                .is_some_and(|t| {
                                    t.shape.len() == limit_indices.len()
                                        && t.shape
                                            .iter()
                                            .zip(limit_indices.iter())
                                            .all(|(d, &lim)| *d == Dim::Static(lim))
                                });
                        if is_full {
                            Some((result, operand))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
                .map(|(res, rep)| {
                    module.replace_all_uses(res, rep);
                    rep
                });

                if replacement.is_some() {
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        if removed > 0 {
            let dce_res = DeadCodeEliminationPass.run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 9. Constant Folding Pass (`fold-constants`)
pub struct FoldConstantsPass;

impl FoldConstantsPass {
    fn const_splat(module: &Module, val_id: ValueId) -> Option<f64> {
        if let castforge_ir::ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                return op_node.attributes.get("value")?.as_splat_f64();
            }
        }
        None
    }
}

impl Pass for FoldConstantsPass {
    fn name(&self) -> &'static str {
        "fold-constants"
    }

    fn description(&self) -> &'static str {
        "Constant folding for arithmetic and shape ops on constant splat tensors"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                let folded: Option<(ValueId, f64)> = match hlo {
                    StableHloOp::Add { lhs, rhs, result } => {
                        match (Self::const_splat(module, lhs), Self::const_splat(module, rhs)) {
                            (Some(a), Some(b)) => Some((result, a + b)),
                            _ => None,
                        }
                    }
                    StableHloOp::Subtract { lhs, rhs, result } => {
                        match (Self::const_splat(module, lhs), Self::const_splat(module, rhs)) {
                            (Some(a), Some(b)) => Some((result, a - b)),
                            _ => None,
                        }
                    }
                    StableHloOp::Multiply { lhs, rhs, result } => {
                        match (Self::const_splat(module, lhs), Self::const_splat(module, rhs)) {
                            (Some(a), Some(b)) => Some((result, a * b)),
                            _ => None,
                        }
                    }
                    StableHloOp::Reshape { operand, result }
                    | StableHloOp::Transpose {
                        operand, result, ..
                    }
                    | StableHloOp::BroadcastInDim {
                        operand, result, ..
                    } => Self::const_splat(module, operand).map(|v| (result, v)),
                    _ => None,
                };

                if let Some((res_val, scalar)) = folded {
                    let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                        continue;
                    };
                    let mut attrs = IndexMap::new();
                    attrs.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty: tensor_ty,
                            raw_literal: format!("{scalar:?}"),
                        },
                    );
                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "constant");
                    op_mut.operands.clear();
                    op_mut.attributes = attrs;
                    modified += 1;
                }
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass.run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// 10. Shape Verification Pass (`verify-shapes`)
pub struct VerifyShapesPass;

impl Pass for VerifyShapesPass {
    fn name(&self) -> &'static str {
        "verify-shapes"
    }

    fn description(&self) -> &'static str {
        "Verifies StableHLO tensor shapes and rank consistency across all ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        for (op_id, _) in &module.ops {
            if let Err(msg) = verify_stablehlo_shapes(module, op_id) {
                return Err(anyhow!(msg));
            }
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: false,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}

/// 11. Weight Quantization Pass (`quantize-weights`)
/// Quantizes floating-point `stablehlo.constant` weights to `i8` or `i4` (`!quant.uniform<...>`)
/// and inserts `stablehlo.uniform_dequantize` ops.
pub struct QuantizeWeightsPass {
    pub bits: u8,
    pub min_elements: usize,
}

impl Default for QuantizeWeightsPass {
    fn default() -> Self {
        Self {
            bits: 8,
            min_elements: 2,
        }
    }
}

impl Pass for QuantizeWeightsPass {
    fn name(&self) -> &'static str {
        "quantize-weights"
    }

    fn description(&self) -> &'static str {
        "Post-training symmetric weight quantization (bits=8|4, min_elements=2) with uniform_dequantize"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut created = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                if !op.is_op("stablehlo", "constant") || op.results.len() != 1 {
                    continue;
                }
                let const_val_id = op.results[0];
                let orig_ty = module.values[const_val_id].ty.clone();
                let Some(tensor_ty) = orig_ty.as_tensor() else {
                    continue;
                };
                if !matches!(tensor_ty.dtype, DType::F32 | DType::F16 | DType::BF16) {
                    continue;
                }
                let num_elems = tensor_ty.num_elements().unwrap_or(0) as usize;
                if num_elems < self.min_elements {
                    continue;
                }

                let Some(Attribute::DenseElements { raw_literal, .. }) =
                    op.attributes.get("value").cloned()
                else {
                    continue;
                };
                let Some(f32_vals) = parse_dense_f32_literal(&raw_literal, num_elems) else {
                    continue;
                };

                let (q_data, scale, zp, q_dtype) =
                    quantize_symmetric_f32(&f32_vals, self.bits, tensor_ty.dtype.clone());
                let q_tensor_ty = TensorType {
                    shape: tensor_ty.shape.clone(),
                    dtype: q_dtype,
                    encoding: None,
                };
                let q_literal = format_dense_i8_literal(&q_data, &tensor_ty.shape);

                // Update TensorStore entry
                let store_key = module.values[const_val_id]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| format!("q_weight_{modified}"));
                module.tensor_store.insert(TensorEntry {
                    name: store_key,
                    ty: q_tensor_ty.clone(),
                    backing: TensorBacking::QuantizedI8 {
                        data: q_data,
                        scale,
                        zero_point: zp,
                    },
                });

                // Update the constant op to produce the quantized tensor type
                let q_ty = Type::Tensor(q_tensor_ty.clone());
                module.values[const_val_id].ty = q_ty;
                module.ops[op_id].attributes.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: q_tensor_ty,
                        raw_literal: q_literal,
                    },
                );
                modified += 1;

                // Create `stablehlo.uniform_dequantize` op and splice it right after the constant op
                let deq_ssa = module.values[const_val_id]
                    .ssa_name
                    .as_ref()
                    .map(|s| format!("{s}_deq"));
                let deq_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "uniform_dequantize"),
                    SmallVec::from_slice(&[const_val_id]),
                    vec![(deq_ssa, orig_ty)],
                    IndexMap::new(),
                    Vec::new(),
                    true,
                    Location::Unknown,
                );
                let deq_val_id = module.ops[deq_op_id].results[0];

                // Move `deq_op_id` from end of block to immediately after `op_id`
                let block_ops = &mut module.blocks[block_id].ops;
                if let Some(last) = block_ops.pop() {
                    if let Some(pos) = block_ops.iter().position(|&id| id == op_id) {
                        block_ops.insert(pos + 1, last);
                    } else {
                        block_ops.push(last);
                    }
                }

                // Redirect all downstream uses of `const_val_id` (except `deq_op_id` itself) to `deq_val_id`
                for (other_op_id, other_op) in &mut module.ops {
                    if other_op_id == deq_op_id {
                        continue;
                    }
                    for operand in &mut other_op.operands {
                        if *operand == const_val_id {
                            *operand = deq_val_id;
                        }
                    }
                }
                created += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// 12. Structured Channel Pruning Pass (`prune-structured`)
/// Prunes low-L1-importance output columns of 2D weight constants in `stablehlo.dot_general`
/// and propagates the reduced dimension downstream.
pub struct StructuredPruningPass {
    pub sparsity: f32,
}

impl Default for StructuredPruningPass {
    fn default() -> Self {
        Self { sparsity: 0.5 }
    }
}

impl Pass for StructuredPruningPass {
    fn name(&self) -> &'static str {
        "prune-structured"
    }

    fn description(&self) -> &'static str {
        "Structured L1 channel pruning on 2D weight matrices in dot_general (sparsity=0.5)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::DotGeneral {
                    rhs,
                    result: dot_res,
                    ..
                }) = StableHloOp::from_op(op)
                else {
                    continue;
                };

                let ValueDef::OpResult {
                    op: weight_op_id, ..
                } = module.values[rhs].def
                else {
                    continue;
                };
                let Some(weight_op) = module.ops.get(weight_op_id) else {
                    continue;
                };
                if !weight_op.is_op("stablehlo", "constant") {
                    continue;
                }

                let Some(w_ty) = module.values[rhs].ty.as_tensor().cloned() else {
                    continue;
                };
                if w_ty.shape.len() != 2 {
                    continue;
                }
                let (Dim::Static(rows_i64), Dim::Static(cols_i64)) =
                    (&w_ty.shape[0], &w_ty.shape[1])
                else {
                    continue;
                };
                let rows = *rows_i64 as usize;
                let cols = *cols_i64 as usize;
                if cols <= 1 {
                    continue;
                }

                let keep_cols = ((cols as f32) * (1.0 - self.sparsity.clamp(0.0, 0.9))).round()
                    as usize;
                let keep_cols = keep_cols.clamp(1, cols);
                if keep_cols == cols {
                    continue;
                }

                let Some(Attribute::DenseElements { raw_literal, .. }) =
                    weight_op.attributes.get("value").cloned()
                else {
                    continue;
                };
                let Some(f32_vals) = parse_dense_f32_literal(&raw_literal, rows * cols) else {
                    continue;
                };

                let (pruned_vals, _kept_cols) =
                    prune_2d_columns_by_l1(&f32_vals, rows, cols, keep_cols);

                let new_w_ty = TensorType {
                    shape: vec![Dim::Static(rows as i64), Dim::Static(keep_cols as i64)],
                    dtype: w_ty.dtype.clone(),
                    encoding: None,
                };
                let new_literal = format_dense_f32_literal(&pruned_vals, &new_w_ty.shape);

                // Update weight constant type, literal, and TensorStore
                module.values[rhs].ty = Type::Tensor(new_w_ty.clone());
                module.ops[weight_op_id].attributes.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: new_w_ty.clone(),
                        raw_literal: new_literal,
                    },
                );
                let store_key = module.values[rhs]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| "pruned_weight".to_string());
                module
                    .tensor_store
                    .insert(TensorEntry::from_f32(store_key, new_w_ty, pruned_vals));

                // Update dot_general output shape and propagate to downstream single-input/return consumers
                if let Some(mut dot_out_ty) = module.values[dot_res].ty.as_tensor().cloned() {
                    if let Some(last_dim) = dot_out_ty.shape.last_mut() {
                        *last_dim = Dim::Static(keep_cols as i64);
                    }
                    let updated_out_ty = Type::Tensor(dot_out_ty);
                    module.values[dot_res].ty = updated_out_ty.clone();

                    // Also update parent func.func return type if dot_res is returned
                    for fn_op_id in module.functions() {
                        if let Some(Attribute::Type(Type::Function { inputs, results })) =
                            module.ops[fn_op_id].attributes.get("function_type").cloned()
                        {
                            if results.len() == 1 {
                                module.ops[fn_op_id].attributes.insert(
                                    "function_type".to_string(),
                                    Attribute::Type(Type::Function {
                                        inputs,
                                        results: vec![updated_out_ty.clone()],
                                    }),
                                );
                            }
                        }
                    }
                }

                modified += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// 13. Weight Merging Pass (`merge-weights`)
/// Fuses `stablehlo.add(w_a, w_b)` on two constant weight tensors using Linear, SLERP, or TIES merging.
pub struct MergeWeightsPass {
    pub method: MergeMethod,
    pub t: f32,
}

impl Pass for MergeWeightsPass {
    fn name(&self) -> &'static str {
        "merge-weights"
    }

    fn description(&self) -> &'static str {
        "Merges paired constant weight tensors using linear, slerp, or ties (method=slerp, t=0.5)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                // Match `stablehlo.add %w1, %w2` or `stablehlo.custom_call @merge(%w1, %w2)`
                let is_merge_target = op.is_op("stablehlo", "add")
                    || (op.is_op("stablehlo", "custom_call")
                        && op
                            .attributes
                            .get("call_target_name")
                            .and_then(|a| a.as_str())
                            == Some("castforge.merge"));
                if !is_merge_target || op.operands.len() != 2 || op.results.len() != 1 {
                    continue;
                }

                let lhs = op.operands[0];
                let rhs = op.operands[1];
                let res_val = op.results[0];

                let (
                    ValueDef::OpResult { op: lhs_op_id, .. },
                    ValueDef::OpResult { op: rhs_op_id, .. },
                ) = (module.values[lhs].def, module.values[rhs].def)
                else {
                    continue;
                };

                let (Some(lhs_op), Some(rhs_op)) =
                    (module.ops.get(lhs_op_id), module.ops.get(rhs_op_id))
                else {
                    continue;
                };
                if !lhs_op.is_op("stablehlo", "constant")
                    || !rhs_op.is_op("stablehlo", "constant")
                {
                    continue;
                }

                let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                    continue;
                };
                let num_elems = tensor_ty.num_elements().unwrap_or(1) as usize;

                let (
                    Some(Attribute::DenseElements {
                        raw_literal: lit_a, ..
                    }),
                    Some(Attribute::DenseElements {
                        raw_literal: lit_b, ..
                    }),
                ) = (
                    lhs_op.attributes.get("value").cloned(),
                    rhs_op.attributes.get("value").cloned(),
                )
                else {
                    continue;
                };

                let (Some(vals_a), Some(vals_b)) = (
                    parse_dense_f32_literal(&lit_a, num_elems),
                    parse_dense_f32_literal(&lit_b, num_elems),
                ) else {
                    continue;
                };

                let merged = merge_f32_tensors(&vals_a, &vals_b, self.method, self.t);
                let merged_lit = format_dense_f32_literal(&merged, &tensor_ty.shape);

                let store_key = module.values[res_val]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| "merged_weight".to_string());
                module.tensor_store.insert(TensorEntry::from_f32(
                    store_key,
                    tensor_ty.clone(),
                    merged,
                ));

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "value".to_string(),
                    Attribute::DenseElements {
                        ty: tensor_ty,
                        raw_literal: merged_lit,
                    },
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "constant");
                op_mut.operands.clear();
                op_mut.attributes = attrs;
                modified += 1;
            }
        }

        if modified > 0 {
            let dce_res = DeadCodeEliminationPass.run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0 || removed > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
