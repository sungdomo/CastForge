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

// ============================================================================
// Built-in Transformation & Optimization Passes
// ============================================================================

/// 1. Dead Code Elimination (DCE — SSA values + optional Symbol DCE)
pub struct DeadCodeEliminationPass {
    pub include_symbols: bool,
}

impl Default for DeadCodeEliminationPass {
    fn default() -> Self {
        Self {
            include_symbols: true,
        }
    }
}

impl DeadCodeEliminationPass {
    pub fn ssa_only() -> Self {
        Self {
            include_symbols: false,
        }
    }
}

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn description(&self) -> &'static str {
        "Dead Code Elimination (removes unused pure SSA ops and dead private symbols)"
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

        if self.include_symbols {
            let sym_res = SymbolDcePass.run(module)?;
            removed += sym_res.ops_removed;
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

/// 3. Fold Transpose Pass (`transpose(transpose(x, p1), p2)` -> `transpose(x, p1[p2])`)
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
                    && current_perm.iter().enumerate().all(|(i, &p)| p == i as i64);

                if is_identity {
                    module.replace_all_uses(result, current_operand);
                    module.remove_op(block_id, op_id);
                    removed += 1;
                }
            }
        }

        // Clean up any producer transpose that became dead
        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
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
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
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
        "Fuses eligible elementwise producer-consumer pairs into stablehlo.composite"
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
                let castforge_ir::ValueDef::OpResult { op: add_op_id, .. } = module.values[lhs].def
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
        "Composite canonicalization pass (constants, algebraic, transpose, reshape, cse, dce)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut total = PassResult {
            pass_name: self.name().to_string(),
            ..Default::default()
        };

        for _ in 0..8 {
            let r0 = FoldConstantsPass::default().run(module)?;
            let r_alg = SimplifyAlgebraicPass.run(module)?;
            let r1 = FoldTransposePass::default().run(module)?;
            let r2 = FoldReshapePass.run(module)?;
            let r3 = CommonSubexpressionEliminationPass.run(module)?;
            let r4 = DeadCodeEliminationPass::default().run(module)?;

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

/// 8. Algebraic Simplification Pass (`x + 0 -> x`, `x * 1 -> x`, `x * 0 -> 0`)
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
                            && module.values[operand].ty.as_tensor().is_some_and(|t| {
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
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
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

/// 9. Unified Constant Folding & Conditional Propagation Pass (`fold-constants`)
pub struct FoldConstantsPass {
    pub include_conditional: bool,
}

impl Default for FoldConstantsPass {
    fn default() -> Self {
        Self {
            include_conditional: true,
        }
    }
}

impl FoldConstantsPass {
    pub fn const_splat(module: &Module, val_id: ValueId) -> Option<f64> {
        if let castforge_ir::ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                return op_node.attributes.get("value")?.as_splat_f64();
            }
        }
        None
    }

    pub fn const_f32_vec(module: &Module, val_id: ValueId) -> Option<Vec<f32>> {
        if let castforge_ir::ValueDef::OpResult { op, .. } = module.values[val_id].def {
            let op_node = module.ops.get(op)?;
            if op_node.is_op("stablehlo", "constant") {
                let n = module.values[val_id]
                    .ty
                    .as_tensor()
                    .and_then(|t| t.num_elements())
                    .unwrap_or(1) as usize;
                if let Some(Attribute::DenseElements { raw_literal, .. }) =
                    op_node.attributes.get("value")
                {
                    return parse_dense_f32_literal(raw_literal, n);
                }
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
        "Constant folding & conditional propagation (arithmetic, shape, compare/select)"
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

                // 1. Conditional folding (`compare` & `select`) when enabled
                if self.include_conditional {
                    match &hlo {
                        StableHloOp::Compare {
                            lhs,
                            rhs,
                            direction,
                            result,
                        } => {
                            if let (Some(a), Some(b)) = (
                                Self::const_splat(module, *lhs),
                                Self::const_splat(module, *rhs),
                            ) {
                                let cond = match direction.as_str() {
                                    "EQ" => (a - b).abs() < 1e-7,
                                    "NE" => (a - b).abs() >= 1e-7,
                                    "LT" => a < b,
                                    "LE" => a <= b,
                                    "GT" => a > b,
                                    "GE" => a >= b,
                                    _ => false,
                                };
                                if let Some(res_ty) = module.values[*result].ty.as_tensor().cloned()
                                {
                                    let lit = if cond { "1" } else { "0" };
                                    let mut attrs = IndexMap::new();
                                    attrs.insert(
                                        "value".to_string(),
                                        Attribute::DenseElements {
                                            ty: res_ty,
                                            raw_literal: lit.to_string(),
                                        },
                                    );
                                    let op_mut = &mut module.ops[op_id];
                                    op_mut.name = OpName::new("stablehlo", "constant");
                                    op_mut.operands.clear();
                                    op_mut.attributes = attrs;
                                    modified += 1;
                                    continue;
                                }
                            }
                        }
                        StableHloOp::Select {
                            pred,
                            on_true,
                            on_false,
                            result,
                        } => {
                            let chosen = if on_true == on_false {
                                Some(*on_true)
                            } else {
                                Self::const_splat(module, *pred)
                                    .map(|v| if v != 0.0 { *on_true } else { *on_false })
                            };
                            if let Some(winner) = chosen {
                                module.replace_all_uses(*result, winner);
                                module.remove_op(block_id, op_id);
                                removed += 1;
                                continue;
                            }
                        }
                        _ => {}
                    }
                }

                // 2. General tensor & splat constant folding
                let folded_vec: Option<(ValueId, Vec<f32>)> = match hlo {
                    StableHloOp::Add { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Subtract { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Multiply { lhs, rhs, result } => {
                        match (
                            Self::const_f32_vec(module, lhs),
                            Self::const_f32_vec(module, rhs),
                        ) {
                            (Some(a), Some(b)) if a.len() == b.len() => {
                                Some((result, a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()))
                            }
                            _ => None,
                        }
                    }
                    StableHloOp::Reshape { operand, result } => {
                        Self::const_f32_vec(module, operand).map(|v| (result, v))
                    }
                    StableHloOp::Transpose {
                        operand, result, ..
                    }
                    | StableHloOp::BroadcastInDim {
                        operand, result, ..
                    } => Self::const_splat(module, operand).map(|v| {
                        let n = module.values[result]
                            .ty
                            .as_tensor()
                            .and_then(|t| t.num_elements())
                            .unwrap_or(1) as usize;
                        (result, vec![v as f32; n])
                    }),
                    _ => None,
                };

                if let Some((res_val, vals)) = folded_vec {
                    let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
                        continue;
                    };
                    let raw_literal = format_dense_f32_literal(&vals, &tensor_ty.shape);
                    let mut attrs = IndexMap::new();
                    attrs.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty: tensor_ty,
                            raw_literal,
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

        if modified > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
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
        "Post-training symmetric weight quantization (bits=8|4) with uniform_dequantize"
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

                // Create `stablehlo.uniform_dequantize` op right after the constant op
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

                // Redirect all downstream uses of `const_val_id` to `deq_val_id`
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

                let keep_cols =
                    ((cols as f32) * (1.0 - self.sparsity.clamp(0.0, 0.9))).round() as usize;
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

                // Update dot_general output shape and propagate to downstream consumers
                if let Some(mut dot_out_ty) = module.values[dot_res].ty.as_tensor().cloned() {
                    if let Some(last_dim) = dot_out_ty.shape.last_mut() {
                        *last_dim = Dim::Static(keep_cols as i64);
                    }
                    let updated_out_ty = Type::Tensor(dot_out_ty);
                    module.values[dot_res].ty = updated_out_ty.clone();

                    // Also update parent func.func return type if dot_res is returned
                    for fn_op_id in module.functions() {
                        if let Some(Attribute::Type(Type::Function { inputs, results })) = module
                            .ops[fn_op_id]
                            .attributes
                            .get("function_type")
                            .cloned()
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
/// Fuses paired constant weight tensors using Linear, SLERP, or TIES merging.
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
                if !lhs_op.is_op("stablehlo", "constant") || !rhs_op.is_op("stablehlo", "constant")
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
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
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

/// 14. Function Call Inliner Pass (`inliner`)
/// Inlines `func.call @callee(...)` operations into their caller blocks.
pub struct InlinerPass;

impl Pass for InlinerPass {
    fn name(&self) -> &'static str {
        "inliner"
    }

    fn description(&self) -> &'static str {
        "Inlines func.call operations by splicing callee bodies into caller blocks"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut removed = 0usize;
        let mut created = 0usize;

        let mut func_map: HashMap<String, OpId> = HashMap::new();
        for fn_op_id in module.functions() {
            if let Some(name) = module.ops[fn_op_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
            {
                func_map.insert(name.to_string(), fn_op_id);
            }
        }

        let blocks: Vec<BlockId> = module.blocks.keys().collect();
        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for call_op_id in ops {
                let Some(call_op) = module.ops.get(call_op_id).cloned() else {
                    continue;
                };
                if !call_op.is_op("func", "call") {
                    continue;
                }
                let Some(callee_name) = call_op.attributes.get("callee").and_then(|a| a.as_str())
                else {
                    continue;
                };
                let Some(&callee_fn_id) = func_map.get(callee_name) else {
                    continue;
                };
                let callee_reg = module.ops[callee_fn_id].regions[0];
                let callee_block = module.regions[callee_reg].blocks[0];
                if callee_block == block_id {
                    continue;
                }

                let callee_args = module.blocks[callee_block].arguments.clone();
                if callee_args.len() != call_op.operands.len() {
                    continue;
                }

                let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
                for (&arg_val, &caller_val) in callee_args.iter().zip(call_op.operands.iter()) {
                    val_map.insert(arg_val, caller_val);
                }

                let callee_ops = module.blocks[callee_block].ops.clone();
                let mut new_op_ids = Vec::new();
                let mut return_values = Vec::new();

                for src_op_id in callee_ops {
                    let src_op = module.ops[src_op_id].clone();
                    if src_op.is_op("func", "return") {
                        for &ret_opnd in &src_op.operands {
                            let mapped = val_map.get(&ret_opnd).copied().unwrap_or(ret_opnd);
                            return_values.push(mapped);
                        }
                        continue;
                    }

                    let mapped_operands: SmallVec<[ValueId; 4]> = src_op
                        .operands
                        .iter()
                        .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                        .collect();

                    let res_types: Vec<(Option<String>, Type)> = src_op
                        .results
                        .iter()
                        .map(|&r| {
                            let v = &module.values[r];
                            let inlined_name = v
                                .ssa_name
                                .as_ref()
                                .map(|n| format!("inl_{callee_name}_{n}"));
                            (inlined_name, v.ty.clone())
                        })
                        .collect();

                    let cloned_id = module.add_operation(
                        block_id,
                        src_op.name.clone(),
                        mapped_operands,
                        res_types,
                        src_op.attributes.clone(),
                        Vec::new(),
                        src_op.custom_syntax,
                        src_op.location.clone(),
                    );
                    module.blocks[block_id].ops.pop();
                    for (&old_res, &new_res) in src_op
                        .results
                        .iter()
                        .zip(module.ops[cloned_id].results.iter())
                    {
                        val_map.insert(old_res, new_res);
                    }
                    new_op_ids.push(cloned_id);
                    created += 1;
                }

                for (&call_res, &ret_val) in call_op.results.iter().zip(return_values.iter()) {
                    module.replace_all_uses(call_res, ret_val);
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == call_op_id)
                {
                    module.blocks[block_id].ops.splice(pos..pos, new_op_ids);
                }
                module.remove_op(block_id, call_op_id);
                removed += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: removed > 0,
            ops_removed: removed,
            ops_modified: 0,
            ops_created: created,
        })
    }
}

/// 15. Symbol Dead Code Elimination Pass (`symbol-dce`)
/// Eliminates unreachable `private` functions from the module.
pub struct SymbolDcePass;

impl Pass for SymbolDcePass {
    fn name(&self) -> &'static str {
        "symbol-dce"
    }

    fn description(&self) -> &'static str {
        "Symbol Dead Code Elimination (removes unreachable private func.func symbols)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut referenced_symbols = std::collections::HashSet::new();
        for (_, op) in &module.ops {
            if op.is_op("func", "call") {
                if let Some(callee) = op.attributes.get("callee").and_then(|a| a.as_str()) {
                    referenced_symbols.insert(callee.to_string());
                }
            }
            if op.is_op("stablehlo", "composite") {
                if let Some(decomp) = op.attributes.get("decomposition").and_then(|a| a.as_str()) {
                    referenced_symbols.insert(decomp.to_string());
                }
            }
        }

        let root_block = module.body_block();
        let mut removed = 0usize;
        for fn_op_id in module.functions() {
            let op = &module.ops[fn_op_id];
            let is_private =
                op.attributes.get("sym_visibility").and_then(|a| a.as_str()) == Some("private");
            let sym_name = op
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .unwrap_or("");
            if is_private && !referenced_symbols.contains(sym_name) {
                let regions = op.regions.clone();
                for reg_id in regions {
                    let blocks = module.regions[reg_id].blocks.clone();
                    for blk_id in blocks {
                        let inner_ops = module.block_ops(blk_id);
                        for inner_op in inner_ops {
                            module.remove_op(blk_id, inner_op);
                        }
                    }
                }
                module.remove_op(root_block, fn_op_id);
                removed += 1;
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

/// 16. Sparse Conditional Constant Propagation (`sccp`)
/// Evaluates `stablehlo.compare` on constants and folds `stablehlo.select` predicates.
/// Note: `fold-constants` includes conditional constant propagation by default; `sccp` runs
/// `FoldConstantsPass { include_conditional: true }` to a fixed point.
pub struct SccpPass;

impl Pass for SccpPass {
    fn name(&self) -> &'static str {
        "sccp"
    }

    fn description(&self) -> &'static str {
        "Sparse Conditional Constant Propagation (evaluates compare/select and constants)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut removed = 0usize;

        for _ in 0..8 {
            let fold_res = (FoldConstantsPass {
                include_conditional: true,
            })
            .run(module)?;
            modified += fold_res.ops_modified;
            removed += fold_res.ops_removed;
            if !fold_res.changed {
                break;
            }
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

/// 17. Topological Sort Pass (`topological-sort`)
/// Reorders operations inside each block so all SSA definitions precede their uses.
pub struct TopologicalSortPass;

impl Pass for TopologicalSortPass {
    fn name(&self) -> &'static str {
        "topological-sort"
    }

    fn description(&self) -> &'static str {
        "Reorders operations inside blocks to satisfy SSA def-before-use ordering"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let orig_ops = module.block_ops(block_id);
            if orig_ops.len() <= 1 {
                continue;
            }
            let block_op_set: std::collections::HashSet<OpId> = orig_ops.iter().copied().collect();
            let mut visited: std::collections::HashSet<OpId> = std::collections::HashSet::new();
            let mut sorted: Vec<OpId> = Vec::with_capacity(orig_ops.len());
            let mut terminators: Vec<OpId> = Vec::new();

            fn visit(
                op_id: OpId,
                module: &Module,
                block_op_set: &std::collections::HashSet<OpId>,
                visited: &mut std::collections::HashSet<OpId>,
                sorted: &mut Vec<OpId>,
            ) {
                if !visited.insert(op_id) {
                    return;
                }
                if let Some(op) = module.ops.get(op_id) {
                    for &operand in &op.operands {
                        if let ValueDef::OpResult { op: prod_op, .. } = module.values[operand].def {
                            if block_op_set.contains(&prod_op) {
                                visit(prod_op, module, block_op_set, visited, sorted);
                            }
                        }
                    }
                }
                sorted.push(op_id);
            }

            for &op_id in &orig_ops {
                let is_term = module
                    .ops
                    .get(op_id)
                    .is_some_and(|o| o.is_op("func", "return") || o.is_op("stablehlo", "return"));
                if is_term {
                    terminators.push(op_id);
                } else {
                    visit(op_id, module, &block_op_set, &mut visited, &mut sorted);
                }
            }
            sorted.extend(terminators);

            if sorted != orig_ops {
                module.blocks[block_id].ops = sorted;
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

/// 18. Strip DebugInfo Pass (`strip-debuginfo`)
/// Resets all operation `Location` annotations to `Location::Unknown`.
pub struct StripDebugInfoPass;

impl Pass for StripDebugInfoPass {
    fn name(&self) -> &'static str {
        "strip-debuginfo"
    }

    fn description(&self) -> &'static str {
        "Strips source location metadata (loc(...)) from all operations in the module"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in &mut module.ops {
            if op.location != Location::Unknown {
                op.location = Location::Unknown;
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

/// 19. Print Op Stats Pass (`print-op-stats`)
/// Computes operation frequency, parameter counts, and FLOP estimates across the module.
pub struct PrintOpStatsPass;

impl Pass for PrintOpStatsPass {
    fn name(&self) -> &'static str {
        "print-op-stats"
    }

    fn description(&self) -> &'static str {
        "Computes and prints operation counts, parameter sizes, and FLOP estimates"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut counts: IndexMap<String, usize> = IndexMap::new();
        let mut total_params = 0i64;
        let mut est_flops = 0i64;

        for (_, op) in &module.ops {
            *counts.entry(op.name.to_string()).or_insert(0) += 1;
            if op.is_op("stablehlo", "constant") {
                if let Some(&res) = op.results.first() {
                    if let Some(t) = module.values[res].ty.as_tensor() {
                        total_params += t.num_elements().unwrap_or(0);
                    }
                }
            } else if op.is_op("stablehlo", "dot_general") && op.operands.len() == 2 {
                let lhs_ty = module.values[op.operands[0]].ty.as_tensor();
                let rhs_ty = module.values[op.operands[1]].ty.as_tensor();
                if let (Some(l), Some(r)) = (lhs_ty, rhs_ty) {
                    if let (Some(Dim::Static(m)), Some(Dim::Static(k)), Some(Dim::Static(n))) =
                        (l.shape.first(), l.shape.get(1), r.shape.get(1))
                    {
                        est_flops += 2 * m * k * n;
                    }
                }
            }
        }

        eprintln!(
            "[castforge:stats] Total Ops: {} | Constant Params: {} | Est. MatMul FLOPs: {}",
            module.ops.len(),
            total_params,
            est_flops
        );
        for (op_name, count) in &counts {
            eprintln!("  - {:<28} : {}", op_name, count);
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
