use crate::{
    BakeLoraPass, CanonicalizePass, CommonSubexpressionEliminationPass, DeadCodeEliminationPass,
    DecomposeCompositesPass, ExternalizeWeightsPass, FoldBroadcastPass, FoldConstantsPass,
    FoldConvertPass, FoldNormWeightsPass, FoldReshapePass, FoldSliceConcatPass, FoldTransposePass,
    FuseElementwisePass, FuseHorizontalDotPass, FuseMatmulEpiloguePass, FuseRmsNormPass,
    FuseSoftmaxPass, FuseSwiGluPass, HoistConstantsPass, InlinerPass, IrEditPass, LicmPass,
    MergeWeightsPass, Pass, PassOptions, PrintOpStatsPass, QuantizeWeightsPass, SccpPass,
    SimplifyAlgebraicPass, SinkTransposePass, SpecializeShapesPass, StripDebugInfoPass,
    StructuredPruningPass, SymbolDcePass, TaskArithmeticPass, TopologicalSortPass,
    VerifyShapesPass,
};
use anyhow::{Result, anyhow};
use castforge_ir::MergeMethod;
use indexmap::IndexMap;

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
            "Computes operation counts, parameter sizes, FLOPs, and Roofline arithmetic intensity",
            |opts| {
                let ridge = opts
                    .get("ridge")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(10.0);
                Ok(Box::new(PrintOpStatsPass {
                    ridge_flops_per_byte: ridge,
                }))
            },
        );
        reg.register(
            "externalize-weights",
            "Converts inline dense<...> weights to dense_resource<key> in TensorStore",
            |opts| {
                let min_elements = opts
                    .get("min_elements")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(2);
                let inline = opts.get("inline").is_some_and(|v| v == "true");
                Ok(Box::new(ExternalizeWeightsPass {
                    min_elements,
                    inline,
                }))
            },
        );
        reg.register(
            "sink-transpose",
            "Sinks stablehlo.transpose through elementwise ops toward constants and producers",
            |_opts| Ok(Box::new(SinkTransposePass)),
        );
        reg.register(
            "fuse-matmul-epilogue",
            "Fuses dot_general + bias add (+ optional relu/tanh) into stablehlo.composite",
            |_opts| Ok(Box::new(FuseMatmulEpiloguePass)),
        );
        reg.register(
            "fold-norm-weights",
            "Absorbs constant RMSNorm/LayerNorm scale multiply into adjacent dot_general weights",
            |_opts| Ok(Box::new(FoldNormWeightsPass)),
        );
        reg.register(
            "fuse-softmax",
            "Fuses exp / broadcast(reduce_sum(exp)) into stablehlo.composite (fused_softmax)",
            |_opts| Ok(Box::new(FuseSoftmaxPass)),
        );
        reg.register(
            "fuse-rmsnorm",
            "Fuses x * rsqrt(mean(x^2) + eps) * weight into stablehlo.composite (fused_rmsnorm)",
            |_opts| Ok(Box::new(FuseRmsNormPass)),
        );
        reg.register(
            "decompose-composites",
            "Expands stablehlo.composite operations back into primitive StableHLO ops",
            |_opts| Ok(Box::new(DecomposeCompositesPass)),
        );
        reg.register(
            "bake-lora",
            "Fuses LoRA adapter branches W + alpha * (A @ B) into base weight matrices",
            |opts| {
                let alpha = opts
                    .get("alpha")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                Ok(Box::new(BakeLoraPass { alpha }))
            },
        );
        reg.register(
            "task-arithmetic",
            "Applies multi-checkpoint task vector arithmetic and DARE pruning (lambda, drop_rate)",
            |opts| {
                let lambda = opts
                    .get("lambda")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let drop_rate = opts
                    .get("drop_rate")
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                Ok(Box::new(TaskArithmeticPass { lambda, drop_rate }))
            },
        );
        reg.register(
            "fold-broadcast",
            "Composes chained broadcast_in_dim ops, removes identity broadcasts, hoists unary ops",
            |_opts| Ok(Box::new(FoldBroadcastPass)),
        );
        reg.register(
            "fold-slice-concat",
            "Composes nested slices and folds slices of concatenate ops directly to source inputs",
            |_opts| Ok(Box::new(FoldSliceConcatPass)),
        );
        reg.register(
            "fuse-horizontal-dot",
            "Horizontally fuses parallel dot_general ops sharing an LHS input (e.g. Q/K/V)",
            |_opts| Ok(Box::new(FuseHorizontalDotPass)),
        );
        reg.register(
            "fold-convert",
            "Eliminates redundant or round-trip stablehlo.convert casts and folds constant casts",
            |_opts| Ok(Box::new(FoldConvertPass)),
        );
        reg.register(
            "hoist-constants",
            "Hoists stablehlo.constant ops to block entry and deduplicates identical constants",
            |_opts| Ok(Box::new(HoistConstantsPass)),
        );
        reg.register(
            "licm",
            "Loop/Region-Invariant Code Motion (hoists invariant pure ops out of nested regions)",
            |_opts| Ok(Box::new(LicmPass)),
        );
        reg.register(
            "fuse-swiglu",
            "Fuses SwiGLU (gate / (1 + exp(-gate)) * up) into stablehlo.composite (fused_swiglu)",
            |_opts| Ok(Box::new(FuseSwiGluPass)),
        );
        reg.register(
            "specialize-shapes",
            "Binds dynamic/symbolic dimensions (?, B, S) and propagates static shapes across IR",
            |opts| {
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
                Ok(Box::new(SpecializeShapesPass {
                    default_dim,
                    bindings,
                }))
            },
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
