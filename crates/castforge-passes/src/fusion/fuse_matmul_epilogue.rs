use crate::fusion::fuse_gelu::match_gelu_op;
use crate::fusion::fuse_swiglu::match_silu_val;
use crate::{DeadCodeEliminationPass, Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, OpName, ValueDef};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// AI Optimization Pass: Generalized MatMul Epilogue Fusion (`fuse-matmul-epilogue`).
///
/// Fuses `stablehlo.dot_general` + bias `stablehlo.add` + optional chained elementwise
/// consumers (`relu`, `tanh`, `silu`, or `scale` multiply) into a single `stablehlo.composite`
/// operation (`"castforge.fused_matmul_epilogue"`), eliminating intermediate memory traffic.
pub struct FuseMatmulEpiloguePass;

impl Pass for FuseMatmulEpiloguePass {
    fn name(&self) -> &'static str {
        "fuse-matmul-epilogue"
    }

    fn description(&self) -> &'static str {
        "Fuses dot_general + bias add (+ relu/tanh/silu/scale epilogue) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let mut removed = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            // Pass 1: Look for multi-op epilogue chains rooted at `dot_general + bias`:
            // - `silu(add(dot, bias))` (`bias_silu`)
            // - `maximum(add(dot, bias), zero)` (`bias_relu`)
            // - `tanh(add(dot, bias))` (`bias_tanh`)
            // - `multiply(add(dot, bias), scale)` (`bias_scale`)
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(hlo) = StableHloOp::from_op(op) else {
                    continue;
                };

                // Check `silu(add(dot, bias))` rooted at `Divide`
                if let StableHloOp::Divide { result, .. } = hlo
                    && let Some(add_val) = match_silu_val(module, result)
                    && let Some(MatmulAddMatch {
                        lhs,
                        rhs,
                        bias,
                        dot_attrs,
                        ..
                    }) = match_add_of_dot_ignoring_uses(module, add_val)
                {
                    let mut attrs = dot_attrs;
                    attrs.insert(
                        "name".to_string(),
                        Attribute::String("castforge.fused_matmul_epilogue".to_string()),
                    );
                    attrs.insert(
                        "decomposition".to_string(),
                        Attribute::SymbolRef("fused_matmul_bias_silu_impl".to_string()),
                    );
                    attrs.insert(
                        "epilogue".to_string(),
                        Attribute::String("bias_silu".to_string()),
                    );

                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "composite");
                    op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
                    op_mut.attributes = attrs;
                    created += 1;
                    continue;
                }

                // Check `gelu(add(dot, bias))` rooted at `Multiply`
                if let Some(add_val) = match_gelu_op(module, op)
                    && let Some(MatmulAddMatch {
                        lhs,
                        rhs,
                        bias,
                        dot_attrs,
                        ..
                    }) = match_add_of_dot_ignoring_uses(module, add_val)
                {
                    let mut attrs = dot_attrs;
                    attrs.insert(
                        "name".to_string(),
                        Attribute::String("castforge.fused_matmul_epilogue".to_string()),
                    );
                    attrs.insert(
                        "decomposition".to_string(),
                        Attribute::SymbolRef("fused_matmul_bias_gelu_impl".to_string()),
                    );
                    attrs.insert(
                        "epilogue".to_string(),
                        Attribute::String("bias_gelu".to_string()),
                    );

                    let op_mut = &mut module.ops[op_id];
                    op_mut.name = OpName::new("stablehlo", "composite");
                    op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
                    op_mut.attributes = attrs;
                    created += 1;
                    continue;
                }

                match hlo {
                    StableHloOp::Maximum {
                        lhs: add_val,
                        rhs: zero_val,
                        ..
                    } => {
                        let Some(MatmulAddMatch {
                            add_op_id,
                            dot_op_id,
                            lhs,
                            rhs,
                            bias,
                            dot_attrs,
                        }) = match_add_of_dot(module, add_val)
                        else {
                            continue;
                        };
                        let mut attrs = dot_attrs;
                        attrs.insert(
                            "name".to_string(),
                            Attribute::String("castforge.fused_matmul_epilogue".to_string()),
                        );
                        attrs.insert(
                            "decomposition".to_string(),
                            Attribute::SymbolRef("fused_matmul_bias_relu_impl".to_string()),
                        );
                        attrs.insert(
                            "epilogue".to_string(),
                            Attribute::String("bias_relu".to_string()),
                        );

                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "composite");
                        op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias, zero_val]);
                        op_mut.attributes = attrs;
                        created += 1;

                        module.remove_op(block_id, add_op_id);
                        module.remove_op(block_id, dot_op_id);
                        removed += 2;
                    }
                    StableHloOp::Tanh {
                        operand: add_val, ..
                    } => {
                        let Some(MatmulAddMatch {
                            add_op_id,
                            dot_op_id,
                            lhs,
                            rhs,
                            bias,
                            dot_attrs,
                        }) = match_add_of_dot(module, add_val)
                        else {
                            continue;
                        };
                        let mut attrs = dot_attrs;
                        attrs.insert(
                            "name".to_string(),
                            Attribute::String("castforge.fused_matmul_epilogue".to_string()),
                        );
                        attrs.insert(
                            "decomposition".to_string(),
                            Attribute::SymbolRef("fused_matmul_bias_tanh_impl".to_string()),
                        );
                        attrs.insert(
                            "epilogue".to_string(),
                            Attribute::String("bias_tanh".to_string()),
                        );

                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "composite");
                        op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
                        op_mut.attributes = attrs;
                        created += 1;

                        module.remove_op(block_id, add_op_id);
                        module.remove_op(block_id, dot_op_id);
                        removed += 2;
                    }
                    StableHloOp::Multiply {
                        lhs: m_lhs,
                        rhs: m_rhs,
                        ..
                    } => {
                        let matched = match_add_of_dot(module, m_lhs)
                            .map(|m| (m, m_rhs))
                            .or_else(|| match_add_of_dot(module, m_rhs).map(|m| (m, m_lhs)));
                        let Some((
                            MatmulAddMatch {
                                add_op_id,
                                dot_op_id,
                                lhs,
                                rhs,
                                bias,
                                dot_attrs,
                            },
                            scale_val,
                        )) = matched
                        else {
                            continue;
                        };
                        let mut attrs = dot_attrs;
                        attrs.insert(
                            "name".to_string(),
                            Attribute::String("castforge.fused_matmul_epilogue".to_string()),
                        );
                        attrs.insert(
                            "decomposition".to_string(),
                            Attribute::SymbolRef("fused_matmul_bias_scale_impl".to_string()),
                        );
                        attrs.insert(
                            "epilogue".to_string(),
                            Attribute::String("bias_scale".to_string()),
                        );

                        let op_mut = &mut module.ops[op_id];
                        op_mut.name = OpName::new("stablehlo", "composite");
                        op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias, scale_val]);
                        op_mut.attributes = attrs;
                        created += 1;

                        module.remove_op(block_id, add_op_id);
                        module.remove_op(block_id, dot_op_id);
                        removed += 2;
                    }
                    _ => {}
                }
            }

            if created > 0 {
                let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
                removed += dce_res.ops_removed;
            }

            // Pass 2: Look for remaining `add(dot_general(x, w), bias)` without activation
            let ops2 = module.block_ops(block_id);
            for op_id in ops2 {
                let Some(op) = module.ops.get(op_id) else {
                    continue;
                };
                let Some(StableHloOp::Add { result, .. }) = StableHloOp::from_op(op) else {
                    continue;
                };
                let Some(MatmulAddMatch {
                    dot_op_id,
                    lhs,
                    rhs,
                    bias,
                    dot_attrs,
                    ..
                }) = match_add_of_dot_ignoring_uses(module, result)
                else {
                    continue;
                };

                let mut attrs = dot_attrs;
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.fused_matmul_epilogue".to_string()),
                );
                attrs.insert(
                    "decomposition".to_string(),
                    Attribute::SymbolRef("fused_matmul_bias_impl".to_string()),
                );
                attrs.insert(
                    "epilogue".to_string(),
                    Attribute::String("bias".to_string()),
                );

                let op_mut = &mut module.ops[op_id];
                op_mut.name = OpName::new("stablehlo", "composite");
                op_mut.operands = SmallVec::from_slice(&[lhs, rhs, bias]);
                op_mut.attributes = attrs;
                created += 1;

                module.remove_op(block_id, dot_op_id);
                removed += 1;
            }
        }

        if created > 0 || removed > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
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

struct MatmulAddMatch {
    add_op_id: castforge_ir::OpId,
    dot_op_id: castforge_ir::OpId,
    lhs: castforge_ir::ValueId,
    rhs: castforge_ir::ValueId,
    bias: castforge_ir::ValueId,
    dot_attrs: IndexMap<String, Attribute>,
}

fn match_add_of_dot(module: &Module, add_val: castforge_ir::ValueId) -> Option<MatmulAddMatch> {
    if module.use_count(add_val) != 1 {
        return None;
    }
    match_add_of_dot_ignoring_uses(module, add_val)
}

fn match_add_of_dot_ignoring_uses(
    module: &Module,
    add_val: castforge_ir::ValueId,
) -> Option<MatmulAddMatch> {
    let ValueDef::OpResult { op: add_op_id, .. } = module.values[add_val].def else {
        return None;
    };
    let add_op = module.ops.get(add_op_id)?;
    let StableHloOp::Add { lhs: a, rhs: b, .. } = StableHloOp::from_op(add_op)? else {
        return None;
    };

    // Check if `a` or `b` is a single-use `dot_general`
    for (dot_cand, bias_cand) in [(a, b), (b, a)] {
        if module.use_count(dot_cand) != 1 {
            continue;
        }
        let ValueDef::OpResult { op: dot_op_id, .. } = module.values[dot_cand].def else {
            continue;
        };
        let Some(dot_op) = module.ops.get(dot_op_id) else {
            continue;
        };
        if let Some(StableHloOp::DotGeneral { lhs, rhs, .. }) = StableHloOp::from_op(dot_op) {
            return Some(MatmulAddMatch {
                add_op_id,
                dot_op_id,
                lhs,
                rhs,
                bias: bias_cand,
                dot_attrs: dot_op.attributes.clone(),
            });
        }
    }
    None
}
