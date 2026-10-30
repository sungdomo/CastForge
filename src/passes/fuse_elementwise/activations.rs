//! Activation, Gated-MLP, and Mixture-of-Experts (MoE) fusion (`FuseActivationsPass`,
//! `FuseSwiGluPass`, `FuseGeluPass`, `FuseGatedMlpPass`, `FuseMoePass`).

use crate::ir::{Attribute, BlockId, Module, OpName, ValueDef, ValueId};
use crate::passes::fuse_gemm::fuse_horizontal_dot_no_dce;
use crate::passes::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Unified Activation, Gated-MLP & Mixture-of-Experts (MoE) Fusion Pass (`fuse-activations`).
pub struct FuseActivationsPass {
    pub kind: String,
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for FuseActivationsPass {
    fn default() -> Self {
        Self {
            kind: "all".to_string(),
            shared_expert: false,
            top_k: 2,
            capacity_factor: 1.25,
        }
    }
}

impl Pass for FuseActivationsPass {
    fn name(&self) -> &'static str {
        "fuse-activations"
    }

    fn description(&self) -> &'static str {
        "Unified activation, gated-MLP & MoE fusion (SwiGLU, GELU, GeGLU, GatedMLP, MoE)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let do_mlp = k == "all" || k == "gated_mlp" || k == "mlp";
        let do_swiglu = do_mlp || k == "swiglu";
        let do_gelu = do_mlp || k == "gelu" || k == "geglu";
        let do_moe = k == "all" || k == "moe" || self.shared_expert;

        let mut created = 0usize;
        let mut modified = 0usize;
        let mut removed = 0usize;

        if do_mlp {
            let (hc, hm) = fuse_horizontal_dot_no_dce(module);
            created += hc;
            modified += hm;
        }
        if do_swiglu {
            created += fuse_swiglu_no_dce(module);
        }
        if do_gelu {
            created += fuse_gelu_no_dce(module);
        }
        if do_mlp {
            let mlp_c = fuse_gated_mlp_no_dce(module);
            created += mlp_c;
            modified += mlp_c;
        }
        if do_moe {
            let moe_c =
                fuse_moe_no_dce(module, self.shared_expert, self.top_k, self.capacity_factor);
            created += moe_c;
            modified += moe_c;
        }

        if created > 0 || modified > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || modified > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

/// SwiGLU Fusion Pass (`fuse-swiglu`).
pub struct FuseSwiGluPass;

impl Pass for FuseSwiGluPass {
    fn name(&self) -> &'static str {
        "fuse-swiglu"
    }

    fn description(&self) -> &'static str {
        "Fuses SwiGLU (gate / (1 + exp(-gate)) * up) into stablehlo.composite (fused_swiglu)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let created = fuse_swiglu_no_dce(module);
        let mut removed = 0usize;
        if created > 0 {
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

fn fuse_swiglu_no_dce(module: &mut Module) -> usize {
    let mut created = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                continue;
            };

            let matched = match_silu_val(module, lhs)
                .map(|g| (g, rhs))
                .or_else(|| match_silu_val(module, rhs).map(|g| (g, lhs)));
            let Some((gate_val, up_val)) = matched else {
                continue;
            };

            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_swiglu".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_swiglu_impl".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[gate_val, up_val]);
            op_mut.attributes = attrs;
            created += 1;
        }
    }
    created
}

/// Matches `silu(x) = divide(x, add(1.0, exponential(subtract(0.0, x))))` and returns `Some(x)`.
pub fn match_silu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: div_op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let div_op = module.ops.get(div_op_id)?;
    let StableHloOp::Divide {
        lhs: gate,
        rhs: denom,
        ..
    } = StableHloOp::from_op(div_op)?
    else {
        return None;
    };

    let ValueDef::OpResult { op: add_op_id, .. } = module.values[denom].def else {
        return None;
    };
    let add_op = module.ops.get(add_op_id)?;
    let StableHloOp::Add {
        lhs: a1, rhs: a2, ..
    } = StableHloOp::from_op(add_op)?
    else {
        return None;
    };

    let exp_val = if FoldConstantsPass::const_splat(module, a2)
        .is_some_and(|v| (v - 1.0).abs() < 1e-6)
    {
        a1
    } else if FoldConstantsPass::const_splat(module, a1).is_some_and(|v| (v - 1.0).abs() < 1e-6) {
        a2
    } else {
        return None;
    };

    let ValueDef::OpResult { op: exp_op_id, .. } = module.values[exp_val].def else {
        return None;
    };
    let exp_op = module.ops.get(exp_op_id)?;
    let StableHloOp::Exp {
        operand: neg_val, ..
    } = StableHloOp::from_op(exp_op)?
    else {
        return None;
    };

    let ValueDef::OpResult { op: sub_op_id, .. } = module.values[neg_val].def else {
        return None;
    };
    let sub_op = module.ops.get(sub_op_id)?;
    let StableHloOp::Subtract {
        lhs: zero_val,
        rhs: sub_rhs,
        ..
    } = StableHloOp::from_op(sub_op)?
    else {
        return None;
    };

    if sub_rhs != gate {
        return None;
    }
    if !FoldConstantsPass::const_splat(module, zero_val).is_some_and(|v| v.abs() < 1e-7) {
        return None;
    }

    Some(gate)
}

/// GELU & GeGLU Fusion Pass (`fuse-gelu`).
pub struct FuseGeluPass;

impl Pass for FuseGeluPass {
    fn name(&self) -> &'static str {
        "fuse-gelu"
    }

    fn description(&self) -> &'static str {
        "Fuses GELU and gated GeGLU (gelu(gate) * up) into stablehlo.composite"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let created = fuse_gelu_no_dce(module);
        let mut removed = 0usize;
        if created > 0 {
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

fn fuse_gelu_no_dce(module: &mut Module) -> usize {
    let mut created = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id) else {
                continue;
            };
            let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
                continue;
            };
            if match_gelu_op(module, op).is_some() {
                continue;
            }
            let matched = match_gelu_val(module, lhs)
                .map(|g| (g, rhs))
                .or_else(|| match_gelu_val(module, rhs).map(|g| (g, lhs)));
            let Some((gate_val, up_val)) = matched else {
                continue;
            };

            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_geglu".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_geglu_impl".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[gate_val, up_val]);
            op_mut.attributes = attrs;
            created += 1;
        }

        let ops2 = module.block_ops(block_id);
        for op_id in ops2 {
            let Some(x_val) = module
                .ops
                .get(op_id)
                .and_then(|op| match_gelu_op(module, op))
            else {
                continue;
            };

            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_gelu".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_gelu_impl".to_string()),
            );

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[x_val]);
            op_mut.attributes = attrs;
            created += 1;
        }
    }
    created
}

/// Matches a value defined by a Tanh-Approximation GELU subgraph and returns its input `x`.
pub fn match_gelu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    match_gelu_op(module, op)
}

pub fn match_gelu_op(module: &Module, op: &crate::ir::Operation) -> Option<ValueId> {
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(op)? else {
        return None;
    };
    let mut factors = Vec::with_capacity(3);
    collect_mul_factors(module, lhs, &mut factors, 2);
    collect_mul_factors(module, rhs, &mut factors, 2);
    if factors.len() != 3 {
        return None;
    }

    let half_idx = factors.iter().position(|&v| {
        FoldConstantsPass::const_splat(module, v).is_some_and(|c| (c - 0.5).abs() < 1e-4)
    })?;

    for i in 0..3 {
        if i == half_idx {
            continue;
        }
        let other_idx = 3 - half_idx - i;
        let x_cand = factors[i];
        let one_plus_tanh_cand = factors[other_idx];
        if match_one_plus_tanh_of_x(module, one_plus_tanh_cand, x_cand) {
            return Some(x_cand);
        }
    }
    None
}

fn collect_mul_factors(module: &Module, val: ValueId, out: &mut Vec<ValueId>, depth: usize) {
    if depth > 0
        && let ValueDef::OpResult { op: op_id, .. } = module.values[val].def
        && let Some(op) = module.ops.get(op_id)
        && let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op)
    {
        collect_mul_factors(module, lhs, out, depth - 1);
        collect_mul_factors(module, rhs, out, depth - 1);
        return;
    }
    out.push(val);
}

fn match_one_plus_tanh_of_x(module: &Module, val_id: ValueId, x_val: ValueId) -> bool {
    let ValueDef::OpResult { op: add_id, .. } = module.values[val_id].def else {
        return false;
    };
    let Some(add_op) = module.ops.get(add_id) else {
        return false;
    };
    let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(add_op) else {
        return false;
    };
    let tanh_val = if FoldConstantsPass::const_splat(module, lhs)
        .is_some_and(|c| (c - 1.0).abs() < 1e-4)
    {
        rhs
    } else if FoldConstantsPass::const_splat(module, rhs).is_some_and(|c| (c - 1.0).abs() < 1e-4) {
        lhs
    } else {
        return false;
    };

    let ValueDef::OpResult { op: tanh_id, .. } = module.values[tanh_val].def else {
        return false;
    };
    let Some(tanh_op) = module.ops.get(tanh_id) else {
        return false;
    };
    let Some(StableHloOp::Tanh { operand: inner, .. }) = StableHloOp::from_op(tanh_op) else {
        return false;
    };

    let ValueDef::OpResult { op: mul_id, .. } = module.values[inner].def else {
        return false;
    };
    let Some(mul_op) = module.ops.get(mul_id) else {
        return false;
    };
    let Some(StableHloOp::Multiply {
        lhs: m1, rhs: m2, ..
    }) = StableHloOp::from_op(mul_op)
    else {
        return false;
    };
    let poly_val = if FoldConstantsPass::const_splat(module, m1)
        .is_some_and(|c| (c - 0.79788456).abs() < 1e-2)
    {
        m2
    } else if FoldConstantsPass::const_splat(module, m2)
        .is_some_and(|c| (c - 0.79788456).abs() < 1e-2)
    {
        m1
    } else {
        return false;
    };

    let ValueDef::OpResult { op: p_add_id, .. } = module.values[poly_val].def else {
        return false;
    };
    let Some(p_add_op) = module.ops.get(p_add_id) else {
        return false;
    };
    let Some(StableHloOp::Add {
        lhs: p1, rhs: p2, ..
    }) = StableHloOp::from_op(p_add_op)
    else {
        return false;
    };
    let cubic_val = if p1 == x_val {
        p2
    } else if p2 == x_val {
        p1
    } else {
        return false;
    };

    let mut cubic_factors = Vec::new();
    collect_mul_factors(module, cubic_val, &mut cubic_factors, 3);
    if cubic_factors.len() != 4 {
        return false;
    }
    let x_count = cubic_factors.iter().filter(|&&v| v == x_val).count();
    let coeff_ok = cubic_factors.iter().any(|&v| {
        FoldConstantsPass::const_splat(module, v).is_some_and(|c| (c - 0.044715).abs() < 1e-3)
    });
    x_count == 3 && coeff_ok
}

/// Gated MLP Macro-Fusion Pass (`fuse-gated-mlp`).
pub struct FuseGatedMlpPass;

impl Pass for FuseGatedMlpPass {
    fn name(&self) -> &'static str {
        "fuse-gated-mlp"
    }

    fn description(&self) -> &'static str {
        "Fuses dual gate/up dot_general + slices + SwiGLU/GeGLU into fused_gated_mlp"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let (hc, hm) = fuse_horizontal_dot_no_dce(module);
        let sw_c = fuse_swiglu_no_dce(module);
        let ge_c = fuse_gelu_no_dce(module);
        let fused_count = fuse_gated_mlp_no_dce(module);

        let created = hc + sw_c + ge_c + fused_count;
        let mut removed = 0usize;
        if created > 0 || hm > 0 {
            let dce_res = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce_res.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0 || hm > 0,
            ops_removed: removed,
            ops_modified: fused_count + hm,
            ops_created: created,
        })
    }
}

fn fuse_gated_mlp_no_dce(module: &mut Module) -> usize {
    let mut fused_count = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op)
            else {
                continue;
            };
            let act_kind = if name == "castforge.fused_swiglu" {
                "swiglu"
            } else if name == "castforge.fused_geglu" {
                "geglu"
            } else {
                continue;
            };
            if inputs.len() < 2 {
                continue;
            }

            let Some((dot_val_0, start_col_0)) = match_2d_col_slice(module, inputs[0]) else {
                continue;
            };
            let Some((dot_val_1, start_col_1)) = match_2d_col_slice(module, inputs[1]) else {
                continue;
            };
            if dot_val_0 != dot_val_1 || start_col_0 != 0 || start_col_1 <= 0 {
                continue;
            }

            let ValueDef::OpResult { op: dot_op_id, .. } = module.values[dot_val_0].def else {
                continue;
            };
            let Some(dot_op) = module.ops.get(dot_op_id) else {
                continue;
            };
            let Some(StableHloOp::DotGeneral {
                lhs: x_val,
                rhs: w_gate_up,
                dimension_numbers: dims,
                ..
            }) = StableHloOp::from_op(dot_op)
            else {
                continue;
            };
            let packed = dims.rhs_contracting_dimensions == [1];

            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_gated_mlp".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_gated_mlp_impl".to_string()),
            );
            attrs.insert(
                "activation".to_string(),
                Attribute::String(act_kind.to_string()),
            );
            if packed {
                attrs.insert("packed_rhs".to_string(), Attribute::Bool(true));
            }

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[x_val, w_gate_up]);
            op_mut.attributes = attrs;
            fused_count += 1;
        }
    }
    fused_count
}

fn match_2d_col_slice(module: &Module, val_id: ValueId) -> Option<(ValueId, i64)> {
    let ValueDef::OpResult { op: s_id, .. } = module.values[val_id].def else {
        return None;
    };
    let s_op = module.ops.get(s_id)?;
    let StableHloOp::Slice {
        operand,
        start_indices,
        ..
    } = StableHloOp::from_op(s_op)?
    else {
        return None;
    };
    if start_indices.len() == 2 && start_indices[0] == 0 {
        Some((operand, start_indices[1]))
    } else {
        None
    }
}

/// Mixture-of-Experts (MoE) Routing, Shared-Expert & Grouped-GEMM Fusion Pass (`fuse-moe`).
pub struct FuseMoePass {
    pub shared_expert: bool,
    pub top_k: usize,
    pub capacity_factor: f32,
}

impl Default for FuseMoePass {
    fn default() -> Self {
        Self {
            shared_expert: false,
            top_k: 2,
            capacity_factor: 1.25,
        }
    }
}

impl Pass for FuseMoePass {
    fn name(&self) -> &'static str {
        "fuse-moe"
    }

    fn description(&self) -> &'static str {
        "Fuses routed + shared Mixture-of-Experts (MoE) GEMMs and gate combine into fused_moe"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let created = fuse_moe_no_dce(module, self.shared_expert, self.top_k, self.capacity_factor);
        let mut removed = 0usize;
        if created > 0 {
            let dce = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: created,
            ops_created: created,
        })
    }
}

fn fuse_moe_no_dce(
    module: &mut Module,
    shared_expert: bool,
    top_k: usize,
    capacity_factor: f32,
) -> usize {
    let mut created = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op) else {
                continue;
            };
            let Some((x0, w0, g0)) = match_gated_expert(module, lhs) else {
                continue;
            };
            let Some((x1, w1, g1)) = match_gated_expert(module, rhs) else {
                continue;
            };
            if x0 != x1 {
                continue;
            }

            let mut attrs = IndexMap::new();
            attrs.insert(
                "name".to_string(),
                Attribute::String("castforge.fused_moe".to_string()),
            );
            attrs.insert(
                "decomposition".to_string(),
                Attribute::SymbolRef("fused_moe_impl".to_string()),
            );
            attrs.insert("num_experts".to_string(), Attribute::Int(2));
            attrs.insert(
                "castforge.moe_top_k".to_string(),
                Attribute::Int(top_k.max(1) as i64),
            );
            attrs.insert(
                "castforge.moe_capacity_factor".to_string(),
                Attribute::Float(capacity_factor.max(0.5) as f64),
            );
            if shared_expert {
                attrs.insert(
                    "castforge.moe_shared_expert".to_string(),
                    Attribute::Bool(true),
                );
            }

            let op_mut = &mut module.ops[op_id];
            op_mut.name = OpName::new("stablehlo", "composite");
            op_mut.operands = SmallVec::from_slice(&[x0, w0, w1, g0, g1]);
            op_mut.attributes = attrs;
            created += 1;
        }
    }
    created
}

fn match_gated_expert(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId, ValueId)> {
    let ValueDef::OpResult { op: mul_id, .. } = module.values[val_id].def else {
        return None;
    };
    let mul_op = module.ops.get(mul_id)?;
    let StableHloOp::Multiply { lhs, rhs, .. } = StableHloOp::from_op(mul_op)? else {
        return None;
    };
    match_dot_2d(module, lhs)
        .map(|(x, w)| (x, w, rhs))
        .or_else(|| match_dot_2d(module, rhs).map(|(x, w)| (x, w, lhs)))
}

fn match_dot_2d(module: &Module, val_id: ValueId) -> Option<(ValueId, ValueId)> {
    let ValueDef::OpResult { op: dot_id, .. } = module.values[val_id].def else {
        return None;
    };
    let dot_op = module.ops.get(dot_id)?;
    let StableHloOp::DotGeneral { lhs, rhs, .. } = StableHloOp::from_op(dot_op)? else {
        return None;
    };
    Some((lhs, rhs))
}
