//! Activation, Gated-MLP, and Mixture-of-Experts (MoE) fusion (`FuseActivationsPass`,
//! `FuseSwiGluPass`, `FuseGeluPass`, `FuseGatedMlpPass`, `FuseMoePass`).

use crate::ir::{Attribute, BlockId, Module, OpId, OpName, ValueDef, ValueId};
use crate::passes::fuse_gemm::{
    DotCandidate, emit_horizontal_dot_group, try_collect_horizontal_candidate,
};
use crate::passes::{DeadCodeEliminationPass, FoldConstantsPass, Pass, PassResult};
use crate::stablehlo::StableHloOp;
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::SmallVec;

type HorizGroupKey = (ValueId, usize, usize);

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

    fn child_passes(&self) -> &'static [&'static str] {
        &["fuse-swiglu", "fuse-gelu", "fuse-gated-mlp", "fuse-moe"]
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let k = self.kind.to_ascii_lowercase();
        let res = match k.as_str() {
            "swiglu" => FuseSwiGluPass.run(module)?,
            "gelu" | "geglu" => FuseGeluPass.run(module)?,
            "gated_mlp" | "mlp" => FuseGatedMlpPass.run(module)?,
            "moe" => FuseMoePass {
                shared_expert: self.shared_expert,
                top_k: self.top_k,
                capacity_factor: self.capacity_factor,
            }
            .run(module)?,
            _ => {
                let steps = [
                    FuseGatedMlpPass.run(module)?,
                    FuseMoePass {
                        shared_expert: self.shared_expert,
                        top_k: self.top_k,
                        capacity_factor: self.capacity_factor,
                    }
                    .run(module)?,
                ];
                PassResult::aggregate(self.name(), &steps)
            }
        };
        Ok(res.with_name(self.name()))
    }
}

/// Executes horizontal gate/up dot pairing, SwiGLU, GELU/GeGLU, Gated-MLP, MoE, `add+max`, and
/// `map+reduce` fusion in a single top-down traversal per block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fuse_elementwise_and_activations_single_traversal(
    module: &mut Module,
    do_mlp: bool,
    do_swiglu: bool,
    do_gelu: bool,
    do_moe: bool,
    do_add_max: bool,
    do_map_reduce: bool,
    shared_expert: bool,
    top_k: usize,
    capacity_factor: f32,
) -> (usize, usize, usize) {
    let mut created = 0usize;
    let mut modified = 0usize;
    let mut removed = 0usize;
    let blocks: Vec<BlockId> = module.blocks.keys().collect();

    for block_id in blocks {
        let ops = module.block_ops(block_id);
        let mut horiz_groups: IndexMap<HorizGroupKey, Vec<DotCandidate>> = IndexMap::new();

        for op_id in ops {
            if !module.ops.contains_key(op_id) {
                continue;
            }
            if do_mlp
                && let Some(op) = module.ops.get(op_id)
                && let Some((key, cand)) = try_collect_horizontal_candidate(module, op)
            {
                horiz_groups.entry(key).or_default().push(cand);
                continue;
            }

            if (do_swiglu && try_fuse_swiglu_at_op(module, op_id))
                || (do_gelu && try_fuse_gelu_or_geglu_at_op(module, op_id))
            {
                created += 1;
            }

            if do_mlp && is_swiglu_or_geglu_composite(module, op_id) {
                flush_horizontal_groups(
                    module,
                    block_id,
                    &mut horiz_groups,
                    &mut created,
                    &mut modified,
                );
                if try_fuse_gated_mlp_at_op(module, op_id) {
                    created += 1;
                    modified += 1;
                }
            }

            if do_moe && try_fuse_moe_at_op(module, op_id, shared_expert, top_k, capacity_factor) {
                created += 1;
                modified += 1;
            }

            if do_add_max && super::try_fuse_add_max_at_op(module, block_id, op_id) {
                created += 1;
                removed += 1;
            }

            if do_map_reduce && super::try_fuse_map_reduce_at_op(module, op_id) {
                created += 1;
            }
        }

        if do_mlp {
            flush_horizontal_groups(
                module,
                block_id,
                &mut horiz_groups,
                &mut created,
                &mut modified,
            );
        }
    }

    (created, modified, removed)
}

fn flush_horizontal_groups(
    module: &mut Module,
    block_id: BlockId,
    horiz_groups: &mut IndexMap<HorizGroupKey, Vec<DotCandidate>>,
    created: &mut usize,
    modified: &mut usize,
) {
    for ((lhs_val, m, k), group) in horiz_groups.drain(..) {
        if group.len() >= 2 {
            let (c, mo) = emit_horizontal_dot_group(module, block_id, lhs_val, m, k, &group);
            *created += c;
            *modified += mo;
        }
    }
}

fn is_swiglu_or_geglu_composite(module: &Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::Composite { name, .. }) = StableHloOp::from_op(op) else {
        return false;
    };
    name == "castforge.fused_swiglu" || name == "castforge.fused_geglu"
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
        let (created, _, mut removed) = fuse_elementwise_and_activations_single_traversal(
            module, false, true, false, false, false, false, false, 2, 1.25,
        );
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

fn try_fuse_swiglu_at_op(module: &mut Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
        return false;
    };

    let matched = match_silu_val(module, lhs)
        .map(|g| (g, rhs))
        .or_else(|| match_silu_val(module, rhs).map(|g| (g, lhs)));
    let Some((gate_val, up_val)) = matched else {
        return false;
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
    true
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
    if let Some(StableHloOp::Negate { operand, .. }) = StableHloOp::from_op(sub_op) {
        return (operand == gate).then_some(gate);
    }
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
        let (created, _, mut removed) = fuse_elementwise_and_activations_single_traversal(
            module, false, false, true, false, false, false, false, 2, 1.25,
        );
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

fn try_fuse_gelu_or_geglu_at_op(module: &mut Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id) else {
        return false;
    };
    let Some(StableHloOp::Multiply { lhs, rhs, .. }) = StableHloOp::from_op(op) else {
        return false;
    };
    if let Some(x_val) = match_gelu_op(module, op) {
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
        return true;
    }

    let matched = match_gelu_val(module, lhs)
        .map(|g| (g, rhs))
        .or_else(|| match_gelu_val(module, rhs).map(|g| (g, lhs)));
    let Some((gate_val, up_val)) = matched else {
        return false;
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
    true
}

/// Matches a value defined by a Tanh-Approximation GELU subgraph or an already-fused
/// `castforge.fused_gelu` composite and returns its input `x`.
pub fn match_gelu_val(module: &Module, val_id: ValueId) -> Option<ValueId> {
    let ValueDef::OpResult { op: op_id, .. } = module.values[val_id].def else {
        return None;
    };
    let op = module.ops.get(op_id)?;
    if let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(op)
        && name == "castforge.fused_gelu"
        && let Some(&x) = inputs.first()
    {
        return Some(x);
    }
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
        let (created, modified, mut removed) = fuse_elementwise_and_activations_single_traversal(
            module, true, true, true, false, false, false, false, 2, 1.25,
        );
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

fn try_fuse_gated_mlp_at_op(module: &mut Module, op_id: OpId) -> bool {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return false;
    };
    let Some(StableHloOp::Composite { name, inputs, .. }) = StableHloOp::from_op(&op) else {
        return false;
    };
    let act_kind = if name == "castforge.fused_swiglu" {
        "swiglu"
    } else if name == "castforge.fused_geglu" {
        "geglu"
    } else {
        return false;
    };
    if inputs.len() < 2 {
        return false;
    }

    let Some((dot_val_0, start_col_0)) = match_2d_col_slice(module, inputs[0]) else {
        return false;
    };
    let Some((dot_val_1, start_col_1)) = match_2d_col_slice(module, inputs[1]) else {
        return false;
    };
    if dot_val_0 != dot_val_1 || start_col_0 != 0 || start_col_1 <= 0 {
        return false;
    }

    let ValueDef::OpResult { op: dot_op_id, .. } = module.values[dot_val_0].def else {
        return false;
    };
    let Some(dot_op) = module.ops.get(dot_op_id) else {
        return false;
    };
    let Some(StableHloOp::DotGeneral {
        lhs: x_val,
        rhs: w_gate_up,
        dimension_numbers: dims,
        ..
    }) = StableHloOp::from_op(dot_op)
    else {
        return false;
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
    true
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
        let (created, modified, mut removed) = fuse_elementwise_and_activations_single_traversal(
            module,
            false,
            false,
            false,
            true,
            false,
            false,
            self.shared_expert,
            self.top_k,
            self.capacity_factor,
        );
        if created > 0 {
            let dce = DeadCodeEliminationPass::ssa_only().run(module)?;
            removed += dce.ops_removed;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: removed,
            ops_modified: modified,
            ops_created: created,
        })
    }
}

fn try_fuse_moe_at_op(
    module: &mut Module,
    op_id: OpId,
    shared_expert: bool,
    top_k: usize,
    capacity_factor: f32,
) -> bool {
    let Some(op) = module.ops.get(op_id).cloned() else {
        return false;
    };
    let Some(StableHloOp::Add { lhs, rhs, .. }) = StableHloOp::from_op(&op) else {
        return false;
    };
    let Some((x0, w0, g0)) = match_gated_expert(module, lhs) else {
        return false;
    };
    let Some((x1, w1, g1)) = match_gated_expert(module, rhs) else {
        return false;
    };
    if x0 != x1 {
        return false;
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
    true
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
