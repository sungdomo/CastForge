pub mod target_c;
pub mod target_rust;
pub mod target_wgsl;

use crate::target_c::emit_c_module;
use crate::target_rust::emit_rust_module;
use crate::target_wgsl::emit_wgsl_module;
use anyhow::Result;
use castforge_ir::{Dim, Module, TensorType, ValueDef, ValueId};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenTarget {
    Rust,
    C,
    Wgsl,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferAllocation {
    pub name: String,
    pub num_elements: usize,
    pub byte_size: usize,
    pub is_constant: bool,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryPlan {
    pub allocations: Vec<BufferAllocation>,
    pub total_constant_bytes: usize,
    pub total_activation_bytes: usize,
}

/// Computes a static memory and buffer allocation plan for `module`.
pub fn build_memory_plan(module: &Module) -> MemoryPlan {
    let mut plan = MemoryPlan::default();
    for (val_id, val) in &module.values {
        let Some(t_ty) = val.ty.as_tensor() else {
            continue;
        };
        let elems = t_ty.num_elements().unwrap_or(1).max(1) as usize;
        let bytes = (elems * t_ty.dtype.bit_width()).div_ceil(8);
        let is_const = match val.def {
            ValueDef::OpResult { op, .. } => module
                .ops
                .get(op)
                .is_some_and(|o| o.is_op("stablehlo", "constant")),
            _ => false,
        };
        let name = val
            .ssa_name
            .clone()
            .unwrap_or_else(|| format!("v_{:?}", val_id));
        if is_const {
            plan.total_constant_bytes += bytes;
        } else {
            plan.total_activation_bytes += bytes;
        }
        plan.allocations.push(BufferAllocation {
            name,
            num_elements: elems,
            byte_size: bytes,
            is_constant: is_const,
        });
    }
    plan
}

/// Generate executable source code from a CastForge `Module` for the requested `CodegenTarget`.
pub fn generate_code(module: &Module, target: CodegenTarget) -> Result<String> {
    match target {
        CodegenTarget::Rust => emit_rust_module(module),
        CodegenTarget::C => emit_c_module(module),
        CodegenTarget::Wgsl => emit_wgsl_module(module),
    }
}

pub(crate) fn sanitize_ident(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "v_anon".to_string()
    } else if cleaned.chars().next().unwrap().is_ascii_digit() {
        format!("v_{cleaned}")
    } else {
        cleaned
    }
}

pub(crate) fn value_ident(
    module: &Module,
    val_id: ValueId,
    names: &mut HashMap<ValueId, String>,
    counter: &mut usize,
) -> String {
    if let Some(existing) = names.get(&val_id) {
        return existing.clone();
    }
    let ident = if let Some(ssa) = &module.values[val_id].ssa_name {
        sanitize_ident(ssa)
    } else {
        *counter += 1;
        format!("v_{counter}")
    };
    names.insert(val_id, ident.clone());
    ident
}

pub(crate) fn tensor_len(module: &Module, val_id: ValueId) -> usize {
    module.values[val_id]
        .ty
        .as_tensor()
        .and_then(|t| t.num_elements())
        .unwrap_or(1)
        .max(1) as usize
}

pub(crate) fn tensor_shape_dims(t: &TensorType) -> Vec<usize> {
    t.shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) if *n > 0 => *n as usize,
            _ => 1,
        })
        .collect()
}
