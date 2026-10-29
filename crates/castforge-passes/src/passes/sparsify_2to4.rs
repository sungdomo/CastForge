//! NVIDIA 2:4 Semi-Structured Weight Sparsifier Pass (`sparsify-2to4`: `Ir -> Ir`).
//!
//! Enforces exact **2:4 structured sparsity** across weight tensors (preserving at most the 2
//! largest-magnitude elements in every contiguous 4-element block and zeroing the remaining 2)
//! and stamps `sparsity = "2:4"` metadata onto `stablehlo.dot_general` operations.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{
    Attribute, DType, Module, TensorEntry, TensorType, ValueDef, ValueId, format_dense_f32_literal,
};
use std::collections::HashSet;

/// NVIDIA 2:4 Semi-Structured Weight Sparsifier Pass (`sparsify-2to4`).
#[derive(Debug, Clone, Default)]
pub struct Sparsify2To4Pass;

impl Pass for Sparsify2To4Pass {
    fn name(&self) -> &'static str {
        "sparsify-2to4"
    }

    fn description(&self) -> &'static str {
        "Enforces 2:4 semi-structured sparsity on weight tensors and annotates dot_general ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let mut rhs_weights: HashSet<ValueId> = HashSet::new();

        for (_, op) in module.ops.iter_mut() {
            if op.is_op("stablehlo", "dot_general") && op.operands.len() >= 2 {
                rhs_weights.insert(op.operands[1]);
                if op.attributes.get("sparsity").and_then(|a| a.as_str()) != Some("2:4") {
                    op.attributes
                        .insert("sparsity".to_string(), Attribute::String("2:4".to_string()));
                    modified += 1;
                }
            }
        }

        let mut const_ops = Vec::new();
        for &vid in &rhs_weights {
            if let ValueDef::OpResult { op: prod_op, .. } = module.values[vid].def
                && module.ops[prod_op].is_op("stablehlo", "constant")
            {
                const_ops.push((prod_op, vid));
            }
        }

        for (op_id, vid) in const_ops {
            let Some(t_ty) = module.values[vid].ty.as_tensor().cloned() else {
                continue;
            };
            let n = t_ty.num_elements().unwrap_or(1).max(1) as usize;
            let Some(val_attr) = module.ops[op_id].attributes.get("value").cloned() else {
                continue;
            };
            let Some(mut vals) = module.tensor_store.resolve_f32_attr(&val_attr, n) else {
                continue;
            };
            apply_2to4_mask(&mut vals);

            match val_attr {
                Attribute::TensorRef { key, ty } => {
                    module.tensor_store.insert(TensorEntry::from_f32(
                        key.clone(),
                        TensorType::new(ty.shape.clone(), DType::F32),
                        vals,
                    ));
                    modified += 1;
                }
                Attribute::DenseElements { ty, .. } => {
                    let lit = format_dense_f32_literal(&vals, &ty.shape);
                    module.ops[op_id].attributes.insert(
                        "value".to_string(),
                        Attribute::DenseElements {
                            ty,
                            raw_literal: lit,
                        },
                    );
                    modified += 1;
                }
                _ => {}
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

/// Zeroes out the 2 smallest-magnitude elements in every contiguous 4-element chunk of `vals`.
pub fn apply_2to4_mask(vals: &mut [f32]) {
    for chunk in vals.chunks_mut(4) {
        if chunk.len() < 4 {
            continue;
        }
        let mut indexed: [(usize, f32); 4] = [
            (0, chunk[0].abs()),
            (1, chunk[1].abs()),
            (2, chunk[2].abs()),
            (3, chunk[3].abs()),
        ];
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        chunk[indexed[0].0] = 0.0;
        chunk[indexed[1].0] = 0.0;
    }
}
