//! Static Batch Dimension Resizer Pass (`resize-batch`: `Ir -> Ir`).
//!
//! Rewrites the leading batch dimension (`batch=N`) across activation parameters, intermediate
//! tensors, and function outputs while preserving 2D weight matrix shapes (`[K, N]`) on
//! `stablehlo.dot_general` RHS operands.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Dim, Module, Type, ValueId};
use std::collections::HashSet;

/// Static Batch Dimension Resizer Pass (`resize-batch`).
#[derive(Debug, Clone)]
pub struct ResizeBatchPass {
    /// Target leading batch size (`>= 1`, default `4`).
    pub batch: i64,
}

impl Default for ResizeBatchPass {
    fn default() -> Self {
        Self { batch: 4 }
    }
}

impl Pass for ResizeBatchPass {
    fn name(&self) -> &'static str {
        "resize-batch"
    }

    fn description(&self) -> &'static str {
        "Rewrites the leading batch dimension across activations while preserving 2D GEMM weights"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let new_batch = self.batch.max(1);
        let mut modified = 0usize;

        for fn_op_id in module.functions() {
            let reg_id = module.ops[fn_op_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let args = module.blocks[block_id].arguments.clone();
            let ops = module.blocks[block_id].ops.clone();

            let Some(&first_arg) = args.first() else {
                continue;
            };
            let Some(old_batch) = leading_static_dim(module, first_arg) else {
                continue;
            };
            if old_batch == new_batch {
                continue;
            }

            // Collect all values used as RHS weight matrices of `dot_general`
            let mut weight_rhs_vals: HashSet<ValueId> = HashSet::new();
            for &op_id in &ops {
                let op = &module.ops[op_id];
                if op.is_op("stablehlo", "dot_general") && op.operands.len() >= 2 {
                    weight_rhs_vals.insert(op.operands[1]);
                }
            }

            for &arg_id in &args {
                if weight_rhs_vals.contains(&arg_id) {
                    continue;
                }
                if set_leading_dim(module, arg_id, old_batch, new_batch) {
                    modified += 1;
                }
            }

            for &op_id in &ops {
                if module.ops[op_id].is_op("func", "return") {
                    continue;
                }
                let res_ids: Vec<ValueId> = module.ops[op_id].results.to_vec();
                for res_id in res_ids {
                    if weight_rhs_vals.contains(&res_id) {
                        continue;
                    }
                    if set_leading_dim(module, res_id, old_batch, new_batch) {
                        modified += 1;
                        if let Some(Attribute::DenseElements { ty, .. }) =
                            module.ops[op_id].attributes.get_mut("value")
                            && !ty.shape.is_empty()
                        {
                            ty.shape[0] = Dim::Static(new_batch);
                        }
                    }
                }
            }

            let inputs: Vec<Type> = module.blocks[block_id]
                .arguments
                .iter()
                .map(|&a| module.values[a].ty.clone())
                .collect();
            let results: Vec<Type> = ops
                .iter()
                .find(|&&id| module.ops[id].is_op("func", "return"))
                .map(|&ret_id| {
                    module.ops[ret_id]
                        .operands
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect()
                })
                .unwrap_or_default();
            module.ops[fn_op_id].attributes.insert(
                "function_type".to_string(),
                Attribute::Type(Type::Function { inputs, results }),
            );
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

fn leading_static_dim(module: &Module, vid: ValueId) -> Option<i64> {
    module.values[vid]
        .ty
        .as_tensor()
        .and_then(|t| match t.shape.first() {
            Some(Dim::Static(d)) => Some(*d),
            _ => None,
        })
}

fn set_leading_dim(module: &mut Module, vid: ValueId, old_batch: i64, new_batch: i64) -> bool {
    if let Type::Tensor(ref mut t) = module.values[vid].ty
        && !t.shape.is_empty()
        && t.shape[0] == Dim::Static(old_batch)
    {
        t.shape[0] = Dim::Static(new_batch);
        return true;
    }
    false
}
