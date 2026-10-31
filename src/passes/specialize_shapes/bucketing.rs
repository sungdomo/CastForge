//! Static Batch Dimension Resizer (`resize-batch`) and Multi-Bucket Shape Cloner
//! (`bucket-shapes`).

use super::{
    SpecializeShapesPass, collect_non_batch_weight_operands, specialize_and_resize_single_traversal,
};
use crate::ir::{Attribute, Dim, Module, OpName, Type, ValueId};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use smallvec::SmallVec;
use std::collections::HashMap;

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
        let dummy = SpecializeShapesPass::default();
        let modified =
            specialize_and_resize_single_traversal(module, &dummy, false, Some(self.batch), None);
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}

/// Multi-Bucket Shape Cloner Pass (`bucket-shapes`).
#[derive(Debug, Clone)]
pub struct BucketShapesPass {
    /// Semicolon/comma-separated batch sizes to specialize (default `"1;4;8"`).
    pub buckets: String,
}

impl Default for BucketShapesPass {
    fn default() -> Self {
        Self {
            buckets: "1;4;8".to_string(),
        }
    }
}

impl Pass for BucketShapesPass {
    fn name(&self) -> &'static str {
        "bucket-shapes"
    }

    fn description(&self) -> &'static str {
        "Clones the entry function into batch-specialized bucket variants (@fn_batch_B)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let Some(&src_fn_id) = module.functions().first() else {
            return Ok(PassResult::unchanged(self.name()));
        };
        let batch_sizes: Vec<i64> = self
            .buckets
            .split([',', ';'])
            .filter_map(|s| s.trim().parse::<i64>().ok())
            .filter(|&b| b >= 1)
            .collect();
        if batch_sizes.is_empty() {
            return Ok(PassResult::unchanged(self.name()));
        }

        let src_sym = module.ops[src_fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("main")
            .to_string();
        let src_reg = module.ops[src_fn_id].regions[0];
        let src_blk = module.regions[src_reg].blocks[0];
        let src_args = module.blocks[src_blk].arguments.clone();
        let src_ops = module.blocks[src_blk].ops.clone();

        let old_batch = src_args
            .first()
            .and_then(|&a| module.values[a].ty.as_tensor())
            .and_then(|t| match t.shape.first() {
                Some(Dim::Static(d)) => Some(*d),
                _ => None,
            })
            .unwrap_or(1);

        let weight_rhs_vals = collect_non_batch_weight_operands(module, &src_ops);

        let mod_body = module.body_block();
        let mut created_fns = 0usize;

        for &b_size in &batch_sizes {
            let dst_blk = module.create_block(None);
            let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
            let mut in_types = Vec::with_capacity(src_args.len());

            for &arg_id in &src_args {
                let ssa = module.values[arg_id].ssa_name.clone();
                let mut ty = module.values[arg_id].ty.clone();
                if !weight_rhs_vals.contains(&arg_id) {
                    resize_type_batch(&mut ty, old_batch, b_size);
                }
                in_types.push(ty.clone());
                let new_arg = module.add_block_argument(dst_blk, ssa, ty);
                val_map.insert(arg_id, new_arg);
            }

            let mut out_types = Vec::new();
            for &op_id in &src_ops {
                let op = module.ops[op_id].clone();
                let mapped_operands: SmallVec<[ValueId; 4]> = op
                    .operands
                    .iter()
                    .map(|&v| val_map.get(&v).copied().unwrap_or(v))
                    .collect();

                if op.is_op("func", "return") {
                    for &opnd in &mapped_operands {
                        out_types.push(module.values[opnd].ty.clone());
                    }
                }

                let mut res_specs = Vec::with_capacity(op.results.len());
                for &old_r in &op.results {
                    let ssa = module.values[old_r].ssa_name.clone();
                    let mut r_ty = module.values[old_r].ty.clone();
                    if !weight_rhs_vals.contains(&old_r) {
                        resize_type_batch(&mut r_ty, old_batch, b_size);
                    }
                    res_specs.push((ssa, r_ty));
                }

                let mut attrs = op.attributes.clone();
                if let Some(Attribute::DenseElements { ty, .. }) = attrs.get_mut("value")
                    && op
                        .results
                        .first()
                        .is_some_and(|r| !weight_rhs_vals.contains(r))
                    && !ty.shape.is_empty()
                    && ty.shape[0] == Dim::Static(old_batch)
                {
                    ty.shape[0] = Dim::Static(b_size);
                }

                let cloned_regions = module.clone_regions_local(&op.regions, &mut val_map);
                let new_op_id = module.add_operation(
                    dst_blk,
                    op.name,
                    mapped_operands,
                    res_specs,
                    attrs,
                    cloned_regions,
                    op.custom_syntax,
                    op.location,
                );
                for (&old_r, &new_r) in op.results.iter().zip(module.ops[new_op_id].results.iter())
                {
                    val_map.insert(old_r, new_r);
                }
            }

            let dst_reg = module.create_region(vec![dst_blk]);
            let mut fn_attrs = module.ops[src_fn_id].attributes.clone();
            fn_attrs.insert(
                "sym_name".to_string(),
                Attribute::String(format!("{src_sym}_batch_{b_size}")),
            );
            fn_attrs.insert(
                "function_type".to_string(),
                Attribute::Type(Type::Function {
                    inputs: in_types,
                    results: out_types,
                }),
            );
            let loc = module.ops[src_fn_id].location.clone();
            module.add_operation(
                mod_body,
                OpName::new("func", "func"),
                SmallVec::new(),
                Vec::new(),
                fn_attrs,
                vec![dst_reg],
                true,
                loc,
            );
            created_fns += 1;
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created_fns > 0,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: created_fns,
        })
    }
}

fn resize_type_batch(ty: &mut Type, old_batch: i64, new_batch: i64) {
    if let Type::Tensor(ref mut t) = *ty
        && !t.shape.is_empty()
        && (t.shape[0] == Dim::Static(old_batch)
            || matches!(t.shape[0], Dim::Dynamic | Dim::Symbolic(_)))
    {
        t.shape[0] = Dim::Static(new_batch);
    }
}
