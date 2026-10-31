//! Spatial & Matrix Layout Conversion Pass (`convert-layout`).

use super::fold_all_views_single_pass;
use crate::ir::{Attribute, Module, OpName, TensorType, Type, ValueDef};
use crate::passes::{Pass, PassResult};
use anyhow::Result;
use indexmap::IndexMap;
use smallvec::smallvec;

/// Spatial & Matrix Layout Conversion Pass (`convert-layout`).
#[derive(Debug, Clone)]
pub struct ConvertLayoutPass {
    pub layout: String,
    pub propagate: bool,
}

impl Default for ConvertLayoutPass {
    fn default() -> Self {
        Self {
            layout: "nhwc".to_string(),
            propagate: false,
        }
    }
}

impl Pass for ConvertLayoutPass {
    fn name(&self) -> &'static str {
        "convert-layout"
    }

    fn description(&self) -> &'static str {
        "Converts spatial/matrix layouts (nhwc/nchw/col_major) and propagates layout across ops"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let tgt = self.layout.trim().to_ascii_lowercase();
        let tgt_layout = if tgt.is_empty() { "nhwc" } else { tgt.as_str() };
        run_convert_layout(module, tgt_layout, self.propagate, self.name())
    }
}

pub(crate) fn run_convert_layout(
    module: &mut Module,
    tgt_layout: &str,
    propagate: bool,
    pass_name: &str,
) -> Result<PassResult> {
    let mut created = 0usize;
    let mut modified = 0usize;
    let mut removed = 0usize;

    for fn_id in module.functions() {
        let reg_id = module.ops[fn_id].regions[0];
        let block_id = module.regions[reg_id].blocks[0];
        let ops = module.blocks[block_id].ops.clone();

        for op_id in ops {
            let Some(op) = module.ops.get(op_id).cloned() else {
                continue;
            };
            if !op.is_op("stablehlo", "convolution") && !op.is_op("stablehlo", "dot_general") {
                continue;
            };
            if op.attributes.get("data_layout").and_then(|a| a.as_str()) == Some(tgt_layout) {
                continue;
            }
            let Some(&in_val) = op.operands.first() else {
                continue;
            };
            let Some(in_tensor) = module.values[in_val].ty.as_tensor().cloned() else {
                continue;
            };

            module.ops[op_id].attributes.insert(
                "data_layout".to_string(),
                Attribute::String(tgt_layout.to_string()),
            );
            if propagate {
                module.ops[op_id].attributes.insert(
                    "castforge.layout_propagated".to_string(),
                    Attribute::Bool(true),
                );
            }
            modified += 1;

            if in_tensor.shape.len() == 2 {
                let mid_ty = Type::Tensor(TensorType {
                    shape: vec![in_tensor.shape[1].clone(), in_tensor.shape[0].clone()],
                    dtype: in_tensor.dtype.clone(),
                    encoding: None,
                });
                let orig_ty = Type::Tensor(in_tensor);
                let mut p_attrs = IndexMap::new();
                p_attrs.insert(
                    "permutation".to_string(),
                    Attribute::DenseIntArray(vec![1, 0]),
                );

                let t1_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "transpose"),
                    smallvec![in_val],
                    vec![(Some("layout_in".to_string()), mid_ty)],
                    p_attrs.clone(),
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let t1_res = module.ops[t1_id].results[0];

                let t2_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "transpose"),
                    smallvec![t1_res],
                    vec![(Some("layout_restored".to_string()), orig_ty)],
                    p_attrs,
                    Vec::new(),
                    true,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let t2_res = module.ops[t2_id].results[0];

                module.ops[op_id].operands[0] = t2_res;
                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id].ops.splice(pos..pos, [t1_id, t2_id]);
                }
                created += 2;
            }
        }
    }

    if propagate {
        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.name.dialect != "stablehlo" || op.is_op("stablehlo", "constant") {
                    continue;
                }
                let has_layout_input = op.operands.iter().any(|&opnd| {
                    if let ValueDef::OpResult { op: def_op, .. } = module.values[opnd].def {
                        module
                            .ops
                            .get(def_op)
                            .is_some_and(|d| d.attributes.contains_key("data_layout"))
                    } else {
                        false
                    }
                });
                if has_layout_input && !op.attributes.contains_key("data_layout") {
                    module.ops[op_id].attributes.insert(
                        "data_layout".to_string(),
                        Attribute::String(tgt_layout.to_string()),
                    );
                    module.ops[op_id].attributes.insert(
                        "castforge.layout_propagated".to_string(),
                        Attribute::Bool(true),
                    );
                    modified += 1;
                }
            }
        }
        let (c, m, r) = fold_all_views_single_pass(module)?;
        created += c;
        modified += m;
        removed += r;
    }

    Ok(PassResult {
        pass_name: pass_name.to_string(),
        changed: modified > 0 || created > 0 || removed > 0,
        ops_removed: removed,
        ops_modified: modified,
        ops_created: created,
    })
}
