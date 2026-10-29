use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_dialect_stablehlo::StableHloOp;
use castforge_ir::{Attribute, BlockId, Module, ValueDef};

/// Zero-Copy Concatenation Elimination Pass (`eliminate-concat-copies`).
///
/// Analyzes `stablehlo.concatenate` operations and annotates their pure producer ops with
/// destination-slice offsets (`castforge.concat_dest`, `castforge.concat_axis`,
/// `castforge.concat_elem_offset`) so codegen and the arena memory planner write producer results
/// directly into the destination buffer without an intermediate copy.
#[derive(Default)]
pub struct EliminateConcatCopiesPass;

impl Pass for EliminateConcatCopiesPass {
    fn name(&self) -> &'static str {
        "eliminate-concat-copies"
    }

    fn description(&self) -> &'static str {
        "Eliminates intermediate concatenate copies by routing producer outputs into dest slices"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        let blocks: Vec<BlockId> = module.blocks.keys().collect();

        for block_id in blocks {
            let ops = module.block_ops(block_id);
            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                let Some(StableHloOp::Concatenate {
                    inputs,
                    dimension,
                    result,
                }) = StableHloOp::from_op(&op)
                else {
                    continue;
                };
                if inputs.is_empty() {
                    continue;
                }

                let dest_name = module.values[result]
                    .ssa_name
                    .clone()
                    .unwrap_or_else(|| format!("{result:?}"));
                let mut elem_offset = 0i64;
                let mut routed_any = false;
                for &inp in &inputs {
                    let inp_elems = module.values[inp]
                        .ty
                        .as_tensor()
                        .and_then(|t| t.num_elements())
                        .unwrap_or(0);
                    if let ValueDef::OpResult { op: prod_id, .. } = module.values[inp].def
                        && let Some(prod_op) = module.ops.get_mut(prod_id)
                        && prod_op.is_pure()
                        && !prod_op.is_op("stablehlo", "constant")
                    {
                        prod_op.attributes.insert(
                            "castforge.concat_dest".to_string(),
                            Attribute::String(dest_name.clone()),
                        );
                        prod_op.attributes.insert(
                            "castforge.concat_axis".to_string(),
                            Attribute::Int(dimension),
                        );
                        prod_op.attributes.insert(
                            "castforge.concat_elem_offset".to_string(),
                            Attribute::Int(elem_offset),
                        );
                        routed_any = true;
                        modified += 1;
                    }
                    elem_offset += inp_elems;
                }

                if routed_any && let Some(cat_op) = module.ops.get_mut(op_id) {
                    cat_op.attributes.insert(
                        "castforge.zero_copy_concat".to_string(),
                        Attribute::Bool(true),
                    );
                }
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
