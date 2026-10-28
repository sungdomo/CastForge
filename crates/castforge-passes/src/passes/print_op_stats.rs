use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Dim, Module};
use indexmap::IndexMap;

/// 19. Print Op Stats Pass (`print-op-stats`)
/// Computes operation frequency, parameter counts, and FLOP estimates across the module.
pub struct PrintOpStatsPass;

impl Pass for PrintOpStatsPass {
    fn name(&self) -> &'static str {
        "print-op-stats"
    }

    fn description(&self) -> &'static str {
        "Computes and prints operation counts, parameter sizes, and FLOP estimates"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut counts: IndexMap<String, usize> = IndexMap::new();
        let mut total_params = 0i64;
        let mut est_flops = 0i64;

        for (_, op) in &module.ops {
            *counts.entry(op.name.to_string()).or_insert(0) += 1;
            if op.is_op("stablehlo", "constant") {
                if let Some(&res) = op.results.first() {
                    if let Some(t) = module.values[res].ty.as_tensor() {
                        total_params += t.num_elements().unwrap_or(0);
                    }
                }
            } else if op.is_op("stablehlo", "dot_general") && op.operands.len() == 2 {
                let lhs_ty = module.values[op.operands[0]].ty.as_tensor();
                let rhs_ty = module.values[op.operands[1]].ty.as_tensor();
                if let (Some(l), Some(r)) = (lhs_ty, rhs_ty) {
                    if let (Some(Dim::Static(m)), Some(Dim::Static(k)), Some(Dim::Static(n))) =
                        (l.shape.first(), l.shape.get(1), r.shape.get(1))
                    {
                        est_flops += 2 * m * k * n;
                    }
                }
            }
        }

        eprintln!(
            "[castforge:stats] Total Ops: {} | Constant Params: {} | Est. MatMul FLOPs: {}",
            module.ops.len(),
            total_params,
            est_flops
        );
        for (op_name, count) in &counts {
            eprintln!("  - {:<28} : {}", op_name, count);
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: false,
            ops_removed: 0,
            ops_modified: 0,
            ops_created: 0,
        })
    }
}
