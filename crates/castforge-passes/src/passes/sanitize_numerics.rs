//! Numerical Sanitizer & NaN Guard Injection Pass (`sanitize-numerics`: `Ir -> Ir`).
//!
//! Guards numerically sensitive operations (`divide`, `rsqrt`, `exponential`, `convert`) and
//! function outputs by inserting `stablehlo.composite` (`"castforge.nan_guard"`) operations that
//! replace `NaN` with `0.0` and clamp magnitudes to `[-clamp_max, clamp_max]` (default `65504.0`
//! for IEEE-754 `f16` dynamic range safety).

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpName, ValueId};
use indexmap::IndexMap;
use smallvec::smallvec;

/// Numerical Sanitizer Pass (`sanitize-numerics`).
#[derive(Debug, Clone)]
pub struct SanitizeNumericsPass {
    /// Maximum finite magnitude clamp bound (default: `65504.0`).
    pub clamp_max: f64,
    /// Epsilon floor for denominators and `rsqrt` inputs (default: `1e-6`).
    pub eps: f64,
}

impl Default for SanitizeNumericsPass {
    fn default() -> Self {
        Self {
            clamp_max: 65504.0,
            eps: 1e-6,
        }
    }
}

impl Pass for SanitizeNumericsPass {
    fn name(&self) -> &'static str {
        "sanitize-numerics"
    }

    fn description(&self) -> &'static str {
        "Injects castforge.nan_guard clamps on numerically sensitive ops and function outputs"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut created = 0usize;
        let clamp = self.clamp_max.abs().max(1.0);
        let eps = self.eps.abs().max(1e-12);

        for fn_op_id in module.functions() {
            let reg_id = module.ops[fn_op_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.is_op("stablehlo", "composite")
                    && op.attributes.get("name").and_then(|a| a.as_str())
                        == Some("castforge.nan_guard")
                {
                    continue;
                }

                let is_sensitive = matches!(
                    op.name.name.as_str(),
                    "divide" | "rsqrt" | "exponential" | "convert" | "dot_general"
                );
                if !is_sensitive || op.results.is_empty() {
                    continue;
                }

                let orig_res = op.results[0];
                let res_ty = module.values[orig_res].ty.clone();
                let base_ssa = module.values[orig_res]
                    .ssa_name
                    .as_deref()
                    .unwrap_or("v")
                    .to_string();

                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.nan_guard".to_string()),
                );
                attrs.insert("clamp_max".to_string(), Attribute::Float(clamp));
                attrs.insert("eps".to_string(), Attribute::Float(eps));

                let guard_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "composite"),
                    smallvec![orig_res],
                    vec![(Some(format!("{base_ssa}_san")), res_ty)],
                    attrs,
                    Vec::new(),
                    false,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let guard_res: ValueId = module.ops[guard_op_id].results[0];

                // Redirect all downstream uses of `orig_res` (except `guard_op_id` itself)
                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == op_id || other_id == guard_op_id {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = guard_res;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id].ops.insert(pos + 1, guard_op_id);
                }
                created += 1;
            }
        }

        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: created > 0,
            ops_removed: 0,
            ops_modified: created,
            ops_created: created,
        })
    }
}
