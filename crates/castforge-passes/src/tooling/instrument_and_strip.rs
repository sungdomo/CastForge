//! Per-Op Profiling Instrumentor (`instrument-profiling`) and Transient Attribute Stripper
//! (`strip-attributes`) Passes (`Ir -> Ir`).
//!
//! - [`InstrumentProfilingPass`] (`instrument-profiling`) wraps compute operations (`dot_general`,
//!   `convolution`, `reduce`, or `all`) in `stablehlo.composite` (`"castforge.profile_region"`)
//!   with `profile_id` and `profile_label` attributes so Rust/C99 codegen emits per-layer hardware
//!   timer/counter annotations.
//! - [`StripAttributesPass`] (`strip-attributes`) strips transient `castforge.*` and loop-schedule
//!   attributes so `print-mlir` emits clean, portable upstream StableHLO MLIR.

use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Module, OpName};
use indexmap::IndexMap;
use smallvec::smallvec;

/// Per-Op Profiling Region Injector Pass (`instrument-profiling`: `Ir -> Ir`).
#[derive(Debug, Clone)]
pub struct InstrumentProfilingPass {
    /// Filter of operations to instrument (`"gemm"` or `"all"`, default `"gemm"`).
    pub filter: String,
}

impl Default for InstrumentProfilingPass {
    fn default() -> Self {
        Self {
            filter: "gemm".to_string(),
        }
    }
}

impl Pass for InstrumentProfilingPass {
    fn name(&self) -> &'static str {
        "instrument-profiling"
    }

    fn description(&self) -> &'static str {
        "Injects castforge.profile_region timing markers around target compute operations"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let all_ops = self.filter.eq_ignore_ascii_case("all");
        let mut created = 0usize;

        for fn_id in module.functions() {
            let reg_id = module.ops[fn_id].regions[0];
            let block_id = module.regions[reg_id].blocks[0];
            let ops = module.blocks[block_id].ops.clone();

            for op_id in ops {
                let Some(op) = module.ops.get(op_id).cloned() else {
                    continue;
                };
                if op.is_op("func", "return")
                    || op.is_op("stablehlo", "constant")
                    || (op.is_op("stablehlo", "composite")
                        && op.attributes.get("name").and_then(|a| a.as_str())
                            == Some("castforge.profile_region"))
                {
                    continue;
                }
                let is_target = if all_ops {
                    !op.results.is_empty()
                } else {
                    op.is_op("stablehlo", "dot_general")
                        || op.is_op("stablehlo", "convolution")
                        || op.is_op("stablehlo", "reduce")
                };
                if !is_target || op.results.is_empty() {
                    continue;
                }

                let orig_res = op.results[0];
                let res_ty = module.values[orig_res].ty.clone();
                let mut attrs = IndexMap::new();
                attrs.insert(
                    "name".to_string(),
                    Attribute::String("castforge.profile_region".to_string()),
                );
                attrs.insert("profile_id".to_string(), Attribute::Int(created as i64));
                attrs.insert(
                    "profile_label".to_string(),
                    Attribute::String(op.name.to_string()),
                );

                let prof_op_id = module.add_operation(
                    block_id,
                    OpName::new("stablehlo", "composite"),
                    smallvec![orig_res],
                    vec![(Some(format!("prof_{created}")), res_ty)],
                    attrs,
                    Vec::new(),
                    false,
                    op.location.clone(),
                );
                module.blocks[block_id].ops.pop();
                let prof_res = module.ops[prof_op_id].results[0];

                for (other_id, other_op) in module.ops.iter_mut() {
                    if other_id == op_id || other_id == prof_op_id {
                        continue;
                    }
                    for opnd in &mut other_op.operands {
                        if *opnd == orig_res {
                            *opnd = prof_res;
                        }
                    }
                }

                if let Some(pos) = module.blocks[block_id]
                    .ops
                    .iter()
                    .position(|&id| id == op_id)
                {
                    module.blocks[block_id].ops.insert(pos + 1, prof_op_id);
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

/// Transient Dialect & Schedule Attribute Cleaner Pass (`strip-attributes`: `Ir -> Ir`).
#[derive(Debug, Clone, Default)]
pub struct StripAttributesPass;

impl Pass for StripAttributesPass {
    fn name(&self) -> &'static str {
        "strip-attributes"
    }

    fn description(&self) -> &'static str {
        "Strips transient castforge.* and schedule attributes for pure portable StableHLO export"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        const TRANSIENT_KEYS: &[&str] = &[
            "tile_m",
            "tile_n",
            "tile_k",
            "tile_size",
            "l2_tile",
            "reg_tile_m",
            "reg_tile_n",
            "loop_order",
            "collapse_loops",
            "skew_loops",
            "unroll_factor",
            "prefetch_distance",
            "split_k",
            "double_buffer",
            "vector_width",
            "simd_isa",
            "parallel_m",
            "inplace_operand",
            "sparsity",
            "data_layout",
            "pad_align",
            "padded_mnk",
            "spmd_sharding",
            "num_devices",
            "async_overlap",
            "coalesced_collectives",
            "stream_id",
            "unroll_step",
        ];

        let mut modified = 0usize;
        for (_, op) in module.ops.iter_mut() {
            let before = op.attributes.len();
            op.attributes.retain(|k, _| {
                !k.starts_with("castforge.") && !TRANSIENT_KEYS.contains(&k.as_str())
            });
            if op.attributes.len() < before {
                modified += 1;
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
