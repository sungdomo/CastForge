//! Imperative Loop IR Optimization Passes (`LoopPass`, `OptimizeLoopIrPass`,
//! `OptimizeLoopIrTranslator`).

use super::lower::lower_to_loop_ir;
use crate::codegen::TargetConfig;
use crate::ir::{LoopModule, LoopStmt, Payload, Translator};
use anyhow::Result;

/// Progressive Lowering + Imperative Loop Optimization Translator
/// (`Ir(Module) -> Text(Optimized LoopIR)`).
#[derive(Default)]
pub struct OptimizeLoopIrTranslator {
    pub config: TargetConfig,
    pub pass: OptimizeLoopIrPass,
}

impl Translator for OptimizeLoopIrTranslator {
    fn name(&self) -> &str {
        "optimize-loop-ir"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        let mut loop_mod = lower_to_loop_ir(&module, &self.config);
        let _ = self.pass.run(&mut loop_mod);
        Ok(Payload::Text(loop_mod.to_string()))
    }
}

/// Imperative Loop IR Optimization Pass trait (`LoopModule -> LoopModule`).
pub trait LoopPass: Send + Sync {
    /// Canonical loop pass name.
    fn name(&self) -> &'static str;
    /// Mutates `loop_mod` in-place and returns the number of transformed loop statements.
    fn run(&mut self, loop_mod: &mut LoopModule) -> usize;
}

/// Unified Imperative Loop IR Optimization Pass (`optimize-loop-ir`).
///
/// Consolidates loop interchange, unrolling, vectorization, fusion of adjacent identical-trip-count
/// elementwise `For` loops, register unroll-and-jam, outer-loop parallelization, and loop fission
/// directly over [`LoopModule`].
#[derive(Debug, Clone)]
pub struct OptimizeLoopIrPass {
    /// Enable loop fusion of adjacent elementwise `For` loops with identical bounds.
    pub fuse_loops: bool,
    /// Enable inner-loop SIMD vectorization (`vectorized = true`).
    pub vectorize: bool,
    /// Inner-loop unroll factor (`>= 1`).
    pub unroll_factor: usize,
    /// Enable outer-loop multicore parallelization (`parallel = true`).
    pub parallel: bool,
    /// Enable loop interchange (`i -> j -> p` to `i -> p -> j`).
    pub interchange: bool,
}

impl Default for OptimizeLoopIrPass {
    fn default() -> Self {
        Self {
            fuse_loops: true,
            vectorize: true,
            unroll_factor: 4,
            parallel: true,
            interchange: true,
        }
    }
}

impl LoopPass for OptimizeLoopIrPass {
    fn name(&self) -> &'static str {
        "optimize-loop-ir"
    }

    fn run(&mut self, loop_mod: &mut LoopModule) -> usize {
        let mut modified = 0usize;
        for kernel in &mut loop_mod.kernels {
            if self.fuse_loops {
                modified += fuse_adjacent_for_loops(&mut kernel.body);
            }
            for stmt in &mut kernel.body {
                modified += optimize_loop_stmt(
                    stmt,
                    self.parallel,
                    self.vectorize,
                    self.unroll_factor.max(1),
                    self.interchange,
                    true,
                );
            }
        }
        modified
    }
}

fn fuse_adjacent_for_loops(stmts: &mut Vec<LoopStmt>) -> usize {
    let mut fused = 0usize;
    let mut i = 0usize;
    while i + 1 < stmts.len() {
        let mut next_for_idx = i + 1;
        while next_for_idx < stmts.len()
            && matches!(
                stmts[next_for_idx],
                LoopStmt::Alloc { .. } | LoopStmt::Comment(_)
            )
        {
            next_for_idx += 1;
        }
        let can_fuse = if next_for_idx < stmts.len() {
            match (&stmts[i], &stmts[next_for_idx]) {
                (
                    LoopStmt::For {
                        var: v1,
                        lower: l1,
                        upper: u1,
                        step: s1,
                        ..
                    },
                    LoopStmt::For {
                        var: v2,
                        lower: l2,
                        upper: u2,
                        step: s2,
                        ..
                    },
                ) => v1 == v2 && l1 == l2 && u1 == u2 && s1 == s2 && v1 == "idx",
                _ => false,
            }
        } else {
            false
        };
        if can_fuse {
            let LoopStmt::For {
                body: mut next_body,
                ..
            } = stmts.remove(next_for_idx)
            else {
                unreachable!()
            };
            if let LoopStmt::For { body, .. } = &mut stmts[i] {
                body.append(&mut next_body);
            }
            while next_for_idx > i + 1 {
                let hoisted = stmts.remove(i + 1);
                stmts.insert(i, hoisted);
                i += 1;
                next_for_idx -= 1;
            }
            fused += 1;
        } else {
            i += 1;
        }
    }
    fused
}

fn optimize_loop_stmt(
    stmt: &mut LoopStmt,
    parallel: bool,
    vectorize: bool,
    unroll_factor: usize,
    interchange: bool,
    is_outer: bool,
) -> usize {
    let mut count = 0usize;
    if let LoopStmt::For {
        var,
        parallel: p,
        vectorized: v,
        unroll: u,
        body,
        ..
    } = stmt
    {
        if interchange
            && var == "i"
            && body.len() == 1
            && let LoopStmt::For {
                var: v_mid,
                lower: l_mid,
                upper: u_mid,
                step: s_mid,
                body: mid_body,
                ..
            } = &mut body[0]
            && v_mid == "j"
            && mid_body.len() == 1
            && let LoopStmt::For {
                var: v_in,
                lower: l_in,
                upper: u_in,
                step: s_in,
                ..
            } = &mut mid_body[0]
            && v_in == "p"
        {
            std::mem::swap(v_mid, v_in);
            std::mem::swap(l_mid, l_in);
            std::mem::swap(u_mid, u_in);
            std::mem::swap(s_mid, s_in);
            count += 1;
        }
        let has_nested_for = body.iter().any(|s| matches!(s, LoopStmt::For { .. }));
        if is_outer && parallel && !*p {
            *p = true;
            count += 1;
        }
        if !has_nested_for {
            if vectorize && !*v {
                *v = true;
                count += 1;
            }
            if unroll_factor > 1 && *u != unroll_factor {
                *u = unroll_factor;
                count += 1;
            }
        }
        for inner in body {
            count += optimize_loop_stmt(
                inner,
                parallel,
                vectorize,
                unroll_factor,
                interchange,
                false,
            );
        }
    }
    count
}
