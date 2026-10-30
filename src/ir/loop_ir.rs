//! Lower-Level Imperative Loop Intermediate Representation (`LoopModule`, `LoopKernel`,
//! `LoopStmt`).
//!
//! While [`crate::ir::Module`] represents high-level SSA tensor operations (`stablehlo.*`,
//! `castforge.*`), [`LoopModule`] represents lowered imperative kernels composed of flat 1D buffer
//! allocations (`Alloc`), bounded affine `For` loops (`parallel`, `vectorized`, `unroll`), and
//! indexed scalar assignments (`Assign`).

use serde::{Deserialize, Serialize};
use std::fmt;

/// Statement in the progressive Loop IR ([`LoopModule`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopStmt {
    /// Comment annotation (`// ...`).
    Comment(String),
    /// Static buffer declaration (`float buf[N];`).
    Alloc { name: String, len: usize },
    /// Bounded affine loop (`for var in (0..upper).step_by(step)`).
    For {
        var: String,
        lower: usize,
        upper: usize,
        step: usize,
        parallel: bool,
        vectorized: bool,
        unroll: usize,
        body: Vec<LoopStmt>,
    },
    /// Indexed scalar assignment (`dst[idx] = expr;`).
    Assign {
        dst: String,
        index: String,
        expr: String,
    },
}

/// Lowered function kernel in [`LoopModule`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopKernel {
    /// Sanitized function symbol name.
    pub name: String,
    /// Input buffer parameters `(name, num_elements)`.
    pub params: Vec<(String, usize)>,
    /// Output buffer parameter `(name, num_elements)`.
    pub output: (String, usize),
    /// Sequence of loop and buffer statements.
    pub body: Vec<LoopStmt>,
}

/// Progressive Loop IR module lowered from a Tensor-level [`crate::ir::Module`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopModule {
    /// Module name.
    pub name: String,
    /// Lowered kernel functions.
    pub kernels: Vec<LoopKernel>,
}

impl fmt::Display for LoopModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// CastForge Progressive Loop IR (module: {})",
            self.name
        )?;
        for kernel in &self.kernels {
            let params_str = kernel
                .params
                .iter()
                .map(|(n, len)| format!("{n}: &f32[{len}]"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                f,
                "kernel @{}({}) -> {}: &mut f32[{}] {{",
                kernel.name, params_str, kernel.output.0, kernel.output.1
            )?;
            for stmt in &kernel.body {
                fmt_stmt(f, stmt, 1)?;
            }
            writeln!(f, "}}")?;
        }
        Ok(())
    }
}

fn fmt_stmt(f: &mut fmt::Formatter<'_>, stmt: &LoopStmt, indent: usize) -> fmt::Result {
    let pad = "  ".repeat(indent);
    match stmt {
        LoopStmt::Comment(c) => writeln!(f, "{pad}// {c}"),
        LoopStmt::Alloc { name, len } => writeln!(f, "{pad}alloc {name}: f32[{len}]"),
        LoopStmt::For {
            var,
            lower,
            upper,
            step,
            parallel,
            vectorized,
            unroll,
            body,
        } => {
            let mut tags = Vec::new();
            if *parallel {
                tags.push("parallel".to_string());
            }
            if *vectorized {
                tags.push("vectorize".to_string());
            }
            if *unroll > 1 {
                tags.push(format!("unroll({unroll})"));
            }
            let attr = if tags.is_empty() {
                String::new()
            } else {
                format!(" #[{}]", tags.join(", "))
            };
            writeln!(f, "{pad}for {var} in {lower}..{upper} step {step}{attr} {{")?;
            for inner in body {
                fmt_stmt(f, inner, indent + 1)?;
            }
            writeln!(f, "{pad}}}")
        }
        LoopStmt::Assign { dst, index, expr } => {
            writeln!(f, "{pad}{dst}[{index}] = {expr}")
        }
    }
}
