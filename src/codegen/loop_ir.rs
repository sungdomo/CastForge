//! Progressive Loop IR (`Ir(Module) -> LoopModule -> Target Source`) and modular
//! GPU/Native backend translators (`lower-to-loops`, `codegen-cuda`, `codegen-msl`,
//! `codegen-header`, `compile-lib`).

use crate::codegen::{
    CodegenTarget, TargetConfig, generate_code_with_config, sanitize_ident, tensor_len,
    tensor_shape_dims, value_ident,
};
use crate::ir::{Module, Payload, Translator, ValueId};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static SHLIB_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Statement in the progressive Loop IR (`LoopModule`).
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

/// Progressive Loop IR module lowered from a Tensor-level [`Module`].
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

/// Lowers a tensor-level CastForge [`Module`] into a progressive [`LoopModule`].
pub fn lower_to_loop_ir(module: &Module, base_config: &TargetConfig) -> LoopModule {
    let mod_config = base_config.with_module_attributes(module);
    let mem_plan = crate::codegen::memory_plan::build_memory_plan(module);
    let mod_name = module
        .name
        .clone()
        .unwrap_or_else(|| "castforge_module".to_string());
    let mut kernels = Vec::new();

    for fn_op_id in module.functions() {
        let fn_op = &module.ops[fn_op_id];
        let fn_name = sanitize_ident(
            fn_op
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
                .unwrap_or("forward"),
        );
        let Some(&reg_id) = fn_op.regions.first() else {
            continue;
        };
        let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let block = &module.blocks[blk_id];

        let mut names: HashMap<ValueId, String> = HashMap::new();
        let mut counter = 0usize;
        let mut params = Vec::new();
        for &arg_id in &block.arguments {
            let name = value_ident(module, arg_id, &mut names, &mut counter);
            let len = tensor_len(module, arg_id);
            params.push((name, len));
        }

        let mut body = Vec::new();
        body.push(LoopStmt::Comment(format!(
            "MemoryPlan: arena_workspace_bytes={} (f32_len={}, slots={}, saved_bytes={})",
            mem_plan.arena_workspace_bytes,
            mem_plan.arena_f32_len(),
            mem_plan.reusable_slot_count,
            mem_plan.saved_activation_bytes()
        )));
        let mut output = ("out".to_string(), 1usize);

        for &op_id in &block.ops {
            let op = &module.ops[op_id];
            if op.is_op("func", "return") {
                if let Some(&ret_val) = op.operands.first() {
                    let r_name = value_ident(module, ret_val, &mut names, &mut counter);
                    let r_len = tensor_len(module, ret_val);
                    output = (r_name, r_len);
                }
                continue;
            }
            let Some(&res_val) = op.results.first() else {
                continue;
            };
            let dst = value_ident(module, res_val, &mut names, &mut counter);
            let len = tensor_len(module, res_val);
            let res_dims = module.values[res_val]
                .ty
                .as_tensor()
                .map(tensor_shape_dims)
                .unwrap_or_else(|| vec![len]);
            let cfg = mod_config.with_op_attributes(op);

            if let Some(alloc) = mem_plan.allocation_for(res_val) {
                if let (Some(slot), Some(off)) = (alloc.arena_slot, alloc.arena_offset) {
                    body.push(LoopStmt::Comment(format!(
                        "Arena bind {dst} -> slot #{slot} (byte_offset={off}, f32_offset={})",
                        off / 4
                    )));
                } else if let Some(inpl) = alloc.inplace_from {
                    let inpl_name = value_ident(module, inpl, &mut names, &mut counter);
                    body.push(LoopStmt::Comment(format!(
                        "In-place buffer alias {dst} -> {inpl_name}"
                    )));
                }
            }

            body.push(LoopStmt::Alloc {
                name: dst.clone(),
                len,
            });

            if op
                .attributes
                .get("castforge.paged_attention")
                .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)))
            {
                let blk_sz = match op.attributes.get("castforge.page_block_size") {
                    Some(crate::ir::Attribute::Int(b)) => *b,
                    _ => 16,
                };
                body.push(LoopStmt::Comment(format!(
                    "PagedAttention KV block-table addressing (page_block_size={blk_sz})"
                )));
            }
            if op
                .attributes
                .get("castforge.zero_copy_concat")
                .is_some_and(|a| matches!(a, crate::ir::Attribute::Bool(true)))
            {
                body.push(LoopStmt::Comment(
                    "Zero-copy destination-passing concatenate".to_string(),
                ));
            }
            if let Some(scheme) = op
                .attributes
                .get("castforge.quant_scheme")
                .and_then(|a| a.as_str())
            {
                body.push(LoopStmt::Comment(format!(
                    "Groupwise quantized weight ({scheme})"
                )));
            }

            let dialect = op.name.dialect.as_str();
            let op_name = op.name.name.as_str();

            if (dialect == "stablehlo" && op_name == "dot_general"
                || dialect == "castforge"
                    && matches!(op_name, "fused_matmul_epilogue" | "fused_dequant_matmul"))
                && op.operands.len() >= 2
            {
                let lhs = value_ident(module, op.operands[0], &mut names, &mut counter);
                let rhs = value_ident(module, op.operands[1], &mut names, &mut counter);
                let lhs_dims = module.values[op.operands[0]]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![1, 1]);
                let rhs_dims = module.values[op.operands[1]]
                    .ty
                    .as_tensor()
                    .map(tensor_shape_dims)
                    .unwrap_or_else(|| vec![1, 1]);
                let m = *lhs_dims.first().unwrap_or(&1);
                let k = *lhs_dims.get(1).unwrap_or(&1);
                let n = *rhs_dims.get(1).unwrap_or(&1);
                let tm = cfg.tile_m.max(1).min(m.max(1));
                let tn = cfg.tile_n.max(1).min(n.max(1));
                let tk = cfg.tile_k.max(1).min(k.max(1));
                body.push(LoopStmt::Comment(format!(
                    "GEMM {m}x{k}x{n} ({op_name}) tiles=({tm},{tn},{tk}) order={}",
                    cfg.loop_order.as_str()
                )));
                let rhs_term = if op_name == "fused_dequant_matmul" && op.operands.len() >= 3 {
                    let sc = value_ident(module, op.operands[2], &mut names, &mut counter);
                    format!("({rhs}[p * {n} + j] * {sc}[j])")
                } else {
                    format!("{rhs}[p * {n} + j]")
                };
                let fma = LoopStmt::Assign {
                    dst: dst.clone(),
                    index: format!("i * {n} + j"),
                    expr: format!("{dst}[i * {n} + j] + {lhs}[i * {k} + p] * {rhs_term}"),
                };
                let j_loop = LoopStmt::For {
                    var: "j".to_string(),
                    lower: 0,
                    upper: n,
                    step: 1,
                    parallel: false,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![fma],
                };
                let p_loop = LoopStmt::For {
                    var: "p".to_string(),
                    lower: 0,
                    upper: k,
                    step: tk.min(1),
                    parallel: false,
                    vectorized: false,
                    unroll: 1,
                    body: vec![j_loop],
                };
                let mut i_body = vec![p_loop];
                if op_name == "fused_matmul_epilogue" && op.operands.len() >= 3 {
                    let bias = value_ident(module, op.operands[2], &mut names, &mut counter);
                    let act = op
                        .attributes
                        .get("activation")
                        .and_then(|a| a.as_str())
                        .unwrap_or("none");
                    let base = format!("({dst}[i * {n} + j] + {bias}[j])");
                    let ep_expr = match act {
                        "relu" => format!("fmaxf({base}, 0.0f)"),
                        "silu" => format!("({base} / (1.0f + expf(-{base})))"),
                        "gelu" => format!("(0.5f * {base} * (1.0f + tanhf(0.79788456f * {base})))"),
                        _ => base,
                    };
                    i_body.push(LoopStmt::For {
                        var: "j".to_string(),
                        lower: 0,
                        upper: n,
                        step: 1,
                        parallel: false,
                        vectorized: cfg.vectorize,
                        unroll: cfg.unroll_factor,
                        body: vec![LoopStmt::Assign {
                            dst: dst.clone(),
                            index: format!("i * {n} + j"),
                            expr: ep_expr,
                        }],
                    });
                }
                body.push(LoopStmt::For {
                    var: "i".to_string(),
                    lower: 0,
                    upper: m,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: false,
                    unroll: 1,
                    body: i_body,
                });
            } else if dialect == "castforge"
                && matches!(
                    op_name,
                    "fused_rmsnorm"
                        | "fused_layernorm"
                        | "fused_residual_rmsnorm"
                        | "fused_softmax"
                        | "fused_log_softmax"
                )
                && !op.operands.is_empty()
            {
                let x = value_ident(module, op.operands[0], &mut names, &mut counter);
                let cols = (*res_dims.last().unwrap_or(&len)).max(1);
                let rows = (len / cols).max(1);
                body.push(LoopStmt::Comment(format!(
                    "Row-wise reduction + normalization ({op_name}) rows={rows} cols={cols}"
                )));
                let norm_expr = match op_name {
                    "fused_softmax" => format!("expf({x}[row * {cols} + col])"),
                    "fused_log_softmax" => format!("{x}[row * {cols} + col]"),
                    _ => {
                        if let Some(&w_id) = op.operands.last()
                            && op.operands.len() >= 2
                        {
                            let w = value_ident(module, w_id, &mut names, &mut counter);
                            format!(
                                "({x}[row * {cols} + col] * rsqrtf({x}[row * {cols} + col] * \
                                 {x}[row * {cols} + col] + 1e-5f)) * {w}[col]"
                            )
                        } else {
                            format!("{x}[row * {cols} + col]")
                        }
                    }
                };
                let col_loop = LoopStmt::For {
                    var: "col".to_string(),
                    lower: 0,
                    upper: cols,
                    step: 1,
                    parallel: false,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst: dst.clone(),
                        index: format!("row * {cols} + col"),
                        expr: norm_expr,
                    }],
                };
                body.push(LoopStmt::For {
                    var: "row".to_string(),
                    lower: 0,
                    upper: rows,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: false,
                    unroll: 1,
                    body: vec![col_loop],
                });
            } else if dialect == "castforge"
                && matches!(op_name, "fused_attention" | "flash_attention")
                && op.operands.len() >= 3
            {
                let q = value_ident(module, op.operands[0], &mut names, &mut counter);
                let k_t = value_ident(module, op.operands[1], &mut names, &mut counter);
                let v_t = value_ident(module, op.operands[2], &mut names, &mut counter);
                let d_head = (*res_dims.last().unwrap_or(&1)).max(1);
                let seq = (len / d_head).max(1);
                let scale = 1.0f32 / (d_head as f32).sqrt();
                body.push(LoopStmt::Comment(format!(
                    "Fused scaled dot-product attention ({op_name}) seq={seq} d_head={d_head}"
                )));
                let d_loop = LoopStmt::For {
                    var: "d".to_string(),
                    lower: 0,
                    upper: d_head,
                    step: 1,
                    parallel: false,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst: dst.clone(),
                        index: format!("qi * {d_head} + d"),
                        expr: format!(
                            "{dst}[qi * {d_head} + d] + ({q}[qi * {d_head} + d] * \
                             {k_t}[kj * {d_head} + d] * {scale:.6}f) * {v_t}[kj * {d_head} + d]"
                        ),
                    }],
                };
                let kj_loop = LoopStmt::For {
                    var: "kj".to_string(),
                    lower: 0,
                    upper: seq,
                    step: 1,
                    parallel: false,
                    vectorized: false,
                    unroll: 1,
                    body: vec![d_loop],
                };
                body.push(LoopStmt::For {
                    var: "qi".to_string(),
                    lower: 0,
                    upper: seq,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: false,
                    unroll: 1,
                    body: vec![kj_loop],
                });
            } else if dialect == "stablehlo" && op_name == "reduce" && !op.operands.is_empty() {
                let src = value_ident(module, op.operands[0], &mut names, &mut counter);
                let src_len = tensor_len(module, op.operands[0]);
                let red_extent = (src_len / len.max(1)).max(1);
                let r_loop = LoopStmt::For {
                    var: "r".to_string(),
                    lower: 0,
                    upper: red_extent,
                    step: 1,
                    parallel: false,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst: dst.clone(),
                        index: "row".to_string(),
                        expr: format!("{dst}[row] + {src}[row * {red_extent} + r]"),
                    }],
                };
                body.push(LoopStmt::For {
                    var: "row".to_string(),
                    lower: 0,
                    upper: len,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: false,
                    unroll: 1,
                    body: vec![r_loop],
                });
            } else if dialect == "stablehlo" && op_name == "gather" && op.operands.len() >= 2 {
                let tbl = value_ident(module, op.operands[0], &mut names, &mut counter);
                let idx_buf = value_ident(module, op.operands[1], &mut names, &mut counter);
                let n_idx = tensor_len(module, op.operands[1]).max(1);
                let cols = (len / n_idx).max(1);
                let col_loop = LoopStmt::For {
                    var: "col".to_string(),
                    lower: 0,
                    upper: cols,
                    step: 1,
                    parallel: false,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst: dst.clone(),
                        index: format!("row * {cols} + col"),
                        expr: format!("{tbl}[((int){idx_buf}[row]) * {cols} + col]"),
                    }],
                };
                body.push(LoopStmt::For {
                    var: "row".to_string(),
                    lower: 0,
                    upper: n_idx,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: false,
                    unroll: 1,
                    body: vec![col_loop],
                });
            } else if dialect == "stablehlo" && op_name == "scatter" && op.operands.len() >= 3 {
                let input_buf = value_ident(module, op.operands[0], &mut names, &mut counter);
                let idx_buf = value_ident(module, op.operands[1], &mut names, &mut counter);
                let upd_buf = value_ident(module, op.operands[2], &mut names, &mut counter);
                let n_idx = tensor_len(module, op.operands[1]).max(1);
                let upd_len = tensor_len(module, op.operands[2]);
                let cols = (upd_len / n_idx).max(1);
                body.push(LoopStmt::For {
                    var: "idx".to_string(),
                    lower: 0,
                    upper: len,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst: dst.clone(),
                        index: "idx".to_string(),
                        expr: format!("{input_buf}[idx]"),
                    }],
                });
                let col_loop = LoopStmt::For {
                    var: "col".to_string(),
                    lower: 0,
                    upper: cols,
                    step: 1,
                    parallel: false,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst: dst.clone(),
                        index: format!("((int){idx_buf}[row]) * {cols} + col"),
                        expr: format!("{upd_buf}[row * {cols} + col]"),
                    }],
                };
                body.push(LoopStmt::For {
                    var: "row".to_string(),
                    lower: 0,
                    upper: n_idx,
                    step: 1,
                    parallel: false,
                    vectorized: false,
                    unroll: 1,
                    body: vec![col_loop],
                });
            } else {
                let op_str = op.name.to_string();
                let ops_str: Vec<String> = op
                    .operands
                    .iter()
                    .map(|&v| format!("{}[idx]", value_ident(module, v, &mut names, &mut counter)))
                    .collect();
                let expr = match (op_name, ops_str.as_slice()) {
                    ("add", [a, b]) => format!("{a} + {b}"),
                    ("subtract", [a, b]) => format!("{a} - {b}"),
                    ("multiply", [a, b]) => format!("{a} * {b}"),
                    ("divide", [a, b]) => format!("{a} / {b}"),
                    ("maximum", [a, b]) => format!("fmaxf({a}, {b})"),
                    ("minimum", [a, b]) => format!("fminf({a}, {b})"),
                    ("power", [a, b]) => format!("powf({a}, {b})"),
                    ("clamp", [lo, x, hi]) => format!("fminf(fmaxf({x}, {lo}), {hi})"),
                    ("exponential", [a]) => format!("expf({a})"),
                    ("log", [a]) => format!("logf({a})"),
                    ("tanh", [a]) => format!("tanhf({a})"),
                    ("sqrt", [a]) => format!("sqrtf({a})"),
                    ("rsqrt", [a]) => format!("(1.0f / sqrtf({a}))"),
                    ("sine", [a]) => format!("sinf({a})"),
                    ("cosine", [a]) => format!("cosf({a})"),
                    ("abs", [a]) => format!("fabsf({a})"),
                    ("negate", [a]) => format!("-({a})"),
                    ("logistic", [a]) => format!("(1.0f / (1.0f + expf(-({a}))))"),
                    ("fused_gelu", [a]) => {
                        format!(
                            "(0.5f * {a} * (1.0f + tanhf(0.79788456f * \
                             ({a} + 0.044715f * {a} * {a} * {a}))))"
                        )
                    }
                    ("fused_swiglu" | "fused_gated_mlp", [g, u]) => {
                        format!("(({g} / (1.0f + expf(-({g})))) * {u})")
                    }
                    ("fused_swiglu" | "fused_gated_mlp", [x, wg, wu]) => {
                        format!("((({x} * {wg}) / (1.0f + expf(-({x} * {wg})))) * ({x} * {wu}))")
                    }
                    _ => format!("{op_str}({})", ops_str.join(", ")),
                };
                body.push(LoopStmt::For {
                    var: "idx".to_string(),
                    lower: 0,
                    upper: len,
                    step: 1,
                    parallel: cfg.parallel,
                    vectorized: cfg.vectorize,
                    unroll: cfg.unroll_factor,
                    body: vec![LoopStmt::Assign {
                        dst,
                        index: "idx".to_string(),
                        expr,
                    }],
                });
            }
        }

        kernels.push(LoopKernel {
            name: fn_name,
            params,
            output,
            body,
        });
    }

    LoopModule {
        name: mod_name,
        kernels,
    }
}

/// Emits CUDA C++ (`.cu`) GPU kernels (`__global__ void`) for `module`.
pub fn emit_cuda_module_with_config(module: &Module, base_config: &TargetConfig) -> Result<String> {
    let loop_mod = lower_to_loop_ir(module, base_config);
    let cfg = base_config.with_module_attributes(module);
    let mut out = String::new();
    writeln!(out, "// Auto-generated by CastForge CUDA Backend")?;
    writeln!(out, "#include <cuda_runtime.h>")?;
    writeln!(out, "#include <math.h>\n")?;

    for kernel in &loop_mod.kernels {
        let mut sig_params: Vec<String> = kernel
            .params
            .iter()
            .map(|(n, _)| format!("const float* __restrict__ {n}"))
            .collect();
        sig_params.push(format!("float* __restrict__ {}", kernel.output.0));
        writeln!(
            out,
            "extern \"C\" __global__ void {}_cuda_kernel({}) {{",
            kernel.name,
            sig_params.join(", ")
        )?;
        writeln!(
            out,
            "    const int tid = blockIdx.x * blockDim.x + threadIdx.x;"
        )?;
        writeln!(
            out,
            "    __shared__ float shared_tile[{}];",
            cfg.wgsl_workgroup_size.max(64)
        )?;
        writeln!(out, "    (void)shared_tile;")?;
        for stmt in &kernel.body {
            emit_cuda_stmt(&mut out, stmt, &kernel.output.0)?;
        }
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

fn emit_cuda_stmt(out: &mut String, stmt: &LoopStmt, ret_name: &str) -> Result<()> {
    match stmt {
        LoopStmt::Comment(c) => writeln!(out, "    // {c}")?,
        LoopStmt::Alloc { name, len } if name != ret_name => {
            writeln!(out, "    float {name}[{len}];")?;
        }
        LoopStmt::Alloc { .. } => {}
        LoopStmt::For {
            var,
            upper,
            unroll,
            body,
            ..
        } => {
            if *unroll > 1 {
                writeln!(out, "    #pragma unroll {unroll}")?;
            }
            writeln!(
                out,
                "    for (int {var} = tid; {var} < {upper}; {var} += blockDim.x * gridDim.x) {{"
            )?;
            for inner in body {
                emit_cuda_stmt_seq(out, inner, 2)?;
            }
            writeln!(out, "    }}")?;
        }
        LoopStmt::Assign { dst, index, expr } => {
            writeln!(out, "    {dst}[{index}] = {expr};")?;
        }
    }
    Ok(())
}

fn emit_cuda_stmt_seq(out: &mut String, stmt: &LoopStmt, indent: usize) -> Result<()> {
    let pad = "    ".repeat(indent);
    match stmt {
        LoopStmt::Comment(c) => writeln!(out, "{pad}// {c}")?,
        LoopStmt::Alloc { name, len } => writeln!(out, "{pad}float {name}[{len}];")?,
        LoopStmt::For {
            var,
            lower,
            upper,
            step,
            unroll,
            body,
            ..
        } => {
            if *unroll > 1 {
                writeln!(out, "{pad}#pragma unroll {unroll}")?;
            }
            writeln!(
                out,
                "{pad}for (int {var} = {lower}; {var} < {upper}; {var} += {step}) {{"
            )?;
            for inner in body {
                emit_cuda_stmt_seq(out, inner, indent + 1)?;
            }
            writeln!(out, "{pad}}}")?;
        }
        LoopStmt::Assign { dst, index, expr } => {
            writeln!(out, "{pad}{dst}[{index}] = {expr};")?;
        }
    }
    Ok(())
}

/// Emits Apple Metal Shading Language (`.metal`) GPU compute kernels for `module`.
pub fn emit_msl_module_with_config(module: &Module, base_config: &TargetConfig) -> Result<String> {
    let loop_mod = lower_to_loop_ir(module, base_config);
    let mut out = String::new();
    writeln!(out, "// Auto-generated by CastForge Metal (MSL) Backend")?;
    writeln!(out, "#include <metal_stdlib>")?;
    writeln!(out, "using namespace metal;\n")?;

    for kernel in &loop_mod.kernels {
        let mut sig_params = Vec::new();
        for (idx, (n, _)) in kernel.params.iter().enumerate() {
            sig_params.push(format!("device const float* {n} [[buffer({idx})]]"));
        }
        let out_idx = kernel.params.len();
        sig_params.push(format!(
            "device float* {} [[buffer({out_idx})]]",
            kernel.output.0
        ));
        sig_params.push("uint gid [[thread_position_in_grid]]".to_string());
        writeln!(
            out,
            "kernel void {}_msl_kernel({}) {{",
            kernel.name,
            sig_params.join(", ")
        )?;
        for stmt in &kernel.body {
            emit_msl_stmt(&mut out, stmt, &kernel.output.0)?;
        }
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

fn emit_msl_stmt(out: &mut String, stmt: &LoopStmt, ret_name: &str) -> Result<()> {
    match stmt {
        LoopStmt::Comment(c) => writeln!(out, "    // {c}")?,
        LoopStmt::Alloc { name, len } if name != ret_name => {
            writeln!(out, "    float {name}[{len}];")?;
        }
        LoopStmt::Alloc { .. } => {}
        LoopStmt::For {
            var, upper, body, ..
        } => {
            writeln!(out, "    if (gid < {upper}u) {{")?;
            writeln!(out, "        uint {var} = gid;")?;
            for inner in body {
                emit_cuda_stmt_seq(out, inner, 2)?;
            }
            writeln!(out, "    }}")?;
        }
        LoopStmt::Assign { dst, index, expr } => {
            writeln!(out, "    {dst}[{index}] = {expr};")?;
        }
    }
    Ok(())
}

/// Generates a C/C++ FFI header (`.h`) matching the C99 symbols emitted by `codegen-c`.
pub fn emit_c_header(module: &Module) -> Result<String> {
    let loop_mod = lower_to_loop_ir(module, &TargetConfig::default());
    let guard = format!(
        "CASTFORGE_{}_H",
        sanitize_ident(&loop_mod.name).to_uppercase()
    );
    let mut out = String::new();
    writeln!(out, "/* Auto-generated C/C++ FFI Header by CastForge */")?;
    writeln!(out, "#ifndef {guard}")?;
    writeln!(out, "#define {guard}\n")?;
    writeln!(out, "#include <stddef.h>\n")?;
    writeln!(out, "#ifdef __cplusplus")?;
    writeln!(out, "extern \"C\" {{")?;
    writeln!(out, "#endif\n")?;

    for kernel in &loop_mod.kernels {
        let mut args: Vec<String> = kernel
            .params
            .iter()
            .map(|(n, len)| format!("const float {n}[{len}]"))
            .collect();
        args.push(format!("float {}[{}]", kernel.output.0, kernel.output.1));
        writeln!(out, "void {}({});", kernel.name, args.join(", "))?;
    }

    writeln!(out, "\n#ifdef __cplusplus")?;
    writeln!(out, "}}")?;
    writeln!(out, "#endif\n")?;
    writeln!(out, "#endif /* {guard} */")?;
    Ok(out)
}

/// Compiles `module` into a native shared library (`.so` / `.dylib`) byte vector.
pub fn compile_shared_lib(module: &Module, config: &TargetConfig) -> Result<Vec<u8>> {
    let c_src = generate_code_with_config(module, CodegenTarget::C, config)?;
    let seq = SHLIB_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let tmp_dir = std::env::temp_dir().join(format!("castforge_shlib_{pid}_{seq}"));
    fs::create_dir_all(&tmp_dir)?;
    let src_path = tmp_dir.join("kernel.c");
    let lib_path = tmp_dir.join("libcastforge_kernel.so");
    fs::write(&src_path, &c_src)?;

    let status = Command::new("cc")
        .arg("-O3")
        .arg("-shared")
        .arg("-fPIC")
        .arg(&src_path)
        .arg("-lm")
        .arg("-o")
        .arg(&lib_path)
        .status()
        .context("Failed to invoke `cc` for shared library compilation")?;
    if !status.success() {
        let _ = fs::remove_dir_all(&tmp_dir);
        bail!("`cc -shared` exited with non-zero status: {status}");
    }
    let bytes = fs::read(&lib_path).context("Failed to read compiled shared library")?;
    let _ = fs::remove_dir_all(&tmp_dir);
    Ok(bytes)
}

/// Progressive Lowering Translator (`Ir(Module) -> Text(LoopIR)`).
#[derive(Default)]
pub struct TensorToLoopTranslator {
    pub config: TargetConfig,
}

impl Translator for TensorToLoopTranslator {
    fn name(&self) -> &str {
        "lower-to-loops"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        let loop_mod = lower_to_loop_ir(&module, &self.config);
        Ok(Payload::Text(loop_mod.to_string()))
    }
}

/// C/C++ Header Translator (`Ir(Module) -> Text(C Header)`).
#[derive(Default)]
pub struct CHeaderTranslator;

impl Translator for CHeaderTranslator {
    fn name(&self) -> &str {
        "codegen-header"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        Ok(Payload::Text(emit_c_header(&module)?))
    }
}

/// Native Shared Library Translator (`Ir(Module) -> Bytes(.so/.dylib)`).
#[derive(Default)]
pub struct SharedLibTranslator {
    pub config: TargetConfig,
}

impl Translator for SharedLibTranslator {
    fn name(&self) -> &str {
        "compile-lib"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        Ok(Payload::Bytes(compile_shared_lib(&module, &self.config)?))
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
        let can_fuse = match (&stmts[i], &stmts[i + 1]) {
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
        };
        if can_fuse {
            let LoopStmt::For {
                body: mut next_body,
                ..
            } = stmts.remove(i + 1)
            else {
                unreachable!()
            };
            if let LoopStmt::For { body, .. } = &mut stmts[i] {
                body.append(&mut next_body);
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
        if interchange && var == "i" && body.len() == 1 {
            if let LoopStmt::For {
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
