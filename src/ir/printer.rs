//! Pure-Rust MLIR printer and `MlirPrintTranslator` (`castforge::ir::printer`).

use crate::ir::{Attribute, Module, OpId, Type, ValueId};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write};

#[derive(Debug, Clone, Copy, Default)]
pub struct PrintOptions {
    /// Print all operations in canonical MLIR Generic syntax (`"dialect.op"(...) : ...`).
    pub generic_form: bool,
    /// Include `loc(...)` annotations on operations.
    pub print_locations: bool,
}

pub struct MlirPrinter<'a> {
    module: &'a Module,
    opts: PrintOptions,
    value_names: HashMap<ValueId, String>,
    used_names: HashSet<String>,
    next_anon_id: usize,
}

impl<'a> MlirPrinter<'a> {
    pub fn new(module: &'a Module, opts: PrintOptions) -> Self {
        Self {
            module,
            opts,
            value_names: HashMap::new(),
            used_names: HashSet::new(),
            next_anon_id: 0,
        }
    }

    pub fn print_module(&mut self) -> Result<String, fmt::Error> {
        let mut out = String::new();
        if let Some(name) = &self.module.name {
            writeln!(out, "module @{name} {{")?;
        } else {
            writeln!(out, "module {{")?;
        }
        let body = self.module.body_block();
        for &op_id in &self.module.blocks[body].ops {
            self.print_op(&mut out, op_id, 1)?;
        }
        writeln!(out, "}}")?;
        Ok(out)
    }

    fn assign_value_name(&mut self, val_id: ValueId) -> String {
        if let Some(existing) = self.value_names.get(&val_id) {
            return existing.clone();
        }
        let val = &self.module.values[val_id];
        let name = if let Some(ssa) = &val.ssa_name {
            let base = format!("%{ssa}");
            if !self.used_names.contains(&base) {
                base
            } else {
                let mut suffix = 1usize;
                loop {
                    let candidate = format!("%{ssa}_{suffix}");
                    if !self.used_names.contains(&candidate) {
                        break candidate;
                    }
                    suffix += 1;
                }
            }
        } else {
            loop {
                let n = format!("%{}", self.next_anon_id);
                self.next_anon_id += 1;
                if !self.used_names.contains(&n) {
                    break n;
                }
            }
        };
        self.used_names.insert(name.clone());
        self.value_names.insert(val_id, name.clone());
        name
    }

    fn print_op(&mut self, out: &mut String, op_id: OpId, indent: usize) -> fmt::Result {
        let pad = "  ".repeat(indent);
        let op = &self.module.ops[op_id];

        if op.is_op("func", "func") {
            return self.print_func_op(out, op_id, indent);
        }

        // Assign result names
        let mut res_names = Vec::with_capacity(op.results.len());
        for &res_id in &op.results {
            res_names.push(self.assign_value_name(res_id));
        }

        write!(out, "{pad}")?;
        if !res_names.is_empty() {
            write!(out, "{} = ", res_names.join(", "))?;
        }

        let operand_names: Vec<String> = op
            .operands
            .iter()
            .map(|&v| self.assign_value_name(v))
            .collect();

        let operand_types: Vec<Type> = op
            .operands
            .iter()
            .map(|&v| self.module.values[v].ty.clone())
            .collect();
        let result_types: Vec<Type> = op
            .results
            .iter()
            .map(|&v| self.module.values[v].ty.clone())
            .collect();

        if self.opts.generic_form {
            write!(out, "\"{}\"({})", op.name, operand_names.join(", "))?;
            if !op.attributes.is_empty() {
                self.print_attr_dict(out, &op.attributes)?;
            }
            let fn_ty = Type::Function {
                inputs: operand_types,
                results: result_types,
            };
            write!(out, " : {fn_ty}")?;
        } else if op.is_op("func", "return") {
            if operand_names.is_empty() {
                write!(out, "return")?;
            } else {
                let types_str: Vec<String> = operand_types.iter().map(|t| t.to_string()).collect();
                write!(
                    out,
                    "return {} : {}",
                    operand_names.join(", "),
                    types_str.join(", ")
                )?;
            }
        } else if op.is_op("func", "call") {
            let callee = op
                .attributes
                .get("callee")
                .and_then(|a| a.as_str())
                .unwrap_or("callee");
            let fn_ty = Type::Function {
                inputs: operand_types,
                results: result_types,
            };
            write!(
                out,
                "func.call @{callee}({}) : {fn_ty}",
                operand_names.join(", ")
            )?;
        } else if op.is_op("stablehlo", "constant") {
            if let Some(val_attr) = op.attributes.get("value") {
                write!(out, "stablehlo.constant {val_attr}")?;
            } else {
                write!(out, "stablehlo.constant")?;
            }
        } else if op.is_op("stablehlo", "transpose") && op.results.len() == 1 {
            let in_ty = &operand_types[0];
            let out_ty = &result_types[0];
            let dims = op
                .attributes
                .get("permutation")
                .and_then(|a| a.as_int_slice())
                .map(|s| {
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            write!(
                out,
                "stablehlo.transpose {}, dims = [{}]",
                operand_names.join(", "),
                dims
            )?;
            self.print_extra_attrs(out, &op.attributes, &["permutation"])?;
            write!(out, " : ({in_ty}) -> {out_ty}")?;
        } else if op.is_op("stablehlo", "reshape") && op.results.len() == 1 {
            let in_ty = &operand_types[0];
            let out_ty = &result_types[0];
            write!(out, "stablehlo.reshape {}", operand_names.join(", "))?;
            self.print_extra_attrs(out, &op.attributes, &[])?;
            write!(out, " : ({in_ty}) -> {out_ty}")?;
        } else if op.is_op("stablehlo", "convert") && op.results.len() == 1 {
            let in_ty = &operand_types[0];
            let out_ty = &result_types[0];
            write!(out, "stablehlo.convert {}", operand_names.join(", "))?;
            self.print_extra_attrs(out, &op.attributes, &[])?;
            write!(out, " : ({in_ty}) -> {out_ty}")?;
        } else if op.is_op("stablehlo", "uniform_dequantize") && op.results.len() == 1 {
            let in_ty = &operand_types[0];
            let out_ty = &result_types[0];
            write!(
                out,
                "stablehlo.uniform_dequantize {}",
                operand_names.join(", ")
            )?;
            self.print_extra_attrs(out, &op.attributes, &[])?;
            write!(out, " : ({in_ty}) -> {out_ty}")?;
        } else if op.is_op("stablehlo", "iota") && op.results.len() == 1 {
            let dim = match op.attributes.get("iota_dimension") {
                Some(Attribute::Int(d)) => *d,
                _ => 0,
            };
            write!(out, "stablehlo.iota dim = {dim}")?;
            self.print_extra_attrs(out, &op.attributes, &["iota_dimension"])?;
            write!(out, " : {}", result_types[0])?;
        } else if op.is_op("stablehlo", "reduce")
            && op.operands.len() == 2
            && op.results.len() == 1
            && op.attributes.contains_key("reducer")
        {
            let reducer = op
                .attributes
                .get("reducer")
                .and_then(|a| a.as_str())
                .unwrap_or("stablehlo.add");
            let dims = op
                .attributes
                .get("dimensions")
                .and_then(|a| a.as_int_slice())
                .map(|s| {
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            let lhs = &operand_names[0];
            let init = &operand_names[1];
            let in0 = &operand_types[0];
            let in1 = &operand_types[1];
            let res0 = &result_types[0];
            write!(
                out,
                "stablehlo.reduce({lhs} init: {init}) applies {reducer} \
                 across dimensions = [{dims}]"
            )?;
            self.print_extra_attrs(out, &op.attributes, &["reducer", "dimensions"])?;
            write!(out, " : ({in0}, {in1}) -> {res0}")?;
        } else if op.is_op("stablehlo", "broadcast_in_dim") && op.results.len() == 1 {
            let in_ty = &operand_types[0];
            let out_ty = &result_types[0];
            let dims = op
                .attributes
                .get("broadcast_dimensions")
                .and_then(|a| a.as_int_slice())
                .map(|s| {
                    s.iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            write!(
                out,
                "stablehlo.broadcast_in_dim {}, dims = [{}]",
                operand_names.join(", "),
                dims
            )?;
            self.print_extra_attrs(out, &op.attributes, &["broadcast_dimensions"])?;
            write!(out, " : ({in_ty}) -> {out_ty}")?;
        } else if op.name.dialect == "stablehlo"
            && op.results.len() == 1
            && !operand_types.is_empty()
            && operand_types.iter().all(|t| t == &result_types[0])
            && op.attributes.is_empty()
        {
            // Homogeneous elementwise/unary op sugar: `stablehlo.add %0, %1 : tensor<...>`
            write!(
                out,
                "{} {} : {}",
                op.name,
                operand_names.join(", "),
                result_types[0]
            )?;
        } else {
            write!(out, "{}", op.name)?;
            if !operand_names.is_empty() {
                write!(out, " {}", operand_names.join(", "))?;
            }
            if !op.attributes.is_empty() {
                write!(out, " ")?;
                self.print_attr_dict(out, &op.attributes)?;
            }
            let fn_ty = Type::Function {
                inputs: operand_types,
                results: result_types,
            };
            write!(out, " : {fn_ty}")?;
        }

        let op_regions = op.regions.clone();
        let op_loc = op.location.clone();
        if !op_regions.is_empty() {
            writeln!(out, " {{")?;
            for reg_id in op_regions {
                if let Some(reg) = self.module.regions.get(reg_id) {
                    let blocks = reg.blocks.clone();
                    for blk_id in blocks {
                        let ops = self.module.blocks[blk_id].ops.clone();
                        for child_op in ops {
                            self.print_op(out, child_op, indent + 1)?;
                        }
                    }
                }
            }
            write!(out, "{pad}}}")?;
        }

        if self.opts.print_locations {
            write!(out, " {op_loc}")?;
        }
        writeln!(out)?;
        Ok(())
    }

    fn print_func_op(&mut self, out: &mut String, op_id: OpId, indent: usize) -> fmt::Result {
        self.value_names.clear();
        self.used_names.clear();
        self.next_anon_id = 0;

        let pad = "  ".repeat(indent);
        let op = &self.module.ops[op_id];
        let sym_name = op
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or("main");

        let Some(&fn_reg) = op.regions.first() else {
            return Ok(());
        };
        let Some(&entry_block) = self.module.regions[fn_reg].blocks.first() else {
            return Ok(());
        };
        let args = &self.module.blocks[entry_block].arguments;

        let mut arg_strs = Vec::with_capacity(args.len());
        for &arg_id in args {
            let name = self.assign_value_name(arg_id);
            let ty = &self.module.values[arg_id].ty;
            arg_strs.push(format!("{name}: {ty}"));
        }

        let ret_str = match op.attributes.get("function_type") {
            Some(Attribute::Type(Type::Function { results, .. })) if !results.is_empty() => {
                if results.len() == 1 {
                    format!(" -> {}", results[0])
                } else {
                    let s: Vec<String> = results.iter().map(|r| r.to_string()).collect();
                    format!(" -> ({})", s.join(", "))
                }
            }
            _ => String::new(),
        };

        let vis_str = match op.attributes.get("sym_visibility").and_then(|a| a.as_str()) {
            Some("private") => "private ",
            _ => "",
        };

        let extra_fn_attrs: Vec<(&String, &Attribute)> = op
            .attributes
            .iter()
            .filter(|(k, _)| {
                !matches!(
                    k.as_str(),
                    "sym_name" | "function_type" | "sym_visibility"
                )
            })
            .collect();
        let mut fn_attrs_str = String::new();
        if !extra_fn_attrs.is_empty() {
            fn_attrs_str.push_str(" attributes {");
            for (i, (k, v)) in extra_fn_attrs.iter().enumerate() {
                if i > 0 {
                    fn_attrs_str.push_str(", ");
                }
                let _ = write!(fn_attrs_str, "{k} = {v}");
            }
            fn_attrs_str.push('}');
        }

        let loc_str = if self.opts.print_locations {
            format!(" {}", op.location)
        } else {
            String::new()
        };

        writeln!(
            out,
            "{pad}func.func {vis_str}@{sym_name}({}){ret_str}{fn_attrs_str}{loc_str} {{",
            arg_strs.join(", ")
        )?;

        let ops = self.module.blocks[entry_block].ops.clone();
        for child_op in ops {
            self.print_op(out, child_op, indent + 1)?;
        }

        writeln!(out, "{pad}}}")?;
        Ok(())
    }

    fn print_extra_attrs(
        &self,
        out: &mut String,
        attrs: &indexmap::IndexMap<String, Attribute>,
        skip_keys: &[&str],
    ) -> fmt::Result {
        let extra: Vec<(&String, &Attribute)> = attrs
            .iter()
            .filter(|(k, _)| !skip_keys.contains(&k.as_str()))
            .collect();
        if extra.is_empty() {
            return Ok(());
        }
        write!(out, " {{")?;
        for (i, (k, v)) in extra.iter().enumerate() {
            if i > 0 {
                write!(out, ", ")?;
            }
            write!(out, "{k} = {v}")?;
        }
        write!(out, "}}")
    }

    fn print_attr_dict(
        &self,
        out: &mut String,
        attrs: &indexmap::IndexMap<String, Attribute>,
    ) -> fmt::Result {
        write!(out, "{{")?;
        for (i, (k, v)) in attrs.iter().enumerate() {
            if i > 0 {
                write!(out, ", ")?;
            }
            write!(out, "{k} = {v}")?;
        }
        write!(out, "}}")
    }
}

/// Format a CastForge `Module` into MLIR text with default pretty syntax.
pub fn print_mlir(module: &Module) -> String {
    let mut printer = MlirPrinter::new(module, PrintOptions::default());
    printer.print_module().unwrap_or_default()
}

/// Format a CastForge `Module` into MLIR text with custom `PrintOptions`.
pub fn print_mlir_with_options(module: &Module, opts: PrintOptions) -> String {
    let mut printer = MlirPrinter::new(module, opts);
    printer.print_module().unwrap_or_default()
}

/// A-to-B Translator (`Ir(Module) -> Text(MLIR)`).
#[derive(Default)]
pub struct MlirPrintTranslator {
    pub options: PrintOptions,
}

impl crate::ir::Translator for MlirPrintTranslator {
    fn name(&self) -> &str {
        "print-mlir"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> anyhow::Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        Ok(crate::ir::Payload::Text(print_mlir_with_options(
            &module,
            self.options,
        )))
    }
}
