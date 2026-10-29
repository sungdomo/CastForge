use crate::parser_types::{
    extract_trailing_loc, find_matching_delim, find_top_level_colon, find_top_level_eq,
    parse_attr_dict, parse_int_list, parse_type, split_top_level,
};
use castforge_ir::{Attribute, BlockId, DType, Module, OpName, TensorType, Type, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Line {line}: {message}")]
    Syntax { line: usize, message: String },

    #[error("Line {line}: Undefined SSA value `%{name}`")]
    UndefinedValue { line: usize, name: String },
}

pub struct MlirParser {
    pub(crate) value_map: HashMap<String, ValueId>,
}

impl Default for MlirParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MlirParser {
    pub fn new() -> Self {
        Self {
            value_map: HashMap::new(),
        }
    }

    pub fn parse_module(&mut self, input: &str) -> Result<Module, ParseError> {
        let mut module = Module::new(None);
        let mut current_block: BlockId = module.body_block();
        let mut in_func = false;

        let mut logical_lines: Vec<(usize, String)> = Vec::new();
        for (idx, raw_line) in input.lines().enumerate() {
            let line_no = idx + 1;
            let line = match raw_line.find("//") {
                Some(pos) => raw_line[..pos].trim(),
                None => raw_line.trim(),
            };
            if line.is_empty() {
                continue;
            }
            let prev_incomplete = logical_lines.last().is_some_and(|(_, prev)| {
                let open_p = prev.chars().filter(|&c| c == '(').count();
                let close_p = prev.chars().filter(|&c| c == ')').count();
                let open_b = prev.chars().filter(|&c| c == '[').count();
                let close_b = prev.chars().filter(|&c| c == ']').count();
                open_p > close_p || open_b > close_b
            });
            let is_continuation = prev_incomplete
                || line.starts_with(':')
                || line.starts_with("->")
                || line.starts_with("applies ")
                || line.starts_with("across ")
                || line.starts_with('{')
                || line.starts_with('[');
            if is_continuation && let Some((_prev_ln, prev_text)) = logical_lines.last_mut() {
                prev_text.push(' ');
                prev_text.push_str(line);
                continue;
            }
            logical_lines.push((line_no, line.to_string()));
        }

        for (line_no, line_buf) in logical_lines {
            let line = line_buf.as_str();

            if let Some(rest) = line.strip_prefix("module") {
                let rest = rest.trim();
                if let Some(at_rest) = rest.strip_prefix('@') {
                    let name_end = at_rest
                        .find(|c: char| c.is_whitespace() || c == '{')
                        .unwrap_or(at_rest.len());
                    module.name = Some(at_rest[..name_end].to_string());
                }
                continue;
            }

            if line == "}" {
                if in_func {
                    in_func = false;
                    current_block = module.body_block();
                }
                continue;
            }

            if line.starts_with("func.func ") {
                current_block = self.parse_func_header(&mut module, line, line_no)?;
                in_func = true;
                continue;
            }

            self.parse_operation_line(&mut module, current_block, line, line_no)?;
        }

        Ok(module)
    }

    fn parse_operation_line(
        &mut self,
        module: &mut Module,
        block: BlockId,
        line: &str,
        line_no: usize,
    ) -> Result<(), ParseError> {
        // Strip optional trailing `loc(...)`
        let (line, location) = extract_trailing_loc(line);

        // Split LHS results if `=` is present before any quotes/braces
        let (lhs_names, rhs) = if let Some(eq_idx) = find_top_level_eq(line) {
            let lhs = line[..eq_idx].trim();
            let names: Vec<String> = lhs
                .split(',')
                .map(|s| s.trim().trim_start_matches('%').to_string())
                .collect();
            (names, line[eq_idx + 1..].trim())
        } else {
            (Vec::new(), line.trim())
        };

        // 1. Check if Generic MLIR form: `"dialect.op"(...)`
        if rhs.starts_with('"') {
            return self.parse_generic_op(module, block, lhs_names, rhs, location, line_no);
        }

        // 2. Check `return` / `func.return`
        if let Some(rest) = rhs
            .strip_prefix("func.return")
            .or_else(|| rhs.strip_prefix("return"))
        {
            let rest = rest.trim();
            let mut operands = SmallVec::new();
            if !rest.is_empty() {
                let (ops_part, _tys_part) = rest.split_once(':').unwrap_or((rest, ""));
                for op_tok in split_top_level(ops_part, ',') {
                    let ssa = op_tok.trim().trim_start_matches('%');
                    if !ssa.is_empty() {
                        let val_id =
                            *self
                                .value_map
                                .get(ssa)
                                .ok_or_else(|| ParseError::UndefinedValue {
                                    line: line_no,
                                    name: ssa.to_string(),
                                })?;
                        operands.push(val_id);
                    }
                }
            }
            module.add_operation(
                block,
                OpName::new("func", "return"),
                operands,
                Vec::new(),
                IndexMap::new(),
                Vec::new(),
                true,
                location,
            );
            return Ok(());
        }

        // 3. Custom / Pretty syntax: `<op_name> ...`
        let op_end = rhs
            .find(|c: char| c.is_whitespace() || c == '(')
            .unwrap_or(rhs.len());
        let op_name = OpName::parse(&rhs[..op_end]);
        let rest = rhs[op_end..].trim();

        // Special case: `stablehlo.reduce(%in init: %init) applies <op> across dimensions = [...]`
        if op_name.dialect == "stablehlo" && op_name.name == "reduce" && rest.starts_with('(') {
            return self.parse_reduce_short_op(module, block, lhs_names, rest, location, line_no);
        }

        // Special case: `stablehlo.iota dim = 0 : tensor<...>`
        if op_name.dialect == "stablehlo" && op_name.name == "iota" && rest.starts_with("dim") {
            let colon_idx = find_top_level_colon(rest).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: "Expected `: <type>` in `stablehlo.iota`".to_string(),
            })?;
            let dim_part = rest[..colon_idx]
                .trim()
                .trim_start_matches("dim")
                .trim()
                .trim_start_matches('=')
                .trim();
            let dim_val = dim_part.parse::<i64>().unwrap_or(0);
            let ty = parse_type(rest[colon_idx + 1..].trim(), line_no)?;
            let mut attrs = IndexMap::new();
            attrs.insert("iota_dimension".to_string(), Attribute::Int(dim_val));
            let res_name = lhs_names.into_iter().next();
            let op_id = module.add_operation(
                block,
                op_name,
                SmallVec::new(),
                vec![(res_name.clone(), ty)],
                attrs,
                Vec::new(),
                true,
                location,
            );
            if let Some(name) = res_name {
                self.value_map.insert(name, module.ops[op_id].results[0]);
            }
            return Ok(());
        }

        // Special case: `func.call @callee(%a, %b) : (...) -> ...`
        if op_name.dialect == "func" && op_name.name == "call" {
            let colon_idx = find_top_level_colon(rest).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: "Expected `:` type signature in `func.call`".to_string(),
            })?;
            let call_part = rest[..colon_idx].trim();
            let ty_part = rest[colon_idx + 1..].trim();
            let open_paren = call_part.find('(').ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: "Expected `(` in `func.call`".to_string(),
            })?;
            let callee = call_part[..open_paren]
                .trim()
                .trim_start_matches('@')
                .to_string();
            let close_paren = find_matching_delim(&call_part[open_paren + 1..], '(', ')')
                .ok_or_else(|| ParseError::Syntax {
                    line: line_no,
                    message: "Unclosed `(` in `func.call`".to_string(),
                })?
                + open_paren
                + 1;
            let args_inner = &call_part[open_paren + 1..close_paren];
            let mut operands = SmallVec::new();
            for tok in split_top_level(args_inner, ',') {
                let ssa = tok.trim().trim_start_matches('%');
                if !ssa.is_empty() {
                    let val_id =
                        *self
                            .value_map
                            .get(ssa)
                            .ok_or_else(|| ParseError::UndefinedValue {
                                line: line_no,
                                name: ssa.to_string(),
                            })?;
                    operands.push(val_id);
                }
            }
            let parsed_ty = parse_type(ty_part, line_no)?;
            let result_types = match parsed_ty {
                Type::Function { results, .. } => results,
                other => vec![other],
            };
            let named_results: Vec<(Option<String>, Type)> = result_types
                .into_iter()
                .enumerate()
                .map(|(i, t)| (lhs_names.get(i).cloned(), t))
                .collect();
            let mut attrs = IndexMap::new();
            attrs.insert("callee".to_string(), Attribute::SymbolRef(callee));
            let op_id = module.add_operation(
                block,
                op_name,
                operands,
                named_results,
                attrs,
                Vec::new(),
                true,
                location,
            );
            for (i, name) in lhs_names.into_iter().enumerate() {
                if let Some(&val_id) = module.ops[op_id].results.get(i) {
                    self.value_map.insert(name, val_id);
                }
            }
            return Ok(());
        }

        // Special case: `stablehlo.constant dense<...> : tensor<...>`
        if op_name.dialect == "stablehlo" && op_name.name == "constant" {
            let colon_idx = find_top_level_colon(rest).ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: "Expected `: <type>` in `stablehlo.constant`".to_string(),
            })?;
            let literal_part = rest[..colon_idx].trim();
            let ty = parse_type(rest[colon_idx + 1..].trim(), line_no)?;
            let tensor_ty = ty
                .as_tensor()
                .cloned()
                .unwrap_or_else(|| TensorType::new(Vec::new(), DType::F32));

            let val_attr = if let Some(inner) = literal_part
                .strip_prefix("dense<")
                .and_then(|s| s.strip_suffix('>'))
            {
                let res_key = lhs_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| format!("const_{}", module.tensor_store.len()));
                let num_elems = tensor_ty.num_elements().unwrap_or(1) as usize;
                if let Some(f32_vals) = castforge_ir::parse_dense_f32_literal(inner, num_elems) {
                    module
                        .tensor_store
                        .insert(castforge_ir::TensorEntry::from_f32(
                            res_key,
                            tensor_ty.clone(),
                            f32_vals,
                        ));
                }
                Attribute::DenseElements {
                    ty: tensor_ty,
                    raw_literal: inner.to_string(),
                }
            } else if let Some(key) = literal_part
                .strip_prefix("dense_resource<")
                .and_then(|s| s.strip_suffix('>'))
            {
                Attribute::TensorRef {
                    key: key.trim().to_string(),
                    ty: tensor_ty,
                }
            } else {
                Attribute::Opaque(literal_part.to_string())
            };

            let mut attrs = IndexMap::new();
            attrs.insert("value".to_string(), val_attr);
            let res_name = lhs_names.into_iter().next();
            let op_id = module.add_operation(
                block,
                op_name,
                SmallVec::new(),
                vec![(res_name.clone(), ty)],
                attrs,
                Vec::new(),
                true,
                location,
            );
            if let Some(name) = res_name {
                self.value_map.insert(name, module.ops[op_id].results[0]);
            }
            return Ok(());
        }

        // General custom syntax: `<operands> [, dims = [...]] [{attrs}] : <type_or_fn_type>`
        let colon_idx = find_top_level_colon(rest).ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: format!("Expected `:` type signature in operation `{rhs}`"),
        })?;
        let before_colon = rest[..colon_idx].trim();
        let after_colon = rest[colon_idx + 1..].trim();

        let mut operands = SmallVec::new();
        let mut attrs = IndexMap::new();

        // Extract optional `{...}` attribute dictionary in `before_colon`
        let before_attrs = if let Some(brace_idx) = before_colon.find('{') {
            let dict_str = &before_colon[brace_idx..];
            parse_attr_dict(dict_str, &mut attrs);
            before_colon[..brace_idx].trim()
        } else {
            before_colon
        };

        for item in split_top_level(before_attrs, ',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            if let Some(dims_rest) = item.strip_prefix("dims") {
                let dims_rest = dims_rest.trim().trim_start_matches('=').trim();
                let nums = parse_int_list(dims_rest);
                let attr_key = if op_name.name == "transpose" {
                    "permutation"
                } else {
                    "broadcast_dimensions"
                };
                attrs.insert(attr_key.to_string(), Attribute::DenseIntArray(nums));
            } else if let Some(ssa) = item.strip_prefix('%') {
                let val_id =
                    *self
                        .value_map
                        .get(ssa.trim())
                        .ok_or_else(|| ParseError::UndefinedValue {
                            line: line_no,
                            name: ssa.trim().to_string(),
                        })?;
                operands.push(val_id);
            } else if op_name.name == "compare"
                && matches!(item, "EQ" | "NE" | "LT" | "LE" | "GT" | "GE")
            {
                attrs.insert(
                    "comparison_direction".to_string(),
                    Attribute::String(item.to_string()),
                );
            }
        }

        let parsed_ty = parse_type(after_colon, line_no)?;
        let result_types: Vec<Type> = match parsed_ty {
            Type::Function { results, .. } => results,
            other => vec![other],
        };

        let named_results: Vec<(Option<String>, Type)> = result_types
            .into_iter()
            .enumerate()
            .map(|(i, t)| (lhs_names.get(i).cloned(), t))
            .collect();

        let op_id = module.add_operation(
            block,
            op_name,
            operands,
            named_results,
            attrs,
            Vec::new(),
            true,
            location,
        );

        for (i, name) in lhs_names.into_iter().enumerate() {
            if let Some(&val_id) = module.ops[op_id].results.get(i) {
                self.value_map.insert(name, val_id);
            }
        }

        Ok(())
    }
}
