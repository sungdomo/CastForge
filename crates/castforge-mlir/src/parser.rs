use castforge_ir::{
    Attribute, BlockId, DType, Dim, Location, Module, OpName, TensorType, Type, ValueId,
};
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

impl MlirParser {
    pub(crate) fn parse_func_header(
        &mut self,
        module: &mut Module,
        line: &str,
        line_no: usize,
    ) -> Result<BlockId, ParseError> {
        self.value_map.clear();
        let after_kw = line.trim_start_matches("func.func").trim();
        let is_private = after_kw.starts_with("private ");
        let at_idx = line.find('@').ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: "Expected `@func_name` in `func.func`".to_string(),
        })?;
        let after_at = &line[at_idx + 1..];
        let open_paren = after_at.find('(').ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: "Expected `(` after function name".to_string(),
        })?;
        let sym_name = after_at[..open_paren].trim().to_string();

        // Find matching close paren for argument list
        let args_part = &after_at[open_paren + 1..];
        let close_paren =
            find_matching_delim(args_part, '(', ')').ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: "Unclosed argument list in `func.func`".to_string(),
            })?;
        let args_str = &args_part[..close_paren];
        let after_args = args_part[close_paren + 1..].trim();

        let fn_block = module.create_block(None);
        let fn_region = module.create_region(vec![fn_block]);

        let mut input_types = Vec::new();
        for arg_item in split_top_level(args_str, ',') {
            let arg_item = arg_item.trim();
            if arg_item.is_empty() {
                continue;
            }
            let (val_str, ty_str) = arg_item.split_once(':').ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("Invalid function argument `{arg_item}`"),
            })?;
            let ssa = val_str.trim().trim_start_matches('%').to_string();
            let ty = parse_type(ty_str.trim(), line_no)?;
            input_types.push(ty.clone());
            let val_id = module.add_block_argument(fn_block, Some(ssa.clone()), ty);
            self.value_map.insert(ssa, val_id);
        }

        let mut result_types = Vec::new();
        if let Some(arrow_rest) = after_args.strip_prefix("->") {
            let ret_part = arrow_rest.trim().trim_end_matches('{').trim();
            if ret_part.starts_with('(') && ret_part.ends_with(')') {
                let inner = &ret_part[1..ret_part.len() - 1];
                for item in split_top_level(inner, ',') {
                    if !item.trim().is_empty() {
                        result_types.push(parse_type(item.trim(), line_no)?);
                    }
                }
            } else if !ret_part.is_empty() {
                result_types.push(parse_type(ret_part, line_no)?);
            }
        }

        let mut attrs = IndexMap::new();
        attrs.insert("sym_name".to_string(), Attribute::String(sym_name));
        if is_private {
            attrs.insert(
                "sym_visibility".to_string(),
                Attribute::String("private".to_string()),
            );
        }
        attrs.insert(
            "function_type".to_string(),
            Attribute::Type(Type::Function {
                inputs: input_types,
                results: result_types,
            }),
        );

        let root_block = module.body_block();
        module.add_operation(
            root_block,
            OpName::new("func", "func"),
            SmallVec::new(),
            Vec::new(),
            attrs,
            vec![fn_region],
            true,
            Location::Unknown,
        );

        Ok(fn_block)
    }

    pub(crate) fn parse_reduce_short_op(
        &mut self,
        module: &mut Module,
        block: BlockId,
        lhs_names: Vec<String>,
        rest: &str,
        location: Location,
        line_no: usize,
    ) -> Result<(), ParseError> {
        let close_paren =
            find_matching_delim(&rest[1..], '(', ')').ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: "Unclosed `(` in `stablehlo.reduce`".to_string(),
            })? + 1;
        let inside_parens = &rest[1..close_paren];
        let (in_part, init_part) =
            inside_parens
                .split_once("init:")
                .ok_or_else(|| ParseError::Syntax {
                    line: line_no,
                    message: "Expected `init:` in `stablehlo.reduce`".to_string(),
                })?;

        let mut operands = SmallVec::new();
        for tok in [in_part.trim(), init_part.trim()] {
            let ssa = tok.trim_start_matches('%').trim();
            let val_id = *self
                .value_map
                .get(ssa)
                .ok_or_else(|| ParseError::UndefinedValue {
                    line: line_no,
                    name: ssa.to_string(),
                })?;
            operands.push(val_id);
        }

        let after_parens = rest[close_paren + 1..].trim();
        let colon_idx = find_top_level_colon(after_parens).ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: "Expected `:` type signature in `stablehlo.reduce`".to_string(),
        })?;
        let spec_part = after_parens[..colon_idx].trim();
        let ty_part = after_parens[colon_idx + 1..].trim();

        let mut attrs = IndexMap::new();
        if let Some(applies_rest) = spec_part.strip_prefix("applies") {
            let applies_rest = applies_rest.trim();
            if let Some((reducer, dims_part)) = applies_rest.split_once("across") {
                attrs.insert(
                    "reducer".to_string(),
                    Attribute::String(reducer.trim().to_string()),
                );
                if let Some((_, bracket_part)) = dims_part.split_once('=') {
                    let dims = parse_int_list(bracket_part.trim());
                    attrs.insert("dimensions".to_string(), Attribute::DenseIntArray(dims));
                }
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

        let op_id = module.add_operation(
            block,
            OpName::new("stablehlo", "reduce"),
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

    pub(crate) fn parse_generic_op(
        &mut self,
        module: &mut Module,
        block: BlockId,
        lhs_names: Vec<String>,
        rhs: &str,
        location: Location,
        line_no: usize,
    ) -> Result<(), ParseError> {
        let second_quote = rhs[1..].find('"').ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: "Unclosed operation name quote".to_string(),
        })? + 1;
        let op_name = OpName::parse(&rhs[1..second_quote]);
        let after_name = rhs[second_quote + 1..].trim();

        let open_paren = after_name.find('(').ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: "Expected `(` after generic operation name".to_string(),
        })?;
        let close_paren =
            find_matching_delim(&after_name[open_paren + 1..], '(', ')').ok_or_else(|| {
                ParseError::Syntax {
                    line: line_no,
                    message: "Unclosed operand list `(`".to_string(),
                }
            })? + open_paren
                + 1;

        let operands_str = &after_name[open_paren + 1..close_paren];
        let mut operands = SmallVec::new();
        for tok in split_top_level(operands_str, ',') {
            let tok = tok.trim();
            if tok.is_empty() {
                continue;
            }
            let ssa = tok.trim_start_matches('%');
            let val_id = *self
                .value_map
                .get(ssa)
                .ok_or_else(|| ParseError::UndefinedValue {
                    line: line_no,
                    name: ssa.to_string(),
                })?;
            operands.push(val_id);
        }

        let after_operands = after_name[close_paren + 1..].trim();
        let colon_idx = find_top_level_colon(after_operands).ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: "Expected `:` type signature in generic op".to_string(),
        })?;

        let attr_part = after_operands[..colon_idx].trim();
        let mut attrs = IndexMap::new();
        if !attr_part.is_empty() {
            parse_attr_dict(attr_part, &mut attrs);
        }

        let fn_ty = parse_type(after_operands[colon_idx + 1..].trim(), line_no)?;
        let result_types = match fn_ty {
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
            false,
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

pub(crate) fn extract_trailing_loc(line: &str) -> (&str, Location) {
    if let Some(idx) = line.rfind(" loc(") {
        let loc_part = line[idx + 5..].trim().trim_end_matches(')');
        let loc = if loc_part == "unknown" {
            Location::Unknown
        } else {
            Location::Named(loc_part.trim_matches('"').to_string())
        };
        (&line[..idx], loc)
    } else {
        (line, Location::Unknown)
    }
}

pub(crate) fn find_top_level_eq(s: &str) -> Option<usize> {
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '{' | '(' | '<' if !in_quotes => return None,
            '=' if !in_quotes => return Some(i),
            _ => {}
        }
    }
    None
}

pub(crate) fn find_top_level_colon(s: &str) -> Option<usize> {
    let mut depth_brace = 0usize;
    let mut depth_angle = 0usize;
    let mut depth_brack = 0usize;
    let mut in_quotes = false;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => in_quotes = !in_quotes,
            b'{' if !in_quotes => depth_brace += 1,
            b'}' if !in_quotes => depth_brace = depth_brace.saturating_sub(1),
            b'<' if !in_quotes => depth_angle += 1,
            b'>' if !in_quotes => depth_angle = depth_angle.saturating_sub(1),
            b'[' if !in_quotes => depth_brack += 1,
            b']' if !in_quotes => depth_brack = depth_brack.saturating_sub(1),
            b':' if !in_quotes && depth_brace == 0 && depth_angle == 0 && depth_brack == 0 => {
                return Some(i);
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn find_matching_delim(s: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 1usize;
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
    }
    None
}

pub(crate) fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            match c {
                '(' | '<' | '[' | '{' => depth += 1,
                ')' | '>' | ']' | '}' => depth = depth.saturating_sub(1),
                c if c == sep && depth == 0 => {
                    out.push(&s[start..i]);
                    start = i + c.len_utf8();
                }
                _ => {}
            }
        }
    }
    out.push(&s[start..]);
    out
}

pub(crate) fn parse_int_list(s: &str) -> Vec<i64> {
    let inner = s
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .filter_map(|tok| tok.trim().parse::<i64>().ok())
        .collect()
}

pub(crate) fn parse_attr_dict(s: &str, attrs: &mut IndexMap<String, Attribute>) {
    let inner = s
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();
    for entry in split_top_level(inner, ',') {
        let entry = entry.trim();
        if let Some((k, v)) = entry.split_once('=') {
            let key = k.trim().to_string();
            let val_str = v.trim();
            let attr = parse_attribute_value(val_str);
            attrs.insert(key, attr);
        }
    }
}

pub(crate) fn parse_attribute_value(val_str: &str) -> Attribute {
    if val_str == "true" {
        Attribute::Bool(true)
    } else if val_str == "false" {
        Attribute::Bool(false)
    } else if let Some(stripped) = val_str.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        Attribute::String(stripped.to_string())
    } else if let Some(sym) = val_str.strip_prefix('@') {
        Attribute::SymbolRef(sym.to_string())
    } else if let Some(arr_inner) = val_str
        .strip_prefix("array<i64:")
        .and_then(|s| s.strip_suffix('>'))
    {
        let nums = arr_inner
            .split(',')
            .filter_map(|x| x.trim().parse::<i64>().ok())
            .collect();
        Attribute::DenseIntArray(nums)
    } else if val_str.starts_with('[') && val_str.ends_with(']') {
        let nums = parse_int_list(val_str);
        Attribute::DenseIntArray(nums)
    } else if let Some(rest) = val_str.strip_prefix("dense_resource<") {
        if let Some((key_part, ty_part)) = rest.split_once('>') {
            let ty_str = ty_part.trim().trim_start_matches(':').trim();
            let ty = parse_type(ty_str, 0)
                .ok()
                .and_then(|t| t.as_tensor().cloned())
                .unwrap_or_else(|| TensorType::new(Vec::new(), DType::F32));
            return Attribute::TensorRef {
                key: key_part.trim().to_string(),
                ty,
            };
        }
        Attribute::Opaque(val_str.to_string())
    } else if let Some(rest) = val_str.strip_prefix("dense<") {
        if let Some(colon_idx) = find_top_level_colon(val_str) {
            let lit_part = val_str[..colon_idx].trim();
            let ty_part = val_str[colon_idx + 1..].trim();
            if let Some(inner) = lit_part
                .strip_prefix("dense<")
                .and_then(|s| s.strip_suffix('>'))
            {
                let ty = parse_type(ty_part, 0)
                    .ok()
                    .and_then(|t| t.as_tensor().cloned())
                    .unwrap_or_else(|| TensorType::new(Vec::new(), DType::F32));
                return Attribute::DenseElements {
                    ty,
                    raw_literal: inner.to_string(),
                };
            }
        }
        let _ = rest;
        Attribute::Opaque(val_str.to_string())
    } else if let Ok(i) = val_str.parse::<i64>() {
        Attribute::Int(i)
    } else if let Ok(f) = val_str.parse::<f64>() {
        Attribute::Float(f)
    } else {
        Attribute::Opaque(val_str.to_string())
    }
}

pub fn parse_type(s: &str, line_no: usize) -> Result<Type, ParseError> {
    let s = s.trim();
    if let Some(after_open) = s.strip_prefix('(') {
        let close =
            find_matching_delim(after_open, '(', ')').ok_or_else(|| ParseError::Syntax {
                line: line_no,
                message: format!("Unclosed `(` in function type `{s}`"),
            })? + 1;
        let args_inner = &s[1..close];
        let mut inputs = Vec::new();
        for item in split_top_level(args_inner, ',') {
            if !item.trim().is_empty() {
                inputs.push(parse_type(item.trim(), line_no)?);
            }
        }
        let rest = s[close + 1..].trim();
        let after_arrow = rest.strip_prefix("->").ok_or_else(|| ParseError::Syntax {
            line: line_no,
            message: format!("Expected `->` in function type `{s}`"),
        })?;
        let after_arrow = after_arrow.trim();
        let mut results = Vec::new();
        if after_arrow.starts_with('(') && after_arrow.ends_with(')') {
            let inner = &after_arrow[1..after_arrow.len() - 1];
            for item in split_top_level(inner, ',') {
                if !item.trim().is_empty() {
                    results.push(parse_type(item.trim(), line_no)?);
                }
            }
        } else if !after_arrow.is_empty() {
            results.push(parse_type(after_arrow, line_no)?);
        }
        return Ok(Type::Function { inputs, results });
    }

    if let Some(inner) = s.strip_prefix("tensor<").and_then(|x| x.strip_suffix('>')) {
        // Check if element type is `!quant.uniform<...>`
        if let Some(q_idx) = inner.find("!quant.uniform<") {
            let dims_prefix = &inner[..q_idx];
            let quant_part = &inner[q_idx..];
            let mut shape = Vec::new();
            for dim_tok in dims_prefix.trim_end_matches('x').split('x') {
                if dim_tok.is_empty() {
                    continue;
                }
                if dim_tok == "?" {
                    shape.push(Dim::Dynamic);
                } else if let Ok(n) = dim_tok.parse::<i64>() {
                    shape.push(Dim::Static(n));
                } else {
                    shape.push(Dim::Symbolic(dim_tok.to_string()));
                }
            }
            let dtype = parse_dtype(quant_part);
            return Ok(Type::Tensor(TensorType {
                shape,
                dtype,
                encoding: None,
            }));
        }

        // Split optional encoding after comma
        let parts = split_top_level(inner, ',');
        let shape_dtype = parts[0].trim();
        let encoding = parts.get(1).map(|e| e.trim().to_string());

        let tokens: Vec<&str> = shape_dtype.split('x').collect();
        let dtype_str = tokens.last().copied().unwrap_or("f32");
        let dtype = parse_dtype(dtype_str);
        let mut shape = Vec::with_capacity(tokens.len().saturating_sub(1));
        for &dim_tok in &tokens[..tokens.len().saturating_sub(1)] {
            if dim_tok == "?" {
                shape.push(Dim::Dynamic);
            } else if let Ok(n) = dim_tok.parse::<i64>() {
                shape.push(Dim::Static(n));
            } else {
                shape.push(Dim::Symbolic(dim_tok.to_string()));
            }
        }
        return Ok(Type::Tensor(TensorType {
            shape,
            dtype,
            encoding,
        }));
    }

    Ok(Type::Scalar(parse_dtype(s)))
}

pub(crate) fn parse_dtype(s: &str) -> DType {
    let s = s.trim();
    if let Some(q_inner) = s
        .strip_prefix("!quant.uniform<")
        .and_then(|x| x.strip_suffix('>'))
        && let Some((types_part, scale_part)) = q_inner.split_once(',')
    {
        let (st_str, ex_str) = types_part.split_once(':').unwrap_or(("i8", "f32"));
        let (sc_str, zp_str) = scale_part.trim().split_once(':').unwrap_or(("1.0", "0"));
        return DType::QuantUniform {
            storage_type: Box::new(parse_dtype(st_str.trim())),
            expressed_type: Box::new(parse_dtype(ex_str.trim())),
            scale: sc_str.trim().to_string(),
            zero_point: zp_str.trim().parse::<i64>().unwrap_or(0),
        };
    }
    match s {
        "i1" => DType::I1,
        "i4" => DType::I4,
        "i8" => DType::I8,
        "i16" => DType::I16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "ui8" => DType::U8,
        "ui16" => DType::U16,
        "ui32" => DType::U32,
        "ui64" => DType::U64,
        "f8E4M3FN" => DType::F8E4M3FN,
        "f8E5M2" => DType::F8E5M2,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        "index" => DType::Index,
        _ => DType::F32,
    }
}
