use crate::parser::{MlirParser, ParseError};
use crate::parser_types::{
    find_matching_delim, find_top_level_colon, parse_attr_dict, parse_int_list, parse_type,
    split_top_level,
};
use castforge_ir::{Attribute, BlockId, Location, Module, OpName, Type};
use indexmap::IndexMap;
use smallvec::SmallVec;

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
