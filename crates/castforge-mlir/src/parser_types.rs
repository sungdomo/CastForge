use crate::parser::ParseError;
use castforge_ir::{Attribute, DType, Dim, Location, TensorType, Type};
use indexmap::IndexMap;

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
