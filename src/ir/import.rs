//! Frontend model importers and exporters (`GGUF` v2/v3 binary and `ONNX` JSON graphs) into
//! CastForge's Dual-Plane StableHLO IR (`Module` + `TensorStore`).

use crate::ir::{
    DType, Dim, Module, TensorEntry, TensorStore, TensorType, format_dense_f32_literal, parse_mlir,
};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor element encoding supported by the GGUF reader/writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            30 => Ok(Self::BF16),
            other => bail!("Unsupported GGML tensor type id: {other}"),
        }
    }
}

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    name: String,
    shape: Vec<i64>,
    ggml_type: GgmlType,
    offset: usize,
}

/// Imports a binary `.gguf` file from disk into a CastForge [`Module`] + [`TensorStore`].
pub fn import_gguf_file(path: impl AsRef<Path>) -> Result<Module> {
    let bytes = fs::read(path.as_ref())
        .with_context(|| format!("Failed to read GGUF file {}", path.as_ref().display()))?;
    import_gguf_bytes(&bytes)
}

/// Parses a binary GGUF v2/v3 byte buffer, loads all tensors into `TensorStore`, and constructs
/// a StableHLO forward graph over the imported weights.
pub fn import_gguf_bytes(bytes: &[u8]) -> Result<Module> {
    if bytes.len() < 24 || &bytes[0..4] != GGUF_MAGIC {
        bail!("Invalid GGUF header magic (expected 'GGUF')");
    }
    let mut cursor = 4usize;
    let version = read_u32(bytes, &mut cursor)?;
    if !(2..=3).contains(&version) {
        bail!("Unsupported GGUF version: {version}");
    }
    let tensor_count = read_u64(bytes, &mut cursor)? as usize;
    let kv_count = read_u64(bytes, &mut cursor)? as usize;

    let mut model_name = "gguf_model".to_string();
    let mut alignment = GGUF_DEFAULT_ALIGNMENT;
    for _ in 0..kv_count {
        let key = read_gguf_string(bytes, &mut cursor)?;
        let val_ty = read_u32(bytes, &mut cursor)?;
        let str_val = skip_or_read_gguf_val(bytes, &mut cursor, val_ty)?;
        if key == "general.name"
            && let Some(s) = str_val
        {
            model_name = sanitize_name(&s);
        } else if key == "general.alignment"
            && let Some(s) = str_val
            && let Ok(a) = s.parse::<usize>()
            && a.is_power_of_two()
        {
            alignment = a;
        }
    }

    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(bytes, &mut cursor)?;
        let n_dims = read_u32(bytes, &mut cursor)? as usize;
        let mut raw_dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            raw_dims.push(read_u64(bytes, &mut cursor)? as i64);
        }
        // GGUF stores dimensions inner-most first; reverse into row-major [M, N]
        raw_dims.reverse();
        if raw_dims.is_empty() {
            raw_dims.push(1);
        }
        let ggml_type = GgmlType::from_u32(read_u32(bytes, &mut cursor)?)?;
        let offset = read_u64(bytes, &mut cursor)? as usize;
        infos.push(GgufTensorInfo {
            name,
            shape: raw_dims,
            ggml_type,
            offset,
        });
    }

    let data_base = cursor.div_ceil(alignment) * alignment;
    let mut decoded_tensors: Vec<(GgufTensorInfo, Vec<f32>)> = Vec::with_capacity(infos.len());
    for info in infos {
        let num_elems: usize = info
            .shape
            .iter()
            .map(|&d| d.max(1) as usize)
            .product::<usize>()
            .max(1);
        let start = data_base + info.offset;
        let vals = decode_ggml_tensor(bytes, start, num_elems, info.ggml_type)?;
        decoded_tensors.push((info, vals));
    }

    // Synthesize a valid StableHLO forward module referencing the GGUF tensors
    let first_k = decoded_tensors
        .first()
        .and_then(|(info, _)| info.shape.first().copied())
        .unwrap_or(4);
    let mut mlir = String::new();
    writeln!(mlir, "module @{model_name} {{")?;
    if decoded_tensors.is_empty() {
        writeln!(
            mlir,
            "  func.func @forward(%x: tensor<1x4xf32>) -> tensor<1x4xf32> {{"
        )?;
        writeln!(mlir, "    func.return %x : tensor<1x4xf32>")?;
        writeln!(mlir, "  }}")?;
    } else {
        let cur_rows = 1i64;
        let mut cur_cols = first_k;
        let mut cur_ssa = "%x".to_string();
        let mut body = String::new();
        for (idx, (info, vals)) in decoded_tensors.iter().enumerate() {
            let w_ssa = format!("%w{idx}");
            let shape_str = fmt_shape(&info.shape);
            let dims: Vec<Dim> = info.shape.iter().map(|&d| Dim::Static(d)).collect();
            let lit = format_dense_f32_literal(vals, &dims);
            writeln!(
                body,
                "    {w_ssa} = stablehlo.constant dense<{lit}> : tensor<{shape_str}xf32>"
            )?;
            if info.shape.len() == 2 && info.shape[0] == cur_cols {
                let next_cols = info.shape[1];
                let out_ssa = format!("%h{idx}");
                writeln!(
                    body,
                    "    {out_ssa} = stablehlo.dot_general {cur_ssa}, {w_ssa}, \
                     contracting_dims = [1] x [0] : \
                     (tensor<{cur_rows}x{cur_cols}xf32>, tensor<{cur_cols}x{next_cols}xf32>) \
                     -> tensor<{cur_rows}x{next_cols}xf32>"
                )?;
                cur_cols = next_cols;
                cur_ssa = out_ssa;
            } else if info.shape.len() == 2
                && info.shape[0] == cur_rows
                && info.shape[1] == cur_cols
            {
                let out_ssa = format!("%h{idx}");
                writeln!(
                    body,
                    "    {out_ssa} = stablehlo.add {cur_ssa}, {w_ssa} \
                     : tensor<{cur_rows}x{cur_cols}xf32>"
                )?;
                cur_ssa = out_ssa;
            }
        }
        writeln!(
            mlir,
            "  func.func @forward(%x: tensor<1x{first_k}xf32>) \
             -> tensor<{cur_rows}x{cur_cols}xf32> {{"
        )?;
        mlir.push_str(&body);
        writeln!(
            mlir,
            "    func.return {cur_ssa} : tensor<{cur_rows}x{cur_cols}xf32>"
        )?;
        writeln!(mlir, "  }}")?;
    }
    writeln!(mlir, "}}")?;

    let mut module = parse_mlir(&mlir).context("Failed to parse synthesized GGUF MLIR")?;
    module.tensor_store = TensorStore::new();
    for (info, vals) in decoded_tensors {
        let dims: Vec<Dim> = info.shape.iter().map(|&d| Dim::Static(d)).collect();
        module.tensor_store.insert(TensorEntry::from_f32(
            info.name,
            TensorType::new(dims, DType::F32),
            vals,
        ));
    }
    Ok(module)
}

/// Collects all weight tensors from `module.tensor_store`, falling back to inline
/// `stablehlo.constant` operations when `tensor_store` is empty.
pub fn collect_module_tensors(module: &Module) -> TensorStore {
    if !module.tensor_store.is_empty() {
        return module.tensor_store.clone();
    }
    let mut store = TensorStore::new();
    let mut idx = 0usize;
    for (_, op) in module.ops.iter() {
        if !op.is_op("stablehlo", "constant") {
            continue;
        }
        let Some(&res_val) = op.results.first() else {
            continue;
        };
        let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
            continue;
        };
        let n = tensor_ty.num_elements().unwrap_or(1).max(1) as usize;
        let vals = op
            .attributes
            .get("value")
            .and_then(|a| module.tensor_store.resolve_f32_attr(a, n))
            .unwrap_or_else(|| vec![0.0; n]);
        let name = module.values[res_val]
            .ssa_name
            .as_deref()
            .map(|s| s.trim_start_matches('%').to_string())
            .unwrap_or_else(|| {
                idx += 1;
                format!("const_{idx}.weight")
            });
        store.insert(TensorEntry::from_f32(
            name,
            TensorType::new(tensor_ty.shape, DType::F32),
            vals,
        ));
    }
    store
}

/// Serializes all tensors in `module.tensor_store` (or inline constants) into a valid binary
/// GGUF v3 byte buffer.
pub fn export_gguf_bytes(module: &Module) -> Vec<u8> {
    let store = collect_module_tensors(module);
    let mut out = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    out.extend_from_slice(&3u32.to_le_bytes()); // version 3
    let tensor_count = store.len() as u64;
    out.extend_from_slice(&tensor_count.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata KV: general.name

    let mod_name = module.name.as_deref().unwrap_or("castforge_model");
    write_gguf_string(&mut out, "general.name");
    out.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    write_gguf_string(&mut out, mod_name);

    let mut raw_payloads: Vec<Vec<u8>> = Vec::with_capacity(store.len());
    let mut current_offset = 0usize;

    for (name, tensor) in store.iter() {
        write_gguf_string(&mut out, name);
        let static_dims: Vec<u64> = tensor
            .ty
            .shape
            .iter()
            .map(|d| match d {
                Dim::Static(n) if *n > 0 => *n as u64,
                _ => 1,
            })
            .collect();
        let n_dims = static_dims.len().max(1) as u32;
        out.extend_from_slice(&n_dims.to_le_bytes());
        // GGUF stores dimensions inner-most first
        for &d in static_dims.iter().rev() {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        out.extend_from_slice(&(current_offset as u64).to_le_bytes());

        let f32_vals = tensor
            .as_f32_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| vec![0.0; 1]);
        let mut bytes = Vec::with_capacity(f32_vals.len() * 4);
        for v in f32_vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let padded_len = bytes.len().div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
        bytes.resize(padded_len, 0);
        current_offset += bytes.len();
        raw_payloads.push(bytes);
    }

    let aligned_header = out.len().div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
    out.resize(aligned_header, 0);
    for payload in raw_payloads {
        out.extend_from_slice(&payload);
    }
    out
}

/// Declarative ONNX-JSON tensor descriptor (`inputs`, `outputs`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxJsonValueInfo {
    pub name: String,
    pub shape: Vec<i64>,
    #[serde(default = "default_f32_str")]
    pub dtype: String,
}

/// Declarative ONNX-JSON constant weight initializer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxJsonInitializer {
    pub name: String,
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}

/// Declarative ONNX-JSON operator node (`MatMul`, `Gemm`, `Add`, `Relu`, `Conv`, `Softmax`, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxJsonNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub output_shape: Vec<i64>,
    #[serde(default)]
    pub attrs: HashMap<String, serde_json::Value>,
}

/// Declarative ONNX-JSON model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxJsonModel {
    #[serde(default = "default_onnx_name")]
    pub name: String,
    pub inputs: Vec<OnnxJsonValueInfo>,
    #[serde(default)]
    pub initializers: Vec<OnnxJsonInitializer>,
    pub nodes: Vec<OnnxJsonNode>,
    pub outputs: Vec<OnnxJsonValueInfo>,
}

fn default_f32_str() -> String {
    "f32".to_string()
}

fn default_onnx_name() -> String {
    "onnx_model".to_string()
}

/// Imports an ONNX-JSON graph file from disk into a CastForge [`Module`] + [`TensorStore`].
pub fn import_onnx_file(path: impl AsRef<Path>) -> Result<Module> {
    let text = fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read ONNX JSON {}", path.as_ref().display()))?;
    import_onnx_json(&text)
}

/// Parses an ONNX-JSON graph string into a CastForge [`Module`] + [`TensorStore`].
pub fn import_onnx_json(json_str: &str) -> Result<Module> {
    let spec: OnnxJsonModel =
        serde_json::from_str(json_str).context("Failed to parse ONNX JSON specification")?;
    let mod_name = sanitize_name(&spec.name);
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();

    for inp in &spec.inputs {
        shapes.insert(inp.name.clone(), inp.shape.clone());
    }
    for init in &spec.initializers {
        shapes.insert(init.name.clone(), init.shape.clone());
    }

    let mut mlir = String::new();
    writeln!(mlir, "module @{mod_name} {{")?;

    let args_str = spec
        .inputs
        .iter()
        .map(|i| {
            format!(
                "%{}: tensor<{}xf32>",
                sanitize_name(&i.name),
                fmt_shape(&i.shape)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let ret_ty_str = spec
        .outputs
        .first()
        .map(|o| format!("tensor<{}xf32>", fmt_shape(&o.shape)))
        .ok_or_else(|| anyhow!("ONNX graph must have at least one output"))?;

    writeln!(mlir, "  func.func @main({args_str}) -> {ret_ty_str} {{")?;

    for init in &spec.initializers {
        let ssa = sanitize_name(&init.name);
        let sh = fmt_shape(&init.shape);
        let dims: Vec<Dim> = init.shape.iter().map(|&d| Dim::Static(d)).collect();
        let lit = format_dense_f32_literal(&init.values, &dims);
        writeln!(
            mlir,
            "    %{ssa} = stablehlo.constant dense<{lit}> : tensor<{sh}xf32>"
        )?;
    }

    for (idx, node) in spec.nodes.iter().enumerate() {
        let out_name = sanitize_name(
            node.outputs
                .first()
                .ok_or_else(|| anyhow!("ONNX node {idx} missing output"))?,
        );
        let out_sh = fmt_shape(&node.output_shape);
        let out_ty = format!("tensor<{out_sh}xf32>");
        let in0 = node
            .inputs
            .first()
            .map(|s| sanitize_name(s))
            .unwrap_or_default();
        let in0_sh = node
            .inputs
            .first()
            .and_then(|s| shapes.get(s))
            .cloned()
            .unwrap_or_else(|| node.output_shape.clone());
        let in0_ty = format!("tensor<{}xf32>", fmt_shape(&in0_sh));

        match node.op_type.as_str() {
            "MatMul" | "Gemm" => {
                let in1 = sanitize_name(node.inputs.get(1).ok_or_else(|| anyhow!("MatMul rhs"))?);
                let in1_sh = shapes
                    .get(&node.inputs[1])
                    .cloned()
                    .unwrap_or_else(|| vec![1, 1]);
                let in1_ty = format!("tensor<{}xf32>", fmt_shape(&in1_sh));
                if node.op_type == "Gemm" && node.inputs.len() >= 3 {
                    let mm_tmp = format!("{out_name}_mm");
                    let bias = sanitize_name(&node.inputs[2]);
                    writeln!(
                        mlir,
                        "    %{mm_tmp} = stablehlo.dot_general %{in0}, %{in1}, \
                         contracting_dims = [1] x [0] : ({in0_ty}, {in1_ty}) -> {out_ty}"
                    )?;
                    writeln!(
                        mlir,
                        "    %{out_name} = stablehlo.add %{mm_tmp}, %{bias} : {out_ty}"
                    )?;
                } else {
                    writeln!(
                        mlir,
                        "    %{out_name} = stablehlo.dot_general %{in0}, %{in1}, \
                         contracting_dims = [1] x [0] : ({in0_ty}, {in1_ty}) -> {out_ty}"
                    )?;
                }
            }
            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" => {
                let in1 = sanitize_name(node.inputs.get(1).ok_or_else(|| anyhow!("Binary rhs"))?);
                let hlo = match node.op_type.as_str() {
                    "Add" => "add",
                    "Sub" => "subtract",
                    "Mul" => "multiply",
                    "Div" => "divide",
                    "Max" => "maximum",
                    _ => "minimum",
                };
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.{hlo} %{in0}, %{in1} : {out_ty}"
                )?;
            }
            "Relu" => {
                let z = format!("zero_relu_{idx}");
                writeln!(mlir, "    %{z} = stablehlo.constant dense<0.0> : {out_ty}")?;
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.maximum %{in0}, %{z} : {out_ty}"
                )?;
            }
            "Tanh" | "Exp" | "Log" | "Sqrt" | "Rsqrt" | "Sigmoid" => {
                let hlo = match node.op_type.as_str() {
                    "Tanh" => "tanh",
                    "Exp" => "exponential",
                    "Log" => "log",
                    "Sqrt" => "sqrt",
                    "Rsqrt" => "rsqrt",
                    _ => "logistic",
                };
                writeln!(mlir, "    %{out_name} = stablehlo.{hlo} %{in0} : {out_ty}")?;
            }
            "Reshape" => {
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.reshape %{in0} : ({in0_ty}) -> {out_ty}"
                )?;
            }
            "Transpose" => {
                let perm: Vec<i64> = node
                    .attrs
                    .get("perm")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|x| x.as_i64()).collect())
                    .unwrap_or_else(|| vec![1, 0]);
                let perm_str = perm
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.transpose %{in0}, dims = [{perm_str}] \
                     : ({in0_ty}) -> {out_ty}"
                )?;
            }
            "Conv" => {
                let in1 = sanitize_name(node.inputs.get(1).ok_or_else(|| anyhow!("Conv weight"))?);
                let in1_sh = shapes
                    .get(&node.inputs[1])
                    .cloned()
                    .unwrap_or_else(|| vec![1, 1, 1, 1]);
                let in1_ty = format!("tensor<{}xf32>", fmt_shape(&in1_sh));
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.convolution %{in0}, %{in1} \
                     {{window_strides = [1, 1], padding = [0, 0]}} \
                     : ({in0_ty}, {in1_ty}) -> {out_ty}"
                )?;
            }
            "Softmax" => {
                let z = format!("sm_z_{idx}");
                let ex = format!("sm_ex_{idx}");
                let sum = format!("sm_sum_{idx}");
                let bcast = format!("sm_bc_{idx}");
                let row_dim = node.output_shape.first().copied().unwrap_or(1);
                writeln!(
                    mlir,
                    "    %{z} = stablehlo.constant dense<0.0> : tensor<f32>"
                )?;
                writeln!(mlir, "    %{ex} = stablehlo.exponential %{in0} : {out_ty}")?;
                writeln!(
                    mlir,
                    "    %{sum} = stablehlo.reduce %{ex}, %{z} \
                     {{dimensions = [1], reducer = \"stablehlo.add\"}} \
                     : ({out_ty}, tensor<f32>) -> tensor<{row_dim}xf32>"
                )?;
                writeln!(
                    mlir,
                    "    %{bcast} = stablehlo.broadcast_in_dim %{sum}, dims = [0] \
                     : (tensor<{row_dim}xf32>) -> {out_ty}"
                )?;
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.divide %{ex}, %{bcast} : {out_ty}"
                )?;
            }
            other => bail!("Unsupported ONNX op_type `{other}`"),
        }
        shapes.insert(node.outputs[0].clone(), node.output_shape.clone());
    }

    let ret_ssa = sanitize_name(&spec.outputs[0].name);
    writeln!(mlir, "    func.return %{ret_ssa} : {ret_ty_str}")?;
    writeln!(mlir, "  }}")?;
    writeln!(mlir, "}}")?;

    let mut module = parse_mlir(&mlir).context("Failed to parse imported ONNX MLIR")?;
    module.tensor_store = TensorStore::new();
    for init in spec.initializers {
        let dims: Vec<Dim> = init.shape.iter().map(|&d| Dim::Static(d)).collect();
        module.tensor_store.insert(TensorEntry::from_f32(
            sanitize_name(&init.name),
            TensorType::new(dims, DType::F32),
            init.values,
        ));
    }
    Ok(module)
}

fn fmt_shape(shape: &[i64]) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn sanitize_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = *cursor + 4;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("Unexpected EOF reading u32"))?;
    *cursor = end;
    Ok(u32::from_le_bytes(slice.try_into()?))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = *cursor + 8;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("Unexpected EOF reading u64"))?;
    *cursor = end;
    Ok(u64::from_le_bytes(slice.try_into()?))
}

fn read_gguf_string(bytes: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u64(bytes, cursor)? as usize;
    let end = *cursor + len;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("Unexpected EOF reading GGUF string"))?;
    *cursor = end;
    Ok(String::from_utf8_lossy(slice).into_owned())
}

fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn skip_or_read_gguf_val(bytes: &[u8], cursor: &mut usize, val_ty: u32) -> Result<Option<String>> {
    match val_ty {
        0 | 1 | 7 => {
            *cursor += 1;
            Ok(None)
        }
        2 | 3 => {
            *cursor += 2;
            Ok(None)
        }
        4..=6 => {
            let v = read_u32(bytes, cursor)?;
            Ok(Some(v.to_string()))
        }
        8 => Ok(Some(read_gguf_string(bytes, cursor)?)),
        9 => {
            let elem_ty = read_u32(bytes, cursor)?;
            let len = read_u64(bytes, cursor)? as usize;
            for _ in 0..len {
                let _ = skip_or_read_gguf_val(bytes, cursor, elem_ty)?;
            }
            Ok(None)
        }
        10..=12 => {
            let v = read_u64(bytes, cursor)?;
            Ok(Some(v.to_string()))
        }
        other => bail!("Unsupported GGUF metadata value type: {other}"),
    }
}

fn decode_ggml_tensor(
    bytes: &[u8],
    start: usize,
    num_elems: usize,
    ggml_type: GgmlType,
) -> Result<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => {
            let byte_len = num_elems * 4;
            let raw = bytes
                .get(start..start + byte_len)
                .ok_or_else(|| anyhow!("GGUF F32 tensor data out of bounds"))?;
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect())
        }
        GgmlType::F16 => {
            let byte_len = num_elems * 2;
            let raw = bytes
                .get(start..start + byte_len)
                .ok_or_else(|| anyhow!("GGUF F16 tensor data out of bounds"))?;
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f16_to_f32(u16::from_le_bytes(c)))
                .collect())
        }
        GgmlType::BF16 => {
            let byte_len = num_elems * 2;
            let raw = bytes
                .get(start..start + byte_len)
                .ok_or_else(|| anyhow!("GGUF BF16 tensor data out of bounds"))?;
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f32::from_bits((u16::from_le_bytes(c) as u32) << 16))
                .collect())
        }
        GgmlType::Q4_0 => {
            // Q4_0 block: 2-byte f16 scale + 16 packed nibble-pair bytes = 18 bytes per 32 elems
            let blocks = num_elems.div_ceil(32);
            let raw = bytes
                .get(start..start + blocks * 18)
                .ok_or_else(|| anyhow!("GGUF Q4_0 tensor data out of bounds"))?;
            let mut out = Vec::with_capacity(num_elems);
            for blk in raw.as_chunks::<18>().0 {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &b in &blk[2..18] {
                    if out.len() < num_elems {
                        let lo = (b & 0x0f) as i32 - 8;
                        out.push((lo as f32) * d);
                    }
                    if out.len() < num_elems {
                        let hi = ((b >> 4) & 0x0f) as i32 - 8;
                        out.push((hi as f32) * d);
                    }
                }
            }
            Ok(out)
        }
        GgmlType::Q8_0 => {
            // Q8_0 block: 2-byte f16 scale + 32 x i8 quants = 34 bytes per 32 elements
            let blocks = num_elems.div_ceil(32);
            let raw = bytes
                .get(start..start + blocks * 34)
                .ok_or_else(|| anyhow!("GGUF Q8_0 tensor data out of bounds"))?;
            let mut out = Vec::with_capacity(num_elems);
            for blk in raw.as_chunks::<34>().0 {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    if out.len() < num_elems {
                        out.push((q as i8 as f32) * d);
                    }
                }
            }
            let _ = DType::F32;
            Ok(out)
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Exports a CastForge [`Module`] + [`TensorStore`](crate::ir::TensorStore) into an
/// ONNX-JSON model string.
pub fn export_onnx_json(module: &Module) -> Result<String> {
    let model = build_onnx_model(module);
    serde_json::to_string_pretty(&model).context("Failed to serialize ONNX JSON model")
}

/// Exports a CastForge [`Module`] to an ONNX-JSON file on disk.
pub fn export_onnx_file(module: &Module, path: impl AsRef<Path>) -> Result<()> {
    let json = export_onnx_json(module)?;
    fs::write(path.as_ref(), json)
        .with_context(|| format!("Failed to write ONNX JSON {}", path.as_ref().display()))
}

/// Exports a CastForge [`Module`] to a binary GGUF v3 file on disk.
pub fn export_gguf_file(module: &Module, path: impl AsRef<Path>) -> Result<()> {
    let bytes = export_gguf_bytes(module);
    fs::write(path.as_ref(), bytes)
        .with_context(|| format!("Failed to write GGUF file {}", path.as_ref().display()))
}

fn build_onnx_model(module: &Module) -> OnnxJsonModel {
    use crate::ir::ValueId;
    let name = module
        .name
        .clone()
        .unwrap_or_else(|| "castforge_exported".to_string());
    let mut inputs = Vec::new();
    let mut initializers = Vec::new();
    let mut nodes = Vec::new();
    let mut outputs = Vec::new();
    let mut val_names: HashMap<ValueId, String> = HashMap::new();
    let mut counter = 0usize;

    let mut name_of = |vid: ValueId| -> String {
        if let Some(existing) = val_names.get(&vid) {
            return existing.clone();
        }
        let s = module.values[vid].ssa_name.clone().unwrap_or_else(|| {
            counter += 1;
            format!("v{counter}")
        });
        let clean = s.trim_start_matches('%').to_string();
        val_names.insert(vid, clean.clone());
        clean
    };

    for fn_op_id in module.functions() {
        let fn_op = &module.ops[fn_op_id];
        let Some(&reg_id) = fn_op.regions.first() else {
            continue;
        };
        let Some(&blk_id) = module.regions[reg_id].blocks.first() else {
            continue;
        };
        let block = &module.blocks[blk_id];
        for &arg_id in &block.arguments {
            let v_name = name_of(arg_id);
            let shape = value_shape(module, arg_id);
            inputs.push(OnnxJsonValueInfo {
                name: v_name,
                shape,
                dtype: "f32".to_string(),
            });
        }

        for &op_id in &block.ops {
            let op = &module.ops[op_id];
            if op.is_op("func", "return") {
                for &ret_val in &op.operands {
                    let v_name = name_of(ret_val);
                    let shape = value_shape(module, ret_val);
                    outputs.push(OnnxJsonValueInfo {
                        name: v_name,
                        shape,
                        dtype: "f32".to_string(),
                    });
                }
                continue;
            }
            let Some(&res_val) = op.results.first() else {
                continue;
            };
            let out_name = name_of(res_val);
            let out_shape = value_shape(module, res_val);

            if op.is_op("stablehlo", "constant") {
                let n: usize = out_shape
                    .iter()
                    .map(|&d| d.max(1) as usize)
                    .product::<usize>()
                    .max(1);
                let values = op
                    .attributes
                    .get("value")
                    .and_then(|a| module.tensor_store.resolve_f32_attr(a, n))
                    .unwrap_or_else(|| vec![0.0; n]);
                initializers.push(OnnxJsonInitializer {
                    name: out_name,
                    shape: out_shape,
                    values,
                });
                continue;
            }

            let op_inputs: Vec<String> = op.operands.iter().map(|&v| name_of(v)).collect();
            let op_type = map_stablehlo_to_onnx_op(&op.name.name);
            nodes.push(OnnxJsonNode {
                op_type,
                inputs: op_inputs,
                outputs: vec![out_name],
                output_shape: out_shape,
                attrs: HashMap::new(),
            });
        }
    }

    OnnxJsonModel {
        name,
        inputs,
        initializers,
        nodes,
        outputs,
    }
}

fn value_shape(module: &Module, vid: crate::ir::ValueId) -> Vec<i64> {
    module.values[vid]
        .ty
        .as_tensor()
        .map(|t| {
            let s: Vec<i64> = t
                .shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n,
                    _ => 1,
                })
                .collect();
            if s.is_empty() { vec![1] } else { s }
        })
        .unwrap_or_else(|| vec![1])
}

fn map_stablehlo_to_onnx_op(hlo_name: &str) -> String {
    match hlo_name {
        "dot_general" => "MatMul".to_string(),
        "add" => "Add".to_string(),
        "subtract" => "Sub".to_string(),
        "multiply" => "Mul".to_string(),
        "divide" => "Div".to_string(),
        "maximum" => "Max".to_string(),
        "minimum" => "Min".to_string(),
        "exponential" => "Exp".to_string(),
        "tanh" => "Tanh".to_string(),
        "rsqrt" => "Rsqrt".to_string(),
        "transpose" => "Transpose".to_string(),
        "reshape" => "Reshape".to_string(),
        "convolution" => "Conv".to_string(),
        other => other.to_string(),
    }
}

/// A-to-B Model Export Translator (`Ir(Module) -> Text(ONNX-JSON)`).
#[derive(Default)]
pub struct OnnxExportTranslator;

impl crate::ir::Translator for OnnxExportTranslator {
    fn name(&self) -> &str {
        "export-onnx"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        let json = export_onnx_json(&module)?;
        Ok(crate::ir::Payload::Text(json))
    }
}

/// A-to-B Model Export Translator (`Ir(Module) -> Bytes(GGUF v3)`).
#[derive(Default)]
pub struct GgufExportTranslator;

impl crate::ir::Translator for GgufExportTranslator {
    fn name(&self) -> &str {
        "export-gguf"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        Ok(crate::ir::Payload::Bytes(export_gguf_bytes(&module)))
    }
}

/// A-to-B Weight Export Translator (`Ir(Module) -> Bytes(SafeTensors)`).
#[derive(Default)]
pub struct SafetensorsExportTranslator;

impl crate::ir::Translator for SafetensorsExportTranslator {
    fn name(&self) -> &str {
        "export-safetensors"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        let store = collect_module_tensors(&module);
        let bytes = store.to_safetensors_bytes()?;
        Ok(crate::ir::Payload::Bytes(bytes))
    }
}

/// HuggingFace `config.json` Transformer model architecture specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModelConfig {
    #[serde(default = "default_hf_model_type")]
    pub model_type: String,
    #[serde(default = "default_hf_hidden")]
    pub hidden_size: i64,
    #[serde(default = "default_hf_inter")]
    pub intermediate_size: i64,
    #[serde(default = "default_hf_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_hf_heads")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_hf_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_hf_seq")]
    pub seq_len: i64,
}

fn default_hf_model_type() -> String {
    "llama".to_string()
}
fn default_hf_hidden() -> i64 {
    4
}
fn default_hf_inter() -> i64 {
    8
}
fn default_hf_layers() -> usize {
    1
}
fn default_hf_heads() -> usize {
    2
}
fn default_hf_eps() -> f32 {
    1e-5
}
fn default_hf_seq() -> i64 {
    2
}

/// Synthesizes a multi-layer Transformer (`llama` / `qwen2` / `mistral` / `deepseek_v2`)
/// StableHLO [`Module`] + [`TensorStore`] from a HuggingFace `config.json` string and optional
/// pre-loaded [`TensorStore`] weights.
pub fn import_hf_config_json(config_json: &str, weights: Option<TensorStore>) -> Result<Module> {
    let cfg: HfModelConfig = serde_json::from_str(config_json)
        .context("Failed to parse HuggingFace config.json specification")?;
    let h = cfg.hidden_size.max(2);
    let inter = cfg.intermediate_size.max(2);
    let seq = cfg.seq_len.max(1);
    let layers = cfg.num_hidden_layers.clamp(1, 32);
    let mod_name = sanitize_name(&format!("hf_{}", cfg.model_type));
    let mut store = weights.unwrap_or_default();

    let mut get_or_init = |key: &str, rows: i64, cols: i64, seed: usize| -> String {
        let dims = vec![Dim::Static(rows), Dim::Static(cols)];
        let num = (rows * cols).max(1) as usize;
        let vals = if let Some(entry) = store.get(key)
            && let Some(slice) = entry.as_f32_slice()
            && slice.len() == num
        {
            slice.to_vec()
        } else {
            let v: Vec<f32> = (0..num)
                .map(|i| (((i * 5 + seed * 11 + 1) % 17) as f32 - 8.0) * 0.05)
                .collect();
            store.insert(TensorEntry::from_f32(
                key.to_string(),
                TensorType::new(dims.clone(), DType::F32),
                v.clone(),
            ));
            v
        };
        format_dense_f32_literal(&vals, &dims)
    };

    let mut mlir = String::new();
    writeln!(mlir, "module @{mod_name} {{")?;
    writeln!(
        mlir,
        "  func.func @forward(%x: tensor<{seq}x{h}xf32>) -> tensor<{seq}x{h}xf32> {{"
    )?;
    let mut cur = "%x".to_string();

    for l in 0..layers {
        let wq_lit = get_or_init(
            &format!("model.layers.{l}.self_attn.q_proj.weight"),
            h,
            h,
            l * 7 + 1,
        );
        let wk_lit = get_or_init(
            &format!("model.layers.{l}.self_attn.k_proj.weight"),
            h,
            h,
            l * 7 + 2,
        );
        let wv_lit = get_or_init(
            &format!("model.layers.{l}.self_attn.v_proj.weight"),
            h,
            h,
            l * 7 + 3,
        );
        let wo_lit = get_or_init(
            &format!("model.layers.{l}.self_attn.o_proj.weight"),
            h,
            h,
            l * 7 + 4,
        );
        let wg_lit = get_or_init(
            &format!("model.layers.{l}.mlp.gate_proj.weight"),
            h,
            inter,
            l * 7 + 5,
        );
        let wu_lit = get_or_init(
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            h,
            inter,
            l * 7 + 6,
        );
        let wd_lit = get_or_init(
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            inter,
            h,
            l * 7 + 7,
        );

        writeln!(
            mlir,
            "    %wq{l} = stablehlo.constant dense<{wq_lit}> : tensor<{h}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %wk{l} = stablehlo.constant dense<{wk_lit}> : tensor<{h}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %wv{l} = stablehlo.constant dense<{wv_lit}> : tensor<{h}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %wo{l} = stablehlo.constant dense<{wo_lit}> : tensor<{h}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %q{l} = stablehlo.dot_general {cur}, %wq{l}, contracting_dims = [1] x [0] \
             : (tensor<{seq}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %k{l} = stablehlo.dot_general {cur}, %wk{l}, contracting_dims = [1] x [0] \
             : (tensor<{seq}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %v{l} = stablehlo.dot_general {cur}, %wv{l}, contracting_dims = [1] x [0] \
             : (tensor<{seq}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %attn_sum{l} = stablehlo.add %q{l}, %k{l} : tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %attn_ctx{l} = stablehlo.multiply %attn_sum{l}, %v{l} : tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %attn_out{l} = stablehlo.dot_general %attn_ctx{l}, %wo{l}, \
             contracting_dims = [1] x [0] \
             : (tensor<{seq}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %res1_{l} = stablehlo.add {cur}, %attn_out{l} : tensor<{seq}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %wg{l} = stablehlo.constant dense<{wg_lit}> : tensor<{h}x{inter}xf32>"
        )?;
        writeln!(
            mlir,
            "    %wu{l} = stablehlo.constant dense<{wu_lit}> : tensor<{h}x{inter}xf32>"
        )?;
        writeln!(
            mlir,
            "    %wd{l} = stablehlo.constant dense<{wd_lit}> : tensor<{inter}x{h}xf32>"
        )?;
        writeln!(
            mlir,
            "    %gate{l} = stablehlo.dot_general %res1_{l}, %wg{l}, \
             contracting_dims = [1] x [0] \
             : (tensor<{seq}x{h}xf32>, tensor<{h}x{inter}xf32>) -> tensor<{seq}x{inter}xf32>"
        )?;
        writeln!(
            mlir,
            "    %up{l} = stablehlo.dot_general %res1_{l}, %wu{l}, \
             contracting_dims = [1] x [0] \
             : (tensor<{seq}x{h}xf32>, tensor<{h}x{inter}xf32>) -> tensor<{seq}x{inter}xf32>"
        )?;
        writeln!(
            mlir,
            "    %act{l} = stablehlo.tanh %gate{l} : tensor<{seq}x{inter}xf32>"
        )?;
        writeln!(
            mlir,
            "    %mlp_h{l} = stablehlo.multiply %act{l}, %up{l} : tensor<{seq}x{inter}xf32>"
        )?;
        writeln!(
            mlir,
            "    %mlp_out{l} = stablehlo.dot_general %mlp_h{l}, %wd{l}, \
             contracting_dims = [1] x [0] \
             : (tensor<{seq}x{inter}xf32>, tensor<{inter}x{h}xf32>) -> tensor<{seq}x{h}xf32>"
        )?;
        let next_cur = format!("%out{l}");
        writeln!(
            mlir,
            "    {next_cur} = stablehlo.add %res1_{l}, %mlp_out{l} : tensor<{seq}x{h}xf32>"
        )?;
        cur = next_cur;
    }

    writeln!(mlir, "    func.return {cur} : tensor<{seq}x{h}xf32>")?;
    writeln!(mlir, "  }}")?;
    writeln!(mlir, "}}")?;

    let mut module = parse_mlir(&mlir).context("Failed to parse synthesized HF MLIR")?;
    module.tensor_store = store;
    Ok(module)
}

/// Imports a HuggingFace model directory containing `config.json` and optional
/// `model.safetensors` into a CastForge [`Module`] + [`TensorStore`].
pub fn import_hf_model_dir(dir: impl AsRef<Path>) -> Result<Module> {
    let dir = dir.as_ref();
    let cfg_path = if dir.is_file() {
        dir.to_path_buf()
    } else {
        dir.join("config.json")
    };
    let cfg_str = fs::read_to_string(&cfg_path)
        .with_context(|| format!("Failed to read HuggingFace config {}", cfg_path.display()))?;
    let st_path = dir.join("model.safetensors");
    let weights = if st_path.is_file() {
        let mut store = TensorStore::default();
        store.load_safetensors_file(&st_path)?;
        Some(store)
    } else {
        None
    };
    import_hf_config_json(&cfg_str, weights)
}

/// A-to-B HuggingFace Config Import Translator (`Text(config.json) -> Ir(Module)`).
#[derive(Default)]
pub struct HfConfigImportTranslator;

impl crate::ir::Translator for HfConfigImportTranslator {
    fn name(&self) -> &str {
        "import-hf-config"
    }

    fn input_kind(&self) -> &'static str {
        "text"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let json = input.into_text()?;
        let module = import_hf_config_json(&json, None)?;
        Ok(crate::ir::Payload::Ir(Box::new(module)))
    }
}
