//! Declarative ONNX-JSON graph importer and exporter.

use super::{fmt_shape, sanitize_name};
use crate::ir::{
    Attribute, DType, Dim, Module, TensorEntry, TensorStore, TensorType, format_dense_f32_literal,
    parse_mlir,
};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

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
                let trans_a = node
                    .attrs
                    .get("transA")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let trans_b = node
                    .attrs
                    .get("transB")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let lhs_c = if trans_a != 0 { 0 } else { 1 };
                let rhs_c = if trans_b != 0 { 1 } else { 0 };
                if node.op_type == "Gemm" && node.inputs.len() >= 3 {
                    let mm_tmp = format!("{out_name}_mm");
                    let bias = sanitize_name(&node.inputs[2]);
                    writeln!(
                        mlir,
                        "    %{mm_tmp} = stablehlo.dot_general %{in0}, %{in1}, \
                         contracting_dims = [{lhs_c}] x [{rhs_c}] \
                         : ({in0_ty}, {in1_ty}) -> {out_ty}"
                    )?;
                    writeln!(
                        mlir,
                        "    %{out_name} = stablehlo.add %{mm_tmp}, %{bias} : {out_ty}"
                    )?;
                } else {
                    writeln!(
                        mlir,
                        "    %{out_name} = stablehlo.dot_general %{in0}, %{in1}, \
                         contracting_dims = [{lhs_c}] x [{rhs_c}] \
                         : ({in0_ty}, {in1_ty}) -> {out_ty}"
                    )?;
                }
            }
            "Add" | "Sub" | "Mul" | "Div" | "Max" | "Min" | "Pow" => {
                let in1 = sanitize_name(node.inputs.get(1).ok_or_else(|| anyhow!("Binary rhs"))?);
                let hlo = match node.op_type.as_str() {
                    "Add" => "add",
                    "Sub" => "subtract",
                    "Mul" => "multiply",
                    "Div" => "divide",
                    "Max" => "maximum",
                    "Pow" => "power",
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
            "Tanh" | "Exp" | "Log" | "Sqrt" | "Rsqrt" | "Sigmoid" | "Sin" | "Cos" | "Abs"
            | "Neg" => {
                let hlo = match node.op_type.as_str() {
                    "Tanh" => "tanh",
                    "Exp" => "exponential",
                    "Log" => "log",
                    "Sqrt" => "sqrt",
                    "Rsqrt" => "rsqrt",
                    "Sin" => "sine",
                    "Cos" => "cosine",
                    "Abs" => "abs",
                    "Neg" => "negate",
                    _ => "logistic",
                };
                writeln!(mlir, "    %{out_name} = stablehlo.{hlo} %{in0} : {out_ty}")?;
            }
            "Clip" => {
                let lo = node
                    .inputs
                    .get(1)
                    .map(|s| sanitize_name(s))
                    .unwrap_or_else(|| in0.clone());
                let hi = node
                    .inputs
                    .get(2)
                    .map(|s| sanitize_name(s))
                    .unwrap_or_else(|| in0.clone());
                writeln!(
                    mlir,
                    "    %{out_name} = \"stablehlo.clamp\"(%{lo}, %{in0}, %{hi}) \
                     : ({out_ty}, {out_ty}, {out_ty}) -> {out_ty}"
                )?;
            }
            "Gather" => {
                let indices = sanitize_name(
                    node.inputs
                        .get(1)
                        .ok_or_else(|| anyhow!("Gather indices"))?,
                );
                let idx_sh = shapes
                    .get(&node.inputs[1])
                    .cloned()
                    .unwrap_or_else(|| vec![1]);
                let idx_ty = format!("tensor<{}xf32>", fmt_shape(&idx_sh));
                let row_w = node.output_shape.get(1).copied().unwrap_or(1);
                writeln!(
                    mlir,
                    "    %{out_name} = \"stablehlo.gather\"(%{in0}, %{indices}) \
                     {{slice_sizes = [1, {row_w}]}} : ({in0_ty}, {idx_ty}) -> {out_ty}"
                )?;
            }
            "Concat" => {
                let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0);
                let ins: Vec<String> = node
                    .inputs
                    .iter()
                    .map(|s| format!("%{}", sanitize_name(s)))
                    .collect();
                let in_tys: Vec<String> = node
                    .inputs
                    .iter()
                    .map(|s| {
                        let sh = shapes
                            .get(s)
                            .cloned()
                            .unwrap_or_else(|| node.output_shape.clone());
                        format!("tensor<{}xf32>", fmt_shape(&sh))
                    })
                    .collect();
                writeln!(
                    mlir,
                    "    %{out_name} = \"stablehlo.concatenate\"({}) \
                     {{dimension = {axis}}} : ({}) -> {out_ty}",
                    ins.join(", "),
                    in_tys.join(", ")
                )?;
            }
            "Where" => {
                let cond = in0.clone();
                let x = sanitize_name(node.inputs.get(1).ok_or_else(|| anyhow!("Where x"))?);
                let y = sanitize_name(node.inputs.get(2).ok_or_else(|| anyhow!("Where y"))?);
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.select %{cond}, %{x}, %{y} \
                     : ({in0_ty}, {out_ty}, {out_ty}) -> {out_ty}"
                )?;
            }
            "ReduceSum" | "ReduceMax" => {
                let z = format!("red_init_{idx}");
                let reducer = if node.op_type == "ReduceMax" {
                    "stablehlo.maximum"
                } else {
                    "stablehlo.add"
                };
                let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(1);
                writeln!(
                    mlir,
                    "    %{z} = stablehlo.constant dense<0.0> : tensor<f32>"
                )?;
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.reduce %{in0}, %{z} \
                     {{dimensions = [{axis}], reducer = \"{reducer}\"}} \
                     : ({in0_ty}, tensor<f32>) -> {out_ty}"
                )?;
            }
            s if s.starts_with("castforge.") => {
                let ins: Vec<String> = node
                    .inputs
                    .iter()
                    .map(|inp| format!("%{}", sanitize_name(inp)))
                    .collect();
                let in_tys: Vec<String> = node
                    .inputs
                    .iter()
                    .map(|inp| {
                        let sh = shapes
                            .get(inp)
                            .cloned()
                            .unwrap_or_else(|| node.output_shape.clone());
                        format!("tensor<{}xf32>", fmt_shape(&sh))
                    })
                    .collect();
                writeln!(
                    mlir,
                    "    %{out_name} = \"{s}\"({}) : ({}) -> {out_ty}",
                    ins.join(", "),
                    in_tys.join(", ")
                )?;
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
                let strides: Vec<String> = node
                    .attrs
                    .get("strides")
                    .or_else(|| node.attrs.get("window_strides"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_i64())
                            .map(|d| d.to_string())
                            .collect()
                    })
                    .unwrap_or_else(|| vec!["1".to_string(), "1".to_string()]);
                let pads: Vec<String> = node
                    .attrs
                    .get("pads")
                    .or_else(|| node.attrs.get("padding"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_i64())
                            .map(|d| d.to_string())
                            .collect()
                    })
                    .unwrap_or_else(|| vec!["0".to_string(), "0".to_string()]);
                writeln!(
                    mlir,
                    "    %{out_name} = stablehlo.convolution %{in0}, %{in1} \
                     {{window_strides = [{}], padding = [{}]}} \
                     : ({in0_ty}, {in1_ty}) -> {out_ty}",
                    strides.join(", "),
                    pads.join(", ")
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
            let dtype = value_dtype(module, arg_id);
            inputs.push(OnnxJsonValueInfo {
                name: v_name,
                shape,
                dtype,
            });
        }

        for &op_id in &block.ops {
            let op = &module.ops[op_id];
            if op.is_op("func", "return") {
                for &ret_val in &op.operands {
                    let v_name = name_of(ret_val);
                    let shape = value_shape(module, ret_val);
                    let dtype = value_dtype(module, ret_val);
                    outputs.push(OnnxJsonValueInfo {
                        name: v_name,
                        shape,
                        dtype,
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
            let op_type = if op.is_op("stablehlo", "composite") {
                match op.attributes.get("name") {
                    Some(Attribute::String(s)) => s.clone(),
                    _ => "castforge.composite".to_string(),
                }
            } else if op.name.dialect == "castforge" {
                op.name.full_name()
            } else {
                map_stablehlo_to_onnx_op(&op.name.name)
            };
            let mut attrs = HashMap::new();
            for (k, v) in &op.attributes {
                match v {
                    Attribute::Int(n) => {
                        attrs.insert(k.clone(), serde_json::json!(n));
                    }
                    Attribute::Float(f) => {
                        attrs.insert(k.clone(), serde_json::json!(f));
                    }
                    Attribute::String(s) => {
                        attrs.insert(k.clone(), serde_json::json!(s));
                    }
                    Attribute::DenseIntArray(arr) => {
                        attrs.insert(k.clone(), serde_json::json!(arr));
                    }
                    _ => {}
                }
            }
            nodes.push(OnnxJsonNode {
                op_type,
                inputs: op_inputs,
                outputs: vec![out_name],
                output_shape: out_shape,
                attrs,
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

fn value_dtype(module: &Module, vid: crate::ir::ValueId) -> String {
    module.values[vid]
        .ty
        .as_tensor()
        .map(|t| t.dtype.to_string())
        .unwrap_or_else(|| "f32".to_string())
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
        "power" => "Pow".to_string(),
        "exponential" => "Exp".to_string(),
        "log" => "Log".to_string(),
        "tanh" => "Tanh".to_string(),
        "sqrt" => "Sqrt".to_string(),
        "rsqrt" => "Rsqrt".to_string(),
        "sine" => "Sin".to_string(),
        "cosine" => "Cos".to_string(),
        "abs" => "Abs".to_string(),
        "negate" => "Neg".to_string(),
        "logistic" => "Sigmoid".to_string(),
        "clamp" => "Clip".to_string(),
        "pad" => "Pad".to_string(),
        "gather" => "Gather".to_string(),
        "scatter" => "ScatterElements".to_string(),
        "concatenate" => "Concat".to_string(),
        "slice" | "dynamic_slice" => "Slice".to_string(),
        "select" => "Where".to_string(),
        "reduce" => "ReduceSum".to_string(),
        "transpose" => "Transpose".to_string(),
        "reshape" => "Reshape".to_string(),
        "convolution" => "Conv".to_string(),
        other => other.to_string(),
    }
}

/// A-to-B Model Import Translator (`Text(ONNX-JSON) -> Ir(Module)`).
#[derive(Default)]
pub struct OnnxImportTranslator;

impl crate::ir::Translator for OnnxImportTranslator {
    fn name(&self) -> &str {
        "import-onnx"
    }

    fn input_kind(&self) -> &'static str {
        "text"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let src = input.into_text()?;
        let module = import_onnx_json(&src)?;
        Ok(crate::ir::Payload::from_ir(module))
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
