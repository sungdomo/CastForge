//! Symmetric model exporters (`IR -> ONNX-JSON`, `IR -> GGUF v3`, `IR -> SafeTensors`)
//! and composable [`Translator`] implementations.

use crate::import::{
    OnnxJsonInitializer, OnnxJsonModel, OnnxJsonNode, OnnxJsonValueInfo, collect_module_tensors,
    export_gguf_bytes,
};
use anyhow::{Context, Result};
use castforge_ir::{Dim, Module, Payload, Translator, ValueId};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Exports a CastForge [`Module`] + [`TensorStore`](castforge_ir::TensorStore) into an
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

fn value_shape(module: &Module, vid: ValueId) -> Vec<i64> {
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

impl Translator for OnnxExportTranslator {
    fn name(&self) -> &str {
        "export-onnx"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        let json = export_onnx_json(&module)?;
        Ok(Payload::Text(json))
    }
}

/// A-to-B Model Export Translator (`Ir(Module) -> Bytes(GGUF v3)`).
#[derive(Default)]
pub struct GgufExportTranslator;

impl Translator for GgufExportTranslator {
    fn name(&self) -> &str {
        "export-gguf"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        Ok(Payload::Bytes(export_gguf_bytes(&module)))
    }
}

/// A-to-B Weight Export Translator (`Ir(Module) -> Bytes(SafeTensors)`).
#[derive(Default)]
pub struct SafetensorsExportTranslator;

impl Translator for SafetensorsExportTranslator {
    fn name(&self) -> &str {
        "export-safetensors"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: Payload) -> Result<Payload> {
        let module = input.into_ir()?;
        let store = collect_module_tensors(&module);
        let bytes = store.to_safetensors_bytes()?;
        Ok(Payload::Bytes(bytes))
    }
}
