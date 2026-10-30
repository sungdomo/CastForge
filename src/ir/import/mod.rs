//! Frontend model importers and exporters (`GGUF` v2/v3 binary, `ONNX` JSON graphs, and
//! HuggingFace `config.json` Transformer definitions) into CastForge's Dual-Plane StableHLO IR
//! (`Module` + `TensorStore`).

pub mod gguf;
pub mod hf_config;
pub mod onnx;

pub use gguf::{
    GgmlType, GgufExportTranslator, export_gguf_bytes, export_gguf_file, import_gguf_bytes,
    import_gguf_file,
};
pub use hf_config::{
    HfConfigImportTranslator, HfModelConfig, import_hf_config_json, import_hf_model_dir,
};
pub use onnx::{
    OnnxExportTranslator, OnnxJsonInitializer, OnnxJsonModel, OnnxJsonNode, OnnxJsonValueInfo,
    export_onnx_file, export_onnx_json, import_onnx_file, import_onnx_json,
};

use crate::ir::{DType, Module, TensorBacking, TensorEntry, TensorStore, TensorType};
use anyhow::Result;

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
        let name = module.values[res_val]
            .ssa_name
            .as_deref()
            .map(|s| s.trim_start_matches('%').to_string())
            .unwrap_or_else(|| {
                idx += 1;
                format!("const_{idx}.weight")
            });
        if let DType::QuantUniform {
            scale, zero_point, ..
        } = &tensor_ty.dtype
        {
            let sc = scale.parse::<f32>().unwrap_or(1.0);
            let zp = *zero_point;
            let q_vals = op
                .attributes
                .get("value")
                .and_then(|a| module.tensor_store.resolve_i8_attr(a, n))
                .unwrap_or_else(|| vec![0; n]);
            store.insert(TensorEntry {
                name,
                ty: tensor_ty,
                backing: TensorBacking::QuantizedI8 {
                    data: q_vals,
                    scale: sc,
                    zero_point: zp,
                },
            });
            continue;
        }
        let vals = op
            .attributes
            .get("value")
            .and_then(|a| module.tensor_store.resolve_f32_attr(a, n))
            .unwrap_or_else(|| vec![0.0; n]);
        let keep_dtype = match tensor_ty.dtype {
            DType::F16 => DType::F16,
            DType::BF16 => DType::BF16,
            _ => DType::F32,
        };
        store.insert(TensorEntry::from_f32(
            name,
            TensorType::new(tensor_ty.shape, keep_dtype),
            vals,
        ));
    }
    store
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

pub(crate) fn fmt_shape(shape: &[i64]) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

pub(crate) fn sanitize_name(s: &str) -> String {
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
