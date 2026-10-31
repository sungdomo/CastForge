//! Model format importers and exporters (`GGUF` v2/v3 binary, `ONNX` JSON graphs,
//! `SafeTensors`, and HuggingFace `config.json` Transformer definitions) for CastForge's
//! Dual-Plane StableHLO IR (`Module` + `TensorStore`).

pub mod gguf;
pub mod hf_config;
pub mod onnx;
pub mod safetensors;

pub use gguf::{
    GgmlType, GgufExportTranslator, GgufImportTranslator, export_gguf_bytes, export_gguf_file,
    import_gguf_bytes, import_gguf_file,
};
pub use hf_config::{
    HfConfigImportTranslator, HfModelConfig, import_hf_config_json, import_hf_model_dir,
};
pub use onnx::{
    OnnxExportTranslator, OnnxImportTranslator, OnnxJsonInitializer, OnnxJsonModel, OnnxJsonNode,
    OnnxJsonValueInfo, export_onnx_file, export_onnx_json, import_onnx_file, import_onnx_json,
};
pub use safetensors::{
    SafetensorsExportTranslator, SafetensorsImportTranslator, collect_module_tensors,
    import_safetensors_bytes, import_safetensors_file,
};
pub(crate) use safetensors::{fmt_shape, sanitize_name};
