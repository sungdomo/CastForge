//! Core MLIR-compatible Dual-Plane Intermediate Representation (`castforge::ir`).
//!
//! Provides arena-allocated SSA IR data structures ([`Module`], [`Operation`], [`Value`],
//! [`Block`], [`Region`], [`Type`], [`Attribute`], [`Location`]), the [`TensorStore`] weight
//! plane, SafeTensors binary I/O, strongly-typed [`StableHloOp`] dialect views, the MLIR
//! parser/printer, ONNX/GGUF importers/exporters, and the composable [`Translator`] /
//! [`TranslatorChain`] A-to-B pipeline abstraction.

pub mod formats;
pub mod loops;
pub mod mlir;
pub mod parser;
pub mod printer;
pub mod stablehlo;
pub mod tensor_store;
pub mod translator;
pub mod viz;

pub use crate::passes::normalize::verify::*;
pub use crate::passes::registry::manager::*;
pub use formats::{
    GgmlType, GgufExportTranslator, GgufImportTranslator, HfConfigImportTranslator, HfModelConfig,
    OnnxExportTranslator, OnnxImportTranslator, OnnxJsonInitializer, OnnxJsonModel, OnnxJsonNode,
    OnnxJsonValueInfo, SafetensorsExportTranslator, SafetensorsImportTranslator,
    collect_module_tensors, export_gguf_bytes, export_gguf_file, export_onnx_file,
    export_onnx_json, import_gguf_bytes, import_gguf_file, import_hf_config_json,
    import_hf_model_dir, import_onnx_file, import_onnx_json, import_safetensors_bytes,
    import_safetensors_file,
};
pub use loops::{LoopKernel, LoopModule, LoopStmt};
pub use mlir::*;
pub use parser::{MlirParseTranslator, MlirParser, ParseError, parse_mlir, parse_type};
pub use printer::{
    MlirPrintTranslator, MlirPrinter, PrintOptions, print_mlir, print_mlir_with_options,
};
pub use stablehlo::{DotDimensionNumbers, FuncView, StableHloOp, verify_stablehlo_shapes};
pub use tensor_store::{
    TensorBacking, TensorEntry, TensorStore, bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits,
    f32_to_f16_bits, format_dense_f32_literal, format_dense_i8_literal, parse_dense_f32_literal,
};
pub use translator::*;
pub use viz::*;
