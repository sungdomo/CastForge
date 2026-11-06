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

pub use crate::passes::normalize::verify::{
    BisectPassesPass, HashIrPass, InstrumentProfilingPass, PrintOpStatsPass, RooflineSummary,
    SanitizeNumericsPass, VerifyEquivPass, VerifyIrPass, VerifyShapesPass, analyze_roofline,
    compute_structural_hash,
};
pub use crate::passes::registry::manager::{
    Pass, PassCatalog, PassCatalogEntry, PassDescriptor, PassFactory, PassManager, PassOptionSpec,
    PassOptions, PassRegistry, PassResult, PassStepConfig, PassTelemetry, PassTraceStep,
    PassTranslator, PipelineConfig, PipelineSettings, parse_pass_spec_item, pass_catalog,
    split_pipeline_spec,
};
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
pub use mlir::{
    Attribute, Block, BlockId, DType, Dim, Location, Module, OpId, OpName, Operation, Region,
    RegionId, TensorType, Type, Value, ValueDef, ValueId, VerifyError,
};
pub use parser::{MlirParseTranslator, MlirParser, ParseError, parse_mlir, parse_type};
pub use printer::{
    MlirPrintTranslator, MlirPrinter, PrintOptions, print_mlir, print_mlir_with_options,
};
pub use stablehlo::{DotDimensionNumbers, FuncView, StableHloOp, verify_stablehlo_shapes};
pub use tensor_store::{
    TensorBacking, TensorEntry, TensorStore, bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits,
    f32_to_f16_bits, format_dense_f32_literal, format_dense_i8_literal, parse_dense_f32_literal,
};
pub use translator::{Payload, Translator, TranslatorChain};
pub use viz::{
    GraphEdgeJson, GraphJson, GraphNodeJson, IrDiffTranslator, SnapshotIrTranslator, TimelineFrame,
    VizFormat, VizTranslator, build_graph_json, diff_mlir_strings, diff_modules, render_dot,
    render_json, render_mermaid, render_module, render_timeline_html,
};
