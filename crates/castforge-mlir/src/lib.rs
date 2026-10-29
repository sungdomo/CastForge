//! Pure-Rust MLIR parser & printer and symmetric ONNX-JSON / GGUF / SafeTensors model translators
//! (`castforge-mlir`).

pub mod export;
pub mod import;
pub mod parser;
pub mod parser_special;
pub mod parser_types;
pub mod printer;

pub use export::{
    GgufExportTranslator, OnnxExportTranslator, SafetensorsExportTranslator, export_gguf_file,
    export_onnx_file, export_onnx_json,
};
pub use import::{
    GgmlType, OnnxJsonInitializer, OnnxJsonModel, OnnxJsonNode, OnnxJsonValueInfo,
    collect_module_tensors, export_gguf_bytes, import_gguf_bytes, import_gguf_file,
    import_onnx_file, import_onnx_json,
};
pub use parser::{MlirParser, ParseError};
pub use parser_types::parse_type;
pub use printer::{MlirPrinter, PrintOptions};

use castforge_ir::Module;

/// Parse an MLIR text string into a CastForge `Module`.
pub fn parse_mlir(input: &str) -> Result<Module, ParseError> {
    let mut parser = MlirParser::new();
    parser.parse_module(input)
}

/// Format a CastForge `Module` into MLIR text with default pretty syntax.
pub fn print_mlir(module: &Module) -> String {
    let mut printer = MlirPrinter::new(module, PrintOptions::default());
    printer.print_module().unwrap_or_default()
}

/// Format a CastForge `Module` into MLIR text with custom `PrintOptions`.
pub fn print_mlir_with_options(module: &Module, opts: PrintOptions) -> String {
    let mut printer = MlirPrinter::new(module, opts);
    printer.print_module().unwrap_or_default()
}

/// A-to-B Translator (`Text(MLIR) -> Ir(Module)`).
#[derive(Default)]
pub struct MlirParseTranslator;

impl castforge_ir::Translator for MlirParseTranslator {
    fn name(&self) -> &str {
        "parse-mlir"
    }

    fn input_kind(&self) -> &'static str {
        "text"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> anyhow::Result<castforge_ir::Payload> {
        let src = input.into_text()?;
        let module = parse_mlir(&src)?;
        Ok(castforge_ir::Payload::from_ir(module))
    }
}

/// A-to-B Translator (`Ir(Module) -> Text(MLIR)`).
#[derive(Default)]
pub struct MlirPrintTranslator {
    pub options: PrintOptions,
}

impl castforge_ir::Translator for MlirPrintTranslator {
    fn name(&self) -> &str {
        "print-mlir"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "text"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> anyhow::Result<castforge_ir::Payload> {
        let module = input.into_ir()?;
        Ok(castforge_ir::Payload::Text(print_mlir_with_options(
            &module,
            self.options,
        )))
    }
}

/// A-to-B Translator (`Text(ONNX-JSON) -> Ir(Module)`).
#[derive(Default)]
pub struct OnnxImportTranslator;

impl castforge_ir::Translator for OnnxImportTranslator {
    fn name(&self) -> &str {
        "import-onnx"
    }

    fn input_kind(&self) -> &'static str {
        "text"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> anyhow::Result<castforge_ir::Payload> {
        let src = input.into_text()?;
        let module = import_onnx_json(&src)?;
        Ok(castforge_ir::Payload::from_ir(module))
    }
}

/// A-to-B Translator (`Bytes(GGUF) -> Ir(Module)`).
#[derive(Default)]
pub struct GgufImportTranslator;

impl castforge_ir::Translator for GgufImportTranslator {
    fn name(&self) -> &str {
        "import-gguf"
    }

    fn input_kind(&self) -> &'static str {
        "bytes"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: castforge_ir::Payload) -> anyhow::Result<castforge_ir::Payload> {
        let bytes = input.into_bytes()?;
        let module = import_gguf_bytes(&bytes)?;
        Ok(castforge_ir::Payload::from_ir(module))
    }
}
