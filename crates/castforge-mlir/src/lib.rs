pub mod parser;
pub mod printer;

pub use parser::{MlirParser, ParseError};
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
