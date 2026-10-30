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
pub mod verify;
pub mod viz;

pub use crate::passes::manager::*;
pub use formats::{
    GgmlType, GgufExportTranslator, GgufImportTranslator, HfConfigImportTranslator, HfModelConfig,
    OnnxExportTranslator, OnnxImportTranslator, OnnxJsonInitializer, OnnxJsonModel, OnnxJsonNode,
    OnnxJsonValueInfo, SafetensorsExportTranslator, collect_module_tensors, export_gguf_bytes,
    export_gguf_file, export_onnx_file, export_onnx_json, import_gguf_bytes, import_gguf_file,
    import_hf_config_json, import_hf_model_dir, import_onnx_file, import_onnx_json,
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
pub use verify::*;
pub use viz::*;

use anyhow::{Result, anyhow};

/// Data payload flowing between concatenated A-to-B [`Translator`] modules.
#[derive(Debug, Clone)]
pub enum Payload {
    /// Raw binary bytes (e.g., GGUF v2/v3 or SafeTensors binary input/output).
    Bytes(Vec<u8>),
    /// UTF-8 text (e.g., MLIR source text, ONNX JSON, or generated Rust / C99 / WGSL / Viz code).
    Text(String),
    /// In-memory Dual-Plane IR [`Module`] (`Compute Graph + TensorStore`).
    Ir(Box<Module>),
}

impl Payload {
    /// Constructs a [`Payload::Ir`] from a [`Module`].
    pub fn from_ir(module: Module) -> Self {
        Self::Ir(Box::new(module))
    }

    /// Returns the human-readable payload kind (`"bytes"`, `"text"`, `"ir"`).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Bytes(_) => "bytes",
            Self::Text(_) => "text",
            Self::Ir(_) => "ir",
        }
    }

    /// Extracts the inner [`Module`] if this payload is [`Payload::Ir`].
    pub fn into_ir(self) -> Result<Module> {
        match self {
            Self::Ir(m) => Ok(*m),
            other => Err(anyhow!("Expected Payload::Ir, found {}", other.kind())),
        }
    }

    /// Extracts the inner [`String`] if this payload is [`Payload::Text`].
    pub fn into_text(self) -> Result<String> {
        match self {
            Self::Text(s) => Ok(s),
            other => Err(anyhow!("Expected Payload::Text, found {}", other.kind())),
        }
    }

    /// Extracts the inner `Vec<u8>` if this payload is [`Payload::Bytes`].
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        match self {
            Self::Bytes(b) => Ok(b),
            other => Err(anyhow!("Expected Payload::Bytes, found {}", other.kind())),
        }
    }
}

/// A single-purpose modular A-to-B translator (`Input Payload -> Output Payload`).
///
/// Complex compiler, weight-surgery, loop-scheduling, and code-generation workflows are
/// built by concatenating a sequence of simple [`Translator`] modules in a [`TranslatorChain`].
pub trait Translator: Send + Sync {
    /// Canonical identifier of this A-to-B translator module (e.g., `"parse-mlir"`,
    /// `"fuse-attention"`, `"schedule-loops"`, `"codegen-rust"`).
    fn name(&self) -> &str;

    /// Expected input representation (`"text"`, `"bytes"`, `"ir"`).
    fn input_kind(&self) -> &'static str;

    /// Produced output representation (`"text"`, `"bytes"`, `"ir"`).
    fn output_kind(&self) -> &'static str;

    /// Translates `input` (`A`) into `output` (`B`).
    fn translate(&mut self, input: Payload) -> Result<Payload>;
}

/// Concatenated sequence of modular A-to-B [`Translator`] stages (`T_1 -> T_2 -> ... -> T_n`).
#[derive(Default)]
pub struct TranslatorChain {
    stages: Vec<Box<dyn Translator>>,
}

impl TranslatorChain {
    /// Creates an empty [`TranslatorChain`].
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Appends an A-to-B [`Translator`] stage to the end of the chain.
    pub fn push(&mut self, stage: Box<dyn Translator>) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Builder-style method to append an A-to-B [`Translator`] stage.
    pub fn then<T: Translator + 'static>(mut self, stage: T) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Returns the number of concatenated translator stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Returns `true` if the chain has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Returns a summary of stage transitions (`name (A -> B)`).
    pub fn describe_stages(&self) -> Vec<String> {
        self.stages
            .iter()
            .map(|s| format!("{} ({} -> {})", s.name(), s.input_kind(), s.output_kind()))
            .collect()
    }

    /// Executes all concatenated [`Translator`] modules sequentially on `initial`.
    pub fn execute(&mut self, initial: Payload) -> Result<Payload> {
        let mut current = initial;
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            if current.kind() != stage.input_kind() {
                return Err(anyhow!(
                    "TranslatorChain stage {idx} (`{}`) expected `{}` input, found `{}`",
                    stage.name(),
                    stage.input_kind(),
                    current.kind()
                ));
            }
            current = stage.translate(current).map_err(|e| {
                anyhow!(
                    "TranslatorChain stage {idx} (`{}`) failed: {e}",
                    stage.name()
                )
            })?;
        }
        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    #[test]
    fn test_ir_fixes_and_roundtrips() {
        // 1. Purity of castforge compute ops vs collective/stateful ops
        let mut m = Module::new(Some("test".to_string()));
        let blk = m.body_block();
        let fma_id = m.add_operation(
            blk,
            OpName::new("castforge", "fma"),
            SmallVec::new(),
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let ar_id = m.add_operation(
            blk,
            OpName::new("castforge", "all_reduce"),
            SmallVec::new(),
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        assert!(m.ops[fma_id].is_pure());
        assert!(!m.ops[ar_id].is_pure());

        // 2. Recursive region cleanup on remove_op
        let sub_blk = m.create_block(None);
        let _arg = m.add_block_argument(sub_blk, Some("a".to_string()), Type::Scalar(DType::F32));
        let sub_reg = m.create_region(vec![sub_blk]);
        let reg_op = m.add_operation(
            blk,
            OpName::new("stablehlo", "while"),
            SmallVec::new(),
            Vec::new(),
            IndexMap::new(),
            vec![sub_reg],
            false,
            Location::Unknown,
        );
        assert!(m.regions.contains_key(sub_reg));
        assert!(m.blocks.contains_key(sub_blk));
        m.remove_op(blk, reg_op);
        assert!(!m.regions.contains_key(sub_reg));
        assert!(!m.blocks.contains_key(sub_blk));
        assert!(m.verify().is_ok());

        // 3. 3D nested literal formatting
        let dims_3d = vec![Dim::Static(2), Dim::Static(1), Dim::Static(2)];
        let lit = format_dense_f32_literal(&[1.0, 2.0, 3.0, 4.0], &dims_3d);
        assert_eq!(lit, "[[[1.0, 2.0]], [[3.0, 4.0]]]");

        // 4. Tuple and Opaque type parsing and FileLineCol location parsing
        let ty = parse_type("tuple<tensor<2xf32>, i32>", 1).expect("tuple parses");
        assert_eq!(ty.to_string(), "tuple<tensor<2xf32>, i32>");
        let opq = parse_type("!castforge.stream", 1).expect("opaque parses");
        assert_eq!(opq, Type::Opaque("!castforge.stream".to_string()));

        let mlir_src = concat!(
            "module @m {\n",
            "  func.func @f1(%x: tensor<2xf32>) -> tensor<2xf32> loc(\"m.py\":10:2) {\n",
            "    %c = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32> loc(\"m.py\":11:4)\n",
            "    %y = stablehlo.add %x, %c : tensor<2xf32>\n",
            "    return %y : tensor<2xf32>\n",
            "  }\n",
            "  func.func @f2(%x: tensor<2xf32>) -> tensor<2xf32> {\n",
            "    %c = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>\n",
            "    %y = stablehlo.add %x, %c : tensor<2xf32>\n",
            "    return %y : tensor<2xf32>\n",
            "  }\n",
            "}\n"
        );
        let parsed = parse_mlir(mlir_src).expect("valid mlir");
        assert_eq!(parsed.tensor_store.len(), 2);
        let fns = parsed.functions();
        assert_eq!(
            parsed.ops[fns[0]].location,
            Location::FileLineCol {
                file: "m.py".to_string(),
                line: 10,
                col: 2,
            }
        );

        // 5. LCS diff check
        let diff = diff_mlir_strings("a\nb\nc\n", "a\nx\nb\nc\n");
        assert!(diff.contains("+ x"));
        assert!(diff.contains("// Summary: -0 removed / +1 added lines"));

        // 6. Module::default() is valid and does not panic on body_block()
        let def_mod = Module::default();
        let _ = def_mod.body_block();
        assert!(def_mod.verify().is_ok());

        // 7. Bracketed uniform splat detection
        let splat_attr = Attribute::DenseElements {
            ty: TensorType::new(vec![Dim::Static(2), Dim::Static(2)], DType::F32),
            raw_literal: "[[1.0, 1.0], [1.0, 1.0]]".to_string(),
        };
        assert!(splat_attr.is_splat_one());
        assert!(!splat_attr.is_splat_zero());

        // 8. Non-default contracting_dims / batching_dims & iota dim round-trip
        let dg_mlir = concat!(
            "module @dg {\n",
            "  func.func @main(%a: tensor<2x3x4xf32>, %b: tensor<2x3x5xf32>)",
            " -> tensor<2x4x5xf32> {\n",
            "    %idx = stablehlo.iota dim = 1 : tensor<2x4xi32>\n",
            "    %out = stablehlo.dot_general %a, %b, batching_dims = [0] x [0],",
            " contracting_dims = [1] x [1]",
            " : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>\n",
            "    return %out : tensor<2x4x5xf32>\n",
            "  }\n",
            "}\n"
        );
        let dg_mod = parse_mlir(dg_mlir).expect("dg_mlir parses");
        let dg_fn = dg_mod.functions()[0];
        let dg_blk = dg_mod.regions[dg_mod.ops[dg_fn].regions[0]].blocks[0];
        let iota_op = &dg_mod.ops[dg_mod.blocks[dg_blk].ops[0]];
        assert_eq!(
            iota_op.attributes.get("iota_dimension"),
            Some(&Attribute::Int(1))
        );
        let dot_op = &dg_mod.ops[dg_mod.blocks[dg_blk].ops[1]];
        assert_eq!(
            dot_op.attributes.get("lhs_batching_dimensions"),
            Some(&Attribute::DenseIntArray(vec![0]))
        );
        assert_eq!(
            dot_op.attributes.get("rhs_contracting_dimensions"),
            Some(&Attribute::DenseIntArray(vec![1]))
        );
        assert!(verify_stablehlo_shapes(&dg_mod, dot_op.id).is_ok());

        // 9. Attribute parsing for array<f64: ...>, nested dict, and non-int array
        let attr_f64 = parser::parse_attribute_value("array<f64: 1.5, 2.5>");
        assert_eq!(attr_f64, Attribute::DenseF64Array(vec![1.5, 2.5]));
        let attr_str_arr = parser::parse_attribute_value("[\"x\", \"y\"]");
        assert_eq!(
            attr_str_arr,
            Attribute::Array(vec![
                Attribute::String("x".to_string()),
                Attribute::String("y".to_string()),
            ])
        );

        // 10. f32_to_f16_bits preserves negative zero sign on underflow
        assert_eq!(f32_to_f16_bits(-1e-10), 0x8000);
    }
}
