//! Core MLIR-compatible Dual-Plane Intermediate Representation (`castforge::ir`).
//!
//! Provides arena-allocated SSA IR data structures ([`Module`], [`Operation`], [`Value`],
//! [`Block`], [`Region`], [`Type`], [`Attribute`], [`Location`]), the [`TensorStore`] weight
//! plane, SafeTensors binary I/O, strongly-typed [`StableHloOp`] dialect views, the MLIR
//! parser/printer, ONNX/GGUF importers/exporters, and the composable [`Translator`] /
//! [`TranslatorChain`] A-to-B pipeline abstraction.

pub mod import;
pub mod mlir;
pub mod parser;
pub mod stablehlo;
pub mod tensor_store;

pub use import::{
    GgmlType, GgufExportTranslator, OnnxExportTranslator, OnnxJsonInitializer, OnnxJsonModel,
    OnnxJsonNode, OnnxJsonValueInfo, SafetensorsExportTranslator, collect_module_tensors,
    export_gguf_bytes, export_gguf_file, export_onnx_file, export_onnx_json, import_gguf_bytes,
    import_gguf_file, import_onnx_file, import_onnx_json,
};
pub use mlir::{
    GgufImportTranslator, MlirParseTranslator, MlirPrintTranslator, MlirPrinter,
    OnnxImportTranslator, PrintOptions, parse_mlir, print_mlir, print_mlir_with_options,
};
pub use parser::{MlirParser, ParseError, parse_type};
pub use stablehlo::{DotDimensionNumbers, FuncView, StableHloOp, verify_stablehlo_shapes};
pub use tensor_store::{
    TensorBacking, TensorEntry, TensorStore, format_dense_f32_literal, format_dense_i8_literal,
    parse_dense_f32_literal,
};

use anyhow::{Result, anyhow};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use std::fmt;
use thiserror::Error;

/// Element data types supported in tensor and scalar types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    I1,
    I4,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F8E4M3FN,
    F8E5M2,
    F16,
    BF16,
    F32,
    F64,
    Index,
    QuantUniform {
        storage_type: Box<DType>,
        expressed_type: Box<DType>,
        scale: String,
        zero_point: i64,
    },
}

impl DType {
    pub fn bit_width(&self) -> usize {
        match self {
            DType::I1 => 1,
            DType::I4 => 4,
            DType::I8 | DType::U8 | DType::F8E4M3FN | DType::F8E5M2 => 8,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 16,
            DType::I32 | DType::U32 | DType::F32 => 32,
            DType::I64 | DType::U64 | DType::F64 | DType::Index => 64,
            DType::QuantUniform { storage_type, .. } => storage_type.bit_width(),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::I1 => write!(f, "i1"),
            DType::I4 => write!(f, "i4"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "ui8"),
            DType::U16 => write!(f, "ui16"),
            DType::U32 => write!(f, "ui32"),
            DType::U64 => write!(f, "ui64"),
            DType::F8E4M3FN => write!(f, "f8E4M3FN"),
            DType::F8E5M2 => write!(f, "f8E5M2"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::Index => write!(f, "index"),
            DType::QuantUniform {
                storage_type,
                expressed_type,
                scale,
                zero_point,
            } => write!(
                f,
                "!quant.uniform<{}:{},{}:{}>",
                storage_type, expressed_type, scale, zero_point
            ),
        }
    }
}

/// Tensor dimension (static, dynamic `?`, or symbolic).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Static(i64),
    Dynamic,
    Symbolic(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(d) => write!(f, "{d}"),
            Dim::Dynamic => write!(f, "?"),
            Dim::Symbolic(s) => write!(f, "{s}"),
        }
    }
}

/// Ranked or unranked tensor type (`tensor<4x8xf32>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<Dim>,
    pub dtype: DType,
    pub encoding: Option<String>,
}

impl TensorType {
    pub fn new(shape: Vec<Dim>, dtype: DType) -> Self {
        Self {
            shape,
            dtype,
            encoding: None,
        }
    }

    pub fn static_shape(dims: &[i64], dtype: DType) -> Self {
        Self {
            shape: dims.iter().copied().map(Dim::Static).collect(),
            dtype,
            encoding: None,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn num_elements(&self) -> Option<i64> {
        let mut prod = 1i64;
        for d in &self.shape {
            match d {
                Dim::Static(n) => prod = prod.checked_mul(*n)?,
                _ => return None,
            }
        }
        Some(prod)
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor<")?;
        for d in &self.shape {
            write!(f, "{d}x")?;
        }
        write!(f, "{}", self.dtype)?;
        if let Some(enc) = &self.encoding {
            write!(f, ", {enc}")?;
        }
        write!(f, ">")
    }
}

/// IR Type representation (compatible with MLIR type system).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    Tensor(TensorType),
    Scalar(DType),
    Tuple(Vec<Type>),
    Function {
        inputs: Vec<Type>,
        results: Vec<Type>,
    },
    Opaque(String),
}

impl Type {
    pub fn as_tensor(&self) -> Option<&TensorType> {
        match self {
            Type::Tensor(t) => Some(t),
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Tensor(t) => write!(f, "{t}"),
            Type::Scalar(d) => write!(f, "{d}"),
            Type::Tuple(items) => {
                write!(f, "tuple<")?;
                for (i, ty) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ">")
            }
            Type::Function { inputs, results } => {
                write!(f, "(")?;
                for (i, inp) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{inp}")?;
                }
                write!(f, ") -> ")?;
                if results.len() == 1 {
                    write!(f, "{}", results[0])
                } else {
                    write!(f, "(")?;
                    for (i, res) in results.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{res}")?;
                    }
                    write!(f, ")")
                }
            }
            Type::Opaque(raw) => write!(f, "{raw}"),
        }
    }
}

/// Compile-time attributes attached to MLIR/StableHLO operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    SymbolRef(String),
    Type(Type),
    Array(Vec<Attribute>),
    DenseIntArray(Vec<i64>),
    DenseF64Array(Vec<f64>),
    DenseElements {
        ty: TensorType,
        raw_literal: String,
    },
    /// Reference to a weight tensor stored in the Module's `TensorStore`
    TensorRef {
        key: String,
        ty: TensorType,
    },
    Dictionary(IndexMap<String, Attribute>),
    Opaque(String),
}

impl Attribute {
    pub fn as_int_slice(&self) -> Option<&[i64]> {
        match self {
            Attribute::DenseIntArray(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Attribute::String(s) | Attribute::SymbolRef(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// If this attribute is a scalar/splat `DenseElements` constant (e.g., `dense<0.0>`),
    /// parses and returns its scalar `f64` value.
    pub fn as_splat_f64(&self) -> Option<f64> {
        match self {
            Attribute::DenseElements { raw_literal, .. } => {
                let s = raw_literal.trim();
                if s.starts_with('[') {
                    return None;
                }
                s.parse::<f64>().ok()
            }
            Attribute::Float(f) => Some(*f),
            Attribute::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn is_splat_zero(&self) -> bool {
        self.as_splat_f64().map(|v| v == 0.0).unwrap_or(false)
    }

    pub fn is_splat_one(&self) -> bool {
        self.as_splat_f64()
            .map(|v| (v - 1.0).abs() < f64::EPSILON)
            .unwrap_or(false)
    }
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Attribute::Bool(b) => write!(f, "{b}"),
            Attribute::Int(i) => write!(f, "{i}"),
            Attribute::Float(v) => write!(f, "{v:?}"),
            Attribute::String(s) => write!(f, "\"{}\"", s.escape_default()),
            Attribute::SymbolRef(s) => write!(f, "@{s}"),
            Attribute::Type(ty) => write!(f, "{ty}"),
            Attribute::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Attribute::DenseIntArray(vals) => {
                write!(f, "array<i64: ")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, ">")
            }
            Attribute::DenseF64Array(vals) => {
                write!(f, "array<f64: ")?;
                for (i, v) in vals.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v:?}")?;
                }
                write!(f, ">")
            }
            Attribute::DenseElements { ty, raw_literal } => {
                write!(f, "dense<{raw_literal}> : {ty}")
            }
            Attribute::TensorRef { key, ty } => {
                write!(f, "dense_resource<{key}> : {ty}")
            }
            Attribute::Dictionary(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k} = {v}")?;
                }
                write!(f, "}}")
            }
            Attribute::Opaque(raw) => write!(f, "{raw}"),
        }
    }
}

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
    /// `"fuse-attention"`, `"tile-loops"`, `"codegen-rust"`).
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
        for stage in &mut self.stages {
            current = stage.translate(current)?;
        }
        Ok(current)
    }
}

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("Operation `{op_name}` references missing operand ValueId")]
    DanglingOperand { op_name: String },

    #[error("Operation `{op_name}` references missing result ValueId")]
    DanglingResult { op_name: String },

    #[error("Region references missing BlockId")]
    DanglingBlock,
}

new_key_type! {
    pub struct OpId;
    pub struct ValueId;
    pub struct BlockId;
    pub struct RegionId;
}

/// Source location metadata (mirrors MLIR `loc(...)`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Location {
    #[default]
    Unknown,
    Named(String),
    FileLineCol {
        file: String,
        line: u32,
        col: u32,
    },
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::Unknown => write!(f, "loc(unknown)"),
            Location::Named(name) => write!(f, "loc(\"{name}\")"),
            Location::FileLineCol { file, line, col } => {
                write!(f, "loc(\"{file}\":{line}:{col})")
            }
        }
    }
}

/// Qualified operation name (`dialect.op_name`, e.g., `stablehlo.add`, `func.func`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpName {
    pub dialect: String,
    pub name: String,
}

impl OpName {
    pub fn new(dialect: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            dialect: dialect.into(),
            name: name.into(),
        }
    }

    pub fn parse(qualified: &str) -> Self {
        if let Some((dialect, name)) = qualified.split_once('.') {
            Self::new(dialect, name)
        } else {
            Self::new("builtin", qualified)
        }
    }

    pub fn full_name(&self) -> String {
        if self.dialect == "builtin" && self.name == "module" {
            "builtin.module".to_string()
        } else {
            format!("{}.{}", self.dialect, self.name)
        }
    }
}

impl fmt::Display for OpName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.dialect, self.name)
    }
}

/// How an SSA Value is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueDef {
    OpResult { op: OpId, index: u32 },
    BlockArgument { block: BlockId, index: u32 },
}

/// An SSA Value in the compute graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    pub id: ValueId,
    /// Optional user-preserved SSA identifier (e.g., `arg0`, `0`, `hidden_states`).
    pub ssa_name: Option<String>,
    pub ty: Type,
    pub def: ValueDef,
}

/// An MLIR-compatible Operation node in the arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: OpId,
    pub name: OpName,
    pub operands: SmallVec<[ValueId; 4]>,
    pub results: SmallVec<[ValueId; 2]>,
    pub attributes: IndexMap<String, Attribute>,
    pub regions: Vec<RegionId>,
    pub custom_syntax: bool,
    pub location: Location,
}

impl Operation {
    pub fn is_op(&self, dialect: &str, name: &str) -> bool {
        self.name.dialect == dialect && self.name.name == name
    }

    /// Returns true if this operation has no side effects and can be safely eliminated when unused.
    pub fn is_pure(&self) -> bool {
        match (self.name.dialect.as_str(), self.name.name.as_str()) {
            ("func", "return") | ("func", "func") | ("func", "call") => false,
            ("stablehlo", "custom_call") => {
                // Unless marked has_side_effect = true, custom_call is pure
                !matches!(
                    self.attributes.get("has_side_effect"),
                    Some(Attribute::Bool(true))
                )
            }
            ("stablehlo", _) => true,
            _ => false,
        }
    }
}

/// A basic block containing a sequence of operations and block arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub id: BlockId,
    pub label: Option<String>,
    pub arguments: Vec<ValueId>,
    pub ops: Vec<OpId>,
}

/// A region containing one or more blocks (e.g., function body, module body, loop body).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub id: RegionId,
    pub blocks: Vec<BlockId>,
}

/// Top-level CastForge Dual-Plane IR Module (SSA Compute Graph + Weight TensorStore).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Module {
    pub name: Option<String>,
    pub ops: SlotMap<OpId, Operation>,
    pub values: SlotMap<ValueId, Value>,
    pub blocks: SlotMap<BlockId, Block>,
    pub regions: SlotMap<RegionId, Region>,
    pub root_region: RegionId,
    pub tensor_store: TensorStore,
}

impl Module {
    pub fn new(name: Option<String>) -> Self {
        let mut module = Self {
            name,
            ..Default::default()
        };
        let entry_block = module.create_block(None);
        let root_region = module.create_region(vec![entry_block]);
        module.root_region = root_region;
        module
    }

    pub fn body_block(&self) -> BlockId {
        self.regions[self.root_region].blocks[0]
    }

    pub fn create_block(&mut self, label: Option<String>) -> BlockId {
        self.blocks.insert_with_key(|id| Block {
            id,
            label,
            arguments: Vec::new(),
            ops: Vec::new(),
        })
    }

    pub fn create_region(&mut self, blocks: Vec<BlockId>) -> RegionId {
        self.regions.insert_with_key(|id| Region { id, blocks })
    }

    pub fn add_block_argument(
        &mut self,
        block: BlockId,
        ssa_name: Option<String>,
        ty: Type,
    ) -> ValueId {
        let index = self.blocks[block].arguments.len() as u32;
        let val_id = self.values.insert_with_key(|id| Value {
            id,
            ssa_name,
            ty,
            def: ValueDef::BlockArgument { block, index },
        });
        self.blocks[block].arguments.push(val_id);
        val_id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_operation(
        &mut self,
        block: BlockId,
        name: OpName,
        operands: SmallVec<[ValueId; 4]>,
        result_types: Vec<(Option<String>, Type)>,
        attributes: IndexMap<String, Attribute>,
        regions: Vec<RegionId>,
        custom_syntax: bool,
        location: Location,
    ) -> OpId {
        let op_id = self.ops.insert_with_key(|id| Operation {
            id,
            name,
            operands,
            results: SmallVec::new(),
            attributes,
            regions,
            custom_syntax,
            location,
        });

        let mut results = SmallVec::with_capacity(result_types.len());
        for (idx, (ssa_name, ty)) in result_types.into_iter().enumerate() {
            let val_id = self.values.insert_with_key(|id| Value {
                id,
                ssa_name,
                ty,
                def: ValueDef::OpResult {
                    op: op_id,
                    index: idx as u32,
                },
            });
            results.push(val_id);
        }
        self.ops[op_id].results = results;
        self.blocks[block].ops.push(op_id);
        op_id
    }

    /// Replace all uses of `from` with `to` across all operations in the module.
    pub fn replace_all_uses(&mut self, from: ValueId, to: ValueId) -> usize {
        if from == to {
            return 0;
        }
        let mut count = 0;
        for (_, op) in &mut self.ops {
            for operand in &mut op.operands {
                if *operand == from {
                    *operand = to;
                    count += 1;
                }
            }
        }
        count
    }

    /// Count how many operation operands reference `val`.
    pub fn use_count(&self, val: ValueId) -> usize {
        let mut count = 0;
        for (_, region) in &self.regions {
            for &block_id in &region.blocks {
                for &op_id in &self.blocks[block_id].ops {
                    if let Some(op) = self.ops.get(op_id) {
                        count += op.operands.iter().filter(|&&v| v == val).count();
                    }
                }
            }
        }
        count
    }

    /// Computes a reference-count map (`ValueId -> use_count`) across all active operations
    /// in the module in a single $O(N)$ pass.
    pub fn compute_use_counts(&self) -> std::collections::HashMap<ValueId, usize> {
        let mut counts = std::collections::HashMap::with_capacity(self.values.len());
        for (_, region) in &self.regions {
            for &block_id in &region.blocks {
                for &op_id in &self.blocks[block_id].ops {
                    if let Some(op) = self.ops.get(op_id) {
                        for &operand in &op.operands {
                            *counts.entry(operand).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        counts
    }

    /// Returns all active `OpId`s in `block`.
    pub fn block_ops(&self, block: BlockId) -> Vec<OpId> {
        self.blocks[block].ops.clone()
    }

    /// Removes `op_id` from `block` and deletes it from the arena.
    pub fn remove_op(&mut self, block: BlockId, op_id: OpId) {
        self.blocks[block].ops.retain(|&id| id != op_id);
        if let Some(op) = self.ops.remove(op_id) {
            for res in op.results {
                self.values.remove(res);
            }
        }
    }

    /// Returns top-level function operations (`func.func`) inside the module.
    pub fn functions(&self) -> Vec<OpId> {
        let body = self.body_block();
        self.blocks[body]
            .ops
            .iter()
            .copied()
            .filter(|&op_id| {
                self.ops
                    .get(op_id)
                    .map(|op| op.is_op("func", "func"))
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Verifies structural integrity of the IR (all SSA ValueIds, OpIds, and BlockIds resolve).
    pub fn verify(&self) -> Result<(), VerifyError> {
        for (_, region) in &self.regions {
            for &block_id in &region.blocks {
                let Some(block) = self.blocks.get(block_id) else {
                    return Err(VerifyError::DanglingBlock);
                };
                for &op_id in &block.ops {
                    let Some(op) = self.ops.get(op_id) else {
                        continue;
                    };
                    for &operand in &op.operands {
                        if !self.values.contains_key(operand) {
                            return Err(VerifyError::DanglingOperand {
                                op_name: op.name.full_name(),
                            });
                        }
                    }
                    for &result in &op.results {
                        if !self.values.contains_key(result) {
                            return Err(VerifyError::DanglingResult {
                                op_name: op.name.full_name(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
