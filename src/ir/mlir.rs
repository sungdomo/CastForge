//! Core MLIR-compatible SSA Compute Graph arena, type system, attributes, operations, blocks,
//! regions, and [`Module`] representation (`castforge::ir::mlir`).

use crate::ir::TensorStore;
use anyhow::Result;
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

    /// If this attribute is a scalar/splat `DenseElements` constant (e.g., `dense<0.0>` or a
    /// bracketed literal where all elements are identical), parses and returns its scalar `f64`
    /// value.
    pub fn as_splat_f64(&self) -> Option<f64> {
        match self {
            Attribute::DenseElements { raw_literal, .. } => {
                let s = raw_literal.trim();
                if !s.starts_with('[') {
                    return s.parse::<f64>().ok();
                }
                let cleaned: String = s
                    .chars()
                    .map(|c| if c == '[' || c == ']' { ' ' } else { c })
                    .collect();
                let mut first: Option<f64> = None;
                for tok in cleaned.split(',') {
                    let tok = tok.trim();
                    if tok.is_empty() {
                        continue;
                    }
                    let val = tok.parse::<f64>().ok()?;
                    match first {
                        None => first = Some(val),
                        Some(f) if (val - f).abs() <= f64::EPSILON => {}
                        Some(_) => return None,
                    }
                }
                first
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

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("Module root_region does not exist in regions arena")]
    InvalidRootRegion,

    #[error("Operation `{op_name}` references missing operand ValueId")]
    DanglingOperand { op_name: String },

    #[error("Operation `{op_name}` references missing result ValueId")]
    DanglingResult { op_name: String },

    #[error("Operation `{op_name}` references missing RegionId")]
    DanglingRegion { op_name: String },

    #[error("Block references missing argument ValueId")]
    DanglingBlockArgument,

    #[error("Region references missing BlockId")]
    DanglingBlock,

    #[error("Block references missing OpId")]
    DanglingOp,

    #[error("SSA Value definition metadata does not match owning Operation or Block")]
    InvalidValueDef,
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
            ("castforge", name) => !matches!(
                name,
                "kv_cache_update"
                    | "all_reduce"
                    | "all_gather"
                    | "reduce_scatter"
                    | "all_to_all"
                    | "send"
                    | "recv"
                    | "stream_wait"
                    | "offload_to_host"
                    | "offload_to_device"
                    | "profile_region"
            ),
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub name: Option<String>,
    pub ops: SlotMap<OpId, Operation>,
    pub values: SlotMap<ValueId, Value>,
    pub blocks: SlotMap<BlockId, Block>,
    pub regions: SlotMap<RegionId, Region>,
    pub root_region: RegionId,
    pub tensor_store: TensorStore,
}

impl Default for Module {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Module {
    pub fn new(name: Option<String>) -> Self {
        let mut module = Self {
            name,
            ops: SlotMap::with_key(),
            values: SlotMap::with_key(),
            blocks: SlotMap::with_key(),
            regions: SlotMap::with_key(),
            root_region: RegionId::default(),
            tensor_store: TensorStore::default(),
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

    /// Removes `op_id` from `block` and deletes it (and any nested regions) from the arena.
    pub fn remove_op(&mut self, block: BlockId, op_id: OpId) {
        if let Some(blk) = self.blocks.get_mut(block) {
            blk.ops.retain(|&id| id != op_id);
        }
        self.remove_op_unlinked(op_id);
    }

    fn remove_op_unlinked(&mut self, op_id: OpId) {
        if let Some(op) = self.ops.remove(op_id) {
            for res in op.results {
                self.values.remove(res);
            }
            for reg_id in op.regions {
                self.remove_region(reg_id);
            }
        }
    }

    fn remove_region(&mut self, region_id: RegionId) {
        if let Some(region) = self.regions.remove(region_id) {
            for blk_id in region.blocks {
                if let Some(blk) = self.blocks.remove(blk_id) {
                    for arg in blk.arguments {
                        self.values.remove(arg);
                    }
                    for op_id in blk.ops {
                        self.remove_op_unlinked(op_id);
                    }
                }
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

    /// Returns all live operations reachable from `self.root_region` in program order.
    pub fn live_ops(&self) -> Vec<OpId> {
        let mut out = Vec::new();
        self.collect_region_live_ops(self.root_region, &mut out);
        out
    }

    fn collect_region_live_ops(&self, region_id: RegionId, out: &mut Vec<OpId>) {
        let Some(region) = self.regions.get(region_id) else {
            return;
        };
        for &blk_id in &region.blocks {
            let Some(blk) = self.blocks.get(blk_id) else {
                continue;
            };
            for &op_id in &blk.ops {
                if let Some(op) = self.ops.get(op_id) {
                    out.push(op_id);
                    for &child_reg in &op.regions {
                        self.collect_region_live_ops(child_reg, out);
                    }
                }
            }
        }
    }

    /// Verifies structural integrity of the IR (all SSA ValueIds, OpIds, BlockIds, and RegionIds
    /// resolve).
    pub fn verify(&self) -> Result<(), VerifyError> {
        let Some(root_reg) = self.regions.get(self.root_region) else {
            return Err(VerifyError::InvalidRootRegion);
        };
        if root_reg.blocks.is_empty() {
            return Err(VerifyError::DanglingBlock);
        }
        for (_, region) in &self.regions {
            for &block_id in &region.blocks {
                let Some(block) = self.blocks.get(block_id) else {
                    return Err(VerifyError::DanglingBlock);
                };
                for (idx, &arg_id) in block.arguments.iter().enumerate() {
                    let Some(val) = self.values.get(arg_id) else {
                        return Err(VerifyError::DanglingBlockArgument);
                    };
                    if val.def
                        != (ValueDef::BlockArgument {
                            block: block_id,
                            index: idx as u32,
                        })
                    {
                        return Err(VerifyError::InvalidValueDef);
                    }
                }
                for &op_id in &block.ops {
                    let Some(op) = self.ops.get(op_id) else {
                        return Err(VerifyError::DanglingOp);
                    };
                    for &operand in &op.operands {
                        if !self.values.contains_key(operand) {
                            return Err(VerifyError::DanglingOperand {
                                op_name: op.name.full_name(),
                            });
                        }
                    }
                    for (idx, &result) in op.results.iter().enumerate() {
                        let Some(val) = self.values.get(result) else {
                            return Err(VerifyError::DanglingResult {
                                op_name: op.name.full_name(),
                            });
                        };
                        if val.def
                            != (ValueDef::OpResult {
                                op: op_id,
                                index: idx as u32,
                            })
                        {
                            return Err(VerifyError::InvalidValueDef);
                        }
                    }
                    for &reg_id in &op.regions {
                        if !self.regions.contains_key(reg_id) {
                            return Err(VerifyError::DanglingRegion {
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
