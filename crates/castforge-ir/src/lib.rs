//! Core MLIR-compatible Dual-Plane Intermediate Representation (`castforge-ir`).
//!
//! Provides arena-allocated SSA IR data structures ([`Module`], [`Operation`], [`Value`],
//! [`Block`], [`Region`], [`Type`], [`Attribute`], [`Location`]), the [`TensorStore`] weight
//! plane, SafeTensors binary I/O, and the composable [`Translator`] / [`TranslatorChain`]
//! A-to-B pipeline abstraction.

pub mod attr;
pub mod safetensors_io;
pub mod tensor_store;
pub mod translator;
pub mod types;
pub mod verify;

pub use attr::Attribute;
pub use tensor_store::{
    MergeMethod, TensorBacking, TensorEntry, TensorStore, format_dense_f32_literal,
    format_dense_i8_literal, merge_f32_tensors, parse_dense_f32_literal, prune_2d_columns_by_l1,
    quantize_symmetric_f32,
};
pub use translator::{Payload, Translator, TranslatorChain};
pub use types::{DType, Dim, TensorType, Type};
pub use verify::VerifyError;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use std::fmt;

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
}
