//! Composable A-to-B [`Translator`] trait, [`Payload`] representation, and concatenated
//! [`TranslatorChain`] execution engine.

use crate::Module;
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
