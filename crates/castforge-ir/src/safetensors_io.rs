use crate::attr::Attribute;
use crate::tensor_store::{TensorBacking, TensorEntry, TensorStore, parse_dense_f32_literal};
use crate::types::{DType, Dim, TensorType};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SafeTensorEntryHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zero_point: Option<i64>,
}

impl TensorStore {
    /// Resolve `f32` values from either an inline `DenseElements` literal or a `TensorRef`
    /// (`dense_resource<key>`) stored inside this `TensorStore`.
    pub fn resolve_f32_attr(&self, attr: &Attribute, expected_len: usize) -> Option<Vec<f32>> {
        match attr {
            Attribute::DenseElements { raw_literal, .. } => {
                parse_dense_f32_literal(raw_literal, expected_len)
            }
            Attribute::TensorRef { key, .. } => {
                let entry = self.get(key)?;
                match &entry.backing {
                    TensorBacking::F32Data(vals) => Some(vals.clone()),
                    TensorBacking::QuantizedI8 {
                        data,
                        scale,
                        zero_point,
                    } => Some(
                        data.iter()
                            .map(|&q| (q as i64 - *zero_point) as f32 * *scale)
                            .collect(),
                    ),
                    TensorBacking::InMemory(bytes) => {
                        if bytes.len() == expected_len * 4 {
                            Some(
                                bytes
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Resolve quantized `i8` values from a `TensorRef` or `DenseElements`.
    pub fn resolve_i8_attr(&self, attr: &Attribute, expected_len: usize) -> Option<Vec<i8>> {
        match attr {
            Attribute::TensorRef { key, .. } => {
                let entry = self.get(key)?;
                match &entry.backing {
                    TensorBacking::QuantizedI8 { data, .. } => Some(data.clone()),
                    TensorBacking::InMemory(bytes) if bytes.len() == expected_len => {
                        Some(bytes.iter().map(|&b| b as i8).collect())
                    }
                    _ => None,
                }
            }
            Attribute::DenseElements { raw_literal, .. } => {
                parse_dense_f32_literal(raw_literal, expected_len)
                    .map(|v| v.into_iter().map(|x| x as i8).collect())
            }
            _ => None,
        }
    }

    /// Serialize all in-memory tensors in `TensorStore` into the binary `.safetensors` format.
    pub fn to_safetensors_bytes(&self) -> Result<Vec<u8>, io::Error> {
        let mut headers: IndexMap<String, SafeTensorEntryHeader> = IndexMap::new();
        let mut data_buf: Vec<u8> = Vec::new();

        for (name, entry) in &self.entries {
            let shape: Vec<usize> = entry
                .ty
                .shape
                .iter()
                .map(|d| match d {
                    Dim::Static(n) if *n > 0 => *n as usize,
                    _ => 1,
                })
                .collect();

            let start = data_buf.len();
            match &entry.backing {
                TensorBacking::F32Data(vals) => {
                    for &v in vals {
                        data_buf.extend_from_slice(&v.to_le_bytes());
                    }
                    let end = data_buf.len();
                    headers.insert(
                        name.clone(),
                        SafeTensorEntryHeader {
                            dtype: "F32".to_string(),
                            shape,
                            data_offsets: [start, end],
                            scale: None,
                            zero_point: None,
                        },
                    );
                }
                TensorBacking::QuantizedI8 {
                    data,
                    scale,
                    zero_point,
                } => {
                    for &q in data {
                        data_buf.push(q as u8);
                    }
                    let end = data_buf.len();
                    headers.insert(
                        name.clone(),
                        SafeTensorEntryHeader {
                            dtype: "I8".to_string(),
                            shape,
                            data_offsets: [start, end],
                            scale: Some(*scale),
                            zero_point: Some(*zero_point),
                        },
                    );
                }
                TensorBacking::InMemory(raw) => {
                    data_buf.extend_from_slice(raw);
                    let end = data_buf.len();
                    headers.insert(
                        name.clone(),
                        SafeTensorEntryHeader {
                            dtype: "U8".to_string(),
                            shape,
                            data_offsets: [start, end],
                            scale: None,
                            zero_point: None,
                        },
                    );
                }
                _ => {}
            }
        }

        let mut header_json = serde_json::to_vec(&headers)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        // Pad header JSON to 8-byte alignment per Safetensors spec
        while !header_json.len().is_multiple_of(8) {
            header_json.push(b' ');
        }

        let header_len = header_json.len() as u64;
        let mut out = Vec::with_capacity(8 + header_json.len() + data_buf.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_json);
        out.extend_from_slice(&data_buf);
        Ok(out)
    }

    /// Load tensors from a `.safetensors` byte slice into this `TensorStore`.
    pub fn load_safetensors_bytes(&mut self, bytes: &[u8]) -> Result<usize, io::Error> {
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Safetensors buffer shorter than 8-byte header length",
            ));
        }
        let header_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize;
        let data_start = 8usize.saturating_add(header_len);
        if bytes.len() < data_start {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Safetensors buffer truncated inside JSON header",
            ));
        }

        let header_slice = &bytes[8..data_start];
        let body_slice = &bytes[data_start..];

        let raw_map: IndexMap<String, serde_json::Value> = serde_json::from_slice(header_slice)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut loaded = 0usize;
        for (name, val) in raw_map {
            if name == "__metadata__" {
                continue;
            }
            let hdr: SafeTensorEntryHeader = serde_json::from_value(val)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let [start, end] = hdr.data_offsets;
            if end > body_slice.len() || start > end {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("Invalid data_offsets [{start}, {end}] for tensor `{name}`"),
                ));
            }
            let slice = &body_slice[start..end];
            let shape: Vec<Dim> = hdr.shape.iter().map(|&d| Dim::Static(d as i64)).collect();

            match hdr.dtype.as_str() {
                "F32" => {
                    let vals: Vec<f32> = slice
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let ty = TensorType::new(shape, DType::F32);
                    self.insert(TensorEntry::from_f32(name, ty, vals));
                    loaded += 1;
                }
                "I8" => {
                    let q_data: Vec<i8> = slice.iter().map(|&b| b as i8).collect();
                    let scale = hdr.scale.unwrap_or(1.0);
                    let zp = hdr.zero_point.unwrap_or(0);
                    let dtype = if hdr.scale.is_some() {
                        DType::QuantUniform {
                            storage_type: Box::new(DType::I8),
                            expressed_type: Box::new(DType::F32),
                            scale: format!("{scale:.6}")
                                .trim_end_matches('0')
                                .trim_end_matches('.')
                                .to_string(),
                            zero_point: zp,
                        }
                    } else {
                        DType::I8
                    };
                    let ty = TensorType::new(shape, dtype);
                    self.insert(TensorEntry {
                        name,
                        ty,
                        backing: TensorBacking::QuantizedI8 {
                            data: q_data,
                            scale,
                            zero_point: zp,
                        },
                    });
                    loaded += 1;
                }
                _ => {
                    let ty = TensorType::new(shape, DType::U8);
                    self.insert(TensorEntry {
                        name,
                        ty,
                        backing: TensorBacking::InMemory(slice.to_vec()),
                    });
                    loaded += 1;
                }
            }
        }
        Ok(loaded)
    }

    /// Save this `TensorStore` to a `.safetensors` file on disk.
    pub fn save_safetensors_file(&self, path: impl AsRef<Path>) -> Result<(), io::Error> {
        let bytes = self.to_safetensors_bytes()?;
        fs::write(path, bytes)
    }

    /// Load a `.safetensors` file from disk into this `TensorStore`.
    pub fn load_safetensors_file(&mut self, path: impl AsRef<Path>) -> Result<usize, io::Error> {
        let bytes = fs::read(path)?;
        self.load_safetensors_bytes(&bytes)
    }
}
