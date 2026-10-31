//! Weight Plane (`TensorStore`), `TensorEntry`, `TensorBacking`, IEEE `f16`/`bf16` conversion
//! helpers, N-D dense literal formatting, and binary `.safetensors` serialization
//! (`castforge::ir::tensor_store`).

use crate::ir::{Attribute, DType, Dim, TensorType};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Backing storage for a weight/constant tensor in the Weight Plane (`TensorStore`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorBacking {
    /// In-memory raw byte buffer.
    InMemory(Vec<u8>),
    /// In-memory typed F32 values (convenient for pruning, merging, and PTQ quantization).
    F32Data(Vec<f32>),
    /// In-memory quantized I8 values with scale and zero_point.
    QuantizedI8 {
        data: Vec<i8>,
        scale: f32,
        zero_point: i64,
    },
    /// External file slice (e.g., Safetensors, GGUF, or MLIR external resource).
    ExternalFile {
        path: PathBuf,
        offset: u64,
        length: u64,
    },
    /// Unallocated symbolic placeholder (graph-only mode).
    Symbolic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub ty: TensorType,
    pub backing: TensorBacking,
}

impl TensorEntry {
    pub fn from_f32(name: impl Into<String>, ty: TensorType, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            ty,
            backing: TensorBacking::F32Data(values),
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.backing {
            TensorBacking::F32Data(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Converts an IEEE 754 `f16` 16-bit pattern to `f32`.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Converts an `f32` value to an IEEE 754 `f16` 16-bit pattern.
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        sign | 0x7c00
    } else if new_exp <= 0 {
        sign
    } else {
        sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

/// Converts a `bf16` 16-bit pattern to `f32`.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts an `f32` value to a `bf16` 16-bit pattern.
pub fn f32_to_bf16_bits(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

/// Parse a flat or nested MLIR dense literal (e.g., `[[1.0, 2.0], [3.0, 4.0]]` or `0.5`)
/// into a vector of `f32` of length `expected_len` (expanding splats automatically).
pub fn parse_dense_f32_literal(raw_literal: &str, expected_len: usize) -> Option<Vec<f32>> {
    let trimmed = raw_literal.trim();
    if !trimmed.starts_with('[') {
        let val = trimmed.parse::<f32>().ok()?;
        return Some(vec![val; expected_len.max(1)]);
    }
    let mut vals = Vec::new();
    let cleaned: String = trimmed
        .chars()
        .map(|c| if c == '[' || c == ']' { ' ' } else { c })
        .collect();
    for tok in cleaned.split(',') {
        let tok = tok.trim();
        if !tok.is_empty() {
            vals.push(tok.parse::<f32>().ok()?);
        }
    }
    if vals.len() == 1 && expected_len > 1 {
        return Some(vec![vals[0]; expected_len]);
    }
    Some(vals)
}

fn format_nd_slice<T, F>(values: &[T], dims: &[usize], fmt_elem: &F) -> String
where
    F: Fn(&T) -> String,
{
    if dims.len() <= 1 {
        let elems: Vec<String> = values.iter().map(fmt_elem).collect();
        return format!("[{}]", elems.join(", "));
    }
    let outer = dims[0];
    let inner_stride: usize = dims[1..].iter().product();
    let mut parts = Vec::with_capacity(outer);
    for i in 0..outer {
        let start = i * inner_stride;
        let end = start + inner_stride;
        parts.push(format_nd_slice(&values[start..end], &dims[1..], fmt_elem));
    }
    format!("[{}]", parts.join(", "))
}

/// Format a slice of `f32` or `i8` values as an MLIR `dense<...>` literal matching `shape`.
pub fn format_dense_f32_literal(values: &[f32], shape: &[Dim]) -> String {
    if values.is_empty() {
        return "0.0".to_string();
    }
    // Check if all elements are identical (splat)
    let first = values[0];
    if values.len() > 1 && values.iter().all(|&v| (v - first).abs() < 1e-7) {
        return format!("{first:?}");
    }
    let static_dims: Vec<usize> = shape
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) if *n > 0 => Some(*n as usize),
            _ => None,
        })
        .collect();
    if static_dims.len() >= 2
        && static_dims.len() == shape.len()
        && static_dims.iter().product::<usize>() == values.len()
    {
        format_nd_slice(values, &static_dims, &|v| format!("{v:?}"))
    } else {
        let elems: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
        format!("[{}]", elems.join(", "))
    }
}

pub fn format_dense_i8_literal(values: &[i8], shape: &[Dim]) -> String {
    if values.is_empty() {
        return "0".to_string();
    }
    let first = values[0];
    if values.len() > 1 && values.iter().all(|&v| v == first) {
        return format!("{first}");
    }
    let static_dims: Vec<usize> = shape
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) if *n > 0 => Some(*n as usize),
            _ => None,
        })
        .collect();
    if static_dims.len() >= 2
        && static_dims.len() == shape.len()
        && static_dims.iter().product::<usize>() == values.len()
    {
        format_nd_slice(values, &static_dims, &|v| v.to_string())
    } else {
        let elems: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        format!("[{}]", elems.join(", "))
    }
}

/// The Weight Plane of CastForge's Dual-Plane IR.
/// Holds weight buffers or lazy external file references alongside the SSA Compute Graph.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStore {
    pub(crate) entries: IndexMap<String, TensorEntry>,
}

impl TensorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, entry: TensorEntry) {
        self.entries.insert(entry.name.clone(), entry);
    }

    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut TensorEntry> {
        self.entries.get_mut(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<TensorEntry> {
        self.entries.shift_remove(name)
    }

    pub fn rename_key(&mut self, old_name: &str, new_name: String) {
        if old_name == new_name {
            return;
        }
        if let Some(mut entry) = self.entries.shift_remove(old_name) {
            entry.name = new_name.clone();
            self.entries.insert(new_name, entry);
        }
    }

    /// Updates a constant's `f32` data in both its operation `Attribute` (`TensorRef` or
    /// `DenseElements`) and any corresponding `TensorStore` entry (keyed by `TensorRef` or
    /// `ssa_name`).
    pub fn update_f32_constant(
        &mut self,
        ssa_name: Option<&str>,
        old_attr: &Attribute,
        new_ty: TensorType,
        new_vals: Vec<f32>,
    ) -> Attribute {
        match old_attr {
            Attribute::TensorRef { key, .. } => {
                if let Some(entry) = self.entries.get_mut(key) {
                    entry.ty = new_ty.clone();
                    entry.backing = TensorBacking::F32Data(new_vals);
                }
                Attribute::TensorRef {
                    key: key.clone(),
                    ty: new_ty,
                }
            }
            _ => {
                let raw_literal = format_dense_f32_literal(&new_vals, &new_ty.shape);
                if let Some(name) = ssa_name
                    && let Some(entry) = self.entries.get_mut(name)
                {
                    entry.ty = new_ty.clone();
                    entry.backing = TensorBacking::F32Data(new_vals);
                }
                Attribute::DenseElements {
                    ty: new_ty,
                    raw_literal,
                }
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorEntry)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

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
                        Self::decode_raw_bytes_to_f32(bytes, &entry.ty.dtype, expected_len)
                    }
                    TensorBacking::ExternalFile {
                        path,
                        offset,
                        length,
                    } => {
                        let slice = Self::read_external_slice(path, *offset, *length).ok()?;
                        Self::decode_raw_bytes_to_f32(&slice, &entry.ty.dtype, expected_len)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn read_external_slice(
        path: &std::path::Path,
        offset: u64,
        length: u64,
    ) -> std::io::Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; length as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn decode_raw_bytes_to_f32(
        bytes: &[u8],
        dtype: &DType,
        expected_len: usize,
    ) -> Option<Vec<f32>> {
        if bytes.len() == expected_len * 4 {
            Some(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        } else if bytes.len() == expected_len * 2 {
            let is_bf16 = matches!(dtype, DType::BF16);
            Some(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        if is_bf16 {
                            bf16_bits_to_f32(bits)
                        } else {
                            f16_bits_to_f32(bits)
                        }
                    })
                    .collect(),
            )
        } else if bytes.len() == expected_len {
            if matches!(dtype, DType::U8) {
                Some(bytes.iter().map(|&b| b as f32).collect())
            } else {
                Some(bytes.iter().map(|&b| (b as i8) as f32).collect())
            }
        } else {
            None
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
    pub fn to_safetensors_bytes(&self) -> Result<Vec<u8>, std::io::Error> {
        use std::io;
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
                    let dtype_str = match entry.ty.dtype {
                        DType::F16 => {
                            for &v in vals {
                                data_buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                            }
                            "F16"
                        }
                        DType::BF16 => {
                            for &v in vals {
                                data_buf.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
                            }
                            "BF16"
                        }
                        _ => {
                            for &v in vals {
                                data_buf.extend_from_slice(&v.to_le_bytes());
                            }
                            "F32"
                        }
                    };
                    let end = data_buf.len();
                    headers.insert(
                        name.clone(),
                        SafeTensorEntryHeader {
                            dtype: dtype_str.to_string(),
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
                TensorBacking::ExternalFile {
                    path,
                    offset,
                    length,
                } => {
                    let slice = Self::read_external_slice(path, *offset, *length)?;
                    data_buf.extend_from_slice(&slice);
                    let dtype_str = match entry.ty.dtype {
                        DType::F16 => "F16",
                        DType::BF16 => "BF16",
                        DType::I8 => "I8",
                        DType::U8 => "U8",
                        _ => "F32",
                    };
                    let end = data_buf.len();
                    headers.insert(
                        name.clone(),
                        SafeTensorEntryHeader {
                            dtype: dtype_str.to_string(),
                            shape,
                            data_offsets: [start, end],
                            scale: None,
                            zero_point: None,
                        },
                    );
                }
                TensorBacking::Symbolic => {}
            }
        }

        let mut header_json = serde_json::to_vec(&headers)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
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

    /// Construct a `TensorStore` from a `.safetensors` byte slice.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> Result<Self, std::io::Error> {
        let mut store = Self::new();
        store.load_safetensors_bytes(bytes)?;
        Ok(store)
    }

    /// Load tensors from a `.safetensors` byte slice into this `TensorStore`.
    pub fn load_safetensors_bytes(&mut self, bytes: &[u8]) -> Result<usize, std::io::Error> {
        use std::io;
        const MAX_HEADER_LEN: usize = 100 * 1024 * 1024;
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Safetensors buffer shorter than 8-byte header length",
            ));
        }
        let header_len_u64 = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let header_len = usize::try_from(header_len_u64).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Safetensors header length overflows usize",
            )
        })?;
        if header_len > MAX_HEADER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Safetensors header length {header_len} exceeds 100 MiB limit"),
            ));
        }
        let data_start = 8usize.checked_add(header_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Safetensors header offset overflow",
            )
        })?;
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
            let mut shape = Vec::with_capacity(hdr.shape.len());
            let mut elem_prod = 1i64;
            for &d in &hdr.shape {
                let di = i64::try_from(d).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Dimension {d} overflows i64 for tensor `{name}`"),
                    )
                })?;
                elem_prod = elem_prod.checked_mul(di.max(1)).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Dimension product overflow for tensor `{name}`"),
                    )
                })?;
                if elem_prod > TensorType::MAX_TENSOR_ELEMENTS {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Tensor `{name}` exceeds max element limit"),
                    ));
                }
                shape.push(Dim::Static(di));
            }
            let slice = &body_slice[start..end];

            match hdr.dtype.as_str() {
                "F32" => {
                    let vals: Vec<f32> = slice
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let ty = TensorType::new(shape, DType::F32);
                    self.insert(TensorEntry::from_f32(name, ty, vals));
                    loaded += 1;
                }
                "F16" => {
                    let vals: Vec<f32> = slice
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    let ty = TensorType::new(shape, DType::F16);
                    self.insert(TensorEntry::from_f32(name, ty, vals));
                    loaded += 1;
                }
                "BF16" => {
                    let vals: Vec<f32> = slice
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect();
                    let ty = TensorType::new(shape, DType::BF16);
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
    pub fn save_safetensors_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), std::io::Error> {
        let bytes = self.to_safetensors_bytes()?;
        std::fs::write(path, bytes)
    }

    /// Load a `.safetensors` file from disk into this `TensorStore`.
    pub fn load_safetensors_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<usize, std::io::Error> {
        let bytes = std::fs::read(path)?;
        self.load_safetensors_bytes(&bytes)
    }

    /// Lazily index a `.safetensors` file on disk by reading ONLY its JSON header and registering
    /// `TensorBacking::ExternalFile` slice descriptors without loading weight payloads into RAM.
    pub fn load_safetensors_lazy_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<usize, std::io::Error> {
        use std::io::{self, Read};
        let path_ref = path.as_ref();
        let mut file = std::fs::File::open(path_ref)?;
        let file_len = file.metadata()?.len();
        if file_len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Safetensors file shorter than 8-byte header",
            ));
        }
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf);
        if header_len > 100 * 1024 * 1024 || 8 + header_len > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid or truncated Safetensors header length",
            ));
        }
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)?;
        let data_base = 8u64 + header_len;
        let raw_map: IndexMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut count = 0usize;
        for (name, val) in raw_map {
            if name == "__metadata__" {
                continue;
            }
            let hdr: SafeTensorEntryHeader = serde_json::from_value(val)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let [start, end] = hdr.data_offsets;
            let abs_start = data_base.saturating_add(start as u64);
            let abs_end = data_base.saturating_add(end as u64);
            if start > end || abs_end > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("Out-of-bounds data_offsets for tensor `{name}`"),
                ));
            }
            let dtype = match hdr.dtype.as_str() {
                "F16" => DType::F16,
                "BF16" => DType::BF16,
                "I8" => DType::I8,
                "U8" => DType::U8,
                _ => DType::F32,
            };
            let shape: Vec<Dim> = hdr.shape.iter().map(|&d| Dim::Static(d as i64)).collect();
            self.insert(TensorEntry {
                name,
                ty: TensorType::new(shape, dtype),
                backing: TensorBacking::ExternalFile {
                    path: path_ref.to_path_buf(),
                    offset: abs_start,
                    length: (end - start) as u64,
                },
            });
            count += 1;
        }
        Ok(count)
    }
}
