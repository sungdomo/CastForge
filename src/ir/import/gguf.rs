//! Binary GGUF v2/v3 importer and exporter (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`).

use super::{collect_module_tensors, fmt_shape, sanitize_name};
use crate::ir::{
    DType, Dim, Module, TensorBacking, TensorEntry, TensorStore, TensorType, f32_to_bf16_bits,
    f32_to_f16_bits, format_dense_f32_literal, parse_mlir,
};
use anyhow::{Context, Result, anyhow, bail};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor element encoding supported by the GGUF reader/writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    BF16 = 30,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            30 => Ok(Self::BF16),
            other => bail!("Unsupported GGML tensor type id: {other}"),
        }
    }
}

#[derive(Debug, Clone)]
struct GgufTensorInfo {
    name: String,
    shape: Vec<i64>,
    ggml_type: GgmlType,
    offset: usize,
}

/// Imports a binary `.gguf` file from disk into a CastForge [`Module`] + [`TensorStore`].
pub fn import_gguf_file(path: impl AsRef<Path>) -> Result<Module> {
    let bytes = fs::read(path.as_ref())
        .with_context(|| format!("Failed to read GGUF file {}", path.as_ref().display()))?;
    import_gguf_bytes(&bytes)
}

/// Parses a binary GGUF v2/v3 byte buffer, loads all tensors into `TensorStore`, and constructs
/// a StableHLO forward graph over the imported weights.
pub fn import_gguf_bytes(bytes: &[u8]) -> Result<Module> {
    if bytes.len() < 24 || &bytes[0..4] != GGUF_MAGIC {
        bail!("Invalid GGUF header magic (expected 'GGUF')");
    }
    let mut cursor = 4usize;
    let version = read_u32(bytes, &mut cursor)?;
    if !(2..=3).contains(&version) {
        bail!("Unsupported GGUF version: {version}");
    }
    let tensor_count = read_u64(bytes, &mut cursor)? as usize;
    let kv_count = read_u64(bytes, &mut cursor)? as usize;

    let mut model_name = "gguf_model".to_string();
    let mut alignment = GGUF_DEFAULT_ALIGNMENT;
    for _ in 0..kv_count {
        let key = read_gguf_string(bytes, &mut cursor)?;
        let val_ty = read_u32(bytes, &mut cursor)?;
        let str_val = skip_or_read_gguf_val(bytes, &mut cursor, val_ty)?;
        if key == "general.name"
            && let Some(s) = str_val
        {
            model_name = sanitize_name(&s);
        } else if key == "general.alignment"
            && let Some(s) = str_val
            && let Ok(a) = s.parse::<usize>()
            && a.is_power_of_two()
        {
            alignment = a;
        }
    }

    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(bytes, &mut cursor)?;
        let n_dims = read_u32(bytes, &mut cursor)? as usize;
        let mut raw_dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            raw_dims.push(read_u64(bytes, &mut cursor)? as i64);
        }
        // GGUF stores dimensions inner-most first; reverse into row-major [M, N]
        raw_dims.reverse();
        if raw_dims.is_empty() {
            raw_dims.push(1);
        }
        let ggml_type = GgmlType::from_u32(read_u32(bytes, &mut cursor)?)?;
        let offset = read_u64(bytes, &mut cursor)? as usize;
        infos.push(GgufTensorInfo {
            name,
            shape: raw_dims,
            ggml_type,
            offset,
        });
    }

    let data_base = cursor.div_ceil(alignment) * alignment;
    let mut decoded_tensors: Vec<(GgufTensorInfo, Vec<f32>)> = Vec::with_capacity(infos.len());
    for info in infos {
        let num_elems: usize = info
            .shape
            .iter()
            .map(|&d| d.max(1) as usize)
            .product::<usize>()
            .max(1);
        let start = data_base + info.offset;
        let vals = decode_ggml_tensor(bytes, start, num_elems, info.ggml_type)?;
        decoded_tensors.push((info, vals));
    }

    // Synthesize a valid StableHLO forward module referencing the GGUF tensors
    let first_k = decoded_tensors
        .first()
        .and_then(|(info, _)| info.shape.first().copied())
        .unwrap_or(4);
    let mut mlir = String::new();
    writeln!(mlir, "module @{model_name} {{")?;
    if decoded_tensors.is_empty() {
        writeln!(
            mlir,
            "  func.func @forward(%x: tensor<1x4xf32>) -> tensor<1x4xf32> {{"
        )?;
        writeln!(mlir, "    func.return %x : tensor<1x4xf32>")?;
        writeln!(mlir, "  }}")?;
    } else {
        let cur_rows = 1i64;
        let mut cur_cols = first_k;
        let mut cur_ssa = "%x".to_string();
        let mut body = String::new();
        for (idx, (info, vals)) in decoded_tensors.iter().enumerate() {
            let w_ssa = format!("%w{idx}");
            let shape_str = fmt_shape(&info.shape);
            let dims: Vec<Dim> = info.shape.iter().map(|&d| Dim::Static(d)).collect();
            let lit = format_dense_f32_literal(vals, &dims);
            writeln!(
                body,
                "    {w_ssa} = stablehlo.constant dense<{lit}> : tensor<{shape_str}xf32>"
            )?;
            if info.shape.len() == 2 && info.shape[0] == cur_cols {
                let next_cols = info.shape[1];
                let out_ssa = format!("%h{idx}");
                writeln!(
                    body,
                    "    {out_ssa} = stablehlo.dot_general {cur_ssa}, {w_ssa}, \
                     contracting_dims = [1] x [0] : \
                     (tensor<{cur_rows}x{cur_cols}xf32>, tensor<{cur_cols}x{next_cols}xf32>) \
                     -> tensor<{cur_rows}x{next_cols}xf32>"
                )?;
                cur_cols = next_cols;
                cur_ssa = out_ssa;
            } else if info.shape.len() == 2
                && info.shape[0] == cur_rows
                && info.shape[1] == cur_cols
            {
                let out_ssa = format!("%h{idx}");
                writeln!(
                    body,
                    "    {out_ssa} = stablehlo.add {cur_ssa}, {w_ssa} \
                     : tensor<{cur_rows}x{cur_cols}xf32>"
                )?;
                cur_ssa = out_ssa;
            }
        }
        writeln!(
            mlir,
            "  func.func @forward(%x: tensor<1x{first_k}xf32>) \
             -> tensor<{cur_rows}x{cur_cols}xf32> {{"
        )?;
        mlir.push_str(&body);
        writeln!(
            mlir,
            "    func.return {cur_ssa} : tensor<{cur_rows}x{cur_cols}xf32>"
        )?;
        writeln!(mlir, "  }}")?;
    }
    writeln!(mlir, "}}")?;

    let mut module = parse_mlir(&mlir).context("Failed to parse synthesized GGUF MLIR")?;
    module.tensor_store = TensorStore::new();
    for (info, vals) in decoded_tensors {
        let dims: Vec<Dim> = info.shape.iter().map(|&d| Dim::Static(d)).collect();
        module.tensor_store.insert(TensorEntry::from_f32(
            info.name,
            TensorType::new(dims, DType::F32),
            vals,
        ));
    }
    Ok(module)
}

/// Serializes all tensors in `module.tensor_store` (or inline constants) into a valid binary
/// GGUF v3 byte buffer.
pub fn export_gguf_bytes(module: &Module) -> Vec<u8> {
    let store = collect_module_tensors(module);
    let mut out = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    out.extend_from_slice(&3u32.to_le_bytes()); // version 3
    let tensor_count = store.len() as u64;
    out.extend_from_slice(&tensor_count.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata KV: general.name

    let mod_name = module.name.as_deref().unwrap_or("castforge_model");
    write_gguf_string(&mut out, "general.name");
    out.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    write_gguf_string(&mut out, mod_name);

    let mut raw_payloads: Vec<Vec<u8>> = Vec::with_capacity(store.len());
    let mut current_offset = 0usize;

    for (name, tensor) in store.iter() {
        write_gguf_string(&mut out, name);
        let static_dims: Vec<u64> = tensor
            .ty
            .shape
            .iter()
            .map(|d| match d {
                Dim::Static(n) if *n > 0 => *n as u64,
                _ => 1,
            })
            .collect();
        let n_dims = static_dims.len().max(1) as u32;
        out.extend_from_slice(&n_dims.to_le_bytes());
        // GGUF stores dimensions inner-most first
        for &d in static_dims.iter().rev() {
            out.extend_from_slice(&d.to_le_bytes());
        }
        let (ggml_ty, mut bytes) = match &tensor.backing {
            TensorBacking::QuantizedI8 { data, scale, .. } => {
                // Q8_0 block: 2-byte f16 scale + 32 x i8 quants = 34 bytes per 32 elements
                let blocks = data.len().max(1).div_ceil(32);
                let mut b = Vec::with_capacity(blocks * 34);
                let sc_bits = f32_to_f16_bits(*scale).to_le_bytes();
                for blk_idx in 0..blocks {
                    b.extend_from_slice(&sc_bits);
                    for i in 0..32 {
                        let q = data.get(blk_idx * 32 + i).copied().unwrap_or(0);
                        b.push(q as u8);
                    }
                }
                (GgmlType::Q8_0, b)
            }
            _ => {
                let f32_vals = tensor
                    .as_f32_slice()
                    .map(|s| s.to_vec())
                    .unwrap_or_else(|| vec![0.0; 1]);
                match tensor.ty.dtype {
                    DType::F16 => {
                        let mut b = Vec::with_capacity(f32_vals.len() * 2);
                        for v in f32_vals {
                            b.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                        }
                        (GgmlType::F16, b)
                    }
                    DType::BF16 => {
                        let mut b = Vec::with_capacity(f32_vals.len() * 2);
                        for v in f32_vals {
                            b.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
                        }
                        (GgmlType::BF16, b)
                    }
                    _ => {
                        let mut b = Vec::with_capacity(f32_vals.len() * 4);
                        for v in f32_vals {
                            b.extend_from_slice(&v.to_le_bytes());
                        }
                        (GgmlType::F32, b)
                    }
                }
            }
        };
        out.extend_from_slice(&(ggml_ty as u32).to_le_bytes());
        out.extend_from_slice(&(current_offset as u64).to_le_bytes());
        let padded_len = bytes.len().div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
        bytes.resize(padded_len, 0);
        current_offset += bytes.len();
        raw_payloads.push(bytes);
    }

    let aligned_header = out.len().div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
    out.resize(aligned_header, 0);
    for payload in raw_payloads {
        out.extend_from_slice(&payload);
    }
    out
}

/// Exports a CastForge [`Module`] to a binary GGUF v3 file on disk.
pub fn export_gguf_file(module: &Module, path: impl AsRef<Path>) -> Result<()> {
    let bytes = export_gguf_bytes(module);
    fs::write(path.as_ref(), bytes)
        .with_context(|| format!("Failed to write GGUF file {}", path.as_ref().display()))
}

/// A-to-B Model Export Translator (`Ir(Module) -> Bytes(GGUF v3)`).
#[derive(Default)]
pub struct GgufExportTranslator;

impl crate::ir::Translator for GgufExportTranslator {
    fn name(&self) -> &str {
        "export-gguf"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        Ok(crate::ir::Payload::Bytes(export_gguf_bytes(&module)))
    }
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let end = *cursor + 4;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("Unexpected EOF reading u32"))?;
    *cursor = end;
    Ok(u32::from_le_bytes(slice.try_into()?))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = *cursor + 8;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("Unexpected EOF reading u64"))?;
    *cursor = end;
    Ok(u64::from_le_bytes(slice.try_into()?))
}

fn read_gguf_string(bytes: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u64(bytes, cursor)? as usize;
    let end = *cursor + len;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("Unexpected EOF reading GGUF string"))?;
    *cursor = end;
    Ok(String::from_utf8_lossy(slice).into_owned())
}

fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn skip_or_read_gguf_val(bytes: &[u8], cursor: &mut usize, val_ty: u32) -> Result<Option<String>> {
    match val_ty {
        0 | 1 | 7 => {
            *cursor += 1;
            Ok(None)
        }
        2 | 3 => {
            *cursor += 2;
            Ok(None)
        }
        4..=6 => {
            let v = read_u32(bytes, cursor)?;
            Ok(Some(v.to_string()))
        }
        8 => Ok(Some(read_gguf_string(bytes, cursor)?)),
        9 => {
            let elem_ty = read_u32(bytes, cursor)?;
            let len = read_u64(bytes, cursor)? as usize;
            for _ in 0..len {
                let _ = skip_or_read_gguf_val(bytes, cursor, elem_ty)?;
            }
            Ok(None)
        }
        10..=12 => {
            let v = read_u64(bytes, cursor)?;
            Ok(Some(v.to_string()))
        }
        other => bail!("Unsupported GGUF metadata value type: {other}"),
    }
}

fn decode_ggml_tensor(
    bytes: &[u8],
    start: usize,
    num_elems: usize,
    ggml_type: GgmlType,
) -> Result<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => {
            let byte_len = num_elems * 4;
            let raw = bytes
                .get(start..start + byte_len)
                .ok_or_else(|| anyhow!("GGUF F32 tensor data out of bounds"))?;
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect())
        }
        GgmlType::F16 => {
            let byte_len = num_elems * 2;
            let raw = bytes
                .get(start..start + byte_len)
                .ok_or_else(|| anyhow!("GGUF F16 tensor data out of bounds"))?;
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f16_to_f32(u16::from_le_bytes(c)))
                .collect())
        }
        GgmlType::BF16 => {
            let byte_len = num_elems * 2;
            let raw = bytes
                .get(start..start + byte_len)
                .ok_or_else(|| anyhow!("GGUF BF16 tensor data out of bounds"))?;
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f32::from_bits((u16::from_le_bytes(c) as u32) << 16))
                .collect())
        }
        GgmlType::Q4_0 => {
            // Q4_0 block: 2-byte f16 scale + 16 packed nibble-pair bytes = 18 bytes per 32 elems
            let blocks = num_elems.div_ceil(32);
            let raw = bytes
                .get(start..start + blocks * 18)
                .ok_or_else(|| anyhow!("GGUF Q4_0 tensor data out of bounds"))?;
            let mut out = Vec::with_capacity(num_elems);
            for blk in raw.as_chunks::<18>().0 {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &b in &blk[2..18] {
                    if out.len() < num_elems {
                        let lo = (b & 0x0f) as i32 - 8;
                        out.push((lo as f32) * d);
                    }
                    if out.len() < num_elems {
                        let hi = ((b >> 4) & 0x0f) as i32 - 8;
                        out.push((hi as f32) * d);
                    }
                }
            }
            Ok(out)
        }
        GgmlType::Q8_0 => {
            // Q8_0 block: 2-byte f16 scale + 32 x i8 quants = 34 bytes per 32 elements
            let blocks = num_elems.div_ceil(32);
            let raw = bytes
                .get(start..start + blocks * 34)
                .ok_or_else(|| anyhow!("GGUF Q8_0 tensor data out of bounds"))?;
            let mut out = Vec::with_capacity(num_elems);
            for blk in raw.as_chunks::<34>().0 {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    if out.len() < num_elems {
                        out.push((q as i8 as f32) * d);
                    }
                }
            }
            let _ = DType::F32;
            Ok(out)
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
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
