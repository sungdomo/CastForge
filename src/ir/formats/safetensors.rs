//! SafeTensors binary model weight importer and exporter (`SafetensorsImportTranslator`,
//! `SafetensorsExportTranslator`).

use crate::ir::{DType, Module, TensorBacking, TensorEntry, TensorStore, TensorType};
use anyhow::Result;

/// Collects all weight tensors from `module.tensor_store`, falling back to inline
/// `stablehlo.constant` operations when `tensor_store` is empty.
pub fn collect_module_tensors(module: &Module) -> TensorStore {
    if !module.tensor_store.is_empty() {
        return module.tensor_store.clone();
    }
    let mut store = TensorStore::new();
    let mut idx = 0usize;
    for op_id in module.live_ops() {
        let op = &module.ops[op_id];
        if !op.is_op("stablehlo", "constant") {
            continue;
        }
        let Some(&res_val) = op.results.first() else {
            continue;
        };
        let Some(tensor_ty) = module.values[res_val].ty.as_tensor().cloned() else {
            continue;
        };
        let n = tensor_ty.num_elements().unwrap_or(1).max(1) as usize;
        let name = module.values[res_val]
            .ssa_name
            .as_deref()
            .map(|s| s.trim_start_matches('%').to_string())
            .unwrap_or_else(|| {
                idx += 1;
                format!("const_{idx}.weight")
            });
        if let DType::QuantUniform {
            scale, zero_point, ..
        } = &tensor_ty.dtype
        {
            let sc = scale.parse::<f32>().unwrap_or(1.0);
            let zp = *zero_point;
            let q_vals = op
                .attributes
                .get("value")
                .and_then(|a| module.tensor_store.resolve_i8_attr(a, n))
                .unwrap_or_else(|| vec![0; n]);
            store.insert(TensorEntry {
                name,
                ty: tensor_ty,
                backing: TensorBacking::QuantizedI8 {
                    data: q_vals,
                    scale: sc,
                    zero_point: zp,
                },
            });
            continue;
        }
        let vals = op
            .attributes
            .get("value")
            .and_then(|a| module.tensor_store.resolve_f32_attr(a, n))
            .unwrap_or_else(|| vec![0.0; n]);
        let keep_dtype = match tensor_ty.dtype {
            DType::F16 => DType::F16,
            DType::BF16 => DType::BF16,
            _ => DType::F32,
        };
        store.insert(TensorEntry::from_f32(
            name,
            TensorType::new(tensor_ty.shape, keep_dtype),
            vals,
        ));
    }
    store
}

/// A-to-B Weight Export Translator (`Ir(Module) -> Bytes(SafeTensors)`).
#[derive(Default)]
pub struct SafetensorsExportTranslator;

impl crate::ir::Translator for SafetensorsExportTranslator {
    fn name(&self) -> &str {
        "export-safetensors"
    }

    fn input_kind(&self) -> &'static str {
        "ir"
    }

    fn output_kind(&self) -> &'static str {
        "bytes"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let module = input.into_ir()?;
        let store = collect_module_tensors(&module);
        let bytes = store.to_safetensors_bytes()?;
        Ok(crate::ir::Payload::Bytes(bytes))
    }
}

/// Imports a `.safetensors` binary buffer into a CastForge [`Module`] with a populated
/// [`TensorStore`] and `dense_resource<...>` constants.
pub fn import_safetensors_bytes(bytes: &[u8]) -> Result<Module> {
    use crate::ir::{Attribute, Location, OpName, Type};
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    let store = TensorStore::from_safetensors_bytes(bytes)?;
    let mut module = Module::new(Some("safetensors_model".to_string()));

    let fn_blk = module.create_block(None);
    let fn_reg = module.create_region(vec![fn_blk]);

    let mut return_vals = SmallVec::new();
    let mut return_tys = Vec::new();

    for (idx, (name, entry)) in store.iter().enumerate() {
        let mut attrs = IndexMap::new();
        attrs.insert(
            "value".to_string(),
            Attribute::TensorRef {
                key: name.clone(),
                ty: entry.ty.clone(),
            },
        );
        let ssa_name = sanitize_name(name);
        let ssa = if ssa_name.is_empty() {
            format!("w{idx}")
        } else {
            ssa_name
        };
        let res_ty = Type::Tensor(entry.ty.clone());
        let op_id = module.add_operation(
            fn_blk,
            OpName::new("stablehlo", "constant"),
            SmallVec::new(),
            vec![(Some(ssa), res_ty.clone())],
            attrs,
            Vec::new(),
            true,
            Location::Unknown,
        );
        return_vals.push(module.ops[op_id].results[0]);
        return_tys.push(res_ty);
    }

    module.add_operation(
        fn_blk,
        OpName::new("func", "return"),
        return_vals,
        Vec::new(),
        IndexMap::new(),
        Vec::new(),
        true,
        Location::Unknown,
    );

    let mut fn_attrs = IndexMap::new();
    fn_attrs.insert(
        "sym_name".to_string(),
        Attribute::String("weights".to_string()),
    );
    fn_attrs.insert(
        "function_type".to_string(),
        Attribute::Type(Type::Function {
            inputs: Vec::new(),
            results: return_tys,
        }),
    );
    let root_blk = module.body_block();
    module.add_operation(
        root_blk,
        OpName::new("func", "func"),
        SmallVec::new(),
        Vec::new(),
        fn_attrs,
        vec![fn_reg],
        true,
        Location::Unknown,
    );
    module.tensor_store = store;
    Ok(module)
}

/// Imports a `.safetensors` file from disk into a CastForge [`Module`].
pub fn import_safetensors_file(path: impl AsRef<std::path::Path>) -> Result<Module> {
    let bytes = std::fs::read(path.as_ref())?;
    import_safetensors_bytes(&bytes)
}

/// A-to-B Weight Import Translator (`Bytes(SafeTensors) -> Ir(Module)`).
#[derive(Default)]
pub struct SafetensorsImportTranslator;

impl crate::ir::Translator for SafetensorsImportTranslator {
    fn name(&self) -> &str {
        "import-safetensors"
    }

    fn input_kind(&self) -> &'static str {
        "bytes"
    }

    fn output_kind(&self) -> &'static str {
        "ir"
    }

    fn translate(&mut self, input: crate::ir::Payload) -> Result<crate::ir::Payload> {
        let bytes = input.into_bytes()?;
        let module = import_safetensors_bytes(&bytes)?;
        Ok(crate::ir::Payload::from_ir(module))
    }
}

pub(crate) fn fmt_shape(shape: &[i64]) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

pub(crate) fn sanitize_name(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
