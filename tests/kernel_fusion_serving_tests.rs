//! Integration tests for model callable outlining, kernel body outlining, mutable load fusion,
//! tenant quota management, and device-graph verification passes.
#![cfg(feature = "serving")]

use castforge::ir::{DType, Location, parse_mlir};

#[test]
fn test_outline_models_to_callables_splits_init_exec_and_propagates_prefixes() {
    use castforge::ir::{
        Attribute, Block, DType, Dim, Location, Module, OpName, Operation, Region, TensorType,
        Type, Value, ValueDef,
    };
    use castforge::passes::outline_models_to_callables;
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    let mut m = Module::new(Some("outline_test".to_string()));
    let root_blk = m.regions[m.root_region].blocks[0];
    let f32_ty = Type::Tensor(TensorType::new(vec![Dim::Static(4)], DType::F32));
    let ch_ty = Type::Opaque("!mef.chain".to_string());
    let str_ty = Type::Opaque("!mgp.opaque<\"mojo_string\">".to_string());

    // Build subgraph `sub_layer` (`is_subgraph = true`, `implicit_args_count = 1`) with an init
    // region containing a placeholder `mgp.buffer.constant_external` and an execute region:
    let sub_init_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let sub_init_ch = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: ch_ty.clone(),
        def: ValueDef::BlockArgument {
            block: sub_init_blk,
            index: 0,
        },
    });
    m.blocks[sub_init_blk].arguments.push(sub_init_ch);
    let raw_name_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mgp", "string.create"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    let raw_name_val = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: str_ty,
        def: ValueDef::OpResult {
            op: raw_name_op,
            index: 0,
        },
    });
    m.ops[raw_name_op].results.push(raw_name_val);
    let ext_const_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mgp", "buffer.constant_external"),
        operands: SmallVec::from_slice(&[raw_name_val]),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert("e_is_placeholder".to_string(), Attribute::Bool(true));
            a
        },
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    let ext_w_val = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::OpResult {
            op: ext_const_op,
            index: 0,
        },
    });
    m.ops[ext_const_op].results.push(ext_w_val);
    let sub_init_out = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mef", "output"),
        operands: SmallVec::from_slice(&[sub_init_ch, ext_w_val]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[sub_init_blk].ops = vec![raw_name_op, ext_const_op, sub_init_out];
    let sub_init_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![sub_init_blk],
    });

    let sub_exec_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let sub_exec_w = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::BlockArgument {
            block: sub_exec_blk,
            index: 0,
        },
    });
    let sub_exec_x = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::BlockArgument {
            block: sub_exec_blk,
            index: 1,
        },
    });
    m.blocks[sub_exec_blk].arguments = vec![sub_exec_w, sub_exec_x];
    let sub_exec_out = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mef", "output"),
        operands: SmallVec::from_slice(&[sub_exec_x]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[sub_exec_blk].ops = vec![sub_exec_out];
    let sub_exec_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![sub_exec_blk],
    });

    let sub_model = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mgp", "model"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "sym_name".to_string(),
                Attribute::String("sub_layer".to_string()),
            );
            a.insert("is_subgraph".to_string(), Attribute::Bool(true));
            a.insert("implicit_args_count".to_string(), Attribute::Int(1));
            a
        },
        regions: vec![sub_init_reg, sub_exec_reg],
        custom_syntax: false,
        location: Location::Unknown,
    });

    // Build entry model `main_model` (`is_subgraph = false`) calling `@sub_layer` with prefix:
    let main_init_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let main_init_ch = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: ch_ty,
        def: ValueDef::BlockArgument {
            block: main_init_blk,
            index: 0,
        },
    });
    m.blocks[main_init_blk].arguments.push(main_init_ch);
    let main_init_out = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mef", "output"),
        operands: SmallVec::from_slice(&[main_init_ch]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[main_init_blk].ops = vec![main_init_out];
    let main_init_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![main_init_blk],
    });

    let main_exec_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let main_exec_x = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::BlockArgument {
            block: main_exec_blk,
            index: 0,
        },
    });
    m.blocks[main_exec_blk].arguments.push(main_exec_x);
    let call_sub = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mgp", "call"),
        operands: SmallVec::from_slice(&[main_exec_x, main_exec_x, main_exec_x]),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "callee".to_string(),
                Attribute::SymbolRef("sub_layer".to_string()),
            );
            a.insert(
                "prefix".to_string(),
                Attribute::String("blocks.0.".to_string()),
            );
            a
        },
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    let call_res = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty,
        def: ValueDef::OpResult {
            op: call_sub,
            index: 0,
        },
    });
    m.ops[call_sub].results.push(call_res);
    let main_exec_out = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mef", "output"),
        operands: SmallVec::from_slice(&[call_res]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[main_exec_blk].ops = vec![call_sub, main_exec_out];
    let main_exec_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![main_exec_blk],
    });

    let main_model = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("mgp", "model"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "sym_name".to_string(),
                Attribute::String("main_model".to_string()),
            );
            a.insert("is_subgraph".to_string(), Attribute::Bool(false));
            a
        },
        regions: vec![main_init_reg, main_exec_reg],
        custom_syntax: false,
        location: Location::Unknown,
    });

    m.blocks[root_blk].ops = vec![sub_model, main_model];

    let outlined = outline_models_to_callables(&mut m).expect("outline succeeds");
    assert_eq!(outlined, 2);

    // Root block now holds: sub_layer$init, sub_layer$exec, main_model$init, main_model$exec,
    // and the wrapper mef.model for main_model (5 ops total):
    let top_ops = &m.blocks[root_blk].ops;
    assert_eq!(top_ops.len(), 5);
    assert!(m.ops[top_ops[0]].is_op("mef", "callable"));
    assert_eq!(
        m.ops[top_ops[0]]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str()),
        Some("sub_layer$init")
    );
    assert!(m.ops[top_ops[4]].is_op("mef", "model"));
    assert!(m.ops[call_sub].is_op("grt", "invoke"));
}

#[test]
fn test_asym_k16_v8_multi_serde_storage_only_and_split_tier_v_only() {
    use castforge::passes::{
        ASYM_K16_V8_ESTIMATE_ALLOWANCE_BYTES, AsymK16V8MultiSerde, AsymK16V8SerdeMode, KvScaleScope,
    };

    let k_data = vec![0.5f32, -1.0, 2.0, -0.25, 1.5, -2.5, 0.0, 3.0];
    let v_data = vec![0.25f32, -0.5, 1.0, -1.25, 0.75, -0.75, 0.0, 1.27];

    // 1. StorageOnlyDequant mode:
    let storage_serde = AsymK16V8MultiSerde::new(
        AsymK16V8SerdeMode::StorageOnlyDequant,
        KvScaleScope::PerTensor,
        1,
    );
    assert_eq!(storage_serde.group_size(), 2);
    assert_eq!(
        storage_serde
            .estimate_serialized_size(Some(8), Some(8))
            .expect("estimate ok"),
        ASYM_K16_V8_ESTIMATE_ALLOWANCE_BYTES + 16 + 8 + 4
    );
    assert!(
        storage_serde
            .estimate_serialized_size(None, Some(8))
            .is_err()
    );

    let blob_kv = storage_serde
        .serialize(Some(&k_data), Some(&v_data), 1, 2, 2, 2)
        .expect("serialize (K, V)");
    let (k_dec, v_dec) = storage_serde
        .deserialize(&blob_kv, true, true)
        .expect("deserialize (K, V)");
    let k_unwrapped = k_dec.expect("K present");
    let v_unwrapped = v_dec.expect("V present");
    assert_eq!(k_unwrapped, k_data);
    for (orig, rec) in v_data.iter().zip(v_unwrapped.iter()) {
        assert!((orig - rec).abs() <= 0.02);
    }

    // Selective V-only read from a full (K, V) blob skips K:
    let (k_skip, v_only) = storage_serde
        .deserialize(&blob_kv, false, true)
        .expect("selective V-only read");
    assert!(k_skip.is_none());
    assert!(v_only.is_some());

    // 2. SplitTierVOnly mode: K must be None on serialize, and k_payload_len == 0:
    let v_only_serde = AsymK16V8MultiSerde::new(
        AsymK16V8SerdeMode::SplitTierVOnly,
        KvScaleScope::PerTensor,
        2,
    );
    assert!(
        v_only_serde
            .serialize(Some(&k_data), Some(&v_data), 1, 2, 2, 2)
            .is_err()
    );
    // Rejects decoding a full (K, V) blob in SplitTierVOnly mode:
    assert!(v_only_serde.deserialize(&blob_kv, false, true).is_err());

    assert_eq!(
        v_only_serde
            .estimate_serialized_size(None, Some(8))
            .expect("split-tier estimate"),
        ASYM_K16_V8_ESTIMATE_ALLOWANCE_BYTES + 8 + 4
    );
    let blob_v = v_only_serde
        .serialize(None, Some(&v_data), 1, 2, 2, 2)
        .expect("serialize (None, V)");
    let (k_out_vmode, v_out_vmode) = v_only_serde
        .deserialize(&blob_v, true, true)
        .expect("deserialize split-tier");
    assert!(k_out_vmode.is_none());
    assert_eq!(v_out_vmode.expect("V present").len(), 8);
}

#[test]
fn test_conform_call_operands_scalar_variadic_trait_and_unfused_epilogue_bypass() {
    use castforge::ir::{Attribute, Location, OpName, Operation};
    use castforge::passes::conform_call_operands;
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    let mlir_src = concat!(
        "module {\n",
        "  func.func @kernel_stub(%x: tensor<4xf32>, %y: tensor<4xf32>, %s: tensor<1xi64>)",
        " -> tensor<4xf32> {\n",
        "    %mut_in = \"castforge.tensor.create\"(%x) {mut = true}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %ep = \"castforge.tensor.create.epilogue\"(%y) {recomputes_value = false}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %res = \"castforge.call.opaque\"(%mut_in, %x, %s, %ep)",
        " {callee = @my_kernel, has_epilogue_fusion = false}",
        " : (tensor<4xf32>, tensor<4xf32>, tensor<1xi64>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %res : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let root_blk = m.regions[m.root_region].blocks[0];
    let func_id = m.functions()[0];
    m.ops[func_id].attributes.insert(
        "device_ref".to_string(),
        Attribute::String("gpu:0".to_string()),
    );

    // Insert top-level `castforge.kernel.decl` with `supports_unfused_execute = true` and
    // param_specs = ["trait:TileTensorable", "variadic:plain", "scalar", "tensor"]:
    let decl_id = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("castforge", "kernel.decl"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "sym_name".to_string(),
                Attribute::String("my_kernel".to_string()),
            );
            a.insert(
                "supports_unfused_execute".to_string(),
                Attribute::Bool(true),
            );
            a.insert(
                "param_specs".to_string(),
                Attribute::Array(vec![
                    Attribute::String("trait:TileTensorable".to_string()),
                    Attribute::String("variadic:plain".to_string()),
                    Attribute::String("scalar".to_string()),
                    Attribute::String("tensor".to_string()),
                ]),
            );
            a
        },
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[root_blk].ops.insert(0, decl_id);

    let conformed = conform_call_operands(&mut m).expect("conform succeeds");
    assert_eq!(conformed, 1);

    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    // The pass-through `tensor.create.epilogue` was bypassed and erased, and the call is preceded
    // by: original `tensor.create(mut=true)`, cloned `tensor.create(mut=false)`,
    // `trait.create`, `tuple.create`, and `scalar.create`:
    assert!(
        !ops.iter()
            .any(|&id| m.ops[id].is_op("castforge", "tensor.create.epilogue"))
    );
    assert!(
        ops.iter()
            .any(|&id| m.ops[id].is_op("castforge", "trait.create"))
    );
    assert!(
        ops.iter()
            .any(|&id| m.ops[id].is_op("castforge", "tuple.create"))
    );
    assert!(
        ops.iter()
            .any(|&id| m.ops[id].is_op("castforge", "scalar.create"))
    );
}

#[test]
fn test_outline_routines_and_outline_async_routines() {
    use castforge::ir::{
        Attribute, Block, DType, Dim, Location, OpName, Operation, Region, TensorType, Type, Value,
        ValueDef,
    };
    use castforge::passes::{outline_async_routines, outline_routines};
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    let mlir_src = concat!(
        "module {\n",
        "  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %r = \"castforge.routine\"(%x) {fusion_info = \"relu_add\", device_ref = \"gpu:0\"}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %a = \"castforge.routine.async\"(%r)",
        " {summary = \"async_norm\", dispatch = \"deferred\", device_ref = \"gpu:0\"}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %a : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let main_id = m.functions()[0];
    let main_blk = m.regions[m.ops[main_id].regions[0]].blocks[0];
    let routine_op = m.blocks[main_blk].ops[0];
    let async_op = m.blocks[main_blk].ops[1];
    let f32_ty = Type::Tensor(TensorType::new(vec![Dim::Static(4)], DType::F32));

    // Populate `routine` body with block arg `%in0`, a `castforge.buffer.constant`, and an add op:
    let r_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let r_arg = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::BlockArgument {
            block: r_blk,
            index: 0,
        },
    });
    m.blocks[r_blk].arguments.push(r_arg);
    let buf_cst = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("castforge", "buffer.constant"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    let cst_val = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::OpResult {
            op: buf_cst,
            index: 0,
        },
    });
    m.ops[buf_cst].results.push(cst_val);
    let add_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("stablehlo", "add"),
        operands: SmallVec::from_slice(&[r_arg, cst_val]),
        results: SmallVec::new(),
        attributes: IndexMap::new(),
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    let add_val = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty.clone(),
        def: ValueDef::OpResult {
            op: add_op,
            index: 0,
        },
    });
    m.ops[add_op].results.push(add_val);
    m.blocks[r_blk].ops = vec![buf_cst, add_op];
    let r_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![r_blk],
    });
    m.ops[routine_op].regions = vec![r_reg];

    // Populate `routine.async` body with a call to `@__relu_add`:
    let a_blk = m.blocks.insert_with_key(|id| Block {
        id,
        label: None,
        arguments: Vec::new(),
        ops: Vec::new(),
    });
    let a_arg = m.values.insert_with_key(|id| Value {
        id,
        ssa_name: None,
        ty: f32_ty,
        def: ValueDef::BlockArgument {
            block: a_blk,
            index: 0,
        },
    });
    m.blocks[a_blk].arguments.push(a_arg);
    let inner_call = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("castforge", "call"),
        operands: SmallVec::from_slice(&[a_arg]),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "callee".to_string(),
                Attribute::SymbolRef("__relu_add".to_string()),
            );
            a
        },
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[a_blk].ops = vec![inner_call];
    let a_reg = m.regions.insert_with_key(|id| Region {
        id,
        blocks: vec![a_blk],
    });
    m.ops[async_op].regions = vec![a_reg];

    let r_outlined = outline_routines(&mut m);
    assert_eq!(r_outlined, 1);

    // Verify `routine_op` kept `buf_cst` at the call site and calls `@__relu_add`:
    let new_r_blk = m.regions[m.ops[routine_op].regions[0]].blocks[0];
    assert_eq!(m.blocks[new_r_blk].ops.len(), 2);
    assert!(m.ops[m.blocks[new_r_blk].ops[0]].is_op("castforge", "buffer.constant"));
    assert!(m.ops[m.blocks[new_r_blk].ops[1]].is_op("castforge", "call"));

    let a_outlined = outline_async_routines(&mut m);
    assert_eq!(a_outlined, 1);

    // Verify `async_op` became `castforge.compiled_call` targeting `@__relu_add_async`:
    assert!(m.ops[async_op].is_op("castforge", "compiled_call"));
    assert_eq!(
        m.ops[async_op]
            .attributes
            .get("callee_symbol")
            .and_then(|a| a.as_str()),
        Some("__relu_add_async")
    );
    assert_eq!(m.functions().len(), 3);
}

#[test]
fn test_wrap_funcs_with_trace_and_rewrite_opaque_types() {
    use castforge::ir::{Attribute, Location, OpName, Operation, Type};
    use castforge::passes::{rewrite_opaque_types, wrap_funcs_with_trace};
    use indexmap::IndexMap;
    use smallvec::SmallVec;

    let mlir_src = concat!(
        "module {\n",
        "  func.func @rt_stub(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %setup = \"castforge.tensor.create\"(%x) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %dispatched = \"castforge.call.opaque\"(%setup) {callee = @k0}",
        " : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %dispatched : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let func_id = m.functions()[0];
    m.ops[func_id]
        .attributes
        .insert("abi".to_string(), Attribute::String("runtime".to_string()));

    let wrapped = wrap_funcs_with_trace(&mut m);
    assert_eq!(wrapped, 1);

    // Entry block now has: [setup_op, castforge.with_trace, func.return]:
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    assert_eq!(m.blocks[blk_id].ops.len(), 3);
    assert!(m.ops[m.blocks[blk_id].ops[0]].is_op("castforge", "tensor.create"));
    let trace_op = m.blocks[blk_id].ops[1];
    assert!(m.ops[trace_op].is_op("castforge", "with_trace"));
    let trace_blk = m.regions[m.ops[trace_op].regions[0]].blocks[0];
    assert_eq!(m.blocks[trace_blk].ops.len(), 2);
    assert!(m.ops[m.blocks[trace_blk].ops[0]].is_op("castforge", "call.opaque"));
    assert!(m.ops[m.blocks[trace_blk].ops[1]].is_op("castforge", "yield"));

    // Test `rewrite_opaque_types`: qualify `!mo.opaque<"KvCollection">` ->
    // `!mo.opaque<"nn.kv_cache.KvCollection">`:
    let arg_val = m.blocks[blk_id].arguments[0];
    m.values[arg_val].ty = Type::Opaque("!mo.opaque<\"KvCollection\">".to_string());
    let mut mapping = IndexMap::new();
    mapping.insert(
        "KvCollection".to_string(),
        Attribute::String("nn.kv_cache.KvCollection".to_string()),
    );
    let rewritten = rewrite_opaque_types(&mut m, &mapping).expect("rewrite succeeds");
    assert_eq!(rewritten, 1);
    assert_eq!(
        m.values[arg_val].ty,
        Type::Opaque("!mo.opaque<\"nn.kv_cache.KvCollection\">".to_string())
    );

    // Registering a kernel.decl `pkg.ops.MyKernelStruct` and referencing it as an unqualified
    // opaque data type `!mo.opaque<"MyKernelStruct">` must be rejected:
    let root_blk = m.regions[m.root_region].blocks[0];
    let decl_op = m.ops.insert_with_key(|id| Operation {
        id,
        name: OpName::new("castforge", "kernel.decl"),
        operands: SmallVec::new(),
        results: SmallVec::new(),
        attributes: {
            let mut a = IndexMap::new();
            a.insert(
                "sym_name".to_string(),
                Attribute::String("pkg.ops.MyKernelStruct".to_string()),
            );
            a
        },
        regions: Vec::new(),
        custom_syntax: false,
        location: Location::Unknown,
    });
    m.blocks[root_blk].ops.push(decl_op);
    m.values[arg_val].ty = Type::Opaque("!mo.opaque<\"MyKernelStruct\">".to_string());
    assert!(rewrite_opaque_types(&mut m, &mapping).is_err());
}

#[test]
fn test_anonymize_dim_names_normalize_all_names_and_semantic_hashing() {
    use castforge::ir::{Attribute, Type};
    use castforge::passes::{
        anonymize_dim_names, apply_semantic_naming, hash_module_contents, normalize_all_names,
    };

    let mlir_src = concat!(
        "module {\n",
        "  func.func @helper_one(%d: tensor<1xi64>) -> tensor<1xi64> {\n",
        "    %p = \"castforge.param.declare\"(%d)",
        " {param_name = \"BLOCK_M\", param_ref = \"BLOCK_M\"}",
        " : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %p : tensor<1xi64>\n",
        "  }\n",
        "  func.func @helper_two(%d: tensor<1xi64>) -> tensor<1xi64> {\n",
        "    %p = \"castforge.param.declare\"(%d)",
        " {param_name = \"TILE_M\", param_ref = \"TILE_M\"}",
        " : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %p : tensor<1xi64>\n",
        "  }\n",
        "  func.func @entry_fn(%d: tensor<1xi64>) -> tensor<1xi64> {\n",
        "    %0 = \"castforge.call\"(%d) {callee = @helper_two}",
        " : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %0 : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_src).expect("valid mlir");
    let funcs = m.functions();
    let f1 = funcs[0];
    let f2 = funcs[1];
    let f_entry = funcs[2];

    // Give f1 and f2 named dimension types (`!mogg.dim<batch_a>` vs `!mogg.dim<batch_b>`):
    let arg1 = m.blocks[m.regions[m.ops[f1].regions[0]].blocks[0]].arguments[0];
    let arg2 = m.blocks[m.regions[m.ops[f2].regions[0]].blocks[0]].arguments[0];
    m.values[arg1].ty = Type::Opaque("!mogg.dim<batch_a>".to_string());
    m.values[arg2].ty = Type::Opaque("!mogg.dim<batch_b>".to_string());

    let anon_count = anonymize_dim_names(&mut m);
    assert_eq!(anon_count, 2);
    assert_eq!(m.values[arg1].ty, Type::Opaque("!mogg.dim".to_string()));
    assert_eq!(m.values[arg2].ty, Type::Opaque("!mogg.dim".to_string()));

    let norm_count = normalize_all_names(&mut m).expect("normalize succeeds");
    assert_eq!(norm_count, 3);

    // After `normalize_all_names`, symbols are `__S0`, `__S1`, `__S2`, and both helper bodies
    // have `param_name = "__P0"` and `param_ref = "__P0"`, so their semantic hashes match:
    let hashed = hash_module_contents(&mut m);
    assert_eq!(hashed, 3);
    let h0 = m.ops[f1]
        .attributes
        .get("semantic_hash")
        .and_then(Attribute::as_str)
        .expect("h0")
        .to_string();
    let h1 = m.ops[f2]
        .attributes
        .get("semantic_hash")
        .and_then(Attribute::as_str)
        .expect("h1")
        .to_string();
    assert_eq!(h0, h1);

    // `apply_semantic_naming` on `["__S0", "__S1"]` deduplicates `__S1` into `kern_<h0>` and
    // rewrites `@entry_fn`'s call target to `kern_<h0>`:
    let renamed = apply_semantic_naming(&mut m, &["__S0".to_string(), "__S1".to_string()], "kern_")
        .expect("semantic naming succeeds");
    let expected_name = format!("kern_{h0}");
    assert_eq!(renamed.get("__S0"), Some(&expected_name));
    assert_eq!(renamed.get("__S1"), Some(&expected_name));
    assert_eq!(m.functions().len(), 2);

    let entry_blk = m.regions[m.ops[f_entry].regions[0]].blocks[0];
    let call_op = m.blocks[entry_blk].ops[0];
    assert_eq!(
        m.ops[call_op]
            .attributes
            .get("callee")
            .and_then(Attribute::as_str),
        Some(expected_name.as_str())
    );
}

#[test]
fn test_bucket_location_object_codec_partial_batch_and_meta_cache() {
    use castforge::passes::{
        BucketLocation, BucketObjectMetaCache, PartialBatchStoreError, decode_bucket_object_name,
        encode_bucket_object_name,
    };

    let loc = BucketLocation::parse("hf://buckets/org-team/kv-tier2/prod/v1/")
        .expect("valid hf bucket URI");
    assert_eq!(loc.namespace, "org-team");
    assert_eq!(loc.bucket, "kv-tier2");
    assert_eq!(loc.prefix, "prod/v1");
    assert_eq!(loc.handle(), "org-team/kv-tier2");

    let key = "llama3/70b@tp4:chunk#42/0";
    let encoded = encode_bucket_object_name(key);
    assert!(encoded.ends_with(".bin"));
    assert!(!encoded.contains('/'));
    assert_eq!(
        loc.remote_object_path(&encoded),
        format!("prod/v1/{encoded}")
    );
    let roundtrip = decode_bucket_object_name(&encoded).expect("decode succeeds");
    assert_eq!(roundtrip, key);

    let err = PartialBatchStoreError {
        succeeded_names: vec!["a.bin".to_string(), "c.bin".to_string()],
        failed_name: "b.bin".to_string(),
        reason: "HTTP 503".to_string(),
    };
    let bitmap = err.build_completion_bitmap(&[
        "a.bin".to_string(),
        "b.bin".to_string(),
        "c.bin".to_string(),
    ]);
    assert_eq!(bitmap, vec![true, false, true]);

    let mut cache = BucketObjectMetaCache::new(10.0, 3);
    cache.put_size("k1.bin", 4096, 100.0);
    cache.put_size("k2.bin", 8192, 105.0);
    assert_eq!(cache.get_size("k1.bin", 108.0), Some(4096));
    // At t = 112.0, `k1.bin` is expired (100 + 10 = 110 < 112), so writing `k3.bin`
    // triggers the periodic prune on the 3rd write and evicts `k1.bin`:
    cache.put_size("k3.bin", 16384, 112.0);
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get_size("k1.bin", 112.0), None);
    assert_eq!(cache.get_size("k2.bin", 112.0), Some(8192));
    assert!(cache.invalidate("k2.bin"));
    assert_eq!(cache.get_size("k2.bin", 112.0), None);
}

#[test]
fn test_canonicalize_chain_sequences_and_elide_redundant_mutable_ops() {
    use castforge::passes::{canonicalize_chain_sequences, elide_redundant_mutable_ops};

    let src = concat!(
        "module {\n",
        "  func.func @chain_and_mutable(\n",
        "    %ch0: !mo.chain,\n",
        "    %b0: !mo.buffer<4xf32>,\n",
        "    %b1: !mo.buffer<4xf32>,\n",
        "    %t0: tensor<4xf32>,\n",
        "    %t1: tensor<4xf32>\n",
        "  ) -> (tensor<4xf32>, !mo.chain) {\n",
        "    %ch1 = \"mo.mutable.store\"(%ch0, %b0, %t0) : ",
        "(!mo.chain, !mo.buffer<4xf32>, tensor<4xf32>) -> !mo.chain\n",
        "    %ch2 = \"mo.mutable.store\"(%ch1, %b0, %t1) : ",
        "(!mo.chain, !mo.buffer<4xf32>, tensor<4xf32>) -> !mo.chain\n",
        "    %loaded, %ch3 = \"mo.mutable.load\"(%ch2, %b0) : ",
        "(!mo.chain, !mo.buffer<4xf32>) -> (tensor<4xf32>, !mo.chain)\n",
        "    %ch4 = \"mo.mutable.store\"(%ch3, %b1, %t0) : ",
        "(!mo.chain, !mo.buffer<4xf32>, tensor<4xf32>) -> !mo.chain\n",
        "    return %loaded, %ch4 : tensor<4xf32>, !mo.chain\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    // 1. Elide dead first store to `%b0` and redundant load from `%b0`:
    let elided = elide_redundant_mutable_ops(&mut m);
    assert_eq!(elided, 2);

    // 2. Now `%ch2 = store(%ch0, %b0, %t1)` and `%ch4 = store(%ch2, %b1, %t0)` write to
    // distinct buffers `%b0` and `%b1`, so `canonicalize_chain_sequences` relaxes their chain
    // and inserts `mo.chain.create`:
    let relaxed = canonicalize_chain_sequences(&mut m);
    assert_eq!(relaxed, 1);

    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mo.mutable.store",
            "mo.mutable.store",
            "mo.chain.create",
            "func.return"
        ]
    );
    // Both stores now take `%ch0` as their input chain:
    let s0 = m.blocks[blk].ops[0];
    let s1 = m.blocks[blk].ops[1];
    assert_eq!(m.ops[s0].operands[0], m.ops[s1].operands[0]);
}

#[test]
fn test_create_parallel_from_unbundle_and_fold_identity_tensor_bundles() {
    use castforge::passes::{create_parallel_from_unbundle, fold_identity_tensor_bundles};

    let src = concat!(
        "module {\n",
        "  func.func @unbundle_peers(\n",
        "    %b_in: !mo.bundle<tensor<4xf32>, tensor<4xf32>>,\n",
        "    %w0: tensor<4xf32>,\n",
        "    %w1: tensor<4xf32>\n",
        "  ) -> (tensor<4xf32>, tensor<4xf32>) {\n",
        "    %u0, %u1 = \"mo.tensor.unbundle\"(%b_in) : ",
        "(!mo.bundle<tensor<4xf32>, tensor<4xf32>>) -> (tensor<4xf32>, tensor<4xf32>)\n",
        "    %a0 = \"mo.add\"(%u0, %w0) {device = \"gpu:0\"} : ",
        "(tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %a1 = \"mo.add\"(%u1, %w1) {device = \"gpu:1\"} : ",
        "(tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r0 = \"mo.relu\"(%a0) {device = \"gpu:0\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = \"mo.relu\"(%a1) {device = \"gpu:1\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r0, %r1 : tensor<4xf32>, tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    // Wrapping runs iteratively: first wraps the two `mo.add` peers into a `mo.parallel`,
    // then wraps the two downstream `mo.relu` peers consuming the new unbundle into a second
    // `mo.parallel`:
    let wrapped = create_parallel_from_unbundle(&mut m);
    assert_eq!(wrapped, 2);

    // Folding identity `bundle(unbundle(X))` eliminates the two transient round-trip bundles:
    let folded = fold_identity_tensor_bundles(&mut m);
    assert_eq!(folded, 2);

    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mo.tensor.bundle",
            "mo.parallel",
            "mo.parallel",
            "mo.tensor.unbundle",
            "func.return"
        ]
    );
}

#[test]
fn test_merge_adjacent_parallels_fuses_producer_consumer_and_drops_internal_yield() {
    use castforge::passes::{
        create_parallel_from_unbundle, fold_identity_tensor_bundles, merge_adjacent_parallels,
    };

    let src = concat!(
        "module {\n",
        "  func.func @pipeline_parallels(\n",
        "    %b_in: !mo.bundle<tensor<4xf32>, tensor<4xf32>>,\n",
        "    %w0: tensor<4xf32>,\n",
        "    %w1: tensor<4xf32>\n",
        "  ) -> (tensor<4xf32>, tensor<4xf32>) {\n",
        "    %u0, %u1 = \"mo.tensor.unbundle\"(%b_in) : ",
        "(!mo.bundle<tensor<4xf32>, tensor<4xf32>>) -> (tensor<4xf32>, tensor<4xf32>)\n",
        "    %a0 = \"mo.add\"(%u0, %w0) {device = \"gpu:0\"} : ",
        "(tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %a1 = \"mo.add\"(%u1, %w1) {device = \"gpu:1\"} : ",
        "(tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r0 = \"mo.relu\"(%a0) {device = \"gpu:0\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = \"mo.relu\"(%a1) {device = \"gpu:1\"} : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r0, %r1 : tensor<4xf32>, tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    assert_eq!(create_parallel_from_unbundle(&mut m), 2);
    assert_eq!(fold_identity_tensor_bundles(&mut m), 2);

    // Now fuse the producer `mo.parallel` (`mo.add`) and consumer `mo.parallel` (`mo.relu`)
    // into a single `mo.parallel` whose body executes `mo.add` -> `mo.relu` -> `mo.yield`:
    let merged = merge_adjacent_parallels(&mut m);
    assert_eq!(merged, 1);

    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mo.tensor.bundle",
            "mo.parallel",
            "mo.tensor.unbundle",
            "func.return"
        ]
    );
    let par_id = m.blocks[blk].ops[1];
    assert_eq!(m.ops[par_id].results.len(), 1);
    let par_blk = m.regions[m.ops[par_id].regions[0]].blocks[0];
    let body_ops: Vec<String> = m.blocks[par_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(body_ops, vec!["mo.add", "mo.relu", "mo.yield"]);
}

#[test]
fn test_infer_tensor_layouts_and_mark_stable_tensors() {
    use castforge::ir::Attribute;
    use castforge::passes::{infer_tensor_layouts, mark_stable_tensors};

    let src = concat!(
        "module {\n",
        "  \"kgen.kernel_decl\"() {sym_name = \"custom_conv_epilogue\"} : () -> ()\n",
        "  func.func @device_conv_graph(\n",
        "    %x: tensor<1x32x32x16xf32>,\n",
        "    %w: tensor<64x16x3x3xf32>,\n",
        "    %b: tensor<1x32x32x64xf32>\n",
        "  ) -> tensor<1x32x32x64xf32> attributes {is_device_graph = true} {\n",
        "    %x_relu = \"mo.relu\"(%x) : (tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>\n",
        "    %conv = \"mo.conv\"(%x_relu, %w) : ",
        "(tensor<1x32x32x16xf32>, tensor<64x16x3x3xf32>) -> tensor<1x32x32x64xf32>\n",
        "    %out = \"mo.custom\"(%conv, %b) {symbol = \"custom_conv_epilogue\"} : ",
        "(tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>\n",
        "    return %out : tensor<1x32x32x64xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    let root_blk = m.regions[m.root_region].blocks[0];
    let decl_op = m.blocks[root_blk].ops[0];
    m.ops[decl_op].attributes.insert(
        "stable_arg_indices".to_string(),
        Attribute::Array(vec![Attribute::Int(0)]),
    );

    // 1. Infer layouts: `mo.conv` seeds `NHWC` on `%x_relu` and `FCRS` on `%w`, and propagates
    // `NHWC` backward across `mo.relu` to `%x`:
    let layouts = infer_tensor_layouts(&mut m).expect("layout inference succeeds");
    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let x_arg = m.blocks[blk].arguments[0];
    let w_arg = m.blocks[blk].arguments[1];
    let relu_op = m.blocks[blk].ops[0];
    let relu_res = m.ops[relu_op].results[0];
    assert_eq!(layouts.get(&x_arg).map(String::as_str), Some("NHWC"));
    assert_eq!(layouts.get(&relu_res).map(String::as_str), Some("NHWC"));
    assert_eq!(layouts.get(&w_arg).map(String::as_str), Some("FCRS"));

    // 2. Mark stable tensors: `custom_conv_epilogue` requires operand 0 (`%conv`) to be wrapped
    // in `mo.stable.create`, and running a second time is idempotent:
    let inserted = mark_stable_tensors(&mut m).expect("mark_stable_tensors succeeds");
    assert_eq!(inserted, 1);
    let inserted_again = mark_stable_tensors(&mut m).expect("idempotent second run");
    assert_eq!(inserted_again, 0);

    let op_names: Vec<String> = m.blocks[blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mo.relu",
            "mo.conv",
            "mo.stable.create",
            "mo.custom",
            "func.return"
        ]
    );
}

#[test]
fn test_aesgcm_frame_serde_tenant_isolation_and_fp8_quantization_serde() {
    use castforge::passes::{
        AESGCM_FRAME_OVERHEAD_BYTES, AesGcmFrameSerde, Fp8E4M3QuantSerde, HkdfTenantKeyProvider,
    };

    let provider = HkdfTenantKeyProvider::new(
        b"0123456789abcdef0123456789abcdef".to_vec(),
        256,
        b"castforge-l2-aesgcm-v1".to_vec(),
    )
    .expect("valid 256-bit HKDF provider");
    let mut serde = AesGcmFrameSerde::new(provider);

    let plaintext = b"kv-cache-layer-0-block-77-payload";
    let iv = [7u8; 12];
    assert_eq!(
        AesGcmFrameSerde::estimate_serialized_size(plaintext.len()),
        plaintext.len() + AESGCM_FRAME_OVERHEAD_BYTES
    );

    let mut frame = serde.serialize(plaintext, &iv, "tenant_alpha");
    assert_eq!(frame.len(), plaintext.len() + 29);
    // Append trailing storage padding to verify `deserialize` uses `dst.len()` as plaintext length:
    frame.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

    let mut decrypted = vec![0u8; plaintext.len()];
    serde
        .deserialize(&frame, &mut decrypted, "tenant_alpha")
        .expect("decrypt with matching tenant salt succeeds");
    assert_eq!(&decrypted, plaintext);

    // Wrong tenant `cache_salt` must fail authentication:
    let mut wrong_out = vec![0u8; plaintext.len()];
    assert!(
        serde
            .deserialize(&frame, &mut wrong_out, "tenant_beta")
            .is_err()
    );

    // Tampered ciphertext byte must fail authentication:
    let mut tampered = frame.clone();
    tampered[15] ^= 0x01;
    assert!(
        serde
            .deserialize(&tampered, &mut wrong_out, "tenant_alpha")
            .is_err()
    );

    // Verify FP8 E4M3 quantization serde:
    let shapes = vec![vec![2, 4], vec![2, 4]];
    assert_eq!(Fp8E4M3QuantSerde::estimate_serialized_size(&shapes), 16);
    let src_f32 = [-224.0f32, -14.0, 0.0, 14.0, 112.0, 448.0];
    let fp8_bytes = Fp8E4M3QuantSerde::serialize_f32_to_fp8(&src_f32);
    assert_eq!(fp8_bytes.len(), src_f32.len());
    let mut restored = vec![0.0f32; src_f32.len()];
    Fp8E4M3QuantSerde::deserialize_fp8_to_f32(&fp8_bytes, &mut restored)
        .expect("fp8 deserialize succeeds");
    for (orig, dec) in src_f32.iter().zip(restored.iter()) {
        assert!((orig - dec).abs() <= 4.0, "orig={orig}, dec={dec}");
    }
}

#[test]
fn test_hoist_constant_subgraphs_into_parent_hoists_invariant_ops_and_moves_chains_last() {
    use castforge::passes::hoist_constant_subgraphs_into_parent;

    let src = concat!(
        "module {\n",
        "  func.func @sub_layer(\n",
        "    %w: tensor<4xf32>,\n",
        "    %ch: !mo.chain,\n",
        "    %x: tensor<4xf32>\n",
        "  ) -> (tensor<4xf32>, !mo.chain) attributes {is_subgraph = true} {\n",
        "    %w_abs = \"mo.abs\"(%w) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %w_scaled = \"mo.relu\"(%w_abs) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %y = \"mo.add\"(%x, %w_scaled) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %y, %ch : tensor<4xf32>, !mo.chain\n",
        "  }\n",
        "  func.func @main_graph(\n",
        "    %w_shared: tensor<4xf32>,\n",
        "    %ch0: !mo.chain,\n",
        "    %x0: tensor<4xf32>,\n",
        "    %x1: tensor<4xf32>\n",
        "  ) -> (tensor<4xf32>, !mo.chain) {\n",
        "    %y0, %ch1 = \"mo.call\"(%w_shared, %ch0, %x0) {callee = @sub_layer} : ",
        "(tensor<4xf32>, !mo.chain, tensor<4xf32>) -> (tensor<4xf32>, !mo.chain)\n",
        "    %y1, %ch2 = \"mo.call\"(%w_shared, %ch1, %x1) {callee = @sub_layer} : ",
        "(tensor<4xf32>, !mo.chain, tensor<4xf32>) -> (tensor<4xf32>, !mo.chain)\n",
        "    return %y1, %ch2 : tensor<4xf32>, !mo.chain\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    // `%w` is passed `%w_shared` at both callsites, so `mo.abs` and `mo.relu` are hoisted into
    // `@main_graph` before each callsite, `%w` is replaced by the boundary value `%w_scaled`,
    // and `%ch: !mo.chain` is moved to the trailing argument slot:
    let hoisted = hoist_constant_subgraphs_into_parent(&mut m);
    assert_eq!(hoisted, 2);

    let fns = m.functions();
    let sub_fn = fns[0];
    let main_fn = fns[1];
    let sub_blk = m.regions[m.ops[sub_fn].regions[0]].blocks[0];
    let sub_op_names: Vec<String> = m.blocks[sub_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(sub_op_names, vec!["mo.add", "func.return"]);
    // Subgraph now has 3 arguments `[%x, %w_scaled, %ch]` with `!mo.chain` moved to the end:
    let sub_arg_types: Vec<String> = m.blocks[sub_blk]
        .arguments
        .iter()
        .map(|&v| format!("{}", m.values[v].ty))
        .collect();
    assert_eq!(
        sub_arg_types,
        vec![
            "tensor<4xf32>".to_string(),
            "tensor<4xf32>".to_string(),
            "!mo.chain".to_string()
        ]
    );

    let main_blk = m.regions[m.ops[main_fn].regions[0]].blocks[0];
    let main_op_names: Vec<String> = m.blocks[main_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        main_op_names,
        vec![
            "mo.abs",
            "mo.relu",
            "mo.call",
            "mo.abs",
            "mo.relu",
            "mo.call",
            "func.return"
        ]
    );
}

#[test]
fn test_hoist_parameter_expressions_and_invoke_shape_packing_functions() {
    use castforge::ir::{Attribute, DType, Dim, TensorType, Type};
    use castforge::passes::{hoist_parameter_expressions, invoke_shape_packing_functions};

    let src = concat!(
        "module {\n",
        "  func.func @param_and_packing(%x: tensor<4xf32>, %w: tensor<64x36xf32>) ",
        "-> tensor<4xf32> attributes {target_tile_width = 16} {\n",
        "    \"mo.invoke_shape_func\"(%w) {shape_func_name = \"pack_matmul_b_shape_func\"} : ",
        "(tensor<64x36xf32>) -> ()\n",
        "    %y = \"mo.relu\"(%x) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %z = \"mo.abs\"(%y) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %z : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let invoke_op = m.blocks[blk].ops[0];
    m.ops[invoke_op].attributes.insert(
        "output_param_decls".to_string(),
        Attribute::Array(vec![
            Attribute::String("P0".to_string()),
            Attribute::String("P1".to_string()),
            Attribute::String("P2".to_string()),
        ]),
    );

    // Give `%x`, `%y`, and `%z` a complex symbolic dimension `mul(B, 2)` so
    // `hoist_parameter_expressions` hoists `param_expr_0 = mul(B, 2)` and folds consecutive
    // `mo.rebind`s between `mo.relu` and `mo.abs`:
    let complex_ty = Type::Tensor(TensorType::new(
        vec![Dim::Symbolic("mul(B, 2)".to_string())],
        DType::F32,
    ));
    let x_arg = m.blocks[blk].arguments[0];
    let relu_op = m.blocks[blk].ops[1];
    let abs_op = m.blocks[blk].ops[2];
    m.values[x_arg].ty = complex_ty.clone();
    m.values[m.ops[relu_op].results[0]].ty = complex_ty.clone();
    m.values[m.ops[abs_op].results[0]].ty = complex_ty;

    let decls = hoist_parameter_expressions(&mut m);
    assert_eq!(decls, 1);

    let resolved = invoke_shape_packing_functions(&mut m, true)
        .expect("invoke_shape_packing_functions succeeds");
    assert_eq!(resolved, 1);

    // Check that 3 packing `kgen.param.declare` ops (`[ceil(36/16)=3, 64, 16]`) and 3
    // `kgen.param.assert_eq` sanity-check ops were materialized:
    let mut pack_vals = Vec::new();
    let mut assert_count = 0usize;
    for &oid in &m.blocks[blk].ops {
        let full = m.ops[oid].name.full_name();
        if full == "kgen.param.declare"
            && let Some(Attribute::Int(v)) = m.ops[oid].attributes.get("value")
        {
            pack_vals.push(*v);
        } else if full == "kgen.param.assert_eq" {
            assert_count += 1;
        }
    }
    assert_eq!(pack_vals, vec![3, 64, 16]);
    assert_eq!(assert_count, 3);
}

#[test]
fn test_validate_device_dtypes_verify_device_graph_kernels_and_splat_large_constants() {
    use castforge::passes::{
        splat_large_constants, validate_device_dtypes, verify_device_graph_kernels,
    };

    let src = concat!(
        "module {\n",
        "  \"kgen.kernel_decl\"() {sym_name = \"fast_gelu_kern\", ",
        "kernel_name = \"fast_gelu\", build_graph = true} : () -> ()\n",
        "  func.func @dev_graph(%x: tensor<16xf16>) -> tensor<16xf16> ",
        "attributes {is_device_graph = true} {\n",
        "    %c = \"mo.constant\"() : () -> tensor<16xf16>\n",
        "    %y = \"mo.custom\"(%x, %c) {symbol = \"fast_gelu\"} : ",
        "(tensor<16xf16>, tensor<16xf16>) -> tensor<16xf16>\n",
        "    return %y : tensor<16xf16>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();

    // 1. `f16` is rejected on `aarch64` CPUs, but permitted on `x86_64`:
    let err = validate_device_dtypes(&mut m, Some("aarch64-unknown-linux-gnu"))
        .expect_err("f16 must be rejected on aarch64");
    assert!(err.contains("f16 data type is not supported"));

    let fn_id = m.functions()[0];
    m.ops[fn_id].attributes.shift_remove("device_target_triple");
    validate_device_dtypes(&mut m, Some("x86_64-unknown-linux-gnu"))
        .expect("f16 is permitted on x86_64");

    // 2. Verify device-graph kernel registration has `build_graph = true`:
    let verified = verify_device_graph_kernels(&m).expect("kernel has build_graph");
    assert_eq!(verified, 1);

    // 3. Splat constants with >= 8 elements into `tensor<1xf16>` + `mo.static.broadcast_to`:
    let splatted = splat_large_constants(&mut m, 8);
    assert_eq!(splatted, 1);

    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let op_names: Vec<String> = m.blocks[blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mo.constant",
            "mo.static.broadcast_to",
            "mo.custom",
            "func.return"
        ]
    );
}

#[test]
fn test_inject_nan_check_instrumentation_with_stride_sampling_and_host_transfer() {
    use castforge::ir::Attribute;
    use castforge::passes::inject_nan_check_instrumentation;

    let src = concat!(
        "module {\n",
        "  func.func @gpu_kernels(%x: tensor<4xf32>) -> tensor<4xf32> {\n",
        "    %k0 = \"map.iter.elementwise\"(%x) {device = \"gpu:0\"} : ",
        "(tensor<4xf32>) -> tensor<4xf32>\n",
        "    %k1 = \"map.iter.elementwise\"(%k0) {device = \"gpu:0\"} : ",
        "(tensor<4xf32>) -> tensor<4xf32>\n",
        "    %k2 = \"map.iter.opaque\"(%k1) {device = \"gpu:0\"} : ",
        "(tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %k2 : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    // With `stride = 2` across 3 float kernel outputs (`%k0`, `%k1`, `%k2`), indices 0 and 2
    // are checked (`checked = 2`, `total = 3`):
    let (checked, total) = inject_nan_check_instrumentation(&mut m, 2);
    assert_eq!(total, 3);
    assert_eq!(checked, 2);

    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let mut count_ops = 0usize;
    let mut xfer_ops = 0usize;
    let mut raise_ops = 0usize;
    for &oid in &m.blocks[blk].ops {
        let op = &m.ops[oid];
        let full = op.name.full_name();
        if full == "mo.transfer" {
            xfer_ops += 1;
        } else if full == "map.iter.opaque" {
            match op.attributes.get("symbol").and_then(Attribute::as_str) {
                Some("nan_check_count") => count_ops += 1,
                Some("nan_check_raise") => {
                    assert_eq!(
                        op.attributes.get("has_write_effect"),
                        Some(&Attribute::Bool(true))
                    );
                    raise_ops += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(count_ops, 2);
    assert_eq!(xfer_ops, 4); // 2 transfers (`nan_count`, `inf_count`) per checked GPU kernel
    assert_eq!(raise_ops, 2);
}

#[test]
fn test_tenant_quota_manager_allowlist_vs_default_semantics_and_multi_serde_arity() {
    use castforge::passes::{TenantQuotaManager, validate_multi_serde_group_size};

    let mut qm = TenantQuotaManager::new("l2_quotas");
    assert_eq!(qm.name(), "l2_quotas");
    assert!(qm.set_quota("bad_tenant", -1).is_err());
    qm.set_quota("tenant_a", 1_048_576).expect("positive quota");
    qm.set_quota("tenant_zero", 0).expect("zero quota allowed");

    // Allowlist semantics (`get_limit_bytes`) returns 0 for unregistered salts:
    assert_eq!(qm.get_limit_bytes("tenant_a"), 1_048_576);
    assert_eq!(qm.get_limit_bytes("unregistered"), 0);
    assert!(qm.has_quota("tenant_zero"));
    assert!(!qm.has_quota("unregistered"));

    // Boot default-quota semantics (`effective_limit_bytes`) returns `None` (exempt) for
    // unregistered salts until `set_default_limit_bytes(Some(0))` arms allowlist enforcement:
    assert_eq!(qm.effective_limit_bytes("tenant_a"), Some(1_048_576));
    assert_eq!(qm.effective_limit_bytes("unregistered"), None);

    qm.set_default_limit_bytes(Some(0))
        .expect("arm allowlist enforcement");
    assert_eq!(qm.effective_limit_bytes("unregistered"), Some(0));

    let snap = qm.capture();
    assert!(qm.delete_quota("tenant_a"));
    assert_eq!(qm.get_limit_bytes("tenant_a"), 0);

    qm.restore(snap);
    assert_eq!(qm.get_limit_bytes("tenant_a"), 1_048_576);
    assert_eq!(qm.list_quotas().len(), 2);

    validate_multi_serde_group_size(&[Some(1u8), None], 2, "src").expect("arity 2 matches");
    let err = validate_multi_serde_group_size(&[Some(1u8)], 2, "dst")
        .expect_err("arity mismatch must error");
    assert!(err.contains("MultiSerde dst group length 1 does not match expected 2"));
}

#[test]
fn test_allocate_accumulating_buffers_mixed_precision_vs_same_dtype() {
    use castforge::ir::Attribute;
    use castforge::ir::mlir::OpName;
    use castforge::passes::allocate_accumulating_buffers;

    let src = concat!(
        "module {\n",
        "  func.func @mixed_and_same(\n",
        "    %a: tensor<4x4xf32>,\n",
        "    %out_f16: tensor<4x4xf16>,\n",
        "    %out_f32: tensor<4x4xf32>,\n",
        "    %ch0: !mo.chain\n",
        "  ) -> (tensor<4x4xf16>, tensor<4x4xf32>, !mo.chain) {\n",
        "    return %out_f16, %out_f32, %ch0",
        " : tensor<4x4xf16>, tensor<4x4xf32>, !mo.chain\n",
        "  }\n",
        "}\n"
    );

    let mut module = parse_mlir(src).unwrap();
    let fn_id = module.functions()[0];
    let fn_blk = module.regions[module.ops[fn_id].regions[0]].blocks[0];
    let fn_args = module.blocks[fn_blk].arguments.clone();
    let (a_arg, out_f16_arg, out_f32_arg, ch0_arg) =
        (fn_args[0], fn_args[1], fn_args[2], fn_args[3]);
    let f32_scalar = castforge::ir::Type::Scalar(DType::F32);
    let f16_scalar = castforge::ir::Type::Scalar(DType::F16);
    let idx_ty = castforge::ir::Type::Scalar(DType::Index);
    let t_f32 = module.values[a_arg].ty.clone();
    let t_f16 = module.values[out_f16_arg].ty.clone();
    let ch_ty = module.values[ch0_arg].ty.clone();

    // Helper to build a `map.iter.opaque` with a `mo.matmul` call and a `map.iter.epilogue`:
    let build_kernel = |m: &mut castforge::ir::Module,
                        out_arg: castforge::ir::ValueId,
                        out_ty: castforge::ir::Type,
                        store_val_ty: castforge::ir::Type| {
        let epi_blk = m.create_block(None);
        let val_in = m.add_block_argument(epi_blk, None, f32_scalar.clone());
        let idx_in = m.add_block_argument(epi_blk, None, idx_ty.clone());
        let cap_out = m.add_block_argument(epi_blk, None, out_ty.clone());
        let store_val = if store_val_ty != f32_scalar {
            let cast_op = m.add_operation(
                epi_blk,
                OpName::new("map", "cast"),
                smallvec::smallvec![val_in],
                vec![(None, store_val_ty)],
                indexmap::IndexMap::new(),
                vec![],
                false,
                Location::Unknown,
            );
            m.ops[cast_op].results[0]
        } else {
            val_in
        };
        m.add_operation(
            epi_blk,
            OpName::new("map", "store"),
            smallvec::smallvec![store_val, cap_out, idx_in],
            vec![],
            indexmap::IndexMap::new(),
            vec![],
            false,
            Location::Unknown,
        );
        let epi_reg = m.create_region(vec![epi_blk]);

        let k_blk = m.create_block(None);
        let in_a = m.add_block_argument(k_blk, None, t_f32.clone());
        let _in_out = m.add_block_argument(k_blk, None, out_ty.clone());
        let mut call_attrs = indexmap::IndexMap::new();
        call_attrs.insert(
            "kernel_name".to_string(),
            Attribute::String("mo.matmul".to_string()),
        );
        m.add_operation(
            k_blk,
            OpName::new("map", "call_opaque"),
            smallvec::smallvec![in_a, in_a],
            vec![(None, t_f32.clone())],
            call_attrs,
            vec![],
            false,
            Location::Unknown,
        );
        m.add_operation(
            k_blk,
            OpName::new("map", "iter.epilogue"),
            smallvec::smallvec![out_arg],
            vec![],
            indexmap::IndexMap::new(),
            vec![epi_reg],
            false,
            Location::Unknown,
        );
        let k_reg = m.create_region(vec![k_blk]);
        m.add_operation(
            fn_blk,
            OpName::new("map", "iter.opaque"),
            smallvec::smallvec![a_arg, out_arg, ch0_arg],
            vec![(None, out_ty), (None, ch_ty.clone())],
            indexmap::IndexMap::new(),
            vec![k_reg],
            false,
            Location::Unknown,
        )
    };

    let k1 = build_kernel(&mut module, out_f16_arg, t_f16, f16_scalar);
    let k2 = build_kernel(&mut module, out_f32_arg, t_f32.clone(), f32_scalar.clone());

    let (assigned, scratch) = allocate_accumulating_buffers(&mut module);
    assert_eq!(assigned, 2);
    assert_eq!(scratch, 1);
    assert_eq!(module.ops[k1].results.len(), 3);
    assert_eq!(module.ops[k2].results.len(), 2);
}

#[test]
fn test_loop_invariant_view_motion_hoists_load_views_and_preserves_store_transforms() {
    use castforge::ir::mlir::OpName;
    use castforge::passes::loop_invariant_view_motion;

    let src = concat!(
        "module {\n",
        "  func.func @view_motion(\n",
        "    %in_t: tensor<8xf32>,\n",
        "    %out_t: tensor<8xf32>\n",
        "  ) -> tensor<8xf32> {\n",
        "    return %out_t : tensor<8xf32>\n",
        "  }\n",
        "}\n"
    );

    let mut m = parse_mlir(src).unwrap();
    let fn_id = m.functions()[0];
    let fn_blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let args = m.blocks[fn_blk].arguments.clone();
    let (in_t, out_t) = (args[0], args[1]);
    let t_ty = m.values[in_t].ty.clone();
    let idx_ty = castforge::ir::Type::Scalar(DType::Index);
    let f32_ty = castforge::ir::Type::Scalar(DType::F32);

    // Build `map.iter.elementwise.body(%in_t, %out_t)` before `func.return`:
    let c_blk = m.create_block(None);
    let loop_idx = m.add_block_argument(c_blk, None, idx_ty.clone());
    let cap_in = m.add_block_argument(c_blk, None, t_ty.clone());
    let cap_out = m.add_block_argument(c_blk, None, t_ty.clone());

    // Store-feeding transform `map.index.reshape(%loop_idx)` -> must stay in closure:
    let reshape_op = m.add_operation(
        c_blk,
        OpName::new("map", "index.reshape"),
        smallvec::smallvec![loop_idx],
        vec![(None, idx_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let reshaped_idx = m.ops[reshape_op].results[0];

    // Load-only transform composed on `reshaped_idx`: `map.index.broadcast(%reshaped_idx)` ->
    // must hoist into `map.index.broadcast.stride` before the closure!
    let bcast_op = m.add_operation(
        c_blk,
        OpName::new("map", "index.broadcast"),
        smallvec::smallvec![reshaped_idx],
        vec![(None, idx_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let bcast_idx = m.ops[bcast_op].results[0];

    let load_op = m.add_operation(
        c_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![cap_in, bcast_idx],
        vec![(None, f32_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let loaded_val = m.ops[load_op].results[0];

    m.add_operation(
        c_blk,
        OpName::new("map", "store"),
        smallvec::smallvec![loaded_val, cap_out, reshaped_idx],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );

    let c_reg = m.create_region(vec![c_blk]);
    let closure_op = m.add_operation(
        fn_blk,
        OpName::new("map", "iter.elementwise.body"),
        smallvec::smallvec![in_t, out_t],
        vec![],
        indexmap::IndexMap::new(),
        vec![c_reg],
        false,
        Location::Unknown,
    );

    let (hoisted, erased) = loop_invariant_view_motion(&mut m);
    assert_eq!(hoisted, 1);
    assert_eq!(erased, 1);

    // Verify `map.index.broadcast.stride` was inserted before `map.iter.elementwise.body`
    // and `map.index.reshape` remained inside the closure:
    let outer_names: Vec<String> = m.blocks[fn_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert!(outer_names.contains(&"map.index.broadcast.stride".to_string()));
    assert_eq!(m.ops[closure_op].operands.len(), 3);

    let inner_names: Vec<String> = m.blocks[c_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        inner_names,
        vec!["map.index.reshape", "map.load", "map.store"]
    );
    // `map.load` now indexes at `reshaped_idx` directly:
    assert_eq!(m.ops[load_op].operands[1], reshaped_idx);
}

#[test]
fn test_stage_device_graphs_and_raise_back_index_constants() {
    use castforge::ir::Attribute;
    use castforge::passes::{raise_back_index_constants, stage_device_graphs};

    let src = concat!(
        "module {\n",
        "  func.func @kernel_single_caller(%buf: !mo.buffer) -> index",
        " attributes {abi = \"mgp\"} {\n",
        "    %rc = \"mgp.resource.constant\"() {resource_slot = 1 : i64} : () -> !mo.buffer\n",
        "    %idx = \"mgp.buffer_to_index\"(%rc) : (!mo.buffer) -> index\n",
        "    return %idx : index\n",
        "  }\n",
        "  func.func @kernel_multi_caller(%buf: !mo.buffer) -> index",
        " attributes {abi = \"mgp\"} {\n",
        "    %rc = \"mgp.resource.constant\"() {resource_slot = 0 : i64} : () -> !mo.buffer\n",
        "    %idx = \"mgp.buffer_to_index\"(%rc) : (!mo.buffer) -> index\n",
        "    return %idx : index\n",
        "  }\n",
        "  func.func @device_graph_model(\n",
        "    %x: tensor<4xf32>,\n",
        "    %ch0: !mo.chain\n",
        "  ) -> (tensor<4xf32>, !mo.chain) attributes {is_device_graph = true} {\n",
        "    %s = \"mgp.device_graph.stable.create\"(%x) : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    %g = \"mgp.guard\"(%ch0, %s) : (!mo.chain, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %y, %ch1 = \"mogg.routine.async\"(%g, %ch0)",
        " : (tensor<4xf32>, !mo.chain) -> (tensor<4xf32>, !mo.chain)\n",
        "    \"mogg.compiled_call\"(%y) {callee = @kernel_single_caller,",
        " constant_resources = array<i64: 10, 42>} : (tensor<4xf32>) -> ()\n",
        "    \"mogg.compiled_call\"(%y) {callee = @kernel_multi_caller,",
        " constant_resources = array<i64: 7>} : (tensor<4xf32>) -> ()\n",
        "    \"mogg.compiled_call\"(%y) {callee = @kernel_multi_caller,",
        " constant_resources = array<i64: 9>} : (tensor<4xf32>) -> ()\n",
        "    return %y, %ch1 : tensor<4xf32>, !mo.chain\n",
        "  }\n",
        "}\n"
    );

    let mut m = parse_mlir(src).unwrap();

    // 1. `raise_back_index_constants` raises `mgp.buffer_to_index` in `@kernel_single_caller`
    // to `index.constant {value = 42}` while leaving `@kernel_multi_caller` untouched:
    let raised = raise_back_index_constants(&mut m);
    assert_eq!(raised, 1);

    let fns = m.functions();
    let single_blk = m.regions[m.ops[fns[0]].regions[0]].blocks[0];
    let single_ops: Vec<String> = m.blocks[single_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        single_ops,
        vec!["mgp.resource.constant", "index.constant", "func.return"]
    );
    let idx_const_id = m.blocks[single_blk].ops[1];
    assert_eq!(
        m.ops[idx_const_id].attributes.get("value"),
        Some(&Attribute::Int(42))
    );

    // 2. `stage_device_graphs` lowers `mgp.guard` to `mogg.routine.async` and stages all 3
    // device-graph ops (`stable.create`, lowered `guard`, `routine.async`) into a single
    // `mgp.device_graph.staging` region:
    let (lowered_guards, staged_ops) = stage_device_graphs(&mut m);
    assert_eq!(lowered_guards, 1);
    assert_eq!(staged_ops, 3);

    let model_blk = m.regions[m.ops[fns[2]].regions[0]].blocks[0];
    let model_op_names: Vec<String> = m.blocks[model_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        model_op_names,
        vec![
            "mogg.compiled_call",
            "mogg.compiled_call",
            "mogg.compiled_call",
            "mgp.device_graph.staging",
            "func.return"
        ]
    );
}

#[test]
fn test_verify_model_arguments_and_fixup_overhanging_chains() {
    use castforge::ir::Attribute;
    use castforge::passes::{fixup_overhanging_chains, verify_model_arguments};

    let src = concat!(
        "module {\n",
        "  func.func @model_entry(\n",
        "    %ctx: !mo.chain,\n",
        "    %x_mixed: tensor<?x16x32xf32>,\n",
        "    %x_dyn: tensor<?x?xf32>\n",
        "  ) -> !mo.chain {\n",
        "    %u0, %u1 = \"mo.async_op_a\"(%x_mixed)",
        " : (tensor<?x16x32xf32>) -> (tensor<16xf32>, !mo.chain)\n",
        "    %v0 = \"mo.async_op_b\"(%x_dyn) : (tensor<?x?xf32>) -> tensor<8xf32>\n",
        "    return %ctx : !mo.chain\n",
        "  }\n",
        "}\n"
    );

    let mut m = parse_mlir(src).unwrap();

    // 1. `verify_model_arguments` with `implicit_arg_count = 1` skips `%ctx`, generates a shape
    // assertion for `%x_mixed` (2 static dims -> `pop.and` + `mgp.assert` with `"any,16,32"`),
    // and skips `%x_dyn` (all dynamic dims):
    let asserts = verify_model_arguments(&mut m, 1);
    assert_eq!(asserts, 1);

    let fn_id = m.functions()[0];
    let blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let assert_op = m.blocks[blk]
        .ops
        .iter()
        .copied()
        .find(|&id| m.ops[id].name.full_name() == "mgp.assert")
        .expect("mgp.assert must exist");
    assert_eq!(
        m.ops[assert_op].attributes.get("msg"),
        Some(&Attribute::String(
            "Tensor type mismatch for argument 0: Expected shape: [any,16,32]".to_string()
        ))
    );

    // 2. `fixup_overhanging_chains` finds 2 non-terminator ops with unused results (`mo.async_op_a`
    // and `mo.async_op_b`), picking at most 1 unused result per op (`%u0` and `%v0`), and joins
    // them with the trailing `%ctx` return chain via `mo.chain.create`:
    let synced = fixup_overhanging_chains(&mut m);
    assert_eq!(synced, 2);

    let ops = &m.blocks[blk].ops;
    let chain_create_id = ops[ops.len() - 2];
    let ret_id = ops[ops.len() - 1];
    assert_eq!(m.ops[chain_create_id].name.full_name(), "mo.chain.create");
    // `mo.chain.create` has 3 operands: 1 from `mo.async_op_a`, 1 from `mo.async_op_b`, and `%ctx`:
    assert_eq!(m.ops[chain_create_id].operands.len(), 3);
    assert_eq!(m.ops[ret_id].operands[0], m.ops[chain_create_id].results[0]);
}

#[test]
fn test_inline_tensor_and_grouped_tensor_wire_envelopes() {
    use castforge::passes::{
        InlineTensorFrame, InlineTensorGroup, decode_grouped_tensor_envelope,
        decode_inline_tensor_envelope, encode_grouped_tensor_envelope,
        encode_inline_tensor_envelope, inline_tensor_dtype_itemsize,
        validate_grouped_chunk_indices,
    };

    assert_eq!(inline_tensor_dtype_itemsize("torch.bfloat16"), Some(2));
    assert_eq!(inline_tensor_dtype_itemsize("float8_e4m3fn"), Some(1));
    assert_eq!(inline_tensor_dtype_itemsize("complex128"), None);

    let f0 = InlineTensorFrame {
        dtype: "float16".to_string(),
        shape: vec![2, 2],
        data: vec![1, 2, 3, 4, 5, 6, 7, 8],
    };
    let f1 = InlineTensorFrame {
        dtype: "torch.float16".to_string(),
        shape: vec![2, 2],
        data: vec![9, 10, 11, 12, 13, 14, 15, 16],
    };

    let flat_wire =
        encode_inline_tensor_envelope(&[f0.clone(), f1.clone()]).expect("flat encode succeeds");
    let decoded_flat =
        decode_inline_tensor_envelope(&flat_wire, &[2, 2], "float16", 2).expect("flat decode");
    assert_eq!(decoded_flat.len(), 2);
    assert_eq!(decoded_flat[1].dtype, "float16");
    assert_eq!(decoded_flat[1].data, f1.data);

    // Disjoint envelope tags: flat wire must not decode as grouped and vice versa:
    let expected_spec = vec![(vec![2usize, 2], "float16".to_string(), 4usize)];
    assert!(decode_grouped_tensor_envelope(&flat_wire, &expected_spec).is_err());

    // Grouped envelope with sparse strictly-increasing chunk indices `[1, 3]` (out of 4 chunks):
    let grp = InlineTensorGroup {
        chunk_indices: vec![1, 3],
        tensors: vec![f0.clone(), f1],
    };
    let grp_wire = encode_grouped_tensor_envelope(std::slice::from_ref(&grp))
        .expect("grouped encode succeeds");
    assert!(decode_inline_tensor_envelope(&grp_wire, &[2, 2], "float16", 2).is_err());

    let decoded_grp =
        decode_grouped_tensor_envelope(&grp_wire, &expected_spec).expect("grouped decode succeeds");
    assert_eq!(decoded_grp.len(), 1);
    assert_eq!(decoded_grp[0].chunk_indices, vec![1, 3]);
    assert_eq!(decoded_grp[0].tensors.len(), 2);

    // Non-monotonic chunk indices `[2, 2]` and out-of-range chunk index `4 >= 4` must fail:
    assert!(validate_grouped_chunk_indices(0, &[2, 2], 2, Some(4)).is_err());
    assert!(validate_grouped_chunk_indices(0, &[1, 4], 2, Some(4)).is_err());
}

#[test]
fn test_outline_structured_kernel_bodies_wraps_elementwise_and_rotates_index_arg() {
    use castforge::ir::Attribute;
    use castforge::ir::mlir::OpName;
    use castforge::passes::outline_structured_map_loops;

    let src = concat!(
        "module {\n",
        "  func.func @structured_loop(\n",
        "    %x: tensor<8xf32>,\n",
        "    %ch0: !mo.chain\n",
        "  ) -> (tensor<8xf32>, !mo.chain) {\n",
        "    return %x, %ch0 : tensor<8xf32>, !mo.chain\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    let fn_id = m.functions()[0];
    let fn_blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let args = m.blocks[fn_blk].arguments.clone();
    let (x_arg, ch0_arg) = (args[0], args[1]);
    let t_ty = m.values[x_arg].ty.clone();
    let ch_ty = m.values[ch0_arg].ty.clone();
    let idx_ty = castforge::ir::Type::Scalar(DType::Index);
    let f32_ty = castforge::ir::Type::Scalar(DType::F32);

    // Build `map.iter.elementwise(%x, %ch0)` whose body block has args `[%in_x, %out_y, %idx]`:
    let body_blk = m.create_block(None);
    let in_x = m.add_block_argument(body_blk, None, t_ty.clone());
    let out_y = m.add_block_argument(body_blk, None, t_ty.clone());
    let loop_idx = m.add_block_argument(body_blk, None, idx_ty.clone());
    let load_op = m.add_operation(
        body_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![in_x, loop_idx],
        vec![(None, f32_ty)],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let loaded = m.ops[load_op].results[0];
    m.add_operation(
        body_blk,
        OpName::new("map", "store"),
        smallvec::smallvec![loaded, out_y, loop_idx],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let body_reg = m.create_region(vec![body_blk]);

    let mut attrs = indexmap::IndexMap::new();
    attrs.insert("has_read_effect".to_string(), Attribute::Bool(true));
    let loop_op = m.add_operation(
        fn_blk,
        OpName::new("map", "iter.elementwise"),
        smallvec::smallvec![x_arg, ch0_arg],
        vec![(None, t_ty), (None, ch_ty)],
        attrs,
        vec![body_reg],
        false,
        Location::Unknown,
    );

    let outlined = outline_structured_map_loops(&mut m);
    assert_eq!(outlined, 1);
    assert_eq!(m.ops[loop_op].name.full_name(), "map.iter.opaque");
    assert_eq!(
        m.ops[loop_op].attributes.get("device"),
        Some(&Attribute::String("cpu:0".to_string()))
    );

    // Outer `map.iter.opaque` body has 2 non-chain block args (input `%x`, output `%y`) and
    // contains a single `map.iter.elementwise.body` closure whose block args are
    // `[%loop_idx, %in_x, %out_y]`:
    let opaque_reg = m.ops[loop_op].regions[0];
    let opaque_blk = m.regions[opaque_reg].blocks[0];
    assert_eq!(m.blocks[opaque_blk].arguments.len(), 2);
    let closure_id = m.blocks[opaque_blk].ops[0];
    assert_eq!(
        m.ops[closure_id].name.full_name(),
        "map.iter.elementwise.body"
    );
    assert_eq!(m.ops[closure_id].operands.len(), 2);
    assert_eq!(m.blocks[body_blk].arguments, vec![loop_idx, in_x, out_y]);
}

#[test]
fn test_attach_python_stack_traces_and_fixup_isolated_from_above_regions() {
    use castforge::ir::Attribute;
    use castforge::ir::mlir::OpName;
    use castforge::passes::{attach_python_stack_traces, fixup_isolated_from_above_regions};

    let src = concat!(
        "module {\n",
        "  func.func @trace_and_isolation(\n",
        "    %cond_in: i1,\n",
        "    %outer_val: tensor<4xf32>\n",
        "  ) -> tensor<4xf32> {\n",
        "    \"mgp.assert\"(%cond_in) {msg = \"Shape mismatch\",",
        " python_stack_trace = \"  File 'model.py', line 42, in forward\"} : (i1) -> ()\n",
        "    return %outer_val : tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    let fn_id = m.functions()[0];
    let fn_blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let assert_op = m.blocks[fn_blk].ops[0];
    let args = m.blocks[fn_blk].arguments.clone();
    let (cond_in, outer_val) = (args[0], args[1]);
    let i1_ty = m.values[cond_in].ty.clone();
    let t_ty = m.values[outer_val].ty.clone();

    // 1. `attach_python_stack_traces` embeds the Python traceback into `mgp.assert`'s `"msg"`
    // and materializes `"source_notes"` while removing `"python_stack_trace"`:
    let attached = attach_python_stack_traces(&mut m);
    assert_eq!(attached, 1);
    assert_eq!(
        m.ops[assert_op].attributes.get("msg"),
        Some(&Attribute::String(
            "\nSource Traceback:\n  File 'model.py', line 42, in forward\nerror: Shape mismatch"
                .to_string()
        ))
    );
    assert_eq!(
        m.ops[assert_op].attributes.get("source_notes"),
        Some(&Attribute::String(
            "  File 'model.py', line 42, in forward".to_string()
        ))
    );
    assert!(
        !m.ops[assert_op]
            .attributes
            .contains_key("python_stack_trace")
    );

    // 2. Build a `grt.while(%outer_val)` with 2 regions (`cond` and `body`) where `body` directly
    // references `%cond_in` as a free value from the outer block:
    let cond_blk = m.create_block(None);
    let cond_arg0 = m.add_block_argument(cond_blk, None, t_ty.clone());
    let cond_out_op = m.add_operation(
        cond_blk,
        OpName::new("mef", "output"),
        smallvec::smallvec![cond_arg0, cond_in],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let cond_reg = m.create_region(vec![cond_blk]);

    let body_blk = m.create_block(None);
    let body_arg0 = m.add_block_argument(body_blk, None, t_ty.clone());
    let body_out_op = m.add_operation(
        body_blk,
        OpName::new("mef", "output"),
        smallvec::smallvec![body_arg0],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let body_reg = m.create_region(vec![body_blk]);

    let while_op = m.add_operation(
        fn_blk,
        OpName::new("grt", "while"),
        smallvec::smallvec![outer_val],
        vec![(None, t_ty)],
        indexmap::IndexMap::new(),
        vec![cond_reg, body_reg],
        false,
        Location::Unknown,
    );

    let fixed = fixup_isolated_from_above_regions(&mut m);
    assert_eq!(fixed, 1);

    // `grt.while` now takes `(%outer_val, %cond_in)` and returns 2 results:
    assert_eq!(m.ops[while_op].operands.as_slice(), &[outer_val, cond_in]);
    assert_eq!(m.ops[while_op].results.len(), 2);
    assert_eq!(m.values[m.ops[while_op].results[1]].ty, i1_ty);

    // In `cond_blk`, the newly threaded block arg is placed before the last condition output:
    assert_eq!(m.blocks[cond_blk].arguments.len(), 2);
    let cond_free_barg = m.blocks[cond_blk].arguments[1];
    assert_eq!(
        m.ops[cond_out_op].operands.as_slice(),
        &[cond_arg0, cond_free_barg, cond_free_barg]
    );

    // In `body_blk`, the newly threaded block arg is appended to `mef.output`:
    assert_eq!(m.blocks[body_blk].arguments.len(), 2);
    let body_free_barg = m.blocks[body_blk].arguments[1];
    assert_eq!(
        m.ops[body_out_op].operands.as_slice(),
        &[body_arg0, body_free_barg]
    );
}

#[test]
fn test_fuse_rowwise_reductions_fuses_producer_and_per_row_consumer() {
    use castforge::ir::Attribute;
    use castforge::ir::mlir::OpName;
    use castforge::passes::fuse_rowwise_reductions;

    let src = concat!(
        "module {\n",
        "  func.func @rowwise_fusion(\n",
        "    %x: tensor<4x8xf32>,\n",
        "    %bias: tensor<4x1xf32>\n",
        "  ) -> tensor<4x1xf32> {\n",
        "    return %bias : tensor<4x1xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    let fn_id = m.functions()[0];
    let fn_blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let ret_id = m.blocks[fn_blk].ops[0];
    let args = m.blocks[fn_blk].arguments.clone();
    let (x_arg, bias_arg) = (args[0], args[1]);
    let full_ty = m.values[x_arg].ty.clone();
    let row_ty = m.values[bias_arg].ty.clone();
    let idx_ty = castforge::ir::Type::Scalar(DType::Index);
    let f32_ty = castforge::ir::Type::Scalar(DType::F32);

    // 1. Build producer `map.iter.elementwise(%x) -> tensor<4x8xf32>` computing `exp(load(%x))`:
    let prod_blk = m.create_block(None);
    let p_in = m.add_block_argument(prod_blk, None, full_ty.clone());
    let p_out = m.add_block_argument(prod_blk, None, full_ty.clone());
    let p_idx = m.add_block_argument(prod_blk, None, idx_ty.clone());
    let p_load = m.add_operation(
        prod_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![p_in, p_idx],
        vec![(None, f32_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let p_loaded = m.ops[p_load].results[0];
    let p_exp = m.add_operation(
        prod_blk,
        OpName::new("rmo", "exp"),
        smallvec::smallvec![p_loaded],
        vec![(None, f32_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let p_exp_val = m.ops[p_exp].results[0];
    m.add_operation(
        prod_blk,
        OpName::new("map", "store"),
        smallvec::smallvec![p_exp_val, p_out, p_idx],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let prod_reg = m.create_region(vec![prod_blk]);
    let mut prod_attrs = indexmap::IndexMap::new();
    prod_attrs.insert("device".to_string(), Attribute::String("gpu:0".to_string()));
    prod_attrs.insert("has_read_effect".to_string(), Attribute::Bool(true));
    let prod_op = m.add_operation(
        fn_blk,
        OpName::new("map", "iter.elementwise"),
        smallvec::smallvec![x_arg],
        vec![(None, full_ty.clone())],
        prod_attrs,
        vec![prod_reg],
        false,
        Location::Unknown,
    );
    let prod_res = m.ops[prod_op].results[0];

    // 2. Build `map.iter.rowwise(%prod_res) {axis = 1, device = "gpu:0"} -> tensor<4x1xf32>`:
    let row_blk = m.create_block(None);
    let r_in = m.add_block_argument(row_blk, None, full_ty.clone());
    let r_out = m.add_block_argument(row_blk, None, row_ty.clone());
    let r_idx = m.add_block_argument(row_blk, None, idx_ty.clone());
    let r_load = m.add_operation(
        row_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![r_in, r_idx],
        vec![(None, f32_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let r_loaded = m.ops[r_load].results[0];
    m.add_operation(
        row_blk,
        OpName::new("map", "store"),
        smallvec::smallvec![r_loaded, r_out, r_idx],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let row_reg = m.create_region(vec![row_blk]);
    let mut row_attrs = indexmap::IndexMap::new();
    row_attrs.insert("axis".to_string(), Attribute::Int(1));
    row_attrs.insert("device".to_string(), Attribute::String("gpu:0".to_string()));
    let row_op = m.add_operation(
        fn_blk,
        OpName::new("map", "iter.rowwise"),
        smallvec::smallvec![prod_res],
        vec![(None, row_ty.clone())],
        row_attrs,
        vec![row_reg],
        false,
        Location::Unknown,
    );
    let row_res = m.ops[row_op].results[0];

    // 3. Build per-row consumer `map.iter.elementwise(%row_res, %bias) -> tensor<4x1xf32>`:
    let cons_blk = m.create_block(None);
    let c_in0 = m.add_block_argument(cons_blk, None, row_ty.clone());
    let c_in1 = m.add_block_argument(cons_blk, None, row_ty.clone());
    let c_out = m.add_block_argument(cons_blk, None, row_ty.clone());
    let c_idx = m.add_block_argument(cons_blk, None, idx_ty);
    let c_load0 = m.add_operation(
        cons_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![c_in0, c_idx],
        vec![(None, f32_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let c_val0 = m.ops[c_load0].results[0];
    let c_load1 = m.add_operation(
        cons_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![c_in1, c_idx],
        vec![(None, f32_ty.clone())],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let c_val1 = m.ops[c_load1].results[0];
    let c_add = m.add_operation(
        cons_blk,
        OpName::new("rmo", "add"),
        smallvec::smallvec![c_val0, c_val1],
        vec![(None, f32_ty)],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let c_sum = m.ops[c_add].results[0];
    m.add_operation(
        cons_blk,
        OpName::new("map", "store"),
        smallvec::smallvec![c_sum, c_out, c_idx],
        vec![],
        indexmap::IndexMap::new(),
        vec![],
        false,
        Location::Unknown,
    );
    let cons_reg = m.create_region(vec![cons_blk]);
    let mut cons_attrs = indexmap::IndexMap::new();
    cons_attrs.insert("device".to_string(), Attribute::String("gpu:0".to_string()));
    let cons_op = m.add_operation(
        fn_blk,
        OpName::new("map", "iter.elementwise"),
        smallvec::smallvec![row_res, bias_arg],
        vec![(None, row_ty)],
        cons_attrs,
        vec![cons_reg],
        false,
        Location::Unknown,
    );
    let cons_res = m.ops[cons_op].results[0];

    // Move `return` to the end of `fn_blk` and make it return `cons_res`:
    m.ops[ret_id].operands[0] = cons_res;
    m.blocks[fn_blk].ops.retain(|&op| op != ret_id);
    m.blocks[fn_blk].ops.push(ret_id);

    let (producers_fused, consumers_fused) = fuse_rowwise_reductions(&mut m);
    assert_eq!(producers_fused, 1);
    assert_eq!(consumers_fused, 1);

    // Only `map.iter.rowwise` and `func.return` remain in `fn_blk`:
    let remaining: Vec<String> = m.blocks[fn_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(remaining, vec!["map.iter.rowwise", "func.return"]);
    assert_eq!(m.ops[row_op].operands.as_slice(), &[x_arg, bias_arg]);
    assert_eq!(m.ops[ret_id].operands[0], row_res);

    // Inside `row_blk`, both producer (`rmo.exp`) and consumer (`rmo.add`) are fused:
    let row_body_ops: Vec<String> = m.blocks[row_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        row_body_ops,
        vec!["map.load", "rmo.exp", "map.load", "rmo.add", "map.store"]
    );
}

#[test]
fn test_add_fallback_shape_functions_inserts_host_transfer_and_maps_params() {
    use castforge::ir::Attribute;
    use castforge::ir::mlir::{Dim, TensorType};
    use castforge::passes::add_fallback_shape_functions;

    let src = concat!(
        "module {\n",
        "  func.func @fallback_shapes(\n",
        "    %x: tensor<512xf32>,\n",
        "    %shape_gpu: tensor<3xi64>\n",
        "  ) -> tensor<16x?x32xf32> {\n",
        "    %y = \"mo.reshape\"(%x, %shape_gpu)",
        " {device = \"gpu:0\", output_param_decls = [\"D_out\"]}",
        " : (tensor<512xf32>, tensor<3xi64>) -> tensor<16x?x32xf32>\n",
        "    return %y : tensor<16x?x32xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(src).unwrap();
    let fn_id = m.functions()[0];
    let fn_blk = m.regions[m.ops[fn_id].regions[0]].blocks[0];
    let reshape_id = m.blocks[fn_blk].ops[0];
    let y_val = m.ops[reshape_id].results[0];

    // Give `%y` a symbolic dimension `Dim::Symbolic("D_out")` at axis 1:
    m.values[y_val].ty = castforge::ir::Type::Tensor(TensorType {
        shape: vec![
            Dim::Static(16),
            Dim::Symbolic("D_out".to_string()),
            Dim::Static(32),
        ],
        dtype: DType::F32,
        encoding: Some("gpu:0".to_string()),
    });

    let (emitted, transfers) =
        add_fallback_shape_functions(&mut m).expect("fallback shape function pass succeeds");
    assert_eq!(emitted, 1);
    assert_eq!(transfers, 1);

    let op_names: Vec<String> = m.blocks[fn_blk]
        .ops
        .iter()
        .map(|&id| m.ops[id].name.full_name())
        .collect();
    assert_eq!(
        op_names,
        vec![
            "mo.chain.create",
            "mo.transfer",
            "mo.invoke_shape_func",
            "mo.reshape",
            "func.return"
        ]
    );

    let xfer_id = m.blocks[fn_blk].ops[1];
    let inv_id = m.blocks[fn_blk].ops[2];
    // `mo.invoke_shape_func` consumes the host-transferred shape operand at index 1:
    assert_eq!(m.ops[inv_id].operands[1], m.ops[xfer_id].results[0]);
    assert_eq!(
        m.ops[inv_id].attributes.get("func_name"),
        Some(&Attribute::String("mo.reshape".to_string()))
    );
    assert_eq!(
        m.ops[inv_id].attributes.get("output_param_decls"),
        Some(&Attribute::Array(vec![
            Attribute::String("F0".to_string()),
            Attribute::String("D_out".to_string()),
            Attribute::String("F1".to_string()),
        ]))
    );
    // `output_param_decls` on `mo.reshape` is cleared:
    assert_eq!(
        m.ops[reshape_id].attributes.get("output_param_decls"),
        Some(&Attribute::Array(vec![]))
    );

    // Verify `mo.custom` without a registered shape function reports a clear error:
    let bad_custom_src = concat!(
        "module {\n",
        "  func.func @unregistered_custom(%x: tensor<4xf32>) -> tensor<?xf32>",
        " attributes {registered_shape_funcs = [\"other_op\"]} {\n",
        "    %y = \"mo.custom\"(%x)",
        " {symbol = \"my_unregistered_op\", output_param_decls = [\"N\"]}",
        " : (tensor<4xf32>) -> tensor<?xf32>\n",
        "    return %y : tensor<?xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut bad_m = parse_mlir(bad_custom_src).unwrap();
    let err = add_fallback_shape_functions(&mut bad_m).unwrap_err();
    assert!(err.contains("@extensibility.register_shape_function(\"my_unregistered_op\")"));
}
