//! End-to-end integration tests for MAP input/output fusion, target-aware weight packing and
//! symbolic parameter range propagation, and multiprocess session/completion/auth subsystems.
#![cfg(feature = "serving")]

use castforge::ir::{Attribute, DType, Location, Module, OpName, TensorType, Type};
use castforge::passes::serving::{
    ApiKeyVerifier, ConstantIntRange, ConvFilterLayout, DeviceHostFuncDispatcher, IpcPolicy,
    MpSessionManager, PackingTargetDesc, SymbolicParamExpr, calculate_matmul_tile_nk,
    can_param_operator_be_nested, consumer_is_pure_reshape, consumer_reads_through_reshape,
    consumer_reads_untransformed, duplication_allowed_for_producer, fuse_map_inputs,
    fuse_map_outputs, fused_kernel_count, grow_opaque_outputs, infer_param_expr_range,
    is_cheap_to_duplicate_map_foreach, is_large_tensor_operand, load_request_client_api_key,
    load_service_api_keys, materialize_pack_conv_filter_shape,
    materialize_pack_conv_transpose_filter_shape, materialize_pack_matmul_b_shape,
    merge_redundant_map_loads, request_client_auth_headers, select_conv_micro_kernel_shape,
    simplify_param_expr_with_ranges, update_graph_signature_from_body,
};
use indexmap::IndexMap;
use smallvec::SmallVec;

fn f32_tensor(dims: &[i64]) -> Type {
    Type::Tensor(TensorType::static_shape(dims, DType::F32))
}

#[test]
fn test_batch38_end_to_end_map_input_and_output_fusion_pipeline() {
    let mut module = Module::new(Some("batch38_fusion_pipeline".to_string()));
    let body_blk = module.body_block();
    let x_in = module.add_block_argument(body_blk, None, f32_tensor(&[1024]));

    // 1. Cheap-to-duplicate producer (`1024 <= 4096` elements) used by TWO consumers.
    let p_blk = module.create_block(None);
    let p_idx = module.add_block_argument(p_blk, None, Type::Scalar(DType::I64));
    let p_ld = module.add_operation(
        p_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![x_in, p_idx],
        vec![(None, Type::Scalar(DType::F32))],
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let p_ld_res = module.ops[p_ld].results[0];
    let p_abs = module.add_operation(
        p_blk,
        OpName::new("rmo", "abs"),
        smallvec::smallvec![p_ld_res],
        vec![(None, Type::Scalar(DType::F32))],
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let p_abs_res = module.ops[p_abs].results[0];
    module.add_operation(
        p_blk,
        OpName::new("map", "yield"),
        smallvec::smallvec![p_abs_res],
        Vec::new(),
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let p_reg = module.create_region(vec![p_blk]);
    let prod_op = module.add_operation(
        body_blk,
        OpName::new("map", "foreach"),
        smallvec::smallvec![x_in],
        vec![(None, f32_tensor(&[1024]))],
        IndexMap::new(),
        vec![p_reg],
        false,
        Location::Unknown,
    );
    let prod_res = module.ops[prod_op].results[0];

    assert!(!is_large_tensor_operand(&f32_tensor(&[1024])));
    assert!(is_cheap_to_duplicate_map_foreach(&module, prod_op));

    // Consumer 1 and Consumer 2 both read `prod_res`.
    let mut consumer_ops = Vec::new();
    for op_short in ["sqrt", "tanh"] {
        let c_blk = module.create_block(None);
        let c_idx = module.add_block_argument(c_blk, None, Type::Scalar(DType::I64));
        let c_ld = module.add_operation(
            c_blk,
            OpName::new("map", "load"),
            smallvec::smallvec![prod_res, c_idx],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let c_ld_res = module.ops[c_ld].results[0];
        let c_compute = module.add_operation(
            c_blk,
            OpName::new("rmo", op_short),
            smallvec::smallvec![c_ld_res],
            vec![(None, Type::Scalar(DType::F32))],
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let c_comp_res = module.ops[c_compute].results[0];
        module.add_operation(
            c_blk,
            OpName::new("map", "yield"),
            smallvec::smallvec![c_comp_res],
            Vec::new(),
            IndexMap::new(),
            Vec::new(),
            false,
            Location::Unknown,
        );
        let c_reg = module.create_region(vec![c_blk]);
        let cons_op = module.add_operation(
            body_blk,
            OpName::new("map", "foreach"),
            smallvec::smallvec![prod_res],
            vec![(None, f32_tensor(&[1024]))],
            IndexMap::new(),
            vec![c_reg],
            false,
            Location::Unknown,
        );
        consumer_ops.push(cons_op);
    }

    assert!(duplication_allowed_for_producer(
        &module,
        body_blk,
        prod_op,
        Some(4)
    ));
    let fused_inputs = fuse_map_inputs(&mut module, Some(4));
    assert_eq!(fused_inputs, 2);
    assert_eq!(module.blocks[body_blk].ops, consumer_ops);
    assert_eq!(fused_kernel_count(&module, consumer_ops[0]), 2);
    assert_eq!(fused_kernel_count(&module, consumer_ops[1]), 2);

    // 2. Output fusion on `consumer_ops[0]` into a downstream untransformed epilogue consumer,
    // plus `grow_opaque_outputs` check.
    let c0_res = module.ops[consumer_ops[0]].results[0];
    let ep_blk = module.create_block(None);
    let ep_idx = module.add_block_argument(ep_blk, None, Type::Scalar(DType::I64));
    let ep_ld = module.add_operation(
        ep_blk,
        OpName::new("map", "load"),
        smallvec::smallvec![c0_res, ep_idx],
        vec![(None, Type::Scalar(DType::F32))],
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let ep_ld_res = module.ops[ep_ld].results[0];
    let ep_neg = module.add_operation(
        ep_blk,
        OpName::new("rmo", "neg"),
        smallvec::smallvec![ep_ld_res],
        vec![(None, Type::Scalar(DType::F32))],
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let ep_neg_res = module.ops[ep_neg].results[0];
    module.add_operation(
        ep_blk,
        OpName::new("map", "yield"),
        smallvec::smallvec![ep_neg_res],
        Vec::new(),
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let ep_reg = module.create_region(vec![ep_blk]);
    let ep_op = module.add_operation(
        body_blk,
        OpName::new("map", "foreach"),
        smallvec::smallvec![c0_res],
        vec![(None, f32_tensor(&[1024]))],
        IndexMap::new(),
        vec![ep_reg],
        false,
        Location::Unknown,
    );

    assert!(consumer_reads_untransformed(&module, ep_op, c0_res));
    assert!(!consumer_is_pure_reshape(&module, ep_op, c0_res));
    assert!(consumer_reads_through_reshape(&module, ep_op, c0_res).is_none());
    assert_eq!(merge_redundant_map_loads(&mut module, ep_reg), 0);

    let fused_outputs = fuse_map_outputs(&mut module);
    assert_eq!(fused_outputs, 1);
    assert_eq!(fused_kernel_count(&module, consumer_ops[0]), 3);

    // Verify `grow_opaque_outputs` appends an output slot and updates `num_outputs`.
    let grown = grow_opaque_outputs(
        &mut module,
        consumer_ops[1],
        std::slice::from_ref(&f32_tensor(&[1024])),
    );
    assert_eq!(grown.len(), 1);
    assert_eq!(
        module.ops[consumer_ops[1]].attributes.get("num_outputs"),
        Some(&Attribute::Int(2))
    );
}

#[test]
fn test_batch39_end_to_end_packing_shapes_and_symbolic_range_propagation() {
    let avx2 = PackingTargetDesc::x86_64_avx2(false);
    let (u_m, u_n) = select_conv_micro_kernel_shape(&avx2, Some(15), 8, 1, 64);
    assert!(u_m >= 1 && (2..=4).contains(&u_n));

    let packed_conv_t = materialize_pack_conv_transpose_filter_shape(
        &[3, 3, 32, 64],
        2,
        ConvFilterLayout::Rscf,
        &DType::F32,
        &avx2,
    )
    .expect("conv_transpose pack");
    assert_eq!(packed_conv_t, vec![8, 3, 3, 32, 8]);

    let macos_arm = PackingTargetDesc::aarch64_neon("apple-m2", true, "macos");
    assert_eq!(
        calculate_matmul_tile_nk(&macos_arm, &DType::F32, &DType::F32, &DType::F32),
        (1, 1)
    );
    assert_eq!(
        materialize_pack_matmul_b_shape(
            &[256, 128],
            true,
            &DType::F32,
            &DType::F32,
            &DType::F32,
            &macos_arm,
        )
        .expect("macos accelerate bypass"),
        vec![128, 256]
    );

    let avx512 = PackingTargetDesc::x86_64_avx512(true);
    let packed_3d = materialize_pack_conv_filter_shape(
        &[2, 3, 3, 16, 96],
        1,
        ConvFilterLayout::Rscf,
        Some(14),
        &DType::F32,
        &avx512,
    )
    .expect("3D conv filter pack");
    assert_eq!(packed_3d, vec![2, 2, 3, 3, 16, 48]);

    let mut sym_bounds = IndexMap::new();
    sym_bounds.insert("seq".to_string(), ConstantIntRange::new(1, 4096));
    sym_bounds.insert("pad".to_string(), ConstantIntRange::new(0, 16));

    let seq = SymbolicParamExpr::Sym("seq".to_string());
    let pad = SymbolicParamExpr::Sym("pad".to_string());
    let mul_nw =
        SymbolicParamExpr::MulNoWrap(Box::new(seq.clone()), Box::new(SymbolicParamExpr::Const(4)));
    assert!(can_param_operator_be_nested(&mul_nw, &pad));
    assert_eq!(
        infer_param_expr_range(&mul_nw, &sym_bounds),
        ConstantIntRange::new(4, 16384)
    );

    let div_expr = SymbolicParamExpr::Div(
        Box::new(SymbolicParamExpr::Add(
            Box::new(seq.clone()),
            Box::new(pad.clone()),
        )),
        Box::new(SymbolicParamExpr::Const(2)),
    );
    assert_eq!(
        infer_param_expr_range(&div_expr, &sym_bounds),
        ConstantIntRange::new(0, 2056)
    );

    let min_expr = SymbolicParamExpr::Min(
        Box::new(SymbolicParamExpr::Add(
            Box::new(pad.clone()),
            Box::new(seq.clone()),
        )),
        Box::new(seq.clone()),
    );
    assert_eq!(simplify_param_expr_with_ranges(&min_expr, &sym_bounds), seq);

    // Verify `update_graph_signature_from_body` propagates refined return types onto `func.func`.
    let mut module = Module::new(Some("sig_sync_test".to_string()));
    let root_blk = module.body_block();
    let fn_blk = module.create_block(None);
    let arg0 = module.add_block_argument(fn_blk, None, f32_tensor(&[16]));
    module.add_operation(
        fn_blk,
        OpName::new("func", "return"),
        smallvec::smallvec![arg0],
        Vec::new(),
        IndexMap::new(),
        Vec::new(),
        false,
        Location::Unknown,
    );
    let fn_reg = module.create_region(vec![fn_blk]);
    let fn_id = module.add_operation(
        root_blk,
        OpName::new("func", "func"),
        SmallVec::new(),
        vec![(None, f32_tensor(&[8]))],
        IndexMap::new(),
        vec![fn_reg],
        false,
        Location::Unknown,
    );
    assert_eq!(update_graph_signature_from_body(&mut module), 1);
    assert_eq!(
        module.ops[fn_id].attributes.get("result_types"),
        Some(&Attribute::Array(vec![Attribute::Type(f32_tensor(&[16]))]))
    );
}

#[test]
fn test_batch40_end_to_end_mp_session_host_completion_and_auth() {
    let mut mgr = MpSessionManager::new(4, 0xCAFE_BABE, 30.0).expect("session manager");
    let sess = mgr.get_or_create("req-42", 10.0);
    let packed: Vec<u8> = (0..8u32).flat_map(u32::to_be_bytes).collect();
    sess.set_tokens(packed).expect("set_tokens");
    sess.begin_lookup("ipc://worker-0");
    let mut owners = IndexMap::new();
    owners.insert("obj://chunk-0".to_string(), 2usize);
    sess.record_prefetch_result(2, vec![10, 11], Some(owners));
    assert_eq!(sess.get_prefetch_owners().get("obj://chunk-0"), Some(&2));
    assert_eq!(sess.get_hashes(0, None).expect("hashes").len(), 2);

    let mut disp = DeviceHostFuncDispatcher::new();
    disp.register("finish_read_prefetched", |payload| {
        if payload == b"ok" {
            Ok(())
        } else {
            Err("invalid payload".to_string())
        }
    });
    disp.record_completion_on_stream(1, "finish_read_prefetched", b"ok".to_vec());
    assert_eq!(disp.stop(), 1);
    assert!(!disp.is_running());

    let keys = load_service_api_keys(&[("k1", "alpha-key")], None, Some("beta-key"))
        .expect("service keys");
    let verifier = ApiKeyVerifier::new(&keys).expect("verifier");
    let client_key =
        load_request_client_api_key(None, &[("k1", "alpha-key")], None, None).expect("client key");
    assert!(verifier.verify_token(client_key.as_deref()));
    let headers =
        request_client_auth_headers(client_key.as_deref(), &[], None, None).expect("headers");
    assert_eq!(
        headers.get("Authorization").map(String::as_str),
        Some("Bearer alpha-key")
    );

    let policy = IpcPolicy::new()
        .with_isolated_ipc(true)
        .with_use_vmm_api(true);
    assert!(policy.isolated_ipc && policy.use_vmm_api);
}
