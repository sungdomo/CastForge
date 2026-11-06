//! Integration tests for tiered KV storage, AEAD frame codecs, CacheBlend fingerprint indexing,
//! fleet instance routing, P2P discovery, and view/transpose sinking passes.
#![cfg(feature = "serving")]

use castforge::ir::{Dim, parse_mlir};
use castforge::passes::{
    AffinitySlotRouter, BLEND_INDEX_MIN_TABLE_SIZE, BLEND_POLY_BASE, BLEND_REORDER_LMIN, Bitmap2D,
    BlendContentFingerprintIndex, BlendFragmentMatch, BlendIndexStats, BlendMatchSpan,
    BlendObjectGroupKind, BlendReadGroupSet, BlendTokenRangeMatcher, BucketHistogram,
    BufferDependencyTree, CBGroupRopeGeometry, CHUNK_COUNT_HISTOGRAM_BOUNDS,
    COORDINATOR_CHECKPOINT_MAGIC, CacheEventBatchSpec, CacheEventEntrySpec, CacheEventOpKind,
    CacheEventTier, CollectiveGraphNodeSpec, ContiguousViewPlan, CoordinatorEventGate,
    CoordinatorKeyDirectory, DaxGenerationSlotArena, DaxPutOutcome, DeviceGraphStage,
    EXCLUDED_ENGINE_GROUP, EngineGroupInfoSpec, EventIngestOutcome, FULL_ATTENTION_WINDOW,
    FdpRawBlockSlotManager, FleetCacheUsageView, FleetInstanceRegistry, FleetMembershipCounters,
    FleetWatermarkEvictionController, KafkaStreamPositionTable, KvStorageKeySpec,
    L2_AEAD_FRAME_OVERHEAD, L2_AEAD_IV_LEN, L2_HKDF_INFO_PREFIX, MemoryTagSpec, MpInstanceSpec,
    MultiTierPrefetchPlan, P2P_DEFAULT_MAX_MISSES, P2PPeerDiscoveryTable, P2PPeerSpec,
    PlacementRecord, PrefetchFetchingPolicy, QuiesceStateMachine, RAW_BLOCK_META_MAGIC,
    RAW_BLOCK_SLOT_MAGIC, RawBlockMetaCheckpointHeader, ReshapeDimMapping, SaltKeyProvider,
    TierMapState, TurboQuantPreset, TurboQuantSerdeConfig, analyze_static_reshape_dim_mapping,
    annotate_squeeze_unsqueeze_reshapes, build_orthonormal_hadamard, chunk_hash_windows_u64,
    classify_blend_read_groups, coalesce_exhaustive_gathers_to_split,
    coalesce_partitioning_slices_to_split, coalesce_unbundle_consumers_to_parallel,
    collect_alloc_uses_in_block, compute_cb_group_rope_geometry, compute_ieee_crc32,
    decode_coordinator_checkpoint_envelope, decode_raw_block_slot_header,
    elide_redundant_mutable_buffer_ops, encode_coordinator_checkpoint_envelope,
    encode_raw_block_meta_checkpoint, encode_raw_block_slot_header, encode_wide_column_row_key,
    engine_group_layer_indices, estimate_l2_framed_size, expand_engine_block_ids,
    expected_contiguous_shape_from_strides, find_squeeze_axis, find_unsqueeze_axis,
    fold_extend_lifetimes_into_async_kernels, fuse_horizontal_shared_input_ops,
    get_engine_group_indices, group_p2p_lookup_keys, hoist_callsite_invariant_subgraph_ops,
    insert_explicit_buffer_frees, interleave_chunk_major_object_keys,
    map_axis_through_static_reshape, merge_parallel_matmuls, num_engine_groups, open_l2_kv_frame,
    plan_cacheblend_token_reorder, plan_device_graph_collective_partition,
    plan_l1_prefetch_retention, plan_multi_tier_prefetch_load, predicate_assert_dependencies,
    recognize_composite_activations, recover_contiguous_or_dim0_padded_view,
    relax_non_aliasing_mutable_chain_sequences, resolve_l1_memory_affinity,
    rolling_hash_windows_u64, seal_l2_kv_frame, select_latest_valid_meta_checkpoint,
    sink_broadcast_through_reduction, sink_split_through_reshape,
    sink_transpose_through_elementwise_and_broadcast, slice_block_ids_per_group,
    solve_lloyd_max_gaussian, swim_reshape_through_slice, unique_blend_token_coverage,
    unpermute_slice,
};

#[test]
fn test_coalesce_partitioning_slices_to_split_contiguity_and_ordering() {
    use castforge::ir::Attribute;

    // `%x` has shape `tensor<4x16xf32>`. Three out-of-order slices along dim 1:
    // - `%s_mid`: [0..4, 4..12] (size 8, start 4)
    // - `%s_hi`:  [0..4, 12..16] (size 4, start 12)
    // - `%s_lo`:  [0..4, 0..4] (size 4, start 0)
    // Together they partition dim 1 (`4 + 8 + 4 = 16`)!
    let mlir = r#"
module @slice_to_split {
  func.func @main(%x: tensor<4x16xf32>) -> tensor<4x4xf32> {
    %s_mid = stablehlo.slice %x
      {start_indices = [0, 4], limit_indices = [4, 12], strides = [1, 1]}
      : (tensor<4x16xf32>) -> tensor<4x8xf32>
    %s_hi = stablehlo.slice %x
      {start_indices = [0, 12], limit_indices = [4, 16], strides = [1, 1]}
      : (tensor<4x16xf32>) -> tensor<4x4xf32>
    %s_lo = stablehlo.slice %x
      {start_indices = [0, 0], limit_indices = [4, 4], strides = [1, 1]}
      : (tensor<4x16xf32>) -> tensor<4x4xf32>
    %out = stablehlo.add %s_lo, %s_hi : tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let groups = coalesce_partitioning_slices_to_split(&mut module);
    assert_eq!(groups, 1);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;

    // `%s_lo` (`ops[2]`) has `start = 0` -> sorted split index 0, sizes `[4, 8, 4]`
    // `%s_mid` (`ops[0]`) has `start = 4` -> sorted split index 1
    // `%s_hi` (`ops[1]`) has `start = 12` -> sorted split index 2
    assert_eq!(
        module.ops[ops[2]]
            .attributes
            .get("castforge.coalesced_split_index"),
        Some(&Attribute::Int(0))
    );
    assert_eq!(
        module.ops[ops[0]]
            .attributes
            .get("castforge.coalesced_split_index"),
        Some(&Attribute::Int(1))
    );
    assert_eq!(
        module.ops[ops[1]]
            .attributes
            .get("castforge.coalesced_split_index"),
        Some(&Attribute::Int(2))
    );
    assert_eq!(
        module.ops[ops[2]]
            .attributes
            .get("castforge.coalesced_split_sizes"),
        Some(&Attribute::DenseIntArray(vec![4, 8, 4]))
    );
}

#[test]
fn test_find_squeeze_unsqueeze_axis_and_reshape_annotation() {
    use castforge::ir::Attribute;

    // Symbolic + static dims example for `sink_reshape_and_rebind_symbolic_dims`:
    // `[D1, 2, 1, D3]` -> `[D1, 2, D3]` => `Some(2)`
    let in_dims = vec![
        Dim::Symbolic("D1".to_string()),
        Dim::Static(2),
        Dim::Static(1),
        Dim::Symbolic("D3".to_string()),
    ];
    let out_dims = vec![
        Dim::Symbolic("D1".to_string()),
        Dim::Static(2),
        Dim::Symbolic("D3".to_string()),
    ];
    assert_eq!(find_squeeze_axis(&in_dims, &out_dims), Some(2));
    assert_eq!(find_unsqueeze_axis(&out_dims, &in_dims), Some(2));

    // Mismatched trailing symbolic dim (`[D1, 2, 1, D4]` -> `[D1, 2, D3]`) => `None`
    let bad_in = vec![
        Dim::Symbolic("D1".to_string()),
        Dim::Static(2),
        Dim::Static(1),
        Dim::Symbolic("D4".to_string()),
    ];
    assert_eq!(find_squeeze_axis(&bad_in, &out_dims), None);

    let mlir = r#"
module @squeeze_unsqueeze {
  func.func @main(%x: tensor<4x1x8xf32>) -> tensor<1x4x8xf32> {
    %sq = stablehlo.reshape %x : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
    %unsq = stablehlo.reshape %sq : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
    return %unsq : tensor<1x4x8xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let count = annotate_squeeze_unsqueeze_reshapes(&mut module);
    assert_eq!(count, 2);

    let func_id = module.functions()[0];
    let func_block = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[func_block].ops;
    assert_eq!(
        module.ops[ops[0]].attributes.get("castforge.squeeze_axis"),
        Some(&Attribute::Int(1))
    );
    assert_eq!(
        module.ops[ops[1]]
            .attributes
            .get("castforge.unsqueeze_axis"),
        Some(&Attribute::Int(0))
    );
}

#[test]
fn test_p2p_peer_discovery_consecutive_misses_and_group_lookup_keys() {
    let mut table = P2PPeerDiscoveryTable::new("node-self", P2P_DEFAULT_MAX_MISSES);
    let peer_a = P2PPeerSpec {
        instance_id: "node-a".to_string(),
        ip: "10.0.0.2".to_string(),
        p2p_advertised_url: "tcp://10.0.0.2:5555".to_string(),
        mq_port: 5556,
    };
    let peer_self = P2PPeerSpec {
        instance_id: "node-self".to_string(),
        ip: "10.0.0.1".to_string(),
        p2p_advertised_url: "tcp://10.0.0.1:5555".to_string(),
        mq_port: 5556,
    };

    // Poll 1: discovers `node-a` (ignores `node-self`) -> creates adapter 1
    assert_eq!(
        table.reconcile_peers(&[peer_self, peer_a.clone()]),
        (vec![1], vec![])
    );
    assert_eq!(table.active_peer_count(), 1);

    // Polls 2 & 3: `node-a` missing -> 2 consecutive misses (< 3), adapter still retained!
    assert_eq!(table.reconcile_peers(&[]), (vec![], vec![]));
    assert_eq!(table.reconcile_peers(&[]), (vec![], vec![]));
    assert_eq!(table.active_peer_count(), 1);

    // Poll 4: 3rd consecutive miss -> adapter 1 is evicted!
    assert_eq!(table.reconcile_peers(&[]), (vec![], vec![1]));
    assert_eq!(table.active_peer_count(), 0);

    // Test `group_p2p_lookup_keys`: equal counts across registered groups [0, 1] succeed,
    // unequal counts or unregistered group IDs return empty!
    let k0 = KvStorageKeySpec {
        model_name: "m".to_string(),
        kv_rank: 0,
        object_group_id: 0,
        chunk_hash: 1,
        cache_salt: String::new(),
    };
    let k1 = KvStorageKeySpec {
        object_group_id: 1,
        chunk_hash: 2,
        ..k0.clone()
    };
    let grouped = group_p2p_lookup_keys(&[k0.clone(), k1.clone()], &[0, 1]);
    assert_eq!(grouped.len(), 2);
    // Unequal group counts (2 keys in group 0, 1 key in group 1) -> treated as all misses
    assert!(group_p2p_lookup_keys(&[k0.clone(), k0, k1], &[0, 1]).is_empty());
}

#[test]
fn test_encode_wide_column_row_key_hash_prefix_and_layer_group_suffix() {
    let key = KvStorageKeySpec {
        model_name: "meta-llama/Llama-3-70B".to_string(),
        kv_rank: 2,
        object_group_id: 1,
        chunk_hash: 0xabcd_ef01_2345_6789,
        cache_salt: "tenant-x".to_string(),
    };
    let row_key = encode_wide_column_row_key(&key, 8, Some(16));
    assert_eq!(
        row_key,
        "abcd@meta-llama/Llama-3-70B@lg8@00000002@1@abcdef0123456789@tenant-x@layer_16"
    );

    let unsalted = KvStorageKeySpec {
        cache_salt: String::new(),
        ..key
    };
    assert_eq!(
        encode_wide_column_row_key(&unsalted, 0, None),
        "abcd@meta-llama/Llama-3-70B@00000002@1@abcdef0123456789"
    );
}

#[test]
fn test_blend_token_range_matcher_sliding_window_and_coverage() {
    let mut matcher = BlendTokenRangeMatcher::new(4, 10, true);
    let stored_tokens = vec![10, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40];
    let chunk_hashes = vec![0x1111, 0x2222, 0x3333];
    // Chunk 2 (`[10, 20, 30, 40]`) duplicates chunk 0's content, so `dedup_content = true`
    // indexes only 2 chunks (`0x1111` at offset 0 and `0x2222` at offset 4).
    assert_eq!(
        matcher.on_new_token_hashes(&stored_tokens, &chunk_hashes, 0),
        2
    );

    // Query contains chunk 1 at offset 2 and chunk 0 at offset 7:
    let query = vec![99, 98, 50, 60, 70, 80, 97, 10, 20, 30, 40, 96];
    let matches = matcher.match_sub_sequence(&query);
    assert_eq!(
        matches,
        vec![
            BlendMatchSpan {
                old_st: 4,
                old_ed: 8,
                cur_st: 2,
                cur_ed: 6,
                chunk_hash: 0x2222,
            },
            BlendMatchSpan {
                old_st: 0,
                old_ed: 4,
                cur_st: 7,
                cur_ed: 11,
                chunk_hash: 0x1111,
            },
        ]
    );
    assert_eq!(unique_blend_token_coverage(&matches), 8);

    // Overlapping spans `[2, 6)` and `[4, 8)` merge to coverage 6 (not 8):
    let overlapping = vec![
        BlendMatchSpan {
            old_st: 0,
            old_ed: 4,
            cur_st: 2,
            cur_ed: 6,
            chunk_hash: 1,
        },
        BlendMatchSpan {
            old_st: 4,
            old_ed: 8,
            cur_st: 4,
            cur_ed: 8,
            chunk_hash: 2,
        },
    ];
    assert_eq!(unique_blend_token_coverage(&overlapping), 6);

    // Evicting chunk `0x2222` removes it from subsequent probes:
    matcher.remove_chunks(&[0x2222]);
    let after_evict = matcher.match_sub_sequence(&query);
    assert_eq!(after_evict.len(), 1);
    assert_eq!(after_evict[0].chunk_hash, 0x1111);
}

#[test]
fn test_plan_cacheblend_token_reorder_glued_pieces_and_pinned_tail() {
    // System prefix (4 tokens): `[1, 2, 3, 4]`
    // Document A (16 tokens): `[100..116]`
    // Glue separator (2 tokens): `[999, 998]`
    // Document B (16 tokens): `[200..216]`
    // Tail question (16 tokens = `BLEND_REORDER_LMIN`): `[500..516]`
    let prefix: Vec<u32> = vec![1, 2, 3, 4];
    let doc_a: Vec<u32> = (100..116).collect();
    let sep: Vec<u32> = vec![999, 998];
    let doc_b: Vec<u32> = (200..216).collect();
    let tail_q: Vec<u32> = (500..500 + BLEND_REORDER_LMIN as u32).collect();

    // Cached candidate H has `[prefix, doc_a, sep, doc_b]`:
    let mut cand_h = Vec::new();
    cand_h.extend_from_slice(&prefix);
    cand_h.extend_from_slice(&doc_a);
    cand_h.extend_from_slice(&sep);
    cand_h.extend_from_slice(&doc_b);

    // Incoming prompt P has `[prefix, doc_b, sep, doc_a, tail_q]` (documents swapped!):
    let mut prompt_p = Vec::new();
    prompt_p.extend_from_slice(&prefix);
    prompt_p.extend_from_slice(&doc_b);
    prompt_p.extend_from_slice(&sep);
    prompt_p.extend_from_slice(&doc_a);
    prompt_p.extend_from_slice(&tail_q);

    // Reorder P with `chunk_size = 8`, `baseline_chunks = 1`, `keep_prefix = 4`:
    let perm = plan_cacheblend_token_reorder(&prompt_p, &[cand_h.clone()], 8, 1, 4)
        .expect("should find a valid token reorder plan");
    assert_eq!(perm.len(), prompt_p.len());

    let reordered: Vec<u32> = perm.iter().map(|&idx| prompt_p[idx]).collect();
    // Reordered prompt starts with exact copy of `cand_h` (`[prefix, doc_a, sep, doc_b]` = 38
    // tokens) and keeps `tail_q` pinned at the very end!
    assert_eq!(&reordered[..cand_h.len()], cand_h.as_slice());
    assert_eq!(
        &reordered[reordered.len() - BLEND_REORDER_LMIN..],
        tail_q.as_slice()
    );
}

#[test]
fn test_coordinator_event_gate_incarnation_fencing_dedup_and_gaps() {
    let mut gate = CoordinatorEventGate::new();

    // 1. First batch (`seq = 1`, `incarnation = 10`) is admitted cleanly:
    let b1 = CacheEventBatchSpec {
        instance_id: "worker-1".to_string(),
        incarnation: 10,
        seq: 1,
        dropped_events: 0,
        tier: CacheEventTier::L1,
    };
    assert_eq!(gate.ingest(&b1), EventIngestOutcome::Admitted);
    // Re-offering `seq = 1` is rejected as `Duplicate`:
    assert_eq!(gate.ingest(&b1), EventIngestOutcome::Duplicate);
    // Older incarnation (`9 < 10`) is rejected as `StaleIncarnation`:
    let stale = CacheEventBatchSpec {
        incarnation: 9,
        seq: 2,
        ..b1.clone()
    };
    assert_eq!(gate.ingest(&stale), EventIngestOutcome::StaleIncarnation);

    // 2. Jumping to `seq = 4` (skipping 2 and 3) with `dropped_events = 5` detects gap & loss:
    let b4 = CacheEventBatchSpec {
        seq: 4,
        dropped_events: 5,
        ..b1.clone()
    };
    assert_eq!(gate.ingest(&b4), EventIngestOutcome::Admitted);
    let c = gate.cursor("worker-1").expect("cursor exists");
    assert!(c.gap_detected);
    assert_eq!(c.missing_batches, 2);
    assert_eq!(c.events_dropped, 5);

    // 3. Newer incarnation (`11 > 10`) fences `worker-1` and resets its cursor:
    let b_restart = CacheEventBatchSpec {
        incarnation: 11,
        seq: 1,
        dropped_events: 0,
        ..b1.clone()
    };
    assert_eq!(gate.ingest(&b_restart), EventIngestOutcome::Admitted);
    assert_eq!(gate.fenced_instances(), &["worker-1".to_string()]);
    assert!(!gate.cursor("worker-1").unwrap().gap_detected);

    // 4. `drop_instance("worker-1")` fences L1 batches at incarnation 11, while L2 passes through:
    gate.drop_instance("worker-1");
    let b_l1_after_drop = CacheEventBatchSpec {
        incarnation: 11,
        seq: 2,
        tier: CacheEventTier::L1,
        ..b1.clone()
    };
    assert_eq!(
        gate.ingest(&b_l1_after_drop),
        EventIngestOutcome::StaleIncarnation
    );
    let b_l2_after_drop = CacheEventBatchSpec {
        incarnation: 11,
        seq: 2,
        tier: CacheEventTier::L2,
        ..b1
    };
    assert_eq!(gate.ingest(&b_l2_after_drop), EventIngestOutcome::Admitted);
}

#[test]
fn test_merge_parallel_matmuls_shared_activation_and_transpose_guard() {
    use castforge::ir::Attribute;

    let mlir = r#"
module @parallel_matmul {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<8x32xf32>) {
    %c1 = stablehlo.dot %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %c2 = stablehlo.dot %a, %c : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    %act1 = stablehlo.tanh %c1 : tensor<4x16xf32>
    %act2 = stablehlo.tanh %c2 : tensor<4x32xf32>
    return %act1 : tensor<4x16xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    let merged = merge_parallel_matmuls(&mut module);
    assert_eq!(merged, 1);

    let func_id = module.functions()[0];
    let block_id = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[block_id].ops;
    // Expected: `stablehlo.concatenate`, merged `stablehlo.dot`, `mo.split`,
    // `stablehlo.tanh`, `stablehlo.tanh`, `func.return`
    assert_eq!(ops.len(), 6);
    assert!(module.ops[ops[0]].is_op("stablehlo", "concatenate"));
    assert!(module.ops[ops[1]].is_op("stablehlo", "dot"));
    assert!(module.ops[ops[2]].is_op("mo", "split"));
    assert_eq!(
        module.ops[ops[2]].attributes.get("split_sizes"),
        Some(&Attribute::DenseIntArray(vec![16, 32]))
    );

    // Verify that when all matmul users only feed directly into `func.return`, merging is skipped:
    let return_only_mlir = r#"
module @parallel_matmul_return_only {
  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<8x32xf32>) {
    %c1 = stablehlo.dot %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %c2 = stablehlo.dot %a, %c : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    return %c1, %c2 : tensor<4x16xf32>, tensor<4x32xf32>
  }
}
"#;
    let mut ret_module = parse_mlir(return_only_mlir).expect("valid MLIR");
    assert_eq!(merge_parallel_matmuls(&mut ret_module), 0);
}

#[test]
fn test_static_reshape_dim_mapping_and_axis_through_reshape() {
    // 1. Static prefix with trailing dynamic dim: [3, 5, 6, -1] -> [15, 3, 2, -1]
    let before = [3, 5, 6, -1];
    let after = [15, 3, 2, -1];
    let mapping = analyze_static_reshape_dim_mapping(&before, &after).expect("supported mapping");
    assert_eq!(
        mapping,
        ReshapeDimMapping {
            forward: vec![vec![0], vec![0], vec![1, 2], vec![]],
            backward: vec![vec![0, 1], vec![2], vec![2], vec![]],
        }
    );

    // Axis 0 (`3`) merges with right neighbor (`5`) into output axis 0 (`15`), which is valid:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 0), Some(0));
    // Axis 1 (`5`) merges with left neighbor (`3`) into output axis 0 (`15`), which is rejected:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 1), None);
    // Axis 2 (`6`) splits into output axes 1 (`3`) and 2 (`2`), mapping to outer split axis 1:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 2), Some(1));
    // Axis 3 (`-1`, dynamic) cannot be statically mapped:
    assert_eq!(map_axis_through_static_reshape(&before, &after, 3), None);

    // 2. Leading dynamic dim (right-to-left scan + reversed order): [-1, 3, 5, 6] -> [-1, 15, 3, 2]
    let dyn_lead_before = [-1, 3, 5, 6];
    let dyn_lead_after = [-1, 15, 3, 2];
    let rev_mapping = analyze_static_reshape_dim_mapping(&dyn_lead_before, &dyn_lead_after)
        .expect("supported right-to-left mapping");
    assert_eq!(
        rev_mapping,
        ReshapeDimMapping {
            forward: vec![vec![], vec![1], vec![1], vec![2, 3]],
            backward: vec![vec![], vec![1, 2], vec![3], vec![3]],
        }
    );
    assert_eq!(
        map_axis_through_static_reshape(&dyn_lead_before, &dyn_lead_after, 3),
        Some(2)
    );

    // 3. Skipping leading unit (`1`) dimensions when mapping an axis:
    let unit_before = [1, 1, 4, 8];
    let unit_after = [32];
    assert_eq!(
        map_axis_through_static_reshape(&unit_before, &unit_after, 0),
        Some(0)
    );

    // 4. Rejections: zero dimensions, mismatched volumes, or static/dynamic mixing:
    assert!(analyze_static_reshape_dim_mapping(&[0, 2], &[0, 4]).is_none());
    assert!(analyze_static_reshape_dim_mapping(&[3, 5], &[16]).is_none());
    assert!(analyze_static_reshape_dim_mapping(&[3, -1, 5], &[15, -1]).is_none());
}

#[test]
fn test_coalesce_exhaustive_gathers_to_split_dominance_and_partial_rejection() {
    use castforge::ir::Attribute;

    let mlir = r#"
module @gathers_to_split {
  func.func @main(%x: tensor<3x8xf32>) {
    %g2 = mo.gather %x {axis = 0, index = 2} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %g0 = mo.gather %x {axis = 0, index = 0} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %g1 = mo.gather %x {axis = 0, index = 1} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %sum01 = stablehlo.add %g0, %g1 : tensor<1x8xf32>
    %sum012 = stablehlo.add %sum01, %g2 : tensor<1x8xf32>
    return %sum012 : tensor<1x8xf32>
  }
}
"#;
    let mut module = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(coalesce_exhaustive_gathers_to_split(&mut module), 1);

    let func_id = module.functions()[0];
    let block_id = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let ops = &module.blocks[block_id].ops;
    // The 3 gathers are replaced by a single `mo.split` at the dominant position (index 0):
    assert_eq!(ops.len(), 4);
    assert!(module.ops[ops[0]].is_op("mo", "split"));
    assert_eq!(
        module.ops[ops[0]].attributes.get("split_sizes"),
        Some(&Attribute::DenseIntArray(vec![1, 1, 1]))
    );
    // Verify %g0, %g1, %g2 map to split results #0, #1, #2 in index order (not block order):
    let split_results = &module.ops[ops[0]].results;
    assert_eq!(split_results.len(), 3);
    assert_eq!(module.ops[ops[1]].operands[0], split_results[0]);
    assert_eq!(module.ops[ops[1]].operands[1], split_results[1]);
    assert_eq!(module.ops[ops[2]].operands[1], split_results[2]);

    // Partial gather coverage (missing index 2 of 3) must NOT be rewritten:
    let partial_mlir = r#"
module @partial_gathers {
  func.func @main(%x: tensor<3x8xf32>) {
    %g0 = mo.gather %x {axis = 0, index = 0} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %g1 = mo.gather %x {axis = 0, index = 1} : (tensor<3x8xf32>) -> tensor<1x8xf32>
    %sum01 = stablehlo.add %g0, %g1 : tensor<1x8xf32>
    return %sum01 : tensor<1x8xf32>
  }
}
"#;
    let mut partial_mod = parse_mlir(partial_mlir).expect("valid MLIR");
    assert_eq!(coalesce_exhaustive_gathers_to_split(&mut partial_mod), 0);
}

#[test]
fn test_recognize_composite_activations_sigmoid_silu_gelu_variants() {
    // 1. Decomposed `x * (1 / (1 + exp(-x)))` folds `1 / (1 + exp(-x))` -> `mo.sigmoid`
    //    and then `x * mo.sigmoid(x)` -> `mo.silu(x)` in a single pass!
    let silu_mlir = r#"
module @decomposed_silu {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %neg = stablehlo.negate %x : tensor<4xf32>
    %exp = stablehlo.exponential %neg : tensor<4xf32>
    %den = stablehlo.add %one, %exp : tensor<4xf32>
    %sig = stablehlo.divide %one, %den : tensor<4xf32>
    %silu = stablehlo.multiply %x, %sig : tensor<4xf32>
    return %silu : tensor<4xf32>
  }
}
"#;
    let mut silu_mod = parse_mlir(silu_mlir).expect("valid MLIR");
    assert_eq!(recognize_composite_activations(&mut silu_mod), 2);
    let func_id = silu_mod.functions()[0];
    let blk_id = silu_mod.regions[silu_mod.ops[func_id].regions[0]].blocks[0];
    let ops = &silu_mod.blocks[blk_id].ops;
    assert!(ops.iter().any(|&id| silu_mod.ops[id].is_op("mo", "silu")));
    assert!(
        !ops.iter()
            .any(|&id| silu_mod.ops[id].is_op("mo", "sigmoid"))
    );

    // 2. Quick-GELU (`x * sigmoid(1.702 * x)`) and Erf-GELU (`0.5 * x * (1 + erf(x / sqrt(2)))`):
    let gelu_mlir = r#"
module @decomposed_gelus {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c1702 = stablehlo.constant dense<1.702> : tensor<4xf32>
    %half = stablehlo.constant dense<0.5> : tensor<4xf32>
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %sqrt2 = stablehlo.constant dense<1.41421356> : tensor<4xf32>
    %scaled_q = stablehlo.multiply %c1702, %x : tensor<4xf32>
    %sig_q = stablehlo.logistic %scaled_q : tensor<4xf32>
    %qgelu = stablehlo.multiply %x, %sig_q : tensor<4xf32>
    %hx = stablehlo.multiply %half, %qgelu : tensor<4xf32>
    %div_s2 = stablehlo.divide %qgelu, %sqrt2 : tensor<4xf32>
    %erf_v = stablehlo.erf %div_s2 : tensor<4xf32>
    %one_p = stablehlo.add %one, %erf_v : tensor<4xf32>
    %gelu = stablehlo.multiply %hx, %one_p : tensor<4xf32>
    return %gelu : tensor<4xf32>
  }
}
"#;
    let mut gelu_mod = parse_mlir(gelu_mlir).expect("valid MLIR");
    assert_eq!(recognize_composite_activations(&mut gelu_mod), 2);
    let g_func = gelu_mod.functions()[0];
    let g_blk = gelu_mod.regions[gelu_mod.ops[g_func].regions[0]].blocks[0];
    let g_ops = &gelu_mod.blocks[g_blk].ops;
    assert!(
        g_ops
            .iter()
            .any(|&id| gelu_mod.ops[id].is_op("mo", "gelu_quick"))
    );
    assert!(g_ops.iter().any(|&id| gelu_mod.ops[id].is_op("mo", "gelu")));

    // 3. Tanh-GELU (`0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * (x * x * x))))`):
    let tanh_gelu_mlir = r#"
module @decomposed_gelu_tanh {
  func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<4xf32>
    %one = stablehlo.constant dense<1.0> : tensor<4xf32>
    %c044 = stablehlo.constant dense<0.044715> : tensor<4xf32>
    %c797 = stablehlo.constant dense<0.79788456> : tensor<4xf32>
    %hx = stablehlo.multiply %half, %x : tensor<4xf32>
    %x2 = stablehlo.multiply %x, %x : tensor<4xf32>
    %x3 = stablehlo.multiply %x2, %x : tensor<4xf32>
    %term = stablehlo.multiply %c044, %x3 : tensor<4xf32>
    %poly = stablehlo.add %x, %term : tensor<4xf32>
    %targ = stablehlo.multiply %c797, %poly : tensor<4xf32>
    %th = stablehlo.tanh %targ : tensor<4xf32>
    %one_p = stablehlo.add %one, %th : tensor<4xf32>
    %gtanh = stablehlo.multiply %hx, %one_p : tensor<4xf32>
    return %gtanh : tensor<4xf32>
  }
}
"#;
    let mut gt_mod = parse_mlir(tanh_gelu_mlir).expect("valid MLIR");
    assert_eq!(recognize_composite_activations(&mut gt_mod), 1);
    let gt_func = gt_mod.functions()[0];
    let gt_blk = gt_mod.regions[gt_mod.ops[gt_func].regions[0]].blocks[0];
    let gt_ops = &gt_mod.blocks[gt_blk].ops;
    assert!(
        gt_ops
            .iter()
            .any(|&id| gt_mod.ops[id].is_op("mo", "gelu_tanh"))
    );
}

#[test]
fn test_sink_transpose_through_unary_binary_and_rank_increasing_broadcast() {
    use castforge::ir::Attribute;

    // Verify `unpermute_slice` inverse property (`perm = [2, 0, 1]` -> `inverse = [1, 2, 0]`):
    assert_eq!(
        unpermute_slice(&[10, 20, 30], &[2, 0, 1]),
        Some(vec![20, 30, 10])
    );

    // 1. Binary + Unary cascade:
    //    `%t1 = transpose(%x, [1, 0])`, `%t2 = transpose(%y, [1, 0])`
    //    `%sum = add(%t1, %t2)` -> sinks to `add(%x, %y)` followed by 1 transpose
    //    `%act = tanh(%sum)`    -> sinks that transpose past `tanh` as well!
    let bin_unary_mlir = r#"
module @sink_transpose_bin_unary {
  func.func @main(%x: tensor<4x8xf32>, %y: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %t1 = stablehlo.transpose %x {permutation = [1, 0]} : (tensor<4x8xf32>) -> tensor<8x4xf32>
    %t2 = stablehlo.transpose %y {permutation = [1, 0]} : (tensor<4x8xf32>) -> tensor<8x4xf32>
    %sum = stablehlo.add %t1, %t2 : tensor<8x4xf32>
    %act = stablehlo.tanh %sum : tensor<8x4xf32>
    return %act : tensor<8x4xf32>
  }
}
"#;
    let mut bu_mod = parse_mlir(bin_unary_mlir).expect("valid MLIR");
    assert_eq!(
        sink_transpose_through_elementwise_and_broadcast(&mut bu_mod),
        2
    );
    let func_id = bu_mod.functions()[0];
    let blk_id = bu_mod.regions[bu_mod.ops[func_id].regions[0]].blocks[0];
    let ops = &bu_mod.blocks[blk_id].ops;
    assert_eq!(ops.len(), 4);
    assert!(bu_mod.ops[ops[0]].is_op("stablehlo", "add"));
    assert!(bu_mod.ops[ops[1]].is_op("stablehlo", "tanh"));
    assert!(bu_mod.ops[ops[2]].is_op("stablehlo", "transpose"));

    // 2. Rank-increasing broadcast (`[1, 3, 2]` -> transpose `[0, 2, 1]` -> `[1, 2, 3]`
    //    -> broadcast `[9, 1, 2, 3]`):
    //    After sinking transpose: broadcast produces `[9, 1, 3, 2]` and transpose uses
    //    padded permutation `[0, 1, 3, 2]` to produce `[9, 1, 2, 3]`!
    let bcast_mlir = r#"
module @sink_transpose_bcast {
  func.func @main(%x: tensor<1x3x2xf32>) -> tensor<9x1x2x3xf32> {
    %t = stablehlo.transpose %x {permutation = [0, 2, 1]}
      : (tensor<1x3x2xf32>) -> tensor<1x2x3xf32>
    %b = mo.static.broadcast_to %t : (tensor<1x2x3xf32>) -> tensor<9x1x2x3xf32>
    return %b : tensor<9x1x2x3xf32>
  }
}
"#;
    let mut bc_mod = parse_mlir(bcast_mlir).expect("valid MLIR");
    assert_eq!(
        sink_transpose_through_elementwise_and_broadcast(&mut bc_mod),
        1
    );
    let b_func = bc_mod.functions()[0];
    let b_blk = bc_mod.regions[bc_mod.ops[b_func].regions[0]].blocks[0];
    let b_ops = &bc_mod.blocks[b_blk].ops;
    assert_eq!(b_ops.len(), 3);
    assert!(bc_mod.ops[b_ops[0]].is_op("mo", "static.broadcast_to"));
    assert!(bc_mod.ops[b_ops[1]].is_op("stablehlo", "transpose"));
    let inter_val = bc_mod.ops[b_ops[0]].results[0];
    assert_eq!(
        bc_mod.values[inter_val].ty.as_tensor().unwrap().shape,
        vec![
            Dim::Static(9),
            Dim::Static(1),
            Dim::Static(3),
            Dim::Static(2)
        ]
    );
    assert_eq!(
        bc_mod.ops[b_ops[1]].attributes.get("permutation"),
        Some(&Attribute::DenseIntArray(vec![0, 1, 3, 2]))
    );
}

#[test]
fn test_classify_blend_read_groups_and_cb_group_rope_geometry() {
    // 1. Single-group fused layout maps both prefix and blend legs to group 0:
    assert_eq!(
        classify_blend_read_groups(1, &[]),
        Some(BlendReadGroupSet {
            blend_gids: vec![0],
            prefix_gids: vec![0],
            recurrent_gids: vec![],
            attn_gid: 0,
        })
    );

    // 2. Multi-group hybrid layout (Recurrent=0, Attention=1, Aux=2):
    let kinds = [
        BlendObjectGroupKind::Recurrent,
        BlendObjectGroupKind::Attention,
        BlendObjectGroupKind::Aux,
    ];
    assert_eq!(
        classify_blend_read_groups(3, &kinds),
        Some(BlendReadGroupSet {
            blend_gids: vec![1, 2],
            prefix_gids: vec![0, 1],
            recurrent_gids: vec![0],
            attn_gid: 1,
        })
    );
    // Rejects multiple attention groups or >1 aux group:
    assert!(
        classify_blend_read_groups(
            2,
            &[
                BlendObjectGroupKind::Attention,
                BlendObjectGroupKind::Attention
            ]
        )
        .is_none()
    );

    // 3. Chunk-major key interleaving (`expansion = 2` keys per chunk across 2 groups):
    let g0 = vec!["c0_g0_a", "c0_g0_b", "c1_g0_a", "c1_g0_b"];
    let g1 = vec!["c0_g1_a", "c0_g1_b", "c1_g1_a", "c1_g1_b"];
    assert_eq!(
        interleave_chunk_major_object_keys(&[g0, g1], 2),
        Some(vec![
            "c0_g0_a", "c0_g0_b", "c0_g1_a", "c0_g1_b", "c1_g0_a", "c1_g0_b", "c1_g1_a", "c1_g1_b"
        ])
    );

    // 4. RoPE geometry: split K/V, fused-packed K/V, and MLA `[content | rope]`:
    assert_eq!(
        compute_cb_group_rope_geometry(16, 16, 2, 1024, 128, false, None, None),
        Some(CBGroupRopeGeometry {
            fused_packed: false,
            per_head_elems: 128,
            num_heads: 8,
            rot_offset: 0,
        })
    );
    assert_eq!(
        compute_cb_group_rope_geometry(16, 16, 1, 2048, 128, true, Some(256), None),
        Some(CBGroupRopeGeometry {
            fused_packed: true,
            per_head_elems: 256,
            num_heads: 8,
            rot_offset: 0,
        })
    );
    // MLA latent (`kv_size = 1`, `hidden_dim = 576 = 512 content + 64 rope`):
    assert_eq!(
        compute_cb_group_rope_geometry(16, 16, 1, 576, 64, false, None, Some((512, 64))),
        Some(CBGroupRopeGeometry {
            fused_packed: false,
            per_head_elems: 576,
            num_heads: 1,
            rot_offset: 512,
        })
    );
    // Rejects MLA window when `kv_size != 1` or window does not end the row:
    assert!(
        compute_cb_group_rope_geometry(16, 16, 2, 576, 64, false, None, Some((512, 64))).is_none()
    );
    assert!(
        compute_cb_group_rope_geometry(16, 16, 1, 576, 64, false, None, Some((500, 64))).is_none()
    );
}

#[test]
fn test_dax_generation_slot_arena_read_borrows_deferred_reclaim_and_shrink() {
    let mut arena = DaxGenerationSlotArena::new(4 * 4096, 4096).expect("4-slot DAX arena");
    assert_eq!(arena.max_slots(), 4);

    // Insert 3 keys into slots 0, 1, 2:
    assert_eq!(arena.put(100, 4096), DaxPutOutcome::Inserted);
    assert_eq!(arena.put(200, 4096), DaxPutOutcome::Inserted);
    assert_eq!(arena.put(300, 4096), DaxPutOutcome::Inserted);
    assert_eq!(arena.put(100, 4096), DaxPutOutcome::AlreadyExists);
    assert_eq!(arena.put(999, 8192), DaxPutOutcome::InvalidSize);

    // Reserve read on key 200 (slot 1, generation 1):
    let (reservations, hits) = arena.reserve_reads(&[200, 404], false);
    assert_eq!(hits, vec![true, false]);
    assert_eq!(reservations.len(), 1);
    assert_eq!(reservations[0].slot_id, 1);
    assert_eq!(reservations[0].generation, 1);

    // Evict key 200 while borrowed by an active read:
    // Key 200 is removed from the index immediately, but slot 1 enters `pending_free`!
    assert_eq!(arena.delete_many(&[200], false), vec![true]);
    assert_eq!(arena.exists_many(&[200], false), vec![false]);

    // `usage()` reports 3/4 = 0.75 current usage (slot 1 still borrowed),
    // and 2/4 = 0.50 post-eviction usage!
    let (cur_u, post_u) = arena.usage();
    assert!((cur_u - 0.75).abs() < 1e-9);
    assert!((post_u - 0.50).abs() < 1e-9);

    // Finalizing the read drains `borrow_count` to 0 and recycles slot 1 into `free_slots`:
    arena.finalize_reads(&reservations);
    let (cur_after, post_after) = arena.usage();
    assert!((cur_after - 0.50).abs() < 1e-9);
    assert!((post_after - 0.50).abs() < 1e-9);

    // Inserting key 400 reuses recycled slot 1 with incremented generation 2:
    let (out, slot_info) = arena.begin_put(400, 2048);
    assert_eq!(out, DaxPutOutcome::Inserted);
    assert_eq!(slot_info, Some((1, 2, 4096)));
    assert!(arena.finalize_put(400, false));

    // Highest occupied slot is slot 2 (`0, 1, 2`), so shrinking to 3 slots is valid,
    // while shrinking to 2 slots is rejected because slot 2 is live:
    assert!(arena.can_shrink_to(3));
    assert!(!arena.can_shrink_to(2));
}

#[test]
fn test_coordinator_key_directory_l1_fencing_shared_l2_and_histogram() {
    let mut dir = CoordinatorKeyDirectory::new();

    // Worker A stores keys 10 and 20 in L1 ("cpu") and key 20 in shared L2 ("redis"):
    dir.consume_batch(
        "worker-a",
        1,
        CacheEventTier::L1,
        "cpu",
        false,
        CacheEventOpKind::Store,
        100,
        &[
            CacheEventEntrySpec {
                key: 10,
                size_bytes: 1024,
            },
            CacheEventEntrySpec {
                key: 20,
                size_bytes: 2048,
            },
        ],
    );
    dir.consume_batch(
        "worker-a",
        1,
        CacheEventTier::L2,
        "redis",
        true,
        CacheEventOpKind::Store,
        101,
        &[CacheEventEntrySpec {
            key: 20,
            size_bytes: 2048,
        }],
    );
    // Worker B re-stores key 20 into the same shared L2 ("redis") pool -> upserts in place!
    dir.consume_batch(
        "worker-b",
        1,
        CacheEventTier::L2,
        "redis",
        true,
        CacheEventOpKind::Store,
        102,
        &[CacheEventEntrySpec {
            key: 20,
            size_bytes: 2048,
        }],
    );
    dir.consume_batch(
        "worker-b",
        1,
        CacheEventTier::L1,
        "cpu",
        false,
        CacheEventOpKind::Access,
        105,
        &[CacheEventEntrySpec {
            key: 20,
            size_bytes: 0,
        }],
    );

    assert_eq!(dir.num_keys(), 2);
    assert_eq!(dir.tier_stats(), (2, 3072, 1, 2048));
    assert_eq!(dir.get_access_counts(&[10, 20, 99]), vec![0, 1, 0]);

    // Fencing `worker-a` drops its 2 L1 placements (removing key 10 completely),
    // while key 20's L2 placement survives!
    assert_eq!(dir.fence_instance("worker-a"), 2);
    assert_eq!(dir.num_keys(), 1);
    assert_eq!(dir.tier_stats(), (0, 0, 1, 2048));
    let looked_up = dir.lookup(&[10, 20]);
    assert!(looked_up[0].is_empty());
    assert_eq!(
        looked_up[1],
        vec![PlacementRecord {
            instance_id: "worker-b".to_string(),
            incarnation: 1,
            tier: CacheEventTier::L2,
            backend: "redis".to_string(),
            size_bytes: 2048,
            shared: true,
        }]
    );

    // Verify `BucketHistogram` observation and Prometheus cumulative bucket series:
    let mut hist = BucketHistogram::with_bounds(&[1.0, 4.0, 16.0]).expect("valid bounds");
    assert!(BucketHistogram::with_bounds(&[4.0, 2.0]).is_none());
    assert!(BucketHistogram::with_bounds(CHUNK_COUNT_HISTOGRAM_BOUNDS).is_some());
    hist.observe(0.5);
    hist.observe(4.0);
    hist.observe(32.0);
    assert_eq!(hist.count(), 3);
    assert!((hist.sum() - 36.5).abs() < 1e-9);
    assert_eq!(
        hist.cumulative_buckets(),
        vec![
            ("1".to_string(), 1),
            ("4".to_string(), 2),
            ("16".to_string(), 2),
            ("+Inf".to_string(), 3),
        ]
    );
}

#[test]
fn test_fdp_raw_block_slot_manager_and_double_buffered_meta_checkpoint() {
    // 1. Verify raw-block per-slot header codec (`LMCBLK01` + slot_identity + payload_len):
    assert_eq!(RAW_BLOCK_SLOT_MAGIC, b"LMCBLK01");
    assert_eq!(RAW_BLOCK_META_MAGIC, b"LMCIDX01");
    assert!(encode_raw_block_slot_header(0x1122_3344_5566_7788, 65536, 16).is_none());
    let slot_hdr =
        encode_raw_block_slot_header(0x1122_3344_5566_7788, 65536, 4096).expect("valid header");
    assert_eq!(slot_hdr.len(), 4096);
    assert_eq!(
        decode_raw_block_slot_header(&slot_hdr),
        Some((0x1122_3344_5566_7788, 65536))
    );

    // 2. Verify double-buffered metadata checkpoint container selection and CRC32 validation:
    let payload_v1 = br#"{"next_slot":3,"entries":[0,2]}"#;
    let payload_v2 = br#"{"next_slot":4,"entries":[0,2,3]}"#;
    let container_a =
        encode_raw_block_meta_checkpoint(1, 10, payload_v1, 4096).expect("valid checkpoint a");
    let container_b =
        encode_raw_block_meta_checkpoint(1, 11, payload_v2, 4096).expect("valid checkpoint b");

    // Both containers valid -> selects higher `seq = 11`:
    let (hdr, chosen_payload) = select_latest_valid_meta_checkpoint(
        &[(0, container_a.clone()), (65536, container_b.clone())],
        4096,
        1,
    )
    .expect("should pick valid checkpoint");
    assert_eq!(
        hdr,
        RawBlockMetaCheckpointHeader {
            seq: 11,
            payload_len: payload_v2.len() as u64,
            crc32: compute_ieee_crc32(payload_v2),
            container_offset: 65536,
        }
    );
    assert_eq!(chosen_payload, payload_v2);

    // Corrupt a byte in container B's payload -> falls back to container A (`seq = 10`):
    let mut corrupted_b = container_b;
    corrupted_b[4100] ^= 0x5A;
    let (fallback_hdr, fallback_payload) =
        select_latest_valid_meta_checkpoint(&[(0, container_a), (65536, corrupted_b)], 4096, 1)
            .expect("should fall back to container A");
    assert_eq!(fallback_hdr.seq, 10);
    assert_eq!(fallback_hdr.container_offset, 0);
    assert_eq!(fallback_payload, payload_v1);

    // 3. Verify NVMe FDP placement-affinity slot recycling and crash-safe free-list rebuild:
    let mut mgr = FdpRawBlockSlotManager::new(4, true);
    let s0 = mgr.allocate_slot(Some(100)).expect("slot 0");
    let s1 = mgr.allocate_slot(Some(200)).expect("slot 1");
    let s2 = mgr.allocate_slot(Some(100)).expect("slot 2");
    assert_eq!((s0, s1, s2), (0, 1, 2));
    assert_eq!(mgr.slot_placement_id(s1), Some(200));

    // Free slots 0 (PID 100) and 1 (PID 200), then request PID 100 -> affinity hit on slot 0!
    mgr.free_slot(s0);
    mgr.free_slot(s1);
    let s_reused_100 = mgr.allocate_slot(Some(100)).expect("affinity hit");
    assert_eq!(s_reused_100, 0);
    assert_eq!(mgr.stats(), (3, 1, 1, 0));

    // Request PID 300 while slot 1 (PID 200) is in free list -> fallback recycle of slot 1!
    let s_fallback_300 = mgr.allocate_slot(Some(300)).expect("fallback recycle");
    assert_eq!(s_fallback_300, 1);
    assert_eq!(mgr.slot_placement_id(1), Some(300));
    assert_eq!(mgr.stats(), (3, 0, 1, 1));

    // Simulate crash recovery where `next_slot = 4` had uncommitted in-flight slot 1 and freed
    // slot 3, with only slots `[0, 2]` committed in the checkpoint:
    assert!(mgr.rebuild_free_slots_from_committed(4, &[0, 2]));
    assert_eq!(mgr.stats(), (4, 2, 1, 1));
    assert_eq!(mgr.allocate_slot(None), Some(3));
    assert_eq!(mgr.allocate_slot(None), Some(1));
    assert_eq!(mgr.allocate_slot(None), None);
}

#[test]
fn test_swim_reshape_through_slice_and_sink_split_through_reshape() {
    // 1. Verify `swim_reshape_through_slice`:
    //    `slice(tensor<8x6xf32> -> tensor<4x6xf32>, start=[2, 0], limit=[6, 6], strides=[1, 1])`
    //    followed by `reshape(tensor<4x6xf32> -> tensor<2x2x6xf32>)`
    //    commutes into `reshape(tensor<8x6xf32> -> tensor<4x2x6xf32>)` followed by
    //    `slice(tensor<4x2x6xf32> -> tensor<2x2x6xf32>, start=[1, 0, 0], limit=[3, 2, 6])`!
    let mlir_slice_reshape = concat!(
        "module @swim_reshape_slice {\n",
        "  func.func @main(%x: tensor<8x6xf32>) -> tensor<2x2x6xf32> {\n",
        "    %s = stablehlo.slice %x ",
        "{start_indices = [2, 0], limit_indices = [6, 6], strides = [1, 1]} ",
        ": (tensor<8x6xf32>) -> tensor<4x6xf32>\n",
        "    %r = stablehlo.reshape %s : (tensor<4x6xf32>) -> tensor<2x2x6xf32>\n",
        "    return %r : tensor<2x2x6xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m1 = parse_mlir(mlir_slice_reshape).expect("valid MLIR");
    assert_eq!(swim_reshape_through_slice(&mut m1), 1);
    let f1 = m1.functions()[0];
    let b1 = m1.regions[m1.ops[f1].regions[0]].blocks[0];
    let ops1 = &m1.blocks[b1].ops;
    assert!(m1.ops[ops1[0]].is_op("stablehlo", "reshape"));
    assert!(m1.ops[ops1[1]].is_op("stablehlo", "slice"));
    let mid_v = m1.ops[ops1[0]].results[0];
    assert_eq!(
        m1.values[mid_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(4), Dim::Static(2), Dim::Static(6)]
    );

    // 2. Verify `sink_split_through_reshape`:
    //    `mo.split %x {axis = 0} : (tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)`
    //    followed by two identical `stablehlo.reshape` ops `tensor<2x6xf32> -> tensor<2x2x3xf32>`
    //    commutes into a single `stablehlo.reshape` (`tensor<4x6xf32> -> tensor<4x2x3xf32>`)
    //    followed by `mo.split` along `axis = 0`!
    let mlir_split_reshape = r#"
module @sink_split_reshape {
  func.func @main(%x: tensor<4x6xf32>) -> tensor<2x2x3xf32> {
    %s0, %s1 = mo.split %x {axis = 0} : (tensor<4x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xf32>)
    %r0 = stablehlo.reshape %s0 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
    %r1 = stablehlo.reshape %s1 : (tensor<2x6xf32>) -> tensor<2x2x3xf32>
    %sum = stablehlo.add %r0, %r1 : tensor<2x2x3xf32>
    return %sum : tensor<2x2x3xf32>
  }
}
"#;
    let mut m2 = parse_mlir(mlir_split_reshape).expect("valid MLIR");
    assert_eq!(sink_split_through_reshape(&mut m2), 1);
    let f2 = m2.functions()[0];
    let b2 = m2.regions[m2.ops[f2].regions[0]].blocks[0];
    let ops2 = &m2.blocks[b2].ops;
    assert_eq!(ops2.len(), 4);
    assert!(m2.ops[ops2[0]].is_op("stablehlo", "reshape"));
    assert!(m2.ops[ops2[1]].is_op("mo", "split"));
    let pre_split_v = m2.ops[ops2[0]].results[0];
    assert_eq!(
        m2.values[pre_split_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(4), Dim::Static(2), Dim::Static(3)]
    );
}

#[test]
fn test_sink_broadcast_through_reduction_non_broadcast_and_idempotent() {
    // 1. Non-broadcasted axis (`axis = 2` on `tensor<3x4x8xf32>` broadcast from `tensor<4x8xf32>`):
    //    Commutes into `mo.reduce_add %x {axis = 1} : (tensor<4x8xf32>) -> tensor<4x1xf32>`
    //    followed by `mo.static.broadcast_to -> tensor<3x4x1xf32>`!
    let mlir_non_bcast = r#"
module @sink_bcast_reduce_non_bcast {
  func.func @main(%x: tensor<4x8xf32>) -> tensor<3x4x1xf32> {
    %b = mo.static.broadcast_to %x : (tensor<4x8xf32>) -> tensor<3x4x8xf32>
    %r = mo.reduce_add %b {axis = 2} : (tensor<3x4x8xf32>) -> tensor<3x4x1xf32>
    return %r : tensor<3x4x1xf32>
  }
}
"#;
    let mut m1 = parse_mlir(mlir_non_bcast).expect("valid MLIR");
    assert_eq!(sink_broadcast_through_reduction(&mut m1), 1);
    let f1 = m1.functions()[0];
    let b1 = m1.regions[m1.ops[f1].regions[0]].blocks[0];
    let ops1 = &m1.blocks[b1].ops;
    assert_eq!(ops1.len(), 3);
    assert!(m1.ops[ops1[0]].is_op("mo", "reduce_add"));
    assert_eq!(
        m1.ops[ops1[0]].attributes.get("axis"),
        Some(&castforge::ir::Attribute::Int(1))
    );
    assert!(m1.ops[ops1[1]].is_op("mo", "static.broadcast_to"));
    let mid_v = m1.ops[ops1[0]].results[0];
    assert_eq!(
        m1.values[mid_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(4), Dim::Static(1)]
    );

    // 2. Broadcasted axis (`axis = 1` on `tensor<3x4x8xf32>` broadcast from `tensor<1x8xf32>`)
    //    with idempotent `mo.reduce_max`: folds directly into a single `mo.static.broadcast_to`!
    let mlir_idempotent = r#"
module @sink_bcast_reduce_idempotent {
  func.func @main(%x: tensor<1x8xf32>) -> tensor<3x1x8xf32> {
    %b = mo.static.broadcast_to %x : (tensor<1x8xf32>) -> tensor<3x4x8xf32>
    %r = mo.reduce_max %b {axis = 1} : (tensor<3x4x8xf32>) -> tensor<3x1x8xf32>
    return %r : tensor<3x1x8xf32>
  }
}
"#;
    let mut m2 = parse_mlir(mlir_idempotent).expect("valid MLIR");
    assert_eq!(sink_broadcast_through_reduction(&mut m2), 1);
    let f2 = m2.functions()[0];
    let b2 = m2.regions[m2.ops[f2].regions[0]].blocks[0];
    let ops2 = &m2.blocks[b2].ops;
    assert_eq!(ops2.len(), 2);
    assert!(m2.ops[ops2[0]].is_op("mo", "static.broadcast_to"));
    let out_v = m2.ops[ops2[0]].results[0];
    assert_eq!(
        m2.values[out_v].ty.as_tensor().unwrap().shape,
        vec![Dim::Static(3), Dim::Static(1), Dim::Static(8)]
    );
}

#[test]
fn test_fold_extend_lifetimes_and_device_graph_collective_partition() {
    // 1. Verify `fold_extend_lifetimes_into_async_kernels`:
    //    `%c1 = mogg.async_exec %c0, %a` followed by `%c2 = mgp.extend_lifetimes %c1, %b, %a`
    //    and a no-op `%c3 = mgp.extend_lifetimes %c2` folds `%b` into `mogg.async_exec`
    //    (deduplicating `%a`) and forwards `%c3` -> `%c1`!
    let mlir_lifetime = concat!(
        "module @simplify_extend_lifetimes {\n",
        "  func.func @main(%c0: tensor<1xi64>, %a: tensor<4xf32>, %b: tensor<4xf32>) ",
        "-> tensor<1xi64> {\n",
        "    %c1 = mogg.async_exec %c0, %a : (tensor<1xi64>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %c2 = mgp.extend_lifetimes %c1, %b, %a ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %c3 = mgp.extend_lifetimes %c2 : (tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %c3 : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir_lifetime).expect("valid MLIR");
    assert_eq!(fold_extend_lifetimes_into_async_kernels(&mut m), 2);
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    assert_eq!(ops.len(), 2);
    assert!(m.ops[ops[0]].is_op("mogg", "async_exec"));
    assert_eq!(m.ops[ops[0]].operands.len(), 3);
    assert_eq!(
        m.ops[ops[0]]
            .attributes
            .get("castforge.extended_lifetime_count"),
        Some(&castforge::ir::Attribute::Int(1))
    );
    assert_eq!(m.ops[ops[1]].operands[0], m.ops[ops[0]].results[0]);

    // 2. Verify `plan_device_graph_collective_partition` across 2 devices:
    let nodes = vec![
        CollectiveGraphNodeSpec {
            node_id: 10,
            devices: vec![0],
            is_collective: false,
            is_graph_capturable: true,
        },
        CollectiveGraphNodeSpec {
            node_id: 11,
            devices: vec![1],
            is_collective: false,
            is_graph_capturable: true,
        },
        CollectiveGraphNodeSpec {
            node_id: 20,
            devices: vec![0, 1],
            is_collective: true,
            is_graph_capturable: true,
        },
        CollectiveGraphNodeSpec {
            node_id: 30,
            devices: vec![0, 1],
            is_collective: false,
            is_graph_capturable: false,
        },
        CollectiveGraphNodeSpec {
            node_id: 40,
            devices: vec![0, 1],
            is_collective: false,
            is_graph_capturable: true,
        },
    ];

    // When `capture_collectives_in_graph == false`, node 20 splits the capture into a
    // pre-collective graph, an `EagerCollectiveStage`, an `EagerHostStage` (node 30), and a
    // post-host graph (node 40):
    let stages_split =
        plan_device_graph_collective_partition(2, &nodes, false).expect("valid partition");
    assert_eq!(
        stages_split,
        vec![
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![10], vec![11]],
            },
            DeviceGraphStage::EagerCollectiveStage { node_ids: vec![20] },
            DeviceGraphStage::EagerHostStage { node_ids: vec![30] },
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![40], vec![40]],
            },
        ]
    );

    // When `capture_collectives_in_graph == true`, the all-device collective (node 20) is captured
    // directly inside the first `DeviceGraphCapture` stage!
    let stages_fused =
        plan_device_graph_collective_partition(2, &nodes, true).expect("valid partition");
    assert_eq!(
        stages_fused,
        vec![
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![10, 20], vec![11, 20]],
            },
            DeviceGraphStage::EagerHostStage { node_ids: vec![30] },
            DeviceGraphStage::DeviceGraphCapture {
                device_subgraphs: vec![vec![40], vec![40]],
            },
        ]
    );
}

#[test]
fn test_turboquant_serde_config_lloyd_max_and_orthonormal_hadamard() {
    assert_eq!(
        TurboQuantPreset::from_name("turboquant_k3v4_nc"),
        Some(TurboQuantPreset::K3V4Nc)
    );
    assert!(TurboQuantPreset::from_name("unknown").is_none());

    // 1. Verify `TurboQuantSerdeConfig` slot geometry and serialized byte calculation:
    //    For `turboquant_k3v4_nc` with `head_dim = 128`:
    //    - `key_packed_size = ceil(128 * 3 / 8) + 2 = 48 + 2 = 50`
    //    - `value_packed_size = ceil(128 * 4 / 8) + 4 = 64 + 4 = 68`
    //    - `slot_size = 118`, `slot_size_aligned = 118`
    let cfg = TurboQuantSerdeConfig::new(TurboQuantPreset::K3V4Nc, 128, 16, 2, 2).expect("valid");
    assert!(!cfg.key_fp8());
    assert_eq!(cfg.key_mse_bits(), 3);
    assert_eq!(cfg.centroid_bits(), 3);
    assert_eq!(cfg.key_packed_size(), 50);
    assert_eq!(cfg.value_packed_size(), 68);
    assert_eq!(cfg.slot_size(), 118);
    assert_eq!(cfg.slot_size_aligned(), 118);
    assert_eq!(cfg.layer_ranges(10), (2, 8));
    // 10 layers (4 raw, 6 quantized), 32 tokens (2 blocks of 16), hidden_dim=512 (4 heads), fp16:
    // raw_bytes = 2 * 4 * 32 * 512 * 2 = 262_144
    // quant_bytes = 6 * 2 * 16 * 4 * 118 = 90_624
    assert_eq!(
        cfg.serialized_nbytes_for_shape(10, 32, 512, 2),
        Some(262_144 + 90_624)
    );

    // 2. Verify `solve_lloyd_max_gaussian` symmetry, strict monotonicity, and midpoint boundaries:
    let (centroids, boundaries) =
        solve_lloyd_max_gaussian(128, 3, 200, 1e-10).expect("valid lloyd-max");
    assert_eq!(centroids.len(), 8);
    assert_eq!(boundaries.len(), 7);
    for i in 0..7 {
        assert!(centroids[i] < boundaries[i]);
        assert!(boundaries[i] < centroids[i + 1]);
    }
    for i in 0..4 {
        assert!((centroids[i] + centroids[7 - i]).abs() < 1e-5);
    }

    // 3. Verify `build_orthonormal_hadamard` produces $H^T H = I$:
    assert!(build_orthonormal_hadamard(6).is_none());
    let h = build_orthonormal_hadamard(8).expect("valid hadamard");
    assert_eq!(h.len(), 64);
    for i in 0..8 {
        for j in 0..8 {
            let dot: f32 = (0..8).map(|k| h[i * 8 + k] * h[j * 8 + k]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((dot - expected).abs() < 1e-5);
        }
    }
}

#[test]
fn test_recover_contiguous_or_dim0_padded_view_and_stride_ties() {
    assert_eq!(
        expected_contiguous_shape_from_strides(&[16, 4, 1], 64),
        Some(vec![4, 4, 4])
    );

    // 1. Transposed HND physical layout `[2, 4, 8, 16, 64]` exposed logically as
    //    `[2, 4, 16, 8, 64]` with strides `[32768, 8192, 64, 1024, 1]`:
    let hnd =
        recover_contiguous_or_dim0_padded_view(&[2, 4, 16, 8, 64], &[32768, 8192, 64, 1024, 1])
            .expect("valid transposed contiguous view");
    assert_eq!(
        hnd,
        ContiguousViewPlan {
            permutation: vec![0, 1, 3, 2, 4],
            permuted_shape: vec![2, 4, 8, 16, 64],
            permuted_strides: vec![32768, 8192, 1024, 64, 1],
            is_strictly_contiguous: true,
            padding_per_block_elems: 0,
            block_stride_elems: 32768,
        }
    );

    // 2. Size-1 dimension tying on stride with neighbor (`[NB=4, NH=1, BS=16, HS=64]` with
    //    strides `[1024, 64, 64, 1]`) places the size-1 dim innermost among tied dims
    //    (`permutation = [0, 2, 1, 3]`, `permuted_shape = [4, 16, 1, 64]`):
    let tied = recover_contiguous_or_dim0_padded_view(&[4, 1, 16, 64], &[1024, 64, 64, 1])
        .expect("valid tied-stride view");
    assert_eq!(tied.permutation, vec![0, 2, 1, 3]);
    assert_eq!(tied.permuted_shape, vec![4, 16, 1, 64]);
    assert!(tied.is_strictly_contiguous);

    // 3. DeepSeek-V4 style `dim-0`-padded view (`shape = [4, 16, 64]`, `inner_tight = 1024`,
    //    `strides = [1536, 64, 1]` -> `padding_per_block_elems = 512`):
    let padded = recover_contiguous_or_dim0_padded_view(&[4, 16, 64], &[1536, 64, 1])
        .expect("valid dim-0-padded view");
    assert!(!padded.is_strictly_contiguous);
    assert_eq!(padded.padding_per_block_elems, 512);
    assert_eq!(padded.block_stride_elems, 1536);

    // 4. Reject interior-dimension padding (`strides = [2048, 128, 1]` for `[4, 16, 64]`) and
    //    overlapping outer blocks (`strides = [512, 64, 1]`):
    assert!(recover_contiguous_or_dim0_padded_view(&[4, 16, 64], &[2048, 128, 1]).is_none());
    assert!(recover_contiguous_or_dim0_padded_view(&[4, 16, 64], &[512, 64, 1]).is_none());
}

#[test]
fn test_fleet_cache_usage_view_and_watermark_eviction_controller() {
    let mut usage = FleetCacheUsageView::new();
    let mut ctrl = FleetWatermarkEvictionController::new(0.5, 0.8, 2);

    // 1. Store 4 keys under `tenant-a` in L2 (each 1000 bytes -> 4000 bytes total), plus key 10
    //    replicated on a second non-shared L2 backend (`nvme`) and in `worker-1` L1:
    let entries_a = vec![
        CacheEventEntrySpec {
            key: 10,
            size_bytes: 1000,
        },
        CacheEventEntrySpec {
            key: 20,
            size_bytes: 1000,
        },
        CacheEventEntrySpec {
            key: 30,
            size_bytes: 1000,
        },
        CacheEventEntrySpec {
            key: 40,
            size_bytes: 1000,
        },
    ];
    usage.consume_batch(
        "worker-1",
        CacheEventTier::L2,
        "redis",
        true,
        CacheEventOpKind::Store,
        "tenant-a",
        &entries_a,
    );
    ctrl.consume_l2_batch(
        &usage,
        CacheEventOpKind::Store,
        "tenant-a",
        &[10, 20, 30, 40],
    );

    usage.consume_batch(
        "worker-2",
        CacheEventTier::L2,
        "nvme",
        false,
        CacheEventOpKind::Store,
        "tenant-a",
        &[CacheEventEntrySpec {
            key: 10,
            size_bytes: 1000,
        }],
    );
    usage.consume_batch(
        "worker-1",
        CacheEventTier::L1,
        "dram",
        false,
        CacheEventOpKind::Store,
        "tenant-a",
        &[CacheEventEntrySpec {
            key: 10,
            size_bytes: 1000,
        }],
    );

    assert_eq!(usage.get_key_bytes(CacheEventTier::L2, 10), 2000);
    assert_eq!(usage.get_salt_bytes(CacheEventTier::L2, "tenant-a"), 5000);
    assert_eq!(usage.get_total_bytes(CacheEventTier::L1), 1000);

    // Fencing `worker-1` drops its L1 bytes (`1000`) while leaving L2 (`5000`) untouched:
    assert_eq!(usage.fence_instance("worker-1"), 1000);
    assert_eq!(usage.get_total_bytes(CacheEventTier::L1), 0);
    assert_eq!(usage.get_salt_bytes(CacheEventTier::L2, "tenant-a"), 5000);

    // Deleting key 10 from `nvme` still leaves its `redis` L2 copy (`1000` bytes), so key 10
    // stays in the LRU!
    usage.consume_batch(
        "worker-2",
        CacheEventTier::L2,
        "nvme",
        false,
        CacheEventOpKind::Delete,
        "tenant-a",
        &[CacheEventEntrySpec {
            key: 10,
            size_bytes: 0,
        }],
    );
    ctrl.consume_l2_batch(&usage, CacheEventOpKind::Delete, "tenant-a", &[10]);
    assert_eq!(ctrl.policy.resident_len_for_salt("tenant-a"), 4);

    // 2. Pin key 40 (the LRU head, since `on_keys_created` inserts in reverse request order
    //    `[40, 30, 20, 10]`) and set quota for `tenant-a` to `5000` (`watermark = 4000`):
    ctrl.pin(&[40]);
    assert_eq!(ctrl.pin_count(40), 1);
    assert_eq!(ctrl.filter_unpinned(&[40, 30, 20]), vec![30, 20]);

    // Without quota configured, `tenant-a` is exempt:
    assert!(ctrl.compute_eviction_plan(&usage).is_empty());

    // Arm quota `5000` -> `current_bytes = 4000 >= 0.8 * 5000` triggers eviction of 2
    // unpinned keys (`[30, 20]`, skipping pinned key `40`):
    ctrl.quota.set_quota("tenant-a", 5000);
    let plan = ctrl.compute_eviction_plan(&usage);
    assert_eq!(plan.get("tenant-a"), Some(&vec![30, 20]));

    // With quota `0`, `effective_ratio = 1.0` selects all 3 unpinned keys (`[30, 20, 10]`) and
    // chunks them into batches of `max_delete_batch = 2`:
    ctrl.quota.set_quota("tenant-a", 0);
    let full_plan = ctrl.compute_eviction_plan(&usage);
    assert_eq!(full_plan.get("tenant-a"), Some(&vec![30, 20, 10]));
    assert_eq!(
        ctrl.chunk_delete_batches(&full_plan),
        vec![
            ("tenant-a".to_string(), vec![30, 20]),
            ("tenant-a".to_string(), vec![10]),
        ]
    );
}

#[test]
fn test_fuse_horizontal_shared_input_ops_and_def_before_use_guard() {
    // 1. Sibling elementwise ops `%a = stablehlo.add %x, %y` and `%b = stablehlo.multiply %x, %z`
    //    share `%x` and have no dependency between them -> fuse into `mo.horizontal_fused`!
    let mlir_fusible = r#"
module @horizontal_fuse_ok {
  func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> tensor<4xf32> {
    %a = stablehlo.add %x, %y : tensor<4xf32>
    %b = stablehlo.multiply %x, %z : tensor<4xf32>
    %out = stablehlo.subtract %a, %b : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
"#;
    let mut m1 = parse_mlir(mlir_fusible).expect("valid MLIR");
    assert_eq!(fuse_horizontal_shared_input_ops(&mut m1), 1);
    let f1 = m1.functions()[0];
    let b1 = m1.regions[m1.ops[f1].regions[0]].blocks[0];
    let ops1 = &m1.blocks[b1].ops;
    assert_eq!(ops1.len(), 3);
    assert!(m1.ops[ops1[0]].is_op("mo", "horizontal_fused"));
    assert_eq!(m1.ops[ops1[0]].operands.len(), 3);
    assert_eq!(m1.ops[ops1[0]].results.len(), 2);

    // 2. When an intermediate op `%mid` between `%a` and `%b` reads `%a`, moving `%a` down to `%b`
    //    would violate def-before-use (`resultsUnusedBefore(a, b) == false`), so horizontal fusion
    //    is rejected!
    let mlir_blocked = r#"
module @horizontal_fuse_blocked {
  func.func @main(%x: tensor<4xf32>, %y: tensor<4xf32>, %z: tensor<4xf32>) -> tensor<4xf32> {
    %a = stablehlo.add %x, %y : tensor<4xf32>
    %mid = stablehlo.tanh %a : tensor<4xf32>
    %b = stablehlo.multiply %x, %z : tensor<4xf32>
    %out = stablehlo.subtract %mid, %b : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
"#;
    let mut m2 = parse_mlir(mlir_blocked).expect("valid MLIR");
    assert_eq!(fuse_horizontal_shared_input_ops(&mut m2), 0);
}

#[test]
fn test_hoist_callsite_invariant_subgraph_ops() {
    // Subgraph `@sub` is called twice from `@main` with the same weight `%w` as argument 0 and
    // varying activations `%x1`, `%x2` as argument 1.
    // Inside `@sub`, `%w_exp = stablehlo.exponential %arg0` and `%w_tan = stablehlo.tanh %w_exp`
    // depend only on callsite-invariant `%arg0`, so both ops are hoisted into `@main` before the
    // first `func.call @sub`, `%arg0` is pruned, and `%w_tan` is passed into `@sub`!
    let mlir = r#"
module @hoist_subgraph_invariant {
  func.func @sub(%w: tensor<4xf32>, %x: tensor<4xf32>) -> tensor<4xf32> {
    %w_exp = stablehlo.exponential %w : tensor<4xf32>
    %w_tan = stablehlo.tanh %w_exp : tensor<4xf32>
    %y = stablehlo.add %x, %w_tan : tensor<4xf32>
    return %y : tensor<4xf32>
  }
  func.func @main(%w: tensor<4xf32>, %x1: tensor<4xf32>, %x2: tensor<4xf32>) -> tensor<4xf32> {
    %r1 = func.call @sub(%w, %x1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %r2 = func.call @sub(%w, %x2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %out = stablehlo.add %r1, %r2 : tensor<4xf32>
    return %out : tensor<4xf32>
  }
}
"#;
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(hoist_callsite_invariant_subgraph_ops(&mut m), 2);

    let fns = m.functions();
    let sub_fn = fns[0];
    let main_fn = fns[1];
    let sub_blk = m.regions[m.ops[sub_fn].regions[0]].blocks[0];
    let main_blk = m.regions[m.ops[main_fn].regions[0]].blocks[0];

    // `@sub` now has only `stablehlo.add` and `func.return` (2 ops instead of 4), and 2 args
    // (`%x` and the hoisted boundary value `%w_tan`):
    assert_eq!(m.blocks[sub_blk].ops.len(), 2);
    assert!(m.ops[m.blocks[sub_blk].ops[0]].is_op("stablehlo", "add"));
    assert_eq!(m.blocks[sub_blk].arguments.len(), 2);

    // `@main` now begins with the 2 hoisted ops (`stablehlo.exponential` and `stablehlo.tanh`)
    // before the 2 `func.call` ops:
    let main_ops = &m.blocks[main_blk].ops;
    assert_eq!(main_ops.len(), 6);
    assert!(m.ops[main_ops[0]].is_op("stablehlo", "exponential"));
    assert!(m.ops[main_ops[1]].is_op("stablehlo", "tanh"));
    assert!(m.ops[main_ops[2]].is_op("func", "call"));
    assert!(m.ops[main_ops[3]].is_op("func", "call"));
}

#[test]
fn test_hybrid_engine_group_view_and_token_range_block_slicer() {
    // 3 storage groups over 2 engine groups (engine group 0 split into two physical views `[0, 2]`
    // and `[1]`, engine group 1 holding `[3]` with layer `4` excluded as a KV-sharing layer):
    let groups = vec![
        EngineGroupInfoSpec {
            engine_group_id: 0,
            layer_indices: vec![0, 2],
            tokens_per_block: 64,
            sw_size_tokens: -1,
            extra_object_group_tag: 0,
            recurrent_state: false,
        },
        EngineGroupInfoSpec {
            engine_group_id: 0,
            layer_indices: vec![1],
            tokens_per_block: 64,
            sw_size_tokens: -1,
            extra_object_group_tag: 0,
            recurrent_state: false,
        },
        EngineGroupInfoSpec {
            engine_group_id: 1,
            layer_indices: vec![3],
            tokens_per_block: 256,
            sw_size_tokens: 1024,
            extra_object_group_tag: 0,
            recurrent_state: true,
        },
    ];

    assert_eq!(num_engine_groups(&groups), 2);
    assert_eq!(
        engine_group_layer_indices(&groups),
        vec![vec![0, 1, 2], vec![3]]
    );
    assert_eq!(
        expand_engine_block_ids(&groups, &[vec![10, 11, 12, 13], vec![99]]),
        Some(vec![vec![10, 11, 12, 13], vec![10, 11, 12, 13], vec![99]])
    );
    assert_eq!(
        get_engine_group_indices(&groups, 5),
        Some(vec![0, 0, 0, 1, EXCLUDED_ENGINE_GROUP])
    );

    // Slice token range `[256, 512)` across engine group 0 (`tpb=64` -> blocks `4..8`),
    // engine group 1 (`tpb=256` -> block `1..2`), and scratch group 2 (`tpb=0` -> `[]`):
    let mut alloc = indexmap::IndexMap::new();
    alloc.insert(0usize, vec![10, 11, 12, 13, 14, 15, 16, 17]);
    alloc.insert(1usize, vec![100, 200]);
    assert_eq!(
        slice_block_ids_per_group(&alloc, &[64, 256, 0], 256, 512),
        Some(vec![vec![14, 15, 16, 17], vec![200], vec![]])
    );
    // Unaligned token range (`[128, 512)` is not divisible by `256`) returns `None`:
    assert!(slice_block_ids_per_group(&alloc, &[64, 256, 0], 128, 512).is_none());
}

#[test]
fn test_affinity_slot_router_and_fleet_instance_registry() {
    // 1. Verify `AffinitySlotRouter`:
    assert!(AffinitySlotRouter::new(0).is_none());
    let mut router = AffinitySlotRouter::new(2).expect("valid router");
    assert_eq!(router.slot_for_key(100), (0, true, false));
    assert_eq!(router.slot_for_key(200), (1, true, false));
    assert_eq!(router.slot_for_key(100), (0, false, false));
    // Third distinct key wraps onto slot 0 and marks `is_overflow = true`:
    assert_eq!(router.slot_for_key(300), (0, true, true));
    assert_eq!(router.stats(), (3, 1));

    // 2. Verify `FleetInstanceRegistry` registration, heartbeat freshness, and stale eviction:
    let mut reg = FleetInstanceRegistry::new();
    let inst_a = MpInstanceSpec {
        instance_id: "mp-a".to_string(),
        ip: "10.0.0.1".to_string(),
        http_port: 8000,
        registration_time_ms: 1000,
        last_heartbeat_time_ms: 1000,
        p2p_advertised_url: "10.0.0.1:8100".to_string(),
        mq_port: 8100,
    };
    let inst_b = MpInstanceSpec {
        instance_id: "mp-b".to_string(),
        ip: "10.0.0.2".to_string(),
        http_port: 8000,
        registration_time_ms: 1000,
        last_heartbeat_time_ms: 1000,
        p2p_advertised_url: String::new(),
        mq_port: 0,
    };
    assert!(!reg.register(inst_a.clone()));
    assert!(reg.register(inst_a));
    assert!(!reg.register(inst_b));
    assert_eq!(reg.len(), 2);

    assert!(reg.update_heartbeat("mp-a", 5000));
    assert!(!reg.update_heartbeat("mp-ghost", 5000));

    // At `now_ms = 6000` with `timeout_ms = 3000`, `mp-b` (`6000 - 1000 = 5000 > 3000`) expires
    // while `mp-a` (`6000 - 5000 = 1000 <= 3000`) stays live:
    assert_eq!(reg.remove_stale(6000, 3000), vec!["mp-b".to_string()]);
    assert_eq!(reg.len(), 1);
    assert!(reg.deregister("mp-a").is_some());
    assert!(reg.is_empty());
    assert_eq!(
        reg.counters(),
        FleetMembershipCounters {
            registered: 2,
            re_registered: 1,
            deregistered: 1,
            timed_out: 1,
            unknown_heartbeat: 1,
        }
    );
}

#[test]
fn test_quiesce_state_machine_stream_position_and_checkpoint_envelope() {
    // 1. Verify `QuiesceStateMachine`:
    let mut q = QuiesceStateMachine::new();
    assert!(q.try_begin_apply());
    assert!(q.try_begin_apply());
    assert_eq!(q.state(), (false, 2));

    // Requesting quiesce while 2 batches are in flight sets `quiesce_requested = true` and parks
    // any new arriving batches:
    assert!(!q.request_quiesce());
    assert!(!q.is_quiesced());
    assert!(!q.try_begin_apply());

    // First in-flight batch finishes -> still 1 in flight:
    assert!(!q.finish_apply());
    // Second in-flight batch finishes -> `in_flight == 0` wakes the capturer!
    assert!(q.finish_apply());
    assert!(q.is_quiesced());

    // Releasing quiesce resumes normal batch ingestion:
    q.release_quiesce();
    assert!(q.try_begin_apply());
    assert!(!q.request_quiesce());
    assert_eq!(q.abort_quiesce_on_timeout(), 1);
    assert!(!q.finish_apply());

    // 2. Verify `KafkaStreamPositionTable` and `encode/decode_coordinator_checkpoint_envelope`:
    let mut pos = KafkaStreamPositionTable::new();
    assert_eq!(pos.next_offset("cache-events", 0), None);
    pos.record("cache-events", 0, 41);
    pos.record("cache-events", 1, 99);
    assert_eq!(pos.next_offset("cache-events", 0), Some(42));
    let snapshot = pos.capture_offsets();
    let mut restored = KafkaStreamPositionTable::new();
    restored.restore_offsets(&snapshot);
    assert_eq!(restored, pos);

    assert_eq!(COORDINATOR_CHECKPOINT_MAGIC, b"LMCKPT\0\0");
    let env = encode_coordinator_checkpoint_envelope(1, b"checkpoint-payload");
    assert_eq!(
        decode_coordinator_checkpoint_envelope(&env, 1),
        Some(&b"checkpoint-payload"[..])
    );
    assert!(decode_coordinator_checkpoint_envelope(&env, 2).is_none());
}

#[test]
fn test_elide_redundant_mutable_buffer_ops_cascade() {
    // Cascade on the same buffer `%buf`:
    //   `%c1 = mo.mutable.store %c0, %x, %buf`
    //   `%val, %c2 = mo.mutable.load %c1, %buf`
    //   `%y = stablehlo.add %val, %x`
    //   `%c3 = mo.mutable.store %c2, %y, %buf`
    // First elides `%val, %c2 = mo.mutable.load` (forwarding `%val -> %x` and `%c2 -> %c1`),
    // which exposes `%c1` directly feeding `%c3 = mo.mutable.store %c1, %y, %buf` on the same
    // buffer, so the overwritten first store `%c1` is also elided in the same fixed-point pass!
    let mlir = concat!(
        "module @elide_mutable_cascade {\n",
        "  func.func @main(%c0: tensor<1xi64>, %x: tensor<4xf32>, %buf: tensor<4xf32>) ",
        "-> tensor<1xi64> {\n",
        "    %c1 = mo.mutable.store %c0, %x, %buf ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %val, %c2 = mo.mutable.load %c1, %buf ",
        ": (tensor<1xi64>, tensor<4xf32>) -> (tensor<4xf32>, tensor<1xi64>)\n",
        "    %y = stablehlo.add %val, %x : tensor<4xf32>\n",
        "    %c3 = mo.mutable.store %c2, %y, %buf ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    return %c3 : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(elide_redundant_mutable_buffer_ops(&mut m), 2);
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    assert_eq!(ops.len(), 3);
    assert!(m.ops[ops[0]].is_op("stablehlo", "add"));
    assert_eq!(m.ops[ops[0]].operands[0], m.ops[ops[0]].operands[1]);
    assert!(m.ops[ops[1]].is_op("mo", "mutable.store"));
    assert_eq!(m.ops[ops[1]].operands[0], m.blocks[blk_id].arguments[0]);
}

#[test]
fn test_coalesce_unbundle_consumers_to_parallel() {
    let mlir = concat!(
        "module @unbundle_to_parallel {\n",
        "  func.func @main(%x: tensor<2x4xf32>, %w0: tensor<4xf32>, %w1: tensor<4xf32>) ",
        "-> (tensor<4xf32>, tensor<4xf32>) {\n",
        "    %u0, %u1 = mo.tensor.unbundle %x ",
        ": (tensor<2x4xf32>) -> (tensor<4xf32>, tensor<4xf32>)\n",
        "    %r0 = mo.add %u0, %w0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = mo.add %u1, %w1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r0, %r1 : tensor<4xf32>, tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(coalesce_unbundle_consumers_to_parallel(&mut m), 1);
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    assert_eq!(ops.len(), 2);
    let par_op = &m.ops[ops[0]];
    assert!(par_op.is_op("mo", "parallel"));
    assert_eq!(par_op.results.len(), 2);
    assert_eq!(par_op.operands.len(), 3);
    assert_eq!(
        par_op.attributes.get("wrapped_op"),
        Some(&castforge::ir::Attribute::String("mo.add".into()))
    );
    assert_eq!(
        par_op.attributes.get("parallel_launch_count"),
        Some(&castforge::ir::Attribute::Int(2))
    );
    assert_eq!(
        par_op.attributes.get("driving_operand_index"),
        Some(&castforge::ir::Attribute::Int(0))
    );

    // Rejects when a non-driving operand (`%w0`) is shared identically across all launches.
    let mlir_shared = concat!(
        "module @unbundle_shared_weight {\n",
        "  func.func @main(%x: tensor<2x4xf32>, %w0: tensor<4xf32>) ",
        "-> (tensor<4xf32>, tensor<4xf32>) {\n",
        "    %u0, %u1 = mo.tensor.unbundle %x ",
        ": (tensor<2x4xf32>) -> (tensor<4xf32>, tensor<4xf32>)\n",
        "    %r0 = mo.add %u0, %w0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    %r1 = mo.add %u1, %w0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r0, %r1 : tensor<4xf32>, tensor<4xf32>\n",
        "  }\n",
        "}\n"
    );
    let mut m_shared = parse_mlir(mlir_shared).expect("valid MLIR");
    assert_eq!(coalesce_unbundle_consumers_to_parallel(&mut m_shared), 0);
}

#[test]
fn test_multi_tier_prefetch_planner_bitmap2d_and_l1_affinity() {
    // 1. L1 memory tag affinity resolution:
    let wanted = vec![
        MemoryTagSpec::new("dax", "persistent"),
        MemoryTagSpec::new("cpu", "pinned"),
    ];
    let l1_mgrs = vec![
        (10, MemoryTagSpec::new("cpu", "pageable")),
        (20, MemoryTagSpec::new("cpu", "pinned")),
        (30, MemoryTagSpec::new("dax", "persistent")),
    ];
    // Prefers "dax/persistent" (manager 30) even though manager 20 appears earlier in list.
    assert_eq!(resolve_l1_memory_affinity(&wanted, &l1_mgrs), Some(30));

    // 2. Multi-tier grouped prefetch planning (2 groups, 4 chunks):
    // Group 0: full attention (`0`), Group 1: sliding window of `2` chunks (`2`).
    let group_windows = vec![FULL_ATTENTION_WINDOW, 2];
    let mut l1_locked = TierMapState::new();
    // L1 manager 0 holds chunks 0, 1 for group 0 and chunk 2 for group 1.
    assert!(
        l1_locked.insert(
            0,
            Bitmap2D::new(vec![
                vec![true, true, false, false],
                vec![false, false, true, false],
            ])
            .unwrap()
        )
    );

    let mut l2_locked = TierMapState::new();
    // L2 adapter 0 holds chunks 1, 2, 3 for group 0 and chunks 2, 3 for group 1.
    // Note: chunk 0, 1 for group 1 were evicted, but because group 1 has sliding window = 2,
    // prefix length 4 only needs chunks [2, 3] for group 1 and [0..4] for group 0!
    assert!(
        l2_locked.insert(
            0,
            Bitmap2D::new(vec![
                vec![false, true, true, true],
                vec![false, false, true, true],
            ])
            .unwrap()
        )
    );

    // Full mode rejects sliding-window groups (`2 > 0`).
    let full_plan = plan_multi_tier_prefetch_load(
        &group_windows,
        4,
        &l1_locked,
        &l2_locked,
        &[0],
        &[0],
        PrefetchFetchingPolicy::Full,
    );
    assert_eq!(full_plan, MultiTierPrefetchPlan::default());

    // Prefix mode recovers full 4-chunk hit, serves overlaps from L1 first, and loads only the
    // remaining needed cells from L2:
    let prefix_plan = plan_multi_tier_prefetch_load(
        &group_windows,
        4,
        &l1_locked,
        &l2_locked,
        &[0],
        &[0],
        PrefetchFetchingPolicy::Prefix,
    );
    let l1_grid = prefix_plan.l1_planned_keys.get(0).expect("l1 planned");
    let l2_grid = prefix_plan.l2_planned_keys.get(0).expect("l2 planned");
    assert_eq!(
        l1_grid.rows(),
        &[
            vec![true, true, false, false],
            vec![false, false, true, false],
        ]
    );
    // Chunk 1 of group 0 and chunk 2 of group 1 were already planned in L1, so L2 only loads
    // chunks [2, 3] of group 0 and chunk [3] of group 1!
    assert_eq!(
        l2_grid.rows(),
        &[
            vec![false, false, true, true],
            vec![false, false, false, true],
        ]
    );
    assert_eq!(l1_grid.intersect(l2_grid).unwrap().popcount(), 0);
    assert_eq!(plan_l1_prefetch_retention(3, false), vec![false; 3]);
    assert_eq!(plan_l1_prefetch_retention(3, true), vec![true; 3]);
}

#[test]
fn test_salt_key_provider_and_l2_authenticated_kv_frame_codec() {
    // Rejects empty master key or unsupported AES key bit-width (`192`).
    assert!(SaltKeyProvider::new(vec![], 128, L2_HKDF_INFO_PREFIX.to_vec()).is_none());
    assert!(SaltKeyProvider::new(b"master".to_vec(), 192, L2_HKDF_INFO_PREFIX.to_vec()).is_none());

    let mut provider = SaltKeyProvider::new(
        b"0123456789abcdef0123456789abcdef".to_vec(),
        256,
        L2_HKDF_INFO_PREFIX.to_vec(),
    )
    .expect("valid 256-bit key provider");
    assert_eq!(provider.key_len(), 32);
    assert_eq!(provider.cached_salt_count(), 0);

    let k_a1 = provider.get_key("tenant-alpha");
    let k_a2 = provider.get_key("tenant-alpha");
    let k_b = provider.get_key("tenant-beta");
    assert_eq!(k_a1.len(), 32);
    assert_eq!(k_a1, k_a2);
    assert_ne!(k_a1, k_b);
    assert_eq!(provider.cached_salt_count(), 2);

    // Exact framed size calculation: (16 elems * 2B) + (8 elems * 4B) = 64B + 29B = 93B.
    assert_eq!(
        estimate_l2_framed_size(&[(16, 2), (8, 4)]),
        64 + L2_AEAD_FRAME_OVERHEAD
    );

    let plaintext = b"kv-cache-layer-block-payload-bytes-0123456789";
    let iv: [u8; L2_AEAD_IV_LEN] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let frame = seal_l2_kv_frame(&mut provider, "tenant-alpha", &iv, plaintext);
    assert_eq!(frame.len(), plaintext.len() + L2_AEAD_FRAME_OVERHEAD);

    // Roundtrip with exact frame buffer AND with trailing slab padding:
    let mut padded = frame.clone();
    padded.extend_from_slice(&[0xAA; 32]);
    let opened = open_l2_kv_frame(&mut provider, "tenant-alpha", &padded, plaintext.len())
        .expect("authentic frame with trailing padding");
    assert_eq!(opened, plaintext);

    // Wrong tenant `cache_salt` fails authentication:
    assert!(open_l2_kv_frame(&mut provider, "tenant-beta", &frame, plaintext.len()).is_none());

    // Tampered ciphertext byte fails authentication:
    let mut tampered = frame.clone();
    tampered[15] ^= 0x01;
    assert!(open_l2_kv_frame(&mut provider, "tenant-alpha", &tampered, plaintext.len()).is_none());
}

#[test]
fn test_relax_non_aliasing_mutable_chain_sequences() {
    // Three serialized mutable ops starting from `%c0`:
    //   `%c1 = mo.mutable.store %c0, %x, %buf1`
    //   `%c2 = mo.mutable.store %c1, %y, %buf2` (disjoint buffer -> relaxes to `%c0`)
    //   `%c3 = mo.mutable.store %c2, %z, %buf1` (bypasses `%buf2` in `mo.chain.create`)
    let mlir = concat!(
        "module @relax_mutable_chains {\n",
        "  func.func @main(%c0: tensor<1xi64>, %x: tensor<4xf32>, %y: tensor<4xf32>, ",
        "%z: tensor<4xf32>, %buf1: tensor<4xf32>, %buf2: tensor<4xf32>) -> tensor<1xi64> {\n",
        "    %c1 = mo.mutable.store %c0, %x, %buf1 ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %c2 = mo.mutable.store %c1, %y, %buf2 ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    %c3 = mo.mutable.store %c2, %z, %buf1 ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xf32>) -> tensor<1xi64>\n",
        "    return %c3 : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(relax_non_aliasing_mutable_chain_sequences(&mut m), 2);
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let c0_arg = m.blocks[blk_id].arguments[0];
    let store_ops: Vec<_> = m.blocks[blk_id]
        .ops
        .iter()
        .copied()
        .filter(|&id| m.ops[id].is_op("mo", "mutable.store"))
        .collect();
    assert_eq!(store_ops.len(), 3);
    // Both the first store (`%buf1`) and second store (`%buf2`) now take `%c0` directly!
    assert_eq!(m.ops[store_ops[0]].operands[0], c0_arg);
    assert_eq!(m.ops[store_ops[1]].operands[0], c0_arg);
    // The third store (`%buf1`) depends directly on the first store's out_chain (`%c1`),
    // bypassing the second store (`%buf2`)!
    assert_eq!(
        m.ops[store_ops[2]].operands[0],
        m.ops[store_ops[0]].results[0]
    );
}

#[test]
fn test_buffer_dependency_tree_lca_and_insert_explicit_buffer_frees() {
    // Buffer `%buf` is allocated by `mgp.buffer.alloc`, viewed via `mgp.view` (`%v`),
    // and used by two parallel branches (`%u1`, `%u2`) whose chains join at `mo.chain.create`
    // (`%c_join`), followed by a downstream op (`%c_end`).
    // The post-dominator LCA of `%buf`'s uses (`%v`, `%u1`, `%u2`) is `%c_join`, which produces a
    // chain output `tensor<1xi64>`, so `mgp.buffer.free` is inserted right after `%c_join` and
    // before `%c_end`!
    let mlir = concat!(
        "module @explicit_buffer_lifetimes {\n",
        "  func.func @main(%c0: tensor<1xi64>) -> tensor<1xi64> {\n",
        "    %buf, %c_alloc = mgp.buffer.alloc %c0 ",
        ": (tensor<1xi64>) -> (tensor<8xf32>, tensor<1xi64>)\n",
        "    %spec = mgp.tensor.extract_tensor_spec %buf : (tensor<8xf32>) -> tensor<1xi64>\n",
        "    %v = mgp.view %buf : (tensor<8xf32>) -> tensor<2x4xf32>\n",
        "    %c1 = mgp.kernel_a %c_alloc, %buf : (tensor<1xi64>, tensor<8xf32>) -> tensor<1xi64>\n",
        "    %c2 = mgp.kernel_b %c_alloc, %v : (tensor<1xi64>, tensor<2x4xf32>) -> tensor<1xi64>\n",
        "    %c_join = mo.chain.create %c1, %c2 ",
        ": (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>\n",
        "    %c_end = mgp.epilogue %c_join, %spec ",
        ": (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>\n",
        "    return %c_end : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops_before = m.blocks[blk_id].ops.clone();
    let alloc_id = ops_before[0];
    let view_id = ops_before[2];
    let ka_id = ops_before[3];
    let kb_id = ops_before[4];
    let join_id = ops_before[5];
    let epilogue_id = ops_before[6];

    // 1. Verify `collect_alloc_uses_in_block` ignores `mgp.tensor.extract_tensor_spec` and follows
    //    `mgp.view` alias to `mgp.kernel_b`:
    let uses = collect_alloc_uses_in_block(&m, blk_id, alloc_id).expect("valid uses");
    assert_eq!(uses, vec![view_id, ka_id, kb_id]);

    // 2. Verify `BufferDependencyTree` post-dominator LCA and strict `executes_after`:
    let tree = BufferDependencyTree::build_for_block(&m, blk_id).expect("DAG block");
    assert_eq!(tree.find_lca(&uses), Some(join_id));
    assert!(!tree.executes_after(ka_id, ka_id));
    assert!(tree.executes_after(join_id, ka_id));
    assert!(tree.executes_after(join_id, kb_id));
    assert!(tree.all_execute_after(&[join_id, epilogue_id], &[ka_id, kb_id]));

    // 3. Run `insert_explicit_buffer_frees` and verify `mgp.buffer.free` is placed right after
    //    `%c_join` and spliced into `%c_end`:
    assert_eq!(insert_explicit_buffer_frees(&mut m), 1);
    let ops_after = &m.blocks[blk_id].ops;
    let join_pos = ops_after.iter().position(|&id| id == join_id).unwrap();
    let free_id = ops_after[join_pos + 1];
    assert!(m.ops[free_id].is_op("mgp", "buffer.free"));
    assert_eq!(m.ops[free_id].operands[0], m.ops[join_id].results[0]);
    assert_eq!(m.ops[free_id].operands[1], m.ops[alloc_id].results[0]);
    assert_eq!(m.ops[epilogue_id].operands[0], m.ops[free_id].results[0]);
}

#[test]
fn test_predicate_assert_dependencies_stops_at_first_chain_consumer() {
    // `%idx` feeds both:
    //   1. `%cond = stablehlo.compare %idx, %limit` -> `%c_assert = mgp.assert %c0, %cond`
    //   2. `%scaled = stablehlo.add %idx, %idx` (pure op, no chain)
    //        -> `%out1, %c1 = mgp.gather %c0, %table, %scaled` (first chain-taking op!)
    //        -> `%out2, %c2 = mgp.kernel %c1, %out1` (second chain-taking op, inherits via `%c1`)
    // `predicate_assert_dependencies` predicates `mgp.gather` on `%c_assert` via `mo.chain.create`,
    // and stops downward propagation at `mgp.gather` so `mgp.kernel` is not redundantly predicated!
    let mlir = concat!(
        "module @predicate_asserts {\n",
        "  func.func @main(%c0: tensor<1xi64>, %idx: tensor<4xi32>, %limit: tensor<4xi32>, ",
        "%table: tensor<4xf32>) -> tensor<1xi64> {\n",
        "    %cond = stablehlo.add %idx, %limit : tensor<4xi32>\n",
        "    %c_assert = mgp.assert %c0, %cond : (tensor<1xi64>, tensor<4xi32>) -> tensor<1xi64>\n",
        "    %scaled = stablehlo.add %idx, %idx : tensor<4xi32>\n",
        "    %out1, %c1 = mgp.gather %c0, %table, %scaled ",
        ": (tensor<1xi64>, tensor<4xf32>, tensor<4xi32>) -> (tensor<4xf32>, tensor<1xi64>)\n",
        "    %out2, %c2 = mgp.kernel %c1, %out1 ",
        ": (tensor<1xi64>, tensor<4xf32>) -> (tensor<4xf32>, tensor<1xi64>)\n",
        "    return %c2 : tensor<1xi64>\n",
        "  }\n",
        "}\n"
    );
    let mut m = parse_mlir(mlir).expect("valid MLIR");
    assert_eq!(predicate_assert_dependencies(&mut m), 1);

    let func_id = m.functions()[0];
    let blk_id = m.regions[m.ops[func_id].regions[0]].blocks[0];
    let ops = &m.blocks[blk_id].ops;
    // Expected block ops:
    //   0: stablehlo.add (%cond)
    //   1: mgp.assert (%c_assert)
    //   2: stablehlo.add (%scaled)
    //   3: mo.chain.create (%c0, %c_assert)
    //   4: mgp.gather
    //   5: mgp.kernel
    //   6: func.return
    assert_eq!(ops.len(), 7);
    assert!(m.ops[ops[3]].is_op("mo", "chain.create"));
    assert_eq!(
        m.ops[ops[3]].operands.as_slice(),
        &[m.blocks[blk_id].arguments[0], m.ops[ops[1]].results[0]]
    );
    assert_eq!(m.ops[ops[4]].operands[0], m.ops[ops[3]].results[0]);
    assert_eq!(m.ops[ops[5]].operands[0], m.ops[ops[4]].results[1]);
}

#[test]
fn test_blend_content_fingerprint_index_and_rolling_hash_windows() {
    let seq = [10u64, 20, 30, 40, 50, 60, 70, 80];
    let rolling = rolling_hash_windows_u64(&seq, 4, BLEND_POLY_BASE);
    let chunked = chunk_hash_windows_u64(&seq, 4, BLEND_POLY_BASE);
    assert_eq!(rolling.len(), 5);
    assert_eq!(chunked.len(), 2);
    // Non-overlapping chunk hashes at offsets 0 and 4 equal rolling hashes at 0 and 4!
    assert_eq!(chunked[0], rolling[0]);
    assert_eq!(chunked[1], rolling[4]);

    let mut idx = BlendContentFingerprintIndex::new(4, 1).expect("valid index");
    let ch1 = [0x11u8; 32];
    let ch2 = [0x22u8; 32];
    idx.add(&[10, 20, 30, 40], ch1, 64, "ns-alpha");
    idx.add(&[10, 20, 30, 40], ch1, 64, "ns-beta");
    idx.add(&[50, 60, 70, 80], ch2, 128, "ns-alpha");

    assert_eq!(
        idx.stats(),
        BlendIndexStats {
            num_contents: 2,
            num_chunks: 2,
            num_claims: 3,
            num_namespaces: 2,
            table_size: BLEND_INDEX_MIN_TABLE_SIZE,
        }
    );

    // Query `[99, 10, 20, 30, 40, 88, 50, 60, 70, 80]` contains `ch1` at `cur_st = 1` and
    // `ch2` at `cur_st = 6`:
    let query = [99u64, 10, 20, 30, 40, 88, 50, 60, 70, 80];
    let matches_alpha = idx.match_tokens(&query, "ns-alpha");
    assert_eq!(
        matches_alpha,
        vec![
            BlendFragmentMatch {
                chunk_hash: ch1,
                old_st: 64,
                cur_st: 1,
            },
            BlendFragmentMatch {
                chunk_hash: ch2,
                old_st: 128,
                cur_st: 6,
            },
        ]
    );
    // `ns-beta` only claimed `ch1`:
    let matches_beta = idx.match_tokens(&query, "ns-beta");
    assert_eq!(
        matches_beta,
        vec![BlendFragmentMatch {
            chunk_hash: ch1,
            old_st: 64,
            cur_st: 1,
        }]
    );

    // Removing `ns-alpha`'s claim on `ch1` keeps `ch1` alive for `ns-beta`, while `remove_chunk`
    // drops `ch2` completely:
    idx.remove_claim(&[10, 20, 30, 80], &ch1, "ns-alpha");
    idx.remove_claim(&[10, 20, 30, 40], &ch1, "ns-alpha");
    idx.remove_chunk(&[50, 60, 70, 80], &ch2);
    assert_eq!(idx.stats().num_contents, 1);
    assert_eq!(idx.stats().num_chunks, 1);
    assert_eq!(idx.stats().num_claims, 1);
    assert_eq!(idx.stats().num_namespaces, 1);
}

#[test]
fn test_fp8_e4m3fn_codec_and_multi_serde_group_planner() {
    use castforge::passes::{
        FP8_E4M3FN_MAX_FINITE, FP8_E4M3FN_NAN, MultiSerdeGroupSchema, MultiSerdeSlotSpec,
        dequantize_fp8_e4m3fn_slice_to_f32, estimate_fp8_serialized_size, f32_to_fp8_e4m3fn,
        fp8_e4m3fn_to_f32, quantize_f32_slice_to_fp8_e4m3fn,
    };

    // 1. Exact values & edge cases in float8_e4m3fn:
    assert_eq!(f32_to_fp8_e4m3fn(0.0), 0x00);
    assert_eq!(f32_to_fp8_e4m3fn(-0.0), 0x80);
    assert_eq!(f32_to_fp8_e4m3fn(f32::NAN), FP8_E4M3FN_NAN);
    assert!(fp8_e4m3fn_to_f32(FP8_E4M3FN_NAN).is_nan());
    assert_eq!(f32_to_fp8_e4m3fn(1000.0), 0x7E);
    assert_eq!(fp8_e4m3fn_to_f32(0x7E), FP8_E4M3FN_MAX_FINITE);
    assert_eq!(f32_to_fp8_e4m3fn(-1000.0), 0xFE);
    assert_eq!(fp8_e4m3fn_to_f32(0xFE), -FP8_E4M3FN_MAX_FINITE);

    let samples = [0.0f32, 0.125, -0.5, 1.0, 1.5, -3.5, 16.0, 448.0];
    let q = quantize_f32_slice_to_fp8_e4m3fn(&samples);
    let dq = dequantize_fp8_e4m3fn_slice_to_f32(&q);
    assert_eq!(dq, samples);

    // 2. Deterministic 1-byte-per-element size estimator:
    let shapes = vec![vec![2, 16, 8], vec![2, 16, 8]];
    assert_eq!(estimate_fp8_serialized_size(&shapes), 512);

    // 3. Multi-output serde group schema (slot 0 = K in BF16 optional, slot 1 = V in FP8 required):
    let schema = MultiSerdeGroupSchema {
        slots: vec![
            MultiSerdeSlotSpec {
                name: "K".to_string(),
                bytes_per_element: 2,
                allow_absent: true,
            },
            MultiSerdeSlotSpec {
                name: "V".to_string(),
                bytes_per_element: 1,
                allow_absent: false,
            },
        ],
    };
    assert_eq!(schema.group_size(), 2);

    // Both K and V present:
    let full_plan = schema
        .plan_group(&[Some(vec![4, 8]), Some(vec![4, 8])], "src")
        .unwrap();
    assert_eq!(full_plan.slot_ranges, vec![Some((0, 64)), Some((64, 32))]);
    assert_eq!(full_plan.total_bytes, 96);

    // Tier-split placement (K absent in L1, V in L2):
    let split_plan = schema.plan_group(&[None, Some(vec![4, 8])], "src").unwrap();
    assert_eq!(split_plan.slot_ranges, vec![None, Some((0, 32))]);
    assert_eq!(split_plan.total_bytes, 32);

    // Missing required slot V or wrong group arity fails cleanly:
    assert!(schema.plan_group(&[Some(vec![4, 8]), None], "src").is_err());
    assert!(schema.plan_group(&[Some(vec![4, 8])], "layout").is_err());
}

#[test]
fn test_parallelize_and_fuse_mutable_loads_with_exclusive_chain_detachment() {
    use castforge::ir::mlir::Attribute;
    use castforge::passes::fuse_mutable_loads_into_kernels;

    let mlir_src = concat!(
        "module @fuse_mutable_loads {\n",
        "  func.func @main(%buf: tensor<4xf32>, %c0: tensor<1xi64>, %other: tensor<1xi64>)",
        " -> (tensor<4xf32>, tensor<4xf32>, tensor<1xi64>) {\n",
        "    %t, %c1 = mo.mutable.load %buf, %c0",
        " : (tensor<4xf32>, tensor<1xi64>) -> (tensor<4xf32>, tensor<1xi64>)\n",
        "    %merged = mo.chain.create %other, %c1",
        " : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>\n",
        "    %r0, %kc0 = map.iter.elementwise %t, %merged",
        " : (tensor<4xf32>, tensor<1xi64>) -> (tensor<4xf32>, tensor<1xi64>)\n",
        "    %r1 = map.iter.elementwise %t : (tensor<4xf32>) -> tensor<4xf32>\n",
        "    return %r0, %r1, %kc0 : tensor<4xf32>, tensor<4xf32>, tensor<1xi64>\n",
        "  }\n",
        "}\n",
    );

    let mut module = parse_mlir(mlir_src).expect("valid MLIR");
    let fused = fuse_mutable_loads_into_kernels(&mut module);
    assert_eq!(fused, 2);

    // Verify all `mo.mutable.load` ops were fused into the two `map.iter.elementwise` kernels:
    let func_id = module.functions()[0];
    let blk_id = module.regions[module.ops[func_id].regions[0]].blocks[0];
    let block = &module.blocks[blk_id];
    let remaining_loads = block
        .ops
        .iter()
        .filter(|&&op_id| module.ops[op_id].name.full_name() == "mo.mutable.load")
        .count();
    assert_eq!(remaining_loads, 0);

    let kernels: Vec<_> = block
        .ops
        .iter()
        .copied()
        .filter(|&op_id| module.ops[op_id].name.full_name() == "map.iter.elementwise")
        .collect();
    assert_eq!(kernels.len(), 2);
    for k_id in kernels {
        assert_eq!(
            module.ops[k_id].attributes.get("hasReadEffect"),
            Some(&Attribute::Bool(true))
        );
        assert_eq!(
            module.ops[k_id].attributes.get("fusionInfo"),
            Some(&Attribute::String("MutableLoad".to_string()))
        );
    }
}
