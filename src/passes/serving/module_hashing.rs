//! Call operand conformance, routine outlining, and module content hashing.

use crate::ir::{Attribute, Module, OpId, Operation, Type, Value, ValueDef, ValueId};
use indexmap::IndexMap;
use smallvec::SmallVec;

/// Conforms `castforge.call.opaque` (`mogg.call.opaque`) and `castforge.call.shape`
/// (`mogg.call.shape`) operands against their declared `castforge.kernel.decl` /
/// `mogg.kernel.decl` or `castforge.shape_func.decl` / `mogg.shape_func.decl` signatures:
/// 1. When a kernel decl sets `"supports_unfused_execute" == true`, the call has
///    `"has_epilogue_fusion" != true`, and every output operand is produced by a single-use,
///    non-recomputing `castforge.tensor.create.epilogue` (`mogg.tensor.create.epilogue`), bypasses
///    the pass-through epilogue to its first capture operand and erases the dead epilogue op.
/// 2. Reconciles mutable `castforge.tensor.create` (`mogg.tensor.create` with `"mut" == true`)
///    producers backing read-only `"tensor"` / `"trait:..."` inputs by cloning an immutable
///    producer (`"mut" = false`).
/// 3. Wraps `"scalar"` operands in `castforge.scalar.create`, non-fused `"trait:<contract>"`
///    operands in `castforge.trait.create` (stamped with `"device"` and `"contract"`), and
///    consecutive `"variadic:<abi>"` (`"plain"`, `"stable"`, `"fused_input"`, `"fused_output"`,
///    `"tensor_tuple"`) runs in `castforge.tuple.create` with `"abi"`.
///
/// Returns the number of calls whose operand lists were conformed.
pub fn conform_call_operands(module: &mut Module) -> Result<usize, String> {
    use crate::ir::mlir::OpName;
    use std::collections::HashMap;

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return Ok(0);
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return Ok(0);
    };

    // Collect top-level kernel declarations and shape function declarations.
    // Decl entry: (param_specs, supports_unfused_execute, is_shape_decl)
    type DeclInfo = (Vec<String>, bool, bool);
    let mut decls: HashMap<String, DeclInfo> = HashMap::new();
    for &op_id in &module.blocks[root_blk].ops {
        let full = module.ops[op_id].name.full_name();
        let is_kernel_decl = full == "castforge.kernel.decl" || full == "mogg.kernel.decl";
        let is_shape_decl = full == "castforge.shape_func.decl" || full == "mogg.shape_func.decl";
        if !is_kernel_decl && !is_shape_decl {
            continue;
        }
        let Some(sym) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        let specs: Vec<String> = match module.ops[op_id].attributes.get("param_specs") {
            Some(Attribute::Array(arr)) => arr
                .iter()
                .filter_map(|a| a.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        let supports_unfused = module.ops[op_id]
            .attributes
            .get("supports_unfused_execute")
            .and_then(|a| a.as_bool())
            .unwrap_or(false);
        decls.insert(sym, (specs, supports_unfused, is_shape_decl));
    }

    let mut conformed_calls = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let device_label = module.ops[func_id]
            .attributes
            .get("device_ref")
            .or_else(|| module.ops[func_id].attributes.get("mo.device_ref"))
            .and_then(|a| a.as_str())
            .map(|s| s.split_once(':').map(|(l, _)| l).unwrap_or(s).to_string())
            .unwrap_or_else(|| "cpu".to_string());

        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let orig_ops = module.blocks[blk_id].ops.clone();
                let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
                for &oid in &orig_ops {
                    for &v in &module.ops[oid].operands {
                        *use_counts.entry(v).or_insert(0) += 1;
                    }
                }

                let mut new_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len() + 8);
                let mut dead_epilogues: std::collections::HashSet<OpId> =
                    std::collections::HashSet::new();

                for &op_id in &orig_ops {
                    let full = module.ops[op_id].name.full_name();
                    let is_opaque_call =
                        full == "castforge.call.opaque" || full == "mogg.call.opaque";
                    let is_shape_call = full == "castforge.call.shape" || full == "mogg.call.shape";
                    if !is_opaque_call && !is_shape_call {
                        new_ops.push(op_id);
                        continue;
                    }

                    let callee = module.ops[op_id]
                        .attributes
                        .get("callee")
                        .and_then(|a| a.as_str())
                        .ok_or_else(|| format!("{full} missing 'callee' attribute"))?
                        .trim_start_matches('@')
                        .to_string();

                    let Some((param_specs, supports_unfused, _)) = decls.get(&callee).cloned()
                    else {
                        return Err(format!("ConformCallOperands: no decl named '{callee}'"));
                    };

                    let mut call_operands: Vec<ValueId> =
                        module.ops[op_id].operands.iter().copied().collect();
                    let mut changed = false;

                    // 1. Check pass-through epilogue bypass for unfused `call.opaque`.
                    let has_epilogue_fusion = module.ops[op_id]
                        .attributes
                        .get("has_epilogue_fusion")
                        .and_then(|a| a.as_bool())
                        .unwrap_or(false);
                    if is_opaque_call && supports_unfused && !has_epilogue_fusion {
                        let mut epilogue_rewrites: Vec<(usize, OpId, ValueId)> = Vec::new();
                        let mut all_valid = true;
                        for (idx, &opd_v) in call_operands.iter().enumerate() {
                            if let ValueDef::OpResult { op: prod_op, .. } = module.values[opd_v].def
                            {
                                let p_name = module.ops[prod_op].name.full_name();
                                if p_name == "castforge.tensor.create.epilogue"
                                    || p_name == "mogg.tensor.create.epilogue"
                                {
                                    let recomputes = module.ops[prod_op]
                                        .attributes
                                        .get("recomputes_value")
                                        .and_then(|a| a.as_bool())
                                        .unwrap_or(false);
                                    let single_use =
                                        use_counts.get(&opd_v).copied().unwrap_or(0) == 1;
                                    if !recomputes
                                        && single_use
                                        && let Some(&cap0) = module.ops[prod_op].operands.first()
                                    {
                                        epilogue_rewrites.push((idx, prod_op, cap0));
                                    } else {
                                        all_valid = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if all_valid && !epilogue_rewrites.is_empty() {
                            for (idx, ep_op, cap0) in epilogue_rewrites {
                                call_operands[idx] = cap0;
                                dead_epilogues.insert(ep_op);
                            }
                            changed = true;
                        }
                    }

                    // 2. Pack operands according to `param_specs`.
                    let loc = module.ops[op_id].location.clone();
                    let mut packed: SmallVec<[ValueId; 4]> = SmallVec::new();
                    let mut i = 0usize;
                    let mut spec_idx = 0usize;

                    while i < call_operands.len() {
                        let spec = param_specs
                            .get(spec_idx)
                            .map(String::as_str)
                            .unwrap_or("tensor");
                        let opd = call_operands[i];

                        // Mut-to-immut reconciliation for read-only kernel inputs:
                        let reconciled_opd = if is_opaque_call && spec != "mut_tensor" {
                            if let ValueDef::OpResult { op: prod_op, .. } = module.values[opd].def {
                                let p_name = module.ops[prod_op].name.full_name();
                                let is_mut = module.ops[prod_op]
                                    .attributes
                                    .get("mut")
                                    .and_then(|a| a.as_bool())
                                    .unwrap_or(false);
                                if (p_name == "castforge.tensor.create"
                                    || p_name == "mogg.tensor.create")
                                    && is_mut
                                {
                                    let p_opds = module.ops[prod_op].operands.clone();
                                    let mut p_attrs = module.ops[prod_op].attributes.clone();
                                    p_attrs.insert("mut".to_string(), Attribute::Bool(false));
                                    let p_ty = module.values[opd].ty.clone();
                                    let immut_loc = loc.clone();
                                    let immut_op = module.ops.insert_with_key(|oid| Operation {
                                        id: oid,
                                        name: OpName::new("castforge", "tensor.create"),
                                        operands: p_opds,
                                        results: SmallVec::new(),
                                        attributes: p_attrs,
                                        regions: Vec::new(),
                                        custom_syntax: false,
                                        location: immut_loc,
                                    });
                                    let immut_val = module.values.insert_with_key(|vid| Value {
                                        id: vid,
                                        ssa_name: None,
                                        ty: p_ty,
                                        def: ValueDef::OpResult {
                                            op: immut_op,
                                            index: 0,
                                        },
                                    });
                                    module.ops[immut_op].results.push(immut_val);
                                    new_ops.push(immut_op);
                                    changed = true;
                                    immut_val
                                } else {
                                    opd
                                }
                            } else {
                                opd
                            }
                        } else {
                            opd
                        };

                        if spec == "scalar" {
                            let s_loc = loc.clone();
                            let s_ty = module.values[reconciled_opd].ty.clone();
                            let sc_op = module.ops.insert_with_key(|oid| Operation {
                                id: oid,
                                name: OpName::new("castforge", "scalar.create"),
                                operands: SmallVec::from_slice(&[reconciled_opd]),
                                results: SmallVec::new(),
                                attributes: IndexMap::new(),
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: s_loc,
                            });
                            let sc_val = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: s_ty,
                                def: ValueDef::OpResult {
                                    op: sc_op,
                                    index: 0,
                                },
                            });
                            module.ops[sc_op].results.push(sc_val);
                            new_ops.push(sc_op);
                            packed.push(sc_val);
                            changed = true;
                            i += 1;
                            spec_idx += 1;
                            continue;
                        }

                        if let Some(contract) = spec.strip_prefix("trait:") {
                            let t_loc = loc.clone();
                            let t_ty = module.values[reconciled_opd].ty.clone();
                            let dev_str = device_label.clone();
                            let contract_str = contract.to_string();
                            let tr_op = module.ops.insert_with_key(|oid| Operation {
                                id: oid,
                                name: OpName::new("castforge", "trait.create"),
                                operands: SmallVec::from_slice(&[reconciled_opd]),
                                results: SmallVec::new(),
                                attributes: {
                                    let mut a = IndexMap::new();
                                    a.insert("device".to_string(), Attribute::String(dev_str));
                                    a.insert(
                                        "contract".to_string(),
                                        Attribute::String(contract_str),
                                    );
                                    a
                                },
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: t_loc,
                            });
                            let tr_val = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: t_ty,
                                def: ValueDef::OpResult {
                                    op: tr_op,
                                    index: 0,
                                },
                            });
                            module.ops[tr_op].results.push(tr_val);
                            new_ops.push(tr_op);
                            packed.push(tr_val);
                            changed = true;
                            i += 1;
                            spec_idx += 1;
                            continue;
                        }

                        if let Some(abi_str) = spec.strip_prefix("variadic:") {
                            // Consume consecutive operands matching this variadic slot.
                            let remaining_specs = param_specs.len().saturating_sub(spec_idx + 1);
                            let run_end = call_operands
                                .len()
                                .saturating_sub(remaining_specs)
                                .max(i + 1);
                            let mut run: SmallVec<[ValueId; 4]> = SmallVec::new();
                            let mut elem_tys: Vec<Type> = Vec::new();
                            while i < run_end && i < call_operands.len() {
                                let mut elem_v = call_operands[i];
                                let elem_ty = module.values[elem_v].ty.clone();
                                if abi_str == "tensor_tuple" {
                                    let tr_loc = loc.clone();
                                    let dev_s = device_label.clone();
                                    let tr_ty = elem_ty.clone();
                                    let tr_op = module.ops.insert_with_key(|oid| Operation {
                                        id: oid,
                                        name: OpName::new("castforge", "trait.create"),
                                        operands: SmallVec::from_slice(&[elem_v]),
                                        results: SmallVec::new(),
                                        attributes: {
                                            let mut a = IndexMap::new();
                                            a.insert(
                                                "device".to_string(),
                                                Attribute::String(dev_s),
                                            );
                                            a.insert(
                                                "contract".to_string(),
                                                Attribute::String("TileTensorable".to_string()),
                                            );
                                            a
                                        },
                                        regions: Vec::new(),
                                        custom_syntax: false,
                                        location: tr_loc,
                                    });
                                    let wrapped_v = module.values.insert_with_key(|vid| Value {
                                        id: vid,
                                        ssa_name: None,
                                        ty: tr_ty,
                                        def: ValueDef::OpResult {
                                            op: tr_op,
                                            index: 0,
                                        },
                                    });
                                    module.ops[tr_op].results.push(wrapped_v);
                                    new_ops.push(tr_op);
                                    elem_v = wrapped_v;
                                }
                                run.push(elem_v);
                                elem_tys.push(elem_ty);
                                i += 1;
                            }
                            let tup_loc = loc.clone();
                            let abi_owned = abi_str.to_string();
                            let tup_op = module.ops.insert_with_key(|oid| Operation {
                                id: oid,
                                name: OpName::new("castforge", "tuple.create"),
                                operands: run,
                                results: SmallVec::new(),
                                attributes: {
                                    let mut a = IndexMap::new();
                                    a.insert("abi".to_string(), Attribute::String(abi_owned));
                                    a
                                },
                                regions: Vec::new(),
                                custom_syntax: false,
                                location: tup_loc,
                            });
                            let tup_val = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: Type::Tuple(elem_tys),
                                def: ValueDef::OpResult {
                                    op: tup_op,
                                    index: 0,
                                },
                            });
                            module.ops[tup_op].results.push(tup_val);
                            new_ops.push(tup_op);
                            packed.push(tup_val);
                            changed = true;
                            spec_idx += 1;
                            continue;
                        }

                        packed.push(reconciled_opd);
                        i += 1;
                        spec_idx += 1;
                    }

                    if changed {
                        module.ops[op_id].operands = packed;
                        module.ops[op_id]
                            .attributes
                            .shift_remove("has_same_variadic_operand_size");
                        conformed_calls += 1;
                    }
                    new_ops.push(op_id);
                }

                if !dead_epilogues.is_empty() {
                    new_ops.retain(|id| !dead_epilogues.contains(id));
                    for dead_id in dead_epilogues {
                        module.ops.remove(dead_id);
                    }
                }
                module.blocks[blk_id].ops = new_ops;
            }
        }
    }

    Ok(conformed_calls)
}

/// Outlines each `castforge.routine` (`mogg.routine`) body into a module-level `func.func` stub
/// (labeled with `"abi" = "runtime"`) while keeping the `routine` container at the call site:
/// - Any `castforge.buffer.constant` (`mogg.buffer.constant`) ops inside the routine body are moved
///   to the rebuilt routine body at the call site, and their uses inside the outlined stub are
///   replaced by trailing `!castforge.pointer` block arguments.
/// - Rebuilds the routine's body as those hoisted buffer constants followed by a single
///   `castforge.call` forwarding fresh block arguments plus the constant pointers to the new stub.
pub fn outline_routines(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, OpName, Region};
    use std::collections::{HashMap, HashSet};

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return 0;
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return 0;
    };

    let mut used_syms: HashSet<String> = HashSet::new();
    for &op_id in &module.blocks[root_blk].ops {
        if let Some(s) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
        {
            used_syms.insert(s.to_string());
        }
    }

    let mut outlined = 0usize;
    let func_ids = module.functions();
    let mut new_stubs: Vec<OpId> = Vec::new();

    for func_id in func_ids {
        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let ops = module.blocks[blk_id].ops.clone();
                for op_id in ops {
                    let full = module.ops[op_id].name.full_name();
                    if full != "castforge.routine" && full != "mogg.routine" {
                        continue;
                    }
                    if module.ops[op_id].regions.is_empty() {
                        continue;
                    }
                    let body_reg = module.ops[op_id].regions[0];
                    if module.regions[body_reg].blocks.is_empty() {
                        continue;
                    }
                    let body_blk = module.regions[body_reg].blocks[0];
                    let body_ops = module.blocks[body_blk].ops.clone();
                    let orig_args = module.blocks[body_blk].arguments.clone();
                    let arg_types: Vec<Type> = orig_args
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();
                    let loc = module.ops[op_id].location.clone();
                    let ptr_ty = Type::Opaque("!castforge.pointer".to_string());

                    // Partition `buffer.constant` ops from the rest of the routine body.
                    let mut const_ops: Vec<OpId> = Vec::new();
                    let mut stub_ops: Vec<OpId> = Vec::new();
                    let mut const_val_to_arg: HashMap<ValueId, ValueId> = HashMap::new();

                    for &b_op in &body_ops {
                        let b_name = module.ops[b_op].name.full_name();
                        if (b_name == "castforge.buffer.constant"
                            || b_name == "mogg.buffer.constant")
                            && let Some(&c_res) = module.ops[b_op].results.first()
                        {
                            let new_idx = module.blocks[body_blk].arguments.len() as u32;
                            let trailing_arg = module.values.insert_with_key(|vid| Value {
                                id: vid,
                                ssa_name: None,
                                ty: ptr_ty.clone(),
                                def: ValueDef::BlockArgument {
                                    block: body_blk,
                                    index: new_idx,
                                },
                            });
                            module.blocks[body_blk].arguments.push(trailing_arg);
                            const_val_to_arg.insert(c_res, trailing_arg);
                            const_ops.push(b_op);
                        } else {
                            stub_ops.push(b_op);
                        }
                    }

                    // Rewrite uses of hoisted constant values inside `stub_ops` to trailing args.
                    for &s_op in &stub_ops {
                        for opd in module.ops[s_op].operands.iter_mut() {
                            if let Some(&rep) = const_val_to_arg.get(opd) {
                                *opd = rep;
                            }
                        }
                    }

                    // Check if stub_ops ends with a terminator (`mogg.output`, `castforge.output`,
                    // or `func.return`), or cap it with `castforge.output`.
                    let mut stub_res_types: Vec<Type> = Vec::new();
                    let has_terminator = stub_ops.last().is_some_and(|&last_op| {
                        let n = module.ops[last_op].name.full_name();
                        n == "mogg.output" || n == "castforge.output" || n == "func.return"
                    });
                    if has_terminator {
                        let term = *stub_ops.last().unwrap();
                        stub_res_types = module.ops[term]
                            .operands
                            .iter()
                            .map(|&v| module.values[v].ty.clone())
                            .collect();
                    } else {
                        let out_loc = loc.clone();
                        let out_op = module.ops.insert_with_key(|oid| Operation {
                            id: oid,
                            name: OpName::new("castforge", "output"),
                            operands: SmallVec::new(),
                            results: SmallVec::new(),
                            attributes: IndexMap::new(),
                            regions: Vec::new(),
                            custom_syntax: false,
                            location: out_loc,
                        });
                        stub_ops.push(out_op);
                    }
                    module.blocks[body_blk].ops = stub_ops;

                    let stub_arg_types: Vec<Type> = module.blocks[body_blk]
                        .arguments
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();

                    let fusion_info = module.ops[op_id]
                        .attributes
                        .get("fusion_info")
                        .and_then(|a| a.as_str())
                        .unwrap_or("routine")
                        .to_string();
                    let base_sym = format!("__{}", fusion_info.trim_start_matches('_'));
                    let mut stub_sym = base_sym.clone();
                    let mut suffix = 0usize;
                    while used_syms.contains(&stub_sym) {
                        suffix += 1;
                        stub_sym = format!("{base_sym}_{suffix}");
                    }
                    used_syms.insert(stub_sym.clone());

                    let device_label = module.ops[op_id]
                        .attributes
                        .get("device_ref")
                        .and_then(|a| a.as_str())
                        .map(|s| s.split_once(':').map(|(l, _)| l).unwrap_or(s).to_string())
                        .unwrap_or_else(|| "cpu".to_string());

                    let stub_loc = loc.clone();
                    let stub_fn_ty = Type::Function {
                        inputs: stub_arg_types,
                        results: stub_res_types.clone(),
                    };
                    let stub_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("func", "func"),
                        operands: SmallVec::new(),
                        results: SmallVec::new(),
                        attributes: {
                            let mut a = IndexMap::new();
                            a.insert("sym_name".to_string(), Attribute::String(stub_sym.clone()));
                            a.insert("function_type".to_string(), Attribute::Type(stub_fn_ty));
                            a.insert("fusion_info".to_string(), Attribute::String(fusion_info));
                            a.insert("abi".to_string(), Attribute::String("runtime".to_string()));
                            a.insert("device".to_string(), Attribute::String(device_label));
                            a.insert(
                                "constant_resources".to_string(),
                                Attribute::Array(Vec::new()),
                            );
                            a
                        },
                        regions: vec![body_reg],
                        custom_syntax: true,
                        location: stub_loc,
                    });
                    new_stubs.push(stub_id);

                    // Rebuild the `routine` body with fresh block args, hoisted constants, and
                    // a single `castforge.call` to `stub_sym`.
                    let call_blk = module.blocks.insert_with_key(|bid| Block {
                        id: bid,
                        label: None,
                        arguments: Vec::new(),
                        ops: Vec::new(),
                    });
                    let mut forwarded: SmallVec<[ValueId; 4]> = SmallVec::new();
                    for (idx, ty) in arg_types.into_iter().enumerate() {
                        let av = module.values.insert_with_key(|vid| Value {
                            id: vid,
                            ssa_name: None,
                            ty,
                            def: ValueDef::BlockArgument {
                                block: call_blk,
                                index: idx as u32,
                            },
                        });
                        module.blocks[call_blk].arguments.push(av);
                        forwarded.push(av);
                    }
                    let mut call_blk_ops: Vec<OpId> = Vec::new();
                    for c_op in const_ops {
                        call_blk_ops.push(c_op);
                        if let Some(&c_res) = module.ops[c_op].results.first() {
                            forwarded.push(c_res);
                        }
                    }
                    let call_loc = loc.clone();
                    let call_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("castforge", "call"),
                        operands: forwarded,
                        results: SmallVec::new(),
                        attributes: {
                            let mut a = IndexMap::new();
                            a.insert("callee".to_string(), Attribute::SymbolRef(stub_sym));
                            a
                        },
                        regions: Vec::new(),
                        custom_syntax: false,
                        location: call_loc,
                    });
                    for (r_idx, r_ty) in stub_res_types.into_iter().enumerate() {
                        let rv = module.values.insert_with_key(|vid| Value {
                            id: vid,
                            ssa_name: None,
                            ty: r_ty,
                            def: ValueDef::OpResult {
                                op: call_id,
                                index: r_idx as u32,
                            },
                        });
                        module.ops[call_id].results.push(rv);
                    }
                    call_blk_ops.push(call_id);
                    module.blocks[call_blk].ops = call_blk_ops;

                    let call_reg = module.regions.insert_with_key(|rid| Region {
                        id: rid,
                        blocks: vec![call_blk],
                    });
                    module.ops[op_id].regions = vec![call_reg];
                    outlined += 1;
                }
            }
        }
    }

    module.blocks[root_blk].ops.extend(new_stubs);
    outlined
}

/// Outlines each `castforge.routine.async` (`mogg.routine.async`) op into a module-level
/// `func.func` stub (`"abi" = "graph"`) named `__<callees>_async` (or `__primitive_<N>_async` when
/// the region contains no inner calls), and replaces the `routine.async` op at the call site with a
/// `castforge.compiled_call` (`"callee_symbol" = @<stub_name>`) forwarding its operands, fusion
/// info, dispatch kind, constant resources, and device reference.
pub fn outline_async_routines(module: &mut Module) -> usize {
    use crate::ir::mlir::OpName;
    use std::collections::HashSet;

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return 0;
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return 0;
    };

    let mut used_syms: HashSet<String> = HashSet::new();
    for &op_id in &module.blocks[root_blk].ops {
        if let Some(s) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
        {
            used_syms.insert(s.to_string());
        }
    }

    let mut stub_number = 0usize;
    let mut outlined = 0usize;
    let func_ids = module.functions();
    let mut new_stubs: Vec<OpId> = Vec::new();

    for func_id in func_ids {
        let regions = module.ops[func_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let orig_ops = module.blocks[blk_id].ops.clone();
                let mut updated_ops: Vec<OpId> = Vec::with_capacity(orig_ops.len());

                for &op_id in &orig_ops {
                    let full = module.ops[op_id].name.full_name();
                    if full != "castforge.routine.async" && full != "mogg.routine.async" {
                        updated_ops.push(op_id);
                        continue;
                    }
                    if module.ops[op_id].regions.is_empty() {
                        updated_ops.push(op_id);
                        continue;
                    }
                    let body_reg = module.ops[op_id].regions[0];
                    if module.regions[body_reg].blocks.is_empty() {
                        updated_ops.push(op_id);
                        continue;
                    }
                    let body_blk = module.regions[body_reg].blocks[0];

                    // Collect callees inside `body_blk` in call order.
                    let mut callees: Vec<String> = Vec::new();
                    for &b_op in &module.blocks[body_blk].ops {
                        if let Some(c) = module.ops[b_op]
                            .attributes
                            .get("callee")
                            .or_else(|| module.ops[b_op].attributes.get("callee_symbol"))
                            .and_then(|a| a.as_str())
                        {
                            callees.push(
                                c.trim_start_matches('@')
                                    .trim_start_matches('_')
                                    .to_string(),
                            );
                        }
                    }
                    let base_part = if callees.is_empty() {
                        format!("primitive_{stub_number}")
                    } else {
                        callees.join("_")
                    };
                    stub_number += 1;
                    let base_sym = format!("__{base_part}_async");
                    let mut stub_sym = base_sym.clone();
                    let mut suffix = 0usize;
                    while used_syms.contains(&stub_sym) {
                        suffix += 1;
                        stub_sym = format!("{base_sym}_{suffix}");
                    }
                    used_syms.insert(stub_sym.clone());

                    let arg_tys: Vec<Type> = module.blocks[body_blk]
                        .arguments
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();
                    let res_tys: Vec<Type> = module.ops[op_id]
                        .results
                        .iter()
                        .map(|&v| module.values[v].ty.clone())
                        .collect();

                    let fusion_info = module.ops[op_id]
                        .attributes
                        .get("fusion_info")
                        .or_else(|| module.ops[op_id].attributes.get("summary"))
                        .and_then(|a| a.as_str())
                        .unwrap_or(&stub_sym)
                        .to_string();
                    let dispatch_attr = module.ops[op_id].attributes.get("dispatch").cloned();
                    let const_res_attr = module.ops[op_id]
                        .attributes
                        .get("constant_resources")
                        .cloned()
                        .unwrap_or_else(|| Attribute::Array(Vec::new()));
                    let dev_ref = module.ops[op_id]
                        .attributes
                        .get("device_ref")
                        .and_then(|a| a.as_str())
                        .unwrap_or("cpu:0")
                        .to_string();
                    let dev_label = dev_ref
                        .split_once(':')
                        .map(|(l, _)| l)
                        .unwrap_or(&dev_ref)
                        .to_string();
                    let loc = module.ops[op_id].location.clone();

                    let stub_loc = loc.clone();
                    let stub_fn_ty = Type::Function {
                        inputs: arg_tys,
                        results: res_tys.clone(),
                    };
                    let mut stub_attrs = IndexMap::new();
                    stub_attrs.insert("sym_name".to_string(), Attribute::String(stub_sym.clone()));
                    stub_attrs.insert("function_type".to_string(), Attribute::Type(stub_fn_ty));
                    stub_attrs.insert(
                        "fusion_info".to_string(),
                        Attribute::String(fusion_info.clone()),
                    );
                    if let Some(d) = dispatch_attr.clone() {
                        stub_attrs.insert("dispatch".to_string(), d);
                    }
                    stub_attrs.insert("constant_resources".to_string(), const_res_attr.clone());
                    stub_attrs.insert("abi".to_string(), Attribute::String("graph".to_string()));
                    stub_attrs.insert("device".to_string(), Attribute::String(dev_label));

                    let stub_id = module.ops.insert_with_key(|oid| Operation {
                        id: oid,
                        name: OpName::new("func", "func"),
                        operands: SmallVec::new(),
                        results: SmallVec::new(),
                        attributes: stub_attrs,
                        regions: vec![body_reg],
                        custom_syntax: true,
                        location: stub_loc,
                    });
                    new_stubs.push(stub_id);

                    // Replace `routine.async` in-place with `castforge.compiled_call`.
                    module.ops[op_id].name = OpName::new("castforge", "compiled_call");
                    module.ops[op_id].regions.clear();
                    module.ops[op_id]
                        .attributes
                        .insert("callee_symbol".to_string(), Attribute::SymbolRef(stub_sym));
                    module.ops[op_id]
                        .attributes
                        .insert("fusion_info".to_string(), Attribute::String(fusion_info));
                    module.ops[op_id]
                        .attributes
                        .insert("constant_resources".to_string(), const_res_attr);
                    module.ops[op_id]
                        .attributes
                        .insert("device_ref".to_string(), Attribute::String(dev_ref));
                    updated_ops.push(op_id);
                    outlined += 1;
                }

                module.blocks[blk_id].ops = updated_ops;
            }
        }
    }

    // Insert new async stubs at the start of the module root block.
    if !new_stubs.is_empty() {
        let mut combined = new_stubs;
        combined.extend(module.blocks[root_blk].ops.iter().copied());
        module.blocks[root_blk].ops = combined;
    }

    outlined
}

/// Wraps the dispatch suffix of runtime-ABI (`"abi" = "runtime"` or `"mogg"`) functions in a
/// `castforge.with_trace` (`mogg.with_trace`) region:
/// - Keeps setup operations (before the first call-like op) outside the trace region so passes
///   that rebuild dispatch from outside can still reach setup values.
/// - Moves all ops starting at the first call-like op (`castforge.call*`, `mogg.call*`,
///   `func.call`) up to (but excluding) the terminator into `castforge.with_trace`'s body block,
///   appends `castforge.yield` with the terminator's original operands, and forwards the
///   `castforge.with_trace` results into the function terminator.
pub fn wrap_funcs_with_trace(module: &mut Module) -> usize {
    use crate::ir::mlir::{Block, OpName, Region};

    let mut wrapped = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        let abi = module.ops[func_id]
            .attributes
            .get("abi")
            .and_then(|a| a.as_str())
            .unwrap_or("");
        if abi != "runtime" && abi != "mogg" {
            continue;
        }
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let body_reg = module.ops[func_id].regions[0];
        if module.regions[body_reg].blocks.is_empty() {
            continue;
        }
        let entry_blk = module.regions[body_reg].blocks[0];
        let block_ops = module.blocks[entry_blk].ops.clone();
        if block_ops.len() < 2 {
            continue;
        }

        // Skip if already wrapped in `with_trace`.
        if block_ops.iter().any(|&oid| {
            let f = module.ops[oid].name.full_name();
            f == "castforge.with_trace" || f == "mogg.with_trace"
        }) {
            continue;
        }

        let term_id = *block_ops.last().unwrap();
        let mut setup_ops: Vec<OpId> = Vec::new();
        let mut traced_ops: Vec<OpId> = Vec::new();
        let mut seen_call = false;

        for &oid in &block_ops[..block_ops.len() - 1] {
            let full = module.ops[oid].name.full_name();
            if full == "func.call"
                || full.starts_with("castforge.call")
                || full.starts_with("mogg.call")
            {
                seen_call = true;
            }
            if seen_call {
                traced_ops.push(oid);
            } else {
                setup_ops.push(oid);
            }
        }

        if traced_ops.is_empty() {
            continue;
        }

        let loc = module.ops[func_id].location.clone();
        let term_operands = module.ops[term_id].operands.clone();
        let term_types: Vec<Type> = term_operands
            .iter()
            .map(|&v| module.values[v].ty.clone())
            .collect();

        let trace_blk = module.blocks.insert_with_key(|bid| Block {
            id: bid,
            label: None,
            arguments: Vec::new(),
            ops: Vec::new(),
        });
        let yield_loc = loc.clone();
        let yield_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("castforge", "yield"),
            operands: term_operands,
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: Vec::new(),
            custom_syntax: false,
            location: yield_loc,
        });
        traced_ops.push(yield_id);
        module.blocks[trace_blk].ops = traced_ops;

        let trace_reg = module.regions.insert_with_key(|rid| Region {
            id: rid,
            blocks: vec![trace_blk],
        });

        let trace_id = module.ops.insert_with_key(|oid| Operation {
            id: oid,
            name: OpName::new("castforge", "with_trace"),
            operands: SmallVec::new(),
            results: SmallVec::new(),
            attributes: IndexMap::new(),
            regions: vec![trace_reg],
            custom_syntax: false,
            location: loc,
        });
        let mut trace_results: SmallVec<[ValueId; 4]> = SmallVec::new();
        for (idx, ty) in term_types.into_iter().enumerate() {
            let rv = module.values.insert_with_key(|vid| Value {
                id: vid,
                ssa_name: None,
                ty,
                def: ValueDef::OpResult {
                    op: trace_id,
                    index: idx as u32,
                },
            });
            module.ops[trace_id].results.push(rv);
            trace_results.push(rv);
        }

        module.ops[term_id].operands = trace_results;
        setup_ops.push(trace_id);
        setup_ops.push(term_id);
        module.blocks[entry_blk].ops = setup_ops;
        wrapped += 1;
    }

    wrapped
}

pub(super) fn rewrite_single_opaque_type(
    ty: &mut Type,
    mapping: &IndexMap<String, Attribute>,
    kernel_leaves: &std::collections::HashSet<String>,
) -> Result<usize, String> {
    match ty {
        Type::Opaque(raw) => {
            // Match `!mo.opaque<"Symbol"...>` or `!castforge.opaque<"Symbol"...>`
            for prefix in ["!mo.opaque<\"", "!castforge.opaque<\""] {
                if let Some(after) = raw.strip_prefix(prefix)
                    && let Some((sym, tail)) = after.split_once('"')
                {
                    if let Some(qual_attr) = mapping.get(sym)
                        && let Some(qual_str) = qual_attr.as_str()
                    {
                        if qual_str != sym {
                            *raw = format!("{prefix}{qual_str}\"{tail}");
                            return Ok(1);
                        }
                        return Ok(0);
                    }
                    if !sym.contains('.') && kernel_leaves.contains(sym) {
                        return Err(format!(
                            "opaque type '{sym}' refers to a registered kernel struct; kernel \
                             structs are implementations resolved by registration, not opaque \
                             data types"
                        ));
                    }
                }
            }
            Ok(0)
        }
        Type::Tuple(items) => {
            let mut sum = 0usize;
            for item in items {
                sum += rewrite_single_opaque_type(item, mapping, kernel_leaves)?;
            }
            Ok(sum)
        }
        Type::Function { inputs, results } => {
            let mut sum = 0usize;
            for inp in inputs {
                sum += rewrite_single_opaque_type(inp, mapping, kernel_leaves)?;
            }
            for res in results {
                sum += rewrite_single_opaque_type(res, mapping, kernel_leaves)?;
            }
            Ok(sum)
        }
        _ => Ok(0),
    }
}

/// Qualifies short `!mo.opaque<"Symbol">` and `!castforge.opaque<"Symbol">` types across the
/// module using a simple-to-qualified dictionary mapping (`mapping`), while rejecting any
/// unqualified opaque type whose symbol matches the leaf name of a registered `kernel.decl` or
/// `primitive.decl` in the module.
pub fn rewrite_opaque_types(
    module: &mut Module,
    mapping: &IndexMap<String, Attribute>,
) -> Result<usize, String> {
    use std::collections::HashSet;

    let mut kernel_leaves: HashSet<String> = HashSet::new();
    if let Some(root_reg) = module.regions.get(module.root_region)
        && let Some(&root_blk) = root_reg.blocks.first()
    {
        for &op_id in &module.blocks[root_blk].ops {
            let full = module.ops[op_id].name.full_name();
            if matches!(
                full.as_str(),
                "castforge.kernel.decl"
                    | "mogg.kernel.decl"
                    | "castforge.primitive.decl"
                    | "mogg.primitive.decl"
            ) && let Some(sym) = module.ops[op_id]
                .attributes
                .get("sym_name")
                .and_then(|a| a.as_str())
            {
                let leaf = sym.rsplit_once('.').map(|(_, r)| r).unwrap_or(sym);
                kernel_leaves.insert(leaf.to_string());
            }
        }
    }

    if mapping.is_empty() && kernel_leaves.is_empty() {
        return Ok(0);
    }

    let mut rewritten = 0usize;
    let val_ids: Vec<ValueId> = module.values.keys().collect();
    for vid in val_ids {
        rewritten +=
            rewrite_single_opaque_type(&mut module.values[vid].ty, mapping, &kernel_leaves)?;
    }

    let op_ids: Vec<OpId> = module.ops.keys().collect();
    for oid in op_ids {
        let attr_keys: Vec<String> = module.ops[oid].attributes.keys().cloned().collect();
        for k in attr_keys {
            if let Some(Attribute::Type(ty)) = module.ops[oid].attributes.get_mut(&k) {
                rewritten += rewrite_single_opaque_type(ty, mapping, &kernel_leaves)?;
            }
        }
    }

    Ok(rewritten)
}

pub(super) fn strip_dim_name_in_type(ty: &mut Type) -> bool {
    if let Type::Opaque(raw) = ty {
        for (prefix, bare) in [
            ("!mogg.dim<", "!mogg.dim"),
            ("!castforge.dim<", "!castforge.dim"),
        ] {
            if raw.starts_with(prefix) && raw.ends_with('>') {
                *raw = bare.to_string();
                return true;
            }
        }
    }
    false
}

/// Drops the parameter-reference name from every `!mogg.dim<name>` or `!castforge.dim<name>`
/// block argument of top-level functions and updates their `"function_type"` attributes
/// accordingly. Returns the number of functions whose signatures were updated.
pub fn anonymize_dim_names(module: &mut Module) -> usize {
    let mut updated_funcs = 0usize;
    let func_ids = module.functions();

    for func_id in func_ids {
        if module.ops[func_id].regions.is_empty() {
            continue;
        }
        let reg_id = module.ops[func_id].regions[0];
        if module.regions[reg_id].blocks.is_empty() {
            continue;
        }
        let blk_id = module.regions[reg_id].blocks[0];
        let args = module.blocks[blk_id].arguments.clone();
        let mut changed = false;
        let mut new_arg_types: Vec<Type> = Vec::with_capacity(args.len());

        for arg_v in args {
            if strip_dim_name_in_type(&mut module.values[arg_v].ty) {
                changed = true;
            }
            new_arg_types.push(module.values[arg_v].ty.clone());
        }

        if !changed {
            continue;
        }

        if let Some(Attribute::Type(Type::Function { inputs, .. })) =
            module.ops[func_id].attributes.get_mut("function_type")
        {
            *inputs = new_arg_types;
        }
        updated_funcs += 1;
    }

    updated_funcs
}

/// Renames all top-level symbols in `module` sequentially to `__S0`, `__S1`, ..., rewrites all
/// `Attribute::SymbolRef` references across the module to match, and normalizes scoped
/// parameter declarations (`input_parameters` arrays and `kgen.param.declare` /
/// `castforge.param.declare` ops) to `__P0`, `__P1`, ... within each top-level operation.
pub fn normalize_all_names(module: &mut Module) -> Result<usize, String> {
    use std::collections::HashMap;

    let Some(root_reg) = module.regions.get(module.root_region) else {
        return Ok(0);
    };
    let Some(&root_blk) = root_reg.blocks.first() else {
        return Ok(0);
    };
    let top_ops = module.blocks[root_blk].ops.clone();

    let mut sym_map: HashMap<String, String> = HashMap::new();
    let mut sym_idx = 0usize;

    for &op_id in &top_ops {
        if let Some(old_name) = module.ops[op_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        {
            let new_name = format!("__S{sym_idx}");
            sym_idx += 1;
            module.ops[op_id]
                .attributes
                .insert("sym_name".to_string(), Attribute::String(new_name.clone()));
            sym_map.insert(old_name, new_name);
        }
    }

    // Scoped parameter uniquing within each top-level operation.
    for &top_id in &top_ops {
        let mut param_counter = 0usize;
        let mut param_map: HashMap<String, String> = HashMap::new();

        if let Some(Attribute::Array(arr)) = module.ops[top_id].attributes.get("input_parameters") {
            let old_params: Vec<String> = arr
                .iter()
                .filter_map(|a| a.as_str().map(str::to_string))
                .collect();
            let mut new_arr: Vec<Attribute> = Vec::with_capacity(old_params.len());
            for old_p in old_params {
                if param_map.contains_key(&old_p) {
                    return Err(format!("Shadowing of parameter name '{old_p}' detected"));
                }
                let new_p = format!("__P{param_counter}");
                param_counter += 1;
                param_map.insert(old_p, new_p.clone());
                new_arr.push(Attribute::String(new_p));
            }
            module.ops[top_id]
                .attributes
                .insert("input_parameters".to_string(), Attribute::Array(new_arr));
        }

        let regions = module.ops[top_id].regions.clone();
        for reg_id in regions {
            let blocks = module.regions[reg_id].blocks.clone();
            for blk_id in blocks {
                let mut block_scope = param_map.clone();
                let ops = module.blocks[blk_id].ops.clone();
                for &oid in &ops {
                    let full = module.ops[oid].name.full_name();
                    if (full == "kgen.param.declare" || full == "castforge.param.declare")
                        && let Some(old_p) = module.ops[oid]
                            .attributes
                            .get("param_name")
                            .and_then(|a| a.as_str())
                            .map(str::to_string)
                    {
                        if block_scope.contains_key(&old_p) {
                            return Err(format!("Shadowing of parameter name '{old_p}' detected"));
                        }
                        let new_p = format!("__P{param_counter}");
                        param_counter += 1;
                        block_scope.insert(old_p, new_p.clone());
                        module.ops[oid]
                            .attributes
                            .insert("param_name".to_string(), Attribute::String(new_p));
                    }
                }
                for &oid in &ops {
                    if let Some(old_ref) = module.ops[oid]
                        .attributes
                        .get("param_ref")
                        .and_then(|a| a.as_str())
                        .map(str::to_string)
                        && let Some(new_ref) = block_scope.get(&old_ref)
                    {
                        module.ops[oid]
                            .attributes
                            .insert("param_ref".to_string(), Attribute::String(new_ref.clone()));
                    }
                }
            }
        }
    }

    // Rewrite all SymbolRef attributes across the module.
    let all_ops: Vec<OpId> = module.ops.keys().collect();
    for oid in all_ops {
        let keys: Vec<String> = module.ops[oid].attributes.keys().cloned().collect();
        for k in keys {
            if let Some(Attribute::SymbolRef(s)) = module.ops[oid].attributes.get_mut(&k) {
                let bare = s.trim_start_matches('@');
                if let Some(mapped) = sym_map.get(bare) {
                    *s = mapped.clone();
                }
            }
        }
    }

    Ok(sym_idx)
}

/// Computes call-graph-aware 64-bit semantic hashes (16-char lowercase hex) for every top-level
/// function in `module` and stamps `"semantic_hash"` onto each function's attributes:
/// - A function's generator hash combines its signature, discardable attributes (excluding
///   `"sym_name"`, `"source_name"`, `"linkage_name"`, and `"semantic_hash"`), and its body's
///   structural hash while masking out direct symbol names of external callees.
/// - Incorporates the resolved `"semantic_hash"` of every called function in call order so two
///   callers are semantically equivalent if and only if both their bodies and their callees'
///   semantic hashes match.
pub fn hash_module_contents(module: &mut Module) -> usize {
    use std::collections::HashMap;

    let func_ids = module.functions();
    if func_ids.is_empty() {
        return 0;
    }

    let mut sym_to_func: HashMap<String, OpId> = HashMap::new();
    let mut gen_hashes: HashMap<String, u64> = HashMap::new();
    let mut call_edges: HashMap<String, Vec<String>> = HashMap::new();

    for &fn_id in &func_ids {
        let Some(sym) = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .map(str::to_string)
        else {
            continue;
        };
        sym_to_func.insert(sym.clone(), fn_id);

        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let fn_mix = |acc: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *acc ^= u64::from(b);
                *acc = acc.wrapping_mul(0x0100_0000_01b3);
            }
        };

        let mut attr_parts: Vec<String> = Vec::new();
        for (k, v) in &module.ops[fn_id].attributes {
            if matches!(
                k.as_str(),
                "sym_name" | "source_name" | "linkage_name" | "semantic_hash"
            ) {
                continue;
            }
            attr_parts.push(format!("{k}={v}"));
        }
        attr_parts.sort();
        for part in &attr_parts {
            fn_mix(&mut h, part.as_bytes());
        }

        let mut callees: Vec<String> = Vec::new();
        if let Some(&reg_id) = module.ops[fn_id].regions.first() {
            for &blk_id in &module.regions[reg_id].blocks {
                for &oid in &module.blocks[blk_id].ops {
                    fn_mix(&mut h, module.ops[oid].name.full_name().as_bytes());
                    for (ak, av) in &module.ops[oid].attributes {
                        if let Attribute::SymbolRef(target) = av {
                            let bare = target.trim_start_matches('@').to_string();
                            callees.push(bare);
                            fn_mix(&mut h, ak.as_bytes());
                        } else {
                            fn_mix(&mut h, format!("{ak}={av}").as_bytes());
                        }
                    }
                }
            }
        }

        gen_hashes.insert(sym.clone(), h);
        call_edges.insert(sym, callees);
    }

    // Iteratively propagate callee hashes (handles both DAGs and SCCs deterministically).
    let mut total_hashes: HashMap<String, u64> = gen_hashes.clone();
    for _ in 0..func_ids.len().max(1) {
        let prev = total_hashes.clone();
        for (sym, &base_h) in &gen_hashes {
            let mut h = base_h;
            if let Some(callees) = call_edges.get(sym) {
                for c in callees {
                    let ch = prev.get(c).copied().unwrap_or(0);
                    for &b in &ch.to_le_bytes() {
                        h ^= u64::from(b);
                        h = h.wrapping_mul(0x0100_0000_01b3);
                    }
                }
            }
            total_hashes.insert(sym.clone(), h);
        }
    }

    let mut stamped = 0usize;
    for (sym, fn_id) in sym_to_func {
        if let Some(&h) = total_hashes.get(&sym) {
            module.ops[fn_id].attributes.insert(
                "semantic_hash".to_string(),
                Attribute::String(format!("{h:016x}")),
            );
            stamped += 1;
        }
    }
    stamped
}

/// Applies semantic hash-based naming (`<name_prefix><semantic_hash>`) to `symbols_to_rename`,
/// deduplicating any functions that share an identical `"semantic_hash"`, erasing their
/// `"source_name"` and `"linkage_name"` attributes, and rewriting all `SymbolRef` references across
/// `module`. Returns the map from original symbol name to final canonical symbol name.
pub fn apply_semantic_naming(
    module: &mut Module,
    symbols_to_rename: &[String],
    name_prefix: &str,
) -> Result<IndexMap<String, String>, String> {
    use std::collections::{HashMap, HashSet};

    if symbols_to_rename.is_empty() {
        return Ok(IndexMap::new());
    }

    // Index functions by `sym_name` and `linkage_name`.
    let mut name_to_fn: HashMap<String, OpId> = HashMap::new();
    for fn_id in module.functions() {
        if let Some(s) = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
        {
            name_to_fn.insert(s.to_string(), fn_id);
        }
        if let Some(ln) = module.ops[fn_id]
            .attributes
            .get("linkage_name")
            .and_then(|a| a.as_str())
        {
            name_to_fn.insert(ln.to_string(), fn_id);
        }
    }

    for sym in symbols_to_rename {
        let &fn_id = name_to_fn
            .get(sym)
            .ok_or_else(|| format!("symbol '{sym}' not found in module"))?;
        module.ops[fn_id].attributes.shift_remove("source_name");
    }

    hash_module_contents(module);

    let mut visited_hash_to_sym: HashMap<String, String> = HashMap::new();
    let mut old_sym_to_new_sym: HashMap<String, String> = HashMap::new();
    let mut result_map: IndexMap<String, String> = IndexMap::new();
    let mut dead_fns: HashSet<OpId> = HashSet::new();

    for sym in symbols_to_rename {
        let fn_id = name_to_fn[sym];
        let hash = module.ops[fn_id]
            .attributes
            .get("semantic_hash")
            .and_then(|a| a.as_str())
            .ok_or_else(|| "module has not had semantic hashes computed".to_string())?
            .to_string();
        let old_sym = module.ops[fn_id]
            .attributes
            .get("sym_name")
            .and_then(|a| a.as_str())
            .unwrap_or(sym)
            .to_string();

        if let Some(existing_sym) = visited_hash_to_sym.get(&hash) {
            old_sym_to_new_sym.insert(old_sym, existing_sym.clone());
            result_map.insert(sym.clone(), existing_sym.clone());
            dead_fns.insert(fn_id);
        } else {
            let new_sym = format!("{name_prefix}{hash}");
            old_sym_to_new_sym.insert(old_sym, new_sym.clone());
            result_map.insert(sym.clone(), new_sym.clone());
            module.ops[fn_id]
                .attributes
                .insert("sym_name".to_string(), Attribute::String(new_sym.clone()));
            module.ops[fn_id].attributes.shift_remove("linkage_name");
            visited_hash_to_sym.insert(hash, new_sym);
        }
    }

    let all_ops: Vec<OpId> = module.ops.keys().collect();
    for oid in all_ops {
        let keys: Vec<String> = module.ops[oid].attributes.keys().cloned().collect();
        for k in keys {
            if let Some(Attribute::SymbolRef(s)) = module.ops[oid].attributes.get_mut(&k) {
                let bare = s.trim_start_matches('@');
                if let Some(mapped) = old_sym_to_new_sym.get(bare) {
                    *s = mapped.clone();
                }
            }
        }
    }

    if !dead_fns.is_empty() {
        if let Some(root_reg) = module.regions.get(module.root_region)
            && let Some(&root_blk) = root_reg.blocks.first()
        {
            module.blocks[root_blk]
                .ops
                .retain(|id| !dead_fns.contains(id));
        }
        for dead_id in dead_fns {
            module.ops.remove(dead_id);
        }
    }

    Ok(result_map)
}

/// Parsed object-store bucket URI location (`hf://buckets/<namespace>/<bucket>[/<prefix>]`
/// or `bucket://<namespace>/<bucket>[/<prefix>]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketLocation {
    /// Bucket owner namespace.
    pub namespace: String,
    /// Bucket identifier.
    pub bucket: String,
    /// Normalized subpath prefix (stripped of leading and trailing `'/'`).
    pub prefix: String,
}

impl BucketLocation {
    /// Parses a bucket URI into a [`BucketLocation`].
    pub fn parse(uri: &str) -> Result<Self, String> {
        let trimmed = uri.trim();
        let rest = if let Some(r) = trimmed.strip_prefix("hf://buckets/") {
            r
        } else if let Some(r) = trimmed.strip_prefix("bucket://") {
            r
        } else {
            return Err(format!("Unsupported bucket URI scheme: '{uri}'"));
        };
        let clean = rest.trim_matches('/');
        let mut parts = clean.splitn(3, '/');
        let namespace = parts.next().unwrap_or("").trim();
        let bucket = parts.next().unwrap_or("").trim();
        if namespace.is_empty() || bucket.is_empty() {
            return Err(format!(
                "Bucket URI must include non-empty namespace and bucket: '{uri}'"
            ));
        }
        let prefix = parts.next().unwrap_or("").trim_matches('/').to_string();
        Ok(Self {
            namespace: namespace.to_string(),
            bucket: bucket.to_string(),
            prefix,
        })
    }

    /// Returns the `<namespace>/<bucket>` handle.
    #[must_use]
    pub fn handle(&self) -> String {
        format!("{}/{}", self.namespace, self.bucket)
    }

    /// Resolves an encoded object filename into a remote bucket path under `prefix`.
    #[must_use]
    pub fn remote_object_path(&self, encoded_name: &str) -> String {
        let name = encoded_name.trim_start_matches('/');
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}/{name}", self.prefix)
        }
    }
}

/// Encodes an arbitrary object key into a reversible URL-safe filename ending with `.bin`.
#[must_use]
pub fn encode_bucket_object_name(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 12);
    for &b in key.as_bytes() {
        let is_unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if is_unreserved {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out.push_str(".bin");
    out
}

/// Decodes a `.bin` bucket object filename produced by [`encode_bucket_object_name`].
pub fn decode_bucket_object_name(encoded: &str) -> Result<String, String> {
    let stem = encoded
        .strip_suffix(".bin")
        .ok_or_else(|| format!("Bucket object name missing '.bin' suffix: '{encoded}'"))?;
    let bytes = stem.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(format!(
                    "Truncated percent-escape in bucket object name: '{encoded}'"
                ));
            }
            let hex_str = std::str::from_utf8(&bytes[i + 1..i + 3])
                .map_err(|e| format!("Invalid UTF-8 in percent-escape: {e}"))?;
            let val = u8::from_str_radix(hex_str, 16)
                .map_err(|_| format!("Invalid hex digits '%{hex_str}' in bucket object name"))?;
            decoded.push(val);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded)
        .map_err(|e| format!("Decoded bucket object key is not valid UTF-8: {e}"))
}

/// Partial failure descriptor when a multi-file batch upload fails partway through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialBatchStoreError {
    /// Object names that were durably uploaded before the failure occurred.
    pub succeeded_names: Vec<String>,
    /// Object name whose upload failed.
    pub failed_name: String,
    /// Diagnostic reason for the failure.
    pub reason: String,
}

impl PartialBatchStoreError {
    /// Builds a per-item completion mask aligned with `requested_names` so L1 callers
    /// can commit succeeded transfers and fail the remainder.
    #[must_use]
    pub fn build_completion_bitmap(&self, requested_names: &[String]) -> Vec<bool> {
        let succeeded: std::collections::HashSet<&str> =
            self.succeeded_names.iter().map(String::as_str).collect();
        requested_names
            .iter()
            .map(|n| succeeded.contains(n.as_str()))
            .collect()
    }
}

/// TTL-bounded L2 bucket object size cache with periodic amortization of expired-entry pruning.
#[derive(Debug, Clone)]
pub struct BucketObjectMetaCache {
    ttl_secs: f64,
    prune_interval: usize,
    writes_since_prune: usize,
    entries: std::collections::HashMap<String, (u64, f64)>,
}

impl BucketObjectMetaCache {
    /// Creates a new metadata cache with the specified `ttl_secs` and `prune_interval`.
    #[must_use]
    pub fn new(ttl_secs: f64, prune_interval: usize) -> Self {
        Self {
            ttl_secs: ttl_secs.max(0.0),
            prune_interval: prune_interval.max(1),
            writes_since_prune: 0,
            entries: std::collections::HashMap::new(),
        }
    }

    /// Inserts or refreshes `name` with `size_bytes` at `now_secs`, triggering a prune
    /// sweep every `prune_interval` writes.
    pub fn put_size(&mut self, name: impl Into<String>, size_bytes: u64, now_secs: f64) {
        self.entries
            .insert(name.into(), (size_bytes, now_secs + self.ttl_secs));
        self.writes_since_prune += 1;
        if self.writes_since_prune >= self.prune_interval {
            self.prune_expired(now_secs);
            self.writes_since_prune = 0;
        }
    }

    /// Looks up the cached object size for `name` at `now_secs`, evicting on expiration.
    pub fn get_size(&mut self, name: &str, now_secs: f64) -> Option<u64> {
        if let Some(&(size, expires_at)) = self.entries.get(name) {
            if now_secs <= expires_at {
                return Some(size);
            }
            self.entries.remove(name);
        }
        None
    }

    /// Removes `name` from the metadata cache.
    pub fn invalidate(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    /// Sweeps all entries whose expiration timestamp is strictly before `now_secs`.
    pub fn prune_expired(&mut self, now_secs: f64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, &mut (_, expires_at)| now_secs <= expires_at);
        before.saturating_sub(self.entries.len())
    }

    /// Returns the current number of cached entries (including not-yet-swept expired entries).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries are currently stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
