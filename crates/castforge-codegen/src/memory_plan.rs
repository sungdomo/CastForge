use crate::liveness::{LiveInterval, compute_module_liveness};
use castforge_ir::{Attribute, Module, ValueDef, ValueId};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct BufferAllocation {
    pub value_id: Option<ValueId>,
    pub name: String,
    pub num_elements: usize,
    pub byte_size: usize,
    pub is_constant: bool,
    pub first_def: usize,
    pub last_use: usize,
    pub arena_offset: Option<usize>,
    pub arena_slot: Option<usize>,
    pub inplace_from: Option<ValueId>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryPlan {
    pub allocations: Vec<BufferAllocation>,
    #[serde(skip)]
    value_to_alloc: HashMap<ValueId, usize>,
    pub total_constant_bytes: usize,
    /// Peak arena workspace bytes required with liveness-guided buffer reuse.
    pub total_activation_bytes: usize,
    /// Naive sum of all non-constant activation buffer bytes without reuse.
    pub naive_activation_bytes: usize,
    /// Number of reusable arena slots allocated.
    pub reusable_slot_count: usize,
    /// Total contiguous arena bytes allocated strictly for reusable intermediate slots.
    pub arena_workspace_bytes: usize,
    /// Number of operations that reused an operand buffer in-place (`inplace_operand`).
    pub inplace_reuse_count: usize,
}

impl MemoryPlan {
    /// Lookup the allocation record for `val_id` in $O(1)$ time.
    pub fn allocation_for(&self, val_id: ValueId) -> Option<&BufferAllocation> {
        if let Some(&idx) = self.value_to_alloc.get(&val_id) {
            self.allocations.get(idx)
        } else {
            self.allocations.iter().find(|a| a.value_id == Some(val_id))
        }
    }

    fn push_allocation(&mut self, alloc: BufferAllocation) {
        let idx = self.allocations.len();
        if let Some(vid) = alloc.value_id {
            self.value_to_alloc.insert(vid, idx);
        }
        self.allocations.push(alloc);
    }

    /// Returns the `f32` element index (`arena_offset / 4`) in the shared `arena` workspace
    /// if `val_id` is allocated in the reusable activation arena.
    pub fn arena_f32_offset(&self, val_id: ValueId) -> Option<usize> {
        let alloc = self.allocation_for(val_id)?;
        alloc.arena_offset.map(|bytes| bytes / 4)
    }

    /// Total number of `f32` elements required by the shared `arena` workspace array.
    pub fn arena_f32_len(&self) -> usize {
        self.arena_workspace_bytes.div_ceil(4)
    }

    /// Bytes saved by liveness-based arena buffer reuse compared to naive allocation.
    pub fn saved_activation_bytes(&self) -> usize {
        self.naive_activation_bytes
            .saturating_sub(self.total_activation_bytes)
    }
}

struct ArenaSlot {
    max_bytes: usize,
    intervals: Vec<(usize, usize)>,
    offset: usize,
}

impl ArenaSlot {
    /// Two values conflict if their live ranges overlap, OR if one is defined at the exact
    /// step `s == e` where the other has its last use (because loops read operands while
    /// writing the result buffer). Thus reuse requires `first_def > e || s > last_use`.
    fn conflicts_with(&self, first_def: usize, last_use: usize) -> bool {
        self.intervals
            .iter()
            .any(|&(s, e)| first_def <= e && s <= last_use)
    }
}

/// Computes a liveness-guided static memory and arena buffer reuse plan for `module`.
pub fn build_memory_plan(module: &Module) -> MemoryPlan {
    let liveness = compute_module_liveness(module);
    let mut plan = MemoryPlan::default();
    let mut slots: Vec<ArenaSlot> = Vec::new();

    for (val_id, val) in &module.values {
        let Some(t_ty) = val.ty.as_tensor() else {
            continue;
        };
        let elems = t_ty.num_elements().unwrap_or(1).max(1) as usize;
        let bytes = (elems * t_ty.dtype.bit_width()).div_ceil(8);
        let iv = liveness.get(&val_id).copied().unwrap_or(LiveInterval {
            value: val_id,
            first_def: 0,
            last_use: 0,
            is_constant: false,
            is_block_arg: false,
            is_returned: false,
        });

        let name = val
            .ssa_name
            .clone()
            .unwrap_or_else(|| format!("v_{:?}", val_id));

        if iv.is_constant {
            plan.total_constant_bytes += bytes;
            plan.push_allocation(BufferAllocation {
                value_id: Some(val_id),
                name,
                num_elements: elems,
                byte_size: bytes,
                is_constant: true,
                first_def: iv.first_def,
                last_use: iv.last_use,
                arena_offset: None,
                arena_slot: None,
                inplace_from: None,
            });
            continue;
        }

        plan.naive_activation_bytes += bytes;

        // Block arguments are caller-provided inputs; intermediate SSA op results share the arena
        if iv.is_block_arg {
            plan.push_allocation(BufferAllocation {
                value_id: Some(val_id),
                name,
                num_elements: elems,
                byte_size: bytes,
                is_constant: false,
                first_def: iv.first_def,
                last_use: iv.last_use,
                arena_offset: None,
                arena_slot: None,
                inplace_from: None,
            });
            continue;
        }

        // Check if this value is defined by an op annotated with `inplace_operand = <idx>`.
        let mut inplace_donor: Option<(ValueId, usize)> = None;
        if let ValueDef::OpResult { op: op_id, .. } = val.def
            && let Some(op) = module.ops.get(op_id)
            && let Some(Attribute::Int(op_idx)) = op.attributes.get("inplace_operand")
            && let Some(&donor_id) = op.operands.get(*op_idx as usize)
            && let Some(donor_slot) = plan.allocation_for(donor_id).and_then(|a| a.arena_slot)
        {
            let can_extend = iv.last_use <= iv.first_def
                || !slots[donor_slot].conflicts_with(iv.first_def + 1, iv.last_use);
            if can_extend {
                inplace_donor = Some((donor_id, donor_slot));
            }
        }

        let (assigned_slot, inplace_from) = if let Some((donor_id, donor_slot)) = inplace_donor {
            slots[donor_slot].max_bytes = slots[donor_slot].max_bytes.max(bytes);
            slots[donor_slot]
                .intervals
                .push((iv.first_def, iv.last_use));
            plan.inplace_reuse_count += 1;
            (donor_slot, Some(donor_id))
        } else {
            // Greedy interval coloring: pick an existing non-overlapping slot (preferring one
            // already large enough, or the one requiring minimal growth), else allocate a new slot.
            let mut best_slot_idx: Option<usize> = None;
            for (idx, slot) in slots.iter().enumerate() {
                if !slot.conflicts_with(iv.first_def, iv.last_use) {
                    match best_slot_idx {
                        None => best_slot_idx = Some(idx),
                        Some(prev_idx) => {
                            let prev_growth = bytes.saturating_sub(slots[prev_idx].max_bytes);
                            let cur_growth = bytes.saturating_sub(slot.max_bytes);
                            if cur_growth < prev_growth {
                                best_slot_idx = Some(idx);
                            }
                        }
                    }
                }
            }

            let slot_idx = if let Some(idx) = best_slot_idx {
                slots[idx].max_bytes = slots[idx].max_bytes.max(bytes);
                slots[idx].intervals.push((iv.first_def, iv.last_use));
                idx
            } else {
                let idx = slots.len();
                slots.push(ArenaSlot {
                    max_bytes: bytes,
                    intervals: vec![(iv.first_def, iv.last_use)],
                    offset: 0,
                });
                idx
            };
            (slot_idx, None)
        };

        plan.push_allocation(BufferAllocation {
            value_id: Some(val_id),
            name,
            num_elements: elems,
            byte_size: bytes,
            is_constant: false,
            first_def: iv.first_def,
            last_use: iv.last_use,
            arena_offset: None,
            arena_slot: Some(assigned_slot),
            inplace_from,
        });
    }

    // Assign contiguous byte offsets to each reusable arena slot
    let mut current_offset = 0usize;
    for slot in &mut slots {
        slot.offset = current_offset;
        current_offset += slot.max_bytes;
    }

    // Include block argument bytes if there are no intermediate ops so total_activation_bytes > 0
    let arg_bytes: usize = plan
        .allocations
        .iter()
        .filter(|a| !a.is_constant && a.arena_slot.is_none())
        .map(|a| a.byte_size)
        .sum();

    for alloc in &mut plan.allocations {
        if let Some(slot_id) = alloc.arena_slot {
            alloc.arena_offset = Some(slots[slot_id].offset);
        }
    }

    plan.reusable_slot_count = slots.len();
    plan.arena_workspace_bytes = current_offset;
    plan.total_activation_bytes = current_offset + arg_bytes;
    plan
}
