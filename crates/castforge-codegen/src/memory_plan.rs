use crate::liveness::{LiveInterval, compute_module_liveness};
use castforge_ir::{Module, ValueId};
use serde::Serialize;

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
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryPlan {
    pub allocations: Vec<BufferAllocation>,
    pub total_constant_bytes: usize,
    /// Peak arena workspace bytes required with liveness-guided buffer reuse.
    pub total_activation_bytes: usize,
    /// Naive sum of all non-constant activation buffer bytes without reuse.
    pub naive_activation_bytes: usize,
    /// Number of reusable arena slots allocated.
    pub reusable_slot_count: usize,
}

impl MemoryPlan {
    /// Lookup the allocation record for `val_id`.
    pub fn allocation_for(&self, val_id: ValueId) -> Option<&BufferAllocation> {
        self.allocations.iter().find(|a| a.value_id == Some(val_id))
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
            plan.allocations.push(BufferAllocation {
                value_id: Some(val_id),
                name,
                num_elements: elems,
                byte_size: bytes,
                is_constant: true,
                first_def: iv.first_def,
                last_use: iv.last_use,
                arena_offset: None,
                arena_slot: None,
            });
            continue;
        }

        plan.naive_activation_bytes += bytes;

        // Block arguments are caller-provided inputs; intermediate SSA op results share the arena
        if iv.is_block_arg {
            plan.allocations.push(BufferAllocation {
                value_id: Some(val_id),
                name,
                num_elements: elems,
                byte_size: bytes,
                is_constant: false,
                first_def: iv.first_def,
                last_use: iv.last_use,
                arena_offset: None,
                arena_slot: None,
            });
            continue;
        }

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

        let assigned_slot = if let Some(idx) = best_slot_idx {
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

        plan.allocations.push(BufferAllocation {
            value_id: Some(val_id),
            name,
            num_elements: elems,
            byte_size: bytes,
            is_constant: false,
            first_def: iv.first_def,
            last_use: iv.last_use,
            arena_offset: None,
            arena_slot: Some(assigned_slot),
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
    plan.total_activation_bytes = current_offset + arg_bytes;
    plan
}
