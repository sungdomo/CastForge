use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Attribute, Location, Module, OpName};

/// 6. Surgical IR Edit Pass (`ir-edit`)
pub struct IrEditPass {
    /// Format: `"old_dialect.old_op:new_dialect.new_op"`
    pub rename_op: Option<String>,
    /// Format: `"dialect.op:attr_key:attr_value"`
    pub annotate_op: Option<String>,
}

impl Pass for IrEditPass {
    fn name(&self) -> &'static str {
        "ir-edit"
    }

    fn description(&self) -> &'static str {
        "Surgical IR editing pass (rename_op=from:to, annotate_op=op:key:val)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;

        if let Some(rule) = &self.rename_op {
            if let Some((from, to)) = rule.split_once(':') {
                let from_name = OpName::parse(from.trim());
                let to_name = OpName::parse(to.trim());
                for (_, op) in &mut module.ops {
                    if op.name == from_name {
                        op.name = to_name.clone();
                        modified += 1;
                    }
                }
            }
        }

        if let Some(rule) = &self.annotate_op {
            let parts: Vec<&str> = rule.splitn(3, ':').collect();
            if parts.len() == 3 {
                let target_op = OpName::parse(parts[0].trim());
                let attr_key = parts[1].trim().to_string();
                let attr_val = Attribute::String(parts[2].trim().to_string());
                for (_, op) in &mut module.ops {
                    if op.name == target_op {
                        op.attributes.insert(attr_key.clone(), attr_val.clone());
                        modified += 1;
                    }
                }
            }
        }

        let _ = Location::Unknown;
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
