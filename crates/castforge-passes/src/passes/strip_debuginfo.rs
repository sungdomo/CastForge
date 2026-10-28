use crate::{Pass, PassResult};
use anyhow::Result;
use castforge_ir::{Location, Module};

/// 18. Strip DebugInfo Pass (`strip-debuginfo`)
/// Resets all operation `Location` annotations to `Location::Unknown`.
pub struct StripDebugInfoPass;

impl Pass for StripDebugInfoPass {
    fn name(&self) -> &'static str {
        "strip-debuginfo"
    }

    fn description(&self) -> &'static str {
        "Strips source location metadata (loc(...)) from all operations in the module"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut modified = 0usize;
        for (_, op) in &mut module.ops {
            if op.location != Location::Unknown {
                op.location = Location::Unknown;
                modified += 1;
            }
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: modified > 0,
            ops_removed: 0,
            ops_modified: modified,
            ops_created: 0,
        })
    }
}
