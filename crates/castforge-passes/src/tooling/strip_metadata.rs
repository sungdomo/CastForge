use crate::{
    FreezeInputsPass, ParameterizeConstantsPass, Pass, PassResult, StripAttributesPass,
    StripDebugInfoPass,
};
use anyhow::Result;
use castforge_ir::Module;

/// Unified Input Freezing & Constant Parameterization Pass (`freeze-inputs`).
pub struct UnifiedFreezeInputsPass {
    pub args: String,
    pub value: f64,
    pub unfreeze: bool,
    pub min_elements: usize,
}

impl Default for UnifiedFreezeInputsPass {
    fn default() -> Self {
        Self {
            args: String::new(),
            value: 0.5,
            unfreeze: false,
            min_elements: 2,
        }
    }
}

impl Pass for UnifiedFreezeInputsPass {
    fn name(&self) -> &'static str {
        "freeze-inputs"
    }

    fn description(&self) -> &'static str {
        "Binds func.func args to constants or hoists constants to args (unfreeze=true)"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut res = if self.unfreeze {
            ParameterizeConstantsPass {
                min_elements: self.min_elements,
            }
            .run(module)?
        } else {
            FreezeInputsPass {
                args: self.args.clone(),
                value: self.value,
            }
            .run(module)?
        };
        res.pass_name = self.name().to_string();
        Ok(res)
    }
}

/// Unified Metadata & Debug Info Stripping Pass (`strip-metadata`).
///
/// Consolidates `strip-debuginfo` and `strip-attributes` into a single pass with configurable
/// flags (`debuginfo=true|false`, `attributes=true|false`).
pub struct StripMetadataPass {
    pub debuginfo: bool,
    pub attributes: bool,
}

impl Default for StripMetadataPass {
    fn default() -> Self {
        Self {
            debuginfo: true,
            attributes: true,
        }
    }
}

impl Pass for StripMetadataPass {
    fn name(&self) -> &'static str {
        "strip-metadata"
    }

    fn description(&self) -> &'static str {
        "Strips debug locations and transient castforge.* attributes for portable export"
    }

    fn run(&mut self, module: &mut Module) -> Result<PassResult> {
        let mut steps = Vec::new();
        if self.debuginfo {
            steps.push(StripDebugInfoPass.run(module)?);
        }
        if self.attributes {
            steps.push(StripAttributesPass.run(module)?);
        }
        Ok(PassResult {
            pass_name: self.name().to_string(),
            changed: steps.iter().any(|r| r.changed),
            ops_removed: steps.iter().map(|r| r.ops_removed).sum(),
            ops_modified: steps.iter().map(|r| r.ops_modified).sum(),
            ops_created: steps.iter().map(|r| r.ops_created).sum(),
        })
    }
}
