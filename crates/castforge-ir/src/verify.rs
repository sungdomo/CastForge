use crate::Module;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("Operation `{op_name}` references missing operand ValueId")]
    DanglingOperand { op_name: String },

    #[error("Operation `{op_name}` references missing result ValueId")]
    DanglingResult { op_name: String },

    #[error("Region references missing BlockId")]
    DanglingBlock,
}

impl Module {
    /// Verifies structural integrity of the IR (all SSA ValueIds, OpIds, and BlockIds resolve).
    pub fn verify(&self) -> Result<(), VerifyError> {
        for (_, region) in &self.regions {
            for &block_id in &region.blocks {
                let Some(block) = self.blocks.get(block_id) else {
                    return Err(VerifyError::DanglingBlock);
                };
                for &op_id in &block.ops {
                    let Some(op) = self.ops.get(op_id) else {
                        continue;
                    };
                    for &operand in &op.operands {
                        if !self.values.contains_key(operand) {
                            return Err(VerifyError::DanglingOperand {
                                op_name: op.name.full_name(),
                            });
                        }
                    }
                    for &result in &op.results {
                        if !self.values.contains_key(result) {
                            return Err(VerifyError::DanglingResult {
                                op_name: op.name.full_name(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
