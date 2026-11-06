//! Unified Quantization, Calibration, Mixed Precision & Outlier Suppression Passes
//! (`quantize`, `rotate-hadamard`).

pub mod formats;
pub mod schemes;

#[cfg(test)]
pub(crate) use formats::symmetric_int_quant_bounds;
pub use formats::{
    CalibrateQuantizePass, InsertQatStePass, LowerMixedPrecisionPass, QuantizeFp4Pass,
    RotateHadamardPass, quantize_symmetric_f32,
};
pub use schemes::{
    QuantizeFp8Pass, QuantizeGroupwisePass, QuantizePass, QuantizeWeightsPass, SmoothQuantizePass,
};
