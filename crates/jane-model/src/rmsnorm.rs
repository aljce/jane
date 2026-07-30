//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! Hand-rolled from tensor primitives: `sqrt(mean(x²) + eps)` then scale by a
//! learnable per-feature gain vector. Pre-norm placement means every sublayer
//! (attention, FFN) sees a normalized input.
//!
//! We intentionally do NOT use `burn::nn::RmsNorm` — the point of this project
//! is to understand and implement these building blocks from first principles.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use burn::{module::Module, nn::Initializer, prelude::Backend, tensor::Tensor};

/// RMS normalization with a learnable gain (no bias).
///
/// The gain parameter `gamma` has shape `[d_model]` and is initialized to ones.
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Per-feature scale, shape `[d_model]`.
    gamma: burn::module::Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability, from `JaneConfig::norm_eps`.
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Create a new RmsNorm with `gamma` initialized to ones.
    ///
    /// # Tests required
    /// - gamma has shape `[d_model]` and all values are 1.0
    /// - eps is stored correctly
    pub fn new(d_model: usize, eps: f64, device: &B::Device) -> Self {
        // Initializer::Ones.init already returns Param<Tensor<B, 1>>.
        let gamma = Initializer::Ones.init([d_model], device);
        Self { gamma, eps }
    }

    /// `x / sqrt(mean(x², dim=-1, keepdim=true) + eps) * gamma`
    ///
    /// Input shape: `[..., d_model]` (works for any leading dimensions).
    /// Output shape: same as input.
    ///
    /// # Tests required
    /// - hand-computed reference: for a known 1×4 input, compute the expected
    ///   output manually and assert within 1e-5
    /// - output shape equals input shape for `[batch, seq, d_model]`
    /// - normalizing a constant vector yields a constant (scaled by gamma)
    /// - the output changes when gamma is modified (proving it's not hardcoded)
    /// - a vector of zeros does not produce NaN (eps prevents it)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute RMS over the last dimension; mean_dim keeps the dim so rms
        // has shape [batch, seq, 1] and broadcasts naturally against x.
        let rms = (x.clone().powf_scalar(2.0).mean_dim(2) + self.eps).sqrt();
        // gamma shape [d_model] → unsqueeze to [1, 1, d_model] for broadcast.
        (x / rms) * self.gamma.val().unsqueeze()
    }
}

#[cfg(test)]
#[path = "rmsnorm_test.rs"]
mod tests;
