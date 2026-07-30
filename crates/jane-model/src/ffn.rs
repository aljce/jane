//! SwiGLU feed-forward network.
//!
//! Three linear projections (gate, up, down) with SiLU gating:
//!
//! ```text
//! FFN(x) = down(SiLU(gate(x)) * up(x))
//! ```
//!
//! where `gate` and `up` project `d_model → d_ff`, and `down` projects
//! `d_ff → d_model`. The `*` is element-wise. This is the "GLU Variants"
//! architecture from Noam Shazeer (arXiv:2002.05202).
//!
//! Uses `burn::nn::Linear` for the weight matrices — those are just tensor
//! wrappers, not "cheating". The composition and gating are hand-rolled.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::Tensor,
};

/// SwiGLU feed-forward block.
#[derive(Module, Debug)]
pub struct Ffn<B: Backend> {
    /// Projects `d_model → d_ff`. Output is passed through SiLU activation.
    gate: Linear<B>,
    /// Projects `d_model → d_ff`. Output is multiplied element-wise with
    /// the gated output.
    up: Linear<B>,
    /// Projects `d_ff → d_model`.
    down: Linear<B>,
}

impl<B: Backend> Ffn<B> {
    /// Create a new SwiGLU FFN.
    ///
    /// All three projections are bias-free (`LinearConfig::new(in, out).init(device)`
    /// — Burn's Linear has no bias by default).
    ///
    /// # Tests required
    /// - output shape is `[batch, seq, d_model]` for input `[batch, seq, d_model]`
    /// - a zero input produces a zero output (SiLU(0) = 0, so gate output is 0)
    /// - parameter count is `3 * d_model * d_ff` (no biases)
    pub fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        // Bias is disabled: the SwiGLU paper and the contract both require
        // bias-free projections. Burn's LinearConfig defaults to bias=true,
        // so we must opt out explicitly.
        let gate = LinearConfig::new(d_model, d_ff)
            .with_bias(false)
            .init(device);
        let up = LinearConfig::new(d_model, d_ff)
            .with_bias(false)
            .init(device);
        let down = LinearConfig::new(d_ff, d_model)
            .with_bias(false)
            .init(device);
        Self { gate, up, down }
    }

    /// `down(SiLU(gate(x)) * up(x))`
    ///
    /// Input: `[batch, seq, d_model]`
    /// Output: `[batch, seq, d_model]`
    ///
    /// # Tests required
    /// - hand-computed reference: for a tiny `d_model=2, d_ff=3` with known
    ///   weights, verify the output matches the formula
    /// - output changes when weights change (not a constant function)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: gate branch is activated with SiLU, then element-wise
        // multiplied with the un-activated up branch before projecting back.
        let gate_out = burn::tensor::activation::silu(self.gate.forward(x.clone()));
        let up_out = self.up.forward(x);
        self.down.forward(gate_out * up_out)
    }
}

#[cfg(test)]
#[path = "ffn_test.rs"]
mod tests;
