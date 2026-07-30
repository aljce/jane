//! Causal multi-head self-attention.
//!
//! Written from raw tensor ops: Q/K/V projections → reshape into heads →
//! scaled dot-product with causal mask → output projection. RoPE is applied
//! to Q and K before the dot product.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::Tensor,
};

use crate::rope::RotaryEmbedding;

/// Multi-head causal self-attention.
#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    /// Projects `d_model → d_model` (all heads packed). Q projection.
    wq: Linear<B>,
    /// K projection.
    wk: Linear<B>,
    /// V projection.
    wv: Linear<B>,
    /// Output projection `d_model → d_model`.
    wo: Linear<B>,
    n_heads: usize,
    head_dim: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    /// Create a new attention layer.
    ///
    /// All four projections are bias-free.
    /// `d_model` must be divisible by `n_heads` (caller ensures via config validation).
    ///
    /// # Tests required
    /// - parameter count is `4 * d_model * d_model` (no biases)
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        todo!()
    }

    /// Causal self-attention forward pass.
    ///
    /// 1. Project input `x` through Wq, Wk, Wv → `[batch, seq, d_model]`
    /// 2. Reshape to `[batch, seq, n_heads, head_dim]`, then transpose to
    ///    `[batch, n_heads, seq, head_dim]`
    /// 3. Apply RoPE to Q and K (using `rope.apply(q, offset)`)
    /// 4. Compute attention scores: `Q @ K^T / sqrt(head_dim)`
    /// 5. Apply causal mask: set upper-triangular entries to `-inf` (so they
    ///    become 0 after softmax). Token `t` may only attend to tokens `<= t`.
    /// 6. Softmax over the last dimension
    /// 7. Multiply by V: `attn_weights @ V`
    /// 8. Transpose back to `[batch, seq, n_heads, head_dim]`, reshape to
    ///    `[batch, seq, d_model]`
    /// 9. Output projection through Wo
    ///
    /// `offset` is the RoPE position offset (0 during training).
    ///
    /// # Tests required
    /// - output shape is `[batch, seq, d_model]`
    /// - **causal mask test**: perturb tokens at positions `> t` and verify
    ///   that the output at position `t` does NOT change. This is the real
    ///   proof there's no future leakage — a model that peeks looks *better*
    ///   on loss, so only an explicit test catches it.
    /// - the attention mask is strictly lower-triangular (no token attends to
    ///   future positions)
    /// - a single-token input (`seq_len=1`) works without error
    /// - output is different for different inputs (not a constant function)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[cfg(test)]
#[path = "attention_test.rs"]
mod tests;
