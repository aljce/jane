//! RoPE — Rotary Positional Encoding.
//!
//! Precomputes a `[max_seq_len, head_dim]` table of cos/sin values, then
//! applies them to query and key tensors by rotating adjacent coordinate pairs.
//! This embeds relative position information without learned position
//! parameters, with better length generalization than absolute embeddings.
//!
//! We intentionally do NOT use `burn::nn::RotaryEncoding` — implementing it
//! from tensor primitives is the point.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use burn::{prelude::Backend, tensor::Tensor};

/// Precomputed rotary embedding table.
///
/// Not a `Module` (no learned parameters). Stored alongside the model and
/// passed into attention at each forward call.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding<B: Backend> {
    /// Shape `[max_seq_len, head_dim]`. Each row is the cosine of the
    /// rotation angle for that (position, dimension-pair) combination.
    cos: Tensor<B, 2>,
    /// Shape `[max_seq_len, head_dim]`. Matching sines.
    sin: Tensor<B, 2>,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Build the cos/sin cache for positions `0..max_seq_len`.
    ///
    /// The frequency for dimension pair `i` is:
    /// `theta_i = 1 / (rope_theta ^ (2i / head_dim))`, for `i = 0, 1, ..., head_dim/2 - 1`.
    ///
    /// Then for each position `p`:
    /// `cos[p, 2i] = cos[p, 2i+1] = cos(p * theta_i)`
    /// `sin[p, 2i] = sin[p, 2i+1] = sin(p * theta_i)`
    ///
    /// The duplication across pairs means the table has shape `[max_seq_len, head_dim]`,
    /// not `[max_seq_len, head_dim/2]`, which simplifies the apply step.
    ///
    /// # Tests required
    /// - output shapes are `[max_seq_len, head_dim]`
    /// - position 0 has all cosines = 1.0 and all sines = 0.0
    /// - values are in `[-1, 1]`
    /// - the table is deterministic (two calls produce identical results)
    pub fn new(max_seq_len: usize, head_dim: usize, rope_theta: f64, device: &B::Device) -> Self {
        todo!()
    }

    /// Apply rotary embedding to a tensor of shape `[batch, n_heads, seq_len, head_dim]`.
    ///
    /// For each position `p` and dimension pair `(2i, 2i+1)`:
    /// ```text
    /// out[..., 2i]   = x[..., 2i]   * cos[p, 2i]   - x[..., 2i+1] * sin[p, 2i]
    /// out[..., 2i+1] = x[..., 2i+1] * cos[p, 2i+1] + x[..., 2i]   * sin[p, 2i+1]
    /// ```
    ///
    /// `offset` shifts the position lookup (used during inference with KV cache,
    /// where position != array index). For training, pass 0.
    ///
    /// # Tests required
    /// - applying to a zero tensor gives zeros
    /// - applying with offset=0 at position 0 is identity (cos=1, sin=0)
    /// - vector norms are preserved (RoPE is a rotation, |Rx| = |x|) —
    ///   check within 1e-4 for several random vectors
    /// - output shape matches input shape
    /// - the rotation changes with position (output at pos 0 != output at pos 10)
    /// - offset shifts which row of cos/sin is used
    pub fn apply(&self, x: Tensor<B, 4>, offset: usize) -> Tensor<B, 4> {
        todo!()
    }
}

#[cfg(test)]
#[path = "rope_test.rs"]
mod tests;
