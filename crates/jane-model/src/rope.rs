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

use burn::{
    prelude::Backend,
    tensor::{Int, Tensor},
};

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
        let half_dim = head_dim / 2;

        // Build frequency vector of shape [half_dim].
        // freq[i] = 1 / (rope_theta ^ (2*i / head_dim))
        // We build as f32 data and create a tensor from it.
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                (1.0 / rope_theta.powf(exponent)) as f32
            })
            .collect();

        let freq = Tensor::<B, 1>::from_data(freqs.as_slice(), device);

        // Build position vector of shape [max_seq_len].
        let pos = Tensor::<B, 1, Int>::arange(0..max_seq_len as i64, device).float();

        // Outer product: angles[p, i] = p * freq[i]
        // pos: [max_seq_len] → [max_seq_len, 1]
        // freq: [half_dim] → [1, half_dim]
        // angles: [max_seq_len, half_dim]
        let angles = pos.reshape([max_seq_len, 1]) * freq.reshape([1, half_dim]);

        // cos and sin over angles, both shape [max_seq_len, half_dim]
        let cos_half = angles.clone().cos();
        let sin_half = angles.sin();

        // Duplicate each value to get [max_seq_len, head_dim]:
        // pair index i occupies columns 2*i and 2*i+1.
        // Stack along a new dim-2 to get [max_seq_len, half_dim, 2], then reshape to
        // [max_seq_len, head_dim].  Using repeat_dim(1, 2) would give [p, 2*half_dim]
        // but in the wrong order (all evens then all odds). Instead we stack so that
        // the two copies interleave.
        let cos_table = Tensor::stack::<3>(vec![cos_half.clone(), cos_half], 2)
            .reshape([max_seq_len, head_dim]);
        let sin_table = Tensor::stack::<3>(vec![sin_half.clone(), sin_half], 2)
            .reshape([max_seq_len, head_dim]);

        Self {
            cos: cos_table,
            sin: sin_table,
        }
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
        let [_batch, _n_heads, seq_len, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        // Slice the cos/sin tables to [seq_len, head_dim] starting at `offset`.
        let cos = self.cos.clone().narrow(0, offset, seq_len);
        let sin = self.sin.clone().narrow(0, offset, seq_len);

        // Reshape for broadcasting with x: [1, 1, seq_len, head_dim]
        let cos = cos.reshape([1, 1, seq_len, head_dim]);
        let sin = sin.reshape([1, 1, seq_len, head_dim]);

        // Extract even-indexed (x[..., 2i]) and odd-indexed (x[..., 2i+1]) slices.
        // We use arange_step to pick every other index.
        //
        // Reshape x to [..., half_dim, 2] so we can split cleanly, then reshape back.
        // x: [batch, n_heads, seq_len, head_dim] → [batch, n_heads, seq_len, half_dim, 2]
        let [batch, n_heads, seq_len_x, _] = x.dims();
        let x5 = x.reshape([batch, n_heads, seq_len_x, half_dim, 2]);

        // x_even = x5[..., 0], x_odd = x5[..., 1], each [batch, n_heads, seq_len, half_dim]
        let x_even: Tensor<B, 4> = x5.clone().narrow(4, 0, 1).squeeze_dim(4);
        let x_odd: Tensor<B, 4> = x5.narrow(4, 1, 1).squeeze_dim(4);

        // Same for cos/sin: [1, 1, seq_len, head_dim] → [1, 1, seq_len, half_dim, 2]
        let cos5 = cos.reshape([1, 1, seq_len, half_dim, 2]);
        let sin5 = sin.reshape([1, 1, seq_len, half_dim, 2]);
        let cos_even: Tensor<B, 4> = cos5.clone().narrow(4, 0, 1).squeeze_dim(4);
        let cos_odd: Tensor<B, 4> = cos5.narrow(4, 1, 1).squeeze_dim(4);
        let sin_even: Tensor<B, 4> = sin5.clone().narrow(4, 0, 1).squeeze_dim(4);
        let sin_odd: Tensor<B, 4> = sin5.narrow(4, 1, 1).squeeze_dim(4);

        // Rotation formula:
        // out_even = x_even * cos_even - x_odd * sin_even
        // out_odd  = x_odd  * cos_odd  + x_even * sin_odd
        //
        // cos_even == cos_odd and sin_even == sin_odd because we duplicated,
        // but we keep the names separate for clarity matching the spec.
        let out_even = x_even.clone() * cos_even - x_odd.clone() * sin_even;
        let out_odd = x_odd * cos_odd + x_even * sin_odd;

        // Interleave back: stack on a new dim-4 to get [batch, n_heads, seq_len, half_dim, 2],
        // then reshape to [batch, n_heads, seq_len, head_dim].
        Tensor::stack::<5>(vec![out_even, out_odd], 4)
            .reshape([batch, n_heads, seq_len_x, head_dim])
    }
}

#[cfg(test)]
#[path = "rope_test.rs"]
mod tests;
