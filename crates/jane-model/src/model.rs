//! The full transformer: Block and Jane.
//!
//! A `Block` is pre-norm attention + residual, then pre-norm FFN + residual.
//! `Jane` stacks N blocks between an embedding layer and a tied output head.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use crate::{
    JaneConfig, attention::CausalSelfAttention, ffn::Ffn, rmsnorm::RmsNorm, rope::RotaryEmbedding,
};

/// One transformer block: pre-norm → attention → residual → pre-norm → FFN → residual.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    attn_norm: RmsNorm<B>,
    attn: CausalSelfAttention<B>,
    ffn_norm: RmsNorm<B>,
    ffn: Ffn<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(cfg: &JaneConfig, device: &B::Device) -> Self {
        todo!()
    }

    /// ```text
    /// h = x + attn(attn_norm(x), rope, offset)
    /// out = h + ffn(ffn_norm(h))
    /// ```
    ///
    /// Input/output: `[batch, seq, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RotaryEmbedding<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

/// The full decoder-only transformer.
///
/// ```text
/// token_ids → Embedding → [N × Block] → RmsNorm → output_head → logits
/// ```
///
/// When `tie_embeddings` is true, the output head reuses the embedding weight
/// matrix (transposed), so `logits = norm_out @ embed_weight^T`.
#[derive(Module, Debug)]
pub struct Jane<B: Backend> {
    embed: Embedding<B>,
    blocks: Vec<Block<B>>,
    final_norm: RmsNorm<B>,
    /// Only present when `tie_embeddings` is false.
    output_head: Option<burn::nn::Linear<B>>,
}

impl<B: Backend> Jane<B> {
    /// Build the full model from config.
    ///
    /// When `tie_embeddings` is true, the output head is `None` and
    /// `forward` multiplies by the embedding weight transposed.
    ///
    /// # Tests required
    /// - parameter count matches `JaneConfig::param_count()` for all four presets
    ///   (this is the formula test from ROADMAP §3.1)
    /// - instantiating every preset on `ndarray` at batch=1 does not panic
    ///   (proves no config hardcodes anything)
    pub fn new(cfg: &JaneConfig, device: &B::Device) -> Self {
        todo!()
    }

    /// Forward pass: token ids → logits.
    ///
    /// Input: `[batch, seq]` integer tensor of token IDs.
    /// Output: `[batch, seq, vocab_size]` float tensor of logits.
    ///
    /// Steps:
    /// 1. Embed token ids → `[batch, seq, d_model]`
    /// 2. Pass through each block with the shared `rope`
    /// 3. Final RmsNorm
    /// 4. If tied: `output @ embed_weight^T` → `[batch, seq, vocab_size]`
    ///    If untied: linear output head → `[batch, seq, vocab_size]`
    ///
    /// # Tests required
    /// - shape test: `[batch, seq]` → `[batch, seq, vocab_size]` on ndarray
    /// - output logits are finite (no NaN/Inf)
    /// - different input ids produce different outputs
    pub fn forward(
        &self,
        ids: Tensor<B, 2, Int>,
        rope: &RotaryEmbedding<B>,
        offset: usize,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[cfg(test)]
#[path = "model_test.rs"]
mod tests;
