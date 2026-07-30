//! Collates [`TokenSample`]s into integer tensors.
//!
//! # CONTRACT — implement the `todo!()` body. Do not change signatures.

use burn::{
    data::dataloader::batcher::Batcher,
    prelude::{Backend, Int, Tensor},
};

use crate::dataset::TokenSample;

/// A batch of next-token-prediction windows, both `[batch_size, seq_len]`.
#[derive(Debug, Clone)]
pub struct LmBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LmBatcher;

impl<B: Backend> Batcher<B, TokenSample, LmBatch<B>> for LmBatcher {
    /// Stack `items` into `[batch_size, seq_len]` integer tensors on `device`.
    ///
    /// Row order must match `items` order — a shuffle that silently reorders
    /// inputs relative to targets would still train, just badly.
    ///
    /// # Tests required (run on `burn::backend::NdArray`)
    /// - shapes are exactly `[items.len(), seq_len]` for both tensors
    /// - values survive the round trip: build samples with distinctive ids
    ///   (including one above 255 and one at `u16::MAX`), batch them, then read
    ///   back via `into_data()` and compare element-wise against the inputs
    /// - row `k` of the batch equals `items[k]` — construct rows that differ
    ///   from each other so a transpose or reorder cannot pass
    /// - the shift property survives batching:
    ///   `targets[k][i] == inputs[k][i + 1]`
    /// - a single-item batch gives `[1, seq_len]`, not `[seq_len]`
    /// - an empty `items` vector: assert whichever behaviour you implement
    ///   (panic or empty tensor) so it is at least pinned down
    fn batch(&self, items: Vec<TokenSample>, device: &B::Device) -> LmBatch<B> {
        let batch_size = items.len();
        assert!(
            !items.is_empty(),
            "LmBatcher::batch called with empty items"
        );
        let seq_len = items[0].input.len();

        // Flatten all inputs and targets into row-major i64 vecs, then create
        // [batch_size * seq_len] tensors and reshape to [batch_size, seq_len].
        // Burn's NdArray Int element type is i64; u16 token ids fit without loss.
        let input_data: Vec<i64> = items
            .iter()
            .flat_map(|s| s.input.iter().map(|&t| t as i64))
            .collect();
        let target_data: Vec<i64> = items
            .iter()
            .flat_map(|s| s.target.iter().map(|&t| t as i64))
            .collect();

        let inputs = Tensor::<B, 1, Int>::from_ints(input_data.as_slice(), device)
            .reshape([batch_size, seq_len]);
        let targets = Tensor::<B, 1, Int>::from_ints(target_data.as_slice(), device)
            .reshape([batch_size, seq_len]);

        LmBatch { inputs, targets }
    }
}

#[cfg(test)]
#[path = "batcher_test.rs"]
mod tests;
