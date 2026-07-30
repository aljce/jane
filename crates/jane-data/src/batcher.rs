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
mod tests {
    use super::*;
    use burn::backend::NdArray;

    // Fix the backend for all tests.
    type B = NdArray;

    fn make_sample(start: u32, seq_len: usize) -> TokenSample {
        // input = [start, start+1, ..., start+seq_len-1]
        // target = [start+1, ..., start+seq_len]  (shift by one)
        let input: Vec<u32> = (start..start + seq_len as u32).collect();
        let target: Vec<u32> = (start + 1..start + seq_len as u32 + 1).collect();
        TokenSample { input, target }
    }

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    // Helper so tests read cleanly without turbofish on every call.
    fn do_batch(items: Vec<TokenSample>) -> LmBatch<B> {
        <LmBatcher as Batcher<B, TokenSample, LmBatch<B>>>::batch(&LmBatcher, items, &device())
    }

    #[test]
    fn shapes_are_batch_size_by_seq_len() {
        let seq_len = 6;
        let items = vec![make_sample(0, seq_len), make_sample(100, seq_len)];
        let batch = do_batch(items);
        assert_eq!(batch.inputs.shape().dims, [2, seq_len]);
        assert_eq!(batch.targets.shape().dims, [2, seq_len]);
    }

    #[test]
    fn single_item_gives_1_by_seq_len() {
        let seq_len = 5;
        let items = vec![make_sample(0, seq_len)];
        let batch = do_batch(items);
        assert_eq!(batch.inputs.shape().dims, [1, seq_len]);
        assert_eq!(batch.targets.shape().dims, [1, seq_len]);
    }

    #[test]
    fn values_survive_round_trip_including_high_ids() {
        // Include id > 255 and u16::MAX to catch byte-order or truncation bugs.
        let items = vec![TokenSample {
            input: vec![256, 1000, 32767, 65535],
            target: vec![1000, 32767, 65535, 0],
        }];
        let batch = do_batch(items.clone());
        let input_data = batch.inputs.into_data();
        let target_data = batch.targets.into_data();
        let input_vals: Vec<i64> = input_data.to_vec().unwrap();
        let target_vals: Vec<i64> = target_data.to_vec().unwrap();

        let expected_in: Vec<i64> = items[0].input.iter().map(|&t| t as i64).collect();
        let expected_tgt: Vec<i64> = items[0].target.iter().map(|&t| t as i64).collect();
        assert_eq!(input_vals, expected_in);
        assert_eq!(target_vals, expected_tgt);
    }

    #[test]
    fn row_k_matches_item_k() {
        // Each row has a distinct value pattern so a transpose cannot pass.
        let seq_len = 4;
        let items = vec![
            make_sample(0, seq_len),   // row 0: 0,1,2,3
            make_sample(100, seq_len), // row 1: 100,101,102,103
            make_sample(200, seq_len), // row 2: 200,201,202,203
        ];
        let batch = do_batch(items.clone());
        let flat: Vec<i64> = batch.inputs.into_data().to_vec().unwrap();

        for (k, item) in items.iter().enumerate() {
            for j in 0..seq_len {
                assert_eq!(
                    flat[k * seq_len + j],
                    item.input[j] as i64,
                    "row {k} col {j}"
                );
            }
        }
    }

    #[test]
    fn shift_property_survives_batching() {
        let seq_len = 5;
        let items = vec![make_sample(10, seq_len), make_sample(50, seq_len)];
        let batch = do_batch(items);
        let input_flat: Vec<i64> = batch.inputs.into_data().to_vec().unwrap();
        let target_flat: Vec<i64> = batch.targets.into_data().to_vec().unwrap();

        let batch_size = 2;
        for k in 0..batch_size {
            for i in 0..seq_len - 1 {
                assert_eq!(
                    target_flat[k * seq_len + i],
                    input_flat[k * seq_len + i + 1],
                    "k={k} i={i}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "LmBatcher::batch called with empty items")]
    fn empty_items_panics() {
        do_batch(vec![]);
    }
}
