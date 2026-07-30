//! Memory-mapped token dataset.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.
//!
//! Reads the format defined in [`crate::meta`] — flat little-endian `u16`, no
//! header, with a `.meta.json` sidecar. Read that module first. You can write
//! test fixtures directly (`std::fs::write` of `u16::to_le_bytes`) without
//! depending on the binarizer.

use memmap2::Mmap;
use std::fs::File;

use burn_dataset::Dataset;

use crate::{DataError, Result, TokenMeta};

/// One training window. `target` is `input` shifted left by one token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSample {
    pub input: Vec<u32>,
    pub target: Vec<u32>,
}

/// A `.bin` token file, memory-mapped and sliced into fixed windows.
#[derive(Debug)]
pub struct TokenDataset {
    mmap: Mmap,
    seq_len: usize,
    stride: usize,
    token_count: usize,
}

impl TokenDataset {
    /// Map `bin`, validating it against its sidecar.
    ///
    /// - `seq_len`: tokens per sample. Each window needs `seq_len + 1` tokens
    ///   because the target is shifted by one.
    /// - `stride`: gap between window starts. `stride == seq_len` gives
    ///   non-overlapping windows (the standard pretraining choice); a smaller
    ///   stride oversamples. Must be >= 1.
    ///
    /// Must error on: missing sidecar, `seq_len == 0`, `stride == 0`, a length
    /// that disagrees with the sidecar (use [`crate::TokenMeta::check_bin_len`]).
    ///
    /// # Tests required
    /// - `seq_len == 0` and `stride == 0` are rejected
    /// - a `.bin` of odd byte length is rejected
    /// - a sidecar whose `token_count` disagrees with the file is rejected
    /// - a missing sidecar is rejected with a path in the message
    pub fn open(bin: impl AsRef<std::path::Path>, seq_len: usize, stride: usize) -> Result<Self> {
        let bin = bin.as_ref();

        if seq_len == 0 {
            return Err(DataError::Meta("seq_len must be >= 1".into()));
        }
        if stride == 0 {
            return Err(DataError::Meta("stride must be >= 1".into()));
        }

        // Load and validate sidecar — propagates a path-bearing Io error if missing.
        let meta = TokenMeta::load_for(bin)?;

        let file = File::open(bin).map_err(|e| DataError::io(bin, e))?;
        let len_bytes = file.metadata().map_err(|e| DataError::io(bin, e))?.len();

        // Validates odd-byte and count-mismatch cases.
        meta.check_bin_len(bin, len_bytes)?;

        // Safety: the file is not modified while we hold the mapping (standard
        // mmap caveat; acceptable in a training data pipeline).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| DataError::io(bin, e))?;

        Ok(Self {
            mmap,
            seq_len,
            stride,
            token_count: meta.token_count as usize,
        })
    }

    /// Total tokens in the file.
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Token at absolute index, decoded little-endian. Panics out of bounds.
    ///
    /// # Tests required
    /// Write a known `u16` sequence to a fixture and assert every index reads
    /// back exactly, including values above 255 (which is where a byte-order or
    /// stride bug shows up).
    pub fn token_at(&self, index: usize) -> u32 {
        assert!(
            index < self.token_count,
            "token index {index} out of bounds"
        );
        let byte = index * 2;
        u16::from_le_bytes([self.mmap[byte], self.mmap[byte + 1]]) as u32
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// Window count:
///
/// ```text
/// if token_count < seq_len + 1 { 0 }
/// else { (token_count - seq_len - 1) / stride + 1 }
/// ```
///
/// # Tests required
/// This arithmetic is the easiest thing here to get wrong by one. Test it
/// directly and exhaustively for small values:
/// - `token_count=11, seq_len=10, stride=10` -> 1
/// - `token_count=10, seq_len=10, stride=10` -> 0 (no room for the shift)
/// - `token_count=21, seq_len=10, stride=10` -> 2
/// - `token_count=20, seq_len=10, stride=10` -> 1
/// - `token_count=13, seq_len=4,  stride=1`  -> 9
/// - `token_count=0`  -> 0
/// - For a brute-force cross-check over `token_count in 0..40`,
///   `seq_len in 1..8`, `stride in 1..8`: the count must equal the number of
///   `start` values where `start + seq_len < token_count`, and the last window
///   must never read past the end.
impl Dataset<TokenSample> for TokenDataset {
    /// Window `index`, starting at token `index * stride`.
    ///
    /// - `input` = tokens `[start, start + seq_len)`
    /// - `target` = tokens `[start + 1, start + seq_len + 1)`
    /// - `None` when `index >= len()`
    ///
    /// # Tests required
    /// - both vectors have length `seq_len`
    /// - **the shift property**: `sample.target[i] == sample.input[i + 1]` for
    ///   all `i < seq_len - 1`, and `target.last()` is the token just past the
    ///   input window
    /// - `get(len())` and `get(len() + 1)` are `None`
    /// - `get(0)` starts at token 0; `get(1)` starts at token `stride`
    /// - the final window's last target index is `< token_count` (never reads
    ///   past the end — this is the bounds bug the window arithmetic invites)
    fn get(&self, index: usize) -> Option<TokenSample> {
        if index >= self.len() {
            return None;
        }
        let start = index * self.stride;
        let input = (start..start + self.seq_len)
            .map(|i| self.token_at(i))
            .collect();
        let target = (start + 1..start + self.seq_len + 1)
            .map(|i| self.token_at(i))
            .collect();
        Some(TokenSample { input, target })
    }

    fn len(&self) -> usize {
        let tc = self.token_count;
        let sl = self.seq_len;
        // Need at least seq_len+1 tokens for one window (input + one-past target).
        if tc < sl + 1 {
            0
        } else {
            (tc - sl - 1) / self.stride + 1
        }
    }
}

#[cfg(test)]
#[path = "dataset_test.rs"]
mod tests;
