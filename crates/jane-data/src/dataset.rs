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
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Write a flat little-endian u16 sequence and a matching sidecar, return the
    /// dir (kept alive) and the bin path.
    fn make_fixture(tokens: &[u16]) -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("train.bin");

        let mut f = File::create(&bin).unwrap();
        for t in tokens {
            f.write_all(&t.to_le_bytes()).unwrap();
        }

        let meta = TokenMeta::new(tokens.len() as u64, 65536, "deadbeef", "test", 0);
        meta.save_for(&bin).unwrap();

        (dir, bin)
    }

    // ---- open() validation ----

    #[test]
    fn rejects_seq_len_zero() {
        let (_dir, bin) = make_fixture(&[1, 2, 3]);
        let err = TokenDataset::open(&bin, 0, 1).unwrap_err();
        assert!(matches!(err, DataError::Meta(_)), "{err:?}");
    }

    #[test]
    fn rejects_stride_zero() {
        let (_dir, bin) = make_fixture(&[1, 2, 3]);
        let err = TokenDataset::open(&bin, 1, 0).unwrap_err();
        assert!(matches!(err, DataError::Meta(_)), "{err:?}");
    }

    #[test]
    fn rejects_odd_byte_length() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("odd.bin");
        // Write 3 bytes — not a whole number of u16s.
        std::fs::write(&bin, &[0u8, 1, 2]).unwrap();

        // Write a sidecar that claims 1 token (2 bytes) so the check fires on
        // the odd-byte guard, not the missing-sidecar guard.
        let meta = TokenMeta::new(1, 256, "x", "s", 0);
        meta.save_for(&bin).unwrap();

        let err = TokenDataset::open(&bin, 1, 1).unwrap_err();
        assert!(matches!(err, DataError::CorruptTokenFile { .. }), "{err:?}");
    }

    #[test]
    fn rejects_token_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("mismatch.bin");
        // Write 4 bytes = 2 tokens.
        std::fs::write(&bin, &[1u8, 0, 2, 0]).unwrap();
        // Sidecar claims 5 tokens.
        let meta = TokenMeta::new(5, 256, "x", "s", 0);
        meta.save_for(&bin).unwrap();

        let err = TokenDataset::open(&bin, 1, 1).unwrap_err();
        assert!(matches!(err, DataError::CorruptTokenFile { .. }), "{err:?}");
    }

    #[test]
    fn rejects_missing_sidecar_with_path_in_message() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("nosidecar.bin");
        std::fs::write(&bin, &[0u8, 0]).unwrap();
        // No sidecar written.

        let err = TokenDataset::open(&bin, 1, 1).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("nosidecar"),
            "path not in error message: {msg}"
        );
    }

    // ---- token_at ----

    #[test]
    fn token_at_roundtrip_including_high_values() {
        let tokens: Vec<u16> = vec![0, 1, 255, 256, 1000, 32767, 65535];
        let (_dir, bin) = make_fixture(&tokens);
        let ds = TokenDataset::open(&bin, 1, 1).unwrap();
        for (i, &t) in tokens.iter().enumerate() {
            assert_eq!(ds.token_at(i), t as u32, "index {i}");
        }
    }

    // ---- window count (len) arithmetic ----

    fn window_count(token_count: usize, seq_len: usize, stride: usize) -> usize {
        if token_count < seq_len + 1 {
            0
        } else {
            (token_count - seq_len - 1) / stride + 1
        }
    }

    #[test]
    fn len_arithmetic_contract_cases() {
        assert_eq!(window_count(11, 10, 10), 1);
        assert_eq!(window_count(10, 10, 10), 0);
        assert_eq!(window_count(21, 10, 10), 2);
        assert_eq!(window_count(20, 10, 10), 1);
        assert_eq!(window_count(13, 4, 1), 9);
        assert_eq!(window_count(0, 1, 1), 0);
    }

    #[test]
    fn len_arithmetic_brute_force_cross_check() {
        // For every combination in the specified range, compare formula against
        // a brute-force count of valid start positions.
        for tc in 0usize..40 {
            for sl in 1usize..8 {
                for st in 1usize..8 {
                    let formula = window_count(tc, sl, st);
                    // Brute force: count starts where start + seq_len < token_count
                    // (i.e. there is room for the shifted target at start + seq_len).
                    let brute: usize = (0..tc).step_by(st).filter(|&s| s + sl < tc).count();
                    assert_eq!(
                        formula, brute,
                        "tc={tc} sl={sl} st={st}: formula={formula} brute={brute}"
                    );
                }
            }
        }
    }

    // ---- Dataset::get ----

    fn make_sequential(n: usize) -> (TempDir, std::path::PathBuf) {
        let tokens: Vec<u16> = (0..n as u16).collect();
        make_fixture(&tokens)
    }

    #[test]
    fn get_returns_none_at_and_beyond_len() {
        // token_count=11, seq_len=10, stride=10 -> len=1
        let (_dir, bin) = make_sequential(11);
        let ds = TokenDataset::open(&bin, 10, 10).unwrap();
        assert_eq!(ds.len(), 1);
        assert!(ds.get(0).is_some());
        assert!(ds.get(1).is_none());
        assert!(ds.get(2).is_none());
    }

    #[test]
    fn get_lengths_are_seq_len() {
        let (_dir, bin) = make_sequential(20);
        let ds = TokenDataset::open(&bin, 5, 3).unwrap();
        for i in 0..ds.len() {
            let s = ds.get(i).unwrap();
            assert_eq!(s.input.len(), 5, "window {i} input len");
            assert_eq!(s.target.len(), 5, "window {i} target len");
        }
    }

    #[test]
    fn get_shift_property() {
        // Tokens 0,1,2,...,19 — easy to verify shifts.
        let (_dir, bin) = make_sequential(20);
        let ds = TokenDataset::open(&bin, 5, 3).unwrap();
        for i in 0..ds.len() {
            let s = ds.get(i).unwrap();
            let seq_len = ds.seq_len();
            // target[j] == input[j+1] for j < seq_len-1
            for j in 0..seq_len - 1 {
                assert_eq!(
                    s.target[j],
                    s.input[j + 1],
                    "window {i}: target[{j}] != input[{}]",
                    j + 1
                );
            }
            // target.last() is the token just past the input window
            let start = i * ds.stride();
            assert_eq!(
                *s.target.last().unwrap(),
                ds.token_at(start + seq_len),
                "window {i}: last target wrong"
            );
        }
    }

    #[test]
    fn get_starts_at_correct_positions() {
        let (_dir, bin) = make_sequential(30);
        let ds = TokenDataset::open(&bin, 4, 5).unwrap();
        // get(0) starts at token 0
        assert_eq!(ds.get(0).unwrap().input[0], 0);
        // get(1) starts at token stride=5
        assert_eq!(ds.get(1).unwrap().input[0], 5);
    }

    #[test]
    fn final_window_never_reads_past_end() {
        // Brute-force: for every valid (tc, sl, st) combo, the last window's
        // last target index must be < token_count.
        for tc in 1usize..40 {
            for sl in 1usize..8 {
                for st in 1usize..8 {
                    let count = window_count(tc, sl, st);
                    if count == 0 {
                        continue;
                    }
                    let last_start = (count - 1) * st;
                    let last_target_end = last_start + sl; // inclusive index of last target token
                    assert!(
                        last_target_end < tc,
                        "tc={tc} sl={sl} st={st}: last target index {last_target_end} >= {tc}"
                    );
                }
            }
        }
    }

    #[test]
    fn get_via_dataset_matches_token_at() {
        // Cross-check get() against token_at() for a realistic configuration.
        let tokens: Vec<u16> = (0..50).map(|i| i * 17 + 3).collect();
        let (_dir, bin) = make_fixture(&tokens);
        let ds = TokenDataset::open(&bin, 7, 3).unwrap();
        for i in 0..ds.len() {
            let s = ds.get(i).unwrap();
            let start = i * 3;
            for j in 0..7 {
                assert_eq!(s.input[j], ds.token_at(start + j));
                assert_eq!(s.target[j], ds.token_at(start + j + 1));
            }
        }
    }
}
