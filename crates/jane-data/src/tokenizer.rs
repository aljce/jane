//! Tokenizer trait and byte-level BPE implementation.
//!
//! We reuse HuggingFace's `tokenizers` crate for the *implementation* but train
//! our own *vocabulary* — see ROADMAP §3. At `d_model=384`, GPT-2's 50257-entry
//! vocabulary would make the embedding table 65% of the whole model.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use std::path::Path;

use crate::Result;

/// Marks document boundaries inside the flat token stream.
pub const EOT_TOKEN: &str = "<|endoftext|>";

/// Backend-agnostic tokenizer interface.
///
/// Encode/decode, vocabulary metadata, and the end-of-text marker. Everything
/// that touches a tokenizer in the pipeline goes through this trait so the
/// byte-level BPE backend can be swapped without rewiring call sites.
pub trait Tokenizer: Send + Sync {
    /// Encode without adding special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode, skipping special tokens.
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Total vocabulary size including special tokens.
    fn vocab_size(&self) -> usize;

    /// Id of [`EOT_TOKEN`].
    fn eot_id(&self) -> u32;
}

/// A trained byte-level BPE tokenizer.
pub struct ByteTokenizer {
    // Add fields as needed; `tokenizers::Tokenizer` is the expected core.
    _private: (),
}

impl ByteTokenizer {
    /// Train byte-level BPE over `files` to exactly `vocab_size` tokens.
    ///
    /// Requirements:
    /// - byte-level pre-tokenizer *and* matching byte-level decoder, so that
    ///   arbitrary bytes round-trip and no token maps to U+FFFD
    /// - [`EOT_TOKEN`] registered as a special token
    /// - `vocab_size` must be >= 256 (byte alphabet); error otherwise
    ///
    /// # Tests required
    /// Train a small tokenizer (vocab ~300) on a temp file of repetitive text,
    /// then assert:
    /// - [`ByteTokenizer::vocab_size`] is <= the requested size and >= 256
    /// - every id returned by [`ByteTokenizer::encode`] is `< vocab_size`
    /// - `vocab_size` below 256 is rejected
    pub fn train_from_files(_files: &[impl AsRef<Path>], _vocab_size: usize) -> Result<Self> {
        todo!("ByteTokenizer::train_from_files")
    }

    pub fn load(_path: impl AsRef<Path>) -> Result<Self> {
        todo!("ByteTokenizer::load")
    }

    /// Serialize to `tokenizer.json`.
    ///
    /// # Tests required
    /// Save then [`ByteTokenizer::load`], and assert the reloaded tokenizer
    /// encodes a sample string to the identical id sequence.
    pub fn save(&self, _path: impl AsRef<Path>) -> Result<()> {
        todo!("ByteTokenizer::save")
    }
}

impl Tokenizer for ByteTokenizer {
    /// # Tests required — this is the important one
    /// `decode(encode(s)) == s` exactly, for at least:
    /// - plain ASCII prose
    /// - the empty string (must give an empty id list)
    /// - leading/trailing/repeated whitespace (byte-level BPE must preserve it)
    /// - non-ASCII: `"héllo wörld"`, `"日本語のテキスト"`, `"🙂🙃"`
    /// - a string containing [`EOT_TOKEN`] literally
    /// - text with `\n` and `\t`
    fn encode(&self, _text: &str) -> Result<Vec<u32>> {
        todo!("ByteTokenizer::encode")
    }

    fn decode(&self, _ids: &[u32]) -> Result<String> {
        todo!("ByteTokenizer::decode")
    }

    fn vocab_size(&self) -> usize {
        todo!("ByteTokenizer::vocab_size")
    }

    /// # Tests required
    /// The id is `< vocab_size`, and decoding `[eot_id]` alone yields an empty
    /// string (special tokens are skipped).
    fn eot_id(&self) -> u32 {
        todo!("ByteTokenizer::eot_id")
    }
}

/// Lowercase hex SHA-256 of a file, for [`crate::TokenMeta::tokenizer_sha256`].
///
/// # Tests required
/// - the digest of a known short input matches a precomputed constant
///   (`""` -> `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`)
/// - two files with identical bytes hash equally; a one-byte change does not
pub fn sha256_of_file(_path: impl AsRef<Path>) -> Result<String> {
    todo!("sha256_of_file")
}
