//! Tokenizer trait and byte-level BPE implementation.
//!
//! We reuse HuggingFace's `tokenizers` crate for the *implementation* but train
//! our own *vocabulary* — see ROADMAP §3. At `d_model=384`, GPT-2's 50257-entry
//! vocabulary would make the embedding table 65% of the whole model.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use std::collections::HashSet;
use std::path::Path;

use sha2::{Digest, Sha256};
use tokenizers::{
    AddedToken, Tokenizer as HfTokenizer,
    models::{
        TrainerWrapper,
        bpe::{BPE, BpeTrainer},
    },
    pre_tokenizers::byte_level::ByteLevel,
};

use crate::{DataError, Result};

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
    inner: HfTokenizer,
}

impl std::fmt::Debug for ByteTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteTokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
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
    pub fn train_from_files(files: &[impl AsRef<Path>], vocab_size: usize) -> Result<Self> {
        if vocab_size < 256 {
            return Err(DataError::Tokenizer(format!(
                "vocab_size must be >= 256 (byte alphabet), got {vocab_size}"
            )));
        }

        let eot_token = AddedToken::from(EOT_TOKEN, true);

        // ByteLevel::alphabet() returns AHashSet; BpeTrainer::initial_alphabet
        // takes std HashSet. Collect to convert.
        let alphabet: HashSet<char> = ByteLevel::alphabet().into_iter().collect();

        let trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(0)
            .show_progress(false)
            .special_tokens(vec![eot_token])
            .initial_alphabet(alphabet)
            .build();
        // Wrap in TrainerWrapper so the type satisfies Trainer<Model = ModelWrapper>.
        let mut trainer: TrainerWrapper = trainer.into();

        let mut tokenizer = HfTokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        // ByteLevel implements both PreTokenizer and Decoder: it maps each byte
        // to a visible unicode character before BPE and reverses that mapping on
        // decode, guaranteeing lossless round-trips for arbitrary byte sequences.
        tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
        // When encode_special_tokens = true, the tokenizer does NOT match special
        // token strings in the input (passing them through the BPE model as
        // ordinary bytes). This is what we want: special tokens like EOT are
        // inserted explicitly by the caller via eot_id(), never via encode().
        // Without this, encode("<|endoftext|>") returns [eot_id] and the
        // round-trip fails for text that literally contains the EOT string.
        tokenizer.set_encode_special_tokens(true);

        let file_paths: Vec<String> = files
            .iter()
            .map(|p| p.as_ref().display().to_string())
            .collect();

        tokenizer
            .train_from_files(&mut trainer, file_paths)
            .map_err(|e| DataError::Tokenizer(e.to_string()))?;

        Ok(Self { inner: tokenizer })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut tokenizer =
            HfTokenizer::from_file(path).map_err(|e| DataError::Tokenizer(e.to_string()))?;
        // Restore the flag that isn't serialized: special tokens in input text
        // should be treated as ordinary bytes, not matched as special tokens.
        tokenizer.set_encode_special_tokens(true);
        Ok(Self { inner: tokenizer })
    }

    /// Serialize to `tokenizer.json`.
    ///
    /// # Tests required
    /// Save then [`ByteTokenizer::load`], and assert the reloaded tokenizer
    /// encodes a sample string to the identical id sequence.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        self.inner
            .save(path, false)
            .map_err(|e| DataError::Tokenizer(e.to_string()))
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
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| DataError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode, skipping special tokens.
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| DataError::Tokenizer(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// # Tests required
    /// The id is `< vocab_size`, and decoding `[eot_id]` alone yields an empty
    /// string (special tokens are skipped).
    fn eot_id(&self) -> u32 {
        self.inner
            .token_to_id(EOT_TOKEN)
            .expect("EOT_TOKEN must be registered as a special token")
    }
}

/// Lowercase hex SHA-256 of a file, for [`crate::TokenMeta::tokenizer_sha256`].
///
/// # Tests required
/// - the digest of a known short input matches a precomputed constant
///   (`""` -> `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`)
/// - two files with identical bytes hash equally; a one-byte change does not
pub fn sha256_of_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path).map_err(|e| DataError::io(path, e))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).map_err(|e| DataError::io(path, e))?;
    let digest = hasher.finalize();
    Ok(format!("{digest:x}"))
}

#[cfg(test)]
#[path = "tokenizer_test.rs"]
mod tests;
