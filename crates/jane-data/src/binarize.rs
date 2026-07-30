//! Text -> flat `u16` token file.
//!
//! Run once, ahead of training. The hot loop must never parse text.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.
//!
//! Output layout is fixed by [`crate::meta`]: flat little-endian `u16`, no
//! header, plus a `.meta.json` sidecar. Read that module first.

use std::path::Path;

use crate::{Result, tokenizer::Tokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BinarizeStats {
    /// Tokens written, including one [`crate::tokenizer::EOT_TOKEN`] per document.
    pub tokens: u64,
    pub docs: u64,
    pub bytes_in: u64,
}

/// Tokenize `text_file` into `out_bin`, writing the sidecar alongside it.
///
/// Behaviour:
/// - **Must stream.** The corpus is 2.2 GB against 15 GB of RAM shared with
///   everything else; never hold the whole file (or the whole token vector) in
///   memory. Read in chunks, flush through a `BufWriter`.
/// - `doc_sep`: when `Some(sep)`, split the input on `sep` and emit exactly one
///   EOT token after each non-empty document. When `None`, treat the whole file
///   as one document and emit a single trailing EOT. TinyStories uses
///   `Some("<|endoftext|>")`.
/// - Reject any id `>= 65536` with [`crate::DataError::TokenTooLarge`] rather
///   than truncating — a silent `as u16` cast here would corrupt the corpus in a
///   way that only shows up as bad samples hours into training.
/// - Chunk boundaries must not split a document separator. Carry a tail buffer.
///
/// # Tests required
/// - a file of two documents separated by the EOT marker produces
///   `docs == 2` and a token stream containing exactly two EOT ids
/// - `tokens` equals the actual `.bin` length / 2, and the sidecar agrees
///   (`TokenMeta::check_bin_len` passes)
/// - the written bytes are little-endian: build a tokenizer, binarize a short
///   string, and compare against `encode()` mapped through `to_le_bytes`
/// - an empty input file yields a valid (possibly zero-token) `.bin` plus sidecar
/// - `doc_sep: None` emits exactly one EOT
/// - consecutive separators do not emit empty documents
/// - **streaming correctness**: binarizing a multi-megabyte input where
///   documents straddle chunk boundaries gives the same token stream as
///   tokenizing the whole string in one call
/// - the sidecar records the tokenizer sha256 and vocab_size
pub fn binarize_text_file(
    _text_file: impl AsRef<Path>,
    _tokenizer: &impl Tokenizer,
    _out_bin: impl AsRef<Path>,
    _doc_sep: Option<&str>,
) -> Result<BinarizeStats> {
    todo!("binarize_text_file")
}
