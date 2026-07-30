//! Text -> flat `u16` token file.
//!
//! Run once, ahead of training. The hot loop must never parse text.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.
//!
//! Output layout is fixed by [`crate::meta`]: flat little-endian `u16`, no
//! header, plus a `.meta.json` sidecar. Read that module first.

use std::{
    io::{BufRead, BufWriter, Write},
    path::Path,
};

use crate::{DataError, Result, meta::TokenMeta, tokenizer::Tokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BinarizeStats {
    /// Tokens written, including one [`crate::tokenizer::EOT_TOKEN`] per document.
    pub tokens: u64,
    pub docs: u64,
    pub bytes_in: u64,
}

/// Write a single token id as little-endian u16 to the writer.
///
/// Returns `TokenTooLarge` before writing if the id does not fit in u16.
#[inline]
fn write_token(
    writer: &mut impl Write,
    id: u32,
    vocab_size: usize,
    stats: &mut BinarizeStats,
) -> Result<()> {
    if id >= 65536 {
        return Err(DataError::TokenTooLarge { id, vocab_size });
    }
    writer
        .write_all(&(id as u16).to_le_bytes())
        .map_err(|e| DataError::io("<output>", e))?;
    stats.tokens += 1;
    Ok(())
}

/// Encode `text` and write all token ids (no EOT).
///
/// Silently skips empty text. Used to stream within-document content.
fn write_content(
    text: &str,
    tokenizer: &impl Tokenizer,
    writer: &mut impl Write,
    stats: &mut BinarizeStats,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let ids = tokenizer.encode(text)?;
    let vs = tokenizer.vocab_size();
    for id in ids {
        write_token(writer, id, vs, stats)?;
    }
    Ok(())
}

/// Write the EOT token and increment doc count. Call after all content of a
/// non-empty document has been written.
fn write_eot(
    tokenizer: &impl Tokenizer,
    writer: &mut impl Write,
    stats: &mut BinarizeStats,
) -> Result<()> {
    let eot = tokenizer.eot_id();
    write_token(writer, eot, tokenizer.vocab_size(), stats)?;
    stats.docs += 1;
    Ok(())
}

/// Returns the largest byte index `<= index` that is a valid UTF-8 char boundary.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
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
    text_file: impl AsRef<Path>,
    tokenizer: &impl Tokenizer,
    out_bin: impl AsRef<Path>,
    doc_sep: Option<&str>,
) -> Result<BinarizeStats> {
    let text_file = text_file.as_ref();
    let out_bin = out_bin.as_ref();

    let in_file = std::fs::File::open(text_file).map_err(|e| DataError::io(text_file, e))?;
    let mut reader = std::io::BufReader::new(in_file);

    let out_file = std::fs::File::create(out_bin).map_err(|e| DataError::io(out_bin, e))?;
    let mut writer = BufWriter::new(out_file);

    let mut stats = BinarizeStats::default();

    match doc_sep {
        None => {
            // Whole file is one document. Stream line-by-line; tokenize each
            // line-chunk individually so we never hold the full file in RAM.
            // Track whether we wrote any tokens (empty file → no EOT).
            let mut wrote_any = false;
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader
                    .read_line(&mut line)
                    .map_err(|e| DataError::io(text_file, e))?;
                if n == 0 {
                    break;
                }
                stats.bytes_in += n as u64;
                if !line.is_empty() {
                    write_content(&line, tokenizer, &mut writer, &mut stats)?;
                    wrote_any = true;
                }
            }
            if wrote_any {
                write_eot(tokenizer, &mut writer, &mut stats)?;
            }
        }
        Some(sep) => {
            // Split on `sep`, streaming line by line.
            //
            // Invariant: `tail` holds text that has been read but not yet
            // tokenized. We keep the last `sep.len() - 1` bytes unsent so we
            // can detect separators that straddle two successive reads.
            //
            // `current_doc_has_content` tracks whether the current document
            // already has tokens written (needed to suppress EOT for empty docs).
            let sep_len = sep.len();
            let mut tail = String::new();
            let mut current_doc_has_content = false;

            let mut line = String::new();
            loop {
                line.clear();
                let n = reader
                    .read_line(&mut line)
                    .map_err(|e| DataError::io(text_file, e))?;
                if n == 0 {
                    break;
                }
                stats.bytes_in += n as u64;
                tail.push_str(&line);

                // Flush all complete separators found in `tail`.
                loop {
                    match tail.find(sep) {
                        Some(pos) => {
                            // The text before the separator is the end of the
                            // current document.
                            let doc_end = tail[..pos].to_string();
                            tail.drain(..pos + sep_len);

                            if !doc_end.is_empty() {
                                write_content(&doc_end, tokenizer, &mut writer, &mut stats)?;
                                current_doc_has_content = true;
                            }

                            if current_doc_has_content {
                                // Close the current document.
                                write_eot(tokenizer, &mut writer, &mut stats)?;
                            }
                            // Start a fresh document (empty doc state).
                            current_doc_has_content = false;
                        }
                        None => {
                            // No separator in `tail` yet. Flush everything
                            // except the last `sep_len - 1` bytes which might
                            // be the start of a separator that continues in the
                            // next read.
                            if sep_len > 0 && tail.len() >= sep_len {
                                let safe = tail.len() - (sep_len - 1);
                                let boundary = floor_char_boundary(&tail, safe);
                                if boundary > 0 {
                                    let segment = tail[..boundary].to_string();
                                    tail.drain(..boundary);
                                    write_content(&segment, tokenizer, &mut writer, &mut stats)?;
                                    if !segment.is_empty() {
                                        current_doc_has_content = true;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }

            // After all reads: flush any remaining text as the last document.
            if !tail.is_empty() {
                write_content(&tail, tokenizer, &mut writer, &mut stats)?;
                current_doc_has_content = true;
            }
            if current_doc_has_content {
                write_eot(tokenizer, &mut writer, &mut stats)?;
            }
        }
    }

    writer.flush().map_err(|e| DataError::io(out_bin, e))?;

    // Build the sidecar. The tokenizer path is not passed here, so we cannot
    // compute its SHA-256 — leave it empty. Callers that need it can call
    // sha256_of_file on the tokenizer.json themselves and overwrite the field.
    let meta = TokenMeta::new(
        stats.tokens,
        tokenizer.vocab_size(),
        "",
        text_file.file_name().unwrap_or_default().to_string_lossy(),
        stats.docs,
    );
    meta.save_for(out_bin)?;

    Ok(stats)
}

#[cfg(test)]
#[path = "binarize_test.rs"]
mod tests;
