//! The on-disk token format.
//!
//! # OWNED MODULE — do not edit.
//!
//! Both the binarizer (writer) and the dataset (reader) depend on this, so it is
//! written in full and fixed. It is the contract between them.
//!
//! ## Format
//!
//! A `.bin` file is a **flat little-endian `u16` array of token ids and nothing
//! else** — no header, no padding, no separators. Document boundaries are
//! ordinary tokens (the tokenizer's end-of-text token) inside the stream. This
//! keeps the file trivially memory-mappable: token `i` is bytes `2i..2i+2`.
//!
//! `u16` bounds `vocab_size` at 65536, which is far above anything in ROADMAP
//! §3.2. The binarizer must reject out-of-range ids rather than truncate.
//!
//! Because the `.bin` carries no header, everything needed to validate it lives
//! in a JSON sidecar at `<name>.bin.meta.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{DataError, Result};

/// Format tag written into every sidecar. Bump when the layout changes.
pub const TOKEN_BIN_FORMAT: &str = "jane-tokens-v1";

/// Sidecar describing a `.bin` token file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenMeta {
    /// Always [`TOKEN_BIN_FORMAT`].
    pub format: String,
    /// Number of `u16` tokens; the `.bin` must be exactly `2 * token_count` bytes.
    pub token_count: u64,
    /// Vocabulary the ids were produced with. Guards against pairing a `.bin`
    /// with a mismatched tokenizer, which otherwise fails as garbage output
    /// rather than an error.
    pub vocab_size: usize,
    /// SHA-256 of the `tokenizer.json` used, for the same reason.
    pub tokenizer_sha256: String,
    /// Human-readable provenance, e.g. `"TinyStoriesV2-GPT4-train.txt"`.
    pub source: String,
    /// Documents seen (end-of-text tokens emitted).
    pub doc_count: u64,
}

impl TokenMeta {
    /// `data/train.bin` -> `data/train.bin.meta.json`.
    ///
    /// Appends rather than replacing the extension, so `train.bin` and
    /// `train.txt` can coexist without their sidecars colliding.
    pub fn sidecar_path(bin: impl AsRef<Path>) -> PathBuf {
        let bin = bin.as_ref();
        let mut name = bin.file_name().unwrap_or_default().to_os_string();
        name.push(".meta.json");
        bin.with_file_name(name)
    }

    pub fn new(
        token_count: u64,
        vocab_size: usize,
        tokenizer_sha256: impl Into<String>,
        source: impl Into<String>,
        doc_count: u64,
    ) -> Self {
        Self {
            format: TOKEN_BIN_FORMAT.to_string(),
            token_count,
            vocab_size,
            tokenizer_sha256: tokenizer_sha256.into(),
            source: source.into(),
            doc_count,
        }
    }

    /// Write the sidecar for `bin`.
    pub fn save_for(&self, bin: impl AsRef<Path>) -> Result<()> {
        let path = Self::sidecar_path(bin);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json).map_err(|e| DataError::io(&path, e))
    }

    /// Read the sidecar for `bin`.
    pub fn load_for(bin: impl AsRef<Path>) -> Result<Self> {
        let path = Self::sidecar_path(bin);
        let raw = std::fs::read_to_string(&path).map_err(|e| DataError::io(&path, e))?;
        let meta: TokenMeta = serde_json::from_str(&raw)?;
        if meta.format != TOKEN_BIN_FORMAT {
            return Err(DataError::Meta(format!(
                "{}: unsupported format {:?}, expected {:?}",
                path.display(),
                meta.format,
                TOKEN_BIN_FORMAT
            )));
        }
        Ok(meta)
    }

    /// Check the sidecar against the actual `.bin` size.
    pub fn check_bin_len(&self, bin: impl AsRef<Path>, len_bytes: u64) -> Result<()> {
        let path = bin.as_ref().display().to_string();
        if len_bytes % 2 != 0 {
            return Err(DataError::CorruptTokenFile {
                path,
                reason: format!("{len_bytes} bytes is not a whole number of u16 tokens"),
            });
        }
        let actual = len_bytes / 2;
        if actual != self.token_count {
            return Err(DataError::CorruptTokenFile {
                path,
                reason: format!(
                    "sidecar declares {} tokens but file holds {actual}",
                    self.token_count
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_path_appends_rather_than_replaces() {
        assert_eq!(
            TokenMeta::sidecar_path("data/train.bin"),
            PathBuf::from("data/train.bin.meta.json")
        );
        // The distinguishing case: replacing the extension would collide here.
        assert_ne!(
            TokenMeta::sidecar_path("data/train.bin"),
            TokenMeta::sidecar_path("data/train.txt")
        );
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("train.bin");
        let meta = TokenMeta::new(42, 8192, "abc123", "test.txt", 7);

        meta.save_for(&bin).unwrap();
        assert_eq!(TokenMeta::load_for(&bin).unwrap(), meta);
    }

    #[test]
    fn rejects_unknown_format_tag() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("train.bin");
        let mut meta = TokenMeta::new(1, 256, "x", "s", 0);
        meta.format = "jane-tokens-v999".into();
        meta.save_for(&bin).unwrap();

        let err = TokenMeta::load_for(&bin).unwrap_err();
        assert!(matches!(err, DataError::Meta(_)), "got {err:?}");
    }

    #[test]
    fn check_bin_len_accepts_exact_match() {
        let meta = TokenMeta::new(10, 256, "x", "s", 0);
        meta.check_bin_len("t.bin", 20).unwrap();
    }

    #[test]
    fn check_bin_len_rejects_odd_byte_count() {
        let meta = TokenMeta::new(10, 256, "x", "s", 0);
        let err = meta.check_bin_len("t.bin", 21).unwrap_err();
        assert!(matches!(err, DataError::CorruptTokenFile { .. }), "got {err:?}");
    }

    #[test]
    fn check_bin_len_rejects_token_count_mismatch() {
        let meta = TokenMeta::new(10, 256, "x", "s", 0);
        let err = meta.check_bin_len("t.bin", 18).unwrap_err();
        match err {
            DataError::CorruptTokenFile { reason, .. } => {
                assert!(reason.contains("10") && reason.contains('9'), "{reason}");
            }
            other => panic!("got {other:?}"),
        }
    }
}
