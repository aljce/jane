//! Corpus fetching, tokenization, binarization and the token dataset.
//!
//! Pipeline:
//!
//! ```text
//! DataSource::fetch  ->  raw .txt
//!   JaneTokenizer::train_from_files  ->  tokenizer.json
//!     binarize_text_file  ->  train.bin  (flat u16 LE)  +  train.bin.meta.json
//!       TokenDataset::open  ->  LmBatcher  ->  LmBatch { inputs, targets }
//! ```
//!
//! This crate takes `vocab_size` and `seq_len` as plain arguments and does not
//! depend on `jane-model`.

pub mod batcher;
pub mod binarize;
pub mod dataset;
pub mod meta;
pub mod source;
pub mod tokenizer;

pub use batcher::{LmBatch, LmBatcher};
pub use binarize::{BinarizeStats, binarize_text_file};
pub use dataset::{TokenDataset, TokenSample};
pub use meta::{TOKEN_BIN_FORMAT, TokenMeta};
pub use source::{DataSource, HfTextSource, RawTextSource};
pub use tokenizer::JaneTokenizer;

/// Errors across the data pipeline.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("download failed for {url}: {reason}")]
    Download { url: String, reason: String },

    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    Checksum {
        path: String,
        expected: String,
        actual: String,
    },

    #[error("huggingface import failed for {dataset}: {reason}")]
    HuggingFace { dataset: String, reason: String },

    #[error(
        "token id {id} does not fit in u16 (vocab_size {vocab_size}); \
         the .bin format is u16 and cannot represent vocabularies above 65536"
    )]
    TokenTooLarge { id: u32, vocab_size: usize },

    #[error("corrupt token file {path}: {reason}")]
    CorruptTokenFile { path: String, reason: String },

    #[error("metadata error: {0}")]
    Meta(String),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DataError>;

impl DataError {
    /// Attach a path to an [`std::io::Error`]. Prefer this over bare `?` on io
    /// operations — "No such file or directory" without a path is not a
    /// diagnostic.
    pub fn io(path: impl AsRef<std::path::Path>, source: std::io::Error) -> Self {
        DataError::Io {
            path: path.as_ref().display().to_string(),
            source,
        }
    }
}
