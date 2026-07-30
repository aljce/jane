//! Corpus acquisition.
//!
//! Two paths, behind one trait so the rest of the pipeline doesn't care which
//! produced the text (ROADMAP §2).
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use std::path::{Path, PathBuf};

use crate::Result;

/// Something that can produce a local plain-text corpus file.
pub trait DataSource: Send + Sync {
    /// Short stable identifier, used in log lines and [`crate::TokenMeta::source`].
    fn name(&self) -> &str;

    /// Materialize the corpus into `cache_dir`, returning the `.txt` path.
    ///
    /// Must be **idempotent and cheap on a second call**: if the target already
    /// exists and passes verification, return it without re-downloading. These
    /// are multi-gigabyte files.
    fn fetch(&self, cache_dir: &Path) -> Result<PathBuf>;
}

/// A plain-text file over HTTP, fetched with `curl`.
///
/// Used for tiny-shakespeare (not an HF dataset) and for the
/// `TinyStoriesV2-GPT4-*.txt` files, which are loose files in the HF repo rather
/// than rows in its parquet conversion.
pub struct RawTextSource {
    pub name: String,
    pub url: String,
    /// Filename within the cache dir.
    pub filename: String,
    /// Optional expected lowercase hex SHA-256.
    pub sha256: Option<String>,
}

impl RawTextSource {
    pub fn new(name: impl Into<String>, url: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            filename: filename.into(),
            sha256: None,
        }
    }

    pub fn with_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.sha256 = Some(sha256.into());
        self
    }

    /// tiny-shakespeare, ~1 MB. ROADMAP data ladder rung 0.
    pub fn tiny_shakespeare() -> Self {
        Self::new(
            "tiny-shakespeare",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "tiny-shakespeare.txt",
        )
    }

    /// TinyStories V2 (GPT-4 only), ~2.2 GB. Ladder rung 2.
    pub fn tinystories_v2_train() -> Self {
        Self::new(
            "tinystories-v2-train",
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
            "TinyStoriesV2-GPT4-train.txt",
        )
    }

    /// TinyStories V2 validation split, ~22 MB.
    pub fn tinystories_v2_valid() -> Self {
        Self::new(
            "tinystories-v2-valid",
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
            "TinyStoriesV2-GPT4-valid.txt",
        )
    }
}

impl DataSource for RawTextSource {
    fn name(&self) -> &str {
        &self.name
    }

    /// Download to a `.part` file and rename only on success, so an interrupted
    /// transfer can never be mistaken for a complete corpus on the next run.
    /// Use `curl -fL --retry 3`. Verify `sha256` when set.
    ///
    /// # Tests required
    /// Network tests must be `#[ignore]`d with a comment saying so. Cover the
    /// offline logic instead:
    /// - a pre-existing correct file short-circuits (no network): create the
    ///   target by hand with a matching `sha256` and assert `fetch` returns it
    /// - a pre-existing file with the *wrong* sha256 is an error, not a silent pass
    /// - a stale `.part` file does not get returned as the corpus
    /// - the returned path is inside `cache_dir` and named `filename`
    /// - `cache_dir` is created when absent
    /// - one `#[ignore]`d live test against tiny-shakespeare (~1 MB)
    fn fetch(&self, _cache_dir: &Path) -> Result<PathBuf> {
        todo!("RawTextSource::fetch")
    }
}

/// A HuggingFace dataset, imported via Burn's `HuggingfaceDatasetLoader` and
/// flattened to a plain-text file.
pub struct HfTextSource {
    pub name: String,
    /// e.g. `"roneneldan/TinyStories"`.
    pub dataset: String,
    pub subset: Option<String>,
    /// e.g. `"train"`.
    pub split: String,
    /// Row field holding the text, e.g. `"text"`.
    pub text_column: String,
    /// Stop after this many rows. Keeps smoke tests quick.
    pub limit: Option<usize>,
}

impl HfTextSource {
    pub fn new(
        name: impl Into<String>,
        dataset: impl Into<String>,
        split: impl Into<String>,
        text_column: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            dataset: dataset.into(),
            subset: None,
            split: split.into(),
            text_column: text_column.into(),
            limit: None,
        }
    }

    pub fn with_subset(mut self, subset: impl Into<String>) -> Self {
        self.subset = Some(subset.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

impl DataSource for HfTextSource {
    fn name(&self) -> &str {
        &self.name
    }

    /// Import via `HuggingfaceDatasetLoader`, then write one document per row to
    /// a `.txt`, separated by [`crate::tokenizer::EOT_TOKEN`].
    ///
    /// **`.with_use_python_venv(false)` is mandatory.** The loader otherwise
    /// builds a venv and `pip install`s pyarrow/datasets, whose binary wheels
    /// have unpatched ELF interpreters and cannot execute on NixOS. The flake
    /// already provides these packages; we must use the ambient `python3`.
    /// Do this in one place so it cannot be forgotten at another call site.
    ///
    /// Point `.with_base_dir()` at `cache_dir` so the SQLite import lands under
    /// `data/` and is removed by `rm -rf data/`.
    ///
    /// Rows deserialize into a `serde_json::Map<String, serde_json::Value>` (or
    /// an equivalent), from which `text_column` is read; a missing column must
    /// be an error naming the column and the columns that were present.
    ///
    /// # Tests required
    /// Anything touching the network or Python must be `#[ignore]`d. Cover:
    /// - the row-to-text flattening as a pure helper over synthetic rows,
    ///   including a missing-column error and non-string column values
    /// - `with_subset` / `with_limit` builders set what they claim
    /// - `limit` truncates
    /// - one `#[ignore]`d end-to-end test against a small real HF dataset that
    ///   asserts the venv is *not* used (no `venv/` directory under `cache_dir`)
    fn fetch(&self, _cache_dir: &Path) -> Result<PathBuf> {
        todo!("HfTextSource::fetch")
    }
}
