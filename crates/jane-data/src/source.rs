//! Corpus acquisition.
//!
//! Two paths, behind one trait so the rest of the pipeline doesn't care which
//! produced the text (ROADMAP §2).
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::tokenizer::EOT_TOKEN;
use crate::{DataError, Result};

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
    pub fn new(
        name: impl Into<String>,
        url: impl Into<String>,
        filename: impl Into<String>,
    ) -> Self {
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
    fn fetch(&self, cache_dir: &Path) -> Result<PathBuf> {
        // Create the cache directory if it does not already exist.
        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir).map_err(|e| DataError::io(cache_dir, e))?;
        }

        let dest = cache_dir.join(&self.filename);
        let part = cache_dir.join(format!("{}.part", self.filename));

        // If the destination file already exists, verify and return early.
        if dest.exists() {
            if let Some(expected) = &self.sha256 {
                let actual = crate::tokenizer::sha256_of_file(&dest)?;
                if actual != *expected {
                    return Err(DataError::Checksum {
                        path: dest.display().to_string(),
                        expected: expected.clone(),
                        actual,
                    });
                }
            }
            return Ok(dest);
        }

        // Remove any stale .part file from a previous interrupted download so
        // curl always starts fresh and we never serve a partial file.
        if part.exists() {
            fs::remove_file(&part).map_err(|e| DataError::io(&part, e))?;
        }

        // Download via curl. -f: fail on server errors, -L: follow redirects,
        // --retry 3: automatic retry on transient failures.
        let status = Command::new("curl")
            .args(["-fL", "--retry", "3", "-o"])
            .arg(&part)
            .arg(&self.url)
            .status()
            .map_err(|e| DataError::Download {
                url: self.url.clone(),
                reason: format!("failed to spawn curl: {e}"),
            })?;

        if !status.success() {
            return Err(DataError::Download {
                url: self.url.clone(),
                reason: format!("curl exited with {status}"),
            });
        }

        // Verify checksum before promoting the .part file.
        if let Some(expected) = &self.sha256 {
            let actual = crate::tokenizer::sha256_of_file(&part)?;
            if actual != *expected {
                // Remove the corrupt download so a retry starts fresh.
                let _ = fs::remove_file(&part);
                return Err(DataError::Checksum {
                    path: part.display().to_string(),
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        // Atomic rename: the final path is only visible once fully downloaded
        // and verified, so a crash here can never leave a truncated corpus.
        fs::rename(&part, &dest).map_err(|e| DataError::io(&dest, e))?;

        Ok(dest)
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

    /// Extract the text value for `column` from a JSON object row.
    ///
    /// Returns an error that names the missing column and all available columns
    /// so the user can diagnose a misconfigured `text_column` immediately.
    fn extract_text(row: &serde_json::Map<String, Value>, column: &str) -> Result<String> {
        match row.get(column) {
            Some(Value::String(s)) => Ok(s.clone()),
            Some(other) => {
                // Non-string values: serialize them so the corpus at least
                // contains something human-readable rather than silently
                // losing the row.
                Ok(other.to_string())
            }
            None => {
                let present: Vec<&str> = row.keys().map(String::as_str).collect();
                Err(DataError::HuggingFace {
                    dataset: column.to_string(),
                    reason: format!(
                        "column '{}' not found; present columns: [{}]",
                        column,
                        present.join(", ")
                    ),
                })
            }
        }
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
    fn fetch(&self, cache_dir: &Path) -> Result<PathBuf> {
        use burn_dataset::Dataset as _;
        use burn_dataset::HuggingfaceDatasetLoader;

        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir).map_err(|e| DataError::io(cache_dir, e))?;
        }

        let out_path = cache_dir.join(format!("{}.txt", self.name));

        // Return early if the file is already present — idempotent, no Python.
        if out_path.exists() {
            return Ok(out_path);
        }

        // Build the loader. with_use_python_venv(false) is mandatory: on NixOS
        // the venv wheels have unpatched ELF interpreters and cannot execute.
        let mut loader = HuggingfaceDatasetLoader::new(&self.dataset)
            .with_use_python_venv(false)
            .with_base_dir(cache_dir.to_str().ok_or_else(|| DataError::HuggingFace {
                dataset: self.dataset.clone(),
                reason: "cache_dir is not valid UTF-8".into(),
            })?);

        if let Some(subset) = &self.subset {
            loader = loader.with_subset(subset);
        }

        // Each row is a JSON object; we deserialize generically rather than
        // binding to a fixed struct so any dataset schema works.
        let ds: burn_dataset::SqliteDataset<serde_json::Map<String, Value>> = loader
            .dataset(&self.split)
            .map_err(|e| DataError::HuggingFace {
                dataset: self.dataset.clone(),
                reason: e.to_string(),
            })?;

        let total = ds.len();
        let take = self.limit.unwrap_or(total).min(total);

        // Write to a .part file and rename atomically on success.
        let part_path = cache_dir.join(format!("{}.txt.part", self.name));
        let mut file = fs::File::create(&part_path).map_err(|e| DataError::io(&part_path, e))?;

        for i in 0..take {
            let row = ds.get(i).ok_or_else(|| DataError::HuggingFace {
                dataset: self.dataset.clone(),
                reason: format!("row {i} disappeared during iteration"),
            })?;

            let text = Self::extract_text(&row, &self.text_column)?;
            file.write_all(text.as_bytes())
                .map_err(|e| DataError::io(&part_path, e))?;
            // Documents are separated by the end-of-text token so the
            // downstream tokenizer can treat each row as an independent sample.
            file.write_all(EOT_TOKEN.as_bytes())
                .map_err(|e| DataError::io(&part_path, e))?;
            file.write_all(b"\n")
                .map_err(|e| DataError::io(&part_path, e))?;
        }

        file.flush().map_err(|e| DataError::io(&part_path, e))?;
        drop(file);

        fs::rename(&part_path, &out_path).map_err(|e| DataError::io(&out_path, e))?;

        Ok(out_path)
    }
}

#[cfg(test)]
#[path = "source_test.rs"]
mod tests;
