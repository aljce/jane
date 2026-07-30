//! Model geometry.
//!
//! Every dimension lives here. No module body anywhere in the project may
//! contain a dimension literal — scaling from `jane-1m` to `jane-150m` must be a
//! config change and nothing else.
//!
//! # CONTRACT — implement the `todo!()` bodies below. Do not change signatures.
//!
//! The struct definition, the constants and the public API are fixed. Fill in
//! the implementations and add the tests described in each doc comment.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Byte-level BPE needs one token per possible byte before any merges, so no
/// vocabulary can be smaller than this.
pub const MIN_VOCAB_SIZE: usize = 256;

/// `d_ff` is rounded to a multiple of this for tensor-core friendliness.
pub const FFN_ALIGN: usize = 64;

/// SwiGLU uses three projections instead of two, so the standard `4 * d_model`
/// hidden size is scaled by 2/3 to keep the parameter count equivalent.
pub const FFN_RATIO: f64 = 8.0 / 3.0;

/// Model geometry. See [`Preset`] for the standard configurations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JaneConfig {
    /// Number of BPE tokens. Must be >= [`MIN_VOCAB_SIZE`].
    pub vocab_size: usize,
    /// Width of the residual stream, and therefore the token-embedding
    /// dimension. Must be divisible by `n_heads`.
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Feed-forward hidden size. `None` derives it via [`aligned_d_ff`], which
    /// is what all four presets do.
    #[serde(default)]
    pub d_ff: Option<usize>,
    /// Training context length; also sizes the RoPE cache.
    pub seq_len: usize,
    #[serde(default)]
    pub dropout: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    /// Share the embedding matrix with the output projection.
    #[serde(default = "default_tie_embeddings")]
    pub tie_embeddings: bool,
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_tie_embeddings() -> bool {
    true
}

/// The standard configurations from ROADMAP §3.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Preset {
    /// 1.28M params. CI and CPU-only tests.
    Jane1m,
    /// 13.77M params. The primary target.
    Jane14m,
    /// 60.06M params. Rung 3 / WikiText-103.
    Jane60m,
    /// 144.3M params. Rung 4, sized to fit 12 GB with AdamW state.
    Jane150m,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("d_model ({d_model}) must be divisible by n_heads ({n_heads})")]
    HeadsNotDivisible { d_model: usize, n_heads: usize },

    /// RoPE rotates coordinate pairs, so an odd head dimension has a leftover.
    #[error("head_dim ({head_dim}) must be even for RoPE (d_model {d_model} / n_heads {n_heads})")]
    OddHeadDim {
        head_dim: usize,
        d_model: usize,
        n_heads: usize,
    },

    #[error("vocab_size ({0}) must be at least {MIN_VOCAB_SIZE} for byte-level BPE")]
    VocabTooSmall(usize),

    #[error("{field} must be greater than zero")]
    Zero { field: &'static str },

    #[error("dropout ({0}) must be in [0.0, 1.0)")]
    DropoutOutOfRange(f64),

    #[error("unknown preset {0:?} (expected one of: 1m, 14m, 60m, 150m)")]
    UnknownPreset(String),

    #[error("failed to parse TOML config: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("failed to serialize TOML config: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[error("config io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// `round(FFN_RATIO * d_model / FFN_ALIGN) * FFN_ALIGN`, never zero.
///
/// # Tests required
/// - `aligned_d_ff(128) == 320`, `(384) == 1024`, `(640) == 1728`, `(896) == 2368`
/// - result is always a non-zero multiple of [`FFN_ALIGN`]
/// - monotonically non-decreasing in `d_model` over `1..=4096`
pub fn aligned_d_ff(_d_model: usize) -> usize {
    todo!("aligned_d_ff")
}

impl Preset {
    /// All presets, for exhaustive testing.
    pub fn all() -> [Preset; 4] {
        [
            Preset::Jane1m,
            Preset::Jane14m,
            Preset::Jane60m,
            Preset::Jane150m,
        ]
    }

    /// Canonical name, e.g. `"jane-14m"`. Must round-trip through [`Preset::parse`].
    pub fn name(&self) -> &'static str {
        match self {
            Preset::Jane1m => "jane-1m",
            Preset::Jane14m => "jane-14m",
            Preset::Jane60m => "jane-60m",
            Preset::Jane150m => "jane-150m",
        }
    }

    /// Accepts `"jane-14m"`, `"14m"` and `"JANE-14M"`.
    ///
    /// # Tests required
    /// - every `Preset::all()` name round-trips
    /// - bare form (`"14m"`) and mixed case both work
    /// - unknown input yields [`ConfigError::UnknownPreset`]
    pub fn parse(_s: &str) -> Result<Preset, ConfigError> {
        todo!("Preset::parse")
    }
}

impl JaneConfig {
    /// Build a preset. Values are fixed by ROADMAP §3.2:
    ///
    /// | preset | vocab | d_model | layers | heads | seq_len |
    /// |---|---|---|---|---|---|
    /// | `jane-1m`   | 4096  | 128 | 4  | 4  | 256  |
    /// | `jane-14m`  | 8192  | 384 | 6  | 6  | 512  |
    /// | `jane-60m`  | 16384 | 640 | 10 | 10 | 1024 |
    /// | `jane-150m` | 32768 | 896 | 12 | 14 | 1024 |
    ///
    /// All presets leave `d_ff: None` (derived), `dropout: 0.0` and
    /// `tie_embeddings: true`.
    pub fn preset(_preset: Preset) -> Self {
        todo!("JaneConfig::preset")
    }

    /// Explicit `d_ff` if set, otherwise [`aligned_d_ff`] of `d_model`.
    pub fn d_ff(&self) -> usize {
        todo!("JaneConfig::d_ff")
    }

    /// `d_model / n_heads`. Call [`JaneConfig::validate`] first — this may
    /// truncate on an invalid config.
    pub fn head_dim(&self) -> usize {
        todo!("JaneConfig::head_dim")
    }

    /// Total trainable parameters:
    ///
    /// ```text
    /// V·D + L·(4·D² + 3·D·F + 2·D) + D   (+ V·D again when not tied)
    /// ```
    ///
    /// where `V`=vocab_size, `D`=d_model, `L`=n_layers, `F`=d_ff(). The two `D`
    /// terms per layer are the RMSNorm gains; the trailing `D` is the final norm.
    ///
    /// # Tests required
    /// Assert against these exact values, and also recompute the formula
    /// independently in the test so a typo can't pass both:
    ///
    /// | preset | tied | untied |
    /// |---|---|---|
    /// | `jane-1m`   | 1_279_104   | 1_803_392   |
    /// | `jane-14m`  | 13_767_552  | 16_913_280  |
    /// | `jane-60m`  | 60_060_800  | 70_546_560  |
    /// | `jane-150m` | 144_299_904 | 173_660_032 |
    ///
    /// Also assert `untied - tied == vocab_size * d_model` for every preset.
    pub fn param_count(&self) -> usize {
        todo!("JaneConfig::param_count")
    }

    /// `vocab_size * d_model` — the embedding table, counted once.
    pub fn embedding_params(&self) -> usize {
        todo!("JaneConfig::embedding_params")
    }

    /// Embedding share of [`JaneConfig::param_count`], in `0.0..=1.0`.
    ///
    /// # Tests required
    /// Within 0.0005 of: `jane-1m` 0.4099, `jane-14m` 0.2285, `jane-60m` 0.1746,
    /// `jane-150m` 0.2035. Assert every preset except `jane-1m` is under 0.25
    /// (see ROADMAP §3 — `jane-1m` is a knowing exception because vocab cannot
    /// shrink below the 256 byte tokens).
    pub fn embedding_fraction(&self) -> f64 {
        todo!("JaneConfig::embedding_fraction")
    }

    /// # Tests required
    /// - all four presets validate
    /// - `d_model=384, n_heads=5` → [`ConfigError::HeadsNotDivisible`]
    /// - `d_model=6, n_heads=4` → `HeadsNotDivisible` (not `OddHeadDim`)
    /// - `d_model=12, n_heads=4` (head_dim 3) → [`ConfigError::OddHeadDim`]
    /// - `vocab_size=255` → [`ConfigError::VocabTooSmall`]
    /// - zero `d_model` / `n_layers` / `n_heads` / `seq_len` → [`ConfigError::Zero`]
    /// - `dropout` of 1.0 and -0.1 → [`ConfigError::DropoutOutOfRange`]; 0.0 and
    ///   0.9 are accepted
    /// - explicit `d_ff: Some(0)` → [`ConfigError::Zero`]
    pub fn validate(&self) -> Result<(), ConfigError> {
        todo!("JaneConfig::validate")
    }

    /// Parse and validate.
    ///
    /// # Tests required
    /// - TOML round-trips through [`JaneConfig::to_toml_string`] for all presets
    /// - omitting `dropout`/`rope_theta`/`norm_eps`/`tie_embeddings`/`d_ff`
    ///   yields the documented defaults
    /// - an unknown key is rejected (the struct is `deny_unknown_fields`)
    /// - a TOML string describing an invalid config is rejected by `validate`,
    ///   not silently accepted
    pub fn from_toml_str(_s: &str) -> Result<Self, ConfigError> {
        todo!("JaneConfig::from_toml_str")
    }

    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        todo!("JaneConfig::to_toml_string")
    }

    /// Read a config file. Errors carry the path — a bare `io::Error` here is
    /// useless for debugging.
    pub fn load(_path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        todo!("JaneConfig::load")
    }

    /// Write this config next to a checkpoint. A checkpoint whose
    /// hyperparameters are unknown is not a result.
    pub fn save(&self, _path: impl AsRef<Path>) -> Result<(), ConfigError> {
        todo!("JaneConfig::save")
    }
}
