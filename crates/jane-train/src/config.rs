//! Training hyperparameters.
//!
//! # CONTRACT — implement the `todo!()` bodies. Do not change signatures.
//!
//! The design point worth understanding before you start: **effective batch size
//! is configured in tokens, not rows.** `tokens_per_step` is fixed and
//! gradient accumulation is *derived* from whatever micro-batch fits in VRAM. A
//! bigger model that forces a smaller micro-batch then trains at the same
//! effective batch, so loss curves stay comparable across presets.

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    /// Peak learning rate, reached at the end of warmup.
    pub lr: f64,
    /// Floor the cosine schedule decays to.
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,

    /// Effective tokens per optimizer step. The invariant to hold across presets.
    pub tokens_per_step: usize,
    /// Rows per forward pass — whatever fits in VRAM.
    pub micro_batch_size: usize,
    /// Must match the model's `seq_len`.
    pub seq_len: usize,

    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub grad_clip: f64,

    pub seed: u64,
    pub eval_every: usize,
    pub checkpoint_every: usize,
    pub num_workers: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum TrainConfigError {
    #[error("{field} must be greater than zero")]
    Zero { field: &'static str },

    #[error(
        "tokens_per_step ({tokens_per_step}) must be a multiple of \
         micro_batch_size * seq_len ({micro_tokens}), otherwise the effective \
         batch size is not what the config claims"
    )]
    IndivisibleBatch {
        tokens_per_step: usize,
        micro_tokens: usize,
    },

    #[error("warmup_steps ({warmup}) must be less than max_steps ({max})")]
    WarmupTooLong { warmup: usize, max: usize },

    #[error("min_lr ({min_lr}) must be >= 0 and <= lr ({lr})")]
    MinLrOutOfRange { min_lr: f64, lr: f64 },

    #[error("{field} ({value}) must be in [0.0, 1.0)")]
    NotProbability { field: &'static str, value: f64 },

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

impl Default for TrainConfig {
    /// The ROADMAP §4 recipe for `jane-14m`: peak lr 3e-4 decaying to 3e-5,
    /// 500 warmup steps, 18000 total, 16384 tokens/step (32 x 512), AdamW
    /// betas (0.9, 0.95), weight decay 0.1, grad clip 1.0.
    fn default() -> Self {
        todo!("TrainConfig::default")
    }
}

impl TrainConfig {
    /// `tokens_per_step / (micro_batch_size * seq_len)`, at least 1.
    ///
    /// # Tests required
    /// - default config: `16384 / (32 * 512) == 1`
    /// - halving `micro_batch_size` doubles it, leaving `tokens_per_step` intact
    ///   (the whole point — assert `grad_accum_steps * micro * seq ==
    ///   tokens_per_step` across several micro-batch sizes)
    /// - a micro-batch larger than `tokens_per_step` still yields >= 1
    pub fn grad_accum_steps(&self) -> usize {
        todo!("TrainConfig::grad_accum_steps")
    }

    /// Learning rate at `step`: linear warmup over `warmup_steps`, then cosine
    /// decay from `lr` to `min_lr` across the remaining steps.
    ///
    /// ```text
    /// step < warmup_steps:  lr * (step + 1) / warmup_steps
    /// otherwise:            min_lr + 0.5 * (lr - min_lr) * (1 + cos(pi * p))
    ///     where p = (step - warmup_steps) / (max_steps - warmup_steps), clamped to 1.0
    /// ```
    ///
    /// # Tests required
    /// This is pure arithmetic and fully testable — cover it properly:
    /// - `lr_at(warmup_steps - 1)` is approximately `lr` (warmup ends at peak)
    /// - `lr_at(warmup_steps)` is approximately `lr` (no discontinuity at the seam)
    /// - `lr_at(0)` is `lr / warmup_steps`, and is > 0 — a zero first step
    ///   wastes it
    /// - `lr_at(max_steps)` is approximately `min_lr`
    /// - beyond `max_steps` it stays clamped at `min_lr`, never negative and
    ///   never rising
    /// - monotonically non-increasing across `warmup_steps..=max_steps`
    /// - monotonically increasing across `0..warmup_steps`
    /// - always within `[min_lr, lr]` for every step in `0..=max_steps * 2`
    /// - `warmup_steps == 0` does not divide by zero: step 0 gives `lr`
    pub fn lr_at(&self, _step: usize) -> f64 {
        todo!("TrainConfig::lr_at")
    }

    /// # Tests required
    /// - the default config validates
    /// - `tokens_per_step` not divisible by `micro * seq` →
    ///   [`TrainConfigError::IndivisibleBatch`]
    /// - `warmup_steps >= max_steps` → [`TrainConfigError::WarmupTooLong`]
    /// - `min_lr > lr`, and negative `min_lr` → [`TrainConfigError::MinLrOutOfRange`]
    /// - zero `lr` / `max_steps` / `micro_batch_size` / `seq_len` /
    ///   `tokens_per_step` → [`TrainConfigError::Zero`]
    /// - `beta1`/`beta2`/`weight_decay` outside `[0, 1)` →
    ///   [`TrainConfigError::NotProbability`]
    /// - `num_workers == 0` → `Zero`
    pub fn validate(&self) -> Result<(), TrainConfigError> {
        todo!("TrainConfig::validate")
    }

    /// Parse and validate.
    ///
    /// # Tests required
    /// - round-trips through [`TrainConfig::to_toml_string`]
    /// - unknown keys are rejected
    pub fn from_toml_str(_s: &str) -> Result<Self, TrainConfigError> {
        todo!("TrainConfig::from_toml_str")
    }

    pub fn to_toml_string(&self) -> Result<String, TrainConfigError> {
        todo!("TrainConfig::to_toml_string")
    }

    pub fn load(_path: impl AsRef<Path>) -> Result<Self, TrainConfigError> {
        todo!("TrainConfig::load")
    }

    pub fn save(&self, _path: impl AsRef<Path>) -> Result<(), TrainConfigError> {
        todo!("TrainConfig::save")
    }
}
