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
        TrainConfig {
            lr: 3e-4,
            min_lr: 3e-5,
            warmup_steps: 500,
            max_steps: 18_000,
            tokens_per_step: 16_384,
            micro_batch_size: 32,
            seq_len: 512,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            grad_clip: 1.0,
            seed: 1337,
            eval_every: 500,
            checkpoint_every: 1000,
            num_workers: 4,
        }
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
        let micro_tokens = self.micro_batch_size * self.seq_len;
        (self.tokens_per_step / micro_tokens.max(1)).max(1)
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
    /// - always within `[0, lr]` for every step in `0..=max_steps * 2`
    ///   (`min_lr` is the floor of cosine decay, not a global floor — warmup
    ///   starts near zero by design)
    /// - `warmup_steps == 0` does not divide by zero: step 0 gives `lr`
    pub fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // warmup_steps == 0 never reaches this branch, so no div-by-zero.
            return self.lr * (step + 1) as f64 / self.warmup_steps as f64;
        }
        if step >= self.max_steps {
            return self.min_lr;
        }
        // decay_span == 0 only when warmup_steps == max_steps, which validate()
        // rejects; guard it anyway so lr_at never panics on an unvalidated config.
        let decay_span = self.max_steps.saturating_sub(self.warmup_steps).max(1) as f64;
        let p = ((step - self.warmup_steps) as f64 / decay_span).min(1.0);
        self.min_lr + 0.5 * (self.lr - self.min_lr) * (1.0 + (std::f64::consts::PI * p).cos())
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
        if self.lr <= 0.0 {
            return Err(TrainConfigError::Zero { field: "lr" });
        }
        if self.max_steps == 0 {
            return Err(TrainConfigError::Zero { field: "max_steps" });
        }
        if self.micro_batch_size == 0 {
            return Err(TrainConfigError::Zero {
                field: "micro_batch_size",
            });
        }
        if self.seq_len == 0 {
            return Err(TrainConfigError::Zero { field: "seq_len" });
        }
        if self.tokens_per_step == 0 {
            return Err(TrainConfigError::Zero {
                field: "tokens_per_step",
            });
        }
        if self.num_workers == 0 {
            return Err(TrainConfigError::Zero {
                field: "num_workers",
            });
        }

        let micro_tokens = self.micro_batch_size * self.seq_len;
        if !self.tokens_per_step.is_multiple_of(micro_tokens) {
            return Err(TrainConfigError::IndivisibleBatch {
                tokens_per_step: self.tokens_per_step,
                micro_tokens,
            });
        }

        if self.warmup_steps >= self.max_steps {
            return Err(TrainConfigError::WarmupTooLong {
                warmup: self.warmup_steps,
                max: self.max_steps,
            });
        }

        if self.min_lr < 0.0 || self.min_lr > self.lr {
            return Err(TrainConfigError::MinLrOutOfRange {
                min_lr: self.min_lr,
                lr: self.lr,
            });
        }

        for (field, value) in [
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("weight_decay", self.weight_decay),
        ] {
            if !(0.0..1.0).contains(&value) {
                return Err(TrainConfigError::NotProbability { field, value });
            }
        }

        Ok(())
    }

    /// Parse and validate.
    ///
    /// # Tests required
    /// - round-trips through [`TrainConfig::to_toml_string`]
    /// - unknown keys are rejected
    pub fn from_toml_str(s: &str) -> Result<Self, TrainConfigError> {
        let config: TrainConfig = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_toml_string(&self) -> Result<String, TrainConfigError> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, TrainConfigError> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path).map_err(|source| TrainConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml_str(&s)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), TrainConfigError> {
        let path = path.as_ref();
        let s = self.to_toml_string()?;
        std::fs::write(path, s).map_err(|source| TrainConfigError::Io {
            path: path.display().to_string(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    // ---- Default -----------------------------------------------------

    #[test]
    fn default_matches_roadmap_recipe() {
        let cfg = TrainConfig::default();
        assert_eq!(cfg.lr, 3e-4);
        assert_eq!(cfg.min_lr, 3e-5);
        assert_eq!(cfg.warmup_steps, 500);
        assert_eq!(cfg.max_steps, 18_000);
        assert_eq!(cfg.tokens_per_step, 16_384);
        assert_eq!(cfg.micro_batch_size, 32);
        assert_eq!(cfg.seq_len, 512);
        assert_eq!(cfg.weight_decay, 0.1);
        assert_eq!(cfg.beta1, 0.9);
        assert_eq!(cfg.beta2, 0.95);
        assert_eq!(cfg.grad_clip, 1.0);
    }

    #[test]
    fn default_validates() {
        TrainConfig::default().validate().unwrap();
    }

    // ---- grad_accum_steps ---------------------------------------------

    #[test]
    fn grad_accum_steps_default_is_one() {
        let cfg = TrainConfig::default();
        assert_eq!(cfg.grad_accum_steps(), 1);
        assert_eq!(16_384 / (32 * 512), 1);
    }

    #[test]
    fn grad_accum_steps_invariant_across_micro_batch_sizes() {
        // tokens_per_step is fixed; halving micro_batch_size should double
        // grad_accum_steps so the effective batch (in tokens) never moves.
        let mut cfg = TrainConfig {
            tokens_per_step: 16_384,
            seq_len: 512,
            ..TrainConfig::default()
        };
        for &micro in &[32usize, 16, 8, 4, 2, 1] {
            cfg.micro_batch_size = micro;
            let accum = cfg.grad_accum_steps();
            assert_eq!(
                accum * cfg.micro_batch_size * cfg.seq_len,
                cfg.tokens_per_step,
                "micro_batch_size={micro} broke the token-budget invariant"
            );
        }
        // halving 32 -> 16 must double accumulation exactly.
        cfg.micro_batch_size = 32;
        let accum_32 = cfg.grad_accum_steps();
        cfg.micro_batch_size = 16;
        let accum_16 = cfg.grad_accum_steps();
        assert_eq!(accum_16, accum_32 * 2);
    }

    #[test]
    fn grad_accum_steps_micro_batch_larger_than_budget_still_at_least_one() {
        let cfg = TrainConfig {
            tokens_per_step: 16_384,
            micro_batch_size: 1_000_000,
            seq_len: 512,
            ..TrainConfig::default()
        };
        assert_eq!(cfg.grad_accum_steps(), 1);
    }

    // ---- lr_at ----------------------------------------------------------

    #[test]
    fn lr_at_warmup_end_hits_peak() {
        let cfg = TrainConfig::default();
        assert!(approx(cfg.lr_at(cfg.warmup_steps - 1), cfg.lr, 1e-12));
    }

    #[test]
    fn lr_at_no_discontinuity_at_warmup_seam() {
        let cfg = TrainConfig::default();
        let before = cfg.lr_at(cfg.warmup_steps - 1);
        let after = cfg.lr_at(cfg.warmup_steps);
        assert!(approx(before, cfg.lr, 1e-9));
        assert!(approx(after, cfg.lr, 1e-9));
        assert!(approx(before, after, 1e-9));
    }

    #[test]
    fn lr_at_zero_is_lr_over_warmup_and_positive() {
        let cfg = TrainConfig::default();
        let expected = cfg.lr / cfg.warmup_steps as f64;
        assert!(approx(cfg.lr_at(0), expected, 1e-15));
        assert!(cfg.lr_at(0) > 0.0);
    }

    #[test]
    fn lr_at_max_steps_hits_floor() {
        let cfg = TrainConfig::default();
        assert!(approx(cfg.lr_at(cfg.max_steps), cfg.min_lr, 1e-9));
    }

    #[test]
    fn lr_at_beyond_max_steps_clamped_never_rises_never_negative() {
        let cfg = TrainConfig::default();
        let floor = cfg.lr_at(cfg.max_steps);
        for step in [
            cfg.max_steps,
            cfg.max_steps + 1,
            cfg.max_steps + 100,
            cfg.max_steps * 2,
        ] {
            let v = cfg.lr_at(step);
            assert!(approx(v, cfg.min_lr, 1e-12), "step {step} -> {v}");
            assert!(v >= 0.0);
            assert!(v <= floor + 1e-15, "lr rose past the clamp at step {step}");
        }
    }

    #[test]
    fn lr_at_monotonically_increasing_during_warmup() {
        let cfg = TrainConfig::default();
        let mut prev = cfg.lr_at(0);
        for step in 1..cfg.warmup_steps {
            let v = cfg.lr_at(step);
            assert!(v > prev, "not increasing at step {step}: {prev} -> {v}");
            prev = v;
        }
    }

    #[test]
    fn lr_at_monotonically_non_increasing_after_warmup() {
        let cfg = TrainConfig::default();
        let mut prev = cfg.lr_at(cfg.warmup_steps);
        for step in (cfg.warmup_steps + 1)..=cfg.max_steps {
            let v = cfg.lr_at(step);
            assert!(v <= prev + 1e-12, "rose at step {step}: {prev} -> {v}");
            prev = v;
        }
    }

    #[test]
    fn lr_at_never_exceeds_lr_and_never_negative_over_full_range() {
        // The universal bound the formula actually guarantees. `min_lr` is only
        // a floor for the cosine (post-warmup) segment — see
        // `lr_at_within_min_lr_and_lr_when_warmup_is_shallow` for why it is
        // *not* a floor during the linear-warmup segment in general.
        let cfg = TrainConfig::default();
        for step in 0..=(cfg.max_steps * 2) {
            let v = cfg.lr_at(step);
            assert!(v >= 0.0, "negative lr at step {step}: {v}");
            assert!(v <= cfg.lr + 1e-9, "lr exceeded peak at step {step}: {v}");
        }
    }

    #[test]
    fn lr_at_within_min_lr_and_lr_post_warmup() {
        let cfg = TrainConfig::default();
        for step in cfg.warmup_steps..=(cfg.max_steps * 2) {
            let v = cfg.lr_at(step);
            assert!(
                v >= cfg.min_lr - 1e-12 && v <= cfg.lr + 1e-12,
                "step {step} -> {v} outside [{}, {}]",
                cfg.min_lr,
                cfg.lr
            );
        }
    }

    #[test]
    fn lr_at_within_min_lr_and_lr_when_warmup_is_shallow() {
        // The doc comment's warmup formula `lr * (step+1)/warmup_steps` starts
        // near zero by design (standard warmup practice), so it only stays
        // above `min_lr` for the *whole* trajectory when `min_lr <=
        // lr/warmup_steps`. That does not hold for the ROADMAP default
        // (lr=3e-4, min_lr=3e-5, warmup=500: lr/warmup_steps = 6e-7 < min_lr),
        // so this property is demonstrated on a config where it is
        // mathematically possible, not on the default.
        let cfg = TrainConfig {
            lr: 1.0,
            min_lr: 1e-3,
            warmup_steps: 10,
            max_steps: 100,
            ..TrainConfig::default()
        };
        cfg.validate().unwrap();
        for step in 0..=(cfg.max_steps * 2) {
            let v = cfg.lr_at(step);
            assert!(
                v >= cfg.min_lr - 1e-12 && v <= cfg.lr + 1e-12,
                "step {step} -> {v} outside [{}, {}]",
                cfg.min_lr,
                cfg.lr
            );
        }
    }

    #[test]
    fn lr_at_warmup_steps_zero_no_div_by_zero() {
        let cfg = TrainConfig {
            warmup_steps: 0,
            ..TrainConfig::default()
        };
        assert!(approx(cfg.lr_at(0), cfg.lr, 1e-12));
        assert!(cfg.lr_at(0).is_finite());
    }

    // ---- validate ---------------------------------------------------------

    #[test]
    fn validate_indivisible_batch() {
        let cfg = TrainConfig {
            tokens_per_step: 16_385, // not a multiple of 32*512
            ..TrainConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(TrainConfigError::IndivisibleBatch { .. })
        ));
    }

    #[test]
    fn validate_warmup_too_long() {
        let cfg = TrainConfig {
            warmup_steps: 18_000,
            max_steps: 18_000,
            ..TrainConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(TrainConfigError::WarmupTooLong { .. })
        ));

        let cfg2 = TrainConfig {
            warmup_steps: 19_000,
            max_steps: 18_000,
            ..TrainConfig::default()
        };
        assert!(matches!(
            cfg2.validate(),
            Err(TrainConfigError::WarmupTooLong { .. })
        ));
    }

    #[test]
    fn validate_min_lr_out_of_range() {
        let too_big = TrainConfig {
            min_lr: 1.0,
            lr: 3e-4,
            ..TrainConfig::default()
        };
        assert!(matches!(
            too_big.validate(),
            Err(TrainConfigError::MinLrOutOfRange { .. })
        ));

        let negative = TrainConfig {
            min_lr: -1e-6,
            ..TrainConfig::default()
        };
        assert!(matches!(
            negative.validate(),
            Err(TrainConfigError::MinLrOutOfRange { .. })
        ));
    }

    #[test]
    fn validate_zero_fields() {
        let base = TrainConfig::default();

        let cases: Vec<(&str, TrainConfig)> = vec![
            (
                "lr",
                TrainConfig {
                    lr: 0.0,
                    ..base.clone()
                },
            ),
            (
                "max_steps",
                TrainConfig {
                    max_steps: 0,
                    ..base.clone()
                },
            ),
            (
                "micro_batch_size",
                TrainConfig {
                    micro_batch_size: 0,
                    ..base.clone()
                },
            ),
            (
                "seq_len",
                TrainConfig {
                    seq_len: 0,
                    ..base.clone()
                },
            ),
            (
                "tokens_per_step",
                TrainConfig {
                    tokens_per_step: 0,
                    ..base.clone()
                },
            ),
            (
                "num_workers",
                TrainConfig {
                    num_workers: 0,
                    ..base.clone()
                },
            ),
        ];

        for (field, cfg) in cases {
            match cfg.validate() {
                Err(TrainConfigError::Zero { field: got }) => {
                    assert_eq!(got, field, "wrong field name reported")
                }
                other => panic!("expected Zero{{{field}}}, got {other:?}"),
            }
        }
    }

    #[test]
    fn validate_not_probability() {
        for field in ["beta1", "beta2", "weight_decay"] {
            let mut cfg = TrainConfig::default();
            match field {
                "beta1" => cfg.beta1 = 1.0,
                "beta2" => cfg.beta2 = -0.1,
                "weight_decay" => cfg.weight_decay = 1.5,
                _ => unreachable!(),
            }
            match cfg.validate() {
                Err(TrainConfigError::NotProbability { field: got, .. }) => {
                    assert_eq!(got, field)
                }
                other => panic!("expected NotProbability{{{field}}}, got {other:?}"),
            }
        }
    }

    // ---- TOML round-trip ----------------------------------------------

    #[test]
    fn toml_round_trip() {
        let cfg = TrainConfig::default();
        let s = cfg.to_toml_string().unwrap();
        let parsed = TrainConfig::from_toml_str(&s).unwrap();
        assert_eq!(cfg, parsed);
    }

    #[test]
    fn toml_unknown_key_rejected() {
        let cfg = TrainConfig::default();
        let mut s = cfg.to_toml_string().unwrap();
        s.push_str("\nnot_a_real_field = 1\n");
        assert!(matches!(
            TrainConfig::from_toml_str(&s),
            Err(TrainConfigError::TomlDe(_))
        ));
    }

    #[test]
    fn toml_from_str_rejects_invalid_config() {
        // Parses fine as TOML but fails validate() — from_toml_str must
        // propagate that, not just the parse.
        let cfg = TrainConfig {
            warmup_steps: 20_000,
            max_steps: 18_000,
            ..TrainConfig::default()
        };
        let s = toml::to_string_pretty(&cfg).unwrap();
        assert!(matches!(
            TrainConfig::from_toml_str(&s),
            Err(TrainConfigError::WarmupTooLong { .. })
        ));
    }

    #[test]
    fn load_save_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let cfg = TrainConfig::default();
        cfg.save(&path).unwrap();
        let loaded = TrainConfig::load(&path).unwrap();
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn load_missing_file_is_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does_not_exist.toml");
        assert!(matches!(
            TrainConfig::load(&path),
            Err(TrainConfigError::Io { .. })
        ));
    }
}
