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
pub fn aligned_d_ff(d_model: usize) -> usize {
    let raw_units = FFN_RATIO * d_model as f64 / FFN_ALIGN as f64;
    // `.max(1.0)` is what keeps this non-zero for tiny `d_model`: round() alone
    // would floor small ratios to 0 units, i.e. an FFN with no hidden layer.
    let units = raw_units.round().max(1.0) as usize;
    units * FFN_ALIGN
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
    pub fn parse(s: &str) -> Result<Preset, ConfigError> {
        let lower = s.to_lowercase();
        let bare = lower.strip_prefix("jane-").unwrap_or(&lower);
        match bare {
            "1m" => Ok(Preset::Jane1m),
            "14m" => Ok(Preset::Jane14m),
            "60m" => Ok(Preset::Jane60m),
            "150m" => Ok(Preset::Jane150m),
            _ => Err(ConfigError::UnknownPreset(s.to_string())),
        }
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
    pub fn preset(preset: Preset) -> Self {
        let (vocab_size, d_model, n_layers, n_heads, seq_len) = match preset {
            Preset::Jane1m => (4096, 128, 4, 4, 256),
            Preset::Jane14m => (8192, 384, 6, 6, 512),
            Preset::Jane60m => (16384, 640, 10, 10, 1024),
            Preset::Jane150m => (32768, 896, 12, 14, 1024),
        };
        JaneConfig {
            vocab_size,
            d_model,
            n_layers,
            n_heads,
            d_ff: None,
            seq_len,
            dropout: 0.0,
            rope_theta: default_rope_theta(),
            norm_eps: default_norm_eps(),
            tie_embeddings: true,
        }
    }

    /// Explicit `d_ff` if set, otherwise [`aligned_d_ff`] of `d_model`.
    pub fn d_ff(&self) -> usize {
        self.d_ff.unwrap_or_else(|| aligned_d_ff(self.d_model))
    }

    /// `d_model / n_heads`. Call [`JaneConfig::validate`] first — this may
    /// truncate on an invalid config.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
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
        let v = self.vocab_size;
        let d = self.d_model;
        let l = self.n_layers;
        let f = self.d_ff();
        let per_layer = 4 * d * d + 3 * d * f + 2 * d;
        let mut total = v * d + l * per_layer + d;
        if !self.tie_embeddings {
            total += v * d;
        }
        total
    }

    /// `vocab_size * d_model` — the embedding table, counted once.
    pub fn embedding_params(&self) -> usize {
        self.vocab_size * self.d_model
    }

    /// Embedding share of [`JaneConfig::param_count`], in `0.0..=1.0`.
    ///
    /// # Tests required
    /// Within 0.0005 of: `jane-1m` 0.4099, `jane-14m` 0.2285, `jane-60m` 0.1746,
    /// `jane-150m` 0.2035. Assert every preset except `jane-1m` is under 0.25
    /// (see ROADMAP §3 — `jane-1m` is a knowing exception because vocab cannot
    /// shrink below the 256 byte tokens).
    pub fn embedding_fraction(&self) -> f64 {
        self.embedding_params() as f64 / self.param_count() as f64
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
        if self.vocab_size < MIN_VOCAB_SIZE {
            return Err(ConfigError::VocabTooSmall(self.vocab_size));
        }
        if self.d_model == 0 {
            return Err(ConfigError::Zero { field: "d_model" });
        }
        if self.n_layers == 0 {
            return Err(ConfigError::Zero { field: "n_layers" });
        }
        if self.n_heads == 0 {
            return Err(ConfigError::Zero { field: "n_heads" });
        }
        if self.seq_len == 0 {
            return Err(ConfigError::Zero { field: "seq_len" });
        }
        if self.d_ff == Some(0) {
            return Err(ConfigError::Zero { field: "d_ff" });
        }
        // Divisibility before head_dim: a zero-checked, non-divisible n_heads
        // (e.g. d_model=6, n_heads=4) must report HeadsNotDivisible, not the
        // truncated-division OddHeadDim that would follow it.
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(ConfigError::HeadsNotDivisible {
                d_model: self.d_model,
                n_heads: self.n_heads,
            });
        }
        let head_dim = self.head_dim();
        if !head_dim.is_multiple_of(2) {
            return Err(ConfigError::OddHeadDim {
                head_dim,
                d_model: self.d_model,
                n_heads: self.n_heads,
            });
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(ConfigError::DropoutOutOfRange(self.dropout));
        }
        Ok(())
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
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        let config: JaneConfig = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_toml_string(&self) -> Result<String, ConfigError> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Read a config file. Errors carry the path — a bare `io::Error` here is
    /// useless for debugging.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml_str(&contents)
    }

    /// Write this config next to a checkpoint. A checkpoint whose
    /// hyperparameters are unknown is not a result.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        let contents = self.to_toml_string()?;
        std::fs::write(path, contents).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------- aligned_d_ff

    #[test]
    fn aligned_d_ff_matches_documented_values() {
        assert_eq!(aligned_d_ff(128), 320);
        assert_eq!(aligned_d_ff(384), 1024);
        assert_eq!(aligned_d_ff(640), 1728);
        assert_eq!(aligned_d_ff(896), 2368);
    }

    #[test]
    fn aligned_d_ff_is_never_zero_and_always_aligned() {
        for d_model in 1..=4096usize {
            let f = aligned_d_ff(d_model);
            assert_ne!(f, 0, "d_model={d_model}");
            assert_eq!(f % FFN_ALIGN, 0, "d_model={d_model} f={f}");
        }
    }

    #[test]
    fn aligned_d_ff_is_monotonic_non_decreasing() {
        let mut prev = aligned_d_ff(1);
        for d_model in 2..=4096usize {
            let cur = aligned_d_ff(d_model);
            assert!(cur >= prev, "d_model={d_model} cur={cur} prev={prev}");
            prev = cur;
        }
    }

    // -------------------------------------------------------------- Preset

    #[test]
    fn preset_names_round_trip() {
        for preset in Preset::all() {
            assert_eq!(Preset::parse(preset.name()).unwrap(), preset);
        }
    }

    #[test]
    fn preset_parse_accepts_bare_and_mixed_case() {
        assert_eq!(Preset::parse("14m").unwrap(), Preset::Jane14m);
        assert_eq!(Preset::parse("JANE-14M").unwrap(), Preset::Jane14m);
        assert_eq!(Preset::parse("Jane-1M").unwrap(), Preset::Jane1m);
        assert_eq!(Preset::parse("60M").unwrap(), Preset::Jane60m);
        assert_eq!(Preset::parse("jane-150m").unwrap(), Preset::Jane150m);
    }

    #[test]
    fn preset_parse_rejects_unknown() {
        let err = Preset::parse("jane-7b").unwrap_err();
        assert!(matches!(err, ConfigError::UnknownPreset(s) if s == "jane-7b"));
    }

    // ------------------------------------------------------ JaneConfig::preset

    #[test]
    fn preset_table_matches_roadmap() {
        let cases = [
            (Preset::Jane1m, 4096, 128, 4, 4, 256),
            (Preset::Jane14m, 8192, 384, 6, 6, 512),
            (Preset::Jane60m, 16384, 640, 10, 10, 1024),
            (Preset::Jane150m, 32768, 896, 12, 14, 1024),
        ];
        for (preset, vocab_size, d_model, n_layers, n_heads, seq_len) in cases {
            let cfg = JaneConfig::preset(preset);
            assert_eq!(cfg.vocab_size, vocab_size);
            assert_eq!(cfg.d_model, d_model);
            assert_eq!(cfg.n_layers, n_layers);
            assert_eq!(cfg.n_heads, n_heads);
            assert_eq!(cfg.seq_len, seq_len);
            assert_eq!(cfg.d_ff, None);
            assert_eq!(cfg.dropout, 0.0);
            assert!(cfg.tie_embeddings);
            assert_eq!(cfg.rope_theta, 10_000.0);
            assert_eq!(cfg.norm_eps, 1e-5);
            cfg.validate().expect("preset must validate");
        }
    }

    // -------------------------------------------------------------- d_ff/head_dim

    #[test]
    fn d_ff_derives_when_unset_and_respects_explicit() {
        let mut cfg = JaneConfig::preset(Preset::Jane14m);
        assert_eq!(cfg.d_ff(), aligned_d_ff(384));
        cfg.d_ff = Some(777);
        assert_eq!(cfg.d_ff(), 777);
    }

    #[test]
    fn head_dim_is_d_model_over_n_heads() {
        assert_eq!(JaneConfig::preset(Preset::Jane1m).head_dim(), 32);
        assert_eq!(JaneConfig::preset(Preset::Jane14m).head_dim(), 64);
        assert_eq!(JaneConfig::preset(Preset::Jane60m).head_dim(), 64);
        assert_eq!(JaneConfig::preset(Preset::Jane150m).head_dim(), 64);
    }

    // ------------------------------------------------------------ param_count

    /// Independent re-derivation of the formula in the doc comment, written
    /// separately from the implementation so a shared typo can't satisfy both
    /// this and `param_count`.
    fn independent_param_count(cfg: &JaneConfig) -> usize {
        let vocab = cfg.vocab_size;
        let dim = cfg.d_model;
        let layers = cfg.n_layers;
        let ffn = cfg.d_ff();

        let embedding = vocab * dim;
        let attn_per_layer = 4 * dim * dim; // Wq, Wk, Wv, Wo
        let ffn_per_layer = 3 * dim * ffn; // gate, up, down (SwiGLU)
        let norm_per_layer = 2 * dim; // two RMSNorm gains per block
        let per_layer_total = attn_per_layer + ffn_per_layer + norm_per_layer;
        let final_norm = dim;

        let mut total = embedding + layers * per_layer_total + final_norm;
        if !cfg.tie_embeddings {
            total += embedding;
        }
        total
    }

    #[test]
    fn param_count_matches_documented_values_tied_and_untied() {
        let cases = [
            (Preset::Jane1m, 1_279_104usize, 1_803_392usize),
            (Preset::Jane14m, 13_767_552, 16_913_280),
            (Preset::Jane60m, 60_060_800, 70_546_560),
            (Preset::Jane150m, 144_299_904, 173_660_032),
        ];
        for (preset, tied, untied) in cases {
            let mut cfg = JaneConfig::preset(preset);

            cfg.tie_embeddings = true;
            assert_eq!(cfg.param_count(), tied, "{preset:?} tied");
            assert_eq!(
                independent_param_count(&cfg),
                tied,
                "{preset:?} tied (independent)"
            );

            cfg.tie_embeddings = false;
            assert_eq!(cfg.param_count(), untied, "{preset:?} untied");
            assert_eq!(
                independent_param_count(&cfg),
                untied,
                "{preset:?} untied (independent)"
            );

            assert_eq!(
                untied - tied,
                cfg.vocab_size * cfg.d_model,
                "{preset:?} untied-tied gap must equal one embedding table"
            );
        }
    }

    #[test]
    fn embedding_params_is_vocab_times_d_model() {
        for preset in Preset::all() {
            let cfg = JaneConfig::preset(preset);
            assert_eq!(cfg.embedding_params(), cfg.vocab_size * cfg.d_model);
        }
    }

    #[test]
    fn embedding_fraction_matches_documented_values() {
        let cases = [
            (Preset::Jane1m, 0.4099),
            (Preset::Jane14m, 0.2285),
            (Preset::Jane60m, 0.1746),
            (Preset::Jane150m, 0.2035),
        ];
        for (preset, expected) in cases {
            let cfg = JaneConfig::preset(preset);
            let actual = cfg.embedding_fraction();
            assert!(
                (actual - expected).abs() < 0.0005,
                "{preset:?}: expected {expected}, got {actual}"
            );
            if preset != Preset::Jane1m {
                assert!(
                    actual < 0.25,
                    "{preset:?}: embedding fraction {actual} must stay under 25%"
                );
            }
        }
    }

    // --------------------------------------------------------------- validate

    #[test]
    fn all_presets_validate() {
        for preset in Preset::all() {
            JaneConfig::preset(preset)
                .validate()
                .unwrap_or_else(|e| panic!("{preset:?} should validate: {e}"));
        }
    }

    #[test]
    fn heads_not_divisible_is_reported() {
        let mut cfg = JaneConfig::preset(Preset::Jane14m);
        cfg.d_model = 384;
        cfg.n_heads = 5;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::HeadsNotDivisible {
                d_model: 384,
                n_heads: 5
            }
        ));
    }

    #[test]
    fn heads_not_divisible_takes_priority_over_odd_head_dim() {
        // d_model=6, n_heads=4: 6 % 4 != 0, so this must be reported as
        // HeadsNotDivisible even though a truncated 6/4=1 head_dim would also
        // be odd — the divisibility check must run first.
        let mut cfg = JaneConfig::preset(Preset::Jane1m);
        cfg.d_model = 6;
        cfg.n_heads = 4;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::HeadsNotDivisible {
                d_model: 6,
                n_heads: 4
            }
        ));
    }

    #[test]
    fn odd_head_dim_is_reported() {
        let mut cfg = JaneConfig::preset(Preset::Jane1m);
        cfg.d_model = 12;
        cfg.n_heads = 4;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::OddHeadDim {
                head_dim: 3,
                d_model: 12,
                n_heads: 4
            }
        ));
    }

    #[test]
    fn vocab_too_small_is_reported() {
        let mut cfg = JaneConfig::preset(Preset::Jane1m);
        cfg.vocab_size = 255;
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::VocabTooSmall(255)));
    }

    #[test]
    fn zero_fields_are_reported() {
        for field in ["d_model", "n_layers", "n_heads", "seq_len"] {
            let mut cfg = JaneConfig::preset(Preset::Jane1m);
            match field {
                "d_model" => cfg.d_model = 0,
                "n_layers" => cfg.n_layers = 0,
                "n_heads" => cfg.n_heads = 0,
                "seq_len" => cfg.seq_len = 0,
                _ => unreachable!(),
            }
            let err = cfg.validate().unwrap_err();
            assert!(
                matches!(err, ConfigError::Zero { field: f } if f == field),
                "expected Zero{{field: {field}}}, got {err:?}"
            );
        }
    }

    #[test]
    fn explicit_zero_d_ff_is_reported() {
        let mut cfg = JaneConfig::preset(Preset::Jane1m);
        cfg.d_ff = Some(0);
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ConfigError::Zero { field: "d_ff" }));
    }

    #[test]
    fn dropout_range_is_enforced() {
        let mut cfg = JaneConfig::preset(Preset::Jane1m);

        cfg.dropout = 1.0;
        assert!(matches!(
            cfg.validate().unwrap_err(),
            ConfigError::DropoutOutOfRange(d) if d == 1.0
        ));

        cfg.dropout = -0.1;
        assert!(matches!(
            cfg.validate().unwrap_err(),
            ConfigError::DropoutOutOfRange(d) if d == -0.1
        ));

        cfg.dropout = 0.0;
        cfg.validate().expect("0.0 dropout is accepted");

        cfg.dropout = 0.9;
        cfg.validate().expect("0.9 dropout is accepted");
    }

    // ------------------------------------------------------------------- TOML

    #[test]
    fn toml_round_trips_for_all_presets() {
        for preset in Preset::all() {
            let cfg = JaneConfig::preset(preset);
            let toml_str = cfg.to_toml_string().unwrap();
            let parsed = JaneConfig::from_toml_str(&toml_str).unwrap();
            assert_eq!(parsed, cfg, "{preset:?} round-trip mismatch");
        }
    }

    #[test]
    fn omitted_optional_fields_get_documented_defaults() {
        let minimal = r#"
            vocab_size = 8192
            d_model = 384
            n_layers = 6
            n_heads = 6
            seq_len = 512
        "#;
        let cfg = JaneConfig::from_toml_str(minimal).unwrap();
        assert_eq!(cfg.d_ff, None);
        assert_eq!(cfg.dropout, 0.0);
        assert_eq!(cfg.rope_theta, 10_000.0);
        assert_eq!(cfg.norm_eps, 1e-5);
        assert!(cfg.tie_embeddings);
    }

    #[test]
    fn unknown_key_is_rejected() {
        let bad = r#"
            vocab_size = 8192
            d_model = 384
            n_layers = 6
            n_heads = 6
            seq_len = 512
            not_a_real_field = 42
        "#;
        let err = JaneConfig::from_toml_str(bad).unwrap_err();
        assert!(matches!(err, ConfigError::TomlDe(_)));
    }

    #[test]
    fn invalid_config_is_rejected_by_from_toml_str_not_silently_accepted() {
        // d_model=384 % n_heads=5 != 0: parses fine as TOML, but validate()
        // must still reject it.
        let bad = r#"
            vocab_size = 8192
            d_model = 384
            n_layers = 6
            n_heads = 5
            seq_len = 512
        "#;
        let err = JaneConfig::from_toml_str(bad).unwrap_err();
        assert!(matches!(err, ConfigError::HeadsNotDivisible { .. }));
    }

    #[test]
    fn load_save_round_trip_through_a_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = JaneConfig::preset(Preset::Jane60m);
        cfg.save(&path).unwrap();
        let loaded = JaneConfig::load(&path).unwrap();
        assert_eq!(loaded, cfg);
    }

    #[test]
    fn load_missing_file_reports_path_in_io_error() {
        let err = JaneConfig::load("/nonexistent/does-not-exist/config.toml").unwrap_err();
        match err {
            ConfigError::Io { path, .. } => {
                assert!(path.contains("does-not-exist"));
            }
            other => panic!("expected ConfigError::Io, got {other:?}"),
        }
    }

    // ------------------------------------------------------- configs/*.toml
    //
    // The whole point of shipping preset TOMLs on disk is that they cannot
    // silently drift from JaneConfig::preset(..). This is the test that
    // would catch that drift.

    fn workspace_configs_dir() -> std::path::PathBuf {
        // crates/jane-model -> workspace root is two levels up.
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("configs")
    }

    #[test]
    fn on_disk_presets_match_jane_config_preset() {
        let cases = [
            ("jane-1m.toml", Preset::Jane1m),
            ("jane-14m.toml", Preset::Jane14m),
            ("jane-60m.toml", Preset::Jane60m),
            ("jane-150m.toml", Preset::Jane150m),
        ];
        let dir = workspace_configs_dir();
        for (filename, preset) in cases {
            let path = dir.join(filename);
            let loaded = JaneConfig::load(&path)
                .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
            assert_eq!(
                loaded,
                JaneConfig::preset(preset),
                "{filename} does not match JaneConfig::preset({preset:?})"
            );
        }
    }
}
