//! Model configuration for `jane`.
//!
//! Phase 0/1 scope: configuration only. The transformer modules (RoPE, RMSNorm,
//! causal attention, SwiGLU, blocks) arrive in Phase 2 alongside the Burn
//! dependency.

pub mod config;

pub use config::{ConfigError, JaneConfig, Preset};
