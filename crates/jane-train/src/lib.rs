//! Training configuration for `jane`.
//!
//! Phase 0/1 scope: configuration only. The learner loop, optimizer wiring and
//! checkpointing arrive in Phase 3.

pub mod config;

pub use config::{TrainConfig, TrainConfigError};
