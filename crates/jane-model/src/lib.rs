//! Model configuration and transformer modules for `jane`.
//!
//! Phase 1: configuration only. Phase 2: the actual model.
//!
//! Module dependency graph:
//! ```text
//! rmsnorm  rope  ffn       ← standalone (Wave 1)
//!     \     |    /
//!      attention            ← uses rope (Wave 2)
//!          |
//!    model (Block, Jane)    ← uses all (Wave 2)
//! ```

pub mod config;

pub mod attention;
pub mod ffn;
pub mod model;
pub mod rmsnorm;
pub mod rope;

pub use config::{ConfigError, JaneConfig, Preset};
pub use model::Jane;
pub use rmsnorm::Norm;
