//! `jane` command-line interface.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

mod smoke;

#[derive(Parser)]
#[command(name = "jane", version, about = "A transformer from scratch in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify a backend can allocate, compute and read back a tensor.
    ///
    /// This is the Phase 0 gate: it proves CubeCL can compile kernels for this
    /// GPU's compute capability (sm_120 on Blackwell) before any effort goes
    /// into the model.
    Smoke {
        #[arg(long, value_enum, default_value_t = BackendChoice::Ndarray)]
        backend: BackendChoice,
        /// Square matrix size for the matmul.
        #[arg(long, default_value_t = 512)]
        size: usize,
    },

    /// Print a config's derived geometry and parameter budget.
    Config {
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        /// A preset name: 1m, 14m, 60m, 150m.
        #[arg(long, conflicts_with = "config")]
        preset: Option<String>,
    },

    /// Fetch a corpus, train a tokenizer and binarize — the whole Phase 1 pipeline.
    Prepare {
        #[arg(long, default_value = "data")]
        data_dir: std::path::PathBuf,
        #[arg(long, default_value = "tiny-shakespeare")]
        corpus: String,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
    },

    /// Encode text with a trained tokenizer and echo the round trip.
    Tokenize {
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: std::path::PathBuf,
        text: String,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    /// CPU reference backend. Always available.
    Ndarray,
    /// Requires `--features cuda`.
    Cuda,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Smoke { backend, size } => smoke::run(backend, size),
        Command::Config { .. } => {
            anyhow::bail!("`jane config` lands once jane-model's config is implemented")
        }
        Command::Prepare { .. } => {
            anyhow::bail!("`jane prepare` lands once jane-data's pipeline is implemented")
        }
        Command::Tokenize { .. } => {
            anyhow::bail!("`jane tokenize` lands once jane-data's tokenizer is implemented")
        }
    }
}
