//! `jane` command-line interface.

use std::path::PathBuf;

use anyhow::{Context, Result};
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
        /// Path to a TOML config file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// A preset name: 1m, 14m, 60m, 150m.
        #[arg(long, conflicts_with = "config")]
        preset: Option<String>,
    },

    /// Fetch a corpus, train a tokenizer and binarize — the whole Phase 1 pipeline.
    Prepare {
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Corpus: tiny-shakespeare, tinystories-v2
        #[arg(long, default_value = "tiny-shakespeare")]
        corpus: String,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
    },

    /// Encode text with a trained tokenizer and echo the round trip.
    Tokenize {
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: PathBuf,
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
        Command::Config { config, preset } => cmd_config(config, preset),
        Command::Prepare {
            data_dir,
            corpus,
            vocab_size,
        } => cmd_prepare(&data_dir, &corpus, vocab_size),
        Command::Tokenize { tokenizer, text } => cmd_tokenize(&tokenizer, &text),
    }
}

fn cmd_config(config: Option<PathBuf>, preset: Option<String>) -> Result<()> {
    let cfg = match (config, preset) {
        (Some(path), _) => jane_model::JaneConfig::load(&path)
            .with_context(|| format!("loading {}", path.display()))?,
        (_, Some(name)) => {
            let p = jane_model::Preset::parse(&name)
                .with_context(|| format!("unknown preset {name:?}"))?;
            jane_model::JaneConfig::preset(p)
        }
        (None, None) => {
            println!("No --config or --preset given; showing all presets:\n");
            for p in jane_model::Preset::all() {
                let c = jane_model::JaneConfig::preset(p);
                print_config(p.name(), &c);
                println!();
            }
            return Ok(());
        }
    };
    print_config("custom", &cfg);
    Ok(())
}

fn print_config(label: &str, cfg: &jane_model::JaneConfig) {
    println!("  preset       : {label}");
    println!("  vocab_size   : {}", cfg.vocab_size);
    println!("  d_model      : {}", cfg.d_model);
    println!("  n_layers     : {}", cfg.n_layers);
    println!("  n_heads      : {}", cfg.n_heads);
    println!("  head_dim     : {}", cfg.head_dim());
    println!("  d_ff         : {}", cfg.d_ff());
    println!("  seq_len      : {}", cfg.seq_len);
    println!("  dropout      : {}", cfg.dropout);
    println!("  rope_theta   : {}", cfg.rope_theta);
    println!("  tie_embed    : {}", cfg.tie_embeddings);
    println!(
        "  params       : {:>12} ({:.2}M)",
        cfg.param_count(),
        cfg.param_count() as f64 / 1e6
    );
    println!(
        "  embed params : {:>12} ({:.1}%)",
        cfg.embedding_params(),
        cfg.embedding_fraction() * 100.0
    );
}

fn cmd_prepare(data_dir: &std::path::Path, corpus: &str, vocab_size: usize) -> Result<()> {
    use jane_data::tokenizer::EOT_TOKEN;
    use jane_data::{ByteTokenizer, DataSource, RawTextSource, Tokenizer, binarize_text_file};

    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("creating {}", data_dir.display()))?;

    // 1. Fetch corpus
    let source: Box<dyn DataSource> = match corpus {
        "tiny-shakespeare" => Box::new(RawTextSource::tiny_shakespeare()),
        "tinystories-v2" => {
            // Fetch both train and valid
            let train_src = RawTextSource::tinystories_v2_train();
            let valid_src = RawTextSource::tinystories_v2_valid();
            println!("[1/4] fetching train split...");
            let train_path = train_src.fetch(data_dir).context("fetching train corpus")?;
            println!("       {}", train_path.display());
            println!("       fetching valid split...");
            let valid_path = valid_src.fetch(data_dir).context("fetching valid corpus")?;
            println!("       {}", valid_path.display());

            // Train tokenizer on train split
            println!("[2/4] training tokenizer (vocab_size={vocab_size})...");
            let tok = ByteTokenizer::train_from_files(&[&train_path], vocab_size)
                .context("training tokenizer")?;
            let tok_path = data_dir.join("tokenizer.json");
            tok.save(&tok_path).context("saving tokenizer")?;
            println!(
                "       {} (vocab_size={})",
                tok_path.display(),
                tok.vocab_size()
            );

            // Binarize both splits
            println!("[3/4] binarizing train...");
            let train_bin = data_dir.join("train.bin");
            let stats = binarize_text_file(&train_path, &tok, &train_bin, Some(EOT_TOKEN))
                .context("binarizing train")?;
            println!("       {} tokens, {} docs", stats.tokens, stats.docs);

            println!("[4/4] binarizing valid...");
            let valid_bin = data_dir.join("valid.bin");
            let stats = binarize_text_file(&valid_path, &tok, &valid_bin, Some(EOT_TOKEN))
                .context("binarizing valid")?;
            println!("       {} tokens, {} docs", stats.tokens, stats.docs);

            println!("\ndone.");
            return Ok(());
        }
        other => {
            anyhow::bail!("unknown corpus {other:?}; available: tiny-shakespeare, tinystories-v2")
        }
    };

    // Single-file corpora (tiny-shakespeare)
    println!("[1/3] fetching corpus...");
    let text_path = source.fetch(data_dir).context("fetching corpus")?;
    println!("       {}", text_path.display());

    println!("[2/3] training tokenizer (vocab_size={vocab_size})...");
    let tok =
        ByteTokenizer::train_from_files(&[&text_path], vocab_size).context("training tokenizer")?;
    let tok_path = data_dir.join("tokenizer.json");
    tok.save(&tok_path).context("saving tokenizer")?;
    println!(
        "       {} (vocab_size={})",
        tok_path.display(),
        tok.vocab_size()
    );

    println!("[3/3] binarizing...");
    let bin_path = data_dir.join("train.bin");
    let doc_sep = if corpus == "tiny-shakespeare" {
        None
    } else {
        Some(EOT_TOKEN)
    };
    let stats = binarize_text_file(&text_path, &tok, &bin_path, doc_sep).context("binarizing")?;
    println!("       {} tokens, {} docs", stats.tokens, stats.docs);

    println!("\ndone.");
    Ok(())
}

fn cmd_tokenize(tokenizer_path: &std::path::Path, text: &str) -> Result<()> {
    use jane_data::{ByteTokenizer, Tokenizer};

    let tok = ByteTokenizer::load(tokenizer_path)
        .with_context(|| format!("loading {}", tokenizer_path.display()))?;

    let ids = tok.encode(text).context("encoding")?;
    let decoded = tok.decode(&ids).context("decoding")?;

    println!("vocab_size : {}", tok.vocab_size());
    println!("input      : {text:?}");
    println!("ids        : {ids:?}");
    println!("n_tokens   : {}", ids.len());
    println!("decoded    : {decoded:?}");
    println!(
        "round-trip : {}",
        if decoded == text { "OK" } else { "MISMATCH" }
    );
    Ok(())
}
