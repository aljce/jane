# jane — a transformer from scratch in Rust

A decoder-only (GPT-style) language model implemented from first principles with
[Burn](https://burn.dev), trained end-to-end on real text on local hardware.

## Goals

- Implement attention, positional encoding, normalization, and the feed-forward
  block **by hand** from Burn tensor primitives — no `burn::nn::transformer::*`.
- Train on a **real dataset** to a checkpoint that generates coherent English.
- Keep the whole thing reproducible: `nix develop` → `cargo run` → trained model.

## Non-goals

- Competing with frontier models, or any multi-GPU / distributed training.
- Fine-tuning or adapting pretrained weights. Everything is trained from random init.
- Novel architecture research. This is a known-good recipe, built carefully.

---

## 1. Environment (verified on this machine)

These were probed, not assumed:

| Fact | Value |
| --- | --- |
| Rust | 1.91.1 (Burn MSRV is 1.89 ✓) |
| Burn (latest stable) | **0.20.1** — `0.22.0-pre.1` is a pre-release, avoid |
| GPU | **NVIDIA RTX 5070 Ti Laptop, 12227 MiB VRAM** |
| Driver / CUDA | 577.05 / CUDA 12.9 |
| nixpkgs CUDA toolkit | 12.8 (`cuda_nvrtc` 12.8.93) |
| CPU / RAM | 16 cores / 15 GB |
| Host | NixOS 25.11 on WSL2, Nix 2.31.4, flakes enabled |
| Measured fp32 matmul | **9.8 TFLOP/s** (4096², warm) via `make smoke-cuda` |

**Three environment gotchas already solved in `flake.nix`:**

1. `nvidia-smi` fails by default here. The WSL driver libraries live in
   `/usr/lib/wsl/lib` (`libcuda.so.1`, `libnvidia-ml.so.1`), which NixOS does not
   put on the dynamic linker path. The dev shell prepends it to
   `LD_LIBRARY_PATH`. On non-WSL NixOS the equivalent path is
   `/run/opengl-driver/lib`.
2. The RTX 5070 Ti is Blackwell (**sm_120**). Frameworks that ship precompiled
   fatbins (e.g. stable PyTorch) still lag on sm_120. Burn sidesteps this: CubeCL
   compiles kernels **at runtime via NVRTC** for the detected compute capability.
   **Verified working** — see §6.
3. `CUDA_PATH` must point at `cudaPackages.cudatoolkit` (the *merged*
   distribution), not `cuda_nvcc` or `cuda_cudart`. CubeCL emits CUDA C
   containing `#include <cuda_runtime.h>` and passes
   `--include-path=$CUDA_PATH/include` to NVRTC, so the header must really be
   there. The split outputs don't have it, and the failure is a runtime
   `catastrophic error: cannot open source file "cuda_runtime.h"` on the first
   kernel launch — nothing at build time warns you.

### Backends

| Backend | Use |
| --- | --- |
| `burn-cuda` | Primary. All real training runs. |
| `burn-ndarray` | Correctness reference. Tiny shapes, unit tests, CI, gradient checks. |
| `burn-wgpu` | Portability only. Weak fallback here — WSL exposes no NVIDIA Vulkan driver (`/run/opengl-driver/lib` is Mesa-only), so it would run on a software rasterizer. |

Every module is generic over `B: Backend` so the same code runs on all three.

---

## 2. Data — where it comes from

The dataset choice is the highest-leverage decision in the project. A 14M-parameter
model trained on the *right* corpus produces fluent English; the same model on
general web text produces mush, because it never sees any distribution often
enough to learn it.

### Primary: TinyStories V2

[`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) —
synthetic short children's stories, constrained to the vocabulary of a 3–4 year
old. It exists precisely to answer "how small can a language model be and still
speak coherent English?" ([paper](https://arxiv.org/abs/2305.07759)). Models
**under 10M parameters** trained on it produce grammatical, logically coherent
prose. That is exactly our budget.

Use the **V2 / GPT-4-only** files — the V1 files include weaker GPT-3.5 output:

```
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt   (~2.2 GB)
https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt   (~22 MB)
```

Stories are separated by `<|endoftext|>`. At ~4.4 chars/token with an 8k BPE
vocab this is roughly **500M tokens** — more than enough (see the budget in §4).

### Ingest: two supported paths

The flake provides Python (`datasets`, `sqlalchemy`, `pyarrow`, `pillow`,
`soundfile`), so Burn's `HuggingfaceDatasetLoader` is available and is the
default path. It downloads a dataset's splits into a single SQLite database
usable through `SqliteDataset`.

**One mandatory adjustment.** By default the loader creates a venv and
`pip install`s its dependencies. That is broken on NixOS: pip's binary wheels
(`pyarrow` especially) ship unpatched ELF interpreters and won't execute. Always
build the loader with:

```rust
HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
    .with_use_python_venv(false) // REQUIRED on NixOS — use the flake's python3
```

That makes it invoke the ambient `python3`, which is the Nix-provided env.

The `curl`-the-raw-`.txt` path stays as a **fallback** for two cases where it is
genuinely better: Rung 0 (tiny-shakespeare isn't an HF dataset) and the V2/GPT-4
files, which are loose `.txt` in the repo rather than rows in the parquet
conversion. Keep both behind one `DataSource` trait so the rest of the pipeline
doesn't care which was used.

### The dataset ladder

Work up this ladder; each rung de-risks the next. Do not start at rung 3.

| # | Corpus | Size | Purpose |
| --- | --- | --- | --- |
| 0 | [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | 1 MB | Smoke test. Char-level, no tokenizer needed. Must overfit a 100-sample slice to ~0 loss — this is how you prove the backward pass is correct. |
| 1 | TinyStories V2 (1% slice) | ~22 MB | Full pipeline shakedown; a real training curve in minutes. |
| 2 | **TinyStories V2 (full)** | 2.2 GB | **The main event.** Target coherent generation. |
| 3 | [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | ~500 MB | Real human prose, much harder. Expect a big loss jump — factual text needs far more capacity. |
| 4 | [FineWeb-Edu `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 10B tokens | Only if scaling up the model. Parquet, so this rung *does* need a parquet reader. |

Licensing: TinyStories is CDLA-Sharing-1.0, WikiText CC-BY-SA-3.0, FineWeb-Edu
ODC-By. All fine for this; note them if anything is ever published.

### On-disk format

Tokenize once, ahead of time, into a flat `u16` little-endian array of token IDs
(8k vocab fits `u16`) written to `data/{train,valid}.bin`, plus a
`tokenizer.json`. Training then memory-maps the file and samples random windows —
no per-epoch tokenization, no text parsing in the hot loop. 500M tokens × 2 bytes
= **~1 GB**, which fits in page cache alongside everything else.

---

## 3. Architecture

### Tokenization and the embedding table

**The token embedding is `[vocab_size, d_model]` = `[8192, 384]`.** Each token ID
indexes one row and comes out as a **384-dim** vector. That 384 is `d_model`, the
width of the residual stream — the same width every block reads and writes, all
the way to the final norm. There is no separate "embedding dimension" knob to
tune; widening the embedding *is* widening the model.

**It is not character tokenization.** It's byte-level **BPE with an 8192 vocab**,
trained on a TinyStories subset. Only Rung 0 (tiny-shakespeare) is char-level,
deliberately: it lets us exercise the training loop before a tokenizer exists.

**On reusing an existing tokenizer** — split the question in two:

- *Implementation*: **reuse.** Use HuggingFace's
  [`tokenizers`](https://crates.io/crates/tokenizers) crate. Hand-rolling BPE
  merge-ranking is fiddly, cache-sensitive, and teaches nothing about
  transformers. It's also pure Rust, so it costs us nothing.
- *Vocabulary*: **train our own.** Don't adopt GPT-2's.

The reason is parameter budget, and at our scale it's decisive:

| Vocab | Embedding params | As % of model |
| --- | --- | --- |
| 8,192 (ours) | 8192 × 384 = **3.1M** | 23% of 13.8M |
| 32,000 (Llama) | 32000 × 384 = **12.3M** | 54% of 22.9M |
| 50,257 (GPT-2) | 50257 × 384 = **19.3M** | **65% of 29.9M** |

With GPT-2's tokenizer, the lookup table would be nearly **twice the size of the
entire rest of the network** (10.6M). Weight tying doesn't rescue this — it's
still one 19.3M table. We'd be training a big sparse dictionary with a small
transformer bolted on.

It's also a poor fit: TinyStories is *deliberately* restricted to a young child's
vocabulary, so most of GPT-2's 50k entries never appear. Those rows receive
essentially zero gradient and sit at initialization forever — dead weight
consuming VRAM and two Adam moments each.

Rule of thumb to carry forward: **keep embedding params under ~25% of the total.**
That ratio, not a fixed number, is what should drive `vocab_size` — see the
presets in §3.2. Once we reach `jane-150m` or move to Rung 3/4, 16k–32k becomes
correct, and adopting GPT-2's tokenizer becomes reasonable.

### Layers

Modern decoder-only stack — the post-GPT-2 consensus, all hand-rolled:

- **Embedding**: learned token embeddings, **weight-tied** to the output head.
- **Positional**: **RoPE** (rotary), applied to Q and K. No learned position
  table — better length generalization, fewer parameters.
- **Attention**: multi-head, **causal mask**, scaled dot-product. Written from
  raw tensor ops: `reshape` → `swap_dims` → `matmul` → mask → `softmax` → `matmul`.
- **Norm**: **RMSNorm**, **pre-norm** placement (pre-norm is what makes deep
  stacks trainable without warmup gymnastics).
- **FFN**: **SwiGLU**, `d_ff = 1024` (three matrices: gate, up, down).
- **Residual** connections around attention and FFN.

### 3.1 Reference config (`jane-14m`)

| Param | Value |
| --- | --- |
| `vocab_size` | 8192 |
| `d_model` | 384 |
| `n_layers` | 6 |
| `n_heads` | 6 (head_dim 64) |
| `d_ff` | 1024 |
| `seq_len` | 512 |
| dropout | 0.0 (single epoch on 500M tokens — no overfitting risk) |

Parameter budget:

```
token embedding   8192 × 384                        =  3,145,728   (tied w/ head)
per block:
  attn Wq,Wk,Wv,Wo  4 × 384×384      =   589,824
  SwiGLU            3 × 384×1024     = 1,179,648
  2 × RMSNorm       2 × 384          =       768
                                       ---------
                                       1,770,240  × 6 layers = 10,621,440
final RMSNorm                                     =         384
                                                    ----------
total                                             ≈ 13.8M parameters
```

As a closed form, with `V`=vocab, `D`=d_model, `L`=layers, `F`=d_ff, tied head:

```
params = V·D  +  L·(4·D² + 3·D·F + 2·D)  +  D
```

This formula — not a literal like `13_767_552` — is what the parameter-count test
asserts against, so it keeps holding as configs change.

### 3.2 Scaling presets

Every preset is the same code with different numbers. `vocab_size` grows with
`d_model` to hold embedding params under ~25% (§3, tokenization).

Counts below are computed from the formula above, not estimated:

| Preset | `vocab` | `d_model` | `L` | `heads` | `d_ff` | `seq` | Params | Embed % | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `jane-1m` | 4,096 | 128 | 4 | 4 | 320 | 256 | 1.28M | 41.0% | CI / CPU-only tests |
| **`jane-14m`** | 8,192 | 384 | 6 | 6 | 1024 | 512 | **13.77M** | 22.8% | **Primary target** |
| `jane-60m` | 16,384 | 640 | 10 | 10 | 1728 | 1024 | 60.06M | 17.5% | Rung 3, WikiText-103 |
| `jane-150m` | 32,768 | 896 | 12 | 14 | 2368 | 1024 | 144.3M | 20.3% | Rung 4, only if warranted |

All four keep `head_dim = 64` (except `jane-1m` at 32) and satisfy
`d_model % n_heads == 0`.

`d_ff = round(8/3 · d_model / 64) · 64` — the standard SwiGLU ratio, which keeps
the FFN's parameter count level with a `4·d_model` GELU FFN despite the third
matrix, rounded to a multiple of 64 for tensor-core alignment.

`jane-1m` knowingly breaks the ≤25%-embedding rule: byte-level BPE needs 256 base
byte tokens before any merges, so vocab can't shrink proportionally at that size.
It's a plumbing test, not a model that has to be good.

`jane-150m` was sized to fit this GPU: AdamW fp32 master weights + 2 moments come
to **2.3 GB**, leaving room for activations inside 12 GB.

### 3.3 Configuration discipline

Scaling to bigger models must be a config change, never an edit to model code.

- **One `JaneConfig`** carrying every dimension, derived via Burn's `#[derive(Config)]`
  (serde-backed, so it round-trips to TOML/JSON). No literal ever appears in a
  module body. Same for `TrainConfig` (lr, warmup, steps, batch, decay, clip,
  seed) and `DataConfig` (paths, corpus, tokenizer, split).
- **Derive, don't duplicate.** `head_dim = d_model / n_heads`, asserted divisible
  at construction with a real error message. `d_ff` defaults to the 8/3 rule but
  stays overridable. Nothing that can be computed gets its own field.
- **Presets are named files**, `configs/jane-{2m,14m,50m,150m}.toml`, selected by
  `jane train --config configs/jane-14m.toml`. Every override is a CLI flag.
- **Serialize the config next to every checkpoint.** A checkpoint whose
  hyperparameters are unknown is not a result. This also makes resume exact.
- **Decouple effective batch from VRAM.** Configure `tokens_per_step`; derive
  grad-accumulation as `tokens_per_step / (micro_batch × seq_len)`. Then a bigger
  model that forces a smaller micro-batch trains at the *same* effective batch,
  and loss curves stay comparable across presets.
- **Generic over `B: Backend`** everywhere, so the CPU reference path costs nothing.
- **Seed everything** from config; log it.

The test suite runs against `jane-1m` on `ndarray` — small enough for CI, and it
proves the config plumbing works at more than one size.

---

## 4. Training plan

| Setting | Value |
| --- | --- |
| Objective | next-token cross-entropy |
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 |
| LR schedule | linear warmup 500 steps → 3e-4 → cosine decay to 3e-5 |
| Grad clip | 1.0 global norm |
| Precision | bf16 compute, fp32 master weights |
| Batch | 32 × 512 = **16,384 tokens/step** (grad-accumulate if VRAM-bound) |
| Steps | ~18,000 (≈300M tokens, ~0.6 epochs) |

**Token budget.** Chinchilla-optimal for 13.8M params is ~20 tokens/param ≈ **276M
tokens**, so ~300M is the right target and TinyStories' ~500M gives comfortable headroom.

**Memory.** Params + grads + 2 Adam moments at fp32 = 13.8M × 16 B ≈ **220 MB**.
Activations at 32×512 are the real consumer but land far inside 12 GB. VRAM is not
the binding constraint; step time is.

**Time.** FLOPs ≈ `6 × N × D` = 6 × 13.8e6 × 300e6 ≈ **2.5e16**. Even at a
pessimistic few TFLOP/s effective through Burn, this is a **single-digit hours**
run. Budget one overnight run, expect less.

**Success criteria:**
- Rung 0: overfit 100 samples to loss < 0.1. *Non-negotiable gate before Rung 2.*
- Rung 2: val loss ≲ 1.5, and sampled continuations are grammatical, on-topic,
  and keep a character consistent across a paragraph.

---

## 5. Phases

Each phase ends in something runnable.

### Phase 0 — Scaffolding ✅
- [x] `flake.nix` with Rust 1.91.1 + CUDA 12.8 + WSL driver path wiring
- [x] `rust-toolchain.toml`, `.envrc`, `.gitignore`
- [x] Python env in the flake for `HuggingfaceDatasetLoader` (`datasets`, `sqlalchemy`, `pyarrow`, `pillow`, `soundfile`)
- [x] Cargo workspace: `jane-model`, `jane-data`, `jane-train`, `jane-cli`
- [x] Agent harness: sccache build isolation, worktree lanes, ownership
      enforcement, `Makefile` (see `harness-soul.md`)
- [x] **Backend smoke test** — `make smoke` (CPU) and `make smoke-cuda` (GPU) both
      pass. This was the Phase 0 gate; sm_120 is confirmed working.
- [ ] `JaneConfig` / `TrainConfig` + the four preset TOMLs (§3.3) — *contracts
      written, implementation delegated to agent lanes*

### Phase 1 — Data pipeline
- [ ] `DataSource` trait with two impls: `HuggingfaceDatasetLoader`
      (**with `.with_use_python_venv(false)`**) and raw-`.txt` via `curl`
- [ ] Verify the HF path end-to-end on a small dataset before pointing it at 2.2 GB
- [ ] Train byte-level BPE with `tokenizers` at the config's `vocab_size`;
      emit `data/tokenizer.json`
- [ ] Tokenize corpus → `data/{train,valid}.bin` (flat `u16`), with progress reporting
- [ ] Memory-mapped `Dataset` impl: random `seq_len+1` windows → `(input, target)`
- [ ] Burn `Batcher` + `DataLoader`, multi-worker
- [ ] Round-trip test: `decode(encode(s)) == s`; assert all IDs < `vocab_size`

### Phase 2 — The model
- [ ] `RmsNorm` + unit test against a hand-computed reference
- [ ] RoPE: precomputed cos/sin cache; test that it preserves vector norms
- [ ] `CausalSelfAttention` — assert the mask is strictly lower-triangular, and
      test that token *t*'s output is unchanged when tokens `>t` are perturbed
      (the real proof there's no future leakage)
- [ ] `SwiGLU` FFN
- [ ] `Block` (pre-norm + 2 residuals), then `Jane` (embed → N blocks → norm → tied head)
- [ ] Parameter-count test against the **§3.1 formula**, checked across all four
      presets (13,767,552 for `jane-14m`) — never a bare literal
- [ ] Shape test: `[batch, seq]` → `[batch, seq, vocab]` on `ndarray`
- [ ] Instantiate every preset on `ndarray` at tiny batch, to prove no config
      hardcodes anything

### Phase 3 — Training
- [ ] `TrainStep`/`ValidStep`, cross-entropy over flattened logits
- [ ] AdamW + warmup-cosine LR scheduler
- [ ] Gradient clipping
- [ ] `Learner`: loss + perplexity metrics, TUI renderer, checkpointing
- [ ] **Gate: overfit 100 samples to ~0 loss** (Rung 0)
- [ ] Rung 1 run on the 1% slice; confirm a sane loss curve
- [ ] Config via CLI/TOML; log every hyperparameter next to each checkpoint
- [ ] Resume-from-checkpoint
- [ ] **Rung 2: the full run**

### Phase 4 — Inference
- [ ] `generate` with temperature, top-k, top-p
- [ ] **KV cache** — turns O(n²) re-computation per token into O(n)
- [ ] `jane sample --prompt "Once upon a time"` CLI
- [ ] Sample-quality log: fixed prompt set, sampled every N checkpoints, committed
      so quality drift over training is visible

### Phase 5 — Evaluation & scaling
- [ ] Held-out val loss / perplexity as the headline number
- [ ] Ablations, one variable at a time: RoPE vs. learned pos, RMSNorm vs.
      LayerNorm, SwiGLU vs. GELU, pre-norm vs. post-norm
- [ ] Tokens/sec and VRAM by config; find the batch-size knee
- [ ] Scale to `jane-60m` and/or Rung 3 (WikiText-103)

### Phase 6 — Stretch
- [ ] Grouped-query attention
- [ ] Fused attention kernel written directly in CubeCL
- [ ] Post-training quantization; `burn-import` ONNX export
- [ ] `wgpu` portability run (accepting software-rasterizer speed here)

---

## 6. Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| ~~CubeCL/NVRTC can't target sm_120~~ | **RETIRED** | Verified on 2026-07-29: `make smoke-cuda` runs a 4096² fp32 matmul at 9.8 TFLOP/s on the RTX 5070 Ti and reads the result back correctly. CubeCL's runtime NVRTC compilation handles sm_120. The one real obstacle was the `CUDA_PATH` header issue in §1, not the architecture. |
| **HF loader pip-installs into a venv** | Medium — fails on NixOS | `.with_use_python_venv(false)` so it uses the flake's `python3`. Wrap loader construction in one helper so this can't be forgotten at a second call site. |
| Burn 0.20 API drift vs. online examples | Medium | Trust `cargo doc` and the version's own `examples/`, not blog posts. Pin the exact version in `Cargo.toml`. |
| Silent causal-mask bug | Medium — trains fine, "too good" val loss | The future-perturbation test in Phase 2. A model that peeks looks *better* on loss, so only an explicit test catches it. |
| WSL2 disk/page-cache pressure | Low | 2.2 GB raw + ~1 GB tokenized against 15 GB RAM. Stream tokenization; never hold the corpus as one `String`. |
| bf16 instability | Low | fp32 master weights; fall back to fp32 compute if loss goes NaN. |

---

## 7. Layout

```
jane/
├── flake.nix              # CUDA + Python + Rust dev shell
├── rust-toolchain.toml
├── ROADMAP.md
├── Cargo.toml             # workspace
├── configs/               # jane-1m.toml, jane-14m.toml, jane-60m.toml, jane-150m.toml
├── crates/
│   ├── jane-model/        # config, RoPE, RMSNorm, attention, SwiGLU, blocks
│   ├── jane-data/         # DataSource (HF + raw), BPE, binarize, memmap dataset
│   ├── jane-train/        # train step, optimizer, scheduler, learner
│   └── jane-cli/          # `jane train|sample|tokenize|smoke`
└── data/                  # gitignored: .hf/ cache, *.txt, *.bin, tokenizer.json
```

## 8. References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [TinyStories](https://arxiv.org/abs/2305.07759) · [dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) · [RMSNorm](https://arxiv.org/abs/1910.07467) · [GLU Variants](https://arxiv.org/abs/2002.05202)
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- [Burn book](https://burn.dev/books/burn/) · [`burn` 0.20.1 docs](https://docs.rs/burn/0.20.1) · [Burn examples](https://github.com/tracel-ai/burn/tree/main/examples)
- [`tokenizers` crate](https://docs.rs/tokenizers)
## 9. Tooling backlog

Developer-experience requests, most from subagents hitting friction mid-lane.

Agents cannot edit this file — `Makefile`, `scripts/` and `ROADMAP.md` are all
orchestrator-owned (`agent-orchestrator/ownership`). That is deliberate: four agents each
"just adding a quick target" is how a build system rots. Instead an agent reports
the friction in its final summary and the orchestrator transcribes it here, so
requests are visible, deduplicated, and decided once.

Request format: what you were doing, what was awkward, what would have helped.

### Done

- [x] `make` targets wrapping `./scripts/x cargo …` — the raw commands were long
      enough to be error-prone and easy to run outside the dev shell by accident
- [x] Per-lane test targets (`make t-model`, `t-tokenizer`, `t-dataset`,
      `t-sources`, `t-train`) so a lane doesn't wait on unrelated crates
- [x] `make gate` as the single pre-review command
- [x] `make sccache-stats` to confirm the harness is actually sharing compilation

### Requested

- [ ] `make t-<lane> WATCH=1` via `cargo-watch` (already in the flake) for a
      continuous loop
- [ ] `make fix` = `cargo clippy --fix` + `cargo fmt`, for mechanical lint churn
- [ ] A `jane inspect-bin <file>` command dumping the first N tokens of a `.bin`
      with their decoded text — the obvious tool for debugging the dataset lane,
      and currently everyone writes it ad hoc in a test
- [ ] `make nextest` — `cargo-nextest` is in the flake and gives better output
      and per-test isolation than `cargo test`
- [ ] Test fixture helper for a tiny deterministic `.bin` + sidecar. Both the
      dataset and tokenizer lanes need one; if both write their own it belongs in
      an orchestrator-owned `jane-data/src/testing.rs` instead (§5 of
      harness-soul.md)
- [ ] `make prepare-tiny` for a fast rung-0/rung-1 end-to-end pipeline run

