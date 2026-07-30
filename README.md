# jane

A decoder-only transformer written from scratch in Rust with [Burn](https://burn.dev),
trained end-to-end on real text ([TinyStories](https://arxiv.org/abs/2305.07759))
on one laptop GPU.

Attention, RoPE, RMSNorm and SwiGLU are all hand-rolled from tensor primitives —
no `burn::nn::transformer::*`. Target is a 13.8M-parameter model that writes
coherent English. Full plan in **[ROADMAP.md](ROADMAP.md)**.

## Quick start

```sh
nix develop          # or `direnv allow`
make hooks           # once per clone — activates harness enforcement
make gate            # fmt + clippy -D warnings + tests
make smoke-cuda      # verify the GPU path works
```

`make` on its own lists every target. Everything runs inside the Nix dev shell via
`scripts/x`, so `cargo` is never invoked bare.

Status: **Phase 1 complete** (114 tests). Model config with four presets
(1M→150M params), byte-level BPE tokenizer with binarization, mmap dataset +
batching, corpus fetcher (tiny-shakespeare / tinystories-v2), and training config
with warmup→cosine schedule — all merged. **Phase 2 in progress**: Wave 1
(RMSNorm, RoPE, SwiGLU FFN) implementing in parallel; Wave 2 (causal
self-attention, full model) follows.

## The agent harness

Phase work is split into **lanes** run by parallel subagents in separate git
worktrees. One main agent — the **orchestrator** — owns everything that spans
lanes: contracts, shared types, sequencing, merges.

Four laws, enforced in code where code can enforce them:

1. **Only the orchestrator commits to `master`.** A `reference-transaction` hook
   defines the orchestrator structurally as the primary worktree; subagents live in
   linked worktrees. `--no-verify` does not bypass it.
2. **Only the orchestrator spawns subagents.** Their definitions omit the `Agent`
   tool — the capability is absent, not forbidden.
3. **Nothing merges unreviewed.** A fresh read-only reviewer agent checks each lane
   before merge. The implementer is the wrong reader of its own work.
4. **Every file has one owner.** A `pre-commit` hook enforces a path manifest, so
   lanes cannot collide and merges are mechanical.

Read **[harness-soul.md](harness-soul.md)** for the reasoning, the escalation
protocol, and an honest account of the two holes that remain.

Build isolation: each worktree keeps its own `target/` (cargo locks a build dir, so
sharing one would serialize agents), with a shared `sccache` so Burn's dependency
tree compiles once. This is why `.cargo/config.toml` sets `incremental = false`.

## Harness files

| Path | What |
| --- | --- |
| [harness-soul.md](harness-soul.md) | The orchestrator's role and the four laws |
| [agent-orchestrator/ownership](agent-orchestrator/ownership) | Lane → owned path globs. Enforced. |
| [agent-orchestrator/flight.md](agent-orchestrator/flight.md) | What's in flight, agent ids, pipeline stage (`make status`) |
| [.githooks/](.githooks/) | `reference-transaction`, `pre-commit`, `pre-push` |
| [.claude/agents/](.claude/agents/) | `rust-impl`, `rust-reviewer` definitions |
| [scripts/x](scripts/x) | Run any command in the dev shell |
| [scripts/agent-worktree.sh](scripts/agent-worktree.sh) | Create / list / remove lane worktrees |
| [Makefile](Makefile) | Every command worth having a short name |

## Layout

```
crates/jane-model/   config, RMSNorm, RoPE, SwiGLU FFN, attention, blocks
crates/jane-data/    corpus fetch, BPE tokenizer, binarize, mmap dataset + batcher
crates/jane-train/   training config (warmup→cosine LR); learner loop in Phase 3
crates/jane-cli/     jane smoke|config|prepare|tokenize
configs/             jane-{1m,14m,60m,150m}.toml
```

## License

MIT OR Apache-2.0
