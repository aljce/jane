# Flight log

Which lanes are in the air, who is flying them, and where each one is in the
impl → review → merge pipeline.

**Orchestrator-owned.** Agents cannot edit this (`agent-orchestrator/ownership`); they report
and the orchestrator records. Kept as a file rather than in the orchestrator's
head for three reasons:

1. **Context is not durable.** A long session gets summarized and in-flight state
   is exactly the kind of detail that evaporates. If the only record of what is
   running lives in the orchestrator's context, one compaction loses the board.
2. **`SendMessage` needs the agent id.** Law 3 routes review findings back to the
   *warm* implementer rather than a cold respawn — that is only possible if the id
   survives. An id nobody wrote down is an agent you can only abandon.
3. **The human can read it.** `make status` beats asking.

`make status` reconciles this file against live git state and complains about
drift, because a stale flight log is worse than none — it lies with authority.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| `queued` | Lane defined, contract committed, not yet spawned |
| `in-flight` | Agent working |
| `blocked` | Escalated; needs an orchestrator decision (see Blocker) |
| `awaiting-review` | Agent reported done, `make gate` green, reviewer not yet spawned |
| `in-review` | `rust-reviewer` reading it |
| `changes-requested` | Findings sent back to the implementer |
| `merged` | On master |
| `abandoned` | Dropped; say why in Blocker |

## Phase 1 lanes

The markers below delimit the machine-parsed region — `scripts/flight.sh` reads
only between them, so the vocabulary table above cannot be mistaken for lanes.
Keep the column order.

<!-- lanes:begin -->
| Lane | Status | Agent | Model | Doing | Blocker |
| --- | --- | --- | --- | --- | --- |
| `model-config` | merged | — | sonnet | `JaneConfig` geometry, `param_count` closed form, the four preset TOMLs, validation | — |
| `tokenizer` | merged | — | opus | Byte-level BPE training/save/load, exact round-trip, streaming text → flat `u16` `.bin` + sidecar | — |
| `dataset` | merged | — | opus | mmap `TokenDataset`, window arithmetic, shifted input/target, `LmBatcher` → `[batch, seq]` Int tensors | — |
| `sources` | merged | — | opus | `DataSource` trait, `curl` raw-text fetch with atomic rename + checksum, HF import with `use_python_venv(false)` | — |
| `train-config` | merged | — | sonnet | `TrainConfig`, `grad_accum_steps` derivation, warmup→cosine `lr_at`, validation | — |
<!-- Phase 2 -->
| `rmsnorm` | merged | — | opus | Hand-rolled RMSNorm from tensor ops, generic Norm trait | — |
| `rope` | merged | — | opus | Rotary positional encoding, precomputed cos/sin cache | — |
| `ffn` | merged | — | opus | SwiGLU feed-forward (gate/up/down projections) | — |
| `attention` | queued | — | — | Causal multi-head self-attention with RoPE | — |
| `model` | queued | — | — | Block (pre-norm + residuals) + Jane (embed → blocks → head) | — |
<!-- lanes:end -->

Model choice is per-lane on purpose: raise it for a lane whose difficulty turns
out to be judgement rather than plumbing, and record the change here.

## Log

Append-only. One line per state transition; newest last. This is the audit trail
for "why is master like this".

- `2026-07-29` — Phase 0 merged. Contracts committed at `320933d`, `make gate`
  green. Five lanes declared in `agent-orchestrator/ownership`. Nothing spawned yet.
- `2026-07-29` — `train-config` split out of `sources` (`5d8caf4`); the two were
  bundled to balance size, which is not a reason.
- `2026-07-29` — All five lanes spawned on sonnet, one per pre-created worktree
  under `../jane-worktrees/`. Two harness features were unavailable because this
  session began before the git repo and the agent definitions existed:
  `.claude/agents/` had not loaded (so `rust-impl` could not be selected, leaving
  Law 2 advisory rather than enforced by tool allowlist), and built-in worktree
  isolation reported "not in a git repository". Worktrees were created with
  `scripts/agent-worktree.sh` instead, which keeps Laws 1, 3 and 4 fully enforced
  — the agents really are in linked worktrees on `agent/*` branches. A Claude Code
  restart restores both features for later waves.
- `2026-07-29` — **All agents stopped on user command.** Surviving state:
  `model-config` committed `5da3b02` (27 tests, gate green) and `train-config`
  committed `da6dc86` (26 tests, gate green) — both complete, both awaiting the
  review Law 3 requires. `dataset` and `sources` were killed mid-work with
  uncommitted edits left in their worktrees (`batcher.rs` + `dataset.rs`, and
  `source.rs`); nothing is lost, but nothing is committed either. `tokenizer` was
  still reading its contract and has no edits. The `train-config` reviewer was
  also stopped before reporting, so no lane has been reviewed and nothing is
  merged.
- `2026-07-29` — Orchestrator ruling on a contract defect `train-config` found and
  I accept: the `lr_at` doc comment demanded the rate stay within `[min_lr, lr]`
  at every step, which is impossible across warmup for the default recipe
  (`lr_at(0)` = 3e-4/500 = 6e-7, well under `min_lr` = 3e-5). `min_lr` is the floor
  of the *cosine decay*, not a global floor, and linear warmup is meant to start
  near zero. The contract wording was wrong, not the implementation. Needs fixing
  in the doc comment before Phase 3 builds the real scheduler against it.
- `2026-07-29` — `model-config` and `train-config` reviewed by fresh
  `rust-reviewer` agents (Law 3). Both clean — no blockers, no warnings. Merged
  to master and `lr_at` doc comment fixed in the same commit. `make gate` green
  on the merged whole (59 tests). Hook updated to allow `worktree-agent-*`
  branches so Agent tool worktree isolation works.
- `2026-07-29` — Restarted `tokenizer`, `dataset`, `sources` with worktree
  isolation (opus). Dataset and sources worktrees rebased onto master and
  partial work preserved. Human review gate added to Law 3: merges >20 lines
  require human approval on the diff.
- `2026-07-29` — All Phase 1 lanes reviewed, approved, and merged. Cross-lane
  fix: source test updated now that `sha256_of_file` is no longer `todo!()`.
  `make gate` green (114 tests). CLI subcommands (`config`, `prepare`,
  `tokenize`) wired up. `RESUME.md` removed (superseded by `make status` +
  flight log). Phase 2 planning started.
