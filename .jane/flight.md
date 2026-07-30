# Flight log

Which lanes are in the air, who is flying them, and where each one is in the
impl → review → merge pipeline.

**Orchestrator-owned.** Agents cannot edit this (`.jane/ownership`); they report
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
| `model-config` | queued | — | sonnet | `JaneConfig` geometry, `param_count` closed form, the four preset TOMLs, validation | — |
| `tokenizer` | queued | — | sonnet | Byte-level BPE training/save/load, exact round-trip, streaming text → flat `u16` `.bin` + sidecar | — |
| `dataset` | queued | — | sonnet | mmap `TokenDataset`, window arithmetic, shifted input/target, `LmBatcher` → `[batch, seq]` Int tensors | — |
| `sources` | queued | — | sonnet | `DataSource` trait, `curl` raw-text fetch with atomic rename + checksum, HF import with `use_python_venv(false)` | — |
| `train-config` | queued | — | sonnet | `TrainConfig`, `grad_accum_steps` derivation, warmup→cosine `lr_at`, validation | — |
<!-- lanes:end -->

Model choice is per-lane on purpose: raise it for a lane whose difficulty turns
out to be judgement rather than plumbing, and record the change here.

## Log

Append-only. One line per state transition; newest last. This is the audit trail
for "why is master like this".

- `2026-07-29` — Phase 0 merged. Contracts committed at `320933d`, `make gate`
  green. Five lanes declared in `.jane/ownership`. Nothing spawned yet.
- `2026-07-29` — `train-config` split out of `sources` (`5d8caf4`); the two were
  bundled to balance size, which is not a reason.
