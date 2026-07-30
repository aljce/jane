# Resume brief

Paste-and-go context for a new orchestrator session. Keep it current — this is
what a fresh session reads instead of re-deriving the board.

## You are the orchestrator

Read `harness-soul.md` first (it is short). You own everything that spans lanes:
contracts, shared types, sequencing, merges. Subagents own one module each and
cannot see one another.

Then run `make status`. That is the live board, reconciled against git.

## Where the project is

**Phase 0: done.** Workspace, agent harness, and a verified CUDA backend —
`make smoke-cuda` gets 9.8 TFLOP/s fp32 on the RTX 5070 Ti, so sm_120 is not a
risk. `make gate` is green on master.

**Phase 1: two of five lanes complete, none merged.**

| Lane | State |
| --- | --- |
| `model-config` | committed `5da3b02`, 27 tests, gate green — **needs review** |
| `train-config` | committed `da6dc86`, 26 tests, gate green — **needs review** |
| `tokenizer` | not started; worktree clean |
| `dataset` | stopped mid-work; **uncommitted** `dataset.rs` + `batcher.rs` in its worktree |
| `sources` | stopped mid-work; **uncommitted** `source.rs` in its worktree |

Nothing has been reviewed and nothing is merged. Law 3 requires a fresh
`rust-reviewer` on each lane before it lands.

## Next actions, in order

1. Review the two finished lanes, then merge. Lanes are disjoint, so merging is
   mechanical.
2. Fix the `lr_at` contract defect below while you are in that file.
3. Restart `tokenizer`, `dataset`, `sources`. The last two have work on disk —
   tell the agent to inspect it before overwriting.
4. Integrate: `jane-cli`'s `config`/`prepare`/`tokenize` subcommands currently
   `bail!` with "lands once <lane> is implemented". They are orchestrator-owned.
5. `make gate` on the merged whole. Five green lanes do not imply a green
   workspace.

## Open decision the orchestrator must carry

**The `lr_at` doc comment in `crates/jane-train/src/config.rs` is wrong.** It
requires the learning rate to stay within `[min_lr, lr]` at every step, which is
impossible across warmup for the shipped recipe: `lr_at(0)` = 3e-4/500 = 6e-7,
far below `min_lr` = 3e-5. `min_lr` is the floor of the *cosine decay*, not a
global floor, and linear warmup is meant to start near zero. The implementation is
correct; the specification was not. Fix the wording before Phase 3 builds the real
scheduler against it.

## Environment facts already established — do not re-derive

- Burn is pinned to **0.20.1**. `0.22.0-pre` is a pre-release. Trust `cargo doc`
  for its API, not blog posts.
- `CUDA_PATH` must point at `cudaPackages.cudatoolkit` (the *merged* output).
  CubeCL emits `#include <cuda_runtime.h>` and compiles via NVRTC at runtime; the
  split `cuda_nvcc`/`cuda_cudart` outputs lack that header and fail at first
  kernel launch, with nothing at build time to warn you.
- WSL's NVIDIA driver lives at `/usr/lib/wsl/lib`, which NixOS does not put on the
  linker path. The flake handles it.
- Burn's `HuggingfaceDatasetLoader` needs `.with_use_python_venv(false)` on NixOS,
  or it pip-installs wheels that cannot execute.
- Never run `cargo` bare — use `make`, or `./scripts/x cargo ...`.
- `make hooks` once per clone, or enforcement is silently off.

## Session hygiene

Start named so it is findable: `claude -n jane-orchestrator`. Sessions persist to
disk and survive a reboot; resume with `claude -c` (most recent here) or
`claude -r` (picker).

Two harness features were unavailable in the first session because it began before
the git repo and `.claude/agents/` existed: custom agent types could not be
selected, and built-in worktree isolation reported "not in a git repository". A
fresh session fixes both — so prefer `subagent_type: rust-impl` with
`isolation: "worktree"`, which restores Law 2's enforcement by tool allowlist.
