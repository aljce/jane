#!/usr/bin/env bash
# Activate the enforcement hooks. Run once per clone: `make hooks`.
#
# Git will not use committed hooks on its own — core.hooksPath must be set. That
# setting lives in .git/config, which is SHARED by every linked worktree, so
# doing this once in the primary repo covers every agent worktree created later.
# An agent cannot opt out of hooks it never had to install.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

# ABSOLUTE, pointing at the primary worktree's copy. This matters more than it
# looks. With a relative ".githooks", every linked worktree resolves the path
# against its own checkout — so an agent runs the hooks from its own working tree
# and can neuter them just by editing the files there. No commit is involved, so
# the ownership hook never gets a chance to object.
#
# Pinning the absolute path means every worktree executes the orchestrator's
# copy, which lives outside any agent's checkout and is therefore not theirs to
# edit. It also means a worktree forked from a commit predating a hook still runs
# that hook.
git config core.hooksPath "$root/.githooks"
chmod +x .githooks/* 2>/dev/null || true

echo "core.hooksPath = $(git config --get core.hooksPath)"
echo
echo "active hooks:"
for h in .githooks/*; do
  [ -f "$h" ] || continue
  printf '  %-24s %s\n' "$(basename "$h")" "$([ -x "$h" ] && echo executable || echo 'NOT EXECUTABLE')"
done
echo
echo "reference-transaction is the un-bypassable gate: --no-verify does not skip it."
echo "pre-commit enforces agent-orchestrator/ownership; skippable with --no-verify."
