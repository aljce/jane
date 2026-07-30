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

git config core.hooksPath .githooks
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
echo "pre-commit enforces .jane/ownership and IS skippable with --no-verify."
