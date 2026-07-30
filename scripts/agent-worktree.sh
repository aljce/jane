#!/usr/bin/env bash
# Manage git worktrees for parallel agent work.
#
# Each agent gets its own checkout at ../jane-worktrees/<name> on branch
# agent/<name>. Separate checkouts mean separate `target/` dirs, so no two
# agents ever contend on cargo's build-directory lock. The expensive part of
# compilation is still shared via sccache (see .cargo/config.toml).
#
#   scripts/agent-worktree.sh create model-config [base-ref]
#   scripts/agent-worktree.sh list
#   scripts/agent-worktree.sh stats
#   scripts/agent-worktree.sh remove model-config
#   scripts/agent-worktree.sh clean
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
wt_root="$(dirname "$root")/$(basename "$root")-worktrees"

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \?//'
  exit 64
}

cmd_create() {
  local name="${1:-}" base="${2:-HEAD}"
  [ -n "$name" ] || usage
  local dir="$wt_root/$name" branch="agent/$name"

  if [ -e "$dir" ]; then
    echo "worktree already exists: $dir" >&2
    exit 1
  fi

  mkdir -p "$wt_root"

  # Reuse the branch if it already exists, otherwise fork it from base.
  if git -C "$root" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$root" worktree add "$dir" "$branch"
  else
    git -C "$root" worktree add -b "$branch" "$dir" "$base"
  fi

  # direnv is per-directory and must be trusted again in each new worktree.
  if command -v direnv >/dev/null 2>&1; then
    (cd "$dir" && direnv allow) || true
  fi

  echo
  echo "worktree : $dir"
  echo "branch   : $branch (from $base)"
  echo "build    : cd $dir && ./scripts/x cargo test --workspace"
}

cmd_list() {
  git -C "$root" worktree list
}

cmd_stats() {
  # Cache hit rate is the number that tells you whether the harness is working.
  ./scripts/x sccache --show-stats 2>/dev/null ||
    nix develop "$root" --command sccache --show-stats
}

cmd_remove() {
  local name="${1:-}"
  [ -n "$name" ] || usage
  git -C "$root" worktree remove --force "$wt_root/$name"
  echo "removed worktree $name (branch agent/$name kept — delete with: git branch -D agent/$name)"
}

cmd_clean() {
  git -C "$root" worktree prune
  echo "pruned stale worktree metadata"
  git -C "$root" worktree list
}

case "${1:-}" in
  create) shift; cmd_create "$@" ;;
  list)   shift; cmd_list ;;
  stats)  shift; cmd_stats ;;
  remove) shift; cmd_remove "$@" ;;
  clean)  shift; cmd_clean ;;
  *)      usage ;;
esac
