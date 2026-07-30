#!/usr/bin/env bash
# Show the flight log and reconcile it against live git state.
#
# The log records *intent*; git records *reality*. A stale log is worse than no
# log — it lies with authority — so this diffs the two and complains. Run via
# `make status`.
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"
flight="agent-orchestrator/flight.md"
ownership="agent-orchestrator/ownership"

bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; yellow=$'\033[33m'
green=$'\033[32m'; reset=$'\033[0m'

[ -f "$flight" ] || { echo "no $flight"; exit 1; }

# --- the table ---------------------------------------------------------------
echo "${bold}Lanes${reset}"
awk -F'|' '
  /lanes:begin/ { inlanes = 1; next }
  /lanes:end/   { inlanes = 0 }
  inlanes && /^\| *`/ {
    gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3)
    gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $6)
    gsub(/`/, "", $2)
    printf "  %-14s %-18s %-22s %s\n", $2, $3, $4, $6
  }
' "$flight"
echo

# --- reconcile ---------------------------------------------------------------
# Lanes declared in the ownership manifest (the authority on what may exist).
declared="$(sed -e 's/#.*//' "$ownership" | grep -oE '^agent/[a-z0-9-]+' | sed 's|^agent/||' | sort -u)"

# Lanes appearing in the flight table.
logged="$(awk -F'|' '/lanes:begin/{i=1;next} /lanes:end/{i=0} i && /^\| *`/ { gsub(/[ `]/, "", $2); print $2 }' "$flight" | sort -u)"

# Live git state.
branches="$(git branch --list 'agent/*' --format='%(refname:short)' | sed 's|^agent/||' | sort -u)"
# Ancestry is the wrong question. `git branch --merged` calls a lane merged
# whenever its tip is an ancestor of master — which is true both for a lane whose
# work landed AND for one that never committed anything and has simply fallen
# behind. Excluding tip==master patched the freshly-created case but broke again
# the moment master moved ahead on its own.
#
# What is always computable and always means something is the count of commits a
# lane holds that master does not:
#   ahead == 0  ->  nothing outstanding (either merged, or never started)
#   ahead  > 0  ->  unmerged work sitting on that branch
# The status checks below compare that against what the log claims.
ahead_of_master() {
  git rev-list --count "master..refs/heads/agent/$1" 2>/dev/null || echo 0
}
unmerged=""
for b in $branches; do
  [ "$(ahead_of_master "$b")" -gt 0 ] && unmerged="$unmerged$b"$'\n'
done
unmerged="$(printf '%s' "$unmerged" | sed '/^$/d' | sort -u)"
worktrees="$(git worktree list --porcelain | awk '/^branch refs\/heads\/agent\//{sub("branch refs/heads/agent/","");print}' | sort -u)"

problems=0
warn() { printf '%s  %s%s\n' "$1" "$2" "$reset"; problems=$((problems + 1)); }

# A lane in the log that nobody declared cannot be committed to — the pre-commit
# hook rejects an undeclared branch outright.
for l in $logged; do
  grep -qx "$l" <<<"$declared" || warn "$red" "logged but NOT in $ownership: '$l' — that lane cannot commit"
done

# A declared lane missing from the log is invisible work waiting to happen.
for l in $declared; do
  grep -qx "$l" <<<"$logged" || warn "$yellow" "declared in $ownership but not in the flight table: '$l'"
done

# A branch with no log entry is the dangerous one: work exists that the
# orchestrator has no record of.
for b in $branches; do
  grep -qx "$b" <<<"$logged" || warn "$red" "branch agent/$b exists but is not in the flight log"
done

# Status vs. reality.
while IFS='|' read -r _ lane status _; do
  lane="${lane//[\` ]/}"; status="$(tr -d ' ' <<<"$status")"
  [ -n "$lane" ] || continue
  case "$status" in
    merged)
      grep -qx "$lane" <<<"$unmerged" &&
        warn "$red" "'$lane' is logged merged but still holds commits master does not"
      ;;
    awaiting-review | in-review | changes-requested)
      # These all assert the agent finished and committed. Zero commits means
      # either the log is ahead of reality or the agent reported work it never
      # committed — both worth stopping for.
      grep -qx "$lane" <<<"$unmerged" ||
        warn "$red" "'$lane' is '$status' but agent/$lane has no commits of its own"
      ;;
    in-flight | blocked)
      grep -qx "$lane" <<<"$branches" ||
        warn "$yellow" "'$lane' is '$status' but branch agent/$lane does not exist yet"
      ;;
    queued)
      # A worktree may legitimately be pre-created before a lane starts, so the
      # branch existing proves nothing. Commits on it do.
      grep -qx "$lane" <<<"$unmerged" &&
        warn "$yellow" "'$lane' is 'queued' but agent/$lane already has commits"
      ;;
  esac
done < <(awk -F'|' '/lanes:begin/{i=1;next} /lanes:end/{i=0} i && /^\| *`/ {print "|" $2 "|" $3 "|"}' "$flight")

echo "${bold}Live git${reset}"
printf '  %-12s %s\n' "branches:"  "${branches//$'\n'/ }"
printf '  %-12s %s\n' "worktrees:" "${worktrees:+${worktrees//$'\n'/ }}"
printf "  %-12s %s\n" "unmerged:"  "${unmerged:+${unmerged//$'\n'/ }}"
echo

if [ "$problems" -eq 0 ]; then
  echo "${green}flight log agrees with git${reset}"
else
  echo "${dim}$problems discrepancy(ies) — the log is intent, git is truth${reset}"
fi
