#!/usr/bin/env bash
# Same test/profile driver, immutable a49338f binary vs deferred denominator.
# Execution only. The two Python extensions never coexist in one process.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sage-int8-denominator-ab-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
before="$out/before-source"
if [[ -e "$before" ]]; then
  echo '[denominator A/B] FAIL: use a new OUT; before-source already exists' >&2
  exit 1
fi
# This commit packages the exact clean a49338f binary; it does not rebuild it.
baseline=97ea065672d3dd5e69c3f568590d9c149bb2c00e
git -C "$repo" cat-file -e "$baseline^{commit}"
git -C "$repo" worktree add --detach "$before" "$baseline"
printf '[denominator A/B] before_package=%s after=%s out=%s\n' "$baseline" "$sha" "$out"
OUT="$out/before" CANDIDATE=alu-candidate CANDIDATE_SOURCE_REPO="$before" \
  PROFILE="${PROFILE:-1}" bash "$repo/tools/run_ppu_int8_alu_candidate_box.sh"
OUT="$out/after" CANDIDATE=deferred-denominator CANDIDATE_SOURCE_REPO="$repo" \
  PROFILE="${PROFILE:-1}" bash "$repo/tools/run_ppu_int8_alu_candidate_box.sh"
printf '[denominator A/B] PASS: sequential before/after; reports under %s; installed_default=UNCHANGED\n' "$out"
