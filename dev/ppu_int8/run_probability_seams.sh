#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk="${PPU_SDK:-}"
if [[ -z "$sdk" || ! -f "$sdk/include/hggc_runtime.h" ]]; then
  echo '[P seams] SKIP: real PPU SDK headers required; no fake runtime headers'
  exit 3
fi
out="${OUT:-/workspace/sageattention-probability-seams}"
mkdir -p "$out"
inc=("-I$repo/third_party/actlize/include" "-I$sdk/include")
for path in "$sdk"/targets/*/include; do
  [[ ! -d "$path" ]] || inc+=("-I$path")
done
for pair in 'probability_mask_oracle unmasked-origin' 'probability_pack_exhaustive missing-input'; do
  read -r source negative <<< "$pair"
  "${CXX:-c++}" -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math \
    "${inc[@]}" "$repo/dev/ppu_int8/$source.cpp" -o "$out/$source"
  "$out/$source" | tee "$out/$source.log"
  if "$out/$source" "--$negative" > "$out/$negative.log" 2>&1; then
    echo "[P seams] FAIL: $negative survived"
    exit 1
  fi
  echo "[P seams] $negative EXPECTED-RED/PASS"
done
echo '[P seams] host numerical proofs PASS; PPU execution NOT RUN'
