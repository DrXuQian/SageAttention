#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${SAGEATTENTION_PPU_SPARSE_OUT:-/workspace/sageattention-ppu-sparse-local}"
mkdir -p "$out"

python "$repo/dev/ppu_sparse/test_plans.py" | tee "$out/plan-oracle.log"
python -m py_compile "$repo"/sageattention/ppu_sparse/*.py

printf '[PPU sparse local] PASS: planner/oracle admission; PPU source admission follows after device ABI lands\n'
