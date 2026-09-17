#!/usr/bin/env bash
# Opt-in only: same DSO contains unchanged raw INT8/FP16 controls. No build or
# install. Numeric gate precedes normal event timing; optional ACU is separate.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE=permuted-key bash "$repo/tools/run_ppu_int8_alu_candidate_box.sh"
