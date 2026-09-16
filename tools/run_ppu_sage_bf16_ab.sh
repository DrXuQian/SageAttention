#!/usr/bin/env bash
# Installed post1 Sage wheel + installed FA2; no build or in-place .so copy.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sdk="${PPU_SDK:-/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK}"
out="${OUT:-/workspace/sage-bf16-ab-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$out"
export LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Use an installed Sage wheel, never an old in-place .so from the source repo.
# If FA2 is only in a prior build, explicitly supply its runtime + Python paths.
export PYTHONPATH="${FA_PYTHONPATH:-}"
cd /workspace
args=(--batch "${BATCH:-1}" --heads "${HEADS:-16}" --seq "${SEQ:-4096}"
      --head-dim "${HEAD_DIM:-128}" --warmup "${WARMUP:-3}" --samples "${SAMPLES:-7}"
      --launches "${LAUNCHES:-20}" --out "$out/result.json")
if [[ "${CAUSAL:-0}" == 1 ]]; then args+=(--causal); fi
python "$script_dir/benchmark_ppu_sage_bf16.py" "${args[@]}" "$@" 2>&1 | tee "$out/run.log"
echo "[Sage/BF16] artifacts: $out"
