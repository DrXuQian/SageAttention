#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sageattention-ppu-perf-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
artifact="$repo/prebuilt/ppu_10/cpython312-torch2.8-cxx11abi1/_qattn_ppu.cpython-312-x86_64-linux-gnu.so"
runtime_dir="${PPU_SDK:-/usr/local/PPU_SDK}/lib"
mkdir -p "$out"

if [[ ! -f "$artifact" ]]; then
  echo "[PPU Sage perf runner] FAIL: prebuilt extension missing: $artifact" >&2
  exit 1
fi
cp -f "$artifact" "$repo/sageattention/"
sha256sum "$artifact" | tee "$out/binary.sha256"
printf '[PPU Sage perf runner] sha=%s out=%s runtime=%s\n' \
  "$sha" "$out" "$runtime_dir"

export LD_LIBRARY_PATH="$runtime_dir:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo"
common=(
  --batch "${BATCH:-1}"
  --heads "${HEADS:-16}"
  --kv-heads "${KV_HEADS:-16}"
  --seq "${SEQ:-4096}"
  --head-dim "${HEAD_DIM:-128}"
  --warmup "${WARMUP:-5}"
  --samples "${SAMPLES:-7}"
  --launches "${LAUNCHES:-20}"
  --peak-tflops "${PPU_PEAK_TFLOPS:-500}"
)

python "$repo/dev/ppu_int8/device_perf.py" "${common[@]}" \
  2>&1 | tee "$out/noncausal.log"
if [[ "${RUN_CAUSAL:-1}" == 1 ]]; then
  python "$repo/dev/ppu_int8/device_perf.py" "${common[@]}" --causal \
    2>&1 | tee "$out/causal.log"
fi
echo "[PPU Sage perf runner] PASS: artifacts=$out"
