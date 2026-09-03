#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sageattention-ppu-sparse-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
prebuilt_dir="$repo/prebuilt/ppu_10/cpython312-torch2.8-cxx11abi1"
manifest="$prebuilt_dir/manifest.json"
extension="$prebuilt_dir/_qattn_ppu.cpython-312-x86_64-linux-gnu.so"
runtime_dir="${PPU_RUNTIME_DIR:-${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}/lib}"
mkdir -p "$out"

printf '[PPU sparse box] mode=EXECUTION-ONLY sha=%s actlize=%s runtime=%s out=%s\n' \
  "$sha" "$(git -C "$repo" rev-parse HEAD:third_party/actlize)" "$runtime_dir" "$out"

python "$repo/tools/verify_ppu_prebuilt.py" \
  --repo "$repo" --manifest "$manifest" --artifact "$extension" \
  --runtime-dir "$runtime_dir" \
  --json-out "$out/prebuilt-identity.json" \
  2>&1 | tee "$out/prebuilt-identity.log"

cp -f "$extension" "$repo/sageattention/"
sha256sum "$extension" | tee "$out/binary.sha256"
python "$repo/dev/ppu_sparse/test_plans.py" | tee "$out/host-plan-admission.log"

export LD_LIBRARY_PATH="$runtime_dir:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo"
python "$repo/dev/ppu_int8/device_smoke.py" 2>&1 | tee "$out/dense-device-anchor.log"
python "$repo/dev/ppu_sparse/device_smoke.py" 2>&1 | tee "$out/device-smoke.log"

if [[ "${RUN_PERF:-1}" == 1 ]]; then
  python "$repo/dev/ppu_sparse/device_perf.py" \
    --batch "${BATCH:-1}" \
    --heads "${HEADS:-16}" \
    --kv-heads "${KV_HEADS:-16}" \
    --seq "${SEQ:-4096}" \
    --top-k "${TOP_K:-8}" \
    --tau "${TAU:-1.0}" \
    --warmup "${WARMUP:-3}" \
    --samples "${SAMPLES:-7}" \
    --launches "${LAUNCHES:-10}" \
    2>&1 | tee "$out/device-perf.log"
fi

printf '[PPU sparse box] PASS: artifacts=%s\n' "$out"
