#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sageattention-ppu-sparse-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
prebuilt_root="$repo/prebuilt/ppu_10"
artifact_path_file="$out/prebuilt-artifact.path"
runtime_dir="${PPU_RUNTIME_DIR:-${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}/lib}"
mkdir -p "$out"

printf '[PPU sparse box] mode=EXECUTION-ONLY sha=%s actlize=%s runtime=%s out=%s\n' \
  "$sha" "$(git -C "$repo" rev-parse HEAD:third_party/actlize)" "$runtime_dir" "$out"

python "$repo/tools/verify_ppu_prebuilt.py" \
  --repo "$repo" --prebuilt-root "$prebuilt_root" \
  --runtime-dir "$runtime_dir" \
  --json-out "$out/prebuilt-identity.json" \
  --artifact-path-out "$artifact_path_file" \
  2>&1 | tee "$out/prebuilt-identity.log"

extension="$(<"$artifact_path_file")"
cp -f "$extension" "$repo/sageattention/"
sha256sum "$extension" | tee "$out/binary.sha256"
python "$repo/dev/ppu_sparse/test_plans.py" | tee "$out/host-plan-admission.log"

export LD_LIBRARY_PATH="$runtime_dir:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo"
python "$repo/dev/ppu_int8/device_smoke.py" 2>&1 | tee "$out/dense-device-anchor.log"
python "$repo/dev/ppu_sparse/device_smoke.py" 2>&1 | tee "$out/device-smoke.log"

if [[ "${RUN_PERF:-1}" == 1 ]]; then
  perf_args=(
    --batch "${BATCH:-1}"
    --heads "${HEADS:-16}"
    --kv-heads "${KV_HEADS:-16}"
    --seq "${SEQ:-4096}"
    --top-k "${TOP_K:-8}"
    --tau "${TAU:-1.0}"
    --warmup "${WARMUP:-3}"
    --samples "${SAMPLES:-7}"
    --launches "${LAUNCHES:-10}"
  )
  if [[ -n "${RADIAL_MASK:-}" ]]; then
    perf_args+=(--radial-mask "$RADIAL_MASK"
      --radial-mask-kind "${RADIAL_MASK_KIND:-compute}")
  fi
  python "$repo/dev/ppu_sparse/device_perf.py" "${perf_args[@]}" \
    2>&1 | tee "$out/device-perf.log"
fi

printf '[PPU sparse box] PASS: artifacts=%s\n' "$out"
