#!/usr/bin/env bash
# Experimental source branch only. Local-built binary; no box compilation or
# pip installation. FP16-PV default remains in the user's installed package.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sage-int8-alu-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
runtime="${PPU_RUNTIME_DIR:-${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}/lib}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
python "$repo/tools/verify_ppu_prebuilt.py" --repo "$repo" \
  --prebuilt-root "$repo/prebuilt/ppu_10/alu-candidate" \
  --runtime-dir "$runtime" --json-out "$out/identity.json" \
  --artifact-path-out "$out/binary.path" 2>&1 | tee "$out/identity.log"
binary="$(<"$out/binary.path")"
python "$repo/tools/stage_ppu_candidate.py" --repo "$repo" \
  --manifest "$(dirname "$binary")/manifest.json" --out "$out/python"
export PYTHONPATH="$out/python"
export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /workspace
python "$repo/dev/ppu_int8/device_all_int8.py" 2>&1 | tee "$out/correctness.log"
if [[ "${PROFILE:-0}" == 1 ]]; then
  "${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}" -f \
    -o "$out/sage-h3-int8-alu" --set full \
    python "$repo/tools/profile_ppu_attention_pipes.py" \
      --arm sage --pv int8 --batch 1 --heads 56 --seq 73774 --head-dim 128 \
      --device "${DEVICE:-0}" --iters "${ITERS:-1}" \
      2>&1 | tee "$out/acu.log"
fi
printf '[PPU ALU candidate] PASS: artifacts=%s installed_default=UNCHANGED\n' "$out"
