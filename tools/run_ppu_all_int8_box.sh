#!/usr/bin/env bash
# Execute the installed wheel. No compiler or source-tree .so copying.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${OUT:-/workspace/sageattention-all-int8-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$out"
if [[ -n "${PPU_SDK:-}" ]]; then
  export LD_LIBRARY_PATH="$PPU_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH=""
cd /workspace
python "$repo/dev/ppu_int8/device_all_int8.py" 2>&1 | tee "$out/correctness.log"
if [[ "${PROFILE:-0}" == 1 ]]; then
  "${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}" \
    -f -o "$out/sage-h3-int8" --set full \
    python "$repo/tools/profile_ppu_attention_pipes.py" \
      --arm sage --pv int8 --batch 1 --heads 56 --seq 73774 --head-dim 128 \
      --device "${DEVICE:-0}" --iters "${ITERS:-1}" 2>&1 | tee "$out/acu.log"
fi
echo "[all-int8 box] artifacts: $out"
