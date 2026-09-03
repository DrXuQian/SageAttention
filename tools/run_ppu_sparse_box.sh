#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk="${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}"
if [[ ! -x "$sdk/bin/hgcc" || ! -x "$sdk/bin/hgobjdump" ]]; then
  echo "[PPU sparse box] FAIL: PPU SDK missing hgcc/hgobjdump at $sdk" >&2
  exit 1
fi
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sageattention-ppu-sparse-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out/build-temp"
printf '[PPU sparse box] sha=%s actlize=%s sdk=%s out=%s\n' \
  "$sha" "$(git -C "$repo/csrc/actlize" rev-parse HEAD)" "$sdk" "$out"

python "$repo/dev/ppu_sparse/test_plans.py" | tee "$out/host-plan-admission.log"

export PPU_SDK="$sdk"
export PATH="$sdk/bin:$PATH"
(
  cd "$repo"
  python setup_ppu.py build_ext --inplace --force --build-temp "$out/build-temp"
) 2>&1 | tee "$out/build.log"

extension="$(find "$repo/sageattention" -maxdepth 1 -type f \
  -name '_qattn_ppu*.so' -print -quit)"
if [[ -z "$extension" ]]; then
  echo '[PPU sparse box] FAIL: build produced no extension' >&2
  exit 1
fi
sha256sum "$extension" | tee "$out/binary.sha256"
"$sdk/bin/hgobjdump" --dump-resource-usage=all "$extension" \
  >"$out/resources.log"
python - "$out/resources.log" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
vregs = [int(x) for x in re.findall(r"vreg_number:(\d+)", text)]
stacks = [int(x) for x in re.findall(r"STACK SIZE:(\d+)", text)]
private = [x for x in stacks if x]
if len(vregs) != 32 or sorted(private) != [8, 8] or max(vregs) > 256:
    raise SystemExit(
        f"[PPU sparse box] FAIL: resource contract kernels={len(vregs)}/32 "
        f"max_vregs={max(vregs, default=-1)} private={private}"
    )
print(
    f"[PPU sparse box] resources kernels=32 max_vregs={max(vregs)} "
    "summary_private_frame=8B/REGISTERED"
)
PY

export LD_LIBRARY_PATH="$sdk/lib:${LD_LIBRARY_PATH:-}"
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
