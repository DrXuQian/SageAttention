#!/usr/bin/env bash
# Experimental source branch only. Local-built binary; no box compilation or
# pip installation. FP16-PV default remains in the user's installed package.
set -euo pipefail
runner_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "${CANDIDATE_SOURCE_REPO:-$runner_repo}" && pwd)"
candidate="${CANDIDATE:-alu-candidate}"
case "$candidate" in
  alu-candidate|deferred-denominator) ;;
  *) echo "[PPU ALU candidate] FAIL: unknown candidate $candidate" >&2; exit 1 ;;
esac
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sage-int8-${candidate}-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
runtime="${PPU_RUNTIME_DIR:-${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}/lib}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
printf '[PPU ALU candidate] role=%s source=%s sha=%s test_authority=%s\n' "$candidate" "$repo" "$sha" "$runner_repo"
python "$runner_repo/tools/verify_ppu_prebuilt.py" --repo "$repo" \
  --prebuilt-root "$repo/prebuilt/ppu_10/$candidate" \
  --runtime-dir "$runtime" --json-out "$out/identity.json" \
  --artifact-path-out "$out/binary.path" 2>&1 | tee "$out/identity.log"
binary="$(<"$out/binary.path")"
python "$runner_repo/tools/stage_ppu_candidate.py" --repo "$repo" \
  --manifest "$(dirname "$binary")/manifest.json" --out "$out/python"
export PYTHONPATH="$out/python"
export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /workspace
python - "$out/python" <<'PY'
from pathlib import Path
import sys
from sageattention import ppu_compile
actual = Path(ppu_compile._qattn_ppu.__file__).resolve()
expected = (Path(sys.argv[1]) / "sageattention").resolve()
if actual.parent != expected:
    raise RuntimeError(f"staged candidate was shadowed: {actual}; expected {expected}")
print(f"[PPU ALU candidate] loaded={actual}")
PY
python "$runner_repo/dev/ppu_int8/device_all_int8.py" 2>&1 | tee "$out/correctness.log"
if [[ "${PROFILE:-0}" == 1 ]]; then
  "${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}" -f \
    -o "$out/sage-h3-int8-$candidate" --set full \
    python "$runner_repo/tools/profile_ppu_attention_pipes.py" \
      --arm sage --pv int8 --batch 1 --heads 56 --seq 73774 --head-dim 128 \
      --device "${DEVICE:-0}" --iters "${ITERS:-1}" \
      2>&1 | tee "$out/acu.log"
fi
printf '[PPU ALU candidate] PASS: artifacts=%s installed_default=UNCHANGED\n' "$out"
