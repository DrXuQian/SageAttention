#!/usr/bin/env bash
# Experimental source branch only. Local-built binary; no box compilation or
# pip installation. FP16-PV default remains in the user's installed package.
set -euo pipefail
runner_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "${CANDIDATE_SOURCE_REPO:-$runner_repo}" && pwd)"
candidate="${CANDIDATE:-alu-candidate}"
case "$candidate" in
  alu-candidate|deferred-denominator|permuted-key) ;;
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
numeric_args=()
bench_args=()
profile_args=()
if [[ "$candidate" == permuted-key ]]; then
  numeric_args+=(--key-layout permuted)
  bench_args+=(--permuted-k)
  profile_args+=(--key-layout permuted)
fi
python "$runner_repo/dev/ppu_int8/device_all_int8.py" "${numeric_args[@]}" 2>&1 | tee "$out/correctness.log"
if [[ "${BENCHMARK:-1}" == 1 ]]; then
  # Normal events in a separate, unprofiled process. Both PV arms use identical
  # prequantized Q/K and the same original V; no preparation inside the timer.
  python "$runner_repo/tools/benchmark_ppu_pv_core.py" \
    --batch 1 --heads 56 --seq 73774 --head-dim 128 --device "${DEVICE:-0}" \
    --warmup "${WARMUP:-2}" --samples "${SAMPLES:-7}" --launches "${LAUNCHES:-1}" \
    --out "$out/pv-core-events.json" "${bench_args[@]}" 2>&1 | tee "$out/pv-core-events.log"
fi
if [[ "${PROFILE:-0}" == 1 ]]; then
  "${ACU:-/sim/eec/shared/junfu.qx/asight/bin/acu}" -f \
    -o "$out/sage-h3-int8-$candidate" --set full \
    python "$runner_repo/tools/profile_ppu_attention_pipes.py" \
      --arm sage --pv int8 --batch 1 --heads 56 --seq 73774 --head-dim 128 \
      --device "${DEVICE:-0}" --iters "${ITERS:-1}" "${profile_args[@]}" \
      2>&1 | tee "$out/acu.log"
fi
printf '[PPU ALU candidate] PASS: artifacts=%s installed_default=UNCHANGED\n' "$out"
