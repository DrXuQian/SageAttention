#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sageattention-ppu-int8-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
prebuilt_dir="$repo/prebuilt/ppu_10/cpython312-torch2.8-cxx11abi1"
manifest="$prebuilt_dir/manifest.json"
extension="$prebuilt_dir/_qattn_ppu.cpython-312-x86_64-linux-gnu.so"
runtime_dir="${PPU_RUNTIME_DIR:-${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}/lib}"
mkdir -p "$out"
printf '[PPU Sage box] mode=EXECUTION-ONLY sha=%s actlize=%s runtime=%s out=%s\n' \
  "$sha" "$(git -C "$repo" rev-parse HEAD:third_party/actlize)" \
  "$runtime_dir" "$out"

python "$repo/tools/verify_ppu_prebuilt.py" \
  --repo "$repo" --manifest "$manifest" --artifact "$extension" \
  --runtime-dir "$runtime_dir" \
  --json-out "$out/prebuilt-identity.json" \
  2>&1 | tee "$out/prebuilt-identity.log"

# The host CuTe proof is generated on a complete NVIDIA CUDA toolchain.  The
# PPU box consumes evidence from this exact result SHA; it does not rerun the
# oracle with a compiler whose nvcc wrapper delegates to ppu_clang++.
git -C "$repo" show \
  "$sha:dev/ppu_int8/layout_oracle.expected.txt" \
  >"$out/layout-oracle.committed.txt"
grep -Fqx '[PPU Sage layout] c_formula_bad=0/512 transposed_formula_bad=240/256 row_peer_bad=0 bridge_bad=0/256 old_order_bad=240/256' \
  "$out/layout-oracle.committed.txt"
grep -Fqx '[PPU Sage layout] PASS: real traits + row peers + C-to-A bridge' \
  "$out/layout-oracle.committed.txt"
printf '[PPU Sage box] host_layout_evidence=COMMITTED/PASS fresh_box_execution=0\n'

cp -f "$extension" "$repo/sageattention/"
sha256sum "$extension" | tee "$out/binary.sha256"
export LD_LIBRARY_PATH="$runtime_dir:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo"
(
  cd "$repo"
  python dev/ppu_int8/device_smoke.py
) 2>&1 | tee "$out/device-smoke.log"
echo "[PPU Sage box] PASS: artifacts=$out"
