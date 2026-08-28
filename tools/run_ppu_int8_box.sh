#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk="${PPU_SDK:-${PPU_HOME:-}}"
if [[ -z "$sdk" || ! -x "$sdk/bin/hgcc" ]]; then
  echo '[PPU Sage box] FAIL: set PPU_SDK to an SDK containing bin/hgcc' >&2
  exit 1
fi
sha="$(git -C "$repo" rev-parse HEAD)"
out="${OUT:-/workspace/sageattention-ppu-int8-${sha:0:8}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out/build-temp"
printf '[PPU Sage box] sha=%s actlize=%s out=%s\n' \
  "$sha" "$(git -C "$repo/csrc/actlize" rev-parse HEAD)" "$out"

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

export PPU_SDK="$sdk"
export PATH="$sdk/bin:$PATH"
export PYTORCH_NVCC="$sdk/bin/hgcc"
(
  cd "$repo"
  python setup_ppu.py build_ext --inplace --force \
    --build-temp "$out/build-temp"
) 2>&1 | tee "$out/build.log"

extension="$(find "$repo/sageattention" -maxdepth 1 -type f \
  -name '_qattn_ppu*.so' -print -quit)"
if [[ -z "$extension" ]]; then
  echo '[PPU Sage box] FAIL: build produced no _qattn_ppu extension' >&2
  exit 1
fi
sha256sum "$extension" | tee "$out/binary.sha256"
(
  cd "$repo"
  python dev/ppu_int8/device_smoke.py
) 2>&1 | tee "$out/device-smoke.log"
echo "[PPU Sage box] PASS: artifacts=$out"
