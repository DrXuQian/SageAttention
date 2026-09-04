#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk="${PPU_SDK:-${PPU_HOME:-}}"
if [[ -z "$sdk" || ! -x "$sdk/bin/hgcc" || ! -x "$sdk/bin/hgobjdump" ]]; then
  echo '[PPU Sage SDK] SKIP: set PPU_SDK to a real SDK with hgcc and hgobjdump' >&2
  exit 3
fi

sha="$(git -C "$repo" rev-parse HEAD)"
out="${SAGEATTENTION_PPU_SDK_OUT:-/workspace/sageattention-ppu-sdk-${sha:0:8}}"
mkdir -p "$out/build-temp" "$out/build-lib"
printf '[PPU Sage SDK] sha=%s sdk=%s out=%s\n' "$sha" "$sdk" "$out"

export PPU_SDK="$sdk"
export PATH="$sdk/bin:$PATH"
(
  cd "$repo"
  python setup_ppu.py build_ext --force \
    --build-temp "$out/build-temp" --build-lib "$out/build-lib"
) 2>&1 | tee "$out/build.log"

extension="$(find "$out/build-lib/sageattention" -maxdepth 1 -type f \
  -name '_qattn_ppu*.so' -print -quit)"
if [[ -z "$extension" ]]; then
  echo '[PPU Sage SDK] FAIL: native build produced no extension' >&2
  exit 1
fi

"$sdk/bin/hgobjdump" --list-all --demangle "$extension" >"$out/device-functions.log"
"$sdk/bin/hgobjdump" --dump-resource-usage=all "$extension" >"$out/resources.log"
"$sdk/bin/hgobjdump" --dump-isa "$extension" >"$out/shipping-isa.log"

attn_count="$(rg -c 'Func [0-9]+ \(kernel\): .*qk_int8_pv_f16_kernel' \
  "$out/device-functions.log")"
quant_count="$(rg -c 'Func [0-9]+ \(kernel\): .*quantize_int8_kernel' \
  "$out/device-functions.log")"
if [[ "$attn_count" -ne 16 || "$quant_count" -ne 12 ]]; then
  printf '[PPU Sage SDK] FAIL: device specialization census dense=%s/16 quant=%s/12\n' \
    "$attn_count" "$quant_count" >&2
  exit 1
fi
if rg -q 'block_sparse_kernel|radial_sparse_kernel' "$out/device-functions.log"; then
  echo '[PPU Sage SDK] FAIL: sparse device symbol remains in dense SageAttention' >&2
  exit 1
fi

python - "$out/resources.log" <<'PY'
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
vregs = [int(value) for value in re.findall(r"vreg_number:(\d+)", text)]
stacks = [int(value) for value in re.findall(r"STACK SIZE:(\d+)", text)]
if len(vregs) != 28 or len(stacks) != 28:
    raise SystemExit(
        f"[PPU Sage SDK] FAIL: resource census vregs={len(vregs)}/28 "
        f"stacks={len(stacks)}/28"
    )
private = [value for value in stacks if value]
if max(vregs) > 256 or private:
    raise SystemExit(
        f"[PPU Sage SDK] FAIL: max_vregs={max(vregs)} nonzero_stacks="
        f"{private}"
    )
print(
    f"[PPU Sage SDK] resources kernels=28 max_vregs={max(vregs)} "
    "spill_stack=0/PASS"
)
PY

# Compile the score-C -> PV-A bridge twice through the fixed actlize atom and
# the device-proven raw spelling, plus the exact historical reference-aliasing
# defect.  This is a generated-ISA gate: a host layout model cannot see whether
# the compiler actually initialized all four hardware accumulator registers.
probe_includes=(
  "-I$repo/csrc/qattn/ppu"
  "-I$repo/third_party/actlize/include"
  "-I$sdk/include"
)
for target_include in "$sdk"/targets/*/include; do
  probe_includes+=("-I$target_include")
done
"$sdk/bin/hgcc" \
  --forward-unknown-to-host-compiler --forward-unknown-to-host-linker \
  -arch=ppu_10 -x hg -DSWITCH_TO_HGGCRT \
  -Xcompiler -ftemplate-depth=8192 -Xllvm -ppu-max-vreg-count=256 \
  --expt-relaxed-constexpr -DUSE_CLANG -DCUTLASS_VERSIONS_GENERATED \
  -DCUTLASS_USE_PACKED_TUPLE=1 -DCUTE_USE_PACKED_TUPLE=1 \
  -DUSE_PPU=1 -DUSE_AIU=1 -O3 -std=c++17 --use_fast_math -fPIC \
  "${probe_includes[@]}" \
  -c "$repo/dev/ppu_int8/bridge_codegen_probe.cu" \
  -o "$out/bridge_codegen_probe.o" \
  >"$out/bridge-codegen-build.log" 2>&1
"$sdk/bin/hgobjdump" --dump-isa "$out/bridge_codegen_probe.o" \
  >"$out/bridge-codegen-isa.log"
python "$repo/dev/ppu_int8/check_bridge_codegen.py" \
  "$out/bridge-codegen-isa.log"
python "$repo/dev/ppu_int8/check_bridge_codegen.py" --shipping \
  "$out/shipping-isa.log"

sha256sum "$extension" | tee "$out/binary.sha256"
printf '[PPU Sage SDK] PASS: native -x hg build; dense=%s quant=%s; no device code executed\n' \
  "$attn_count" "$quant_count"
