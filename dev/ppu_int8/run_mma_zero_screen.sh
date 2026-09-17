#!/usr/bin/env bash
# Local compiler screen only. No GPU workload and no change to a kernel.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk="${PPU_SDK:-${PPU_HOME:-}}"
if [[ -z "$sdk" || ! -x "$sdk/bin/hgcc" || ! -x "$sdk/bin/hgobjdump" ]]; then
  echo '[MMA zero screen] SKIP: real PPU SDK compiler/disassembler required'
  exit 3
fi
out="${OUT:-/workspace/sage-mma-zero-screen}"
shipping="${SHIPPING_ISA:-}"
if [[ -z "$shipping" || ! -f "$shipping" ]]; then
  echo '[MMA zero screen] FAIL: SHIPPING_ISA must identify the clean711ec99 native body'
  exit 1
fi
mkdir -p "$out/compiler-tmp"
out="$(cd "$out" && pwd)"
export TMPDIR="$out/compiler-tmp"
flags=(--forward-unknown-to-host-compiler --forward-unknown-to-host-linker
  -arch=ppu_10 -x hg -DSWITCH_TO_HGGCRT -Xcompiler -ftemplate-depth=8192
  -Xllvm -wno-loop-miss-transform -Xllvm -ppu-simt-branch=false
  -Xllvm -ppu-patch-fence-ppu=false -Xllvm -ppu-cg-to-kp1=true
  -Xllvm -ppu-fix-uninit=true -Xllvm -ppu-max-vreg-count=256
  -Xllvm -ppu-sink-matrix-addr=true -Xllvm -ppu-sink-async-addr=true
  -Xllvm -ppu-sink-load-addr=true -Xllvm -ppu-sink-store-addr=true
  --expt-relaxed-constexpr -DUSE_CLANG
  -DCUTLASS_VERSIONS_GENERATED -DCUTLASS_USE_PACKED_TUPLE=1 -DCUTE_USE_PACKED_TUPLE=1
  -DUSE_PPU=1 -DUSE_AIU=1 -O3 -std=c++17 --use_fast_math -fPIC
  "-I$repo/csrc/qattn/ppu" "-I$repo/third_party/actlize/include" "-I$sdk/include")
for inc in "$sdk"/targets/*/include; do flags+=("-I$inc"); done
for variant in ordinary tuple; do
  extra=()
  if [[ "$variant" == tuple ]]; then extra+=(-DPROBE_LITERAL_TUPLE=1); fi
  "$sdk/bin/hgcc" "${flags[@]}" "${extra[@]}" \
    -c "$repo/dev/ppu_int8/mma_zero_codegen_probe.cu" -o "$out/$variant.o" \
    >"$out/$variant-build.log" 2>&1
  "$sdk/bin/hgobjdump" --dump-isa "$out/$variant.o" >"$out/$variant-isa.log"
  "$sdk/bin/hgobjdump" --dump-resource-usage=all "$out/$variant.o" >"$out/$variant-resources.log"
done
if "$sdk/bin/hgcc" "${flags[@]}" -DPROBE_LITERAL_SCALAR=1 \
    -c "$repo/dev/ppu_int8/mma_zero_codegen_probe.cu" -o "$out/scalar.o" \
    >"$out/scalar-build.log" 2>&1; then
  echo '[MMA zero screen] FAIL: scalar spelling now compiles; re-evaluate the prior toolchain boundary'
  exit 1
fi
if ! grep -q 'Call parameter type does not match function signature' "$out/scalar-build.log"; then
  echo '[MMA zero screen] FAIL: scalar spelling failed for an unexpected cause'
  exit 1
fi
python - "$out" <<'PY'
from pathlib import Path
import re,sys
root=Path(sys.argv[1])
for name,want in (("ordinary",9),("tuple",2)):
    t=(root/f"{name}-resources.log").read_text()
    stacks=[int(x) for x in re.findall(r"STACK SIZE:(\d+)",t)]
    if len(stacks)!=want or any(stacks):
        raise SystemExit("resource census/spill failure: "+name)
print('[MMA zero screen] resources9+2 no private stack; scalar form compiler EXPECTED-RED')
PY
common=(--probes "$out/ordinary-isa.log" --literal "$out/tuple-isa.log" --shipping "$shipping")
python "$repo/dev/ppu_int8/check_mma_zero_codegen.py" "${common[@]}" --out "$out/account.json"
for plant in missing-type missing-clear missing-mma shipping-clear-missing; do
  if python "$repo/dev/ppu_int8/check_mma_zero_codegen.py" "${common[@]}" \
      --plant "$plant" >"$out/$plant.log" 2>&1; then
    echo "[MMA zero screen] FAIL: negative survived: $plant"
    exit 1
  fi
  grep -q '\[MMA zero screen\] FAIL:' "$out/$plant.log"
done
printf '[MMA zero screen] PASS: compiler evidence only; artifacts=%s\n' "$out"
