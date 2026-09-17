#!/usr/bin/env bash
# Host-only execution of real SDK/CuTe traits; no NVIDIA replacement headers.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sdk="${PPU_SDK:-}"
out="${OUT:-/workspace/sageattention-all-int8-layout}"
if [[ -z "$sdk" || ! -x "$sdk/bin/hgcc" ]]; then
  echo '[all-int8 layout] SKIP: PPU_SDK with hgcc is required' >&2
  exit 3
fi
mkdir -p "$out"
includes=("-I$repo/third_party/actlize/include" "-I$sdk/include")
for target_include in "$sdk"/targets/*/include; do
  if [[ -d "$target_include" ]]; then includes+=("-I$target_include"); fi
done
"$sdk/bin/hgcc" -arch=ppu_10 -x hg -DSWITCH_TO_HGGCRT -std=c++17 -O2 \
  "${includes[@]}" -c "$repo/dev/ppu_int8/all_int8_layout_oracle.cu" \
  -o "$out/oracle.o" >"$out/compile.log" 2>&1
# On older build hosts the SDK DSOs require a newer glibc. Allow deferred DSO
# symbol resolution only at this host-oracle link, then actually execute the
# oracle with the selected loader; unresolved symbols cannot become a PASS.
"${CXX:-c++}" "$out/oracle.o" -L"$sdk/lib" -Wl,-rpath-link,"$sdk/lib" \
  -Wl,--allow-shlib-undefined -lhggcrt1 -lhggc -lalippu -o "$out/oracle"
if [[ -n "${PPU_HOST_LOADER:-}" ]]; then
  : "${PPU_HOST_LIBRARY_PATH:?set the library path for PPU_HOST_LOADER}"
  "$PPU_HOST_LOADER" --library-path "$PPU_HOST_LIBRARY_PATH" "$out/oracle" \
    | tee "$out/result.log"
else
  LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$out/oracle" | tee "$out/result.log"
fi
diff -u "$repo/dev/ppu_int8/all_int8_layout_oracle.expected.txt" "$out/result.log"
echo '[all-int8 layout] native SDK host oracle executed; device execution NOT RUN'
