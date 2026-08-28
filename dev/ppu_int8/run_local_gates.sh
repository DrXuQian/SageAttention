#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${SAGEATTENTION_PPU_ORACLE_OUT:-/workspace/sageattention-ppu-local}"
mkdir -p "$out"

nvcc_bin="${NVCC:-nvcc}"
probe_log="$out/compiler-probe.log"
if ! "$nvcc_bin" -std=c++17 -arch=sm_80 \
    -o "$out/compiler-probe" "$repo/dev/ppu_int8/compiler_probe.cu" \
    >"$probe_log" 2>&1; then
  printf '[PPU Sage local] SKIP: NVIDIA host-oracle compiler capability unavailable\n'
  sed -n '1,20p' "$probe_log"
  exit 3
fi

base=("$nvcc_bin" -std=c++17 -arch=sm_80 -x cu
      -I "$repo/dev/ppu_int8/stub_inc"
      -I "$repo/csrc/actlize/include")
"${base[@]}" -o "$out/layout-oracle" "$repo/dev/ppu_int8/layout_oracle.cu"
"$out/layout-oracle" | tee "$out/layout-oracle.log"
diff -u "$repo/dev/ppu_int8/layout_oracle.expected.txt" \
  "$out/layout-oracle.log"

"${base[@]}" -DSAGE_PPU_BREAK_BRIDGE_BIT=1 \
  -o "$out/layout-oracle-bad-bridge" "$repo/dev/ppu_int8/layout_oracle.cu"
if "$out/layout-oracle-bad-bridge" >"$out/layout-oracle-bad-bridge.log" 2>&1; then
  echo '[PPU Sage local] FAIL: bridge-bit negative did not turn red' >&2
  exit 1
fi
grep -q 'EXPECTED-RED' "$out/layout-oracle-bad-bridge.log"

python "$repo/dev/ppu_int8/check_ppu_port.py"
for plant in bridge-order v-write row-formula scheduler; do
  if python "$repo/dev/ppu_int8/check_ppu_port.py" --plant "$plant" \
      >"$out/source-$plant.log" 2>&1; then
    echo "[PPU Sage local] FAIL: source plant $plant did not turn red" >&2
    exit 1
  fi
done

python -m py_compile \
  "$repo/setup_ppu.py" \
  "$repo/sageattention/core.py" \
  "$repo/sageattention/ppu_compile.py" \
  "$repo/sageattention/quant.py" \
  "$repo/sageattention/__init__.py"

# NVIDIA nvcc cannot compile several host-marked functions in the PPU actlize
# fork.  Compile the same header boundary and the instantiated shipping TU, and
# require the latter to add no diagnostics.  A real hgcc build remains the PPU
# device admission and is never replaced by this differential gate.
mapfile -t torch_includes < <(python - <<'PY'
from torch.utils.cpp_extension import include_paths, CUDA_HOME
import sysconfig
for path in include_paths(): print(path)
print(sysconfig.get_paths()["include"])
print(f"{CUDA_HOME}/include")
PY
)
front_inc=(-I "$repo/dev/ppu_int8/stub_inc"
           -I "$repo/csrc/actlize/include"
           -I "$repo/csrc/qattn/ppu")
for path in "${torch_includes[@]}"; do front_inc+=(-I "$path"); done
front=("$nvcc_bin" -std=c++17 -arch=sm_80 --expt-relaxed-constexpr
       --expt-extended-lambda -D_GLIBCXX_USE_CXX11_ABI=1 "${front_inc[@]}" -c)
"${front[@]}" "$repo/dev/ppu_int8/front_end_baseline.cu" \
  -o "$out/front-end-baseline.o" >"$out/front-end-baseline.log" 2>&1 || true
"${front[@]}" "$repo/csrc/qattn/ppu/qk_int_sv_f16_ppu.cu" \
  -o "$out/front-end-shipping.o" >"$out/front-end-shipping.log" 2>&1 || true
frontend_skips=0
set +e
python "$repo/dev/ppu_int8/check_front_end.py" \
  "$out/front-end-baseline.log" "$out/front-end-shipping.log"
rc=$?
set -e
if [[ $rc -eq 3 ]]; then frontend_skips=$((frontend_skips + 1));
elif [[ $rc -ne 0 ]]; then exit "$rc"; fi
"${front[@]}" "$repo/csrc/qattn/ppu/quant_ppu.cu" \
  -o "$out/front-end-quant.o" >"$out/front-end-quant.log" 2>&1 || true
set +e
python "$repo/dev/ppu_int8/check_front_end.py" \
  "$out/front-end-baseline.log" "$out/front-end-quant.log"
rc=$?
set -e
if [[ $rc -eq 3 ]]; then frontend_skips=$((frontend_skips + 1));
elif [[ $rc -ne 0 ]]; then exit "$rc"; fi

cxx_inc=()
for path in "${torch_includes[@]}"; do cxx_inc+=(-I "$path"); done
"${CXX:-c++}" -std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
  "${cxx_inc[@]}" -I "$repo/csrc/qattn/ppu" \
  -c "$repo/csrc/qattn/ppu/pybind_ppu.cpp" -o "$out/pybind-ppu.o"

printf '[PPU Sage local] PASS: layout oracle + five negatives + Python + pybind; device_frontend_skips=%d (fresh hgcc required)\n' \
  "$frontend_skips"
