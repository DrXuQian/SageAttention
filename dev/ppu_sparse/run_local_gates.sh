#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${SAGEATTENTION_PPU_SPARSE_OUT:-/workspace/sageattention-ppu-sparse-local}"
mkdir -p "$out"

python "$repo/dev/ppu_sparse/test_plans.py" | tee "$out/plan-oracle.log"
python -m py_compile "$repo"/sageattention/ppu_sparse/*.py
python "$repo/tools/verify_ppu_prebuilt.py" \
  --repo "$repo" \
  --manifest "$repo/prebuilt/ppu_10/cpython312-torch2.8-cxx11abi1/manifest.json" \
  --artifact "$repo/prebuilt/ppu_10/cpython312-torch2.8-cxx11abi1/_qattn_ppu.cpython-312-x86_64-linux-gnu.so" \
  --json-out "$out/prebuilt-identity.json" --self-test \
  | tee "$out/prebuilt-identity.log"

python - "$repo" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
runners = [
    root / "tools/run_ppu_int8_box.sh",
    root / "tools/run_ppu_int8_perf_box.sh",
    root / "tools/run_ppu_sparse_box.sh",
]
forbidden = ("hgcc", "hgobjdump", "build_ext", "setup_ppu.py")
bad = {
    str(path.relative_to(root)): [token for token in forbidden if token in path.read_text()]
    for path in runners
}
bad = {path: tokens for path, tokens in bad.items() if tokens}
if bad:
    raise SystemExit(f"[PPU sparse local] FAIL: box runner can compile: {bad}")
print("[PPU sparse local] box-runners=3 mode=EXECUTION-ONLY/PASS")
PY

printf '[PPU sparse local] PASS: planner/oracle + prebuilt identity + execution-only runners\n'
