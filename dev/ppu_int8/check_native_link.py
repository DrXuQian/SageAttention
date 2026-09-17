#!/usr/bin/env python3
"""Import the real extension with/without the SDK's globally loaded legacy shim.

No device work is submitted. The old extension must reproduce the missing-12.0
error, and the candidate must import successfully even with that same shim.
"""

import argparse
import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("native_link", ROOT / "tools/ppu_native_link.py")
link = importlib.util.module_from_spec(spec)
spec.loader.exec_module(link)

IMPORT = """
import ctypes, importlib.util, os, sys, torch
if sys.argv[2] == 'preload':
    ctypes.CDLL(sys.argv[3], mode=os.RTLD_NOW | os.RTLD_GLOBAL)
spec = importlib.util.spec_from_file_location('_qattn_ppu', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert all(hasattr(module, name) for name in (
    'qk_int8_sv_f16_accum_f32_attn', 'quant_per_block_int8', 'quant_per_warp_int8',
    'qk_int8_sv_int8_accum_f32_attn', 'quant_value_int8'))
try:
    module.quant_per_warp_int8(torch.zeros((1,1,32,64), dtype=torch.float16),
        torch.empty((1,1,32,64), dtype=torch.int8), torch.ones((1,1,1)), 128, 32, 0)
except RuntimeError as error:
    assert 'must be device tensors' in str(error), str(error)
else:
    raise AssertionError('CPU input must be rejected before a device launch')
try:
    module.quant_value_int8(torch.zeros((1,1,64,64), dtype=torch.float16),
        torch.empty((1,1,1,64,64), dtype=torch.int8), torch.ones((1,1,1,64)), 1)
except RuntimeError as error:
    assert 'on the same device' in str(error), str(error)
else:
    raise AssertionError('new V quantizer must also reject CPU input')
print('NATIVE_IMPORT_PASS', torch.__version__)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--unbound", type=Path,
                        help="Direct-native relink without hidden bindings (second negative)")
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--loader")
    parser.add_argument("--library-path")
    args = parser.parse_args()
    prefix = [args.python]
    if args.loader:
        if not args.library_path:
            parser.error("--loader requires --library-path")
        prefix = [args.loader, "--library-path", args.library_path, args.python]
    soname = link.runtime_soname(args.sdk / "lib/libhggcrt1.so")
    print(link.check_native_linkage(args.candidate, soname))
    try:
        link.check_native_linkage(args.old, soname)
    except RuntimeError:
        print("[PPU link negative] old-wrapper-ELF EXPECTED-RED/PASS")
    else:
        raise RuntimeError("old wrapper link was incorrectly admitted")
    cases = [
        ("old-wrapper", args.old, "plain", False),
        ("candidate", args.candidate, "plain", True),
        ("global-wrapper-interposition", args.candidate, "preload", True),
    ]
    if args.unbound:
        try:
            link.check_native_linkage(args.unbound, soname)
        except RuntimeError as error:
            if "interposition" not in str(error):
                raise
            print("[PPU link negative] unbound-native-entries EXPECTED-RED/PASS")
        else:
            raise RuntimeError("unbound native entry points were incorrectly admitted")
        cases.append(("unbound-native-interposition", args.unbound, "preload", False))
    for role, path, preload, expected in cases:
        result = subprocess.run(
            [*prefix, "-c", IMPORT, str(path.resolve()), preload,
             str(args.sdk / "lib/libhggc_wrapper.so")],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        if expected:
            ok = result.returncode == 0 and "NATIVE_IMPORT_PASS" in result.stdout
        else:
            ok = result.returncode != 0 and "libhggcrt.12.0.so" in result.stdout
        if not ok:
            raise RuntimeError(f"{role} unexpected rc={result.returncode}: {result.stdout}")
        print(f"[PPU link import] {role} {'PASS' if expected else 'EXPECTED-RED/PASS'}")
    print("[PPU link] runtime registration/import only; device NOT RUN")


if __name__ == "__main__":
    main()
