#!/usr/bin/env python3
"""CPU packaging negatives; no import of the PPU extension or device claim."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wheel_guard", ROOT / "sageattention/_ppu_wheel.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/sageattention-wheel-contract"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.out / "fixture.so"
    binary.write_bytes(b"CPU contract fixture; not a native extension")
    native = {"artifact": binary.name, "artifact_size": binary.stat().st_size,
              "artifact_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "build": {"python_cache_tag": sys.implementation.cache_tag,
                        "torch_public_version": "2.9.0", "cxx11_abi": True}}
    manifest = args.out / "_ppu_wheel_manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "native": native}))
    fake = SimpleNamespace(__version__="2.9.0+fixture", _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True))
    module.verify_installed_wheel(fake, args.out)
    for plant in ("torch-version", "cxx11-abi", "payload"):
        fake.__version__ = "2.8.0" if plant == "torch-version" else "2.9.0+fixture"
        fake._C._GLIBCXX_USE_CXX11_ABI = plant != "cxx11-abi"
        if plant == "payload":
            binary.write_bytes(b"X" * native["artifact_size"])
        try:
            module.verify_installed_wheel(fake, args.out)
        except RuntimeError as error:
            print(f"[PPU Sage wheel negative] {plant} EXPECTED-RED/PASS: {error}")
        else:
            raise RuntimeError(f"planted {plant} was accepted")
    print("[PPU Sage wheel] CPU manifest contracts PASS; device NOT RUN")


if __name__ == "__main__":
    main()
