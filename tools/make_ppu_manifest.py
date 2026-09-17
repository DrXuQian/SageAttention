#!/usr/bin/env python3
"""Bind an SDK-built PPU extension to source, target ABI and native runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
from ppu_native_link import check_native_linkage, runtime_soname, wrapped_symbols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact = args.artifact.resolve()
    soname = runtime_soname(args.sdk / "lib/libhggcrt1.so")
    linkage = check_native_linkage(artifact, soname)
    linkage["binding"] = "hidden --wrap entries with handle-scoped native lookup"
    linkage["symbols"] = wrapped_symbols(root / "csrc/qattn/ppu/native_runtime.cpp")
    resources = args.resources.read_text()
    regs = [int(x) for x in re.findall(r"vreg_number:(\d+)", resources)]
    stacks = [int(x) for x in re.findall(r"STACK SIZE:(\d+)", resources)]
    if len(regs) != 72 or len(stacks) != 72 or any(stacks):
        raise RuntimeError("PPU resource evidence must cover all 72 dense/quant/V kernels, no spills")
    git = lambda *words: subprocess.check_output(["git", *words], cwd=root, text=True).strip()
    sources = ["setup_ppu.py", "tools/ppu_native_link.py"]
    sources += [str(p.relative_to(root)) for p in sorted((root / "csrc/qattn/ppu").iterdir())
                if p.suffix in (".cpp", ".cu", ".cuh", ".h")]
    compiler = subprocess.check_output([str(args.sdk / "bin/hgcc"), "--version"], text=True)
    manifest = {
        "schema_version": 1,
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_size": artifact.stat().st_size,
        "build": {
            "source_git_commit": git("rev-parse", "HEAD"),
            "actlize_git_commit": git("rev-parse", "HEAD:third_party/actlize"),
            "ppu_arch": "ppu_10", "compiler_identity": compiler,
            "python_cache_tag": sys.implementation.cache_tag,
            "torch_public_version": torch.__version__.split("+", 1)[0],
            "torch_build_version": torch.__version__,
            "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        },
        "resource_evidence": {"specializations": len(regs), "max_vector_registers": max(regs),
                              "nonzero_private_frames": [], "scope": "local hgobjdump; no device run"},
        "attention_pv_modes": ["fp16", "int8"],
        "attention_key_layouts": ["raw", "permuted-full-k64"],
        "all_int8_device_admission": "NOT RUN; execute dev/ppu_int8/device_all_int8.py before timing",
        "required_ppu_runtime_libraries": [soname, "libhggc.so", "libalippu.so"],
        "native_runtime_linkage": linkage,
        "source_sha256": {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in sources},
    }
    path = artifact.parent / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PPU manifest] {path} sha256={manifest['artifact_sha256']} runtime={soname}")


if __name__ == "__main__":
    main()
