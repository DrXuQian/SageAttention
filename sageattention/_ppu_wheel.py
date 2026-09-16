"""Validate an installed PPU wheel before loading a Torch-linked extension."""

import hashlib
import json
from pathlib import Path
import sys


def verify_installed_wheel(torch_module, directory=None):
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    path = directory / "_ppu_wheel_manifest.json"
    if not path.is_file():
        return  # Source/prebuilt-tree installs keep their existing verifier.
    data = json.loads(path.read_text())
    if data.get("schema") != 1:
        raise RuntimeError("unknown PPU Sage wheel manifest schema")
    native = data["native"]
    expected = native["build"]
    actual = {
        "python_cache_tag": sys.implementation.cache_tag,
        "torch_public_version": torch_module.__version__.split("+", 1)[0],
        "cxx11_abi": bool(torch_module._C._GLIBCXX_USE_CXX11_ABI),
    }
    for key, value in actual.items():
        if value != expected[key]:
            raise RuntimeError(f"PPU Sage wheel {key}: got {value!r}, requires {expected[key]!r}")
    filename = native["artifact"]
    if Path(filename).name != filename:
        raise RuntimeError("unsafe PPU Sage wheel artifact filename")
    binary = directory / filename
    if not binary.is_file() or binary.stat().st_size != native["artifact_size"]:
        raise RuntimeError("PPU Sage wheel native payload missing or truncated")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != native["artifact_sha256"]:
        raise RuntimeError("PPU Sage wheel native payload SHA256 mismatch")
