#!/usr/bin/env python3
"""Make an isolated import directory for a verified experimental prebuilt.

Does not install a wheel or overwrite an installed/default SageAttention.
"""
import argparse
import json
from pathlib import Path
import shutil

from verify_ppu_prebuilt import runtime_identity, verify


def stage(repo, manifest_path, destination):
    manifest = json.loads(manifest_path.read_text())
    binary = manifest_path.parent / manifest["artifact"]
    evidence = verify(manifest, repo, binary, runtime_identity())
    destination.mkdir(parents=True, exist_ok=False)
    package = destination / "sageattention"
    for source in (repo / "sageattention").rglob("*.py"):
        target = package / source.relative_to(repo / "sageattention")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(binary, package / binary.name)
    (package / "_ppu_wheel_manifest.json").write_text(json.dumps(
        {"schema": 1, "scope": "isolated experimental directory; not installed", "native": manifest}, indent=2) + "\n")
    (destination / "identity.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"[PPU ALU candidate] isolated={destination} sha256={evidence['artifact_sha256']} default_install=UNCHANGED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stage(args.repo.resolve(), args.manifest.resolve(), args.out.resolve())
