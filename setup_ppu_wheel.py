"""Package one verified PPU prebuilt into a wheel, without rebuilding kernels.

The manifest selects the target Torch ABI, not the host's installed Torch.
No NVIDIA/PPU compiler is invoked by this packaging-only entry point.
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from tools.ppu_native_link import check_native_linkage

ROOT = Path(__file__).resolve().parent
VARIANT = os.environ.get("SAGEATTENTION_PPU_WHEEL_VARIANT", "cpython312-torch2.9-cxx11abi1")
override = os.environ.get("SAGEATTENTION_PPU_PREBUILT_DIR")
directory = Path(override).resolve() if override else ROOT / "prebuilt/ppu_10" / VARIANT
if not (directory / "manifest.json").is_file():
    raise RuntimeError(f"unknown PPU prebuilt variant: {VARIANT}")
manifest = json.loads((directory / "manifest.json").read_text())
artifact = directory / manifest["artifact"]
spec = importlib.util.spec_from_file_location("ppu_prebuilt", ROOT / "tools/verify_ppu_prebuilt.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
target = manifest["build"]
if sys.implementation.cache_tag != target["python_cache_tag"]:
    raise RuntimeError("wheel packager Python ABI differs from the selected prebuilt")
verifier.verify(manifest, ROOT, artifact, target)
linkage = manifest.get("native_runtime_linkage", {})
if not linkage.get("soname"):
    raise RuntimeError("Prebuilt lacks native-runtime linkage evidence; rebuild with setup_ppu.py")
check_native_linkage(artifact, linkage["soname"])


class VerifiedPrebuilt(build_ext):
    def run(self):
        verifier.verify(manifest, ROOT, artifact, target)
        check_native_linkage(artifact, linkage["soname"])
        for extension in self.extensions:
            output = Path(self.get_ext_fullpath(extension.name))
            if output.name != artifact.name:
                raise RuntimeError("wheel extension suffix differs from the verified prebuilt")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, output)
            (output.parent / "_ppu_wheel_manifest.json").write_text(
                json.dumps({
                    "schema": 1,
                    "scope": "verified-prebuilt-repack-no-new-device-run",
                    "variant": VARIANT,
                    "package_source_sha": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "native": manifest,
                }, indent=2) + "\n"
            )
            print(f"[PPU Sage wheel] exact prebuilt {manifest['artifact_sha256']} -> {output}")


torch_tag = ".".join(target["torch_public_version"].split(".")[:2]).replace(".", "")
setup(
    name="sageattention",
    version=f"2.2.0.post1+ppu.torch{torch_tag}",
    description="Dense actlize PPU SageAttention, verified prebuilt wheel",
    license="Apache-2.0",
    license_files=["LICENSE"],
    packages=find_packages(include=["sageattention", "sageattention.*"]),
    python_requires=">=3.12,<3.13",
    install_requires=[f"torch=={target['torch_public_version']}"],
    ext_modules=[Extension("sageattention._qattn_ppu", sources=[])],
    cmdclass={"build_ext": VerifiedPrebuilt},
)
