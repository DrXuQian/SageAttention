"""Build the actlize-backed PPU SageAttention extension.

This is deliberately separate from setup.py: NVIDIA's SM80/89/90 source graph
contains PTX contracts that are not valid PPU input.  A PPU build must opt in to
the PPU source graph rather than compile everything and hope unused PTX drops.
"""

import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
PPU_SDK = os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME")
if not PPU_SDK:
    raise RuntimeError("Set PPU_SDK (or PPU_HOME) to the PPU SDK root")
PPU_SDK = str(Path(PPU_SDK).resolve())
HGCC = Path(PPU_SDK) / "bin" / "hgcc"
if not HGCC.is_file():
    raise RuntimeError(f"hgcc not found at {HGCC}")
os.environ["PATH"] = f"{HGCC.parent}:{os.environ.get('PATH', '')}"
os.environ.setdefault("PYTORCH_NVCC", str(HGCC))

ACTLIZE = ROOT / "csrc" / "actlize"
if not (ACTLIZE / "include" / "cute" / "tensor.hpp").is_file():
    raise RuntimeError(
        "actlize submodule is missing; run: git submodule update --init csrc/actlize"
    )
SDK_INCLUDE = Path(PPU_SDK) / "include"
TARGET_INCLUDES = sorted((Path(PPU_SDK) / "targets").glob("*/include"))
if not SDK_INCLUDE.is_dir() or not TARGET_INCLUDES:
    raise RuntimeError(
        "PPU SDK must provide include/ and targets/<triple>/include; "
        f"got SDK_INCLUDE={SDK_INCLUDE} targets={TARGET_INCLUDES}"
    )

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS = ["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
HGCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-mllvm", "-ppu-max-vreg-count=256",
    "-mllvm", "-ppu-sink-matrix-addr=true",
    "-mllvm", "-ppu-sink-async-addr=true",
    "-mllvm", "-ppu-sink-load-addr=true",
    "-mllvm", "-ppu-sink-store-addr=true",
    "-DUSE_PPU=1",
    "-DUSE_AIU=1",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
    "-arch=ppu_10",
]

extension = CUDAExtension(
    name="sageattention._qattn_ppu",
    sources=[
        "csrc/qattn/ppu/pybind_ppu.cpp",
        "csrc/qattn/ppu/qk_int_sv_f16_ppu.cu",
        "csrc/qattn/ppu/quant_ppu.cu",
    ],
    include_dirs=[
        str(ROOT / "csrc" / "qattn" / "ppu"),
        str(ACTLIZE / "include"),
        str(SDK_INCLUDE),
        *[str(path) for path in TARGET_INCLUDES],
    ],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": HGCC_FLAGS},
)

setup(
    name="sageattention",
    version="2.2.0+ppu",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
