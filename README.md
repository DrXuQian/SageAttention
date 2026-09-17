# Dense SageAttention PPU wheel

Source main: `fbf3d0f` (all-INT8 QK/PV). This independent artifact branch
stores the approximately 497 KB wheel as an ordinary Git blob; **Git LFS is not
required**. The current version is **2.2.0.post2+ppu.torch29**, for **Python 3.12 / Torch 2.9.0 / C++11
ABI=1 / PPU0010**. No NVIDIA binaries or sparse/Radial kernels are built by
this packaging step.

Post2 changes the PPU `sageattn()` default to **S8 QK + U8xS8 PV**, with FP32
softmax/cross-block accumulation. The named `sageattn_qk_int8_pv_fp16_ppu` path
remains available. **PPU device correctness/performance are not yet measured**;
run the new numeric admission before profiling or model deployment. This adds
lossy P/V quantization; local random-input error is not model-quality evidence.

Post2 retains the native `libhggcrt.13.0.so` bindings fixed in post1. **Do not
symlink 13.0 to 12.0.** Older wheels are retained for reproducibility/rollback,
but `install.sh` selects post2 from `release.json`. No device speedup is claimed.

```bash
git clone --single-branch --branch ppu-wheels \
  https://github.com/DrXuQian/SageAttention.git /workspace/ppu-wheel-sage
export LD_LIBRARY_PATH=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
bash /workspace/ppu-wheel-sage/install.sh
(cd /workspace && python -c 'import sageattention; from sageattention import core; print(sageattention.__file__, "PPU_ENABLED=", core.PPU_ENABLED); assert core.PPU_ENABLED')
```

If you already cloned this artifact branch, use
`git -C /workspace/ppu-wheel-sage pull --ff-only` instead of cloning again.

Change the SDK path if needed. Run Python outside old source checkouts so their
in-place `.so` files cannot shadow the wheel. `install.sh` checks the target ABI
and wheel hash before installing. Import verifies the native payload hash and
ABI again. `--no-deps` deliberately preserves the PPU Torch installation instead
of downloading a public CUDA Torch wheel.

`release.json` and the embedded `_ppu_wheel_manifest.json` distinguish packaging
identity from the native kernel build identity. Source/ABI/hash checks, three
negative controls, local pip installation and installed-version rejection pass.
The fixed extension also imports with the actual legacy SDK wrapper preloaded;
the old extension reproduces the exact failure. The unprotected direct-relink
negative was established in post1 and was not rerun for post2.
A fresh box execution is not claimed. The SDK and Torch runtimes are required,
not bundled with the wheel.

Local hgcc/hgobjdump compiled all 48 kernels without private stack; all 16
integer-PV attention bodies contain S8 QK/U8xS8 PV and zero floating MMA. Real
CuTe traits anchor the exhaustive probability map. CPU numeric admission's
worst relative RMSE is 1.55%; a single-head/five-query sample of S73774 is 1.49%.
These do not replace a PPU numeric result or downstream model validation.

After installing, restart any existing Python/ComfyUI process. In the source
checkout (not this artifact branch):

```bash
git pull --ff-only
PROFILE=1 bash tools/run_ppu_all_int8_box.sh
```

This runs correctness first, then captures B1/H56/S73774/D128/full Sage ACU,
using the installed wheel. It never compiles on the box. Use
`tools/run_ppu_sage_bf16_ab.sh --pv int8` or `--pv fp16` for the controlled
latency comparison; preparation and core are separate reported spans.
