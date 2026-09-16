# Dense SageAttention PPU wheel

Source main: `4ca7d9f`. This independent artifact branch stores the approximately
284 KB wheel as an ordinary Git blob; **Git LFS is not required**. It contains
the unchanged verified dense prebuilt for **Python 3.12 / Torch 2.9.0 / C++11
ABI=1 / PPU0010**. No NVIDIA binaries or sparse/Radial kernels are built by
this packaging step.

```bash
git clone --single-branch --branch ppu-wheels \
  https://github.com/DrXuQian/SageAttention.git /workspace/ppu-wheel-sage
bash /workspace/ppu-wheel-sage/install.sh
export LD_LIBRARY_PATH=/workspace/ppu-sdk-2.1.1-a5c56e/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
(cd /workspace && python -c 'import sageattention; from sageattention import core; print(sageattention.__file__, "PPU_ENABLED=", core.PPU_ENABLED); assert core.PPU_ENABLED')
```

Change the SDK path if needed. Run Python outside old source checkouts so their
in-place `.so` files cannot shadow the wheel. `install.sh` checks the target ABI
and wheel hash before installing. Import verifies the native payload hash and
ABI again. `--no-deps` deliberately preserves the PPU Torch installation instead
of downloading a public CUDA Torch wheel.

`release.json` and the embedded `_ppu_wheel_manifest.json` distinguish packaging
identity from the original kernel build identity. Source/ABI/hash checks, three
negative controls, local pip installation and installed-version rejection pass.
A fresh box execution is not claimed. The SDK and Torch runtimes are required,
not bundled with the wheel.
