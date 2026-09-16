# Dense SageAttention PPU wheel

Source main: `11a60bf` (runtime repair `a9181bf`). This independent artifact branch
stores the approximately 287 KB wheel as an ordinary Git blob; **Git LFS is not
required**. The current version is **2.2.0.post1+ppu.torch29**, for **Python 3.12 / Torch 2.9.0 / C++11
ABI=1 / PPU0010**. No NVIDIA binaries or sparse/Radial kernels are built by
this packaging step.

The previous wheel requested missing `libhggcrt.12.0.so` through the SDK's
legacy wrapper. **Use post1; do not symlink 13.0 to 12.0.** This repair binds the
actual `libhggcrt.13.0.so` runtime and protects its entry points from a globally
loaded old wrapper. All 28 device kernels / 43,024 instructions are unchanged.
The old wheel is retained for reproducibility but `install.sh` selects only post1.

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
identity from the original kernel build identity. Source/ABI/hash checks, three
negative controls, local pip installation and installed-version rejection pass.
The fixed extension also imports with the actual legacy SDK wrapper preloaded;
the old extension and an unprotected direct relink reproduce the exact failure.
A fresh box execution is not claimed. The SDK and Torch runtimes are required,
not bundled with the wheel.
