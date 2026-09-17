# Dense SageAttention PPU wheels

The default is **INT8 QK / FP16 PV / FP32 accumulation**, the existing
`2.2.0.post1+ppu.torch29` wheel. It retains the native HGGC 13 runtime fix.
Python 3.12, PPU Torch 2.9.0, C++11 ABI=1 and PPU0010 are required.
The FP16-PV payload is unchanged; the INT8-PV experiment is retired.

```bash
git clone --single-branch --branch ppu-wheels \
  https://github.com/DrXuQian/SageAttention.git /workspace/ppu-wheel-sage
# If already cloned: git -C /workspace/ppu-wheel-sage pull --ff-only
bash /workspace/ppu-wheel-sage/install.sh
```

Use the existing PPU SDK runtime environment. Do not symlink HGGC 13 as HGGC 12.
The installer checks Python/Torch/C++ ABI, the complete wheel/native hashes,
and the actual PPU dispatch in the wheel's Python code. It uses `--no-deps` and
does not download or replace Torch. `--verify-only` checks without installing.
Restart Python/ComfyUI after changing a wheel; pulling source alone does not
replace an installed post2 default. Run imports outside a source checkout.

`release.json` is the only supported release authority. The INT8-PV post2
wheel, its manifest and `--experimental-pv-int8` option have been removed.
The installer rejects that old option instead of silently selecting a mode.

`python verify_release.py` verifies the real FP16 wheel and tests precision,
payload-hash and installed-route negatives. It submits no device work.
The retained payload stays on this dedicated artifact branch; its bytes are
unchanged. Git history retains the retired experiment, not the current tree.
