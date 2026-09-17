# Dense SageAttention PPU wheels

The default is **INT8 QK / FP16 PV / FP32 accumulation**, the existing
`2.2.0.post1+ppu.torch29` wheel. It retains the native HGGC 13 runtime fix.
Python 3.12, PPU Torch 2.9.0, C++11 ABI=1 and PPU0010 are required.
This selection change does not rebuild or change either wheel payload.

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

`release.json` is the FP16-PV authority. `release-pv-int8.json` retains the
original all-INT8 post2 baseline. The source experiment is isolated on
`experiment/ppu-pv-int8`; **its later local requant ALU optimization is not in
the post2 wheel**. That optimization has no new PPU timing/model-quality result.
Do not treat quantized-oracle correctness as full-model quality admission.

Only to select that original experiment explicitly:

```bash
bash /workspace/ppu-wheel-sage/install.sh --experimental-pv-int8
```

`python verify_release.py` checks both real wheels and rejects swapped precision
labels and a correctly hashed INT8 wheel disguised as the main FP16 release.
It submits no device work. Original payloads remain on this dedicated artifact
branch for reproducibility; no wheel bytes are added or modified by this change.
