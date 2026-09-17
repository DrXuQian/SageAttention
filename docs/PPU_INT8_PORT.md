# SageAttention INT8 on PPU through actlize

This port follows the shipping SM80 FP32-accumulation algorithm in
`csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`; it does not translate its NVIDIA PTX.
The PPU source graph is independent and contains no private inline assembly.

## Algorithm crosswalk

| Sage semantic step | PPU implementation |
|---|---|
| Q per-warp INT8 quantization (`BLKQ=128`, `WARPQ=32`) | `quant_ppu.cu`, 32-row blocks |
| K per-block INT8 quantization (`BLKK=64`), optional K-mean subtraction | `quant_ppu.cu`, 64-row blocks |
| `Q_int8 @ K_int8^T -> int32` | actlize `PPU0010_16x16x32_S32S8S8S32_TN` |
| `q_scale * k_scale * softmax_scale * log2(e)` | unchanged, once per Q warp / K block |
| online row max, denominator, and old-output rescale | unchanged mathematically; PPU row peers are derived from `MMA_Traits::CLayout` |
| score FP32 to probability FP16 | actlize f16-output MMA bridge, proven over all 256 fragment values |
| `P_fp16 @ V_fp16 -> O_fp32` | actlize `PPU0010_16x16x16_F32F16F16F32_TN` |
| normalize and optional base-2 LSE | unchanged; Python applies the original natural-log and K-mean corrections |

Q and K use non-transposed `.swzl` AIU writes and non-transposed TSM reads. V
uses the independently device-anchored pair from `fattn_ppu.cu`: a
non-transposed AIU write followed by a transposed TSM read. Head dimension 128
is two proven 64-wide actlize slices, not a new 128-wide layout assumption.

## The fragment seam that is not portable from NVIDIA

PPU0010's accumulator `CLayout` and FP16 MMA `ALayout` are not the same
physical register layout. CuTe linearizes the logical M mode first; treating
the returned offset as row-major transposes the logical coordinates and is
240/256 wrong. Passing the fixed bridge weights as A and the score as B is
also 240/256 wrong. The production direction is score-as-A, weights-as-B,
matching the measured `fattn_ppu.cu` implementation.

`dev/ppu_int8/layout_oracle.cu` uses the real actlize traits to prove:

- the closed-form PPU row/column map (0/512 differences);
- each four-lane row peer set (exact 16-column coverage);
- the one-MMA C-to-next-A bridge (0/256 differences);
- the historical reversed bridge as an exact negative (240/256 differences).

The oracle is host-only.  It never claims to replace a fresh hgcc build or a
device numeric admission. Its canonical output is committed as
`dev/ppu_int8/layout_oracle.expected.txt`; the box runner consumes that file
from its own result SHA and prints `fresh_box_execution=0`. NVIDIA nvcc cannot
instantiate actlize's HGGCCC-only device atoms, so that frontend is reported as
an explicit SKIP, never as a successful PPU body compile.

The first device admission exposed a compiler seam that the layout algebra
could not see: the initial actlize fp16-output atom treated four independent
reference arguments as one contiguous array.  A caller passing the same zero
object as all four C operands therefore loaded three neighboring values.  The
generated PPU ISA initialized only 1/4 accumulator registers and produced a
stable but grossly wrong PV result while QK/LSE remained correct.  The fixed
atom binds every reference independently.  `bridge_codegen_probe.cu` now
compares it with the device-proven raw spelling and retains the exact old
implementation as a negative; generated ISA must report `4/4`, `4/4`, and
`1/4 EXPECTED-RED`.  The shipping binary is additionally checked at all 128
static bridge sites.

## First shipping scope

- forward, fixed-length HND and NHD tensors;
- PPU0010 (`-arch=ppu_10`);
- FP16/BF16 Q/K/V input, INT8 QK, FP16 V, FP32 PV accumulation;
- head dimension up to 128 (padded to 64 or 128);
- GQA where query heads are divisible by KV heads;
- full attention and equal-length causal attention;
- optional LSE and optional K smoothing.

Not included: arbitrary masks, varlen, backward, INT4 QK, per-thread Q/K
quantization, PPU0015, or fused quantization.  These are explicit unsupported
contracts, not silent fallbacks.

## Performance invariants and device-only questions

For each K64 block and warp, D64/D128 execute respectively:

- QK MMA: 16 / 32;
- C-to-A bridge MMA: 8 / 8;
- PV MMA: 32 / 64.

The bridge repair did not add an MMA relative to the first draft.  Matched
64-wide Q/K slices make D128 issue two known AIU operations instead of one
unproved 128-wide operation.  The loop has four CTA barriers per K64 block.

Registers, spills, and generated bridge initialization are established by the
local PPU SDK build.  Latency, achieved occupancy, and ACU instruction mix
remain device-only verdicts; none is inferred from the host layout oracle.

The PPU extension does not use PyTorch's `CUDAExtension`: that route selects
the SDK's CUDA-compatibility `nvcc` wrapper and is not the shipping PPU device
language.  `setup_ppu.py` compiles device TUs with the SDK's native
`hgcc -x hg -DSWITCH_TO_HGGCRT`, compiles the pybind TU with the host C++
compiler, and links both against Torch and the PPU runtime.  The dynamic-shared
memory opt-in is checked once per kernel specialization instead of being
reissued for every attention call; the before/after PPU ISA is byte-identical.

The PV traversal is K-block-major.  That keeps only one K block's FP16
probability operands live while preserving the exact `kb=0,1,2,3` accumulation
order of every `(d,qb)` output.  The host proof exhausts D64 and D128 and its
reversed-K negative turns red.  With SDK 2.1.1 this removes the only two 8-byte
stack spills (D128 causal/no-LSE, FP16 and BF16 output); all 28 device
specializations now report stack size zero.  Static instruction footprint grows
slightly (D64 full/no-LSE 2095 to 2103, D128 2822 to 2845), so latency remains a
device measurement rather than being inferred from the resource improvement.

## Commands

### Verified Torch 2.9 wheel

The original `2.2.0+ppu.torch29` wheel must not be used with the SDK 2.1.1
13.0 runtime: it linked `libhggc_wrapper.so`, whose loader requests the absent
`libhggcrt.12.0.so`. **Do not symlink 13.0 under the 12.0 name.** The replacement
version is `2.2.0.post1+ppu.torch29`; only its host runtime binding changes.

`setup_ppu.py` now reads the SDK runtime's actual ELF SONAME, keeps that direct
dependency, and binds all ten HGGC registration/launch entry points through
hidden wrappers and handle-scoped lookup. This remains correct if Torch has
already loaded the SDK's old shim globally. The Torch/pybind extension itself
is **not** deep-bound: its ATen/C++ objects retain the existing process ABI.
The post-link gate rejects both shim dependencies and any remaining unbound
HGGC entry. `check_native_link.py` reproduces the original failure locally and
checks the fixed extension with the same legacy shim preloaded. A direct-only
relink without hidden bindings is the second negative; it still reproduces the
missing-12.0 error. All 28 device kernels / 43,024 instructions compare unchanged
against the previous Torch 2.9 prebuilt. No new device performance is claimed.

The SDK build now emits `manifest.json` beside the extension. To package that
fresh output rather than a historical source-tree prebuilt, set
`SAGEATTENTION_PPU_PREBUILT_DIR=<build-lib/sageattention>` when running
`setup_ppu_wheel.py`. Packaging rejects a prebuilt without the native-link
contract. The box receives an already-built wheel, not a rebuild command.

The verified wheel is published on the independent
[`ppu-wheels` artifact branch](https://github.com/DrXuQian/SageAttention/tree/ppu-wheels)
at `be05a56`. It is an ordinary Git blob (no LFS required); `main` contains no new
wheel payload. The branch includes `release.json` and a checksum/ABI-verifying
`install.sh`. Its exact filename is
`sageattention-2.2.0.post1+ppu.torch29-cp312-cp312-linux_x86_64.whl`, SHA256
`70409a902755efcbff7d06124ad871b627ad9e59ad448c8e69f2e4d5fba9f366`.

`setup_ppu_wheel.py` packages the exact dense-only SDK output for
`cpython312-torch2.9-cxx11abi1`. It verifies the compiled source hashes,
actlize gitlink, payload SHA256 and target ABI before copying that library into
the wheel; it does **not** compile a replacement kernel. The packager's own Torch
installation does not select the target ABI.

The runtime-fixed wheel is `sageattention==2.2.0.post1+ppu.torch29` for Python 3.12, Torch 2.9.0,
C++11 ABI=1, PPU0010. Set `LD_LIBRARY_PATH` to the PPU SDK `lib` directory and
install with `python -m pip install --no-deps --force-reinstall <wheel>`.
Test imports outside the source checkout to avoid loading an old in-place `.so`.
At import, `_ppu_wheel_manifest.json` validates Python/Torch/C++ ABI and the
installed native payload hash before loading the extension. ABI or payload
mismatches are errors, not silent fallbacks to a different backend.

Local packaging and three planted ABI/hash failures are tested by
`python dev/ppu_int8/check_wheel_contract.py`. These are CPU checks, not a new
device correctness/performance measurement. The native manifest retains the
original kernel build identity separately from the wheel's packaging commit.

Local, no PPU execution:

```bash
SAGEATTENTION_PPU_ORACLE_OUT=/workspace/sageattention-ppu-local \
  dev/ppu_int8/run_local_gates.sh
```

Native PPU SDK compile, link, device-symbol census, and resource admission
(also no device execution):

```bash
PPU_SDK=/usr/local/PPU_SDK \
SAGEATTENTION_PPU_SDK_OUT=/workspace/sageattention-ppu-sdk \
  bash dev/ppu_int8/run_sdk_compile.sh
```

PPU build and four-case numeric admission:

```bash
PPU_SDK=/usr/local/PPU_SDK \
OUT=/workspace/sageattention-ppu-int8 \
  bash tools/run_ppu_int8_box.sh
```

The device admission covers full/causal, a tail, GQA, NHD, D64, D128
multi-slice, LSE, and three raw-bit-stable replays.  Its reference consumes the
actual quantized Q/K and scale tensors, so quantization error cannot disguise a
layout error.

## PPU0010 device verdict (2026-09-03)

### Same-input BF16 FlashAttention comparison

`tools/run_ppu_sage_bf16_ab.sh` uses the installed **post1** Sage wheel and an
installed `flash_attn_2_cuda`. It never builds, copies an old source-tree `.so`,
or falls back to SDPA. It rotates three sequential event-timed arms on the same
BF16 Q/K/V: native FA2 BF16 forward, prequantized Sage core, and Sage Q/K
quantization + V cast + core. Outputs are BF16; Sage's internal V remains FP16.
Input/output/quantization buffers are preallocated; FA's internal LSE/RNG buffer
bookkeeping remains inside its native forward call. Kernel spans include host
launch idle and are not profiler-derived device-only durations.

```bash
# Historical attention SHAPE, newly measured BF16 comparison (not the old FP16 run):
bash tools/run_ppu_sage_bf16_ab.sh
# The earlier MiniMax H3 attention shape; one launch/sample avoids a long run:
HEADS=56 SEQ=73774 LAUNCHES=1 WARMUP=2 bash tools/run_ppu_sage_bf16_ab.sh
# Optional matched causal comparison:
CAUSAL=1 bash tools/run_ppu_sage_bf16_ab.sh
```

Default: B1/H16/S4096/D128, full attention. Both arms use NHD, unlike the old
FP16/HND Sage anchor below. No historical timing is relabelled as a BF16 baseline.
Q/K smoothing is off. Per-role raw samples, binary hashes, device identity and
quantization-error diagnostics are written to `/workspace/.../result.json`.
Overlapping min/max envelopes are UNRESOLVED. Replay/error checks are sampled
and do not constitute a new whole-domain correctness proof. MFU is explicitly
**BF16-equivalent** normalization, not mixed INT8/FP16 hardware SOL.
If FA is available only in an earlier build, set `FA_PYTHONPATH` to that native
runtime directory plus its matching `flash-attention-for-sail` source directory.
CPU plan/arithmetic contracts are in `dev/ppu_int8/check_sage_bf16_bench.py`;
device performance must be measured on the box.

### Single-launch MiniMax H3 ACU capture

`tools/profile_ppu_attention_pipes.py` is committed separately from the uploaded
experimental kernel patch. It uses the installed native extension, prints its
path and SHA256, and never builds or changes the installed wheel. `--arm sage`
does not import FlashAttention. BF16/NHD inputs, quantization and scale settings
match the A/B benchmark above. No warmup or timed benchmark loop is performed.

```bash
mkdir -p /workspace/sage-h3-acu
cd /workspace
/sim/eec/shared/junfu.qx/asight/bin/acu \
  -f -o /workspace/sage-h3-acu/sage-h3 --set full \
  python /sim/eec/shared/junfu.qx/SageAttention/tools/profile_ppu_attention_pipes.py \
    --arm sage --batch 1 --heads 56 --seq 73774 --head-dim 128 \
    --device "${DEVICE:-0}" --iters 1
```

Use the already-working PPU runtime environment. `DEVICE` is the Torch-visible
ordinal. The expected Sage kernel is `qk_int8_pv_f16_kernel`, grid `(577,56,1)`,
128 threads/CTA, noncausal. ACU can also list input initialization, Q/K
quantization and the V cast: do not add those to the core kernel duration.
ACU replay may execute the single requested call multiple times for counters.
`--describe` and `dev/ppu_int8/check_attention_profile_target.py` validate the
host-side plan only, not device correctness or performance.

### Recorded FP16-input result

The prebuilt extension from `44a641a` (SHA256
`059c9948f439abc1663103a3279a02ed7c40d598606b42895b307fb13fbe4168`)
passed all four numeric admissions.  The worst output error was
`9.221e-5`, the worst LSE error was `4.8e-7`, and every replay was raw-bit
stable.

Performance commit `3794335` measured `B=1, H=Hkv=16, N=4096, D=128` with
seven samples and 20 launches per sample.  Logical MFU uses the explicitly
printed 500 TFLOP/s PPU denominator:

| Mode | Scope | Median | Logical TFLOP/s | Logical MFU |
|---|---|---:|---:|---:|
| full | prequantized core | 405.972 us | 338.543 | 67.71% |
| full | Q/K quantization + core | 529.114 us | 259.753 | 51.95% |
| causal | prequantized core | 289.606 us | 237.344 | 47.47% |
| causal | Q/K quantization + core | 400.608 us | 171.580 | 34.32% |

The registered headline result is the **prequantized attention core**:
`405.972 us / 338.543 TFLOP/s / 67.71% logical MFU` for full attention.
Q/K quantization is deliberately outside that result; `529.114 us / 51.95%`
is retained only as a separately labelled end-to-end diagnostic.  The full
and causal output fingerprints were respectively `5837302b21552a97` and
`070b28c772e1d76e7`.  No NVIDIA measurement is part of this verdict.

The full-attention core exceeds the device-proven fp16 `fattn_ppu.cu` anchor
of 294 TFLOP/s by 15.2%.  The causal MFU uses triangular useful FLOPs and is
not compared directly with full attention; its 289.606 us latency is 20.7%
below the existing 365 us causal anchor.  Quantization adds 123.142 us to full
attention and 111.002 us to causal attention.  Therefore the next performance
bisection is Q quantization versus K quantization versus the already-admitted
core, not an unmeasured rewrite of the attention mainloop.

Reproduce the timing without rebuilding:

```bash
RUN_CAUSAL=1 bash tools/run_ppu_int8_perf_box.sh
```
