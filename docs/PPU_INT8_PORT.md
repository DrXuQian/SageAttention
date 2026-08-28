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

Latency, registers, spills, achieved occupancy, and ACU instruction mix are
device-only verdicts.  No performance claim is made from the local oracle.
After correctness admission, the first optimization targets are the K/V
pipeline depth and barrier count; the algorithm and actlize fragment maps stay
fixed.

## Commands

Local, no PPU execution:

```bash
SAGEATTENTION_PPU_ORACLE_OUT=/workspace/sageattention-ppu-local \
  dev/ppu_int8/run_local_gates.sh
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
