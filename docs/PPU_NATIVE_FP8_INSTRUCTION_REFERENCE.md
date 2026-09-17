# RTX 5070 native FP8 reference for the PPU integer-PV experiment

## Verdict and boundary

The reference is useful: the measured PPU integer/bit/select stream is about
**9.09x** the native NVIDIA FP8 stream for the same matrix work. Integer PV
does not by itself justify that excess. This is an instruction-structure
comparison, **not** equal-precision accuracy, equal-input latency, or a claim
that PPU can remove every extra instruction.

Main is restored to INT8 QK / FP16 PV (`12a4135`); the default artifact channel
is the existing post1 wheel (`ppu-wheels` at `d713b01`). Integer PV and these
measurements remain on `experiment/ppu-pv-int8`. No new candidate wheel is
published and no new PPU run was performed here.

## Measured identities

Common work: B1, Hq=Hkv=56, S=73774, D=128, noncausal, BF16 input/output, NHD,
Q128/KV64, 128 threads, grid `(577,56,1)`. Preparation is outside the profiled
core. The prior PPU Q/K/V bytes were not recovered: matching shapes and matrix
counts do not claim byte-identical input or model-quality parity.

- PPU: uploaded `sage-h3-int8.acurep`, source `fbf3d0f`, original post2 binary
  `98c43ca4d30c16ef260114588b51ba9f81ef66dd37f5296f6b3de06bc7b4077f`.
  Report SHA256 `0e120a27783b9766acf5ffb5c3361207ea7b0f0e6bbe25cbe7db933938d9febc`.
  Both QK and PV are integer; P is U8, V is S8, block/channel scale, S32
  partials and FP32 online accumulation.
- NVIDIA: RTX 5070, UUID `GPU-46dc05e0-dac4-dc09-3a8b-30e380f3acc7`, driver
  595.59, WSL, Torch 2.11.0+cu128, CUDA 12.8, NCU 2025.1.1.
  Unmodified official SageAttention source
  `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`; `_qattn_sm89` binary SHA256
  `6f43b7052afc68c21c95907c1faf46691cf5a0c47ee99e50f5adec8304a54d56`.
  QK is S8; PV operands are E4M3. Two native calls were profiled sequentially:
  the SM120-default FP16 instruction buffer / FP32 output mode, and direct
  FP32 accumulation. These are not native INT8-PV calls.
- Both NVIDIA arms report 255 registers/thread and a 1024-byte stack attribute;
  their exported SASS has zero LDL/STL sites. Do not relabel the stack
  attribute as measured spill traffic. No latency/resource superiority claim.

All raw CSV, logs, opcode counts and extraction driver are committed under
`dev/ppu_int8/results/5070-fp8-reference/` and `dev/ppu_int8/`.
Full local reports are retained in
`/workspace/sage-nvidia-5070-reference-20260917/ncu-fp8-reference/`:

| Report | SHA256 |
|---|---|
| `h3-native-fp8-fp16-buffer.ncu-rep` | `0032974811e0109c7b2f62d6c5a3f0d4a4408c966672544c9e5382c2ecb5e6bd` |
| `h3-native-fp8-fp32.ncu-rep` | `0f78ed1794906a9a80c55731abd69ce7773d8ac368f1ad809941e9185dd4c28f` |

## Common denominator, before comparing instructions

There are `577*56*4*1153 = 149,022,944` warp/K64 visits. Each visit covers
the same QK and PV matrix work on both devices. The PPU emits 32 QK + 32 PV
`m16n16k32` instructions; NVIDIA emits 64 QK + 64 PV `m16n8k32` instructions.
Both observed MMA families independently match these exact totals. Thus an
unadjusted MMA-count ratio of two is geometry, not duplicated computation.

NCU's per-opcode sum closes to its per-PC `inst_executed` sum. PPU uses the
ACU export's `warp_executed_count` for every PC, totaling 337,494,189,592;
the separate ACU aggregate is 337,941,719,752 and is deliberately not substituted.
See [NCU instruction metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#instructions-per-opcode-metrics).

The parser prints its full opcode membership for integer/bit/select: arithmetic,
bitfields, comparisons and selects on integer/word operands; excludes moves,
loads, shuffles, numeric casts, floating arithmetic, MMA, waits and branches.
`FSEL` is included as NVIDIA's word-selection counterpart. Do not treat this
software category as an identical hardware execution-pipe metric.

### Measured warp instructions per K64 visit

| Category | PPU post2 INT8 PV | NVIDIA FP8, FP16 buffer | NVIDIA FP8, FP32 |
|---|---:|---:|---:|
| Integer / bit / select | 566.20 | 62.32 | 62.31 |
| Cross-lane shuffle | 80.00 | 8.01 | 8.01 |
| Explicit numeric conversion | 288.12 | 96.06 | 96.06 |
| MMA (different atom shapes) | 64.00 | 128.00 | 128.00 |
| Total, including NOP | 2264.71 | 986.05 | 845.96 |
| Total, excluding NOP | 2257.71 | 908.09 | 777.98 |

The NVIDIA FP16-buffer arm additionally uses **128 `HADD2.F32` per visit**:
it combines conversion of a half value with FP32 accumulation. That operation
is not included in the explicit-conversion row. The direct-FP32 arm avoids
this buffer; neither comparison hides it inside a generic INT8/FP8 label.

## Where to work next, without mixing causes

1. **P packing:** native FP8 uses 32 packed `F2FP...E4M3.F32` instructions per
   visit for 64 probabilities/lane, not scalar clamp/shift/OR chains. PPU post2
   uses 64 F32-to-S32 conversions, 64 min + 64 max, byte packing and widening.
   This is the first avoidable cost; it is not intrinsic dot-product work.
   The existing local candidate removes the clamps and reduces packing ALU,
   but its dynamic counts are not measured in the uploaded PPU report.
2. **Register placement:** NVIDIA's score-to-FP8-A mapping needs no P cross-lane
   shuffle in this geometry; its roughly 8 shuffles/visit belong to reductions.
   PPU post2 has 64 P shuffles plus 16 row-reduction shuffles. The candidate's
   two-stage byte butterfly halves the P shuffles (64 to 32). This difference
   comes from the real C/A layouts, not from an integer-versus-floating label.
3. **PV partial restoration:** PPU adds 128 S32-to-FP32 conversions per visit
   beyond the 64 QK conversions common to both devices. Its per-K64/channel V
   scale and local-P scale also require partial scaling. NVIDIA keeps V scale
   per channel across the sequence and applies it at the end; direct FP32 PV
   already produces float accumulators. Moving our scale to the end changes
   the quantization contract and needs its own accuracy experiment.
4. **General addressing/control/moves:** retain this as a separate account.
   Neither equal 8-bit operand width nor equal MMA work guarantees identical
   integer address generation across PPU and NVIDIA. The full opcode table is
   preserved instead of attributing all 9.09x to requantization.

The inverse-dequant / magic-bias byte-extraction idea belongs to item 1 on the
experimental branch. It cannot simply reinterpret an arbitrary float as U8.
Keep the existing multiply then round-to-nearest-even semantics. A host IEEE
threshold-neighbour probe gives separate-multiply/add `0/765` mismatches, but
fusing multiply+bias changes `128/765`; that fused candidate is not admitted.

The local optimized PPU static integer/bit count `1136 -> 832` is a different
measurement (whole compiled body). It must not be subtracted from the measured
dynamic 566.20 above or reported as a new PPU speedup.

## Reproduce the NVIDIA reference

Use the unmodified upstream checkout and its built SM120 extension. The driver
hash used for these reports is
`6e8e9c4394f5bae90d599ce65aec6e75bc6ee47cc75c5d9d36716c5dd34350f1`.
No global package install or changes to the upstream implementation are needed.

```bash
mkdir -p /workspace/sage-native-reference
PYTHONPATH=/path/to/upstream /usr/local/cuda-12.8/bin/ncu \
  --target-processes all --clock-control none --cache-control none \
  --nvtx --nvtx-include 'sage_native_profile/' \
  --metrics sass__inst_executed_per_opcode \
  -o /workspace/sage-native-reference/fp8-fp16-buffer \
  python /path/to/experiment/dev/ppu_int8/profile_native_sage_ncu.py \
    --seq 73774 --heads 56 --dim 128 --pv fp32+fp16
# Separate run: change --pv to fp32 and use another report filename.
```

Export with `ncu --import REPORT --page raw --csv --print-metric-instances details`.
Do not use `values`: the opcode/PC correlation IDs are required. This NCU
version does not accept `--units`; no unsupported flag is part of the command.
The runs disabled clock/cache control and required two replay passes with host
memory backup. Instrumented duration is not a normal latency measurement.

```bash
python dev/ppu_int8/compare_native_sage_opcodes.py \
  --nvidia-raw dev/ppu_int8/results/5070-fp8-reference/*-raw.csv \
  --ppu-sass /workspace/sage-h3-int8-acu-analysis/sass.json \
  --out /workspace/sage-native-reference/comparison.json
```

Wrong aggregate sums, duplicate correlation IDs and a missing MMA instruction
must reject before reporting ratios. The raw NVIDIA exports and the PPU PC
export are independent measured inputs, not reconstructed instruction models.
