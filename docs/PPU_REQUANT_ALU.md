# PPU probability requant: local ALU reduction

Baseline is `fbf3d0feb1d356d7f4bad4713955de9a6e46f864`, all-INT8 QK/PV.
This is **local native code generation**, not a new PPU device timing or
correctness verdict. The original FP16-PV API and all its native instruction
streams remain unchanged. No candidate wheel has been published by this work.

This work lives on `experiment/ppu-pv-int8`; main retains FP16 PV. The next
packing investigation is an inverse-dequant / magic-bias byte extraction, not
an assumption that bit reinterpretation replaces numeric rounding. It must
preserve `RN_float(p * 255)` then RNE, masked zero, and all four packed bytes.
Combining the multiply and bias in an FMA changes threshold rounding and is
not admitted. Keep threshold-neighbour and byte-map negatives with any candidate.

The native RTX 5070 INT8-QK / FP8-PV path may be used as an instruction-structure
reference, explicitly not a same-precision performance comparison. Normalize
matrix work and separate P packing, PV accumulator conversion/scaling, register
retile, and control. Different integer / floating MMA geometries cannot be
compared by raw instruction count alone.

## Scope fixed before comparison

Only two byte-level seams change:

1. Keep `RN_float(p * 255)` followed by RNE-to-S32. Replace the four separate
   integer clamps and shift/OR assembly with actlize's saturated U8 vector
   converter. Native PPU lowering has two `v.pcnvt.b16.u8x2` instructions per
   four probabilities. No bias-FMA or exponent shift is substituted.
2. Replace each four-lane byte gather with a two-stage butterfly (xor 1,
   xor 2) and two `__byte_perm` calls. It presents exactly the same MMA-A
   bytes, without changing the quantized format or inserting shared staging.

QK/V quantizers, local-block P scaling, online softmax, mask handling, V-scale
granularity, FP32 PV accumulation, shared layout, MMA order and barriers are
unchanged. FP32 multiply/convert/exp counts are checked independently.

## Native SDK 2.1.1 results

Same `ppu_10`, Torch 2.9 / Python 3.12 / C++11 ABI 1 build. Before DSO SHA256:
`98c43ca4d30c16ef260114588b51ba9f81ef66dd37f5296f6b3de06bc7b4077f`.
Candidate DSO SHA256:
`6f47f372231a8bfad02df681817113877545f76a1ed7b727ed9971be858ca0e6`.

H3 specialization: D128, noncausal, BF16 output, no LSE, INT8 PV. Its runtime
shape is B1/H56/S73774; static ISA counts do **not** multiply by this shape.

| Static count, reachable complete kernel | Before | After | Change |
|---|---:|---:|---:|
| Integer/bit ALU | 1136 | 832 | -304 (-26.76%) |
| Shuffle | 80 | 48 | -32 (-40%); P part 64 -> 32 |
| Conversions, all types | 485 | 453 | -32 byte-widening conversions |
| All instructions | 4046 | 3672 | -374 (-9.24%) |
| FP32 multiplies | 583 | 583 | unchanged |
| QK/PV MMA | 32 / 32 | 32 / 32 | unchanged |
| CTA barriers | 5 | 5 | unchanged |
| Vector registers / private stack | 250 / 0 | 250 / 0 | unchanged |

The fixed integer/bit scope includes integer arithmetic, compares/selects,
bit operations and byte permute/saturating pack; it excludes shuffle, moves,
conversions, branches/waits and loads. `check_requant_codegen.py` defines it.
Counts include prologue, tail/epilogue and static paths that may not execute;
they are not the dynamic ACU opcode sums or latency predictions.

The totals were corrected on 2026-09-17 to exclude 37 dead/foreign records
per H3 body. See the parser erratum in `PPU_INT8_ALU_CLOSURE.md`; integer ALU,
the before/after instruction delta, source and binaries did not change.

Principal opcode deltas: `v.min.i32 -64`, `v.max.i32 -64`, `v.shll.b32 -97`,
`v.shrl.b32 -64`, `v.or.b32 -35`, `v.lop3.b32 -32`, `v.and.b32 -16`;
replacement `v.pcnvt +32`, `v.byte.prmt +32`. Shuffle changes from 64 indexed
gathers to 32 butterfly exchanges; the 16 softmax exchanges are retained.
There are also two extra compares/selects for lane selectors and 17 extra
vector moves: these are included in the totals, not hidden.

Per four-P probe, requant arithmetic goes from 21 to 10 instructions (4
multiplies + 4 RNE conversions remain). Including identical load/store/setup,
the standalone body goes 33 -> 22. Per four-word transpose probe the whole
body goes 83 -> 39, including selector setup, with 16 -> 8 exchanges.
These small-probe ratios must not be presented as whole-kernel gains.

## Local admission and reproducibility

- Real actlize MMA A/C traits: all 512 byte destinations, all 256 byte values,
  and the complete independent 4096-bit basis / 524,288 output-word checks.
- Wrong byte selector and missing one basis vector both fail. Existing wrong
  lane/half/bit/V-layout/signed-P negatives remain active.
- Native helper codegen uses the actual production functions. Reintroduced
  scalar clamp and extra gather negatives both fail.
- All 48 shipping kernels build with zero private stack; all 16 integer-PV
  specializations preserve MMA/conversion/exp/barrier counts. All 16 FP16-PV
  instruction/operand streams are unchanged (only BB label spelling ignored).
- Independent CPU numeric suite still passes 24 cases plus H3 sampled rows
  and cancellation/zero cases. This does not execute the PPU candidate.

```bash
PPU_SDK=/path/to/PPU_SDK bash dev/ppu_int8/run_all_int8_layout.sh
PPU_SDK=/path/to/PPU_SDK bash dev/ppu_int8/run_sdk_compile.sh
python dev/ppu_int8/check_requant_codegen.py \
  --probe /path/to/candidate/requant-codegen-isa.log \
  --before /path/to/post2/shipping-isa.log \
  --after /path/to/candidate/shipping-isa.log \
  --out /workspace/sage-requant-static-ab.json
```

Local artifacts: `/workspace/sageattention-requant-candidate`,
`/workspace/sageattention-requant-layout`,
`/workspace/sage-ppu-requant-analysis`.

## Why this does not yet make integer PV equivalent to native FP8 PV

Upstream NVIDIA's FP8 path folds QK scale/subtraction into FMA and the FP8 P
scale into an exponent offset; it converts/assembles operand registers with
packed saturation and applies sequence-wide V channel scales after the loop.
FP8 MMA accumulates floating output (some variants use local accumulator
buffers), unlike our per-block S32 partial followed by FP32 scaling. Those
differences remain. Moving scales or changing P rounding is a separate numeric
experiment, not bundled into this byte-equivalent reduction.

The requested original-Sage RTX 5070 NCU measurement is separate: report its
exact upstream SHA, kernel, shape, precision, GPU and warp-level dynamic opcode
counts. Different MMA atom shapes and FP8/S32 accumulator semantics must be
explicit when comparing with PPU. No NVIDIA device count has been inferred
from this PPU static result.
