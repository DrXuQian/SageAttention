# Integer PV: attributable overhead, not a missing INT8 instruction

Work is isolated on `experiment/ppu-pv-int8`. Main and its default wheel keep
QK INT8 / PV FP16. This document extends `PPU_REQUANT_ALU.md`; the measured
5070 reference is `PPU_NATIVE_FP8_INSTRUCTION_REFERENCE.md`. No new PPU device
time or ACU result has been inferred from local compilation.

The 5070 native FP8 path is a useful structural reference, but replacing its
MMA datatype is not the only difference in the current integer-PV algorithm:

| Source of work | PPU integer PV | Native NVIDIA FP8 PV | This round |
|---|---|---|---|
| P packing | RN float multiply by 255, RNE to integer, U8 pack | packed saturating FP8 conversion | keep the shorter actlize saturated pack |
| Score C -> PV A ownership | four source lanes per packed word | native FP8 register delivery | earlier two-stage byte butterfly retained |
| P mask | a predicate/select for every P value | different native masked exponent path | one exponent-origin classification per row |
| PV scale addressing | per K64 block/channel, repeated by each warp | whole-sequence/channel scale | cooperate within CTA, without changing scale values |
| PV accumulation | S32 partial -> FP32, scale each K64 block | floating-point MMA accumulator | keep one exact integer-to-float cast per partial |

The emitted integer specializations have zero floating MMA instructions. The
extra ALU does **not** mean the compiler secretly implements unsupported INT8
MMA with FP16 MMA.

## Fixed arithmetic, concrete removals

`RN_float(p*255)` then round-to-nearest-even must remain unchanged. A raw byte
pick from an ordinary float is not that operation: for example `p=0.25` has
zero low byte, but its quantized code is 64. Magic-bias packing is valid only
after the correct multiply and rounding. Exhausting all 1,065,353,217 FP32
bit patterns from +0 through 1, plus -0, proved the separate-multiply variant
equivalent. In the native probe it costs 23 instructions versus 22 for the
current saturated pack. Fusing the multiply/bias makes 128 of 765 rounding
neighbours differ. Neither variant is adopted.

The S32 partial is bounded by `64*255*127 = 2,072,640 < 2^24`, so its numeric
FP32 conversion is exact. This is nevertheless a **conversion**, not a bit
reinterpretation. The native cast is one instruction; integer magic-bias plus
floating subtraction would require two. Moving conversion/scaling outside
the K loop would change the result because V scales vary by K64 block.

For masking, valid scores are <= the row maximum. With threshold `-1e29`, a
masked finite score and a valid row maximum are separated by at least
`9.44473e21`: the exponent is unconditionally zero. All-masked rows instead
use zero as the exponent origin, avoiding `exp2(s-s)=1`. The production helper
passes 1,065,024 raw-bit host pairs; that all-masked mistake is a live negative.
The real SDK probe changes four predicates/selects to one, with the same four
subtracts and exponentials.

Value-scale staging uses one float/channel after the V payload. Its owner is
the linear CTA thread ID, and its source is the quantizer's declared
`[batch, kv_head, K64_block, channel]` tensor. It is published by the existing
issue-V barrier and reused by all four warps. The next loop's existing barrier
protects it from overwrite. No additional barrier or new quantization format.
The host executes the actual helper for 69,600 blocks, 6,681,600 publications,
and 213,811,200 consumers. Missing owner, wrong head pitch, payload overlap,
and wrong consumer column are all red.

## Native instruction accounting

SDK 2.1.1, `ppu_10`, D128/noncausal/BF16-output/no-LSE/INT8-PV specialization.
Counts are static instructions of the complete emitted body, not ACU dynamic
counts or predicted latency. The category definition remains fixed in
`dev/ppu_int8/check_requant_codegen.py`.

| Complete-body count | Original post2 | Pack + butterfly | + row mask | + V staging |
|---|---:|---:|---:|---:|
| Integer / bit ALU | 1136 | 832 | 772 | 663 |
| All instructions | 4083 | 3709 | 3592 | 3492 |
| Shuffles | 80 | 48 | 48 | 48 |
| All explicit conversions | 485 | 453 | 453 | 448 |
| FP32 multiplies | 583 | 583 | 583 | 583 |
| QK / PV MMA | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |
| CTA barriers | 5 | 5 | 5 | 5 |
| Vector registers | 250 | 250 | 250 | 246 |
| Scalar registers | — | 112 | 112 | 128 |
| Private stack | 0 | 0 | 0 | 0 |

V staging replaces 32 per-channel global load sites with one cooperative
load site plus 32 shared reads and one shared store. The complete body's
`vmem.ld.b32` count is 33 -> 2 because the K-scale load remains. Address-op
deltas include carry-add pairs -21/-21, wide unsigned MAD -20, wide shift -21,
OR -22; `s.wait` -21. Five removed conversions are **address widening**, not
PV numeric casts. H3 staging costs 512 additional shared bytes and 16 extra
scalar registers. These costs are reported, not hidden behind the lower
vector-register count.

Staging is **not universal**: the initial all-format SDK census showed D64
integer ALU +3/+6 and causal D128 +1; those compiler paths already hoist more
address work. That candidate was rejected for those geometries. Only D128
noncausal selects staging; other INT8 paths keep their original V delivery.
All 16 FP16-PV native streams remain unchanged. This is a static candidate
selection, not an assertion that the chosen geometry is faster on device.

## What remains different

Compared with the published post2, the composed target removes 473 static
integer/bit instructions (-41.64%) and 591 total instructions (-14.47%). It
does not establish NVIDIA instruction parity. PPU C/A byte ownership, U8
packing instead of FP8 encoding, S32 restoration, and block-local V/P scales
remain distinct. Changing scale granularity or exponent-based quantization is
a separate accuracy experiment, not a byte-equivalent cleanup.

The next PPU run must first pass unchanged device numeric/replay criteria,
then profile this exact binary against post2 on the same H3 work. Record both
normal core latency and per-opcode ACU counts. Static ALU reduction with flat
or worse latency is an admissible outcome; do not promote integer PV to main
on the strength of this document. Main's FP16 default is unaffected.

The execution-only handoff is `tools/run_ppu_int8_alu_candidate_box.sh` on the
experimental branch. It verifies the dedicated candidate binary, stages a
private Python import directory beneath `OUT`, runs the unchanged numerical
gate, and optionally profiles H3 with `PROFILE=1`. It does not install a wheel,
compile on the box, or change the user's FP16-PV default. Select the
`qk_int8_pv_kernel<128,false,false,bfloat16_t,true>` report, not preparation
kernels, for the before/after opcode comparison.
