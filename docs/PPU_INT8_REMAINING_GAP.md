# Remaining INT8-PV / native FP8 differences: a hot-loop account

2026-09-17; reporting and offline arithmetic witnesses only. No production
code, quantization rule, default FP16-PV route, or `a49338f` candidate binary
changed. No new PPU or NVIDIA launch. This follows the question whether
INT32-to-FP32 conversion exists on **both** sides: yes, their QK conversion is
common work and must cancel before attributing any PV difference.

## Identity and independent anchor

Use the same H3 work as `PPU_NATIVE_FP8_INSTRUCTION_REFERENCE.md`:
B1/H56/S73774/D128/full, Q128/KV64, four warps, 129,248 warps and 1,153 K64
iterations = 149,022,944 warp/iteration visits. This is not equal-precision
or byte-identical-input benchmarking. NVIDIA's measured source remains
`d1a57a5`; current PPU candidate source remains `a49338f`.

The old PPU native body matches the independent uploaded ACU export at all
**4,046 PCs, opcodes AND operands**, after removing only symbolic branch labels
and whitespace. All five extra ACU PCs executed zero times. The live-PC sum
closes exactly to **337,494,189,592**. This anchors the parser on observed
code; it is not another mathematical implementation of attention.

`native_isa.py` bounds each kernel at the next ELF section and follows its
real branches. It excludes unreachable padding, not arbitrary NOPs. An early
exit cannot hide a live target after it. Its seven tests include both cases.
The old static totals contained 37 dead/foreign records; the corrected
post2-to-candidate totals are 4046 -> 3455, not 4083 -> 3492. The integer
category and absolute optimization deltas are unaffected.

## What is actually inside the current K loop

`account_hot_loop.py` derives the strongly connected component containing
both MMA families from the generated code, not source-loop indentation.
There is one entry and no nested cycle after cutting its backedge. Cold code
is acyclic. Direct execution-mask branches are then-only regions that rejoin;
each K64 path contains exactly 32 QK and 32 PV MMAs.

| Current candidate category | Whole reachable body | Once-only union | Loop union | Single CFG walk min..max |
|---|---:|---:|---:|---:|
| Integer/bit/select | 663 | 365 | 298 | 93..298 |
| All instructions | 3455 | 1402 | 2053 | 1636..1872 |
| Shuffle | 48 | 0 | 48 | 48..48 |
| S32 -> F32 | 252 | 0 | 252 | 192..192 |
| F32 -> S32 | 64 | 0 | 64 | 64..64 |
| FP32 multiply | 583 | 128 | 455 | 395..395 |

Union counts include alternative edge/masked code. For example, the 252
S32 conversion sites are **not** 252 executed casts per iteration: 60 sites
are alternative copies. Likewise, 663 is not the candidate's dynamic integer
count. The minimum path is an inventory, not a measured device lower bound;
masked regions can execute on only some warps. A conservative CFG-derived
integer upper bound, including all cold sites once, is 298.317/visit. Actual
candidate PC weights and latency still require its ACU run.

The static integer category additionally includes `s.not`, `s.mull`,
`s.mulh`, unlike the original dynamic comparison's category. In the old
report these are all once-only: just 0.011275/visit. The historical measured
566.1967 remains unchanged; the broader static-scope PC account is 566.2080.
Neither number is a measurement of the optimized candidate.

## Four remaining mechanisms, kept separate

### 1. Common QK casts cancel; PV restoration must be counted as a chain

Per warp/K64:

| Work | PPU integer PV | Native FP8, FP16 buffer | Native FP8, FP32 |
|---|---:|---:|---:|
| QK S32 -> FP32 | 64 | 64 | 64 |
| PV S32 -> FP32 | 128 | 0 | 0 |
| PV half restore + FP32 add (`HADD2.F32`) | 0 | 128 | 0 |

Both QK and integer-PV dot products are bounded below 2^24 here, so their
S32-to-FP32 casts are exact. But bit reinterpretation is still not conversion.
It is wrong to charge all 192 PPU casts as extra, and wrong to call native
FP16-buffer PV restoration free. These casts are excluded from the old 9.09x
integer/bit/select statistic anyway; they cannot explain that statistic.

### 2. P representation and fragment ownership

For 64 probabilities/lane, PPU preserves `RN(p*255)` followed by RNE and
saturated U8 packing: 64 multiplies, 64 conversions, 32 native pack operations.
Native uses 32 packed E4M3 conversions after scaling its exponent range.
The prior inverse-magic experiment found no shorter bit-equivalent PPU pack.

The current PPU C-to-A mapping still needs **32 byte permutations plus 32
cross-lane shuffles** per visit. It already replaced the old 64 shuffles and
their mask/shift/or chains. Native needs no P cross-lane exchange for its
MMA layout. This is an actual ownership/representation difference, not an
unsupported integer MMA emulated with floating MMA. Any further remap must
retain the independent real-traits basis/roundtrip tests.

### 3. Denominator reduction is unnecessarily inside our K loop

PPU source `qk_int_sv_f16_ppu.cu` reduces each lane-local `tile_sum` twice with
XOR shuffles before updating `row_sum`. Four rows/lane give 8 sum shuffles per
K64, in addition to 8 maximum shuffles.

Native `attn_utils.cuh::accumulate_d` keeps lane-local denominator state.
`normalize_d` reduces it once, after the K loop. The measured template selects
`ComputeUnit::kCudaCore` (enum 1), **not** tensor-core denominator accumulation.
Both native NCU modes close exactly to:

`8 * 149,022,944 + 8 * 129,248 = 1,193,217,536` shuffle instructions.

Deferring our sum reduction can remove **1,191,149,568** warp shuffles over
this launch, plus corresponding adds. Candidate shuffle work would move
from 48/visit toward 40.00694, not to native's 8.00694: P retiling remains.
This is source/codegen opportunity, not measured acceleration.

The mathematical recurrence allows deferral because maximum/rescale is
common across row peers. **FP32 arithmetic order does not remain identical.**
With 1,153 blocks, lane sums `[16, 2^-20, 0, 0]`, and rescale=block_scale=1,
the actual host witness prints eager `18448 / 0x46902000` versus deferred
`18448.001953125 / 0x46902001`. One-block and exact-integer controls agree.
Therefore this needs its own numerical admission; it was not silently folded
into the bit-preserving candidate.

### 4. Scaling and FMA structure, not just `mul` counts

Native `update_mdo` reduces unscaled scores, scales row maxima, and computes
score*scale-origin with FMA before exp2. Our code materializes all scaled
scores before the reduction, then subtracts the origin. A native-style
form can replace 64 multiplies + 64 subtracts with 64 FMAs + four row-scale
operations (roughly 60 arithmetic instructions/visit saved).

The rounding witness is concrete: score=1001, scale=FP32(0.1),
origin=RN(score*scale). Separate multiply/subtract gives zero; FMA gives
`0x364a8000` (about 3.02e-6). This disproves raw-equivalent intermediates,
not acceptable model accuracy. A new numerical gate is required.

V granularity is also different: native applies its sequence/channel V scale
at the epilogue; ours applies K64/channel V scale and local-P scale to each
partial. Moving either outside K without changing the representation is
invalid. Current PPU codegen already fuses output rescale with partial addition;
do not count the source's rescale multiply again as an extra emitted multiply.

Measured **old** PPU / native instruction sums for the whole mul/add/FMA/half-
restore chain are **671.111 / 395.338 / 267.338 per visit**, respectively.
Native FP16-buffer uses 200 FFMA and only ~2.33 FMUL; direct-FP32 uses 72 FFMA
and ~130.33 FMUL. Both implement output rescaling. Comparing PPU's 395 FMUL
against native's 2.33 FMUL would therefore be misleading. These chain totals
are instruction counts, not equal per-instruction costs across devices.

## Next experiment and unchanged boundaries

First obtain the pending `a49338f` PPU result to replace static bounds with
measured counts; main remains PV FP16. Of the newly identified changes, test
**deferred denominator alone first**: it is a small, attributable reduction
in cross-lane work, without changing the P/V quantization format. Register
error bounds before running it; retain the exact rounding witness and the
existing numeric fixtures. FMA score formation comes second. Changing P/V
scale granularity is a separate accuracy experiment, not the same patch.

No claim that these changes close all FP8 overhead, remove every ALU, or win
latency. The original 9.09x is a post2 measurement, not a verdict on a49338f.

## Reproduce without a GPU

```bash
mkdir -p /workspace/sage-int8-hot-account-20260917
python dev/ppu_int8/test_native_isa.py
python dev/ppu_int8/account_hot_loop.py \
  --before /workspace/sageattention-all-int8-final/shipping-isa.log \
  --after /workspace/sage-int8-alu-closure-20260917/final-a49338f/sdk/shipping-isa.log \
  --acu /workspace/sage-h3-int8-acu-analysis/sass.json \
  --reference dev/ppu_int8/results/5070-fp8-reference/comparison.json \
  --out /workspace/sage-int8-hot-account-20260917/account.json
python dev/ppu_int8/test_hot_account.py \
  --before /workspace/sageattention-all-int8-final/shipping-isa.log \
  --acu /workspace/sage-h3-int8-acu-analysis/sass.json \
  --reference dev/ppu_int8/results/5070-fp8-reference/comparison.json
c++ -O2 -std=c++17 -ffp-contract=off \
  dev/ppu_int8/softmax_reorder_witness.cpp \
  -o /workspace/sage-int8-hot-account-20260917/softmax-reorder-witness
/workspace/sage-int8-hot-account-20260917/softmax-reorder-witness
```

Seven hot-account negatives reject missing PC, wrong register at a matching
opcode, missing measured/generated MMA, nested cycle, wrong specialization,
and changed reference denominator. The two rounding witnesses must remain
red to a claim of bit equivalence. Result JSON is hash-bound; archived raw
reports remain unchanged.
