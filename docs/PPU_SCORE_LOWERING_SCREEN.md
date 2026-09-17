# After the user-reported 570 ms: local screens, not a device verdict

2026-09-17; parent 41f68d5, experimental INT8-PV branch. The roughly 570 ms
observation is encouraging but still slower than the user's FP16 control.
Its binary/report and ACU versus event timer are not yet identified. Do not
assign it to a49338f or 567bba7, mix it with an older timer, or declare FP8
instruction parity. This checkpoint changes **no shipping kernel, quantizer,
main route, or published binary**.

## A cheap port of FP8 score FMA is not numerically safe

The constant-V anchor is independent: equal logits and V=1 must give O=1.
The current path rounds score*scale first, then subtracts that same maximum;
its P is exactly 1. Fusing the product/subtraction can instead leave a small
positive residual. The U8 numerator saturates at 255 while the unquantized
denominator still sums the probability greater than 1.

Actual host witness, compiled without implicit FP contraction:

| Quantity | Value |
|---|---|
| Integer QK score | 1001 (within the admitted INT8 dot range) |
| Positive scale | `0x1.99999ap+9` |
| FMA residual | `0x1.95p-6` |
| Probability / saturated P code | 1.01728165 / 255 |
| Constant-V output | **0.98301193**, expected **1** |
| Existing O gate | atol .002 + rtol .01: **fails** |

138/1025 equal-logit cases in this deliberately wide finite-scale screen
fail that existing tolerance. This is not the error rate on the H3 fixture.
It rejects **unconditional** replacement, not every possible compensated or
guarded FMA design. A raw-rounding difference alone was not the rejection.

## Removing QK casts is representationally possible, not yet admitted

Seed the actual S32 MMA accumulator with `0x4b400000`. For all **4,194,305**
integer scores in [-2,097,152, 2,097,152], including signed input code -128,
interpreting `(seed + score)` as FP32 and subtracting 12,582,912 recovers the
integer exactly. Wrong seed and wrong bias negatives each fail all values.

The real SDK2.1.1/ppu_10 microprobe uses the production actlize integer MMA,
four K32 steps and four N16 fragments. It exports all 32 per-lane probabilities
and both row maxima so the latter's restoration cannot be optimized away:

| Native microprobe | Current | Biased difference-first |
|---|---:|---:|
| S32 -> FP32 casts | 32 | 0 |
| FP32 add / multiply | 32 / 32 | 34 / 34 |
| Integer MMA / exp2 / shuffles | 16 / 32 / 4 | 16 / 32 / 4 |
| Whole reachable static instructions | 298 | 273 |
| Vector/scalar registers | 62 / 80 | 62 / 80 |
| Private stack | 0 | 0 |

The net reduction is **25**, not 32: row restoration and compiler scheduling
must be counted. This is an unmasked positive-scale microprobe, not an H3 loop
or a latency measurement. Its difference-before-scale also changes rounding:
scores 2,097,152 and 2,097,151 at FP32(.1) give old delta -.09375 versus new
-.10000000149. Full attention tolerance, all-masked/negative-scale handling,
full-body codegen and speed still need admission. **No production adoption.**
Native negatives cover hidden casts, omitted MMA and omitted maximum restore.

## FP8's sequence V scale cannot simply be copied to INT8

An offline-only screen holds Q/K/P and physical V packing fixed, changing
only V's scale granularity from K64/channel to sequence/channel. Both use the
existing independent logical attention oracle and relative-RMSE <=.02.

All 16 sampled random D64/D128, N65/257/4096/73774, causal/full cases pass.
But a second K64 block with V=16384 erases the preceding ordinary V=1 when
the entire sequence shares one INT8 scale. The current K64-local quantizer
preserves it:

| Witness | K64-local INT8 | Sequence-global INT8 | Independent answer |
|---|---:|---:|---:|
| Causal query before the large-value block | 1 | 0 | 1 |
| Full attention assigning negligible weight to that block | 1 | 3.63e-16 | 1 |

This is a finite BF16 input, with no overflow or unsupported shape. Both
counterexamples fail by 100% relative error; random-only testing would have
missed them. Therefore moving V scale completely out of K is **not an
unconditionally safe INT8 optimization**. This is not a claim that calibrated
or guarded variants can never work, and not a claim that all instruction gaps
are unavoidable.

## Compare speed to the retained FP16 path, not just instruction counts

`tools/benchmark_ppu_pv_core.py` measures INT8-PV and FP16-PV entrypoints from
the same identified DSO, on the same BF16 Q/K/V. Q/K quantized buffers are
shared. Preparation is done once, outside events. The arms alternate on the
default stream, with two warmups, seven samples and one H3 launch/sample by
default. Finite preflight and sampled raw-bit replay are mandatory. Separate
`device_all_int8.py` remains the independent correctness authority.

Report the raw event samples, binary hash, shape, device, median and envelopes.
Disjoint envelopes decide INT8-FASTER or FP16-FASTER; overlap is UNRESOLVED.
P/V precision differs, and these core times exclude preparation. This does
not require FlashAttention or a new wheel, and performs no compilation.

After updating the experimental branch:

```bash
git switch experiment/ppu-pv-int8
git pull --ff-only
CANDIDATE=deferred-denominator PROFILE=0 \
  bash tools/run_ppu_int8_alu_candidate_box.sh
```

Use `CANDIDATE=alu-candidate` for the earlier immutable binary instead.
`tools/run_ppu_int8_denominator_ab_box.sh` runs both candidates sequentially,
each with its FP16 control. Ordinary event measurement is separate from ACU;
`BENCHMARK=0` explicitly disables timing. This checkpoint has **no new PPU
device correctness or performance result**.

## Local reproduction

```bash
mkdir -p /workspace/sage-score-lowering-local
c++ -O2 -std=c++17 -ffp-contract=off dev/ppu_int8/score_lowering_screen.cpp \
  -o /workspace/sage-score-lowering-local/score-screen
/workspace/sage-score-lowering-local/score-screen
# These must return nonzero: zero-seed, wrong-bias, allow-fma.
python dev/ppu_int8/screen_global_v_scale.py
python dev/ppu_int8/test_pv_core_bench.py
```

The codegen TU is `dev/ppu_int8/score_lowering_codegen_probe.cu`, compiled with
the same ppu_10 hgcc flags/includes as `run_sdk_compile.sh`'s probes. Feed
its hgobjdump ISA to `check_score_lowering_codegen.py`; this never launches it.
