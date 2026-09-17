# PPU QK/PV eight-bit forward contract

This is the opt-in `experiment/ppu-pv-int8` branch, not the mainline algorithm.
Main retains INT8 QK / FP16 PV. This branch preserves the integer-PV prototype,
its post2 baseline, and the locally verified requant/byte-permutation reduction;
the latter has no PPU timing or model-quality admission yet.

Upstream scope: the original SageAttention paper's Table 6 includes all-INT8
SAGEAttn-vB/vT and section 4.5 selects them only on sufficiently accurate layers.
The current Sage2 snapshot inspected for the NVIDIA comparison does not expose
a ready INT8-PV entry point. These are different statements; neither FP16 nor
FP8 PV is an all-INT8 reference. See the
[original paper](https://arxiv.org/html/2410.02367v4#S4.S5).

Baseline: source `0e30f92`, QK S8xS8->S32, PV F16xF16->F32.
The upstream SM80 path uses this mixed-precision algorithm; upstream SM89/90
also supplies S8 QK + FP8 PV. The latter is not the current PPU implementation.
This work implements an explicit PPU integer PV, not an automatic FP fallback.
The initial port followed the already-understood SM80 algorithm to establish
correct PPU layouts; it did not establish that INT8 PV was impossible. Upstream
precision paths are documented in [SageAttention](https://github.com/thu-ml/SageAttention)
and [its dispatch](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py).

## Fixed numerical rules, before implementation measurements

- Q/K quantization is unchanged (Q per 32 rows, K per 64 rows).
- V is symmetric signed INT8, per (batch, KV head, K64 block, output channel).
  `s_v = max(amax(abs(V)), 1e-7)/127`, round-to-nearest-even, clamp [-127,127].
  The quantizer also packs V into `[B,Hkv,ceil(N/64),D,64]`, padding tail K
  with zeros. This is required because the current PPU0010 TSM atom has no
  INT8 transposed-read implementation. The attention reader uses the already
  supported non-transposed INT8 load, just as QK does. No floating MMA fallback.
- For each query row in each K64 block, use its local score maximum `t`:
  `p_u8 = round_even(255 * exp2(score_log2 - t))`, clamp [0,255]. Masked
  probabilities are zero. Local scaling preserves small-probability blocks in
  long sequences; a single fixed quantum for globally normalized P is rejected.
- Online softmax retains FP32 max/denominator and output. If `m` is the new
  running maximum, integer PV is dequantized by `exp2(t-m)/255 * s_v`.
  The denominator uses unquantized exponentials, not the sum of U8 codes.
- QK uses actlize S8xS8->S32. PV uses actlize U8xS8->S32 over a whole K64 block,
  then converts that exact integer partial to FP32 before scaling/accumulation.
  `64*255*127 = 2,072,640 < 2^24 < 2^31`: neither INT32 overflow nor rounding
  in this integer-to-FP32 conversion is possible. Do not carry INT32 across
  blocks with different V scales or online-softmax maxima.
- Convert two score-C fragments directly into the U8 MMA-A fragment by packed
  integer shuffles. No FP16 MMA is permitted as a layout bridge in this arm.

"All INT8" refers to both matrix products (P is unsigned); it does not mean
integer softmax, integer scale factors or integer accumulation across KV blocks.
Original FP16/BF16 inputs and output types, layouts, GQA, tails and causal scope
are retained. The named FP16-PV API remains an explicit reference arm.

## Pre-registered local/device admission

1. Exhaust the real actlize S8/U8 MMA traits: all 512 probability bytes in a
   16x32 fragment; unique coordinate tags and all 256 byte values. A wrong
   source lane, a one-bit byte permutation, a missing half and signed treatment
   of P>=128 must turn red. Packed V must reconstruct every source coordinate,
   including its zero-padded tail; wrong channel/block scale must turn red.
2. Independent dense integer matmul/FP32 softmax oracle, both causal/noncausal,
   D64/D128, HND/NHD, GQA, positive and signed V, zero V, nonuniform block and
   channel ranges, and tails. The low-level integer dot/layout gates are exact;
   float end-to-end error is separately reported, never called raw-bit equality
   against BF16 or the old FP16-PV algorithm. Fixed seeded distribution tests
   require relative RMSE <= 0.02 against unquantized attention; cancellation
   fixtures also report absolute error instead of a misleading relative ratio.
3. Native SDK full-body build, no private NVIDIA/PTX shortcuts, no spill; inspect
   generated all-INT8 symbols for S8 and U8/S8 MMA, and absence of every floating
   MMA. Host mapping/CPU numeric results are not a PPU device numeric verdict.
4. Box: quantizer and quantized-operand oracle first; replay stability; then
   same-input core and prepare+core timings against the retained FP16-PV and
   BF16 FA arms. Quantization setup is excluded only from the named core span.
   Report error and latency together. No promised speedup before measurement.

## Work/space accounting (D128, Q128/KV64, four warps)

Per warp/K64: QK stays 32 S8 MMA; PV becomes 32 U8/S8 MMA instead of 64 F16 MMA;
the eight floating bridge MMA become integer register permutations. Shared Q
16 KiB + K 8 KiB + V 8 KiB = 32 KiB (old 40 KiB), before any explicit scale
staging. Additional work: P quantization/permutation, per-block integer partial
conversion and V scales. No pipeline patch or double buffering is bundled in.

For B1/H56/S73774/D128 full attention: Q tiles 577, K tiles 1153, CTAs 32312.
QK and PV each predict 4,768,734,208 integer MMA; floating MMA predicts zero.

## Local evidence / remaining device boundary

- Real-SDK host oracle: map `0/512`, packed words `0/32768`, V transpose/tail
  `0/73728`. Wrong lane, missing half, wrong bit, untransposed V and signed P
  all turn red. QK and integer PV share the real actlize B/C traits.
- CPU logical-matrix oracle: 24 fixed cases, worst relative RMSE `0.01548991`.
  H3-length `S=73774,D128`: **one head, five sampled query rows**, relative RMSE
  `0.01486595`. This is not exhaustive H3 attention or model-output validation.
  Four zero/cancellation cases in HND/NHD use exact absolute zero, not RMSE.
- SDK 2.1.1 real `ppu_10` build: all 48 attention/QK-quant/V-quant kernels have
  zero private stack. All 16 integer-PV attention specializations have **zero
  floating MMA**. D64 static QK/PV counts are `16/16`, D128 `32/32`. Injected
  floating MMA and a missing specialization both reject. D128 integer-PV uses
  248 vector registers (causal) or 250 (full); no occupancy claim is inferred.
- Native runtime imports, installed-wheel ABI/hash checks and CPU-tensor
  rejection are local checks. **PPU numeric correctness and performance are
  NOT RUN.** Random-input RMSE does not establish diffusion/model quality.

Reproduce the local numeric and layout gates (no GPU work):

```bash
python dev/ppu_int8/check_all_int8_numerics.py
python dev/ppu_int8/check_all_int8_api.py
PPU_SDK=/path/to/PPU_SDK bash dev/ppu_int8/run_all_int8_layout.sh
```

An older host glibc can supply `PPU_HOST_LOADER` and
`PPU_HOST_LIBRARY_PATH` for the SDK host oracle. Missing SDK is an explicit
SKIP, not a passing device compile. `run_sdk_compile.sh` supplies the actual
shipping device-symbol/ISA checks separately.

## Box: install, admit, then profile

The artifact branch retains post2 for this experiment, while its default
`release.json` / installer selects the mainline post1 FP16-PV wheel. Post2 does
NOT contain the later requant ALU reduction. Select the experimental release
explicitly; never infer the algorithm from the package name alone. On this
experimental source branch the default PPU API uses INT8 PV;
`sageattn_qk_int8_pv_fp16_ppu` remains the named old precision mode.

```bash
git -C /workspace/ppu-wheel-sage pull --ff-only
bash /workspace/ppu-wheel-sage/install.sh --experimental-pv-int8
git pull --ff-only
PROFILE=1 bash tools/run_ppu_all_int8_box.sh
```

Run the last two commands in the SageAttention source checkout. The runner
executes the **installed wheel**, never copies an old in-tree `.so`, does not
compile, and submits ACU only after the numeric admission passes. Restart a
Python/ComfyUI process that already imported the old wheel. It checks exact
uniform-P/signed-V witnesses, quantized-operand oracles, tails, GQA, output
dtypes/layouts, causal/full and eight-repeat raw-bit stability. The ACU target
is B1/H56/S73774/D128/full, prepared INT8 Q/K/V, one profiled call.

For measured core and preparation-inclusive latency against BF16 FA, use the
same benchmark and select `--pv int8` or `--pv fp16`; the JSON records that mode:

```bash
bash tools/run_ppu_sage_bf16_ab.sh --pv int8
bash tools/run_ppu_sage_bf16_ab.sh --pv fp16
```

Do not equate fewer MMA instructions with a guaranteed speedup. Quantizing P,
packing/permuting operands, converting S32 partials and reading V scales cost
instructions too; prequantized core and prepare+core must be reported separately.
