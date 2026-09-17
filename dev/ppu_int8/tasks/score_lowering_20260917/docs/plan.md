# Score lowering screen (before any production edit)

Parent 41f68d5, experiment/ppu-pv-int8. Main and both published experimental
binaries remain immutable. User reports about 570 ms for the preceding run,
still slower than FP16; source/report identity and ACU/event timing scope are
not yet supplied. Do not bind that number to a candidate or claim a speedup.

Priority: H3 B1/H56/S73774/D128/full, Q128/KV64/128 threads. Keep the existing
O atol=.002/rtol=.01, LSE atol=.002/rtol=.001, original-input relative RMSE
<=.02 and eight same-variant bit-stable replays. No changed P/V representation,
clipping, FP16-PV changes or relaxed numerical admission.

Screen two score ideas locally, before touching the kernel:

1. Native-style score FMA. Test equal-logit/constant-V witnesses, not only
   random attention. P must be <=1 before U8 saturation, or numerator and
   denominator no longer describe the same probability. A rounding difference
   alone is not a rejection; exceeding the existing O tolerance is.
2. Integer-MMA biased accumulator. Exhaust the full D128 signed-INT8 score
   range [-2097152,2097152]: seed 0x4b400000, reinterpret as FP32, then subtract
   12582912 must exactly recover the integer. This proves a representation,
   NOT an attention implementation or a speedup. Negative seed and missing
   bias controls must fail. Screen difference-before-scale separately against
   the unchanged logical oracle; exact representation does not prove exact
   scaled-logit rounding.

Compile small probes with the real SDK only if their arithmetic passes; count
the whole helper, not only removed conversions. Any promoted kernel needs a
clean full-body build, unchanged FP16 streams, zero spills and device latency.
Do not adopt an instruction reduction at the expense of speed.

Measurement gap: add a core-only INT8-PV/FP16-PV paired event measurement to
the existing isolated execution runner. Same original inputs, quantize once,
alternating arms on one stream, no ACU in the timed process. Keep sample
envelopes and UNRESOLVED ties; preparation is excluded explicitly. The two
paths differ in V/P quantization, so this is not a raw-equivalence comparison.
No device result can be established locally.

Additional offline-only representation screen: native FP8 puts its sequence/
channel V scale after K. Test that same scale granularity with our INT8 codes,
without changing physical packing, P, Q/K or the oracle. Include both the
existing random/error cohort and a causal prefix whose future K64 block has
a large V outlier. The current block-local quantizer must pass that witness;
if sequence-global INT8 loses the prefix, reject it as an unconditional port.
This is not authorization to weaken the error gate or change the shipped V
representation. Production/kernel modifications remain zero in this screen.
