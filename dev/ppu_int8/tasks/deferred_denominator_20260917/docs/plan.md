# Deferred INT8-PV denominator: one-variable experiment

Parent e33d68c; experiment/ppu-pv-int8 only. Immutable codegen/binary baseline
is a49338f (DSO dc50962536e79bf8ec636ece3c0d2a5070d8a416ae05d82270f11c0aee6e126c).
Priority work B1/H56/S73774/D128/full, Q128/KV64/four warps. Regression scope:
D64/D128, causal/full, BF16/FP16 output, LSE/no-LSE, tail and GQA.

Change ONLY the integer-PV denominator schedule: accumulate each lane's sum
through K, then XOR-1/XOR-2 reduce once before both O normalization and LSE.
Row maxima and both rescale factors are common to four peers. Keep QK/PV
arithmetic and K order, P/V quantizers/scales, masks, fragment mapping, AIU
delivery, scheduler and ABI. FP16-PV must remain native-stream identical.
No score-FMA change and no alternate V/P representation in this candidate.

Numerical contract registered before editing: reassociation is intentional.
Do not require old/new FP32 denominator bit equality or weaken the independent
attention oracle. Preserve device O atol=0.002/rtol=0.01, LSE atol=0.002/
rtol=0.001, unquantized relative-RMSE <=0.02, finite output and eight raw-bit
identical replays of EACH variant. Existing uniform-P/cancellation exact
fixtures remain exact. Add long-K sampled queries (K length 73774) so the
device gate cannot pass by testing only 1-5 K64 blocks.

Host seam: execute the production four-peer finalization helper, preserve
the one-ULP reorder witness, and compare both schedules to long-double
recurrence for distinct nonnegative lane sums/common rescaling, up to 1153
blocks. Fixed relative denominator bound 5e-4 (zero stays exactly zero),
including unequal lanes, zero/masked blocks and changing maxima. This is a
local seam test, not execution of the PPU kernel. Negatives: omit finalization,
use wrong XOR peer, double-finalize, and reorder-witness wrongly demanded raw.

Native gates: 32 attention/48 total types, zero private stack, H3 vector regs
<=246; shared bytes unchanged; all 16 FP16 bodies unchanged. Integer QK/PV
MMA inventories unchanged, no floating fallback, P packing unchanged, barriers
unchanged. CFG must put 40 shuffles in K loop and 8 outside (baseline 48/0).
Plant old placement and missing/extra final shuffle: gate must reject.

Performance verdict waits for same-device baseline/candidate numeric gate,
normal event latency, and ACU. Static reduction is not an observed speedup.
Build locally with SDK2.1.1/Torch2.9; publish only a separate experimental
candidate, never overwrite a49338f or the main FP16-PV package. No deadline
specified; scope is this candidate plus local validation and direct box handoff.
