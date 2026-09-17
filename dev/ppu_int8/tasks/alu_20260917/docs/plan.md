# Integer-PV ALU closure: bounded local candidates

Start: `f3e4dd7`, `experiment/ppu-pv-int8`. Main FP16 PV and the default post1
wheel remain untouched. Workload priority is B1/H56/S73774/D128/full; local
regression inventory is D64/D128, full/causal, FP16/BF16 output, LSE off/on.
Baseline PPU post2 ACU and native RTX 5070 FP8 comparison are retained. The
already-built local requant candidate is a codegen baseline, not a timing win.

Arithmetic contract: keep Q/K and V quantizers, per-K64/channel V scale,
local-row P scale, `RN_float(p*255)` then RNE U8, FP32 softmax sums, MMA/K order,
FP32 partial scaling/accumulation, masks, AIU layout, public ABI and scheduler.
Only finite admitted inputs are numerical-performance evidence. Masked rows,
threshold neighbours and cancellation must have explicit coverage.

Candidate inventory, tried separately before composition:

1. Row-level mask classification: remove each probability's repeated sentinel
   compare/select by choosing the exponent origin once per row. Prove both
   partial-mask and all-masked rows, including sentinel-threshold neighbours.
2. Inverse-dequant probability packing: bounded magic addition + byte extraction
   versus the existing saturated pack. Exhaust all finite nonnegative FP32
   inputs <=1; retain a fused-multiply/bias rounding negative. Keep only if
   actual native codegen improves the relevant instruction/latency chain.
3. Value-scale address delivery: inspect real instruction PCs and hoist/vectorize
   only where generated arithmetic is actually duplicated. No layout or scale
   granularity changes. Require exact owner/address/value coverage.

INT32 partial -> FP32 is audited, not presumed redundant. Current S32 MMA and
distinct per-block scales require float restoration before online accumulation;
an alternative must show exact arithmetic and a shorter native sequence.

Local admission: real SDK helper/full-body builds; original FP16-PV bodies
unchanged; complete 32 attention / 48 total symbol census; zero private stack;
MMA/barrier inventory preserved; exact physical byte maps and negatives;
numeric CPU cases plus boundary-specific checks; `git diff --check`.
Do not claim independent tensor-oracle tests execute the new PPU body.

Performance decision: retain a candidate only with smaller measured native
instruction cost in its intended hot chain and no spill/critical footprint
regression. Normal latency and ACU dynamic counters still require a fresh PPU
run of the composed binary. No numerical tolerance is relaxed to obtain a win.
No user wall-clock deadline was specified; scope is these three candidates,
then verification and a reproducible handoff, not an unrestricted sweep.
