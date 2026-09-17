# Resume

Read docs/plan.md before touching the arithmetic. Parent e33d68c, baseline
a49338f immutable. Only move integer-PV denominator XOR reduction outside K.
Output root /workspace/sage-deferred-denominator-20260917 (mkdir).
Keep main FP16-PV untouched; no local PPU device admission is possible.

Implementation and first real SDK build done. Kernel-only change is ~22 lines
plus a shared host/device finalization helper. P/V/numerator unchanged.
2560 traces / 478720 updates pass bound 5e-4, worst 4.47630923e-7; four host
negatives red. 48/48 compiled, all spill stacks zero. Native placement passes
all 16 INT8 variants: K48->40, final0->8; FP16/quant 32 bodies unchanged.
H3 vregs/sregs246/128 unchanged. Four native-placement negatives red.
Independent CPU tensor suite: 24 cases, five H3 rows, four zero/cancellation
cases pass unchanged thresholds. These are not device results.
Complete: source 567bba7 clean SDK build passed, 48 native streams identical
to first candidate. Published separate prebuilt/ppu_10/deferred-denominator;
DSO 0c41e28d02c491b61992fd33de75eacafd76a967cc937b2d882f23f7029ad22d.
Both before/after staged imports and native CPU-input rejections pass.
Pinned before-package worktree 97ea065 verified (DSO from clean a49338f).
Long-K CPU fixture NQ5/NK73774/GQA2:1 passes unchanged .02 RMSE threshold for
FP16 (.01445358) and BF16 (.01515601); device gate still mandatory.
Next is user PPU execution: tools/run_ppu_int8_denominator_ab_box.sh.
Do not infer a latency win or dynamic instruction count from the static gate.
