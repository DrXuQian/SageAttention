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
Next: commit source, clean-source SDK rebuild, verify 48 native streams equal
to first candidate, publish separate prebuilt and test isolated import. Run
tools/run_ppu_int8_denominator_ab_box.sh on PPU only after that handoff.
