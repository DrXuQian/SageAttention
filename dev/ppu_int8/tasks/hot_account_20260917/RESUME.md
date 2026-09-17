# Resume

Read docs/plan.md. Source baseline 97ea065; final INT8 candidate a49338f binary
is immutable. PC-preserving native parser now bounds kernels at the next ELF
section and follows reachable branches, including targets after an early
exit. Seven parser tests pass. The old H3 total included 37 dead/foreign
records: 4083 is 4046, and final 3492 is 3455. Integer ALU is unaffected.
New source finding: native scalar denominator stays lane-local over K and
reduces once in normalize_d; ours reduces every K64 iteration. Host rounding
witnesses must gate any claim of raw equivalence.
No new PPU/NVIDIA run started. Native reference files and prior PC counters
are under /workspace/sage-nvidia-5070-reference-20260917 and
/workspace/sage-h3-int8-acu-analysis. Output root for this investigation:
/workspace/sage-int8-hot-account-20260917 (mkdir, not temporary directories).

Complete: docs/PPU_INT8_REMAINING_GAP.md and results/hot_account_20260917.
All 4046 native old PCs/opcodes/operands match ACU; sum closes. Candidate
integer whole body 663 = 365 cold + 298 loop-union; CFG paths 93..298 are
NOT measured counts. Original dynamic category is 0.011275/visit narrower
than the static category (three once-only opcode families); explicitly noted.
7 parser tests, 7 hot-account negatives, 7 retained negatives passed.
Denominator deferral differs 0x46902000 vs 0x46902001 for 1153 blocks;
score FMA differs 0 vs 0x364a8000. Exact controls agree. Neither is installed.
Next numerical candidate: defer denominator only, preregister tolerance,
keep P/V format. Pending a49338f device verdict remains independent.
