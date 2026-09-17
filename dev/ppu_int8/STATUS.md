# PPU Sage main checkpoint

- updated-at: 2026-09-17 12:46:37 UTC
- working-on: stable INT8-QK/FP16-SV path verified; retire the integer-SV experiment and its opt-in artifacts, no further FP8 instruction-alignment work
- baseline: 0e30f92 (INT8 QK / FP16 PV); restored device and Python compute sources are identical
- experiment: cancelled by user; remove current-tree candidates, scripts and binary payloads with a forward restoration commit, not a history rewrite
- blocked-on: none
- last-commit: 2d890e7 (prior main checkpoint; retirement commit follows)
- local verification: fresh SDK 28/28 kernels no spill and bridge128/128; CPU routing4 tests (public/named,HND/NHD,two planted routing errors,retired-entrypoint absence), profile6 and benchmark7 PASS; packaging3 negatives PASS; no new device run
- artifacts: ppu-wheels943ac7b removes post2 manifest/payload and experimental installer option; post1 wheel SHA25670409a90 unchanged;4 release negatives and retired-option rejection PASS
- device verdict: no new PPU execution or speedup claimed; reusing the existing post1 FP16-PV wheel
