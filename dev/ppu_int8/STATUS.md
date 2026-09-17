# PPU Sage main checkpoint

- updated-at: 2026-09-17 12:50:02 UTC
- working-on: complete; stable INT8-QK/FP16-SV restored and experiment retired, no further FP8 instruction-alignment work
- baseline: 0e30f92 (INT8 QK / FP16 PV); restored device and Python compute sources are identical
- experiment: bdf89be restores experiment/ppu-pv-int8 to main0a89c2f exactly;116 experiment files removed,including3 candidate DSOs; history remains recoverable
- blocked-on: none
- last-commit: 0a89c2f (main retirement; this metadata checkpoint excluded)
- local verification: fresh SDK 28/28 kernels no spill and bridge128/128; CPU routing4 tests (public/named,HND/NHD,two planted routing errors,retired-entrypoint absence), profile6 and benchmark7 PASS; packaging3 negatives PASS; no new device run
- artifacts: ppu-wheels943ac7b removes post2 manifest/payload and experimental installer option; post1 wheel SHA25670409a90 unchanged;4 release negatives and retired-option rejection PASS
- device verdict: no new PPU execution or speedup claimed; reusing the existing post1 FP16-PV wheel
