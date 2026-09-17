# PPU Sage main checkpoint

- updated-at: 2026-09-17 12:50:02 UTC
- working-on: retired; use main38d2839 and default post1 FP16-SV wheel, no active INT8-SV optimization
- baseline: 0e30f92 (INT8 QK / FP16 PV); restored device and Python compute sources are identical
- experiment: bdf89be restored the tree to stable main0a89c2f exactly;116 experiment files including3 candidate DSOs removed, history recoverable
- blocked-on: none
- last-commit: bdf89be (forward restoration; this retirement-status checkpoint excluded)
- local verification: fresh SDK 28/28 kernels no spill and bridge128/128; CPU routing4 tests (public/named,HND/NHD,two planted routing errors,retired-entrypoint absence), profile6 and benchmark7 PASS; packaging3 negatives PASS; no new device run
- artifacts: ppu-wheels943ac7b removes post2 manifest/payload and experimental installer option; post1 wheel SHA25670409a90 unchanged;4 release negatives and retired-option rejection PASS
- device verdict: no new PPU execution or speedup claimed; reusing the existing post1 FP16-PV wheel
