# PPU Sage main checkpoint

- updated-at: 2026-09-17 06:18:30 UTC
- working-on: FP16-PV routing and two negatives PASS; main/installed-wheel restoration; fresh SDK build and RTX 5070 instruction reference running next
- baseline: 0e30f92 (INT8 QK / FP16 PV); restored device and Python compute sources are identical
- experiment: experiment/ppu-pv-int8 at 7139657 (pushed); all integer-PV code, ALU gates and evidence preserved there
- blocked-on: none
- last-commit: 96e2a07
- device verdict: no new PPU execution or speedup claimed; reusing the existing post1 FP16-PV wheel
