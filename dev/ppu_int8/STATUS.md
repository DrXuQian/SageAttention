# PPU Sage main checkpoint

- updated-at: 2026-09-17 06:35:21 UTC
- working-on: complete; FP16-PV main/default-wheel restoration and local SDK gates closed; measured RTX 5070 instruction comparison delivered on experiment branch
- baseline: 0e30f92 (INT8 QK / FP16 PV); restored device and Python compute sources are identical
- experiment: experiment/ppu-pv-int8 at f3e4dd7 (pushed); integer-PV code/ALU work, two RTX 5070 NCU runs, raw exports and normalized comparison preserved there
- blocked-on: none
- last-commit: 12a4135
- local verification: fresh SDK 28/28 kernels no spill, bridge 128/128, historical accumulator negative red; routing 4 paths/two negatives; profile 6 tests and benchmark 7 tests PASS; native imports with legacy shim and CPU rejection PASS
- artifacts: ppu-wheels d713b01 restores the existing post1 default; unchanged wheel bytes; swapped-mode and valid-wrong-wheel negatives PASS
- device verdict: no new PPU execution or speedup claimed; reusing the existing post1 FP16-PV wheel
