# PPU Sage main checkpoint

- updated-at: 2026-09-17 06:23:27 UTC
- working-on: main FP16-PV restoration complete; RTX 5070 H3 native FP8 NCU captured, opcode analysis isolated on experiment branch next
- baseline: 0e30f92 (INT8 QK / FP16 PV); restored device and Python compute sources are identical
- experiment: experiment/ppu-pv-int8 at 7139657 (pushed); all integer-PV code, ALU gates and evidence preserved there
- blocked-on: none
- last-commit: e09b305
- local verification: fresh SDK 28/28 kernels no spill, bridge 128/128, historical accumulator negative red; routing 4 paths/two negatives; profile 6 tests and benchmark 7 tests PASS; native imports with legacy shim and CPU rejection PASS
- artifacts: ppu-wheels d713b01 restores the existing post1 default; unchanged wheel bytes; swapped-mode and valid-wrong-wheel negatives PASS
- device verdict: no new PPU execution or speedup claimed; reusing the existing post1 FP16-PV wheel
