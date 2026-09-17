# Explicit integer-PV ALU candidate

This is **not** the default SageAttention wheel. It is the local SDK 2.1.1
`ppu_10` build of clean source `a49338f8c86e5ec7a4c05eb9543b5e35af6f7e26`,
for Python 3.12 / Torch 2.9 / C++11 ABI 1 / `libhggcrt.13.0.so`.

All 48 specializations built with zero private stack; local numerical/layout
and code-generation gates passed. PPU device correctness, latency and ACU:
**NOT RUN**. Main/default installation retains FP16 PV.

From a separate checkout of `experiment/ppu-pv-int8`, run:

```bash
PROFILE=1 bash tools/run_ppu_int8_alu_candidate_box.sh
```

The runner validates source/binary/runtime, stages a private import directory
under `/workspace`, runs the existing INT8 numerical/replay gate, then profiles
B1/H56/S73774/D128/full with ACU. No box compilation or pip installation.
Do not use the legacy generic wheel installer to select this candidate.
See `docs/PPU_INT8_ALU_CLOSURE.md` for the count changes and their limits.
