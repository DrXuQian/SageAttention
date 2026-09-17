# PPU Sage requant checkpoint

- updated-at: 2026-09-17 05:58:25 UTC
- working-on: local requant ALU A/B closed; building upstream Sage2 on RTX 5070 for NCU dynamic-instruction comparison
- baseline: fbf3d0feb1d356d7f4bad4713955de9a6e46f864; uploaded all-INT8 H3 ACU
- numerical contract: unchanged P multiply/RNE/clamp, local P scale, V block/channel scale, FP32 softmax and accumulation order
- blocked-on: no blocker; upstream sm120 build running in isolated WSL directory, no global package changes
- last-commit: fbf3d0f
- device verdict: NOT RUN; no speedup claimed and no wheel published for this candidate
- local results: H3 static integer/bit ALU 1136->832 (-26.76%), total 4083->3709 (-9.16%), regs 250->250, spill 0; 16 legacy FP16 bodies unchanged; real-traits 4096-bit basis and all negative gates PASS; full SDK 48/48 no spill
