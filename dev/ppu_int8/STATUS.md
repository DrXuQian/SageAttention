# PPU Sage requant checkpoint

- updated-at: 2026-09-17 06:01:55 UTC
- working-on: local requant ALU A/B closed; NVIDIA FP8 profile cancelled before execution per user's INT8 requirement
- baseline: fbf3d0feb1d356d7f4bad4713955de9a6e46f864; uploaded all-INT8 H3 ACU
- numerical contract: unchanged P multiply/RNE/clamp, local P scale, V block/channel scale, FP32 softmax and accumulation order
- blocked-on: upstream Sage2 has no integer PV branch; distinguish existing INT8-QK/FP16-PV from a new all-INT8 NVIDIA counterpart before profiling
- last-commit: 5a3fc7d
- device verdict: NOT RUN; no speedup claimed and no wheel published for this candidate
- local results: H3 static integer/bit ALU 1136->832 (-26.76%), total 4083->3709 (-9.16%), regs 250->250, spill 0; 16 legacy FP16 bodies unchanged; real-traits 4096-bit basis and all negative gates PASS; full SDK 48/48 no spill
- remote state: upstream d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 built successfully in isolated WSL directory; no package installed globally, no Sage GPU/NCU measurement run
