# PPU Sage requant checkpoint

- updated-at: 2026-09-17 06:15:00 UTC
- working-on: preserve INT8 PV on experiment/ppu-pv-int8; restore FP16 PV on main; RTX 5070 native FP8 now authorized as a cross-precision instruction-structure reference
- baseline: fbf3d0feb1d356d7f4bad4713955de9a6e46f864; uploaded all-INT8 H3 ACU
- numerical contract: unchanged P multiply/RNE/clamp, local P scale, V block/channel scale, FP32 softmax and accumulation order
- blocked-on: none for source isolation; NVIDIA / PPU work and precision must be labelled separately
- last-commit: 96e2a07
- device verdict: NOT RUN; no speedup claimed and no wheel published for this candidate
- local results: H3 static integer/bit ALU 1136->832 (-26.76%), total 4083->3709 (-9.16%), regs 250->250, spill 0; 16 legacy FP16 bodies unchanged; real-traits 4096-bit basis and all negative gates PASS; full SDK 48/48 no spill
- remote state: upstream d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 built successfully in isolated WSL directory; no package installed globally, no Sage GPU/NCU measurement run
