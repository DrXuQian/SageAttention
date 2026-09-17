# PPU Sage requant checkpoint

- updated-at: 2026-09-17 06:34:13 UTC
- working-on: RTX 5070 H3 native FP8 two-mode NCU reference complete; write measured instruction account and preserve raw exports; main already restored to FP16 PV
- baseline: fbf3d0feb1d356d7f4bad4713955de9a6e46f864; uploaded all-INT8 H3 ACU
- numerical contract: unchanged P multiply/RNE/clamp, local P scale, V block/channel scale, FP32 softmax and accumulation order
- blocked-on: none for source isolation; NVIDIA / PPU work and precision must be labelled separately
- last-commit: 7139657
- PPU candidate verdict: NOT RUN; no speedup claimed and no wheel published for the requant candidate; the NVIDIA reference below was measured
- local results: H3 static integer/bit ALU 1136->832 (-26.76%), total 4083->3709 (-9.16%), regs 250->250, spill 0; 16 legacy FP16 bodies unchanged; real-traits 4096-bit basis and all negative gates PASS; full SDK 48/48 no spill
- remote state: unmodified upstream d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 profiled on RTX 5070; two sequential FP8-PV modes, common 149022944 warp/K64 visits; no global installation; no equal-precision or latency claim
- measured comparison: PPU post2 integer/bit/select 566.20 per warp/K64 vs NVIDIA FP8 62.32; full counts 2264.71 vs 986.05 (FP16 buffer) / 845.96 (FP32); different MMA shapes explicitly normalized
- main/artifacts: main 12a4135 and ppu-wheels d713b01 pushed; FP16-PV default and unchanged post1 wheel; INT8 branch + post2 opt-in retained
