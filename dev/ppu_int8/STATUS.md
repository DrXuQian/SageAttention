# PPU Sage requant checkpoint

- updated-at: 2026-09-17 07:16:25 UTC
- working-on: scoped native A/B passed (4 selected / 12 unchanged INT8 / 16 unchanged FP16); source checkpoint then clean-SHA SDK build and explicit experimental prebuilt handoff
- baseline: fbf3d0feb1d356d7f4bad4713955de9a6e46f864; uploaded all-INT8 H3 ACU
- numerical contract: unchanged P multiply/RNE/clamp, local P scale, V block/channel scale, FP32 softmax and accumulation order
- blocked-on: none for source isolation; NVIDIA / PPU work and precision must be labelled separately
- last-commit: 52b37e3
- PPU candidate verdict: NOT RUN; no speedup claimed and no wheel published for the requant candidate; the NVIDIA reference below was measured
- local results: H3 static integer/bit ALU 1136->832 (-26.76%), total 4083->3709 (-9.16%), regs 250->250, spill 0; 16 legacy FP16 bodies unchanged; real-traits 4096-bit basis and all negative gates PASS; full SDK 48/48 no spill
- remote state: unmodified upstream d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 profiled on RTX 5070; two sequential FP8-PV modes, common 149022944 warp/K64 visits; no global installation; no equal-precision or latency claim
- measured comparison: PPU post2 integer/bit/select 566.20 per warp/K64 vs NVIDIA FP8 62.32; full counts 2264.71 vs 986.05 (FP16 buffer) / 845.96 (FP32); different MMA shapes explicitly normalized
- main/artifacts: main 2d890e7 and ppu-wheels d713b01 pushed; FP16-PV default and unchanged post1 wheel; INT8 branch + post2 opt-in retained
- row-mask checkpoint: 1065024 host pairs raw-equal, all-masked negative red; shipping H3 static instructions 3709->3592, integer/bit ALU 832->772; all 16 FP16 instruction streams identical; all 16 INT8 arithmetic/barrier inventories preserved; SDK 48/48 zero private stack; device timing NOT RUN
- inverse-pack verdict: 1065353217 FP32 inputs exhausted, strict magic pack raw-equal but native probe 23 vs 22 instructions; fused version differs at 128/765 rounding-boundary neighbours; NOT ADOPTED
- V-scale checkpoint: host 69600 blocks / 6681600 publications / 213811200 reads raw-equal; missing owner, wrong head pitch, payload overlap and wrong column negatives red. Unscoped native H3 total 3592->3492, integer ALU 772->663, vregs 250->246, sregs 112->128, shared +512 B, stack 0; do not apply to D64/causal (integer ALU slightly increases).
