# Resume

Branch `experiment/ppu-pv-int8`, starting `f3e4dd7`. Main is not the worktree.
Read `docs/plan.md` in this directory before modifying an arithmetic seam.
Row-mask candidate passed 1,065,024 cases and exact all-masked negative; fresh
SDK all 48 types zero spill, FP16 16/16 unchanged; H3 static total 3709->3592.
Inverse magic pack passes all 1,065,353,217 bounded FP32 values but takes one
more probe instruction; do not replace the shorter current saturated pack.
V-scale staging all-format native build passed, but integer ALU grew in D64
and causal D128. Rejected blanket application. D128/noncausal candidate alone
passed native A/B against mask-only: 4 selected / 12 unchanged INT8
and all 16 unchanged FP16 bodies. H3 ALU 772->663, total 3592->3492,
vector regs 250->246 but scalar regs 112->128 and shared +512 B; stack 0.
Host staging 213811200 consumers raw-equal, all four planted defects red.
24 numeric cases + sampled H3 + zero/cancellation and complete real-traits
4096-bit basis passed; they are not an execution of the new PPU body.
PPU SDK: `/root/ppu-sdk/2.1.1`, Torch 2.9 build Python:
`/root/autodl-tmp/sageattention-build-envs/torch29/bin/python`.
Build/test output root: `/workspace/sage-int8-alu-closure-20260917` (mkdir).
No PPU GPU job or new timing is running.
Next: source checkpoint, fresh clean-SHA full SDK build and an explicit
experimental prebuilt handoff. Do not replace the default FP16 wheel.
