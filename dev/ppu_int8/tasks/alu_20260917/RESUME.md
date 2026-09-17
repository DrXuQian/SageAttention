# Resume

Branch `experiment/ppu-pv-int8`, starting `f3e4dd7`. Main is not the worktree.
Read `docs/plan.md` in this directory before modifying an arithmetic seam.
Row-mask candidate passed 1,065,024 cases and exact all-masked negative; fresh
SDK all 48 types zero spill, FP16 16/16 unchanged; H3 static total 3709->3592.
Inverse magic pack passes all 1,065,353,217 bounded FP32 values but takes one
more probe instruction; do not replace the shorter current saturated pack.
Next candidate: CTA-cooperative V-scale staging with no extra barrier.
PPU SDK: `/root/ppu-sdk/2.1.1`, Torch 2.9 build Python:
`/root/autodl-tmp/sageattention-build-envs/torch29/bin/python`.
Build/test output root: `/workspace/sage-int8-alu-closure-20260917` (mkdir).
No PPU GPU job or new timing is running.
