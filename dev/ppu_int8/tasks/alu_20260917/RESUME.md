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
Final clean source a49338f was rebuilt: native streams identical 48/48,
all codegen/negative gates pass and all private stacks are zero. The dedicated
prebuilt is under `prebuilt/ppu_10/alu-candidate`, DSO sha256
`dc50962536e79bf8ec636ece3c0d2a5070d8a416ae05d82270f11c0aee6e126c`.
Isolated package import and native runtime/CPU-rejection tests passed. Runner:
`PROFILE=1 bash tools/run_ppu_int8_alu_candidate_box.sh` from this experiment
branch. It does not install or overwrite the default FP16 wheel. Next evidence
must be device numeric/latency/ACU; this local pass is not performance admission.
