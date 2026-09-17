# Output rescale: native savings exist, but these three candidates are rejected

2026-09-17, experimental INT8 PV only. Parent9004faa; immutable shipping
source711ec99 / binary42702fd4. No new device result, no changed default,
no replacement of an existing artifact. The live kernel was restored exactly
after the three local compiler experiments.

## What is actually redundant

Each warp owns128 persistent FP32 O values. The current K64 loop multiplies
all128 by the online-softmax rescale even when that factor is exactly1.
Only skip EXACT1, never a nearby value. This leaves P RNE/saturation, local
V scales, integer MMA, denominator and update ordering unchanged.

A native microprobe proves that the compiler does produce a bypass, not just
predicate every multiply. One32-value row has the following CFG counts:

| Microprobe | identity path | update path | multiplies |
|---|---:|---:|---:|
| current |141|141|32|
| lane conditional |115|178|0 or32|
| warp-uniform vote |118|181|0 or32|

Loads/stores are included in these small probes. Their scheduling changes
the wait count; these are not the production-loop savings. All use exactly
the production HGGC lowering flags, not a simplified-O3 surrogate.

## The full kernel decides

Same SDK2.1.1/ppu_10, same Q128/KV64 geometry and generated72-type target.
Change only full-D128/noncausal/permuted-K:4 specializations. Other68 native
opcode+operand streams are unchanged, including every FP16-PV control and
all quantizers. Stack0 in all72 types. Only the third candidate moves the
independent O rescale after the row's P packing to shorten temporary lifetimes.

| Real H3 body | shortest CFG walk | longest CFG walk | whole static | vregs/sregs | decision |
|---|---:|---:|---:|---|---|
| retained711ec99 |1569|1876|3454|244/104|retain|
| conditional before P |1456|1891|3469|250/104|reject register cap|
| uniform before P |1468|1903|3481|250/104|reject register cap|
| conditional after P |1459|1894|3472|248/104|reject register cap|

All three change FP32-multiply path counts395..395 to267..395: exactly128
can be bypassed. The real predicates compare the SAME factor used by each
group of32 multiplies to literal FP32 one. The gate follows branch/join
regions and binds those operands, not just two totals. QK/PV MMA, both
conversion types, FMA/add, P pack, shuffle and barrier inventories are intact.

These are generated-path inventories, NOT fresh dynamic averages. Different
Q rows can update their maxima in different iterations; an active lane can
keep a whole warp's predicated body executing. Do not claim that every late
iteration takes the short path, or divide a static body by a measured FP8
average and call it a measured ratio.

The244-vector-register cap was registered before editing the kernel. It was
not relaxed after seeing250/248. Rejection here is conservative resource
admission, not evidence that248 registers necessarily slows this device.
None of these candidates received device latency admission.

## Numeric boundaries and negative controls

CPU screen:16,711,680 factor/value pairs, including2,088,960 exact-unit pairs,
both signs, all finite FP32 exponent classes, subnormals and signed zero;
the multiply and following restoration FMA are raw-equal. Exhaustive256
eight-row change masks give2,048 additional uniform-predicate checks.
Skip-nonunit, approximate-one and missing-output plants each fail.

An independent floating-environment witness catches a real qualification:
FTZ-only can flush the old x*1 but not the skipped x before a following FMA
produces a normal value. Thus the identity is NOT extrapolated to every
floating mode. The actual kernel resource descriptors specify
`fp_denorm_flush:0`; that is an explicit native gate, not a host assumption.

Native negatives: change one FP16 instruction, introduce private stack,
enable FTZ, keep the128 multiplies on every path, or change the compared
one literal. Each must fail. The old/current report-binding11 negative
controls also still fail as designed; no old report is relabelled as new.

## Reproduce / retained evidence

Task folder: `dev/ppu_int8/tasks/identity_rescale_20260917/`.
Each rejected patch is preserved there; apply only in a new experimental
worktree based on9004faa, never to the retained runner worktree. The first
two use zero-context diffs (`git apply --unidiff-zero`); the late patch uses
normal context. All three dry-run checks pass against the restored source.

Use `run_sdk_compile.sh` with a realSDK and a separate `/workspace` output
per variant. Feed the retained711ec99 `shipping-isa.log`, variant ISA and
resource output to `check_identity_rescale_codegen.py`. The checker exits
normally when an experiment is successfully classified, but its decision
field remains `REJECT_REGISTER_REGRESSION`, not PASS/admitted.

CPU (from the repository root):

```bash
mkdir -p /workspace/sage-identity-rescale-local
c++ -O2 -std=c++17 -ffp-contract=off dev/ppu_int8/identity_rescale_screen.cpp \
  -o /workspace/sage-identity-rescale-local/screen
/workspace/sage-identity-rescale-local/screen
```

Native negatives use `--plant legacy-changed|spill|ftz|no-bypass|wrong-unit`.
Raw compiler logs/objects/ISA are in
`/workspace/sage-identity-rescale-20260917/{probe,conditional-sdk,uniform-sdk,late-sdk}`;
committed candidate records bind the ISA/resource hashes and keep timing empty.
Trial build manifests are NOT published source authorities; conditional's
source checkout moved to the uniform variant before its final manifest step.
Use the saved patch and clean replay, not that trial manifest, for packaging.

The FP8 instruction-alignment goal is still open. The large remaining
restoration/scaling chain and MMA-zero lowering are recorded separately in
`PPU_MMA_ZERO_SCREEN.md`; this experiment does not prove them unavoidable.
