# Skip only the identity output rescale

Parent 9004faa, immutable native baseline source711ec99 / DSO42702fd4.
Scope: compile/local proof on experiment/ppu-pv-int8; default FP16 PV and all
already-published binaries remain unchanged. Priority H3 B1/H56/S73774/D128,
Q128/KV64/four warps, full attention with the explicit permuted-K entrypoint.

Hypothesis: the output's128 multiplications by online-softmax rescale can be
bypassed when that already-computed factor is exactly1. Do not approximate
the comparison, skip a factor close to1, alter P codes, change V-scale
granularity, or change the denominator. Keep the general multiply path.
This is not the previously-rejected score FMA or global V quantizer.

First compile a32-element row helper with current, lane-conditional and
warp-uniform-all-identity alternatives under all shipping HGGC flags. Inspect
native CFG, not just source branch presence. Reject if the compiler keeps
the multiplications on every path or adds spills. Only then test a narrowly
scoped real-body candidate. Final H3 vregs must not exceed244, stack=0;
other compiled instances and FP16 native bodies must remain unchanged.

Numerics: skipping RN(x*1) is an IEEE identity for finite FP32 x (including
signed zero/subnormal without FTZ); explicitly test the subsequent FMA under
FTZ/DAZ as well so a floating-environment assumption cannot hide a difference.
Cover mixed row factors, underflow, signed/cancelling numerators and repeated
updates. Existing O/LSE tolerance and raw-stable per-variant replay unchanged;
no device result is implied by a CPU check.

Negative controls: skip a nonunit rescale; compare approximately rather than
exactly; hide one output from the denominator; retain multiplication on the
claimed bypass path; modify a legacy specialization; add private stack.

Keep worst/update path and best/identity path counts separately. Frequency of
identity factors is not asserted for the unknown device fixture. A native
instruction win without timing is only a candidate. No unmeasured speedup,
automatic routing, or replacement of the delivered prebuilt. If native code
does not improve, record rejection and leave production untouched.

11:57 UTC checkpoint: early lane-conditional and warp-uniform candidates both
raise H3 vector registers244->250 and are rejected. Lane-conditional has the
smaller added control path (best1456/worst1891 versus1569/1876); uniform is
1468/1903. A final, bounded reschedule is now proposed BEFORE compiling it:
do the same conditional output rescale after this row's P packing, not before
it. The operations are independent; their arithmetic is unchanged. This may
end the score-fragment live ranges before the conditional O update. Keep the
244-register cap and every other gate; do not waive it if the third fails.
