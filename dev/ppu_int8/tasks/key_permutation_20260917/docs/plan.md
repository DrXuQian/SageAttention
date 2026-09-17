# Move P's repeated lane permutation to K preparation

Parent 4486f03. Experimental branch only; main FP16-PV and existing entrypoints
keep their semantics. Priority B1/H56/S73774/D128/full, Q128/KV64/128 threads;
regress D64/D128, causal/full, LSE/no-LSE, FP16/BF16, HND/NHD, tails and GQA.

The real PPU0010 C layout owns column `lane%4 + 4*value%4`. U8 A needs
`4*(lane%4) + byte`. Swapping the two low two-bit groups of K's row index
within each 16 rows is an involution that bridges these coordinates BEFORE
QK. Quantize the same input values/scales, but store K codes at those permuted
row positions. In a complete K64 block P packs then need no lane shuffle or
byte permutation. V packing and semantic K order at PV stay unchanged.

Incomplete K64 blocks MUST keep their original row order and the original P
butterfly. This avoids extra padding/storage or treating a hole as a valid
key. Causal masks compare ORIGINAL logical key indices after undoing the
permutation; never the physical stored row. No change to Q or V quantization,
P RN(p*255)/RNE/saturation, MMA, block geometry, scheduler or public old ABI.
Expose new, explicit paired K-quantizer/attention entrypoints. Never silently
interpret an old K tensor as the new representation. No auto-selector change.

Numerical contract: each semantic QK dot, P code and PV operand is unchanged.
The lane-local denominator summation order changes; explicitly admit only
that FP32 reassociation under the existing O atol=.002/rtol=.01, LSE atol=.002/
rtol=.001, original-input relative RMSE<=.02 and eight raw-stable replays of
each variant. Keep exact uniform-P/constant and cancellation witnesses.

First gates: actual QK CLayout and U8 PV ALayout, full 4096-bit basis, row
permutation roundtrip, every tail residue, batch/head stride semantics and
causal masks. Plants: omitted permutation, wrong permutation bit, permuted
tail, mask using physical index and incomplete coverage. Full SDK afterward:
old 48 native instruction streams unchanged modulo template spelling; new
types have no floating MMA, no spill, no H3 register regression beyond246.
P bridge shuffle/byte-perm can survive only on incomplete-tail paths.

Performance: same-device alternating normal event times, including an FP16
control. Compare preparation cost separately; do not hide a moved cost.
ACU counts bind the exact new symbol and binary. No static count establishes
speedup or closes the goal. Retain old binaries and publish only a separate
experimental artifact after local gates. No requested deadline.
