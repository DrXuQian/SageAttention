# Explicit INT8 K preparation / direct P operand experiment

Parent4486f03, experimental branch only. The installed/main FP16-PV path is
not replaced. This is a local-proof/codegen result, **not** a new PPU numerical,
latency or NVIDIA instruction-parity verdict.

## Change and scope

PPU0010 QK's C layout places columns at `lane%4 + 4*column_slot`, whereas U8
PV's A layout consumes `4*(lane%4) + byte`. Store quantized K rows with the
4-by-4 low-row transpose within each16 rows. Complete K64 blocks then need
only register renaming between packed P and operand A. V remains unchanged.

The permutation occurs inside K's existing quantizer store, with the same
codes, scale computation and numerical operations. It is not an additional
reorder kernel. A partial final K64 block stays in ordinary row order and
uses the original P butterfly. Causal masks recover logical key coordinates.
For H3 N73774,1152 blocks use the direct path and one K46 tail uses the old
path. No padding or out-of-bounds key representation is introduced.

The new native entrypoints must be used together:

- `quant_per_block_int8_permuted_k`
- `qk_int8_sv_int8_permuted_k_attn`

Passing old raw K to the second function is invalid. There is no automatic
selector or default API change. Existing low-level tensor APIs cannot identify
layout from tensor dtype/shape; the independent numeric test deliberately feeds
raw K to the new reader and requires an error in the resulting values.

Individual QK dots, P codes, V scale and PV operands retain their semantic
values. The denominator's lane-local FP32 order changes. The numerical gate
therefore keeps the existing O atol=.002/rtol=.01, LSE atol=.002/rtol=.001,
original-input relative RMSE<=.02, exact constant/cancellation witnesses and
eight bit-stable replays per variant. It does not silently claim raw equality
between two FP32 accumulation orders.

## Local evidence

`key_permutation_oracle.cu` calls the production row map and independently
anchors packed words to actual QK CLayout / PV ALayout traits. It exhausts
4096 input bits /524288 words,18528 row placements and2377760 causal decisions.
Omitted permutation, one wrong bit, permuted tail, physical-coordinate masking,
and an omitted basis case all fail. The independent tensor reshape/transpose
reference checks768 cases /42688512 cells across all64 tail residues, two
head dimensions, two layouts, batches and heads.

Real SDK2.1.1 compilation has72 instances: old48 plus16 explicit attention
and8 K quantizers. All old48 opcode+operand streams are unchanged (only the
default template argument spelling changes). All new attention types preserve
integer MMA / P conversion / barrier inventories, with zero private stack.

| H3 resource / emitted path | Deferred raw control | K-permuted r3 |
|---|---:|---:|
| Vector registers |246|244|
| Scalar registers |128|104|
| Private stack bytes |0|0|
| K64 shuffle sites on complete-block path |40|8|
| K64 shuffle sites on incomplete-tail path |40|40|
| Final denominator shuffles |8|8|
| Reachable static instructions, including tail/control |3450|3454|
| All-instruction CFG path min/max |1617/1853|1569/1876|

Do **not** call the last two rows dynamic instruction counts. The tail code is
still emitted but only the last partial block executes it; masked paths and
once-only setup require separate accounting. Native FP8's measured count is
not directly comparable to a minimum/maximum emitted path.

Two earlier placements (r1/r2) used248 vector registers and were rejected
under the predeclared246 ceiling. r3 performs the tail-only word transpose as
each P row finishes, shortening the merge lifetime. No arithmetic was removed.
The codegen gate has six negatives: missing type, changed old body, register
regression, private stack, FP16 MMA substitution, and a surviving full-path
transpose. All are required red before publishing.

## Device decision is still pending

Run the isolated prebuilt, without compiling or installing on the box:

```bash
PROFILE=1 bash tools/run_ppu_int8_key_permutation_box.sh
```

The runner first compares actual K preparation bytes and scales (including
mean subtraction), runs quantized attention oracles, tails/causal/GQA/layouts,
long1153-step denominator checks and eight replays. Only then does it time
the same-DSO FP16, raw INT8 and permuted-K INT8 controls with normal events.
K quantization alone and K-prepare+core are also timed: a moved preparation
cost must not disappear from the result. ACU is a separate single launch.

Candidate adoption requires no numeric regression and a resolved latency win,
both core and K-prepare+core versus raw INT8. Overlapping sample envelopes are
UNRESOLVED; slower is rejected regardless of fewer instructions. Report the
FP16 control even if it remains faster. The user570ms observation remains
unbound to SHA/report/timer and is not assigned to this candidate.
