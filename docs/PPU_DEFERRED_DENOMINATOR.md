# Experimental INT8-PV: denominator reduction after K

Parent `e33d68c`; codegen/binary control is the immutable clean-source
`a49338f` ALU candidate, packaged at `97ea065`. Main/default remains PV FP16.
This is one numerical-scheduling change, not a new quantization format.

`tile_sum` stays lane-local for INT8 PV. Each lane updates its FP32 denominator
with the same row-uniform `rescale` and `block_rescale` as before. After the
last K block, the production `finalize_row_denominator` does XOR-1 then XOR-2
addition, once, before **both** output division and LSE. FP16 PV keeps its
original in-loop reduction. QK, P packing, V scales, PV/numerator accumulation,
AIU delivery, masking, scheduler and public ABI are unchanged.

## Numerical admission, registered before editing

This intentionally changes FP32 summation order. The known 1-ULP witness is
retained, not hidden by a weakened raw comparator. Both variants must still
pass the existing independent quantized-output gate (atol .002, rtol .01),
LSE gate (atol .002, rtol .001), unquantized relative RMSE <=.02, finite values
and eight raw-bit stable replays per variant. Uniform-P/exact cancellation
controls remain exact. The device runner now also covers five queries over
73,774 keys, FP16 and BF16 output, GQA, LSE and replay. It cannot admit this
change using only short one-to-five-block examples.

The host test executes the actual production finalization with a four-peer
exchange. An independent long-double recurrence covers 2,560 complete traces,
478,720 updates, lengths 0/1/2/4/16/64/256/1153, unequal positive lanes, tiny
contributions, masked/zero blocks and changing row maxima. Predeclared
relative denominator bound: 5e-4; exact zero stays zero. Observed worst
deferred error 4.47631e-7, versus eager 7.57679e-7 on **this test set**, not a
general accuracy advantage. Omit finalization, wrong XOR peer, double finalize,
and wrongly demanding old/new raw equality are all red.

## Local SDK result (no PPU device run)

SDK 2.1.1, ppu_10, Python3.12/Torch2.9/ABI1:

| Native property | a49338f control | Deferred candidate |
|---|---:|---:|
| K-loop shuffle sites / iteration | 48 | 40 |
| Post-loop shuffle sites | 0 | 8 |
| H3 vector / scalar registers | 246 / 128 | 246 / 128 |
| H3 private stack | 0 | 0 |
| H3 complete reachable instructions | 3455 | 3450 |
| H3 per-K CFG path inventory | 1636..1872 | 1617..1853 |

All 16 INT8 instances have the 48->40 / 0->8 placement. All 16 FP16 attention
bodies and 16 quantizer bodies are native-stream identical to control. No
extra barrier/shared allocation, changed MMA/P-pack inventory, or floating
MMA fallback. All 48 types compile with zero private stack. Native negatives
restore old placement, omit a final shuffle, add a hot shuffle, or alter FP16;
each must reject.

For causal code, the compiler has a pre-MMA loop-exit branch: a zero-iteration
exit is not a K64 visit. The checker proves placement using the actual MMA
loop SCC and requires each final shuffle to have no path back into K. The
stricter single-path bound is additionally checked on the noncausal H3 target.
It does not call an unsupported causal path model a PASS.

At H3 length 1153 blocks, moved cross-lane work saves
`8 * (149022944 - 129248) = 1191149568` warp shuffles. It also removes eight
per-iteration sum adds (adds now occur once after K); generated H3 paths are
19 instructions shorter overall. This is an instruction opportunity, **not**
a measured latency win. Current candidate's conservative CFG upper bound is
1854.23 instructions/warp/K64 versus NVIDIA FP8 measured 986.05 (FP16 buffer)
or 845.96 (direct FP32). Neither bound is a new ACU measurement. Substantial P
packing/retiling and block-local scale work remains.

## Build and box handoff

Local build uses `dev/ppu_int8/run_sdk_compile.sh` followed by
`check_deferred_denominator_codegen.py` against the retained a49338f ISA.
The experimental prebuilt is published separately under
`prebuilt/ppu_10/deferred-denominator`; the original `alu-candidate` is not
overwritten. No replacement main wheel.

Clean candidate source: `567bba728393209c4ac13f38279be93f6e56e830`.
DSO SHA256: `0c41e28d02c491b61992fd33de75eacafd76a967cc937b2d882f23f7029ad22d`.
All 48 native streams match the first local candidate; artifact/source/ABI
verification and both isolated imports pass. Runtime SONAME is
`libhggcrt.13.0.so`; the manifest requires Torch2.9/CPython3.12/C++11 ABI1.

From `experiment/ppu-pv-int8`, the execution-only A/B is:

```bash
git pull --ff-only
bash tools/run_ppu_int8_denominator_ab_box.sh
```

It creates a fresh `/workspace` output, checks out the pinned **before source**
in a separate worktree, stages each verified binary in a private Python import
directory, and runs the **same current** numerical/profile driver for both.
Baseline and candidate run sequentially in separate processes. It requires
no SDK compiler, pip install, or change to the default FP16-PV package. Both
use the same seeds and shape. Each arm records source and binary identities.
`PROFILE=0` runs only correctness; default `PROFILE=1` adds two `acu -o ...
--set full` reports beneath `before/` and `after/`.

Compare the exact integer-PV H3 core symbol, not preparation kernels. ACU
instrumented duration is diagnostic; normal event latency is still needed
before declaring a performance winner. No device result is claimed here.
