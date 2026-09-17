# Remaining move instructions: MMA zero initialization

2026-09-17. The delivered K-permuted source711ec99/binary42702fd4 is unchanged.
This is a compile-only screen against SDK2.1.1/ppu_10, not a new device result
or proof that every possible PPU compiler/ISA implementation is constrained.

## A distinct instruction cost, not P retiling or S32 conversion

The real H3 native K-loop contains **192 literal-zero vector moves on every
CFG path**, not merely192 sites in a union of alternatives. They match:

- QK:8 independent accumulator fragments *8 S32 values =64 clears.
- PV:16 independent partial fragments *8 S32 values =128 clears.

This is also checked by native def-use propagation, intersecting known-zero
definitions at every CFG join. All192 definitions reach exactly those8+16
first-C groups; deleting one real shipping clear makes the check fail. It is
not a coincidence between two aggregate counts.

They are moves, excluded from the earlier integer/bit/select category. Moving
K's row permutation into preparation does not remove them. NVIDIA's recorded
FP8 stream has about5 MOV/warp/K64 in total, not this per-fragment clearing
sequence. This is one reason equal layout/bit width need not give equal
instruction counts; it is not an excuse to stop examining our implementation.

## Controlled real-compiler experiments

Each small probe exports all eight S32 outputs after two same-type MMA steps.
S8xS8 and U8xS8 are both checked. The first C operand must be eight initialized
zeros; the second must consume that product, not restart from zero.

| First-product source | Reachable instructions | C clears | Resource |
|---|---:|---:|---|
| Existing in-place initialized atom |29|8|32 vector/80 scalar,stack0|
| Same actlize atom, separate zero C operands |29|8|same|
| SDK `mma_dense_sync`, zero vector |29|8|same|
| Literal `{0,0,0,0,0,0,0,0}` in PTX |29|8|same|

**The entire opcode+operand streams are identical**, not only these totals.
There are three additional zero moves for setup in each probe; they are not
charged to C. The checker follows native register definitions and rejects a missing
first-C clear, missing MMA and a missing specialization.

A scalar `0` instead of the vector does not supply a native zero source in
this SDK. Compilation rejects the lowered call:

```text
Call parameter type does not match function signature!
i32 0
<8 x i32> ... @__ppu_mma_m16n16k32_mma_s32_s8_s8(..., i32 0, i32 0)
```

This is specifically a frontend/lowering type failure. It does not prove
that the silicon lacks a zero-accumulate instruction mode.

One-product tests alone cannot rule out sharing a zero vector. A second
probe retains8 independent products through4 MMA steps. In-place zeros,
one shared constant vector, and one shared vector produced by eight opaque
inline-asm zero moves all emit the **same197-instruction body**,32 MMAs and
64 C clears,76 vector/80 scalar registers,stack0. The current lowering did
not retain the requested shared zero source; each first product uses its
own destination group as C. No source-only spelling from this screen is a win.

These final counts use all relevant production HGGC lowering flags, including
SIMT-branch/fence/fix-uninit and matrix/async/load/store sinking. The initial
bare-O3 screen had24/192 instructions and32/78 vector registers, respectively;
it had the same no-gain conclusion but is not substituted for the shipping-
flags replay. Both evidence directories are retained locally.

## The other large chain remains separate

Per warp/K64, the current native CFG has fixed arithmetic counts on all paths:

| Restoration/scaling operation | PPU source711ec99 hot path | NVIDIA FP8 FP16-buffer measured average |
|---|---:|---:|
| S32 to FP32 |192|64|
| FP32 multiply |395|~3.33|
| FP32 add |136|~64.01|
| FP32 FMA |132|200|
| Half restore plus add (`HADD2.F32`) |0|128|
| Sum of these disjoint categories |855|~459.34|

The common64 QK casts cancel. Native's128 half-restore/add instructions must
also be counted; PV restoration is not free there. The roughly396-instruction
chain difference contains P scaling, separately materialized score arithmetic,
and K64-local V/P scaling. It is not all integer conversion or all unavoidable.
The zero-clear sequence adds another192 **outside** this table.

These are specific component accounts: current PPU generated hot paths versus
the recorded NVIDIA dynamic average (which includes amortized cold work).
They are **not** a new PPU dynamic total or an exact closure of a measured gap.
See the constant-V FMA and global-V-scale counterexamples in
`PPU_SCORE_LOWERING_SCREEN.md`; do not silently port those shortcuts to win
an instruction comparison. The shipped candidate's own ACU/normal latency
and numeric gate remain pending.

## Reproduce and decision

```bash
PPU_SDK=/path/to/PPU_SDK \
SHIPPING_ISA=/workspace/clean711ec99/shipping-isa.log \
OUT=/workspace/sage-mma-zero-screen \
  bash dev/ppu_int8/run_mma_zero_screen.sh
```

Missing compiler/disassembler is SKIP, not PASS. Unexpected compile failures
are FAIL. Literal-scalar rejection is accepted only for the observed type
failure; if it starts compiling, the screen demands re-evaluation. Successful
probes cover9+2 native instances without a private stack; four planted
evidence defects must be red.

Decision: **do not change production for these zero-source spellings**.
Preserve this reproducer as a compiler optimization request/boundary. The
current box candidate and default FP16 PV remain unchanged. No speed inferred
from the192 clears, and no claim that the overall FP8-alignment goal is done.
