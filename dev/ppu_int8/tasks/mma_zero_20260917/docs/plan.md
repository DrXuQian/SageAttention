# MMA first-product zero-source screen

Parent37ac724. Do not change published K-permuted candidate or FP16 default.
Priority: same H3/Q128/K64/D128 geometry and arithmetic as source711ec99.

Hypothesis: many remaining vector moves initialize S32 accumulator groups.
Compare the existing in-place zeroed atom, the same production atom with a
separate zero C operand, SDK vector builtin with zero C, and two explicit
PTX zero spellings. Full output remains observable; first+second MMAs must
remain present. Real SDK compile/disassembly only, no device performance claim.

Admission: exactly two same-type integer MMA calls, no floating fallback,
no private stack, all eight first C registers provably zero (or a literal
native zero source), second C comes from the first product. Count full native
instruction streams, not just zero moves. If every accepted spelling emits
the same eight clears, record a toolchain boundary, not proof that no future
ISA/compiler design can do better. Unsupported raw spellings are evidence,
not permission to weaken atom/operand checks.

Do not publish another production binary without a substantive win and full
body verification. Existing box command remains bound to42702fd4/711ec99.
