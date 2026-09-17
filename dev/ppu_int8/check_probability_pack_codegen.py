#!/usr/bin/env python3
"""Retain the native counterfactual behind the inverse-magic pack rejection."""
from pathlib import Path
import sys
from check_requant_codegen import sections, counters, summary

rows = sections(Path(sys.argv[1]).read_text())
assert set(rows) == {"p_pack_current", "p_pack_magic_strict", "p_pack_magic_fused"}
a, b, c = (counters(rows[n]) for n in ("p_pack_current", "p_pack_magic_strict", "p_pack_magic_fused"))
assert a["v.mul.f32"] == 4 and a["v.cnvt.i32.f32.rtte"] == 4 and a["v.pcnvt.b16.u8x2"] == 2
assert b["v.mul.f32"] == 4 and b["v.add.f32"] == 4 and b["v.byte.prmt.b32"] == 3
assert c["v.fma.f32.rtte"] == 4 and c["v.byte.prmt.b32"] == 3
for name in rows:
    print("[P inverse pack ISA]", name, summary(counters(rows[name])))
assert sum(b.values()) > sum(a.values()), "revisit the rejected candidate: codegen changed"
print("[P inverse pack ISA] strict magic is not shorter; fused is a changed-rounding negative, not admitted; PASS")
