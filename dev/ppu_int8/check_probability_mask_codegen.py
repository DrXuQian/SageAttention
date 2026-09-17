#!/usr/bin/env python3
"""Native mask-hoisting gate; keep arithmetic/cadence and FP16 PV unchanged."""
import argparse
from pathlib import Path
import re
import subprocess
from check_requant_codegen import sections, counters, summary


def check_probe(text, plant=False):
    rows = sections(text)
    if set(rows) != {"probability_mask_before", "probability_mask_after"}:
        raise ValueError("mask probe inventory changed")
    old = counters(rows["probability_mask_before"])
    new = counters(rows["probability_mask_after"])
    if plant:
        new["v.csel.b32"] += 1
    comparisons = lambda c: sum(n for op, n in c.items()
        if re.match(r"v\.(u?cmp)\..*f32", op))
    if comparisons(old) != 4 or comparisons(new) != 1:
        raise ValueError("expected four original predicates and one row predicate")
    if old["v.csel.b32"] != 4 or new["v.csel.b32"] != 1:
        raise ValueError("per-P mask select returned")
    for op in ("v.add.f32", "v.exp2.f32"):
        if old[op] != 4 or new[op] != 4:
            raise ValueError("mask optimization changed exponent arithmetic")
    print("[P mask ISA] probe predicates/selects=4->1 exp/add=4->4 PASS")


def check_shipping(before, after):
    old, new = sections(before), sections(after)
    if old.keys() != new.keys():
        raise ValueError("native specialization denominator changed")
    fp16, int8 = 0, 0
    normalize = lambda rows: [(op, re.sub(r"<BB\w+>", "<BB>", arg)) for op, arg in rows]
    for name in old:
        if "qk_int8_pv_kernel" not in name:
            continue
        sig = subprocess.check_output(["c++filt", name], text=True)
        m = re.search(r"qk_int8_pv_kernel<(64|128), (true|false), (true|false), "
                      r"cutlass::(half_t|bfloat16_t), (true|false)>", sig)
        if not m:
            raise ValueError("unknown attention symbol")
        if m[5] == "false":
            if normalize(old[name]) != normalize(new[name]):
                raise ValueError("FP16 PV instruction stream changed")
            fp16 += 1
            continue
        a, b = counters(old[name]), counters(new[name])
        for op in ("v.mma.i32.i8.i8.m16n16k32", "v.mma.i32.u8.i8.m16n16k32",
                   "v.cnvt.i32.f32.rtte", "v.cnvt.f32.i32.rtte", "v.exp2.f32",
                   "v.mul.f32", "v.add.f32", "v.fma.f32.rtte", "s.blksyn.defer"):
            if a[op] != b[op]:
                raise ValueError(f"math/cadence changed: {op}, {sig}")
        if b["v.ucmp.gt.f32"] or b["v.cmp.gt.f32"] != 4:
            raise ValueError("shipping row-level mask contract changed")
        int8 += 1
        if m.groups() == ("128", "false", "false", "bfloat16_t", "true"):
            print("[P mask ISA] H3 before=", summary(a), "after=", summary(b))
    if fp16 != 16 or int8 != 16:
        raise ValueError("expected 16 FP16 plus 16 INT8 attention symbols")
    print("[P mask ISA] FP16 unchanged=16/16 integer math/cadence=16/16 PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    check_probe(args.probe.read_text(), args.plant)
    if bool(args.before) != bool(args.after):
        parser.error("both shipping operands required")
    if args.before:
        check_shipping(args.before.read_text(), args.after.read_text())


if __name__ == "__main__":
    main()
