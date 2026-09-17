#!/usr/bin/env python3
"""Check the real shipping V-scale A/B, not a model of global addresses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from check_probability_mask_codegen import check_shipping
from check_requant_codegen import counters, sections, summary


def resources(text):
    result = {}
    pattern = r"^Func \d+ (\S+) RESOURCE INFO:\n(.*?)(?=^Func \d+ .* RESOURCE INFO:|\Z)"
    for match in re.finditer(pattern, text, re.M | re.S):
        if match[1] in result:
            raise ValueError("duplicate resource symbol")
        result[match[1]] = {key: int(re.search(regex, match[2])[1]) for key, regex in (
            ("vector_registers", r"vreg_number:(\d+)"),
            ("scalar_registers", r"sreg_number:(\d+)"),
            ("private_stack", r"STACK SIZE:(\d+)"))}
    if len(result) != 48:
        raise ValueError("expected 48 unique resource records")
    return result


def compare(before, after, old_resources, new_resources, plant=None):
    check_shipping(before, after)  # All FP16 streams + INT8 math/barriers.
    old, new = sections(before), sections(after)
    if plant == "missing-target":
        del new[next(n for n in new if "qk_int8_pv_kernel" in n)]
    if old.keys() != new.keys() or old.keys() != old_resources.keys() or old.keys() != new_resources.keys():
        raise ValueError("ISA/resource denominator mismatch")
    rows = []
    unchanged = 0
    for name in old:
        if new_resources[name]["private_stack"]:
            raise ValueError("new private stack is not admitted")
        sig = subprocess.check_output(["c++filt", name], text=True)
        match = re.search(r"qk_int8_pv_kernel<(64|128), (true|false), (true|false), "
                          r"cutlass::(half_t|bfloat16_t), true>", sig)
        if not match:
            continue
        a, b = counters(old[name]), counters(new[name])
        if plant == "repeated-global-load":
            b["vmem.ld.b32"] += 1
        dim = int(match[1])
        if dim != 128 or match[2] == "true":
            normalize = lambda lines: [(op, re.sub(r"<BB\w+>", "<BB>", arg)) for op, arg in lines]
            if normalize(old[name]) != normalize(new[name]):
                raise ValueError("an unselected D64/causal stream changed")
            unchanged += 1
            continue
        if a["vmem.ld.b32"] - b["vmem.ld.b32"] != dim // 4 - 1:
            raise ValueError("V scale must replace D/4 scalar global loads with one")
        if (b["tsm.ld.b32"] - a["tsm.ld.b32"] != dim // 4 or
                b["tsm.st.b32"] - a["tsm.st.b32"] != 1):
            raise ValueError("one shared publication and D/4 plain float loads required")
        prior, now = summary(a), summary(b)
        if now["integer_bit_alu"] >= prior["integer_bit_alu"]:
            raise ValueError("V-scale delivery did not reduce integer/address overhead")
        # Record both register classes. Do not hide scalar-register growth by
        # citing only fewer vector registers. Occupancy needs device evidence.
        row = dict(specialization=match.groups(), before=prior, after=now,
                   resources_before=old_resources[name], resources_after=new_resources[name],
                   shared_bytes_added=dim * 4,
                   opcode_delta={op: b[op] - a[op] for op in sorted(a.keys() | b.keys()) if a[op] != b[op]})
        rows.append(row)
        if match.groups() == ("128", "false", "false", "bfloat16_t"):
            print("[V-scale ISA H3]", json.dumps(row))
    if len(rows) != 4 or unchanged != 12:
        raise ValueError("INT8 denominator must be 4 selected plus 12 unchanged")
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("before", "after", "before-resources", "after-resources"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plant", choices=("repeated-global-load", "missing-target"))
    args = parser.parse_args()
    try:
        paths = {name: getattr(args, name) for name in (
            "before", "after", "before_resources", "after_resources")}
        rows = compare(args.before.read_text(), args.after.read_text(),
                       resources(args.before_resources.read_text()),
                       resources(args.after_resources.read_text()), args.plant)
        result = dict(scope="STATIC_NATIVE_ISA_ONLY; PPU candidate NOT RUN", specializations=rows,
                      evidence_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for name, path in paths.items()})
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("[V-scale ISA] D128/full chains reduced=4/4; other INT8 unchanged=12/12; FP16 unchanged=16/16; PASS")
        return 0
    except (ValueError, OSError) as error:
        print(f"[V-scale ISA] FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
