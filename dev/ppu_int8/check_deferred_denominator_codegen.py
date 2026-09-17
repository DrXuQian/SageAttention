#!/usr/bin/env python3
"""Real-body placement gate: 8 denominator shuffles move from K to epilogue."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess

from account_hot_loop import loop_inventory, parents, reachable, topological
from check_value_scale_codegen import resources
from native_isa import parse_native


def stream(kernel):
    return [(r.opcode, re.sub(r"<BB\w+>", "<BB>", r.operands))
            for r in kernel.instructions.values()]


def placement(kernel, mma_per_visit):
    """Causal loops can exit before any MMA; classify their real SCC directly.

    This proves placement, not a shortest-path dynamic count. A zero-iteration
    exit is not a K64 visit. The strict H3 path bound is checked separately.
    """
    mma = {pc for pc, r in kernel.instructions.items() if r.opcode.startswith("v.mma.")}
    if not mma:
        raise ValueError("no integer MMA body")
    first = min(mma)
    loop = reachable(kernel.edges, first) & reachable(parents(kernel.edges), first)
    if not mma <= loop:
        raise ValueError("MMA outside the one K-loop SCC")
    reverse = parents(kernel.edges)
    header, = [n for n in loop if reverse[n] - loop]
    topological({n: (kernel.edges[n] & loop) - {header} for n in loop})
    cold = kernel.instructions.keys() - loop
    topological({n: kernel.edges[n] & cold for n in cold})
    for op in ("v.mma.i32.i8.i8.m16n16k32", "v.mma.i32.u8.i8.m16n16k32"):
        if sum(kernel.instructions[n].opcode == op for n in loop) != mma_per_visit:
            raise ValueError("integer MMA body denominator differs")
    hot = {n for n in loop if kernel.instructions[n].opcode.startswith("v.shuffle.")}
    final = {n for n in cold if kernel.instructions[n].opcode.startswith("v.shuffle.")}
    if any(reachable(kernel.edges, n) & loop for n in final):
        raise ValueError("a claimed final shuffle precedes or reenters K")
    return dict(loop_sites=len(hot), post_loop_sites=len(final)), loop


def check(before, after, resource_text, plant=None):
    old, new = parse_native(before), parse_native(after)
    regs = resources(resource_text)
    if old.keys() != new.keys() or new.keys() != regs.keys() or len(new) != 48:
        raise ValueError("complete native/resource denominator must be 48/48")
    if any(r["private_stack"] for r in regs.values()):
        raise ValueError("nonzero private stack")
    if plant == "old-placement":
        new = old
    rows = []
    for name, kernel in new.items():
        if "qk_int8_pv_kernel" not in name:
            if stream(old[name]) != stream(kernel):
                raise ValueError("quantization native body changed")
            continue
        sig = subprocess.check_output(["c++filt", name], text=True)
        m = re.search(r"qk_int8_pv_kernel<(64|128), (true|false), (true|false), "
                      r"cutlass::(half_t|bfloat16_t), (true|false)>", sig)
        if not m:
            raise ValueError("unknown attention specialization")
        if m[5] == "false":
            actual = stream(kernel)
            if plant == "fp16-changed":
                actual[0] = ("WRONG", actual[0][1])
            if actual != stream(old[name]):
                raise ValueError("legacy FP16 PV body changed")
            continue
        a, _ = placement(old[name], int(m[1]) // 4)
        b, loop = placement(kernel, int(m[1]) // 4)
        if plant in ("missing-final", "extra-hot"):
            points = [pc for pc, r in kernel.instructions.items() if
                      r.opcode.startswith("v.shuffle.") and (pc not in loop)]
            if not points:
                raise ValueError("no finalization to mutate")
            pc = points[0] if plant == "missing-final" else min(loop)
            row = kernel.instructions[pc]
            op = "v.mov.b32" if plant == "missing-final" else "v.shuffle.bfly.b32"
            kernel.instructions[pc] = replace(row, opcode=op)
            b, loop = placement(kernel, int(m[1]) // 4)
        if (a["loop_sites"], a["post_loop_sites"]) != (48, 0):
            raise ValueError("baseline is not the a49338f eager denominator")
        if (b["loop_sites"], b["post_loop_sites"]) != (40, 8):
            raise ValueError("expected K-loop shuffles 48->40 and exactly 8 final shuffles")
        counts = [Counter(r.opcode for r in k.instructions.values()) for k in (old[name], kernel)]
        for op in ("v.mma.i32.i8.i8.m16n16k32", "v.mma.i32.u8.i8.m16n16k32",
                   "v.cnvt.i32.f32.rtte", "v.pcnvt.b16.u8x2", "s.blksyn.defer"):
            if counts[0][op] != counts[1][op]:
                raise ValueError(f"unchanged MMA/P-pack/cadence changed: {op}")
        rows.append(dict(specialization=m.groups(), shuffle_before=a, shuffle_after=b,
                         resources=regs[name]))
        if m.groups() == ("128", "false", "false", "bfloat16_t", "true"):
            prior, _ = loop_inventory(old[name])
            current, _ = loop_inventory(kernel)
            b = current["groups"]["shuffle"]
            if (b["cfg_path_min"], b["cfg_path_max"], b["once_only_union_static"]) != (40, 40, 8):
                raise ValueError("H3 executed-path placement differs")
            rows[-1].update(all_before=prior["groups"]["all"], all_after=current["groups"]["all"])
            if regs[name]["vector_registers"] > 246:
                raise ValueError("H3 vector registers exceed registered 246 baseline")
            print("[deferred denominator ISA H3]", json.dumps(rows[-1]))
    if len(rows) != 16:
        raise ValueError("integer-PV census must be 16/16")
    return rows


def main():
    p = argparse.ArgumentParser()
    for field in ("before", "after", "resources"):
        p.add_argument("--" + field, required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--plant", choices=("old-placement", "missing-final", "extra-hot", "fp16-changed"))
    args = p.parse_args()
    try:
        rows = check(args.before.read_text(), args.after.read_text(), args.resources.read_text(), args.plant)
        result = dict(scope="NATIVE_CODEGEN_ONLY; DEVICE_NOT_RUN", rows=rows,
                      sha256={name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                              for name in ("before", "after", "resources")})
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("[deferred denominator ISA] PASS: INT8=16/16 K-loop48->40 + final8; FP16=16/16 unchanged; quant=16/16 unchanged")
        return 0
    except ValueError as error:
        print("[deferred denominator ISA] FAIL:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
