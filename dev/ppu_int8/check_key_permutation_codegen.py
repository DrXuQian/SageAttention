#!/usr/bin/env python3
"""Compare complete native bodies against the immutable deferred-denominator build.

Counts describe emitted CFG paths, not measured instructions or device speed.
The 48 old bodies must remain exact; the 24 new bodies are explicit opt-ins.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

from account_hot_loop import loop_inventory
from check_deferred_denominator_codegen import placement, stream
from native_isa import parse_native


def identity(name):
    text = subprocess.check_output(["c++filt", name], text=True)
    specs = (
        ("attention", r"qk_int8_pv_kernel<(64|128), (true|false), (true|false), "
         r"cutlass::(half_t|bfloat16_t), (true|false)(?:, (true|false))?>"),
        ("quant", r"quantize_int8_kernel<cutlass::(half_t|bfloat16_t), "
         r"(64|128), (32|64), (true|false)(?:, (true|false))?>"),
        ("value", r"quantize_value_int8_kernel<cutlass::(half_t|bfloat16_t), (64|128)>"),
    )
    for kind, pattern in specs:
        m = re.search(pattern, text)
        if m:
            return (kind,) + tuple(x or "false" for x in m.groups())
    raise ValueError("unrecognized kernel: " + text)


def expected(permuted):
    dims, flags, types = ("64", "128"), ("false", "true"), ("half_t", "bfloat16_t")
    rows = set()
    for d, causal, lse, dtype, integer in itertools.product(dims, flags, flags, types, flags):
        rows.add(("attention", d, causal, lse, dtype, integer, "false"))
        if permuted and integer == "true":
            rows.add(("attention", d, causal, lse, dtype, integer, "true"))
    for dtype, d in itertools.product(types, dims):
        rows.add(("value", dtype, d))
        rows.add(("quant", dtype, d, "32", "false", "false"))
        for mean in flags:
            rows.add(("quant", dtype, d, "64", mean, "false"))
            if permuted:
                rows.add(("quant", dtype, d, "64", mean, "true"))
    return rows


def indexed(text, permuted):
    kernels, names = {}, {}
    for name, body in parse_native(text).items():
        key = identity(name)
        if key in kernels:
            raise ValueError("duplicate specialization")
        kernels[key], names[key] = body, name
    if kernels.keys() != expected(permuted):
        raise ValueError(f"native specialization denominator differs: {len(kernels)}")
    return kernels, names


def read_resources(text, names):
    by_symbol = {}
    for m in re.finditer(r"^Func \d+ (\S+) RESOURCE INFO:\n(.*?)"
                         r"(?=^Func \d+ .* RESOURCE INFO:|\Z)", text, re.M | re.S):
        if m[1] in by_symbol:
            raise ValueError("duplicate resource record")
        by_symbol[m[1]] = {key: int(re.search(pattern, m[2])[1]) for key, pattern in (
            ("vector_registers", r"vreg_number:(\d+)"),
            ("scalar_registers", r"sreg_number:(\d+)"),
            ("private_stack", r"STACK SIZE:(\d+)"))}
    if by_symbol.keys() != set(names.values()):
        raise ValueError("resource/native denominator differs")
    return {key: by_symbol[symbol] for key, symbol in names.items()}


def check(before, after, resource_text, plant=None):
    old, _ = indexed(before, False)
    new, names = indexed(after, True)
    regs = read_resources(resource_text, names)
    target = ("attention", "128", "false", "false", "bfloat16_t", "true", "true")
    if plant == "missing-type":
        new.pop(target)
    if new.keys() != expected(True):
        raise ValueError("new denominator must be 72/72")
    if plant == "old-body-changed":
        k = new[("attention", "128", "false", "false", "bfloat16_t", "false", "false")]
        pc = min(k.instructions)
        k.instructions[pc] = replace(k.instructions[pc], opcode="WRONG")
    for key, old_body in old.items():
        if stream(old_body) != stream(new[key]):
            raise ValueError("old native body changed: " + str(key))
    if plant == "register-regression":
        regs[target]["vector_registers"] = 248
    if plant == "spill":
        regs[target]["private_stack"] = 16
    if any(r["private_stack"] for r in regs.values()):
        raise ValueError("nonzero private stack")
    if regs[target]["vector_registers"] > 246:
        raise ValueError("H3 vector registers exceed preregistered246")
    if plant == "floating-mma":
        k = new[target]
        pc = next(pc for pc, r in k.instructions.items() if r.opcode.startswith("v.mma."))
        k.instructions[pc] = replace(k.instructions[pc], opcode="v.mma.f32.f16.m16n16k16")
    rows = []
    for key in sorted(new.keys() - old.keys()):
        if key[0] != "attention":
            continue
        body = new[key]
        prior = old[key[:-1] + ("false",)]
        counts = [Counter(r.opcode for r in k.instructions.values()) for k in (prior, body)]
        if any(op.startswith("v.mma.f") for op in counts[1]):
            raise ValueError("floating MMA in explicit integer path")
        for op in ("v.mma.i32.i8.i8.m16n16k32", "v.mma.i32.u8.i8.m16n16k32",
                   "v.cnvt.i32.f32.rtte", "v.pcnvt.b16.u8x2", "s.blksyn.defer"):
            if counts[0][op] != counts[1][op]:
                raise ValueError("MMA/P-pack/cadence changed: " + op)
        positions, _ = placement(body, int(key[1]) // 4)
        if positions != dict(loop_sites=40, post_loop_sites=8):
            raise ValueError("max/denominator/tail shuffle inventory differs")
        rows.append(dict(specialization=key, resources=regs[key], shuffles=positions))
    if len(rows) != 16:
        raise ValueError("new attention denominator must be16")
    info, _ = loop_inventory(new[target])
    before_info, _ = loop_inventory(old[target[:-1] + ("false",)])
    if plant == "full-path-not-removed":
        info = before_info
    shuffle = info["groups"]["shuffle"]
    if (shuffle["cfg_path_min"], shuffle["cfg_path_max"], shuffle["once_only_union_static"]) != (8, 40, 8):
        raise ValueError("H3 paths must separate full-block8 from tail40 and final8")
    for group in ("s32_to_f32", "f32_to_s32", "fp32_mul", "fp32_add", "fp32_fma", "qk_mma", "pv_mma"):
        fields = ("cfg_path_min", "cfg_path_max")
        if any(info["groups"][group][f] != before_info["groups"][group][f] for f in fields):
            raise ValueError("H3 arithmetic path changed: " + group)
    return dict(old_native_unchanged=48, new_attention=16, new_quant=8,
                rows=rows, h3_before=before_info, h3_after=info)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "resources"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--plant", choices=("missing-type", "old-body-changed", "register-regression",
                                      "spill", "floating-mma", "full-path-not-removed"))
    args = p.parse_args()
    try:
        paths = {key: getattr(args, key) for key in ("before", "after", "resources")}
        result = check(*(paths[k].read_text() for k in paths), args.plant)
        result["scope"] = "NATIVE_CFG_ONLY; DEVICE_NUMERICS/LATENCY_NOT_RUN"
        result["input_sha256"] = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("[K permutation ISA] PASS: old48 unchanged; explicit16 attention+8 quant; H3 no register/stack regression; CFG shuffle8..40")
        return 0
    except (ValueError, OSError) as error:
        print("[K permutation ISA] FAIL:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
