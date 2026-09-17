#!/usr/bin/env python3
"""Prove each compiled integer-PV specialization has no floating MMA fallback."""
import argparse
from pathlib import Path
import re
import subprocess
import itertools


def check(text, plant=None, include_permuted=False):
    headers = list(re.finditer(r"Disassembly of section \.text\.kernel\.([^:\n]+):", text))
    rows = []
    for i, match in enumerate(headers):
        if "qk_int8_pv_kernel" not in match[1]:
            continue
        symbol = subprocess.check_output(["c++filt", match[1]], text=True).strip()
        template = re.search(r"qk_int8_pv_kernel<(64|128), (true|false), (true|false), cutlass::(half_t|bfloat16_t), (true|false)(?:, (true|false))?>", symbol)
        if not template:
            raise ValueError(f"unrecognized attention specialization: {symbol}")
        if template[5] != "true":
            continue
        body = text[match.end():headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        if plant == "floating-mma" and not rows:
            body += "\tv.mma.f32.f16.m16n16k16\tvreg[0:7]\n"
        opcodes = re.findall(r"\bv\.mma\.[^\s]+", body)
        floating = [op for op in opcodes if op.startswith("v.mma.f")]
        qk = sum(op == "v.mma.i32.i8.i8.m16n16k32" for op in opcodes)
        pv = sum(op == "v.mma.i32.u8.i8.m16n16k32" for op in opcodes)
        expected = int(template[1]) // 4
        if floating or qk != expected or pv != expected:
            raise ValueError(f"INT8 PV codegen mismatch {template[0]}: QK={qk}/{expected} PV={pv}/{expected} floating={floating}")
        params = template.groups()[:5] + (template[6] or "false",)
        rows.append((params, qk, pv))
    if plant == "missing-specialization" and rows:
        rows.pop()
    expected_keys = ("false", "true") if include_permuted else ("false",)
    expected = set(itertools.product(("64", "128"), ("false", "true"),
                   ("false", "true"), ("half_t", "bfloat16_t"), ("true",), expected_keys))
    if len(rows) != len(expected) or {row[0] for row in rows} != expected:
        raise ValueError(f"integer-PV specialization census {len(rows)}/{len(expected)}")
    for params, qk, pv in rows:
        print(f"[all-int8 ISA] D={params[0]} causal={params[1]} lse={params[2]} "
              f"out={params[3]} permuted_K={params[5]} static_QK={qk} static_PV={pv} floating_MMA=0")
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("isa", type=Path)
    p.add_argument("--plant", choices=("floating-mma", "missing-specialization"))
    p.add_argument("--permuted-k", action="store_true", help="Require all 16 additional explicit K-layout types")
    args = p.parse_args()
    try:
        count = check(args.isa.read_text(), args.plant, args.permuted_k)
    except ValueError as e:
        print(f"[all-int8 ISA] FAIL: {e}")
        return 1
    print(f"[all-int8 ISA] PASS: {count}/{32 if args.permuted_k else 16} exact shipping specializations; device execution NOT RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
