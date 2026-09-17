#!/usr/bin/env python3
"""Native PPU static-ISA A/B. Never treats static counts as device time."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from native_isa import parse_native


def sections(text):
    return {name: [(row.opcode, row.operands) for row in kernel.instructions.values()]
            for name, kernel in parse_native(text).items()}


def counters(lines):
    return Counter(op for op, _ in lines)


def integer_alu(op):
    # Fixed scope: integer arithmetic, compares/selects, and bit/byte operations.
    # Conversion, shuffle, mov, scalar branches/waits and loads are separate.
    if re.search(r"\.(?:f16|f32|f64|bf16)(?:\.|$)", op):
        return False
    return re.match(r"(?:v|s)\.(?:add|sub|mul|mull|mulh|madl|madw|mad|shll|shrl|shra|"
                    r"and|or|xor|xnor|not|lop3|bfi|bfe|min|max|cmp|ucmp|csel|"
                    r"byte\.prmt|pcnvt)(?:\.|$)", op) is not None


def summary(counts):
    return dict(static_instructions=sum(counts.values()),
                integer_bit_alu=sum(n for op, n in counts.items() if integer_alu(op)),
                shuffles=sum(n for op, n in counts.items() if op.startswith("v.shuffle.")),
                conversions=sum(n for op, n in counts.items() if op.startswith("v.cnvt.")),
                float_multiply=counts["v.mul.f32"],
                mma=sum(n for op, n in counts.items() if op.startswith("v.mma.")),
                barriers=counts["s.blksyn.defer"])


def probe_check(text, plant=None):
    raw = sections(text)
    names = {"requant_before", "requant_after", "transpose_before", "transpose_after"}
    if set(raw) != names:
        raise ValueError(f"probe census {sorted(raw)} != {sorted(names)}")
    data = {name: counters(lines) for name, lines in raw.items()}
    if plant == "scalar-clamp":
        data["requant_after"]["v.min.i32"] += 1
    if plant == "extra-shuffle":
        data["transpose_after"]["v.shuffle.idx.b32"] += 1
    old, new = data["requant_before"], data["requant_after"]
    if old["v.min.i32"] != 4 or old["v.max.i32"] != 4:
        raise ValueError("the retained scalar negative no longer reproduces four clamps")
    if (new["v.pcnvt.b16.u8x2"] != 2 or new["v.min.i32"] or new["v.max.i32"] or
            new["v.mul.f32"] != 4 or new["v.cnvt.i32.f32.rtte"] != 4):
        raise ValueError("quantization must keep 4 FP32 multiplies + 4 RN conversions + 2 U8 packs")
    before = summary(data["transpose_before"])
    after = summary(data["transpose_after"])
    if before["shuffles"] != 16 or after["shuffles"] != 8:
        raise ValueError("four-word transpose must shrink 16 gathers to 8 butterfly exchanges")
    if data["transpose_after"]["v.byte.prmt.b32"] != 8:
        raise ValueError("four-word transpose lost its 8 native byte permutations")
    return {name: dict(summary=summary(c), opcodes=dict(sorted(c.items()))) for name, c in data.items()}


def shipping_compare(before, after):
    old, new = sections(before), sections(after)
    if set(old) != set(new):
        raise ValueError("shipping symbol inventories differ")
    results = []
    seen = set()
    for name in old:
        if "qk_int8_pv_kernel" not in name:
            continue
        demangled = subprocess.check_output(["c++filt", name], text=True).strip()
        match = re.search(r"qk_int8_pv_kernel<(64|128), (true|false), (true|false), "
                          r"cutlass::(half_t|bfloat16_t), (true|false)>", demangled)
        if not match:
            raise ValueError(f"unrecognized attention specialization {demangled}")
        seen.add(match.groups())
        a, b = counters(old[name]), counters(new[name])
        if match[5] == "false":
            # Exact native instruction/operand stream, allowing renamed BB labels.
            normalize = lambda rows: [(op, re.sub(r"<BB\w+>", "<BB>", operands)) for op, operands in rows]
            if normalize(old[name]) != normalize(new[name]):
                raise ValueError(f"unrelated FP16 PV code changed: {demangled}")
        else:
            for op in ("v.mma.i32.i8.i8.m16n16k32", "v.mma.i32.u8.i8.m16n16k32",
                       "v.cnvt.i32.f32.rtte", "v.cnvt.f32.i32.rtte", "v.exp2.f32",
                       "v.mul.f32", "v.add.f32", "v.fma.f32.rtte",
                       "s.blksyn.defer"):
                if a[op] != b[op]:
                    raise ValueError(f"math/cadence invariant changed: {op} in {demangled}")
            # Causal grid/K-loop bounds have an unrelated min/max pair. Only
            # the 64 per-P clamps are removed; do not require zero globally.
            if (a["v.min.i32"] - b["v.min.i32"] != 64 or
                    a["v.max.i32"] - b["v.max.i32"] != 64 or
                    b["v.pcnvt.b16.u8x2"] != 32):
                raise ValueError(f"shipping P requant lost the 32-pack/no-clamp body: {demangled}")
            if a["v.shuffle.idx.b32"] != 64 or b["v.shuffle.idx.b32"] != 0:
                raise ValueError(f"shipping P gather was not removed: {demangled}")
        results.append(dict(specialization=match.groups(), before=summary(a), after=summary(b),
                            opcode_delta={op: b[op] - a[op] for op in sorted(a.keys() | b.keys()) if a[op] != b[op]}))
    if len(seen) != 32:
        raise ValueError(f"attention census {len(seen)}/32")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plant", choices=("scalar-clamp", "extra-shuffle"))
    args = parser.parse_args()
    try:
        result = dict(scope="STATIC_ISA_ONLY; no new device result", probes=probe_check(args.probe.read_text(), args.plant))
        if bool(args.before) != bool(args.after):
            raise ValueError("shipping comparison requires both --before and --after")
        if args.before:
            result["shipping"] = shipping_compare(args.before.read_text(), args.after.read_text())
            result["isa_sha256"] = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for label, path in (("before", args.before), ("after", args.after))}
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        for name, row in result["probes"].items():
            print("[requant probe]", name, json.dumps(row["summary"]))
        for row in result.get("shipping", []):
            if row["specialization"] == ("128", "false", "false", "bfloat16_t", "true"):
                print("[requant shipping H3]", json.dumps(row))
        print("[requant ISA] PASS: no timing or device numeric verdict inferred")
        return 0
    except ValueError as error:
        print(f"[requant ISA] FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
