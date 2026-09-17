#!/usr/bin/env python3
"""Real SDK score microprobe screen; no full-kernel or latency inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from native_isa import parse_native


def inspect(path, plant=None):
    kernels = parse_native(path.read_text())
    if set(kernels) != {"score_current", "score_biased_difference"}:
        raise ValueError("score microprobe census must be exactly two")
    data = {name: Counter(i.opcode for i in kernel.instructions.values())
            for name, kernel in kernels.items()}
    old, new = data["score_current"], data["score_biased_difference"]
    if plant == "hidden-casts":
        new["v.cnvt.f32.i32.rtte"] += 1
    if plant == "missing-mma":
        new["v.mma.i32.i8.i8.m16n16k32"] -= 1
    if plant == "omit-maximum-restore":
        new["v.mul.f32"] -= 2
    if old["v.cnvt.f32.i32.rtte"] != 32 or new["v.cnvt.f32.i32.rtte"] != 0:
        raise ValueError("score casts did not change 32 -> 0")
    for name, row in data.items():
        if row["v.mma.i32.i8.i8.m16n16k32"] != 16:
            raise ValueError(f"real integer QK MMA missing: {name}")
        if row["v.exp2.f32"] != 32 or row["v.shuffle.bfly.b32"] != 4:
            raise ValueError("probabilities or full row maximum were omitted")
    if (old["v.mul.f32"], new["v.mul.f32"], old["v.add.f32"], new["v.add.f32"]) != (32, 34, 32, 34):
        raise ValueError("count both restored/scaled maxima, not only removed casts")
    seed_sites = sum(i.opcode == "v.mov.b32" and "0x4b400000" in i.operands
                     for i in kernels["score_biased_difference"].instructions.values())
    if seed_sites != 32:
        raise ValueError("all 32 actual MMA accumulator registers must be seeded")
    return dict(scope="STATIC_MICROPROBE_POSITIVE_SCALE_UNMASKED; NOT ATTENTION_ADMISSION",
                isa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                kernels={name: dict(total=sum(row.values()), opcodes=dict(sorted(row.items())))
                         for name, row in data.items()},
                net_static_reduction=sum(old.values()) - sum(new.values()),
                seed_sites=seed_sites, device="NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isa", type=Path)
    parser.add_argument("--plant", choices=("hidden-casts", "missing-mma", "omit-maximum-restore"))
    args = parser.parse_args()
    try:
        result = inspect(args.isa, args.plant)
    except ValueError as error:
        print(f"[score native screen] FAIL: {error}")
        return 1
    print("[score native screen] " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
