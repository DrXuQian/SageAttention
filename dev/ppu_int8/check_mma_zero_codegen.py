#!/usr/bin/env python3
"""Account exact native zeroing and prove that each initial MMA sees eight zeros."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from account_hot_loop import GROUPS, loop_inventory, topological
from check_key_permutation_codegen import indexed
from check_deferred_denominator_codegen import stream
from native_isa import parse_native


def registers(operand):
    m = re.fullmatch(r"vreg\[(\d+):(\d+)\](?:\.reuse)?", operand)
    if m:
        return list(range(int(m[1]), int(m[2]) + 1))
    m = re.fullmatch(r"vreg(\d+)(?:\.reuse)?", operand)
    return [int(m[1])] if m else []


def initial_zeroes(kernel, products, steps):
    known, accumulators = {}, []
    count = initial = 0
    for pc, row in kernel.instructions.items():
        args = row.operands.split(", ")
        if row.opcode.startswith("v.mma."):
            if len(args) != 4 or len(registers(args[3])) != 8:
                raise ValueError("unknown MMA accumulator form")
            dest, source = registers(args[0]), registers(args[3])
            if count < products:
                if not all(known.get(r) == 0 for r in source):
                    raise ValueError("initial C was not eight initialized zeros")
                accumulators.append(dest)
                initial += len(source)
            elif source != accumulators[count % products]:
                raise ValueError("later MMA did not consume its own prior partial")
            accumulators[count % products] = dest
            count += 1
            for reg in dest:
                known[reg] = None
        elif row.opcode.startswith("vmem.st"):
            continue  # The first operand of a store is a source.
        else:
            for reg in registers(args[0]):
                known[reg] = 0 if row.opcode == "v.mov.b32" and args[1:] == ["0x0"] else None
    if count != products * steps or initial != products * 8:
        raise ValueError("MMA/first-C denominator mismatch")
    return dict(mma=count, first_c_initialized=initial,
                instructions=len(kernel.instructions),
                explicit_zero_moves=sum(r.opcode == "v.mov.b32" and r.operands.endswith(", 0x0")
                                        for r in kernel.instructions.values()))


def shipping_clear_provenance(kernel):
    info, loop = loop_inventory(kernel)
    header = int(info["loop_header"], 16)
    dag = {n: (kernel.edges[n] & loop) - {header} for n in loop}
    order, incoming = topological(dag)
    states, used, first_products = {}, set(), {"QK": 0, "PV": 0}
    clears = {pc for pc in loop if kernel.instructions[pc].opcode == "v.mov.b32"
              and kernel.instructions[pc].operands.endswith(", 0x0")}
    for pc in order:
        prior = [states[p] for p in incoming[pc]]
        state = {}
        if prior:
            # A C register is known zero only if EVERY incoming path defines
            # it as zero. Keep the PCs, not just a coincident instruction count.
            for reg in set.intersection(*(set(p) for p in prior)):
                state[reg] = frozenset().union(*(p[reg] for p in prior))
        row = kernel.instructions[pc]
        args = row.operands.split(", ")
        if row.opcode.startswith("v.mma."):
            source = registers(args[3])
            if len(source) == 8 and all(reg in state for reg in source):
                used.update(frozenset().union(*(state[r] for r in source)))
                role = "PV" if ".u8.i8." in row.opcode else "QK"
                first_products[role] += 1
        if not row.opcode.startswith("vmem.st"):
            for reg in registers(args[0]):
                state.pop(reg, None)
                if pc in clears:
                    state[reg] = frozenset((pc,))
        states[pc] = state
    if first_products != {"QK": 8, "PV": 16} or used != clears or len(used) != 192:
        raise ValueError(f"shipping clear def-use not closed: {first_products} used={len(used)}/{len(clears)}")
    return dict(initial_products=first_products, all_path_zero_definitions=len(used),
                scope="one real native K-loop DAG; known-zero intersection at every join")


def check(probes, literal, shipping, plant=None):
    small, raw = parse_native(probes), parse_native(literal)
    required = {f"zero_{mode}_{kind}" for mode in ("inplace", "operand", "builtin") for kind in ("s8", "u8")}
    required |= {"many_zero_inplace", "many_zero_shared_const", "many_zero_shared_opaque"}
    if plant == "missing-type":
        small.pop("zero_operand_s8", None)
    if small.keys() != required or raw.keys() != {"zero_literal_s8", "zero_literal_u8"}:
        raise ValueError("probe census differs")
    if plant == "missing-clear":
        k = small["zero_operand_s8"]
        first = next(r for r in k.instructions.values() if r.opcode.startswith("v.mma."))
        reg = registers(first.operands.split(", ")[3])[0]
        pc = next(pc for pc, r in k.instructions.items() if r.opcode == "v.mov.b32" and r.operands == f"vreg{reg}, 0x0")
        k.instructions[pc] = replace(k.instructions[pc], opcode="s.nop", operands="")
    if plant == "missing-mma":
        k = small["zero_operand_s8"]
        pc = next(pc for pc, r in k.instructions.items() if r.opcode.startswith("v.mma."))
        k.instructions[pc] = replace(k.instructions[pc], opcode="s.nop", operands="")
    rows = {}
    for name, kernel in (small | raw).items():
        many = name.startswith("many_")
        rows[name] = initial_zeroes(kernel, 8 if many else 1, 4 if many else 2)
    for kind in ("s8", "u8"):
        before = stream(small[f"zero_inplace_{kind}"])
        for mode in ("operand", "builtin"):
            if before != stream(small[f"zero_{mode}_{kind}"]):
                raise ValueError("zero source alternative differs: re-evaluate instead of claiming identical")
        if before != stream(raw[f"zero_literal_{kind}"]):
            raise ValueError("literal tuple differs: re-evaluate")
    before = stream(small["many_zero_inplace"])
    for mode in ("shared_const", "shared_opaque"):
        if before != stream(small[f"many_zero_{mode}"]):
            raise ValueError("shared zero changed native output: re-evaluate")

    kernels, _ = indexed(shipping, True)
    target = ("attention", "128", "false", "false", "bfloat16_t", "true", "true")
    k = kernels[target]
    if plant == "shipping-clear-missing":
        _, loop = loop_inventory(k)
        pc = next(pc for pc in sorted(loop) if k.instructions[pc].opcode == "v.mov.b32"
                  and k.instructions[pc].operands.endswith(", 0x0"))
        k.instructions[pc] = replace(k.instructions[pc], opcode="s.nop", operands="")
    provenance = shipping_clear_provenance(k)
    for pc, r in tuple(k.instructions.items()):
        if r.opcode == "v.mov.b32" and r.operands.endswith(", 0x0"):
            k.instructions[pc] = replace(r, opcode="ACCOUNT_ZERO_MOVE")
    GROUPS["literal_zero_move"] = lambda op: op == "ACCOUNT_ZERO_MOVE"
    result, _ = loop_inventory(k)
    count = result["groups"]["literal_zero_move"]
    if (count["loop_union_static"], count["cfg_path_min"], count["cfg_path_max"]) != (192, 192, 192):
        raise ValueError("shipping K-loop zero denominator is not192 on every CFG path")
    arithmetic = {g: result["groups"][g] for g in
                  ("s32_to_f32", "f32_to_s32", "fp32_mul", "fp32_add", "fp32_fma")}
    return dict(probes=rows, shipping_zero_cfg=count, shipping_clear_provenance=provenance,
                shipping_arithmetic_cfg=arithmetic,
                scope="STATIC_NATIVE_EVIDENCE_NOT_NEW_DEVICE_COUNTS; SDK2.1.1 alternatives only")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("probes", "literal", "shipping"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--plant", choices=("missing-type", "missing-clear", "missing-mma", "shipping-clear-missing"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    try:
        paths = {k: getattr(args, k) for k in ("probes", "literal", "shipping")}
        result = check(*(p.read_text() for p in paths.values()), args.plant)
        result["input_sha256"] = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("[MMA zero screen] PASS: four first-product spellings identical, shared-zero alternatives identical; shipping CFG192 zero moves/K64; no production change")
        return 0
    except (ValueError, OSError) as error:
        print("[MMA zero screen] FAIL:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
