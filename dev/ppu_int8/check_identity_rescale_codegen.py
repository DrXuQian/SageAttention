#!/usr/bin/env python3
"""Account a real full-body rescale experiment; rejected resources stay rejected.

This is a local compiler screen, not latency or device numeric admission.
The result records both path extremes, never an invented branch frequency.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from account_hot_loop import loop_inventory
from check_deferred_denominator_codegen import stream
from check_key_permutation_codegen import indexed, read_resources


TARGET = ("attention", "128", "false", "false", "bfloat16_t", "true", "true")


def selected(key):
    return key[0] == "attention" and key[1:3] == ("128", "false") and key[-2:] == ("true", "true")


def identity_guards(kernel, loop):
    """Bind each real predicate's factor to exactly the32 multiplies it skips."""
    rows = list(kernel.instructions.values())
    following = {a.pc: b.pc for a, b in zip(rows, rows[1:])}
    guards, covered = [], set()
    for i, row in enumerate(rows):
        if row.pc not in loop or row.opcode != "v.ucmp.ne.f32":
            continue
        m = re.fullmatch(r"vcc, (vreg\d+), 0x3f800000", row.operands)
        if not m:
            continue
        # Lane-conditional: compare, emsk, branch. Uniform also has the
        # __any_sync vote lowering before its single scalar branch.
        branch = next((r for r in rows[i + 1:i + 9] if r.opcode.startswith("s.cbr")), None)
        if branch is None or len(kernel.edges[branch.pc]) != 2:
            raise ValueError("identity compare has no native bypass branch")
        first = following[branch.pc]
        join, = kernel.edges[branch.pc] - {first}
        seen, todo = set(), [first]
        while todo:
            pc = todo.pop()
            if pc == join or pc in seen:
                continue
            if pc not in loop:
                raise ValueError("identity branch escapes its K iteration")
            seen.add(pc)
            todo.extend(kernel.edges[pc])
        multiplies = {pc for pc in seen if kernel.instructions[pc].opcode == "v.mul.f32"}
        if len(multiplies) != 32 or covered & multiplies:
            raise ValueError("guard must bypass32 distinct output multiplies")
        for pc in multiplies:
            operands = kernel.instructions[pc].operands.replace(".reuse", "").split(", ")
            if len(operands) != 3 or sorted(operands[1:]) != sorted((operands[0], m[1])):
                raise ValueError("bypassed multiply does not use the compared exact-unit factor")
        covered.update(multiplies)
        guards.append(dict(compare=hex(row.pc), branch=hex(branch.pc), join=hex(join),
                           factor=m[1], multiplies=32))
    if len(guards) != 4 or len(covered) != 128:
        raise ValueError("expected four exact-unit predicates bound to128 multiplies")
    return guards


def compare(before, after, resource_text, plant=None):
    old, _ = indexed(before, True)
    new, names = indexed(after, True)
    resources = read_resources(resource_text, names)
    if plant == "legacy-changed":
        k = new[("attention", "128", "false", "false", "bfloat16_t", "false", "false")]
        pc = min(k.instructions)
        k.instructions[pc] = replace(k.instructions[pc], opcode="WRONG")
    unchanged = 0
    for key in old:
        if not selected(key):
            if stream(old[key]) != stream(new[key]):
                raise ValueError("unselected native body changed: " + str(key))
            unchanged += 1
        elif stream(old[key]) == stream(new[key]):
            raise ValueError("claimed rescale candidate is unchanged")
    if unchanged != 68:
        raise ValueError("unchanged specialization denominator must be68/72")
    if plant == "spill":
        resources[TARGET]["private_stack"] = 16
    if any(r["private_stack"] for r in resources.values()):
        raise ValueError("private stack/spill appeared")
    # Skipping multiply-by-one is not the same operation under FTZ-only.
    # Check the real kernel descriptor, not a host floating-mode assumption.
    for key in old:
        if selected(key):
            symbol = names[key]
            m = re.search(r"^Func \d+ " + re.escape(symbol) + r" RESOURCE INFO:\n(.*?)"
                          r"(?=^Func \d+ .* RESOURCE INFO:|\Z)", resource_text, re.M | re.S)
            if not m or not re.search(r"^fp_denorm_flush:0$", m[1], re.M) or plant == "ftz":
                raise ValueError("identity experiment requires native fp_denorm_flush=0")
    prior, _ = loop_inventory(old[TARGET])
    current, loop = loop_inventory(new[TARGET])
    if plant == "wrong-unit":
        k = new[TARGET]
        pc = next(pc for pc in loop if k.instructions[pc].opcode == "v.ucmp.ne.f32" and
                  "0x3f800000" in k.instructions[pc].operands)
        k.instructions[pc] = replace(k.instructions[pc], operands=k.instructions[pc].operands.replace(
            "0x3f800000", "0x3f7fffff"))
    guards = identity_guards(new[TARGET], loop)
    if plant == "no-bypass":
        current = prior
    old_mul, new_mul = prior["groups"]["fp32_mul"], current["groups"]["fp32_mul"]
    if (old_mul["cfg_path_min"], old_mul["cfg_path_max"]) != (395, 395):
        raise ValueError("wrong immutable K-permutation baseline")
    if (new_mul["cfg_path_min"], new_mul["cfg_path_max"]) != (267, 395):
        raise ValueError("must bypass exactly128 O multiplies and retain full update")
    for name in ("s32_to_f32", "f32_to_s32", "fp32_add", "fp32_fma", "shuffle", "qk_mma", "pv_mma"):
        for field in ("cfg_path_min", "cfg_path_max"):
            if prior["groups"][name][field] != current["groups"][name][field]:
                raise ValueError("another arithmetic operation/path changed: " + name)
    for key in old:
        if selected(key):
            a = Counter(r.opcode for r in old[key].instructions.values())
            b = Counter(r.opcode for r in new[key].instructions.values())
            for opcode in ("v.mma.i32.i8.i8.m16n16k32", "v.mma.i32.u8.i8.m16n16k32",
                           "v.cnvt.i32.f32.rtte", "v.pcnvt.b16.u8x2", "s.blksyn.defer",
                           "vmem.acp.commit.grp"):
                if a[opcode] != b[opcode]:
                    raise ValueError("MMA/P quantization/barrier cadence changed")
    rs = resources[TARGET]
    accepted_resource = rs["vector_registers"] <= 244
    return dict(scope="NATIVE_CFG_ONLY; NO_DEVICE_NUMERICS_OR_SPEED_VERDICT",
                immutable_source="711ec99", unchanged_native=unchanged, changed_native=4,
                h3_resources=rs, required_native_denorm_flush=0,
                before=prior, after=current, generated_factor_to_multiply_proofs=guards,
                path_delta={f: current["groups"]["all"][f] - prior["groups"]["all"][f]
                            for f in ("whole_static", "cfg_path_min", "cfg_path_max")},
                decision="LOCAL_CANDIDATE_ONLY" if accepted_resource else "REJECT_REGISTER_REGRESSION",
                resource_cap=244)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("before", "after", "resources"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--plant", choices=("legacy-changed", "spill", "ftz", "no-bypass", "wrong-unit"))
    args = p.parse_args()
    try:
        result = compare(args.before.read_text(), args.after.read_text(), args.resources.read_text(), args.plant)
        result["input_sha256"] = {f: hashlib.sha256(getattr(args, f).read_bytes()).hexdigest()
                                  for f in ("before", "after", "resources")}
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("[identity rescale native]", result["decision"],
              "unchanged68/72; H3", result["h3_resources"], "delta", result["path_delta"])
        return 0
    except (ValueError, OSError) as error:
        print("[identity rescale native] FAIL:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
