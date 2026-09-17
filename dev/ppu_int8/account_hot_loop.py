#!/usr/bin/env python3
"""Separate emitted hot-loop paths from cold code; anchor old ISA to ACU PCs.

Candidate path inventories and conservative upper bounds are NOT measured
execution counts. Masked then-only regions may execute on fewer warps, so the
shortest CFG path is deliberately not called a device lower bound.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_requant_codegen import integer_alu
from compare_native_sage_opcodes import require_mma
from native_isa import parse_native


TARGET = "_ZN13sageattention3ppu6detail17qk_int8_pv_kernelILi128ELb0ELb0EN7cutlass10bfloat16_tELb1E"
WARPS = 577 * 56 * 4
ITERATIONS = 1153
VISITS = WARPS * ITERATIONS
GROUPS = {
    "all": lambda op: True,
    "integer_bit_select": integer_alu,
    "shuffle": lambda op: op.startswith("v.shuffle."),
    "s32_to_f32": lambda op: op == "v.cnvt.f32.i32.rtte",
    "f32_to_s32": lambda op: op == "v.cnvt.i32.f32.rtte",
    "fp32_mul": lambda op: op == "v.mul.f32",
    "fp32_add": lambda op: op == "v.add.f32",
    "fp32_fma": lambda op: op == "v.fma.f32.rtte",
    "qk_mma": lambda op: op == "v.mma.i32.i8.i8.m16n16k32",
    "pv_mma": lambda op: op == "v.mma.i32.u8.i8.m16n16k32",
}


def select(text):
    rows = [k for name, k in parse_native(text).items() if name.startswith(TARGET)]
    if len(rows) != 1:
        raise ValueError("expected one exact H3 integer-PV specialization")
    return rows[0]


def reachable(edges, root):
    seen, pending = set(), [root]
    while pending:
        n = pending.pop()
        if n not in seen:
            seen.add(n)
            pending.extend(edges[n])
    return seen


def parents(edges):
    result = {n: set() for n in edges}
    for n, targets in edges.items():
        for target in targets:
            result[target].add(n)
    return result


def topological(edges):
    incoming = parents(edges)
    degrees = {n: len(p) for n, p in incoming.items()}
    pending = sorted(n for n, degree in degrees.items() if not degree)
    order = []
    while pending:
        n = pending.pop()
        order.append(n)
        for target in sorted(edges[n]):
            degrees[target] -= 1
            if not degrees[target]:
                pending.append(target)
    if len(order) != len(edges):
        raise ValueError("unmodelled nested cycle")
    return order, incoming


def loop_inventory(kernel):
    mma = [pc for pc, r in kernel.instructions.items() if r.opcode.startswith("v.mma.")]
    if not mma:
        raise ValueError("missing MMA work")
    reverse = parents(kernel.edges)
    loop = reachable(kernel.edges, mma[0]) & reachable(reverse, mma[0])
    if not set(mma) <= loop:
        raise ValueError("MMA work outside the single modelled loop")
    headers = [n for n in loop if reverse[n] - loop]
    if len(headers) != 1:
        raise ValueError("expected single-entry K loop")
    header = headers[0]
    dag = {n: (kernel.edges[n] & loop) - {header} for n in loop}
    order, incoming = topological(dag)
    ends = {n for n in loop if not dag[n] or kernel.edges[n] - loop or header in kernel.edges[n]}
    cold = kernel.instructions.keys() - loop
    topological({n: kernel.edges[n] & cold for n in cold})

    # No divergent if/else may invalidate a single-walk upper bound. For each
    # emsk branch, prove that the skipped body rejoins its branch target on
    # every path. It is then a then-only region, not two serialized arms.
    mask_branches = 0
    pcs = list(kernel.instructions)
    following = dict(zip(pcs, pcs[1:]))
    for n in sorted(loop):
        row = kernel.instructions[n]
        if not row.opcode.startswith("s.cbr") or not row.operands.startswith("emsk,"):
            continue
        if row.opcode != "s.cbr.az" or len(kernel.edges[n]) != 2:
            raise ValueError("unproved divergent branch form")
        fallthrough = following[n]
        target, = kernel.edges[n] - {fallthrough}
        if target not in loop:
            raise ValueError("unproved divergent loop exit")
        pending, visited = [fallthrough], set()
        while pending:
            node = pending.pop()
            if node == target or node in visited:
                continue
            if node in ends:
                raise ValueError("divergent arms do not immediately rejoin")
            visited.add(node)
            pending.extend(dag[node])
        mask_branches += 1

    groups = {}
    for name, predicate in GROUPS.items():
        weights = {n: int(predicate(r.opcode)) for n, r in kernel.instructions.items()}
        lo, hi = {}, {}
        for n in order:
            lo[n] = weights[n] + min((lo[p] for p in incoming[n]), default=0)
            hi[n] = weights[n] + max((hi[p] for p in incoming[n]), default=0)
        cold_count = sum(weights[n] for n in cold)
        groups[name] = dict(whole_static=sum(weights.values()),
                            loop_union_static=sum(weights[n] for n in loop),
                            once_only_union_static=cold_count,
                            cfg_path_min=min(lo[n] for n in ends),
                            cfg_path_max=max(hi[n] for n in ends),
                            conservative_max_per_visit=max(hi[n] for n in ends) + cold_count / ITERATIONS)
    for name in ("qk_mma", "pv_mma"):
        if groups[name]["cfg_path_min"] != 32 or groups[name]["cfg_path_max"] != 32:
            raise ValueError(f"{name}: K64 path does not contain exactly 32 MMAs")
    return dict(kernel=kernel.name, loop_header=hex(header), mask_then_only_proofs=mask_branches,
                reachable_instructions=len(kernel.instructions), discarded=kernel.discarded_instructions,
                groups=groups,
                loop_opcode_union=dict(sorted(Counter(kernel.instructions[n].opcode for n in loop).items()))), loop


def anchor_acu(kernel, loop, raw):
    if len(raw) != 1 or "qk_int8_pv_kernel<128, false, false, cutlass::bfloat16_t, true>" not in raw[0]["name"]:
        raise ValueError("ACU target is not the exact H3 specialization")
    instructions = raw[0]["instructions"]
    base = min(r["pc"] for r in instructions)
    rows = {r["pc"] - base: r for r in instructions}
    if len(rows) != len(instructions) or any(r["executed"] < 0 for r in instructions):
        raise ValueError("duplicate PC or negative ACU execution count")
    normalize = lambda s: re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", s))
    for pc, row in kernel.instructions.items():
        if pc not in rows or normalize(row.opcode + " " + row.operands) != normalize(rows[pc]["sass"]):
            raise ValueError(f"ACU/native instruction or operand mismatch at {pc:x}")
    if any(row["executed"] for pc, row in rows.items() if pc not in kernel.instructions):
        raise ValueError("ACU executes a PC outside the reachable native body")
    groups = {}
    for name, predicate in GROUPS.items():
        total = sum(rows[pc]["executed"] for pc, r in kernel.instructions.items() if predicate(r.opcode))
        hot = sum(rows[pc]["executed"] for pc in loop if predicate(kernel.instructions[pc].opcode))
        groups[name] = dict(measured=total, measured_hot=hot, measured_once_only=total-hot,
                            measured_per_visit=total/VISITS)
    if groups["all"]["measured"] != sum(r["executed"] for r in instructions):
        raise ValueError("ACU PC denominator did not close")
    for name in ("qk_mma", "pv_mma"):
        if groups[name]["measured"] != 32 * VISITS:
            raise ValueError("ACU MMA does not match registered work")
    return dict(instruction_and_operand_matches=len(kernel.instructions),
                nonexecuted_external_pcs=len(rows)-len(kernel.instructions), groups=groups)


def reference_account(reference, acu_sha):
    if reference["warp_k64_visits"] != VISITS:
        raise ValueError("reference visit denominator changed")
    rows = reference["results"]
    expected = {"sass.json", "h3-native-fp8-fp16-buffer-raw.csv", "h3-native-fp8-fp32-raw.csv"}
    if set(rows) != expected or rows["sass.json"]["identity"]["source_sha256"] != acu_sha:
        raise ValueError("reference identity/census differs")
    result = {}
    for name, row in rows.items():
        c = row["opcodes"]
        native = name != "sass.json"
        require_mma(c, "nvidia" if native else "ppu", VISITS)
        if native:
            shuffles = sum(n for op, n in c.items() if op.startswith("SHFL."))
            if shuffles != 8 * VISITS + 8 * WARPS:
                raise ValueError("native denominator-reduction shuffle count did not close")
            conversions = c["I2FP.F32.S32.RZ"]
            if conversions != 64 * VISITS:
                raise ValueError("native shared QK conversion count differs")
        else:
            conversions = c["v.cnvt.f32.i32.rtte"]
            if conversions != 192 * VISITS:
                raise ValueError("PPU QK+PV conversion count differs")
        chosen = {op: n for op, n in c.items() if op in ("v.mul.f32", "v.add.f32", "v.fma.f32.rtte") or
                  op.startswith(("FMUL.", "UFMUL.", "FADD.", "FFMA.", "HADD2."))}
        result[name] = dict(s32_to_f32_per_visit=conversions/VISITS,
                            half_restore_add_per_visit=c.get("HADD2.F32", 0)/VISITS,
                            mul_add_fma_half_restore_per_visit=sum(chosen.values())/VISITS,
                            selected_arithmetic_opcodes=chosen)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--acu", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        old, new = select(args.before.read_text()), select(args.after.read_text())
        before, loop = loop_inventory(old)
        after, _ = loop_inventory(new)
        anchor = anchor_acu(old, loop, json.loads(args.acu.read_text()))
        acu_sha = hashlib.sha256(args.acu.read_bytes()).hexdigest()
        reference = reference_account(json.loads(args.reference.read_text()), acu_sha)
        result = dict(scope="STATIC_CFG_BOUND_FOR_CANDIDATE; OLD_ACU_ONLY; NO_NEW_DEVICE_RESULT",
                      warps=WARPS, k64_iterations=ITERATIONS, visits=VISITS,
                      input_sha256={key: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for key, path in (("before", args.before), ("after", args.after),
                                                      ("acu", args.acu), ("reference", args.reference))},
                      integer_category_note="fixed static scope also includes s.not/s.mull/s.mulh; old reference excludes these once-only ops (0.011275 per visit)",
                      before=before, after=after, old_acu_anchor=anchor, measured_reference=reference)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("[hot account] old PC/operand anchor", anchor["instruction_and_operand_matches"], "PASS")
        for name, counts in after["groups"].items():
            print("[hot account] candidate", name, json.dumps(counts))
        print("[hot account] PASS: bounds are not a new dynamic count or latency")
        return 0
    except ValueError as error:
        print("[hot account] FAIL:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
