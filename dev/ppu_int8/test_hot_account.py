#!/usr/bin/env python3
"""Evidence-dependent negatives for the hot-loop account (no device run)."""
import argparse
import copy
import json
from pathlib import Path

from account_hot_loop import anchor_acu, loop_inventory, select, reference_account


def expect_red(label, action):
    try:
        action()
    except ValueError as error:
        print(f"[hot negative] {label}: EXPECTED-RED/PASS ({error})")
        return
    raise AssertionError(f"{label}: plant went green")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--acu", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    text = args.before.read_text()
    kernel = select(text)
    _, loop = loop_inventory(kernel)
    raw = json.loads(args.acu.read_text())
    anchor_acu(kernel, loop, raw)
    damaged = copy.deepcopy(raw)
    damaged[0]["instructions"].pop(0)
    expect_red("missing-PC", lambda: anchor_acu(kernel, loop, damaged))
    damaged_operand = copy.deepcopy(raw)
    damaged_operand[0]["instructions"][0]["sass"] += ",WRONG_REGISTER"
    expect_red("same-opcode-wrong-operand", lambda: anchor_acu(kernel, loop, damaged_operand))
    wrong_count = copy.deepcopy(raw)
    next(r for r in wrong_count[0]["instructions"] if r["sass"].startswith("v.mma."))["executed"] -= 1
    expect_red("missing-measured-MMA", lambda: anchor_acu(kernel, loop, wrong_count))
    damaged_kernel = copy.deepcopy(kernel)
    pc = next(n for n, r in kernel.instructions.items() if r.opcode.startswith("v.mma."))
    row = damaged_kernel.instructions[pc]
    damaged_kernel.instructions[pc] = type(row)(row.pc, "v.mov.b32", row.operands)
    expect_red("missing-generated-MMA", lambda: loop_inventory(damaged_kernel))
    nested = copy.deepcopy(kernel)
    nested.edges[pc].add(pc)
    expect_red("unmodelled-loop", lambda: loop_inventory(nested))
    expect_red("wrong-specialization", lambda: select(text.replace("ELb1E", "ELb0E")))
    reference = json.loads(args.reference.read_text())
    sha = reference["results"]["sass.json"]["identity"]["source_sha256"]
    reference_account(reference, sha)
    reference["warp_k64_visits"] -= 1
    expect_red("wrong-reference-denominator", lambda: reference_account(reference, sha))
    print("[hot negatives] PASS: seven independent evidence/CFG mutations rejected")


if __name__ == "__main__":
    main()
