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
    parser.add_argument("--candidate", type=Path, help="Check both explicit K layouts in a generated72-type binary")
    args = parser.parse_args()
    text = args.before.read_text()
    kernel = select(text)
    _, loop = loop_inventory(kernel)
    raw = json.loads(args.acu.read_text())
    anchor_acu(kernel, loop, raw)
    mislabeled = copy.deepcopy(raw)
    mislabeled[0]["name"] = mislabeled[0]["name"].replace("bfloat16_t, true>", "bfloat16_t, true, true>")
    if mislabeled[0]["name"] == raw[0]["name"]:
        raise ValueError("could not plant a K-layout swap in the old report")
    expect_red("report-key-layout-swap", lambda: anchor_acu(kernel, loop, mislabeled))
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
    if args.candidate:
        candidate = args.candidate.read_text()
        raw_k, permuted_k = select(candidate, "raw"), select(candidate, "permuted")
        if raw_k.name == permuted_k.name:
            raise AssertionError("two K layouts collapsed into one native symbol")
        _, new_loop = loop_inventory(permuted_k)
        expect_red("old-report-on-new-key-layout", lambda: anchor_acu(permuted_k, new_loop, raw))
        _, raw_loop = loop_inventory(raw_k)
        expect_red("old-report-same-layout-different-body", lambda: anchor_acu(raw_k, raw_loop, raw))
        expect_red("absent-permuted-specialization", lambda: select(text, "permuted"))
        print("[hot selection] both explicit H3 key layouts select distinct native bodies; PASS")
    print("[hot negatives] PASS: eight core evidence/CFG mutations rejected; optional candidate tests separately printed")


if __name__ == "__main__":
    main()
