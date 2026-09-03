#!/usr/bin/env python3
"""Check the generated PPU accumulator-zero contract of the PV bridge."""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path


def zeroed_destination_registers(section: str) -> tuple[int, int]:
    lines = section.splitlines()
    mma_index = next(
        (i for i, line in enumerate(lines) if "v.mma.f16.f16.m16n16k16" in line),
        None,
    )
    if mma_index is None:
        raise ValueError("missing fp16 bridge MMA")
    match = re.search(r"vreg\[(\d+):(\d+)\]", lines[mma_index])
    if match is None:
        raise ValueError("bridge MMA destination range is not visible")
    first, last = map(int, match.groups())
    expected = set(range(first, last + 1))
    zeroed: set[int] = set()
    for line in lines[max(0, mma_index - 20):mma_index]:
        move = re.search(r"v\.mov\.b32\s+vreg(\d+), 0x0", line)
        if move and int(move.group(1)) in expected:
            zeroed.add(int(move.group(1)))
    return len(zeroed), len(expected)


def written_vector_registers(line: str) -> set[int]:
    """Return the first-operand vregs written by one hgobjdump line."""
    if "\t" not in line:
        return set()
    fields = line.split("\t")
    if len(fields) < 3:
        return set()
    opcode, operands = fields[-2], fields[-1]
    if ".st" in opcode or opcode.startswith("s."):
        return set()
    first = operands.split(",", 1)[0].strip()
    vector = re.fullmatch(r"vreg\[(\d+):(\d+)\]", first)
    if vector:
        begin, end = map(int, vector.groups())
        return set(range(begin, end + 1))
    scalar = re.fullmatch(r"vreg(\d+)", first)
    return {int(scalar.group(1))} if scalar else set()


def check_shipping(text: str) -> int:
    sections = re.split(r"(?=Disassembly of section \.text\.kernel\.)", text)
    bridge_count = 0
    failures: list[str] = []
    for section in sections:
        lines = section.splitlines()
        for mma_index, line in enumerate(lines):
            if "v.mma.f16.f16.m16n16k16" not in line:
                continue
            bridge_count += 1
            match = re.search(r"vreg\[(\d+):(\d+)\]", line)
            if match is None:
                failures.append(f"bridge {bridge_count}: destination range missing")
                continue
            begin, end = map(int, match.groups())
            pending = set(range(begin, end + 1))
            nonzero: dict[int, str] = {}
            for previous in reversed(lines[:mma_index]):
                written = written_vector_registers(previous) & pending
                for register in written:
                    if not re.search(
                        rf"v\.mov\.b32\s+vreg{register}, 0x0", previous
                    ):
                        nonzero[register] = previous.strip()
                pending -= written
                if not pending:
                    break
            if pending or nonzero:
                failures.append(
                    f"bridge {bridge_count}: destination={begin}:{end} "
                    f"no-writer={sorted(pending)} nonzero-writer={nonzero}"
                )
    if bridge_count == 0:
        failures.append("shipping ISA contains no fp16 bridge MMA")
    if failures:
        for failure in failures:
            print(f"[PPU Sage shipping bridge ISA] FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"[PPU Sage shipping bridge ISA] PASS: bridges={bridge_count} "
        "accumulator_zeroed=4/4-before-each"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("isa", type=Path)
    parser.add_argument("--shipping", action="store_true")
    args = parser.parse_args()
    text = args.isa.read_text()
    if args.shipping:
        return check_shipping(text)
    sections = re.split(r"(?=Disassembly of section \.text\.kernel\.)", text)
    expected = {"atom": 4, "raw": 4, "legacy": 1}
    observed: dict[str, tuple[int, int]] = {}
    for role in expected:
        section = next(
            (part for part in sections if f"bridge_{role}_codegen" in part), None
        )
        if section is None:
            print(f"[PPU Sage bridge ISA] FAIL: missing {role} kernel", file=sys.stderr)
            return 1
        try:
            observed[role] = zeroed_destination_registers(section)
        except ValueError as error:
            print(f"[PPU Sage bridge ISA] FAIL: {role}: {error}", file=sys.stderr)
            return 1

    for role, (zeroed, width) in observed.items():
        print(
            f"[PPU Sage bridge ISA] role={role} accumulator_zeroed="
            f"{zeroed}/{width}"
        )
        if width != 4 or zeroed != expected[role]:
            print(
                f"[PPU Sage bridge ISA] FAIL: {role} expected "
                f"{expected[role]}/4 zeroed destination registers",
                file=sys.stderr,
            )
            return 1
    print(
        "[PPU Sage bridge ISA] PASS: actlize=4/4 raw=4/4; "
        "legacy-contiguous-reference=1/4 EXPECTED-RED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
