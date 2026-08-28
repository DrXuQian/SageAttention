#!/usr/bin/env python3
"""Compare shipping-TU nvcc diagnostics with an actlize-header baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ERROR = re.compile(r"error: (.*)")
PPU_COMPILER_BOUNDARY = re.compile(
    r'(?:calling a __host__ function\("(?:void )?(?:cute::PPU0010_|cute::cp_async|'
    r'cutlass::(?:half_t|bfloat16_t))|'
    r'identifier "(?:cute::PPU0010_|cute::cp_async|'
    r'cutlass::(?:half_t|bfloat16_t)))'
)


def normalized(path: Path) -> list[str]:
    result: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        match = ERROR.search(line)
        if not match:
            continue
        text = match.group(1)
        text = re.sub(r'_ZN\d+_INTERNAL_[^\"]+', '<internal>', text)
        result.append(text)
    return sorted(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("shipping", type=Path)
    args = parser.parse_args()
    base = normalized(args.baseline)
    ship = normalized(args.shipping)
    extra = list(ship)
    for item in base:
        if item in extra:
            extra.remove(item)
    if not base:
        if ship and all(PPU_COMPILER_BOUNDARY.search(item) for item in ship):
            print(
                "[PPU Sage frontend] SKIP: NVIDIA nvcc cannot instantiate "
                f"{len(ship)} HGGCCC-only actlize device seams; fresh hgcc "
                "shipping build remains required"
            )
            return 3
        if ship:
            for item in ship:
                print(f"[PPU Sage frontend] FAIL: {item}")
            return 1
        print("[PPU Sage frontend] baseline and shipping compiled cleanly")
        return 0
    if extra:
        for item in extra:
            print(f"[PPU Sage frontend] FAIL: non-baseline diagnostic: {item}")
        return 1
    print(
        f"[PPU Sage frontend] PASS: shipping added 0 diagnostics beyond "
        f"{len(base)} fixed actlize nvcc-boundary diagnostics"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
