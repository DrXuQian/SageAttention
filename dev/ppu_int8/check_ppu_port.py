#!/usr/bin/env python3
"""Fail-closed source/algorithm contract for the actlize PPU port."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def scheduler_bad(batches: int, heads: int, qtiles: int, causal: bool) -> int:
    seen: set[tuple[int, int, int]] = set()
    bad = 0
    for bz in range(batches):
        for by in range(heads):
            for bx in range(qtiles):
                if causal:
                    linear = ((bz * heads + by) * qtiles) + bx
                    z = linear % (heads * batches)
                    batch = z // heads
                    head = z % heads
                    qtile = qtiles - 1 - linear // (heads * batches)
                else:
                    batch, head, qtile = bz, by, qtiles - 1 - bx
                item = (batch, head, qtile)
                bad += item in seen
                seen.add(item)
    return bad + (batches * heads * qtiles - len(seen))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plant",
        choices=("bridge-order", "v-write", "row-formula", "scheduler"),
    )
    args = parser.parse_args()

    ops = (ROOT / "csrc/qattn/ppu/attn_ppu_ops.cuh").read_text()
    kernel = (ROOT / "csrc/qattn/ppu/qk_int_sv_f16_ppu.cu").read_text()
    layout = (ROOT / "csrc/qattn/ppu/attn_ppu_layout.cuh").read_text()
    setup = (ROOT / "setup_ppu.py").read_text()
    mma_arch = (
        ROOT / "csrc/actlize/include/cute/arch/mma_ppu0010.hpp"
    ).read_text()
    if args.plant == "bridge-order":
        ops = ops.replace(
            "dst[0], dst[1], dst[2], dst[3],\n"
            "      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],",
            "dst[0], dst[1], dst[2], dst[3],\n"
            "      b[0], b[1], b[2], b[3], a[0], a[1], a[2], a[3],",
        )
    elif args.plant == "v-write":
        kernel = kernel.replace(
            "aiu_load_swizzled_64<cutlass::half_t, kBlockKV>",
            "aiu_load_swizzled_transposed_64<cutlass::half_t, kBlockKV>",
        )
    elif args.plant == "row-formula":
        layout = layout.replace(
            "return lane / 4 + 8 * (value >> 2);",
            "return (lane & 3) + 4 * (value & 3);",
        )

    failures: list[str] = []
    ppu_sources = list((ROOT / "csrc/qattn/ppu").glob("*"))
    ppu_text = "\n".join(
        p.read_text(errors="replace") for p in ppu_sources if p.is_file()
    )
    require(not re.search(r"\basm\s+volatile\b", ppu_text),
            "private inline assembly entered the Sage PPU source graph", failures)
    require("PPU0010_16x16x32_S32S8S8S32_TN" in ops,
            "actlize INT8 MMA atom missing", failures)
    require("PPU0010_16x16x16_F32F16F16F32_TN" in ops,
            "actlize FP16/FP32 MMA atom missing", failures)
    require(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16" in mma_arch,
        "actlize f16-output bridge lacks the device-anchored PPU0010 opcode",
        failures,
    )
    require("PPU0010_AIU_LOAD" in ops and "PPU0010_TSM_LD_SWZL" in ops,
            "actlize AIU write/read atoms missing", failures)
    require("Element, false, true" in ops,
            "non-transposed AIU write atom is not modeled", failures)
    require("false, true, Cubes" in ops,
            "transposed V TSM read is not modeled", failures)
    require("aiu_load_swizzled_64<cutlass::half_t, kBlockKV>" in kernel and
            "aiu_load_swizzled_transposed_64<cutlass::half_t, kBlockKV>" not in kernel,
            "V departed from the device-anchored nontrans-write/trans-read path",
            failures)
    require("(step & 1) * 32, step >> 1" in kernel,
            "Q/K head_dim=128 is not expressed as proven 64-wide slices", failures)
    require("layout::accumulator_row(lane, e)" in kernel and
            "layout::accumulator_column(lane, e)" in kernel,
            "kernel bypasses the trait-checked accumulator map", failures)
    require("return lane / 4 + 8 * (value >> 2);" in layout and
            "return (lane & 3) + 4 * (value & 3);" in layout,
            "PPU CLayout row formula drifted", failures)
    require(
        "dst[0], dst[1], dst[2], dst[3],\n"
        "      a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]," in ops,
        "C-to-A bridge operands are reversed",
        failures,
    )

    shipping_sources = {
        "csrc/qattn/ppu/pybind_ppu.cpp",
        "csrc/qattn/ppu/qk_int_sv_f16_ppu.cu",
        "csrc/qattn/ppu/quant_ppu.cu",
    }
    for source in shipping_sources:
        require(f'"{source}"' in setup, f"shipping source missing: {source}", failures)
    require("qk_int_sv_f16_cuda_sm80.cu" not in setup and "sm90" not in setup,
            "NVIDIA source entered the PPU build graph", failures)

    total_scheduler = 0
    for batches in range(1, 4):
        for heads in range(1, 18):
            for qtiles in range(1, 18):
                for causal in (False, True):
                    if args.plant == "scheduler" and causal and qtiles > 1:
                        total_scheduler += 1
                    else:
                        total_scheduler += scheduler_bad(
                            batches, heads, qtiles, causal
                        )
    require(total_scheduler == 0,
            f"scheduler bijection failed: bad={total_scheduler}", failures)

    if failures:
        for failure in failures:
            print(f"[PPU Sage source] FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "[PPU Sage source] PASS: actlize-only graph; device-anchored V path; "
        "trait-bound maps; scheduler exhaustive"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
