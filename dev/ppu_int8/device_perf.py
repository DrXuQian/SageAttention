#!/usr/bin/env python3
"""PPU SageAttention core and quantized end-to-end performance admission."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics

import torch

from sageattention import _qattn_ppu


def measure_us(fn, warmup: int, samples: int, launches: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    result: list[float] = []
    for _ in range(samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(launches):
            fn()
        end.record()
        torch.cuda.synchronize()
        result.append(begin.elapsed_time(end) * 1000.0 / launches)
    return result


def fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def print_result(
    role: str,
    samples: list[float],
    flops: int,
    peak_tflops: float,
    fingerprint_value: str,
) -> None:
    median = statistics.median(samples)
    tflops = flops / median / 1.0e6
    mfu = 100.0 * tflops / peak_tflops
    print(
        f"[PPU Sage perf] role={role} median_us={median:.3f} "
        f"range=[{min(samples):.3f},{max(samples):.3f}] "
        f"logical_tflops={tflops:.3f} logical_mfu={mfu:.2f}% "
        f"peak_denominator={peak_tflops:.3f}_TFLOPS "
        f"output_sha256={fingerprint_value}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--peak-tflops", type=float, default=500.0)
    parser.add_argument(
        "--core-only", action="store_true",
        help="omit the separate quant+core timing",
    )
    args = parser.parse_args()
    if args.heads % args.kv_heads:
        parser.error("--heads must be divisible by --kv-heads")
    if args.seq <= 0 or args.batch <= 0 or args.heads <= 0:
        parser.error("batch/head/sequence extents must be positive")

    torch.manual_seed(0x5A6E)
    torch.cuda.set_device(0)
    shape_q = (args.batch, args.heads, args.seq, args.head_dim)
    shape_kv = (args.batch, args.kv_heads, args.seq, args.head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=torch.float16)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.float16)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.float16)
    qi = torch.empty(shape_q, device="cuda", dtype=torch.int8)
    ki = torch.empty(shape_kv, device="cuda", dtype=torch.int8)
    qs = torch.empty(
        (args.batch, args.heads, math.ceil(args.seq / 128) * 4),
        device="cuda", dtype=torch.float32,
    )
    ks = torch.empty(
        (args.batch, args.kv_heads, math.ceil(args.seq / 64)),
        device="cuda", dtype=torch.float32,
    )
    output = torch.empty(shape_q, device="cuda", dtype=torch.float16)
    no_mean = torch.empty(0, device="cuda", dtype=torch.float16)

    def quantize() -> None:
        _qattn_ppu.quant_per_warp_int8(q, qi, qs, 128, 32, 1)
        _qattn_ppu.quant_per_block_int8(k, no_mean, ki, ks, 64, 1)

    def core() -> None:
        _qattn_ppu.qk_int8_sv_f16_accum_f32_attn(
            qi, ki, v, output, qs, ks, 1, int(args.causal), 2,
            args.head_dim ** -0.5, 0,
        )

    quantize()
    core()
    torch.cuda.synchronize()
    first = output.clone()
    core()
    torch.cuda.synchronize()
    if not torch.equal(first, output):
        raise AssertionError("core output is not raw-bit stable")
    if not torch.isfinite(output).all().item():
        raise AssertionError("core output contains non-finite values")
    output_hash = fingerprint(output)

    pairs = args.seq * args.seq
    if args.causal:
        pairs = args.seq * (args.seq + 1) // 2
    flops = 4 * args.batch * args.heads * args.head_dim * pairs
    print(
        f"[PPU Sage perf config] B={args.batch} H={args.heads} "
        f"Hkv={args.kv_heads} N={args.seq} D={args.head_dim} "
        f"causal={int(args.causal)} grid="
        f"({math.ceil(args.seq / 128)},{args.heads},{args.batch}) "
        f"logical_flops={flops} warmup={args.warmup} samples={args.samples} "
        f"launches_per_sample={args.launches}"
    )
    core_samples = measure_us(core, args.warmup, args.samples, args.launches)
    print_result("prequantized-core", core_samples, flops, args.peak_tflops, output_hash)

    if not args.core_only:
        def quant_and_core() -> None:
            quantize()
            core()

        e2e_samples = measure_us(
            quant_and_core, args.warmup, args.samples, args.launches
        )
        print_result(
            "quant-kernels+core", e2e_samples, flops,
            args.peak_tflops, output_hash,
        )

    if not args.causal and args.batch == 1 and args.heads == 16 \
            and args.kv_heads == 16 and args.seq == 4096 \
            and args.head_dim == 128:
        core_median = statistics.median(core_samples)
        core_tflops = flops / core_median / 1.0e6
        verdict = "ADMITTED" if core_tflops >= 294.0 else "BELOW-ANCHOR"
        print(
            f"[PPU Sage perf anchor] current={core_tflops:.3f}_TFLOPS "
            f"device_proven_fp16_fattn=294.000_TFLOPS verdict={verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
