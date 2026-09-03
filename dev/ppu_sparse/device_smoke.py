#!/usr/bin/env python3
"""PPU admission for full-CSR, H3 summary and Sol summary paths."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace

import torch

from sageattention import ppu_compile
from sageattention.core import sageattn_qk_int8_pv_fp16_ppu
from sageattention.ppu_sparse import (
    make_h3_topk_plan,
    make_sol_plan,
    quantized_sparse_attention_reference,
    sageattn_block_sparse_ppu,
)


def fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().view(torch.uint8).numpy()).hexdigest()[:16]


def operands(q, k, v, layout):
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(
        q, k, None, tensor_layout=layout
    )
    return qi, qs, ki, ks, q.to(torch.float16).contiguous(), v.to(torch.float16).contiguous()


def quantized_error(q, k, v, plan, actual, layout):
    qi, qs, ki, ks, qf, vf = operands(q, k, v, layout)
    expected = quantized_sparse_attention_reference(
        qi, qs, ki, ks, qf, vf, plan, tensor_layout=layout
    )
    return (actual.float() - expected.float()).abs().max().item()


def run_full_density(layout: str, query_block: int) -> None:
    torch.manual_seed(1000 + query_block)
    if layout == "NHD":
        shape = (1, 256, 2, 128)
    else:
        shape = (1, 2, 256, 128)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.25
        for _ in range(3)
    )
    dense = sageattn_qk_int8_pv_fp16_ppu(
        q, k, v, tensor_layout=layout, smooth_k=False
    )
    if query_block == 128:
        plan = make_h3_topk_plan(
            q, k, v, top_k=2, tensor_layout=layout, taylor=False
        )
    else:
        plan = make_sol_plan(
            q,
            k,
            v,
            tensor_layout=layout,
            sink_query_blocks=(0, 4),
        )
        plan = replace(plan, use_summary=False, algorithm="sol-full-density")
        plan.validate(deep=True)
    sparse = sageattn_block_sparse_ppu(q, k, v, plan, tensor_layout=layout)
    raw_bad = int((dense.view(torch.int16) != sparse.view(torch.int16)).sum().item())
    print(
        f"[PPU sparse device] role=full-density Q={query_block} layout={layout} "
        f"raw_bad={raw_bad}/{dense.numel()} fingerprint={fingerprint(sparse)}"
    )
    if raw_bad:
        raise AssertionError("full-density sparse path is not raw-bit dense-equivalent")


def run_h3_summary() -> None:
    torch.manual_seed(2128)
    q = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn((1, 257, 1, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k) * 0.2
    plan = make_h3_topk_plan(q, k, v, top_k=1, tensor_layout="NHD", taylor=True)
    actual = sageattn_block_sparse_ppu(q, k, v, plan, tensor_layout="NHD")
    error = quantized_error(q, k, v, plan, actual, "NHD")
    first = actual.clone()
    for _ in range(3):
        replay = sageattn_block_sparse_ppu(q, k, v, plan, tensor_layout="NHD")
        torch.cuda.synchronize()
        if not torch.equal(first, replay):
            raise AssertionError("H3 sparse replay fingerprint is unstable")
    print(
        f"[PPU sparse device] role=h3-topk-summary shape=B1,N257,Hq2,Hkv1,D128 "
        f"top_k=1 exact_kv64={plan.exact_kv64.numel()} max_quant_oracle={error:.8f} "
        f"fingerprint={fingerprint(actual)} replay=RAW-BIT/STABLE"
    )
    if error > 0.01:
        raise AssertionError("H3 summary path disagrees with its quantized oracle")

    zero_summary = replace(
        plan,
        value_mean=torch.zeros_like(plan.value_mean),
        algorithm="h3-summary-value-zero-plant",
    )
    zero_summary.validate(deep=True)
    planted = sageattn_block_sparse_ppu(
        q, k, v, zero_summary, tensor_layout="NHD"
    )
    changed = int((planted.view(torch.int16) != actual.view(torch.int16)).sum().item())
    print(
        f"[PPU sparse negative] plant=summary-value-zero bitdiff={changed}/{actual.numel()} "
        "EXPECTED-RED/PASS"
    )
    if changed == 0:
        raise AssertionError("summary-value plant did not change device output")


def run_sol_summary() -> None:
    torch.manual_seed(3064)
    q = torch.randn((1, 2, 321, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn_like(q) * 0.2
    plan = make_sol_plan(
        q,
        k,
        v,
        tau=1.0,
        thresh_type="diag",
        tensor_layout="HND",
        sink_blocks=(0, 1),
    )
    actual = sageattn_block_sparse_ppu(q, k, v, plan, tensor_layout="HND")
    error = quantized_error(q, k, v, plan, actual, "HND")
    selected = plan.exact_kv64.numel()
    full = plan.rows * plan.kv_blocks
    print(
        f"[PPU sparse device] role=sol-diag-summary shape=B1,N321,H2,D128 "
        f"exact_kv64={selected}/{full} max_quant_oracle={error:.8f} "
        f"fingerprint={fingerprint(actual)}"
    )
    if error > 0.01:
        raise AssertionError("Sol summary path disagrees with its quantized oracle")


def run_negative() -> None:
    q = torch.zeros((1, 128, 1, 128), device="cuda", dtype=torch.bfloat16)
    plan = make_h3_topk_plan(q, q, q, top_k=1, tensor_layout="NHD")
    bad = replace(plan, exact_kv64=torch.full_like(plan.exact_kv64, 99))
    try:
        sageattn_block_sparse_ppu(q, q, q, bad, tensor_layout="NHD")
    except ValueError as error:
        if "not produced by an admitted planner" not in str(error):
            raise
        print("[PPU sparse negative] plant=unadmitted-mutated-plan EXPECTED-RED/PASS")
        return
    raise AssertionError("mutated sparse plan was not rejected")


def main() -> int:
    if not torch.cuda.is_available():
        print("[PPU sparse device] FAIL: torch device runtime unavailable", file=sys.stderr)
        return 1
    run_full_density("NHD", 128)
    run_full_density("HND", 64)
    run_h3_summary()
    run_sol_summary()
    run_negative()
    print(
        "[PPU sparse device] PASS: dense identity + H3 top-k summary + "
        "Sol diag summary + tail/GQA/layout/replay"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
