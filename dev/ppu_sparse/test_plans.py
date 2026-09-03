#!/usr/bin/env python3
"""Host admission for PPU sparse plans; no PPU runtime is required."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sageattention.ppu_sparse.plan import _pack_bits, unpack_bits
from sageattention.ppu_sparse.planners import (
    _block_means_bthd,
    make_h3_topk_plan,
    make_sol_plan,
)
from sageattention.ppu_sparse.reference import sparse_attention_reference


def expect_red(name, fn, contains: str) -> None:
    try:
        fn()
    except (TypeError, ValueError) as error:
        if contains not in str(error):
            raise AssertionError(f"{name}: wrong diagnostic: {error}") from error
        print(f"[PPU sparse negative] plant={name} EXPECTED-RED/PASS")
        return
    raise AssertionError(f"{name}: plant did not turn red")


def test_bitset() -> None:
    source = torch.zeros((3, 65), dtype=torch.bool)
    source[0, [0, 31, 32, 64]] = True
    source[1, 1::3] = True
    source[2] = True
    packed = _pack_bits(source)
    recovered = unpack_bits(packed, 65)
    assert torch.equal(source, recovered)
    assert packed.shape == (3, 3)
    print("[PPU sparse plan] bitset=65-boundary ROUNDTRIP/PASS")


def test_h3() -> None:
    torch.manual_seed(7)
    q = torch.randn((1, 257, 2, 64), dtype=torch.float16)
    k = torch.randn((1, 257, 1, 64), dtype=torch.float16)
    v = torch.randn_like(k)
    plan = make_h3_topk_plan(q, k, v, top_k=2, tensor_layout="NHD")
    plan.validate(deep=True)
    assert plan.query_block == 128 and plan.route_block == 128
    assert plan.query_blocks == 3 and plan.route_blocks == 3
    assert plan.rows == 6

    qm = _block_means_bthd(q, 128)
    km = _block_means_bthd(k, 128).repeat_interleave(2, dim=1)
    expected_routes = (qm @ km.transpose(-1, -2)).topk(2, dim=-1).indices
    expected_mask = torch.zeros((1, 2, 3, 3), dtype=torch.bool)
    expected_mask.scatter_(-1, expected_routes, True)
    assert torch.equal(
        unpack_bits(plan.selected_route_bits, 3).reshape(1, 2, 3, 3),
        expected_mask,
    )
    # Every selected K128 owns both KV64s except route 2, whose tail has one.
    ptr = plan.exact_row_ptr.to(torch.int64)
    for row in range(plan.rows):
        selected = expected_mask.reshape(plan.rows, 3)[row].nonzero().flatten()
        expected = []
        for route in selected.tolist():
            expected.extend(tile for tile in (2 * route, 2 * route + 1) if tile < 5)
        got = plan.exact_kv64[ptr[row] : ptr[row + 1]].tolist()
        assert got == sorted(expected)

    dense = make_h3_topk_plan(q, k, v, top_k=3, tensor_layout="NHD", taylor=False)
    actual = sparse_attention_reference(q, k, v, dense, tensor_layout="NHD")
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=64**-0.5,
    ).transpose(1, 2)
    # PyTorch SDPA and the explicit gather/matmul oracle use different host
    # reduction kernels.  This gate proves the plan's mathematical coverage;
    # raw-bit equality belongs to the PPU dense-vs-full-CSR device admission.
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0)
    print(
        "[PPU sparse plan] algorithm=h3-topk rows=6 route_blocks=3 "
        "K128-to-KV64=EXACT full-density=NUMERIC/PASS"
    )

    bad_bits = plan.selected_route_bits.clone()
    bad_bits[0, 0] ^= 1
    expect_red(
        "h3-route-bit",
        lambda: replace(plan, selected_route_bits=bad_bits).validate(deep=True),
        "route",
    )
    bad_ptr = plan.exact_row_ptr.clone()
    bad_ptr[-1] -= 1
    expect_red(
        "csr-denominator",
        lambda: replace(plan, exact_row_ptr=bad_ptr).validate(deep=True),
        "endpoints",
    )
    duplicate = plan.exact_kv64.clone()
    duplicate[1] = duplicate[0]
    expect_red(
        "csr-duplicate",
        lambda: replace(plan, exact_kv64=duplicate).validate(deep=True),
        "duplicate",
    )


def test_sol() -> None:
    torch.manual_seed(11)
    q = torch.randn((1, 193, 2, 64), dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    tau = 0.7
    plan = make_sol_plan(
        q,
        k,
        v,
        tau=tau,
        thresh_type="diag",
        tensor_layout="NHD",
        sink_blocks=(3, 4),
        sink_query_blocks=(1, 2),
    )
    plan.validate(deep=True)
    assert plan.query_block == 64 and plan.route_block == 64
    assert plan.query_blocks == 4 and plan.route_blocks == 4
    mask = unpack_bits(plan.selected_route_bits, 4).reshape(1, 2, 4, 4)
    assert bool(mask[..., 3].all())
    assert bool(mask[:, :, 1, :].all())
    for query_block in range(4):
        lo, hi = max(0, query_block - 1), min(4, query_block + 2)
        assert bool(mask[:, :, query_block, lo:hi].all())

    # Independent scalar transcription of Sol's diagonal threshold.
    qm = _block_means_bthd(q, 64).float()
    km = _block_means_bthd(k, 64).float()
    scale_log2 = 64**-0.5 / math.log(2.0)
    scores = qm @ km.transpose(-1, -2) * scale_log2
    stat_mean = km.mean(2)
    stat_var = (km - stat_mean[:, :, None]).square().mean(2)
    mean = (qm * stat_mean[:, :, None]).sum(-1) * scale_log2
    variance = (qm.square() * stat_var[:, :, None]).sum(-1) * scale_log2**2
    routed = scores > (mean + tau * torch.sqrt(variance + 1.0e-6))[..., None]
    # Compare only non-neighbor/non-sink entries, where the threshold is the
    # sole route reason.
    for query_block in (0, 2, 3):
        for key_block in range(4):
            if abs(query_block - key_block) <= 1 or key_block == 3:
                continue
            assert torch.equal(mask[:, :, query_block, key_block], routed[:, :, query_block, key_block])
    print(
        "[PPU sparse plan] algorithm=sol-diag rows=8 route_blocks=4 "
        "neighbor=EXACT sink=EXACT threshold=INDEPENDENT/PASS"
    )

    expect_red(
        "sol-sink-range",
        lambda: make_sol_plan(q, k, v, sink_blocks=(0, 5)),
        "outside",
    )


def main() -> int:
    test_bitset()
    test_h3()
    test_sol()
    print("[PPU sparse plan] PASS: H3 top-k + Sol diag + 4 negative contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
