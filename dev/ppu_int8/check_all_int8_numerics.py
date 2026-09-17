#!/usr/bin/env python3
"""Fixed CPU numeric admission; no PPU kernel execution or performance claim."""
import itertools
import torch
from all_int8_reference import quantize_qk, quantize_value, from_quantized, unquantized


def main():
    torch.set_num_threads(4)
    cases = 0
    worst = 0.0
    for d, n, causal, positive in itertools.product((64, 128), (65, 257, 4096), (False, True), (False, True)):
        torch.manual_seed(0x5A6E + d + n)
        q, k, v = [torch.randn(1, h, n, d).to(torch.bfloat16) for h in (2, 1, 1)]
        if positive:
            v = v.abs()
        # Distinct per-channel and per-K64 scales expose metadata addressing.
        v = (v.float() * torch.linspace(0.25, 2, d)).to(torch.bfloat16)
        qi, qs = quantize_qk(q, 32, is_query=True)
        ki, ks = quantize_qk(k, 64)
        vi, vs = quantize_value(v)
        rows = sorted({0, 31, 63, 64, n // 2, n - 1})
        got, _ = from_quantized(qi, qs, ki, ks, vi, vs, causal=causal, query_rows=rows)
        expected = unquantized(q, k, v, causal=causal, query_rows=rows)
        relative = float((got - expected).square().mean().sqrt() / expected.square().mean().sqrt())
        assert torch.isfinite(got).all() and relative <= 0.02, (d, n, causal, positive, relative)
        bad, _ = from_quantized(qi, qs, ki, ks, vi, vs, causal=causal, query_rows=rows, plant="signed-p")
        assert not torch.equal(got, bad), "signed-P negative survived"
        bad, _ = from_quantized(qi, qs, ki, ks, vi, vs, causal=causal, query_rows=rows, plant="v-scale")
        assert not torch.equal(got, bad), "wrong V-channel scale negative survived"
        worst = max(worst, relative)
        cases += 1
    # Long-sequence sampled query rows, the exact S from the requested H3 case.
    torch.manual_seed(0x5A6E)
    q, k, v = [torch.randn(1, 1, 73774, 128).to(torch.bfloat16) for _ in range(3)]
    qi, qs = quantize_qk(q, 32, is_query=True)
    ki, ks = quantize_qk(k, 64)
    vi, vs = quantize_value(v)
    rows = [0, 31, 127, 36887, 73773]
    got, _ = from_quantized(qi, qs, ki, ks, vi, vs, query_rows=rows)
    ref = unquantized(q, k, v, query_rows=rows)
    relative = float((got - ref).square().mean().sqrt() / ref.square().mean().sqrt())
    assert relative <= 0.02, relative
    print(f"[all-int8 CPU numeric] cases={cases}/24 worst_relative_rmse={worst:.8f} "
          f"H3_sampled_rows=5 H3_heads=1 relative_rmse={relative:.8f} threshold=0.02 PASS")
    # Zero and exact cancellation have no useful relative-error denominator.
    # Exercise both tensor layouts against the logical oracle, with an exact
    # absolute-zero criterion; do not hide cancellation behind relative RMSE.
    for layout, zero in itertools.product(("HND", "NHD"), (False, True)):
        q = torch.zeros(1, 2, 128, 64, dtype=torch.bfloat16)
        k = torch.zeros(1, 1, 128, 64, dtype=torch.bfloat16)
        v = ((torch.arange(128) % 2) * 2 - 1).to(torch.bfloat16)[None, None, :, None].expand(1, 1, 128, 64).contiguous()
        if zero:
            v.zero_()
        if layout == "NHD":
            q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
        qi, qs = quantize_qk(q, 32, layout, is_query=True)
        ki, ks = quantize_qk(k, 64, layout)
        vi, vs = quantize_value(v, layout)
        got, _ = from_quantized(qi, qs, ki, ks, vi, vs, layout=layout)
        assert torch.count_nonzero(got).item() == 0
    print("[all-int8 CPU zero/cancellation] HND+NHD cases=4/4 max_abs=0 PASS")
    print("[all-int8 CPU negatives] signed_P + wrong_V_channel_scale EXPECTED-RED/PASS; device=NOT-RUN")


if __name__ == "__main__":
    main()
