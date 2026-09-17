#!/usr/bin/env python3
"""Offline test of an FP8-style sequence V scale applied to INT8 codes.

Not a new oracle or a GPU implementation: use the existing independent dense
reference for both representations and keep all error thresholds unchanged.
"""
from __future__ import annotations
import itertools
import json
import torch
import torch.nn.functional as F
from all_int8_reference import quantize_qk, quantize_value, from_quantized, unquantized


def sequence_value(value):
    value = value.float()
    batch, heads, length, dim = value.shape
    blocks = (length + 63) // 64
    maximum = value.abs().amax(dim=2).clamp_min(1e-7)
    rows = F.pad(value, (0, 0, 0, blocks * 64 - length)).reshape(batch, heads, blocks, 64, dim)
    codes = torch.round(rows * (127.0 / maximum)[:, :, None, None, :]).clamp(-127, 127).to(torch.int8)
    scales = (maximum / 127)[:, :, None, :].expand(batch, heads, blocks, dim).contiguous()
    return codes.transpose(-1, -2).contiguous(), scales


def relative(got, expected):
    return float((got - expected).square().mean().sqrt() / expected.square().mean().sqrt())


def main():
    torch.set_num_threads(4)
    results = []
    for dim, length, causal in itertools.product((64, 128), (65, 257, 4096, 73774), (False, True)):
        torch.manual_seed(0x5A6E + dim + length)
        q, k, v = [torch.randn(1, 1, length, dim).to(torch.bfloat16) for _ in range(3)]
        v = (v.float() * torch.linspace(.25, 2, dim)).to(torch.bfloat16)
        qi, qs = quantize_qk(q, 32, is_query=True)
        ki, ks = quantize_qk(k, 64)
        rows = sorted({0, 31, 63, 64, length // 2, length - 1})
        expected = unquantized(q, k, v, causal=causal, query_rows=rows)
        row = dict(D=dim, N=length, causal=causal)
        for name, quantize in (("block", quantize_value), ("sequence", sequence_value)):
            vi, vs = quantize(v)
            got, _ = from_quantized(qi, qs, ki, ks, vi, vs, causal=causal, query_rows=rows)
            if not torch.isfinite(got).all():
                raise AssertionError(f"{name}: unexpected nonfinite")
            row[name + "_relative_rmse"] = relative(got, expected)
        if row["block_relative_rmse"] > .02:
            raise AssertionError(f"existing contract failed; stop this screen: {row}")
        row["sequence_gate"] = "PASS" if row["sequence_relative_rmse"] <= .02 else "FAIL"
        results.append(row)

    # Same BF16 input; only V-scale granularity changes. A future value cannot
    # change causal output for query 31. Its scale MUST not erase earlier ones.
    q = torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16)
    k = q.clone()
    v = torch.ones_like(q)
    v[:, :, 64:, :] = 16384
    witness = {}
    for causal in (True, False):
        if not causal:
            # Same failure without a mask: attend ordinary V=1 keys, not the
            # remote large values. This is relevant to the noncausal H3 route.
            q[:, :, :, 0] = 16
            k[:, :, :64, 0] = 16
            k[:, :, 64:, 0] = -16
        qi, qs = quantize_qk(q, 32, is_query=True)
        ki, ks = quantize_qk(k, 64)
        expected = unquantized(q, k, v, causal=causal, query_rows=[31])
        row = {}
        for name, quantize in (("block", quantize_value), ("sequence", sequence_value)):
            vi, vs = quantize(v)
            got, _ = from_quantized(qi, qs, ki, ks, vi, vs, causal=causal, query_rows=[31])
            row[name] = dict(got=float(got[0, 0, 0, 0]), want=1., relative_rmse=relative(got, expected))
        assert row["block"]["relative_rmse"] <= .02
        assert row["sequence"]["relative_rmse"] > .02
        witness["causal" if causal else "full"] = row
    print("[global V screen] " + json.dumps(dict(
        device="NOT_RUN", production="UNCHANGED", threshold=.02, random_cases=results,
        unattended_outlier=witness,
        verdict="REJECT_UNCONDITIONAL_SEQUENCE_INT8_SCALE; random-only evidence is insufficient")), flush=True)


if __name__ == "__main__":
    main()
