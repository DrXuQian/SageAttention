#!/usr/bin/env python3
"""PPU device admission against an oracle built from the quantized operands."""

from __future__ import annotations

import math
import sys

import torch

from sageattention import ppu_compile


def quantized_reference(
    qi, qs, ki, ks, v, layout: str, causal: bool, softmax_scale: float
):
    if layout == "NHD":
        qi_h = qi.transpose(1, 2)
        ki_h = ki.transpose(1, 2)
        v_h = v.transpose(1, 2)
    else:
        qi_h, ki_h, v_h = qi, ki, v
    batch, q_heads, q_len, _ = qi_h.shape
    _, kv_heads, kv_len, _ = ki_h.shape
    group = q_heads // kv_heads
    q_index = (torch.arange(q_len, device=qi.device) // 128) * 4
    q_index += (torch.arange(q_len, device=qi.device) % 128) // 32
    k_index = torch.arange(kv_len, device=ki.device) // 64
    rows = []
    lses = []
    for head in range(q_heads):
        kv_head = head // group
        qf = qi_h[:, head].float() * qs[:, head, q_index].unsqueeze(-1)
        kf = ki_h[:, kv_head].float() * ks[:, kv_head, k_index].unsqueeze(-1)
        logits = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale
        if causal:
            mask = torch.ones(q_len, kv_len, device=qi.device, dtype=torch.bool).triu(1)
            logits = logits.masked_fill(mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        rows.append(torch.matmul(probs, v_h[:, kv_head].float()))
        lses.append(torch.logsumexp(logits, dim=-1))
    out = torch.stack(rows, dim=1)
    lse = torch.stack(lses, dim=1)
    if layout == "NHD":
        out = out.transpose(1, 2)
    # The C++ ABI publishes the online-softmax accumulator in base 2.  The
    # high-level Python API converts it to natural log and then adds K-mean
    # correction, exactly like Sage's SM80 path.
    return out, lse / math.log(2.0)


def run_case(name: str, head_dim: int, seq: int, q_heads: int,
             layout: str, causal: bool) -> None:
    base = torch.zeros((seq, head_dim), device="cuda", dtype=torch.float16)
    base[torch.arange(seq, device="cuda"), torch.arange(seq, device="cuda")] = 1
    if layout == "HND":
        q = base[None, None].repeat(1, q_heads, 1, 1)
        k = base[None, None]
        v = base[None, None]
    else:
        q = base[None, :, None, :].repeat(1, 1, q_heads, 1).contiguous()
        k = base[None, :, None, :].contiguous()
        v = base[None, :, None, :].contiguous()
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(
        q, k, None, tensor_layout=layout
    )
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    layout_id = 0 if layout == "NHD" else 1
    softmax_scale = head_dim ** -0.5
    lse = ppu_compile.qk_int8_sv_f16_accum_f32_attn(
        qi, ki, v, output, qs, ks, layout_id, int(causal), 2,
        softmax_scale, 1,
    )
    expected, expected_lse = quantized_reference(
        qi, qs, ki, ks, v, layout, causal, softmax_scale
    )
    output_error = (output.float() - expected).abs().max().item()
    lse_error = (lse - expected_lse).abs().max().item()
    first = output.clone()
    for _ in range(3):
        ppu_compile.qk_int8_sv_f16_accum_f32_attn(
            qi, ki, v, output, qs, ks, layout_id, int(causal), 2,
            softmax_scale, 1,
        )
        torch.cuda.synchronize()
        if not torch.equal(first, output):
            raise AssertionError(f"{name}: output fingerprint is not stable")
    print(
        f"[PPU Sage device] case={name} shape=B1,H{q_heads},N{seq},D{head_dim} "
        f"layout={layout} causal={int(causal)} max_o={output_error:.8f} "
        f"max_lse={lse_error:.8f} replay=RAW-BIT/STABLE"
    )
    if output_error > 0.002 or lse_error > 0.002:
        raise AssertionError(f"{name}: quantized-oracle mismatch")


def main() -> int:
    if not torch.cuda.is_available():
        print("[PPU Sage device] FAIL: torch device runtime unavailable", file=sys.stderr)
        return 1
    run_case("h64-full", 64, 64, 1, "HND", False)
    run_case("h64-causal", 64, 64, 1, "HND", True)
    run_case("h64-tail-gqa-nhd", 64, 33, 2, "NHD", False)
    run_case("h128-multislice", 128, 128, 1, "HND", False)
    print("[PPU Sage device] PASS: QK/PV/LSE + causal + tail + GQA + NHD + D128")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
