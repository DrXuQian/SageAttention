#!/usr/bin/env python3
"""Execution-only admission of the installed all-INT8 PPU attention wheel."""
import itertools
import os
import torch
from sageattention import ppu_compile, sageattn_qk_int8_pv_int8_ppu
from all_int8_reference import from_quantized, quantize_value, unquantized


def run_case(d, n, causal, layout, dtype, return_lse):
    torch.manual_seed(0x5A6E + d + n)
    # GQA 2:1, signs, distinct channels and a K64 boundary/tail.
    q, k, v = [torch.randn(1, h, n, d, dtype=dtype, device="cuda") for h in (2, 1, 1)]
    if layout == "NHD":
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(q, k, None, tensor_layout=layout)
    vi, vs = ppu_compile.quant_value_int8(v, tensor_layout=layout)
    expected_vi, expected_vs = quantize_value(v, layout)
    # Away from exact rounding ties the quantizer matches; a possible one-code
    # tie difference is reported, while the attention oracle uses actual vi/vs.
    code_delta = (vi.cpu().int() - expected_vi.int()).abs().max().item()
    torch.testing.assert_close(vs.cpu(), expected_vs, atol=1e-7, rtol=2e-6)
    if code_delta > 1:
        raise AssertionError(f"V quantization differs by {code_delta} codes")
    valid = n % 64
    if valid and torch.count_nonzero(vi[:, :, -1, :, valid:]).item():
        raise AssertionError("V tail padding is not zero")
    out = torch.full(q.shape, float("nan"), dtype=dtype, device="cuda")
    args = (qi, ki, vi, out, qs, ks, vs, 0 if layout == "NHD" else 1,
            int(causal), 2, d ** -0.5, int(return_lse))
    lse = ppu_compile.qk_int8_sv_int8_accum_f32_attn(*args)
    torch.cuda.synchronize()
    expected, expected_lse = from_quantized(qi, qs, ki, ks, vi, vs, layout=layout, causal=causal)
    torch.testing.assert_close(out.float().cpu(), expected, atol=0.002, rtol=0.01)
    if return_lse:
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0.001)
    elif lse.numel() != 0:
        raise AssertionError("no-LSE specialization returned a nonempty LSE")
    reference = unquantized(q, k, v, layout=layout, causal=causal)
    relative = float((out.cpu().float() - reference).square().mean().sqrt() /
                     reference.square().mean().sqrt())
    if relative > 0.02:
        raise AssertionError(f"relative RMSE {relative} exceeds fixed 0.02")
    anchor = out.clone()
    lse_anchor = lse.clone()
    for _ in range(8):
        replay_lse = ppu_compile.qk_int8_sv_int8_accum_f32_attn(*args)
        torch.cuda.synchronize()
        if not torch.equal(out.view(torch.int16), anchor.view(torch.int16)):
            raise AssertionError("all-INT8 replay bit pattern changed")
        if not torch.equal(replay_lse.view(torch.int32), lse_anchor.view(torch.int32)):
            raise AssertionError("all-INT8 LSE replay bit pattern changed")
    print(f"[all-int8 device] D={d} N={n} causal={int(causal)} layout={layout} "
          f"dtype={dtype} lse={int(return_lse)} GQA=2:1 V_code_max_delta={code_delta} "
          f"max_quantized_oracle={(out.cpu().float()-expected).abs().max().item():.8g} "
          f"relative_rmse={relative:.8g} replay=8/8 PASS", flush=True)


def exact_zero_score():
    # All logits zero => exact uniform P, including U8=255. Signed-P bugs flip
    # the result. Each V channel differs, so a transposed/stale V read is red.
    for layout in ("HND", "NHD"):
        q = torch.zeros((1, 2, 64, 128), device="cuda", dtype=torch.float16)
        k = torch.zeros((1, 1, 64, 128), device="cuda", dtype=torch.float16)
        v = ((torch.arange(128, device="cuda") % 15 - 7) / 8).half()[None, None, None].expand(1, 1, 64, 128).contiguous()
        wanted = v.expand(1, 2, 64, 128).contiguous()
        if layout == "NHD":
            q, k, v, wanted = [x.transpose(1, 2).contiguous() for x in (q, k, v, wanted)]
        out = sageattn_qk_int8_pv_int8_ppu(q, k, v, tensor_layout=layout, smooth_k=False)
        torch.cuda.synchronize()
        bad = torch.count_nonzero(out.view(torch.int16) != wanted.view(torch.int16)).item()
        if bad:
            raise AssertionError(f"uniform-P distinct-V exact witness failed {bad}/{out.numel()}")
        print(f"[all-int8 exact] uniform_P=255 signed_distinct_V layout={layout} raw_bad=0/{out.numel()} PASS", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PPU device required; not a CPU/NVIDIA numeric claim")
    torch.cuda.set_device(int(os.environ.get("DEVICE", "0")))
    if "PPU" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("PPU device required; not a CPU/NVIDIA numeric claim")
    torch.set_num_threads(4)
    exact_zero_score()
    for d, n, causal, layout, lse in itertools.product((64, 128), (65, 128), (False, True), ("HND", "NHD"), (False, True)):
        run_case(d, n, causal, layout, torch.bfloat16, lse)
    for lse in (False, True):
        run_case(128, 257, False, "NHD", torch.float16, lse)
    print("[all-int8 device] PASS: integer-PV oracle + V pack/tail + exact witness + GQA + dtype/layout/causal + replay")


if __name__ == "__main__":
    main()
