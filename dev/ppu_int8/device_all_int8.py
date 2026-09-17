#!/usr/bin/env python3
"""Execution-only admission of the installed all-INT8 PPU attention wheel."""
import itertools
import os
import argparse
import torch
from sageattention import ppu_compile, sageattn_qk_int8_pv_int8_ppu
from all_int8_reference import from_quantized, quantize_value, unquantized
from key_permutation_reference import permute_key_reference


def run_case(d, n, causal, layout, dtype, return_lse, query_len=None,
             key_layout="raw", batch=1, kv_heads=1):
    torch.manual_seed(0x5A6E + d + n)
    # GQA 2:1, signs, distinct channels and a K64 boundary/tail.
    nq = n if query_len is None else query_len
    if causal and nq != n:
        raise ValueError("this oracle uses equal-length causal coordinates")
    q, k, v = [torch.randn(batch, h, length, d, dtype=dtype, device="cuda")
               for h, length in ((2 * kv_heads, nq), (kv_heads, n), (kv_heads, n))]
    if layout == "NHD":
        q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(q, k, None, tensor_layout=layout)
    prepared_k, prepared_ks = ki, ks
    attention = ppu_compile.qk_int8_sv_int8_accum_f32_attn
    if key_layout == "permuted":
        prepared_k, prepared_ks = torch.empty_like(ki), torch.empty_like(ks)
        native = ppu_compile._qattn_ppu
        no_mean = torch.empty(0, device=k.device, dtype=k.dtype)
        native.quant_per_block_int8_permuted_k(k, no_mean, prepared_k, prepared_ks,
                                             64, 0 if layout == "NHD" else 1)
        torch.cuda.synchronize()
        if not torch.equal(prepared_k.cpu(), permute_key_reference(ki.cpu(), layout)):
            raise AssertionError("permuted K differs from the independent transpose of raw K codes")
        if not torch.equal(prepared_ks.view(torch.int32), ks.view(torch.int32)):
            raise AssertionError("K permutation changed scale bits")
        attention = native.qk_int8_sv_int8_permuted_k_attn
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
    args = (qi, prepared_k, vi, out, qs, prepared_ks, vs, 0 if layout == "NHD" else 1,
            int(causal), 2, d ** -0.5, int(return_lse))
    lse = attention(*args)
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
    if key_layout == "permuted" and (d, n, causal, layout, return_lse) == (128, 128, False, "NHD", False):
        wrong_args = (qi, ki, vi, out, qs, ks, vs, 0, 0, 2, d ** -0.5, 0)
        attention(*wrong_args)
        torch.cuda.synchronize()
        if torch.allclose(out.float().cpu(), expected, atol=.002, rtol=.01):
            raise AssertionError("raw K mistakenly used with the permuted-K consumer survived")
        print("[permuted K negative] unpermuted-input EXPECTED-RED/PASS", flush=True)
    for _ in range(8):
        replay_lse = attention(*args)
        torch.cuda.synchronize()
        if not torch.equal(out.view(torch.int16), anchor.view(torch.int16)):
            raise AssertionError("all-INT8 replay bit pattern changed")
        if not torch.equal(replay_lse.view(torch.int32), lse_anchor.view(torch.int32)):
            raise AssertionError("all-INT8 LSE replay bit pattern changed")
    print(f"[all-int8 device] B={batch} Hkv={kv_heads} D={d} N={n} NQ={nq} causal={int(causal)} layout={layout} key_layout={key_layout} "
          f"dtype={dtype} lse={int(return_lse)} GQA=2:1 V_code_max_delta={code_delta} "
          f"max_quantized_oracle={(out.cpu().float()-expected).abs().max().item():.8g} "
          f"relative_rmse={relative:.8g} replay=8/8 PASS", flush=True)


def exact_zero_score(key_layout="raw"):
    # All logits zero => exact uniform P, including U8=255. Signed-P bugs flip
    # the result. Each V channel differs, so a transposed/stale V read is red.
    for layout, cancellation in itertools.product(("HND", "NHD"), (False, True)):
        q = torch.zeros((1, 2, 64, 128), device="cuda", dtype=torch.float16)
        k = torch.zeros((1, 1, 64, 128), device="cuda", dtype=torch.float16)
        v = ((torch.arange(128, device="cuda") % 15 - 7) / 8).half()[None, None, None].expand(1, 1, 64, 128).contiguous()
        if cancellation:
            sign = (torch.arange(64, device="cuda") % 2 * 2 - 1).half()
            v *= sign[None, None, :, None]
            wanted = torch.zeros((1, 2, 64, 128), device="cuda", dtype=torch.float16)
        else:
            wanted = v.expand(1, 2, 64, 128).contiguous()
        if layout == "NHD":
            q, k, v, wanted = [x.transpose(1, 2).contiguous() for x in (q, k, v, wanted)]
        if key_layout == "raw":
            out = sageattn_qk_int8_pv_int8_ppu(q, k, v, tensor_layout=layout, smooth_k=False)
        else:
            qi, qs, ki, ks = ppu_compile.quant_per_warp_int8(q, k, None, tensor_layout=layout)
            vi, vs = ppu_compile.quant_value_int8(v, tensor_layout=layout)
            native = ppu_compile._qattn_ppu
            native.quant_per_block_int8_permuted_k(k, torch.empty(0, device=k.device, dtype=k.dtype),
                                                 ki, ks, 64, 0 if layout == "NHD" else 1)
            out = torch.empty_like(q)
            native.qk_int8_sv_int8_permuted_k_attn(qi, ki, vi, out, qs, ks, vs,
                                                0 if layout == "NHD" else 1, 0, 2, 128 ** -.5, 0)
        torch.cuda.synchronize()
        bad = torch.count_nonzero(out.view(torch.int16) != wanted.view(torch.int16)).item()
        if bad:
            raise AssertionError(f"uniform-P distinct-V exact witness failed {bad}/{out.numel()}")
        print(f"[all-int8 exact] uniform_P=255 signed_distinct_V cancellation={int(cancellation)} layout={layout} raw_bad=0/{out.numel()} PASS", flush=True)


def key_quantization_parity():
    # Cover all eight new quantizer types, including smoothing. Distinct batch
    # and head values expose pitch mistakes; K129 includes two full blocks+tail.
    native = ppu_compile._qattn_ppu
    cases = 0
    for d, dtype, layout, mean_on in itertools.product(
            (64, 128), (torch.float16, torch.bfloat16), ("HND", "NHD"), (False, True)):
        torch.manual_seed(90210 + d)
        x = torch.randn(2, 3, 129, d, dtype=dtype, device="cuda")
        mean = x.float().mean(2).to(dtype).contiguous() if mean_on else torch.empty(0, dtype=dtype, device="cuda")
        if layout == "NHD":
            x = x.transpose(1, 2).contiguous()
        a, b = torch.empty_like(x, dtype=torch.int8), torch.empty_like(x, dtype=torch.int8)
        sa = torch.empty((2, 3, 3), dtype=torch.float32, device="cuda")
        sb = torch.empty_like(sa)
        native.quant_per_block_int8(x, mean, a, sa, 64, int(layout == "HND"))
        native.quant_per_block_int8_permuted_k(x, mean, b, sb, 64, int(layout == "HND"))
        torch.cuda.synchronize()
        if not torch.equal(b.cpu(), permute_key_reference(a.cpu(), layout)):
            raise AssertionError("new K quantizer changed codes beyond the declared permutation")
        if not torch.equal(sa.view(torch.int32), sb.view(torch.int32)):
            raise AssertionError("new K quantizer changed scale bits")
        cases += 1
    print(f"[permuted K quantization] types=8/8 cases={cases}/16 mean/layout/batch/head raw-equal PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-layout", choices=("raw", "permuted"), default="raw")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PPU device required; not a CPU/NVIDIA numeric claim")
    torch.cuda.set_device(int(os.environ.get("DEVICE", "0")))
    if "PPU" not in torch.cuda.get_device_name().upper():
        raise RuntimeError("PPU device required; not a CPU/NVIDIA numeric claim")
    torch.set_num_threads(4)
    if args.key_layout == "permuted":
        key_quantization_parity()
    exact_zero_score(args.key_layout)
    for d, n, causal, layout, lse in itertools.product((64, 128), (65, 128), (False, True), ("HND", "NHD"), (False, True)):
        run_case(d, n, causal, layout, torch.bfloat16, lse, key_layout=args.key_layout)
    for lse in (False, True):
        run_case(128, 257, False, "NHD", torch.float16, lse, key_layout=args.key_layout)
    # A denominator changes over 1153 K64 blocks in H3. Short fixtures cannot
    # admit this reassociation. Sample five query rows, not a quadratic H3
    # tensor, but execute the real long K loop, LSE and eight device replays.
    for dtype in (torch.float16, torch.bfloat16):
        run_case(128, 73774, False, "NHD", dtype, True, query_len=5, key_layout=args.key_layout)
    if args.key_layout == "permuted":
        run_case(128, 193, False, "NHD", torch.bfloat16, True,
                 key_layout=args.key_layout, batch=2, kv_heads=3)
        run_case(64, 129, True, "HND", torch.bfloat16, True,
                 key_layout=args.key_layout, batch=2, kv_heads=2)
    print("[all-int8 device] PASS: integer-PV oracle + V pack/tail + exact witness + GQA + dtype/layout/causal + replay")


if __name__ == "__main__":
    main()
