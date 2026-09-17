#!/usr/bin/env python3
"""Single-launch ACU target for installed PPU Sage / BF16 FlashAttention.

Adapted from the uploaded agent's profiling patch, without its kernel changes.
No build, backend fallback, warmup, or benchmark timing loop. Setup kernels can
also appear in ACU; select qk_int8_pv_f16_kernel for the Sage core counters.
"""
from __future__ import annotations

import argparse
import importlib
import json

from benchmark_ppu_sage_bf16 import file_identity


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 1), ("heads", 16), ("seq", 4096),
                          ("head-dim", 128), ("iters", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arm", choices=("sage", "flash", "both"), default="sage")
    parser.add_argument("--seed", type=int, default=0x5A6E)
    parser.add_argument("--describe", action="store_true", help="Print plan without importing Torch")
    args = parser.parse_args(argv)
    if any(getattr(args, name) <= 0 for name in ("batch", "heads", "seq", "head_dim", "iters")):
        parser.error("shape dimensions and --iters must be positive")
    if args.head_dim not in (64, 128) or args.device < 0:
        parser.error("head-dim must be 64/128 and device must be nonnegative")
    return args


def selected_arms(arm):
    return ("flash", "sage") if arm == "both" else (arm,)


def load_backends(arms):
    # A Sage-only profile must not require an installed FlashAttention module.
    modules = {"sage": "sageattention._qattn_ppu", "flash": "flash_attn_2_cuda"}
    return {arm: importlib.import_module(modules[arm]) for arm in arms}


def main(argv=None):
    args = arguments(argv)
    arms = selected_arms(args.arm)
    plan = dict(B=args.batch, H=args.heads, Hkv=args.heads, S=args.seq, D=args.head_dim,
                causal=args.causal, input_dtype="bf16", output_dtype="bf16", layout="NHD",
                smooth_k=False, arms=arms, core_launches_per_arm=args.iters,
                warmup=0, compile=False, device_ordinal=args.device, seed=args.seed,
                sage_expected_grid=[(args.seq + 127) // 128, args.heads, args.batch],
                sage_expected_threads=128, sage_kernel="qk_int8_pv_f16_kernel",
                scope="profile only; not a new numerical correctness verdict")
    print("[profile-target plan] " + json.dumps(plan), flush=True)
    if args.describe:
        return 0

    import torch

    if not torch.cuda.is_available() or "PPU" not in torch.cuda.get_device_name(args.device).upper():
        raise RuntimeError("this profile requires a PPU device; no backend fallback")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    backends = load_backends(arms)
    print("[profile-target binaries] " + json.dumps(
        {arm: file_identity(module.__file__) for arm, module in backends.items()}), flush=True)
    print(f"[profile-target device] {torch.cuda.get_device_name(args.device)} "
          f"torch={torch.__version__}", flush=True)
    shape = (args.batch, args.seq, args.heads, args.head_dim)
    scale = args.head_dim ** -0.5

    # Use the default stream, matching the installed PPU extension's launch ABI.
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.default_stream(args.device)):
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        output = {arm: torch.empty(shape, device="cuda", dtype=torch.bfloat16) for arm in arms}
        if "sage" in arms:
            sage = backends["sage"]
            qi, ki = [torch.empty(shape, device="cuda", dtype=torch.int8) for _ in range(2)]
            qs = torch.empty((args.batch, args.heads, ((args.seq + 127) // 128) * 4),
                             device="cuda", dtype=torch.float32)
            ks = torch.empty((args.batch, args.heads, (args.seq + 63) // 64),
                             device="cuda", dtype=torch.float32)
            vh = torch.empty(shape, device="cuda", dtype=torch.float16)
            no_mean = torch.empty(0, device="cuda", dtype=torch.bfloat16)
            # One preparation, not one per attention invocation. ACU may report
            # these separately; they are not part of the Sage core kernel.
            sage.quant_per_warp_int8(q, qi, qs, 128, 32, 0)
            sage.quant_per_block_int8(k, no_mean, ki, ks, 64, 0)
            vh.copy_(v)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            for arm in arms:
                if arm == "flash":
                    backends[arm].fwd(q, k, v, output[arm], None, 0.0, scale,
                                      args.causal, -1, -1, 0.0, False, None)
                else:
                    sage.qk_int8_sv_f16_accum_f32_attn(qi, ki, vh, output[arm], qs, ks,
                                                     0, int(args.causal), 2, scale, 0)
        torch.cuda.synchronize()
    print(f"[profile-target] COMPLETE arms={','.join(arms)} "
          f"core_launches_per_arm={args.iters}; no device correctness verdict", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
