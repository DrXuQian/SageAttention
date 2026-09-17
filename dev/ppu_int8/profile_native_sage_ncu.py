#!/usr/bin/env python3
"""Profile unmodified upstream NVIDIA SageAttention2 on matching PPU shapes.

Run with PYTHONPATH pointing at the separately built upstream checkout. This
does not install packages or compile, and never dispatches to the PPU fork.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=73774)
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0x5A6E)
    p.add_argument("--pv", choices=("fp32", "fp32+fp16"), default="fp32+fp16")
    p.add_argument("--describe", action="store_true")
    args = p.parse_args()
    if min(args.seq, args.heads, args.batch) < 1 or args.dim not in (64, 128):
        p.error("positive B/H/S and D64/D128 required")
    plan = dict(B=args.batch, H=args.heads, S=args.seq, D=args.dim,
                input="bf16", output="bf16", layout="NHD", causal=False,
                qk="S8-S8-S32", pv="E4M3-E4M3", pv_accum_mode=args.pv,
                qk_quant_gran="per_warp", smooth_k=False, smooth_v=False,
                seed=args.seed, warmup=1, profile_launches=1,
                nvtx_range="sage_native_profile", preparation="outside NVTX range",
                comparison="same shape; prior PPU inputs not recovered, no byte-identical-input claim")
    print("[native Sage plan] " + json.dumps(plan), flush=True)
    if args.describe:
        return 0

    import torch
    import sageattention
    from sageattention import _qattn_sm89 as backend
    from sageattention.quant import per_warp_int8, per_channel_fp8

    torch.cuda.set_device(args.device)
    device_name = torch.cuda.get_device_name(args.device)
    if "NVIDIA" not in device_name.upper() or "PPU" in device_name.upper():
        raise RuntimeError("this is an NVIDIA-only upstream comparison")
    root = Path(sageattention.__file__).resolve().parent.parent
    source_sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "diff", "--", "csrc", "sageattention"], text=True)
    if dirty:
        raise RuntimeError("upstream algorithm sources are modified")
    selected = "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn" if args.pv == "fp32" else "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf"
    fn = getattr(backend, selected)
    print("[native Sage identity] " + json.dumps(dict(
        source_sha=source_sha, package=identity(sageattention.__file__),
        native=identity(backend.__file__), selected=selected, device=device_name,
        capability=torch.cuda.get_device_capability(args.device),
        torch=torch.__version__, cuda=torch.version.cuda,
        memory_free_total=torch.cuda.mem_get_info(args.device))), flush=True)

    torch.manual_seed(args.seed)
    shape = (args.batch, args.seq, args.heads, args.dim)
    with torch.inference_mode():
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        qi, qs, ki, ks = per_warp_int8(q, k, km=None, tensor_layout="NHD")
        # Match upstream sm120 default FP16-buffer rule, not a PPU V quantizer.
        v_limit = 448.0 if args.pv == "fp32" else 2.25
        vi, vs, _ = per_channel_fp8(v, tensor_layout="NHD", scale_max=v_limit, smooth_v=False)
        out = torch.empty_like(q)
        del q, k, v
        call = (qi, ki, vi, out, qs, ks, vs, 0, 0, 2, args.dim ** -0.5, 0)
        fn(*call)
        torch.cuda.synchronize()
        if not torch.isfinite(out).all().item():
            raise RuntimeError("nonfinite native output at preflight")
        torch.cuda.nvtx.range_push("sage_native_profile")
        fn(*call)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        sample = out.reshape(-1)[:4096].view(torch.int16).cpu().numpy().tobytes()
        print("[native Sage complete] " + json.dumps(dict(
            sampled_output_sha256=hashlib.sha256(sample).hexdigest(),
            scope="profile-only, not a new full numerical admission")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
