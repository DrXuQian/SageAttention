#!/usr/bin/env python3
"""Paired Sage INT8-PV/FP16-PV core event timing, without ACU or preparation.

Both entrypoints come from one identified DSO; the candidate's FP16-PV native
body is separately checked against the retained baseline at build time. This
does not require FlashAttention, install a wheel, or rebuild a kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmark_ppu_sage_bf16 import file_identity

ROLES = ("fp16-pv", "int8-pv")
PERMUTED_ROLES = ROLES + ("int8-pv-permuted-k", "K-quant-raw", "K-quant-permuted",
                         "K-prepare+int8-pv", "K-prepare+int8-pv-permuted-k")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 1), ("heads", 56), ("seq", 73774),
                          ("head-dim", 128), ("warmup", 2), ("samples", 7),
                          ("launches", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0x5A6E)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--permuted-k", action="store_true", help="Add explicit K-layout candidate and its moved preparation cost")
    parser.add_argument("--out", type=Path, default=Path("/workspace/sage-pv-core.json"))
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    if any(getattr(args, name) <= 0 for name in
           ("batch", "heads", "seq", "head_dim", "warmup", "samples", "launches")):
        parser.error("all shape and measurement counts must be positive")
    if args.head_dim not in (64, 128) or args.device < 0 or args.samples < 3:
        parser.error("head-dim must be 64/128, device >=0 and samples >=3")
    return args


def roles_for_sample(index, roles=ROLES):
    shift = index % len(roles)
    return roles[shift:] + roles[:shift]


def summary(samples):
    if len(samples) < 3 or not all(math.isfinite(x) and x > 0 for x in samples):
        raise ValueError("invalid or insufficient event samples")
    return dict(median_us=statistics.median(samples), min_us=min(samples),
                max_us=max(samples), samples_us=samples)


def verdict(fp16, int8):
    if int8["max_us"] < fp16["min_us"]:
        return "INT8-FASTER"
    if fp16["max_us"] < int8["min_us"]:
        return "FP16-FASTER"
    return "UNRESOLVED"


def launch_core(pv, extension, tensors, causal, scale):
    if pv == "int8-pv-permuted-k":
        return extension.qk_int8_sv_int8_permuted_k_attn(
            tensors["qi"], tensors["kp"], tensors["vi"], tensors[pv],
            tensors["qs"], tensors["kps"], tensors["vs"], 0, int(causal), 2, scale, 0)
    if pv == "int8-pv":
        return extension.qk_int8_sv_int8_accum_f32_attn(
            tensors["qi"], tensors["ki"], tensors["vi"], tensors["int8-pv"],
            tensors["qs"], tensors["ks"], tensors["vs"], 0, int(causal), 2, scale, 0)
    if pv == "fp16-pv":
        return extension.qk_int8_sv_f16_accum_f32_attn(
            tensors["qi"], tensors["ki"], tensors["vf"], tensors["fp16-pv"],
            tensors["qs"], tensors["ks"], 0, int(causal), 2, scale, 0)
    raise ValueError(f"unknown PV arm: {pv}")


def main(argv=None):
    args = arguments(argv)
    roles = PERMUTED_ROLES if args.permuted_k else ROLES
    plan = dict(B=args.batch, H=args.heads, S=args.seq, D=args.head_dim,
                causal=args.causal, input_dtype="bf16", output_dtype="bf16", layout="NHD",
                roles=roles, QK="same codes/scales; explicit K row permutation" if args.permuted_k else "shared-identical-quantized-buffers", smooth_k=False,
                timing="normal-device-events/launch-span/alternating-arms",
                preparation="once outside core; K-only and K-prepare+core also measured" if args.permuted_k else "once; outside events", warmup=args.warmup,
                samples=args.samples, launches_per_sample=args.launches, seed=args.seed,
                acu=False, compile=False, scope="core-only; unequal P/V quantization",
                verdict="strictly disjoint sample envelopes, otherwise UNRESOLVED")
    print("[PV core plan] " + json.dumps(plan), flush=True)
    if args.describe:
        return 0
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite result: {args.out}")

    import torch
    from sageattention import _qattn_ppu as sage

    if not torch.cuda.is_available() or "PPU" not in torch.cuda.get_device_name(args.device).upper():
        raise RuntimeError("PPU device required; no backend fallback")
    torch.cuda.set_device(args.device)
    properties = torch.cuda.get_device_properties(args.device)
    binary = file_identity(sage.__file__)
    print("[PV core identity] " + json.dumps(dict(binary=binary, torch=torch.__version__,
          device=properties.name, ordinal=args.device, cu=properties.multi_processor_count)), flush=True)
    torch.manual_seed(args.seed)
    shape = (args.batch, args.seq, args.heads, args.head_dim)
    rows = sorted({0, min(31, args.seq - 1), min(127, args.seq - 1), args.seq // 2, args.seq - 1})
    scale = args.head_dim ** -.5
    timings = {role: [] for role in roles}
    execution_order = []

    # The native extension launches on the default stream; record events there.
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.default_stream(args.device)):
        q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        t = {key: torch.empty(shape, device="cuda", dtype=torch.int8) for key in ("qi", "ki")}
        core_roles = ROLES + (("int8-pv-permuted-k",) if args.permuted_k else ())
        t.update({role: torch.empty(shape, device="cuda", dtype=torch.bfloat16) for role in core_roles})
        t["qs"] = torch.empty((args.batch, args.heads, ((args.seq + 127) // 128) * 4),
                              device="cuda", dtype=torch.float32)
        t["ks"] = torch.empty((args.batch, args.heads, (args.seq + 63) // 64),
                              device="cuda", dtype=torch.float32)
        t["vi"] = torch.empty((args.batch, args.heads, (args.seq + 63) // 64, args.head_dim, 64),
                              device="cuda", dtype=torch.int8)
        t["vs"] = torch.empty(t["vi"].shape[:-1], device="cuda", dtype=torch.float32)
        t["vf"] = v.to(torch.float16)
        no_mean = torch.empty(0, device="cuda", dtype=torch.bfloat16)
        sage.quant_per_warp_int8(q, t["qi"], t["qs"], 128, 32, 0)
        sage.quant_per_block_int8(k, no_mean, t["ki"], t["ks"], 64, 0)
        sage.quant_value_int8(v, t["vi"], t["vs"], 0)
        functions = {role: (lambda r=role: launch_core(r, sage, t, args.causal, scale))
                     for role in core_roles}
        if args.permuted_k:
            if not hasattr(sage, "qk_int8_sv_int8_permuted_k_attn"):
                raise RuntimeError("candidate lacks explicit permuted-K entrypoint; no fallback")
            t["kp"] = torch.empty_like(t["ki"])
            t["kps"] = torch.empty_like(t["ks"])
            def quant_raw():
                sage.quant_per_block_int8(k, no_mean, t["ki"], t["ks"], 64, 0)
            def quant_permuted():
                sage.quant_per_block_int8_permuted_k(k, no_mean, t["kp"], t["kps"], 64, 0)
            def prepare_raw_core():
                quant_raw()
                functions["int8-pv"]()
            def prepare_permuted_core():
                quant_permuted()
                functions["int8-pv-permuted-k"]()
            quant_permuted()
            functions.update({"K-quant-raw": quant_raw, "K-quant-permuted": quant_permuted,
                              "K-prepare+int8-pv": prepare_raw_core,
                              "K-prepare+int8-pv-permuted-k": prepare_permuted_core})
            # Reuse each arm's O, not extra copies or an omitted output check.
            t["K-quant-raw"], t["K-quant-permuted"] = t["ki"], t["kp"]
            t["K-prepare+int8-pv"] = t["int8-pv"]
            t["K-prepare+int8-pv-permuted-k"] = t["int8-pv-permuted-k"]
        torch.cuda.synchronize()

        def sample(role):
            return t[role][:, rows].cpu().contiguous()

        anchors = {}
        for role in roles:
            if not role.startswith("K-quant-"):
                t[role].fill_(float("nan"))
            functions[role]()
            torch.cuda.synchronize()
            if not torch.isfinite(t[role]).all().item():
                raise RuntimeError(f"{role}: nonfinite or unwritten preflight output")
            anchors[role] = sample(role)
        if args.permuted_k:
            torch.testing.assert_close(anchors["int8-pv-permuted-k"].float(),
                                       anchors["int8-pv"].float(), atol=.002, rtol=.01)
        for iteration in range(args.warmup):
            for role in roles_for_sample(iteration, roles):
                functions[role]()
        torch.cuda.synchronize()
        for iteration in range(args.samples):
            order = roles_for_sample(iteration, roles)
            execution_order.append(order)
            for role in order:
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(args.launches):
                    functions[role]()
                end.record()
                end.synchronize()
                timings[role].append(begin.elapsed_time(end) * 1000 / args.launches)
                if not torch.equal(sample(role).view(torch.int16), anchors[role].view(torch.int16)):
                    raise RuntimeError(f"{role}: sampled raw replay changed")

    results = {role: summary(values) for role, values in timings.items()}
    decision = verdict(results["fp16-pv"], results["int8-pv"])
    speedup = results["fp16-pv"]["median_us"] / results["int8-pv"]["median_us"]
    fingerprints = {role: hashlib.sha256(tensor.view(torch.int16).numpy().tobytes()).hexdigest()
                    for role, tensor in anchors.items()}
    delta = anchors["int8-pv"].float() - anchors["fp16-pv"].float()
    report = dict(schema=1, plan=plan, binary=binary, benchmark=file_identity(__file__),
                  device=properties.name, ordinal=args.device, cu=properties.multi_processor_count,
                  torch=torch.__version__, results=results, execution_order=execution_order,
                  verdict=decision, speedup_int8_vs_fp16=speedup, sampled_rows=rows,
                  output_sha256_sampled=fingerprints, replay="sampled-RAW-BIT/STABLE",
                  between_arm_max_abs_diagnostic=delta.abs().max().item(),
                  numerical_authority="separate device_all_int8 gate; this timing is not an independent numeric oracle")
    for role in roles:
        print("[PV core result] " + json.dumps(dict(role=role, **results[role])), flush=True)
    print(f"[PV core verdict] {decision} speedup_int8_vs_fp16={speedup:.6f}x "
          "scope=PREQUANTIZED-CORE event_timing=NOT-ACU", flush=True)
    if args.permuted_k:
        comparisons = {}
        for label, old, new in (
                ("permuted-core-vs-raw", "int8-pv", "int8-pv-permuted-k"),
                ("permuted-core-vs-fp16", "fp16-pv", "int8-pv-permuted-k"),
                ("K-prepare+core-vs-raw", "K-prepare+int8-pv", "K-prepare+int8-pv-permuted-k")):
            speed = results[old]["median_us"] / results[new]["median_us"]
            v = verdict(results[old], results[new]).replace("INT8-FASTER", "CANDIDATE-FASTER").replace("FP16-FASTER", "CONTROL-FASTER")
            comparisons[label] = dict(verdict=v, speedup=speed, control=old, candidate=new)
            print(f"[K permutation verdict] {label} {v} speedup={speed:.6f}x", flush=True)
        report["key_permutation_comparisons"] = comparisons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
