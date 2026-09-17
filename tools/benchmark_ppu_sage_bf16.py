#!/usr/bin/env python3
"""Same-input PPU Sage INT8-QK/PV versus native BF16 FlashAttention forward.

No build, backend fallback, SDPA dispatch, backward, or parallel benchmark arms.
Q/K/V preparation is excluded from the headline Sage core timing. --pv fp16
selects the retained reference instead of the default U8/S8 integer PV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROLES = ("flash-bf16-core", "sage-prequantized-core", "sage-prepare+core")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("batch", 1), ("heads", 16), ("seq", 4096),
                          ("head-dim", 128), ("warmup", 3), ("samples", 7),
                          ("launches", 20)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--pv", choices=("int8", "fp16"), default="int8")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0x5A6E)
    p.add_argument("--peak-bf16-tflops", type=float, default=500.0)
    p.add_argument("--out", type=Path, default=Path("/workspace/sage-bf16-ab.json"))
    p.add_argument("--describe", action="store_true", help="Print plan without importing Torch")
    args = p.parse_args(argv)
    if any(getattr(args, n) <= 0 for n in ("batch", "heads", "seq", "head_dim", "warmup", "samples", "launches")):
        p.error("shape and measurement counts must be positive")
    if args.head_dim not in (64, 128) or args.device < 0 or not math.isfinite(args.peak_bf16_tflops) or args.peak_bf16_tflops <= 0:
        p.error("head-dim must be 64/128, device nonnegative and BF16 peak positive")
    return args


def roles_for_sample(index):
    # Rotate the leader, not all FA followed by all Sage (clock-drift confound).
    shift = index % len(ROLES)
    return ROLES[shift:] + ROLES[:shift]


def summarize(values, flops, peak):
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("invalid event timing samples")
    median = statistics.median(values)
    tflops = flops / median / 1e6
    return dict(median_us=median, min_us=min(values), max_us=max(values),
                samples_us=values, logical_tflops=tflops,
                bf16_equivalent_mfu_percent=100 * tflops / peak)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def main(argv=None):
    args = arguments(argv)
    pairs = args.seq * (args.seq + 1) // 2 if args.causal else args.seq ** 2
    flops = 4 * args.batch * args.heads * args.head_dim * pairs
    plan = dict(B=args.batch, H=args.heads, Hkv=args.heads, S=args.seq, D=args.head_dim,
                causal=args.causal, input_dtype="bf16", output_dtype="bf16", layout="NHD",
                dropout=0, mask="causal" if args.causal else "none", smooth_k=False,
                logical_flops=flops, roles=ROLES, compile=False,
                sage_pv=args.pv,
                sage_core_excludes="Q/K quantization + V quantization/transpose" if args.pv == "int8" else "Q/K quantization + V bf16-to-fp16 cast",
                preparation="Q/K/V preparation; input/output/quant buffers preallocated",
                fa_bookkeeping="native fwd still allocates its internal LSE/RNG buffers")
    print("[Sage/BF16 plan] " + json.dumps(plan), flush=True)
    if args.describe:
        return 0
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {args.out}")

    import torch
    import flash_attn_2_cuda as fa
    from sageattention import _qattn_ppu as sage

    if not torch.cuda.is_available() or "PPU" not in torch.cuda.get_device_name(args.device).upper():
        raise RuntimeError("this comparison requires the PPU device, not a CUDA backend fallback")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties(args.device)
    shape = (args.batch, args.seq, args.heads, args.head_dim)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    qi, ki = [torch.empty(shape, device="cuda", dtype=torch.int8) for _ in range(2)]
    qs = torch.empty((args.batch, args.heads, math.ceil(args.seq / 128) * 4), device="cuda", dtype=torch.float32)
    ks = torch.empty((args.batch, args.heads, math.ceil(args.seq / 64)), device="cuda", dtype=torch.float32)
    if args.pv == "int8":
        if not hasattr(sage, "quant_value_int8"):
            raise RuntimeError("installed Sage wheel lacks INT8 PV; install the all-INT8 wheel or use --pv fp16")
        vh = torch.empty((args.batch, args.heads, (args.seq + 63) // 64, args.head_dim, 64),
                         device="cuda", dtype=torch.int8)
        vs = torch.empty(vh.shape[:-1], device="cuda", dtype=torch.float32)
    else:
        vh = torch.empty(shape, device="cuda", dtype=torch.float16)
    no_mean = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    output = {name: torch.empty(shape, device="cuda", dtype=torch.bfloat16)
              for name in ("flash", "sage")}
    scale = args.head_dim ** -0.5

    def prepare():
        sage.quant_per_warp_int8(q, qi, qs, 128, 32, 0)
        sage.quant_per_block_int8(k, no_mean, ki, ks, 64, 0)
        if args.pv == "int8":
            sage.quant_value_int8(v, vh, vs, 0)
        else:
            vh.copy_(v)

    def sage_core():
        if args.pv == "int8":
            sage.qk_int8_sv_int8_accum_f32_attn(qi, ki, vh, output["sage"], qs, ks, vs,
                                             0, int(args.causal), 2, scale, 0)
        else:
            sage.qk_int8_sv_f16_accum_f32_attn(qi, ki, vh, output["sage"], qs, ks,
                                            0, int(args.causal), 2, scale, 0)

    def fa_core():
        # Public FA2 fwd extension ABI, verified against flash-attention-for-sail.
        # Preallocate O for both arms; FA's internal LSE/RNG bookkeeping remains.
        return fa.fwd(q, k, v, output["flash"], None, 0.0, scale,
                      args.causal, -1, -1, 0.0, False, None)

    def prepare_and_core():
        prepare()
        sage_core()

    functions = dict(zip(ROLES, (fa_core, sage_core, prepare_and_core)))
    rows = sorted({0, min(31, args.seq - 1), min(127, args.seq - 1), args.seq // 2, args.seq - 1})

    def sample(tensor):
        return tensor[:, rows].float().cpu().contiguous()

    timings = {role: [] for role in ROLES}
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.default_stream(args.device)):
        prepare()
        for tensor in output.values():
            tensor.fill_(float("nan"))
        fa_result = fa_core()
        if not fa_result or fa_result[0].data_ptr() != output["flash"].data_ptr():
            raise RuntimeError("installed FA2 does not honor the preallocated output ABI")
        del fa_result
        sage_core()
        torch.cuda.synchronize()
        for role, tensor in output.items():
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(f"{role} preflight nonfinite/unwritten output")
        anchor = {role: sample(tensor) for role, tensor in output.items()}
        for _ in range(args.warmup):
            for fn in functions.values():
                fn()
        torch.cuda.synchronize()
        for index in range(args.samples):
            for role in roles_for_sample(index):
                begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                begin.record()
                for _ in range(args.launches):
                    functions[role]()
                end.record()
                end.synchronize()
                timings[role].append(begin.elapsed_time(end) * 1000 / args.launches)
                key = "flash" if role == ROLES[0] else "sage"
                if not torch.equal(sample(output[key]).view(torch.int32), anchor[key].view(torch.int32)):
                    raise RuntimeError(f"{role} sampled replay changed")

    delta = anchor["sage"] - anchor["flash"]
    rms = delta.square().mean().sqrt().item()
    reference_rms = anchor["flash"].square().mean().sqrt().item()
    summaries = {role: summarize(values, flops, args.peak_bf16_tflops) for role, values in timings.items()}
    bf16 = summaries[ROLES[0]]
    for role, result in summaries.items():
        result["speedup_vs_bf16"] = bf16["median_us"] / result["median_us"]
        if role != ROLES[0]:
            result["sample_envelopes"] = (
                "SAGE-FASTER" if result["max_us"] < bf16["min_us"] else
                "BF16-FASTER" if bf16["max_us"] < result["min_us"] else "UNRESOLVED")
        print("[Sage/BF16 result] " + json.dumps(dict(role=role, **result)), flush=True)
    report = dict(schema=1, plan=plan, torch=torch.__version__, device=properties.name,
                  cu=properties.multi_processor_count, device_ordinal=args.device,
                  binaries={"sage": file_identity(sage.__file__), "flash": file_identity(fa.__file__)},
                  benchmark=file_identity(__file__), protocol="sequential-rotating-arms/device-events/launch-span",
                  warmup=args.warmup, samples=args.samples, launches=args.launches, seed=args.seed,
                  bf16_peak_tflops=args.peak_bf16_tflops,
                  mfu_scope="BF16-equivalent normalization; NOT hardware SOL; Sage PV type is recorded in plan",
                  sampled_rows=rows, sampled_replay="STABLE", comparison_scope="quantization error diagnostic, not raw-bit equality",
                  sampled_max_abs=delta.abs().max().item(), sampled_rmse=rms,
                  sampled_relative_rmse=rms / max(reference_rms, 1e-30), results=summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[Sage/BF16] sampled_quantization_max_abs={report['sampled_max_abs']:.6g} "
          f"relative_rmse={report['sampled_relative_rmse']:.6g} output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
