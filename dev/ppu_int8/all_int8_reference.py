"""Independent CPU tensor reference for the PPU U8/S8 PV contract.

This never imports the extension or its physical layout helpers. Matmuls and
probability quantization are expressed on logical dense matrices.
"""
import math
import torch
import torch.nn.functional as F


def hnd(tensor, layout):
    return tensor.transpose(1, 2) if layout == "NHD" else tensor


def quantize_value(value, layout="HND"):
    value = hnd(value.cpu(), layout).float()
    b, h, length, d = value.shape
    blocks = (length + 63) // 64
    rows = F.pad(value, (0, 0, 0, blocks * 64 - length)).reshape(b, h, blocks, 64, d)
    maximum = rows.abs().amax(dim=-2).clamp_min(1e-7)
    scale = maximum / 127.0
    codes = torch.round(rows * (127.0 / maximum).unsqueeze(-2)).clamp(-127, 127).to(torch.int8)
    return codes.transpose(-1, -2).contiguous(), scale


def quantize_qk(value, rows, layout="HND", is_query=False):
    value = hnd(value.cpu(), layout).float()
    b, h, length, d = value.shape
    groups = ((length + 127) // 128) * 4 if is_query else (length + rows - 1) // rows
    block = F.pad(value, (0, 0, 0, groups * rows - length)).reshape(b, h, groups, rows, d)
    maximum = block.abs().amax(dim=(-1, -2)).clamp_min(1e-7)
    scale = maximum / 127.0
    codes = torch.round(block * (127.0 / maximum)[..., None, None]).clamp(-127, 127).to(torch.int8)
    codes = codes.reshape(b, h, groups * rows, d)[..., :length, :].contiguous()
    return (codes.transpose(1, 2).contiguous() if layout == "NHD" else codes), scale


def from_quantized(qi, qs, ki, ks, packed_v, vs, *, layout="HND", causal=False,
                   sm_scale=None, query_rows=None, plant=None):
    q, k = hnd(qi.cpu(), layout).float(), hnd(ki.cpu(), layout).float()
    qs, ks, vs, packed_v = [x.cpu().float() for x in (qs, ks, vs, packed_v)]
    b, hq, nq, d = q.shape
    _, hk, nk, _ = k.shape
    rows = torch.arange(nq) if query_rows is None else torch.tensor(query_rows)
    block_count = (nk + 63) // 64
    q = q[:, :, rows] * qs[:, :, rows // 32].unsqueeze(-1)
    k = k * ks[:, :, torch.arange(nk) // 64].unsqueeze(-1)
    group = hq // hk
    k = k.repeat_interleave(group, dim=1)
    packed_v = packed_v.repeat_interleave(group, dim=1)
    vs = vs.repeat_interleave(group, dim=1)
    if plant == "v-scale":
        vs = vs.roll(1, dims=-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (d ** -0.5 if sm_scale is None else sm_scale)
    if causal:
        scores = scores.masked_fill(torch.arange(nk)[None, :] > rows[:, None], -float("inf"))
    lse = torch.logsumexp(scores, dim=-1) / math.log(2.0)
    padded = F.pad(scores, (0, block_count * 64 - nk), value=-float("inf"))
    chunk = padded.reshape(b, hq, len(rows), block_count, 64)
    local_max = chunk.amax(-1)
    valid = torch.isfinite(local_max)
    shifted = chunk - torch.where(valid, local_max, 0.0)[..., None]
    local_p = shifted.exp()
    pcodes = torch.round(local_p * 255.0).clamp(0, 255)
    if plant == "signed-p":
        pcodes = pcodes.to(torch.uint8).to(torch.int8).float()
    global_max = scores.amax(-1)
    factor = (local_max - global_max[..., None]).exp()
    denominator = (local_p.sum(-1) * factor).sum(-1)
    # [B,H,chunk,query,K] @ [B,H,chunk,K,D], then per-channel V scale.
    dots = torch.matmul(pcodes.transpose(2, 3), packed_v.transpose(-1, -2))
    value = (dots * vs.unsqueeze(-2)).transpose(2, 3)
    out = (value * (factor / 255.0)[..., None]).sum(-2) / denominator[..., None]
    if layout == "NHD":
        out = out.transpose(1, 2)
    return out, lse


def unquantized(q, k, v, *, layout="HND", causal=False, query_rows=None):
    q, k, v = [hnd(x.cpu(), layout).float() for x in (q, k, v)]
    nq = q.shape[2]
    rows = torch.arange(nq) if query_rows is None else torch.tensor(query_rows)
    groups = q.shape[1] // k.shape[1]
    scores = torch.matmul(q[:, :, rows], k.repeat_interleave(groups, 1).transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        scores = scores.masked_fill(torch.arange(k.shape[2])[None, :] > rows[:, None], -float("inf"))
    out = torch.matmul(scores.softmax(-1), v.repeat_interleave(groups, 1))
    return out.transpose(1, 2) if layout == "NHD" else out
