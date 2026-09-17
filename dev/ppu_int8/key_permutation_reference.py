"""Independent tensor transpose for checking the native K preparation bytes.

Use reshape/transpose, not the production bit formulas or physical MMA map.
The final incomplete K64 block is left alone. The operation is its own inverse.
"""
import torch


def permute_key_reference(codes, layout):
    if layout not in ("HND", "NHD"):
        raise ValueError("K layout must be HND or NHD")
    axis = 2 if layout == "HND" else 1
    tokens = codes.shape[axis]
    indexes = torch.arange(tokens, device=codes.device)
    full = tokens // 64 * 64
    indexes = torch.cat((indexes[:full].reshape(-1, 4, 4).transpose(1, 2).flatten(),
                         indexes[full:]))
    return codes.index_select(axis, indexes)
