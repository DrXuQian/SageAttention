#!/usr/bin/env python3
"""Execute the actual public route on CPU tensors with a recorded native boundary."""

import ast
from pathlib import Path
from typing import Any, Optional
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
FUNCTIONS = {"sageattn", "sageattn_qk_int8_pv_fp16_ppu"}


def run_route(*, named=False, layout="HND", plant=None):
    tree = ast.parse((ROOT / "sageattention/core.py").read_text())
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in body} != FUNCTIONS:
        raise AssertionError("public FP16 route inventory changed")
    if plant == "integer-default":
        route = next(node for node in body if node.name == "sageattn")
        calls = [node for node in ast.walk(route)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "sageattn_qk_int8_pv_fp16_ppu"]
        if len(calls) != 1:
            raise AssertionError("negative did not identify exactly one PPU dispatch")
        calls[0].func.id = "integer_default"
    elif plant == "quantize-v":
        route = next(node for node in body if node.name == "sageattn_qk_int8_pv_fp16_ppu")
        route.body.insert(1, ast.parse("ppu_compile.quant_value_int8(v)").body[0])
    backend = Mock()
    backend.quant_per_warp_int8.return_value = (Mock(), Mock(), Mock(), Mock())
    scope = dict(torch=torch, F=F, Any=Any, Optional=Optional,
                 PPU_ENABLED=True, ppu_compile=backend,
                 integer_default=lambda q, k, v, **kwargs: q)
    code = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(code, "core.py", "exec"), scope)
    qshape = (1, 2, 65, 128) if layout == "HND" else (1, 65, 2, 128)
    kvshape = (1, 1, 65, 128) if layout == "HND" else (1, 65, 1, 128)
    q = torch.zeros(qshape, dtype=torch.bfloat16)
    k = torch.zeros(kvshape, dtype=torch.bfloat16)
    v = torch.arange(k.numel(), dtype=torch.float32).reshape(kvshape).to(torch.bfloat16)
    name = "sageattn_qk_int8_pv_fp16_ppu" if named else "sageattn"
    with patch.object(torch.cuda, "set_device"):
        result = scope[name](q, k, v, tensor_layout=layout, smooth_k=False)
    backend.qk_int8_sv_f16_accum_f32_attn.assert_called_once()
    backend.quant_per_warp_int8.assert_called_once()
    backend.quant_value_int8.assert_not_called()
    backend.qk_int8_sv_int8_accum_f32_attn.assert_not_called()
    value = backend.qk_int8_sv_f16_accum_f32_attn.call_args.args[2]
    if value.dtype != torch.float16 or not torch.equal(value, v.to(torch.float16)):
        raise AssertionError("FP16 V was quantized or reinterpreted")
    if result.shape != q.shape or result.dtype != q.dtype:
        raise AssertionError("output contract changed")


class Contract(unittest.TestCase):
    def test_public_and_named_paths(self):
        for named in (False, True):
            for layout in ("HND", "NHD"):
                with self.subTest(named=named, layout=layout):
                    run_route(named=named, layout=layout)

    def test_integer_default_is_red(self):
        with self.assertRaises(AssertionError):
            run_route(plant="integer-default")

    def test_hidden_value_quantizer_is_red(self):
        with self.assertRaises(AssertionError):
            run_route(plant="quantize-v")


if __name__ == "__main__":
    unittest.main()
