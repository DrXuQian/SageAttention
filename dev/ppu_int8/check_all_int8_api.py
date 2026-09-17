#!/usr/bin/env python3
"""CPU routing/argument tests using the actual public Python function AST."""
import ast
from pathlib import Path
from typing import Any, Optional
import unittest
from unittest.mock import Mock, patch
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT / "sageattention/core.py").read_text())
names = {"sageattn", "_sageattn_ppu", "sageattn_qk_int8_pv_int8_ppu", "sageattn_qk_int8_pv_fp16_ppu"}
code = compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
                          type_ignores=[]), "core.py", "exec")


class Contract(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.backend.quant_per_warp_int8.return_value = (Mock(), Mock(), Mock(), Mock())
        self.backend.quant_value_int8.return_value = (Mock(), Mock())
        self.scope = dict(torch=torch, F=F, Any=Any, Optional=Optional,
                          PPU_ENABLED=True, ppu_compile=self.backend)
        exec(code, self.scope)
        self.q = torch.zeros(1, 2, 65, 128, dtype=torch.bfloat16)
        self.k = self.v = torch.zeros(1, 1, 65, 128, dtype=torch.bfloat16)

    def test_default_uses_integer_pv(self):
        with patch.object(torch.cuda, "set_device"):
            out = self.scope["sageattn"](self.q, self.k, self.v, smooth_k=False)
        self.assertEqual(out.shape, self.q.shape)
        self.backend.quant_value_int8.assert_called_once()
        self.backend.qk_int8_sv_int8_accum_f32_attn.assert_called_once()
        self.backend.qk_int8_sv_f16_accum_f32_attn.assert_not_called()

    def test_named_legacy_remains_fp16(self):
        with patch.object(torch.cuda, "set_device"):
            self.scope["sageattn_qk_int8_pv_fp16_ppu"](self.q, self.k, self.v, smooth_k=False)
        self.backend.quant_value_int8.assert_not_called()
        self.backend.qk_int8_sv_f16_accum_f32_attn.assert_called_once()
        self.backend.qk_int8_sv_int8_accum_f32_attn.assert_not_called()

    def test_invalid_layout_does_not_reach_native(self):
        with self.assertRaises(ValueError):
            self.scope["sageattn_qk_int8_pv_int8_ppu"](self.q, self.k, self.v, tensor_layout="bad")
        self.backend.quant_per_warp_int8.assert_not_called()

    def test_unknown_option_is_not_silently_ignored(self):
        with self.assertRaises(TypeError):
            self.scope["sageattn_qk_int8_pv_int8_ppu"](self.q, self.k, self.v, made_up=True)
        self.backend.quant_per_warp_int8.assert_not_called()


if __name__ == "__main__":
    unittest.main()
