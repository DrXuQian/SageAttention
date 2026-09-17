#!/usr/bin/env python3
"""CPU-only profile target contracts; no device performance verdict."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import profile_ppu_attention_pipes as target


class ProfileContract(unittest.TestCase):
    def test_default_is_one_sage_launch(self):
        args = target.arguments([])
        self.assertEqual((args.arm, args.iters), ("sage", 1))

    def test_h3_plan_without_importing_torch(self):
        with patch.object(target, "load_backends", side_effect=AssertionError("GPU import")), \
                patch.dict(sys.modules, {"torch": None}), \
                contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(target.main(["--heads", "56", "--seq", "73774", "--describe"]), 0)
        plan = json.loads(capture.getvalue().split("] ", 1)[1])
        self.assertEqual((plan["B"], plan["H"], plan["Hkv"], plan["S"], plan["D"]),
                         (1, 56, 56, 73774, 128))
        self.assertEqual(plan["sage_expected_grid"], [577, 56, 1])
        self.assertEqual(plan["sage_expected_threads"], 128)
        self.assertEqual(plan["input_dtype"], "bf16")
        self.assertFalse(plan["causal"])
        self.assertFalse(plan["compile"])
        self.assertEqual(plan["warmup"], 0)

    def test_sage_never_imports_flash(self):
        with patch.object(target.importlib, "import_module") as load:
            target.load_backends(target.selected_arms("sage"))
        load.assert_called_once_with("sageattention._qattn_ppu")

    def test_flash_never_imports_sage(self):
        with patch.object(target.importlib, "import_module") as load:
            target.load_backends(target.selected_arms("flash"))
        load.assert_called_once_with("flash_attn_2_cuda")

    def test_both_selects_each_arm_once(self):
        self.assertEqual(target.selected_arms("both"), ("flash", "sage"))

    def test_key_layout_is_explicit_and_int8_only(self):
        self.assertEqual(target.arguments([]).key_layout, "raw")
        self.assertEqual(target.arguments(["--key-layout", "permuted"]).key_layout, "permuted")
        for argv in (["--pv", "fp16"], ["--arm", "flash"], ["--arm", "both"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                target.arguments(argv + ["--key-layout", "permuted"])

    def test_bad_work_fails_before_device_import(self):
        for argv in (["--batch", "0"], ["--heads", "-1"], ["--seq", "0"],
                     ["--head-dim", "96"], ["--iters", "0"], ["--device", "-1"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                target.arguments(argv)


if __name__ == "__main__":
    unittest.main()
